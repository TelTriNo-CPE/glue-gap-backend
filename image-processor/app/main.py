import asyncio
import io
import json
import logging
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import openpyxl
import pyvips
from fastapi import FastAPI, HTTPException
from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-processor")

app = FastAPI(title="image-processor")

BUCKET = os.environ.get("MINIO_BUCKET", "glue-analysis")
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".svs", ".ndpi"}
MIN_GAP_AREA_PX       = float(os.environ.get("MIN_GAP_AREA_PX", "20"))        # absolute floor in px²
POLY_SIMPLIFY_EPSILON = float(os.environ.get("POLY_SIMPLIFY_EPSILON", "0.001")) # fraction of arc length
GAP_PAYLOAD_LIMIT     = int(os.environ.get("GAP_PAYLOAD_LIMIT", "2000"))        # max gaps sent to frontend
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", "60"))
RETENTION_SECONDS = int(os.environ.get("RETENTION_SECONDS", "3600"))


def get_minio_client() -> Minio:
    return Minio(
        endpoint=f"{os.environ.get('MINIO_ENDPOINT', 'minio')}:{os.environ.get('MINIO_PORT', '9090')}",
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )


def dzi_exists(client: Minio, stem: str) -> bool:
    try:
        client.stat_object(BUCKET, f"tiles/{stem}/{stem}.dzi")
        return True
    except S3Error:
        return False


def result_exists(client: Minio, stem: str) -> bool:
    try:
        client.stat_object(BUCKET, f"results/{stem}.json")
        return True
    except S3Error:
        return False


def export_exists(client: Minio, key: str) -> bool:
    try:
        client.stat_object(BUCKET, key)
        return True
    except S3Error:
        return False


def upload_directory(client: Minio, local_dir: Path, minio_prefix: str):
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir)
        object_name = f"{minio_prefix}/{relative.as_posix()}"
        suffix = file_path.suffix.lower()
        content_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
        client.fput_object(BUCKET, object_name, str(file_path), content_type=content_type)


def process_image(client: Minio, object_name: str):
    stem = Path(object_name).stem
    ext = Path(object_name).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        return

    if dzi_exists(client, stem):
        logger.info("DZI already exists for %s, skipping", stem)
        return

    logger.info("Processing %s → DZI tiles", object_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_path = tmp_path / object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.fget_object(BUCKET, object_name, str(local_path))

        dzi_base = str(tmp_path / stem)
        image = pyvips.Image.new_from_file(str(local_path))
        image.dzsave(dzi_base, suffix=".jpeg", overlap=1, tile_size=256)

        dzi_file = tmp_path / f"{stem}.dzi"
        tiles_dir = tmp_path / f"{stem}_files"

        # Upload tile images BEFORE the DZI descriptor so the frontend never
        # sees the descriptor without its tiles (avoids a 404 race condition).
        upload_directory(client, tiles_dir, f"tiles/{stem}/{stem}_files")
        client.fput_object(
            BUCKET,
            f"tiles/{stem}/{stem}.dzi",
            str(dzi_file),
            content_type="application/xml",
        )

    logger.info("Done tiling %s", object_name)


def analyze_gaps(client: Minio, object_name: str):
    stem = Path(object_name).stem
    ext = Path(object_name).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        return None

    if result_exists(client, stem):
        # Schema migration: invalidate cached results that are stale —
        # missing 'coordinates' (pre-Phase 3) or missing 'version' (pre-v2,
        # generated with the coarse 0.015 epsilon).
        try:
            cached = fetch_result(client, stem)
            first_gap = (cached.get("gaps") or [{}])[0]
            needs_refresh = (
                "coordinates" not in first_gap
                or cached.get("version", 0) < 7
            )
            if needs_refresh:
                logger.info(
                    "Cached result for %s is stale — re-analysing", stem
                )
                client.remove_object(BUCKET, f"results/{stem}.json")
                # fall through to re-analyse
            else:
                logger.info("Gap analysis already exists for %s, skipping", stem)
                return None
        except Exception:
            logger.info("Could not validate cached result for %s — re-analysing", stem)
            # fall through to re-analyse

    logger.info("Analyzing gaps for %s — starting", object_name)
    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_path = tmp_path / object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.fget_object(BUCKET, object_name, str(local_path))

        img = cv2.imread(str(local_path))
        if img is None:
            logger.warning("cv2.imread returned None for %s, skipping", object_name)
            return None
        h, w = img.shape[:2]
        if w == 0 or h == 0:
            logger.warning("Zero-dimension image for %s, skipping", object_name)
            return None

        # Dynamic noise floor: scales with image area so that microscopic noise
        # contours are suppressed proportionally on high-resolution images.
        # Formula: 1 px² floor per 40,000 image pixels, but never below MIN_GAP_AREA_PX.
        # e.g. 4 000×4 000 → 400 px²;  2 000×2 000 → 100 px²;  1 000×1 000 → 50 px²
        min_area = max(MIN_GAP_AREA_PX, (w * h) / 40_000)

        # --- Illumination correction (CLAHE) + strict adaptive threshold ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Bilateral filter smooths noise while preserving sharp edges.
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        # Large-block adaptive threshold: blockSize large enough to prevent
        # hollowing; low C value to stay sensitive to faint thin veins.
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=201,
            C=5,
        )

        # Opening disconnects thin noise bridges that merge separate gaps
        # into one giant blob; closing heals small internal specks.
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        # CHAIN_APPROX_NONE retains every boundary pixel so approxPolyDP has
        # the full shape to work with before fine-grained simplification.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # --- Per-contour processing ---
        max_area = 0.25 * w * h  # 25% of image area — anything larger is a bleed
        gaps = []
        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))

            # Strict noise filter (dynamic threshold)
            if area_px < min_area:
                continue
            # Discard unrealistically large contours (false-positive bleeds)
            if area_px > max_area:
                continue

            # Fine-grained simplification — epsilon ≈ 0.2 % of arc length.
            # Produces high-fidelity polygons that hug the organic gap edges
            # closely, at the cost of more vertices per contour.
            perimeter = cv2.arcLength(cnt, True)
            epsilon = POLY_SIMPLIFY_EPSILON * perimeter if perimeter > 0 else 1.0
            simplified = cv2.approxPolyDP(cnt, epsilon, True)

            # Equivalent radius — useful for size-distribution statistics
            equiv_radius_px = math.sqrt(area_px / math.pi)

            # Centroid via image moments; fall back to bounding-box centre
            m = cv2.moments(cnt)
            if m["m00"] != 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx = bx + bw / 2.0
                cy = by + bh / 2.0

            # Flat normalized coordinate array [x1_norm, y1_norm, x2_norm, y2_norm, …]
            # Normalised to [0, 1] so the frontend scales to any viewport without
            # needing the original pixel dimensions; eliminates per-vertex JSON keys.
            pts = simplified.reshape(-1, 2)
            coordinates: list[float] = []
            for pt in pts:
                coordinates.append(round(float(pt[0]) / w, 4))
                coordinates.append(round(float(pt[1]) / h, 4))

            gaps.append({
                "area_px":         round(area_px, 1),
                "equiv_radius_px": round(equiv_radius_px, 2),
                "centroid_norm":   [round(cx / w, 4), round(cy / h, 4)],
                "coordinates":     coordinates,
            })

        total_gap_count = len(gaps)
        logger.info(
            "  %s: %d contours passed noise filter (min_area=%.1f px²)",
            stem, total_gap_count, min_area,
        )

        # --- Statistics over ALL valid gaps (computed before the payload cap) ---
        if gaps:
            radii = [g["equiv_radius_px"] for g in gaps]
            radius_stats: dict | None = {
                "min":    round(float(np.min(radii)),    2),
                "max":    round(float(np.max(radii)),    2),
                "mean":   round(float(np.mean(radii)),   2),
                "median": round(float(np.median(radii)), 2),
                "std":    round(float(np.std(radii)),    2),
            }
        else:
            radius_stats = None

        # --- Hard payload cap: largest gaps first, frontend receives at most GAP_PAYLOAD_LIMIT ---
        gaps.sort(key=lambda g: g["area_px"], reverse=True)
        payload_gaps = gaps[:GAP_PAYLOAD_LIMIT]

        result = {
            "version":      7,                  # bump when pipeline changes
            "stem":         stem,
            "image_size":   {"width": w, "height": h},
            "gap_count":    total_gap_count,   # all valid gaps, not capped
            "gaps":         payload_gaps,       # top GAP_PAYLOAD_LIMIT by area
            "radius_stats": radius_stats,       # stats over all valid gaps
        }

        data = json.dumps(result).encode("utf-8")
        client.put_object(
            BUCKET,
            f"results/{stem}.json",
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )

    elapsed = time.time() - t_start
    logger.info(
        "Gap analysis done for %s: %d total, %d returned (cap=%d), %.1fs elapsed",
        object_name, total_gap_count, len(payload_gaps), GAP_PAYLOAD_LIMIT, elapsed,
    )
    return result


def fetch_result(client: Minio, stem: str) -> dict:
    response = client.get_object(BUCKET, f"results/{stem}.json")
    try:
        return json.loads(response.read())
    finally:
        response.close()
        response.release_conn()


def generate_excel(client: Minio, stem: str):
    export_key = f"exports/{stem}.xlsx"
    if export_exists(client, export_key):
        logger.info("Excel already exists for %s, skipping", stem)
        return

    if not result_exists(client, stem):
        logger.info("No result yet for %s, skipping Excel", stem)
        return

    result = fetch_result(client, stem)
    gaps = result.get("gaps", [])
    image_size = result.get("image_size", {})
    w = image_size.get("width", 1)
    h = image_size.get("height", 1)
    radius_stats = result.get("radius_stats") or {}

    wb = openpyxl.Workbook()

    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_summary.append(["Field", "Value"])
    ws_summary.append(["Stem", result.get("stem", stem)])
    ws_summary.append(["Image Width (px)", w])
    ws_summary.append(["Image Height (px)", h])
    ws_summary.append(["Gap Count", result.get("gap_count", 0)])
    ws_summary.append(["Radius Min (px)", radius_stats.get("min", "")])
    ws_summary.append(["Radius Max (px)", radius_stats.get("max", "")])
    ws_summary.append(["Radius Mean (px)", radius_stats.get("mean", "")])
    ws_summary.append(["Radius Median (px)", radius_stats.get("median", "")])
    ws_summary.append(["Radius Std (px)", radius_stats.get("std", "")])

    ws_gaps = wb.create_sheet("Gaps")
    ws_gaps.append([
        "ID",
        "Area (px²)",
        "Equiv. Radius (px)",
        "Centroid X (norm)",
        "Centroid Y (norm)",
        "Centroid X (px)",
        "Centroid Y (px)",
    ])
    for i, gap in enumerate(gaps, start=1):
        cn = gap.get("centroid_norm", [0, 0])
        ws_gaps.append([
            i,
            gap.get("area_px", 0),
            gap.get("equiv_radius_px", 0),
            cn[0],
            cn[1],
            cn[0] * w,
            cn[1] * h,
        ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    data = buf.getvalue()
    client.put_object(
        BUCKET,
        export_key,
        io.BytesIO(data),
        length=len(data),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    logger.info("Excel export done for %s", stem)


def generate_annotated_image(client: Minio, object_name: str):
    stem = Path(object_name).stem
    ext = Path(object_name).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        return None

    export_key = f"exports/{stem}-annotated.jpg"
    if export_exists(client, export_key):
        logger.info("Annotated image already exists for %s, skipping", stem)
        return export_key

    logger.info("Generating annotated image for %s", object_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_path = tmp_path / object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.fget_object(BUCKET, object_name, str(local_path))

        img = cv2.imread(str(local_path))
        if img is None:
            logger.warning("cv2.imread returned None for %s", object_name)
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=201,
            C=5,
        )
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_ann, w_ann = img.shape[:2]
        min_area_ann = max(MIN_GAP_AREA_PX, (w_ann * h_ann) / 40_000)
        max_area_ann = 0.25 * w_ann * h_ann
        filtered = [cnt for cnt in contours
                    if min_area_ann <= cv2.contourArea(cnt) <= max_area_ann]
        cv2.drawContours(img, filtered, -1, (0, 0, 255), 2)

        lo, hi, best_buf = 1, 95, None
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, mid])
            if ok and len(buf) <= 15 * 1024 * 1024:
                best_buf = buf
                lo = mid + 1
            else:
                hi = mid - 1
        if best_buf is None:
            _, best_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 1])

        data = best_buf.tobytes()
        client.put_object(
            BUCKET,
            export_key,
            io.BytesIO(data),
            length=len(data),
            content_type="image/jpeg",
        )

    logger.info("Annotated image done for %s", object_name)
    return export_key


def delete_stem(client: Minio, original_object_name: str, stem: str):
    try:
        client.remove_object(BUCKET, original_object_name)
    except S3Error:
        pass

    for key in [
        f"results/{stem}.json",
        f"exports/{stem}.xlsx",
        f"exports/{stem}-annotated.jpg",
    ]:
        try:
            client.remove_object(BUCKET, key)
        except S3Error:
            pass

    tile_objs = list(client.list_objects(BUCKET, prefix=f"tiles/{stem}/", recursive=True))
    if tile_objs:
        errors = list(client.remove_objects(
            BUCKET,
            iter(DeleteObject(o.object_name) for o in tile_objs),
        ))
        for e in errors:
            logger.error("Tile deletion error: %s", e)
    logger.info("Cleanup complete for stem %s", stem)


async def cleanup_loop():
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    while True:
        try:
            if client.bucket_exists(BUCKET):
                now = datetime.now(timezone.utc)
                for obj in list(client.list_objects(BUCKET, recursive=False)):
                    name = obj.object_name
                    if name.startswith(("tiles/", "results/", "exports/")):
                        continue
                    age = (now - obj.last_modified).total_seconds()
                    if age >= RETENTION_SECONDS:
                        stem = Path(name).stem
                        logger.info("Expiring stem %s (age %.0fs)", stem, age)
                        await loop.run_in_executor(None, delete_stem, client, name, stem)
        except Exception:
            logger.exception("Error in cleanup loop")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


async def poll_loop():
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    while True:
        try:
            if client.bucket_exists(BUCKET):
                objects = list(client.list_objects(BUCKET, recursive=False))
                for obj in objects:
                    name = obj.object_name
                    if name.startswith(("tiles/", "results/", "exports/")):
                        continue
                    dzi_fut = loop.run_in_executor(None, process_image, client, name)
                    gap_fut = loop.run_in_executor(None, analyze_gaps, client, name)
                    try:
                        await dzi_fut
                    except Exception:
                        logger.exception("DZI error for %s", name)
                    try:
                        await gap_fut
                    except Exception:
                        logger.exception("Gap analysis error for %s", name)
                    excel_fut = loop.run_in_executor(None, generate_excel, client, Path(name).stem)
                    try:
                        await excel_fut
                    except Exception:
                        logger.exception("Excel export error for %s", name)
        except Exception:
            logger.exception("Error in poll loop sweep")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup():
    logger.info(
        "image-processor STARTED — MIN_GAP_AREA_PX=%.0f  GAP_PAYLOAD_LIMIT=%d  "
        "POLY_SIMPLIFY_EPSILON=%.3f  (coordinates field: ENABLED)",
        MIN_GAP_AREA_PX, GAP_PAYLOAD_LIMIT, POLY_SIMPLIFY_EPSILON,
    )
    asyncio.create_task(poll_loop())
    asyncio.create_task(cleanup_loop())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/files")
def list_files():
    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        return {"files": []}
    objects = client.list_objects(BUCKET, recursive=False)
    files = [
        obj.object_name for obj in objects
        if not obj.object_name.startswith(("tiles/", "results/", "exports/"))
    ]
    return {"files": files}


@app.get("/tiles")
def list_tiles():
    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        return {"tiles": []}
    objects = client.list_objects(BUCKET, prefix="tiles/", recursive=False)
    stems = []
    for obj in objects:
        name = obj.object_name.rstrip("/")
        stem = name[len("tiles/"):]
        if stem:
            stems.append(stem)
    return {"tiles": stems}


class AnalyzeRequest(BaseModel):
    key: str


class GapItem(BaseModel):
    area_px: float
    equiv_radius_px: float
    centroid_norm: list[float]
    coordinates: list[float]   # flat [x1_norm, y1_norm, x2_norm, y2_norm, …]


class RadiusStats(BaseModel):
    min: float
    max: float
    mean: float
    median: float
    std: float


class ImageSize(BaseModel):
    width: int
    height: int


class AnalysisResult(BaseModel):
    stem: str
    image_size: ImageSize
    gap_count: int
    gaps: list[GapItem]
    radius_stats: RadiusStats | None


@app.post("/analyze-gaps", response_model=AnalysisResult)
async def analyze_gaps_endpoint(body: AnalyzeRequest):
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, analyze_gaps, client, body.key)
        if result is None:
            stem = Path(body.key).stem
            try:
                return await loop.run_in_executor(None, fetch_result, client, stem)
            except S3Error:
                raise HTTPException(status_code=404, detail=f"No result found for {stem}")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /analyze-gaps for %s", body.key)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/results")
def list_results():
    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        return {"results": []}
    objects = client.list_objects(BUCKET, prefix="results/", recursive=False)
    stems = [
        Path(obj.object_name).stem for obj in objects
        if obj.object_name.endswith(".json")
    ]
    return {"results": stems}


class ExportRequest(BaseModel):
    key: str


@app.post("/exports/excel")
async def export_excel_endpoint(body: ExportRequest):
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    stem = Path(body.key).stem
    await loop.run_in_executor(None, generate_excel, client, stem)
    return {"key": f"exports/{stem}.xlsx"}


@app.post("/exports/image")
async def export_image_endpoint(body: ExportRequest):
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    key = await loop.run_in_executor(None, generate_annotated_image, client, body.key)
    if key is None:
        raise HTTPException(status_code=422, detail="Failed to generate annotated image")
    return {"key": key}


@app.get("/exports")
def list_exports():
    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        return {"exports": []}
    return {"exports": [o.object_name for o in
                        client.list_objects(BUCKET, prefix="exports/", recursive=True)]}
