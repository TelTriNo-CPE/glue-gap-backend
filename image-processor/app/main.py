import asyncio
import io
import json
import logging
import math
import os
import tempfile
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
MIN_GAP_AREA_PX = float(os.environ.get("MIN_GAP_AREA_PX", "10"))
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

        client.fput_object(
            BUCKET,
            f"tiles/{stem}/{stem}.dzi",
            str(dzi_file),
            content_type="application/xml",
        )
        upload_directory(client, tiles_dir, f"tiles/{stem}/{stem}_files")

    logger.info("Done tiling %s", object_name)


def analyze_gaps(client: Minio, object_name: str):
    stem = Path(object_name).stem
    ext = Path(object_name).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        return None

    if result_exists(client, stem):
        logger.info("Gap analysis already exists for %s, skipping", stem)
        return None

    logger.info("Analyzing gaps for %s", object_name)

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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gaps = []
        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            if area_px < MIN_GAP_AREA_PX:
                continue
            equiv_radius_px = math.sqrt(area_px / math.pi)
            m = cv2.moments(cnt)
            if m["m00"] != 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
            else:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = x + bw / 2
                cy = y + bh / 2
            gaps.append({
                "area_px": area_px,
                "equiv_radius_px": equiv_radius_px,
                "centroid_norm": [cx / w, cy / h],
            })

        radii = [g["equiv_radius_px"] for g in gaps]
        radius_stats = {
            "min": float(np.min(radii)),
            "max": float(np.max(radii)),
            "mean": float(np.mean(radii)),
            "median": float(np.median(radii)),
            "std": float(np.std(radii)),
        } if radii else None

        result = {
            "stem": stem,
            "image_size": {"width": w, "height": h},
            "gap_count": len(gaps),
            "gaps": gaps,
            "radius_stats": radius_stats,
        }

        data = json.dumps(result).encode("utf-8")
        client.put_object(
            BUCKET,
            f"results/{stem}.json",
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )

    logger.info("Gap analysis done for %s: %d gaps found", object_name, len(gaps))
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
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_GAP_AREA_PX]
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


@app.post("/analyze-gaps")
async def analyze_gaps_endpoint(body: AnalyzeRequest):
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, analyze_gaps, client, body.key)
    if result is None:
        stem = Path(body.key).stem
        try:
            return await loop.run_in_executor(None, fetch_result, client, stem)
        except S3Error:
            raise HTTPException(status_code=404, detail=f"No result found for {stem}")
    return result


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
