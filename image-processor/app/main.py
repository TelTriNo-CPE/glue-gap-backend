from __future__ import annotations

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

# Raise pixel limits BEFORE importing image libraries so that
# high-resolution images (400 MB+) are not rejected as "decompression bombs".
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2  # noqa: E402
import PIL.Image  # noqa: E402
PIL.Image.MAX_IMAGE_PIXELS = None
import numpy as np
import openpyxl
import pyvips

# Restrict pyvips memory behaviour BEFORE any image operations.
#
# By default pyvips caches every completed operation tile so it can reuse
# results if the same pipeline is evaluated again.  On a 1 GB TIFF this
# cache can silently retain the entire decompressed raster in RAM *after*
# write_to_memory() returns, because pyvips does not know we will never
# re-run the same pipeline.  Setting the limits to 0 disables caching
# completely so memory is freed as soon as each strip is written out.
#
# concurrency_set(1) forces a single worker thread.  With the default
# thread pool pyvips processes N strips simultaneously (one per thread),
# each backed by a slice of the full-resolution source.  On a large image
# this multiplies peak RAM by the thread count.  Single-threaded keeps
# exactly one strip in flight at any moment.
pyvips.cache_set_max(0)        # zero cached operation entries
pyvips.cache_set_max_mem(0)    # zero bytes of cached pixel data
pyvips.cache_set_max_files(0)  # zero cached open file handles
pyvips.concurrency_set(1)      # one worker thread → one strip in RAM at a time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
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
# Maximum pixel dimension (width or height) used for gap-analysis.
# pyvips downscales the image to this size in a streaming, tile-based pass
# so the peak working-set RAM is bounded to ~few_strips + output raster
# regardless of the source file size (a 1 GB TIFF stays under ~200 MB).
MAX_ANALYSIS_DIM = int(os.environ.get("MAX_ANALYSIS_DIM", "4096"))


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


def _load_for_analysis(
    local_path: Path,
    bbox: "BBoxModel | None" = None,
) -> tuple[np.ndarray, int, int, int, int, float]:
    """Load an image for gap analysis via pyvips — memory-safe for files of any size.

    pyvips evaluates lazily in tile-strips, so only a narrow band of the
    source image is resident at any time.  For a 1 GB TIFF the peak working
    set is roughly (one_strip × full_width × bands) + output_raster ≈ 50 MB,
    compared with ~3-4 GB if the whole file were decoded with cv2.imread.

    The returned BGR array has at most MAX_ANALYSIS_DIM pixels on its longest
    side.  All pixel-space quantities (areas, radii, coordinates) must be
    scaled back to full-image space by the caller using the returned *scale*.

    Parameters
    ----------
    local_path : path to the image file on disk (already downloaded from MinIO)
    bbox       : optional crop region in full-image pixel coordinates

    Returns
    -------
    bgr      : uint8 BGR numpy array ready for OpenCV
    orig_w   : original image width  (pixels)
    orig_h   : original image height (pixels)
    bbox_x   : left edge of the crop in full-image pixels (0 when bbox is None)
    bbox_y   : top  edge of the crop in full-image pixels (0 when bbox is None)
    scale    : analysis_pixels / original_pixels  (≤ 1.0).
               Multiply analysis-space pixel coords by (1 / scale) to recover
               full-image pixel coordinates before normalising.
    """
    img_v = pyvips.Image.new_from_file(str(local_path))

    # Flatten any alpha channel onto a white background so we always have
    # opaque pixels; OpenCV does not understand transparency.
    if img_v.hasalpha():
        img_v = img_v.flatten(background=255)

    # Normalise to sRGB so we always get a 3-band uint8 image.
    # This handles greyscale TIFFs, 16-bit images, CMYK scans, etc.
    if img_v.interpretation != "srgb":
        img_v = img_v.colourspace("srgb")

    # Ensure exactly 3 bands after colour-space conversion.
    if img_v.bands > 3:
        img_v = img_v[:3]
    elif img_v.bands < 3:
        # Replicate single band to fill R/G/B
        img_v = img_v.bandjoin([img_v, img_v])[:3]

    # Ensure 8-bit unsigned output (pyvips keeps high-bit-depth formats as
    # uint16 even after colourspace(); we need uint8 for OpenCV).
    if img_v.format != "uchar":
        img_v = img_v.cast("uchar")

    orig_w, orig_h = img_v.width, img_v.height
    bbox_x, bbox_y = 0, 0

    # Crop to the requested bounding box *before* downscaling so we only
    # work on pixels inside the region of interest.
    if bbox is not None:
        x1 = max(0, int(bbox.x))
        y1 = max(0, int(bbox.y))
        x2 = min(orig_w, x1 + max(1, int(bbox.width)))
        y2 = min(orig_h, y1 + max(1, int(bbox.height)))
        img_v = img_v.crop(x1, y1, x2 - x1, y2 - y1)
        bbox_x, bbox_y = x1, y1

    # Compute a uniform downscale factor so the longer edge ≤ MAX_ANALYSIS_DIM.
    roi_w, roi_h = img_v.width, img_v.height
    scale = 1.0
    if max(roi_w, roi_h) > MAX_ANALYSIS_DIM:
        scale = MAX_ANALYSIS_DIM / max(roi_w, roi_h)
        img_v = img_v.resize(scale)

    # Materialise only the (small) analysis-resolution raster.
    # write_to_memory() triggers the lazy pyvips pipeline: it processes the
    # source in strips (one at a time with concurrency=1), so peak RAM during
    # evaluation ≈ one_source_strip + output_buffer.
    #
    # Buffer ownership must be explicit:
    #   raw_buf  — the bytes object returned by pyvips (output-size only)
    #   np_rgb   — a zero-copy numpy *view* into raw_buf (no extra allocation)
    #   bgr      — one owned numpy copy with channels reversed
    # We then del both np_rgb and raw_buf so the pyvips allocation is freed
    # before we hand the array to OpenCV.  The old pattern of
    #   np.ndarray(buffer=write_to_memory(), ...).copy()
    # kept raw_buf alive until the GC collected the anonymous ndarray, meaning
    # write_to_memory_result + np_rgb_copy + bgr_copy were all resident at once
    # (3× the output size).  The explicit del below keeps peak at 2×.
    out_h, out_w = img_v.height, img_v.width
    raw_buf = img_v.write_to_memory()
    np_rgb  = np.frombuffer(raw_buf, dtype=np.uint8).reshape(out_h, out_w, 3)
    bgr     = np_rgb[:, :, ::-1].copy()  # one owned allocation; channels R↔B swapped
    del np_rgb, raw_buf                  # release pyvips buffer before OpenCV pipeline

    logger.info(
        "_load_for_analysis: orig=%dx%d  roi=%dx%d  out=%dx%d  scale=%.4f",
        orig_w, orig_h, roi_w, roi_h, out_w, out_h, scale,
    )
    return bgr, orig_w, orig_h, bbox_x, bbox_y, scale


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


def analyze_gaps(client: Minio, object_name: str,
                  sensitivity: float = 50.0, min_area_param: float = 20.0,
                  bbox: BBoxModel | None = None):
    stem = Path(object_name).stem
    ext = Path(object_name).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        return None

    logger.info("Analyzing gaps for %s — starting", object_name)
    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_path = tmp_path / object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.fget_object(BUCKET, object_name, str(local_path))

        # Memory-safe load: pyvips streams the source in tile-strips and
        # downscales to MAX_ANALYSIS_DIM before handing a small numpy array
        # to OpenCV.  Peak RAM stays under ~200 MB even for 1 GB source files,
        # compared with 3-4 GB if cv2.imread decoded the full raster.
        # bbox cropping and downscaling are both handled inside the helper.
        try:
            img, full_w, full_h, bbox_x, bbox_y, scale = _load_for_analysis(
                local_path, bbox=bbox
            )
        except Exception as exc:
            logger.warning("pyvips load failed for %s (%s), skipping", object_name, exc)
            return None

        h, w = img.shape[:2]
        if w == 0 or h == 0:
            logger.warning("Zero-dimension analysis image for %s, skipping", object_name)
            return None

        # Guard against a degenerate bbox that collapsed to zero after clamping.
        if bbox is not None and (full_w == 0 or full_h == 0):
            return {
                "version": 10,
                "stem": stem,
                "image_size": {"width": full_w, "height": full_h},
                "gap_count": 0,
                "gaps": [],
                "radius_stats": None,
                "sensitivity": sensitivity,
                "min_area": min_area_param,
            }

        # --- Red scale-bar exclusion ---
        # Red text/bar becomes dark in grayscale and triggers false-positive gaps.
        # Detect red pixels in HSV (hue wraps: 0-10 and 160-180) and mark them
        # as background so they are invisible to the thresholding step.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_lo1 = np.array([0, 70, 50], dtype=np.uint8)
        red_hi1 = np.array([10, 255, 255], dtype=np.uint8)
        red_lo2 = np.array([160, 70, 50], dtype=np.uint8)
        red_hi2 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lo1, red_hi1),
            cv2.inRange(hsv, red_lo2, red_hi2),
        )
        # Dilate slightly to cover anti-aliased edges of the red text/bar
        red_mask = cv2.dilate(
            red_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )

        # Dynamic noise floor in *analysis-image* pixel units.
        # min_area_param is expressed in full-image pixels², so we scale it
        # down by scale² to stay consistent with the downscaled raster.
        # The dynamic floor (w*h)/40_000 is already in analysis pixels.
        min_area = max(min_area_param * (scale ** 2), (w * h) / 40_000)

        # --- Background-division illumination normalization ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Fast background estimation: downscale → large morphological close
        # → upscale.  The close fills in dark gaps so the result is a smooth
        # approximation of the rock's illumination field only.
        small_gray = cv2.resize(gray, None, fx=0.1, fy=0.1,
                                interpolation=cv2.INTER_AREA)
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        small_bg = cv2.morphologyEx(small_gray, cv2.MORPH_CLOSE, bg_kernel)
        bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)

        # Divide original by background: rock becomes ~255 (white), gaps
        # stay distinctly dark — lighting variations are erased.
        norm = cv2.divide(gray, bg, scale=255)

        # Neutralise red scale-bar pixels so they read as background (white).
        norm[red_mask > 0] = 255

        # Get Otsu's optimal threshold without applying it, then adjust
        # based on the sensitivity parameter (50 = baseline Otsu).
        # Higher sensitivity → higher threshold → catches lighter (subtler) gaps.
        # Lower sensitivity → lower threshold → only very dark gaps detected.
        T, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        offset = (sensitivity - 50) * 1.5
        adjusted_T = max(0, min(255, T + offset))
        _, thresh = cv2.threshold(norm, adjusted_T, 255, cv2.THRESH_BINARY_INV)

        # Closing fuses porous regions into solid blobs; opening removes
        # single-pixel dust.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)

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

            # Project analysis-space pixel coords back to full-image space:
            #   1. multiply by (1/scale) to undo the downscale
            #   2. add the bbox offset (0 when no bbox was provided)
            #   3. divide by full image dimensions to normalise to [0, 1]
            inv_scale = 1.0 / scale
            pts = simplified.reshape(-1, 2)
            coordinates: list[float] = []
            for pt in pts:
                coordinates.append(round((float(pt[0]) * inv_scale + bbox_x) / full_w, 4))
                coordinates.append(round((float(pt[1]) * inv_scale + bbox_y) / full_h, 4))

            gaps.append({
                # Scale area and radius back to full-image pixel units so the
                # reported values match what users see at the original resolution.
                "area_px":         round(area_px * (inv_scale ** 2), 1),
                "equiv_radius_px": round(equiv_radius_px * inv_scale, 2),
                "centroid_norm":   [round((cx * inv_scale + bbox_x) / full_w, 4),
                                    round((cy * inv_scale + bbox_y) / full_h, 4)],
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
            "version":      10,                 # bump when pipeline changes
            "stem":         stem,
            "image_size":   {"width": full_w, "height": full_h},
            "gap_count":    total_gap_count,   # all valid gaps, not capped
            "gaps":         payload_gaps,       # top GAP_PAYLOAD_LIMIT by area
            "radius_stats": radius_stats,       # stats over all valid gaps
            "sensitivity":  sensitivity,
            "min_area":     min_area_param,
        }

        # Bbox-based (partial) analysis: never persist — the result covers only
        # a sub-region and should not overwrite the full-image baseline.
        if bbox is not None:
            return result

        # Ensure we don't wipe existing saved manual gaps if this is a re-run.
        # The frontend will handle merging the auto-detected gaps with the existing ones.
        # But we must persist a baseline if one doesn't exist so the frontend
        # can PUT /results/:stem/gaps later (which expects the file to exist).
        try:
            fetch_result(client, stem)
            # Result exists, we do NOT overwrite it.
        except Exception:
            # Result doesn't exist, safe to save this as the baseline.
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
        "Gap analysis done for %s: %d total, %d returned (cap=%d), "
        "sensitivity=%.0f, min_area=%.0f, %.1fs elapsed",
        object_name, total_gap_count, len(payload_gaps), GAP_PAYLOAD_LIMIT,
        sensitivity, min_area_param, elapsed,
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

        # Use the pyvips streaming loader — never decompresses the full raster.
        # The annotated output is JPEG-capped at 15 MB so working at analysis
        # resolution (≤ MAX_ANALYSIS_DIM) is sufficient and much safer.
        try:
            img, _full_w, _full_h, _bx, _by, scale_ann = _load_for_analysis(local_path)
        except Exception as exc:
            logger.warning("pyvips load failed for annotated image %s (%s)", object_name, exc)
            return None

        h_img, w_img = img.shape[:2]

        # Red scale-bar exclusion (same logic as analyze_gaps)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 70, 50], dtype=np.uint8),
                             np.array([10, 255, 255], dtype=np.uint8)),
            cv2.inRange(hsv, np.array([160, 70, 50], dtype=np.uint8),
                             np.array([180, 255, 255], dtype=np.uint8)),
        )
        red_mask = cv2.dilate(
            red_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        small_gray = cv2.resize(gray, None, fx=0.1, fy=0.1,
                                interpolation=cv2.INTER_AREA)
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        small_bg = cv2.morphologyEx(small_gray, cv2.MORPH_CLOSE, bg_kernel)
        bg = cv2.resize(small_bg, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        norm = cv2.divide(gray, bg, scale=255)
        norm[red_mask > 0] = 255
        _, thresh = cv2.threshold(
            norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Scale the absolute floor to analysis-pixel units (same logic as analyze_gaps).
        min_area_ann = max(MIN_GAP_AREA_PX * (scale_ann ** 2), (w_img * h_img) / 40_000)
        max_area_ann = 0.25 * w_img * h_img
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
                    stem = Path(name).stem
                    skip_dzi = dzi_exists(client, stem)
                    skip_gap = result_exists(client, stem)
                    if skip_dzi and skip_gap:
                        continue
                    if not skip_dzi:
                        dzi_fut = loop.run_in_executor(None, process_image, client, name)
                        try:
                            await dzi_fut
                        except Exception:
                            logger.exception("DZI error for %s", name)
                    if not skip_gap:
                        gap_fut = loop.run_in_executor(None, analyze_gaps, client, name)
                        try:
                            await gap_fut
                        except Exception:
                            logger.exception("Gap analysis error for %s", name)
                    excel_fut = loop.run_in_executor(None, generate_excel, client, stem)
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


@app.api_route("/tiles/{filename:path}", methods=["GET", "HEAD"])
def serve_tile(filename: str, request: Request):
    """Stream DZI descriptor or tile images from MinIO.

    Frontend requests:
      HEAD/GET /tiles/{stem}.dzi          → tiles/{stem}/{stem}.dzi in MinIO
      HEAD/GET /tiles/{stem}_files/{rest} → tiles/{stem}/{stem}_files/{rest} in MinIO
    """
    if filename.endswith(".dzi"):
        stem = filename[:-4]
        minio_key = f"tiles/{stem}/{stem}.dzi"
        content_type = "application/xml"
    elif "_files/" in filename:
        stem = filename.split("_files/")[0]
        minio_key = f"tiles/{stem}/{filename}"
        content_type = "image/jpeg"
    else:
        raise HTTPException(status_code=404, detail=f"Unknown tile path: {filename}")

    client = get_minio_client()

    if request.method == "HEAD":
        try:
            stat = client.stat_object(BUCKET, minio_key)
        except S3Error:
            raise HTTPException(status_code=404, detail=f"Tile not found: {minio_key}")
        return Response(
            status_code=200,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(stat.size),
                "Cache-Control": "public, max-age=86400",
            },
        )

    try:
        minio_response = client.get_object(BUCKET, minio_key)
    except S3Error:
        raise HTTPException(status_code=404, detail=f"Tile not found: {minio_key}")

    def _stream():
        try:
            for chunk in minio_response.stream(amt=65536):
                yield chunk
        finally:
            minio_response.close()
            minio_response.release_conn()

    return StreamingResponse(
        _stream(),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


class BBoxModel(BaseModel):
    x: int
    y: int
    width: int
    height: int


class AnalyzeRequest(BaseModel):
    key: str
    sensitivity: float = 50.0
    min_area: float = 20.0
    bbox: BBoxModel | None = None


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
    sensitivity: float | None = None
    min_area: float | None = None


@app.post("/analyze-gaps", response_model=AnalysisResult)
async def analyze_gaps_endpoint(body: AnalyzeRequest):
    client = get_minio_client()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, analyze_gaps, client, body.key,
            body.sensitivity, body.min_area, body.bbox,
        )
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
