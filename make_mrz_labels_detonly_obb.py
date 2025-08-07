#!/usr/bin/env python3
"""
make_mrz_labels_detonly_obb.py

Detect MRZ-like regions using PaddleOCR's detection-only mode 
and save YOLO-OBB labels for up to 3 largest boxes, allowing
a configurable small amount of overlap.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Any

import cv2
from paddleocr import PaddleOCR
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)
log = logging.getLogger("make_mrz_labels_obb")

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Point = Tuple[float, float]
Quad  = List[Point]

# -----------------------------------------------------------------------------
# OCR engine
# -----------------------------------------------------------------------------
def ocr_engine_from_args(args: argparse.Namespace) -> PaddleOCR:
    """Initialize PaddleOCR in detection-only mode."""
    return PaddleOCR(
        det_model_dir=str(args.det_model_dir),
        lang="en"
    )

# -----------------------------------------------------------------------------
# IoU calculation
# -----------------------------------------------------------------------------
def iou_aabb(box_a: Quad, box_b: Quad) -> float:
    """Compute IoU of two quadrilaterals via axis-aligned bboxes."""
    xa = [p[0] for p in box_a]
    ya = [p[1] for p in box_a]
    xb = [p[0] for p in box_b]
    yb = [p[1] for p in box_b]

    xA = max(min(xa), min(xb))
    yA = max(min(ya), min(yb))
    xB = min(max(xa), max(xb))
    yB = min(max(ya), max(yb))

    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (max(xa) - min(xa)) * (max(ya) - min(ya))
    area_b = (max(xb) - min(xb)) * (max(yb) - min(yb))
    union  = area_a + area_b - inter

    return inter / union if union else 0.0

# -----------------------------------------------------------------------------
# MRZ box detection + filtering
# -----------------------------------------------------------------------------
def find_mrz_boxes_by_detection(
    img_path: Path,
    ocr: PaddleOCR,
    max_iou: float,
    debug: bool = False
) -> List[Quad]:
    """
    1) Run detection-only
    2) Keep quads in bottom 40% of image
    3) Sort by area, pick up to 3 boxes with IoU < max_iou
    """
    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("❌ Failed to read image: %s", img_path.name)
        return []

    h, w = img.shape[:2]
    result = ocr.predict(str(img_path))  # rec=False, cls=False => detection-only

    if not result:
        log.info("⏭️  No text boxes detected in %s", img_path.name)
        return []

    # normalize nested vs flat return
    lines = result[0] if isinstance(result[0], list) else result

    quads: List[Tuple[float, Quad]] = []
    for entry in lines:
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 1):
            continue
        quad = entry[0]
        if not (isinstance(quad, list) and len(quad) == 4):
            continue

        # bottom-40% filter
        y_coords = [pt[1] for pt in quad]
        if min(y_coords) < 0.60 * h:
            continue

        # compute area
        x_coords = [pt[0] for pt in quad]
        area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        quads.append((area, quad))

    if not quads:
        log.info("⏭️  No quads passed bottom-40%% filter in %s", img_path.name)
        return []

    # pick top-3 by area, ensuring IoU < max_iou
    selected: List[Quad] = []
    for area, quad in sorted(quads, key=lambda t: -t[0]):
        if len(selected) == 3:
            break
        if all(iou_aabb(quad, other) < max_iou for other in selected):
            selected.append(quad)

    if not selected:
        log.info("⏭️  All quads overlapped > %.2f in %s", max_iou, img_path.name)
    elif debug:
        _draw_debug(img_path, selected)

    return selected

# -----------------------------------------------------------------------------
# Debug drawing
# -----------------------------------------------------------------------------
def _draw_debug(img_path: Path, boxes: List[Quad]):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for idx, quad in enumerate(boxes):
        pts = [(int(x), int(y)) for x, y in quad]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i+1)%4], (0,255,0), 2)
        cv2.putText(img, str(idx), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    dbg_path = img_path.with_suffix(".mrz_debug.jpg")
    cv2.imwrite(str(dbg_path), img)
    log.info("🖼️  Debug saved: %s", dbg_path.name)

# -----------------------------------------------------------------------------
# Save YOLO-OBB labels
# -----------------------------------------------------------------------------
def save_yolo_labels(
    img_path: Path,
    boxes: List[Quad],
    out_dir: Path
):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # sort top-to-bottom for consistent ordering
    boxes = sorted(boxes, key=lambda b: min(pt[1] for pt in b))
    lines: List[str] = []
    for quad in boxes:
        normalized = [(x/w, y/h) for x,y in quad]
        flat = " ".join(f"{x:.6f} {y:.6f}" for x,y in normalized)
        lines.append(f"0 {flat}")

    out_dir.mkdir(parents=True, exist_ok=True)
    label_file = out_dir / f"{img_path.stem}.txt"
    label_file.write_text("\n".join(lines))
    log.info("📝 Label saved: %s", label_file.name)

# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create YOLO-OBB labels for MRZ lines (detection only)"
    )
    parser.add_argument("--img_dir",       type=Path,   required=True, help="Folder of images")
    parser.add_argument("--out_dir",       type=Path,   required=True, help="Where to save .txt labels")
    parser.add_argument("--det_model_dir", type=Path,   required=True, help="PaddleOCR det model dir")
    parser.add_argument("--max_iou",       type=float, default=0.30, help="Max IoU allowed between selected boxes")
    parser.add_argument("--debug",         action="store_true",      help="Save debug jpgs")
    args = parser.parse_args()

    # Create the output folder unconditionally
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("📁 Output directory: %s", args.out_dir.resolve())

    ocr = ocr_engine_from_args(args)

    img_paths = sorted(
        p for p in args.img_dir.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    )
    if not img_paths:
        log.error("No images found in %s", args.img_dir)
        sys.exit(1)

    written = 0
    log.info("Starting on %d images…", len(img_paths))
    for img_path in tqdm(img_paths, unit="img"):
        boxes = find_mrz_boxes_by_detection(img_path, ocr, args.max_iou, debug=args.debug)
        if boxes:
            save_yolo_labels(img_path, boxes, args.out_dir)
            written += 1
        else:
            log.info("⏭️  Skipped (no valid boxes): %s", img_path.name)

    log.info("✅ Done. %d labels written to %s", written, args.out_dir)


if __name__ == "__main__":
    main()
