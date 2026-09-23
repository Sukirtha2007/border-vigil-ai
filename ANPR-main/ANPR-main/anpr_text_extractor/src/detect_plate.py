"""
detect_plate.py  (updated)
Provides:
  - detect_plate(image)             — image mode (backward-compatible)
  - detect_plates_in_region(image, x1, y1, x2, y2) — runs LP detector on a sub-region,
                                     returns plate boxes in ORIGINAL image coordinates

Both functions use the same pre-loaded license_plate_detector.pt model.
The COCO YOLOv8n vehicle model is managed externally (video_pipeline.py).
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ── Thresholds ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # minimum LP detection confidence

# ── Model path ───────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE_DIR, 'models', 'license_plate_detector.pt')

# ── Singleton ─────────────────────────────────────────────────────────────────
_lp_model = None


def _load_lp_model() -> YOLO:
    global _lp_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"License plate model not found: {MODEL_PATH}\n"
            "Run `python download_model.py` from the project root."
        )
    if _lp_model is None:
        _lp_model = YOLO(MODEL_PATH)
    return _lp_model


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _is_valid_plate_box(x_min: int, y_min: int, x_max: int, y_max: int,
                         image_shape: tuple) -> bool:
    """
    Returns True only if the box looks like a license plate:
    - Aspect ratio between 1.5 and 8.0 (wide, short)
    - Box covers ≤ 60% of the input image area
    - Minimum size: at least 20×8 pixels
    """
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_h <= 0 or box_w <= 0:
        return False
    if box_w < 20 or box_h < 8:
        return False
    aspect = box_w / box_h
    if aspect < 1.5 or aspect > 8.0:
        return False
    h, w = image_shape[:2]
    area_fraction = (box_w * box_h) / (w * h)
    if area_fraction > 0.60:
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def detect_plates_in_region(image: np.ndarray,
                             vx1: int, vy1: int, vx2: int, vy2: int
                             ) -> list[dict]:
    """
    Runs the license plate detector on a vehicle sub-region of `image`.

    Args:
        image:  Full original frame/image (BGR NumPy array)
        vx1, vy1, vx2, vy2: Vehicle bounding box in original image coordinates

    Returns:
        List of dicts (sorted by confidence descending):
            {
                'x1', 'y1', 'x2', 'y2': int  — in ORIGINAL image coordinates
                'conf': float,
                'crop': np.ndarray            — cropped plate from original image
            }
        Empty list if no valid plates found.
    """
    model = _load_lp_model()

    # Clip vehicle region to image bounds
    h_img, w_img = image.shape[:2]
    vx1 = max(0, vx1); vy1 = max(0, vy1)
    vx2 = min(w_img, vx2); vy2 = min(h_img, vy2)

    if vx2 <= vx1 or vy2 <= vy1:
        return []

    vehicle_roi = image[vy1:vy2, vx1:vx2]
    if vehicle_roi.size == 0:
        return []

    try:
        results = model.predict(vehicle_roi, conf=CONFIDENCE_THRESHOLD,
                                device='cpu', verbose=False)
    except Exception as e:
        print(f"[detect_plate] LP model predict error: {e}")
        return []

    plates = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].cpu().numpy())
            # ROI-relative coordinates
            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0].cpu().numpy())

            # Convert back to ORIGINAL image coordinates
            ox1 = vx1 + rx1; oy1 = vy1 + ry1
            ox2 = vx1 + rx2; oy2 = vy1 + ry2

            # Clip to image bounds
            ox1 = max(0, ox1); oy1 = max(0, oy1)
            ox2 = min(w_img, ox2); oy2 = min(h_img, oy2)

            # Validate geometry (ensures ONLY the plate, not the whole car)
            if not _is_valid_plate_box(ox1, oy1, ox2, oy2, image.shape):
                continue

            crop = image[oy1:oy2, ox1:ox2]
            if crop.size == 0:
                continue

            plates.append({
                'x1': ox1, 'y1': oy1, 'x2': ox2, 'y2': oy2,
                'conf': conf,
                'crop': crop,
            })

    # Sort by confidence descending; return best first
    plates.sort(key=lambda p: p['conf'], reverse=True)
    return plates


def detect_plate(image: np.ndarray) -> tuple[list, np.ndarray]:
    """
    Backward-compatible image-mode API.
    Runs LP detector on the full image (no vehicle pre-detection step).

    Args:
        image: BGR NumPy array

    Returns:
        (cropped_plates, annotated_image)
          cropped_plates: list of np.ndarray (plate crops)
          annotated_image: copy of image with plate bounding boxes drawn
    """
    model = _load_lp_model()
    h_img, w_img = image.shape[:2]

    try:
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD,
                                device='cpu', verbose=False)
    except Exception as e:
        print(f"[detect_plate] model predict error: {e}")
        return [], image.copy()

    cropped_plates = []
    boxed_image = image.copy()
    rejected = 0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)

            if not _is_valid_plate_box(x1, y1, x2, y2, image.shape):
                rejected += 1
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cropped_plates.append(crop)

            cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(boxed_image, f"Plate {conf:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if rejected:
        print(f"[detect_plate] Rejected {rejected} non-plate detection(s).")

    return cropped_plates, boxed_image
