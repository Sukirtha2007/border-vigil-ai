"""
video_pipeline.py
Full video ANPR pipeline:
  VIDEO → frame-by-frame → YOLOv8n vehicle tracking → LP detection per vehicle →
  plate crop → OCR → aggregate per track ID → save best crops → JSON + annotated video

Models are loaded ONCE at the top of process_video() and reused across all frames.
"""
import os
import sys
import json
import glob
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_plate import detect_plates_in_region, _load_lp_model
from ocr_reader import extract_text, clean_plate_text

# ── Vehicle classes from COCO yolov8n ─────────────────────────────────────────
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEHICLE_MODEL_PATH = os.path.join(_BASE_DIR, 'models', 'yolov8n.pt')


def _blur_score(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _best_ocr_result(observations: list[dict]) -> dict:
    """
    Choose the most reliable OCR result / crop from a list of per-frame observations.
    Each observation: {text, confidence, crop, blur, lp_conf}
    """
    if not observations:
        return {'plate_number': None, 'confidence': 0.0, 'crop': None}

    # Filter observations that yielded OCR text
    text_obs = [obs for obs in observations if obs.get('text')]

    if text_obs:
        # Group by text
        text_groups: dict[str, list] = defaultdict(list)
        for obs in text_obs:
            text_groups[obs['text']].append(obs)

        best_text = None
        best_score = -1.0
        best_crop = None
        best_conf = 0.0

        for text, group in text_groups.items():
            frequency = len(group)
            blurs = [g['blur'] for g in group]
            max_blur = max(blurs) if max(blurs) > 0 else 1.0
            for obs in group:
                blur_norm = obs['blur'] / max_blur
                combined = obs['confidence'] * (0.5 + 0.3 * blur_norm + 0.2 * min(frequency / 5, 1.0))
                if combined > best_score:
                    best_score = combined
                    best_text = text
                    best_conf = obs['confidence']
                    best_crop = obs['crop']

        return {
            'plate_number': best_text,
            'confidence': round(best_conf, 4),
            'crop': best_crop,
        }

    # If license plate crops exist but OCR returned no text, return the sharpest crop
    best_obs = max(observations, key=lambda obs: (obs.get('lp_conf', 0.0), obs.get('blur', 0.0)))
    return {
        'plate_number': None,
        'confidence': 0.0,
        'crop': best_obs['crop'],
    }


def process_video(video_path: str) -> None:
    """
    Run the full ANPR pipeline on a video file.

    Args:
        video_path: Path to the input video file.
    """
    base_dir = _BASE_DIR
    plate_output_dir = os.path.join(base_dir, 'plate_output')
    output_dir       = os.path.join(base_dir, 'output')
    os.makedirs(plate_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[Video] ERROR: Video not found: {video_path}")
        return

    # ── Load models ONCE ────────────────────────────────────────────────────
    print("[Video] Loading vehicle detection model (yolov8n.pt)...")
    if not os.path.exists(VEHICLE_MODEL_PATH):
        print(f"[Video] ERROR: Vehicle model not found: {VEHICLE_MODEL_PATH}")
        return
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    print("[Video] Loading license plate detection model...")
    _load_lp_model()   # warms up singleton in detect_plate.py

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video] ERROR: Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}×{height}  {fps:.1f} FPS  {total} frames")

    # ── Video writer ────────────────────────────────────────────────────────
    annotated_path = os.path.join(output_dir, 'annotated_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    # ── Per-track data store ─────────────────────────────────────────────────
    # track_id → {'vehicle_class': str, 'observations': list[dict]}
    track_data: dict[int, dict] = {}

    frame_idx = 0
    print("[Video] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()

        # ── Vehicle detection + tracking ─────────────────────────────────────
        try:
            track_results = vehicle_model.track(
                frame,
                persist=True,
                conf=0.4,
                classes=list(VEHICLE_CLASSES.keys()),
                device='cpu',
                verbose=False,
                tracker='bytetrack.yaml',
            )
        except Exception as e:
            print(f"[Video] Frame {frame_idx}: vehicle track error: {e}")
            writer.write(annotated)
            continue

        if not track_results or track_results[0].boxes is None:
            writer.write(annotated)
            if frame_idx % 50 == 0:
                print(f"[Video] Frame {frame_idx}/{total} — no vehicles")
            continue

        result = track_results[0]
        boxes  = result.boxes

        for box in boxes:
            # Track ID (may be None if tracker hasn't assigned one yet)
            if box.id is None:
                continue
            track_id   = int(box.id[0].cpu().numpy())
            class_id   = int(box.cls[0].cpu().numpy())
            veh_conf   = float(box.conf[0].cpu().numpy())
            veh_class  = VEHICLE_CLASSES.get(class_id, 'vehicle')
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].cpu().numpy())

            # Initialise track entry
            if track_id not in track_data:
                track_data[track_id] = {
                    'vehicle_class': veh_class,
                    'observations': [],
                }

            # ── License plate detection in vehicle ROI ────────────────────
            plates = detect_plates_in_region(frame, vx1, vy1, vx2, vy2)

            best_plate = plates[0] if plates else None
            plate_text = None
            plate_conf = 0.0

            if best_plate is not None:
                crop = best_plate['crop']
                blur = _blur_score(crop)

                # Run OCR
                plate_text, plate_conf = extract_text(crop)

                track_data[track_id]['observations'].append({
                    'text': plate_text,
                    'confidence': plate_conf if plate_text else 0.0,
                    'crop': crop.copy(),
                    'blur': blur,
                    'lp_conf': best_plate.get('conf', 0.0),
                })

                # Draw plate bounding box (green)
                px1, py1, px2, py2 = (best_plate['x1'], best_plate['y1'],
                                      best_plate['x2'], best_plate['y2'])
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                plate_label = plate_text if plate_text else "Plate"
                cv2.putText(annotated, plate_label,
                            (px1, max(py1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Draw vehicle bounding box (blue)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 100, 0), 2)
            veh_label = f"Vehicle {track_id} ({veh_class})"
            cv2.putText(annotated, veh_label,
                        (vx1, max(vy1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        writer.write(annotated)
        if frame_idx % 50 == 0:
            print(f"[Video] Frame {frame_idx}/{total}  tracked: {len(track_data)} vehicles")

    cap.release()
    writer.release()
    print(f"[Video] Annotated video saved → {annotated_path}")

    # ── Build JSON from aggregated track data ────────────────────────────────
    json_output = {}
    vehicle_counter = 1

    for track_id, data in sorted(track_data.items()):
        vehicle_key   = f"vehicle_{vehicle_counter}"
        vehicle_counter += 1
        veh_class     = data['vehicle_class']
        observations  = data['observations']

        best = _best_ocr_result(observations)
        plate_img_path = None

        # Save best plate crop
        if best['crop'] is not None:
            fname = f"{vehicle_key}_plate.jpg"
            fpath = os.path.join(plate_output_dir, fname)
            cv2.imwrite(fpath, best['crop'])
            plate_img_path = os.path.join('plate_output', fname)
            print(f"[Video] Saved best plate crop → {fpath}")

        json_output[vehicle_key] = {
            'vehicle_class': veh_class,
            'plate_number':  best['plate_number'],
            'confidence':    best['confidence'],
            'plate_image':   plate_img_path,
        }
        status = best['plate_number'] or 'null'
        print(f"[Video] {vehicle_key} ({veh_class}): {status}  conf={best['confidence']:.2%}")

    json_path = os.path.join(output_dir, 'vehicle_plates.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)

    print(f"\n[Video] JSON saved → {json_path}")
    print(json.dumps(json_output, indent=4, ensure_ascii=False))
