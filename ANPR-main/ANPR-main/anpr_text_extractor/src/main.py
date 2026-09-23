"""
main.py  (updated)
Entry point for the ANPR pipeline.

Usage:
    # Image mode (default)
    python src/main.py

    # Image mode — explicit path
    python src/main.py --image sample_images/test2.jpg

    # Video mode
    python src/main.py --video "sample video/Nissan_altima_2013_black_02 (1).mp4"

    # Auto-detect video from sample_video/ folder
    python src/main.py --video auto
"""
import os
import sys
import json
import glob
import argparse
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_plate import detect_plate
from ocr_reader import extract_text

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PIPELINE  (original functionality — preserved intact)
# ═══════════════════════════════════════════════════════════════════════════════

def run_image_pipeline(image_path: str) -> None:
    """
    Runs the full ANPR pipeline on a single image.
    Uses detect_plate() directly on the full image (no vehicle pre-detection).
    """
    plate_output_dir = os.path.join(_BASE_DIR, 'plate_output')
    output_dir       = os.path.join(_BASE_DIR, 'output')
    yolo_output_dir  = os.path.join(_BASE_DIR, 'yolo_output')

    os.makedirs(plate_output_dir, exist_ok=True)
    os.makedirs(output_dir,       exist_ok=True)
    os.makedirs(yolo_output_dir,  exist_ok=True)

    print(f"\n[Image Pipeline] Loading: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return

    print("[Image Pipeline] Running license plate detection...")
    cropped_plates, boxed_image = detect_plate(img)

    boxed_path = os.path.join(yolo_output_dir, 'full_image_with_boxes.jpg')
    cv2.imwrite(boxed_path, boxed_image)
    print(f"[Image Pipeline] Annotated image saved → {boxed_path}")
    print(f"[Image Pipeline] Detected {len(cropped_plates)} plate(s).")

    if not cropped_plates:
        print("[Image Pipeline] No plates detected.")
        return

    results_json: dict = {}

    for idx, plate_img in enumerate(cropped_plates, start=1):
        vehicle_key    = f"vehicle_{idx}"
        plate_filename = f"vehicle_{idx}_plate.jpg"
        plate_path     = os.path.join(plate_output_dir, plate_filename)
        plate_rel      = os.path.join('plate_output', plate_filename)

        cv2.imwrite(plate_path, plate_img)
        print(f"[Image Pipeline] Plate crop saved → {plate_path}")

        print(f"[Image Pipeline] Running OCR on {vehicle_key}...")
        text, confidence = extract_text(plate_img)

        if text:
            print(f"[Image Pipeline]   ✓ '{text}'  conf={confidence:.2%}")
        else:
            print(f"[Image Pipeline]   ✗ OCR could not read plate.")

        results_json[vehicle_key] = {
            'vehicle_class': 'unknown',
            'plate_number':  text,
            'confidence':    confidence,
            'plate_image':   plate_rel,
        }

    json_path = os.path.join(output_dir, 'vehicle_plates.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"\n[Image Pipeline] JSON saved → {json_path}")
    print("[Image Pipeline] Done.\n")
    print(json.dumps(results_json, indent=4, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _find_sample_video() -> str | None:
    """Auto-detect a video file inside the 'sample video/' or 'sample_video/' folder."""
    for folder in ['sample video', 'sample_video']:
        folder_path = os.path.join(_BASE_DIR, folder)
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            matches = glob.glob(os.path.join(folder_path, ext))
            if matches:
                return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='ANPR Pipeline')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    group.add_argument('--video', type=str, default=None,
                       help='Path to input video, or "auto" to find in sample_video/')
    args = parser.parse_args()

    if args.video is not None:
        # Video mode
        from video_pipeline import process_video

        if args.video.lower() == 'auto':
            video_path = _find_sample_video()
            if not video_path:
                print("[ERROR] No video found in 'sample video/' or 'sample_video/'.")
                sys.exit(1)
        else:
            video_path = os.path.join(_BASE_DIR, args.video) \
                         if not os.path.isabs(args.video) else args.video
            if not os.path.exists(video_path):
                video_path = args.video  # try as-is

        print(f"[Main] Video mode: {video_path}")
        process_video(video_path)

    else:
        # Image mode (default)
        if args.image:
            image_path = os.path.join(_BASE_DIR, args.image) \
                         if not os.path.isabs(args.image) else args.image
        else:
            image_path = os.path.join(_BASE_DIR, 'sample_images', 'test2.jpg')

        print(f"[Main] Image mode: {image_path}")
        run_image_pipeline(image_path)


if __name__ == '__main__':
    main()
