"""
Downloads a dedicated YOLOv8 license plate detection model.
Model: keremberke/yolov8n-license-plate-detection (trained ONLY on license plates)
"""
import urllib.request
import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "license_plate_detector.pt")

# Direct download URL — YOLOv8 model trained ONLY on license plates (1 class: "license_plate")
# Source: Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
DOWNLOAD_URL = (
    "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8"
    "/raw/main/license_plate_detector.pt"
)

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at: {MODEL_PATH}")
        return

    print(f"Downloading dedicated license plate detection model...")
    print(f"Source: {DOWNLOAD_URL}")
    print(f"Destination: {MODEL_PATH}\n")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct / 2)
            sys.stdout.write(f"\r[{'#' * bar}{'-' * (50 - bar)}] {pct:.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(DOWNLOAD_URL, MODEL_PATH, reporthook=progress)
    print(f"\n\nDownload complete! Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    download_model()
