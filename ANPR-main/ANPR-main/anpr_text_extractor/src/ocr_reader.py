"""
ocr_reader.py  (updated)
Extracts text from a cropped license plate image using PaddleOCR 3.x (CPU-only, PP-OCRv4 mobile).

PaddleOCR is initialized ONCE at module level to avoid reloading on every call.

NOTE: PaddleOCR 3.x API:
  - Use `device='cpu'` instead of `use_gpu`.
  - Use `.predict()` instead of `.ocr()`.
  - Results use keys `rec_texts` / `rec_scores`.
  - Doc-orientation classify disabled (causes 180-deg flip on license plates).
"""
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# Module-level singleton — initialized ONCE, reused on every extract_text() call
# ---------------------------------------------------------------------------
_ocr = PaddleOCR(
    lang='en',
    ocr_version='PP-OCRv4',             # lightweight PP-OCRv4 mobile model
    device='cpu',                       # CPU-only, no GPU
    enable_mkldnn=False,                # Disabled: NotImplementedError on PaddlePaddle 3.x + Windows
    cpu_threads=4,
    use_doc_orientation_classify=False, # Disabled: causes 180° rotation on license plates
    use_doc_unwarping=False,            # Disabled: plates are already flat
    use_textline_orientation=False,     # Disabled: not needed for single-line plates
)
# ---------------------------------------------------------------------------

# Characters that are definitely OCR artifacts on license plates
_ARTIFACT_RE = re.compile(r'[^A-Z0-9\s\-]')


def clean_plate_text(text: str) -> str:
    """
    Normalize raw OCR output to a clean license plate string.
    - Uppercase
    - Remove non-alphanumeric characters (except hyphens)
    - Collapse internal whitespace
    - Strip leading/trailing whitespace
    Does NOT invent or substitute characters.
    """
    if not text:
        return text
    text = text.upper()
    text = _ARTIFACT_RE.sub('', text)   # remove symbols like !, @, #, etc.
    text = re.sub(r'\s+', ' ', text)    # collapse multiple spaces
    return text.strip()


def extract_text(plate_image: np.ndarray) -> tuple:
    """
    Preprocess a cropped license plate image and run PaddleOCR to extract text.

    Pipeline:
        1. Upscale 2× with INTER_CUBIC  (better character resolution)
        2. Convert to grayscale, back to 3-ch BGR (PaddleOCR needs BGR)
        3. Run PaddleOCR via .predict() with low score threshold
        4. Aggregate rec_texts / rec_scores, clean, return

    Args:
        plate_image: BGR NumPy array — must be a cropped license plate ONLY.

    Returns:
        (str, float)  — cleaned text and mean confidence (0.0–1.0)
        (None, 0.0)   — if OCR returns no text
    """
    if plate_image is None or plate_image.size == 0:
        return None, 0.0

    # 1. Upscale 2× for better character resolution
    h, w = plate_image.shape[:2]
    upscaled = cv2.resize(plate_image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale → 3-ch BGR
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 3. Run OCR
    try:
        results = _ocr.predict(gray_3ch, text_rec_score_thresh=0.1)
    except Exception as e:
        print(f"[OCR] predict() raised: {e}")
        return None, 0.0

    if not results:
        return None, 0.0

    texts = []
    confidences = []

    for result in results:
        if not isinstance(result, dict):
            continue
        rec_texts  = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        for text, score in zip(rec_texts, rec_scores):
            text = str(text).strip()
            if text:
                texts.append(text)
                confidences.append(float(score))

    if not texts:
        return None, 0.0

    raw_text = ' '.join(texts)
    cleaned  = clean_plate_text(raw_text)

    if not cleaned:
        return None, 0.0

    avg_confidence = sum(confidences) / len(confidences)
    return cleaned, round(avg_confidence, 4)
