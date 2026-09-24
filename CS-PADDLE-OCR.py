"""
CS-PADDLE-OCR.py

Quick side-by-side test: run PaddleOCR on the same essay photo(s) you've
been testing with TrOCR, so you can directly compare recognition quality.

PaddleOCR runs fully locally after the first model download (models are
cached to disk, ~10-15 MB total for the default English models). No API
key, no account, no per-request cost, no rate limit -- test as many times
as you want.

You already have the newer PaddleOCR 3.x installed and its models
downloaded/cached -- no reinstall needed. This version of the script:
  - disables mkldnn (enable_mkldnn=False), which works around a CPU
    inference crash some Windows setups hit with this paddlepaddle/
    PaddleOCR combo (the "ConvertPirAttribute2RuntimeAttribute" error).
    Kept OFF here for stability -- once you've confirmed things work,
    you can try flipping it to True to see if it's faster and still
    stable on your machine.
  - turns off doc-orientation-classification and page-unwarping, which
    you don't need for a flat upright phone photo -- this just skips
    using those already-downloaded models at runtime, no re-download
  - downscales the image to a max side of 2000px before running OCR
    (MAX_SIDE below) -- your original 3456x4608 photo was pushing well
    past PaddleOCR's own resize limit and taking ~3 minutes on CPU;
    a smaller, still-legible input should cut that down substantially


Usage:
    python CS-PADDLE-OCR.py path/to/test_img3.jpg
"""

import sys
import os
import cv2
from paddleocr import PaddleOCR

# Longest side to downscale to before running OCR. Smaller = faster, but
# too small can hurt recognition of small/tight handwriting. 1800-2200
# is a reasonable starting point for phone photos of notebook pages.
MAX_SIDE = 2000


def downscale_if_needed(image_path, max_side=MAX_SIDE):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_path

    scale = max_side / longest
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    base = os.path.splitext(image_path)[0]
    resized_path = f"{base}_resized_for_ocr.jpg"
    cv2.imwrite(resized_path, resized)
    print(f"Downscaled {w}x{h} -> {resized.shape[1]}x{resized.shape[0]}, saved to {resized_path}")
    return resized_path


def main(image_path):
    base = os.path.splitext(image_path)[0]

    image_path = downscale_if_needed(image_path)

    print("Loading PaddleOCR (models already cached from your last run)...")
    # enable_mkldnn=False works around a CPU/oneDNN inference bug in this
    # PaddleOCR/paddlepaddle combo (the crash you hit). It's slightly
    # slower per image but far more reliable, and fine for one-off tests.
    # The doc-orientation/unwarping stages are switched off since you
    # don't need them for a flat, upright phone photo -- this just skips
    # using those already-downloaded models, no new downloads triggered.
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang='en',
        enable_mkldnn=False,
    )

    print(f"Running OCR on {image_path}...")
    result = ocr.predict(image_path)

    # PaddleOCR 3.x returns a list of result objects (one per image),
    # each behaving like a dict with 'rec_texts' / 'rec_scores' keys.
    page = result[0] if result else {}
    texts = page.get("rec_texts", []) if hasattr(page, "get") else page["rec_texts"]
    scores = page.get("rec_scores", []) if hasattr(page, "get") else page["rec_scores"]

    print(f"\nDetected {len(texts)} line(s)/text region(s).\n")

    recognized_lines = list(texts)
    for i, (text, conf) in enumerate(zip(texts, scores)):
        print(f"Line {i + 1}/{len(texts)} (conf {conf:.2f}): {text}")

    out_path = f"{base}_extracted_text_paddle.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(recognized_lines))

    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python CS-PADDLE-OCR.py path/to/essay_photo.jpg")
        sys.exit(1)
    main(sys.argv[1])