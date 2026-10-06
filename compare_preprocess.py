"""
compare_preprocess.py -- does cleaning the page image help PaddleOCR?

Runs the SAME PaddleOCR setup on every image in an eval folder twice (or more):
  baseline        the photo as-is (downscaled like CS-PADDLE-OCR.py does)
  clean@<frac>    the photo after preprocess.clean_page(ink_frac=<frac>)
and prints CER/WER per image, side by side, plus overall numbers and how many
images got better / worse. Cleaned images are saved so you can look at them.

Put this file next to ocr_utils.py and preprocess.py (the repo root), then:

    python compare_preprocess.py eval_set --rec-thresh 0.7
    python compare_preprocess.py eval_set --rec-thresh 0.7 --ink-fracs 0.3 0.4 0.5

Notes
  * Compare the columns WITHIN one run. This script does not replicate
    evaluate_ocr.py's --edge-margin, so absolute numbers can differ from your
    usual baseline.
  * Each variant = one full OCR pass over the folder (about a minute per image
    on CPU), so every extra --ink-fracs value adds that time.
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

try:
    import jiwer
except ImportError:
    print("Needs jiwer: pip install jiwer")
    sys.exit(1)

from paddleocr import PaddleOCR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from ocr_utils import filter_by_score, ordered_lines  # noqa: E402
from preprocess import clean_page  # noqa: E402


def stock_detector_name(lang="en"):
    try:
        det_name, _ = PaddleOCR._get_ocr_model_names(None, lang, None)
        return det_name
    except Exception:
        return None


def build_ocr(rec_model_dir=None):
    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=True, lang="en", enable_mkldnn=False)
    if rec_model_dir:
        kw.pop("lang", None)
        kw["text_recognition_model_dir"] = rec_model_dir
        kw["text_recognition_model_name"] = "en_PP-OCRv4_mobile_rec"
        det = stock_detector_name("en")
        if det:
            kw["text_detection_model_name"] = det
    return PaddleOCR(**kw)


def load_pairs(eval_dir):
    pairs = []
    for fn in sorted(os.listdir(eval_dir)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            gt = os.path.join(eval_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.exists(gt):
                pairs.append((os.path.join(eval_dir, fn), open(gt, encoding="utf-8").read().strip()))
    return pairs


def load_resized(path, max_side):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def read_text(ocr, img, rec_thresh):
    result = ocr.predict(img)
    page = result[0] if result else {}
    lines, _ = ordered_lines(filter_by_score(page, rec_thresh))
    return "\n".join(lines)


def norm(t):
    return re.sub(r"[^\w\s]", "", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--rec-model-dir", default=None)
    ap.add_argument("--rec-thresh", type=float, default=0.0)
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--ink-fracs", type=float, nargs="+", default=[0.4],
                    help="ink_frac values to try (default 0.4). Lower keeps fainter strokes.")
    ap.add_argument("--no-rules", action="store_true", help="skip ruled-line removal")
    args = ap.parse_args()

    pairs = load_pairs(args.eval_dir)
    if not pairs:
        sys.exit(f"No image + .txt pairs found in {args.eval_dir}")

    print("Loading PaddleOCR...")
    ocr = build_ocr(args.rec_model_dir)

    variants = ["baseline"] + [f"clean@{f:g}" for f in args.ink_fracs]
    out_dir = os.path.join(args.eval_dir, "debug_output")
    os.makedirs(out_dir, exist_ok=True)
    preds = {v: [] for v in variants}
    gts = [gt for _, gt in pairs]

    for path, _ in pairs:
        name = os.path.splitext(os.path.basename(path))[0]
        base_img = load_resized(path, args.max_side)
        for v in variants:
            if v == "baseline":
                img = base_img
            else:
                frac = float(v.split("@")[1])
                img = clean_page(base_img, ink_frac=frac, remove_rules=not args.no_rules)
                cv2.imwrite(os.path.join(out_dir, f"{name}_{v}.png"), img)
            text = read_text(ocr, img, args.rec_thresh)
            preds[v].append(text)
            with open(os.path.join(out_dir, f"{name}_{v}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
        print(f"done {name}")

    cer = {v: [jiwer.cer(g, p) for g, p in zip(gts, preds[v])] for v in variants}
    print("\nCER per image (lower is better)")
    print(f"{'image':<16}" + "".join(f"{v:>14}" for v in variants))
    for i, (path, _) in enumerate(pairs):
        print(f"{os.path.basename(path):<16}" + "".join(f"{cer[v][i]:>14.3f}" for v in variants))

    print("\nOverall")
    for v in variants:
        o_cer = jiwer.cer(gts, preds[v]); o_wer = jiwer.wer(gts, preds[v])
        o_n = jiwer.cer([norm(g) for g in gts], [norm(p) for p in preds[v]])
        line = f"{v:<14} CER={o_cer:.3f}  WER={o_wer:.3f}  CER(no case/punct)={o_n:.3f}"
        if v != "baseline":
            better = sum(c < b - 1e-9 for c, b in zip(cer[v], cer["baseline"]))
            worse = sum(c > b + 1e-9 for c, b in zip(cer[v], cer["baseline"]))
            line += f"   images better/worse vs baseline: {better}/{worse}"
        print(line)
    print(f"\nCleaned images and OCR text saved in {out_dir}")


if __name__ == "__main__":
    main()