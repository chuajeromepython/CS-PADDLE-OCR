"""
evaluate_ocr.py

Measures how well a PaddleOCR model (stock or fine-tuned) actually performs
on YOUR handwriting samples, using Character Error Rate (CER) and Word Error
Rate (WER) against ground-truth transcriptions you provide.

This turns "the fine-tuned model feels better" into an actual number you can
compare before/after, the same way accuracy_tests/ works in BERT-Spellchecker.

Setup:
    Create a folder of image/ground-truth pairs, e.g.:
        eval_set/
          test_img6.jpg
          test_img6.txt      <- the correct transcription, one line per text line
          test_img7.jpg
          test_img7.txt
          ...

    Ground truth .txt files should have one line of text per line of
    handwriting, in reading order, matching how PaddleOCR will detect lines.

Usage:
    python evaluate_ocr.py eval_set/ --rec-model-dir ./iam_finetune_infer
    python evaluate_ocr.py eval_set/                     # uses the stock models
    python evaluate_ocr.py eval_set/ --rec-model-dir ./iam_finetune_infer --compare-stock
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
    print("This script needs jiwer for CER/WER computation: pip install jiwer")
    sys.exit(1)

from paddleocr import PaddleOCR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ocr_utils import drop_edge_fragments, filter_by_score, ordered_lines  # noqa: E402


def stock_detector_name(lang="en"):
    """Name of the detector PaddleOCR picks for `lang` when no model options are
    given. PaddleOCR ignores `lang` as soon as you pass a recognition model dir,
    so we pass this detector explicitly to keep stock and fine-tuned runs
    comparable (same detector, only the recognizer differs)."""
    try:
        det_name, _ = PaddleOCR._get_ocr_model_names(None, lang, None)
        return det_name
    except Exception:
        return None


def load_eval_pairs(eval_dir):
    """Finds every <name>.jpg/.png with a matching <name>.txt ground truth."""
    pairs = []
    for fname in sorted(os.listdir(eval_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        gt_path = os.path.join(eval_dir, f"{base}.txt")
        if not os.path.exists(gt_path):
            print(f"Skipping {fname}: no matching {base}.txt ground truth found.")
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt_text = f.read().strip()
        pairs.append((os.path.join(eval_dir, fname), gt_text))
    return pairs


DEFAULT_MAX_SIDE = 2000  # same default as CS-PADDLE-OCR.py


def load_resized(image_path, max_side=DEFAULT_MAX_SIDE):
    """Read an image and downscale so its longest side is <= max_side.
    Mirrors CS-PADDLE-OCR.py so the eval measures the same pipeline you run."""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    longest = max(h, w)
    if max_side and longest > max_side:
        scale = max_side / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def normalise(text):
    """Lowercase and strip punctuation, to see how much error is only case/punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower())


def run_ocr_on_image(ocr, image_path, max_side=DEFAULT_MAX_SIDE, rec_thresh=0.0, edge_margin=0.0):
    img = load_resized(image_path, max_side)
    result = ocr.predict(img)
    page = result[0] if result else {}
    page = drop_edge_fragments(filter_by_score(page, rec_thresh), img.shape[1], edge_margin)
    lines, _ = ordered_lines(page)
    return "\n".join(lines)


def score(ocr, pairs, label, max_side=DEFAULT_MAX_SIDE, rec_thresh=0.0, edge_margin=0.0):
    print(f"\n=== {label} ===")
    all_gt, all_pred = [], []
    for image_path, gt_text in pairs:
        pred_text = run_ocr_on_image(ocr, image_path, max_side, rec_thresh, edge_margin)
        cer = jiwer.cer(gt_text, pred_text)
        wer = jiwer.wer(gt_text, pred_text)
        cer_n = jiwer.cer(normalise(gt_text), normalise(pred_text))
        all_gt.append(gt_text)
        all_pred.append(pred_text)
        print(f"{os.path.basename(image_path)}: CER={cer:.3f}  WER={wer:.3f}  CER(no case/punct)={cer_n:.3f}")

    overall_cer = jiwer.cer(all_gt, all_pred)
    overall_wer = jiwer.wer(all_gt, all_pred)
    overall_cer_n = jiwer.cer([normalise(t) for t in all_gt], [normalise(t) for t in all_pred])
    print(f"\n{label} OVERALL: CER={overall_cer:.3f}  WER={overall_wer:.3f}  "
          f"CER(no case/punct)={overall_cer_n:.3f}  "
          f"(lower is better; 0 = perfect)")
    return overall_cer, overall_wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="Folder of image + matching .txt ground-truth pairs")
    ap.add_argument("--rec-model-dir", default=None,
                     help="Path to a fine-tuned recognition model (omit to use the stock models)")
    ap.add_argument("--compare-stock", action="store_true",
                     help="Also run stock for a side-by-side comparison")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                     help=f"Downscale images so the longest side is at most this many px "
                          f"(default {DEFAULT_MAX_SIDE}, same as CS-PADDLE-OCR.py; 0 = no resize)")
    ap.add_argument("--rec-thresh", type=float, default=0.0,
                     help="Drop recognized regions below this confidence (default 0 = keep all)")
    ap.add_argument("--edge-margin", type=float, default=0.0,
                     help="Drop short (<=3 char) text within this fraction of the left/right "
                          "image edge, e.g. 0.08 (default 0 = off)")
    args = ap.parse_args()

    pairs = load_eval_pairs(args.eval_dir)
    if not pairs:
        print(f"No image/ground-truth pairs found in {args.eval_dir}.")
        sys.exit(1)
    print(f"Loaded {len(pairs)} evaluation pairs from {args.eval_dir}.")

    common_kwargs = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="en",
        enable_mkldnn=False,
    )

    if args.rec_model_dir:
        print(f"Loading fine-tuned model from {args.rec_model_dir} ...")
        ft_kwargs = {k: v for k, v in common_kwargs.items() if k != "lang"}
        ft_kwargs["text_recognition_model_dir"] = args.rec_model_dir
        ft_kwargs["text_recognition_model_name"] = "en_PP-OCRv4_mobile_rec"
        det_name = stock_detector_name("en")
        if det_name:
            ft_kwargs["text_detection_model_name"] = det_name
        ft_ocr = PaddleOCR(**ft_kwargs)
        ft_cer, ft_wer = score(ft_ocr, pairs, "Fine-tuned model", args.max_side, args.rec_thresh, args.edge_margin)

    if args.compare_stock or not args.rec_model_dir:
        print("Loading stock PaddleOCR English models (PP-OCRv6 medium) ...")
        stock_ocr = PaddleOCR(**common_kwargs)
        stock_cer, stock_wer = score(stock_ocr, pairs, "Stock PP-OCRv6 medium", args.max_side, args.rec_thresh, args.edge_margin)

    if args.rec_model_dir and args.compare_stock:
        print("\n=== Comparison ===")
        print(f"CER: stock {stock_cer:.3f} -> fine-tuned {ft_cer:.3f} "
              f"({'improved' if ft_cer < stock_cer else 'worse'})")
        print(f"WER: stock {stock_wer:.3f} -> fine-tuned {ft_wer:.3f} "
              f"({'improved' if ft_wer < stock_wer else 'worse'})")


if __name__ == "__main__":
    main()