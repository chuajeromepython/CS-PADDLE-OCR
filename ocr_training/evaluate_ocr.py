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
    python evaluate_ocr.py eval_set/                     # uses stock PP-OCRv4
    python evaluate_ocr.py eval_set/ --rec-model-dir ./iam_finetune_infer --compare-stock
"""

import argparse
import os
import sys

try:
    import jiwer
except ImportError:
    print("This script needs jiwer for CER/WER computation: pip install jiwer")
    sys.exit(1)

from paddleocr import PaddleOCR


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


def run_ocr_on_image(ocr, image_path):
    result = ocr.predict(image_path)
    page = result[0] if result else {}
    texts = page.get("rec_texts", []) if hasattr(page, "get") else page["rec_texts"]
    return "\n".join(texts)


def score(ocr, pairs, label):
    print(f"\n=== {label} ===")
    all_gt, all_pred = [], []
    for image_path, gt_text in pairs:
        pred_text = run_ocr_on_image(ocr, image_path)
        cer = jiwer.cer(gt_text, pred_text)
        wer = jiwer.wer(gt_text, pred_text)
        all_gt.append(gt_text)
        all_pred.append(pred_text)
        print(f"{os.path.basename(image_path)}: CER={cer:.3f}  WER={wer:.3f}")

    overall_cer = jiwer.cer(all_gt, all_pred)
    overall_wer = jiwer.wer(all_gt, all_pred)
    print(f"\n{label} OVERALL: CER={overall_cer:.3f}  WER={overall_wer:.3f}  "
          f"(lower is better; 0 = perfect)")
    return overall_cer, overall_wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="Folder of image + matching .txt ground-truth pairs")
    ap.add_argument("--rec-model-dir", default=None,
                     help="Path to a fine-tuned recognition model (omit to use stock PP-OCRv4)")
    ap.add_argument("--compare-stock", action="store_true",
                     help="Also run stock PP-OCRv4 for a side-by-side comparison")
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
        ft_ocr = PaddleOCR(rec_model_dir=args.rec_model_dir, **common_kwargs)
        ft_cer, ft_wer = score(ft_ocr, pairs, "Fine-tuned model")

    if args.compare_stock or not args.rec_model_dir:
        print("Loading stock PP-OCRv4 model ...")
        stock_ocr = PaddleOCR(**common_kwargs)
        stock_cer, stock_wer = score(stock_ocr, pairs, "Stock PP-OCRv4")

    if args.rec_model_dir and args.compare_stock:
        print("\n=== Comparison ===")
        print(f"CER: stock {stock_cer:.3f} -> fine-tuned {ft_cer:.3f} "
              f"({'improved' if ft_cer < stock_cer else 'worse'})")
        print(f"WER: stock {stock_wer:.3f} -> fine-tuned {ft_wer:.3f} "
              f"({'improved' if ft_wer < stock_wer else 'worse'})")


if __name__ == "__main__":
    main()