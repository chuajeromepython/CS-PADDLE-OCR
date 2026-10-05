"""
sweep_stock.py

Finds the best input size / detection / confidence-filter settings for STOCK
PaddleOCR on your own handwriting, by scoring every combination against
ground-truth .txt files.

    python ocr_training/sweep_stock.py eval_set
    python ocr_training/sweep_stock.py eval_set --quick       # sizes only
    python ocr_training/sweep_stock.py eval_set --sizes 2000 --unclip 1.5 2.0 --rec-thresh 0 0.6 0.75 0.85

Results are printed best-first (by strict CER) and saved to sweep_results.csv.
At the end the best config's per-image predictions are printed next to the
ground truth, so label problems are easy to spot.

Settings swept:
  max_side              downscale target for the longest image side
  text_det_unclip_ratio how much detected boxes are expanded
  text_det_box_thresh   minimum box score kept by the detector
  rec_thresh            drop recognized regions below this confidence. This is
                        applied AFTER OCR, so it adds no extra OCR time.
"""

import argparse
import csv
import itertools
import os
import sys
import time

try:
    import jiwer
except ImportError:
    sys.exit("This script needs jiwer: pip install jiwer")

from paddleocr import PaddleOCR

from evaluate_ocr import load_eval_pairs, load_resized, normalise  # noqa: F401
from ocr_utils import filter_by_score, ordered_lines  # noqa: E402 (path set up by evaluate_ocr)


def render(page, rec_thresh):
    lines, _ = ordered_lines(filter_by_score(page, rec_thresh))
    return "\n".join(lines)


def run_detector_config(ocr, pairs, img_cache, max_side, unclip, box_thresh):
    """One OCR pass per image; returns the raw pages so several rec_thresh values
    can be scored from the same run."""
    det_kwargs = dict(text_det_unclip_ratio=unclip, text_det_box_thresh=box_thresh)
    t0 = time.time()
    pages = []
    for path, _ in pairs:
        key = (path, max_side)
        if key not in img_cache:
            img_cache[key] = load_resized(path, max_side)
        result = ocr.predict(img_cache[key], **det_kwargs)
        pages.append(result[0] if result else {})
    return pages, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="Folder of image + matching .txt ground-truth pairs")
    ap.add_argument("--quick", action="store_true",
                    help="Only sweep image size (default detector settings)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1500, 2000, 2500, 3000])
    ap.add_argument("--unclip", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    ap.add_argument("--box-thresh", type=float, nargs="+", default=[0.5, 0.6])
    ap.add_argument("--rec-thresh", type=float, nargs="+", default=[0.0],
                    help="Confidence cut-offs to try (applied after OCR, so they cost nothing)")
    ap.add_argument("--textline-orientation", choices=["on", "off"], default="on",
                    help="Run the line-orientation classifier (default on, as in CS-PADDLE-OCR.py)")
    ap.add_argument("--out", default="sweep_results.csv")
    args = ap.parse_args()

    pairs = load_eval_pairs(args.eval_dir)
    if not pairs:
        sys.exit(f"No image/ground-truth pairs found in {args.eval_dir}.")
    print(f"Loaded {len(pairs)} evaluation pairs from {args.eval_dir}.")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=(args.textline_orientation == "on"),
        lang="en",
        enable_mkldnn=False,
    )

    if args.quick:
        det_grid = [(s, 1.5, 0.6) for s in args.sizes]
    else:
        det_grid = list(itertools.product(args.sizes, args.unclip, args.box_thresh))
    n_total = len(det_grid) * len(args.rec_thresh)
    print(f"{len(det_grid)} OCR pass(es) x {len(args.rec_thresh)} confidence cut-off(s) = "
          f"{n_total} configuration(s) (textline orientation {args.textline_orientation}). "
          f"Each OCR pass can take minutes on CPU.\n")

    gts = [gt for _, gt in pairs]
    img_cache, rows = {}, []
    best, best_preds = None, None
    for i, (size, unclip, box) in enumerate(det_grid, 1):
        pages, secs = run_detector_config(ocr, pairs, img_cache, size, unclip, box)
        print(f"[OCR pass {i}/{len(det_grid)}] side={size} unclip={unclip} box={box} ({secs:.0f}s)")
        for rt in args.rec_thresh:
            preds = [render(p, rt) for p in pages]
            row = {
                "max_side": size, "unclip": unclip, "box_thresh": box, "rec_thresh": rt,
                "cer": jiwer.cer(gts, preds),
                "cer_nocase": jiwer.cer([normalise(t) for t in gts], [normalise(t) for t in preds]),
                "wer": jiwer.wer(gts, preds),
            }
            rows.append(row)
            print(f"    rec>={rt:<4}  CER={row['cer']:.3f}  CER(no case)={row['cer_nocase']:.3f}  "
                  f"WER={row['wer']:.3f}")
            if best is None or row["cer"] < best["cer"]:
                best, best_preds = row, preds

    rows.sort(key=lambda r: r["cer"])
    print("\n=== Top 5 (lower is better) ===")
    for r in rows[:5]:
        print(f"side={r['max_side']:<5} unclip={r['unclip']:<4} box={r['box_thresh']:<4} "
              f"rec>={r['rec_thresh']:<4} CER={r['cer']:.3f}  "
              f"CER(no case)={r['cer_nocase']:.3f}  WER={r['wer']:.3f}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved all results to {args.out}")

    print(f"\n=== Best config: side={best['max_side']} unclip={best['unclip']} "
          f"box={best['box_thresh']} rec>={best['rec_thresh']} -- per-image prediction vs ground truth ===")
    for (path, gt), pred in zip(pairs, best_preds):
        print(f"\n--- {os.path.basename(path)}  (CER={jiwer.cer(gt, pred):.3f}) ---")
        print("GROUND TRUTH:\n" + gt)
        print("PREDICTED:\n" + pred)


if __name__ == "__main__":
    main()