"""
calibrate_flagging.py

Does low OCR confidence actually predict OCR errors?

The teacher-review design only works if the words PaddleOCR is unsure about
are the words it gets wrong. This script measures that on your labeled eval
set. It keeps EVERY recognized region (no confidence filtering), aligns the
OCR words against your ground-truth words, labels each OCR word right/wrong,
and then reports, for each confidence threshold:

    flagged   share of all OCR words that would be sent to the teacher
              (the teacher's workload)
    caught    share of the real errors those flags catch (recall)
    precision share of flagged words that really are errors

It does this for two flagging rules:
    conf      flag if region confidence < threshold
    conf+oov  flag if confidence < threshold OR the word is not in a dictionary

It also lists the errors that sit at HIGH confidence, which are the ones the
teacher would never see. That list is the direct test of the worry that
errors like "abont" can score 0.95+.

Notes on what this can and cannot tell you:
  - Paddle gives one confidence per detected REGION (often a phrase), not per
    word. Every word in a region shares that region's score.
  - Alignment is word-level sequence matching against line-level labels, so a
    wrongly split or merged word can be counted as an error. Treat small
    differences between rows of the table as noise, especially with few images.
  - Words the OCR missed entirely (present in the labels, absent from the OCR
    output) cannot be flagged by confidence. They are reported separately.
  - The dictionary rule will flag proper nouns (e.g. "Thoreau"). That is a
    real cost of the rule, and it shows up in the precision column.

Setup: same as evaluate_ocr.py (labeled folder of image + .txt pairs).

Usage:
    python calibrate_flagging.py eval_set/
    python calibrate_flagging.py eval_set/ --rec-model-dir ./iam_finetune_infer
    python calibrate_flagging.py eval_set/ --wordlist words.txt --dump-csv words.csv

Dictionary: pass --wordlist (one word per line, or "word count" per line).
If omitted, the script tries the frequency dictionary bundled with symspellpy
(pip install symspellpy). If neither is found, the conf+oov rule is skipped.
"""

import argparse
import csv
import os
import re
import sys
from difflib import SequenceMatcher

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ocr_utils import _boxes_from_page, group_rows  # noqa: E402

DEFAULT_MAX_SIDE = 2000
DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
HIGH_CONF = 0.90  # an error at or above this is reported as "high-confidence"


# ----------------------------------------------------------------- helpers --

def norm_word(w):
    """Lowercase and strip everything except letters/digits."""
    return re.sub(r"[^\w]", "", w.lower())


def stock_detector_name(lang="en"):
    try:
        from paddleocr import PaddleOCR
        det_name, _ = PaddleOCR._get_ocr_model_names(None, lang, None)
        return det_name
    except Exception:
        return None


def load_eval_pairs(eval_dir):
    pairs = []
    for fname in sorted(os.listdir(eval_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        gt_path = os.path.join(eval_dir, f"{base}.txt")
        if not os.path.exists(gt_path):
            print(f"Skipping {fname}: no matching {base}.txt")
            continue
        with open(gt_path, encoding="utf-8") as f:
            pairs.append((os.path.join(eval_dir, fname), f.read().strip()))
    return pairs


def load_resized(image_path, max_side=DEFAULT_MAX_SIDE):
    import cv2
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    longest = max(h, w)
    if max_side and longest > max_side:
        scale = max_side / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def load_vocab(wordlist_path):
    """Return a set of normalised words, or None if no dictionary is available."""
    path = wordlist_path
    if not path:
        try:
            import symspellpy
            cand = os.path.join(os.path.dirname(symspellpy.__file__),
                                "frequency_dictionary_en_82_765.txt")
            if os.path.exists(cand):
                path = cand
        except ImportError:
            pass
    if not path or not os.path.exists(path):
        return None
    vocab = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab.add(norm_word(parts[0]))
    print(f"Loaded dictionary ({len(vocab)} words) from {path}")
    return vocab


# ------------------------------------------------------- OCR -> word list --

def extract_words(page):
    """Flatten a PaddleOCR page into words in reading order, KEEPING every
    region regardless of confidence. Each word carries its region's score."""
    getter = page.get if hasattr(page, "get") else (lambda k, d=None: page[k])
    texts = list(getter("rec_texts", []))
    scores = list(getter("rec_scores", []))
    boxes = _boxes_from_page(page, len(texts)) if texts else None
    rows = group_rows(boxes) if boxes is not None else [[i] for i in range(len(texts))]

    words = []
    for r, row in enumerate(rows):
        for i in row:
            for w in texts[i].split():
                n = norm_word(w)
                if n:
                    words.append({
                        "raw": w, "norm": n, "score": float(scores[i]),
                        "row": r, "region": i,
                        "box": [float(v) for v in boxes[i]] if boxes is not None else None,
                    })
    return words


def align(gt_norm, ocr_norm):
    """Word-level alignment. Returns (is_error[], gt_word_for_error[], n_missed).
    is_error[j] = 1 if OCR word j does not match the ground truth, else 0.
    n_missed    = ground-truth words with no OCR counterpart (cannot be flagged)."""
    sm = SequenceMatcher(None, gt_norm, ocr_norm, autojunk=False)
    is_error = [0] * len(ocr_norm)
    gt_for = [""] * len(ocr_norm)
    missed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            missed += i2 - i1
        elif tag == "insert":
            for j in range(j1, j2):
                is_error[j] = 1
        else:  # replace
            span = gt_norm[i1:i2]
            for j in range(j1, j2):
                is_error[j] = 1
                if span:  # display only: closest truth word in the mismatched span
                    gt_for[j] = max(span, key=lambda g: SequenceMatcher(None, g, ocr_norm[j]).ratio())
            missed += max(0, (i2 - i1) - (j2 - j1))
    return is_error, gt_for, missed


# ------------------------------------------------------------------ report --

def rule_stats(flag, err):
    n = len(err)
    n_err = int(err.sum())
    n_flag = int(flag.sum())
    hit = int((flag & (err == 1)).sum())
    flagged = n_flag / n if n else 0.0
    caught = hit / n_err if n_err else 0.0
    precision = hit / n_flag if n_flag else 0.0
    return flagged, caught, precision


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'thresh':>7} {'flagged':>9} {'caught':>9} {'precision':>10}")
    for t, f, c, p in rows:
        print(f"{t:>7.2f} {f:>8.1%} {c:>9.1%} {p:>10.1%}")


def run(all_words, thresholds, vocab, csv_out=None, dump_csv=None):
    score = np.array([w["score"] for w in all_words])
    err = np.array([w["is_error"] for w in all_words])
    n, n_err = len(err), int(err.sum())

    print("\n" + "=" * 60)
    print(f"OCR words: {n}   wrong: {n_err} ({n_err / n:.1%})" if n else "No OCR words.")
    if not n:
        return
    if n_err == 0:
        print("No errors found, so there is nothing to catch. Use harder images.")
        return

    oov = None
    if vocab is not None:
        oov = np.array([w["norm"] not in vocab and not w["norm"].isdigit() for w in all_words])
        _, c, p = rule_stats(oov, err)
        print(f"Dictionary rule alone: flags {oov.mean():.1%} of words, "
              f"catches {c:.1%} of errors, precision {p:.1%}")

    conf_rows, comb_rows, csv_rows = [], [], []
    for t in thresholds:
        low = score < t
        f, c, p = rule_stats(low, err)
        conf_rows.append((t, f, c, p))
        row = {"thresh": t, "conf_flagged": f, "conf_caught": c, "conf_precision": p}
        if oov is not None:
            f2, c2, p2 = rule_stats(low | oov, err)
            comb_rows.append((t, f2, c2, p2))
            row.update({"comb_flagged": f2, "comb_caught": c2, "comb_precision": p2})
        csv_rows.append(row)

    print_table("Rule: confidence only (flag if score < thresh)", conf_rows)
    if comb_rows:
        print_table("Rule: confidence OR not-in-dictionary", comb_rows)
    else:
        print("\n(conf+oov rule skipped: no dictionary. Pass --wordlist or pip install symspellpy.)")

    # The errors the teacher would never see under a confidence-only rule.
    hi = [w for w in all_words if w["is_error"] and w["score"] >= HIGH_CONF]
    print(f"\nHigh-confidence errors (score >= {HIGH_CONF}): {len(hi)} of {n_err} "
          f"({len(hi) / n_err:.1%} of all errors)")
    for w in sorted(hi, key=lambda x: -x["score"])[:20]:
        print(f"  {w['score']:.2f}  OCR '{w['raw']}'  ->  truth '{w['gt']}'   [{w['image']}]")

    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=sorted({k for r in csv_rows for k in r}))
            wr.writeheader()
            wr.writerows(csv_rows)
        print(f"\nSaved threshold table: {csv_out}")
    if dump_csv:
        with open(dump_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["image", "row", "ocr_word", "score", "is_error", "truth_word", "box"])
            for w in all_words:
                wr.writerow([w["image"], w["row"], w["raw"], f"{w['score']:.4f}",
                             w["is_error"], w["gt"], w["box"]])
        print(f"Saved per-word dump: {dump_csv}")


# -------------------------------------------------------------------- main --

def collect(ocr, pairs, max_side):
    all_words = []
    total_missed = total_gt = 0
    for image_path, gt_text in pairs:
        name = os.path.basename(image_path)
        img = load_resized(image_path, max_side)
        result = ocr.predict(img)
        page = result[0] if result else {}
        words = extract_words(page)

        gt_norm = [g for g in (norm_word(w) for w in gt_text.split()) if g]
        is_error, gt_for, missed = align(gt_norm, [w["norm"] for w in words])
        for w, e, g in zip(words, is_error, gt_for):
            w.update({"is_error": e, "gt": g, "image": name})
        all_words.extend(words)

        total_missed += missed
        total_gt += len(gt_norm)
        print(f"{name}: {len(words)} OCR words, {sum(is_error)} wrong, "
              f"{missed} truth words missed")
    return all_words, total_missed, total_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="Folder of image + matching .txt ground-truth pairs")
    ap.add_argument("--rec-model-dir", default=None,
                    help="Fine-tuned recognition model folder (omit for stock)")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--wordlist", default=None, help="Dictionary file for the OOV rule")
    ap.add_argument("--csv", default=None, help="Save the threshold table to this CSV")
    ap.add_argument("--dump-csv", default=None, help="Save every OCR word with its label")
    args = ap.parse_args()

    from paddleocr import PaddleOCR  # imported late so the helpers above are testable

    pairs = load_eval_pairs(args.eval_dir)
    if not pairs:
        print(f"No image/ground-truth pairs found in {args.eval_dir}.")
        sys.exit(1)
    print(f"Loaded {len(pairs)} evaluation pairs.")

    kwargs = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=True, lang="en", enable_mkldnn=False)
    if args.rec_model_dir:
        kwargs.pop("lang")
        kwargs["text_recognition_model_dir"] = args.rec_model_dir
        kwargs["text_recognition_model_name"] = "en_PP-OCRv4_mobile_rec"
        det = stock_detector_name("en")
        if det:
            kwargs["text_detection_model_name"] = det
        print(f"Using fine-tuned model: {args.rec_model_dir}")
    else:
        print("Using stock PaddleOCR models.")
    ocr = PaddleOCR(**kwargs)

    all_words, missed, total_gt = collect(ocr, pairs, args.max_side)
    print(f"\nTruth words the OCR missed entirely (unflaggable): {missed} of {total_gt} "
          f"({missed / total_gt:.1%})" if total_gt else "")

    vocab = load_vocab(args.wordlist)
    run(all_words, sorted(args.thresholds), vocab, args.csv, args.dump_csv)


if __name__ == "__main__":
    main()