r"""
prepare_smhd_eval.py

Builds a held-out test folder (image + cleaned .txt pairs) from the Student Messy
Handwritten Dataset (SMHD), in the format ocr_training/evaluate_ocr.py reads.

What it does
  1. Picks one contiguous block of labeled pages per subject (writers usually wrote
     consecutive pages, so a block + a gap keeps the same writer out of training).
  2. Cleans the transcriptions so scores are not distorted by the dataset's style:
       - "word , word"  ->  "word, word"   (spaces around punctuation removed)
       - '#' strikethrough markers removed
       - broken-encoding characters removed
     Spelling mistakes are NOT corrected.
  3. Writes eval_others/split_manifest.json listing the test pages, the gap pages
     (must stay out of training) and the remaining pool pages (safe for training).

Usage (from the repo root):
    python ocr_training/prepare_smhd_eval.py --smhd SMHD --out eval_others
Then:
    python ocr_training/evaluate_ocr.py eval_others --rec-model-dir journal_v6_infer --compare-stock
"""
import argparse
import json
import os
import re
import shutil

# subject folder prefix -> (short name, number of test pages)
SUBJECTS = {
    "essaybased":    ("essay",   8),
    "summary-based": ("summary", 10),
    "english":       ("english", 8),
    "biology":       ("biology", 6),
    "notes based":   ("notes",   4),
}
GAP = 5   # pages on each side of the test block (by page number) kept out of training


def fix_quotes(s):
    """Standalone quote marks: the opening one sticks to the next word, the closing one to the previous word."""
    out, open_q, pending = [], True, False
    for t in s.split(" "):
        if t == '"':
            if open_q:
                pending = True
            elif out:
                out[-1] += '"'
            else:
                out.append('"')
            open_q = not open_q
        else:
            if pending:
                t, pending = '"' + t, False
            out.append(t)
    if pending:
        out.append('"')
    return " ".join(out)


def clean_line(s):
    s = s.replace("\ufffd", "")
    s = re.sub(r"#\s*", "", s)                      # strikethrough markers
    s = fix_quotes(s)
    s = re.sub(r"\s+([,.:;!?)\]])", r"\1", s)       # no space before punctuation / closing bracket
    s = re.sub(r"([(\[])\s+", r"\1", s)             # no space after opening bracket
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def clean_text(raw):
    lines = [clean_line(l) for l in raw.splitlines()]
    return "\n".join(l for l in lines if l)


def find_subject(folder):
    low = folder.lower()
    for key, val in SUBJECTS.items():
        if low.startswith(key):
            return val
    return None


def page_id(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smhd", default="SMHD", help="path to the SMHD folder")
    ap.add_argument("--out", default="eval_others", help="output folder")
    ap.add_argument("--max-side", type=int, default=2000,
                    help="shrink photos so the longest side is at most this (0 = copy as-is)")
    args = ap.parse_args()

    if args.max_side:
        import cv2

    os.makedirs(args.out, exist_ok=True)
    manifest = {"gap_pages": GAP, "test": [], "gap": [], "train_pool": [], "excluded_subjects": []}
    n_lines = 0

    for folder in sorted(os.listdir(args.smhd)):
        fdir = os.path.join(args.smhd, folder)
        if not os.path.isdir(fdir):
            continue
        # labeled pages = image with a matching .txt
        labeled = []
        for fn in sorted(os.listdir(fdir)):
            base, ext = os.path.splitext(fn)
            if ext.lower() in (".jpg", ".jpeg") and os.path.exists(os.path.join(fdir, base + ".txt")):
                labeled.append(fn)
        subj = find_subject(folder)
        if subj is None:
            manifest["excluded_subjects"].append(folder)
            continue
        short, quota = subj
        quota = min(quota, len(labeled))
        start = min(int(0.6 * len(labeled)), len(labeled) - quota)
        block = labeled[start:start + quota]
        lo, hi = page_id(block[0]) - GAP, page_id(block[-1]) + GAP
        for fn in labeled:
            rel = f"{folder}/{fn}"
            if fn in block:
                manifest["test"].append(rel)
            elif lo <= page_id(fn) <= hi:
                manifest["gap"].append(rel)
            else:
                manifest["train_pool"].append(rel)

        for fn in block:
            base, ext = os.path.splitext(fn)
            name = f"smhd_{short}_{base}"
            src_img = os.path.join(fdir, fn)
            dst_img = os.path.join(args.out, name + ".jpg")
            if args.max_side:
                im = cv2.imread(src_img)
                h, w = im.shape[:2]
                s = args.max_side / max(h, w)
                if s < 1:
                    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_img, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copy(src_img, dst_img)
            raw = open(os.path.join(fdir, base + ".txt"), encoding="utf-8", errors="replace").read()
            cleaned = clean_text(raw)
            n_lines += len(cleaned.splitlines())
            with open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(cleaned + "\n")
        print(f"{short:8} test pages {block[0]} .. {block[-1]}  ({len(block)} pages)")

    with open(os.path.join(args.out, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nTest set: {len(manifest['test'])} pages, {n_lines} text lines -> {args.out}/")
    print(f"Kept out of training as a gap: {len(manifest['gap'])} pages")
    print(f"Safe training pool: {len(manifest['train_pool'])} labeled pages")
    print(f"Not used (symbols/equations): {manifest['excluded_subjects']}")


if __name__ == "__main__":
    main()