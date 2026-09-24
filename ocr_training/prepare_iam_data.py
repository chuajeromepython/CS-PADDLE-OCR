"""
prepare_iam_data.py

Converts the raw IAM Handwriting Database (line-level) into the label
format PaddleOCR's recognition trainer expects:

    <image_path>\t<transcription text>

one pair per line, split into train_list.txt / val_list.txt.

Requires (from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database,
free registration required):
  - ascii/lines.txt           (metadata + transcriptions)
  - data/lines.tgz, extracted (line-level PNGs, nested by form/subform)

Usage:
    python prepare_iam_data.py \
        --lines-txt /path/to/ascii/lines.txt \
        --img-root  /path/to/lines_extracted \
        --out-dir   /path/to/output \
        --val-split 0.1
"""

import argparse
import os
import random


def parse_lines_txt(lines_txt_path):
    """Yields (line_id, status, text) for every entry in IAM's lines.txt."""
    with open(lines_txt_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(" ")
            line_id = parts[0]           # e.g. a01-000u-00
            status = parts[1]            # "ok" or "err" segmentation
            # Transcription is everything after the 8 metadata fields,
            # with IAM's "|" standing in for spaces.
            text = " ".join(parts[8:]).replace("|", " ")
            yield line_id, status, text


def resolve_image_path(line_id, img_root):
    """IAM nests images as <form>/<form-subform>/<line_id>.png"""
    form = line_id.split("-")[0]
    subform = "-".join(line_id.split("-")[:2])
    return os.path.join(img_root, form, subform, f"{line_id}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines-txt", required=True, help="Path to IAM's ascii/lines.txt")
    ap.add_argument("--img-root", required=True, help="Root folder of extracted line PNGs")
    ap.add_argument("--out-dir", default=".", help="Where to write train_list.txt / val_list.txt")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--include-err-segmentation", action="store_true",
                     help="Include lines IAM flagged as badly segmented (default: skip them)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    kept, skipped_status, skipped_missing = 0, 0, 0
    pairs = []

    for line_id, status, text in parse_lines_txt(args.lines_txt):
        if status != "ok" and not args.include_err_segmentation:
            skipped_status += 1
            continue

        img_path = resolve_image_path(line_id, args.img_root)
        if not os.path.exists(img_path):
            skipped_missing += 1
            continue

        if not text.strip():
            continue

        pairs.append((img_path, text))
        kept += 1

    print(f"Parsed {kept} usable lines "
          f"({skipped_status} skipped for bad segmentation, "
          f"{skipped_missing} skipped for missing image files).")

    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.val_split))
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]

    def write(pairs, filename):
        out_path = os.path.join(args.out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for img_path, text in pairs:
                f.write(f"{img_path}\t{text}\n")
        print(f"Wrote {len(pairs)} lines to {out_path}")

    write(train_pairs, "train_list.txt")
    write(val_pairs, "val_list.txt")


if __name__ == "__main__":
    main()