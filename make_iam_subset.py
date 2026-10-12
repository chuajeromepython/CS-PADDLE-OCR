r"""
make_iam_subset.py

Picks a small, diverse sample of IAM *line* images and packs them into one
zip that is small enough to upload straight into Colab (about 60-100 MB).

Step 1 - make line-level labels with your existing script (IAM line files,
         NOT the word files you used for the old IAM model). Download
         ascii/lines.txt and data/lines.tgz from the IAM site (same account as
         words.zip), extract lines.tgz, then:

    python ocr_training/prepare_iam_data.py ^
        --lines-txt C:\IAM\ascii\lines.txt --format line ^
        --img-root C:\IAM\lines --out-dir iam_lines_labels --val-split 0.0

Step 2 - sample and zip:

    python ocr_training/make_iam_subset.py ^
        --lists iam_lines_labels\train_list.txt --n 2500 --out iam_subset

Result: iam_subset.zip  (upload this file in the notebook's IAM step).
Inside: iam_labels.txt and iam_lines/<name>.png
"""
import argparse
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True,
                    help="label list(s) from prepare_iam_data.py (path<TAB>text per line)")
    ap.add_argument("--n", type=int, default=2500, help="how many lines to keep")
    ap.add_argument("--out", default="iam_subset", help="output folder name (zip gets the same name)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for lst in args.lists:
        with open(lst, encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if "\t" not in ln:
                    continue
                path, text = ln.split("\t", 1)
                if text.strip() and os.path.exists(path):
                    rows.append((path, text))
    print(f"{len(rows)} usable IAM lines found")
    if not rows:
        raise SystemExit("No usable lines. Check that the image paths in the list files exist on this PC.")

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    img_dir = os.path.join(args.out, "iam_lines")
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(img_dir)

    seen, kept = set(), []
    for path, text in rows:
        name = os.path.basename(path)
        if name in seen:
            continue
        seen.add(name)
        shutil.copy(path, os.path.join(img_dir, name))
        kept.append((f"iam_lines/{name}", text))

    with open(os.path.join(args.out, "iam_labels.txt"), "w", encoding="utf-8") as f:
        for p, t in kept:
            f.write(f"{p}\t{t}\n")

    zip_path = shutil.make_archive(args.out, "zip", root_dir=args.out)
    mb = os.path.getsize(zip_path) / 1e6
    print(f"Wrote {len(kept)} lines -> {zip_path} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
