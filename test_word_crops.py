"""
Sanity-check an exported recognition model on IAM *word crops*, the data it was
fine-tuned on, reading images straight out of words.zip (no unzipping needed).

  python test_word_crops.py --words-dir C:\\path\\to\\iam_data\\words --labels val_list.txt ^
         --rec-model-dir .\\iam_finetune_infer --n 300

(--words-dir is the already-unzipped folder that contains a01, a02, ...;
 use --zip words.zip instead if you still have it zipped.)

Interpreting the result:
  ~80-85% exact match  -> export is fine; failures on your photos are a
                          domain/format problem (see notes), not a broken model.
  Far lower (<30%)     -> export/preprocessing problem; stop and debug that first.
"""
import argparse, random, zipfile, os, sys
import numpy as np
import cv2


def load_samples(zip_path, words_dir, label_file, n, seed):
    z = zipfile.ZipFile(zip_path) if zip_path else None
    names = set(z.namelist()) if z else None
    rows = []
    for line in open(label_file, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        path, label = line.split("\t", 1)
        # '/content/drive/MyDrive/iam/words/e06/..' or '/content/words/e06/..' -> 'e06/..'
        if "/words/" not in path:
            continue
        rest = path.split("/words/", 1)[1]
        if z:
            member = "words/" + rest
            if member in names:
                rows.append((member, label))
        else:
            full = os.path.join(words_dir, *rest.split("/"))
            if os.path.exists(full):
                rows.append((full, label))
    if not rows:
        sys.exit("No label rows matched image files. Check --words-dir/--zip and --labels.")
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    imgs, labels = [], []
    for src, label in rows:
        if z:
            im = cv2.imdecode(np.frombuffer(z.read(src), np.uint8), cv2.IMREAD_COLOR)
        else:
            im = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
            labels.append(label)
    return imgs, labels


def run(model_name, model_dir, imgs):
    from paddleocr import TextRecognition
    kw = {"model_name": model_name}
    if model_dir:
        kw["model_dir"] = model_dir
    rec = TextRecognition(**kw)
    out = []
    for i in range(0, len(imgs), 16):
        for r in rec.predict(imgs[i:i + 16]):
            out.append(r["rec_text"])
    return out


def score(preds, labels):
    exact = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    exact_ci = sum(p.lower() == l.lower() for p, l in zip(preds, labels)) / len(labels)
    try:
        import jiwer
        cer = jiwer.cer(labels, preds)
    except Exception:
        cer = float("nan")
    return exact, exact_ci, cer


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--words-dir", help="unzipped words folder (contains a01, a02, ...)")
    src.add_argument("--zip", help="path to words.zip")
    ap.add_argument("--labels", required=True, help="val_list.txt (path<TAB>label)")
    ap.add_argument("--rec-model-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stock", action="store_true", help="skip the stock-model comparison")
    ap.add_argument("--stock-name", default="en_PP-OCRv4_mobile_rec",
                    help="stock model to compare against (the base you fine-tuned from)")
    args = ap.parse_args()

    imgs, labels = load_samples(args.zip, args.words_dir, args.labels, args.n, args.seed)
    print(f"Loaded {len(imgs)} word crops.\n")

    ft = run("en_PP-OCRv4_mobile_rec", args.rec_model_dir, imgs)
    e, eci, cer = score(ft, labels)
    print(f"FINE-TUNED   exact {e:6.1%} | case-insens {eci:6.1%} | CER {cer:.3f}")

    if not args.no_stock:
        st = run(args.stock_name, None, imgs)
        e2, eci2, cer2 = score(st, labels)
        print(f"STOCK ({args.stock_name}) exact {e2:6.1%} | case-insens {eci2:6.1%} | CER {cer2:.3f}")

    print("\nFirst 15 examples (truth | fine-tuned" + ("" if args.no_stock else " | stock") + "):")
    for i in range(min(15, len(labels))):
        row = f"  {labels[i]!r:18} | {ft[i]!r:18}"
        if not args.no_stock:
            row += f" | {st[i]!r}"
        print(row)


if __name__ == "__main__":
    main()