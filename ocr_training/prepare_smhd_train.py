r"""
prepare_smhd_train.py

Turns the SAFE TRAINING POOL of SMHD (see prepare_smhd_eval.py) into line crops +
label lists for the v7 Colab notebook:

    other_writers/crops/*.png
    other_writers/train_list.txt      (path<TAB>text, paths relative to the repo root)
    other_writers/val_list.txt
    other_writers/prep_report.json    (what was accepted / skipped and why)

How it works
  1. For every pool page, run the same PaddleOCR detector your other scripts use.
  2. Group detected boxes into rows (ocr_utils.group_rows) and merge each row into ONE
     line crop (all boxes of the row, cut with the same minAreaRect + perspective warp
     your inference pipeline uses). Wide word spacing therefore does not split a line.
  3. Rows are matched to transcription lines. Strict mode: row count equals line count and
     the stock OCR roughly agrees with the text (no line is filtered). Fuzzy fallback: rows
     are matched to lines by text similarity, so lines the stock OCR gets very wrong are
     left out. Fuzzy pages therefore lean toward easier lines; the report counts both.
  4. Lines containing '#' (crossed-out words) are dropped. Labels are cleaned the same
     way as in prepare_smhd_eval.py. Spelling mistakes are kept.
  5. Pages are split into train / val in consecutive blocks with a gap between them,
     because SMHD has no writer IDs and neighbouring pages are probably the same writer.

Usage (repo root, inside your paddle-env):
    python ocr_training/prepare_smhd_train.py --smhd SMHD --manifest eval_others/split_manifest.json --out other_writers

It can be stopped (Ctrl+C) and restarted; finished pages are remembered.
Add --limit 10 to try it on 10 pages first.
"""
import argparse
import difflib
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from ocr_utils import _boxes_from_page, group_rows                                  # noqa: E402
from reexport_crops_polygon import load_resized, minarea_rect_crop, stock_detector_name  # noqa: E402
from prepare_smhd_eval import clean_line, find_subject, page_id                      # noqa: E402


def make_ocr():
    from paddleocr import PaddleOCR
    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=True, enable_mkldnn=False)
    det = stock_detector_name("en")
    if det:
        kw["text_detection_model_name"] = det
    else:
        kw["lang"] = "en"
    return PaddleOCR(**kw)


def detect_rows(ocr, img):
    """Returns [(points Nx2, text), ...] one entry per visual row, top to bottom; or None."""
    res = ocr.predict(img)
    page = res[0] if res else {}
    texts = list(page.get("rec_texts", []))
    polys = page.get("rec_polys", None)
    boxes = _boxes_from_page(page, len(texts)) if texts else None
    if boxes is None or polys is None or len(polys) != len(texts):
        return None
    rows = []
    for row in group_rows(boxes):
        pts = np.concatenate([np.asarray(polys[i], dtype=float).reshape(-1, 2) for i in row])
        rows.append((pts, " ".join(texts[i] for i in row if texts[i].strip())))
    return rows


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def mean_similarity(row_texts, labels):
    sims = [difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
            for a, b in zip(row_texts, labels)]
    return sum(sims) / max(len(sims), 1)


def fuzzy_align(row_texts, labels, min_sim):
    """Monotone matching of detected rows to transcription lines by text similarity.
    Rows or lines may be left unmatched (split/merged lines, headers, noise). Only
    pairs with similarity >= min_sim and a similar text length are returned, as
    (row_index, label_index)."""
    R, T = len(row_texts), len(labels)
    def sim(r, l):
        nr, nl = norm(r), norm(l)
        if not nr or not nl or not (0.65 <= len(nr) / len(nl) <= 1.5):
            return 0.0      # a row much longer/shorter than the line is a merged or split line, never a match
        return difflib.SequenceMatcher(None, nr, nl).ratio()
    S = [[sim(r, l) for l in labels] for r in row_texts]
    dp = [[0.0] * (T + 1) for _ in range(R + 1)]
    for i in range(1, R + 1):
        for j in range(1, T + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            if S[i - 1][j - 1] >= min_sim:
                best = max(best, dp[i - 1][j - 1] + (S[i - 1][j - 1] - min_sim + 0.01))
            dp[i][j] = best
    pairs, i, j = [], R, T
    while i > 0 and j > 0:
        if S[i - 1][j - 1] >= min_sim and abs(dp[i][j] - (dp[i - 1][j - 1] + S[i - 1][j - 1] - min_sim + 0.01)) < 1e-9:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def plausible(crop, row_text, label):
    """Cheap per-line guards that do not depend on how well the OCR read the handwriting:
    the OCR text must have a similar length to the transcription line (catches fragments and
    merged lines), and the crop must be wide enough for that many characters."""
    nr, nl = len(norm(row_text)), len(norm(label))
    if nl == 0 or not (0.5 <= nr / nl <= 2.0):
        return False
    h, w = crop.shape[:2]
    return (w / max(h, 1)) / max(len(label), 1) >= 0.12


def read_label_lines(path):
    raw = [l.rstrip("\n") for l in open(path, encoding="utf-8", errors="replace")]
    return [l for l in raw if l.strip()]


def split_pages(pages, val_frac, gap):
    """pages: list of 'Folder/0123.jpg'. Returns (train_set, val_set, gap_set)."""
    by_subj = {}
    for rel in pages:
        by_subj.setdefault(rel.split("/")[0], []).append(rel)
    train, val, gaps = set(), set(), set()
    for folder, items in by_subj.items():
        items = sorted(items, key=lambda r: page_id(os.path.basename(r)))
        n = len(items)
        k = max(2, round(n * val_frac)) if n >= 6 else 0
        start = min(int(0.85 * n), n - k) if k else n
        block = items[start:start + k]
        lo = page_id(os.path.basename(block[0])) - gap if block else None
        hi = page_id(os.path.basename(block[-1])) + gap if block else None
        for rel in items:
            if rel in block:
                val.add(rel)
            elif block and lo <= page_id(os.path.basename(rel)) <= hi:
                gaps.add(rel)
            else:
                train.add(rel)
    return train, val, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smhd", default="SMHD")
    ap.add_argument("--manifest", default="eval_others/split_manifest.json")
    ap.add_argument("--out", default="other_writers")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pool pages (0 = all)")
    ap.add_argument("--min-sim", type=float, default=0.45,
                    help="min mean similarity between stock OCR and transcription for a page to be accepted")
    ap.add_argument("--align", choices=["auto", "strict", "fuzzy"], default="auto",
                    help="strict = only pages whose row count equals the transcription line count; "
                         "auto = strict first, then match rows to lines by text similarity")
    ap.add_argument("--fuzzy-sim", type=float, default=0.5,
                    help="min similarity for a single row<->line match in fuzzy mode")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--gap", type=int, default=3)
    a = ap.parse_args()

    manifest = json.load(open(a.manifest, encoding="utf-8"))
    pool = manifest["train_pool"]
    if a.limit:
        pool = pool[: a.limit]
    crops_dir = os.path.join(a.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    done_path = os.path.join(a.out, "pages_done.json")
    done = json.load(open(done_path, encoding="utf-8")) if os.path.exists(done_path) else {}

    todo = [p for p in pool if p not in done]
    print(f"{len(pool)} pool pages, {len(done)} already done, {len(todo)} to process")
    ocr = make_ocr() if todo else None

    for n, rel in enumerate(todo, 1):
        folder, fn = rel.split("/")
        base = os.path.splitext(fn)[0]
        short = find_subject(folder)[0]
        labels_raw = read_label_lines(os.path.join(a.smhd, folder, base + ".txt"))
        img = load_resized(os.path.join(a.smhd, folder, fn))
        rows = detect_rows(ocr, img)
        entry = {"labels": len(labels_raw)}
        pairs, mode = None, None
        if rows is None:
            entry.update(status="skipped: no aligned polygons")
        else:
            cleaned = [clean_line(l) for l in labels_raw]
            entry.update(rows=len(rows))
            if len(rows) == len(labels_raw):
                sim = mean_similarity([t for _, t in rows], cleaned)
                entry.update(sim=round(sim, 3))
                if sim >= a.min_sim:
                    pairs, mode = [(i, i) for i in range(len(rows))], "strict"
            if pairs is None and a.align != "strict":
                fz = fuzzy_align([t for _, t in rows], cleaned, a.fuzzy_sim)
                if len(fz) >= max(3, int(0.4 * len(labels_raw))):
                    pairs, mode = fz, "fuzzy"
            if pairs is None:
                entry.update(status=f"skipped: {len(rows)} rows vs {len(labels_raw)} transcription lines, no confident match")
            else:
                lines, rejected = [], 0
                for ri, lj in pairs:
                    if "#" in labels_raw[lj] or not cleaned[lj]:
                        continue
                    crop = minarea_rect_crop(img, rows[ri][0])
                    if crop.shape[0] < 8 or crop.shape[1] < 8:
                        continue
                    if not plausible(crop, rows[ri][1], cleaned[lj]):
                        rejected += 1
                        continue
                    name = f"smhd_{short}_{base}_r{lj:02d}.png"
                    ok, buf = cv2.imencode(".png", crop)
                    if ok:
                        buf.tofile(os.path.join(crops_dir, name))
                        lines.append([name, cleaned[lj]])
                entry.update(status="accepted", mode=mode, lines=lines, rejected=rejected)
        done[rel] = entry
        json.dump(done, open(done_path, "w", encoding="utf-8"))
        print(f"[{n}/{len(todo)}] {rel}: {entry['status']}" + (f" [{entry.get('mode')}] ({len(entry.get('lines', []))} lines)" if entry["status"] == "accepted" else ""))

    # ------- split accepted pages into train / val and write the lists
    accepted = [p for p in pool if done.get(p, {}).get("status") == "accepted"]
    train, val, gaps = split_pages(accepted, a.val_frac, a.gap)
    rel_dir = os.path.basename(os.path.normpath(a.out))

    def write(path, pages):
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for p in sorted(pages):
                for name, text in done[p]["lines"]:
                    f.write(f"{rel_dir}/crops/{name}\t{text}\n")
                    n += 1
        return n

    n_tr = write(os.path.join(a.out, "train_list.txt"), train)
    n_va = write(os.path.join(a.out, "val_list.txt"), val)

    reasons = {}
    for p in pool:
        st = done.get(p, {}).get("status", "not processed")
        key = st.split(" (")[0]
        key = re.sub(r"\d+ rows vs \d+ transcription lines", "row count differs from transcription", key)
        key = re.sub(r"similarity [\d.]+", "similarity too low", key)
        reasons[key] = reasons.get(key, 0) + 1
    report = {"pool_pages": len(pool), "page_status": reasons, "train_pages": len(train), "val_pages": len(val),
              "accepted_strict": sum(1 for p in pool if done.get(p, {}).get("mode") == "strict"),
              "accepted_fuzzy": sum(1 for p in pool if done.get(p, {}).get("mode") == "fuzzy"),
              "lines_rejected_by_guards": sum(done.get(p, {}).get("rejected", 0) for p in pool),
              "gap_pages_dropped": len(gaps), "train_lines": n_tr, "val_lines": n_va}
    json.dump(report, open(os.path.join(a.out, "prep_report.json"), "w"), indent=1)
    print("\n" + json.dumps(report, indent=1))
    print(f"\nWrote {a.out}/train_list.txt ({n_tr} lines) and {a.out}/val_list.txt ({n_va} lines).")
    print("Push the 'other_writers' folder to GitHub, then run the v7 notebook.")


if __name__ == "__main__":
    main()