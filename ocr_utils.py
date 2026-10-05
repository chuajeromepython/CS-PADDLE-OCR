"""
ocr_utils.py

Reading-order fix for PaddleOCR results on handwriting.

PaddleOCR returns boxes roughly sorted top-to-bottom. In handwriting, words on
the same line sit at slightly different heights, so a later word can sort first
(e.g. "Chua / Jerome / S." instead of "Jerome S. Chua"). `ordered_lines` groups
boxes into rows by vertical overlap, then sorts each row left-to-right and
joins its words with spaces.
"""

import numpy as np


def _boxes_from_page(page, n):
    """Return an (n, 4) array of x0, y0, x1, y1 boxes aligned with rec_texts, or None."""
    getter = page.get if hasattr(page, "get") else (lambda k, d=None: page[k] if k in page else d)
    for key in ("rec_boxes", "rec_polys", "dt_polys"):
        raw = getter(key, None)
        if raw is None or len(raw) != n:
            continue
        boxes = []
        for b in raw:
            a = np.asarray(b, dtype=float)
            if a.ndim == 1 and a.size == 4:          # already x0, y0, x1, y1
                boxes.append(a)
            else:                                     # polygon of points
                pts = a.reshape(-1, 2)
                boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        return np.array(boxes)
    return None


def group_rows(boxes, overlap_thresh=0.5):
    """Group box indices into rows. A box joins a row if it overlaps the row's
    vertical band by >= overlap_thresh of the smaller height."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows = []  # each: {"idx": [...], "y0": mean y0, "y1": mean y1}
    for i in order:
        y0, y1 = boxes[i][1], boxes[i][3]
        h = max(y1 - y0, 1e-6)
        placed = False
        for row in rows:
            rh = max(row["y1"] - row["y0"], 1e-6)
            overlap = min(y1, row["y1"]) - max(y0, row["y0"])
            if overlap / min(h, rh) >= overlap_thresh:
                row["idx"].append(i)
                k = len(row["idx"])
                row["y0"] += (y0 - row["y0"]) / k
                row["y1"] += (y1 - row["y1"]) / k
                placed = True
                break
        if not placed:
            rows.append({"idx": [i], "y0": y0, "y1": y1})
    rows.sort(key=lambda r: (r["y0"] + r["y1"]) / 2)
    return [sorted(r["idx"], key=lambda i: boxes[i][0]) for r in rows]


def ordered_lines(page):
    """Return (lines, confidences): one string per visual row, in reading order.
    Falls back to PaddleOCR's own order if box geometry isn't available."""
    getter = page.get if hasattr(page, "get") else (lambda k, d=None: page[k])
    texts = list(getter("rec_texts", []))
    scores = list(getter("rec_scores", []))
    boxes = _boxes_from_page(page, len(texts)) if texts else None
    if boxes is None:
        return texts, scores
    lines, confs = [], []
    for row in group_rows(boxes):
        lines.append(" ".join(texts[i] for i in row if texts[i].strip()))
        confs.append(float(np.mean([scores[i] for i in row])))
    return lines, confs


def filter_by_score(page, thresh):
    """Drop recognized regions whose confidence is below `thresh`.
    Doing this after prediction (instead of via PaddleOCR's text_rec_score_thresh)
    lets one OCR run be re-scored at many thresholds. Returns a plain dict that
    ordered_lines() accepts."""
    getter = page.get if hasattr(page, "get") else (lambda k, d=None: page[k])
    texts = list(getter("rec_texts", []))
    scores = list(getter("rec_scores", []))
    boxes = _boxes_from_page(page, len(texts)) if texts else None
    keep = [i for i, s in enumerate(scores) if s >= thresh]
    out = {
        "rec_texts": [texts[i] for i in keep],
        "rec_scores": [scores[i] for i in keep],
    }
    if boxes is not None:
        out["rec_boxes"] = [boxes[i] for i in keep]
    return out

def drop_edge_fragments(page, img_width, margin=0.08, max_chars=3):
    """Drop short text (<= max_chars) whose box sits in the outer `margin` fraction
    of the image on the left or right. Catches words cut off at the page edge, e.g.
    the neighbouring page peeking in beside a notebook spread. margin <= 0 disables."""
    if margin <= 0:
        return page
    getter = page.get if hasattr(page, "get") else (lambda k, d=None: page[k])
    texts = list(getter("rec_texts", []))
    scores = list(getter("rec_scores", []))
    boxes = _boxes_from_page(page, len(texts)) if texts else None
    if boxes is None:
        return page
    lo, hi = margin * img_width, (1 - margin) * img_width
    keep = [i for i, t in enumerate(texts)
            if not (len(t.strip()) <= max_chars and (boxes[i][0] >= hi or boxes[i][2] <= lo))]
    return {
        "rec_texts": [texts[i] for i in keep],
        "rec_scores": [scores[i] for i in keep],
        "rec_boxes": [boxes[i] for i in keep],
    }