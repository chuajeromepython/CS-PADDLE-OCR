"""
preprocess.py -- optional page clean-up before PaddleOCR.

Targets two problems seen on notebook photos:
  1. Ruled lines cross the handwriting (descenders of y/p/g, tops of t/d).
  2. Faint bleed-through writing from the back of the page gets read as text
     (e.g. the junk token "1008256" above the date on test_img3).

Idea: flatten the lighting, measure how DARK each pixel is relative to the
paper, then keep only pixels about as dark as real ink. Bleed-through and most
ruled lines are much lighter than pen ink. Remaining ruled lines (long, thin,
horizontal) are removed unless the pixel is strong ink, so strokes that cross
a line stay intact.
"""
import cv2
import numpy as np


def flatten_background(gray):
    k = max(31, (min(gray.shape) // 20) | 1)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), k)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)
    return 255.0 - norm  # darkness: 0 = paper, 255 = black


def _drop_edge_strips(keep):
    """Remove tall dark blobs touching the left/right/top/bottom border: that is
    the page edge or the table behind it, never handwriting."""
    H, W = keep.shape
    mask = (keep > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = keep.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if touches and (h > 0.08 * H or w > 0.25 * W) and area > 0.002 * W * H * 0.1:
            out[lab == i] = 0
    return out


def paper_mask(gray):
    """Boolean mask of the paper (largest bright region), or None if no clear page.
    Used to whiten wood grain / fabric / hands around the page."""
    H, W = gray.shape
    small = cv2.resize(gray, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.uint8) * 255
    if comp.sum() / 255 < 0.25 * comp.size:
        return None
    # fill holes (text, pen) so they stay inside the page
    ff = comp.copy(); fm = np.zeros((comp.shape[0] + 2, comp.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, fm, (0, 0), 255)
    comp = comp | cv2.bitwise_not(ff)
    # no erosion: text can sit within a few pixels of the paper edge (e.g. a P.S. line)
    return cv2.resize(comp, (W, H), interpolation=cv2.INTER_NEAREST) > 0


def clean_page(img_bgr, ink_frac=0.40, remove_rules=True, crop_to_paper=True):
    """Return a cleaned BGR image (black ink on white paper).

    ink_frac: pixels darker than ink_frac * (page's typical ink darkness) are
      kept; lighter ones (bleed-through, faint rules) become white. Lower it if
      light pen strokes disappear; raise it if ghost text survives.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    d = flatten_background(gray)

    pm = paper_mask(gray) if crop_to_paper else None
    if pm is not None:
        d = np.where(pm, d, 0.0).astype(np.float32)
    ink_level = float(np.percentile(d[pm] if pm is not None else d, 99.5))  # what "real ink" looks like on this page
    t_ink = max(25.0, ink_frac * ink_level)

    keep = d.copy()
    if remove_rules:
        H, W = d.shape
        cand = (d > 0.18 * ink_level).astype(np.uint8)   # ink + rules + ghost
        klen = max(25, W // 14)
        horiz = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
        rules = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        strong = d > 0.75 * ink_level                   # strokes crossing a rule survive
        keep[(rules > 0) & ~strong] = 0

    keep[keep < t_ink] = 0
    keep = _drop_edge_strips(keep)
    out = 255.0 - np.clip((keep / max(ink_level, 1.0)) * 255.0 * 1.15, 0, 255)
    out = out.astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def preprocess_image(image_path, out_dir=None, max_side=2000, **kwargs):
    """Drop-in for a path-based pipeline: downscale, clean, save a PNG next to the
    original (in debug_output/) and return its path. kwargs go to clean_page()."""
    import os
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = clean_page(img, **kwargs)
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(image_path)), "debug_output")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}_clean.png")
    cv2.imwrite(out_path, out)
    return out_path