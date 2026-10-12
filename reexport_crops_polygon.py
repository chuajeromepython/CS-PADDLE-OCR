"""
Re-export journal_crops so they look like what the recognizer sees at inference.

Why: the existing crops were saved as axis-aligned boxes (+4 px pad) from
`rec_boxes`. Those boxes pick up slices of the line above and below (see
test_img16_r03_00.png). At inference PaddleOCR instead cuts the detector's
polygon with cv2.minAreaRect + a perspective warp, no padding, so the
recognizer never sees that bleed. Training on bleed and testing on clean
crops is a train/inference mismatch.

This script runs the same detector and the same row grouping as
ocr_training/calibrate_flagging.py, so the file names (<image>_rRR_KK.png)
come out identical and train_list / val_list / test_list stay valid. Only
the pixels change. It prints a warning for any photo whose crop count or
names differ from what is already in journal_crops/.

Usage (from the repo root, inside your paddle-env):
    python reexport_crops_polygon.py journal_photos journal_crops_poly
Then check a few crops by eye, and if they look right:
    - move journal_crops_poly/*.png over journal_crops/ (keep the *_list.txt files)
    - commit, then rerun the Colab notebook.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                              # repo root (ocr_utils.py)
sys.path.insert(0, os.path.join(HERE, ".."))          # in case this lives in ocr_training/

from ocr_utils import _boxes_from_page, group_rows  # noqa: E402

MAX_SIDE = 2000  # CS-PADDLE-OCR.py downsizes to this before OCR; keep identical


def stock_detector_name(lang="en"):
    """Same helper as in CS-PADDLE-OCR.py / calibrate_flagging.py."""
    try:
        from paddleocr import PaddleOCR
        det_name, _ = PaddleOCR._get_ocr_model_names(None, lang, None)
        return det_name
    except Exception:
        return None


def load_resized(path, max_side=MAX_SIDE):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def rotate_crop(img, pts):
    """Perspective-warp 4 ordered points to a rectangle (PaddleX get_rotate_crop_image)."""
    pts = np.asarray(pts, dtype=np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts, std)
    dst = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    if dst.shape[0] / max(dst.shape[1], 1) >= 1.5:
        dst = np.rot90(dst)
    return dst


def minarea_rect_crop(img, poly):
    """Same as PaddleX CropByPolys.get_minarea_rect_crop (what the pipeline uses)."""
    rect = cv2.minAreaRect(np.array(poly).astype(np.int32))
    p = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
    ia, id_ = (0, 1) if p[1][1] > p[0][1] else (1, 0)
    ib, ic = (2, 3) if p[3][1] > p[2][1] else (3, 2)
    return rotate_crop(img, np.array([p[ia], p[ib], p[ic], p[id_]]))


def main(image_dir, out_dir, existing_dir="journal_crops"):
    from paddleocr import PaddleOCR

    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=True, enable_mkldnn=False)
    det = stock_detector_name("en")
    if det:
        kw["text_detection_model_name"] = det
    else:
        kw["lang"] = "en"
    ocr = PaddleOCR(**kw)

    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        img = load_resized(os.path.join(image_dir, fname))
        res = ocr.predict(img)
        page = res[0] if res else {}
        texts = list(page.get("rec_texts", []))
        polys = page.get("rec_polys", None)
        boxes = _boxes_from_page(page, len(texts)) if texts else None
        if boxes is None or polys is None or len(polys) != len(texts):
            print(f"{fname}: no aligned polygons, skipped")
            continue

        names = []
        for r, row in enumerate(group_rows(boxes)):
            for k, i in enumerate(row):
                name = f"{base}_r{r:02d}_{k:02d}.png"
                crop = minarea_rect_crop(img, polys[i])
                if crop.shape[0] < 4 or crop.shape[1] < 4:
                    continue
                ok, buf = cv2.imencode(".png", crop)
                if ok:
                    buf.tofile(os.path.join(out_dir, name))
                    names.append(name)

        old = sorted(n for n in os.listdir(existing_dir)
                     if n.startswith(base + "_r") and n.endswith(".png")) \
            if os.path.isdir(existing_dir) else []
        flag = "" if (not old or sorted(names) == old) else \
            f"   WARNING: names differ from existing ({len(old)} old crops) - check labels!"
        print(f"{fname}: {len(names)} crops{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", help="folder of journal photos, e.g. journal_photos")
    ap.add_argument("out_dir", help="where to write the new crops, e.g. journal_crops_poly")
    ap.add_argument("--existing", default="journal_crops",
                    help="folder with the current crops, used only for the name check")
    a = ap.parse_args()
    main(a.image_dir, a.out_dir, a.existing)
