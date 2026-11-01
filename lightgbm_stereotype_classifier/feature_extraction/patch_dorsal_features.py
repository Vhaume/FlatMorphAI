# -*- coding: utf-8 -*-
# Descriptors of the dorsal dark patch (patchint family of the final model).
#
# Motivation. A strongly contracted Walnut carries on its back a compact muscular bulge that appears as a single,
# compact dark focus close to the body centre. The folded C-shape that resembles it shows, at the same location, a dark
# focus shifted towards the edge and elongated (the seam of the fold). This detail lies inside the worm and is not
# captured when working on the whole-well image, where the worm is under-resolved. The worm is therefore cropped on its
# mask, resized to 256 pixels and its local contrast equalised by CLAHE. Then, within the worm interior (eroded mask),
# the largest component clearly darker than the interior median is isolated and described.
#
# Outputs (one row per image, semicolon separator):
#   pdist     : distance from the patch centroid to the body centroid, normalised by the equivalent radius (low = central).
#   psolid    : patch solidity (area / convex hull area), high = compact.
#   pfrac     : patch area as a fraction of the body area.
#   pcontrast : luminance difference between the interior median and the patch, in units of interior standard deviation.
# AUC of pdist for separating Walnut from C-shape on the validation set = 0.805.
#
# Sources.
#   CLAHE : Zuiderveld K., Contrast Limited Adaptive Histogram Equalization, Graphics Gems IV, 1994,
#           DOI 10.5555/180895.180940.
#   Connected components (labelling) : Rosenfeld A., Pfaltz J. L., Sequential Operations in Digital Picture
#           Processing, Journal of the ACM, 1966, DOI 10.1145/321356.321357.
#
# Reproducibility: random_state 42, semicolon CSV separator, deterministic processing (no random sampling).
import os, glob, sys
import numpy as np
import cv2
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Folders. Adjust DATASET if needed. IMAGES holds the RGB images per class, MASKS the binary worm masks.
# The application exports FLATMORPH_DATASET before importing this module
# (lightgbm_bundle._ensure_dataset_stub), so the environment always decides. The
# path that used to sit here named one developer's machine and never existed
# anywhere else.
DATASET = os.environ.get('DATASET_DIR') or os.environ.get('FLATMORPH_DATASET') or (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_training_stereotype'))
IMAGES = os.path.join(DATASET, 'images')
MASKS = os.path.join(DATASET, 'mask_model_Mask2Former_20260324_0451')
# Reference files listing the images of each split (image column). The existing CSV files are reused.
NM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_metrics_out')


def _index(folder):
    idx = {}
    for p in glob.glob(os.path.join(folder, '**', '*.png'), recursive=True):
        idx.setdefault(os.path.basename(p), p)
    return idx


def _crop(img, b, pad=0.10, size=256):
    ys, xs = np.where(b > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    py, px = int(h * pad), int(w * pad)
    y0, y1 = max(0, y0 - py), min(img.shape[0], y1 + py)
    x0, x1 = max(0, x0 - px), min(img.shape[1], x1 + px)
    ci = cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC)
    cm = cv2.resize(b[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
    return ci, cm


def patch_features(image_path, mask_path):
    nan = (np.nan, np.nan, np.nan, np.nan)
    img = cv2.imread(image_path)
    m = cv2.imread(mask_path, 0)
    if img is None or m is None:
        return nan
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = (m > 127).astype(np.uint8)
    if b.sum() < 200:
        return nan
    img, b = _crop(img, b)
    eqR = np.sqrt(b.sum() / np.pi)
    er = cv2.erode(b, np.ones((3, 3), np.uint8), iterations=max(1, int(0.12 * eqR)))
    if er.sum() < 80:
        er = cv2.erode(b, np.ones((3, 3), np.uint8), iterations=max(1, int(0.06 * eqR)))
    if er.sum() < 40:
        return nan
    V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    V = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(V).astype(float)
    Vs = cv2.GaussianBlur(V, (0, 0), 2)
    med = np.median(Vs[er > 0])
    sd = Vs[er > 0].std() + 1e-6
    dark = np.clip((med - Vs) / sd, 0, None) * er
    darkn = dark / (dark.max() + 1e-6)
    thr = np.percentile(darkn[er > 0], 80)
    binm = cv2.morphologyEx(((darkn >= thr) & (er > 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    if n <= 1:
        return nan
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    ys, xs = np.where(comp > 0)
    by, bx = np.where(b > 0)
    pdist = float(np.sqrt((ys.mean() - by.mean()) ** 2 + (xs.mean() - bx.mean()) ** 2) / eqR)
    pfrac = float(comp.sum() / b.sum())
    pcontrast = float((med - Vs[comp > 0].mean()) / sd)
    psolid = np.nan
    cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    if len(cnt) >= 5:
        hull = cv2.contourArea(cv2.convexHull(cnt))
        psolid = float(cv2.contourArea(cnt) / (hull + 1e-6))
    return (pdist, psolid, pfrac, pcontrast)


def run_split(split, img_index, msk_index):
    ref = pd.read_csv(os.path.join(NM, 'valley_%s.csv' % split), sep=';')
    rows = []
    for im in ref['image']:
        base = os.path.basename(str(im))
        ip, mp = img_index.get(base), msk_index.get(base)
        feats = patch_features(ip, mp) if (ip and mp) else (np.nan, np.nan, np.nan, np.nan)
        rows.append([base, *feats])
    out = pd.DataFrame(rows, columns=['image', 'pdist', 'psolid', 'pfrac', 'pcontrast'])
    out.to_csv(os.path.join(NM, 'patchint_%s.csv' % split), sep=';', index=False)
    print('%s : %d rows written -> new_metrics_out/patchint_%s.csv' % (split, len(out), split))


if __name__ == '__main__':
    img_index = _index(IMAGES)
    msk_index = _index(MASKS)
    splits = sys.argv[1:] or [s for s in os.environ.get('FLATMORPH_SPLITS', 'train,valid,test').split(',') if s]   # FLATMORPH_SPLITS, default train,valid,test
    for sp in splits:
        run_split(sp, img_index, msk_index)
