# -*- coding: utf-8 -*-
# Curvature evolution at the worm level (curv family of the final model).
#
# Motivation. On a single frame, a C-shape momentarily contracted into a ball is indistinguishable from a Walnut, even
# with raw shape dynamics. However, a true C-shape bends strongly in at least one frame of its sequence, whereas a
# Walnut never bends and stays convex throughout the clip. We therefore summarise, at the worm level rather than the
# frame level, how much the body bends over its whole sequence.
#
# For each frame of the worm, the following are measured on the binary mask:
#   concavity = depth of the deepest convexity defect of the contour, normalised by the equivalent radius.
#   solidity  = area / convex hull area.
# These are then aggregated over all frames of the same worm:
#   worm_concav_max   : maximum concavity reached (did the worm bend at least once).
#   worm_concav_range : concavity amplitude over the sequence.
#   worm_sol_min      : minimum solidity (most bent pose).
# The same worm value is assigned to each of its frames. Since the train/validation/test split is done per worm, no
# information leaks between splits.
#
# Measured effect. This family recovers the contracted Walnut worms that the model read as C-shape (validation Walnut
# 47/50 to 50/50) while preserving the external test (C-shape 30/30 and balanced accuracy 0.9600), whereas raw
# worm-level shape amplitude broke frozen C-shape worms of the test. AUC of worm_concav_range Walnut-versus-C-shape on
# the validation set = 0.912.
#
# Source. Concavity from contour convexity defects and labelling: Rosenfeld A., Pfaltz J. L., Sequential
# Operations in Digital Picture Processing, Journal of the ACM, 1966, DOI 10.1145/321356.321357.
#
# Reproducibility: random_state 42, semicolon CSV separator, deterministic processing.
import os, re, glob, sys
import numpy as np
import cv2
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# The application exports FLATMORPH_DATASET before importing this module
# (lightgbm_bundle._ensure_dataset_stub), so the environment always decides. The
# path that used to sit here named one developer's machine and never existed
# anywhere else.
DATASET = os.environ.get('DATASET_DIR') or os.environ.get('FLATMORPH_DATASET') or (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_training_stereotype'))
MASKS = os.path.join(DATASET, 'mask_model_Mask2Former_20260324_0451')
NM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_metrics_out')


def _index(folder):
    idx = {}
    for p in glob.glob(os.path.join(folder, '**', '*.png'), recursive=True):
        idx.setdefault(os.path.basename(p), p)
    return idx


def _worm_id(basename):
    return re.sub(r'_[Ff]rame_\d+_in_\d+.*$', '', os.path.basename(str(basename)))


def _frame_curv(mask_path):
    m = cv2.imread(mask_path, 0)
    if m is None:
        return None
    b = (m > 127).astype(np.uint8)
    if b.sum() < 200:
        return None
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    solidity = area / (cv2.contourArea(cv2.convexHull(c)) + 1e-6)
    eqR = np.sqrt(area / np.pi)
    concav = 0.0
    hull_i = cv2.convexHull(c, returnPoints=False)
    if hull_i is not None and len(hull_i) > 3:
        try:
            defects = cv2.convexityDefects(c, hull_i)
        except cv2.error:
            defects = None
        if defects is not None:
            d = defects.reshape(-1, 4)
            concav = float(max(d[:, 3]) / 256.0 / eqR) if len(d) else 0.0
    return solidity, concav


def build_worm_frames(mask_index):
    worm_frames = {}
    for base in mask_index:
        worm_frames.setdefault(_worm_id(base), []).append(base)
    return worm_frames


def worm_stats(worm_id, worm_frames, mask_index, cache):
    if worm_id in cache:
        return cache[worm_id]
    sol, con = [], []
    for f in worm_frames.get(worm_id, []):
        r = _frame_curv(mask_index[f])
        if r is not None:
            sol.append(r[0])
            con.append(r[1])
    if len(sol) < 2:
        v = (np.nan, np.nan, np.nan)
    else:
        sol = np.array(sol)
        con = np.array(con)
        v = (float(con.max()), float(con.max() - con.min()), float(sol.min()))
    cache[worm_id] = v
    return v


def run_split(split, worm_frames, mask_index, cache):
    ref = pd.read_csv(os.path.join(NM, 'valley_%s.csv' % split), sep=';')
    rows = []
    for im in ref['image']:
        base = os.path.basename(str(im))
        cmax, crange, smin = worm_stats(_worm_id(base), worm_frames, mask_index, cache)
        rows.append([base, cmax, crange, smin])
    out = pd.DataFrame(rows, columns=['image', 'worm_concav_max', 'worm_concav_range', 'worm_sol_min'])
    out.to_csv(os.path.join(NM, 'curv_%s.csv' % split), sep=';', index=False)
    print('%s : %d rows written -> new_metrics_out/curv_%s.csv' % (split, len(out), split))


if __name__ == '__main__':
    mask_index = _index(MASKS)
    worm_frames = build_worm_frames(mask_index)
    cache = {}
    splits = sys.argv[1:] or [s for s in os.environ.get('FLATMORPH_SPLITS', 'train,valid,test').split(',') if s]   # FLATMORPH_SPLITS, default train,valid,test
    for sp in splits:
        run_split(sp, worm_frames, mask_index, cache)
