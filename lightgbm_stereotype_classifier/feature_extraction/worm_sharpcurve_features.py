# -*- coding: utf-8 -*-
# Number of sharp turns of the contour, aggregated at the worm level (nsharp family of the final model).
#
# Motivation. A C-shape that is contracted and motionless in its clip is indistinguishable from a Walnut by frame shape,
# global border curvature and dynamics. However, its contour keeps several sharp turns (kinks) where the body folds
# onto itself, typically 2 to 4, whereas a Walnut is round and has none. These sharp turns are therefore counted per
# frame and then aggregated over the whole sequence of the worm.
#
# For each frame, on the smoothed binary mask, the contour is resampled at 200 uniform points. At each point the local
# turning angle is measured over a window (about 4.5 percent of the perimeter), and turning peaks whose angle exceeds
# 0.9 radian and that are at least 8 percent of the perimeter apart are counted. At the worm level, the maximum and the
# median of this count over its frames are kept (worm_nsharp_max, worm_nsharp_med). Since the split is done per worm,
# no information leaks between splits.
#
# Measured effect. This family corrects the last contracted C-shape read as Walnut, without affecting the recovered
# Walnut worms or the external test. Validation C-shape 47/50 to 48/50 and Walnut 50/50, test unchanged at 0.9600. AUC
# of worm_nsharp_max Walnut-versus-C-shape on the validation set 0.867, median number of turns 0 for Walnut versus 3
# for C-shape.
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

N_POINTS = 200
ANGLE_THR = 0.9          # radian, sharp-turn threshold
MIN_SEP = 0.08           # minimum separation between peaks, as a fraction of the perimeter
WIN = 0.045              # angle measurement window, as a fraction of the perimeter


def _index(folder):
    idx = {}
    for p in glob.glob(os.path.join(folder, '**', '*.png'), recursive=True):
        idx.setdefault(os.path.basename(p), p)
    return idx


def _worm_id(basename):
    return re.sub(r'_[Ff]rame_\d+_in_\d+.*$', '', os.path.basename(str(basename)))


def frame_n_sharp(mask_path):
    m = cv2.imread(mask_path, 0)
    if m is None:
        return None
    b = (m > 127).astype(np.uint8)
    if b.sum() < 300:
        return None
    b = (cv2.GaussianBlur(b * 255, (5, 5), 0) > 127).astype(np.uint8)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).squeeze()
    if c.ndim != 2 or len(c) < 40:
        return None
    n = N_POINTS
    idx = np.linspace(0, len(c), n, endpoint=False).astype(int)
    P = c[idx].astype(float)
    step = max(2, int(WIN * n))
    ang = np.zeros(n)
    for i in range(n):
        v1 = P[i] - P[(i - step) % n]
        v2 = P[(i + step) % n] - P[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        ang[i] = np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
    k = 2
    angs = np.convolve(np.r_[ang[-k:], ang, ang[:k]], np.ones(2 * k + 1) / (2 * k + 1), 'same')[k:-k]
    peaks = []
    mind = int(MIN_SEP * n)
    for i in np.argsort(-angs):
        if angs[i] < ANGLE_THR:
            break
        if all(min(abs(i - p), n - abs(i - p)) >= mind for p in peaks):
            peaks.append(i)
    return len(peaks)


def build_worm_frames(mask_index):
    wf = {}
    for base in mask_index:
        wf.setdefault(_worm_id(base), []).append(base)
    return wf


def worm_stats(worm_id, worm_frames, mask_index, cache):
    if worm_id in cache:
        return cache[worm_id]
    vals = []
    for f in worm_frames.get(worm_id, []):
        v = frame_n_sharp(mask_index[f])
        if v is not None:
            vals.append(v)
    v = (np.nan, np.nan) if len(vals) < 2 else (float(np.max(vals)), float(np.median(vals)))
    cache[worm_id] = v
    return v


def run_split(split, worm_frames, mask_index, cache):
    ref = pd.read_csv(os.path.join(NM, 'valley_%s.csv' % split), sep=';')
    rows = []
    for im in ref['image']:
        base = os.path.basename(str(im))
        nmax, nmed = worm_stats(_worm_id(base), worm_frames, mask_index, cache)
        rows.append([base, nmax, nmed])
    out = pd.DataFrame(rows, columns=['image', 'worm_nsharp_max', 'worm_nsharp_med'])
    out.to_csv(os.path.join(NM, 'nsharp_%s.csv' % split), sep=';', index=False)
    print('%s : %d rows written -> new_metrics_out/nsharp_%s.csv' % (split, len(out), split))


if __name__ == '__main__':
    mask_index = _index(MASKS)
    worm_frames = build_worm_frames(mask_index)
    cache = {}
    splits = sys.argv[1:] or [s for s in os.environ.get('FLATMORPH_SPLITS', 'train,valid,test').split(',') if s]   # FLATMORPH_SPLITS, default train,valid,test
    for sp in splits:
        run_split(sp, worm_frames, mask_index, cache)
