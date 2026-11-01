#!/usr/bin/env python
# coding: utf-8
"""centerline_texture.py -- the pigment/texture profile read ALONG the true body centerline, single frame.

Why this exists: axial_pigment bins intensity along the straight principal axis, which on a curved or bent worm cuts
across the bend and mislabels head/middle/tail. This instead traces the medial-axis centerline (skeleton longest path,
the same tracer the border-evolution group uses), samples the grayscale head-to-tail along the real body, and reports
the SAME extremity-lightness cue (lighter head and tail vs the darker middle) plus the along-body texture banding. It is
the curvature-correct version of the lighter-tip cue from the expert review. Orientation-invariant, scale-free, single
frame (processed grayscale, the worm on black). No GPU. Backing: skeleton medial axis, Hu, IRE Trans. Info. Theory 1962,
DOI 10.1109/TIT.1962.1057692; dense profile resampling as in the project border-evolution group.
"""
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
from twist_border_evolution import _longest_path

CTEX_COLS = ['ctex_ends_vs_mid', 'ctex_ends_both_light', 'ctex_max_end_light', 'ctex_end_asym', 'ctex_profile_u',
             'ctex_band_std', 'ctex_band_amp']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def _centerline_intensity(gray):
    g = np.asarray(gray)
    worm = _largest((g > 8).astype(np.uint8))
    if worm.sum() < 60:
        return None
    path = _longest_path(skeletonize(worm > 0))
    if path is None or len(path) < 12:
        return None
    P = np.array(path, np.float32)
    inten = np.array([float(g[int(np.clip(y, 0, g.shape[0] - 1)), int(np.clip(x, 0, g.shape[1] - 1))]) for y, x in P], np.float32)
    return gaussian_filter1d(inten, 2.0)


def centerline_texture(gray):
    out = {k: np.nan for k in CTEX_COLS}
    prof = _centerline_intensity(gray)
    if prof is None or len(prof) < 12:
        return out
    n = len(prof); overall = float(prof.mean()) + 1e-6
    k = max(2, n // 5)
    end1 = float(prof[:k].mean()); end2 = float(prof[-k:].mean())
    mid = float(prof[k:-k].mean()) if n > 2 * k else float(prof.mean())
    out['ctex_ends_vs_mid'] = float((0.5 * (end1 + end2) - mid) / overall)            # both ends lighter than middle
    out['ctex_ends_both_light'] = float((min(end1, end2) - mid) / overall)            # both, not one
    out['ctex_max_end_light'] = float((max(end1, end2) - mid) / overall)              # at least one tip lighter (tail-only too)
    out['ctex_end_asym'] = float(abs(end1 - end2) / overall)                         # one tip much lighter than the other (orientation-free)
    pos = (np.arange(n) - (n - 1) / 2.0); pos = pos / (pos.std() + 1e-9)
    out['ctex_profile_u'] = float(np.polyfit(pos, prof, 2)[0] / overall)              # U-shape along the body
    out['ctex_band_std'] = float(prof.std() / overall)                                # overall texture variation along the body
    d = prof - gaussian_filter1d(prof, max(2, n // 4))
    out['ctex_band_amp'] = float(d.std() / overall)                                   # higher-frequency banding (twist/segment texture)
    return out
