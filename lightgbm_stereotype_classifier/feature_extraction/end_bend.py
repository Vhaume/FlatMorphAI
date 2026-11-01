#!/usr/bin/env python
# coding: utf-8
"""end_bend.py -- is the bend a SHARP turn AT A TIP or a smooth/middle curve? single frame, centerline curvature.

Expert cue on the Snake-versus-Wildtype failures: "the head is a sharp curve while a normal head has a smooth curve",
and "a sharp turn at the head". The existing curvature metrics summarize bending along the whole body, but do not say
WHERE the sharp bend sits. A Snake-like head often has an acute kink right at a tip, while a Wildtype head curves gently,
and a C-shape bends in the MIDDLE not at a tip. This traces the centerline (same skeleton longest-path tracer as the
border-evolution group), measures local curvature along it, and reports the sharpest bend near a tip, the tip-to-tip
asymmetry of that sharpness, the sharpest bend in the middle, and the tip-versus-middle contrast. Orientation-free (max
over the two ends), scale-free. Single frame, processed grayscale. No GPU. Backing: skeleton medial axis, Hu, IRE Trans.
Info. Theory 1962, DOI 10.1109/TIT.1962.1057692.
"""
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
from twist_border_evolution import _longest_path

EBS_COLS = ['ebs_max_end_curv', 'ebs_end_asym', 'ebs_mid_curv', 'ebs_end_vs_mid']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def _curv(P, k):
    L = len(P); c = np.zeros(L, np.float32)
    for i in range(k, L - k):
        a = P[i] - P[i - k]; b = P[i + k] - P[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
        if na > 1e-6:
            c[i] = np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1)))
    return c


def end_bend(gray):
    out = {k: np.nan for k in EBS_COLS}
    worm = _largest((np.asarray(gray) > 8).astype(np.uint8))
    if worm.sum() < 60:
        return out
    path = _longest_path(skeletonize(worm > 0))
    if path is None or len(path) < 16:
        return out
    P = np.array(path, np.float32)
    P = np.stack([gaussian_filter1d(P[:, 0], 2.0), gaussian_filter1d(P[:, 1], 2.0)], 1)
    n = len(P); k = max(3, n // 25)
    cur = _curv(P, k)
    e = max(3, n // 4)                                  # the end-quarter at each tip
    end1 = float(cur[k:e].max()) if e > k else 0.0
    end2 = float(cur[-e:-k].max()) if e > k else 0.0
    mid = float(cur[e:-e].max()) if n > 2 * e else float(cur.max())
    out['ebs_max_end_curv'] = max(end1, end2)                          # sharpest bend at a tip (the sharp head turn)
    out['ebs_end_asym'] = float(abs(end1 - end2))                      # one tip sharper than the other (orientation-free)
    out['ebs_mid_curv'] = mid                                          # sharpest bend in the middle (C-shape bends here)
    out['ebs_end_vs_mid'] = float(max(end1, end2) - mid)               # >0 sharp at a tip, <0 sharp in the middle
    return out
