#!/usr/bin/env python
# coding: utf-8
"""shape_persistence.py -- does the bend LAST? Temporal persistence of the C-shape over the +/-2 window (masks only).

Expert cue on the Wildtype-vs-C-shape failures: a real C-shape holds its curl across several frames, while a momentary
micro-bend in a Wildtype worm is transient. The model, seeing one frame, cannot tell a sustained curl from a passing
one. This measures the worm's curl per frame as 1 - solidity (mask area over convex-hull area, a C-shape leaves a large
empty hull so solidity is low) across the T-2..T+2 window the user authorizes, then reports how sustained and stable the
curl is: the floor curl across the window, the mean, the temporal stability, the fraction of frames clearly bent, and a
sustained-curl score. A real C-shape scores high on the floor and the stability, a transient bend does not. Boundary-
clamped at clip ends. Masks only, orientation-free, scale-free. No GPU. Backing: Hu, IRE Trans. Info. Theory 1962,
DOI 10.1109/TIT.1962.1057692 (shape from moments / convexity).
"""
import numpy as np, cv2

SPER_COLS = ['sper_bend_min', 'sper_bend_mean', 'sper_stability', 'sper_frac_bent', 'sper_score']
BENT = 0.12   # 1-solidity above this = clearly curled


def _curl(mask):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < 40:
        return np.nan
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.nan
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    hull = float(cv2.contourArea(cv2.convexHull(c)))
    if hull < 1e-6:
        return np.nan
    return float(1.0 - area / hull)     # 0 = straight/convex, higher = more curled (C-shape)


def shape_persistence(masks):
    """masks: list of 0/1 masks across the +/-2 window (T0 included, neighbours clamped at clip ends)."""
    out = {k: np.nan for k in SPER_COLS}
    curls = [c for c in (_curl(m) for m in masks if m is not None) if np.isfinite(c)]
    if len(curls) < 3:
        return out
    a = np.array(curls, float)
    out['sper_bend_min'] = float(a.min())                                   # even the least-curled frame is still bent -> sustained
    out['sper_bend_mean'] = float(a.mean())
    out['sper_stability'] = float(1.0 - a.std() / (a.mean() + 1e-6))        # low variation = the shape holds
    out['sper_frac_bent'] = float(np.mean(a > BENT))                        # how much of the window is clearly curled
    out['sper_score'] = float(a.min() * max(0.0, out['sper_stability']))    # sustained AND stable curl
    return out
