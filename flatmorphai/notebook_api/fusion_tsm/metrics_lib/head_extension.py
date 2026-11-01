#!/usr/bin/env python
# coding: utf-8
"""head_extension.py -- temporal head-extension descriptors over a 3-mask window (T-1, T0, T+1).

Built from the expert cue on the ambiguous Snake-like frames: the head endpoint extends and retracts over time while
the trunk stays still, which the existing motion metrics (global magnitude, lateral travel) do not capture. These
descriptors measure, along the T0 principal axis, how much the more-mobile tip (the head) changes its reach across the
window and how localized that motion is to the head versus the body. Everything is scale-free, normalized by sqrt(area).

Backing: Hu, Visual pattern recognition by moment invariants, IRE Trans. Information Theory 1962, DOI 10.1109/TIT.1962.1057692
(principal-axis orientation from second moments). No GPU.
"""
import numpy as np, cv2

HEADEXT_COLS = ['hext_range', 'hext_net', 'hext_osc', 'hbody_asym']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def _axis(m):
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]                       # centroid, major-axis unit vector


def head_extension(m_prev, m_cur, m_next):
    """Each argument is a binary worm mask (HxW). Returns a dict over HEADEXT_COLS."""
    out = {k: np.nan for k in HEADEXT_COLS}
    masks = [_largest(m) for m in (m_prev, m_cur, m_next)]
    if any(m.sum() < 20 for m in masks):
        return out
    c0, ax = _axis(masks[1])              # T0 is the common reference frame, so the two tips stay the same physical ends
    s = np.sqrt(float(masks[1].sum())) + 1e-6
    tipmin, tipmax, cents = [], [], []
    for m in masks:
        ys, xs = np.where(m > 0)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        proj = (pts - c0) @ ax
        tipmin.append(float(proj.min())); tipmax.append(float(proj.max())); cents.append(pts.mean(0))
    tipmin = np.array(tipmin); tipmax = np.array(tipmax); cents = np.array(cents)
    head = tipmin if tipmin.std() >= tipmax.std() else tipmax     # the head is the more mobile tip
    sign = -1.0 if tipmin.std() >= tipmax.std() else 1.0          # so + means net extension outward
    out['hext_range'] = float((head.max() - head.min()) / s)
    out['hext_net'] = float(sign * (head[2] - head[0]) / s)
    out['hext_osc'] = float(head.std() / s)
    cdisp = float(np.linalg.norm(cents[2] - cents[0]))           # body/centroid displacement over the window
    out['hbody_asym'] = float(abs(head[2] - head[0]) / (cdisp + 1e-6))   # head reach change relative to body motion
    return out
