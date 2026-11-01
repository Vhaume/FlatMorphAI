#!/usr/bin/env python
# coding: utf-8
"""body_evolution.py -- mid-body (trunk) temporal descriptors over a 3-mask window (T-1, T0, T+1).

Completes the head_extension / tail_shape pair with the central segment, the trunk between the two tip bands. It
measures the trunk's lateral undulation, how much that wave moves over the window, and how the trunk width and axial
extent change, which separates a Snake-like travelling body wave from a rigid Screw twist or a still trunk. Scale-free,
normalized by sqrt(area). No GPU.

Backing: Hu, Visual pattern recognition by moment invariants, IRE Trans. Information Theory 1962,
DOI 10.1109/TIT.1962.1057692 (principal axis). van der Walt et al., scikit-image, PeerJ 2014, DOI 10.7717/peerj.453.
"""
import numpy as np, cv2

BODY_COLS = ['body_lat_wave', 'body_lat_chg', 'body_width_osc', 'body_axial_chg', 'body_bend_chg']


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
    return c, vt[0]


def _midbody_profiles(pts, c0, ax, perp, lo, hi, nbins=12):
    """Per-slice lateral offset and width across the mid-body axial band lo..hi."""
    a = (pts - c0) @ ax
    p = (pts - c0) @ perp
    edges = np.linspace(lo, hi, nbins + 1)
    lat, wid = [], []
    for i in range(nbins):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 3:
            lat.append(float(p[sel].mean())); wid.append(float(p[sel].max() - p[sel].min()))
    return np.array(lat), np.array(wid)


def body_evolution(m_prev, m_cur, m_next):
    out = {k: np.nan for k in BODY_COLS}
    masks = [_largest(m) for m in (m_prev, m_cur, m_next)]
    if any(m.sum() < 20 for m in masks):
        return out
    c0, ax = _axis(masks[1])
    perp = np.array([-ax[1], ax[0]], np.float32)
    s = np.sqrt(float(masks[1].sum())) + 1e-6
    ys, xs = np.where(masks[1] > 0)
    a0 = (np.stack([xs, ys], 1).astype(np.float32) - c0) @ ax
    amin, amax = float(np.percentile(a0, 2)), float(np.percentile(a0, 98))
    rng = amax - amin + 1e-6
    lo, hi = amin + 0.25 * rng, amax - 0.25 * rng     # central 50% = the trunk, excluding the head and tail bands
    lats, wids, extents = [], [], []
    for m in masks:
        ys, xs = np.where(m > 0)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        lat, wid = _midbody_profiles(pts, c0, ax, perp, lo, hi)
        if len(lat) < 4:
            return out
        lats.append(lat); wids.append(wid)
        a = (pts - c0) @ ax
        aa = a[(a >= lo) & (a < hi)]
        extents.append(float(aa.max() - aa.min()) if aa.size else np.nan)
    L = min(len(x) for x in lats)
    lats = [x[:L] for x in lats]; wids = [x[:L] for x in wids]
    out['body_lat_wave'] = float(np.std(lats[1]) / s)                                  # trunk undulation amplitude at T0
    out['body_lat_chg'] = float(max(np.abs(lats[0] - lats[1]).mean(), np.abs(lats[2] - lats[1]).mean()) / s)
    wmeans = np.array([w.mean() for w in wids])
    out['body_width_osc'] = float((wmeans.max() - wmeans.min()) / s)                   # trunk thickening/thinning
    ext = np.array(extents, float)
    out['body_axial_chg'] = float((np.nanmax(ext) - np.nanmin(ext)) / s)               # trunk stretch/compression
    def _bend(lat):
        return float(np.std(np.diff(lat, 2))) if len(lat) >= 3 else 0.0
    out['body_bend_chg'] = float(abs(_bend(lats[2]) - _bend(lats[0])) / s)
    return out
