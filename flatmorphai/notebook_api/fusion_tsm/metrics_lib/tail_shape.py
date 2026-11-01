#!/usr/bin/env python
# coding: utf-8
"""tail_shape.py -- tail-end shape descriptors over a 3-mask window (T-1, T0, T+1).

Complements head_extension. The head is the more-mobile tip, the tail is the other. At the tail end this measures how
pointed/acuminate it is (taper), how the tail bends, and how its width and reach change over the window. The expert
notes on Snake-like frames describe a thin band running to an acuminate tip, which is a tail-shape cue the head metrics
do not capture. Everything is scale-free, normalized by sqrt(area). No GPU.

Backing: Hu, Visual pattern recognition by moment invariants, IRE Trans. Information Theory 1962,
DOI 10.1109/TIT.1962.1057692 (principal-axis orientation). van der Walt et al., scikit-image, PeerJ 2014,
DOI 10.7717/peerj.453 (mask geometry).
"""
import numpy as np, cv2

TAIL_COLS = ['tail_taper', 'tail_point', 'tail_curv', 'tail_width_chg', 'tail_ext_net']


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
    return c, vt[0], pts


def _width_profile(pts, c0, ax, perp, lo, hi, nbins=10):
    """Mean perpendicular width of the worm in axial slices between axial coords lo..hi."""
    a = (pts - c0) @ ax
    p = (pts - c0) @ perp
    edges = np.linspace(lo, hi, nbins + 1)
    w = []
    for i in range(nbins):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 3:
            w.append(float(p[sel].max() - p[sel].min()))
    return np.array(w)


def tail_shape(m_prev, m_cur, m_next):
    out = {k: np.nan for k in TAIL_COLS}
    masks = [_largest(m) for m in (m_prev, m_cur, m_next)]
    if any(m.sum() < 20 for m in masks):
        return out
    c0, ax, pts0 = _axis(masks[1])
    perp = np.array([-ax[1], ax[0]], np.float32)
    s = np.sqrt(float(masks[1].sum())) + 1e-6
    # tail = the LESS-mobile tip (head moves more). Find the two axial extremes per frame, pick the steadier one.
    tipmin, tipmax = [], []
    projs = []
    for m in masks:
        ys, xs = np.where(m > 0)
        pp = np.stack([xs, ys], 1).astype(np.float32)
        a = (pp - c0) @ ax
        tipmin.append(float(a.min())); tipmax.append(float(a.max())); projs.append((pp, a))
    tipmin = np.array(tipmin); tipmax = np.array(tipmax)
    tail_is_min = tipmin.std() <= tipmax.std()             # steadier extreme = tail
    rng = float(np.percentile((pts0 - c0) @ ax, 98) - np.percentile((pts0 - c0) @ ax, 2)) + 1e-6
    # T0 tail band = the 25% of the axial range nearest the tail tip
    a0 = (pts0 - c0) @ ax
    if tail_is_min:
        lo, hi = a0.min(), a0.min() + 0.25 * rng
    else:
        lo, hi = a0.max() - 0.25 * rng, a0.max()
    w0 = _width_profile(pts0, c0, ax, perp, lo, hi)
    if len(w0) >= 3:
        midw = float(np.median(_width_profile(pts0, c0, ax, perp, a0.min(), a0.max()))) + 1e-6
        tip_w = float(w0[0] if tail_is_min else w0[-1])
        out['tail_taper'] = float((w0.max() - w0.min()) / midw)        # how much it narrows over the tail band
        out['tail_point'] = float(tip_w / midw)                         # tip width vs body, low = acuminate
        # tail curvature: deviation of the tail-band centroid path from the straight axis
        ya = []
        a = (pts0 - c0) @ ax; p = (pts0 - c0) @ perp
        for e0, e1 in zip(np.linspace(lo, hi, 6)[:-1], np.linspace(lo, hi, 6)[1:]):
            sel = (a >= e0) & (a < e1)
            if sel.sum() >= 3: ya.append(float(p[sel].mean()))
        out['tail_curv'] = float(np.std(ya) / s) if len(ya) >= 3 else 0.0
    # temporal: tail-band mean width change and tail-tip reach change over the window
    tippos = tipmin if tail_is_min else tipmax
    out['tail_ext_net'] = float((tippos[2] - tippos[0]) / s) * (-1.0 if tail_is_min else 1.0)
    bw = []
    for (pp, a) in projs:
        if tail_is_min:
            lo2, hi2 = a.min(), a.min() + 0.25 * rng
        else:
            lo2, hi2 = a.max() - 0.25 * rng, a.max()
        wp = _width_profile(pp, c0, ax, perp, lo2, hi2)
        bw.append(float(wp.mean()) if len(wp) else np.nan)
    bw = np.array(bw, float)
    if np.isfinite(bw).all():
        out['tail_width_chg'] = float((bw.max() - bw.min()) / s)
    return out
