#!/usr/bin/env python
# coding: utf-8
"""tail_twist.py -- thin-hooked-tail twist signature, single frame (mask), over a 3-mask window for tail identification.

From the reviewed comments: experts call a Screw on a tail that is thin, sharply curved or hooked, with a spiky or
scalloped end, and the twist sits at that tail. twist_point measures the mid-body neck and tail_shape measures taper, but
neither encodes a THIN tail that also turns SHARPLY and hooks back. This identifies the tail (the steadier tip across the
window) and scores its thinness, the sharp turning of the centerline near it, and how far it swings laterally (the hook).
Scale-free. No GPU. Backing: Hu, moment invariants, IRE Trans. Info. Theory 1962, DOI 10.1109/TIT.1962.1057692.
"""
import numpy as np, cv2

TAILTWIST_COLS = ['tailtw_thin', 'tailtw_turn', 'tailtw_hook', 'tailtw_score']


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


def tail_twist(m_prev, m_cur, m_next):
    out = {k: np.nan for k in TAILTWIST_COLS}
    masks = [_largest(m) for m in (m_prev, m_cur, m_next)]
    if any(m.sum() < 25 for m in masks):
        return out
    c0, ax = _axis(masks[1])
    perp = np.array([-ax[1], ax[0]], np.float32)
    s = np.sqrt(float(masks[1].sum())) + 1e-6
    # tail = the steadier axial extreme across the window (the head leads, the tail trails)
    tmin, tmax = [], []
    for m in masks:
        ys, xs = np.where(m > 0)
        a = (np.stack([xs, ys], 1).astype(np.float32) - c0) @ ax
        tmin.append(float(a.min())); tmax.append(float(a.max()))
    tail_is_min = np.std(tmin) <= np.std(tmax)
    ys, xs = np.where(masks[1] > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    a = (pts - c0) @ ax; p = (pts - c0) @ perp
    amin, amax = float(np.percentile(a, 2)), float(np.percentile(a, 98)); rng = amax - amin + 1e-6
    NB = 20
    edges = np.linspace(amin, amax, NB + 1)
    width = np.full(NB, np.nan); cen = np.full(NB, np.nan)
    for i in range(NB):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 3:
            width[i] = float(p[sel].max() - p[sel].min()); cen[i] = float(p[sel].mean())
    if np.isfinite(width).sum() < 8:
        return out
    medw = float(np.nanmedian(width)) + 1e-6
    tail_bins = range(0, max(1, NB // 4)) if tail_is_min else range(NB - max(1, NB // 4), NB)
    tb = [i for i in tail_bins if np.isfinite(width[i])]
    if len(tb) < 3:
        return out
    out['tailtw_thin'] = float(medw / (np.nanmean([width[i] for i in tb]) + 1e-6))     # body / tail width, high = thin tail
    cc = np.array([cen[i] if np.isfinite(cen[i]) else np.nanmedian(cen) for i in range(NB)])
    seg = cc[tb[0]:tb[-1] + 1]
    out['tailtw_turn'] = float(np.abs(np.diff(seg, 2)).sum() / s) if len(seg) >= 3 else 0.0  # sharp curving of the tail centerline
    out['tailtw_hook'] = float((np.nanmax(seg) - np.nanmin(seg)) / s) if len(seg) else 0.0   # lateral swing of the tail (the hook)
    th = out['tailtw_thin']; tu = out['tailtw_turn']; hk = out['tailtw_hook']
    out['tailtw_score'] = float(th * (1.0 + 8.0 * tu) * (1.0 + 4.0 * hk))                # thin AND sharply-turning AND hooked
    return out
