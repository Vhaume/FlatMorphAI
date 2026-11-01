#!/usr/bin/env python
# coding: utf-8
"""terminal_shape.py -- dedicated detectors for the LOCALIZED silhouettes the expert named on Screw frames, single frame.

The expert never describes a Screw with a global number. They name a specific shape at a body end: a bowling pin or
golf club or filled-P (a thin neck then a bulb), a shark fin or crescent or falciform lobe (a sharp lobe offset to one
side), an omega or epsilon or hook (a re-entrant curl), and a scalloped edge AT a specific position (head, mid, tail).
Generic border roughness or centerline curvature cannot represent these, so this builds a dedicated detector for each.
Everything is in the principal-axis frame, at each of the two body ends (no head/tail call needed, max over ends and the
per-end values are exposed), and scale-free. No GPU.

Backing: Hu, Visual pattern recognition by moment invariants, IRE Trans. Information Theory 1962,
DOI 10.1109/TIT.1962.1057692 (principal axis). Mokhtarian and Mackworth, A theory of multiscale, curvature-based shape
representation for planar curves, IEEE TPAMI 1992, DOI 10.1109/34.149591 (curvature along a contour for scalloping).
"""
import numpy as np, cv2

TERMINAL_COLS = [
    'tsh_stembulb_e0', 'tsh_stembulb_e1', 'tsh_stembulb_max',
    'tsh_fin_e0', 'tsh_fin_e1', 'tsh_fin_max',
    'tsh_hook_e0', 'tsh_hook_e1', 'tsh_hook_max',
    'tsh_scallop_e0', 'tsh_scallop_mid', 'tsh_scallop_e1', 'tsh_scallop_max',
]


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def _profile(a, p, NB):
    amin, amax = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    edges = np.linspace(amin, amax, NB + 1)
    width = np.full(NB, np.nan); cen = np.full(NB, np.nan)
    for i in range(NB):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 3:
            width[i] = float(p[sel].max() - p[sel].min()); cen[i] = float(p[sel].mean())
    return width, cen, edges


def _stembulb(prof, medw):
    """thin neck then a bulb toward the tip. prof ordered tip -> body. plain taper -> 0."""
    o = prof[np.isfinite(prof)]
    if len(o) < 4:
        return 0.0
    er = max(3, len(o) // 2)
    seg = o[:er]
    nk = int(np.argmin(seg))
    bulb = float(np.max(seg[:nk + 1]))
    return max(0.0, (bulb - float(seg[nk])) / (medw + 1e-6))


def _fin(prof, cen, medw, s):
    """sharp lobe offset to one side. OFFSET-GATED: a symmetric point (bowling-pin cap) scores ~0, only an off-axis
    lobe (the shark fin) fires."""
    w = prof[np.isfinite(prof)]; c = cen[np.isfinite(prof)]
    if len(w) < 4:
        return 0.0
    er = max(2, len(w) // 4)
    tip_w = float(np.nanmean(w[:er]))
    sharp = max(0.0, (medw - tip_w) / (medw + 1e-6))
    offset = float(abs(np.nanmean(c[:er])) / (s + 1e-6))
    return offset * (1.0 + 2.0 * sharp)


def _hook(cen, s):
    """re-entrant curl: the centreline reverses within the end region (omega / hook)."""
    c = cen[np.isfinite(cen)]
    if len(c) < 5:
        return 0.0
    er = max(3, len(c) // 2)
    seg = c[:er]
    d2 = np.abs(np.diff(seg, 2))
    return float(np.max(d2) / (s + 1e-6)) if len(d2) else 0.0


def _resample_smooth(cnt, N=240, sigma=2.0):
    d = np.sqrt(((np.roll(cnt, -1, 0) - cnt) ** 2).sum(1))
    cl = np.concatenate([[0], np.cumsum(d)]); total = cl[-1]
    if total < 1e-6:
        return cnt, cnt
    u = np.linspace(0, total, N, endpoint=False)
    rx = np.interp(u, cl[:-1], cnt[:, 0], period=total); ry = np.interp(u, cl[:-1], cnt[:, 1], period=total)
    raw = np.stack([rx, ry], 1).astype(np.float32)
    def smooth(arr, sg):
        g = np.exp(-0.5 * (np.arange(-3 * int(sg), 3 * int(sg) + 1) / sg) ** 2); g /= g.sum()
        pad = len(g)
        sx = np.convolve(np.r_[arr[-pad:, 0], arr[:, 0], arr[:pad, 0]], g, 'same')[pad:-pad]
        sy = np.convolve(np.r_[arr[-pad:, 1], arr[:, 1], arr[:pad, 1]], g, 'same')[pad:-pad]
        return np.stack([sx, sy], 1).astype(np.float32)
    return raw, smooth(raw, max(4.0, N / 30.0))   # base outline = heavily smoothed border


def _scallop_by_third(m, a_of_xy, s):
    """AMPLITUDE of the border's deviation from its own heavily-smoothed outline, per axial third. Threshold-free, so
    fine scallops register, and a smooth taper has near-zero residual (the smooth outline already follows the taper)."""
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.nan, np.nan, np.nan
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(cnt) < 40:
        return np.nan, np.nan, np.nan
    raw, base = _resample_smooth(cnt)
    resid = np.sqrt(((raw - base) ** 2).sum(1)) / (s + 1e-6)   # how far the border bulges from its smoothed self
    av = a_of_xy(raw[:, 0], raw[:, 1])
    lo, hi = np.percentile(av, 5), np.percentile(av, 95); rng = hi - lo + 1e-6
    t = (av - lo) / rng
    out = []
    for a0, a1 in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
        sel = (t >= a0) & (t < a1)
        out.append(float(np.sqrt(np.mean(resid[sel] ** 2)) * 100.0) if sel.sum() >= 8 else np.nan)
    return out[0], out[1], out[2]


def _reentrance(m, a_of_xy, medw, s):
    """re-entrant curl (omega / epsilon / hook) = ONE deep concavity carved by the fold. Convexity-defect depth captures
    it, and a deep single notch separates a curl from a thin waist (which only makes shallow opposing indents)."""
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 8:
        return 0.0, 0.0, 0.0
    try:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
    except Exception:
        return 0.0, 0.0, 0.0
    if defects is None:
        return 0.0, 0.0, 0.0
    lo, hi = np.percentile(a_of_xy(cnt[:, 0, 0], cnt[:, 0, 1]), [5, 95]); mid = 0.5 * (lo + hi)
    he = [0.0, 0.0]
    for k in range(defects.shape[0]):
        _dk = defects.reshape(-1, 4)[k]; f_i = int(_dk[2]); depth = float(_dk[3]) / 256.0 / (s + 1e-6)
        fp = cnt[f_i, 0]
        af = float(a_of_xy(np.array([fp[0]], np.float32), np.array([fp[1]], np.float32))[0])
        end = 0 if af < mid else 1
        he[end] = max(he[end], depth)
    return he[0], he[1], max(he)


def terminal_shape(mask, gray=None):
    out = {k: np.nan for k in TERMINAL_COLS}
    m = _largest(mask)
    if m.sum() < 60:
        return out
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]; perp = np.array([-ax[1], ax[0]], np.float32)
    a = (pts - c) @ ax; p = (pts - c) @ perp
    s = float(np.sqrt(m.sum())) + 1e-6
    NB = 24
    width, cen, _ = _profile(a, p, NB)
    if np.isfinite(width).sum() < 8:
        return out
    medw = float(np.nanmedian(width)) + 1e-6
    w0, c0 = width, cen
    w1, c1 = width[::-1], cen[::-1]
    out['tsh_stembulb_e0'] = _stembulb(w0, medw)
    out['tsh_stembulb_e1'] = _stembulb(w1, medw)
    out['tsh_stembulb_max'] = max(out['tsh_stembulb_e0'], out['tsh_stembulb_e1'])
    def a_of_xy(px, py):
        return (np.stack([px, py], 1).astype(np.float32) - c) @ ax
    out['tsh_hook_e0'], out['tsh_hook_e1'], out['tsh_hook_max'] = _reentrance(m, a_of_xy, medw, s)
    out['tsh_fin_e0'] = max(0.0, _fin(w0, c0, medw, s) - 1.0 * out['tsh_hook_e0'])   # a reversing curl is a hook, not a fin
    out['tsh_fin_e1'] = max(0.0, _fin(w1, c1, medw, s) - 1.0 * out['tsh_hook_e1'])
    out['tsh_fin_max'] = max(out['tsh_fin_e0'], out['tsh_fin_e1'])

    se0, smid, se1 = _scallop_by_third(m, a_of_xy, s)
    out['tsh_scallop_e0'] = se0; out['tsh_scallop_mid'] = smid; out['tsh_scallop_e1'] = se1
    vals = [v for v in (se0, smid, se1) if np.isfinite(v)]
    out['tsh_scallop_max'] = max(vals) if vals else np.nan
    return out
