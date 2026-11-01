#!/usr/bin/env python
# coding: utf-8
"""neck_extent.py -- sustained thin-region descriptor for the Screw twist, single frame (mask).

From the reviewed Screw-vs-Wildtype misses, the expert cue is not a single pinch but a sustained one: "the width of
the body is drastically reduced where the twist happened and is pursued all along the head". The existing neck metrics
(twist_neck_depth, be_neck_depth) measure the depth of the single narrowest point, which a transient pinch and a long
twisted-thin head share. This measures the EXTENT of the thinning instead: the longest run of body where the width
stays below a fraction of the worm median, how thin that run gets, where it sits along the axis, and the left-right
width asymmetry (a Screw is thin at one end while the other stays full, a Wildtype is near-uniform). Scale-free,
orientation-invariant, no GPU.

Backing: Hu, Visual pattern recognition by moment invariants, IRE Trans. Information Theory 1962,
DOI 10.1109/TIT.1962.1057692 (principal axis from second moments).
"""
import numpy as np, cv2

NECKEXT_COLS = ['nke_thin_len', 'nke_thin_depth', 'nke_thin_loc', 'nke_asym']
THIN_FRAC = 0.62          # a bin counts as "thin" if its width < THIN_FRAC * median width


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def neck_extent(mask, gray=None):
    out = {k: np.nan for k in NECKEXT_COLS}
    m = _largest(mask)
    if m.sum() < 60:
        return out
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]; perp = np.array([-ax[1], ax[0]], np.float32)
    a = (pts - c) @ ax; p = (pts - c) @ perp
    amin, amax = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    NB = 24
    edges = np.linspace(amin, amax, NB + 1)
    width = np.full(NB, np.nan)
    for i in range(NB):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 3:
            width[i] = float(p[sel].max() - p[sel].min())
    ok = np.isfinite(width)
    if ok.sum() < 10:
        return out
    w = width.copy(); w[~ok] = np.nanmedian(width)
    medw = float(np.nanmedian(width)) + 1e-6
    thin = w < (THIN_FRAC * medw)                       # which bins are sub-normal width
    # longest consecutive thin run, and its location
    best_len, best_lo = 0, 0; cur = 0; lo = 0
    for i in range(NB):
        if thin[i]:
            if cur == 0: lo = i
            cur += 1
            if cur > best_len: best_len, best_lo = cur, lo
        else:
            cur = 0
    out['nke_thin_len'] = float(best_len / NB)          # fraction of the body that is a continuous thin stretch
    if best_len > 0:
        run = w[best_lo:best_lo + best_len]
        out['nke_thin_depth'] = float((medw - run.min()) / medw)          # how thin the thinnest point of that run is
        mid = (best_lo + best_len / 2.0) / NB
        out['nke_thin_loc'] = float(abs(mid - 0.5) * 2.0)                 # 0 = thin run centred, 1 = thin run at a tip
    else:
        out['nke_thin_depth'] = 0.0; out['nke_thin_loc'] = 0.0
    half = NB // 2
    wl, wr = np.nanmean(w[:half]), np.nanmean(w[half:])
    out['nke_asym'] = float(abs(wl - wr) / (medw + 1e-6))                 # one end thin, the other full = high
    return out
