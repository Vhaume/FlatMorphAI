#!/usr/bin/env python
# coding: utf-8
"""twist_point.py -- localized twist-point descriptor for the Screw-like signature, single frame (mask + grayscale).

The experts never call a Screw on the hourglass alone, they describe a co-location: the body narrows (an hourglass
neck) AND the centerline turns sharply AND the texture shows banding, all at the same point, the twist. Each ingredient
already exists as a global metric, but none measures that the three coincide. This finds the deepest mid-body width
minimum (the neck), then scores the centerline turning and the texture anisotropy at that same location, and combines
them into one twist-point strength. Scale-free where it matters (depth is a width ratio, turn is normalized, texture
coherence is bounded). No GPU.

Backing: Hu, moment invariants, IRE Trans. Info. Theory 1962, DOI 10.1109/TIT.1962.1057692 (principal axis). Jahne,
Digital Image Processing, structure-tensor coherence for oriented texture (banding), Springer 2005, ISBN 3-540-24035-7.
"""
import numpy as np, cv2

TWIST_COLS = ['twist_neck_depth', 'twist_turn_at_neck', 'twist_tex_at_neck', 'twist_point_score']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def twist_point(mask, gray):
    out = {k: np.nan for k in TWIST_COLS}
    m = _largest(mask)
    if m.sum() < 40:
        return out
    g = np.asarray(gray)
    if g.shape != m.shape:
        g = cv2.resize(g, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]; perp = np.array([-ax[1], ax[0]], np.float32)
    a = (pts - c) @ ax
    p = (pts - c) @ perp
    s = np.sqrt(float(m.sum())) + 1e-6
    amin, amax = float(np.percentile(a, 2)), float(np.percentile(a, 98))
    NB = 24
    edges = np.linspace(amin, amax, NB + 1)
    width = np.full(NB, np.nan); center = np.full(NB, np.nan); amid = np.zeros(NB)
    for i in range(NB):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        amid[i] = 0.5 * (edges[i] + edges[i + 1])
        if sel.sum() >= 3:
            width[i] = float(p[sel].max() - p[sel].min()); center[i] = float(p[sel].mean())
    ok = np.isfinite(width)
    if ok.sum() < 10:
        return out
    w = width.copy()
    w[~ok] = np.nanmedian(width)
    medw = float(np.nanmedian(width)) + 1e-6
    # neck = deepest local width minimum in the mid-body (exclude the 20% nearest each tip)
    lo, hi = int(0.2 * NB), int(0.8 * NB)
    mid = np.arange(lo, hi)
    nk = mid[int(np.argmin(w[lo:hi]))]
    shoulder = max(w[max(lo, nk - 4):nk].max() if nk > lo else w[nk],
                   w[nk + 1:min(hi, nk + 5)].max() if nk < hi - 1 else w[nk])
    out['twist_neck_depth'] = float((shoulder - w[nk]) / medw)              # how much it pinches relative to its shoulders
    # turn at the neck = centerline curvature (second difference of the lateral offset) at the neck
    cc = center.copy(); cc[~np.isfinite(cc)] = np.nanmedian(center)
    j = int(np.clip(nk, 1, NB - 2))
    out['twist_turn_at_neck'] = float(abs(cc[j - 1] - 2 * cc[j] + cc[j + 1]) / s)
    # texture anisotropy (banding) at the neck band, structure-tensor coherence of the masked grayscale
    band = (a >= edges[max(0, nk - 1)]) & (a < edges[min(NB, nk + 2)])
    bx = xs[band]; by = ys[band]
    if len(bx) >= 30:
        x0, x1, y0, y1 = bx.min(), bx.max() + 1, by.min(), by.max() + 1
        patch = g[y0:y1, x0:x1]; pm = (m[y0:y1, x0:x1] > 0)
        gxx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gyy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        sel = pm & np.isfinite(gxx)
        if sel.sum() >= 20:
            gmag = np.sqrt(gxx[sel] ** 2 + gyy[sel] ** 2)
            out['twist_tex_at_neck'] = float(gmag.mean() / (abs(patch[sel]).mean() + 1e-6))   # banding = gradient energy at the neck, normalized; smooth body = low
    nd = out['twist_neck_depth'] if np.isfinite(out['twist_neck_depth']) else 0.0
    tn = out['twist_turn_at_neck'] if np.isfinite(out['twist_turn_at_neck']) else 0.0
    tx = out['twist_tex_at_neck'] if np.isfinite(out['twist_tex_at_neck']) else 0.0
    out['twist_point_score'] = float(nd * (1.0 + 5.0 * tn) * (1.0 + tx))     # all three coincide -> high
    return out
