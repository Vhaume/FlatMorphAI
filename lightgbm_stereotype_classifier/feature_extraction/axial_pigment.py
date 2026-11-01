#!/usr/bin/env python
# coding: utf-8
"""axial_pigment.py -- where the darkness sits ALONG the body, single frame (mask + grayscale).

From the reviewed comments: experts read "the first half is darker than the second", "the head is lighter than the
body", "the twisted zone has a darker pigment", and "lighter pigment at the head, also visible at the tail". This bins
the masked grayscale along the principal axis and measures the head-half versus tail-half intensity asymmetry, the total
axial intensity range, how dark the darkest band is, where it sits, and the EXTREMITY-versus-middle lightness (both tips
lighter than the body), orientation-invariant (no head/tail call needed). Scale-free (all ratios to the worm mean). No
GPU. Backing: Hu, moment invariants, IRE Trans. Info. Theory 1962, DOI 10.1109/TIT.1962.1057692 (principal axis).
"""
import numpy as np, cv2

PIGMENT_COLS = ['pig_half_asym', 'pig_axial_range', 'pig_dark_depth', 'pig_dark_loc',
                'pig_ends_vs_mid', 'pig_ends_both_light', 'pig_max_end_light', 'pig_light_loc', 'pig_profile_u']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def axial_pigment(mask, gray):
    out = {k: np.nan for k in PIGMENT_COLS}
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
    ax = vt[0]
    a = (pts - c) @ ax
    gv = g[ys, xs]
    overall = float(gv.mean()) + 1e-6
    amin, amax = float(np.percentile(a, 2)), float(np.percentile(a, 98))
    NB = 20
    edges = np.linspace(amin, amax, NB + 1)
    bmean = np.full(NB, np.nan)
    for i in range(NB):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 5:
            bmean[i] = float(gv[sel].mean())
    ok = np.isfinite(bmean)
    if ok.sum() < 8:
        return out
    bm = bmean[ok]; idx = np.where(ok)[0]
    half = len(bm) // 2
    out['pig_half_asym'] = float(abs(bm[:half].mean() - bm[half:].mean()) / overall)   # one half darker than the other
    out['pig_axial_range'] = float((bm.max() - bm.min()) / overall)                    # total dark-to-light spread along the body
    out['pig_dark_depth'] = float((np.median(bm) - bm.min()) / overall)                # how dark the darkest band is vs the body
    darkest = idx[int(np.argmin(bm))]                                                  # axial position of the darkest band
    out['pig_dark_loc'] = float(abs(darkest / (NB - 1) - 0.5) * 2.0)                   # 0 = centre, 1 = at a tip

    # EXTREMITY lightness (the expert cue: lighter pigment at head AND tail vs the middle), orientation-invariant.
    L = len(bm); k = max(2, L // 5)
    end1 = float(bm[:k].mean()); end2 = float(bm[-k:].mean())
    mid = float(bm[k:-k].mean()) if L > 2 * k else float(bm.mean())
    out['pig_ends_vs_mid'] = float((0.5 * (end1 + end2) - mid) / overall)              # >0 = both ends lighter than middle
    out['pig_ends_both_light'] = float((min(end1, end2) - mid) / overall)              # >0 only when BOTH ends, not just one
    out['pig_max_end_light'] = float((max(end1, end2) - mid) / overall)                # >0 when AT LEAST ONE tip is lighter (tail-only too)
    lightest = idx[int(np.argmax(bm))]
    out['pig_light_loc'] = float(abs(lightest / (NB - 1) - 0.5) * 2.0)                 # 0 = centre, 1 = at a tip
    pos = (idx - idx.mean()) / (idx.std() + 1e-9)
    out['pig_profile_u'] = float(np.polyfit(pos, bm, 2)[0] / overall)                  # >0 = U-shaped (ends light, middle dark)
    return out
