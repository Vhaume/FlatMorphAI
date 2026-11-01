#!/usr/bin/env python
# coding: utf-8
"""peristalsis.py -- detect a localized ROUND bulge (a sphere) travelling along the body, from the 3-mask window.

Corrected definition: peristalsis here is a distinct round swelling, a circle, that moves along the body axis from one
frame to the next, not merely a width change. So the metric (1) finds the most prominent NARROW, ROUND local width peak
in each frame (a broad swelling or a global width change is rejected), (2) tracks that peak's axial position across the
three frames, and (3) fires only when a round sphere is present and its position shifts monotonically by a consistent
amount with a consistent size. Centreline / principal-axis based, scale-free. No GPU.

Backing: Hu, moment invariants, IRE Trans. Info. Theory 1962, DOI 10.1109/TIT.1962.1057692 (principal axis). Kymograph /
reslice quantification of crawling-worm peristaltic waves, Frontiers in Computational Neuroscience 2013,
DOI 10.3389/fncom.2013.00024 (a travelling feature is a slanted line in the body-axis-versus-time plot).
"""
import numpy as np, cv2

PERI_COLS = ['peri_wave_shift', 'peri_coherence', 'peri_n_bulges', 'peri_amp', 'peri_amp_consist', 'peri_round', 'peri_score']
NB = 36


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
    if len(xs) < 20:
        return None, None
    pts = np.stack([xs, ys], 1).astype(np.float32); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _profile(m, c, ax):
    """width per axial bin and the bin length, on a GIVEN oriented axis (shared across the 3 frames)."""
    ys, xs = np.where(m > 0)
    if len(xs) < 20:
        return None, None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    a = (pts - c) @ ax; p = (pts - c) @ np.array([-ax[1], ax[0]], np.float32)
    amin, amax = float(np.percentile(a, 2)), float(np.percentile(a, 98))
    edges = np.linspace(amin, amax, NB + 1); binlen = (amax - amin) / NB
    w = np.full(NB, np.nan)
    for i in range(NB):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 3:
            w[i] = float(p[sel].max() - p[sel].min())
    ok = np.isfinite(w)
    if ok.sum() < NB * 0.6:
        return None, None
    w[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), w[ok])
    return w, binlen


def _smooth(x, k):
    k = max(3, k | 1); g = np.ones(k) / k
    return np.convolve(np.r_[x[:k][::-1], x, x[-k:][::-1]], g, 'same')[k:-k]


def _best_bulge(w, binlen):
    """the most prominent NARROW round local peak: returns (position bin, amplitude, roundness) or None.
    roundness ~ 1 when the bulge's axial length matches its lateral width (a circle), low for a broad swelling."""
    base = _smooth(w, max(5, NB // 3)); medw = float(np.median(w)) + 1e-6
    r = (w - base) / medw                                       # detrended residual = the bump over the body taper
    best = None
    for i in range(1, NB - 1):
        if r[i] <= 0.05 or r[i] < r[i - 1] or r[i] < r[i + 1]:
            continue
        lo = i
        while lo > 0 and r[lo - 1] < r[lo] and r[lo - 1] > 0: lo -= 1
        hi = i
        while hi < NB - 1 and r[hi + 1] < r[hi] and r[hi + 1] > 0: hi += 1
        prom = r[i] - 0.5 * (r[lo] + r[hi])
        axlen = (hi - lo + 1) * binlen                          # physical axial length of the bulge
        latw = float(w[i])                                      # local lateral width at the peak
        roundness = float(min(axlen, latw) / (max(axlen, latw) + 1e-6))   # 1 = round blob, low = long ridge
        narrow = (hi - lo + 1) <= NB // 4                       # localized, not a broad swelling
        score = prom * roundness * (1.0 if narrow else 0.3)
        if prom > 0.06 and (best is None or score > best[3]):
            best = (i, float(r[i]), roundness, score)
    return best


def peristalsis(m_prev, m_cur, m_next):
    out = {k: np.nan for k in PERI_COLS}
    masks = [_largest(m) for m in (m_prev, m_cur, m_next)]
    axes = [_axis(m) for m in masks]
    if any(c is None for c, _ in axes):
        return out
    ax0 = axes[1][1]                                            # T0 axis is the shared reference
    oriented = []
    for c, ax in axes:
        if float(ax @ ax0) < 0: ax = -ax                       # sign-align each frame axis to T0 (frames barely rotate)
        oriented.append((c, ax))
    profs = [_profile(m, c, ax) for m, (c, ax) in zip(masks, oriented)]
    if any(w is None for w, _ in profs):
        return out
    ws = [w for w, _ in profs]; bl = [b for _, b in profs]
    bulges = [_best_bulge(ws[t], bl[t]) for t in range(3)]
    out['peri_n_bulges'] = float(sum(b is not None for b in bulges))
    if any(b is None for b in bulges):                          # a sphere must be present in all three frames
        out['peri_wave_shift'] = 0.0; out['peri_coherence'] = 0.0
        out['peri_amp'] = float(max([b[1] for b in bulges if b], default=0.0))
        out['peri_amp_consist'] = 0.0; out['peri_round'] = float(max([b[2] for b in bulges if b], default=0.0))
        out['peri_score'] = 0.0
        return out
    pos = np.array([b[0] for b in bulges], float); amp = np.array([b[1] for b in bulges]); rnd = np.array([b[2] for b in bulges])
    d1, d2 = pos[1] - pos[0], pos[2] - pos[1]                   # peak displacement over the two transitions
    out['peri_wave_shift'] = float((d1 + d2) / (2.0 * NB))
    moving = (abs(d1) >= 1 and abs(d2) >= 1 and np.sign(d1) == np.sign(d2))   # monotonic travel of the SAME sphere
    out['peri_coherence'] = float(1.0 - abs(abs(d1) - abs(d2)) / (abs(d1) + abs(d2) + 1e-6)) if moving else 0.0
    out['peri_amp'] = float(amp.mean())
    out['peri_amp_consist'] = float(max(0.0, 1.0 - amp.std() / (amp.mean() + 1e-6)))
    out['peri_round'] = float(rnd.mean())
    out['peri_score'] = float(abs(out['peri_wave_shift']) * out['peri_coherence'] * amp.mean() *
                              out['peri_amp_consist'] * out['peri_round'] * 100.0)
    return out
