#!/usr/bin/env python
# coding: utf-8
"""axial_torsion.py -- a metric that reads axial rotation (the Screw cue) from a masked grayscale worm.

Physics: a Screw-like worm rotates about its long body axis, so its pigmented/shaded side spirals along the body and
that spiral travels along the axis over time. A Snake-like worm undulates its centreline laterally in the image plane,
which moves the silhouette but does NOT twist the shading. So axial rotation shows up as a transverse offset of the
bright (pigmented) centroid that OSCILLATES along the body (a helix) and TRAVELS along the body across the 3-frame
window (rotation), independent of the silhouette undulation that the shape metrics already see.

Pipeline: principal-axis frame, then per cross-section the intensity-weighted transverse centroid minus the geometric
centre, normalized by half-width, giving a signed offset profile c(s) along the body. Single-frame metrics read the
helical banding of c(s); the 3-frame metrics read how far c(s) travels along the axis (the rotation).

Backing: principal axis from image moments, Hu, Visual pattern recognition by moment invariants, IRE Trans. Information
Theory 1962, DOI 10.1109/TIT.1962.1057692. Travelling-pattern (kymograph cross-correlation) quantification of motion,
as used for peristalsis, Front. Comput. Neurosci. 2013, DOI 10.3389/fncom.2013.00024. No GPU, single worm per frame.
"""
import numpy as np
import cv2

TORSION_COLS = ['tor_zcross', 'tor_amp', 'tor_fft_frac', 'tor_oscillation',   # single frame (T0)
                'tor_travel', 'tor_phase_adv', 'tor_amp_chg', 'tor_score']     # temporal (T-1,T0,T+1)
NB = 24


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def axis_profile(gray, mask, nb=NB):
    """Signed transverse offset of the intensity-weighted centroid from the geometric centre, per cross-section along
    the principal axis. Returns an array of length nb in roughly [-1, 1], or None."""
    g = np.asarray(gray).astype(np.float32)
    m = _largest(mask)
    if m.sum() < 60:
        return None
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]; perp = np.array([-ax[1], ax[0]], np.float32)
    a = (pts - c) @ ax; p = (pts - c) @ perp
    gi = g[ys, xs]
    gi = gi - gi.mean()                                   # remove DC so only the lateral shading bias matters
    amin, amax = np.percentile(a, 2), np.percentile(a, 98)
    edges = np.linspace(amin, amax, nb + 1)
    prof = np.full(nb, np.nan)
    for i in range(nb):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() >= 5:
            pp = p[sel]; gg = gi[sel]
            geo = float(pp.mean())
            w = gg - gg.min() + 1e-6                       # brighter pixels weigh more -> bright-side position
            wc = float((pp * w).sum() / w.sum())
            half = (float(pp.max()) - float(pp.min())) / 2.0 + 1e-6
            prof[i] = (wc - geo) / half
    good = np.isfinite(prof)
    if good.sum() < nb // 2:
        return None
    prof = np.interp(np.arange(nb), np.where(good)[0], prof[good])
    return prof.astype(np.float32)


def _single(prof):
    if prof is None:
        return {k: np.nan for k in ('tor_zcross', 'tor_amp', 'tor_fft_frac', 'tor_oscillation')}
    p = prof - prof.mean()
    s = np.sign(p); s[s == 0] = 1
    zc = float((np.abs(np.diff(s)) > 0).sum()) / (len(p) - 1)        # sign changes per step = helical banding
    amp = float(np.sqrt(np.mean(p ** 2)))
    f = np.abs(np.fft.rfft(p))[1:]                                   # drop DC
    fft_frac = float((f.max() ** 2) / (np.sum(f ** 2) + 1e-9)) if len(f) else np.nan
    d = np.diff(prof)
    osc = float(1.0 - abs(np.sign(d).sum()) / (len(d) + 1e-9))       # 1 = oscillatory, 0 = monotone
    return dict(tor_zcross=zc, tor_amp=amp, tor_fft_frac=fft_frac, tor_oscillation=osc)


def axial_torsion(grays3, masks3):
    """grays3, masks3: (T-1, T0, T+1) masked grayscale + masks. Returns the full TORSION_COLS dict."""
    out = {k: np.nan for k in TORSION_COLS}
    profs = [axis_profile(g, m) if (g is not None and m is not None) else None for g, m in zip(grays3, masks3)]
    out.update(_single(profs[1]))                                   # single-frame metrics on T0
    p0, p2 = profs[0], profs[2]
    if p0 is not None and p2 is not None and len(p0) == len(p2):
        a0 = p0 - p0.mean(); a2 = p2 - p2.mean()
        xc = np.correlate(a2, np.concatenate([a0, a0]), mode='valid')[:len(a0)]   # circular-ish cross-correlation
        lag = int(np.argmax(xc))
        if lag > len(a0) // 2:
            lag -= len(a0)
        out['tor_travel'] = float(lag) / (len(a0) / 2.0)            # signed fraction of body length the spiral moved
        F0 = np.fft.rfft(a0); F2 = np.fft.rfft(a2)
        k = 1 + int(np.argmax(np.abs(F0)[1:])) if len(F0) > 1 else 0
        out['tor_phase_adv'] = float(np.angle(F2[k]) - np.angle(F0[k])) if k else np.nan
        out['tor_amp_chg'] = float(abs(np.sqrt(np.mean(a2 ** 2)) - np.sqrt(np.mean(a0 ** 2))))
    z = out.get('tor_zcross', np.nan); am = out.get('tor_amp', np.nan); tv = out.get('tor_travel', np.nan)
    if np.isfinite(z) and np.isfinite(am):
        out['tor_score'] = float(am * (0.3 + z) * (1.0 + (abs(tv) if np.isfinite(tv) else 0.0)))
    return out
