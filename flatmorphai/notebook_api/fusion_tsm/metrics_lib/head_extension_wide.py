#!/usr/bin/env python
# coding: utf-8
"""head_extension_wide.py -- the Snake head-extend/retract cue over a WIDE mask window (default +/-5 frames).

The expert calls the missed Snake-like frames by a temporal cue the 3-frame window cannot see, the head reaching out
and scrunching back over time while the trunk stays put, visible across frames beyond T-1..T+1. head_extension.py
measures this over 3 masks and saturates on a mid-transition frame, so this measures the SAME cue over a wide window.
For each mask it takes the worm length on the mask's OWN principal axis (translation- and rotation-robust, no
registration), then summarizes the length trajectory: its swing, its oscillation (scrunch cycles), its net change, and
the mobility asymmetry of the two tips (a Snake moves one tip a lot while the body stays, a Screw twists as a whole).

Backing: Hu, Visual pattern recognition by moment invariants, IRE Trans. Information Theory 1962,
DOI 10.1109/TIT.1962.1057692 (principal-axis from second moments). No GPU.
"""
import numpy as np, cv2

HEADWIDE_COLS = ['hxw_range', 'hxw_osc', 'hxw_net', 'hxw_reversals', 'hxw_nframes']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def _length_tips(m):
    """worm length and the two tip extents on the mask's own principal axis, plus sqrt(area) scale."""
    ys, xs = np.where(m > 0)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    a = (pts - c) @ vt[0]
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    s = float(np.sqrt(m.sum())) + 1e-6
    return (hi - lo), hi, lo, s            # length, top-tip, bottom-tip (relative to centroid), scale


def head_extension_wide(masks, t0=None):
    """masks: time-ordered list of binary worm masks spanning a wide window. t0 is unused (kept for call symmetry)."""
    out = {k: np.nan for k in HEADWIDE_COLS}
    vals = [_length_tips(_largest(m)) for m in masks if m is not None]
    vals = [v for v in vals if v is not None]
    nfr = len(vals)
    out['hxw_nframes'] = float(nfr)
    if nfr < 4:
        return out
    L = np.array([v[0] for v in vals]); top = np.array([v[1] for v in vals]); bot = np.array([v[2] for v in vals])
    s = float(np.mean([v[3] for v in vals])) + 1e-6
    out['hxw_range'] = float((L.max() - L.min()) / s)                       # total length swing over the window
    d2 = np.abs(np.diff(L, 2)) if nfr >= 3 else np.array([0.0])
    out['hxw_osc'] = float(d2.sum() / (s * max(1, nfr - 2)))                # mean curvature of the length trajectory = scrunch
    out['hxw_net'] = float(abs(L[-1] - L[0]) / s)                           # net extend/retract across the window
    dl = np.diff(L)
    sig = np.sign(np.where(np.abs(dl) > 0.05 * s, dl, 0.0)); nz = sig[sig != 0]
    out['hxw_reversals'] = float(np.sum(nz[1:] != nz[:-1]) if len(nz) > 1 else 0)   # scrunch cycles: Snake oscillates (>=2), Wildtype monotonic (0), Screw flat (0)
    return out
