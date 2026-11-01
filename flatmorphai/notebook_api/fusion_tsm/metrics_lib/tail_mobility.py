#!/usr/bin/env python
# coding: utf-8
"""tail_mobility.py -- region-specific motion of the body ENDS vs the middle, over T-1, T0, T+1 (masks only).

From expert review: "a large width just before the tail that is moving a lot", "only the head is moving, the rest of the
body is not moving". We have a head-extension group (hext) but no symmetric measure of one END moving while the body
holds still, which is the Snake-like scrunch/glide signature. This splits the worm along its principal axis into the two
end-fifths and the middle, and measures per-region motion as the symmetric-difference (XOR) area between consecutive
frames normalized by region area. It reports the most-mobile tip, the end-to-end asymmetry (one tip moving more than the
other), the middle motion, and the tip-vs-middle ratio. Orientation-invariant (no head/tail call). Scale-free. No GPU.
Backing: Hu, moment invariants, IRE Trans. Info. Theory 1962, DOI 10.1109/TIT.1962.1057692 (principal axis).
"""
import numpy as np, cv2

TMOB_COLS = ['tmob_max_end', 'tmob_end_asym', 'tmob_mid', 'tmob_tip_vs_mid', 'tmob_score']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def tail_mobility(m_prev, m_cur, m_next):
    out = {k: np.nan for k in TMOB_COLS}
    mc = _largest(m_cur)
    if mc.sum() < 60 or m_prev is None or m_next is None:
        return out
    H, W = mc.shape
    mp = (np.asarray(m_prev) > 0).astype(np.uint8); mn = (np.asarray(m_next) > 0).astype(np.uint8)
    if mp.shape != mc.shape: mp = cv2.resize(mp, (W, H), interpolation=cv2.INTER_NEAREST)
    if mn.shape != mc.shape: mn = cv2.resize(mn, (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mc > 0)
    c = np.array([xs.mean(), ys.mean()], np.float32)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    # axial coordinate of EVERY pixel in the frame, relative to the current centroid + axis
    yy, xx = np.mgrid[0:H, 0:W]
    A = (xx - c[0]) * ax[0] + (yy - c[1]) * ax[1]
    acur = (pts - c) @ ax
    lo, hi = np.percentile(acur, 20), np.percentile(acur, 80)        # end-fifths vs central 60%
    end1 = A <= lo; end2 = A >= hi; mid = (A > lo) & (A < hi)
    xor1 = (mp ^ mc); xor2 = (mc ^ mn)                              # change in occupancy between consecutive frames

    def region_motion(reg):
        area = float((mc[reg] > 0).sum()) + 1e-6
        return float((xor1[reg].sum() + xor2[reg].sum()) / (2.0 * area))
    me1, me2, mmid = region_motion(end1), region_motion(end2), region_motion(mid)
    out['tmob_max_end'] = max(me1, me2)
    out['tmob_end_asym'] = abs(me1 - me2)
    out['tmob_mid'] = mmid
    out['tmob_tip_vs_mid'] = float(max(me1, me2) / (mmid + 1e-3))
    out['tmob_score'] = float(max(me1, me2) * (abs(me1 - me2) / (max(me1, me2) + 1e-6)))   # mobile tip weighted by asymmetry
    return out
