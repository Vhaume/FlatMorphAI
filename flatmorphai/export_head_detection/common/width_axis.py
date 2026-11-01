# -*- coding: utf-8 -*-
"""Width-aligned medial axis, for stubby shapes.

    The morphological skeleton of an ovoid worm is short, bent and does not cross the body, so the
    width profile read along it does not describe the animal. Here the axis is the minimum-cost path
    between the two farthest points in the intra-mask geodesic distance, with a cost decreasing with
    the distance to the border, which constrains the path to follow the ridge of the distance transform,
    that is the locus of the maximal inscribed discs. It therefore crosses the body from tip to tip and
    its normal is perpendicular to the real axis of the worm.

    References. Medial axis as the locus of maximal inscribed discs, H. Blum, A transformation for
    extracting new descriptors of shape, Models for the Perception of Speech and Visual Form, 1967.
    Exact distance transform, C. R. Maurer, R. Qi, V. Raghavan, A linear time algorithm for computing
    exact Euclidean distance transforms of binary images in arbitrary dimensions, IEEE Transactions on
    Pattern Analysis and Machine Intelligence, 2003, DOI 10.1109/TPAMI.2003.1177156.
"""
import numpy as np, cv2
from skimage.graph import MCP_Geometric

def medial_axis(mask, alpha=12.0, k=2.0):
    """Return the (y, x) path of the distance ridge, from one tip to the other."""
    mk = (mask > 0)
    if mk.sum() < 60: return None
    dt = cv2.distanceTransform(mk.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(mk)
    # 1) the two tips, in the pure geodesic distance
    c1 = np.where(mk, 1.0, np.inf)
    m = MCP_Geometric(c1, fully_connected=True)
    d, _ = m.find_costs([(int(ys[0]), int(xs[0]))])
    d = np.where(mk & np.isfinite(d), d, -1.0)
    A = np.unravel_index(int(np.argmax(d)), d.shape)
    m = MCP_Geometric(c1, fully_connected=True)
    d, _ = m.find_costs([A]); d = np.where(mk & np.isfinite(d), d, -1.0)
    B = np.unravel_index(int(np.argmax(d)), d.shape)
    # 2) path between the two tips, constrained on the ridge of the distance map
    w = 1.0 + alpha * (1.0 - dt / (dt.max() + 1e-9)) ** k
    m2 = MCP_Geometric(np.where(mk, w, np.inf), fully_connected=True)
    m2.find_costs([A])
    try: path = np.array(m2.traceback(B), float)
    except Exception: return None
    return path if len(path) >= 12 else None

def smooth(path, n=200, win=9):
    """Resample to n points then smooth by moving average, the axis stays inside the body."""
    d = np.r_[0.0, np.cumsum(np.hypot(*(np.diff(path, axis=0).T)))]
    if d[-1] <= 0: return None, 0.0
    u = np.linspace(0, d[-1], n)
    P = np.stack([np.interp(u, d, path[:, 0]), np.interp(u, d, path[:, 1])], 1)
    if win > 1:
        ker = np.ones(win) / win
        pad = win // 2
        for j in (0, 1):
            v = np.r_[np.repeat(P[0, j], pad), P[:, j], np.repeat(P[-1, j], pad)]
            P[:, j] = np.convolve(v, ker, mode="valid")[:n]
    return P, float(d[-1])

def width_profile(P, mask, half):
    """Mask width on the cut perpendicular to the axis, at every point."""
    H, W = mask.shape; n = len(P); wid = np.zeros(n)
    for i in range(n):
        j0, j1 = max(0, i - 2), min(n - 1, i + 2)
        ty, tx = P[j1] - P[j0]; nn = np.hypot(ty, tx) + 1e-9
        py, px = -tx / nn, ty / nn
        cnt = 0
        for r in range(-half, half + 1):
            yy, xx = int(round(P[i][0] + py * r)), int(round(P[i][1] + px * r))
            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] > 0: cnt += 1
        wid[i] = cnt
    return wid
