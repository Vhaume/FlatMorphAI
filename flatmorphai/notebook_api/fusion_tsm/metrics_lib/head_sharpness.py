#!/usr/bin/env python
# coding: utf-8
"""head_sharpness.py -- centerline curving sharpness at the HEAD, over a 3-mask window (T-1, T0, T+1).

The existing be_head_curv reports the sharper END's curvature and cannot tell head from tail, so a sharp tail fools it.
This identifies the head as the more-mobile tip (the worm leads with its head), skeletonizes the T0 mask, orders the
centerline head-to-tail, and measures how sharply the centerline curves in the HEAD region versus the tail. The
hypothesis: a Snake-like head curves sharply while a Wildtype head is flatter. Angles are scale-free, the contrast is a
bounded fraction. No GPU.

Backing: Zhang, Suen, A fast parallel algorithm for thinning digital patterns, Communications of the ACM 1984,
DOI 10.1145/357994.358023 (skeleton). van der Walt et al., scikit-image, PeerJ 2014, DOI 10.7717/peerj.453.
"""
import numpy as np, cv2
from skimage.morphology import skeletonize

HEADSHARP_COLS = ['head_sharp', 'head_tip_angle', 'head_sharp_frac', 'head_sharp_chg']


def _largest(m):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))).astype(np.uint8)


def _tips_axis(m):
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _ordered_centerline(m):
    sk = skeletonize(m > 0)
    ys, xs = np.where(sk)
    if len(xs) < 10:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    order = np.argsort((pts - c) @ ax)
    return pts[order]


def _region_turning(path, n):
    p = path[:max(4, n)]
    if len(p) < 4:
        return 0.0
    seg = np.diff(p, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    return float(np.abs(np.diff(np.unwrap(ang))).sum())


def _tip_angle(path, n=8):
    if len(path) < 2 * n + 1:
        n = max(2, len(path) // 4)
    a = path[n] - path[0]
    b = path[2 * n] - path[0]
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-6 or db < 1e-6:
        return float(np.pi)
    cos = float(np.clip(np.dot(a, b) / (da * db), -1, 1))
    return float(np.arccos(cos))


def _head_metrics(m, head_is_min):
    path = _ordered_centerline(m)
    if path is None:
        return None
    if not head_is_min:
        path = path[::-1]
    L = len(path)
    k = max(5, L // 4)
    head_turn = _region_turning(path, k)
    tail_turn = _region_turning(path[::-1], k)
    tip_ang = _tip_angle(path)
    return head_turn, tail_turn, tip_ang


def head_sharpness(m_prev, m_cur, m_next):
    out = {kk: np.nan for kk in HEADSHARP_COLS}
    masks = [_largest(m) for m in (m_prev, m_cur, m_next)]
    if any(m.sum() < 25 for m in masks):
        return out
    c0, ax = _tips_axis(masks[1])
    tmin, tmax = [], []
    for m in masks:
        ys, xs = np.where(m > 0)
        a = (np.stack([xs, ys], 1).astype(np.float32) - c0) @ ax
        tmin.append(float(a.min()))
        tmax.append(float(a.max()))
    head_is_min = np.std(tmin) >= np.std(tmax)
    per = [_head_metrics(m, head_is_min) for m in masks]
    if per[1] is None:
        return out
    head_turn, tail_turn, tip_ang = per[1]
    out['head_sharp'] = float(head_turn)
    out['head_tip_angle'] = float(tip_ang)
    out['head_sharp_frac'] = float(head_turn / (head_turn + tail_turn + 1e-6))
    hs = [p[0] for p in per if p is not None]
    out['head_sharp_chg'] = float(max(hs) - min(hs)) if len(hs) >= 2 else 0.0
    return out
