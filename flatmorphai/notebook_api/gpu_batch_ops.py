# -*- coding: utf-8 -*-
"""Batched twins of the two remaining dense image operations of Step 4, written
ONCE on a tiny array backend so the very same code runs in numpy (CPU
reference, used to validate the algorithm against OpenCV / skimage) and in
torch (GPU, many images per call).

1. ``farneback_batch``  : port of OpenCV 5/4.x ``calcOpticalFlowFarneback``
   (CPU path, flags=0: box-filter update, no initial flow). Same pyramid
   (GaussianBlur ksize/sigma rule incl. the fixed [1 2 1]/4 kernel at level 0,
   INTER_LINEAR resize), same polynomial expansion (replicate borders, same
   G^-1 coefficients), same matrix update (bilinear sampling, 5-pixel border
   attenuation) and same box-blur/solve iterations. Not bit-identical (float
   rounding order differs), validated by end-point error vs cv2.

2. ``zhang_skeleton_batch`` : skimage ``skeletonize`` (Zhang-Suen, 2-D).
   Each sub-pass reads the image of the previous sub-pass only, so it is a
   pure 3x3 neighbourhood code + look-up table: BIT-IDENTICAL, any batch.

Nothing in the app imports this module yet (test_gpu_lot.py only).
"""
from __future__ import annotations

import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
class NumpyBackend:
    name = "numpy"

    def __init__(self, dtype="float64"):
        self.f = np.float64 if dtype == "float64" else np.float32

    def asarray(self, a):
        return np.asarray(a, dtype=self.f)

    def to_numpy(self, a):
        return np.asarray(a)

    def zeros(self, shape):
        return np.zeros(shape, dtype=self.f)

    def iarray(self, a):
        return np.asarray(a, dtype=np.int64)

    def pad_edge(self, x, py, px):
        return np.pad(x, ((0, 0), (py, py), (px, px)), mode="edge")

    def pad_reflect101(self, x, py, px):
        return np.pad(x, ((0, 0), (py, py), (px, px)), mode="reflect")

    def floor_int(self, x):
        return np.floor(x).astype(np.int64)

    def to_float(self, x):
        return x.astype(self.f)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def where(self, c, a, b):
        return np.where(c, a, b)

    def bidx(self, B):
        return np.arange(B).reshape(B, 1, 1)

    def u8(self, a):
        return np.asarray(a, dtype=np.uint8)

    def any(self, x):
        return bool(np.any(x))


class TorchBackend:
    name = "torch"

    def __init__(self, dtype="float32", device=None):
        import torch
        self.t = torch
        self.F = torch.nn.functional
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.f = torch.float32 if dtype == "float32" else torch.float64

    def asarray(self, a):
        if isinstance(a, self.t.Tensor):
            return a.to(self.dev, self.f)
        return self.t.as_tensor(np.asarray(a), dtype=self.f, device=self.dev)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()

    def zeros(self, shape):
        return self.t.zeros(shape, dtype=self.f, device=self.dev)

    def iarray(self, a):
        return self.t.as_tensor(np.asarray(a), dtype=self.t.int64, device=self.dev)

    def pad_edge(self, x, py, px):
        return self.F.pad(x[:, None], (px, px, py, py), mode="replicate")[:, 0]

    def pad_reflect101(self, x, py, px):
        return self.F.pad(x[:, None], (px, px, py, py), mode="reflect")[:, 0]

    def floor_int(self, x):
        return self.t.floor(x).long()

    def to_float(self, x):
        return x.to(self.f)

    def clip(self, x, lo, hi):
        return self.t.clamp(x, lo, hi)

    def where(self, c, a, b):
        return self.t.where(c, a, b)

    def bidx(self, B):
        return self.t.arange(B, device=self.dev).view(B, 1, 1)

    def u8(self, a):
        return self.t.as_tensor(np.asarray(a, dtype=np.uint8), device=self.dev)

    def any(self, x):
        return bool(x.any().item())


def _f32(v):
    return float(np.float32(v))


# ---------------------------------------------------------------------------
# OpenCV helpers (same formulas as imgproc)
# ---------------------------------------------------------------------------
_SMALL_GAUSS = {1: [1.0], 3: [0.25, 0.5, 0.25],
                5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
                7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]}


def _cv_gauss_kernel(n, sigma):
    if sigma <= 0 and n in _SMALL_GAUSS:
        return [_f32(v) for v in _SMALL_GAUSS[n]]
    s = sigma if sigma > 0 else ((n - 1) * 0.5 - 1) * 0.3 + 0.8
    sc = -0.5 / (s * s)
    cf = [_f32(math.exp(sc * (i - (n - 1) * 0.5) ** 2)) for i in range(n)]
    tot = 1.0 / sum(cf)
    return [_f32(c * tot) for c in cf]


def _sep_filter(be, x, k, reflect=True):
    """OpenCV sepFilter2D, rows first then columns, BORDER_REFLECT_101."""
    r = len(k) // 2
    H, W = x.shape[1], x.shape[2]
    xp = be.pad_reflect101(x, 0, r)
    t = k[0] * xp[:, :, 0:W]
    for j in range(1, len(k)):
        t = t + k[j] * xp[:, :, j:j + W]
    tp = be.pad_reflect101(t, r, 0)
    o = k[0] * tp[:, 0:H, :]
    for j in range(1, len(k)):
        o = o + k[j] * tp[:, j:j + H, :]
    return o


def _lin_axis(S, D):
    scale = S / float(D)
    d = np.arange(D, dtype=np.float64)
    fx = (d + 0.5) * scale - 0.5
    sx = np.floor(fx).astype(np.int64)
    f = (fx - sx).astype(np.float32)
    neg = sx < 0
    f[neg] = 0.0
    sx[neg] = 0
    big = sx >= S - 1
    f[big] = 0.0
    sx[big] = S - 1
    return sx, np.minimum(sx + 1, S - 1), f


def _resize_linear(be, x, h, w):
    H, W = x.shape[1], x.shape[2]
    if (H, W) == (h, w):
        return x
    x0, x1, fx = _lin_axis(W, w)
    y0, y1, fy = _lin_axis(H, h)
    X0, X1 = be.iarray(x0), be.iarray(x1)
    FX = be.asarray(fx).reshape(1, 1, w)
    t = x[:, :, X0] * (1 - FX) + x[:, :, X1] * FX
    Y0, Y1 = be.iarray(y0), be.iarray(y1)
    FY = be.asarray(fy).reshape(1, h, 1)
    return t[:, Y0, :] * (1 - FY) + t[:, Y1, :] * FY


def _poly_prepare(n, sigma):
    if sigma < 1.1920929e-07:
        sigma = n * 0.3
    xs = range(-n, n + 1)
    g = {x: _f32(math.exp(-x * x / (2 * sigma * sigma))) for x in xs}
    s = 1.0 / sum(g.values())
    g = {x: _f32(g[x] * s) for x in xs}
    xg = {x: _f32(x * g[x]) for x in xs}
    xxg = {x: _f32(x * x * g[x]) for x in xs}
    G = np.zeros((6, 6))
    for y in xs:
        for x in xs:
            G[0, 0] += g[y] * g[x]
            G[1, 1] += g[y] * g[x] * x * x
            G[3, 3] += g[y] * g[x] * x ** 4
            G[5, 5] += g[y] * g[x] * x * x * y * y
    G[2, 2] = G[0, 3] = G[0, 4] = G[3, 0] = G[4, 0] = G[1, 1]
    G[4, 4] = G[3, 3]
    G[3, 4] = G[4, 3] = G[5, 5]
    iG = np.linalg.inv(G)
    return g, xg, xxg, iG[1, 1], iG[0, 3], iG[3, 3], iG[5, 5]


def _poly_exp(be, I, n, sigma):
    g, xg, xxg, ig11, ig03, ig33, ig55 = _poly_prepare(n, sigma)
    H, W = I.shape[1], I.shape[2]
    P = be.pad_edge(I, n, 0)
    sl = lambda k: P[:, n + k:n + k + H, :]          # noqa: E731
    r0 = g[0] * sl(0)
    r1 = None
    r2 = None
    for k in range(1, n + 1):
        a, b = sl(-k), sl(k)
        p = a + b
        r0 = r0 + g[k] * p
        t1 = xg[k] * (b - a)
        t2 = xxg[k] * p
        r1 = t1 if r1 is None else r1 + t1
        r2 = t2 if r2 is None else r2 + t2
    Q0, Q1, Q2 = (be.pad_edge(r, 0, n) for r in (r0, r1, r2))
    hs = lambda Q, k: Q[:, :, n + k:n + k + W]       # noqa: E731
    b1 = g[0] * hs(Q0, 0)
    b3 = g[0] * hs(Q1, 0)
    b5 = g[0] * hs(Q2, 0)
    b2 = b4 = b6 = None
    for k in range(1, n + 1):
        pr, mr = hs(Q0, k), hs(Q0, -k)
        tg = pr + mr
        b1 = b1 + g[k] * tg
        t4 = xxg[k] * tg
        t2 = xg[k] * (pr - mr)
        b3 = b3 + g[k] * (hs(Q1, k) + hs(Q1, -k))
        t6 = xg[k] * (hs(Q1, k) - hs(Q1, -k))
        b5 = b5 + g[k] * (hs(Q2, k) + hs(Q2, -k))
        b2 = t2 if b2 is None else b2 + t2
        b4 = t4 if b4 is None else b4 + t4
        b6 = t6 if b6 is None else b6 + t6
    return [b3 * ig11, b2 * ig11, b1 * ig03 + b5 * ig33, b1 * ig03 + b4 * ig33, b6 * ig55]


_BORDER = [0.14, 0.14, 0.4472, 0.4472, 0.4472]


def _border_map(be, H, W):
    def axis(n):
        v = np.ones(n, dtype=np.float32)
        for i in range(min(5, n)):
            v[i] *= np.float32(_BORDER[i])
        for i in range(max(0, n - 5), n):
            v[i] *= np.float32(_BORDER[n - i - 1])
        return v
    return be.asarray(np.outer(axis(H), axis(W)).astype(np.float32))[None]


def _update_matrices(be, R0, R1, dx, dy, grid):
    X, Y, bmap, bi = grid
    B, H, W = dx.shape
    fx = X + dx
    fy = Y + dy
    x1 = be.floor_int(fx)
    y1 = be.floor_int(fy)
    ax = fx - be.to_float(x1)
    ay = fy - be.to_float(y1)
    ok = (x1 >= 0) & (x1 < W - 1) & (y1 >= 0) & (y1 < H - 1)
    xc = be.clip(x1, 0, W - 2)
    yc = be.clip(y1, 0, H - 2)
    a00 = (1 - ax) * (1 - ay)
    a01 = ax * (1 - ay)
    a10 = (1 - ax) * ay
    a11 = ax * ay

    def samp(c):
        R = R1[c]
        return (a00 * R[bi, yc, xc] + a01 * R[bi, yc, xc + 1]
                + a10 * R[bi, yc + 1, xc] + a11 * R[bi, yc + 1, xc + 1])
    s = [samp(c) for c in range(5)]
    r2 = be.where(ok, s[0], 0 * s[0])
    r3 = be.where(ok, s[1], 0 * s[1])
    r4 = be.where(ok, (R0[2] + s[2]) * 0.5, R0[2])
    r5 = be.where(ok, (R0[3] + s[3]) * 0.5, R0[3])
    r6 = be.where(ok, (R0[4] + s[4]) * 0.25, R0[4] * 0.5)
    r2 = (R0[0] - r2) * 0.5
    r3 = (R0[1] - r3) * 0.5
    r2 = r2 + r4 * dy + r6 * dx
    r3 = r3 + r6 * dy + r5 * dx
    r2, r3, r4, r5, r6 = (r * bmap for r in (r2, r3, r4, r5, r6))
    return [r4 * r4 + r6 * r6, (r4 + r5) * r6, r5 * r5 + r6 * r6,
            r4 * r2 + r6 * r3, r6 * r2 + r5 * r3]


def _box(be, x, m):
    H, W = x.shape[1], x.shape[2]
    xp = be.pad_edge(x, m, 0)
    v = xp[:, 0:H, :]
    for j in range(1, 2 * m + 1):
        v = v + xp[:, j:j + H, :]
    vp = be.pad_edge(v, 0, m)
    o = vp[:, :, 0:W]
    for j in range(1, 2 * m + 1):
        o = o + vp[:, :, j:j + W]
    return o


def _solve(be, M, block):
    m = block // 2
    sc = 1.0 / (block * block)
    g11, g12, g22, h1, h2 = (_box(be, c, m) * sc for c in M)
    idet = 1.0 / (g11 * g22 - g12 * g12 + 1e-3)
    return (g11 * h2 - g12 * h1) * idet, (g22 * h1 - g12 * h2) * idet


def farneback_batch(prevs, nexts, pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, backend=None):
    """prevs/nexts: sequences of same-shape 2-D uint8/float images.
    Returns a numpy array (B, H, W, 2) float32 (dx, dy), like cv2."""
    be = backend or NumpyBackend("float64")
    I0 = be.asarray(np.stack([np.asarray(p) for p in prevs]))
    I1 = be.asarray(np.stack([np.asarray(p) for p in nexts]))
    B, H, W = I0.shape
    k, scale = 0, 1.0
    while k < levels:
        scale *= pyr_scale
        if W * scale < 32 or H * scale < 32:
            break
        k += 1
    levels = k
    dx = dy = None
    for k in range(levels, -1, -1):
        scale = 1.0
        for _i in range(k):
            scale *= pyr_scale
        sigma = (1.0 / scale - 1) * 0.5
        ksz = max(int(np.rint(sigma * 5)) | 1, 3)
        w = int(np.rint(W * scale))
        h = int(np.rint(H * scale))
        kern = _cv_gauss_kernel(ksz, sigma)
        R = []
        for I in (I0, I1):
            R.append(_poly_exp(be, _resize_linear(be, _sep_filter(be, I, kern), h, w),
                               poly_n, poly_sigma))
        if dx is None:
            dx = be.zeros((B, h, w))
            dy = be.zeros((B, h, w))
        else:
            dx = _resize_linear(be, dx, h, w) * (1.0 / pyr_scale)
            dy = _resize_linear(be, dy, h, w) * (1.0 / pyr_scale)
        X = be.asarray(np.arange(w, dtype=np.float32)).reshape(1, 1, w)
        Y = be.asarray(np.arange(h, dtype=np.float32)).reshape(1, h, 1)
        grid = (X, Y, _border_map(be, h, w), be.bidx(B))
        M = _update_matrices(be, R[0], R[1], dx, dy, grid)
        for it in range(iterations):
            dx, dy = _solve(be, M, winsize)
            if it < iterations - 1:
                M = _update_matrices(be, R[0], R[1], dx, dy, grid)
    out = np.stack([be.to_numpy(dx), be.to_numpy(dy)], axis=-1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Zhang-Suen skeleton (skimage weights: 1 NW, 2 N, 4 NE, 8 E, 16 SE, 32 S,
# 64 SW, 128 W). 1 = removable in pass 1 only, 2 = pass 2 only, 3 = both.
# ---------------------------------------------------------------------------
# Verbatim look-up table of scikit-image 0.25 _fast_skeletonize
# (skimage/morphology/_skeletonize_various_cy.pyx, BSD-3-Clause).
_SKIMAGE_ZHANG_LUT = [
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0,
    1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
    3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
    2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3,
    0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0,
    2, 0, 0, 0]


def zhang_lut():
    return np.asarray(_SKIMAGE_ZHANG_LUT, dtype=np.uint8)


_W8 = ((-1, -1, 1), (-1, 0, 2), (-1, 1, 4), (0, 1, 8),
       (1, 1, 16), (1, 0, 32), (1, -1, 64), (0, -1, 128))


def zhang_skeleton_batch(masks, backend=None, max_iter=10000, max_pixels=40_000_000):
    """masks: sequence of 2-D arrays (any sizes). Returns list of bool arrays,
    identical to skimage.morphology.skeletonize(mask) (2-D, Zhang). Masks are
    sorted by size and packed into canvases of <= max_pixels (uint8)."""
    be = backend or NumpyBackend("float32")
    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1])
    out = [None] * len(masks)
    k = 0
    while k < len(order):
        grp = [order[k]]
        Hm, Wm = masks[order[k]].shape
        k += 1
        while k < len(order):
            h, w = masks[order[k]].shape
            H2, W2 = max(Hm, h), max(Wm, w)
            if (len(grp) + 1) * (H2 + 2) * (W2 + 2) > max_pixels:
                break
            grp.append(order[k]); Hm, Wm = H2, W2
            k += 1
        res = _zhang_canvas(be, [masks[i] for i in grp], Hm, Wm, max_iter)
        for i, r in zip(grp, res):
            out[i] = r
    return out


def _zhang_canvas(be, masks, Hm, Wm, max_iter):
    canvas = np.zeros((len(masks), Hm + 2, Wm + 2), dtype=np.uint8)
    for i, m in enumerate(masks):
        canvas[i, 1:1 + m.shape[0], 1:1 + m.shape[1]] = (np.asarray(m) != 0)
    S = be.u8(canvas)
    lut = be.u8(zhang_lut())
    if be.name == "torch":
        lut = lut.long()
    H, W = Hm, Wm
    for _it in range(max_iter):
        changed = False
        for first in (True, False):
            c = S[:, 1:H + 1, 1:W + 1]
            if be.name == "torch":
                code = c.to(be.t.int32) * 0
            else:
                code = np.zeros(c.shape, dtype=np.int32)
            for dy, dx, wgt in _W8:
                code += wgt * S[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            v = lut[code.long()] if be.name == "torch" else lut[code]
            rm = (c != 0) & ((v == 3) | (v == (1 if first else 2)))
            if be.any(rm):
                changed = True
                S[:, 1:H + 1, 1:W + 1][rm] = 0
        if not changed:
            break
    S = be.to_numpy(S)
    return [S[i, 1:1 + m.shape[0], 1:1 + m.shape[1]].astype(bool) for i, m in enumerate(masks)]


# ---------------------------------------------------------------------------
# Farneback v2 (torch only): same algorithm as farneback_batch, but the filters
# are FUSED into a handful of cuDNN calls (few kernel launches, which dominate
# under WSL): Gaussian pyramid = 2 conv2d, polynomial expansion = 2 conv2d
# (the G^-1 combination folded into the horizontal weights), box blur =
# avg_pool2d, resize = F.interpolate (bilinear, align_corners=False: the same
# source-index formula as cv::resize INTER_LINEAR), bilinear sampling = 4
# gathers on the 5 stacked channels.
#   store="float32" : everything float32.
#   store="float16" : big tensors (images, R, M) stored in float16, images
#                     pre-scaled by 1/16 (M would overflow float16 otherwise;
#                     the flow is scale-invariant once the 1e-3 regulariser is
#                     scaled by s^4), the 2x2 solve in float32.
# gpu_lock(): one process at a time on the GPU + empty_cache() after, so 6
# worker processes never hold 6 working sets in VRAM (WSL pages to host RAM).
# ---------------------------------------------------------------------------
import contextlib as _ctxlib


import threading as _thr
_GL_LOCAL = _thr.local()
_GL_THREAD = _thr.Lock()


@_ctxlib.contextmanager
def gpu_lock(path=None):
    """Cross-process GPU lock (flock), RE-ENTRANT in the same thread (a second
    flock on a new descriptor of the same process would deadlock)."""
    depth = getattr(_GL_LOCAL, "depth", 0)
    if depth:
        _GL_LOCAL.depth = depth + 1
        try:
            yield
        finally:
            _GL_LOCAL.depth -= 1
        return
    if os.name == "nt":
        # Native Windows has no flock: the same exclusive lock with msvcrt, on
        # the first byte of the same lock file (LK_LOCK gives up after ten
        # one-second attempts, hence the loop: the lock must wait as flock does).
        import msvcrt
        import tempfile
        import time as _time
        p = path or os.path.join(tempfile.gettempdir(), "flatmorphai_gpu.lock")
        with _GL_THREAD:
            fh = open(p, "a+b")
            try:
                while True:
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        _time.sleep(0.05)
                _GL_LOCAL.depth = 1
                try:
                    yield
                finally:
                    _GL_LOCAL.depth = 0
            finally:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
                finally:
                    fh.close()
        return
    import fcntl
    import tempfile
    p = path or os.path.join(tempfile.gettempdir(), "flatmorphai_gpu.lock")
    with _GL_THREAD:
        fh = open(p, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            _GL_LOCAL.depth = 1
            try:
                yield
            finally:
                _GL_LOCAL.depth = 0
        finally:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            finally:
                fh.close()


_V2_CACHE = {}


def _v2_poly_weights(torch, n, sigma, dev, dt):
    key = ("poly", n, sigma, str(dev), str(dt))
    if key in _V2_CACHE:
        return _V2_CACHE[key]
    g, xg, xxg, ig11, ig03, ig33, ig55 = _poly_prepare(n, sigma)
    L = 2 * n + 1
    gs = np.array([g[abs(k)] for k in range(-n, n + 1)], np.float64)
    xa = np.array([(1 if k > 0 else -1 if k < 0 else 0) * xg[abs(k)] for k in range(-n, n + 1)], np.float64)
    xxs = np.array([xxg[abs(k)] if k else 0.0 for k in range(-n, n + 1)], np.float64)
    Wv = np.stack([gs, xa, xxs])[:, None, :, None]                  # (3,1,L,1)
    Wh = np.zeros((5, 3, 1, L))
    Wh[0, 1, 0] = ig11 * gs
    Wh[1, 0, 0] = ig11 * xa
    Wh[2, 0, 0] = ig03 * gs
    Wh[2, 2, 0] = ig33 * gs
    Wh[3, 0, 0] = ig03 * gs + ig33 * xxs
    Wh[4, 1, 0] = ig55 * xa
    out = (torch.as_tensor(Wv, dtype=dt, device=dev), torch.as_tensor(Wh, dtype=dt, device=dev))
    _V2_CACHE[key] = out
    return out


def farneback_v2(prevs, nexts, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, store="float32", device=None):
    import torch
    F = torch.nn.functional
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    st = torch.float16 if store == "float16" else torch.float32
    s_img = 1.0 / 16.0 if store == "float16" else 1.0
    reg = 1e-3 * s_img ** 4
    with torch.inference_mode():
        I = torch.as_tensor(np.stack([np.asarray(p) for p in prevs] + [np.asarray(p) for p in nexts]))
        I = (I.to(dev, torch.float32) * s_img)[:, None]               # (2B,1,H,W) f32
        B2, _, H, W = I.shape
        B = B2 // 2
        k, scale = 0, 1.0
        while k < levels:
            scale *= pyr_scale
            if W * scale < 32 or H * scale < 32:
                break
            k += 1
        levels = k
        Wv, Wh = _v2_poly_weights(torch, poly_n, poly_sigma, dev, st)
        n = poly_n
        flow = None
        for k in range(levels, -1, -1):
            scale = 1.0
            for _i in range(k):
                scale *= pyr_scale
            sigma = (1.0 / scale - 1) * 0.5
            ksz = max(int(np.rint(sigma * 5)) | 1, 3)
            w = int(np.rint(W * scale))
            h = int(np.rint(H * scale))
            kern = torch.as_tensor(_cv_gauss_kernel(ksz, sigma), dtype=torch.float32, device=dev)
            r = ksz // 2
            x = F.conv2d(F.pad(I, (r, r, 0, 0), mode="reflect"), kern.view(1, 1, 1, ksz))
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kern.view(1, 1, ksz, 1))
            if (h, w) != (H, W):
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            x = x.to(st)
            rows = F.conv2d(F.pad(x, (0, 0, n, n), mode="replicate"), Wv)       # (2B,3,h,w)
            R = F.conv2d(F.pad(rows, (n, n, 0, 0), mode="replicate"), Wh)       # (2B,5,h,w)
            del rows, x
            R0, R1 = R[:B], R[B:]
            if flow is None:
                flow = torch.zeros((B, 2, h, w), dtype=torch.float32, device=dev)
            else:
                flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * (1.0 / pyr_scale)
            X = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, w)
            Y = torch.arange(h, device=dev, dtype=torch.float32).view(1, h, 1)
            bmap = _border_map(TorchBackend("float32", device=str(dev)), h, w)[:, None]   # (1,1,h,w)
            R1f = R1.reshape(B, 5, h * w)

            def update(fl):
                dx, dy = fl[:, 0], fl[:, 1]
                fx = X + dx
                fy = Y + dy
                x1 = torch.floor(fx)
                y1 = torch.floor(fy)
                ax = (fx - x1)[:, None]
                ay = (fy - y1)[:, None]
                x1 = x1.long()
                y1 = y1.long()
                ok = ((x1 >= 0) & (x1 < w - 1) & (y1 >= 0) & (y1 < h - 1))[:, None]
                xc = x1.clamp(0, w - 2)
                yc = y1.clamp(0, h - 2)
                base = (yc * w + xc).view(B, 1, h * w).expand(B, 5, h * w)
                g00 = R1f.gather(2, base).view(B, 5, h, w).float()
                g01 = R1f.gather(2, base + 1).view(B, 5, h, w).float()
                g10 = R1f.gather(2, base + w).view(B, 5, h, w).float()
                g11 = R1f.gather(2, base + w + 1).view(B, 5, h, w).float()
                s = ((1 - ax) * (1 - ay)) * g00 + (ax * (1 - ay)) * g01 + ((1 - ax) * ay) * g10 + (ax * ay) * g11
                R0c = R0.float()
                zero = torch.zeros_like(s[:, 0:1])
                r2 = torch.where(ok, s[:, 0:1], zero)
                r3 = torch.where(ok, s[:, 1:2], zero)
                r4 = torch.where(ok, (R0c[:, 2:3] + s[:, 2:3]) * 0.5, R0c[:, 2:3])
                r5 = torch.where(ok, (R0c[:, 3:4] + s[:, 3:4]) * 0.5, R0c[:, 3:4])
                r6 = torch.where(ok, (R0c[:, 4:5] + s[:, 4:5]) * 0.25, R0c[:, 4:5] * 0.5)
                ddx, ddy = dx[:, None], dy[:, None]
                r2 = (R0c[:, 0:1] - r2) * 0.5 + r4 * ddy + r6 * ddx
                r3 = (R0c[:, 1:2] - r3) * 0.5 + r6 * ddy + r5 * ddx
                r2, r3, r4, r5, r6 = (t * bmap for t in (r2, r3, r4, r5, r6))
                return torch.cat([r4 * r4 + r6 * r6, (r4 + r5) * r6, r5 * r5 + r6 * r6,
                                  r4 * r2 + r6 * r3, r6 * r2 + r5 * r3], 1).to(st)

            m = winsize // 2
            M = update(flow)
            for it in range(iterations):
                Mb = F.avg_pool2d(F.pad(M, (m, m, m, m), mode="replicate"), 2 * m + 1, stride=1).float()
                g11_, g12_, g22_, h1_, h2_ = Mb[:, 0], Mb[:, 1], Mb[:, 2], Mb[:, 3], Mb[:, 4]
                idet = 1.0 / (g11_ * g22_ - g12_ * g12_ + reg)
                flow = torch.stack([(g11_ * h2_ - g12_ * h1_) * idet, (g22_ * h1_ - g12_ * h2_) * idet], 1)
                if it < iterations - 1:
                    M = update(flow)
            del R, R0, R1, R1f, M
        out = flow.permute(0, 2, 3, 1).contiguous().float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Gabor bank by FFT on the GPU (torch.fft, float32). Same maths and the same
# FFT-size rule as gabor_gpu._fft_gabor_many (CPU float64): half-sample
# symmetric padding, convolution by the complex kernels, valid crop.
# ---------------------------------------------------------------------------
_GAB_T = {}


def gabor_bank_torch(image, specs, device=None):
    import torch
    from skimage.filters import gabor_kernel
    import scipy.fft as sfft
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape
    key0 = tuple((round(float(f), 6), round(float(t), 6)) for f, t in specs)
    ks = _GAB_T.get(("k", key0))
    if ks is None:
        ks = [gabor_kernel(f, theta=t) for f, t in specs]
        _GAB_T[("k", key0)] = ks
    KH = max(k.shape[0] for k in ks)
    KW = max(k.shape[1] for k in ks)
    fs = tuple(sfft.next_fast_len(int(-(-(n + 2 * (K - 1)) // 64) * 64), True)
               for n, K in ((H, KH), (W, KW)))
    KF = _GAB_T.get(("f", key0, fs, str(dev)))
    with torch.inference_mode():
        if KF is None:
            Z = np.zeros((2 * len(ks), KH, KW), dtype=np.float32)
            for i, k in enumerate(ks):
                kh, kw = k.shape
                oy, ox = (KH - kh) // 2, (KW - kw) // 2
                Z[2 * i, oy:oy + kh, ox:ox + kw] = k.real
                Z[2 * i + 1, oy:oy + kh, ox:ox + kw] = k.imag
            KF = torch.fft.rfft2(torch.as_tensor(Z, device=dev), s=fs)
            if len(_GAB_T) > 24:
                _GAB_T.clear()
            _GAB_T[("f", key0, fs, str(dev))] = KF
        p = np.pad(img, ((KH // 2, KH - 1 - KH // 2), (KW // 2, KW - 1 - KW // 2)), mode="symmetric")
        P = torch.fft.rfft2(torch.as_tensor(p.astype(np.float32), device=dev), s=fs)
        r = torch.fft.irfft2(KF * P[None], s=fs)[:, KH - 1:KH - 1 + H, KW - 1:KW - 1 + W]
        out = r.cpu().numpy().astype(np.float64)
    return [(out[2 * i], out[2 * i + 1]) for i in range(len(specs))]
