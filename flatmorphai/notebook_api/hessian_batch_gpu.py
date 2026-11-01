# -*- coding: utf-8 -*-
"""Batched GPU twin of ``skimage.feature.corner._hessian_matrix_with_gaussian``.

Used by fold_twist (curated dense maps) and by skimage's frangi / sato /
meijering (all call ``hessian_matrix(..., use_gaussian_derivatives=True)``).
skimage computes, for one sigma, two successive first-order Gaussian-derivative
filters (sigma/sqrt(2), truncate 8, or 100 when sigma <= 1, mode 'reflect'):

    grad_r = G_c^0( G_r^1(I) )          grad_c = G_c^1( G_r^0(I) )
    H_rr   = G_c^0( G_r^1(grad_r) )     H_rc   = G_c^1( G_r^0(grad_r) )
    H_cc   = G_c^1( G_r^0(grad_c) )

Here ALL the sigmas requested for one image are done in ONE batched pass per
stage (4 grouped convolutions in total), with the very same scipy kernels
(``_gaussian_kernel1d``, reversed as ``gaussian_filter1d`` does) zero-padded to
a common length, and the same half-sample-symmetric ('reflect') borders.

Precision: float32 on the GPU (relative error ~1e-7 vs skimage). The families
fed by these maps (frangi, hess, sato, meijering, curated) were measured to
tolerate >= 1e-3 relative without any class change. ``hessian_batch_numpy`` is
the reference of the same algorithm (float64, per-pass rounding like scipy) used
to validate the batching logic itself.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from scipy.ndimage._filters import _gaussian_kernel1d
except Exception:  # noqa: BLE001
    _gaussian_kernel1d = None

_SQ1_2 = 1.0 / math.sqrt(2.0)


def prepare_image(image):
    """Same conversion as skimage (img_as_float + supported float dtype)."""
    from skimage.util import img_as_float
    from skimage._shared.utils import _supported_float_type
    im = img_as_float(image)
    dt = _supported_float_type(im.dtype)
    return im.astype(dt, copy=False), dt


def _plan(sigmas):
    """Per sigma: (w0, w1) correlate weights (kernel[::-1]) and radius."""
    out = []
    for s in sigmas:
        s = float(s)
        truncate = 8 if s > 1 else 100
        ss = _SQ1_2 * s
        r = int(truncate * float(ss) + 0.5)
        w0 = _gaussian_kernel1d(ss, 0, r)[::-1].copy()
        w1 = _gaussian_kernel1d(ss, 1, r)[::-1].copy()
        out.append((w0, w1, r))
    return out


def _padded(w, R):
    """Zero-pad a (2r+1) weight vector to length 2R+1, centred."""
    r = (len(w) - 1) // 2
    z = np.zeros(2 * R + 1, dtype=np.float64)
    z[R - r:R + r + 1] = w
    return z


def _sym_index(n, R):
    """Indices of a half-sample symmetric ('reflect') padding of R on n."""
    i = np.arange(-R, n + R)
    i = np.where(i < 0, -i - 1, i)
    i = np.where(i >= n, 2 * n - i - 1, i)
    return i


def _corr_axis_np(x, w, axis, R, out_dtype):
    """scipy correlate1d(mode='reflect', origin=0) along axis, float64 accum,
    result stored in out_dtype (scipy stores each pass in the input dtype)."""
    n = x.shape[axis]
    xp = np.take(x.astype(np.float64), _sym_index(n, R), axis=axis)
    acc = np.zeros(x.shape, dtype=np.float64)
    for j in range(2 * R + 1):
        if w[j] == 0.0:
            continue
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(j, j + n)
        acc += w[j] * xp[tuple(sl)]
    return acc.astype(out_dtype)


def hessian_batch_numpy(image, sigmas):
    """Reference of the batched algorithm (CPU, numpy). {sigma: [Hrr, Hrc, Hcc]}."""
    im, dt = prepare_image(image)
    plan = _plan(sigmas)
    R = max(r for _w0, _w1, r in plan)
    out = {}
    for s, (w0, w1, _r) in zip(sigmas, plan):
        p0, p1 = _padded(w0, R), _padded(w1, R)
        g_r = _corr_axis_np(_corr_axis_np(im, p1, 0, R, dt), p0, 1, R, dt)
        g_c = _corr_axis_np(_corr_axis_np(im, p0, 0, R, dt), p1, 1, R, dt)
        h_rr = _corr_axis_np(_corr_axis_np(g_r, p1, 0, R, dt), p0, 1, R, dt)
        h_rc = _corr_axis_np(_corr_axis_np(g_r, p0, 0, R, dt), p1, 1, R, dt)
        h_cc = _corr_axis_np(_corr_axis_np(g_c, p0, 0, R, dt), p1, 1, R, dt)
        out[s] = [h_rr, h_rc, h_cc]
    return out


_TORCH = None
_DEV = None


def _torch():
    global _TORCH, _DEV
    if _TORCH is None:
        import torch
        _TORCH = torch
        _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _TORCH, _DEV


def hessian_batch_torch(image, sigmas, dtype="float32"):
    """Batched GPU version (same stages as hessian_batch_numpy).
    Returns {sigma: [Hrr, Hrc, Hcc]} as numpy arrays in skimage's float dtype,
    or None when the case is not supported (caller falls back to skimage)."""
    torch, dev = _torch()
    F = torch.nn.functional
    im, dt = prepare_image(image)
    if im.ndim != 2:
        return None
    plan = _plan(sigmas)
    R = max(r for _w0, _w1, r in plan)
    H, W = im.shape
    if min(H, W) <= R:            # single symmetric reflection only
        return None
    K = len(sigmas)
    tdt = torch.float32 if dtype == "float32" else torch.float16
    W0 = np.stack([_padded(w0, R) for w0, _w1, _r in plan])   # (K, L)
    W1 = np.stack([_padded(w1, R) for _w0, w1, _r in plan])
    w0 = torch.from_numpy(W0).to(dev, tdt)
    w1 = torch.from_numpy(W1).to(dev, tdt)
    ir = torch.from_numpy(_sym_index(H, R)).to(dev)
    ic = torch.from_numpy(_sym_index(W, R)).to(dev)
    L = 2 * R + 1

    def pass_r(x, wts):                # x: (1,C,H,W), wts: (C,L) -> along rows
        xp = x.index_select(2, ir)
        return F.conv2d(xp, wts.view(-1, 1, L, 1), groups=x.shape[1])

    def pass_c(x, wts):                # along columns
        xp = x.index_select(3, ic)
        return F.conv2d(xp, wts.view(-1, 1, 1, L), groups=x.shape[1])

    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(im)).to(dev, tdt)[None, None]
        # stage A: rows with [w1_s, w0_s] for every sigma, then columns with [w0_s, w1_s]
        a = pass_r(x.expand(1, 2 * K, H, W), torch.cat([w1, w0], 0))
        g = pass_c(a, torch.cat([w0, w1], 0))           # channels: [g_r(s)..., g_c(s)...]
        g_r, g_c = g[:, :K], g[:, K:]
        # stage B: inputs [g_r, g_r, g_c] ; rows [w1, w0, w0] ; columns [w0, w1, w1]
        b_in = torch.cat([g_r, g_r, g_c], 1)
        b = pass_r(b_in, torch.cat([w1, w0, w0], 0))
        h = pass_c(b, torch.cat([w0, w1, w1], 0))
        hn = h[0].float().cpu().numpy()
    out = {}
    for k, s in enumerate(sigmas):
        out[s] = [hn[k].astype(dt), hn[K + k].astype(dt), hn[2 * K + k].astype(dt)]
    return out


def max_rel_err(ref_list, new_list):
    """max |new-ref| / max|ref| over the three components."""
    e = 0.0
    for a, b in zip(ref_list, new_list):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        e = max(e, float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-30)))
    return e
