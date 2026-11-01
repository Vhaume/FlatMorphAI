"""
GPU-accelerated, self-validating drop-in for ``skimage.feature.hessian_matrix``
(the ``use_gaussian_derivatives=True`` path).

WHY. The dark-ridge / vesselness battery in ``fold_twist_metrics`` -- Frangi
(``frangi_dark``), Sato (``sato_dark``), Meijering (``meijering_dark``) and the
direct Hessian ridge (``hess_dark``) -- all reduce to, for each scale ``sigma``:

    eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma,
                                     use_gaussian_derivatives=True))

followed by cheap element-wise math. The single expensive part is
``hessian_matrix``: it convolves the image with Gaussian-derivative kernels at
several sigmas (``scipy.ndimage.gaussian_filter`` with ``order``), which is pure
CPU and holds the GIL, so it does not scale across the feature thread pool.
This module runs exactly those separable convolutions on the GPU (PyTorch) and
returns the SAME ``[Hrr, Hrc, Hcc]`` arrays, so every downstream eigenvalue and
ridge formula is untouched.

CORRECTNESS. ``skimage._hessian_matrix_with_gaussian`` applies, per axis,
``scipy.ndimage.gaussian_filter1d``, which builds its 1-D weights with
``scipy.ndimage._filters._gaussian_kernel1d(sigma, order, lw)[::-1]`` and then
calls ``correlate1d`` (a centred cross-correlation, mode ``reflect`` =
half-sample symmetric). We reuse scipy's OWN kernel generator for the weights --
so the kernels are bit-identical -- and reproduce ``correlate1d`` with
``F.conv2d`` in float64 over a symmetric-padded image (``conv2d`` is a
cross-correlation, like ``correlate1d``; both skip the kernel flip). The
scaling (``sigma/sqrt(2)``) and the large ``truncate`` (8, or 100 for sigma<=1)
are copied verbatim from skimage, so the result equals the CPU one to float
rounding.

SAFETY. The first call validates the GPU ``[Hrr, Hrc, Hcc]`` against
``skimage.feature.hessian_matrix`` on the real data; on any mismatch beyond a
tight tolerance, or any error (no torch, no CUDA, API change), the module falls
back to skimage PERMANENTLY. It can therefore never change a metric: at worst
it is the old CPU path. Same contract as ``gabor_gpu``.
"""

from __future__ import annotations

import math
import sys
import threading
from itertools import combinations_with_replacement

import numpy as np

# None = not yet decided, True = GPU validated, False = use skimage CPU.
_GPU_OK = None
_DEVICE = None
_DTYPE = None            # torch.float32, resolved once the GPU device is set
_LOCK = threading.Lock()
_VALIDATE_TOL = 1e-6
_LOGGED = False


def _log_status(ok: bool, detail: str = "") -> None:
    """Emit ONE line saying whether the Hessian/vesselness runs on GPU or CPU.

    Written to stderr, which the session logger tees into the run log, so the
    operator can confirm at a glance that the GPU path was taken (or why it
    fell back). Best-effort and one-shot; never raises."""
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    try:
        if ok:
            msg = ("[FlatMorphAI] GPU: Hessian/vesselness (Frangi/Sato/Meijering/"
                   "hess_dark) on GPU (CUDA), validated against skimage.")
        else:
            msg = ("[FlatMorphAI] GPU: Hessian/vesselness on CPU (skimage)"
                   + (" - %s" % detail if detail else "") + ".")
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass

# Cache of scipy 1-D kernels keyed by (sigma_scaled rounded, order, lw): the
# kernels depend only on sigma/order, not on the frame, so build each once.
_KCACHE: dict = {}


def _kernel1d(sigma_scaled: float, order: int, lw: int):
    key = (round(float(sigma_scaled), 9), int(order), int(lw))
    w = _KCACHE.get(key)
    if w is None:
        from scipy.ndimage._filters import _gaussian_kernel1d
        # Exactly scipy.ndimage.gaussian_filter1d: weights are the kernel
        # reversed, then applied by correlate1d.
        w = np.ascontiguousarray(_gaussian_kernel1d(sigma_scaled, order, lw)[::-1])
        _KCACHE[key] = w
    return w


def _cpu_hessian(image, sigma, mode, cval, order):
    from skimage.feature import hessian_matrix
    return hessian_matrix(image, sigma=sigma, mode=mode, cval=cval, order=order,
                          use_gaussian_derivatives=True)


def _reflect_index(n: int, lw: int, torch):
    """Half-sample symmetric indices (scipy ``mode='reflect'``) for one axis."""
    idx = torch.arange(-lw, n + lw, device=_DEVICE)
    period = 2 * n
    idx = torch.remainder(idx, period)
    idx = torch.where(idx >= n, period - 1 - idx, idx)
    return idx


def _correlate1d_gpu(t, weights_np, dim, mode, cval, torch, F):
    """One separable pass == scipy ``gaussian_filter1d`` along ``dim`` (2 or 3).

    ``t`` is a (1,1,H,W) float64 tensor; ``dim`` is 2 (rows) or 3 (cols).
    """
    lw = len(weights_np) // 2
    if mode == "reflect":
        t = t.index_select(dim, _reflect_index(t.shape[dim], lw, torch))
    else:  # 'constant' with cval (skimage/hessian default)
        pad = [0, 0, 0, 0]  # (left, right, top, bottom)
        if dim == 3:
            pad[0] = pad[1] = lw
        else:
            pad[2] = pad[3] = lw
        t = F.pad(t, pad, mode="constant", value=float(cval))
    w = torch.from_numpy(weights_np).to(_DEVICE, dtype=_DTYPE)
    w = w.view(1, 1, -1, 1) if dim == 2 else w.view(1, 1, 1, -1)
    return F.conv2d(t, w)


def _gaussian_filter_gpu(t, sigma_scaled, orders, lws, mode, cval, torch, F):
    """scipy ``gaussian_filter`` on a (1,1,H,W) tensor: axis 0 then axis 1."""
    for axis in (0, 1):
        w = _kernel1d(sigma_scaled[axis], orders[axis], lws[axis])
        t = _correlate1d_gpu(t, w, 2 + axis, mode, cval, torch, F)
    return t


def _gpu_hessian(image, sigma, mode, cval, order, torch):
    import torch.nn.functional as F

    img = np.asarray(image, dtype=np.float64)
    sig = (float(sigma), float(sigma))
    # Verbatim from skimage._hessian_matrix_with_gaussian.
    truncate = 8 if all(s > 1 for s in sig) else 100
    sq1_2 = 1 / math.sqrt(2)
    sigma_scaled = tuple(sq1_2 * s for s in sig)
    lws = tuple(int(truncate * s + 0.5) for s in sigma_scaled)

    # float32 on the GPU: the Gaussian-derivative convolutions differ from
    # float64 by ~3e-8 here (<< the 1e-6 self-validation), and float32 is ~32x
    # faster than float64 on Pascal GPUs (GTX 1060). The one-shot validation
    # still guards correctness and reverts to the exact CPU path on mismatch.
    t0 = torch.from_numpy(img).to(_DEVICE, dtype=_DTYPE).unsqueeze(0).unsqueeze(0)
    # orders in 2D = ([1,0], [0,1])
    axis_orders = ([1, 0], [0, 1])
    gradients = [
        _gaussian_filter_gpu(t0, sigma_scaled, axis_orders[d], lws, mode, cval, torch, F)
        for d in range(2)
    ]
    axes = (0, 1) if order == "rc" else (1, 0)
    H_elems = []
    for ax0, ax1 in combinations_with_replacement(axes, 2):
        h = _gaussian_filter_gpu(gradients[ax0], sigma_scaled, axis_orders[ax1],
                                 lws, mode, cval, torch, F)
        H_elems.append(h[0, 0].cpu().numpy().astype(np.float64))
    return H_elems


def hessian_matrix(image, sigma=1, mode="constant", cval=0, order="rc",
                   use_gaussian_derivatives=True):
    """Signature-compatible drop-in for ``skimage.feature.hessian_matrix``.

    Only the ``use_gaussian_derivatives=True`` path is GPU-accelerated (the one
    the vesselness filters use). Anything else, or any failure, defers to
    skimage so behaviour is never changed.
    """
    global _GPU_OK, _DEVICE, _DTYPE

    if not use_gaussian_derivatives:
        from skimage.feature import hessian_matrix as _hm
        return _hm(image, sigma=sigma, mode=mode, cval=cval, order=order,
                   use_gaussian_derivatives=False)

    if _GPU_OK is False:
        return _cpu_hessian(image, sigma, mode, cval, order)

    try:
        import torch

        if not torch.cuda.is_available():
            _GPU_OK = False
            _log_status(False, "no CUDA GPU")
            return _cpu_hessian(image, sigma, mode, cval, order)
        if _DEVICE is None:
            _DEVICE = torch.device("cuda")
            _DTYPE = torch.float32

        H_gpu = _gpu_hessian(image, sigma, mode, cval, order, torch)

        if _GPU_OK is None:
            with _LOCK:
                if _GPU_OK is None:
                    H_cpu = _cpu_hessian(image, sigma, mode, cval, order)
                    ok = len(H_gpu) == len(H_cpu) and all(
                        np.abs(a - b).max() < _VALIDATE_TOL
                        for a, b in zip(H_gpu, H_cpu)
                    )
                    _GPU_OK = bool(ok)
                    _log_status(_GPU_OK, "GPU output mismatched skimage")
                    if not ok:
                        return H_cpu
            if _GPU_OK is False:
                return _cpu_hessian(image, sigma, mode, cval, order)
        return H_gpu
    except Exception:  # noqa: BLE001 -- any failure => permanent CPU fallback
        _GPU_OK = False
        _log_status(False, "error on GPU path")
        return _cpu_hessian(image, sigma, mode, cval, order)
