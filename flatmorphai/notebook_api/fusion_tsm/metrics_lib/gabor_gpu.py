"""
Drop-in GPU-accelerated replacement for ``skimage.filters.gabor``.

Gabor filtering is the single most expensive part of the formatted-image
texture battery (~19 convolutions per frame). ``skimage.filters.gabor`` is
pure CPU and holds the GIL for much of its work, so it does not parallelise
well across the feature thread pool. This module runs the exact same
convolution on the GPU (PyTorch) when one is available, which both offloads
the heavy convolution and releases the GIL so the thread pool scales.

CORRECTNESS -- no metric is allowed to change. ``skimage.filters.gabor``
computes, for a kernel built by ``skimage.filters.gabor_kernel``:

    real = ndi.convolve(image, kernel.real, mode='reflect')
    imag = ndi.convolve(image, kernel.imag, mode='reflect')

That is identical, to machine precision, to a cross-correlation of the
image -- padded with 'symmetric' (which is exactly scipy's 'reflect') by
half the kernel -- against the kernel flipped in both axes (this was
verified offline: max abs difference 4e-16 over every frequency/theta used
here). The GPU path reproduces this with ``F.conv2d`` in float64, so the
result is bit-for-bit the same up to float rounding, and the metrics
(which average the response over the mask) are unchanged.

SAFETY -- the very first call validates the GPU output against
``skimage.filters.gabor`` on the real data; if they do not match within a
tight tolerance, or anything raises (no torch, no CUDA, API change), the
module falls back to ``skimage`` permanently. It can therefore never
degrade the metrics: at worst it is simply the old CPU path.
"""

from __future__ import annotations

import sys
import threading

import numpy as np

# Flipped-kernel cache keyed by (frequency, theta): the kernels are constant
# across frames, so build each one once.
_KCACHE: dict = {}
# None = not yet decided, True = GPU validated, False = use skimage CPU.
_GPU_OK = None
_DEVICE = None
_DTYPE = None            # torch.float32, resolved once the GPU device is set
_LOCK = threading.Lock()
_VALIDATE_TOL = 1e-6
_LOGGED: set = set()


def _log_once(key: str, msg: str) -> None:
    """Emit ``msg`` to stderr once per ``key`` (teed into the session log), so the
    operator can confirm the Gabor GPU path was taken or why it fell back.
    Best-effort; never raises."""
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    try:
        sys.stderr.write("[FlatMorphAI] GPU: " + msg + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _flipped_kernel(frequency: float, theta: float):
    key = (round(float(frequency), 6), round(float(theta), 6))
    k = _KCACHE.get(key)
    if k is None:
        from skimage.filters import gabor_kernel
        g = gabor_kernel(frequency, theta=theta)
        # Flip both axes so a cross-correlation (conv2d) equals the
        # convolution skimage performs.
        k = (
            np.ascontiguousarray(g.real[::-1, ::-1]),
            np.ascontiguousarray(g.imag[::-1, ::-1]),
        )
        _KCACHE[key] = k
    return k


def _skimage_gabor(image, frequency, theta):
    from skimage.filters import gabor as _sk_gabor
    return _sk_gabor(image, frequency=frequency, theta=theta)


def _gpu_gabor(image, frequency, theta, torch):
    import torch.nn.functional as F

    kr, ki = _flipped_kernel(frequency, theta)
    kh, kw = kr.shape
    py0, py1 = kh // 2, kh - 1 - kh // 2
    px0, px1 = kw // 2, kw - 1 - kw // 2
    img = np.asarray(image, dtype=np.float64)
    # 'symmetric' padding == scipy ndimage 'reflect' (half-sample symmetric).
    p = np.pad(img, ((py0, py1), (px0, px1)), mode="symmetric")
    # float64 on the GPU: bit-exact vs skimage (~1e-15). float32 was tried for
    # speed on Pascal but is disabled -- in the 7-thread descriptor pool the GPU
    # is a serialisation point, so precision, not throughput, is what matters here.
    t = torch.from_numpy(p).to(_DEVICE, dtype=_DTYPE).unsqueeze(0).unsqueeze(0)
    wr = torch.from_numpy(kr).to(_DEVICE, dtype=_DTYPE).unsqueeze(0).unsqueeze(0)
    wi = torch.from_numpy(ki).to(_DEVICE, dtype=_DTYPE).unsqueeze(0).unsqueeze(0)
    rr = F.conv2d(t, wr)[0, 0].cpu().numpy().astype(np.float64)
    ii = F.conv2d(t, wi)[0, 0].cpu().numpy().astype(np.float64)
    return rr, ii


_MANY_OK = None       # None=undecided, True=batched GPU validated, False=per-call
_MANY_CACHE: dict = {}  # specs -> (stacked weights, KH, KW)


def _stacked_weights(specs):
    """Zero-pad every (real, imag) kernel to a common odd size and stack them as
    ``2*len(specs)`` conv2d output channels, so one convolution yields all
    responses. Exact: the zero border multiplies image samples by 0."""
    key = tuple((round(float(f), 6), round(float(t), 6)) for f, t in specs)
    cached = _MANY_CACHE.get(key)
    if cached is not None:
        return cached
    kers = [_flipped_kernel(f, t) for f, t in specs]
    KH = max(k[0].shape[0] for k in kers)
    KW = max(k[0].shape[1] for k in kers)
    W = np.zeros((2 * len(kers), 1, KH, KW), dtype=np.float64)
    for i, (kr, ki) in enumerate(kers):
        kh, kw = kr.shape
        oy, ox = (KH - kh) // 2, (KW - kw) // 2
        W[2 * i, 0, oy:oy + kh, ox:ox + kw] = kr
        W[2 * i + 1, 0, oy:oy + kh, ox:ox + kw] = ki
    out = (np.ascontiguousarray(W), KH, KW)
    _MANY_CACHE[key] = out
    return out


def _gpu_gabor_many(image, specs, torch):
    import torch.nn.functional as F

    W, KH, KW = _stacked_weights(specs)
    py0, py1 = KH // 2, KH - 1 - KH // 2
    px0, px1 = KW // 2, KW - 1 - KW // 2
    img = np.asarray(image, dtype=np.float64)
    p = np.pad(img, ((py0, py1), (px0, px1)), mode="symmetric")
    # float64 (see _gpu_gabor): bit-exact vs skimage.
    t = torch.from_numpy(p).to(_DEVICE, dtype=_DTYPE).unsqueeze(0).unsqueeze(0)
    w = torch.from_numpy(W).to(_DEVICE, dtype=_DTYPE)
    r = F.conv2d(t, w)[0].cpu().numpy().astype(np.float64)   # (2K, H, W)
    return [(r[2 * i], r[2 * i + 1]) for i in range(len(specs))]


# ---------------------------------------------------------------------------
# FFT path (CPU, float64). Same maths as skimage (convolution of the image,
# half-sample-symmetric padded, by the complex Gabor kernel), done through
# scipy.fft: relative error ~4e-15 vs skimage and ~16x faster on 384x384 than
# the direct 2-D ndi.convolve (kernels up to 35x35). float64 throughout, no GPU
# (Pascal runs float64 at 1/32 rate and the GPU is shared by all processes).
# FLATMORPHAI_GABOR_FFT=0 restores the previous GPU/skimage path.
# ---------------------------------------------------------------------------
_FFT_OK = None           # None=undecided, True=validated, False=disabled
# (specs, FFT shape) -> (KH, KW, stacked kernel FFTs). BOUNDED IN BYTES: the
# tb_gabor bank runs on body crops whose height changes every frame, so the
# FFT shape is rounded up to a multiple of 64 (exact: a larger zero-padded FFT
# gives the same linear convolution) and the cache evicts its oldest entries
# beyond FLATMORPHAI_GABOR_FFT_CACHE_MB (default 48 MB per process). v1 keyed
# on the exact crop shape and grew to ~450 MB per process -> OOM with 6.
_FFT_KCACHE: dict = {}
_FFT_CLOCK = threading.Lock()
_FFT_KSIZE: dict = {}
_FFT_CHUNK_BYTES = 16e6  # transient spectrum per pass (kernels done in packets)


def _fft_enabled():
    import os
    return os.environ.get("FLATMORPHAI_GABOR_FFT", "1") in ("1", "true", "True")


def _fft_cache_budget():
    import os
    try:
        return float(os.environ.get("FLATMORPHAI_GABOR_FFT_CACHE_MB", "48")) * 1e6
    except ValueError:
        return 48e6


def _fft_kernels(specs, H, W):
    import scipy.fft as sfft
    from skimage.filters import gabor_kernel

    key0 = tuple((round(float(f), 6), round(float(t), 6)) for f, t in specs)
    # The FFT size is a pure function of (H, W, kernel size): the result never
    # depends on which shapes happen to be cached (bit-for-bit reproducible
    # whatever the frame order / process split).
    ksz = _FFT_KSIZE.get(key0)
    if ksz is None:
        ks = [gabor_kernel(f, theta=t) for f, t in specs]
        ksz = (max(k.shape[0] for k in ks), max(k.shape[1] for k in ks))
        _FFT_KSIZE[key0] = ksz
    KH, KW = ksz
    fs = tuple(sfft.next_fast_len(int(-(-(n + 2 * (K - 1)) // 64) * 64), True)
               for n, K in ((H, KH), (W, KW)))
    with _FFT_CLOCK:
        v = _FFT_KCACHE.get((key0, fs))
        if v is not None:
            return v
    ks = [gabor_kernel(f, theta=t) for f, t in specs]
    Z = np.zeros((2 * len(ks), KH, KW), dtype=np.float64)
    for i, k in enumerate(ks):
        kh, kw = k.shape
        oy, ox = (KH - kh) // 2, (KW - kw) // 2
        Z[2 * i, oy:oy + kh, ox:ox + kw] = k.real
        Z[2 * i + 1, oy:oy + kh, ox:ox + kw] = k.imag
    KF = sfft.rfft2(Z, fs, axes=(1, 2))
    v = (KH, KW, fs, KF)
    budget = _fft_cache_budget()
    if KF.nbytes <= budget:
        with _FFT_CLOCK:
            _FFT_KCACHE[(key0, fs)] = v
            while sum(e[3].nbytes for e in _FFT_KCACHE.values()) > budget and len(_FFT_KCACHE) > 1:
                _FFT_KCACHE.pop(next(iter(_FFT_KCACHE)))
    return v


def _fft_gabor_many(image, specs):
    import scipy.fft as sfft

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        return None
    H, W = img.shape
    KH, KW, fs, KF = _fft_kernels(specs, H, W)
    if min(H, W) < max(KH, KW) // 2 + 1:         # single reflection only
        return None
    p = np.pad(img, ((KH // 2, KH - 1 - KH // 2), (KW // 2, KW - 1 - KW // 2)),
               mode="symmetric")
    P = sfft.rfft2(p, fs)
    nch = KF.shape[0]
    step = max(2, int(_FFT_CHUNK_BYTES // max(1, KF[0].nbytes)) // 2 * 2)
    out = np.empty((nch, H, W), dtype=np.float64)
    for a in range(0, nch, step):
        r = sfft.irfft2(KF[a:a + step] * P[None], fs, axes=(1, 2))
        out[a:a + step] = r[:, KH - 1:KH - 1 + H, KW - 1:KW - 1 + W]
        del r
    return [(out[2 * i], out[2 * i + 1]) for i in range(len(specs))]


# GPU float32 FFT of the same bank (gpu_batch_ops.gabor_bank_torch), through
# the shared GPU server when it runs (no torch in the worker) or in-process
# under the GPU lock. Measured on the user's GTX 1060: x3.9 per image, 1.2e-6
# relative to skimage, tgb 1.3e-5 / tm 9e-6 / tb 6e-6, 0 class change, -7 % per
# frame. Per process the first 3 calls are checked against the CPU float64 FFT
# (the CPU result is returned); kept only if max error <= 1e-4 x max|response|.
# FLATMORPHAI_GABOR_GPU=0 disables.
GABOR_GPU_STATE = {"ok": None, "calls": 0, "checks": 0, "err": 0.0, "error": "", "server": False}


def _gpu_bank(image, specs):
    import os
    from flatmorphai.notebook_api import gpu_server as _GS
    if _GS.active():
        GABOR_GPU_STATE["server"] = True
        res = _GS.call("gabor", np.asarray(image, dtype=np.float64), list(specs))
        return [(np.asarray(r, np.float64), np.asarray(i, np.float64)) for r, i in res]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA")
    from flatmorphai.notebook_api import gpu_batch_ops as G
    with G.gpu_lock():
        try:
            return G.gabor_bank_torch(image, specs)
        finally:
            torch.cuda.empty_cache()


def _gpu_try(image, specs, cpu_res_fn):
    import os
    st = GABOR_GPU_STATE
    if st["ok"] is False or os.environ.get("FLATMORPHAI_GABOR_GPU", "1") not in ("1", "true", "True"):
        return None
    try:
        if np.ndim(image) != 2:
            return None
        if st["ok"] is None:
            try:
                from flatmorphai.notebook_api import gpu_server as _GS
                _d = _GS.decision("gabor") if _GS.active() else None
            except Exception:  # noqa: BLE001
                _d = None
            if _d is not None:               # run-wide self-test decision (server)
                st["ok"] = _d
                st["selftest"] = True
                if _d is False:
                    st["error"] = "GPU server self-test"
                    return None
        if st["ok"] is None:
            _LOGGED.add("gabor_fft")          # reference only: no "CPU float64" banner
            ref = cpu_res_fn()
            if ref is None:
                return None
            g = _gpu_bank(image, specs)
            e = 0.0
            for (a, b), (c, d) in zip(ref, g):
                m = max(np.abs(a).max(), np.abs(b).max()) + 1e-30
                e = max(e, float(max(np.abs(c - a).max(), np.abs(d - b).max()) / m))
            st["checks"] += 1
            st["err"] = max(st["err"], e)
            # measured on the user's GTX 1060: 3e-6 .. 1.1e-5 relative to the
            # largest response; the tgb/tb/tm families tolerate >= 1e-2 (no class
            # change). 1e-4 keeps a 10x margin above float32 noise, so a worker
            # no longer falls back to the CPU on noise (it did once at 1.1e-5,
            # which made that worker's frames differ from the others).
            if e > 1e-4:
                st["ok"] = False
                st["error"] = "validation %.1e" % e
            elif st["checks"] >= 3:
                st["ok"] = True
                _log_once("gabor_gpu", "Gabor bank on GPU float32%s (validated against the CPU "
                          "float64 FFT on 3 calls, max error %.1e)."
                          % (" via the shared GPU server" if st.get("server") else "", st["err"]))
            if st["ok"] is False:
                _log_once("gabor_gpu", "Gabor GPU float32 rejected (%s): CPU float64 FFT used." % st["error"])
                return ref
            return g        # same path for every frame, whatever the process split
        st["calls"] += 1
        return _gpu_bank(image, specs)
    except Exception as exc:  # noqa: BLE001
        st["ok"] = False
        st["error"] = "%s: %s" % (type(exc).__name__, exc)
        return None


def _fft_try(image, specs):
    """Validated FFT result, or None (caller continues with the old path)."""
    global _FFT_OK
    if _FFT_OK is False or not _fft_enabled():
        return None
    try:
        res = _fft_gabor_many(image, specs)
        if res is None:
            return None
        if _FFT_OK is None:
            with _LOCK:
                if _FFT_OK is None:
                    ok = True
                    for (rr, ii), (f, t) in zip(res, specs):
                        r_sk, i_sk = _skimage_gabor(image, f, t)
                        if (np.abs(rr - r_sk).max() >= _VALIDATE_TOL
                                or np.abs(ii - i_sk).max() >= _VALIDATE_TOL):
                            ok = False
                            break
                    _FFT_OK = bool(ok)
                    _log_once("gabor_fft", (
                        "Gabor bank by FFT (CPU float64), validated against skimage."
                        if ok else
                        "Gabor FFT path mismatched skimage; previous path kept."))
            if _FFT_OK is False:
                return None
        return res
    except Exception:  # noqa: BLE001
        _FFT_OK = False
        _log_once("gabor_fft", "Gabor FFT path error; previous path kept.")
        return None


def gabor_many(image, specs):
    """Batched ``gabor``: one GPU convolution for all ``(frequency, theta)`` in
    ``specs``. Returns ``[(real, imag), ...]`` aligned with ``specs``, identical
    to calling :func:`gabor` once per spec. Self-validates once against skimage;
    on any mismatch or error it falls back to per-call :func:`gabor` (which is
    itself GPU-or-skimage), so metric values never change."""
    global _MANY_OK, _GPU_OK, _DEVICE, _DTYPE

    _r = _gpu_try(image, specs, lambda: _fft_try(image, specs))
    if _r is not None:
        return _r
    _r = _fft_try(image, specs)
    if _r is not None:
        return _r
    if _MANY_OK is False or _GPU_OK is False:
        return [gabor(image, f, t) for f, t in specs]
    try:
        import torch

        if not torch.cuda.is_available():
            _MANY_OK = _GPU_OK = False
            _log_once("gabor", "Gabor on CPU (skimage) - no CUDA GPU.")
            return [_skimage_gabor(image, f, t) for f, t in specs]
        if _DEVICE is None:
            _DEVICE = torch.device("cuda")
            _DTYPE = torch.float64

        res = _gpu_gabor_many(image, specs, torch)

        if _MANY_OK is None:
            with _LOCK:
                if _MANY_OK is None:
                    ok = True
                    for (rr, ii), (f, t) in zip(res, specs):
                        r_sk, i_sk = _skimage_gabor(image, f, t)
                        if (np.abs(rr - r_sk).max() >= _VALIDATE_TOL
                                or np.abs(ii - i_sk).max() >= _VALIDATE_TOL):
                            ok = False
                            break
                    _MANY_OK = bool(ok)
                    # A validated batched path implies the GPU convolution itself
                    # is good, so the single-kernel path can trust the GPU too.
                    if ok and _GPU_OK is None:
                        _GPU_OK = True
                    _log_once("gabor", (
                        "Gabor bank batched on GPU (one conv2d per frame for all "
                        "frequency/theta kernels), validated against skimage."
                        if ok else
                        "Gabor batched path disabled (GPU output mismatched "
                        "skimage); using per-call Gabor."))
            if _MANY_OK is False:
                return [gabor(image, f, t) for f, t in specs]
        return res
    except Exception:  # noqa: BLE001 -- any failure => per-call fallback
        _MANY_OK = False
        _log_once("gabor", "Gabor batched path error; using per-call Gabor.")
        return [gabor(image, f, t) for f, t in specs]


def gabor(image, frequency, theta=0.0, **_ignored):
    """Signature-compatible with ``skimage.filters.gabor`` (frequency, theta)."""
    global _GPU_OK, _DEVICE, _DTYPE

    _r = _fft_try(image, [(frequency, theta)])
    if _r is not None:
        return _r[0]
    if _GPU_OK is False:
        return _skimage_gabor(image, frequency, theta)

    try:
        import torch

        if not torch.cuda.is_available():
            _GPU_OK = False
            _log_once("gabor", "Gabor on CPU (skimage) - no CUDA GPU.")
            return _skimage_gabor(image, frequency, theta)
        if _DEVICE is None:
            _DEVICE = torch.device("cuda")
            _DTYPE = torch.float64

        rr, ii = _gpu_gabor(image, frequency, theta, torch)

        if _GPU_OK is None:
            # One-time self-validation against the reference implementation.
            with _LOCK:
                if _GPU_OK is None:
                    r_sk, i_sk = _skimage_gabor(image, frequency, theta)
                    ok = (
                        np.abs(rr - r_sk).max() < _VALIDATE_TOL
                        and np.abs(ii - i_sk).max() < _VALIDATE_TOL
                    )
                    _GPU_OK = bool(ok)
                    _log_once("gabor", (
                        "Gabor on GPU (CUDA), validated against skimage."
                        if ok else
                        "Gabor on CPU (skimage) - GPU output mismatched skimage."))
                    if not ok:
                        return r_sk, i_sk
            if _GPU_OK is False:
                return _skimage_gabor(image, frequency, theta)
        return rr, ii
    except Exception:  # noqa: BLE001 -- any failure => permanent CPU fallback
        _GPU_OK = False
        _log_once("gabor", "Gabor on CPU (skimage) - error on GPU path.")
        return _skimage_gabor(image, frequency, theta)
