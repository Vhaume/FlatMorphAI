# -*- coding: utf-8 -*-
"""Bit-identical memoization of the pure, heavily repeated calls of the
stereotype descriptor stage.

Measured on the user's videos (4 consecutive frames, real per-frame compute):

    imdecode      49 calls/frame,  9 distinct  -> 82 % duplicates
    skeletonize   48 calls/frame, 20 distinct  -> 59 % duplicates
    hessian       28 calls/frame, 20 distinct  -> 29 % duplicates
    farneback      6 calls/frame, 4.5 distinct -> 25 % duplicates

Many duplicates are ACROSS frames (frame t and t+1 both read the T-1/T0/T+1
masks and images), so the cache is shared by all descriptor threads.

Why it is exact: each wrapped function is pure (same input bytes + same
parameters -> same output). The key is a hash of the input bytes, shape, dtype
and every parameter; on a hit a COPY is returned (and a copy is stored), so no
caller can ever see or corrupt another caller's array.

Why it cannot OOM: one LRU per function with a hard byte budget (default
384 MB in total, FLATMORPHAI_DESC_MEMO_MB), least-recently-used entries are
evicted first. The previous OOM came from holding whole clips unbounded.

Opt-in with FLATMORPHAI_DESC_MEMO=1 (validated bit-identical per frame before
being proposed as default). Nothing changes unless install() is called.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict

import numpy as np

STATS: "dict[str, list]" = {}          # name -> [calls, hits]
HESSIAN_GPU_STATE: dict = {}           # batched GPU Hessian: ok / batches / max_err
FARNEBACK_GPU_STATE: dict = {}         # GPU Farneback v2 float32: ok / calls / epe
_installed = False
_lock_install = threading.Lock()
_ZHANG_CORE = [None]     # Zhang core called by the skeleton memo (crop_exact wraps it)


def _nbytes(v) -> int:
    if isinstance(v, np.ndarray):
        return int(v.nbytes)
    if isinstance(v, (list, tuple)):
        return sum(_nbytes(x) for x in v)
    return 64


def _copy(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, list):
        return [_copy(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_copy(x) for x in v)
    return v


class _LRU:
    def __init__(self, name: str, budget: int):
        self.name = name
        self.budget = int(budget)
        self.used = 0
        self.d: "OrderedDict[str, tuple]" = OrderedDict()
        self.lock = threading.Lock()
        STATS[name] = [0, 0]

    def get(self, k):
        with self.lock:
            STATS[self.name][0] += 1
            hit = self.d.get(k)
            if hit is None:
                return None
            self.d.move_to_end(k)
            STATS[self.name][1] += 1
            return hit[0]

    def put(self, k, v):
        n = _nbytes(v)
        if n > self.budget // 2:      # never let one entry flush the cache
            return
        with self.lock:
            if k in self.d:
                return
            self.d[k] = (v, n)
            self.used += n
            while self.used > self.budget and self.d:
                _k, (_v, _n) = self.d.popitem(last=False)
                self.used -= _n


# Content digest of an array's bytes: a Numba 2x4-lane xxh64-style 128-bit
# hash (~8x faster than blake2b on the 2 MB full-resolution masks, which made
# hashing ~5 % of the descriptor time). Falls back to blake2b without numba.
try:
    from numba import njit as _njit
    _U = np.uint64
    _P1, _P2, _P3, _P4, _P5 = (_U(11400714785074694791), _U(14029467366897019727),
                               _U(1609587929392839161), _U(9650029242287828579),
                               _U(2870177450012600261))

    @_njit(nogil=True, cache=False, inline="always")
    def _rotl(x, r):
        return (x << _U(r)) | (x >> _U(64 - r))

    @_njit(nogil=True, cache=False, inline="always")
    def _rnd(acc, w):
        return _rotl(acc + w * _P2, 31) * _P1

    @_njit(nogil=True, cache=False, inline="always")
    def _aval(h):
        h ^= h >> _U(33)
        h *= _P2
        h ^= h >> _U(29)
        h *= _P3
        h ^= h >> _U(32)
        return h

    @_njit(nogil=True, cache=False)
    def _h128(words, tail):  # pragma: no cover - compiled
        v1 = _P1 + _P2; v2 = _P2; v3 = _U(0); v4 = _U(0) - _P1
        u1 = _P4; u2 = _P5; u3 = _P3; u4 = _P2
        n = words.shape[0]
        i = 0
        while i + 4 <= n:
            a = words[i]; b = words[i + 1]; c = words[i + 2]; d = words[i + 3]
            v1 = _rnd(v1, a); v2 = _rnd(v2, b); v3 = _rnd(v3, c); v4 = _rnd(v4, d)
            u1 = _rnd(u1, b ^ _P5); u2 = _rnd(u2, c ^ _P4); u3 = _rnd(u3, d ^ _P3); u4 = _rnd(u4, a ^ _P1)
            i += 4
        while i < n:
            v1 = _rnd(v1, words[i]); u2 = _rnd(u2, words[i] ^ _P5)
            i += 1
        for j in range(tail.shape[0]):
            t = _U(tail[j]) + _U(j + 1) * _P3
            v2 = _rnd(v2, t); u3 = _rnd(u3, t ^ _P4)
        L = _U(n * 8 + tail.shape[0])
        h1 = _rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18) + L
        h2 = _rotl(u1, 1) + _rotl(u2, 7) + _rotl(u3, 12) + _rotl(u4, 18) + (L ^ _P5)
        return (_aval(h1 ^ _aval(v3 * _P4) ^ _aval(v4 * _P5)),
                _aval(h2 ^ _aval(u1 * _P3) ^ _aval(u2 * _P1)))

    def _digest(a) -> bytes:
        b = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        n8 = b.size // 8
        h1, h2 = _h128(b[:n8 * 8].view(np.uint64), b[n8 * 8:])
        return int(h1).to_bytes(8, "little") + int(h2).to_bytes(8, "little")
    _digest(np.zeros(13, np.uint8))            # compile once, at import
except Exception:  # noqa: BLE001 - no numba: cryptographic hash
    def _digest(a) -> bytes:
        return hashlib.blake2b(memoryview(np.ascontiguousarray(a)).cast("B"), digest_size=16).digest()


def _key(arrays, params) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.asarray(a)
        h.update(repr((a.shape, a.dtype.str)).encode())
        h.update(_digest(a) if a.size else b"")
    h.update(repr(params).encode())
    return h.hexdigest()


_FB_PARAMS = (0.5, 3, 15, 3, 5, 1.2, 0)


def _farneback_gpu(orig):
    """GPU twin (gpu_batch_ops.farneback_v2, float32) of cv2 Farneback for the
    one parameter set the descriptors use. Validated on the user's GTX 1060:
    EPE 5.6e-05 px mean, txf 8e-4 / opt 2e-4 / ax 2e-5 relative, 0 class change,
    -7.5 % per frame. Per process: the first 3 calls are computed BOTH ways
    (the cv2 result is returned) and the GPU path is kept only if mean EPE
    <= 1e-3 px and <= 0.5 % of pixels differ by > 0.01 px; any error -> cv2
    for good. One process at a time on the GPU (file lock) + empty_cache.
    FLATMORPHAI_FARNEBACK_GPU=0 disables. float16 was rejected (txf +120 %)."""
    st = FARNEBACK_GPU_STATE
    st.update(on=True, ok=None, calls=0, checks=0, epe=0.0, frac=0.0, error="")
    from . import gpu_server as _GS
    if _GS.active():
        # shared GPU process: this worker never imports torch (~0.8 GB saved)
        st["server"] = True

        def _run(a, b):
            return _GS.call("farneback", a, b, _FB_PARAMS)
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                st.update(ok=False, error="no CUDA")
                return orig
            from . import gpu_batch_ops as _G
        except Exception as exc:  # noqa: BLE001
            st.update(ok=False, error="%s: %s" % (type(exc).__name__, exc))
            return orig

        def _run(a, b):
            with _G.gpu_lock():
                try:
                    return _G.farneback_v2([a], [b], *_FB_PARAMS[:6], store="float32")[0]
                finally:
                    torch.cuda.empty_cache()

    def fb(prev, nxt, flow, *a, **kw):
        if (st["ok"] is False or flow is not None or kw or tuple(a) != _FB_PARAMS
                or np.ndim(prev) != 2 or np.shape(prev) != np.shape(nxt)
                or np.asarray(prev).dtype != np.uint8 or np.asarray(nxt).dtype != np.uint8
                or min(np.shape(prev)) < 32):
            return orig(prev, nxt, flow, *a, **kw)
        try:
            if st["ok"] is None and _GS.active() and _GS.decision("farneback") is not None:
                st["ok"] = _GS.decision("farneback")       # run-wide self-test decision
                st["error"] = "" if st["ok"] else "GPU server self-test"
                st["selftest"] = True
                if st["ok"] is False:
                    return orig(prev, nxt, flow, *a, **kw)
            if st["ok"] is None:
                ref = orig(prev, nxt, flow, *a, **kw)
                g = _run(np.asarray(prev), np.asarray(nxt))
                e = np.sqrt(((g - ref) ** 2).sum(-1))
                st["checks"] += 1
                st["epe"] = max(st["epe"], float(e.mean()))
                st["frac"] = max(st["frac"], float((e > 0.01).mean()))
                if st["epe"] > 1e-3 or st["frac"] > 5e-3:
                    st["ok"] = False
                    st["error"] = "validation EPE %.1e px, %.2f %% > 0.01 px" % (st["epe"], 100 * st["frac"])
                elif st["checks"] >= 3:
                    st["ok"] = True
                # validated -> the GPU result is used from the very first call,
                # so every frame goes through the same path whatever the
                # process split (the reference is only a check)
                return ref if st["ok"] is False else g
            st["calls"] += 1
            return _run(np.asarray(prev), np.asarray(nxt))
        except Exception as exc:  # noqa: BLE001
            st.update(ok=False, error="%s: %s" % (type(exc).__name__, exc))
            return orig(prev, nxt, flow, *a, **kw)
    return fb


def install(log=None) -> bool:
    """Wrap skeletonize (Zhang core), the Gaussian Hessian, Farneback optical
    flow and imdecode with the bounded, copy-on-hit LRU caches."""
    global _installed
    _log = log or (lambda _m: None)
    with _lock_install:
        if _installed:
            return True
        total_mb = float(os.environ.get("FLATMORPHAI_DESC_MEMO_MB", "384") or "384")
        budget = int(total_mb * 1024 * 1024)
        # Shares of the per-process memo. At the nominal 160 MB: 40 MB each for
        # imdecode / hessian / farneback, 16 skeleton, 24 readbytes. When the RAM
        # planner shrinks the memo to fit one more process, the cache with the
        # worst value per MB goes first: Farneback (12-14 % of calls avoided, one
        # full-resolution flow = 15 MB, and an LRU never stores an entry larger
        # than half its budget, so a smaller share is just dead weight); imdecode
        # keeps its 40 MB (80 % avoided at 40 MB, 54 % at 25 MB, measured).
        _fb = 0.25 * total_mb if total_mb >= 160.0 else 0.0
        _dec = max(0.25 * total_mb, min(40.0, 0.45 * total_mb))
        _rest = max(0.0, total_mb - _fb - _dec)
        parts = {"imdecode": _dec / total_mb, "farneback": _fb / total_mb,
                 "hessian": _rest * 0.50 / total_mb, "skeletonize": _rest * 0.20 / total_mb,
                 "readbytes": _rest * 0.30 / total_mb}
        done = []

        try:  # skeletonize: skimage calls the module-global _skeletonize_zhang
            import skimage.morphology._skeletonize as _SK
            _ZHANG_CORE[0] = _SK._skeletonize_zhang
            c_sk = _LRU("skeletonize", budget * parts["skeletonize"])

            def _sk(image):
                k = _key((image,), ())
                v = c_sk.get(k)
                if v is not None:
                    return _copy(v)
                v = _ZHANG_CORE[0](image)
                c_sk.put(k, _copy(v))
                return v
            _SK._skeletonize_zhang = _sk
            done.append("skeletonize")
        except Exception:  # noqa: BLE001
            pass

        try:  # Hessian: hessian_matrix calls the module-global helper
            import skimage.feature.corner as _CO
            _orig_he = _CO._hessian_matrix_with_gaussian
            c_he = _LRU("hessian", budget * parts["hessian"])

            # Optional batched GPU path (FLATMORPHAI_HESSIAN_GPU=1): on a miss,
            # every sigma already seen for this image shape is computed in ONE
            # batched GPU pass and stored, so the following sigmas of the same
            # image are cache hits. Self-validated against skimage on the first
            # calls (max relative error <= 1e-4), permanent fallback otherwise.
            # ON by default since validated in the app (180 frames, identical
            # classes; max relative error 4e-7). FLATMORPHAI_HESSIAN_GPU=0 to disable.
            _hg = {"on": os.environ.get("FLATMORPHAI_HESSIAN_GPU", "1") in ("1", "true", "True"),
                   "ok": None, "checks": 0, "seen": {}, "lock": threading.Lock()}
            _hg_prec = os.environ.get("FLATMORPHAI_HESSIAN_GPU_DTYPE", "float32")
            HESSIAN_GPU_STATE.update(_hg)

            def _he_key(image, s, mode, cval, order, kw):
                if np.isscalar(s):
                    s = float(s)       # sigma=2 and sigma=2.0 must share an entry
                return _key((image,), (s, mode, cval, order, sorted(kw.items())))

            def _he(image, sigma=1, mode="reflect", cval=0, order="rc", **kw):
                k = _he_key(image, sigma, mode, cval, order, kw)
                v = c_he.get(k)
                if v is not None:
                    return _copy(v)
                if (_hg["on"] and _hg["ok"] is not False and mode == "reflect"
                        and order == "rc" and not kw and np.isscalar(sigma)
                        and np.ndim(image) == 2):
                    try:
                        from . import hessian_batch_gpu as _HB
                        shp = (np.shape(image), np.asarray(image).dtype.str)
                        with _hg["lock"]:
                            sig = sorted(set(_hg["seen"].get(shp, ())) | {float(sigma)})
                        from . import gpu_server as _GS
                        if _GS.active():
                            res = _GS.call("hessian", np.asarray(image), sig, _hg_prec)
                            HESSIAN_GPU_STATE["server"] = True
                        else:
                            res = _HB.hessian_batch_torch(image, sig, dtype=_hg_prec)
                        if res is not None and _hg["ok"] is None and _GS.active() \
                                and _GS.decision("hessian") is not None:
                            _hg["ok"] = _GS.decision("hessian")    # run-wide self-test decision
                            _hg["checks"] = 3
                            HESSIAN_GPU_STATE["ok"] = _hg["ok"]
                            HESSIAN_GPU_STATE["selftest"] = True
                            if _hg["ok"] is False:
                                res = None
                        if res is not None:
                            if _hg["checks"] < 3:          # self-validation
                                ref = _orig_he(image, sigma=sigma, mode=mode, cval=cval, order=order)
                                err = _HB.max_rel_err(ref, res[float(sigma)])
                                _hg["checks"] += 1
                                HESSIAN_GPU_STATE["max_err"] = max(HESSIAN_GPU_STATE.get("max_err", 0.0), err)
                                if err > 1e-4:
                                    _hg["ok"] = False
                                    HESSIAN_GPU_STATE["ok"] = False
                                    c_he.put(k, _copy(ref))
                                    return ref
                                _hg["ok"] = True
                                HESSIAN_GPU_STATE["ok"] = True
                            for s, hv in res.items():
                                c_he.put(_he_key(image, s, mode, cval, order, kw), _copy(hv))
                            with _hg["lock"]:
                                _hg["seen"].setdefault(shp, set()).add(float(sigma))
                            HESSIAN_GPU_STATE["batches"] = HESSIAN_GPU_STATE.get("batches", 0) + 1
                            return _copy(res[float(sigma)])
                    except Exception as exc:  # noqa: BLE001 - never fail here
                        _hg["ok"] = False
                        HESSIAN_GPU_STATE["ok"] = False
                        HESSIAN_GPU_STATE["error"] = "%s: %s" % (type(exc).__name__, exc)
                v = _orig_he(image, sigma=sigma, mode=mode, cval=cval, order=order, **kw)
                c_he.put(k, _copy(v))
                if np.isscalar(sigma) and np.ndim(image) == 2:
                    with _hg["lock"]:
                        _hg["seen"].setdefault((np.shape(image), np.asarray(image).dtype.str), set()).add(float(sigma))
                return v
            _CO._hessian_matrix_with_gaussian = _he
            done.append("hessian")
        except Exception:  # noqa: BLE001
            pass

        import cv2
        try:  # Farneback: only cache when no initial flow buffer is passed
            _orig_fb = cv2.calcOpticalFlowFarneback
            if os.environ.get("FLATMORPHAI_FARNEBACK_GPU", "1") in ("1", "true", "True"):
                _orig_fb = _farneback_gpu(_orig_fb)      # memo caches GPU results too
            c_fb = _LRU("farneback", budget * parts["farneback"])

            def _fb(prev, nxt, flow, *a, **kw):
                if flow is not None:
                    return _orig_fb(prev, nxt, flow, *a, **kw)
                k = _key((prev, nxt), (a, sorted(kw.items())))
                v = c_fb.get(k)
                if v is not None:
                    return _copy(v)
                v = _orig_fb(prev, nxt, flow, *a, **kw)
                c_fb.put(k, _copy(v))
                return v
            cv2.calcOpticalFlowFarneback = _fb
            done.append("farneback")
        except Exception:  # noqa: BLE001
            pass

        try:  # imdecode: key = encoded bytes + flag
            _orig_dec = cv2.imdecode
            c_dec = _LRU("imdecode", budget * parts["imdecode"])

            def _dec(buf, flags):
                k = _key((np.asarray(buf),), (int(flags),))
                v = c_dec.get(k)
                if v is not None:
                    return _copy(v)
                v = _orig_dec(buf, flags)
                if v is not None:
                    c_dec.put(k, _copy(v))
                return v
            cv2.imdecode = _dec
            done.append("imdecode")
        except Exception:  # noqa: BLE001
            pass

        try:  # raw file reads (images / masks / formatted crops): /mnt/c under
            # WSL is slow and each frame re-reads its T-1/T0/T+1 files several
            # times. Key = path + mtime + size, so a rewritten file is re-read.
            from . import lightgbm_bundle as _LB
            _orig_rb = _LB._readbytes
            c_rb = _LRU("readbytes", budget * parts["readbytes"])

            _rb_keys = {}

            def _rb(path):
                k = _rb_keys.get(path)
                if k is None:
                    try:
                        st = os.stat(path)
                        k = repr((os.path.abspath(path), st.st_mtime_ns, st.st_size))
                    except OSError:
                        return _orig_rb(path)
                    if len(_rb_keys) > 20000:
                        _rb_keys.clear()
                    _rb_keys[path] = k
                v = c_rb.get(k)
                if v is not None:
                    return v.copy()
                v = _orig_rb(path)
                c_rb.put(k, v.copy())
                return v
            _LB._readbytes = _rb
            done.append("readbytes")
        except Exception:  # noqa: BLE001
            pass

        _installed = True
        _log("  descriptor memo: ON (%s; bounded %.0f MB, copy-on-hit, bit-identical)"
             % (", ".join(done), total_mb))
        return True


_FN_CACHES: "dict[str, _LRU]" = {}
_FN_LOCK = threading.Lock()


def _code_sig(fn) -> str:
    """Identity of a function's implementation (bytecode + constants + names),
    so two copies of the SAME code (e.g. axe_pointes exec'd separately for
    v4c / v4d / v4e) share a cache while a different implementation never does."""
    c = getattr(fn, "__code__", None)
    if c is None:
        return repr(fn)
    h = hashlib.blake2b(digest_size=12)
    h.update(c.co_code)
    h.update(repr(c.co_consts).encode())
    h.update(repr(c.co_names).encode())
    h.update(repr(getattr(fn, "__defaults__", None)).encode())
    return h.hexdigest()


def memoize(fn, name: str, budget_mb: float = 16.0):
    """Bit-identical memo wrapper for a PURE function of numpy arrays (and
    simple scalars): key = input bytes/shape/dtype + scalars + the function's
    code signature; copy-on-hit / copy-on-store; exceptions are never cached
    (re-raised, recomputed next time). One shared LRU per ``name``."""
    if getattr(fn, "_memo_wrapped", False):
        return fn
    sig = _code_sig(fn)
    with _FN_LOCK:
        cache = _FN_CACHES.get(name)
        if cache is None:
            cache = _LRU(name, budget_mb * 1024 * 1024)
            _FN_CACHES[name] = cache

    def wrapper(*args, **kwargs):
        arrs = [a for a in args if isinstance(a, np.ndarray)]
        scal = [("arr" if isinstance(a, np.ndarray) else a) for a in args]
        try:
            k = _key(arrs, (sig, scal, sorted(kwargs.items())))
        except Exception:  # noqa: BLE001 - unhashable input: no caching
            return fn(*args, **kwargs)
        v = cache.get(k)
        if v is not None:
            return _copy(v)
        v = fn(*args, **kwargs)
        cache.put(k, _copy(v))
        return v

    wrapper._memo_wrapped = True
    wrapper.__wrapped__ = fn
    return wrapper


def summary() -> str:
    parts = []
    for k, (calls, hits) in STATS.items():
        if calls:
            parts.append("%s %d/%d avoided (%.0f%%)" % (k, hits, calls, 100.0 * hits / calls))
    g = HESSIAN_GPU_STATE
    if g.get("on"):
        parts.append("hessian GPU%s %s (%d batches, max error %.1e%s)" % (
            " (server)" if g.get("server") else "",
            "active" if g.get("ok") else ("CPU FALLBACK" if g.get("ok") is False else "not requested"),
            g.get("batches", 0), g.get("max_err", 0.0),
            (", " + g["error"]) if g.get("error") else ""))
    f = FARNEBACK_GPU_STATE
    if f.get("on"):
        parts.append("farneback GPU%s %s (%d calls, validation EPE %.1e px%s)" % (
            " (server)" if f.get("server") else "",
            "active" if f.get("ok") else ("CPU FALLBACK" if f.get("ok") is False else "not requested"),
            f.get("calls", 0), f.get("epe", 0.0),
            (", " + f["error"]) if f.get("error") else ""))
    return ", ".join(parts) if parts else "no call"


# ---------------------------------------------------------------------------
# Per-image memo of the shared single-mask helpers (skeleton, medial axis,
# longest path, largest component, shape / curvature / circle descriptors).
#
# The temporal descriptors of second t read the masks of t-1, t, t+1 (and
# t-5..t+5 for the wide window): every mask goes through the SAME helper
# 3 to 11 times (as T-1, T0, T+1 of three neighbouring seconds), and inside
# one second several families recompute the same skeleton / longest path of
# the same mask (curv_dynamic even calls _curv_profile twice per mask).
# Measured on 8 consecutive frames: 46 of 56 _curv_profile calls, 22 of 32
# shape_desc, 15 of 48 axe_chemin, 69 of 85 _length_tips were repeats.
#
# Exactness: the key is the exact BYTES (+ shape + dtype) of every array
# argument and the exact repr of every scalar, plus the function's code
# signature. A hit therefore means "this very image again", never "a
# similar image": nothing is ever borrowed from another frame. Results are
# copied on store and on hit; a function that mutates its input is detected
# on its first calls and left unwrapped; exceptions are never cached.
# FLATMORPHAI_FRAME_MEMO=0 disables.
# ---------------------------------------------------------------------------
_SCALARS = (int, float, bool, str, bytes, type(None), np.generic)


class _NoKey(Exception):
    pass


def _hash_into(h, x):
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:
            raise _NoKey()
        h.update(b"A" + repr((x.shape, x.dtype.str)).encode())
        h.update(_digest(x) if x.size else b"")
    elif isinstance(x, _SCALARS):
        h.update(b"S" + repr((type(x).__name__, x)).encode())
    elif isinstance(x, (list, tuple)):
        h.update(b"L" if isinstance(x, list) else b"T")
        h.update(str(len(x)).encode())
        for y in x:
            _hash_into(h, y)
    elif isinstance(x, dict):
        h.update(b"D" + str(len(x)).encode())
        for k in sorted(x, key=repr):
            _hash_into(h, k)
            _hash_into(h, x[k])
    else:
        raise _NoKey()


def _deep_copy(v):
    """Independent copy of a result made of arrays / containers / scalars;
    raises _NoKey for anything else (such a result is simply not cached)."""
    if isinstance(v, np.ndarray):
        if v.dtype.hasobject:
            raise _NoKey()
        return v.copy()
    if isinstance(v, _SCALARS):
        return v
    if isinstance(v, list):
        return [_deep_copy(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_deep_copy(x) for x in v)
    if isinstance(v, dict):
        return {k: _deep_copy(x) for k, x in v.items()}
    raise _NoKey()


def _deep_nbytes(v) -> int:
    if isinstance(v, np.ndarray):
        return int(v.nbytes) + 96
    if isinstance(v, (list, tuple)):
        return 56 + sum(_deep_nbytes(x) for x in v)
    if isinstance(v, dict):
        return 232 + sum(_deep_nbytes(x) + 50 for x in v.values())
    return 32


class _DeepLRU(_LRU):
    def put(self, k, v):
        n = _deep_nbytes(v)
        if n > self.budget // 2:
            return
        with self.lock:
            if k in self.d:
                return
            self.d[k] = (v, n)
            self.used += n
            while self.used > self.budget and self.d:
                _k, (_v, _n) = self.d.popitem(last=False)
                self.used -= _n


FRAME_MEMO_STATE: dict = {"wrapped": [], "disabled": []}
_MUT_CHECKS = 3          # first misses: verify the inputs were not modified


def memoize_pure(fn, name: str, budget_mb: float = 8.0):
    """Exact memo of a pure function of arrays / containers / scalars."""
    if getattr(fn, "_memo_wrapped", False):
        return fn
    sig = _code_sig(fn)
    with _FN_LOCK:
        cache = _FN_CACHES.get("pure:" + name)
        if cache is None:
            cache = _DeepLRU(name, budget_mb * 1024 * 1024)
            _FN_CACHES["pure:" + name] = cache
    st = {"checks": 0, "off": False}

    def _k(args, kwargs):
        h = hashlib.blake2b(digest_size=20)
        h.update(sig.encode())
        _hash_into(h, args)
        _hash_into(h, kwargs)
        return h.digest()

    def wrapper(*args, **kwargs):
        if st["off"]:
            return fn(*args, **kwargs)
        try:
            k = _k(args, kwargs)
        except _NoKey:
            return fn(*args, **kwargs)
        v = cache.get(k)
        if v is not None:
            return _deep_copy(v[0])
        r = fn(*args, **kwargs)
        if st["checks"] < _MUT_CHECKS:
            st["checks"] += 1
            try:
                same = _k(args, kwargs) == k
            except _NoKey:
                same = False
            if not same:          # the function modifies its input: never cache it
                st["off"] = True
                FRAME_MEMO_STATE["disabled"].append(name)
                return r
        try:
            cache.put(k, (_deep_copy(r),))    # 1-tuple: a cached None is a hit too
        except _NoKey:
            pass
        return r

    wrapper._memo_wrapped = True
    wrapper.__wrapped__ = fn
    for a in ("__name__", "__qualname__", "__doc__", "__module__"):
        try:
            setattr(wrapper, a, getattr(fn, a))
        except Exception:  # noqa: BLE001
            pass
    return wrapper


# (module file basename, function name, shared cache name, budget MB).
# The "largest component" helpers are NOT memoised: measured 2-17 % hits in
# the app for ~32 MB of full-resolution masks per process (RAM > gain).
# Only pure single-image helpers: their output is a function of the given
# mask / image bytes alone.
_FRAME_TARGETS = [
    ("core_shape_metrics.py", "shape_desc", "shape_desc", 4),
    ("core_shape_metrics.py", "_curv_profile", "_curv_profile", 2),
    ("core_shape_metrics.py", "_circle_fit", "_circle_fit", 2),
    ("core_shape_metrics.py", "curv_metrics_T0", "curv_metrics_T0", 2),
    ("core_shape_metrics.py", "_spath", "_spath", 6),
    ("core_shape_metrics.py", "skeletonize", "skeletonize_full", 12),
    ("metrics_3mask.py", "shape_desc", "shape_desc", 4),
    ("metrics_3mask.py", "_curv_profile", "_curv_profile", 2),
    ("precompute_mask3_metrics.py", "shape_desc", "shape_desc", 4),
    ("precompute_mask3_metrics.py", "_curv_profile", "_curv_profile", 2),
    ("head_extension_wide.py", "_length_tips", "_length_tips", 2),
    ("topology_motion.py", "_skelloops", "_skelloops", 1),
    ("nonrigid_motion.py", "_state", "nrm_state", 12),
    ("peristalsis.py", "_axis", "per_axis", 1),
    ("peristalsis.py", "_profile", "per_profile", 2),
    ("shape_persistence.py", "_curl", "_curl", 1),
    ("twist_border_evolution.py", "_lr_radii", "_lr_radii", 4),
    ("v3c_descriptor_patches.py", "axe_chemin", "axe_chemin", 8),
]


def frame_memo_enabled() -> bool:
    return os.environ.get("FLATMORPHAI_FRAME_MEMO", "1") in ("1", "true", "True")


def patch_frame_memo(log=None, namespaces=()):
    """Wrap the _FRAME_TARGETS helpers in every loaded module (and the given
    exec namespaces) with memoize_pure. Idempotent; returns what was wrapped."""
    if not frame_memo_enabled():
        return []
    import sys
    done = []
    # per-image caches follow the per-process memo chosen by the RAM planner
    # (160 MB nominal): fewer MB when it had to shrink the memo to fit processes
    # NOT scaled with the planner's memo: these caches are small (~45 MB in
    # all) and an LRU never stores an entry larger than half its budget, so a
    # scaled-down shape_desc cache (2.5 MB < 2 x one full-resolution mask) got
    # 0 % hits in the app.
    _scale = 1.0
    targets = [(getattr(m, "__file__", None) or "", getattr(m, "__dict__", None))
               for m in list(sys.modules.values()) if m is not None]
    targets += [(str(ns.get("__file__", "")), ns) for ns in namespaces]
    for fpath, d in targets:
        if not isinstance(d, dict) or not fpath:
            continue
        base = os.path.basename(fpath)
        for tfile, attr, cname, mb in _FRAME_TARGETS:
            if base != tfile:
                continue
            fn = d.get(attr)
            if fn is None or not callable(fn) or getattr(fn, "_memo_wrapped", False):
                continue
            try:
                d[attr] = memoize_pure(fn, cname, mb * _scale)
                done.append("%s.%s" % (base[:-3], attr))
            except Exception:  # noqa: BLE001
                pass
    FRAME_MEMO_STATE["wrapped"].extend(done)
    if done and log is not None:
        log("  per-image memo: %d helpers (skeleton / axis / shape, exact, same image only)" % len(done))
    return done
