# -*- coding: utf-8 -*-
"""Exact numba twin of the "ray marching" inner loops of three descriptor
families (pure-Python loops calling round() ~490 000 times per frame):

  * twist_motion_masks._frame_state_impl   (tm_*  : left/right border radii)
  * twist_border_evolution._lr_radii        (be_*  : left/right border radii)
  * compute_shape_lum_metrics.bilateral     (bilat_*: march(s) closure)

Each loop walks r = 1..59 along a normal, from (y0, x0) by r*(sy, sx), and
keeps the last r whose rounded pixel is inside the mask. Only that inner walk
moves to numba; everything else of the three functions is the ORIGINAL source
(re-compiled from inspect.getsource with the walk replaced by one call), so a
mismatch in the pattern simply leaves the function untouched.

Exactness: the Python expression ``y0 + sy * r`` is evaluated by NumPy in
float32 or float64 depending on the operand dtypes AND on the NumPy version
(NEP 50 in NumPy >= 2 keeps np.float32 * python-int in float32; NumPy 1.x
promotes to float64). The wrapper evaluates the very same expression once on
the real scalars (r = 1) to read the precision NumPy uses for the product and
for the sum, and the kernel does the same arithmetic in that precision, then
rounds half-to-even like round()/np.round. The first walks of each process are
also checked against the Python loop (self-validation); any mismatch or
non-finite input disables the kernel for good (Python loop used).

FLATMORPHAI_RAY_NUMBA=0 disables it.
"""
from __future__ import annotations

import os
import textwrap
import threading

import numpy as np

STATS = {"numba": 0, "python": 0, "checked": 0, "patched": []}
_STATE = {"ok": None, "error": "", "kernel": None}
_LOCK = threading.Lock()
_CHECKS = 64


def enabled():
    return os.environ.get("FLATMORPHAI_RAY_NUMBA", "1") in ("1", "true", "True")


def _build():
    if _STATE["kernel"] is not None:
        return _STATE["kernel"]
    from numba import njit

    @njit(nogil=True, cache=False)
    def walk(worm, y0, x0, sy, sx, rmax, p32, s32):
        H, W = worm.shape
        d = 0
        for r in range(1, rmax):
            if p32:
                py = np.float32(sy) * np.float32(r)
                px = np.float32(sx) * np.float32(r)
            else:
                py = sy * np.float64(r)
                px = sx * np.float64(r)
            if s32:
                fy = np.float32(np.float32(y0) + np.float32(py))
                fx = np.float32(np.float32(x0) + np.float32(px))
                if not (np.isfinite(fy) and np.isfinite(fx)):
                    return -1
                yy = int(np.rint(fy))
                xx = int(np.rint(fx))
            else:
                fy = y0 + np.float64(py)
                fx = x0 + np.float64(px)
                if not (np.isfinite(fy) and np.isfinite(fx)):
                    return -1
                yy = int(np.rint(fy))
                xx = int(np.rint(fx))
            if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0:
                break
            d = r
        return d

    _STATE["kernel"] = walk
    return walk


def _py_walk(worm, y0, x0, sy, sx, rmax, r_kind):
    """The original Python walk (reference)."""
    H, W = worm.shape
    d = 0
    for r in range(1, rmax):
        rr = np.int64(r) if r_kind == "np" else r
        yy = int(round(y0 + sy * rr))
        xx = int(round(x0 + sx * rr))
        if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0:
            break
        d = r
    return d


def _is32(v):
    return getattr(v, "dtype", None) == np.float32


def march(worm, y0, x0, sy, sx, rmax=60, r_kind="py"):
    """Last r in 1..rmax-1 whose round(y0 + sy*r), round(x0 + sx*r) pixel is
    inside ``worm`` (0 if the first step is already outside). ``r_kind`` is
    "np" when the original loop iterates over np.arange (np.int64 r)."""
    if _STATE["ok"] is False or not enabled():
        STATS["python"] += 1
        return _py_walk(worm, y0, x0, sy, sx, rmax, r_kind)
    try:
        one = np.int64(1) if r_kind == "np" else 1
        py_, px_ = sy * one, sx * one                 # NumPy's own precision
        p32 = _is32(py_) and _is32(px_)
        s32 = _is32(y0 + py_) and _is32(x0 + px_)
        if (_is32(py_) != _is32(px_)) or (_is32(y0 + py_) != _is32(x0 + px_)):
            STATS["python"] += 1
            return _py_walk(worm, y0, x0, sy, sx, rmax, r_kind)
        k = _build()
        d = k(worm, float(y0), float(x0), float(sy), float(sx), int(rmax), p32, s32)
        if d < 0:
            STATS["python"] += 1
            return _py_walk(worm, y0, x0, sy, sx, rmax, r_kind)
        if STATS["checked"] < _CHECKS:
            ref = _py_walk(worm, y0, x0, sy, sx, rmax, r_kind)
            with _LOCK:
                STATS["checked"] += 1
                if ref != d:
                    _STATE["ok"] = False
                    _STATE["error"] = "mismatch %r vs %r" % (d, ref)
                    return ref
                if STATS["checked"] >= _CHECKS:
                    _STATE["ok"] = True
        STATS["numba"] += 1
        return d
    except Exception as exc:  # noqa: BLE001
        _STATE["ok"] = False
        _STATE["error"] = "%s: %s" % (type(exc).__name__, exc)
        STATS["python"] += 1
        return _py_walk(worm, y0, x0, sy, sx, rmax, r_kind)


# ---------------------------------------------------------------------------
# source patches: (old text, new text) applied to the dedented source
# ---------------------------------------------------------------------------
_PATCH_FRAME_STATE = (
    """            d = 0
            for r in rr:
                yy = int(round(Ps[i, 0] + sgn * nv[0] * r)); xx = int(round(Ps[i, 1] + sgn * nv[1] * r))
                if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0: break
                d = r
""",
    """            d = _RAYN_.march(worm, Ps[i, 0], Ps[i, 1], sgn * nv[0], sgn * nv[1], int(rr[-1]) + 1, "np")
""")

_PATCH_LR_RADII = (
    """            for r in range(1, 60):
                yy = int(round(y0 + sign * n[0] * r)); xx = int(round(x0 + sign * n[1] * r))
                if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0: break
                d[s] = r
""",
    """            _dd = _RAYN_.march(worm, y0, x0, sign * n[0], sign * n[1], 60, "py")
            if _dd: d[s] = _dd
""")

_PATCH_BILATERAL = (
    """        def march(s):
            for r in range(1,60):
                yy=int(round(y0+s*r*n[0])); xx=int(round(x0+s*r*n[1]))
                if yy<0 or xx<0 or yy>=b.shape[0] or xx>=b.shape[1] or b[yy,xx]==0: return r-1
            return 59
""",
    """        def march(s):
            return _RAYN_.march(b, y0, x0, s*n[0], s*n[1], 60, "py")
""")


def _patched(fn, patch, name):
    """Re-compile ``fn`` from its own source with ``patch`` applied, in its
    own globals. Returns the new function or None (pattern not found)."""
    import inspect
    src = textwrap.dedent(inspect.getsource(fn))
    old, new = patch
    if src.count(old) != 1:
        return None
    src = src.replace(old, new)
    g = fn.__globals__
    g["_RAYN_"] = __import__(__name__, fromlist=["march"])
    loc = {}
    code = compile(src, getattr(fn.__code__, "co_filename", "<ray_numba>"), "exec")
    exec(code, g, loc)  # noqa: S102
    new_fn = loc.get(fn.__name__)
    if new_fn is None:
        return None
    new_fn.__wrapped_original__ = fn
    STATS["patched"].append(name)
    return new_fn


def patch_modules(log=None):
    """Patch _frame_state_impl / _lr_radii in every loaded module that has
    them (both copies of metrics_lib). Idempotent."""
    if not enabled():
        return []
    done = []
    import sys
    for mname, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for attr, patch in (("_frame_state_impl", _PATCH_FRAME_STATE),
                            ("_lr_radii", _PATCH_LR_RADII)):
            fn = getattr(mod, attr, None)
            if fn is None or not callable(fn) or getattr(fn, "__wrapped_original__", None):
                continue
            if getattr(fn, "__module__", None) != getattr(mod, "__name__", None):
                continue
            try:
                nf = _patched(fn, patch, "%s.%s" % (mname, attr))
            except Exception:  # noqa: BLE001
                nf = None
            if nf is not None:
                setattr(mod, attr, nf)
                done.append("%s.%s" % (mname, attr))
    if done and log is not None:
        log("  ray numba: %s (exact twin of the round() ray walks)" % ", ".join(done))
    return done


def patch_bilateral(fn):
    """Return the patched ``bilateral`` (or the original)."""
    if not enabled() or fn is None or getattr(fn, "__wrapped_original__", None):
        return fn
    try:
        nf = _patched(fn, _PATCH_BILATERAL, "bilateral")
    except Exception:  # noqa: BLE001
        nf = None
    return nf or fn


def status():
    return "ray numba %s (%d numba / %d python, %d checked%s)" % (
        "ON" if _STATE["ok"] is not False else "OFF",
        STATS["numba"], STATS["python"], STATS["checked"],
        (", " + _STATE["error"]) if _STATE["error"] else "")
