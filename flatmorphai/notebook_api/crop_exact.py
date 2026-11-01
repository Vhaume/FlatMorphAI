# -*- coding: utf-8 -*-
"""Run the full-frame mask operators on the bounding box of the object only.

The masks are full resolution (e.g. 1200x1600 = 1.9 Mpx) while the worm is a
few 10 kpx: the Zhang skeleton and scipy's exact Euclidean distance transform
spend almost all their time on background (measured on 120 real masks:
skeleton x13, EDT x22 faster when cropped, 0 differing pixel).

Why the cropped result is EXACTLY the full one (then embedded back in zeros):
* Zhang thinning is a local 3x3 rule; a pixel whose 3x3 neighbourhood is all
  background never changes. The crop keeps a >= 2 px background margin (or
  stops at the image edge, where it coincides with the full image's edge), so
  every foreground pixel sees the same neighbourhood in every pass.
* Euclidean distance to the nearest zero: if that zero q lies outside the
  crop, clamping q onto the crop border gives a zero pixel (the margin is
  background) that is no farther in both coordinates, so the minimum is
  reached inside the crop; distances are computed from integer offsets.
(OpenCV's distanceTransform was tested too and is NOT bit-identical when
cropped, so it is left untouched.)
Every wrapper also compares its first calls bit for bit with the original
and switches itself off for good on any difference.
FLATMORPHAI_CROP_EXACT=0 disables.
"""
from __future__ import annotations

import os
import threading

import numpy as np

STATE: dict = {}
CHECKS = 3
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("FLATMORPHAI_CROP_EXACT", "1") in ("1", "true", "True")


def _box(a, margin):
    """(y0, y1, x0, x1) of the nonzero pixels of 2-D ``a`` widened by
    ``margin`` and clipped to the image, or None when cropping is pointless."""
    if a.ndim != 2:
        return None
    H, W = a.shape
    if H * W < 40000:
        return None
    nz = a if a.dtype == np.bool_ else (a != 0)
    rows = np.flatnonzero(nz.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(nz.any(axis=0))
    y0, y1 = max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, H)
    x0, x1 = max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, W)
    if (y1 - y0) * (x1 - x0) > 0.6 * H * W:
        return None
    return y0, y1, x0, x1


def _same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and \
            np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return a is b or a == b


def _guarded(name, fast, orig):
    st = STATE.setdefault(name, {"ok": None, "checks": 0, "fast": 0, "error": ""})

    def call(*a, **k):
        if st["ok"] is False:
            return orig(*a, **k)
        try:
            r = fast(*a, **k)
        except _Skip:
            return orig(*a, **k)
        except Exception as exc:  # noqa: BLE001 - never fail where the original would not
            st["error"] = "%s: %s" % (type(exc).__name__, exc)
            return orig(*a, **k)
        if st["checks"] < CHECKS:
            ref = orig(*a, **k)
            with _lock:
                st["checks"] += 1
                if not _same(ref, r):
                    st["ok"] = False
                    st["error"] = "difference on check %d" % st["checks"]
                    return ref
                if st["checks"] >= CHECKS:
                    st["ok"] = True
            return ref
        st["fast"] += 1
        return r
    call.__crop_exact__ = True
    call.__wrapped__ = orig
    return call


class _Skip(Exception):
    pass


def zhang(orig):
    """Wrap skimage's _skeletonize_zhang(image)."""
    def fast(image):
        a = np.asarray(image)
        if a.dtype != np.bool_:        # skeletonize always passes a C bool array
            raise _Skip()
        b = _box(a, 2)
        if b is None:
            raise _Skip()
        y0, y1, x0, x1 = b
        sub = orig(np.ascontiguousarray(a[y0:y1, x0:x1]))
        out = np.zeros(a.shape, dtype=sub.dtype)
        out[y0:y1, x0:x1] = sub
        return out
    return _guarded("skeleton_zhang", fast, orig)


def edt(orig):
    """Wrap scipy.ndimage.distance_transform_edt (distances only)."""
    def fast(input, sampling=None, return_distances=True, return_indices=False,
             distances=None, indices=None):
        if (sampling is not None or not return_distances or return_indices
                or distances is not None or indices is not None):
            raise _Skip()
        a = np.asarray(input)
        b = _box(a, 1)
        if b is None:
            raise _Skip()
        y0, y1, x0, x1 = b
        sub = orig(np.ascontiguousarray(a[y0:y1, x0:x1]))
        out = np.zeros(a.shape, dtype=sub.dtype)
        out[y0:y1, x0:x1] = sub
        return out
    return _guarded("edt", fast, orig)


def cv_dist(orig):
    """Wrap cv2.distanceTransform(src, distanceType, maskSize[, dst[, dstType]])."""
    import cv2
    ok_types = {cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_C}

    def fast(src, distanceType, maskSize, *rest, **kw):
        if rest or kw or distanceType not in ok_types or maskSize not in (3, 5):
            raise _Skip()
        a = np.asarray(src)
        if a.dtype != np.uint8:
            raise _Skip()
        b = _box(a, 2)
        if b is None:
            raise _Skip()
        y0, y1, x0, x1 = b
        sub = orig(np.ascontiguousarray(a[y0:y1, x0:x1]), distanceType, maskSize)
        out = np.zeros(a.shape, dtype=sub.dtype)
        out[y0:y1, x0:x1] = sub
        return out
    return _guarded("cv_distance", fast, orig)


_installed = {"done": False, "edt_orig": None}


def install(log=None):
    """Patch the Zhang core (module global used by skeletonize / the memo),
    scipy's distance_transform_edt (scipy.ndimage + every module that imported
    it by name). Idempotent."""
    if not enabled():
        return []
    done = []
    with _lock:
        if not _installed["done"]:
            _installed["done"] = True
            try:
                import skimage.morphology._skeletonize as _SK
                cur = _SK._skeletonize_zhang
                # the memo wrapper (memo_cache) calls the Zhang core it captured:
                # patch underneath it when present, else the module global
                import sys as _sys
                mc = _sys.modules.get("flatmorphai.notebook_api.memo_cache")
                if mc is not None and getattr(mc, "_ZHANG_CORE", None) is not None:
                    mc._ZHANG_CORE[0] = zhang(mc._ZHANG_CORE[0])
                elif not getattr(cur, "__crop_exact__", False):
                    _SK._skeletonize_zhang = zhang(cur)
                done.append("skeleton")
            except Exception:  # noqa: BLE001
                pass
            try:
                import scipy.ndimage as _nd
                orig = _nd.distance_transform_edt
                _installed["edt_orig"] = orig
                _installed["edt_new"] = edt(orig)
                done.append("edt")
            except Exception:  # noqa: BLE001
                pass
            # cv2.distanceTransform is NOT cropped: measured not bit-identical
            # on the user's masks (OpenCV's float chamfer depends on the frame
            # extent); cv_dist() is kept for reference only.
    # rebind distance_transform_edt wherever it was imported by name (cheap,
    # repeated so lazily imported modules are covered too)
    orig, new = _installed.get("edt_orig"), _installed.get("edt_new")
    if orig is not None and new is not None:
        import sys as _sys
        for m in list(_sys.modules.values()):
            d = getattr(m, "__dict__", None)
            if isinstance(d, dict):
                for k in ("distance_transform_edt", "_edt", "edt"):
                    if d.get(k) is orig:
                        d[k] = new
    if done and log is not None:
        log("  bounding-box crop (exact, self-checked): %s" % ", ".join(done))
    return done
