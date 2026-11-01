# -*- coding: utf-8 -*-
"""GIL-free, bit-identical drop-in for ``skimage.graph.MCP_Geometric``.

WHY: the v4j stereotype chain spends most of its time in weighted geodesics
(``MCP_Geometric`` in ``_geod_crop``, ``axe_pointes`` (v4c), ``geod`` (v3z)).
skimage's Cython MCP HOLDS THE GIL (measured: 4 threads slower than 1), so the
descriptor thread pool cannot parallelise the dominant cost. This module is a
Numba ``@njit(nogil=True)`` Dijkstra that reproduces MCP_Geometric EXACTLY:

* 8-connectivity, edge a->b = 0.5 * (cost[a] + cost[b]) * |offset|
  (the MCP_Geometric travel cost), start cumulative cost 0;
* costs cast to float64 first, as skimage does (OpenCV's distanceTransform is
  float32 -- summing in float32 was the only source of a 3e-8 residual);
* traceback through the stored predecessors.

Measured in the sandbox against skimage 0.25.2 (384x384 worm-like mask):
uniform cost -> max abs diff 0.0; weighted 1/dt^2 cost -> 0.0; traceback path
identical; ~2x faster single-threaded, and it SCALES with threads (nogil).

Opt-in: ``install()`` patches ``skimage.graph.MCP_Geometric`` (so the golden
programs, which do ``from skimage.graph import MCP_Geometric`` when they are
exec'd, pick it up) and self-validates against the original on the first call;
any mismatch or error -> permanent fallback to skimage. Nothing changes unless
``install()`` is called.
"""
from __future__ import annotations

import heapq
import threading

import numpy as np

try:  # numba is optional: without it the module is inert (install() -> False)
    from numba import njit
    _NUMBA = True
except Exception:  # noqa: BLE001
    _NUMBA = False

_SQ2 = 1.4142135623730951

if _NUMBA:
    _DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
    _DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
    _LEN = np.array([_SQ2, 1.0, _SQ2, 1.0, 1.0, _SQ2, 1.0, _SQ2], np.float64)

    @njit(nogil=True, cache=False)
    def _dijkstra(cost, starts):  # pragma: no cover - compiled
        H, W = cost.shape
        cum = np.full((H, W), np.inf)
        pred = np.full(H * W, -1, np.int64)
        done = np.zeros((H, W), np.bool_)
        h = [(0.0, np.int64(0))]
        h.pop()
        for s in range(starts.shape[0]):
            sy = starts[s, 0]
            sx = starts[s, 1]
            if 0 <= sy < H and 0 <= sx < W and np.isfinite(cost[sy, sx]):
                cum[sy, sx] = 0.0
                heapq.heappush(h, (0.0, sy * W + sx))
        while len(h) > 0:
            d, idx = heapq.heappop(h)
            y = idx // W
            x = idx - y * W
            if done[y, x]:
                continue
            done[y, x] = True
            ca = cost[y, x]
            for k in range(8):
                ny = y + _DY[k]
                nx = x + _DX[k]
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                cb = cost[ny, nx]
                if done[ny, nx] or not np.isfinite(cb):
                    continue
                nd = d + 0.5 * (ca + cb) * _LEN[k]
                if nd < cum[ny, nx]:
                    cum[ny, nx] = nd
                    pred[ny * W + nx] = idx
                    heapq.heappush(h, (nd, ny * W + nx))
        return cum, pred


#: Above this many pixels the call is delegated to skimage: the lazy-deletion
#: heap of the Numba Dijkstra uses far more memory than skimage's indexed heap
#: on full-resolution masks (x7 threads -> the process was OOM-killed). The v4j
#: hot paths run on 384x384 crops / bounding boxes (<= ~150k px), well below.
_MAX_PX = 400_000

#: The original skimage class, captured by install() (needed for delegation).
_ORIG_CLS = None

#: Diagnostic counters: calls served by Numba vs delegated to skimage.
STATS = {"numba": 0, "delegated": 0}


if _NUMBA:
    @njit(nogil=True, cache=False)
    def _dijkstra_targets(cost, starts, targets):  # pragma: no cover - compiled
        """Same Dijkstra as _dijkstra, but stops as soon as every target cell is
        settled. Predecessors of settled cells never change afterwards, so the
        traceback to any target is identical to the full run's."""
        H, W = cost.shape
        cum = np.full((H, W), np.inf)
        pred = np.full(H * W, -1, np.int64)
        done = np.zeros((H, W), np.bool_)
        is_t = np.zeros((H, W), np.bool_)
        n_t = 0
        for t in range(targets.shape[0]):
            ty = targets[t, 0]
            tx = targets[t, 1]
            if 0 <= ty < H and 0 <= tx < W and not is_t[ty, tx]:
                is_t[ty, tx] = True
                n_t += 1
        h = [(0.0, np.int64(0))]
        h.pop()
        for s in range(starts.shape[0]):
            sy = starts[s, 0]
            sx = starts[s, 1]
            if 0 <= sy < H and 0 <= sx < W and np.isfinite(cost[sy, sx]):
                cum[sy, sx] = 0.0
                heapq.heappush(h, (0.0, sy * W + sx))
        got = 0
        while len(h) > 0 and got < n_t:
            d, idx = heapq.heappop(h)
            y = idx // W
            x = idx - y * W
            if done[y, x]:
                continue
            done[y, x] = True
            if is_t[y, x]:
                got += 1
                if got >= n_t:
                    break
            ca = cost[y, x]
            for k in range(8):
                ny = y + _DY[k]
                nx = x + _DX[k]
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                cb = cost[ny, nx]
                if done[ny, nx] or not np.isfinite(cb):
                    continue
                nd = d + 0.5 * (ca + cb) * _LEN[k]
                if nd < cum[ny, nx]:
                    cum[ny, nx] = nd
                    pred[ny * W + nx] = idx
                    heapq.heappush(h, (nd, ny * W + nx))
        return cum, pred, done


def make_fast_axe_pointes(orig):
    """Exact, faster twin of v4c ``axe_pointes`` (the dominant Dijkstra user):
    same cost map, same tips (``pointes``), same ``dedans``, same pair loop and
    resampling, but (1) no Dijkstra from the LAST tip (it has no pair left) and
    (2) each Dijkstra stops once all remaining target tips are settled (paths to
    settled cells are final). Run on the bounding box of the mask. Falls back to
    ``orig`` whenever numba is unavailable."""
    if not _NUMBA:
        return orig
    g = orig.__globals__
    pointes = g["pointes"]
    NB = g["NB"]
    import cv2
    from scipy.ndimage import gaussian_filter1d

    def axe_pointes(m):
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        cost = np.where(m, 1.0 / (dt / max(dt.max(), 1e-6) + 0.05) ** 2, np.inf)
        P, S = pointes(m)
        H, W = m.shape

        def dedans(p):
            y, x = int(np.clip(round(p[0]), 0, H - 1)), int(np.clip(round(p[1]), 0, W - 1))
            if m[y, x]:
                return (y, x)
            ys, xs = np.nonzero(m)
            k = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2))
            return (int(ys[k]), int(xs[k]))

        pts = [dedans(p) for p in P]
        best = None
        fin = np.isfinite(cost)
        rows = np.flatnonzero(fin.any(axis=1))
        cols = np.flatnonzero(fin.any(axis=0))
        y0, x0 = (int(rows[0]), int(cols[0])) if rows.size else (0, 0)
        y1, x1 = (int(rows[-1]) + 1, int(cols[-1]) + 1) if rows.size else (H, W)
        cc = np.ascontiguousarray(cost[y0:y1, x0:x1].astype(np.float64))
        Wc = x1 - x0
        for i in range(len(pts) - 1):          # last tip: no pair left
            tg = [(pts[j][0] - y0, pts[j][1] - x0) for j in range(i + 1, len(pts))]
            _cum, pred, done = _dijkstra_targets(
                cc, np.asarray([(pts[i][0] - y0, pts[i][1] - x0)], np.int64),
                np.asarray(tg, np.int64))
            for j in range(i + 1, len(pts)):
                ty, tx = pts[j][0] - y0, pts[j][1] - x0
                if not (0 <= ty < cc.shape[0] and 0 <= tx < Wc) or not done[ty, tx]:
                    raise ValueError("no minimum-cost path was found to the specified end point")
                idx = ty * Wc + tx
                path = [(ty + y0, tx + x0)]
                while pred[idx] >= 0:
                    idx = int(pred[idx])
                    path.append((idx // Wc + y0, idx % Wc + x0))
                path = np.array(path[::-1], float)
                L = float(np.sum(np.hypot(*np.diff(path, axis=0).T))) if len(path) > 1 else 0.0
                if best is None or L > best[0]:
                    best = (L, path)
        assert best is not None and best[0] > 10, "no pair of tips"
        path = best[1]
        seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        tt = np.linspace(0, seg[-1], NB)
        Q = np.c_[np.interp(tt, seg, path[:, 0]), np.interp(tt, seg, path[:, 1])]
        Q = gaussian_filter1d(Q, 1.5, axis=0, mode="nearest")
        return Q, P

    return axe_pointes


# ---------------------------------------------------------------------------
# Same Dijkstra on a cost map padded with a +inf border: the 8 neighbours of
# a cell are then plain index offsets, with no bounds test (an outside cell
# is +inf, i.e. skipped exactly like an out-of-image one). Row-major order is
# kept in the padded grid, so the (cost, index) heap ties break the same way
# and cum / pred / done are bit-identical to _dijkstra / _dijkstra_targets
# (checked on 225 real calls of 4 frames: identical; x1.5 to x1.7 faster
# alone, but NO measurable gain inside the real multi-threaded descriptor
# run: 6.65 -> 6.57 s per 10 frames). Kept OPT-IN only:
# FLATMORPHAI_DIJKSTRA_PAD=1 enables it (self-checked on the first calls).
# ---------------------------------------------------------------------------
if _NUMBA:
    @njit(nogil=True, cache=False)
    def _dij_pad_core(cp, H, W, starts, targets, use_t):  # pragma: no cover - compiled
        W2 = W + 2
        N2 = (H + 2) * W2
        cum = np.full(N2, np.inf)
        pred = np.full(N2, -1, np.int64)
        done = np.zeros(N2, np.bool_)
        is_t = np.zeros(N2, np.bool_)
        off = np.array([-W2 - 1, -W2, -W2 + 1, -1, 1, W2 - 1, W2, W2 + 1], np.int64)
        n_t = 0
        if use_t:
            for t in range(targets.shape[0]):
                ty = targets[t, 0]
                tx = targets[t, 1]
                if 0 <= ty < H and 0 <= tx < W:
                    q = (ty + 1) * W2 + tx + 1
                    if not is_t[q]:
                        is_t[q] = True
                        n_t += 1
        h = [(0.0, np.int64(0))]
        h.pop()
        for s in range(starts.shape[0]):
            sy = starts[s, 0]
            sx = starts[s, 1]
            if 0 <= sy < H and 0 <= sx < W:
                q = (sy + 1) * W2 + sx + 1
                if np.isfinite(cp[q]):
                    cum[q] = 0.0
                    heapq.heappush(h, (0.0, q))
        got = 0
        while len(h) > 0:
            if use_t and got >= n_t:
                break
            d, q = heapq.heappop(h)
            if done[q]:
                continue
            done[q] = True
            if use_t and is_t[q]:
                got += 1
                if got >= n_t:
                    break
            ca = cp[q]
            for k in range(8):
                r = q + off[k]
                if done[r]:
                    continue
                cb = cp[r]
                if not np.isfinite(cb):
                    continue
                nd = d + 0.5 * (ca + cb) * _LEN[k]
                if nd < cum[r]:
                    cum[r] = nd
                    pred[r] = q
                    heapq.heappush(h, (nd, r))
        cum_o = np.empty((H, W))
        pred_o = np.empty(H * W, np.int64)
        done_o = np.empty((H, W), np.bool_)
        for y in range(H):
            for x in range(W):
                q = (y + 1) * W2 + x + 1
                cum_o[y, x] = cum[q]
                done_o[y, x] = done[q]
                p = pred[q]
                if p >= 0:
                    py = p // W2
                    pred_o[y * W + x] = (py - 1) * W + (p - py * W2 - 1)
                else:
                    pred_o[y * W + x] = -1
        return cum_o, pred_o, done_o

    _NO_T = np.zeros((0, 2), np.int64)

    def _padded(cost):
        H, W = cost.shape
        cp = np.full((H + 2, W + 2), np.inf)
        cp[1:-1, 1:-1] = cost
        return cp.reshape(-1), H, W

    def _dijkstra_pad(cost, starts):
        cp, H, W = _padded(cost)
        cum, pred, _done = _dij_pad_core(cp, H, W, starts, _NO_T, False)
        return cum, pred

    def _dijkstra_targets_pad(cost, starts, targets):
        cp, H, W = _padded(cost)
        return _dij_pad_core(cp, H, W, starts, targets, True)

    PAD_STATE = {"ok": None, "checks": 0, "error": ""}
    _PAD_CHECKS = 3
    _dijkstra_ref = _dijkstra
    _dijkstra_targets_ref = _dijkstra_targets

    def _same_out(a, b):
        return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))

    def _dijkstra_sel(cost, starts):
        st = PAD_STATE
        if st["ok"] is False:
            return _dijkstra_ref(cost, starts)
        if st["ok"] is None:
            ref = _dijkstra_ref(cost, starts)
            try:
                new = _dijkstra_pad(cost, starts)
                good = _same_out(ref, new)
            except Exception as exc:  # noqa: BLE001
                good, st["error"] = False, "%s: %s" % (type(exc).__name__, exc)
            st["checks"] += 1
            if not good:
                st["ok"] = False
                st["error"] = st["error"] or "difference on check %d" % st["checks"]
            elif st["checks"] >= 2 * _PAD_CHECKS:
                st["ok"] = True
            return ref
        return _dijkstra_pad(cost, starts)

    def _dijkstra_targets_sel(cost, starts, targets):
        st = PAD_STATE
        if st["ok"] is False:
            return _dijkstra_targets_ref(cost, starts, targets)
        if st["ok"] is None:
            ref = _dijkstra_targets_ref(cost, starts, targets)
            try:
                new = _dijkstra_targets_pad(cost, starts, targets)
                good = _same_out(ref, new)
            except Exception as exc:  # noqa: BLE001
                good, st["error"] = False, "%s: %s" % (type(exc).__name__, exc)
            st["checks"] += 1
            if not good:
                st["ok"] = False
                st["error"] = st["error"] or "difference on check %d" % st["checks"]
            elif st["checks"] >= 2 * _PAD_CHECKS:
                st["ok"] = True
            return ref
        return _dijkstra_targets_pad(cost, starts, targets)

    import os as _os
    if _os.environ.get("FLATMORPHAI_DIJKSTRA_PAD", "0") in ("1", "true", "True"):
        _dijkstra = _dijkstra_sel                    # module globals used below / by axe_pointes
        _dijkstra_targets = _dijkstra_targets_sel


class MCP_Geometric:  # noqa: N801 - mirrors the skimage class name on purpose
    """Drop-in for ``skimage.graph.MCP_Geometric``.

    1. CROP: the cost is +inf outside the mask, so no path can leave the
       bounding box of the finite cells. Every call is run on that box (the
       raw masks are 1200x1600 = 1.92 Mpx for a ~17 kpx worm) and the result is
       embedded back into a full-size array -- same values, same paths
       (row-major order, hence tie-breaking, is preserved inside the box).
    2. On the box: ``find_costs(starts)`` without ``ends`` runs the GIL-free
       Numba Dijkstra (bit-identical); ``ends``/other options, a box still
       larger than _MAX_PX, a non-finite start, or any error run the REAL
       skimage MCP on the box -- i.e. exactly skimage's behaviour, just smaller.
    Options other than ``fully_connected=True`` (the default) go to skimage on
    the full array, untouched."""

    def __new__(cls, costs, *args, **kwargs):
        ok_kwargs = (not args) and set(kwargs) <= {"fully_connected"} and \
            kwargs.get("fully_connected", True) is True
        arr = np.asarray(costs)
        if (not _NUMBA) or (not ok_kwargs) or arr.ndim != 2:
            STATS["delegated"] += 1
            return _ORIG_CLS(costs, *args, **kwargs)   # real skimage object
        return super().__new__(cls)

    def __init__(self, costs, *args, **kwargs):
        c = np.asarray(costs, dtype=np.float64)
        self._shape = c.shape
        fin = np.isfinite(c)
        if fin.any() and not fin.all():
            rows = np.flatnonzero(fin.any(axis=1))
            cols = np.flatnonzero(fin.any(axis=0))
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
        else:
            y0, y1, x0, x1 = 0, c.shape[0], 0, c.shape[1]
        self._off = (y0, x0)
        self._box = (y1 - y0, x1 - x0)
        self._cost = np.ascontiguousarray(c[y0:y1, x0:x1])
        self._pred = None
        self._cumc = None
        self._sk = None            # skimage object on the box, when used

    def _inbox(self, p):
        y, x = int(p[0]) - self._off[0], int(p[1]) - self._off[1]
        if 0 <= y < self._box[0] and 0 <= x < self._box[1]:
            return (y, x)
        return None

    def _embed(self, cumc):
        out = np.full(self._shape, np.inf)
        y0, x0 = self._off
        out[y0:y0 + self._box[0], x0:x0 + self._box[1]] = cumc
        return out

    def find_costs(self, starts, ends=None, find_all_ends=True, **kwargs):
        self._sk = None
        sc = [self._inbox(p) for p in starts]
        ec = None if ends is None else [self._inbox(p) for p in ends]
        if any(p is None for p in sc) or (ec is not None and any(p is None for p in ec)):
            # a point outside the finite box: let skimage handle the full array
            STATS["delegated"] += 1
            full = np.full(self._shape, np.inf)
            y0, x0 = self._off
            full[y0:y0 + self._box[0], x0:x0 + self._box[1]] = self._cost
            self._sk = ("full", _ORIG_CLS(full))
            return self._sk[1].find_costs(starts, ends, find_all_ends, **kwargs)
        start_ok = all(np.isfinite(self._cost[p]) for p in sc)
        use_skimage = (ends is not None) or bool(kwargs) or (not start_ok) or \
            (self._cost.size > _MAX_PX)
        if not use_skimage:
            try:
                st = np.asarray(sc, dtype=np.int64)
                cumc, pred = _dijkstra(self._cost, st)
                STATS["numba"] += 1
                self._pred, self._cumc = pred, cumc
                return self._embed(cumc), None
            except Exception:  # noqa: BLE001 - never fail where skimage would not
                pass
        STATS["delegated"] += 1
        o = _ORIG_CLS(self._cost)
        cumc, _tb = o.find_costs(sc, ec, find_all_ends, **kwargs)
        self._sk = ("box", o)
        return self._embed(cumc), None

    def traceback(self, end):
        if self._sk is not None:
            kind, o = self._sk
            if kind == "full":
                return o.traceback(end)
            e = self._inbox(end)
            if e is None:
                raise ValueError("no minimum-cost path was found to the specified end point")
            y0, x0 = self._off
            return [(int(y) + y0, int(x) + x0) for y, x in o.traceback(e)]
        e = self._inbox(end)
        if e is None or not np.isfinite(self._cumc[e]):
            # same behaviour as skimage when the end is unreachable
            raise ValueError("no minimum-cost path was found to the specified end point")
        W = self._box[1]
        y0, x0 = self._off
        idx = e[0] * W + e[1]
        path = [(e[0] + y0, e[1] + x0)]
        pred = self._pred
        guard = pred.shape[0] + 1
        while pred[idx] >= 0 and guard > 0:
            idx = int(pred[idx])
            path.append((idx // W + y0, idx % W + x0))
            guard -= 1
        path.reverse()
        return path


_lock = threading.Lock()
_state = {"installed": False, "validated": None, "orig": None, "reason": ""}


def _validate(orig_cls):
    """Compare against skimage on a synthetic worm-like mask (uniform and
    weighted costs, field + traceback). True only if bit-identical."""
    import cv2
    H = W = 160
    yy, xx = np.mgrid[:H, :W]
    m = ((yy - 80) / 65.0) ** 2 + ((xx - 80) / 24.0) ** 2 < 1
    m |= ((yy - 45) / 14.0) ** 2 + ((xx - 110) / 38.0) ** 2 < 1
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    weighted = np.where(m, 1.0 / (dt / max(dt.max(), 1e-6) + 0.05) ** 2, np.inf)
    uniform = np.where(m, 1.0, np.inf)
    src, tgt = (80, 80), (20, 80)
    for cost in (uniform, weighted):
        ref = orig_cls(cost)
        rc, _ = ref.find_costs([src])
        new = MCP_Geometric(cost)
        nc, _ = new.find_costs([src])
        fin = np.isfinite(rc)
        if not np.array_equal(fin, np.isfinite(nc)):
            return False, "finite support differs"
        if not np.array_equal(rc[fin], nc[fin]):
            return False, "cost field differs (max %.3e)" % float(np.max(np.abs(rc[fin] - nc[fin])))
        if [tuple(p) for p in ref.traceback(tgt)] != new.traceback(tgt):
            return False, "traceback differs"
    # Usages seen in the app (v3c_descriptor_patches): fully_connected=True and
    # find_costs(starts, ends) -> must behave exactly like skimage.
    o = orig_cls(weighted, fully_connected=True); o.find_costs([src], [tgt])
    n = MCP_Geometric(weighted, fully_connected=True); n.find_costs([src], [tgt])
    if [tuple(p) for p in o.traceback(tgt)] != [tuple(p) for p in n.traceback(tgt)]:
        return False, "find_costs(starts, ends) path differs"
    # full-resolution frame with a small worm: cropped path == skimage full
    big_c = np.full((900, 1200), np.inf)
    big_c[300:300 + H, 500:500 + W] = weighted
    s2, t2 = (src[0] + 300, src[1] + 500), (tgt[0] + 300, tgt[1] + 500)
    r = orig_cls(big_c); rc, _ = r.find_costs([s2])
    n = MCP_Geometric(big_c); nc, _ = n.find_costs([s2])
    if not np.array_equal(rc, nc):
        return False, "cropped full-res field differs"
    if [tuple(p) for p in r.traceback(t2)] != n.traceback(t2):
        return False, "cropped full-res traceback differs"
    r = orig_cls(big_c); r.find_costs([s2], [t2])
    n = MCP_Geometric(big_c); n.find_costs([s2], [t2])
    if [tuple(p) for p in r.traceback(t2)] != [tuple(p) for p in n.traceback(t2)]:
        return False, "cropped full-res ends path differs"
    return True, ("bit-identical to skimage MCP_Geometric (uniform, weighted, "
                  "full-res cropped, ends)")


def install(log=None) -> bool:
    """Patch skimage.graph.MCP_Geometric with the GIL-free version, after a
    bit-exactness self-check. Returns True if active."""
    _log = log or (lambda _m: None)
    with _lock:
        if _state["installed"]:
            return bool(_state["validated"])
        if not _NUMBA:
            _state["reason"] = "numba not installed"
            _log("  MCP numba: OFF (numba not installed)")
            return False
        import skimage.graph as _sg
        global _ORIG_CLS
        orig = _sg.MCP_Geometric
        if orig is MCP_Geometric:          # re-install after uninstall
            orig = _state["orig"] or orig
        _ORIG_CLS = orig
        try:
            ok, why = _validate(orig)
        except Exception as exc:  # noqa: BLE001
            ok, why = False, "validation raised %s: %s" % (type(exc).__name__, exc)
        _state.update(installed=True, validated=ok, orig=orig, reason=why)
        if ok:
            _sg.MCP_Geometric = MCP_Geometric
            try:
                import skimage.graph.mcp as _sgm  # some code imports from here
                _sgm.MCP_Geometric = MCP_Geometric
            except Exception:  # noqa: BLE001
                pass
        _log("  MCP numba: %s (%s)" % ("ON, GIL-free" if ok else "OFF", why))
        return ok


def uninstall() -> None:
    with _lock:
        if _state["installed"] and _state["orig"] is not None:
            import skimage.graph as _sg
            _sg.MCP_Geometric = _state["orig"]
        _state.update(installed=False, validated=None)


def status() -> str:
    return _state["reason"] or "not installed"
