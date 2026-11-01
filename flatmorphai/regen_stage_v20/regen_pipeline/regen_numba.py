# -*- coding: utf-8 -*-
"""CPU fast path of the regeneration chain: exact Numba twins and per-image memo.

The profile of the chain (12 worms, reference machine) puts a third of the time
in a handful of pure helpers that the production code calls again and again on
the SAME array within one worm, and in one Python pixel loop:

    nonzero                       655 calls   1.25 s
    connectedComponentsWithStats  289 calls   1.02 s
    builtins.round            574 595 calls   0.52 s
    skeletonize (Zhang)            48 calls   0.50 s

``masque_ver`` alone runs about 24 times per worm, always on the refined mask of
that worm: the pipeline, the head cascade, the composite builder, the refiner
and the descriptor block each call it for themselves. ``extremites`` (hence
``squelette``, hence the skeleton and the skan/Dijkstra longest path) runs
twice. ``_histogrammes`` walks 120 perpendicular cuts pixel by pixel in Python.

This module removes that repetition WITHOUT touching the production files:

* a per-image memo keyed on the EXACT BYTES of the input. A hit means "this very
  array again", never "a similar one", so nothing is ever borrowed from another
  worm; the helpers are pure (same bytes in -> same bytes out), results are
  copied on store and on hit, and the first calls check that the function does
  not modify its own input (it would be left unwrapped if it did);
* a Numba twin of ``_histogrammes``. It is the same walk in the same order, and
  it only ever COUNTS pixels into integer histograms, so there is no floating
  point to round differently. Numba's ``round`` was verified to match Python's
  (half to even) on 200 000 values, and the twin was checked bit-for-bit against
  the original on 25 synthetic worms (x536 faster).

Every patch checks itself against the original on its first calls and switches
itself off for good on any difference. FLATMORPHAI_REGEN_NUMBA=0 disables.
"""

from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

from ._env import AC, CS

STATE: dict = {"twins": [], "memo": [], "disabled": [], "error": ""}
_CHECKS = 3
_lock = threading.Lock()
_installed = False


def enabled() -> bool:
    return os.environ.get("FLATMORPHAI_REGEN_NUMBA", "1") in ("1", "true", "True")


# --------------------------------------------------------------- content hash
# A 2 Mpx mask takes ~3 ms with blake2b, which is the cost of the
# connectedComponents call we are trying to avoid. The Numba 128-bit hash
# (xxh64-style, two independent 4-lane streams) does it in ~0.35 ms.
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
    _digest(np.zeros(13, np.uint8))              # compile once, at import
    _NUMBA = True
except Exception:  # noqa: BLE001 - no numba: cryptographic hash, twins disabled
    _NUMBA = False

    def _digest(a) -> bytes:
        return hashlib.blake2b(memoryview(np.ascontiguousarray(a)).cast("B"),
                               digest_size=16).digest()


class _NoKey(Exception):
    pass


def _feed(h, x):
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:
            raise _NoKey()
        h.update(b"A" + repr((x.shape, x.dtype.str)).encode())
        h.update(_digest(x) if x.size else b"")
    elif isinstance(x, (int, float, bool, str, bytes, type(None), np.generic)):
        h.update(b"S" + repr((type(x).__name__, x)).encode())
    elif isinstance(x, (list, tuple)):
        h.update(b"L" if isinstance(x, list) else b"T")
        h.update(str(len(x)).encode())
        for y in x:
            _feed(h, y)
    else:
        raise _NoKey()


def _copy(v):
    if isinstance(v, np.ndarray):
        if v.dtype.hasobject:
            raise _NoKey()
        return v.copy()
    if isinstance(v, (int, float, bool, str, bytes, type(None), np.generic)):
        return v
    if isinstance(v, list):
        return [_copy(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_copy(x) for x in v)
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    raise _NoKey()


# ----------------------------------------------------------------- the memo
def memoize(fn, name, max_entries=8):
    """Exact memo of a pure function of arrays / scalars (per-image reuse)."""
    if getattr(fn, "_regen_memo", False):
        return fn
    cache: "dict[bytes, object]" = {}
    order: "list[bytes]" = []
    st = {"calls": 0, "hits": 0, "checks": 0, "off": False}
    STATE.setdefault("counters", {})[name] = st

    def _key(args, kwargs):
        h = hashlib.blake2b(digest_size=20)
        h.update(name.encode())
        _feed(h, args)
        _feed(h, tuple(sorted(kwargs.items())))
        return h.digest()

    def wrapper(*args, **kwargs):
        if st["off"]:
            return fn(*args, **kwargs)
        try:
            k = _key(args, kwargs)
        except _NoKey:
            return fn(*args, **kwargs)
        st["calls"] += 1
        with _lock:
            hit = cache.get(k)
        if hit is not None:
            st["hits"] += 1
            return _copy(hit[0])
        out = fn(*args, **kwargs)
        if st["checks"] < _CHECKS:
            st["checks"] += 1
            try:
                same = _key(args, kwargs) == k
            except _NoKey:
                same = False
            if not same:               # the function modifies its input: leave it alone
                st["off"] = True
                STATE["disabled"].append(name)
                return out
        try:
            stored = (_copy(out),)
        except _NoKey:
            return out
        with _lock:
            cache[k] = stored
            order.append(k)
            while len(order) > max_entries:
                cache.pop(order.pop(0), None)
        return out

    wrapper._regen_memo = True
    wrapper.__wrapped__ = fn
    for a in ("__name__", "__qualname__", "__doc__", "__module__"):
        try:
            setattr(wrapper, a, getattr(fn, a))
        except Exception:  # noqa: BLE001
            pass
    return wrapper


# ------------------------------------------------- Numba twin: _histogrammes
if _NUMBA:
    @_njit(nogil=True, cache=False)
    def _hist_nb(P, msk, gf8, half, nb_coupes):  # pragma: no cover - compiled
        H, W = msk.shape
        n = P.shape[0]
        h0 = np.zeros(256, np.int64)
        h1 = np.zeros(256, np.int64)
        for i in range(n):
            if nb_coupes <= i < n - nb_coupes:
                continue
            j0 = i - 2
            if j0 < 0:
                j0 = 0
            j1 = i + 2
            if j1 > n - 1:
                j1 = n - 1
            ty = P[j1, 0] - P[j0, 0]
            tx = P[j1, 1] - P[j0, 1]
            nn = np.sqrt(ty * ty + tx * tx) + 1e-9
            py = -tx / nn
            px = ty / nn
            for r in range(-half, half + 1):
                Y = int(round(P[i, 0] + py * r))
                X = int(round(P[i, 1] + px * r))
                if 0 <= Y < H and 0 <= X < W and msk[Y, X] > 0:
                    g = np.int64(gf8[Y, X])
                    if i < nb_coupes:
                        h0[g] += 1
                    else:
                        h1[g] += 1
        return h0, h1


def _twin_histogrammes(orig):
    """Same walk, same order, integer counting only -> bit-identical."""
    st = {"ok": None, "checks": 0, "calls": 0}
    STATE.setdefault("counters", {})["_histogrammes"] = st

    def call(P, msk, gf8, half):
        if st["ok"] is False or not _NUMBA:
            return orig(P, msk, gf8, half)
        try:
            Pa = np.ascontiguousarray(np.asarray(P, np.float64))
            mk = np.ascontiguousarray(np.asarray(msk))
            gg = np.ascontiguousarray(np.asarray(gf8))
            if Pa.ndim != 2 or Pa.shape[1] != 2 or mk.ndim != 2 or gg.ndim != 2:
                return orig(P, msk, gf8, half)
            new = _hist_nb(Pa, mk, gg, int(half), int(AC.NB_COUPES))
        except Exception as exc:  # noqa: BLE001 - never fail where the original would not
            st["ok"] = False
            STATE["error"] = "%s: %s" % (type(exc).__name__, exc)
            return orig(P, msk, gf8, half)
        if st["ok"] is None:
            ref = orig(P, msk, gf8, half)
            st["checks"] += 1
            if not (np.array_equal(ref[0], new[0]) and np.array_equal(ref[1], new[1])):
                st["ok"] = False
                STATE["error"] = "difference on check %d" % st["checks"]
                return ref
            if st["checks"] >= _CHECKS:
                st["ok"] = True
            return ref
        st["calls"] += 1
        return new
    call.__wrapped__ = orig
    return call


# -------------------------------------------------------------------- install
#: (module, attribute, cache name, entries). Pure helpers of ONE worm's mask.
_MEMO_TARGETS = (
    (CS, "masque_ver", "masque_ver", 4),
    (CS, "skeleton", "skeleton", 4),
    (CS, "extremites", "extremites", 4),
    (CS, "_boite", "_boite", 8),
    (CS, "masque_dilate", "masque_dilate", 4),
)


def install(log=None):
    """Patch the repeated helpers and the cut loop. Idempotent; returns a list."""
    global _installed
    if _installed or not enabled():
        return []
    _installed = True
    done = []
    for mod, attr, name, n in _MEMO_TARGETS:
        try:
            fn = getattr(mod, attr, None)
            if callable(fn) and not getattr(fn, "_regen_memo", False):
                setattr(mod, attr, memoize(fn, name, n))
                done.append(name)
        except Exception:  # noqa: BLE001
            pass
    STATE["memo"] = list(done)
    try:
        if _NUMBA and not getattr(AC._histogrammes, "__wrapped__", None):
            AC._histogrammes = _twin_histogrammes(AC._histogrammes)
            STATE["twins"].append("_histogrammes")
    except Exception:  # noqa: BLE001
        pass
    if log is not None and (done or STATE["twins"]):
        log("  regeneration fast path: memo %s | numba %s"
            % (", ".join(done) or "-", ", ".join(STATE["twins"]) or "-"))
    return done


def clear():
    """Nothing to drop between folders: the caches hold at most a few arrays
    each and are keyed by content, so a new worm simply misses."""
    return None


def stats():
    """{name: {calls, hits, ...}} since the process started."""
    return {k: dict(v) for k, v in STATE.get("counters", {}).items()}
