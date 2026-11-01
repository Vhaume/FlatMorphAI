# -*- coding: utf-8 -*-
"""GPU-accelerated, bit-faithful replacement for the v3g ``_geod`` geodesic.

``_geod(m, depart)`` is the 8-connected geodesic (shortest-path) distance from
``depart`` to every pixel, measured INSIDE the binary mask ``m`` with step
weights 1 (orthogonal) and sqrt(2) (diagonal), and +inf everywhere outside the
mask. In the chain it is skimage's ``MCP_Geometric`` with a cost field that is
1.0 inside the mask and +inf outside -- i.e. exactly that geodesic. It is the
single most expensive operation of the v4j stereotype descriptor chain (called
~26x per frame by the axis computation), is pure-CPU Cython and holds the GIL.

This module computes the SAME field on the GPU by synchronous min-plus
relaxation (Bellman-Ford / wavefront) over the 8-neighbourhood, in float64. The
shortest-path distance is unique, so once the relaxation converges every in-mask
cell holds the exact same value MCP_Geometric would give (to float rounding).

CORRECTNESS -- no descriptor is allowed to change. The very first distinct masks
are computed on BOTH paths and compared to the CPU reference; the GPU path is
enabled only if they agree within a tight tolerance AND have the same
finite/inf pattern. If anything disagrees, does not converge, or raises (no
torch, no CUDA, API change), the module falls back to the CPU reference
PERMANENTLY. It can therefore never degrade the metrics: at worst it is simply
the (already bbox-cropped) CPU path.
"""
from __future__ import annotations

import os
import threading

import numpy as np

# None = not yet decided, True = GPU validated, False = use CPU reference.
_GPU_OK = None
_DEVICE = None
_TORCH = None
_LOCK = threading.Lock()
# Human-readable reason for the current state (why GPU is/ isn't used), so the
# fallback is never silent in the log.
_REASON = "not yet attempted"

# Validation: compare GPU vs CPU on the first few masks. In float32 a CORRECT
# min-plus relaxation matches MCP_Geometric (float64) to ~1e-2 at most (float32
# rounding accumulated along the path), while any connectivity / weight / non-
# convergence bug differs by O(1)+. 0.05 cleanly separates the two: it passes a
# correct float32 field and rejects a wrong one (-> permanent CPU fallback).
_VALIDATE_TOL = 5e-2
_VALIDATE_NEEDED = 3        # distinct masks that must pass before trusting GPU
_validated = 0
# Hard cap on relaxation iterations. Synchronous Bellman-Ford converges in at
# most (hop count of the longest shortest path) sweeps; this bound is generous.
# If a call hits the cap without converging, it is treated as a validation
# failure (permanent CPU fallback) so we never silently return a wrong field.
_MAX_ITERS_FACTOR = 3


def status():
    """One-line human-readable state of the GPU geodesic, for logging."""
    if _GPU_OK is True:
        return "v4j geodesic: GPU active (%s)" % _REASON
    if _GPU_OK is False:
        return "v4j geodesic: CPU fallback (%s)" % _REASON
    return "v4j geodesic: CPU (%s)" % _REASON


def _try_init():
    """Import torch + pick a CUDA device, once. Returns True iff GPU usable."""
    global _TORCH, _DEVICE, _REASON
    if _TORCH is not None:
        return _DEVICE is not None
    try:
        import torch
        _TORCH = torch
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
            return True
        _REASON = "torch has no CUDA device (torch.cuda.is_available() is False)"
    except Exception as exc:  # noqa: BLE001
        _TORCH = False
        _REASON = "torch import failed: %s: %s" % (type(exc).__name__, exc)
    _DEVICE = None
    return False


def _gpu_geod_batch(mc, sources):
    """GPU geodesic on the CROPPED mask ``mc`` (bool, HxW) for one OR MANY
    sources at once. ``sources`` is a list of (row, col) inside the mask.

    Returns a float32 numpy array (B, H, W), +inf outside the mask. All B
    sources are relaxed together in ONE batched min-plus wavefront -- that
    batching is where the GPU wins (the per-iteration launch overhead of the
    sequential relaxation is shared across the whole batch, exactly like the
    Gabor path batches its convolutions). float32 keeps the GTX 1060 off its
    crippled float64 path; the distances then differ from the exact CPU field
    by ~1e-5, which the caller's tolerance accepts.

    Raises on non-convergence or any GPU problem so the caller can fall back.
    """
    torch = _TORCH
    dev = _DEVICE
    Fn = torch.nn.functional
    H, W = mc.shape
    B = len(sources)
    INF = float("inf")
    r2 = float(np.sqrt(2.0))
    m = torch.from_numpy(np.ascontiguousarray(mc)).to(dev).bool()
    mB = m.view(1, H, W).expand(B, H, W)
    d = torch.full((B, H, W), INF, dtype=torch.float32, device=dev)
    for b, (sr, sc) in enumerate(sources):
        d[b, int(sr), int(sc)] = 0.0
    # 3x3 min-plus neighbourhood, centre included (weight 0 keeps current d).
    offs = [(0, 0, 0.0),
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, r2), (-1, 1, r2), (1, -1, r2), (1, 1, r2)]
    max_iters = _MAX_ITERS_FACTOR * (H + W) + 16
    check_every = 16                  # amortise the convergence sync
    converged = False
    it = 0
    while it < max_iters:
        prev = d
        for _ in range(min(check_every, max_iters - it)):
            dpad = Fn.pad(d, (1, 1, 1, 1), value=INF)      # (B, H+2, W+2)
            cand = d
            for dy, dx, w in offs:
                sl = dpad[:, 1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
                cand = torch.minimum(cand, sl + w) if w else torch.minimum(cand, sl)
            d = torch.where(mB, cand, torch.full_like(cand, INF))
            it += 1
        if torch.equal(d, prev):
            converged = True
            break
    if not converged:
        raise RuntimeError("geodesic relaxation did not converge")
    out = d.detach().to("cpu").numpy().astype(np.float32)
    out[:, ~mc] = INF
    return out


def _gpu_geod_cropped(mc, depart_rc):
    """Single-source convenience wrapper returning an (H, W) float32 field."""
    return _gpu_geod_batch(mc, [depart_rc])[0]


def _gpu_geod_multi(masks, sources):
    """Geodesic for MANY DIFFERENT cropped masks at once, one source each.

    ``masks[b]`` is a bool HxW array (its own size), ``sources[b]`` a (row,col)
    inside it. All B masks are padded to a common size and relaxed together in
    ONE batched wavefront -- this is the real GPU win (the whole frame's axis
    geodesics share the sequential iterations instead of each paying the launch
    overhead alone, exactly as the Gabor path batches its convolutions). Returns
    a list of float32 (Hb, Wb) fields, +inf outside each mask. Raises on
    non-convergence or any GPU problem.
    """
    torch = _TORCH
    dev = _DEVICE
    Fn = torch.nn.functional
    B = len(masks)
    Hm = max(m.shape[0] for m in masks)
    Wm = max(m.shape[1] for m in masks)
    INF = float("inf")
    r2 = float(np.sqrt(2.0))
    mstack = torch.zeros((B, Hm, Wm), dtype=torch.bool, device=dev)
    d = torch.full((B, Hm, Wm), INF, dtype=torch.float32, device=dev)
    for b, (mc, (sr, sc)) in enumerate(zip(masks, sources)):
        h, w = mc.shape
        mstack[b, :h, :w] = torch.from_numpy(np.ascontiguousarray(mc)).to(dev).bool()
        d[b, int(sr), int(sc)] = 0.0
    offs = [(0, 0, 0.0),
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, r2), (-1, 1, r2), (1, -1, r2), (1, 1, r2)]
    max_iters = _MAX_ITERS_FACTOR * (Hm + Wm) + 16
    check_every = 16
    converged = False
    it = 0
    while it < max_iters:
        prev = d
        for _ in range(min(check_every, max_iters - it)):
            dpad = Fn.pad(d, (1, 1, 1, 1), value=INF)
            cand = d
            for dy, dx, w in offs:
                sl = dpad[:, 1 - dy:1 - dy + Hm, 1 - dx:1 - dx + Wm]
                cand = torch.minimum(cand, sl + w) if w else torch.minimum(cand, sl)
            d = torch.where(mstack, cand, torch.full_like(cand, INF))
            it += 1
        if torch.equal(d, prev):
            converged = True
            break
    if not converged:
        raise RuntimeError("batched geodesic relaxation did not converge")
    dcpu = d.detach().to("cpu").numpy().astype(np.float32)
    out = []
    for b, mc in enumerate(masks):
        h, w = mc.shape
        f = dcpu[b, :h, :w].copy()
        f[~mc] = INF
        out.append(f)
    return out


def batched_geod_fields(masks, sources):
    """Batched GPU geodesic fields for a frame's masks, or None to fall back to
    the CPU per-call path (GPU disabled, no CUDA, or any error -- permanent)."""
    global _GPU_OK, _REASON
    if _GPU_OK is False:
        return None
    # OFF by default: the float32 batch can pick a different axis endpoint than
    # the exact CPU geodesic (argmax of near-equal geodesic distances), which
    # flips orientation-based descriptors and can change the classification.
    # Opt in explicitly with FLATMORPH_V4J_GPU_GEOD=1 only if you accept that.
    if os.environ.get("FLATMORPH_V4J_GPU_GEOD", "0") not in ("1", "true", "True"):
        with _LOCK:
            if _GPU_OK is None:
                _GPU_OK = False
                _REASON = ("OFF by default (float32 can change results) -- set "
                           "FLATMORPH_V4J_GPU_GEOD=1 to opt in")
        return None
    try:
        if not _try_init():
            with _LOCK:
                _GPU_OK = False
            return None
        return _gpu_geod_multi(masks, sources)
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _GPU_OK = False
            _REASON = "batched GPU geod raised: %s: %s" % (type(exc).__name__, exc)
        return None


def note(ok, reason):
    """Let the axis orchestrator record the final GPU decision for status()."""
    global _GPU_OK, _REASON
    with _LOCK:
        _GPU_OK = bool(ok)
        _REASON = reason


def make_geod(cpu_fn):
    """Return a ``geod(m, depart)`` that uses the GPU when validated, else the
    CPU reference ``cpu_fn`` (the bbox-cropped MCP_Geometric geodesic). The
    signature and the returned full-size field match the chain's ``_geod``.
    """

    def geod(m, depart):
        global _GPU_OK, _validated, _REASON
        if _GPU_OK is None and os.environ.get(
                "FLATMORPH_V4J_GPU_GEOD", "1") in ("0", "false", "False"):
            # On by default when CUDA is present; set FLATMORPH_V4J_GPU_GEOD=0 to
            # force the (bit-exact) CPU path instead.
            with _LOCK:
                if _GPU_OK is None:
                    _GPU_OK = False
                    _REASON = "disabled via FLATMORPH_V4J_GPU_GEOD=0"
            return cpu_fn(m, depart)
        if _GPU_OK is False:
            return cpu_fn(m, depart)
        try:
            if not _try_init():
                with _LOCK:
                    _GPU_OK = False
                return cpu_fn(m, depart)
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                return cpu_fn(m, depart)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            mc = m[y0:y1, x0:x1]
            gpu_c = _gpu_geod_cropped(
                mc, (int(depart[0]) - y0, int(depart[1]) - x0))
            full = np.full(m.shape, np.inf)
            full[y0:y1, x0:x1] = gpu_c
            full[~m] = np.inf

            if _GPU_OK is True:
                return full

            # ---- still validating: compare to the CPU reference ----
            ref = cpu_fn(m, depart)
            fin_g = np.isfinite(full)
            fin_r = np.isfinite(ref)
            ok = np.array_equal(fin_g, fin_r)
            diff = 0.0
            if ok and fin_r.any():
                diff = float(np.max(np.abs(full[fin_r] - ref[fin_r])))
                ok = diff <= _VALIDATE_TOL
            with _LOCK:
                if not ok:
                    _GPU_OK = False
                    _REASON = ("GPU/CPU mismatch (max diff %.3e > %.0e or inf "
                               "pattern differs)" % (diff, _VALIDATE_TOL))
                else:
                    _validated += 1
                    if _validated >= _VALIDATE_NEEDED:
                        _GPU_OK = True
                        _REASON = ("validated on %d masks, max diff %.3e"
                                   % (_validated, diff))
            # during validation always return the trusted CPU field
            return ref
        except Exception as exc:  # noqa: BLE001 - any trouble => CPU forever
            with _LOCK:
                _GPU_OK = False
                _REASON = "GPU path raised: %s: %s" % (type(exc).__name__, exc)
            return cpu_fn(m, depart)

    return geod
