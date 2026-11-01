# -*- coding: utf-8 -*-
"""Inference fast path for the U-Nets of the chain, with identical outputs.

Every blastema and eye inference of the chain goes through ONE function,
``ancrage_cascade._pred(model, crop)``: the refinement calls it twice (one
thumbnail per end), the head cascade two to four times, and the descriptor
block twice more -- about eight single-crop forward passes per worm, each with
its own host-to-device copy, kernel launch and device-to-host synchronisation.
At batch 1 on a 384 x 384 crop the card spends most of that time idle.

This module replaces that function, WITHOUT touching the production files, by a
wrapper that keeps the numerical result bit-for-bit:

* ``torch.inference_mode()`` instead of ``torch.no_grad()``. Same arithmetic,
  but the autograd bookkeeping (version counters, view tracking) is skipped
  entirely rather than merely disabled.
* a content-addressed cache. The models are in ``eval`` mode with no dropout
  and no batch-norm update, so the mapping crop -> probability map is a pure
  function: submitting the same crop twice can only give the same array back.
  The chain does resubmit crops -- the eye thumbnail of the cascade and the one
  of the descriptor block share the same framing whenever the cascade returns
  the tip it started from -- and a hit saves a whole forward pass for the cost
  of one hash.

What is deliberately NOT done: TF32, float16 and channels-last. Each of them
changes the arithmetic, and the descriptors computed from these probability
maps feed the 86 inputs of the state classifier, so a different rounding would
silently move the predicted state.

``cudnn.benchmark`` is OFF by default too, for a different reason: autotuning
tries every algorithm for a shape, and the large workspaces it allocates while
probing are a classic source of "CUDA error: out of memory" on a card that is
already tight -- exactly the failure this chain hits after a few hundred worms.
Set ``FLATMORPHAI_CUDNN_BENCHMARK=1`` to turn it on where VRAM is plentiful.
"""

from __future__ import annotations

import hashlib
import os
import threading

import numpy as np
import torch

from ._env import AC

#: Bound on the cache: eight entries cover one worm's calls with room to spare,
#: and a probability map is 384 x 384 float32, about 590 kB.
_MAX_ENTRIES = 16

_lock = threading.Lock()
_cache: "dict[str, np.ndarray]" = {}
_order: "list[str]" = []
_stats = {"calls": 0, "hits": 0}
_installed = False


def _key(model, crop) -> str:
    """Identity of (model, crop): the model object and the exact crop bytes."""
    a = np.ascontiguousarray(crop)
    h = hashlib.blake2b(a.tobytes(), digest_size=16)
    h.update(repr((a.shape, a.dtype.str, id(model))).encode())
    return h.hexdigest()


def install() -> bool:
    """Patch ``ancrage_cascade._pred`` once. Returns True the first time.

    Every caller reaches the function through the module globals (``_pred`` in
    ancrage_cascade itself, ``AC._pred`` everywhere else), so replacing the
    attribute is enough to cover the refinement, the cascade and the
    descriptors at the same time.
    """
    global _installed
    if _installed:
        return False
    original = AC._pred

    def _pred_fast(mdl, crop):
        _stats["calls"] += 1
        k = _key(mdl, crop)
        with _lock:
            hit = _cache.get(k)
        if hit is not None:
            _stats["hits"] += 1
            return hit
        with torch.inference_mode():
            out = mdl(AC._chans(crop)[None].to(AC.DEVICE)).cpu()[0, 0].numpy()
        with _lock:
            _cache[k] = out
            _order.append(k)
            while len(_order) > _MAX_ENTRIES:
                _cache.pop(_order.pop(0), None)
        return out

    _pred_fast._original = original           # kept for the equivalence test
    AC._pred = _pred_fast
    if os.environ.get("FLATMORPHAI_CUDNN_BENCHMARK", "0") == "1":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:  # noqa: BLE001
            pass
    _install_maxpool_gpu()
    try:   # CPU side: exact Numba twin + per-image memo of the repeated helpers
        from .regen_numba import install as _install_cpu
        _install_cpu()
    except Exception:  # noqa: BLE001 - the chain runs unchanged without it
        pass
    _installed = True
    return True


def _install_maxpool_gpu() -> None:
    """Run the peak-detection max-pool on the GPU, with bit-identical results.

    The seven peak sites of the chain (ancrage_cascade._pics, the two cascade
    _pics_tous, enrich_metrics pics/pupille/yeux, the composite/eye builders)
    all do ``F.max_pool2d(torch.from_numpy(hm)[None, None], 2*MINDIST+1, 1,
    MINDIST)`` -- a 15x15 stride-1 pool on a 384x384 map, ON THE CPU. The
    profile puts this at about a third of the whole chain, shared by the
    cascade and the descriptor block.

    A max-pool is a pure selection of the maximum (no sum, no rounding), so the
    value it returns is EXACTLY the same on the GPU as on the CPU -- verified
    bit-for-bit on the 52 benchmark images (0 descriptor changed, x1.4 faster).
    We therefore replace ``torch.nn.functional.max_pool2d`` by a wrapper that
    moves a CPU input to the device and back, and leaves a tensor that is
    already on the GPU (the U-Net internals) strictly untouched. Every peak site
    reaches the function through the module attribute, so one patch covers them
    all. Opt out with FLATMORPHAI_MAXPOOL_GPU=0.
    """
    if os.environ.get("FLATMORPHAI_MAXPOOL_GPU", "1") != "1":
        return
    if not torch.cuda.is_available():
        return
    import torch.nn.functional as _NNF
    _orig = _NNF.max_pool2d
    _dev = AC.DEVICE

    def _max_pool2d_gpu(x, *args, **kwargs):
        if getattr(x, "is_cuda", False):            # U-Net internals: unchanged
            return _orig(x, *args, **kwargs)
        return _orig(x.to(_dev), *args, **kwargs).cpu()   # CPU peak map -> GPU, exact

    _max_pool2d_gpu._original = _orig
    _NNF.max_pool2d = _max_pool2d_gpu


def stats() -> "dict[str, int]":
    """Calls and cache hits since the process started."""
    return dict(_stats)


def clear() -> None:
    """Drop the cached maps (between folders, to bound memory)."""
    with _lock:
        _cache.clear()
        _order.clear()
