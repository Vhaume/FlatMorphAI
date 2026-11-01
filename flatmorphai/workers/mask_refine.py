# -*- coding: utf-8 -*-
"""Shared worm-mask refinement helpers for the subprocess drivers.

Two optional post-processing steps are offered on a Mask2Former mask, used by
both Pipeline C and the Forecast regeneration states pipeline:

* ``union_blasteme`` grows the mask with the blastema prediction so a pale
  blastema the segmenter cut off is recovered ;
* ``smooth_and_binarize`` rounds jagged edges and drops stray pixels with a
  Gaussian blur followed by a re-binarization.

The module is pure and imports OpenCV lazily, so it is safe to import from a
standalone driver script (its directory is on ``sys.path``).
"""
from __future__ import annotations

import os


def odd_kernel(k) -> int:
    """Return an odd kernel size >= 1 (Gaussian kernels must be odd)."""
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def smooth_and_binarize(mask, kernel):
    """Gaussian-blur a 0/255 mask then threshold back to 0/255.

    Rounds staircase edges and removes isolated stray pixels while keeping the
    mask strictly binary for the downstream steps. A kernel of 1 is a no-op.
    """
    import cv2
    k = odd_kernel(kernel)
    if k <= 1:
        return mask
    blur = cv2.GaussianBlur(mask, (k, k), 0)
    return (blur > 127).astype("uint8") * 255


def _declared_size(processor):
    """The processor's ``size`` dict, including when it is nested one level."""
    for obj in (processor, getattr(processor, "image_processor", None)):
        size = getattr(obj, "size", None) if obj is not None else None
        if isinstance(size, dict) and size:
            return size
    return None


def _measure_fed_hw(img_bgr, processor, log=None):
    """Run the processor on one image to get the real fed (H, W).

    Fed EXACTLY as production does -- a PIL image, like
    ``segment_mask._prob_batch`` -- because that detail decided the whole VRAM
    budget. Passing a raw numpy array made the call raise, the exception was
    swallowed, and the caller silently fell back to a generic
    ``shortest_edge=800`` heuristic. For a 1600x1200 plate that invents
    1088x800 where this checkpoint declares ``{"height": 384, "width": 384}``
    and really feeds 384x384: six times the pixels, hence a per-image estimate
    six times too large and a batch pinned to 1 for no reason.

    A failure is no longer silent: it is logged, and the processor's declared
    size is preferred to any guess.
    """
    if processor is None:
        return None, None
    import numpy as np
    rgb = img_bgr[:, :, ::-1] if getattr(img_bgr, "ndim", 2) == 3 else img_bgr
    rgb = np.ascontiguousarray(rgb)
    last = None
    for build in (lambda a: __import__("PIL.Image", fromlist=["Image"]).fromarray(a),
                  lambda a: a):
        try:
            out = processor(images=[build(rgb)], return_tensors="pt")
            pv = out["pixel_values"]
            shape = tuple(pv.shape)
            if len(shape) >= 2:
                return int(shape[-2]), int(shape[-1])
        except Exception as exc:  # noqa: BLE001
            last = exc
    # No measurement: fall back on what the processor SAYS, never on a guess.
    size = _declared_size(processor)
    if isinstance(size, dict) and "height" in size and "width" in size:
        d = int(getattr(processor, "size_divisor", 32) or 32)
        pad = lambda x: int((x + d - 1) // d) * d  # noqa: E731
        if log is not None:
            log("  Could not run the processor to measure the fed size (%s); "
                "using its declared %dx%d." % (last, size["width"], size["height"]))
        return pad(int(size["height"])), pad(int(size["width"]))
    if log is not None:
        log("  Could not determine the fed resolution (%s); falling back to a "
            "conservative estimate, the batch may be smaller than necessary."
            % (last,))
    return None, None


# Remembers the last VRAM-budget line emitted so it is logged once per distinct
# budget instead of once per segmentation chunk.


def _call_suppressed(seg, imgs, suppress):
    with suppress():
        return seg.segment_batch(imgs)


def preflight_vram(log=None, min_free_mb=1200):
    """Report the GPU budget BEFORE the model is loaded.

    Loading Mask2Former and running one 384x384 forward needs on the order of
    a gigabyte. When the card is already full, every later message is about a
    symptom: the batch search halves down to 1 and fails there, minutes into
    the run. Saying it up front, and naming who holds the memory, turns that
    into one actionable line. Returns the free MiB, or None when unknown.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            if log is not None:
                log("No CUDA device: the segmentation will run on the CPU "
                    "(very slow).")
            return None
        free, total = torch.cuda.mem_get_info()
        mb = 1024 * 1024
        free_mb, total_mb = int(free) // mb, int(total) // mb
        if log is not None:
            log("GPU memory before loading the model: %d MiB free of %d MiB"
                % (free_mb, total_mb))
            if free_mb < int(min_free_mb):
                log("WARNING: only %d MiB are free, which is below the ~%d MiB "
                    "the segmentation model needs. Another program is very "
                    "likely holding this card - a TensorFlow process reserves "
                    "all of it by default. Close it (or set "
                    "TF_FORCE_GPU_ALLOW_GROWTH=1 in that environment) before "
                    "running this pipeline." % (free_mb, int(min_free_mb)))
        return free_mb
    except Exception:  # noqa: BLE001
        return None


def _vram_holder_note():
    """Say WHO holds the card, and what the operator can do about it.

    ``mem_get_info`` reports what the driver has free across all processes,
    ``memory_reserved`` what this process's allocator holds. Comparing the two
    separates three very different situations, which call for three different
    answers -- the previous version collapsed them and could claim another
    program was to blame while this run itself held five sixths of the card.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        reserved = int(torch.cuda.memory_reserved())
        allocated = int(torch.cuda.memory_allocated())
        mb = 1024 * 1024
        elsewhere = max(0, int(total) - int(free) - reserved)
        parts = [" Of the %d MiB card: %d MiB held by this run (%d MiB live), "
                 "%d MiB by other processes, %d MiB free."
                 % (int(total) // mb, reserved // mb, allocated // mb,
                    elsewhere // mb, int(free) // mb)]
        if reserved > elsewhere and reserved > 0:
            parts.append(
                " Most of the card is held by THIS run, so the forward pass "
                "itself is what does not fit: the fed resolution above is the "
                "lever, not the batch size, which is already at one.")
        elif elsewhere > reserved:
            parts.append(
                " Most of the card is held by ANOTHER process, so no batch "
                "size would help; free it first.")
        return "".join(parts)
    except Exception:  # noqa: BLE001
        return ""


_WARNED_BATCH = None


def _forced_batch():
    """Batch pinned by the operator through FLATMORPHAI_SEG_BATCH, or 0."""
    try:
        return max(0, int(os.environ.get("FLATMORPHAI_SEG_BATCH", "0") or 0))
    except Exception:  # noqa: BLE001
        return 0


def _warn_batch_differs(effective, configured, log):
    """Note, once, that the batch was reduced. It does NOT change the masks.

    The processor resizes every image (full frame OR second-pass crop) to a
    fixed square, and the model runs each one independently along the batch
    axis, so the mask of a given image is the same whatever else shares its
    forward. The batch is therefore a memory-and-speed knob only: a reduced
    one is slower but byte-for-byte identical. The earlier, alarming version of
    this message -- "the masks will not match the reference" -- was wrong, and
    is the reason staging and temp files kept being chased for a problem that
    was simply a too-large forward.
    """
    global _WARNED_BATCH
    key = (effective, configured)
    if key == _WARNED_BATCH:
        return
    _WARNED_BATCH = key
    try:
        log("  Segmentation batch reduced to %d (from %d) to fit the free GPU "
            "memory. The processor resizes every image to a fixed size and runs "
            "each independently, so the masks are identical to batch %d -- only "
            "slower. Free GPU memory or set FLATMORPHAI_SEG_BATCH to raise it."
            % (effective, configured, configured))
    except Exception:  # noqa: BLE001
        pass


def _batch_cache_path():
    return os.path.join(os.path.expanduser("~"), ".flatmorphai", "seg_batch")


def load_cached_batch(default):
    """Last segmentation batch that fit this card, or ``default``.

    Lets a run (and every folder after the first) start from a batch that is
    known to fit instead of re-probing the configured maximum and paying a
    doomed 8 -> 4 -> 2 descent -- each failed attempt is a full forward -- at
    every folder. Best-effort; a missing/unreadable cache returns ``default``.
    """
    try:
        with open(_batch_cache_path(), encoding="utf-8") as f:
            v = int(f.read().strip())
        return v if v > 0 else int(default)
    except Exception:  # noqa: BLE001
        return int(default)


def save_cached_batch(value):
    try:
        os.makedirs(os.path.dirname(_batch_cache_path()), exist_ok=True)
        with open(_batch_cache_path(), "w", encoding="utf-8") as f:
            f.write(str(int(value)))
    except Exception:  # noqa: BLE001
        pass


# Adaptive batch state for the whole run (one segmentation subprocess):
# bs = current batch, ceiling = smallest batch that went out of memory in this
# run (never retried), best = largest batch that succeeded (saved for the next
# run as its starting point), told = start message already logged.
_BS = {"bs": None, "ceiling": None, "best": 0, "told": False, "speed": {}}


def max_batch(configured):
    """Upper bound of the adaptive batch: FLATMORPHAI_SEG_BATCH_MAX (default
    32), never below the configured value."""
    try:
        m = int(os.environ.get("FLATMORPHAI_SEG_BATCH_MAX", "32") or 32)
    except ValueError:
        m = 32
    return max(int(configured), m)


def _vram_room_bytes():
    """Bytes a larger forward could still use INSIDE THE PHYSICAL VRAM.

    Under WSL/Windows (WDDM) the driver does not fail when the card is full:
    it spills into "shared GPU memory" (host RAM) and mem_get_info() then
    reports more free memory than the card has (10 GB "free" on a 6 GB GTX
    1060, measured) -- and everything becomes several times slower. So the
    budget is 95 % of the card's physical memory minus what other programs
    use, minus what this process already holds."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        phys = int(torch.cuda.get_device_properties(0).total_memory)
        reserved = int(torch.cuda.memory_reserved())
        others = max(0, min(phys, int(total)) - int(min(free, phys)) - reserved)
        return max(0, int(0.95 * phys) - others - int(torch.cuda.memory_allocated()))
    except Exception:  # noqa: BLE001
        return 0


class _BatchPlan:
    """Decides the Mask2Former batch ONCE per run (no drift between chunks).

    Calibration (first images of the run, real work, masks kept): one untimed
    warm-up forward, then batches of 1, 2, 4, 8, ... while the predicted peak
    (fixed + n x per-image, measured) stays within 75 % of the card's PHYSICAL
    memory minus the other programs measured at the start. The size with the
    best images/second is then FIXED for the whole run and saved per GPU model
    (~/.flatmorphai/seg_batch.json) so later runs skip the calibration.
    """

    def __init__(self, cap, log):
        self.cap, self.log = cap, log
        self.fixed = None
        self.queue = None
        self.res = {}          # n -> (sec per image, need bytes)
        self.warm = False
        self.budget = None
        self.gpu = "?"

    def _store(self):
        return os.path.join(os.path.expanduser("~"), ".flatmorphai", "seg_batch.json")

    def start(self):
        try:
            import json
            import torch
            props = torch.cuda.get_device_properties(0)
            self.gpu = props.name
            phys = int(props.total_memory)
            free, total = torch.cuda.mem_get_info()
            others = max(0, min(phys, int(total)) - int(min(free, phys)) - int(torch.cuda.memory_reserved()))
            self.budget = 0.75 * (phys - others) - torch.cuda.memory_allocated()
            try:
                with open(self._store(), encoding="utf-8") as fh:
                    d = json.load(fh).get(self.gpu)
                if d and int(d.get("batch", 0)) > 0 and float(d.get("need_per_image", 1e30)) * int(d["batch"]) < self.budget:
                    self.fixed = int(d["batch"])
                    if self.log:
                        self.log("  Segmentation batch %d (best measured on this %s, %.3f s/image; "
                                 "delete ~/.flatmorphai/seg_batch.json to re-measure)"
                                 % (self.fixed, self.gpu, float(d.get("sec_per_image", 0))))
                    return
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            self.budget = None
        self.queue = [1, 2]
        if self.log:
            self.log("  Segmentation batch: calibrating on this GPU (1, 2, 4, ... images while they fit "
                     "the card), then the fastest size is kept for the whole run.")

    def size(self):
        if self.fixed:
            return self.fixed
        if not self.warm:
            return 1
        return self.queue[0] if self.queue else 1

    def record(self, n, sec, need):
        if self.fixed:
            return
        if not self.warm:
            self.warm = True            # untimed warm-up (CUDA/cuDNN init)
            return
        if not self.queue or n != self.queue[0]:
            return
        self.queue.pop(0)
        self.res[n] = (sec / max(1, n), float(need))
        if not self.queue:
            ns = sorted(self.res)
            if len(ns) >= 2:
                c = max((self.res[q][1] - self.res[p_][1]) / (q - p_) for p_ in ns for q in ns if q > p_)
                c = max(c, 1.0)
                fx = max(0.0, max(self.res[m][1] - c * m for m in ns))
            else:
                c, fx = self.res[ns[0]][1], 0.0
            nxt = ns[-1] * 2
            fits = self.budget is None or fx + c * nxt <= self.budget
            better = len(ns) < 2 or self.res[ns[-1]][0] < 0.97 * self.res[ns[-2]][0]
            if nxt <= self.cap and fits and better:
                self.queue = [nxt]
                return
            fastest = min(self.res[m][0] for m in ns)
            best = min(m for m in ns if self.res[m][0] <= 1.02 * fastest)   # same speed, less VRAM
            self.fixed = best
            if self.log:
                self.log("  Segmentation batch FIXED to %d for this run: %s (%d MiB per image measured, "
                         "card budget %d MiB)" % (best, ", ".join("%d img -> %.3f s/img" % (m, self.res[m][0]) for m in ns),
                                                  c // 2**20, (self.budget or 0) // 2**20))
            try:
                import json
                d = {}
                try:
                    with open(self._store(), encoding="utf-8") as fh:
                        d = json.load(fh)
                except Exception:  # noqa: BLE001
                    d = {}
                d[self.gpu] = {"batch": best, "sec_per_image": self.res[best][0], "need_per_image": c}
                os.makedirs(os.path.dirname(self._store()), exist_ok=True)
                with open(self._store(), "w", encoding="utf-8") as fh:
                    json.dump(d, fh, indent=1)
            except Exception:  # noqa: BLE001
                pass

    def oom(self, n):
        self.fixed = max(1, n // 2)
        self.queue = None


def segment_oom_safe(seg, imgs, start_batch=None, log=None, suppress=None):
    """Segment a list of BGR images, auto-adapting the sub-batch to the VRAM.

    ``seg.segment_batch`` runs the first Mask2Former pass over its whole input
    at once; for full-size images that is the memory-hungry step. This wrapper
    starts at ``start_batch`` images (the configured batch) and halves the
    sub-batch only on a real CUDA out-of-memory error, emptying the cache
    between tries. No resolution-based estimate is used -- running the batch
    is the only reliable measure of whether it fits. The batch that succeeds
    is kept for the rest of the list and returned so the caller can start the
    next chunk there.

    Returns ``(masks, working_batch)`` with masks in the same order as ``imgs``.
    Raises a clear error if even a single image does not fit.
    """
    if not imgs:
        return [], max(1, int(start_batch) if start_batch else 1)
    masks = []
    i = 0
    configured = int(start_batch) if start_batch else len(imgs)
    # Start at the FULL configured batch and only shrink on a real OOM.
    #
    # There is no analytic pre-cap any more. Three successive attempts to
    # predict the forward's memory from the fed resolution (26 GiB, then
    # 4.5 GiB, then 41 GiB for the same images) were all wrong, and each one
    # forced the batch down -- the last to 1 -- before a single image had been
    # tried. Mask2Former's real footprint is not something to guess: the
    # try/except below measures it for real by running the batch, and halves
    # only if the card actually says no. ``FLATMORPHAI_SEG_BATCH`` still pins a
    # value for anyone who wants to override the configured one.
    forced = _forced_batch()
    cap = max_batch(configured)
    if forced:
        bs = max(1, forced)
        if log is not None and not _BS["told"]:
            log("  Segmentation batch pinned to %d by FLATMORPHAI_SEG_BATCH" % bs)
            _BS["told"] = True
    else:
        if _BS.get("plan") is None:
            _BS["plan"] = _BatchPlan(cap, log)
            _BS["plan"].start()
        plan = _BS["plan"]
        bs = plan.size()
    while i < len(imgs):
        if not forced:
            bs = _BS["plan"].size()
        sub = imgs[i:i + bs]
        if bs <= 1:
            # At one image per forward there is nothing left to trim, so give
            # the driver back whatever the allocator is merely caching before
            # trying: the gap between "reserved" and "live" is often a few
            # hundred MiB, which is exactly the margin missing here.
            _free_vram_mb_after_cleanup()
        try:
            _base = _peak = None
            try:
                import torch
                if torch.cuda.is_available():
                    _base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                _base = None
            import time as _tm
            _t0 = _tm.time()
            if suppress is not None:
                # Hide the model's first-forward-pass debug banner (the
                # "==="/"FIRST FORWARD PASS" print()s from the Export wrapper)
                # without touching the driver's own log protocol.
                with suppress():
                    sub_masks = seg.segment_batch(sub)
            else:
                sub_masks = seg.segment_batch(sub)
            masks.extend(sub_masks)
            i += len(sub)
            if not forced and _base is not None:
                try:
                    import torch
                    _pk_abs = torch.cuda.max_memory_allocated()
                    _phys = int(torch.cuda.get_device_properties(0).total_memory)
                    if _pk_abs > 0.95 * _phys and len(sub) > 1:
                        _BS["plan"].oom(len(sub))
                        if log is not None:
                            log("  Segmentation batch reduced to %d: the forward needed %d MiB of the card's "
                                "%d MiB" % (_BS["plan"].size(), _pk_abs // 2**20, _phys // 2**20))
                    else:
                        _BS["plan"].record(len(sub), _tm.time() - _t0, max(1, _pk_abs - _base))
                    _BS["best"] = max(_BS["best"], len(sub))
                except Exception:  # noqa: BLE001
                    pass
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            free_mb = _free_vram_mb_after_cleanup()
            if bs <= 1:
                h, w = imgs[i].shape[0], imgs[i].shape[1]
                # The model never sees the source resolution: the processor
                # resizes to its own input size, so quoting 1600x1200 here was
                # misleading. Report both.
                fh, fw = _measure_fed_hw(imgs[i], getattr(seg, "processor", None), log=log)
                fed = ("" if fh is None or fw is None
                       else ", fed to the model at %dx%d" % (fw, fh))
                vram = ("" if free_mb is None
                        else " Only %d MiB of GPU memory are free." % free_mb)
                raise RuntimeError(
                    "CUDA out of memory on a SINGLE image (%dx%d%s), with both "
                    "passes already down to one item per forward.%s%s "
                    "Free GPU memory (close other GPU programs; a TensorFlow "
                    "process in this environment reserves all VRAM by default), "
                    "or use a GPU with more memory. Note: a driver-level "
                    "'CUDA driver error: out of memory' corrupts the CUDA "
                    "context, so smaller batches cannot recover within the same "
                    "run." % (w, h, fed, vram, _vram_holder_note())
                )
            # Ceil-halving, so the search settles on the largest batch
            # that fits instead of overshooting below it.
            _BS["ceiling"] = bs if _BS["ceiling"] is None else min(_BS["ceiling"], bs)
            if _BS.get("plan") is not None and not forced:
                _BS["plan"].oom(bs)
                bs = _BS["plan"].size()
            else:
                bs = max(1, (bs + 1) // 2)
            _BS["bs"] = bs
            if log is not None:
                try:
                    extra = "" if free_mb is None else " (%d MiB free)" % free_mb
                    log("  GPU out of memory: reducing segmentation batch to "
                        "%d%s" % (bs, extra))
                    if bs < configured:
                        _warn_batch_differs(bs, configured, log)
                except Exception:  # noqa: BLE001
                    pass
    if not forced:
        save_cached_batch(max(_BS["best"], 1) if _BS["best"] else bs)
    return masks, bs


def _free_vram_mb_after_cleanup():
    """Release cached GPU memory and return the free VRAM in MiB (or None).

    Mirrors what the GUI does when the operator cancels a run or closes the
    window (``MainWindow._release_gpu_resources``): collect first, synchronise
    so no kernel is still writing into a block, empty the allocator cache and
    reclaim any IPC handle. Called between batches, after an out-of-memory
    error, and on a fatal error through :func:`release_gpu_on_crash`.
    """
    try:
        import gc
        import torch
        gc.collect()
        if not torch.cuda.is_available():
            return None
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
        free, _total = torch.cuda.mem_get_info()
        return int(free) // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None


def release_gpu_on_crash(log=None, models=()):
    """Drop the models and empty the VRAM after a fatal error.

    A driver that dies mid-run would otherwise hold its models until the
    process is reaped, and on WSL the parent can observe the card still busy
    for a while. Dropping the references first means ``empty_cache`` actually
    has something to give back, so the next attempt -- often launched straight
    away by the operator -- starts from a clean card instead of inheriting the
    footprint of the run that just failed.
    """
    try:
        for m in models:
            try:
                if hasattr(m, "to"):
                    m.to("cpu")
            except Exception:  # noqa: BLE001
                pass
        free = _free_vram_mb_after_cleanup()
        if log is not None:
            log("GPU memory released after the error%s"
                % ("" if free is None else " (%d MiB free)" % free))
        return free
    except Exception:  # noqa: BLE001
        return None


def park_models(objects, log=None):
    """Move models off the GPU and release their blocks. Returns a restore list.

    Mask2Former, the blastema U-Net and the head models used to sit on the card
    at the same time while the folder was processed image by image, so the peak
    was the sum of all three plus the activations of whichever was running.
    Staging the work means only one family is needed at a time; parking the
    others on the host gives their memory back for the duration of the stage.

    ``objects`` is any mix of modules, dicts of modules and torch modules; the
    ones that expose ``.to`` are moved, the rest are ignored. The returned list
    is passed to :func:`unpark_models` to put them back.
    """
    moved = []
    try:
        import torch
        if not torch.cuda.is_available():
            return moved
    except Exception:  # noqa: BLE001
        return moved
    for m in _iter_torch_modules(objects):
        try:
            dev = next(m.parameters()).device
        except Exception:  # noqa: BLE001
            continue
        if getattr(dev, "type", "") != "cuda":
            continue
        try:
            m.to("cpu")
            moved.append(m)
        except Exception:  # noqa: BLE001
            pass
    free = _free_vram_mb_after_cleanup() if moved else None
    if log is not None:
        if moved:
            log("  %d model(s) parked on the host%s"
                % (len(moved), "" if free is None else ", %d MiB free" % free))
        else:
            # Zero parked is normal at the start of a stage: the models load
            # lazily on their first forward, so there is simply nothing on the
            # card to move yet. Not an error, and not the cause of an OOM.
            log("  (nothing to park yet - models load on first use)")
    return moved


def unpark_models(moved, log=None):
    """Put back on the GPU the models :func:`park_models` moved off it."""
    if not moved:
        return
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return
    for m in moved:
        try:
            m.to(dev)
        except Exception:  # noqa: BLE001
            pass


def _is_torch_module(obj):
    """True for a torch.nn.Module, WITHOUT touching the object's attributes.

    ``hasattr`` is not safe here: it runs ``__getattr__``, and a module
    namespace contains things like ``ctypes.cdll`` whose ``__getattr__``
    tries to ``dlopen`` the attribute name -- asking it for ``parameters``
    raised ``OSError: parameters: cannot open shared object file``. Reading
    the type's MRO answers the question without calling anything.
    """
    try:
        for cls in type(obj).__mro__:
            if cls.__name__ == "Module" and cls.__module__.startswith("torch.nn"):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _is_project_module(mod):
    """True for a module defined by this project, not by a library.

    Bounds the search: the model caches live in the Export / flatmorphai
    modules, and descending into numpy, torch or ctypes is both pointless and,
    as the ctypes crash showed, unsafe.
    """
    try:
        f = getattr(mod, "__file__", None) or ""
    except Exception:  # noqa: BLE001
        return False
    if not f:
        return False
    f = f.replace("\\", "/")
    if "/site-packages/" in f or "/dist-packages/" in f:
        return False
    return "/lib/python" not in f


def _iter_torch_modules(objects, _depth=0, _seen=None):
    """Yield every torch module reachable from modules, dicts, lists or models.

    Written after the first version found nothing: it looked for a fixed set of
    attribute names on the module it was handed, but ``blastema_detection``
    holds no model at all -- it calls ``anchor_cascade.load_models()`` and the
    networks live in ``anchor_cascade._MDL``. The search now follows the
    modules a module imported, but only this project's own modules, and it
    never probes an object with ``hasattr``.
    """
    if _depth > 3 or objects is None:
        return
    import types as _types
    if _seen is None:
        _seen = set()
    items = objects if isinstance(objects, (list, tuple, set)) else [objects]
    for obj in items:
        if obj is None:
            continue
        try:
            key = id(obj)
        except Exception:  # noqa: BLE001
            continue
        if key in _seen:
            continue
        _seen.add(key)
        if _is_torch_module(obj):
            yield obj
            continue
        if isinstance(obj, dict):
            for v in list(obj.values()):
                yield from _iter_torch_modules(v, _depth + 1, _seen)
            continue
        if isinstance(obj, (list, tuple, set)):
            for v in list(obj):
                yield from _iter_torch_modules(v, _depth + 1, _seen)
            continue
        if isinstance(obj, _types.ModuleType):
            if _depth and not _is_project_module(obj):
                continue
            try:
                namespace = list(vars(obj).items())
            except Exception:  # noqa: BLE001
                continue
            for name, value in namespace:
                if name.startswith("__"):
                    continue
                if isinstance(value, _types.ModuleType):
                    yield from _iter_torch_modules(value, _depth + 1, _seen)
                elif isinstance(value, (dict, list, tuple, set)) or _is_torch_module(value):
                    yield from _iter_torch_modules(value, _depth + 1, _seen)
            continue


def vram_report(label, log=None):
    """Log what the card holds right now. Returns the free MiB, or None.

    A stage boundary is the only place where the question "who is holding the
    memory" has a clean answer, and the previous rounds were lost guessing at
    it. Reserved is this process's allocator, allocated is what is live inside
    it, and the difference with the driver's free memory is everyone else.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        reserved = int(torch.cuda.memory_reserved())
        allocated = int(torch.cuda.memory_allocated())
        mb = 1024 * 1024
        elsewhere = max(0, int(total) - int(free) - reserved)
        if log is not None:
            log("  [VRAM] %-22s free %5d | this run reserved %5d (live %5d) | "
                "others %5d | total %5d MiB"
                % (label, int(free) // mb, reserved // mb, allocated // mb,
                   elsewhere // mb, int(total) // mb))
        return int(free) // mb
    except Exception:  # noqa: BLE001
        return None


def configure_fed_size(seg, sample_bgr, target=None, log=None):
    """Make the segmenter feed the model at its TRAINING resolution.

    The checkpoint was trained at 320x320, but ``segment_mask`` feeds pass 1
    the WHOLE image and the HuggingFace processor (shortest-edge 800) resizes a
    1600x1200 plate to ~1088x800 -- about nine times the pixels the model saw.
    That single forward is what overflows a 6 GiB card no matter how small the
    batch is. This pins ``processor.size`` to a fixed ``target`` square so every
    image is resized to the training input before the forward.

    It is deliberately NEVER silent: every branch logs, so a run that still
    OOMs tells us exactly what the processor is doing. Returns the effective
    fed (H, W), best-effort.
    """
    _l = log if log is not None else (lambda _m: None)
    # ONE policy for every page (stereotype, image segmentation, eyespots,
    # regeneration): feed the model at its 320x320 training input. The
    # Mask2Former checkpoint was trained on 320x320 square crops, so both passes
    # run at 320 -- pass 1 resizes the whole image to 320 (coarse localisation),
    # pass 2 the oriented crop to 320 (detail). The whole plate is NEVER fed at
    # the processor's 800 fallback: that is off-distribution AND the single
    # forward that overflows the card. Callers pass target=320 so the pages are
    # byte-identical to the validated production run. FLATMORPHAI_SEG_FED is the
    # power-user override (a pixel size to pin; 0 to keep the processor default).
    env = os.environ.get("FLATMORPHAI_SEG_FED", "").strip()
    if env == "0":
        _l("  Fed resolution: pinning disabled (FLATMORPHAI_SEG_FED=0); "
           "processor default kept.")
        return None
    if env.isdigit() and int(env) > 0:
        target = int(env)
    if not target or int(target) <= 0:
        _l("  Fed resolution: processor default kept (full detail).")
        return None
    target = int(target)

    processor = getattr(seg, "processor", None)
    if processor is None:
        _l("  Fed resolution: segmenter has no 'processor' attribute - cannot "
           "pin the input size; the model keeps its default (likely large) "
           "input, which is memory-heavy.")
        return None

    # Pin a fixed square matching the training input. We override the
    # processor's input size to the training size regardless of what it would
    # feed otherwise, so there is no point measuring that (larger) size first.
    try:
        processor.size = {"height": target, "width": target}
    except Exception as exc:  # noqa: BLE001
        _l("  Fed resolution: could NOT set processor.size (%s); the model "
           "keeps its default input." % exc)
        return None
    # Make sure resize is on and padding will not re-inflate the size.
    for attr, val in (("do_resize", True), ("do_pad", False)):
        if hasattr(processor, attr):
            try:
                setattr(processor, attr, val)
            except Exception:  # noqa: BLE001
                pass

    # Confirm the pin took effect (a few processor types ignore processor.size).
    after = (target, target)
    try:
        measured = _measure_fed_hw(sample_bgr, processor)
        if measured and measured[0]:
            after = measured
    except Exception:  # noqa: BLE001
        pass
    if after[0] <= target and after[1] <= target:
        _l("  Fed resolution pinned to the model's %dx%d training input."
           % (target, target))
    else:
        _l("  Fed resolution: asked for %dx%d but the processor still feeds "
           "%dx%d -- setting processor.size had no effect on this processor "
           "type." % (target, target, after[1], after[0]))
    return after


def decode_prefetch(keys, chunk_size, decode_one, interrupted=None, workers=2):
    """Yield ``(chunk_keys, decoded)`` with the NEXT chunk decoded in advance.

    Image decoding is disk + libjpeg work that releases the GIL, so it overlaps
    with the GPU forward of the current chunk on a worker thread instead of
    stalling the card between batches. One chunk is held ahead, so the extra
    host memory is a single chunk of decoded frames, nothing on the GPU.

    ``decode_one(key)`` returns the decoded object (or None); ``decoded`` keeps
    the same order as ``chunk_keys`` with None for anything that failed. The
    chunking itself is unchanged -- same ``chunk_size``, same order -- so the
    model still sees exactly the groups it did before.
    """
    from concurrent.futures import ThreadPoolExecutor

    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
    if not chunks:
        return

    def _load(chunk):
        w = max(1, int(workers))
        if w == 1 or len(chunk) <= 1:
            return [decode_one(k) for k in chunk]
        with ThreadPoolExecutor(max_workers=min(w, len(chunk))) as inner:
            return list(inner.map(decode_one, chunk))

    with ThreadPoolExecutor(max_workers=1) as pump:
        ahead = pump.submit(_load, chunks[0])
        for idx, chunk in enumerate(chunks):
            decoded = ahead.result()
            nxt = chunks[idx + 1] if idx + 1 < len(chunks) else None
            # Kick off the next chunk's decode BEFORE yielding, so it runs
            # while the caller has the GPU.
            ahead = pump.submit(_load, nxt) if nxt is not None else None
            if interrupted is not None and interrupted():
                return
            yield chunk, decoded


def keep_largest_component(mask):
    """Keep only the largest connected component of a 0/255 binary mask.

    Drops smaller spurious detections (e.g. a reflection). On any failure the
    input mask is returned unchanged.
    """
    import cv2
    import numpy as np
    try:
        if mask is None:
            return mask
        m = mask
        if getattr(m, "ndim", 2) == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        _th, bw = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        n, labels, stats, _c = cv2.connectedComponentsWithStats(bw, 8)
        if n <= 2:  # background + at most one object
            return bw
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(areas))
        return np.where(labels == biggest, 255, 0).astype(np.uint8)
    except Exception:  # noqa: BLE001
        return mask


def _weld_gaps(b, image, thr, close_r=5, marge=100, half=True, min_area=12):
    """Fill a gap in the mask only when the IMAGE says it is tissue.

    A morphological closing does not know what it fills: it plugs a notch
    because the notch is narrow, whether that notch is missing tissue or the
    plate seen between two arcs of a coiled worm. On the 180 stereotype masks
    a closing of radius 5 proposes 28 such bridges; exactly two of them are
    missing tissue and the other 26 are real holes, so the closing alone does
    more harm than good.

    What tells them apart is a figure/ground decision, and it is taken here by
    KNN matting (Chen, Li & Tang, "KNN Matting", CVPR 2012 / IEEE TPAMI 2013,
    DOI 10.1109/TPAMI.2013.18), as implemented by the ``pymatting`` package.
    Its affinities come from the k nearest neighbours in a colour + position
    feature space rather than from the pixel lattice, which is exactly what
    this problem needs: the holes are ENCLOSED by the worm, so every method
    that propagates along the lattice (random walker, watershed, closed-form
    matting) has to cross the body to reach a background seed and answers
    "foreground" for all of them. Measured against the operator's own
    judgement on the 13 largest bridges: KNN matting 0 errors, GrabCut 3,
    CascadePSP 6, random walker 11.

    Over the whole set the alpha of a bridge is 0.99 and 1.00 for the two to
    weld and at most 0.76 for the 26 others, so the threshold sits in an empty
    interval. Cost: 0.11 s per bridge at half resolution, about 3 s for a
    180-image run, since only the images that have a bridge are touched.

    ``image`` is the original frame (BGR or RGB, it makes no difference: the
    three channels are only used as a feature space). Returns ``b`` unchanged
    whenever anything is missing or fails -- a refinement must never break a
    run, and never invent tissue on a doubt.
    """
    import cv2
    import numpy as np
    try:
        from pymatting import estimate_alpha_knn
    except Exception:  # noqa: BLE001 - package absent: silently do nothing
        return b
    try:
        img = np.asarray(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim != 3 or img.shape[:2] != b.shape:
            return b
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_r + 1, 2 * close_r + 1))
        add = ((cv2.morphologyEx(b * 255, cv2.MORPH_CLOSE, k) > 0)
               & (b == 0)).astype(np.uint8)
        if not add.any():
            return b
        n, lab, st, _c = cv2.connectedComponentsWithStats(add, 8)
        comps = [i for i in range(1, n)
                 if int(st[i, cv2.CC_STAT_AREA]) >= min_area]
        if not comps:
            return b
        sur = cv2.erode(b, np.ones((7, 7), np.uint8))
        far = cv2.dilate(b, np.ones((41, 41), np.uint8)) == 0
        out = b.copy()
        for i in comps:
            x = int(st[i, cv2.CC_STAT_LEFT]); y = int(st[i, cv2.CC_STAT_TOP])
            w = int(st[i, cv2.CC_STAT_WIDTH]); h = int(st[i, cv2.CC_STAT_HEIGHT])
            # The vignette is cut around the BRIDGE, not around the worm: the
            # cost is proportional to the pixel count. But the method draws its
            # strength from non-locality, so it needs sure background inside
            # the vignette; when there is too little, the vignette grows.
            mg = marge
            while True:
                y0 = max(0, y - mg); y1 = min(b.shape[0], y + h + mg)
                x0 = max(0, x - mg); x1 = min(b.shape[1], x + w + mg)
                bg = far[y0:y1, x0:x1]
                if bg.sum() > 200 or mg > 400:
                    break
                mg = int(mg * 1.6)
            if bg.sum() <= 200:
                continue
            rgb = img[y0:y1, x0:x1].astype(np.float64) / 255.0
            tri = np.full(bg.shape, 0.5)
            tri[sur[y0:y1, x0:x1] > 0] = 1.0
            tri[bg] = 0.0
            if half and min(tri.shape) >= 24:
                h2, w2 = tri.shape[0] // 2, tri.shape[1] // 2
                r2 = np.clip(cv2.resize(rgb, (w2, h2),
                                        interpolation=cv2.INTER_AREA), 0.0, 1.0)
                t2 = cv2.resize(tri, (w2, h2), interpolation=cv2.INTER_NEAREST)
                t2[(t2 > 0.2) & (t2 < 0.8)] = 0.5
                al = cv2.resize(np.clip(estimate_alpha_knn(r2, t2), 0.0, 1.0),
                                (tri.shape[1], tri.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            else:
                al = np.clip(estimate_alpha_knn(np.clip(rgb, 0.0, 1.0), tri),
                             0.0, 1.0)
            if float(al[lab[y0:y1, x0:x1] == i].mean()) >= thr:
                out[lab == i] = 1
        return out
    except Exception:  # noqa: BLE001
        return b


def clean_seg_mask(mask, image=None):
    """Opt-in cleanup of a Mask2Former worm mask, shared by every segmentation
    page (Stereotype classification, Image segmentation, Eyespots, Regeneration)
    so they all run the SAME pipeline. Each step is OFF unless its environment
    variable is set, so the validated masks are unchanged by default:

    * FLATMORPHAI_SEG_FILLHOLES = 1 (or a max hole area in px): fill interior
      holes -- a hole fully enclosed by the worm is always a segmentation gap.
    * FLATMORPHAI_SEG_KEEPMAIN = <radius_px>: cut the connections thinner than
      the radius, then keep the single largest object -- and give it back its
      exact original outline. Three steps: open at the radius (every part
      thinner than it disappears, so a stub hanging by a thread is severed
      from the body), keep the largest piece of that opening, then intersect
      the ORIGINAL mask with that piece grown back by the same radius. The
      body therefore keeps its contour pixel for pixel; only what was hanging
      by a thin connection, or lying detached, is dropped. Nothing is removed
      when the opening leaves nothing (a worm thinner than the radius
      everywhere) or when more than a third of the mask would go: those are
      the signs of a radius too large for that image, and a cleanup must
      never eat the subject.
    * FLATMORPHAI_SEG_PRUNE = <radius_px>: remove ONLY thin TERMINAL protrusions
      (a spur) while keeping the body pixel-exact AND keeping bridges (a thin
      part joining two body masses). The spur is ``b AND NOT dilate(open(b,r),r)``
      subtracted from the original mask, with a connectivity guard that never
      drops a piece whose removal would split the worm.
    * FLATMORPHAI_SEG_CLOSE = <radius_px>: closing -- thickens an under-segmented
      thin part and fills small gaps (the opposite of opening).
    * FLATMORPHAI_SEG_OPEN = <radius_px>: plain opening, for manual tests.
    * FLATMORPHAI_SEG_WELD = <threshold x 100>: weld a gap ONLY when the image
      says it is tissue -- see :func:`_weld_gaps`. Needs ``image``; without it
      the step does nothing. FLATMORPHAI_SEG_WELD_FULL=1 runs the matting at
      full resolution (wider decision margin, about four times slower).

    Applied in that order, the weld last because it reads the final mask.
    Never returns an empty mask.
    """
    if mask is None:
        return mask
    import cv2
    import numpy as np

    def _int(name):
        try:
            return int((os.environ.get(name, "") or "0").strip() or "0")
        except ValueError:
            return 0

    fill = (os.environ.get("FLATMORPHAI_SEG_FILLHOLES", "") or "").strip()
    keep_r = _int("FLATMORPHAI_SEG_KEEPMAIN")
    prune_r = _int("FLATMORPHAI_SEG_PRUNE")
    close_r = _int("FLATMORPHAI_SEG_CLOSE")
    open_r = _int("FLATMORPHAI_SEG_OPEN")
    weld_t = _int("FLATMORPHAI_SEG_WELD") / 100.0
    weld = weld_t > 0 and image is not None
    if not (fill or keep_r > 0 or prune_r > 0 or close_r > 0 or open_r > 0
            or weld):
        return mask
    try:
        b = (np.asarray(mask) > 127).astype(np.uint8)
        if b.ndim != 2:
            return mask
        h, w = b.shape

        if fill:
            ff = b.copy()
            m2 = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, m2, (0, 0), 1)
            holes = ((ff == 0) & (b == 0)).astype(np.uint8)
            if fill not in ("1", "all", "yes", "true", "on"):
                cap = _int("FLATMORPHAI_SEG_FILLHOLES")
                if cap > 0:
                    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 8)
                    keep = np.zeros_like(holes)
                    for i in range(1, n):
                        if st[i, cv2.CC_STAT_AREA] <= cap:
                            keep[lab == i] = 1
                    holes = keep
            b = (b | holes).astype(np.uint8)

        if keep_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * keep_r + 1, 2 * keep_r + 1))
            opened = (cv2.morphologyEx(b * 255, cv2.MORPH_OPEN, k) > 0).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(opened, 8)
            if n > 1:
                main = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                grown = (cv2.dilate(main * 255, k) > 0).astype(np.uint8)
                cand = (b & grown).astype(np.uint8)
                # Guard: keep the result only if it still holds the bulk of the
                # worm. An empty or gutted mask means the radius is too large
                # for this image, and then doing nothing is the right answer.
                if cand.sum() >= 0.66 * b.sum():
                    removed = (b & ~cand).astype(np.uint8)
                    b = cand
                    # Smooth the SCAR, and only the scar. Cutting an appendage
                    # off leaves a notch where it was attached -- a few pixels
                    # of staircase that are still, visually, a growth. An
                    # opening-closing erases it, but applied to the whole mask
                    # it also rounds every other contour: measured, a radius of
                    # 2 changes 148 of the 180 stereotype masks for nothing.
                    # So it is applied only within a few pixels of what was
                    # just removed: a mask with no appendage comes out
                    # untouched, bit for bit.
                    if removed.any():
                        ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        # The zone is wider than the smoothing kernel, so the
                        # smoothing is not clipped halfway through the notch;
                        # measured, it converges from about 8 px outwards.
                        kz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                        zone = (cv2.dilate(removed * 255, kz) > 0)
                        sm = (cv2.morphologyEx(b * 255, cv2.MORPH_OPEN, ks) > 0).astype(np.uint8)
                        sm = (cv2.morphologyEx(sm * 255, cv2.MORPH_CLOSE, ks) > 0).astype(np.uint8)
                        b = np.where(zone, sm, b).astype(np.uint8)

        if prune_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * prune_r + 1, 2 * prune_r + 1))
            core = (cv2.morphologyEx(b * 255, cv2.MORPH_OPEN, k) > 0).astype(np.uint8)
            grown = (cv2.dilate(core * 255, k) > 0).astype(np.uint8)
            spur = (b & ~grown).astype(np.uint8)
            if spur.any():
                n, lab, _st, _ = cv2.connectedComponentsWithStats(spur, 8)
                out = b.copy()
                for i in range(1, n):
                    trial = out.copy()
                    trial[lab == i] = 0
                    if trial.sum() > 0 and \
                            cv2.connectedComponents(trial, 8)[0] <= \
                            cv2.connectedComponents(out, 8)[0]:
                        out = trial
                b = out

        if close_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * close_r + 1, 2 * close_r + 1))
            b = (cv2.morphologyEx(b * 255, cv2.MORPH_CLOSE, k) > 0).astype(np.uint8)

        if open_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * open_r + 1, 2 * open_r + 1))
            b = (cv2.morphologyEx(b * 255, cv2.MORPH_OPEN, k) > 0).astype(np.uint8)

        if weld:
            b = _weld_gaps(
                b, image, weld_t,
                half=not (os.environ.get("FLATMORPHAI_SEG_WELD_FULL", "")
                          or "").strip() in ("1", "true", "True"),
            )

        if b.sum() == 0:
            return mask
        return (b * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001 - cleanup must never break a run
        return mask


def union_blasteme(bl_mod, bgr, mask, keep_biggest=False):
    """Return the union of a Mask2Former mask with the blastema prediction.

    ``bl_mod`` is the Export ``blastema_detection`` module; ``bgr`` the original
    image and ``mask`` its 0/255 Mask2Former mask. When ``keep_biggest`` is
    true, only the largest connected component of the blastema mask is kept
    before the union. On any failure the original mask is returned unchanged.
    """
    import cv2
    try:
        res = bl_mod.detect_blastema(bgr, mask)
    except Exception:  # noqa: BLE001
        return mask
    if not res.get("ok"):
        return mask
    bmask = res.get("blastema_mask_original")
    if bmask is None:
        return mask
    if keep_biggest:
        bmask = keep_largest_component(bmask)
    return cv2.bitwise_or(mask, bmask)


def tail_from_mask(mask, head_xy):
    """Point of the worm furthest from the head, FOLLOWING THE BODY.

    ``head_position.csv`` records three points, all of them about the head:
    the tip, the centre of the head framing 0.12 of the worm's length away,
    and the head point itself. None of them is the tail, so it is taken from
    the body: a breadth-first walk inside the mask, starting at the head, and
    the last pixel reached is the tail.

    The walk follows the body rather than straight lines on purpose. On a
    worm curled into a C, the pixel furthest from the head AS THE CROW FLIES
    can be on the near side of the curve, while the tail is further along the
    body; a geodesic walk cannot make that mistake.

    ``mask`` is a 0/255 or boolean array, ``head_xy`` an (x, y) pair in the
    same coordinates. Returns (x, y) as floats, or None when the mask is
    empty or the head falls outside it.
    """
    import numpy as np
    try:
        b = (np.asarray(mask) > 127) if np.asarray(mask).dtype != bool else np.asarray(mask)
        if b.ndim != 2 or not b.any() or head_xy is None:
            return None
        h, w = b.shape
        hx, hy = int(round(float(head_xy[0]))), int(round(float(head_xy[1])))
        hx, hy = max(0, min(w - 1, hx)), max(0, min(h - 1, hy))
        if not b[hy, hx]:
            # The head sits just outside the mask (it is the tip of a pointed
            # end, and the mask has been cleaned): start from the nearest
            # body pixel instead of giving up.
            ys, xs = np.nonzero(b)
            k = int(np.argmin((xs - hx) ** 2 + (ys - hy) ** 2))
            hx, hy = int(xs[k]), int(ys[k])
        # The walk runs on the bounding box of the worm, not on the whole
        # plate: a 1920x1080 frame holds a worm of a few hundred pixels, and
        # one pass over the full image per step of the walk is what made a
        # first, straightforward version take 0.4 s per image.
        import cv2
        ys0, xs0 = np.nonzero(b)
        x0, x1 = int(xs0.min()), int(xs0.max()) + 1
        y0, y1 = int(ys0.min()), int(ys0.max()) + 1
        sub = b[y0:y1, x0:x1].astype(np.uint8)
        sx, sy = hx - x0, hy - y0
        if not (0 <= sx < sub.shape[1] and 0 <= sy < sub.shape[0]):
            return None
        # Full resolution. Running the walk on a half-size mask was tried --
        # four times cheaper -- but it moved the answer by up to 13 pixels on
        # the 52 reference images: the tail is blunt, several pixels are
        # reached on the last step, and halving changes which. 93 ms per
        # image, once per image, is not worth a point that moves.
        k3 = np.ones((3, 3), np.uint8)
        seen = np.zeros_like(sub)
        cur = np.zeros_like(sub)
        cur[sy, sx] = 1
        seen[sy, sx] = 1
        last = cur
        while True:
            nxt = cv2.dilate(cur, k3) & sub & (1 - seen)
            if not nxt.any():
                break
            seen |= nxt
            last = nxt
            cur = nxt
        # Among the pixels reached last, the one furthest from the head in a
        # straight line, so the answer does not depend on the scan order.
        ys, xs = np.nonzero(last)
        k = int(np.argmax((xs - sx) ** 2 + (ys - sy) ** 2))
        return (float(xs[k] + x0), float(ys[k] + y0))
    except Exception:  # noqa: BLE001 - never break a run for a tail point
        return None
