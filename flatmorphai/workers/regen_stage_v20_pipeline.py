# -*- coding: utf-8 -*-
"""Driver for the UPGRADED regeneration chain (Export_regeneration_stage_v20).

Runs in a dedicated subprocess, like the previous regeneration driver, and
streams the same stdout protocol (STEP / PROGRESS / LOG / DONE / ERROR) so the
page and its timeline are unchanged. Instead of the old Export_head_detection
cascade it drives the self-contained ``regen_pipeline.RegenerationPipeline`` of
the v20 bundle (skan2 skeleton axis, head cascade v3c13 with the border-radius
tip correction, blastema refinement, and the two-branch state classifier that
reads the 86 selected descriptors), vendored locally by
``flatmorphai.regen_stage_bundle``.

Outputs written next to each ``images/`` folder, identical in name and columns
to the previous driver so the regeneration editor keeps working:
  * ``regeneration_states.csv``  (required by the editor) - decimal and rounded
    state from the two-branch classifier, planaria index, Day, monotone
    correction;
  * ``head_position.csv``        (optional overlay) - head point of the cascade;
  * ``eyespots_detection.csv``   (optional overlay) - eyespot count from the new
    eye detector (``eye_count`` descriptor), with the monotone correction.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
import sys
import time
import traceback

# PyTorch allocator. ``expandable_segments:True`` reduces fragmentation and is
# what this project's production scripts use, but it routes allocations through
# the CUDA virtual-memory API, which is where a failure surfaces as the
# driver-level "CUDA driver error: out of memory" instead of the allocator's
# recoverable "CUDA out of memory". That path has been seen to fail under WSL,
# so it is NOT forced here: set FLATMORPHAI_CUDA_ALLOC_CONF (for example
# "expandable_segments:True", or "max_split_size_mb:128") to choose it, and the
# plain PyTorch default applies otherwise.
_alloc_conf = os.environ.get("FLATMORPHAI_CUDA_ALLOC_CONF", "").strip()
if _alloc_conf:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc_conf)

import numpy as np

# The driver runs as a script from this folder; make the sibling helpers
# importable the same way eyespots_counter_pipeline.py does.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _quiet_env  # noqa: E402,F401  (silences model-loading banners/warnings)
import mask_refine  # noqa: E402  (VRAM-aware batching of the segmenter)
import linux_workdir  # noqa: E402  (write the outputs on the Linux disk under WSL)


# ---------------------------------------------------------------------
# stdout protocol
# ---------------------------------------------------------------------
def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    _emit("LOG " + str(msg).replace("\n", " "))


def progress(done: int, total: int) -> None:
    _emit("PROGRESS %d %d" % (int(done), int(total)))


# --- benchmark mode -------------------------------------------------------
# FLATMORPHAI_QUIET=1 silences the explanatory LOG lines (adaptive choices,
# per-step profile, cache hit rates) for a clean timing run. The decision is
# taken ONCE here, by REBINDING the name to a function that does nothing: no
# test is left on the calling path, so quiet mode cannot itself cost anything.
# STEP / PROGRESS / DONE / ERROR are never silenced -- the page needs them to
# follow the run and to report a failure.
if os.environ.get("FLATMORPHAI_QUIET", "0") in ("1", "true", "True"):
    def log(msg: str) -> None:  # noqa: F811
        return None


# ---------------------------------------------------------------------
# Image-name convention: Species_Compound_Concentration_Plate_Position_Date[_Comment]
# (identical to the previous driver)
# ---------------------------------------------------------------------
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")
META_FIELDS = ("species", "compound", "concentration", "plate", "position", "date", "comment")


def empty_meta():
    return {k: "" for k in META_FIELDS}


def parse_image_name(filename: str):
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in _IMAGE_EXTS:
        return None
    parts = stem.split("_")
    if len(parts) < 6:
        return None
    species, compound, concentration, plate, position, date = parts[:6]
    if not (species and compound and concentration and plate and position and date):
        return None
    comment = "_".join(parts[6:]) if len(parts) > 6 else ""
    return dict(plate=plate, position=position, date=date, species=species,
                compound=compound, concentration=concentration, comment=comment)


def _date_key(t: str):
    t = str(t).strip()
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", t)
    if m:
        d, mo, y = (int(g) for g in m.groups()); return (2000 + y, mo, d)
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t)
    if m:
        y, mo, d = (int(g) for g in m.groups()); return (y, mo, d)
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)
    if m:
        y, mo, d = (int(g) for g in m.groups()); return (y, mo, d)
    return (9999, 99, 99, t)


def _date_ordinal(t):
    key = _date_key(str(t))
    if len(key) != 3:
        return None
    y, mo, d = key
    try:
        return datetime.date(y, mo, d).toordinal()
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------
_SKIP_DIRS = ("masks", "masks_refined", "temp_formated", "eyespots heatmaps",
              "blasteme_masks")
_SKIP_PREFIXES = ("masks_", "thumbnails_", "segmented_")


def find_image_folders(root: str):
    found = []
    for dirpath, dirnames, _files in os.walk(root):
        low = os.path.basename(dirpath).lower()
        if low in _SKIP_DIRS or low.startswith(_SKIP_PREFIXES):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in sorted(dirnames)
                       if d.lower() not in _SKIP_DIRS
                       and not d.lower().startswith(_SKIP_PREFIXES)]
        if "images" in dirnames:
            found.append(dirpath)
    return found


def list_images(images_dir: str):
    return [n for n in sorted(os.listdir(images_dir))
            if os.path.splitext(n)[1].lower() in _IMAGE_EXTS
            and os.path.isfile(os.path.join(images_dir, n))]


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
#: Images prepared (and segmented) per round. Matches BATCH in
#: parameters/segmentation.csv, the batch size the production masks were
#: produced with -- running the segmenter one image at a time, as the loop did
#: before, leaves the card idle between kernel launches.
_CHUNK = 8
# Wall-clock accumulated in the (batched) segmentation calls, for the per-step
# timing breakdown logged at the end. Diagnostic only.
_SEG_TIME = 0.0


def _is_oom(exc) -> bool:
    """True for both OOM shapes torch reports.

    The allocator raises "CUDA out of memory. Tried to allocate ...", but a raw
    driver allocation that fails (a cuDNN workspace, a new context) surfaces as
    "CUDA error: out of memory" instead, and only the second one appears once
    the card is merely fragmented rather than genuinely full.
    """
    return "out of memory" in str(exc).lower()


def _force_raw_segmentation(seg):
    """Feed the segmenter the raw image, as the model in service was trained.

    Shared by the regeneration chain and the eyespots counter, which used to
    disagree: one forced the enhancement off, the other left the bundle's
    default on, so the same model was fed two different inputs depending on
    the page. Returns True when the setting was changed.
    """
    import os as _os
    if seg is None or getattr(seg, "P", None) is None:
        return False
    try:
        if _os.environ.get("FLATMORPHAI_SEG_ENHANCE", "").strip() == "1":
            return False
        if str(seg.P.get("ENHANCE", "")).lower() != "none":
            seg.P["ENHANCE"] = "none"
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _physical_cores():
    """Physical cores (hyper-threads excluded), or None when unknown. Same
    reading as the stereotype pool, verified on the reference machine
    (4 physical / 8 logical). ``None`` means "do not cap on the CPU": a wrong
    count must never silently force the chain back to one worm."""
    # The reading lives in flatmorphai.cpu_topology (shared with the
    # stereotype pool); the "at least 2" rule stays here, because for THIS
    # chain a count of 1 means "do not cap" rather than "one worm at a time".
    try:
        try:
            from flatmorphai.cpu_topology import physical_cores as _pc
        except Exception:  # noqa: BLE001 - driver started from workers/
            import os as _os
            import sys as _sys
            _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__))))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from flatmorphai.cpu_topology import physical_cores as _pc
        n = _pc() or 0
        return n if n >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def _resolve_regen_workers(log, n_images=0):
    """How many worms to analyse at the same time.

    The chain is GPU-bound for about three quarters of its work, so while one
    worm waits for a U-Net the next one can run its skan axis and its descriptor
    formulas on the CPU. That overlap used to be opt-in because the production
    modules shared state between worms -- the descriptor block read its image
    from the globals ENTREE / J, the composite builder stashed the head of the
    current worm on itself, and one histogram-equalisation instance (which keeps internal
    buffers) served everybody -- so two worms at once corrupted each other's
    eye, pupil and blastema descriptors. Each of those three is now private to
    the thread that uses it (regen_pipeline/descriptors.py, composite.py), and
    the values were checked IDENTICAL BIT FOR BIT at 1, 2 and 3 threads on the
    reference images (test_concurrence_regeneration.py): x1.35 at 2 threads,
    x1.40 at 3.

    Default: 2 threads, which is where the gain is; the third adds 4 % for
    another model-sized footprint. Capped by the free VRAM (~1.6 GiB per extra
    worm), by the physical cores and by the number of images. Override with
    FLATMORPHAI_REGEN_WORKERS=N (1 disables the overlap entirely).
    """
    env = os.environ.get("FLATMORPHAI_REGEN_WORKERS", "").strip()
    forced = False
    if env:
        try:
            n = max(1, int(env))
            forced = True
        except ValueError:
            n = 2
    else:
        n = 2
    if n <= 1:
        if forced:
            log("  concurrency: 1 worker (FLATMORPHAI_REGEN_WORKERS=%s)" % env)
        return 1
    why = []
    try:
        import torch
        if torch.cuda.is_available():
            free_gib = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            cap = max(1, 1 + int(max(0.0, (free_gib - 1.0)) // 1.6))
            if cap < n:
                why.append("VRAM libre %.1f Gio" % free_gib)
            n = min(n, cap)
    except Exception:  # noqa: BLE001
        pass
    cores = _physical_cores()          # None when /proc/cpuinfo says nothing usable
    if cores and cores < n:
        why.append("%d physical core(s)" % cores)
        n = max(1, cores)
    if n_images:
        n = min(n, max(1, n_images))
    if n > 1:
        log("  concurrency: %d worker(s)%s; values bit-identical to the sequential "
            "chain (per-thread descriptor namespace and composite)"
            % (n, (" (limited by %s)" % ", ".join(why)) if why else ""))
        return n
    log("  concurrency: 1 worker%s" % ((" (limited by %s)" % ", ".join(why)) if why else ""))
    return 1


def _analyse_job(pipe, item, full_composite):
    """Run ONE image's heavy analysis (segmentation already done upstream). Pure
    w.r.t. shared state -- returns (item, result, error) so the caller does the
    ordered post-processing (writes + CSV rows). Safe to run in a thread pool."""
    name, stem, mpath, bgr, mask_m2f, fresh = item
    if bgr is None:
        return item, None, ValueError("image could not be decoded")
    try:
        mk = mask_refine.clean_seg_mask(mask_m2f, bgr)
        res = _analyse_with_retry(pipe, bgr, mk, name, full_composite)
        return item, res, None
    except Exception as exc:  # noqa: BLE001
        return item, None, exc


def _analyse_with_retry(pipe, bgr, mask_m2f, name, full_composite):
    """Run one worm through the chain, retrying once after freeing the VRAM.

    Several hundred worms can go through before the card is fragmented enough
    to refuse an allocation. Emptying the cache and trying the same image again
    recovers that case, so a run that is 90 % done is not thrown away.
    """
    try:
        return pipe.analyse(bgr, mask_m2f=mask_m2f, name=name,
                            full_composite=full_composite)
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        free = mask_refine._free_vram_mb_after_cleanup()
        log("  GPU memory exhausted on %s, cache released (%s MiB free), retrying"
            % (name, "?" if free is None else free))
        return pipe.analyse(bgr, mask_m2f=mask_m2f, name=name,
                            full_composite=full_composite)


def _prepare_one(images_dir, masks_dir, name, overwrite_masks):
    """Decode one image and its cached mask. Pure I/O, safe on a worker thread.

    cv2 is imported here rather than at module level: the driver delays that
    import until after _quiet_env and the allocator settings have been applied,
    and this helper runs on a worker thread that may reach it first. After the
    first call the import is a dictionary lookup in sys.modules.
    """
    import cv2

    stem = os.path.splitext(name)[0]
    path = os.path.join(images_dir, name)
    bgr = cv2.imread(path)
    if bgr is None:
        try:
            bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            bgr = None
    mpath = os.path.join(masks_dir, stem + ".png")
    mask = None
    if (not overwrite_masks) and os.path.isfile(mpath):
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    return [name, stem, mpath, bgr, mask, False]


#: Non-empty once the eyespot counting has failed, so the warning is said
#: once per run rather than once per image.
_EYE_FAIL = []


def _eye_positions(r3, t_bal, t_pos, pair_rule=False):
    """(positions at the strict threshold, positions at the permissive one).

    The count is the eyespot counter's, to the comma: every local maximum of
    the head-end heat map above the threshold. This chain used to read
    ``c["yeux"]``, which the cascade caps at the TWO strongest -- that cap
    exists to choose the two peaks whose probability goes to the state model,
    not to count eyespots, and it made the CSV plateau at 2 whatever the
    reality. The threshold, on the other hand, stays the STRICT one here: the
    operator only selects it for the counter.
    """
    try:
        import eyespot_shared  # noqa: PLC0415
        from regen_pipeline._env import A3b  # noqa: PLC0415
        return eyespot_shared.count_peaks(r3, A3b, t_bal, t_pos, pair_rule)
    except Exception as exc:  # noqa: BLE001 - never lose a run over a count
        # Said once and loudly: silently returning zero would fill the whole
        # CSV with "no eyespot" and look like a biological result.
        if not _EYE_FAIL:
            _EYE_FAIL.append(1)
            log("  WARNING: eyespot counting unavailable (%s: %s); the counts "
                "of this run are empty" % (type(exc).__name__, exc))
        return [], []


def _save_blasteme_mask(target, stem, m2f, refined, r3, keep_biggest, cv2, np):
    """Write the DETECTED blastema region for the editor's overlay.

    The blastema is the region the blastema U-Net predicts at the worm ends
    (probability >= SEUIL_BL = 0.40), exposed by the cascade as
    ``r3["blasteme_mask"]`` (same detection the pre-v20 pipeline showed). Using
    ``refined - m2f`` was wrong: that is only the pixels the refiner ADDED, which
    is empty whenever Mask2Former already covered the blastema -- so nothing was
    detected. The region is clipped to the refined worm mask. Falls back to the
    added pixels only when the cascade region is unavailable.
    """
    try:
        region = None
        if r3 and r3.get("blasteme_mask") is not None:
            bm_src = np.asarray(r3["blasteme_mask"])
            body = (np.asarray(refined) > 127)
            if bm_src.shape == body.shape:
                region = (bm_src > 127) & body
        if region is None:
            region = (np.asarray(refined) > 127) & ~(np.asarray(m2f) > 127)
        bm = region.astype(np.uint8) * 255
        if bm.sum() == 0:
            return
        if keep_biggest:
            bm = mask_refine.keep_largest_component(bm)
        d = os.path.join(target, "blasteme_masks")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, stem + ".png"), bm)
    except Exception:  # noqa: BLE001 - an optional overlay must never stop the run
        pass


def _save_head_outputs(target, stem, r3, save_heat, save_th_blast, save_th_cup,
                       cv2, np):
    """Write the eyespot heatmaps and the blastema / eyespot head-end thumbnails.

    Matches the pre-v20 pipeline's optional outputs, now sourced from the v20
    cascade (r3): the eye U-Net heatmap and thumbnail per candidate, and the
    per-end blastema thumbnail. The head end is the retained candidate
    (r3["i_head"]); heatmaps are written for every end (top/bottom by ``end``).
    """
    if not r3:
        return
    cands = r3.get("candidates") or []
    ends_bl = r3.get("ends_blasteme") or []
    i_head = int(r3.get("i_head", 0))
    try:
        if save_heat:
            d = os.path.join(target, "Eyespots heatmaps")
            os.makedirs(d, exist_ok=True)
            for c in cands:
                hm = c.get("eye_heatmap")
                if hm is not None:
                    suffix = "top" if int(c.get("end", 0)) == 0 else "bottom"
                    cv2.imwrite(os.path.join(d, "%s_%s.png" % (stem, suffix)),
                                (255 * np.clip(np.asarray(hm), 0, 1)).astype(np.uint8))
        if save_th_cup and 0 <= i_head < len(cands):
            tv = cands[i_head].get("eye_thumbnail")
            if tv is not None:
                d = os.path.join(target, "thumbnails_eyespots")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, stem + ".png"), np.asarray(tv))
        if save_th_blast and 0 <= i_head < len(cands):
            end = int(cands[i_head].get("end", 0))
            if 0 <= end < len(ends_bl):
                tb = ends_bl[end].get("thumbnail")
                if tb is not None:
                    d = os.path.join(target, "thumbnails_blastemes")
                    os.makedirs(d, exist_ok=True)
                    cv2.imwrite(os.path.join(d, stem + ".png"), np.asarray(tb))
    except Exception:  # noqa: BLE001 - optional outputs must never stop the run
        pass


def _blasteme_areas(r3):
    """(surface_area_up, surface_area_down) from the cascade's per-end blastema,
    for blasteme_detection.csv. Empty strings when a value is unavailable."""
    ends_bl = (r3 or {}).get("ends_blasteme") or []
    up = int(ends_bl[0]["aire"]) if len(ends_bl) > 0 else ""
    down = int(ends_bl[1]["aire"]) if len(ends_bl) > 1 else ""
    return up, down


def _save_composite_thumbnails(target, stem, comp, save_regen, save_body,
                               save_hz, cv2, np):
    """Write the composite-derived thumbnails the operator asked for.

    ``comp`` is the 384x384x3 state composite (BGR): channel 0 whole worm,
    channel 1 zoom, channel 2 super-zoom. The editor and the user know these as
    thumbnails_regeneration (the whole composite), thumbnails_all_body
    (channel 0) and thumbnails_head_zoomed (channel 1).
    """
    if comp is None:
        return
    try:
        comp = np.asarray(comp)
        if comp.ndim != 3 or comp.shape[2] < 3:
            return
        for want, sub, arr in (
                (save_regen, "thumbnails_regeneration", comp),
                (save_body, "thumbnails_all_body", comp[:, :, 0]),
                (save_hz, "thumbnails_head_zoomed", comp[:, :, 1])):
            if not want:
                continue
            d = os.path.join(target, sub)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, stem + ".png"), arr)
    except Exception:  # noqa: BLE001 - an optional thumbnail must never stop the run
        pass


def _first_sample_bgr(folders):
    """Decode the first readable image across the folders, for calibration."""
    import cv2
    for d in folders:
        images_dir = os.path.join(d, "images")
        for n in list_images(images_dir):
            try:
                bgr = cv2.imdecode(
                    np.fromfile(os.path.join(images_dir, n), np.uint8),
                    cv2.IMREAD_COLOR)
                if bgr is not None:
                    return bgr
            except Exception:  # noqa: BLE001
                continue
    return None


def _iter_prepared(pipe, images_dir, masks_dir, names, overwrite_masks):
    """Yield ``(name, stem, mask_path, bgr, mask_m2f, fresh)`` ready to analyse.

    Two things happen here that the plain per-image loop could not do.

    JPEG decoding of the next round runs on a worker thread while the current
    round is on the GPU, so the card no longer waits for the disk between
    images.

    The images that have no cached mask are segmented TOGETHER, through
    ``Segmenter.segment_batch`` -- the entry point the production masks were
    generated with. Mask2Former then sees a real batch instead of one image per
    forward pass, which is where most of the idle time was. ``fresh`` tells the
    caller the mask was just computed and is worth writing to disk.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _load(chunk):
        return [_prepare_one(images_dir, masks_dir, n, overwrite_masks) for n in chunk]

    segmenter = getattr(pipe, "segmenter", None)
    # Start from the batch that last worked, not always the configured maximum:
    # carried across folders on the segmenter, and across runs through a small
    # cache, so a card that only fits 2 does not waste a doomed 8 -> 4 -> 2
    # probe (each failed attempt costs a full forward) at every folder.
    try:
        _configured = max(1, int(segmenter.P["BATCH"]))
    except Exception:  # noqa: BLE001
        _configured = _CHUNK
    # segment_oom_safe starts from the last batch that fit (cross-folder and
    # cross-run cache) and caps it to this maximum, so we just pass the
    # configured value and carry the result it returns across chunks.
    batch = _configured
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for k in range(0, len(names), _CHUNK):
            fut = pending if pending is not None else pool.submit(_load, names[k:k + _CHUNK])
            nxt = names[k + _CHUNK:k + 2 * _CHUNK]
            pending = pool.submit(_load, nxt) if nxt else None
            items = fut.result()
            todo = [i for i, it in enumerate(items) if it[4] is None and it[3] is not None]
            if todo and segmenter is not None:
                try:
                    _t_seg = time.time()
                    masks, batch = mask_refine.segment_oom_safe(
                        segmenter, [items[i][3] for i in todo],
                        start_batch=batch, log=log,
                        suppress=_quiet_env.suppressed_output)
                    global _SEG_TIME
                    _SEG_TIME += time.time() - _t_seg
                    for i, mk in zip(todo, masks):
                        items[i][4] = mk[0] if isinstance(mk, tuple) else mk
                        items[i][5] = True
                    # Hand the blocks of this batch back to the driver before
                    # the next one: a long folder otherwise ends with a cache
                    # too fragmented for the next raw allocation.
                    mask_refine._free_vram_mb_after_cleanup()
                except Exception as exc:  # noqa: BLE001
                    # Never lose the folder over a batching problem: the items
                    # go out without a mask and analyse() segments them one by
                    # one, exactly as before this optimisation.
                    log("  batch segmentation unavailable (%s: %s), "
                        "falling back to one image at a time"
                        % (type(exc).__name__, exc))
                    segmenter = None
            for it in items:
                yield tuple(it)


def main() -> int:
    if len(sys.argv) < 2:
        _emit("ERROR no config file given")
        return 1
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    import cv2  # noqa: E402

    bundle_root = cfg.get("bundle_root", "")
    if not bundle_root or not os.path.isdir(bundle_root):
        _emit("ERROR regeneration v20 bundle not found: " + str(bundle_root))
        return 1
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)

    root = cfg["root"]
    cancel_file = cfg.get("cancel_file", "")

    def _interrupted():
        return bool(cancel_file and os.path.exists(cancel_file))

    t_bal = float(cfg.get("eyespots_balanced_threshold", 0.80))
    t_pos = float(cfg.get("eyespots_positive_threshold", 0.60))
    save_blasteme_masks = bool(cfg.get("save_blasteme_masks", False))
    overwrite_masks = bool(cfg.get("overwrite_masks", False))
    # "Full body mask refined with the blastema mask": when on, the mask written
    # to masks/ is the REFINED mask (body + blastema the refiner added), not the
    # raw Mask2Former mask. The refiner always runs for the state classifier, so
    # this only changes which mask is saved. The raw mask is never cached in that
    # case (see _iter_prepared below), so a re-run re-segments instead of feeding
    # the already-refined mask back into the refiner.
    blasteme_refined = bool(cfg.get("blasteme_refined", False))
    # The state classifier reads the composite through ch_map = (0, 2, 2):
    # whole worm once, super-zoom twice, channel 1 never. Building channel 1
    # costs two warpAffine passes over the full-size image per worm for
    # pixels nothing consumes, so it is skipped unless the composite PNG is
    # actually written to disk, where the three production channels matter.
    # Build the full 3-channel composite whenever a composite-derived thumbnail
    # is requested (head-zoomed uses channel 1, which the fast path does not
    # build), not only when the composite PNG itself is saved.
    _save_regen = bool(cfg.get("save_thumb_regeneration", False))
    _save_body = bool(cfg.get("save_thumb_all_body", False))
    _save_hz = bool(cfg.get("save_thumb_head_zoomed", False))
    full_composite = _save_regen or _save_body or _save_hz
    # Head-detection optional outputs (previously ignored by this driver, so the
    # checkboxes did nothing). Produced from the cascade result r3, which now
    # exposes the eye thumbnail/heatmap per candidate and the per-end blastema
    # thumbnail + area.
    _save_heat = bool(cfg.get("save_eyespots_heatmaps", False))
    _save_bl_csv = bool(cfg.get("save_blasteme_csv", False))
    _save_th_blast = bool(cfg.get("save_thumb_blastemes", False))
    _save_th_cup = bool(cfg.get("save_thumb_eyespots", False))

    try:
        folders = find_image_folders(root)
        if not folders:
            _emit("ERROR no 'images' folder found under the processing root")
            return 1
        grand_total = sum(len(list_images(os.path.join(d, "images"))) for d in folders)
        if grand_total == 0:
            _emit("ERROR the 'images' folders contain no readable image")
            return 1
        log("%d image(s) to analyse in total" % grand_total)

        # --- load the pipeline once (segments, refines, cascade, state) ---
        _emit("STEP 0")
        from regen_pipeline import RegenerationPipeline  # noqa: E402
        mask_refine.preflight_vram(log=log)
        log("Loading the regeneration pipeline (4 models)...")
        # Suppress the model-loading chatter (ThinWorm DDP note, torch.load
        # FutureWarning, pixel-decoder / resolution_upgrade_784 / refine_1x
        # warnings, the loss-initialisation banner). They are printed by the
        # vendored training code at load time and carry no operational info.
        with _quiet_env.suppressed_output():
            pipe = RegenerationPipeline(segmentation=True, verbose=False)

        # Per-image concurrency: analyse several images at once so the CPU work
        # (skan axis + descriptor formulas, ~56% of the time) of one image
        # overlaps the GPU work (U-Nets, ~44%) of another. Validated on the
        # reference GTX 1060: x1.7 faster with the STATE unchanged (the only
        # descriptors that move under concurrency are the eye/pupil and blastema
        # peak detections, already GPU-sensitive and not read for the state). We
        # pin cuDNN to deterministic algorithms so the convolutions are stable.
        try:
            import torch as _torch
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except Exception:  # noqa: BLE001
            pass
        _workers = _resolve_regen_workers(log, grand_total)

        # Feed the segmentation model at its training resolution (320x320), the
        # SAME segmentation as the Stereotype classification page. Without this
        # the processor resizes the whole frame to ~1088x800 and one forward
        # overflows the card -- the exact OOM churn seen before this fix there.
        _seg = getattr(pipe, "segmenter", None)
        # The local histogram equalisation the bundle could apply BEFORE
        # segmenting is forced off: the model in service
        # (model_Mask2Former_20260324_0451) was trained on raw images, so
        # enhancing the input would feed it a distribution it never saw and
        # degrade the masks. Every page does the same, so the four of them
        # segment identically. Silent: this is settled, not a choice made per
        # run, and the log is for what the run DID. FLATMORPHAI_SEG_ENHANCE=1
        # puts it back, and is only meaningful with a model trained that way.
        _force_raw_segmentation(_seg)
        if _seg is not None and getattr(_seg, "processor", None) is not None:
            _sample = _first_sample_bgr(folders)
            if _sample is not None:
                mask_refine.configure_fed_size(_seg, _sample, target=320, log=log)
            else:
                log("  Fed resolution: no readable frame to calibrate from.")

        done = 0
        _t_start = time.time()
        # Reset the per-step timers so the breakdown logged at the end covers
        # only this run.
        global _SEG_TIME
        _SEG_TIME = 0.0
        try:
            from regen_pipeline.pipeline import reset_step_stats as _reset_steps
            _reset_steps()
        except Exception:  # noqa: BLE001
            pass
        from concurrent.futures import ThreadPoolExecutor as _TPE
        from collections import deque as _deque
        # The pool analyses several images concurrently; results are consumed
        # strictly IN ORDER below so the CSV rows and files are unchanged.
        _pool = _TPE(max_workers=_workers) if _workers > 1 else None
        # Under WSL the output folder is on /mnt/c, where writing one
        # full-resolution PNG costs ~0.30 s against ~0.07 s on the Linux disk;
        # this chain writes a mask (and optionally several thumbnails) per worm.
        # The inputs are still READ from the user's folder -- they have to be
        # read once anyway -- only the outputs go to a Linux mirror, copied back
        # in the background while the run continues and then at the end. Same
        # files, same names, same bytes. FLATMORPHAI_LINUX_WORKDIR=0 writes in
        # place, as before.
        work = linux_workdir.work_path(root)
        if work:
            try:
                if os.path.isdir(work):      # leftovers of an interrupted run
                    import shutil as _sh
                    _sh.rmtree(work, ignore_errors=True)
                os.makedirs(work, exist_ok=True)
                log("Writing the results on the Linux disk, copied back to %s "
                    "during and at the end of the run" % root)
            except Exception as exc:  # noqa: BLE001
                log("  Linux work folder unavailable (%s); writing in place" % exc)
                work = None

        def _out_dir(path):
            """Where to WRITE ``path`` (a folder under root): the Linux mirror,
            or ``path`` itself when the mirror is off."""
            if not work:
                return path
            rel = os.path.relpath(path, root)
            q = work if rel == "." else os.path.join(work, rel)
            os.makedirs(q, exist_ok=True)
            return q

        _sync_started = [False]
        for d in folders:
            if _interrupted():
                _emit("ERROR cancelled")
                if _pool is not None:
                    _pool.shutdown(wait=False, cancel_futures=True)
                return 1
            images_dir = os.path.join(d, "images")
            target = d
            out_target = _out_dir(d)
            masks_dir = os.path.join(target, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            out_masks = _out_dir(masks_dir)
            names = list_images(images_dir)
            rows_state, rows_head, rows_cup, rows_blast = [], [], [], []

            _emit("STEP 1")

            def _post(item, res, err):
                """Ordered post-processing of one analysed image: mask/thumbnail
                writes + CSV rows. Called from the MAIN thread in submission
                order, so the outputs are byte-for-byte what the sequential path
                produced; only the analysis (_analyse_job) ran in the pool."""
                nonlocal done
                name, stem, mpath, bgr, mask_m2f, fresh = item
                meta = parse_image_name(name) or empty_meta()
                meta_cols = {k: meta[k] for k in
                             ("species", "compound", "concentration", "comment")}
                base = {"image_name": name, "plate": meta["plate"],
                        "position": meta["position"], "date": meta["date"], **meta_cols}
                if err is not None or res is None:
                    if err is not None:
                        log("  ERROR on %s: %s: %s"
                            % (name, type(err).__name__, err))
                        if _is_oom(err):
                            mask_refine._free_vram_mb_after_cleanup()
                    done += 1
                    progress(done, grand_total)
                    return
                # Write the mask for masks/: the REFINED mask when the refine
                # option is on, else the raw Mask2Former mask. Only when fresh.
                if fresh:
                    _mask_out = res.get("mask_refined") if (
                        blasteme_refined and res.get("mask_refined") is not None
                    ) else res.get("mask_m2f")
                    if _mask_out is not None:
                        cv2.imwrite(os.path.join(out_masks, stem + ".png"), _mask_out)
                stage = res.get("stage") or {}
                head = res.get("head") or {}
                eye_bal, eye_pos = _eye_positions(
                    res.get("r3"), t_bal, t_pos,
                    bool(cfg.get("eyespots_pair_rule", False)))
                eye_pts = eye_bal
                eye_count = len(eye_bal)
                if save_blasteme_masks and res.get("mask_refined") is not None \
                        and res.get("mask_m2f") is not None:
                    _save_blasteme_mask(out_target, stem, res["mask_m2f"],
                                        res["mask_refined"], res.get("r3"),
                                        True, cv2, np)
                if _save_regen or _save_body or _save_hz:
                    _save_composite_thumbnails(
                        out_target, stem, res.get("composite"),
                        _save_regen, _save_body, _save_hz, cv2, np)
                if _save_heat or _save_th_blast or _save_th_cup:
                    _save_head_outputs(out_target, stem, res.get("r3"),
                                       _save_heat, _save_th_blast, _save_th_cup,
                                       cv2, np)
                if _save_bl_csv:
                    _up, _down = _blasteme_areas(res.get("r3"))
                    rows_blast.append(dict(base, surface_area_up=_up,
                                           surface_area_down=_down))
                dec = stage.get("stage_decimal")
                rnd = stage.get("stage_class")
                if dec is not None and rnd is not None:
                    rnd = int(rnd)
                    if rnd == 4 and eye_count < 1:
                        rnd = 3
                    rows_state.append(dict(base, index_planaria="", Day="",
                                           state_round=rnd, state_decimal=round(float(dec), 4),
                                           state_round_corrected=rnd, state_corrected=""))
                else:
                    rows_state.append(dict(base, index_planaria="", Day="",
                                           state_round="", state_decimal="",
                                           state_round_corrected="", state_corrected=""))
                if head:
                    # The file describes the MEDIAL AXIS, as its column names
                    # say and as the first implementation of the chain did:
                    # the axis has two tips, up_position is the higher one in
                    # the image and bottom_position the lower, one of them is
                    # the head and the other the tail. The v20 driver had
                    # broken that: it wrote the tip AND the framing centre of
                    # the head end alone -- two points of the same extremity,
                    # 0.12 of the worm's length apart -- so the second tip was
                    # missing from the file entirely.
                    # The tips come from the cascade, which means they carry
                    # its blastema correction where it applied one.
                    _ends = res.get("ends")
                    _tips = []
                    try:
                        for _pointe, _centre in (_ends or []):
                            if _pointe is not None:
                                _tips.append((float(_pointe[1]), float(_pointe[0])))  # (y,x)->(x,y)
                    except Exception:  # noqa: BLE001
                        _tips = []
                    # The head is the tip of the end the cascade chose.
                    _r3 = res.get("r3") or {}
                    _head_tip = None
                    try:
                        _p = _r3.get("tip")
                        if _p is not None:
                            _head_tip = (float(_p[1]), float(_p[0]))
                    except Exception:  # noqa: BLE001
                        _head_tip = None
                    # The cascade's own anchor is the centre of the eyespots
                    # when it found some, a point INSIDE the head rather than
                    # its tip. It only serves to tell which tip is the head.
                    _anchor = (float(head.get("x_tete", 0.0)), float(head.get("y_tete", 0.0)))
                    if len(_tips) == 2:
                        _ref = _head_tip if _head_tip is not None else _anchor
                        _d = [((t[0] - _ref[0]) ** 2 + (t[1] - _ref[1]) ** 2) for t in _tips]
                        _head_pt = _tips[0] if _d[0] <= _d[1] else _tips[1]
                        _tail_pt = _tips[1] if _d[0] <= _d[1] else _tips[0]
                        _up, _bottom = sorted(_tips, key=lambda t: t[1])   # higher first
                    else:
                        # No usable axis: the anchor is the head, and the far
                        # end is measured on the body so the row still holds
                        # two ends.
                        _far = mask_refine.tail_from_mask(
                            res.get("mask_refined") if res.get("mask_refined") is not None
                            else res.get("mask_m2f"), _anchor) or _anchor
                        _head_pt = _anchor
                        _tail_pt = _far
                        _up, _bottom = sorted((_anchor, _far), key=lambda t: t[1])
                    rows_head.append(dict(
                        base,
                        up_position="(%.1f, %.1f)" % _up,
                        bottom_position="(%.1f, %.1f)" % _bottom,
                        head_position="(%.1f, %.1f)" % _head_pt,
                        # The tail is the tip that is not the head, so the
                        # column always holds one of the two above.
                        tail_position="(%.1f, %.1f)" % _tail_pt))
                rows_cup.append(dict(base, n_eyespots=eye_count, _eye_pts=eye_pts,
                                     _eye_pos=eye_pos))
                done += 1
                progress(done, grand_total)
                if work and not _sync_started[0] and done >= max(4, grand_total // 3):
                    _sync_started[0] = True
                    linux_workdir.start_background_sync(work, root, log)

            _inflight = _deque()
            for item in _iter_prepared(
                    pipe, images_dir, masks_dir, names,
                    overwrite_masks or blasteme_refined):
                if _interrupted():
                    _emit("ERROR cancelled")
                    if _pool is not None:
                        _pool.shutdown(wait=False, cancel_futures=True)
                    return 1
                if _pool is not None:
                    # Submit this image's analysis; keep up to `_workers` in
                    # flight so CPU and GPU of consecutive images overlap, and
                    # consume the OLDEST in order to keep output order exact.
                    _inflight.append(_pool.submit(_analyse_job, pipe, item,
                                                  full_composite))
                    if len(_inflight) >= _workers:
                        _post(*_inflight.popleft().result())
                else:
                    _post(*_analyse_job(pipe, item, full_composite))
            while _inflight:
                _post(*_inflight.popleft().result())

            _write_folder_csvs(out_target, rows_state, rows_head, rows_cup, rows_blast, t_bal, t_pos, cfg)
            # The next folder starts from a clean allocator rather than
            # inheriting this one's fragmentation.
            mask_refine._free_vram_mb_after_cleanup()
            try:
                from regen_pipeline.fastpath import clear as _fp_clear
                _fp_clear()
            except Exception:  # noqa: BLE001
                pass

        if _pool is not None:
            _pool.shutdown()
        if work:
            linux_workdir.sync_back(work, root, log)

        try:
            from regen_pipeline.fastpath import stats as _fp_stats
            _fp = _fp_stats()
            if _fp.get("calls"):
                log("U-Net inference: %d call(s), %d served from the "
                    "repeated-crop cache (%.0f%%)"
                    % (_fp["calls"], _fp["hits"],
                       100.0 * _fp["hits"] / max(1, _fp["calls"])))
        except Exception:  # noqa: BLE001
            pass
        if grand_total:
            _elapsed = time.time() - _t_start
            # Be explicit about what this number covers, since it is easy to
            # misread: it is the WHOLE pipeline per image (segmentation + the 4
            # U-Nets + descriptors + state), wall-clock. With >1 worker the
            # images overlap, so the per-step totals below SUM TO MORE than the
            # wall-clock (they run in parallel) -- read them as "where the work
            # is", not as a partition of the elapsed time.
            _mode = ("concurrent, %d workers" % _workers) if _workers > 1 else "sequential"
            log("%d image(s) in %.1f s | %.2f s/image, full pipeline "
                "(segmentation + U-Nets + descriptors + state), %s"
                % (grand_total, _elapsed, _elapsed / max(1, grand_total), _mode))
            # Per-step breakdown: where the time actually goes, to target the
            # right bottleneck before any pipelining work.
            try:
                from regen_pipeline.pipeline import step_stats as _step_stats
                rows = [("segmentation (batched)", _SEG_TIME, grand_total)]
                for k, (tot, n) in _step_stats().items():
                    rows.append((k, tot, n))
                rows.sort(key=lambda r: -r[1])
                # Share is computed over the TOTAL WORK (sum of steps), not over
                # the wall-clock, so it stays a valid 0-100% split even when the
                # images overlap (concurrent mode).
                _work = sum(t for _nm, t, _ in rows) or 1e-9
                log("  --- per-stage profile (total s | s/image | share of the work) ---")
                for name, tot, n in rows:
                    log("    %-32s %7.1f s | %.3f s/img | %4.1f%%"
                        % (name, tot, tot / max(1, grand_total),
                           100.0 * tot / _work))
                _gpu = sum(t for nm, t, _ in rows if any(
                    w in nm for w in ("segmentation", "U-Net", "ConvNeXt")))
                log("  GPU (segmentation + U-Nets + state) ~%.0f%% | CPU remainder "
                    "(skan + formulas + I/O) ~%.0f%% of the work"
                    % (100.0 * _gpu / _work, 100.0 * (_work - _gpu) / _work))
            except Exception as _e:  # noqa: BLE001
                log("  (per-stage profile unavailable: %s)" % _e)
        _emit("DONE %d" % grand_total)
        return 0
    except SystemExit as exc:
        _emit("ERROR " + str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        _emit("ERROR %s: %s" % (type(exc).__name__, exc))
        traceback.print_exc()
        # Same release as on cancel: drop what the models hold so the card is
        # clean for the next attempt instead of carrying this run's footprint.
        mask_refine.release_gpu_on_crash(log=log)
        return 1


def _write_folder_csvs(target, rows_state, rows_head, rows_cup, rows_blast, t_bal, t_pos, cfg):
    # planaria index, Day, monotone state correction (one worm = plate+position
    # + species+compound+concentration, as in the previous driver)
    def _worm_key(r):
        return (str(r["plate"]), str(r["position"]), str(r.get("species", "")),
                str(r.get("compound", "")), str(r.get("concentration", "")))

    groups = {}
    for r in rows_state:
        groups.setdefault(_worm_key(r), []).append(r)
    for i, key in enumerate(sorted(groups), start=1):
        members = sorted(groups[key], key=lambda r: _date_key(r["date"]))
        ords = [_date_ordinal(r["date"]) for r in members]
        valid = [o for o in ords if o is not None]
        day0 = min(valid) if valid else None
        for rank, (r, o) in enumerate(zip(members, ords), start=1):
            r["index_planaria"] = i
            r["Day"] = (o - day0 + 1) if (day0 is not None and o is not None) else rank
        if cfg.get("auto_state_increase_correction", True):
            running = None
            for r in members:
                v = r["state_round_corrected"]
                if v == "" or v is None:
                    continue
                v = int(v)
                if running is not None and v < running:
                    v = running
                r["state_round_corrected"] = v
                running = v

    # Editor-compatible eyespot columns (count + (x, y) positions, per
    # threshold), produced by the shared helper so the regeneration editor
    # reads them exactly as it reads the eyespots counter's output.
    import eyespot_shared  # noqa: E402
    col_bal, col_bal_pos, col_pos, col_pos_pos = eyespot_shared.eyespot_columns(
        t_bal, t_pos)
    col_bal_corr = col_bal + "_corrected"
    for r in rows_cup:
        _pts = r.pop("_eye_pts", []) or []
        n = int(r.pop("n_eyespots", len(_pts)) or 0)
        _pts_pos = r.pop("_eye_pos", None)
        if _pts_pos is None:
            _pts_pos = _pts
        r[col_bal] = n
        r[col_bal_pos] = eyespot_shared.positions_str(_pts)
        # Each column its own threshold. Both used to carry the strict count,
        # so the permissive column said nothing it did not already say.
        r[col_pos] = len(_pts_pos)
        r[col_pos_pos] = eyespot_shared.positions_str(_pts_pos)
    if cfg.get("auto_eyespots_increase_correction", True):
        cg = {}
        for r in rows_cup:
            cg.setdefault(_worm_key(r), []).append(r)
        for key in cg:
            members = sorted(cg[key], key=lambda r: _date_key(r["date"]))
            running = None
            for r in members:
                v = int(r.get(col_bal, 0) or 0)
                if running is not None and v < running:
                    v = running
                r[col_bal_corr] = v
                running = v
    else:
        for r in rows_cup:
            r[col_bal_corr] = r.get(col_bal, 0)

    def _write(path, rows, columns):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, delimiter=";")
            w.writeheader()
            w.writerows(rows)
        log("  wrote %s (%d rows)" % (os.path.basename(path), len(rows)))

    meta_hdr = ["species", "compound", "concentration", "comment"]
    base_hdr = ["image_name", "plate", "position", "date"] + meta_hdr
    _write(os.path.join(target, "regeneration_states.csv"), rows_state,
           ["image_name", "index_planaria", "plate", "position", "date"] + meta_hdr
           + ["Day", "state_round", "state_decimal", "state_round_corrected", "state_corrected"])
    _write(os.path.join(target, "head_position.csv"), rows_head,
           base_hdr + ["up_position", "bottom_position", "head_position",
                       "tail_position"])
    _write(os.path.join(target, "eyespots_detection.csv"), rows_cup,
           base_hdr + [col_bal, col_bal_corr, col_bal_pos, col_pos, col_pos_pos])
    if cfg.get("save_blasteme_csv") and rows_blast:
        _write(os.path.join(target, "blasteme_detection.csv"), rows_blast,
               base_hdr + ["surface_area_up", "surface_area_down"])


if __name__ == "__main__":
    raise SystemExit(main())
