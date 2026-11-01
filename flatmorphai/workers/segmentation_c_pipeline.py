# -*- coding: utf-8 -*-
"""Driver for Image segmentation - segmentation only (runs in a dedicated subprocess).

Why a subprocess.  Image segmentation reuses the self-contained ``Export_head_detection``
bundle for two reasons: its ``Segmenter`` applies exactly the same two-pass
Mask2Former as the stereotype-classification and regeneration chains,
and its blastema model is required for the optional "Blasteme refined" step.
Several bundle modules use bare, path-based imports and carry names that clash
with the main application (``evaluate_test_images_v2``, ``mask2former_api``,
``preprocessing``, ``anchor_cascade``).  Importing them inside the running Qt
process would pick up the wrong module from ``sys.modules`` and the GPU setup
would fight the application's own Mask2Former.  A fresh subprocess gets an empty
``sys.modules`` so the bundle versions load cleanly, isolates the CUDA context
and lets the worker cancel the run by terminating the process.

What it does, for every ``images/`` folder found under the processing root:
  1. segments each frame with the bundle's Mask2Former (two-pass, raw input),
     using the SAME checkpoint the application resolves from its Settings page ;
  2. (optional) "Blasteme refined": on the Mask2Former mask it computes the worm
     skeleton/ends and runs the blastema segmenter on BOTH extremities, exactly
     like the blastema pipeline.  The two 384 thumbnails it builds are written to
     a temporary ``temp_thumbnails_blasteme/`` folder next to ``images/`` and that
     folder is deleted at the end of the folder.  The blastema prediction (binary
     mask in the original frame) is then combined (union) with the Mask2Former
     mask so a blastema that Mask2Former cut off is recovered.
The result is written to ``masks_<model>/`` next to ``images/`` - exactly where
Pipelines A and B expect the masks - with the same file names as the images.

It prints machine-readable lines on stdout for the worker:
    LOG <message>
    PROGRESS <done> <total>
    DONE / ERROR <message>
Every other line is treated as a plain log line.

Seed 42.  A GPU is required in practice for the models.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

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
import traceback

# ``mask_refine`` lives next to this driver; make its directory importable
# whether this module is run as a script or imported for testing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _quiet_env  # noqa: E402,F401  (silence noisy third-party notices first)
import mask_refine  # noqa: E402
import linux_workdir  # noqa: E402  (write the outputs on the Linux disk under WSL)


# ---------------------------------------------------------------------
# stdout protocol helpers
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
# Folder / image discovery (same rules as the other pipelines)
# ---------------------------------------------------------------------
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")
_SKIP_DIRS = ("temp_formated", "temp_thumbnails_blasteme")
_SKIP_PREFIXES = ("masks_", "masks", "thumbnails_", "segmented_")


def find_image_folders(root: str):
    """Every folder containing an ``images/`` sub-directory, at any depth.

    Mask and thumbnail folders are skipped so a second run never descends
    into the output of a previous one.
    """
    found = []
    for dirpath, dirnames, _files in os.walk(root):
        low = os.path.basename(dirpath).lower()
        if low in _SKIP_DIRS or low.startswith(_SKIP_PREFIXES):
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d.lower() not in _SKIP_DIRS and not d.lower().startswith(_SKIP_PREFIXES)
        ]
        if "images" in dirnames:
            found.append(dirpath)
    return found


def _training_fed_size(cfg):
    """The resolution the segmentation model was trained at (pixels), or 320.

    Read from the checkpoint parameters file (``image_size`` / ``input_shape``)
    so a future model retrained at another size is honoured automatically.
    """
    try:
        import glob
        import pickle
        root = cfg.get("seg_checkpoint") or ""
        pkls = glob.glob(os.path.join(root, "*_parameters.pkl")) if root else []
        if pkls:
            d = pickle.load(open(pkls[0], "rb"))
            v = d.get("image_size")
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v)
            sh = d.get("input_shape")
            if isinstance(sh, (list, tuple)) and sh and int(sh[0]) > 0:
                return int(sh[0])
    except Exception:  # noqa: BLE001
        pass
    return 320


def _first_sample_bgr(folders):
    """Decode the first readable image across the folders, for calibration."""
    import cv2
    import numpy as np
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


def list_images(images_dir: str):
    out = []
    for name in sorted(os.listdir(images_dir)):
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS and os.path.isfile(
            os.path.join(images_dir, name)
        ):
            out.append(name)
    return out


# ---------------------------------------------------------------------
# Export_head_detection import (subprocess only -> clean sys.modules)
# ---------------------------------------------------------------------
def _add_export_to_path(export_root: str) -> None:
    for d in ("common", "parameters", "eyes", "blastema", "darkness",
              "head", "segmentation", "stage"):
        p = os.path.join(export_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------
# Gaussian edge smoothing + re-binarization
# ---------------------------------------------------------------------
def _odd_kernel(k) -> int:
    """Return an odd kernel size >= 1 (Gaussian kernels must be odd)."""
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _smooth_and_binarize(mask, kernel):
    """Gaussian-blur a 0/255 mask then threshold back to 0/255.

    Rounds staircase edges and removes isolated stray pixels, while keeping
    the mask strictly binary for the downstream pipelines.
    """
    import cv2
    k = _odd_kernel(kernel)
    if k <= 1:
        return mask
    blur = cv2.GaussianBlur(mask, (k, k), 0)
    return (blur > 127).astype("uint8") * 255


# ---------------------------------------------------------------------
# cancellation flag via a sentinel file written by the worker
# ---------------------------------------------------------------------
_CANCEL_FILE = None


def _interrupted():
    return bool(_CANCEL_FILE) and os.path.exists(_CANCEL_FILE)


# ---------------------------------------------------------------------
# Per-folder segmentation
# ---------------------------------------------------------------------
#: Segmenter modules parked on the host by stage 2, restored by the next
#: folder's stage 1. Module level because it must outlive one call.
_SEG_PARKED = []


#: When the processing root is on /mnt/<x> (WSL view of a Windows drive), the
#: FINAL outputs (masks, blastema masks, thumbnails, CSV) are written to a
#: mirror on the Linux disk and copied back at the end: one full-resolution PNG
#: costs ~0.30 s on /mnt/c against ~0.07 s there, and this driver writes one or
#: more per image. The inputs keep being read from the user's folder, and the
#: scratch of the three stages already lived on the local disk.
#: Stereotype classification's step 2 already runs inside such a work folder, so work_path()
#: returns None there and nothing is mirrored twice.
_WORK = {"root": None, "work": None}


def _out(path):
    """Where to WRITE ``path``: the Linux mirror, or ``path`` itself."""
    w, r = _WORK["work"], _WORK["root"]
    if not w or not r:
        return path
    try:
        rel = os.path.relpath(path, r)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    q = w if rel == "." else os.path.join(w, rel)
    d = q if not os.path.splitext(q)[1] else os.path.dirname(q)
    if d:
        os.makedirs(d, exist_ok=True)
    return q


def process_all(folders, cfg, seg, bl_mod, counters, head_mods=None):
    """Segment EVERY folder's frames as a single staged run.

    The previous version walked folder by folder and each folder re-ran the
    three stages and re-parked the auxiliary models (the per-folder VRAM
    churn seen in the log). Here the whole set of frames across ALL folders
    goes through stage 1 (Mask2Former, pass 1 then pass 2) together, THEN all
    through stage 2 (blastema, only when selected), THEN all through stage 3
    (union, head detection, final writes). Each model family is loaded onto the
    card once for the entire run instead of once per folder, and "all pass 1
    then all pass 2" holds across folders rather than within one. Outputs,
    columns and paths are unchanged, so the editors keep working.
    """
    import numpy as np  # noqa: F401
    import cv2

    save_body = bool(cfg.get("save_body_masks", True))
    save_bl_masks = bool(cfg.get("save_blasteme_masks", False))
    save_bl_thumbs = bool(cfg.get("save_blasteme_thumbnails", False))
    save_head_csv = bool(cfg.get("save_head_csv"))
    save_body_thumbs = bool(cfg.get("save_body_thumbnails"))
    refine = bool(cfg.get("blasteme_refined"))
    keep_biggest = bool(cfg.get("blasteme_keep_biggest"))
    # The blastema detector runs when any blastema output/refinement is wanted.
    need_blasteme = (refine or save_bl_masks or save_bl_thumbs) and bl_mod is not None
    # The head detector runs for the head CSV and the full body thumbnails.
    need_head = (save_head_csv or save_body_thumbs) and head_mods is not None
    overwrite = bool(cfg.get("overwrite"))
    # Adaptive batch: start at the configured BATCH; segment_oom_safe shrinks it
    # only on CUDA OOM. The batch is a memory/speed knob only -- identical masks.
    configured_batch = max(1, int(seg.P.get("BATCH", 8)) if hasattr(seg, "P") else 8)

    done_total, grand_total = counters["done"], counters["total"]

    # ---- build the global entry list (every folder) ---------------------
    # Each entry carries its own output paths so the three stages below never
    # need to know which folder an image came from.
    entries = []
    ctx = {}            # target -> per-folder scratch dirs + its head rows
    for d in folders:
        images_dir = os.path.join(d, "images")
        target = os.path.dirname(images_dir)
        masks_dir = _out(os.path.join(target, "masks"))
        bl_masks_dir = _out(os.path.join(target, "blasteme_masks"))
        body_thumb_dir = _out(os.path.join(target, "thumbnails_all_body"))
        if save_bl_thumbs:
            thumb_dir = _out(os.path.join(target, "thumbnails_blastemes"))
            temp_thumb = None
        else:
            thumb_dir = os.path.join(target, "temp_thumbnails_blasteme")
            temp_thumb = thumb_dir
        # Scratch masks (written by stage 1/2, read back by stage 2/3, deleted
        # at the end) go to the LOCAL temp disk: under WSL the app folder is on
        # /mnt/c, where reading back a just-written file costs ~0.3 s/image
        # (measured in the stage-2 timing). FLATMORPHAI_SEG_SCRATCH_LOCAL=0
        # keeps them next to the images as before.
        if os.environ.get("FLATMORPHAI_SEG_SCRATCH_LOCAL", "1") in ("1", "true", "True"):
            import hashlib as _hl
            import tempfile as _tf
            _sc = os.path.join(_tf.gettempdir(), "flatmorphai_seg_scratch",
                               _hl.md5(os.path.abspath(target).encode("utf-8")).hexdigest()[:16])
            tmp_seg = os.path.join(_sc, "temp_segmentation")
            tmp_bl = os.path.join(_sc, "temp_blasteme_mask")
        else:
            tmp_seg = os.path.join(target, "temp_segmentation")
            tmp_bl = os.path.join(target, "temp_blasteme_mask")
        if save_body:
            os.makedirs(masks_dir, exist_ok=True)
        if save_bl_masks:
            os.makedirs(bl_masks_dir, exist_ok=True)
        if save_body_thumbs:
            os.makedirs(body_thumb_dir, exist_ok=True)
        if need_blasteme:
            os.makedirs(thumb_dir, exist_ok=True)
        ctx[target] = dict(head_rows=[], temp_thumb=temp_thumb,
                           tmp_seg=tmp_seg, tmp_bl=tmp_bl)
        made_tmp = False
        for n in list_images(images_dir):
            stem = os.path.splitext(n)[0]
            body_out = os.path.join(masks_dir, stem + ".png")
            bl_out = os.path.join(bl_masks_dir, stem + ".png")
            if not overwrite and not save_bl_thumbs and not need_head:
                body_done = (not save_body) or os.path.exists(body_out)
                bl_done = (not save_bl_masks) or os.path.exists(bl_out)
                if body_done and bl_done:
                    done_total += 1
                    continue
            if not made_tmp:
                os.makedirs(tmp_seg, exist_ok=True)
                if need_blasteme:
                    os.makedirs(tmp_bl, exist_ok=True)
                made_tmp = True
            entries.append(dict(
                name=n, stem=stem, images_dir=images_dir, target=target,
                body_out=body_out, bl_out=bl_out,
                seg_path=os.path.join(tmp_seg, stem + ".png"),
                bl_tmp=os.path.join(tmp_bl, stem + ".png"),
                thumb_dir=thumb_dir, body_thumb_dir=body_thumb_dir))
    progress(done_total, grand_total)
    if not entries:
        counters["done"] = done_total
        return

    def _read(images_dir, name):
        return cv2.imdecode(
            np.fromfile(os.path.join(images_dir, name), np.uint8),
            cv2.IMREAD_COLOR)

    def _read_mask(path):
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)

    try:
        # ---- stage 1: Mask2Former over EVERY folder ---------------------
        mask_refine.unpark_models(_SEG_PARKED)
        _SEG_PARKED.clear()
        mask_refine.vram_report("stage 1 start", log=log)
        parked = mask_refine.park_models([bl_mod, head_mods], log=log)
        seg_done = 0
        # Decode the next chunk on a worker thread while the GPU runs the
        # current one, so the card never waits for the disk.
        for chunk, decoded in mask_refine.decode_prefetch(
                entries, mask_refine.max_batch(configured_batch),
                lambda e: _read(e["images_dir"], e["name"]),
                interrupted=_interrupted):
            todo, bgrs = [], []
            for e, bgr in zip(chunk, decoded):
                if bgr is None:
                    log("  WARNING: unreadable image, skipped: %s" % e["name"])
                    done_total += 1
                    continue
                todo.append(e)
                bgrs.append(bgr)
            if not todo:
                continue
            masks, _working = mask_refine.segment_oom_safe(
                seg, bgrs, start_batch=configured_batch, log=log,
                suppress=_quiet_env.suppressed_output)
            for e, mask, _bgr in zip(todo, masks, bgrs):
                # Same opt-in cleanup as every other segmentation page
                # (fill holes / prune spurs / close / weld), off by default.
                # The frame is passed because the weld reads the image.
                mask = mask_refine.clean_seg_mask(mask, _bgr)
                cv2.imwrite(e["seg_path"], mask)
            seg_done += len(todo)
            del bgrs, masks, todo
            # Segmentation is the long stage; counted at 60% of an image's work
            # so the bar still only ever grows when stage 3 takes over.
            progress(done_total + int(0.6 * seg_done), grand_total)
        mask_refine._free_vram_mb_after_cleanup()
        mask_refine.vram_report("stage 1 done", log=log)

        # ---- stage 2: blastema over EVERY folder ------------------------
        if need_blasteme:
            _SEG_PARKED.extend(mask_refine.park_models([seg.model], log=log))
            mask_refine.unpark_models(parked)
            import time as _tm
            _t2 = {"read": 0.0, "detect": 0.0, "write": 0.0, "n": 0}
            for e in entries:
                if _interrupted():
                    return
                if not os.path.exists(e["seg_path"]):
                    continue
                _a = _tm.time()
                mask = _read_mask(e["seg_path"])
                bgr = _read(e["images_dir"], e["name"])
                _b = _tm.time(); _t2["read"] += _b - _a
                if bgr is None or mask is None:
                    continue
                bmask = _detect_blasteme(
                    bl_mod, bgr, mask, e["thumb_dir"], e["stem"], cv2,
                    keep_biggest, save_bl_thumbs)
                _c = _tm.time(); _t2["detect"] += _c - _b
                if bmask is not None:
                    cv2.imwrite(e["bl_tmp"], bmask)
                _t2["write"] += _tm.time() - _c; _t2["n"] += 1
                del bgr, mask, bmask
            if _t2["n"]:
                log("  stage 2 timing per image: read %.3f s | blastema detection %.3f s | write %.3f s"
                    % (_t2["read"] / _t2["n"], _t2["detect"] / _t2["n"], _t2["write"] / _t2["n"]))
            mask_refine._free_vram_mb_after_cleanup()
            mask_refine.vram_report("stage 2 done", log=log)
        else:
            mask_refine.unpark_models(parked)

        # ---- stage 3: union, head detection, final writes --------------
        import time as _tm3
        _t3 = {"head": 0.0, "n": 0, "t0": _tm3.time()}
        for e in entries:
            if _interrupted():
                return
            if not os.path.exists(e["seg_path"]):
                done_total += 1
                continue
            mask = _read_mask(e["seg_path"])
            if mask is None:
                done_total += 1
                continue
            bmask = _read_mask(e["bl_tmp"]) if os.path.exists(e["bl_tmp"]) else None
            if refine and bmask is not None:
                mask = cv2.bitwise_or(mask, bmask)
            if save_body:
                cv2.imwrite(e["body_out"], mask)
            if save_bl_masks and bmask is not None:
                cv2.imwrite(e["bl_out"], bmask)
            if need_head:
                bgr = _read(e["images_dir"], e["name"])
                if bgr is not None:
                    _h0 = _tm3.time()
                    row = _detect_head(
                        head_mods, bgr, mask, e["body_thumb_dir"], e["stem"],
                        cv2, save_body_thumbs, save_head_csv)
                    _t3["head"] += _tm3.time() - _h0; _t3["n"] += 1
                    if row is not None:
                        row["image_name"] = e["name"]
                        ctx[e["target"]]["head_rows"].append(row)
                del bgr
            del mask, bmask
            done_total += 1
            if done_total % 10 == 0:
                progress(done_total, grand_total)
        progress(done_total, grand_total)
        if _t3["n"]:
            log("  stage 3 timing per image: head detection %.3f s | everything else %.3f s"
                % (_t3["head"] / _t3["n"], (_tm3.time() - _t3["t0"] - _t3["head"]) / _t3["n"]))
        mask_refine._free_vram_mb_after_cleanup()
        mask_refine.vram_report("stage 3 done", log=log)
    finally:
        # Always remove the temporary blastema thumbnails (kept ones stay) and
        # the per-stage scratch folders, for every folder touched.
        for _target, c in ctx.items():
            if c["temp_thumb"] and os.path.isdir(c["temp_thumb"]):
                shutil.rmtree(c["temp_thumb"], ignore_errors=True)
            for scratch in (c["tmp_seg"], c["tmp_bl"]):
                if os.path.isdir(scratch):
                    shutil.rmtree(scratch, ignore_errors=True)

    if save_head_csv:
        for _target, c in ctx.items():
            if c["head_rows"]:
                _write_head_csv(_out(os.path.join(_target, "head_position.csv")),
                                c["head_rows"])
    counters["done"] = done_total


def _detect_head(head_mods, bgr, mask, body_thumb_dir, stem, cv2,
                 save_body_thumbs, save_head_csv):
    """Run head detection; save the full body thumbnail and/or return a CSV row.

    Mirrors the Regeneration pipeline: the composite channel 0 is the
    straightened whole-body thumbnail, and the two extremity positions (mapped
    back to the original frame) plus which end is the head make the CSV row.
    Returns ``{up_position, bottom_position, head_position}`` or ``None``.
    """
    try:
        h = head_mods["head"].detect_head(bgr, mask, with_composite=True)
    except Exception as exc:  # noqa: BLE001
        log("  WARNING: head detection failed on %s (%s)" % (stem, exc))
        return None
    if not h or not h.get("ok"):
        if h:
            log("  head skipped for %s: %s" % (stem, h.get("reason", "?")))
        return None
    if save_body_thumbs:
        comp = h.get("composite")
        if comp is not None:
            try:
                cv2.imwrite(os.path.join(body_thumb_dir, stem + ".png"),
                            comp[:, :, 0])
            except Exception:  # noqa: BLE001
                pass
    if not save_head_csv:
        return None
    try:
        ac = head_mods["ac"]
        pr = head_mods["pr"]
        pts, _ends = ac.ends(h["rm"])
        M = h["M"]
        end = int(h["end"])
        up_o = pr.to_original(pts[0][0], pts[0][1], M)
        bo_o = pr.to_original(pts[1][0], pts[1][1], M)
        return {
            "up_position": "(%.1f, %.1f)" % (up_o[0], up_o[1]),
            "bottom_position": "(%.1f, %.1f)" % (bo_o[0], bo_o[1]),
            "head_position": "up" if end == 0 else "down",
        }
    except Exception as exc:  # noqa: BLE001
        log("  WARNING: head position failed on %s (%s)" % (stem, exc))
        return None


def _write_head_csv(path, rows):
    """Write head_position.csv with the shared ';' separator."""
    import csv as _csv
    cols = ["image_name", "up_position", "bottom_position", "head_position"]
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, delimiter=";")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
    except Exception as exc:  # noqa: BLE001
        log("  WARNING: could not write head_position.csv (%s)" % exc)


def _detect_blasteme(bl_mod, bgr, mask, thumb_dir, stem, cv2,
                     keep_biggest, save_thumbs):
    """Run the blastema detector; return its 0/255 mask (original frame), or None.

    ``bl_mod.detect_blastema`` straightens the worm from ``mask``, segments the
    blastema at the two extremities and returns ``blastema_mask_original``
    together with the two end thumbnails. The thumbnails are saved when
    ``save_thumbs`` is true; the mask keeps only its largest component when
    ``keep_biggest`` is true.
    """
    try:
        res = bl_mod.detect_blastema(bgr, mask)
    except Exception as exc:  # noqa: BLE001 - never abort the run on a failure
        log("  WARNING: blastema detection failed on %s (%s)" % (stem, exc))
        return None
    if not res.get("ok"):
        log("  blastema skipped for %s: %s" % (stem, res.get("reason", "?")))
        return None
    if save_thumbs:
        for end_idx, end in enumerate(res.get("ends", [])):
            thumb = end.get("thumbnail")
            if thumb is not None:
                suffix = "top" if end_idx == 0 else "bottom"
                cv2.imwrite(
                    os.path.join(thumb_dir, "%s_%s.png" % (stem, suffix)), thumb
                )
    bmask = res.get("blastema_mask_original")
    if bmask is None:
        return None
    if keep_biggest:
        bmask = mask_refine.keep_largest_component(bmask)
    return bmask


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    global _CANCEL_FILE
    raw = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    cfg = json.loads(raw)
    _CANCEL_FILE = cfg.get("cancel_file")
    export_root = cfg["export_root"]
    root = cfg["root"]

    _add_export_to_path(export_root)

    # --- light phase (no torch): folders, CLAHE and checkpoint settings ---
    import importlib
    reader = importlib.import_module("parameter_reader")

    # Image segmentation uses the application's own segmentation checkpoint (the one
    # resolved from the Settings page) so its masks match Pipelines A and B.
    # We point the bundle's models.csv at it for the duration of the run and
    # restore the column afterwards.  The CLAHE toggle is written the same way
    # and is independent from the regeneration pipeline's own setting.
    seg_ckpt = cfg.get("seg_checkpoint", "")
    prev_model = None
    if seg_ckpt:
        prev_model = reader.read_table("models")
        reader.write_correction("models", "mask2former_segmentation", seg_ckpt)
    # CLAHE before segmentation is disabled application-wide: the full-body
    # segmentation never runs CLAHE anymore (it could turn reflections into
    # false objects). Force the bundle flag OFF regardless of any legacy config.
    reader.write_correction("clahe_segmentation", "ENABLE_CLAHE", False)

    def _restore():
        try:
            reader.write_correction("clahe_segmentation", "ENABLE_CLAHE", None)
        except Exception:  # noqa: BLE001
            pass
        if prev_model is not None:
            try:
                # Restore the original corrected_value of the model role.
                orig = ""
                for r in prev_model:
                    if r["parameter"].strip() == "mask2former_segmentation":
                        orig = r.get("corrected_value", "") or ""
                reader.write_correction("models", "mask2former_segmentation",
                                        orig if orig else None)
            except Exception:  # noqa: BLE001
                pass

    try:
        folders = find_image_folders(root)
        if not folders:
            _emit("ERROR no 'images' folder found under the processing root")
            return
        log("%d images folder(s) found" % len(folders))
        grand_total = sum(len(list_images(os.path.join(d, "images"))) for d in folders)
        if grand_total == 0:
            _emit("ERROR the 'images' folders contain no readable image")
            return
        log("%d image(s) to segment in total" % grand_total)

        # --- heavy phase: segmenter and (optional) blastema model ---
        # Wrapped so the model-loading chatter does not reach the log.
        from segment_mask import Segmenter  # noqa: E402 (Export on sys.path)
        # The blastema model is needed for refinement AND for saving the
        # blastema masks / thumbnails.
        need_bl = bool(
            cfg.get("blasteme_refined")
            or cfg.get("save_blasteme_masks")
            or cfg.get("save_blasteme_thumbnails")
        )
        # Head detection is needed for the head-position CSV and the full body
        # (straightened) thumbnails.
        need_head = bool(
            cfg.get("save_head_csv") or cfg.get("save_body_thumbnails")
        )
        with _quiet_env.suppressed_output():
            mask_refine.preflight_vram(log=log)
            seg = Segmenter(verbose=False)
            bl_mod = importlib.import_module("blastema_detection") if need_bl else None
            head_mods = None
            if need_head:
                head_mods = {
                    "pr": importlib.import_module("preprocessing"),
                    "ac": importlib.import_module("anchor_cascade"),
                    "head": importlib.import_module("head_detection"),
                }
        if cfg.get("blasteme_refined"):
            log("Full body mask refined with the blastema mask: ON (union on "
                "both extremities)")
        if cfg.get("save_blasteme_masks"):
            log("Saving blastema masks to 'blasteme_masks/'")
        if cfg.get("save_head_csv"):
            log("Saving head positions to 'head_position.csv'")
        if cfg.get("save_body_thumbnails"):
            log("Saving full body thumbnails to 'thumbnails_all_body/'")
        if not cfg.get("save_body_masks", True):
            log("Full body masks will NOT be saved (blastema outputs only)")

        counters = {"done": 0, "total": grand_total}

        # Final outputs on the Linux disk when the root is on /mnt/<x>.
        _WORK["root"] = root
        _WORK["work"] = linux_workdir.work_path(root)
        if _WORK["work"]:
            try:
                if os.path.isdir(_WORK["work"]):      # leftovers of an interrupted run
                    shutil.rmtree(_WORK["work"], ignore_errors=True)
                os.makedirs(_WORK["work"], exist_ok=True)
                log("Writing the masks on the Linux disk, copied back to %s at "
                    "the end of the run" % root)
            except Exception as exc:  # noqa: BLE001
                log("  Linux work folder unavailable (%s); writing in place" % exc)
                _WORK["work"] = None

        # Feed the model at its training resolution (320x320 for this
        # checkpoint). segment_mask feeds pass 1 the whole image, so without
        # this the processor resizes a 1600x1200 plate to ~1088x800 -- nine
        # times the pixels the model trained on, which is what overflows the
        # card. Done once, from the first readable frame.
        # Same policy as every other page: feed the model at its 320x320
        # training input (both passes), so the four pages are one pipeline.
        # See configure_fed_size; FLATMORPHAI_SEG_FED overrides.
        _sample = _first_sample_bgr(folders)
        if _sample is not None:
            mask_refine.configure_fed_size(seg, _sample, target=320, log=log)
        else:
            log("  Fed resolution: no readable frame to calibrate from; the "
                "processor keeps its default input size.")

        if _interrupted():
            _emit("ERROR cancelled")
            return
        log("Processing %d folder(s) as one staged run: all segmentations, "
            "then all blastemas, then all head detection." % len(folders))
        # Per-folder mode (default): each video goes through the three stages
        # and is announced as soon as its final masks are written, so the
        # descriptor stage can start on it while the GPU segments the next one.
        # Same functions, same outputs. FLATMORPHAI_SEG_PER_FOLDER=0 -> one
        # staged run over all folders as before.
        if os.environ.get("FLATMORPHAI_SEG_PER_FOLDER", "1") in ("1", "true", "True") and len(folders) > 1:
            for d in folders:
                if _interrupted():
                    break
                process_all([d], cfg, seg, bl_mod, counters, head_mods)
                log("Folder segmented: %s" % os.path.abspath(d))
        else:
            process_all(folders, cfg, seg, bl_mod, counters, head_mods)
        if _interrupted():
            _emit("ERROR cancelled")
            return
        if _WORK["work"]:
            linux_workdir.sync_back(_WORK["work"], root, log)
        _emit("DONE %d folder(s) processed, %d mask(s) written"
              % (len(folders), counters["done"]))
    finally:
        _restore()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        _emit("ERROR %s: %s" % (type(exc).__name__, exc))
        mask_refine.release_gpu_on_crash(log=log)
        sys.exit(1)
