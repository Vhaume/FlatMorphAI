# -*- coding: utf-8 -*-
"""Driver for the regeneration-states pipeline (runs in a dedicated subprocess).

Why a subprocess.  The regeneration models live in the self-contained folder
``Export_head_detection`` (README there).  Several of its modules use bare,
path-based imports and carry names that clash with the main application
(``evaluate_test_images_v2``, ``mask2former_api``, ``preprocessing``,
``anchor_cascade``).  Importing them inside the running Qt process would pick
up the wrong module from ``sys.modules`` and the GPU/thread setup would fight
the application's own Mask2Former.  Running this driver as a fresh subprocess
gives it an empty ``sys.modules`` so the ``Export_head_detection`` versions
load cleanly, isolates the CUDA context and lets the worker cancel the run by
terminating the process.

The driver reads one JSON config on ``argv[1]`` (or stdin), discovers every
``images/`` folder under the processing root, segments the frames with the
bundle's Mask2Former (two-pass, raw input), then for each image runs the
head cascade (eye cups, blastema, darkness), builds the state composite and
predicts the regeneration state.  It writes, next to each ``images/`` folder,
``eyespots_detection.csv``, ``head_position.csv``, ``regeneration_states.csv``
and, on request, the blastema CSV, the blastema masks, the eyespot heat maps
and the four thumbnail folders.

It prints machine-readable lines on stdout for the worker:
    LOG <message>
    PROGRESS <done> <total>
    DONE / ERROR <message>
Every other line is treated as a plain log line.

CSV separator ";".  Seed 42.  A GPU is required in practice for the models.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import traceback

# ``eyespot_shared`` lives next to this driver. Make sure the driver's own
# directory is importable whether this module is run as a script (script dir is
# already on sys.path) or imported from elsewhere for testing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _quiet_env  # noqa: E402,F401  (silence noisy third-party notices first)
import eyespot_shared  # noqa: E402
import mask_refine  # noqa: E402


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


# ---------------------------------------------------------------------
# Image-name convention: Species_Compound_Concentration_Plate_Position_Date[_Comment].<ext>
# ---------------------------------------------------------------------
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")


#: The ordered meta fields carried by an image name, each stored in its own CSV
#: column. ``comment`` is the only optional one.
META_FIELDS = ("species", "compound", "concentration", "plate", "position",
               "date", "comment")


def empty_meta():
    """A meta dict with every field blank (used when a name does not parse)."""
    return {k: "" for k in META_FIELDS}


def parse_image_name(filename: str):
    """Parse the image name into its meta fields, or None when it does not
    carry the six mandatory underscore-separated fields.

    Convention: ``Species_Compound_Concentration_Plate_Position_Date[_Comment].<ext>``.
    Every image extension is accepted. The six fields before the optional
    comment are mandatory; the comment is everything after them and may itself
    contain underscores.
    """
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


def _date_key(date_token: str):
    """Best-effort sort key for the DATE field so images of one planaria are
    ordered chronologically.  Tries DD MM YY, YYYY MM DD and YYYY-MM-DD, then
    falls back to the raw string.  Change here if your dates use another format.
    """
    t = str(date_token).strip()
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", t)      # DDMMYY
    if m:
        d, mo, y = (int(g) for g in m.groups())
        return (2000 + y, mo, d)
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t)      # YYYYMMDD
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return (y, mo, d)
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)  # YYYY-MM-DD
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return (y, mo, d)
    return (9999, 99, 99, t)


def _date_ordinal(date_token):
    """Proleptic day number of the DATE field, or None when unparseable.

    Used to compute the ``Day`` column as a real calendar-day difference: the
    oldest date of a planaria is Day 1 and a date two calendar days later is
    Day 3 (it counts the number of days, not the acquisition rank). Returns
    ``datetime.date.toordinal()`` so a plain subtraction gives the day gap.
    """
    import datetime

    key = _date_key(str(date_token))
    if len(key) != 3:                      # the (9999, 99, 99, raw) fallback
        return None
    y, mo, d = key
    try:
        return datetime.date(y, mo, d).toordinal()
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------
_SKIP_DIRS = ("masks", "temp_formated", "eyespots heatmaps", "blasteme_masks")
_SKIP_PREFIXES = ("masks_", "thumbnails_", "segmented_")


def find_image_folders(root: str):
    """Every folder containing an ``images/`` sub-directory, at any depth."""
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
    """Prepend every Export_head_detection sub-package to sys.path so its own
    (bare-import) modules resolve from this fresh subprocess."""
    for d in ("common", "parameters", "eyes", "blastema", "darkness",
              "head", "segmentation", "stage"):
        p = os.path.join(export_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_reader():
    """Light import (no torch): parameter reader only, for the bundle settings."""
    import importlib
    return importlib.import_module("parameter_reader")


def _import_models():
    """Heavy imports (torch, segmentation-models-pytorch): the detection modules."""
    import importlib
    mods = {}
    mods["pr"] = importlib.import_module("preprocessing")
    mods["ac"] = importlib.import_module("anchor_cascade")
    mods["head"] = importlib.import_module("head_detection")
    mods["eye"] = importlib.import_module("eye_detection")
    mods["bl"] = importlib.import_module("blastema_detection")
    mods["dk"] = importlib.import_module("darkness_index")
    mods["reader"] = importlib.import_module("parameter_reader")
    return mods


# ---------------------------------------------------------------------
# Eyespot peaks at a given threshold (scipy, independent of the cascade)
# ---------------------------------------------------------------------
def _peaks_at(heatmap, thr, min_dist):
    """Heights of the local maxima of ``heatmap`` (384 x 384 in [0, 1]) that
    reach ``thr``, window 2*min_dist+1.  Returns a list of floats (descending)."""
    import numpy as np
    from scipy.ndimage import maximum_filter
    size = 2 * int(min_dist) + 1
    mx = maximum_filter(heatmap, size=size, mode="nearest")
    ys, xs = np.nonzero((heatmap >= mx) & (heatmap >= float(thr)))
    vals = sorted((float(heatmap[y, x]) for y, x in zip(ys, xs)), reverse=True)
    return vals


# ---------------------------------------------------------------------
# Main per-folder analysis
# ---------------------------------------------------------------------
def _free_gpu(*objs):
    """Drop references to GPU models and release cached CUDA memory."""
    try:
        import gc
        import torch
        for _o in objs:
            del _o
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def process_folder(images_dir, cfg, mods, stage_clf=None):
    import numpy as np
    import cv2

    pr, ac, HEAD, EYE, BL, DK, reader = (
        mods["pr"], mods["ac"], mods["head"], mods["eye"],
        mods["bl"], mods["dk"], mods["reader"],
    )
    target = os.path.dirname(images_dir)                 # folder holding images/
    names = list_images(images_dir)
    # Body masks are always written to a plain "masks" folder next to images/
    # (same naming as every other pipeline).
    masks_dir = os.path.join(target, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # thumbnail folders: create only the ones the user keeps
    thumb_dirs = {}
    for key, flag, folder in (
        ("all_body", "save_thumb_all_body", "thumbnails_all_body"),
        ("head_zoomed", "save_thumb_head_zoomed", "thumbnails_head_zoomed"),
        ("regeneration", "save_thumb_regeneration", "thumbnails_regeneration"),
        ("blastemes", "save_thumb_blastemes", "thumbnails_blastemes"),
        ("eyespots", "save_thumb_eyespots", "thumbnails_eyespots"),
    ):
        d = os.path.join(target, folder)
        thumb_dirs[key] = d
        if cfg.get(flag):
            os.makedirs(d, exist_ok=True)
    if cfg.get("save_eyespots_heatmaps"):
        os.makedirs(os.path.join(target, "Eyespots heatmaps"), exist_ok=True)
    if cfg.get("save_blasteme_masks"):
        os.makedirs(os.path.join(target, "blasteme_masks"), exist_ok=True)

    # thresholds for the two eyespot counts (column names carry the value).
    # The eyespots are counted on BOTH extremities and their (x, y) positions
    # are saved, using the shared helper so this CSV has the exact same schema
    # as the Eyespots counter page - the Regeneration states editor reads it.
    t_bal = float(cfg.get("eyespots_balanced_threshold", 0.80))
    t_pos = float(cfg.get("eyespots_positive_threshold", 0.60))
    col_bal, col_bal_pos, col_pos, col_pos_pos = eyespot_shared.eyespot_columns(
        t_bal, t_pos
    )

    rows_cup, rows_head, rows_blast, rows_state = [], [], [], []
    total = len(names)
    log("  %d image(s) in %s" % (total, os.path.relpath(images_dir, cfg["root"])))

    # Optional mask refinement: grow the mask with the blastema prediction.
    refine_blasteme = bool(cfg.get("blasteme_refined"))
    blasteme_keep_biggest = bool(cfg.get("blasteme_keep_biggest"))

    # Three visible steps (segmentation -> head detection -> state
    # classification). STEP markers let the page advance its timeline.
    # Progress runs over three equal thirds (total * 3).

    # --- Step 1: segmentation of every frame into masks/ ---
    _emit("STEP 0")
    from segment_mask import Segmenter  # noqa: E402 (Export on sys.path)
    seg = globals().get("_SEG")
    if seg is None:
        with _quiet_env.suppressed_output():
            seg = Segmenter(verbose=False)
        globals()["_SEG"] = seg
    # Adaptive batch: start at the configured BATCH and let segment_oom_safe
    # shrink it only if CUDA runs out of memory. Now that the GUI keeps a zero
    # GPU footprint, this subprocess has the whole card, matching the
    # in-process stereotype pipeline (so batch 8 fits where it used to OOM).
    configured_batch = max(1, int(reader.read("clahe_segmentation").get("BATCH", 8)))
    _free = mask_refine._free_vram_mb_after_cleanup()
    if _free is not None:
        log("  GPU free memory before segmentation: %d MiB" % _free)
    done_seg = 0
    while done_seg < total:
        if _interrupted():
            return None
        # Always request the configured batch for the chunk. The cache is
        # freed after each chunk (below), so there is no reason to carry a
        # once-shrunk batch forward -- that is what made the run crawl at 1.
        chunk = names[done_seg:done_seg + configured_batch]
        imgs = [pr.load_image(os.path.join(images_dir, n)) for n in chunk]
        seg_masks, _working = mask_refine.segment_oom_safe(
            seg, imgs, start_batch=configured_batch, log=log,
            suppress=_quiet_env.suppressed_output)
        for n, bgr_img, m in zip(chunk, imgs, seg_masks):
            if refine_blasteme:
                m = mask_refine.union_blasteme(
                    BL, bgr_img, m, keep_biggest=blasteme_keep_biggest
                )
            cv2.imwrite(os.path.join(masks_dir, os.path.splitext(n)[0] + ".png"), m)
        done_seg += len(chunk)
        progress(done_seg, total * 3)
        # Release the CUDA cache after every chunk. Without this, cached/
        # fragmented memory piles up across chunks and eventually triggers an
        # out-of-memory error, which makes segment_oom_safe halve the batch
        # again and again until it crawls at 1 image. Freeing here keeps the
        # footprint flat so the batch stays at its healthy size for the whole
        # run.
        del imgs, seg_masks
        mask_refine._free_vram_mb_after_cleanup()

    # Segmentation done: free the Mask2Former model NOW (not at the end of the
    # run) so the head-detection U-Nets and the heavy state classifier loaded
    # next have the GPU to themselves.
    globals().pop("_SEG", None)
    _free_gpu(seg)
    seg = None
    log("  freed the segmentation model (GPU memory released)")

    # --- Step 2: head detection, eyespots, blastema, composite ---
    # The state composite of every image is kept so the heavier state
    # classification can run as its own clearly-separated step afterwards.
    _emit("STEP 1")
    to_classify = []
    for idx, name in enumerate(names):
        if _interrupted():
            return None
        meta = parse_image_name(name) or empty_meta()
        # Species / Compound / Concentration (+ optional Comment) are new
        # dedicated columns carried by every per-image CSV row.
        meta_cols = {
            "species": meta["species"], "compound": meta["compound"],
            "concentration": meta["concentration"], "comment": meta["comment"],
        }
        stem = os.path.splitext(name)[0]
        bgr = pr.load_image(os.path.join(images_dir, name))
        mpath = os.path.join(masks_dir, stem + ".png")
        mask = pr.load_mask(mpath) if os.path.exists(mpath) else None
        if mask is None:
            log("  WARNING: no mask for %s, skipped" % name)
            continue
        h = HEAD.detect_head(bgr, mask, with_composite=True)
        if not h.get("ok"):
            log("  %s skipped: %s" % (name, h.get("reason", "?")))
            continue
        rbgr, rm, span, M, end = h["rbgr"], h["rm"], h["span"], h["M"], int(h["end"])

        # --- eyespots (eye cups) on BOTH extremities ---
        # One model pass (at the lower threshold) gives the heat map of each
        # end; the shared helper counts the peaks of both ends at the two
        # thresholds and returns their (x, y) positions in the original frame.
        ey = EYE.eye_maps(rbgr, rm, span, threshold=min(t_bal, t_pos))
        pos_bal, pos_pos = eyespot_shared.count_from_ends(ey, pr, M, t_bal, t_pos)
        rows_cup.append({
            "image_name": name, "plate": meta["plate"], "position": meta["position"],
            "date": meta["date"], **meta_cols,
            col_bal: len(pos_bal), col_bal_pos: eyespot_shared.positions_str(pos_bal),
            col_pos: len(pos_pos), col_pos_pos: eyespot_shared.positions_str(pos_pos),
        })
        if cfg.get("save_eyespots_heatmaps"):
            for end_idx, e in enumerate(ey):
                if e.get("heatmap") is not None:
                    suffix = "top" if end_idx == 0 else "bottom"
                    cv2.imwrite(
                        os.path.join(target, "Eyespots heatmaps",
                                     "%s_%s.png" % (stem, suffix)),
                        (255 * np.clip(e["heatmap"], 0, 1)).astype(np.uint8))
        if cfg.get("save_thumb_eyespots") and ey[end]["thumbnail"] is not None:
            cv2.imwrite(os.path.join(thumb_dirs["eyespots"], stem + ".png"),
                        ey[end]["thumbnail"])

        # --- blastema ---
        bl = BL.blastema_maps(rbgr, rm, span)
        area_up = int(bl[0]["binary_area_px"]); area_down = int(bl[1]["binary_area_px"])
        if cfg.get("save_blasteme_csv"):
            rows_blast.append({
                "image_name": name, "plate": meta["plate"], "position": meta["position"],
                "date": meta["date"], **meta_cols,
                "surface_area_up": area_up, "surface_area_down": area_down,
            })
        if cfg.get("save_blasteme_masks"):
            br = BL.detect_blastema(bgr, mask)
            if br.get("ok"):
                bm = br["blastema_mask_original"]
                if blasteme_keep_biggest:
                    bm = mask_refine.keep_largest_component(bm)
                cv2.imwrite(os.path.join(target, "blasteme_masks", stem + ".png"),
                            bm)
        if cfg.get("save_thumb_blastemes"):
            tb = bl[end]["thumbnail"]
            if tb is not None:
                cv2.imwrite(os.path.join(thumb_dirs["blastemes"], stem + ".png"), tb)

        # --- head position (medial axis / terminal bands, original frame) ---
        pts, _ = ac.ends(rm)               # [(cx, cy) top, (cx, cy) bottom] straightened
        up_o = pr.to_original(pts[0][0], pts[0][1], M)
        bo_o = pr.to_original(pts[1][0], pts[1][1], M)
        rows_head.append({
            "image_name": name, "plate": meta["plate"], "position": meta["position"],
            "date": meta["date"], **meta_cols,
            "up_position": "(%.1f, %.1f)" % (up_o[0], up_o[1]),
            "bottom_position": "(%.1f, %.1f)" % (bo_o[0], bo_o[1]),
            "head_position": "up" if end == 0 else "down",
        })

        # --- state composite (prediction deferred to Step 3) ---
        comp = h["composite"]              # BGR 384 (c0, c1, c2)
        if cfg.get("save_thumb_all_body"):
            cv2.imwrite(os.path.join(thumb_dirs["all_body"], stem + ".png"), comp[:, :, 0])
        if cfg.get("save_thumb_head_zoomed"):
            cv2.imwrite(os.path.join(thumb_dirs["head_zoomed"], stem + ".png"), comp[:, :, 1])
        if cfg.get("save_thumb_regeneration"):
            cv2.imwrite(os.path.join(thumb_dirs["regeneration"], stem + ".png"), comp)
        state_row = {
            "image_name": name, "index_planaria": "", "plate": meta["plate"],
            "position": meta["position"], "date": meta["date"], **meta_cols,
            "state_round": "", "state_decimal": "",
            "state_round_corrected": "", "state_corrected": "",
        }
        rows_state.append(state_row)
        # Keep the balanced eye count alongside the composite so the state-4
        # rule below can require at least one detected eye. Private key, not a
        # written column.
        to_classify.append((state_row, comp, len(pos_bal)))
        progress(total + idx + 1, total * 3)
        # Periodically release the CUDA cache so head/eyespot/blastema inference
        # does not let fragmented memory build up across the whole folder.
        if (idx + 1) % 8 == 0:
            mask_refine._free_vram_mb_after_cleanup()

    # --- Step 3: regeneration state classification of each composite ---
    # Load the state classifier ONLY now (after the segmenter was freed), so
    # the heavy ConvNeXt has the GPU to itself, then free it before the next
    # folder / the end of the run.
    _emit("STEP 2")
    if stage_clf is None and to_classify:
        try:
            from stage_prediction import StageClassifier  # noqa: E402
            with _quiet_env.suppressed_output():
                stage_clf = StageClassifier(verbose=False)
        except Exception as exc:  # noqa: BLE001
            log("  state classifier unavailable (%s); state columns left empty" % exc)
            stage_clf = None
    for i, (state_row, comp, eye_count) in enumerate(to_classify):
        if _interrupted():
            _free_gpu(stage_clf)
            return None
        if stage_clf is not None:
            sp = stage_clf.predict(comp, {})     # metrics defaulted to training medians
            decimal = float(sp["decimal_stage"])
            # Round to the nearest integer (half up): 4.2 -> 4, 4.7 -> 5.
            round_val = int(min(6, max(1, int(np.floor(decimal + 0.5)))))
            # State 4 is the first state with regenerated eyes, so a frame only
            # qualifies as state 4 when at least one eye cup was detected;
            # otherwise it is held one state below.
            if round_val == 4 and int(eye_count) < 1:
                round_val = 3
            state_row["state_round"] = round_val
            state_row["state_decimal"] = decimal
            state_row["state_round_corrected"] = round_val
        progress(total * 2 + i + 1, total * 3)
        if (i + 1) % 8 == 0:
            mask_refine._free_vram_mb_after_cleanup()
    # Free the state classifier now, not at the end of the whole run.
    _free_gpu(stage_clf)
    stage_clf = None

    # 3) index_planaria, Day and monotonic state correction.
    # One planaria = one DISTINCT worm = one (plate, position, species,
    # compound, concentration). Plate+position alone is NOT enough: the same
    # well position on the same plate can hold two different treatments (e.g.
    # Caf 500 and Ctrl 0) photographed on the same dates, which are two
    # different worms - grouping on plate+position only would merge them into
    # one index_planaria and produce "duplicate" days and zig-zag graphs.
    # The index starts at 1 and follows the sorted order of the identity key.
    # Within a planaria the images are sorted by date: the oldest is Day 1 and
    # every later date is numbered by its calendar-day gap to that oldest date
    # (a date two days later is Day 3), so Day counts days, not acquisition rank.
    def _worm_key(r):
        return (
            str(r["plate"]), str(r["position"]),
            str(r.get("species", "")), str(r.get("compound", "")),
            str(r.get("concentration", "")),
        )

    groups = {}
    for r in rows_state:
        groups.setdefault(_worm_key(r), []).append(r)
    for i, key in enumerate(sorted(groups), start=1):
        members = sorted(groups[key], key=lambda r: _date_key(r["date"]))
        ords = [_date_ordinal(r["date"]) for r in members]
        valid = [o for o in ords if o is not None]
        base = min(valid) if valid else None
        for rank, (r, o) in enumerate(zip(members, ords), start=1):
            r["index_planaria"] = i
            # Calendar-day number when the date parses, else the 1-based rank.
            r["Day"] = (o - base + 1) if (base is not None and o is not None) else rank
        if cfg.get("auto_state_increase_correction"):
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

    # 3b) monotonic eyespot-count correction (optional). An eye cup, once
    # regenerated, does not disappear, so the balanced eyespot count must not
    # decrease over time within a worm. A transient under-count on one day is
    # lifted to the running maximum. The raw count stays in ``col_bal``; the
    # corrected value is written to ``col_bal + "_corrected"`` so the editor
    # can show the monotonic curve while the raw detection is preserved.
    col_bal_corr = col_bal + "_corrected"
    if cfg.get("auto_eyespots_increase_correction"):
        cup_groups = {}
        for r in rows_cup:
            cup_groups.setdefault(_worm_key(r), []).append(r)
        for key in cup_groups:
            members = sorted(cup_groups[key], key=lambda r: _date_key(r["date"]))
            running = None
            for r in members:
                try:
                    v = int(r.get(col_bal, 0) or 0)
                except (TypeError, ValueError):
                    v = 0
                if running is not None and v < running:
                    v = running
                r[col_bal_corr] = v
                running = v
    else:
        for r in rows_cup:
            r[col_bal_corr] = r.get(col_bal, 0)

    # 4) write the CSVs next to images/
    def _write(path, rows, columns):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, delimiter=";")
            w.writeheader()
            w.writerows(rows)
        log("  wrote %s (%d rows)" % (os.path.basename(path), len(rows)))

    meta_hdr = ["species", "compound", "concentration", "comment"]
    _write(os.path.join(target, "eyespots_detection.csv"), rows_cup,
           ["image_name", "plate", "position", "date"] + meta_hdr
           + [col_bal, col_bal_corr, col_bal_pos, col_pos, col_pos_pos])
    _write(os.path.join(target, "head_position.csv"), rows_head,
           ["image_name", "plate", "position", "date"] + meta_hdr
           + ["up_position", "bottom_position", "head_position"])
    if cfg.get("save_blasteme_csv"):
        _write(os.path.join(target, "blasteme_detection.csv"), rows_blast,
               ["image_name", "plate", "position", "date"] + meta_hdr
               + ["surface_area_up", "surface_area_down"])
    _write(os.path.join(target, "regeneration_states.csv"), rows_state,
           ["image_name", "index_planaria", "plate", "position", "date"] + meta_hdr
           + ["Day", "state_round", "state_decimal", "state_round_corrected", "state_corrected"])

    # 5) remove the thumbnail folders the user did not keep (empty ones)
    for key, flag in (("all_body", "save_thumb_all_body"),
                      ("head_zoomed", "save_thumb_head_zoomed"),
                      ("regeneration", "save_thumb_regeneration"),
                      ("blastemes", "save_thumb_blastemes"),
                      ("eyespots", "save_thumb_eyespots")):
        if not cfg.get(flag):
            d = thumb_dirs[key]
            if os.path.isdir(d) and not os.listdir(d):
                try:
                    os.rmdir(d)
                except OSError:
                    pass

    # Body masks are always written to masks_<model>/ because head detection,
    # eyespot counting and blastema detection all read them. When the user did
    # not ask to keep them, drop the whole folder now that every step is done.
    if not cfg.get("save_body_masks", True):
        if os.path.isdir(masks_dir):
            try:
                shutil.rmtree(masks_dir)
                log("  removed %s/ (Save body masks was off)"
                    % os.path.basename(masks_dir))
            except OSError:
                pass
    return len(names)


# ---------------------------------------------------------------------
# cancellation flag via a sentinel file written by the worker
# ---------------------------------------------------------------------
_CANCEL_FILE = None


def _interrupted():
    return bool(_CANCEL_FILE) and os.path.exists(_CANCEL_FILE)


def main():
    global _CANCEL_FILE
    raw = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    cfg = json.loads(raw)
    _CANCEL_FILE = cfg.get("cancel_file")
    export_root = cfg["export_root"]
    root = cfg["root"]

    _add_export_to_path(export_root)

    # --- light phase (no torch): folders, naming ---
    # CLAHE before segmentation is disabled application-wide (it could turn a
    # reflection into a false object); force the bundle flag OFF.
    reader = _import_reader()
    reader.write_correction("clahe_segmentation", "ENABLE_CLAHE", False)

    folders = find_image_folders(root)
    if not folders:
        _emit("ERROR no 'images' folder found under the processing root")
        return
    log("%d images folder(s) found" % len(folders))

    # naming check: report only the problematic names
    bad = []
    for d in folders:
        images_dir = os.path.join(d, "images")
        for n in list_images(images_dir):
            if parse_image_name(n) is None:
                bad.append(os.path.join(os.path.relpath(d, root), "images", n))
    if bad:
        log("NAMING: %d image name(s) do not follow "
            "PLATE_POSITION_DATE_Species_Compound_Concentration[_Comment].<ext>:" % len(bad))
        for b in bad[:200]:
            log("   - " + b)
    else:
        log("NAMING: all image names follow the convention")

    # --- heavy phase: detection modules only ---
    # The state classifier is NOT loaded here: each folder loads it inside
    # process_folder's Step 3, after freeing the segmenter, so only one heavy
    # model occupies the GPU at a time (critical on small-VRAM GPUs).
    # Wrapped so the libraries' import-time chatter (huggingface_hub hf_api
    # notice, etc.) does not reach the log.
    with _quiet_env.suppressed_output():
        mods = _import_models()

    done = 0
    for d in folders:
        if _interrupted():
            _emit("ERROR cancelled")
            return
        log("folder %d/%d: %s" % (done + 1, len(folders), os.path.relpath(d, root)))
        process_folder(os.path.join(d, "images"), cfg, mods)
        done += 1
    _emit("DONE %d folder(s) processed" % done)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        _emit("ERROR %s: %s" % (type(exc).__name__, exc))
        sys.exit(1)
