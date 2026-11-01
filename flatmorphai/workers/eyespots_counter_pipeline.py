# -*- coding: utf-8 -*-
"""Eyespots counter driver -- v20 head cascade (same detection as regeneration).

The previous driver used the Export_head_detection eye model, which was
returning ZERO eyespots. This one uses the SAME detection as the "Forecast
regeneration states" page: the v20 head cascade (ancrage_cascade_v3c13) runs the
eye U-Net on an oriented vignette at each extremity. The segmentation is also the
SAME as the other pages -- it reuses the regeneration driver's helpers
(``_iter_prepared``: Mask2Former two passes at 320, clean_seg_mask, models loaded
once, masks cached so an already-segmented folder is reused almost instantly).

Only the v20 bundle is loaded (never Export at the same time): the two bundles
share module names (preprocessing, parameter_reader, ...) and would collide.

Output: ``eyespots_detection.csv`` next to each ``images/`` folder, with the
balanced and positive counts + (x, y) positions -- the exact schema the
Regeneration states editor and the Stereotype editor read.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback

# Make the sibling workers importable the same way the other drivers do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _quiet_env  # noqa: E402,F401  (silence noisy third-party notices first)
import eyespot_shared  # noqa: E402
import mask_refine  # noqa: E402
import linux_workdir  # noqa: E402  (write the outputs on the Linux disk under WSL)
import regen_stage_v20_pipeline as R20  # noqa: E402  (reuse the proven v20 helpers)

# Reuse the exact stdout protocol the worker parses (LOG / PROGRESS / STEP / ...).
_emit, log, progress = R20._emit, R20.log, R20.progress
# R20.log is already a no-op under FLATMORPHAI_QUIET=1 (see that module):
# the name is bound once here, so nothing is tested per call.


def _count_eyes(r3, A3b, t_bal, t_pos, pair_rule=False):
    """Eyespot count: the version shared with the regeneration chain.

    The computation used to live here in duplicate; both chains read the same
    heat map of the same extremity, so they must count identically and differ
    only by the threshold.
    """
    return eyespot_shared.count_peaks(r3, A3b, t_bal, t_pos, pair_rule)


def _save_eye_outputs(target, stem, r3, save_heat, save_thumb, cv2, np):
    """Optional outputs: eye heatmaps (both ends) and the head-end eye thumbnail."""
    if not r3:
        return
    cands = r3.get("candidates") or []
    i_head = int(r3.get("i_head", 0))
    try:
        if save_heat:
            d = os.path.join(target, "Eyespots heatmaps")
            os.makedirs(d, exist_ok=True)
            for c in cands:
                hm = c.get("eye_heatmap")
                if hm is not None:
                    sfx = "top" if int(c.get("end", 0)) == 0 else "bottom"
                    cv2.imwrite(os.path.join(d, "%s_%s.png" % (stem, sfx)),
                                (255 * np.clip(np.asarray(hm), 0, 1)).astype(np.uint8))
        if save_thumb and 0 <= i_head < len(cands):
            tv = cands[i_head].get("eye_thumbnail")
            if tv is not None:
                d = os.path.join(target, "thumbnails_eyespots")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, stem + ".png"), np.asarray(tv))
    except Exception:  # noqa: BLE001 - optional outputs must never stop the run
        pass


def main() -> int:
    raw = (open(sys.argv[1], encoding="utf-8").read()
           if len(sys.argv) > 1 else sys.stdin.read())
    cfg = json.loads(raw)

    bundle_root = cfg.get("bundle_root", "")
    if not bundle_root or not os.path.isdir(bundle_root):
        _emit("ERROR regeneration v20 bundle not found: " + str(bundle_root))
        return 1
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402

    root = cfg["root"]
    cancel_file = cfg.get("cancel_file", "")

    def _interrupted():
        return bool(cancel_file and os.path.exists(cancel_file))

    t_bal = float(cfg.get("threshold_balanced", 0.80))
    t_pos = float(cfg.get("threshold_positive", 0.60))
    # Each threshold can be switched off from the page; at least one is always
    # on. A disabled threshold is not counted and its two columns are not
    # written, so the detection runs at the enabled threshold(s) only.
    use_bal = bool(cfg.get("use_balanced", True))
    use_pos = bool(cfg.get("use_positive", True))
    if not (use_bal or use_pos):
        use_bal = True
    save_heat = bool(cfg.get("save_heatmaps", False))
    save_thumb = bool(cfg.get("save_thumbnails", False))
    overwrite = bool(cfg.get("overwrite", False))
    col_bal, col_bal_pos, col_pos, col_pos_pos = eyespot_shared.eyespot_columns(
        t_bal, t_pos)
    # Columns of the CSV, in order, for the enabled thresholds only.
    _cols = []
    if use_bal:
        _cols += [col_bal, col_bal_pos]
    if use_pos:
        _cols += [col_pos, col_pos_pos]

    try:
        folders = R20.find_image_folders(root)
        if not folders:
            _emit("ERROR no 'images' folder found under the processing root")
            return 1
        grand_total = sum(len(R20.list_images(os.path.join(d, "images")))
                          for d in folders)
        if grand_total == 0:
            _emit("ERROR the 'images' folders contain no readable image")
            return 1
        log("%d image(s) to analyse in total" % grand_total)

        _emit("STEP 0")
        # Only the two models the eyespot count needs: the Mask2Former segmenter
        # and the head cascade (which carries the eye U-Net). The refiner, the
        # composite builder and the state classifier of the full pipeline are
        # NOT loaded, keeping GPU memory low.
        from regen_pipeline.segmentation import Segmenter as _Segmenter  # noqa: E402
        from regen_pipeline.head_cascade import HeadCascade as _HeadCascade  # noqa: E402
        import regen_pipeline._env as _v20env  # noqa: E402
        A3b = _v20env.A3b                       # cascade module (_pics_tous, _vers_image)
        mask_refine.preflight_vram(log=log)
        log("Loading the models (segmentation + head cascade)...")
        with _quiet_env.suppressed_output():
            _seg_obj = _Segmenter(verbose=False)
            _cascade_obj = _HeadCascade(verbose=False)
            try:
                from regen_pipeline.fastpath import install as _install_fp
                _install_fp()
            except Exception:  # noqa: BLE001
                pass

        class _Pipe:
            pass

        pipe = _Pipe()
        pipe.segmenter = _seg_obj
        pipe.cascade = _cascade_obj

        _seg = getattr(pipe, "segmenter", None)
        # Same input as every other page: raw image, no pre-segmentation
        # enhancement. This page used to need an environment variable to do
        # it, so by default it fed the segmentation model an enhanced image
        # while the regeneration chain and Pipeline C fed it a raw one -- the
        # same model, two different inputs, two different masks, depending on
        # which page the operator opened.
        R20._force_raw_segmentation(_seg)
        if _seg is not None and getattr(_seg, "processor", None) is not None:
            _sample = R20._first_sample_bgr(folders)
            if _sample is not None:
                mask_refine.configure_fed_size(_seg, _sample, target=320, log=log)

        # Under WSL the folder is on /mnt/c, where writing a full-resolution PNG
        # costs ~0.30 s against ~0.07 s on the Linux disk. The masks, heat maps
        # and thumbnails are written to a mirror there and copied back at the
        # end; the images keep being read from the user's folder, and an
        # already-computed mask is still read from it.
        work = linux_workdir.work_path(root)
        if work:
            try:
                if os.path.isdir(work):
                    import shutil as _sh
                    _sh.rmtree(work, ignore_errors=True)
                os.makedirs(work, exist_ok=True)
                log("Writing the results on the Linux disk, copied back to %s "
                    "at the end of the run" % root)
            except Exception as exc:  # noqa: BLE001
                log("  Linux work folder unavailable (%s); writing in place" % exc)
                work = None

        def _out(path):
            """Where to WRITE ``path`` (a folder under root): the mirror, or itself."""
            if not work:
                return path
            rel = os.path.relpath(path, root)
            if rel.startswith(".."):
                return path
            q = work if rel == "." else os.path.join(work, rel)
            os.makedirs(q if not os.path.splitext(q)[1] else os.path.dirname(q),
                        exist_ok=True)
            return q

        # Flat job list across EVERY folder, so the detection runs on all images
        # at once (not folder by folder), like the other pipelines.
        jobs = []                      # (target, images_dir, masks_dir, name)
        for d in folders:
            images_dir = os.path.join(d, "images")
            masks_dir = os.path.join(d, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for n in R20.list_images(images_dir):
                jobs.append((d, images_dir, masks_dir, n))

        # ---- Stage 1: segment every MISSING mask, all folders together ----
        _emit("STEP 1")
        _to_make = sum(1 for (_t, _i, _m, n) in jobs
                       if overwrite or not os.path.isfile(
                           os.path.join(_m, os.path.splitext(n)[0] + ".png")))
        log("Masks: %d to generate, %d reused (fast check over %d image(s))"
            % (_to_make, len(jobs) - _to_make, len(jobs)))
        if _to_make:
            for d in folders:
                if _interrupted():
                    _emit("ERROR cancelled")
                    return 1
                images_dir = os.path.join(d, "images")
                masks_dir = os.path.join(d, "masks")
                names = R20.list_images(images_dir)
                if all(os.path.isfile(os.path.join(masks_dir, os.path.splitext(n)[0] + ".png"))
                       for n in names) and not overwrite:
                    continue
                for name, stem, mpath, bgr, mask_m2f, fresh in R20._iter_prepared(
                        pipe, images_dir, masks_dir, names, overwrite):
                    if _interrupted():
                        _emit("ERROR cancelled")
                        return 1
                    if fresh and mask_m2f is not None:
                        cv2.imwrite(_out(mpath),
                                    mask_refine.clean_seg_mask(mask_m2f, bgr))
            mask_refine._free_vram_mb_after_cleanup()

        # ---- Stage 2: eye detection over EVERY image at once ----
        # The cascade is GPU-bound (eye U-Net) while the axis and the peak
        # extraction are CPU work, so analysing two images at once overlaps
        # them. The cascade keeps no state between images -- the shared pieces
        # of the v20 chain were made private to each thread for the
        # regeneration page (per-thread namespaces), and the counts
        # were checked identical at 1, 2 and 3 threads there. Reading the image
        # and its mask also happens on the worker, so the disk no longer stalls
        # the card. Rows are reassembled in submission order, so the CSV is
        # byte-for-byte what the sequential run produced.
        _emit("STEP 2")
        from concurrent.futures import ThreadPoolExecutor
        _workers = R20._resolve_regen_workers(log, grand_total)

        def _one(job):
            """Analyse one image. Returns (job, positions_balanced, positions_positive)."""
            target, images_dir, masks_dir, name = job
            stem = os.path.splitext(name)[0]
            bal, pos = [], []
            try:
                _ip = os.path.join(images_dir, name)
                bgr = cv2.imread(_ip)
                if bgr is None:
                    bgr = cv2.imdecode(np.fromfile(_ip, np.uint8), cv2.IMREAD_COLOR)
                _mp = os.path.join(masks_dir, stem + ".png")
                mask = cv2.imread(_mp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(_mp) else None
                if bgr is not None and mask is not None:
                    r3 = pipe.cascade.anchor(bgr, mask)
                    if r3:
                        # Gate: the threshold SELECTED on the page.
                        # Admission: ALWAYS the permissive threshold from the
                        # Settings, even when it is not written to the CSV.
                        # Passing the selected threshold on both sides (what
                        # the previous version did when a single threshold was
                        # ticked) made the pair rule inert, since it requires
                        # t_pos < t_bal -- the log announced "pair rule on"
                        # for a rule that never applied.
                        bal, pos = _count_eyes(
                            r3, A3b,
                            t_bal if use_bal else t_pos,
                            t_pos,
                            bool(cfg.get("eyespots_pair_rule", False)))
                        if not use_bal:
                            bal = []
                        if not use_pos:
                            pos = []
                        if save_heat or save_thumb:
                            _save_eye_outputs(_out(target), stem, r3,
                                              save_heat, save_thumb, cv2, np)
            except Exception as exc:  # noqa: BLE001
                log("  ERROR on %s: %s: %s" % (name, type(exc).__name__, exc))
                if R20._is_oom(exc):
                    mask_refine._free_vram_mb_after_cleanup()
            return job, bal, pos

        rows_by_folder = {}
        done = 0

        def _row(job, bal, pos):
            nonlocal done
            target, _images_dir, _masks_dir, name = job
            meta = R20.parse_image_name(name) or R20.empty_meta()
            row = {"image_name": name, "plate": meta["plate"],
                   "position": meta["position"], "date": meta["date"]}
            if use_bal:
                row[col_bal] = len(bal)
                row[col_bal_pos] = eyespot_shared.positions_str(bal)
            if use_pos:
                row[col_pos] = len(pos)
                row[col_pos_pos] = eyespot_shared.positions_str(pos)
            rows_by_folder.setdefault(target, []).append(row)
            done += 1
            progress(done, grand_total)

        if _workers > 1:
            from collections import deque as _dq
            with ThreadPoolExecutor(_workers) as _pool:
                inflight = _dq()
                for job in jobs:
                    if _interrupted():
                        _emit("ERROR cancelled")
                        return 1
                    inflight.append(_pool.submit(_one, job))
                    if len(inflight) >= _workers:
                        _row(*inflight.popleft().result())
                while inflight:
                    _row(*inflight.popleft().result())
        else:
            for job in jobs:
                if _interrupted():
                    _emit("ERROR cancelled")
                    return 1
                _row(*_one(job))
        mask_refine._free_vram_mb_after_cleanup()

        # ---- write one eyespots_detection.csv per folder ----
        for target, rows in rows_by_folder.items():
            if not rows:
                continue
            path = _out(os.path.join(target, "eyespots_detection.csv"))
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["image_name", "plate", "position", "date"] + _cols,
                    delimiter=";")
                w.writeheader()
                w.writerows(rows)
            log("  wrote %s/eyespots_detection.csv (%d rows)"
                % (os.path.basename(target), len(rows)))

        if work:
            linux_workdir.sync_back(work, root, log)
        _emit("DONE %d" % grand_total)
        return 0
    except SystemExit as exc:
        _emit("ERROR " + str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        _emit("ERROR %s: %s" % (type(exc).__name__, exc))
        sys.stderr.write(traceback.format_exc())
        mask_refine.release_gpu_on_crash(log=log)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
