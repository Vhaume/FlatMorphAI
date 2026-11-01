"""
Regeneration-state pipeline, one ``images/`` folder at a time.

For every folder ``<P>/images/`` found under the processing root the
pipeline works in ``<P>``:

1. ``masks/``                 Mask2Former masks (two passes, raw input),
                              one PNG per image, same stem as the image;
2. eye cups                   heat map of the head end, peak counts at the
                              two thresholds of ``eye_thresholds.csv``,
                              ``eyespots_detection.csv`` and, on request,
                              ``Eyespots heatmaps/``;
3. blastema                   soft and binary areas at both ends, on request
                              ``blasteme_masks/`` and ``blastema_detection.csv``;
4. head                       cascade (eyes, blastema, darkness) and the
                              medial-axis tips, ``head_position.csv``;
5. thumbnails                 ``thumbnails_all_body``, ``thumbnails_head_zoomed``,
                              ``thumbnails_blastemes``, ``thumbnails_eyespots``,
                              written during the run and deleted at the end
                              unless the corresponding "Save images" flag is set;
6. state                      97 descriptors + state classifier,
                              ``regeneration_states.csv`` with ``index_planaria``
                              and the monotone correction of the rounded state.

Every function of the export is called through
:mod:`flatmorphai.notebook_api.regeneration.export_bridge`; the parameter
files of the export (thresholds, fractions, model paths) are the single
source of the numeric settings, the GUI only toggles the options below.

Method references (export README, section 10): Mask2Former, Cheng et al.,
CVPR 2022, arXiv 2112.01527. Local histogram equalisation, Zuiderveld, Graphics Gems IV, 1994,
DOI 10.1016/B978-0-12-336156-1.50061-6. U-Net, Ronneberger et al., MICCAI
2015, DOI 10.1007/978-3-319-24574-4_28. ConvNeXt V2, Woo et al., CVPR 2023,
arXiv 2301.00808.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...constants import CSV_SEPARATOR
from . import export_bridge as EB
from .naming import (
    ImageName, audit_folder, day_index, scan_images_folders, sort_key,
)

Log = Callable[[str], None]
Progress = Callable[[int], None]
Interrupt = Callable[[], bool]

THUMB_FOLDERS = {
    "all_body": "thumbnails_all_body",
    "head_zoomed": "thumbnails_head_zoomed",
    "blastemes": "thumbnails_blastemes",
    "eyespots": "thumbnails_eyespots",
}
HEATMAP_FOLDER = "Eyespots heatmaps"
BLASTEMA_MASK_FOLDER = "blasteme_masks"
MASK_FOLDER = "masks"
PEAK_EXISTENCE_FLOOR = 0.15  # floor of the p_eyespots list (FLOOR of enrich_metrics_yeux_unet_v3)


@dataclass
class RegenerationOptions:
    root: str
    use_clahe: bool = True
    overwrite_masks: bool = False
    save_eyespot_heatmaps: bool = False
    save_blastema_masks: bool = False
    save_blastema_csv: bool = False
    keep_thumbnails: Dict[str, bool] = field(default_factory=lambda: {k: False for k in THUMB_FOLDERS})
    auto_state_increase_correction: bool = True
    export_dir: Optional[str] = None


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _imread(path: str, flags: int):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def _imwrite(path: str, img) -> None:
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)


def _write_csv(path: str, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter=CSV_SEPARATOR, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.6g}"
    return v


def _pt(x: float, y: float) -> str:
    return f"{x:.1f},{y:.1f}"


def _peaks_above(hm: np.ndarray, thr: float, mindist: int) -> List[Tuple[int, int, float]]:
    """Local maxima of the heat map above ``thr`` (same rule as the export)."""
    import torch
    t = torch.from_numpy(hm)[None, None]
    mx = torch.nn.functional.max_pool2d(t, 2 * mindist + 1, 1, mindist)[0, 0].numpy()
    ys, xs = np.nonzero((hm == mx) & (hm >= thr))
    return sorted(zip(xs.tolist(), ys.tolist(), hm[ys, xs].tolist()), key=lambda q: -q[2])


def monotone_correction(rows: List[Dict[str, Any]]) -> None:
    """``state_round_corrected`` = non-decreasing copy of ``state_round`` along
    time inside each planaria (same plate and position).  A value lower than
    the previous one is raised to the previous one."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r["plate"]), str(r["position"])), []).append(r)
    for members in groups.values():
        members.sort(key=lambda r: r["_sort"])
        prev: Optional[float] = None
        for r in members:
            v = r.get("state_round")
            if v in ("", None) or (isinstance(v, float) and math.isnan(v)):
                r["state_round_corrected"] = ""
                continue
            v = int(v)
            if prev is not None and v < prev:
                v = prev
            r["state_round_corrected"] = v
            prev = v


def assign_planaria_index(rows: List[Dict[str, Any]]) -> None:
    """1-based index shared by every image with the same (plate, position),
    in order of first appearance."""
    idx: Dict[Tuple[str, str], int] = {}
    for r in rows:
        k = (str(r["plate"]), str(r["position"]))
        if k not in idx:
            idx[k] = len(idx) + 1
        r["index_planaria"] = idx[k]


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------
def run_regeneration(opts: RegenerationOptions, log: Log, progress: Progress,
                     is_interrupted: Interrupt) -> Dict[str, Any]:
    import cv2
    import torch

    t_start = time.time()
    EB.activate(opts.export_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected. The regeneration pipeline requires a GPU.")

    folders = scan_images_folders(opts.root)
    if not folders:
        raise FileNotFoundError(f"No 'images' folder found under {opts.root}")
    log(f"[Regeneration] {len(folders)} images folder(s) under {opts.root}")

    # ---- models (loaded once) ------------------------------------------
    seg_mod = EB.module("segment_mask")
    segmenter = seg_mod.Segmenter(verbose=False)
    segmenter.P["ENABLE_CLAHE"] = bool(opts.use_clahe)
    log(f"   Mask2Former {os.path.basename(EB.model_path('mask2former_segmentation'))}")
    AC = EB.module("anchor_cascade"); AC.load_models(verbose=False)
    HD = EB.module("head_detection"); EYE = EB.module("eye_detection"); BL = EB.module("blastema_detection")
    PRE = EB.module("preprocessing"); WA = EB.module("width_axis")
    stage_mod = EB.module("stage_prediction")
    classifier = stage_mod.StageClassifier(verbose=False)
    from .descriptors import DescriptorEngine
    engine = DescriptorEngine(log=log)
    thr = EB.parameters("eye_thresholds")
    t_bal = float(thr["head_decision_threshold"]); t_pos = float(thr["counting_threshold"])
    mindist = int(thr["min_peak_distance"])
    col_bal = f"n_eyespots_strong_balanced(T_Min={t_bal:g})"
    col_pos = f"n_eyespots_strong_positive(T_Min={t_pos:g})"
    log(f"   eye thresholds: balanced {t_bal:g}, positive {t_pos:g} | state classifier "
        f"{os.path.basename(EB.model_path('stage_classifier'))} ({len(classifier.cols)} descriptors)")

    # ---- audit ---------------------------------------------------------
    work: List[Tuple[str, List[ImageName]]] = []
    total = 0
    for imgdir in folders:
        ok, bad = audit_folder(imgdir)
        for b in bad:
            log(f"   WARNING non-compliant image name skipped: {os.path.join(os.path.basename(os.path.dirname(imgdir)), 'images', b)}")
        if ok:
            work.append((imgdir, ok)); total += len(ok)
    if total == 0:
        raise FileNotFoundError("No compliant image (PLATE_POSITION_DATE[_comment].ext) found.")
    done = 0
    summary: Dict[str, Any] = {"folders": [], "images": total, "failures": 0}

    for imgdir, names in work:
        if is_interrupted():
            break
        parent = os.path.dirname(imgdir)
        log(f"[Regeneration] {parent} : {len(names)} image(s)")
        mask_dir = os.path.join(parent, MASK_FOLDER); os.makedirs(mask_dir, exist_ok=True)
        thumb_dirs = {k: os.path.join(parent, v) for k, v in THUMB_FOLDERS.items()}
        for d in thumb_dirs.values():
            os.makedirs(d, exist_ok=True)
        heat_dir = os.path.join(parent, HEATMAP_FOLDER)
        if opts.save_eyespot_heatmaps:
            os.makedirs(heat_dir, exist_ok=True)
        bmask_dir = os.path.join(parent, BLASTEMA_MASK_FOLDER)
        if opts.save_blastema_masks:
            os.makedirs(bmask_dir, exist_ok=True)
        days = day_index(names)

        # ---- 1. masks -------------------------------------------------
        todo = [n for n in names if opts.overwrite_masks or not os.path.isfile(os.path.join(mask_dir, os.path.splitext(n.file_name)[0] + ".png"))]
        log(f"   masks: {len(names) - len(todo)} reused, {len(todo)} to segment")
        batch = int(segmenter.P.get("BATCH", 8))
        for s in range(0, len(todo), batch):
            if is_interrupted():
                break
            chunk = todo[s:s + batch]
            ims = [_imread(os.path.join(imgdir, n.file_name), cv2.IMREAD_COLOR) for n in chunk]
            keep = [(n, im) for n, im in zip(chunk, ims) if im is not None]
            if not keep:
                continue
            masks = segmenter.segment_batch([im for _, im in keep])
            for (n, _), m in zip(keep, masks):
                _imwrite(os.path.join(mask_dir, os.path.splitext(n.file_name)[0] + ".png"), m)
            log(f"   segmented {min(s + batch, len(todo))}/{len(todo)}")

        # ---- 2 to 6. per image ------------------------------------------
        cup_rows: List[Dict[str, Any]] = []; bl_rows: List[Dict[str, Any]] = []
        head_rows: List[Dict[str, Any]] = []; state_rows: List[Dict[str, Any]] = []
        for n in names:
            if is_interrupted():
                break
            stem = os.path.splitext(n.file_name)[0]
            base = {"image_name": n.file_name, "plate": n.plate, "position": n.position, "date": n.date}
            cup = dict(base, day=days.get(n.file_name, ""), **{col_bal: "", col_pos: "", "p_eyespots": "[]"})
            blr = dict(base, surface_area_up="", surface_area_down="")
            hdr = dict(base, up_position="", bottom_position="", head_position="")
            st = dict(base, index_planaria="", state_round="", state_decimal="", state_round_corrected="", _sort=sort_key(n))
            try:
                bgr = _imread(os.path.join(imgdir, n.file_name), cv2.IMREAD_COLOR)
                raw_mask = _imread(os.path.join(mask_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
                if bgr is None or raw_mask is None:
                    raise RuntimeError("image or mask unreadable")
                h = HD.detect_head(bgr, raw_mask, with_composite=True)
                if not h["ok"]:
                    raise RuntimeError(h["reason"])
                rbgr, rm, M, span = h["rbgr"], h["rm"], h["M"], h["span"]
                head_end = int(h["end"])

                # eye cups at the head end
                ends = EYE.eye_maps(rbgr, rm, span, threshold=PEAK_EXISTENCE_FLOOR)
                e = ends[head_end]
                if e["heatmap"] is not None:
                    hm = e["heatmap"]
                    pk = _peaks_above(hm, PEAK_EXISTENCE_FLOOR, mindist)
                    cup[col_bal] = sum(1 for p in pk if p[2] >= t_bal)
                    cup[col_pos] = sum(1 for p in pk if p[2] >= t_pos)
                    cup["p_eyespots"] = "[" + ", ".join(f"{p[2]:.3f}" for p in pk) + "]"
                    _imwrite(os.path.join(thumb_dirs["eyespots"], stem + ".png"), e["thumbnail"])
                    if opts.save_eyespot_heatmaps:
                        ms = PRE.project_map(hm, e["box"], rm.shape)
                        Mi = cv2.invertAffineTransform(M); H, W = rm.shape
                        mo = cv2.warpAffine(ms, Mi, (W, H))
                        col = cv2.applyColorMap((np.clip(mo, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        alpha = np.clip(mo, 0, 1)[..., None] * 0.65
                        over = (bgr * (1 - alpha) + col * alpha).astype(np.uint8)
                        _imwrite(os.path.join(heat_dir, stem + ".png"), over)
                else:
                    cup[col_bal] = 0; cup[col_pos] = 0

                # blastema at both ends (0 top, 1 bottom)
                bl = BL.blastema_maps(rbgr, rm, span)
                blr["surface_area_up"] = int(bl[0]["binary_area_px"]); blr["surface_area_down"] = int(bl[1]["binary_area_px"])
                ib = 0 if bl[0]["soft_area"] >= bl[1]["soft_area"] else 1
                if bl[ib]["thumbnail"] is not None:
                    _imwrite(os.path.join(thumb_dirs["blastemes"], stem + ".png"), bl[ib]["thumbnail"])
                if opts.save_blastema_masks and bl[ib]["probmap"] is not None:
                    ms = PRE.project_map(bl[ib]["probmap"], bl[ib]["box"], rm.shape)
                    Mi = cv2.invertAffineTransform(M); H, W = rm.shape
                    mo = cv2.warpAffine(ms, Mi, (W, H))
                    bm = ((mo >= 0.5) & (PRE.worm_mask(raw_mask) > 0)).astype(np.uint8) * 255
                    _imwrite(os.path.join(bmask_dir, stem + ".png"), bm)

                # head and tip positions along the medial axis (original frame)
                pa = WA.medial_axis(rm.astype(np.uint8))
                if pa is not None:
                    P, _L = WA.smooth(pa, int(AC.N_AXIS))
                    if P is not None and len(P) >= 2:
                        a, b = P[0], P[-1]                      # rows are (y, x)
                        up, bottom = (a, b) if a[0] <= b[0] else (b, a)
                        ux, uy = PRE.to_original(float(up[1]), float(up[0]), M)
                        bx, by = PRE.to_original(float(bottom[1]), float(bottom[0]), M)
                        hdr["up_position"] = _pt(ux, uy); hdr["bottom_position"] = _pt(bx, by)
                hdr["head_position"] = _pt(float(h["x_original"]), float(h["y_original"]))

                # thumbnails of the state composite
                comp = h["composite"]
                _imwrite(os.path.join(thumb_dirs["all_body"], stem + ".png"), comp[:, :, 0])
                _imwrite(os.path.join(thumb_dirs["head_zoomed"], stem + ".png"), comp[:, :, 1])

                # descriptors and state
                desc = engine.compute(bgr, raw_mask, (h["x_original"], h["y_original"]))
                pred = classifier.predict(comp, desc)
                s_dec = float(pred["decimal_stage"])
                st["state_decimal"] = round(s_dec, 4)
                st["state_round"] = int(min(6, max(1, math.floor(s_dec))))
                if pred.get("missing_descriptors") and len(pred["missing_descriptors"]) > 1:
                    log(f"   {n.file_name}: {len(pred['missing_descriptors'])} descriptors imputed")
            except Exception as exc:  # noqa: BLE001
                summary["failures"] += 1
                log(f"   ERROR {n.file_name}: {type(exc).__name__}: {exc}")
            cup_rows.append(cup); bl_rows.append(blr); head_rows.append(hdr); state_rows.append(st)
            done += 1
            progress(int(100 * done / max(1, total)))

        # ---- CSV outputs ------------------------------------------------
        _write_csv(os.path.join(parent, "eyespots_detection.csv"),
                   ["image_name", "plate", "position", "day", "date", col_bal, col_pos, "p_eyespots"], cup_rows)
        if opts.save_blastema_csv:
            _write_csv(os.path.join(parent, "blastema_detection.csv"),
                       ["image_name", "plate", "position", "date", "surface_area_up", "surface_area_down"], bl_rows)
        _write_csv(os.path.join(parent, "head_position.csv"),
                   ["image_name", "plate", "position", "date", "up_position", "bottom_position", "head_position"], head_rows)
        assign_planaria_index(state_rows)
        if opts.auto_state_increase_correction:
            monotone_correction(state_rows)
        else:
            for r in state_rows:
                r["state_round_corrected"] = r["state_round"]
        _write_csv(os.path.join(parent, "regeneration_states.csv"),
                   ["image_name", "index_planaria", "plate", "position", "date", "state_round", "state_decimal", "state_round_corrected"],
                   state_rows)

        # ---- thumbnails clean-up -----------------------------------------
        for k, d in thumb_dirs.items():
            if not opts.keep_thumbnails.get(k, False):
                shutil.rmtree(d, ignore_errors=True)
        summary["folders"].append(parent)
        log(f"   written: eyespots_detection.csv, head_position.csv, regeneration_states.csv"
            + (", blastema_detection.csv" if opts.save_blastema_csv else ""))

    summary["elapsed_s"] = round(time.time() - t_start, 1)
    log(f"[Regeneration] finished in {summary['elapsed_s']} s, {summary['failures']} failure(s)")
    return summary
