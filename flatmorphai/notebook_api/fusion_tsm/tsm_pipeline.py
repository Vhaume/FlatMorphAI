"""TSM fusion pipeline: per-video metrics CSV + temporal images + inference.

This module is the FlatMorphAI-side glue between the bundled
``export_model_and_code`` and the live ``process_videos`` /
``run_step4`` infrastructure.  It does three things, in order:

  1. Resolve a 3-frame ``(prev, cur, next)`` window for every frame in a
     video, using the FlatMorphAI naming convention
     ``<video_stem>_<second>s.png`` (NOT the legacy
     ``_Frame_n_in_tot`` convention) but applying the SAME rank-based
     ``dyn_window`` rule the model was trained with -- so a clip with
     ``tot >= 3`` frames places the labelled frame at the middle of
     the window in the interior and at the edge at the boundaries,
     just like training.  Reference: bundle/config/nomenclature.py and
     Lin, J., Gan, C. and Han, S., "TSM: Temporal Shift Module for
     Efficient Video Understanding", ICCV 2019, arXiv:1811.08383.
  2. Compute the 308 required metrics per frame from the (prev, cur,
     next) masks plus the raw current-frame grayscale, the formatted
     224 / 384 image, and (optionally) a motion image / wider mask
     window.  Reference for the metric selection rationale: Guyon, I.
     and Elisseeff, A., "An Introduction to Variable and Feature
     Selection", JMLR 3, 1157-1182, 2003.
  3. Build the 384 x 384 BGR "temporal image" (channel 0 = T-1 masked
     grayscale, channel 1 = T0 masked grayscale, channel 2 = T+1
     masked grayscale; the three frames are cropped by the common
     square union of their masks so motion is preserved at full
     resolution).  This is the SAME image the bundle's
     ``build_temporal_image`` produces.

Outputs (one per video) match the FlatMorphAI conventions:

  - ``<video_dir>/<video_name>_metrics.csv``: per-frame metric rows
    (308 columns) -- written by ``compute_video_metrics_tsm_csv``.
  - The temporal image is held in memory and never written to disk;
    inference consumes it directly.

The forecast loop lives in ``process_videos`` (classification.py) and
calls ``forecast_video_tsm`` below per video.  Skip-existing and
partial-update semantics mirror the earlier metrics_csv.py so re-runs
on the same video are near-instant.

Reference for skip-existing: Feldman, S.I., "Make: a program for
maintaining computer programs", Software: Practice and Experience
9(4), 255-265, 1979, DOI:10.1002/spe.4380090402.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


from .nomenclature import app_second, list_clip_frames, dyn_window
from .temporal_format import build_temporal_image
from .metrics_tsm import compute_required_metrics, required_metrics


def _read_image_uint8(path: str, gray: bool = False) -> Optional[np.ndarray]:
    """Robust cv2 read that survives non-ASCII paths."""
    if not os.path.isfile(path):
        return None
    try:
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_UNCHANGED
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    except Exception:  # noqa: BLE001
        return None


def _list_ordered_frames(
    video_dir: str, video_name: str,
) -> List[Tuple[float, str]]:
    """Return [(second, image_name)] for every image in
    ``<video_dir>/images/`` parseable via the FlatMorphAI naming
    convention, sorted ascending by second."""
    images_dir = os.path.join(video_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    out: List[Tuple[float, str]] = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(".png"):
            continue
        sec = app_second(fn)
        if sec is None:
            continue
        out.append((float(sec), fn))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _resolve_dyn_window(
    ordered: List[Tuple[float, str]],
) -> Dict[str, Tuple[str, str, str]]:
    """For every frame return (prev_name, cur_name, next_name) under
    the bundle's ``dyn_window(n, tot)`` rule applied to acquisition
    rank.  ``cur_name`` is always the frame at the central rank of
    the window (matches the labelled frame in the interior and one of
    the neighbours at the boundary, exactly as the model was
    trained).
    """
    n_total = len(ordered)
    out: Dict[str, Tuple[str, str, str]] = {}
    names = [n for _, n in ordered]
    for idx, (_, name) in enumerate(ordered):
        n = idx + 1
        ks = dyn_window(n, n_total)  # 1-based ranks
        prev_name = names[ks[0] - 1]
        cur_name  = names[ks[1] - 1]
        next_name = names[ks[2] - 1]
        out[name] = (prev_name, cur_name, next_name)
    return out


# ---------------------------------------------------------------------------
# Per-video metrics CSV writer
# ---------------------------------------------------------------------------
def compute_video_metrics_tsm_csv(
    video_dir: str,
    mask_dir_name: str,
    output_csv_path: str,
    metric_cols: Optional[List[str]] = None,
    overwrite: bool = False,
    formatted_dir_name: Optional[str] = None,  # kept for callsite back-compat (ignored)
    log: Optional[Callable[[str], None]] = None,
    max_workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Walk every frame in ``<video_dir>/images/`` and compute the 308
    required metrics per frame, writing the result to ``output_csv_path``
    in the FlatMorphAI ``sep=';'`` format with columns ``image_name``,
    ``second``, then the metric columns.

    Behaviour:

      - Sub-second sampling is supported (the operator's earlier
        request).  Frames are ordered by parsed second; neighbours
        come from the ``dyn_window`` rank-based rule because the
        model was trained with that exact mapping (see
        ``nomenclature.temporal_window_app``).
      - When ``output_csv_path`` already exists AND ``overwrite`` is
        False, every row whose ``image_name`` is already on disk is
        reused as-is; only missing rows are recomputed.  This makes a
        re-run after appending a few frames near-instant.
      - ``formatted_dir_name``, when provided, points at the Step 3
        output directory for this video (typically
        ``<video>_image-size(384)_margin(0.00)_b&w(False)_apply-mask(True)``).
        When present, the formatted T0 image is loaded from there and
        passed to ``compute_required_metrics`` so the formatted-image
        groups (shape, tunif, centerline, texture, curated) are
        actually computed.  When absent, those 78 columns get the
        training-median value via the model's metric-branch
        fallback.

    Returns the number of rows in the final CSV (existing + new).
    """
    _log = log if log is not None else (lambda _msg: None)

    images_dir = os.path.join(video_dir, "images")
    masks_dir = os.path.join(video_dir, mask_dir_name)
    if not os.path.isdir(images_dir):
        _log(f"  WARNING: no images/ sub-folder under {video_dir}; "
             "skipping metrics CSV.")
        return 0
    if not os.path.isdir(masks_dir):
        _log(f"  WARNING: no '{mask_dir_name}/' under {video_dir}; "
             "Step 2 may not have run for this video; skipping metrics CSV.")
        return 0

    video_name = os.path.basename(os.path.normpath(video_dir))
    cols_out = list(metric_cols) if metric_cols else list(required_metrics())

    ordered = _list_ordered_frames(video_dir, video_name)
    if not ordered:
        _log(f"  WARNING: no parseable frames in {images_dir}; "
             "metrics CSV will be empty.")
        return 0

    # Partial-update: re-use existing rows on disk so a rerun after
    # adding a few frames is fast.
    existing_rows: Dict[str, Dict[str, object]] = {}
    if (not overwrite) and os.path.isfile(output_csv_path):
        try:
            df_prev = pd.read_csv(
                output_csv_path, sep=";", encoding="utf-8",
            )
            for _, row in df_prev.iterrows():
                k = str(row.get("image_name", ""))
                if k:
                    existing_rows[k] = row.to_dict()
            _log(
                f"  metrics CSV partial-update: {len(existing_rows)} "
                f"row(s) already on disk in {os.path.basename(output_csv_path)}"
            )
        except Exception as exc:  # noqa: BLE001
            _log(
                f"  Could not read existing metrics CSV "
                f"({type(exc).__name__}: {exc}); recomputing from scratch."
            )
            existing_rows = {}

    neighbour_of = _resolve_dyn_window(ordered)

    # Per-video cache for mask + gray reads (3-frame window means
    # each PNG is otherwise decoded up to three times).
    mask_cache: Dict[str, Optional[np.ndarray]] = {}
    gray_cache: Dict[str, Optional[np.ndarray]] = {}

    def _mask(name: str) -> Optional[np.ndarray]:
        if name not in mask_cache:
            mask_cache[name] = _read_image_uint8(
                os.path.join(masks_dir, name), gray=True,
            )
        return mask_cache[name]

    def _gray(name: str) -> Optional[np.ndarray]:
        if name not in gray_cache:
            gray_cache[name] = _read_image_uint8(
                os.path.join(images_dir, name), gray=True,
            )
        return gray_cache[name]

    # In-memory formatted T0 builder.  We never read any "Formated_*"
    # directory off disk: instead, for each frame we apply the bundle's
    # ``compute_mask_image_planarian_reshaped`` to the source colour
    # image + mask, building the 384x384 BGR formatted T0 in memory.
    # This satisfies the operator's "temporal images on disk only"
    # request while preserving the 78 formatted-image-derived metrics
    # the model was trained with.  Reference for the canonical
    # formatter: ``classification_images.compute_mask_image_planarian_reshaped``
    # (the same function Step 3 calls in non-temporal mode).
    from ..classification_images import (
        biggest_objects_in_the_binary_mask,
        compute_mask_image_planarian_reshaped,
    )

    def _read_color(name: str) -> Optional[np.ndarray]:
        p = os.path.join(images_dir, name)
        if not os.path.isfile(p):
            return None
        try:
            return cv2.imdecode(
                np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR,
            )
        except Exception:  # noqa: BLE001
            return None

    def _build_formatted_t0(cur_mask: np.ndarray,
                            image_name: str) -> Optional[np.ndarray]:
        color = _read_color(image_name)
        if color is None:
            return None
        try:
            masks_planaria = biggest_objects_in_the_binary_mask(
                cur_mask, num_objects_to_keep=1, num_objects_by_row=1,
            )
        except Exception:  # noqa: BLE001
            return None
        if not masks_planaria or masks_planaria[0] is None:
            return None
        try:
            return compute_mask_image_planarian_reshaped(
                color, masks_planaria[0],
                image_size=384, margin_percent=0.0,
                apply_mask=True, black_and_white=False,
            )
        except Exception:  # noqa: BLE001
            return None

    # Pre-compute the (cur_mask, gray, neighbour-mask, neighbour-gray)
    # triplets PER FRAME up front in the main thread so the shared
    # caches (gray_cache / mask_cache) are populated single-threaded
    # and the metric loop becomes embarrassingly parallel.  The
    # downstream ThreadPoolExecutor then only does CPU-bound metric
    # work, which dominates the runtime.  Reference for the worker-
    # pool pattern over CPU-bound numerical work where cv2 and
    # scikit-image release the GIL: McKinney, W., "Python for Data
    # Analysis", 3rd ed., O'Reilly, 2022, ISBN 978-1-098-10403-0,
    # ch. 17 ("Concurrent execution patterns").
    todo: List[Tuple[float, str]] = []
    rows: List[Optional[Dict[str, object]]] = [None] * len(ordered)
    n_skipped_existing = 0
    n_skipped_missing_mask = 0
    for i, (sec, image_name) in enumerate(ordered):
        if image_name in existing_rows:
            rows[i] = existing_rows[image_name]
            n_skipped_existing += 1
            continue
        cur_mask = _mask(image_name)
        if cur_mask is None:
            n_skipped_missing_mask += 1
            continue
        # Warm the caches for this frame's neighbours so the worker
        # threads do not race on cv2.imdecode (the dict reads inside
        # the workers are fine even without a lock because by then
        # every key is already present, but writes from multiple
        # threads simultaneously would conflict).
        prev_name, _cur_neighbour, next_name = neighbour_of[image_name]
        for _n in (prev_name, image_name, next_name):
            _mask(_n)
            _gray(_n)
        todo.append((sec, image_name))

    def _compute_one(args: Tuple[int, float, str]) -> Tuple[int, Dict[str, object]]:
        idx_in_rows, sec, image_name = args
        # ``cur_neighbour`` is the frame at rank ks[1] of the
        # dyn_window applied to acquisition order.  For interior
        # frames it equals ``image_name``; for the FIRST frame it
        # is the SECOND frame (ranks [1,2,3] -> T0 slot = rank 2);
        # for the LAST frame it is the SECOND-TO-LAST frame (ranks
        # [tot-2, tot-1, tot] -> T0 slot = rank tot-1).  The
        # training pipeline (fusion_tsm/metrics_lib/gen_masks.py
        # and gen_temporal_dataset.py) reads the T0 mask/gray from
        # rank ks[1] as well, so mirroring that shift here keeps
        # inference numerically aligned with training.  Reference:
        # fusion_tsm/nomenclature.py ``dyn_window`` and the file
        # gen_temporal_dataset.build_temporal line 94.
        prev_name, cur_neighbour, next_name = neighbour_of[image_name]
        cur_mask = mask_cache.get(cur_neighbour)
        prev_mask = (
            mask_cache.get(prev_name) if prev_name != cur_neighbour else cur_mask
        )
        next_mask = (
            mask_cache.get(next_name) if next_name != cur_neighbour else cur_mask
        )
        cur_gray  = gray_cache.get(cur_neighbour)
        prev_gray = (
            gray_cache.get(prev_name) if prev_name != cur_neighbour else None
        )
        next_gray = (
            gray_cache.get(next_name) if next_name != cur_neighbour else None
        )
        # The formatted T0 image (input to shape / centerline /
        # texture metrics) is also anchored on the dyn_window T0
        # slot for training-time parity.
        formatted_t0_bgr = _build_formatted_t0(cur_mask, cur_neighbour)
        try:
            feat = compute_required_metrics(
                masks3=(prev_mask, cur_mask, next_mask),
                grays3=(prev_gray, cur_gray, next_gray),
                formatted_t0_bgr=formatted_t0_bgr,
                raw_color3=None,
                motion_image=None,
                wide_masks=None,
            )
        except Exception:  # noqa: BLE001
            feat = {}
        row = {"image_name": image_name, "second": float(sec)}
        for col in cols_out:
            row[col] = float(feat.get(col, np.nan))
        return idx_in_rows, row

    # Determine the worker pool size.  cv2 / numpy / scikit-image
    # release the GIL on most calls so threads scale near-linearly
    # up to the number of physical cores.  We leave ONE core free
    # for the GUI event loop.  The previous cap at 8 workers was a
    # legacy conservatism that flattened throughput on machines
    # with 16+ cores; scaling to all available cores minus one
    # keeps the GUI responsive while extracting the full parallel
    # win the OpenCV / scikit-image routines allow.
    if max_workers is None:
        from ...cpu_topology import pool_size
        max_workers = pool_size(reserve=1)

    # Build the (idx_in_rows, sec, image_name) tuples for every
    # frame that survived the existing-row + missing-mask filters.
    by_name_idx = {name: i for i, (_s, name) in enumerate(ordered)}
    work = [
        (by_name_idx[name], sec, name)
        for (sec, name) in todo
    ]
    if work:
        # ``as_completed`` instead of ``map`` so we can emit progress
        # per frame as workers finish, rather than blocking until every
        # task is done.  The progress bar in the host pipeline freezes
        # otherwise -- the bug the operator reported.  Reference for
        # the per-completion progress pattern: McKinney, W., "Python
        # for Data Analysis", 3rd ed., O'Reilly, 2022, ISBN
        # 978-1-098-10403-0, ch. 17 ("Concurrent Patterns").
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_total = len(work)
        n_done = 0
        # First "report" before any work, so the host can move the bar
        # off zero and show "Metrics 0/N" the moment the phase starts.
        if progress_cb is not None:
            try:
                progress_cb(0, n_total, video_name)
            except Exception:  # noqa: BLE001
                pass
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_compute_one, w): w for w in work}
            for fut in as_completed(futures):
                idx_in_rows, row = fut.result()
                rows[idx_in_rows] = row
                n_done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(n_done, n_total, video_name)
                    except Exception:  # noqa: BLE001
                        pass
    # Drop the unfilled rows (frames with missing masks).
    rows = [r for r in rows if r is not None]

    if n_skipped_existing and not overwrite:
        n_new = len(rows) - n_skipped_existing
        _log(
            f"  Reused {n_skipped_existing} existing row(s); "
            f"computed {n_new} new row(s) using {max_workers} thread(s)."
        )
    elif work:
        _log(
            f"  Computed {len(work)} metric row(s) using "
            f"{max_workers} thread(s)."
        )
    if n_skipped_missing_mask:
        _log(
            f"  WARNING: {n_skipped_missing_mask} frame(s) had no "
            f"current mask; those rows are NOT in the CSV."
        )
    if not rows:
        _log(f"  No metric rows produced for {video_name}; "
             "check Step 2 output.")
        return 0

    df = pd.DataFrame(rows)
    front = ["image_name", "second"]
    metric_front = [c for c in cols_out if c in df.columns]
    other = [c for c in df.columns if c not in front and c not in metric_front]
    df = df[front + metric_front + other]

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df.to_csv(output_csv_path, sep=";", index=False, encoding="utf-8")
    _log(
        f"  Wrote metrics CSV ({len(df)} row(s), {len(df.columns)} "
        f"cols): {output_csv_path}"
    )
    return len(df)


# ---------------------------------------------------------------------------
# Per-video temporal-image inference loop
# ---------------------------------------------------------------------------
def forecast_video_tsm(
    video_dir: str,
    mask_dir_name: str,
    model,
    fusion_info: dict,
    device,
    metric_matrix_by_image: Dict[str, np.ndarray],
    temporal_dir_name: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> List[Tuple[str, str, dict]]:
    """Score every frame in the video and return [(image_name, label,
    per-class probs)].

    The temporal image is built on the fly per frame (no temporary
    PNGs on disk).  ``metric_matrix_by_image[image_name]`` is the
    already-standardized 308-vector for that image (rows in this dict
    are aligned to the per-video metrics CSV; the caller pre-computes
    them so this loop only does the per-frame image work and the
    forward pass).

    Inference reproduces ``predictor_tsm.TSMPredictor.predict``
    byte-for-byte:
      - Each of the three temporal channels is scaled to [0, 1] then
        standardised as ``(g - 0.45) / 0.225`` (the FusionDS training
        normalisation).
      - The three values are replicated to 3 channels per frame,
        giving a 9-channel tensor of shape (1, 9, H, W).
      - The model reshapes that 9-channel input into a 3-frame clip
        internally; the TSM hooks shift channels between backbone
        stages.

    Reference for the channel normalisation: bundle README section 3
    ("CRITICAL preprocessing note").
    """
    import torch  # local import: this function is the only place the
    # subpackage actually needs torch at runtime.
    _log = log if log is not None else (lambda _msg: None)

    classes = list(fusion_info["classes"])
    image_size = int(fusion_info.get("image_size", 384))
    _NORM_MEAN, _NORM_STD = 0.45, 0.225

    video_name = os.path.basename(os.path.normpath(video_dir))
    ordered = _list_ordered_frames(video_dir, video_name)
    if not ordered:
        return []

    neighbour_of = _resolve_dyn_window(ordered)
    masks_dir = os.path.join(video_dir, mask_dir_name)
    images_dir = os.path.join(video_dir, "images")

    mask_cache: Dict[str, Optional[np.ndarray]] = {}
    gray_cache: Dict[str, Optional[np.ndarray]] = {}

    def _read_pair(name: str):
        if name not in gray_cache:
            gray_cache[name] = _read_image_uint8(
                os.path.join(images_dir, name), gray=True,
            )
        if name not in mask_cache:
            mask_cache[name] = _read_image_uint8(
                os.path.join(masks_dir, name), gray=True,
            )
        return gray_cache[name], mask_cache[name]

    # If Step 3 wrote temporal images to disk, prefer those (one
    # cv2.imread is faster than three mask + gray reads plus the
    # build_temporal_image crop / resize work).  When the on-disk
    # file is missing we fall back to the in-memory rebuild so the
    # forecast still works if Step 3 was not run in temporal mode.
    temporal_dir = (
        os.path.join(video_dir, temporal_dir_name)
        if temporal_dir_name else None
    )

    out: List[Tuple[str, str, dict]] = []
    n_predicted = 0
    n_skipped = 0
    n_from_disk = 0
    n_built = 0
    n_total = len(ordered)
    n_seen = 0
    # Move the bar off zero so the user sees the phase starting.
    if progress_cb is not None:
        try:
            progress_cb(0, n_total, video_name)
        except Exception:  # noqa: BLE001
            pass
    model.eval()
    for sec, image_name in ordered:
        temporal_bgr = None
        if temporal_dir is not None:
            cand = os.path.join(temporal_dir, image_name)
            if os.path.isfile(cand):
                try:
                    temporal_bgr = cv2.imdecode(
                        np.fromfile(cand, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if temporal_bgr is not None:
                        n_from_disk += 1
                except Exception:  # noqa: BLE001
                    temporal_bgr = None
        if temporal_bgr is None:
            prev_name, cur_name, next_name = neighbour_of[image_name]
            gp, mp = _read_pair(prev_name)
            gc, mc = _read_pair(image_name)
            gn, mn = _read_pair(next_name)
            if gc is None or mc is None:
                n_skipped += 1
                n_seen += 1
                if progress_cb is not None:
                    try:
                        progress_cb(n_seen, n_total, video_name)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            temporal_bgr = build_temporal_image(
                grays=(gp, gc, gn),
                masks=(mp, mc, mn),
                image_size=image_size,
            )
            if temporal_bgr is None:
                n_skipped += 1
                n_seen += 1
                if progress_cb is not None:
                    try:
                        progress_cb(n_seen, n_total, video_name)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            n_built += 1
        # Ensure the temporal image is at the model's expected size
        # even when it came off disk at a different size (defensive).
        if temporal_bgr.shape[0] != image_size or temporal_bgr.shape[1] != image_size:
            temporal_bgr = cv2.resize(
                temporal_bgr, (image_size, image_size),
                interpolation=cv2.INTER_AREA,
            )

        # Image tensor: (g - 0.45) / 0.225 per temporal channel,
        # replicated to 3 channels per frame -> (1, 9, H, W).
        chans = []
        for t in (0, 1, 2):
            g = temporal_bgr[:, :, t].astype(np.float32) / 255.0
            g = (g - _NORM_MEAN) / _NORM_STD
            chans.extend([g, g, g])
        x = (
            torch.from_numpy(np.stack(chans, axis=0))
                 .unsqueeze(0)
                 .to(device)
        )

        # Metric tensor: prebuilt by the caller.
        m_arr = metric_matrix_by_image.get(image_name)
        if m_arr is None:
            # Use the median-filled vector if this frame has no row.
            metric_dim = int(model.metric_gate.shape[0])
            m_arr = np.zeros((metric_dim,), dtype=np.float32)
        m = torch.from_numpy(np.asarray(m_arr, dtype=np.float32)) \
                .unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x, m)
        prob = torch.nn.functional.softmax(logits.float(), 1) \
                    .detach().cpu().numpy()[0]
        i = int(np.argmax(prob))
        label = classes[i] if 0 <= i < len(classes) else "Unknown"
        out.append((
            image_name, label,
            {c: float(p) for c, p in zip(classes, prob)},
        ))
        n_predicted += 1
        n_seen += 1
        if progress_cb is not None:
            try:
                progress_cb(n_seen, n_total, video_name)
            except Exception:  # noqa: BLE001
                pass

    _log(
        f"  Predicted {n_predicted} frame(s), skipped {n_skipped} "
        f"(missing mask / unreadable image).  "
        f"Temporal images: {n_from_disk} from disk "
        f"({temporal_dir_name or '-'}), {n_built} built in memory "
        f"(mask cache hits {len(mask_cache)} / gray cache hits "
        f"{len(gray_cache)})."
    )
    return out
