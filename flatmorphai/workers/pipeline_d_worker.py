"""
Mask metrics worker: extract behavioural and morphometric metrics from
binary planarian masks.

Workflow
--------
1. Resolve the per-video directory layout under ``processing_root``.
   When the requested ``step_seconds`` differs from what the existing
   directory name encodes, or when the directory does not exist, the
   worker regenerates frames (Step 1) and masks (Step 2) so the
   metric run always operates on a sampling rate the user actually
   asked for.
2. For each per-video directory it loads every mask, computes
   per-frame morphometrics, and accumulates a sequence of
   :class:`~flatmorphai.notebook_api.metrics.FrameMetrics`.
3. Once every video has been processed it builds the dataset-level
   curvature matrix, runs PCA-based eigenposture decomposition with
   ``random_state=42`` (Stephens, G. J., et al., "Dimensionality and
   dynamics in the behavior of C. elegans", PLOS Computational
   Biology, 2008, DOI: 10.1371/journal.pcbi.1000028) and writes the
   per-frame and per-video CSVs to ``processing_root`` with
   ``sep=';'``.

References
----------
The morphometric and behavioural feature list mirrors the consensus
descriptors used in nematode behavioural genomics, validated for
flatworms by analogy with similar undulatory body plans:

  - Yemini, E. et al., "A database of Caenorhabditis elegans
    behavioral phenotypes", Nature Methods 10, 877-879, 2013,
    DOI: 10.1038/nmeth.2560.
  - Javer, A. et al., "An open-source platform for analyzing and
    sharing worm-behavior data", Nature Methods 15, 645-646, 2018,
    DOI: 10.1038/s41592-018-0112-1 (Tierpsy Tracker).
  - Swierczek, N. A., Giles, A. C., Rankin, C. H. and Kerr, R. A.,
    "High-throughput behavioral analysis in C. elegans", Nature
    Methods 8, 592-598, 2011, DOI: 10.1038/nmeth.1625
    (Multi-Worm Tracker / WormTracker 2.0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..base_worker import BaseWorker
from ..constants import CHECKPOINTS_SEGMENTATION_DIR
from ..model_cache import ModelCache
from ..notebook_api.metrics import (
    FrameMetrics,
    VideoTemporalMetrics,
    WellGeometry,
    compute_frame_metrics,
    detect_well_circle,
    eigenposture_decomposition,
    read_mask,
    temporal_metrics,
    write_per_frame_csv,
    write_per_video_csv,
)
from ..notebook_api.naming import (
    VIDEO_EXTENSIONS,
    parse_directory_name,
    parse_image_name,
    parse_video_name,
)
from ..notebook_api.segmentation import (
    forecast_mask_with_a_mask2former,
    load_model_for_segmentation,
)
from ..notebook_api.video_frames import process_all_videos_in_directory


# Default mask-folder prefix produced by Step 2.
_MASK_DIR_PREFIX = "masks"


@dataclass
class PipelineDParams:
    """Inputs for :class:`PipelineDWorker`.

    Attributes
    ----------
    input_videos_directory:
        Optional directory holding the raw .mp4 files. Required only
        when frames or masks need to be regenerated.
    processing_root:
        Directory holding (or destined to hold) the per-video
        sub-directories.
    output_directory:
        Where the metrics CSVs will be written. Defaults to the
        processing root so the files sit next to the per-video
        folders.
    step_seconds:
        Target sampling step. When the existing per-video directory
        name encodes a different value, the worker triggers a Step 1
        + Step 2 regeneration with the requested rate.
    seg_checkpoint_root, seg_model_name:
        Segmentation model used during regeneration. Required only
        when at least one video does not already have masks at the
        requested sampling rate.
    detect_well:
        When ``True``, the worker detects the well circle from the
        first frame of each video via the Hough transform and uses
        it to compute well-normalised spatial descriptors.
    eigenposture_components:
        Number of PCA components retained for the eigenposture
        projection. Stephens et al., 2008 (DOI: 10.1371/journal.pcbi
        .1000028) reported >95 percent of postural variance is
        captured by 4 to 6 components for nematodes; we default to 6.
    """

    processing_root: str
    input_videos_directory: str = ""
    output_directory: str = ""
    step_seconds: float = 1.0
    seg_checkpoint_root: str = CHECKPOINTS_SEGMENTATION_DIR
    seg_model_name: str = ""
    detect_well: bool = True
    eigenposture_components: int = 6
    output_per_frame_csv: str = "metrics_per_frame.csv"
    output_per_video_csv: str = "metrics_per_video.csv"


def _existing_step_seconds(dir_name: str) -> Optional[float]:
    parsed = parse_directory_name(dir_name)
    if parsed is None:
        return None
    _meta, _total, step = parsed
    return float(step)


def _list_existing_video_dirs(processing_root: str) -> List[str]:
    if not os.path.isdir(processing_root):
        return []
    return sorted(
        name for name in os.listdir(processing_root)
        if os.path.isdir(os.path.join(processing_root, name))
        and parse_directory_name(name) is not None
    )


def _list_input_videos(input_dir: str) -> List[str]:
    if not input_dir or not os.path.isdir(input_dir):
        return []
    files: List[str] = []
    for root, _dirs, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith(VIDEO_EXTENSIONS):
                files.append(os.path.join(root, name))
    return sorted(files)


def _video_has_masks_at_rate(
    processing_root: str, video_stem: str, step_seconds: float,
) -> Optional[str]:
    """Return the per-video directory whose sampling step matches.

    Returns the directory name (relative to ``processing_root``) on
    success, ``None`` when no matching directory exists.
    """
    if not os.path.isdir(processing_root):
        return None
    for entry in os.listdir(processing_root):
        full = os.path.join(processing_root, entry)
        if not os.path.isdir(full):
            continue
        parsed = parse_directory_name(entry)
        if parsed is None:
            continue
        meta, _total, step = parsed
        if meta.video_stem != video_stem:
            continue
        if abs(step - step_seconds) > 1e-6:
            continue
        # Also require at least one mask folder to be populated.
        for sub in os.listdir(full):
            if sub.startswith(_MASK_DIR_PREFIX) and os.path.isdir(
                os.path.join(full, sub)
            ):
                if any(
                    fname.lower().endswith(".png")
                    for fname in os.listdir(os.path.join(full, sub))
                ):
                    return entry
    return None


def _first_mask_dir(video_dir: str) -> Optional[str]:
    """Return the first ``masks_<model>`` sub-directory of ``video_dir``."""
    for entry in sorted(os.listdir(video_dir)):
        if entry.startswith(_MASK_DIR_PREFIX) and os.path.isdir(
            os.path.join(video_dir, entry)
        ):
            return entry
    return None


def _first_image_path(video_dir: str) -> Optional[str]:
    images_dir = os.path.join(video_dir, "images")
    if not os.path.isdir(images_dir):
        return None
    for entry in sorted(os.listdir(images_dir)):
        if entry.lower().endswith(".png"):
            return os.path.join(images_dir, entry)
    return None


class PipelineDWorker(BaseWorker):
    """End-to-end metric extraction worker."""

    def __init__(self, params: PipelineDParams) -> None:
        super().__init__()
        self.params = params

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 - override
        params = self.params
        processing_root = params.processing_root
        if not processing_root:
            self.error_signal.emit("Mask metrics: processing root is empty")
            return
        os.makedirs(processing_root, exist_ok=True)
        out_dir = params.output_directory or processing_root
        os.makedirs(out_dir, exist_ok=True)
        self.log(
            f"Mask metrics: target step={params.step_seconds}s, "
            f"processing_root={processing_root}"
        )

        try:
            self._maybe_regenerate(processing_root)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log_signal.emit(
                f"TRACEBACK:\n{traceback.format_exc()}"
            )
            self.error_signal.emit(
                f"Regeneration failed: {type(exc).__name__}: {exc}"
            )
            return

        # Enumerate per-video directories that match the requested rate.
        eligible = self._select_video_dirs(processing_root)
        if not eligible:
            self.error_signal.emit(
                "Mask metrics: no per-video directory matched the "
                f"requested step={params.step_seconds}s under "
                f"{processing_root}"
            )
            return
        self.log(
            f"Computing metrics over {len(eligible)} video(s) at "
            f"step={params.step_seconds}s"
        )

        all_frames: List[FrameMetrics] = []
        all_video_origin: List[str] = []
        per_video_aggregates: List[VideoTemporalMetrics] = []
        total = max(1, len(eligible))
        for idx, video_dir_name in enumerate(eligible):
            if self.is_interrupted():
                self.log("Interrupted by user.")
                break
            video_dir = os.path.join(processing_root, video_dir_name)
            try:
                frames = self._process_video(video_dir)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log_signal.emit(
                    f"ERROR processing {video_dir_name}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                continue
            stem = parse_directory_name(video_dir_name)[0].video_stem
            all_frames.extend(frames)
            all_video_origin.extend([stem] * len(frames))
            per_video_aggregates.append(
                temporal_metrics(
                    frames,
                    video_name=stem,
                    step_seconds=params.step_seconds,
                )
            )
            self.progress(int(((idx + 1) / total) * 80))

        if not all_frames:
            self.error_signal.emit(
                "Mask metrics produced no frames; cannot continue."
            )
            return

        # Eigenposture decomposition on the dataset-level curvature
        # matrix. The PCA seed is fixed for reproducibility.
        self.log("Running eigenposture decomposition (random_state=42)")
        curvature_matrix = np.array(
            [list(f.curvature_profile) for f in all_frames if f.curvature_profile],
            dtype=np.float64,
        )
        eigenposture_columns: Dict[str, List[float]] = {}
        eigen_explained: Optional[np.ndarray] = None
        if curvature_matrix.size > 0:
            projections, _components, eigen_explained = (
                eigenposture_decomposition(
                    curvature_matrix,
                    n_components=params.eigenposture_components,
                )
            )
            for k in range(projections.shape[1]):
                eigenposture_columns[f"eigenposture_{k + 1}"] = (
                    projections[:, k].tolist()
                )
            self.log(
                "Eigenposture explained variance ratio: "
                + ", ".join(f"{v:.3f}" for v in eigen_explained)
            )

        # Write the CSVs.
        per_frame_path = os.path.join(out_dir, params.output_per_frame_csv)
        per_video_path = os.path.join(out_dir, params.output_per_video_csv)
        # Pad the eigenposture columns so they align row-for-row when
        # some frames had no curvature profile.
        if eigenposture_columns:
            full_eigen: Dict[str, List[float]] = {}
            for name in eigenposture_columns:
                full_eigen[name] = []
            for f in all_frames:
                if f.curvature_profile:
                    for name in eigenposture_columns:
                        full_eigen[name].append(
                            eigenposture_columns[name].pop(0)
                        )
                else:
                    for name in eigenposture_columns:
                        full_eigen[name].append(float("nan"))
            extra = {
                "video_name": all_video_origin,
                **full_eigen,
            }
        else:
            extra = {"video_name": all_video_origin}

        write_per_frame_csv(all_frames, per_frame_path, extra_columns=extra)
        write_per_video_csv(per_video_aggregates, per_video_path)
        self.log(f"Wrote {per_frame_path}")
        self.log(f"Wrote {per_video_path}")
        self.progress(100)
        self.result_signal.emit(
            {
                "per_frame_csv": params.output_per_frame_csv,
                "per_video_csv": params.output_per_video_csv,
                "n_frames": int(len(all_frames)),
                "n_videos": int(len(per_video_aggregates)),
            }
        )

    # ------------------------------------------------------------------
    # Mask regeneration (Step 1 + Step 2)
    # ------------------------------------------------------------------
    def _maybe_regenerate(self, processing_root: str) -> None:
        """Regenerate frames and masks for any video missing the requested rate."""
        params = self.params
        videos = _list_input_videos(params.input_videos_directory)
        if not videos:
            # Nothing to regenerate: assume the user has already run
            # Step 1 + Step 2 outside the GUI and the existing dirs
            # carry the requested step.
            self.log(
                "Mask metrics: no input video directory provided; skipping "
                "regeneration and using existing per-video directories."
            )
            return
        to_generate: List[str] = []
        for video_path in videos:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            video_meta = parse_video_name(stem)
            if video_meta is None:
                self.log(
                    f"Mask metrics: skipping video '{stem}' which does "
                    "not match the v1.2 naming convention."
                )
                continue
            existing = _video_has_masks_at_rate(
                processing_root, video_meta.video_stem, params.step_seconds,
            )
            if existing is None:
                to_generate.append(video_path)
        if not to_generate:
            self.log(
                "Mask metrics: every input video already has masks at "
                f"step={params.step_seconds}s; skipping regeneration."
            )
            return

        self.log(
            f"Mask metrics: regenerating frames + masks for "
            f"{len(to_generate)} video(s) at step={params.step_seconds}s"
        )

        # Step 1: write frames into temporary per-video directories
        # carrying the requested step in their name.
        tmp_input_dir = os.path.join(
            params.input_videos_directory, "_pipeline_d_input"
        )
        # We do not actually move the videos; we ask the helper to
        # walk the parent directory of each chosen file, but we want
        # to filter to ``to_generate``. The simplest reliable path is
        # to rerun the standard helper on the parent directory of the
        # input list and let unaffected videos that are already in
        # cache be reprocessed (idempotent, just slightly more I/O).
        process_all_videos_in_directory(
            input_directory=params.input_videos_directory,
            output_directory=processing_root,
            step_seconds=params.step_seconds,
            log=self.log,
            progress_cb=lambda done, total: self.progress(
                int((done / total) * 25) if total else 0
            ),
            is_interrupted=self.is_interrupted,
        )

        # Step 2: load segmentation model and produce masks.
        model_name = params.seg_model_name
        if not model_name:
            self.error_signal.emit(
                "Mask metrics: a segmentation model name is required "
                "to regenerate masks. Configure it on the Settings page."
            )
            return
        cache_key = f"seg::{params.seg_checkpoint_root}::{model_name}"
        cache = ModelCache()
        if cache.has(cache_key):
            model, processor = cache.get(cache_key)
            self.log("Mask metrics: using cached segmentation model")
        else:
            self.log(
                f"Mask metrics: loading segmentation model '{model_name}'"
            )
            device = self.resolve_device()
            model, processor = load_model_for_segmentation(
                model_name, device=device,
                checkpoint_root=params.seg_checkpoint_root,
            )
            cache.set(cache_key, (model, processor))
        device = self.resolve_device()
        forecast_mask_with_a_mask2former(
            directory_all_images=processing_root,
            model=model,
            processor=processor,
            device=device,
            model_name=model_name,
            log=self.log,
            progress_cb=lambda done, total, _label="": self.progress(
                25 + int((done / total) * 25) if total else 25
            ),
            is_interrupted=self.is_interrupted,
        )

    # ------------------------------------------------------------------
    # Per-video frame iteration
    # ------------------------------------------------------------------
    def _select_video_dirs(self, processing_root: str) -> List[str]:
        params = self.params
        out: List[str] = []
        for entry in _list_existing_video_dirs(processing_root):
            step = _existing_step_seconds(entry)
            if step is None:
                continue
            if abs(step - params.step_seconds) > 1e-6:
                continue
            out.append(entry)
        return out

    def _process_video(self, video_dir: str) -> List[FrameMetrics]:
        mask_dir_name = _first_mask_dir(video_dir)
        if mask_dir_name is None:
            self.log(
                f"  No mask folder under {os.path.basename(video_dir)}; "
                "skipping."
            )
            return []
        mask_dir = os.path.join(video_dir, mask_dir_name)
        well = self._resolve_well(video_dir)
        frames: List[FrameMetrics] = []
        png_files = sorted(
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(".png")
        )
        if not png_files:
            return []
        self.log(
            f"  {os.path.basename(video_dir)}: {len(png_files)} mask(s)"
        )
        for image_name in png_files:
            if self.is_interrupted():
                break
            try:
                mask = read_mask(os.path.join(mask_dir, image_name))
            except Exception as exc:  # noqa: BLE001
                self.log(
                    f"    Skipped {image_name}: {type(exc).__name__}: {exc}"
                )
                continue
            second = self._second_for(image_name, video_dir)
            try:
                frame = compute_frame_metrics(
                    mask,
                    image_name=image_name,
                    second=second,
                    well=well,
                )
            except Exception as exc:  # noqa: BLE001
                self.log(
                    f"    Skipped {image_name}: {type(exc).__name__}: {exc}"
                )
                continue
            frames.append(frame)
        return frames

    def _resolve_well(self, video_dir: str) -> Optional[WellGeometry]:
        if not self.params.detect_well:
            return None
        first = _first_image_path(video_dir)
        if first is None:
            return None
        try:
            import cv2  # local import
            img = cv2.imdecode(
                np.fromfile(first, dtype=np.uint8), cv2.IMREAD_UNCHANGED,
            )
        except Exception:  # noqa: BLE001
            return None
        if img is None:
            return None
        well = detect_well_circle(img)
        if well is not None:
            self.log(
                f"  Detected well: cx={well.cx:.0f}, cy={well.cy:.0f}, "
                f"r={well.axis_a:.0f} px"
            )
        return well

    @staticmethod
    def _second_for(image_name: str, video_dir: str) -> float:
        parent = os.path.basename(video_dir.rstrip("/\\"))
        meta = parse_image_name(image_name, parent_directory_name=parent)
        if meta is not None:
            return float(meta.second)
        # Fallback: parse a trailing ``_<second>s`` from the stem.
        stem = os.path.splitext(image_name)[0]
        trailing = stem.rsplit("_", 1)[-1]
        if trailing.endswith("s"):
            trailing = trailing[:-1]
        try:
            return float(trailing)
        except ValueError:
            return 0.0
