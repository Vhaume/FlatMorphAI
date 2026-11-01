"""
Scrunching detector worker: scrunching versus gliding state per frame from masks.

The worker reuses Mask metrics's mask-loading and regeneration logic to
guarantee the masks were sampled at a step compatible with the
scrunching frequency band (default 0.2-0.8 Hz, see
:mod:`flatmorphai.notebook_api.scrunching`). For each per-video
directory it:

  1. Loads the binary mask of every frame.
  2. Computes the body length (skeleton arc length) and centroid via
     :func:`flatmorphai.notebook_api.metrics.compute_frame_metrics`.
  3. Builds the body-length time series ``L(t)`` and centroid path.
  4. Runs the sliding-window FFT classifier
     :func:`flatmorphai.notebook_api.scrunching.analyse_video` and
     records, per frame, the state and a continuous scrunching
     intensity score.
  5. Aggregates the result into a per-video summary row via
     :func:`flatmorphai.notebook_api.scrunching.summarise_video`.

References
----------
- Cochet-Escartin, O., Mickolajczyk, K. J. and Collins, E. M.-S.,
  "Scrunching: a novel escape gait in planarians", Physical Biology,
  2015, DOI: 10.1088/1478-3975/12/5/056010 (algorithm).
- Sabry, Z. et al., "Planarian Scrunching as a Quantitative
  Behavioral Readout for Noxious Stimuli Sensing", Journal of
  Visualized Experiments 161: e61549, 2020,
  DOI: 10.3791/61549 (protocol).
- Reho, G., Lelievre, V. and Cadiou, H., "Planarian nociception:
  Lessons from a scrunching flatworm", Frontiers in Molecular
  Neuroscience, 2022, DOI: 10.3389/fnmol.2022.935918 (review).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..base_worker import BaseWorker
from ..constants import CHECKPOINTS_SEGMENTATION_DIR
from ..model_cache import ModelCache
from ..notebook_api.metrics import (
    compute_frame_metrics,
    read_mask,
)
from ..notebook_api.naming import (
    parse_directory_name,
    parse_image_name,
    parse_video_name,
    VIDEO_EXTENSIONS,
)
from ..notebook_api.scrunching import (
    DEFAULT_WINDOW_SECONDS,
    GLIDING_AMPLITUDE_THRESHOLD,
    GLIDING_DIRECTIONALITY,
    IMMOBILE_SPEED_PX_PER_S,
    SCRUNCHING_AMPLITUDE_THRESHOLD,
    SCRUNCHING_FREQ_BAND_HZ,
    analyse_video,
    step_seconds_warning,
    summarise_video,
    write_per_frame_csv,
    write_per_video_csv,
)
from ..notebook_api.segmentation import (
    forecast_mask_with_a_mask2former,
    load_model_for_segmentation,
)
from ..notebook_api.video_frames import process_all_videos_in_directory


_MASK_DIR_PREFIX = "masks"


@dataclass
class PipelineEParams:
    """Inputs for :class:`PipelineEWorker`.

    Attributes
    ----------
    processing_root:
        Directory holding the per-video sub-directories.
    input_videos_directory:
        Optional source directory of raw .mp4 files used when masks
        must be regenerated at the requested ``step_seconds``.
    output_directory:
        Where the two CSVs are written. Defaults to the processing
        root.
    step_seconds:
        Target sampling step. Per Shannon's sampling theorem and the
        0.2-0.8 Hz scrunching band, values above 0.5 s alias the
        oscillation; the worker emits a warning when the value is too
        coarse but still runs the analysis.
    window_seconds:
        Sliding-window width. Default 5.0 s covers two to four full
        cycles at the typical 0.4-0.6 Hz centre frequency, which is
        the Bracewell minimum for a stable FFT magnitude estimate.
    freq_band_hz:
        Band used to localise the dominant scrunching frequency.
        Defaults to (0.2, 0.8) per Cochet-Escartin et al., 2015.
    scrunching_amplitude_threshold:
        Minimum window-level relative elongation amplitude to enter
        the scrunching state. Default 0.20.
    gliding_amplitude_threshold:
        Maximum amplitude tolerated for gliding. Default 0.05.
    immobile_speed_threshold:
        Window-level centroid speed below which the animal is treated
        as immobile (pixels per second). Default 1.0.
    gliding_directionality_threshold:
        Window-level directionality (net displacement / total path)
        above which the animal is treated as gliding. Default 0.7.
    seg_checkpoint_root, seg_model_name:
        Segmentation model used during regeneration. Required only
        when masks are missing or were sampled at a different step.
    """

    processing_root: str
    input_videos_directory: str = ""
    output_directory: str = ""
    step_seconds: float = 0.25
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    freq_band_hz: Tuple[float, float] = SCRUNCHING_FREQ_BAND_HZ
    scrunching_amplitude_threshold: float = SCRUNCHING_AMPLITUDE_THRESHOLD
    gliding_amplitude_threshold: float = GLIDING_AMPLITUDE_THRESHOLD
    immobile_speed_threshold: float = IMMOBILE_SPEED_PX_PER_S
    gliding_directionality_threshold: float = GLIDING_DIRECTIONALITY
    seg_checkpoint_root: str = CHECKPOINTS_SEGMENTATION_DIR
    seg_model_name: str = ""
    output_per_frame_csv: str = "scrunching_per_frame.csv"
    output_per_video_csv: str = "scrunching_per_video.csv"


def _list_existing_video_dirs(processing_root: str) -> List[str]:
    if not os.path.isdir(processing_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(processing_root)):
        full = os.path.join(processing_root, name)
        if os.path.isdir(full) and parse_directory_name(name) is not None:
            out.append(name)
    return out


def _existing_step_seconds(dir_name: str) -> Optional[float]:
    parsed = parse_directory_name(dir_name)
    if parsed is None:
        return None
    _meta, _total, step = parsed
    return float(step)


def _first_mask_dir(video_dir: str) -> Optional[str]:
    for entry in sorted(os.listdir(video_dir)):
        if entry.startswith(_MASK_DIR_PREFIX) and os.path.isdir(
            os.path.join(video_dir, entry)
        ):
            return entry
    return None


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


class PipelineEWorker(BaseWorker):
    """End-to-end scrunching analysis."""

    def __init__(self, params: PipelineEParams) -> None:
        super().__init__()
        self.params = params

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 - override
        params = self.params
        if not params.processing_root:
            self.error_signal.emit("Scrunching detector: processing root is empty")
            return
        os.makedirs(params.processing_root, exist_ok=True)
        out_dir = params.output_directory or params.processing_root
        os.makedirs(out_dir, exist_ok=True)

        warning = step_seconds_warning(
            params.step_seconds, freq_band=params.freq_band_hz,
        )
        if warning:
            self.log("WARNING: " + warning)

        try:
            self._maybe_regenerate(params.processing_root)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log_signal.emit(
                f"TRACEBACK:\n{traceback.format_exc()}"
            )
            self.error_signal.emit(
                f"Regeneration failed: {type(exc).__name__}: {exc}"
            )
            return

        eligible = self._select_video_dirs(params.processing_root)
        if not eligible:
            self.error_signal.emit(
                "Scrunching detector: no per-video directory matched the "
                f"requested step={params.step_seconds}s under "
                f"{params.processing_root}"
            )
            return
        self.log(
            f"Scrunching detector: analysing {len(eligible)} video(s) at "
            f"step={params.step_seconds}s with window="
            f"{params.window_seconds}s"
        )

        all_states = []
        all_video_origin = []
        per_video_summaries = []
        total = max(1, len(eligible))
        for idx, video_dir_name in enumerate(eligible):
            if self.is_interrupted():
                self.log("Interrupted by user.")
                break
            video_dir = os.path.join(params.processing_root, video_dir_name)
            try:
                states = self._process_video(video_dir)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log_signal.emit(
                    f"ERROR processing {video_dir_name}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                continue
            stem = parse_directory_name(video_dir_name)[0].video_stem
            all_states.extend(states)
            all_video_origin.extend([stem] * len(states))
            per_video_summaries.append(
                summarise_video(
                    states, video_name=stem, step_seconds=params.step_seconds,
                )
            )
            self.progress(int(((idx + 1) / total) * 90))

        if not all_states:
            self.error_signal.emit(
                "Scrunching detector produced no labelled frames; cannot continue."
            )
            return

        per_frame_path = os.path.join(out_dir, params.output_per_frame_csv)
        per_video_path = os.path.join(out_dir, params.output_per_video_csv)
        write_per_frame_csv(
            all_states, per_frame_path,
            extra_columns={"video_name": all_video_origin},
        )
        write_per_video_csv(per_video_summaries, per_video_path)
        self.log(f"Wrote {per_frame_path}")
        self.log(f"Wrote {per_video_path}")
        self.progress(100)
        self.result_signal.emit({
            "per_frame_csv": params.output_per_frame_csv,
            "per_video_csv": params.output_per_video_csv,
            "n_frames": int(len(all_states)),
            "n_videos": int(len(per_video_summaries)),
        })

    # ------------------------------------------------------------------
    # Mask regeneration (Step 1 + Step 2) when needed
    # ------------------------------------------------------------------
    def _maybe_regenerate(self, processing_root: str) -> None:
        params = self.params
        videos = _list_input_videos(params.input_videos_directory)
        if not videos:
            self.log(
                "Scrunching detector: no input video directory provided; using "
                "existing per-video directories without regeneration."
            )
            return
        to_generate: List[str] = []
        for video_path in videos:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            video_meta = parse_video_name(stem)
            if video_meta is None:
                self.log(
                    f"Scrunching detector: skipping video '{stem}' which does "
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
                "Scrunching detector: every input video already has masks at "
                f"step={params.step_seconds}s; skipping regeneration."
            )
            return

        self.log(
            "Scrunching detector: regenerating frames + masks for "
            f"{len(to_generate)} video(s) at step={params.step_seconds}s"
        )
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
        if not params.seg_model_name:
            self.error_signal.emit(
                "Scrunching detector: a segmentation model name is required to "
                "regenerate masks at the requested step. Configure it "
                "on the Settings page."
            )
            return
        cache_key = f"seg::{params.seg_checkpoint_root}::{params.seg_model_name}"
        cache = ModelCache()
        if cache.has(cache_key):
            model, processor = cache.get(cache_key)
            self.log("Scrunching detector: using cached segmentation model")
        else:
            device = self.resolve_device()
            self.log(
                f"Scrunching detector: loading segmentation model "
                f"'{params.seg_model_name}'"
            )
            model, processor = load_model_for_segmentation(
                params.seg_model_name, device=device,
                checkpoint_root=params.seg_checkpoint_root,
            )
            cache.set(cache_key, (model, processor))
        device = self.resolve_device()
        forecast_mask_with_a_mask2former(
            directory_all_images=processing_root,
            model=model,
            processor=processor,
            device=device,
            model_name=params.seg_model_name,
            log=self.log,
            progress_cb=lambda done, total, _label="": self.progress(
                25 + int((done / total) * 25) if total else 25
            ),
            is_interrupted=self.is_interrupted,
        )

    # ------------------------------------------------------------------
    # Per-video state computation
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

    def _process_video(self, video_dir: str):
        params = self.params
        mask_dir_name = _first_mask_dir(video_dir)
        if mask_dir_name is None:
            self.log(
                f"  No mask folder under {os.path.basename(video_dir)}; "
                "skipping."
            )
            return []
        mask_dir = os.path.join(video_dir, mask_dir_name)
        png_files = sorted(
            f for f in os.listdir(mask_dir) if f.lower().endswith(".png")
        )
        if not png_files:
            return []
        self.log(
            f"  {os.path.basename(video_dir)}: {len(png_files)} mask(s)"
        )

        image_names: List[str] = []
        seconds: List[float] = []
        body_lengths: List[float] = []
        centroids: List = []
        areas: List[float] = []
        parent = os.path.basename(video_dir.rstrip("/\\"))
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
            try:
                frame = compute_frame_metrics(
                    mask, image_name=image_name,
                    second=self._second_for(image_name, parent),
                )
            except Exception as exc:  # noqa: BLE001
                self.log(
                    f"    Skipped {image_name}: {type(exc).__name__}: {exc}"
                )
                continue
            image_names.append(image_name)
            seconds.append(frame.second)
            body_lengths.append(frame.skeleton_length_px)
            centroids.append((frame.centroid_x, frame.centroid_y))
            areas.append(frame.area_px)

        states = analyse_video(
            image_names=image_names,
            seconds=seconds,
            body_lengths=body_lengths,
            centroids=centroids,
            areas=areas,
            step_seconds=params.step_seconds,
            window_seconds=params.window_seconds,
            freq_band=params.freq_band_hz,
            scrunching_amplitude_threshold=params.scrunching_amplitude_threshold,
            gliding_amplitude_threshold=params.gliding_amplitude_threshold,
            immobile_speed_threshold=params.immobile_speed_threshold,
            gliding_directionality_threshold=params.gliding_directionality_threshold,
        )
        return states

    @staticmethod
    def _second_for(image_name: str, parent: str) -> float:
        meta = parse_image_name(image_name, parent_directory_name=parent)
        if meta is not None:
            return float(meta.second)
        stem = os.path.splitext(image_name)[0]
        trailing = stem.rsplit("_", 1)[-1]
        if trailing.endswith("s"):
            trailing = trailing[:-1]
        try:
            return float(trailing)
        except ValueError:
            return 0.0
