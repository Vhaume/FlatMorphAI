"""
Image segmentation worker: segmentation-only on videos or image folders.

Unlike Stereotype classification and Stereotype classification from images, Image segmentation accepts either a folder
of videos or a folder of already-extracted images, and it does not
run the classification / forecasting stages. The output is a sibling
directory called ``segmented_<source name>`` containing one
binary-mask PNG per input frame.

Rationale
---------
Biologists often need a quick "silhouette only" preview of a
candidate video before committing to the full stereotype pipeline.
Exposing a dedicated fast path for this use case follows the
recommendation of

  - Carpenter, A. E. et al., "CellProfiler: image analysis software
    for identifying and quantifying cell phenotypes", Genome
    Biology, 2006, DOI: 10.1186/gb-2006-7-10-r100, which documents
    the need for a mask-first preview workflow when bulk phenotype
    scoring is expensive.

Sampling
--------
When the input is a video, the user picks a ``[start_second,
end_second]`` window and a ``step_seconds`` cadence (default
``1.0``), following the v1.0 naming convention module. When the
input is a pre-extracted image directory, every image is processed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..base_worker import BaseWorker
from ..constants import CHECKPOINTS_SEGMENTATION_DIR
from ..model_cache import ModelCache
from ..notebook_api.naming import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from ..notebook_api.segmentation import (
    forecast_mask_with_a_mask2former,
    load_model_for_segmentation,
    resolve_most_recent_checkpoint,
)
from ..notebook_api.video_frames import process_all_videos_in_directory


@dataclass
class PipelineCParams:
    """User-configurable parameters for Image segmentation.

    Attributes
    ----------
    source_directory:
        Either a directory full of videos or a directory full of
        images. Detection falls back to file extensions when
        ``source_type`` is ``"auto"``.
    output_directory:
        Parent directory that will receive the
        ``segmented_<source name>`` result folder.
    source_type:
        ``"videos"`` or ``"images"`` when the user chose explicitly
        through the Image segmentation page radio buttons, or ``"auto"`` for
        the historical extension-based detection.
    start_second, end_second, step_seconds:
        Video-sampling window used when the source is a video folder.
        Ignored when the source is an image folder. ``end_second``
        defaults to ``floor(duration)`` at run time.
    checkpoint_root, model_name:
        Segmentation checkpoint selection. ``model_name`` defaults to
        the most recently trained checkpoint.
    batch_size, two_pass, two_pass_merge, keep_largest_component,
    overwrite:
        Same semantics as Step 2.
    """

    source_directory: str
    output_directory: str
    source_type: str = "auto"
    start_second: float = 0.0
    end_second: Optional[float] = None
    step_seconds: float = 1.0
    checkpoint_root: str = CHECKPOINTS_SEGMENTATION_DIR
    model_name: str = ""
    batch_size: int = 8
    two_pass: bool = True
    two_pass_merge: str = "skeleton_guided"
    keep_largest_component: bool = True
    overwrite: bool = False


def _is_video_directory(path: str) -> bool:
    """Return True when ``path`` holds at least one supported video file."""
    if not os.path.isdir(path):
        return False
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXTENSIONS):
                return True
    return False


def _is_image_directory(path: str) -> bool:
    """Return True when ``path`` holds at least one supported image file."""
    if not os.path.isdir(path):
        return False
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(IMAGE_EXTENSIONS):
                return True
    return False


class PipelineCWorker(BaseWorker):
    """Segment every frame or every image under ``source_directory``."""

    def __init__(self, params: PipelineCParams) -> None:
        super().__init__()
        self.params = params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _progress_cb(self, done: int, total: int, label: str = "") -> None:
        if total > 0:
            self.progress(int(done / total * 100))

    def _per_frame_cb(self, done: int, total: int, label: str = "") -> None:
        if total > 0:
            # Extraction and segmentation each get half of the bar.
            self.progress(int(done / total * 50))

    def _staging_directory(self) -> str:
        src = os.path.basename(self.params.source_directory.rstrip("/\\"))
        staging_name = f"segmented_{src}"
        return os.path.join(self.params.output_directory, staging_name)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        params = self.params
        if not os.path.isdir(params.source_directory):
            self.error_signal.emit(
                f"Source directory does not exist: {params.source_directory}"
            )
            return

        device = self.resolve_device()
        if device is None or device.type != "cuda":
            self.error_signal.emit(
                "No CUDA device detected. Image segmentation requires a GPU."
            )
            return

        model_name = params.model_name or resolve_most_recent_checkpoint(
            params.checkpoint_root
        )
        if not model_name:
            self.error_signal.emit(
                f"No segmentation checkpoint under {params.checkpoint_root}"
            )
            return

        staging = self._staging_directory()
        os.makedirs(staging, exist_ok=True)

        # ----- Stage 1: extract frames from videos (if any) ------------
        # Honour the explicit radio-button choice from the page when
        # ``source_type`` is set, so a folder that happens to contain
        # both videos and images is treated according to the user's
        # intent. ``auto`` preserves the historical extension-scan.
        stype = (params.source_type or "auto").lower()
        if stype == "videos":
            is_video = True
            is_image = False
        elif stype == "images":
            is_video = False
            is_image = True
        else:
            is_video = _is_video_directory(params.source_directory)
            is_image = _is_image_directory(params.source_directory)
        if is_video:
            self.log(
                f"Image segmentation stage 1: extracting frames from videos in "
                f"{params.source_directory}"
            )
            process_all_videos_in_directory(
                input_directory=params.source_directory,
                output_directory=staging,
                start_second=params.start_second,
                end_second=params.end_second,
                step_seconds=params.step_seconds,
                log=self.log,
                progress_cb=None,
                per_frame_cb=self._per_frame_cb,
                is_interrupted=self.is_interrupted,
            )
        elif is_image:
            # Mirror the image directory directly into staging so the
            # downstream segmentation function can operate on the same
            # shape it always does.
            self.log(
                f"Image segmentation stage 1: using images from "
                f"{params.source_directory}"
            )
            mirror_name = os.path.basename(
                params.source_directory.rstrip("/\\")
            )
            mirror_root = os.path.join(staging, mirror_name)
            mirror_images = os.path.join(mirror_root, "images")
            os.makedirs(mirror_images, exist_ok=True)
            # Symlink when available, copy otherwise. Symlinks avoid a
            # duplicate write on large datasets.
            for fn in os.listdir(params.source_directory):
                if not fn.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                src = os.path.join(params.source_directory, fn)
                dst = os.path.join(mirror_images, fn)
                if os.path.exists(dst):
                    continue
                try:
                    os.symlink(src, dst)
                except OSError:
                    # Windows non-admin may not allow symlinks.
                    import shutil

                    shutil.copy2(src, dst)
        else:
            self.error_signal.emit(
                f"No videos or images found under {params.source_directory}"
            )
            return

        # ----- Stage 2: segment every extracted frame ------------------
        folders = sorted(
            name for name in os.listdir(staging)
            if os.path.isdir(os.path.join(staging, name, "images"))
        )
        if not folders:
            self.error_signal.emit(
                f"Stage 1 wrote no frames under {staging}"
            )
            return

        cache = ModelCache()
        cache_key = f"seg::{params.checkpoint_root}::{model_name}"
        if cache.has(cache_key):
            self.log("Using cached segmentation model")
            model, processor = cache.get(cache_key)
        else:
            self.log(
                f"Image segmentation loading segmentation model '{model_name}'"
            )
            model, processor = load_model_for_segmentation(
                model_name, device=device
            )
            cache.set(cache_key, (model, processor))

        # Ensure the model is on the requested device (user preference:
        # verify code correctly works with the device). Use the shared
        # equivalence helper so cuda vs cuda:0 are treated as the same
        # physical device on single-GPU workstations.
        try:
            param_device = next(model.parameters()).device
            if not BaseWorker.devices_equivalent(param_device, device):
                self.log(
                    f"WARN: model on {param_device}, moving to {device}"
                )
                model.to(device)
        except Exception:  # noqa: BLE001
            pass

        self.log(
            f"Image segmentation stage 2: running Mask2Former on {len(folders)} folder(s)"
        )
        try:
            total = forecast_mask_with_a_mask2former(
                directory_all_images=staging,
                model=model,
                processor=processor,
                device=device,
                model_name=model_name,
                folders_stereotypie=folders,
                batch_size=params.batch_size,
                two_pass=params.two_pass,
                two_pass_merge=params.two_pass_merge,
                keep_largest_component=params.keep_largest_component,
                overwrite=params.overwrite,
                log=self.log,
                progress_cb=self._progress_cb,
                is_interrupted=self.is_interrupted,
            )
        except Exception as exc:  # noqa: BLE001
            try:
                import torch

                if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
                    self.error_signal.emit("CUDA OOM: reduce batch size")
                    return
            except Exception:  # noqa: BLE001
                pass
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")
            return

        self.progress(100)
        self.log(f"Image segmentation finished: {total} frame(s) segmented under {staging}")
        self.result_signal.emit(
            {
                "staging_directory": staging,
                "model_name": model_name,
                "total_processed": total,
            }
        )
