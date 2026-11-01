"""
Worker for Step 2: Mask2Former semantic segmentation inference.

All heavy work is delegated to :mod:`flatmorphai.notebook_api.segmentation`
which imports the project's existing inference modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..base_worker import BaseWorker
from ..constants import CHECKPOINTS_SEGMENTATION_DIR
from ..model_cache import ModelCache
from ..notebook_api.segmentation import (
    forecast_mask_with_a_mask2former,
    load_model_for_segmentation,
)


@dataclass
class Step2Params:
    checkpoint_root: str = CHECKPOINTS_SEGMENTATION_DIR
    model_name: str = ""
    processing_root: str = "Processed_videos_Frame_by_frame"
    batch_size: int = 8
    two_pass: bool = True
    two_pass_context_fill: bool = True
    two_pass_model_size: int = 320
    two_pass_merge: str = "skeleton_guided"
    keep_largest_component: bool = True
    overwrite: bool = False
    folders_stereotypie: List[str] = field(default_factory=list)
    # Mask refinement. When ``blasteme_refined`` is on, segmentation is routed
    # through the Pipeline C subprocess (which grows the full-body mask with the
    # blastema prediction); otherwise the fast in-process segmentation runs.
    blasteme_refined: bool = False


class Step2Worker(BaseWorker):
    """Generate binary masks for every frame in ``processing_root``."""

    def __init__(self, params: Step2Params) -> None:
        super().__init__()
        self.params = params

    def _resolve_model_name(self) -> Optional[str]:
        if self.params.model_name:
            return self.params.model_name
        root = self.params.checkpoint_root
        if not os.path.isdir(root):
            return None
        entries = [
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]
        if not entries:
            return None
        entries.sort(
            key=lambda n: os.path.getmtime(os.path.join(root, n)),
            reverse=True,
        )
        return entries[0]

    def _resolve_videos(self) -> List[str]:
        if self.params.folders_stereotypie:
            return list(self.params.folders_stereotypie)
        root = self.params.processing_root
        if not os.path.isdir(root):
            return []
        return sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name, "images"))
        )

    def _progress_cb(self, done: int, total: int, video_name: str) -> None:
        if total > 0:
            self.progress(int(done / total * 100))

    def run(self) -> None:
        device = self.resolve_device()
        if device is None or device.type != "cuda":
            self.error_signal.emit(
                "No CUDA device detected. Step 2 requires a GPU."
            )
            return

        model_name = self._resolve_model_name()
        if not model_name:
            self.error_signal.emit(
                f"No checkpoint found under {self.params.checkpoint_root}"
            )
            return

        folders = self._resolve_videos()
        if not folders:
            self.error_signal.emit(
                f"No video subfolders with images/ under {self.params.processing_root}"
            )
            return

        self.log(f"Step 2 starting with model '{model_name}' on {device}")

        cache_key = f"seg::{self.params.checkpoint_root}::{model_name}"
        cache = ModelCache()
        if cache.has(cache_key):
            self.log("Using cached segmentation model")
            model, processor = cache.get(cache_key)
        else:
            self.log("Loading segmentation model...")
            model, processor = load_model_for_segmentation(model_name, device=device)
            cache.set(cache_key, (model, processor))

        # Ensure model is on CUDA (spec section 1.3). Use the shared
        # equivalence helper so cuda vs cuda:0 do not trigger a redundant
        # move on every worker run.
        try:
            param_device = next(model.parameters()).device
            if not BaseWorker.devices_equivalent(param_device, device):
                self.log(
                    f"WARN: model on {param_device}, moving to {device}"
                )
                model.to(device)
        except Exception:
            pass

        try:
            total = forecast_mask_with_a_mask2former(
                directory_all_images=self.params.processing_root,
                model=model,
                processor=processor,
                device=device,
                model_name=model_name,
                folders_stereotypie=folders,
                batch_size=self.params.batch_size,
                two_pass=self.params.two_pass,
                two_pass_context_fill=self.params.two_pass_context_fill,
                two_pass_model_size=self.params.two_pass_model_size,
                two_pass_merge=self.params.two_pass_merge,
                keep_largest_component=self.params.keep_largest_component,
                overwrite=self.params.overwrite,
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
            except Exception:
                pass
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")
            return

        self.progress(100)
        self.log(f"Step 2 finished, {total} frame(s) processed")
        self.result_signal.emit(
            {
                "mask_directory_name": "masks",
                "model_name": model_name,
                "total_processed": total,
            }
        )
