"""
Worker for Step 3: generate classification-ready per-planarian images
from paired frames and masks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..base_worker import BaseWorker
from ..notebook_api.classification_images import (
    build_processed_dir_name,
    build_temporal_dir_name,
    generate_stereotypie_image_for_classification,
    generate_temporal_images_for_classification,
)


@dataclass
class Step3Params:
    """Parameters for Step 3 (classification-ready image generation).

    The former ``number_rows_of_planaria_on_image`` parameter has been
    removed: with the v1.2 naming convention, every video targets a
    single well and therefore a single planarian. The arena position
    (``B4``, ``arena3``, ...) is encoded in the ``position`` field of
    the filename so no extra row count is required. Removing unused
    parameters follows rule 4 of Wilson, G. et al., "Good Enough
    Practices in Scientific Computing", PLOS Computational Biology,
    2017, DOI: 10.1371/journal.pcbi.1005510.
    """

    processing_root: str
    name_stereotypie_dir: str = "Formated"
    image_size: int = 224
    # The deployed prototype in this workspace was built at
    # margin=0.0 (user-confirmed empirically); the resolver still
    # matches on the directory's _margin(M) token so a re-trained
    # model with a different margin can be picked up by simply
    # rerunning Step 3 at the new value.
    margin_percent: float = 0.0
    black_and_white: bool = False
    apply_mask: bool = True
    number_of_planaria_on_image: int = 1
    mask_dir: Optional[str] = None
    folders_stereotypie_to_process: List[str] = field(default_factory=list)
    # When False (default), every frame whose formatted output file
    # already exists on disk is skipped.  This mirrors Step 2's
    # overwrite=False default and avoids rewriting hundreds of frames
    # on every re-run.  Reference: Feldman, S.I., "Make: a program for
    # maintaining computer programs", SPE 9(4), 255-265, 1979,
    # DOI:10.1002/spe.4380090402 (file-timestamp invalidation).
    overwrite: bool = False
    # When True, Step 3 writes the TEMPORAL 3-channel BGR image
    # (channel 0 = T-1 masked grayscale, channel 1 = T0, channel 2 =
    # T+1) instead of the standard single-frame formatted image.
    # Pipeline A auto-sets this to True when the selected classifier
    # is the fusion + TSM model (ConvNeXtV2_tsm_3f_v2_size384), which
    # consumes a temporal 9-channel input at inference.  See
    # classification_images.generate_temporal_images_for_classification.
    temporal_mode: bool = False
    # Worker threads for the formatting pass.  0 (default) means
    # "auto": cpu_count() - 1.  Surfaced as ``step3_max_workers``.
    max_workers: int = 0


class Step3Worker(BaseWorker):
    """Crop each planarian and resize to ``(image_size, image_size)``."""

    def __init__(self, params: Step3Params) -> None:
        super().__init__()
        self.params = params

    def _progress_cb(self, done: int, total: int, folder_name: str) -> None:
        if total > 0:
            self.progress(int(done / total * 100))

    def run(self) -> None:
        params = self.params
        if not os.path.isdir(params.processing_root):
            self.error_signal.emit(
                f"Processing root does not exist: {params.processing_root}"
            )
            return

        self.log(
            f"Step 3 starting: size={params.image_size}, "
            f"margin={params.margin_percent}, "
            f"bw={params.black_and_white}, apply_mask={params.apply_mask}, "
            f"overwrite={params.overwrite}, "
            f"temporal_mode={params.temporal_mode}, "
            f"max_workers={params.max_workers or 'auto'}"
        )
        mask_dirs = [params.mask_dir] if params.mask_dir else None
        try:
            if params.temporal_mode:
                total_written = generate_temporal_images_for_classification(
                    path_dataset=params.processing_root,
                    name_stereotypie_dir=params.name_stereotypie_dir,
                    name_mask_dir=mask_dirs,
                    image_size=params.image_size,
                    margin_percent=params.margin_percent,
                    apply_mask=params.apply_mask,
                    folders_stereotypie_to_process=(
                        params.folders_stereotypie_to_process or None
                    ),
                    overwrite=params.overwrite,
                    log=self.log,
                    progress_cb=self._progress_cb,
                    is_interrupted=self.is_interrupted,
                )
            else:
                total_written = generate_stereotypie_image_for_classification(
                    path_dataset=params.processing_root,
                    name_stereotypie_dir=params.name_stereotypie_dir,
                    name_mask_dir=mask_dirs,
                    number_of_planaria_on_image=params.number_of_planaria_on_image,
                    image_size=params.image_size,
                    margin_percent=params.margin_percent,
                    # The former number_rows_of_planaria_on_image has been
                    # dropped (see Step3Params docstring). A constant 1 is
                    # passed to the legacy helper so the grid split becomes a
                    # single row, matching the one-planarian-per-video layout
                    # of the v1.2 naming convention.
                    number_rows_of_planaria_on_image=1,
                    folders_stereotypie_to_process=(
                        params.folders_stereotypie_to_process or None
                    ),
                    apply_mask=params.apply_mask,
                    black_and_white=params.black_and_white,
                    overwrite=params.overwrite,
                    log=self.log,
                    progress_cb=self._progress_cb,
                    is_interrupted=self.is_interrupted,
                    max_workers=params.max_workers or None,
                )
        except Exception as exc:  # noqa: BLE001
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")
            return

        self.progress(100)
        if params.temporal_mode:
            processed_dir_name = build_temporal_dir_name(
                params.name_stereotypie_dir,
                params.image_size,
                params.margin_percent,
                params.apply_mask,
            )
        else:
            processed_dir_name = build_processed_dir_name(
                params.name_stereotypie_dir,
                params.image_size,
                params.margin_percent,
                params.black_and_white,
                params.apply_mask,
            )
        self.log(f"Step 3 finished: {total_written} image(s) written")
        self.result_signal.emit(
            {
                "processed_dir_name": processed_dir_name,
                "total_written": total_written,
            }
        )
