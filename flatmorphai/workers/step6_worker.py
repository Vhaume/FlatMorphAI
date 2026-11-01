"""
Worker for Step 6: merge per-video stereotype CSVs into one
subject-level CSV.

The v1.2 naming convention (see :mod:`flatmorphai.notebook_api.naming`)
encodes the species, compound, concentration, plate identifier, well
or arena position, and acquisition period in every per-video
directory name. A "subject" is one physical planarian, identified by
the first five underscore-separated tokens of the directory name
(``species_compound_concentration_plate_position``); all of its period
acquisitions share that prefix.

The worker therefore walks the processing root, groups per-video
directories by the five-token subject prefix, reads the most recent
``results_*.csv`` in each, assigns a fresh ``planaria_index`` per
subject and writes a single ``all_videos.csv`` next to the per-video
folders. References:

  - Sandve, G. K. et al., "Ten Simple Rules for Reproducible
    Computational Research", PLOS Comput. Biol., 2013,
    DOI: 10.1371/journal.pcbi.1003285, rule 1 (one identifier per
    entity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base_worker import BaseWorker
from ..constants import CLASSES_NAMES_STEREOTYPE
from ..notebook_api.periods import (
    compute_grouped_statistics,
    fuse_periods_by_subject,
)


@dataclass
class Step6Params:
    """Inputs for :class:`Step6Worker`.

    Attributes
    ----------
    processing_root:
        Directory holding the per-video subdirectories produced by
        Step 1 / Step 2 / Step 4 (typically
        ``Processed_videos_Frame_by_frame``).
    output_directory:
        Where the merged CSV is written. Defaults to
        ``processing_root`` so the file sits alongside the per-video
        folders.
    new_planaria_index:
        First integer assigned to the first subject encountered. The
        next subject gets ``new_planaria_index + 1`` and so on.
    output_csv_name:
        Filename of the merged CSV. Defaults to ``all_videos.csv``.
    grouped_csv_name:
        Filename of the per-class summary CSV. Defaults to
        ``grouped_periods.csv``.
    """

    processing_root: str
    output_directory: str
    new_planaria_index: int = 1
    output_csv_name: str = "all_videos.csv"
    grouped_csv_name: str = "grouped_periods.csv"


class Step6Worker(BaseWorker):
    """Produce ``all_videos.csv`` and ``grouped_periods.csv``."""

    def __init__(self, params: Step6Params) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        params = self.params
        out_dir = params.output_directory or params.processing_root
        self.log(
            "Step 6 starting: scanning "
            f"{params.processing_root} for per-video CSVs"
        )

        try:
            df = fuse_periods_by_subject(
                processing_root=params.processing_root,
                output_directory=out_dir,
                output_csv_name=params.output_csv_name,
                new_planaria_start_index=params.new_planaria_index,
                log=self.log,
                progress_cb=lambda done, total: self.progress(
                    int((done / total) * 50) if total else 0
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")
            return

        if df is None:
            self.error_signal.emit(
                "No per-video CSVs found under processing root"
            )
            return

        self.log("Computing grouped statistics...")
        try:
            import os
            periods_csv_path = os.path.join(out_dir, params.output_csv_name)
            grouped_df: Optional[object] = compute_grouped_statistics(
                periods_csv_path=periods_csv_path,
                class_names=CLASSES_NAMES_STEREOTYPE,
                class_type="Stereotype",
                output_directory=out_dir,
                output_csv_name=params.grouped_csv_name,
                log=self.log,
            )
        except Exception as exc:  # noqa: BLE001
            # The grouped summary is optional; do not fail Step 6 if
            # the column layout differs from the legacy expectation.
            self.log(
                f"Grouped statistics skipped ({type(exc).__name__}: {exc})"
            )
            grouped_df = None

        self.progress(100)
        self.log("Step 6 finished")
        self.result_signal.emit(
            {
                "periods_csv": params.output_csv_name,
                "grouped_csv": (
                    params.grouped_csv_name if grouped_df is not None
                    else ""
                ),
                "rows": int(len(df)),
            }
        )
