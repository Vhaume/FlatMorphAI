"""
Post-process helper for the Step 4 classification CSV.

After :func:`process_videos` in notebook 4 has written
``results_Stereotype_[model].csv`` inside each per-video directory
the dashboard augments every row with the experimental parameters
encoded in the source filename, following the convention documented
in :mod:`flatmorphai.notebook_api.naming`.

The rationale is given in

  - Wilkinson, M. D. et al., "The FAIR Guiding Principles for
    scientific data management and stewardship", Scientific Data,
    2016, DOI: 10.1038/sdata.2016.18, which asks for rich metadata
    to travel with every analytical output, not just the raw data.
  - Sandve, G. K. et al., "Ten Simple Rules for Reproducible
    Computational Research", PLOS Computational Biology, 2013,
    DOI: 10.1371/journal.pcbi.1003285, rule 4 (capture provenance)
    and rule 5 (archive intermediate outputs).

CSV separator
-------------
Writer uses the FlatMorphAI project convention ``sep=";"``, as
requested in the project style guide. Loading tolerates both ``;``
and ``,`` so downstream helpers keep working on legacy files.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from .naming import ImageMeta, parse_directory_name, parse_image_name


METADATA_COLUMNS = (
    "compound",
    "concentration",
    "plate",
    "position",
    "period",
    "second",
    "total_seconds",
    "step_seconds",
)


def _infer_image_column(df: pd.DataFrame) -> Optional[str]:
    """Return the most likely column that holds the image filename."""
    for candidate in ("image", "image_name", "filename", "file", "path"):
        if candidate in df.columns:
            return candidate
    # Fallback: first object-typed column whose first value looks like a file.
    for col in df.columns:
        if df[col].dtype == object and df[col].astype(str).str.contains(".png|.jpg|.jpeg", case=False).any():
            return col
    return None


def augment_forecast_csv(
    csv_path: str,
    parent_directory_name: Optional[str] = None,
    separator: str = ";",
) -> int:
    """Add the FlatMorphAI naming-convention columns to a forecast CSV.

    Parameters
    ----------
    csv_path:
        Absolute path to the ``results_Stereotype_[model].csv`` produced
        by notebook 4 for one video.
    parent_directory_name:
        Used as a fallback when the row stores only the bare second
        index. When ``None`` it is inferred from ``csv_path``.
    separator:
        Output CSV separator. Defaults to ``";"`` per project style
        guide.

    Returns
    -------
    int
        Number of rows that could be enriched. Rows with an
        unparseable filename are kept verbatim.
    """
    if not os.path.isfile(csv_path):
        return 0
    # Tolerate both ";" and "," on input so legacy CSVs keep working.
    try:
        df = pd.read_csv(csv_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(csv_path, sep=",")
    except Exception:
        df = pd.read_csv(csv_path)

    if parent_directory_name is None:
        parent_directory_name = os.path.basename(os.path.dirname(csv_path.rstrip("/\\")))

    image_column = _infer_image_column(df)
    if image_column is None:
        # Nothing we can key off. Still rewrite with the expected
        # separator so downstream consumers see a consistent format.
        df.to_csv(csv_path, sep=separator, index=False)
        return 0

    parsed = []
    for value in df[image_column].astype(str):
        meta: Optional[ImageMeta] = parse_image_name(
            value, parent_directory_name=parent_directory_name
        )
        if meta is None:
            parsed.append({col: "" for col in METADATA_COLUMNS})
            continue
        parsed.append(
            {
                "compound": meta.video.compound,
                "concentration": meta.video.concentration,
                "plate": meta.video.plate,
                "position": meta.video.position,
                "period": meta.video.period,
                "second": meta.second,
                "total_seconds": meta.total_seconds,
                "step_seconds": meta.step_seconds,
            }
        )

    # Inject the new columns right after the image column so the user
    # reads "which image -> which parameters" left-to-right.
    meta_df = pd.DataFrame(parsed)
    existing = [c for c in df.columns if c not in METADATA_COLUMNS]
    image_idx = existing.index(image_column)
    left = existing[: image_idx + 1]
    right = existing[image_idx + 1 :]
    out = pd.concat([df[left].reset_index(drop=True), meta_df, df[right].reset_index(drop=True)], axis=1)
    out.to_csv(csv_path, sep=separator, index=False)
    return int((meta_df[list(METADATA_COLUMNS)] != "").all(axis=1).sum())


def augment_all_forecast_csvs(
    processing_root: str,
    filename_contains: str = "results_Stereotype_",
    separator: str = ";",
) -> List[str]:
    """Apply :func:`augment_forecast_csv` to every forecast CSV under a root.

    Returns the list of CSV paths that were rewritten.
    """
    written: List[str] = []
    if not os.path.isdir(processing_root):
        return written
    for video_dir in os.listdir(processing_root):
        video_path = os.path.join(processing_root, video_dir)
        if not os.path.isdir(video_path):
            continue
        parent = video_dir
        # Resolve the video stem (strip "_total_<T>s_step_<S>s" if present).
        parsed_dir = parse_directory_name(video_dir)
        for fn in os.listdir(video_path):
            if filename_contains in fn and fn.lower().endswith(".csv"):
                csv_path = os.path.join(video_path, fn)
                augment_forecast_csv(
                    csv_path=csv_path,
                    parent_directory_name=parent,
                    separator=separator,
                )
                written.append(csv_path)
    return written


__all__ = [
    "METADATA_COLUMNS",
    "augment_forecast_csv",
    "augment_all_forecast_csvs",
]
