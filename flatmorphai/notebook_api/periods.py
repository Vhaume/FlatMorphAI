"""
Notebook 6 port: merge per-video CSVs into period-level CSVs and
compute aggregated statistics.

Public entry points:
- :func:`fuse_periods_by_subject` (preferred, v1.2.1): concatenates
  every per-video ``results_Stereotype_[...].csv`` under the
  processing root into a single ``all_videos.csv`` written to the
  same root. Subjects are identified by the first five underscore
  tokens of the video directory name
  (``species_compound_concentration_plate_position``) and each
  subject receives a fresh ``planaria_index`` value so rows from the
  same animal across different periods are linked.
- :func:`fuse_videos_into_periods` (legacy): the previous list-of-
  lists API kept for callers that have not migrated yet.
- :func:`compute_grouped_statistics`: counts seconds and segments per
  class for every (species, plate, concentration, period, arena) tuple
  into ``grouped_{molecule}.csv``.

All writers use ``sep=';'`` and ``encoding='latin1'`` as verified
from notebook 6 cells 4 and 9.

Step 6 simplification (v1.2.1)
------------------------------
Since the v1.2 naming convention (see
:mod:`flatmorphai.notebook_api.naming`) the per-video directory name
encodes ``<species>_<compound>_<concentration>_<plate>_<position>_<period>``.
The new fusion routine derives the subject identifier from the first
five tokens of that directory name and joins all periods of the same
subject into one CSV. This follows the "derive, don't duplicate" rule
of

  - Sandve, G. K. et al., "Ten Simple Rules for Reproducible
    Computational Research", PLOS Computational Biology, 2013,
    DOI: 10.1371/journal.pcbi.1003285, rule 1 (track how every
    result was produced) and rule 6 (avoid manual data manipulation
    steps).

Reference
---------
  - Pandas Development Team (2020), "pandas-dev/pandas: Pandas"
    DOI 10.5281/zenodo.3509134 - CSV I/O and group-by semantics.
"""

from __future__ import annotations

import glob
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .naming import parse_directory_name, parse_video_name


def get_most_recent_csv(directory: str) -> Optional[str]:
    """Return the basename of the most recently modified CSV in ``directory``."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        return None
    most_recent_file = max(csv_files, key=os.path.getmtime)
    return os.path.basename(most_recent_file)


def divide_seconds_into_periods(total_seconds: int, periods: int) -> List[List[List[int]]]:
    """Distribute ``total_seconds`` evenly among ``periods`` 1-based ranges."""
    seconds_per_period = total_seconds // periods
    remaining_seconds = total_seconds % periods
    seconds_by_periods: List[List[List[int]]] = []
    start = 1
    for _ in range(periods):
        end = start + seconds_per_period - 1
        if remaining_seconds > 0:
            end += 1
            remaining_seconds -= 1
        seconds_by_periods.append([[start, end]])
        start = end + 1
    return seconds_by_periods


def _derive_video_metadata(video_name: str):
    """Return ``(compound, concentration, plate, position)`` for ``video_name``.

    Tries the v1.1 directory layout first, then falls back to the video
    stem alone, and returns empty strings when neither convention parses.
    """
    parsed_dir = parse_directory_name(video_name)
    meta = None
    if parsed_dir is not None:
        meta = parsed_dir[0]
    else:
        meta = parse_video_name(video_name)
    if meta is None:
        return "", "", "", ""
    return meta.compound, meta.concentration, meta.plate, meta.position


def _subject_key_from_dirname(dirname: str) -> Optional[str]:
    """Return the five-token subject prefix of ``dirname`` or ``None``.

    A subject is the first five underscore-separated tokens of the
    per-video directory name, that is
    ``<species>_<compound>_<concentration>_<plate>_<position>``. The
    sixth token is the period (``T0h``, ``T24h`` ...) and is the only
    field that varies across acquisitions of the same physical
    planarian. Returns ``None`` when the parse fails so callers can
    skip non-conforming directories.
    """
    parsed = parse_directory_name(dirname)
    if parsed is None:
        return None
    meta = parsed[0]
    if not (meta.species and meta.compound and meta.concentration
            and meta.plate and meta.position):
        return None
    return "_".join((
        meta.species, meta.compound, meta.concentration,
        meta.plate, meta.position,
    ))


def fuse_periods_by_subject(
    processing_root: str,
    output_directory: Optional[str] = None,
    output_csv_name: str = "all_videos.csv",
    new_planaria_start_index: int = 1,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    csv_filename_filter: str = "results_",
) -> Optional[pd.DataFrame]:
    """Concatenate per-video CSVs into one CSV grouped by subject.

    The function walks ``processing_root`` for per-video directories
    that match the v1.2 naming convention
    ``<species>_<compound>_<concentration>_<plate>_<position>_<period>_total_<T>s_step_<S>s``,
    reads the most recent ``results_*.csv`` in each, assigns a fresh
    ``planaria_index`` per subject (first five tokens of the directory
    name), and writes a single concatenated CSV next to the
    per-video folders.

    Parameters
    ----------
    processing_root:
        Directory holding the per-video subdirectories produced by
        Step 1 / Step 2 / Step 4. The default location in the
        FlatMorphAI app is ``Processed_videos_Frame_by_frame``.
    output_directory:
        Where the merged CSV is written. Defaults to
        ``processing_root`` so the merged file sits alongside the
        per-video folders, matching the layout the user requested.
    output_csv_name:
        Filename of the merged CSV. Defaults to ``all_videos.csv``.
    new_planaria_start_index:
        First integer assigned to the first subject encountered. The
        next subject gets ``new_planaria_start_index + 1`` and so on.
        Subjects are sorted alphabetically by their five-token prefix
        so the assignment is deterministic across runs.
    csv_filename_filter:
        Substring that must appear in the per-video CSV filename for
        it to be picked up. Defaults to ``"results_"`` to match the
        Step 4 output naming pattern. Pass an empty string to accept
        every CSV in the per-video directory.

    Returns
    -------
    pandas.DataFrame or None
        The concatenated dataframe, or ``None`` when no eligible
        directory was found. The dataframe is also written to
        ``{output_directory}/{output_csv_name}`` with ``sep=';'``.

    References
    ----------
    Sandve, G. K. et al., "Ten Simple Rules for Reproducible
    Computational Research", PLOS Comput. Biol., 2013,
    DOI: 10.1371/journal.pcbi.1003285, rule 1 (one identifier per
    entity).
    """
    _log = log if log is not None else (lambda _msg: None)
    _progress = progress_cb if progress_cb is not None else (lambda *_: None)

    if output_directory is None:
        output_directory = processing_root
    os.makedirs(output_directory, exist_ok=True)

    if not os.path.isdir(processing_root):
        _log(f"Processing root does not exist: {processing_root}")
        return None

    # Enumerate every immediate subdirectory and bucket them by subject
    # key. ``subjects`` maps the five-token prefix to a list of
    # directory names (one per period acquisition for that subject).
    subjects: Dict[str, List[str]] = {}
    skipped: List[str] = []
    for entry in sorted(os.listdir(processing_root)):
        full = os.path.join(processing_root, entry)
        if not os.path.isdir(full):
            continue
        key = _subject_key_from_dirname(entry)
        if key is None:
            skipped.append(entry)
            continue
        subjects.setdefault(key, []).append(entry)

    if skipped:
        _log(
            f"Skipped {len(skipped)} non-conforming directory(ies): "
            f"{', '.join(skipped[:5])}"
            + (" ..." if len(skipped) > 5 else "")
        )

    if not subjects:
        _log("No subject directories found under " + processing_root)
        return None

    _log(
        f"Found {sum(len(v) for v in subjects.values())} period folder(s) "
        f"across {len(subjects)} subject(s)."
    )

    # ``planaria_index`` is reassigned per subject so rows belonging to
    # the same physical planarian (same five-token prefix) share one
    # value across all of its period acquisitions.
    sorted_subject_keys = sorted(subjects.keys())
    subject_to_index: Dict[str, int] = {
        key: new_planaria_start_index + offset
        for offset, key in enumerate(sorted_subject_keys)
    }

    df_results: Optional[pd.DataFrame] = None
    total_dirs = sum(len(v) for v in subjects.values())
    processed = 0

    for subject_key in sorted_subject_keys:
        new_index = subject_to_index[subject_key]
        for dirname in sorted(subjects[subject_key]):
            full = os.path.join(processing_root, dirname)
            csv_filename = get_most_recent_csv(full)
            if csv_filename is None or (
                csv_filename_filter and csv_filename_filter not in csv_filename
            ):
                _log(f"  WARN: no matching CSV in {dirname}, skipping")
                processed += 1
                _progress(processed, total_dirs)
                continue
            csv_path = os.path.join(full, csv_filename)
            try:
                df_current = pd.read_csv(csv_path, sep=";", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                _log(f"  ERROR reading {csv_path}: {exc}")
                processed += 1
                _progress(processed, total_dirs)
                continue

            parsed = parse_directory_name(dirname)
            if parsed is not None:
                meta, _total, _step = parsed
                # Backfill the period column from the directory name
                # for any row that arrived without one (older Step 4
                # outputs predate the in-row period column).
                if "period" not in df_current.columns:
                    df_current["period"] = meta.period
                else:
                    df_current.loc[
                        df_current["period"].isna()
                        | (df_current["period"].astype(str) == ""),
                        "period",
                    ] = meta.period
                # Provide species and per-row metadata so the merged
                # CSV is self-describing without referring back to the
                # directory tree.
                df_current["planaria_species"] = meta.species
                df_current["compound"] = meta.compound
                df_current["concentration"] = meta.concentration
                df_current["plate"] = meta.plate
                df_current["position"] = meta.position
                df_current["video_name"] = meta.video_stem
            else:
                df_current["video_name"] = dirname

            # Preserve the source planaria_index (typically ``1`` from
            # Step 4 since v1.2 keeps one planarian per video) before
            # rewriting the column with the subject-level identifier.
            if "planaria_index" in df_current.columns:
                df_current = df_current.rename(
                    {"planaria_index": "original_planaria_index"}, axis=1,
                )
            df_current["planaria_index"] = new_index

            df_results = (
                df_current if df_results is None
                else pd.concat([df_results, df_current], ignore_index=True)
            )
            processed += 1
            _progress(processed, total_dirs)

    if df_results is None or df_results.empty:
        _log("No rows produced - nothing written")
        return None

    # Cast the planaria index columns to nullable Int64 so empty or
    # missing values stay representable in the CSV without forcing a
    # float coercion.
    for col in (c for c in df_results.columns if "planaria_index" in c):
        df_results[col] = pd.to_numeric(df_results[col], errors="coerce").astype("Int64")

    # Order the leading columns so a human reading the CSV sees the
    # subject identity, the period and the second value first.
    front = [c for c in (
        "video_name", "image_name", "planaria_species", "planaria_index",
        "compound", "concentration", "plate", "position",
        "period", "second_value", "total_seconds",
    ) if c in df_results.columns]
    rest = [c for c in df_results.columns if c not in front]
    df_results = df_results[front + rest]

    output_csv_path = os.path.join(output_directory, output_csv_name)
    df_results.to_csv(
        output_csv_path, sep=";", index=False, encoding="latin1",
    )
    _log(
        f"Wrote {output_csv_path} "
        f"({len(df_results)} rows, {len(subjects)} subject(s))"
    )
    return df_results


def fuse_videos_into_periods(
    periods_name: Sequence[str],
    seconds_by_periods: Sequence[Sequence[Sequence[int]]],
    videos_to_fuse: Sequence[Sequence[str]],
    planaria_indexes_to_process: Sequence[Sequence[int]],
    directory_processed_video: str,
    output_directory: str,
    output_csv_name: str,
    new_planaria_index: int,
    planaria_species: str,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Optional[pd.DataFrame]:
    """Merge per-video CSVs into one period-level CSV.

    The tested compound, the concentration, the plate number and the
    well/arena position are derived per-video from the v1.1 directory
    name (see :mod:`flatmorphai.notebook_api.naming`). The only
    experimental metadata the caller still provides is the planarian
    species, which is not encoded in the filename, plus the
    ``new_planaria_index`` renumbering anchor.

    Returns the merged DataFrame and writes it to
    ``{output_directory}/{output_csv_name}`` with ``sep=';'`` and
    ``encoding='latin1'``.
    """
    os.makedirs(output_directory, exist_ok=True)
    if len(seconds_by_periods) != len(videos_to_fuse):
        raise ValueError(
            "Number of periods does not match number of video groups"
        )
    if len(seconds_by_periods) != len(periods_name):
        raise ValueError(
            "Number of periods does not match number of period names"
        )

    df_results: Optional[pd.DataFrame] = None
    total_periods = len(periods_name)

    for i, period_label in enumerate(periods_name):
        if log is not None:
            log(f"Processing period: {period_label}")

        total_seconds_in_period = 0
        for j in range(len(seconds_by_periods[i])):
            total_seconds_in_period += int(
                seconds_by_periods[i][j][1] - seconds_by_periods[i][j][0]
            ) + 1

        start_second = 0
        for j, video_name in enumerate(videos_to_fuse[i]):
            video_path = os.path.join(directory_processed_video, video_name)
            csv_name = get_most_recent_csv(video_path)
            if csv_name is None:
                if log is not None:
                    log(f"  WARN: no CSV in {video_path}, skipping")
                continue
            csv_path = os.path.join(video_path, csv_name)
            try:
                df_current = pd.read_csv(csv_path, sep=";", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                if log is not None:
                    log(f"  ERROR reading {csv_path}: {exc}")
                continue

            selected_index = planaria_indexes_to_process[i][j]
            df_current = df_current[df_current["planaria_index"] == selected_index].copy()
            df_current["original_planaria_index"] = df_current["planaria_index"]
            df_current["planaria_index"] = new_planaria_index

            df_current = df_current[
                (df_current["second_value"] >= seconds_by_periods[i][j][0])
                & (df_current["second_value"] <= seconds_by_periods[i][j][1])
            ].copy()
            df_current["original_second_value"] = df_current["second_value"]
            df_current["second_value"] = (
                df_current["second_value"] - seconds_by_periods[i][j][0] + start_second + 1
            )

            number_of_seconds_to_add = int(
                seconds_by_periods[i][j][1] - seconds_by_periods[i][j][0]
            ) + 1

            # Per-row metadata derived from the video directory name so
            # the fused CSV is self-describing even when different videos
            # in the same period come from different plates or wells.
            compound, concentration, plate, position = _derive_video_metadata(video_name)
            df_current["Acquisition time"] = period_label
            df_current["video_name"] = video_name
            df_current["total_seconds_in_period"] = total_seconds_in_period
            df_current["compound"] = compound
            df_current["Concentration"] = concentration
            df_current["plate_number"] = plate
            df_current["Arena"] = position
            df_current = df_current.rename(
                {"total_seconds": "original_total_seconds"}, axis=1,
            )

            df_results = df_current if df_results is None else pd.concat([df_results, df_current])
            start_second += number_of_seconds_to_add

        if progress_cb is not None:
            progress_cb(i + 1, total_periods)

    if df_results is None or df_results.empty:
        if log is not None:
            log("No rows produced - nothing written")
        return None

    # Integer-cast any index-bearing columns but skip the plate_number,
    # which is stored as a "P<digit>" token under v1.1 naming.
    for col in (c for c in df_results.columns if "index" in c):
        df_results[col] = pd.to_numeric(df_results[col], errors="coerce").astype("Int64")

    df_results["planaria_species"] = planaria_species

    columns_forecasted = [c for c in df_results.columns if "forecasted" in c]
    columns_corrected = [c for c in df_results.columns if "corrected" in c]
    columns_to_move = [
        "video_name", "image_name", "planaria_species", "compound",
        "Concentration", "plate_number", "planaria_index", "Arena",
        "second_value", "Acquisition time", "total_seconds_in_period",
        *columns_forecasted,
        *columns_corrected,
    ]
    new_order = columns_to_move + [c for c in df_results.columns if c not in columns_to_move]
    df_results = df_results[new_order]

    output_csv_path = os.path.join(output_directory, output_csv_name)
    df_results.to_csv(output_csv_path, sep=";", index=False, encoding="latin1")
    if log is not None:
        log(f"Saved {output_csv_path}")
    return df_results


def compute_grouped_statistics(
    periods_csv_path: str,
    class_names: Sequence[str],
    class_type: str = "Stereotype",
    output_directory: Optional[str] = None,
    output_csv_name: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[pd.DataFrame]:
    """Aggregate per-class time statistics from the periods CSV.

    Produces rows with columns:
    ``planaria_species, Plate (number), Concentration, Acquisition time,
    Arena, {class_type} (index), {class_type},
    {class_type} (sec), {class_type} (count)``.

    Both the v1.2.1 schema (``period``, ``concentration``, ``plate``,
    ``position``) and the legacy schema (``Acquisition time``,
    ``Concentration``, ``plate_number``, ``Arena``) are accepted. The
    column lookup is performed once and the loop body uses the
    resolved names so the function tolerates either layout without
    duplicating logic.
    """
    if not os.path.isfile(periods_csv_path):
        raise FileNotFoundError(periods_csv_path)
    df = pd.read_csv(periods_csv_path, sep=";", encoding="latin1")

    # Resolve column aliases between v1.2.1 (new fusion output) and the
    # legacy schema produced by the old period-list fusion.
    def _pick(*candidates: str) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    period_col = _pick("Acquisition time", "period")
    plate_col = _pick("plate_number", "plate", "Plate (number)")
    concentration_col = _pick("Concentration", "concentration")
    arena_col = _pick("Arena", "position")
    species_col = _pick("planaria_species")

    missing = [
        name for name, col in (
            ("period", period_col),
            ("plate", plate_col),
            ("concentration", concentration_col),
            ("arena/position", arena_col),
            ("planaria_species", species_col),
        ) if col is None
    ]
    if missing:
        raise KeyError(
            "compute_grouped_statistics: required column(s) missing in "
            f"{periods_csv_path}: {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    periods = df[period_col].unique()
    arenas = df[arena_col].unique()
    plates = df[plate_col].unique()
    species = df[species_col].unique()

    df_columns = [
        "planaria_species", "Plate (number)", "Concentration", "Acquisition time",
        "Arena", f"{class_type} (index)", class_type,
        f"{class_type} (sec)", f"{class_type} (count)",
    ]
    df_results: Optional[pd.DataFrame] = None

    corrected_col = f"corrected_{class_type}_index"
    forecasted_col = f"forecasted_{class_type}_index"

    for sp in species:
        for plate in plates:
            for arena in arenas:
                df_arena = df[
                    (df[arena_col] == arena)
                    & (df[plate_col] == plate)
                    & (df[species_col] == sp)
                ]
                for period in periods:
                    concentrations = df_arena[concentration_col].unique()
                    for conc in concentrations:
                        df_current = df_arena[
                            (df_arena[period_col] == period)
                            & (df_arena[concentration_col] == conc)
                        ].copy()
                        if df_current.empty:
                            continue

                        if corrected_col in df_current.columns:
                            df_current["final_class_value"] = df_current[corrected_col]
                            df_current["final_class_value"] = (
                                df_current["final_class_value"].replace("", None)
                            )
                            if forecasted_col in df_current.columns:
                                df_current["final_class_value"] = (
                                    df_current["final_class_value"]
                                    .fillna(df_current[forecasted_col])
                                )
                        elif forecasted_col in df_current.columns:
                            df_current["final_class_value"] = df_current[forecasted_col]
                        else:
                            continue

                        df_current = df_current.sort_values(by=["second_value"]).reset_index()
                        number_segment_class = np.zeros(len(class_names))
                        number_element_class = np.zeros(len(class_names))
                        previous_class_value = -1

                        for k in range(len(df_current)):
                            current_class_value = df_current.loc[k, "final_class_value"]
                            if pd.notna(current_class_value):
                                idx = int(current_class_value)
                                if 0 <= idx < len(class_names):
                                    number_element_class[idx] += 1
                            if current_class_value != previous_class_value:
                                if pd.notna(current_class_value):
                                    idx = int(current_class_value)
                                    if 0 <= idx < len(class_names):
                                        number_segment_class[idx] += 1
                                previous_class_value = current_class_value

                        for class_index in range(len(class_names)):
                            row = {
                                "planaria_species": sp,
                                "Plate (number)": plate,
                                "Concentration": conc,
                                "Acquisition time": period,
                                "Arena": arena,
                                f"{class_type} (index)": class_index,
                                class_type: class_names[class_index],
                                f"{class_type} (sec)": int(number_element_class[class_index]),
                                f"{class_type} (count)": int(number_segment_class[class_index]),
                            }
                            row_df = pd.DataFrame([row], columns=df_columns)
                            df_results = row_df if df_results is None else pd.concat(
                                [df_results, row_df], ignore_index=True,
                            )

    if df_results is None:
        return None

    for col in (f"{class_type} (sec)", f"{class_type} (count)"):
        df_results[col] = pd.to_numeric(df_results[col], errors="coerce").astype("Int64")

    if output_directory and output_csv_name:
        os.makedirs(output_directory, exist_ok=True)
        output_csv_path = os.path.join(output_directory, output_csv_name)
        df_results.to_csv(output_csv_path, sep=";", index=False, encoding="latin1")
        if log is not None:
            log(f"Saved {output_csv_path}")
    return df_results
