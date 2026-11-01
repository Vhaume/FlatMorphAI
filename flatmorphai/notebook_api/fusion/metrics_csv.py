"""Per-video metrics CSV generation for FusionNetV2 inference.

The FusionNetV2 inference pathway (Export_fusion_v2 -> FlatMorphAI) needs
a metric vector per frame, computed from the (prev, cur, next) mask
window plus the raw current-frame grayscale.  This module produces the
``<video>_metrics.csv`` file the rest of the pipeline reads at Step 4.

Why a separate CSV (rather than computing the metrics on the fly per
batch in process_videos):

  1. The metric computation reads three masks plus the original grayscale
     per frame and is independent of the classifier.  Caching it once
     per video means that re-running Step 4 with a different fusion
     checkpoint (or after toggling a setting) does not redo the
     numpy / scikit-image work.
  2. The aggregated ``all_videos_metrics.csv`` produced by Step 6 is
     useful as a standalone dataset for downstream regression / latent-
     space analysis and as a sanity-check artifact for the reviewer.

Frame ordering and the "second" token.  The operator's requirement is
that the second is parsed from the FlatMorphAI v1.2 filename
convention (``<video_stem>_<second>s.png``) via the existing
``naming.parse_image_name`` helper, NOT from the legacy notebook 4
token positions.

Sub-second frames and the closest-to-target rule.  When the operator
samples at sub-second cadence (for example T0 = 0.5s, next = 1.5s),
the (prev, cur, next) window must still be ``one second wide``
because the FusionNetV2 was trained on that physical interval.  For
each cur with second ``s``:

   target_prev = s - 1.0
   target_next = s + 1.0

If a frame with second exactly equal to the target exists on disk it
is picked.  Otherwise the frame whose second is CLOSEST to the target
(excluding cur itself) is picked.  When NO other frame is available
the cur mask is reused, matching the boundary-frame convention used
during training (Export_fusion_v2 README, "Boundary handling").

Worked example for cur=0.5s on a {0.0, 0.5, 1.5, 2.5} sampling:
   target_prev = -0.5  -> existing seconds {0.0, 1.5, 2.5} -> 0.0 wins
   target_next =  1.5  -> 1.5 exists exactly -> 1.5 wins

Performance optimisations included here:

  - Per-video mask cache: each PNG is decoded at most once per video,
    not once per (prev, cur, next) appearance.  On a 1000-frame video
    with one-second sampling this drops the cv2.imdecode count from
    ~3000 to ~1000.
  - Partial-update mode: when the output CSV already exists, only
    frames whose ``image_name`` is missing from it are recomputed;
    existing rows are kept as-is.  This makes re-runs after adding a
    handful of frames instantaneous instead of N seconds.
  - Pre-resolved neighbours: the (prev, cur, next) selection is done
    ONCE up-front into three name lists, then the inner loop just
    looks up cached masks.

References
----------
- Wickham, H., "Tidy Data", Journal of Statistical Software 59(10),
  2014, DOI:10.18637/jss.v059.i10 (one row per observation).
- Sandve, G.K. et al., "Ten Simple Rules for Reproducible
  Computational Research", PLOS Comp Biol, 2013,
  DOI:10.1371/journal.pcbi.1003285 (rule 1: store intermediate
  products so the pipeline is restartable).
- Feldman, S.I., "Make: a program for maintaining computer programs",
  Software: Practice and Experience 9(4), 255-265, 1979,
  DOI:10.1002/spe.4380090402 (skip-existing on output presence).
- Wilson, G. et al., "Good Enough Practices in Scientific Computing",
  PLOS Comp Biol 13(6):e1005510, 2017,
  DOI:10.1371/journal.pcbi.1005510 (per-folder summaries).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .metrics_3mask import METRIC_COLS, compute_metric_vector


def _read_image_uint8(path: str, gray: bool = False) -> Optional[np.ndarray]:
    """Robust cv2 read that survives non-ASCII paths."""
    if not os.path.isfile(path):
        return None
    try:
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_UNCHANGED
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    except Exception:  # noqa: BLE001
        return None


def _parse_second(image_name: str, video_name: str) -> Optional[float]:
    """Return the (float) second token for an image filename, using the
    FlatMorphAI v1.2 naming convention.

    Returns ``None`` when the filename does not match the convention,
    in which case the caller treats the frame as un-orderable and
    skips it for the 3-mask metric window.

    The previous version of this helper rounded the second to an
    integer, which silently merged sub-second frames (T=0.5s and
    T=1.5s would both land at second=1 under round-half-to-even).
    Float keeps every frame addressable.
    """
    try:
        from ..naming import parse_image_name
    except Exception:  # noqa: BLE001
        return None
    meta = parse_image_name(image_name, parent_directory_name=video_name)
    if meta is None:
        return None
    try:
        return float(meta.second)
    except Exception:  # noqa: BLE001
        return None


def _resolve_neighbours(
    ordered: List[Tuple[float, str]],
) -> Dict[str, Tuple[str, str]]:
    """For every (second, image_name) in ``ordered`` return the
    ``image_name`` of the chosen prev and next neighbours.

    Selection rule (matches the training-time one-second window):

      1. Candidates for ``prev`` are restricted to frames STRICTLY
         BEFORE cur (i.e. second < cur.second).  Among those, the one
         whose second is closest to ``cur.second - 1.0`` wins.  When
         no earlier frame exists at all (first-frame boundary), prev
         falls back to cur_mask, matching Export_fusion_v2/README.md
         ("at the first frame pass prev_mask = cur_mask").
      2. Same rule for ``next`` but candidates are frames STRICTLY
         AFTER cur (second > cur.second), target is
         ``cur.second + 1.0``, and the boundary fallback is to cur.

    Worked examples:

      Operator's example, cur=0.5, sampling {0.0, 0.5, 1.5, 2.5}:
        prev candidates {0.0}: 0.0 wins (only earlier frame)
        next candidates {1.5, 2.5}: 1.5 wins (closest to target 1.5)

      First-frame boundary, cur=0.0, sampling {0.0, 1.0, 2.0}:
        prev candidates {}: fallback prev = cur
        next candidates {1.0, 2.0}: 1.0 wins

      Irregular sampling, cur=2.0, sampling {0.3, 2.0, 4.0}:
        prev candidates {0.3}: 0.3 wins (only earlier frame)
        next candidates {4.0}: 4.0 wins (only later frame)

    Pre-resolving the neighbours once up-front means the inner loop
    only needs three cache lookups, not three closest-search calls,
    which keeps the per-frame work down to the actual metric
    computation.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for sec, name in ordered:
        tp = sec - 1.0
        tn = sec + 1.0
        prev_best: Optional[str] = None
        prev_d = float("inf")
        next_best: Optional[str] = None
        next_d = float("inf")
        for s2, n2 in ordered:
            if n2 == name:
                continue
            if s2 < sec:
                d = abs(s2 - tp)
                if d < prev_d:
                    prev_d = d
                    prev_best = n2
            elif s2 > sec:
                d = abs(s2 - tn)
                if d < next_d:
                    next_d = d
                    next_best = n2
            # s2 == sec is ignored: a different frame at the same
            # second is not earlier OR later, so it cannot serve as
            # prev or next.  This is rare in practice (one PNG per
            # second under v1.2) but harmless when it occurs.
        if prev_best is None:
            prev_best = name  # boundary fallback to cur mask
        if next_best is None:
            next_best = name  # boundary fallback to cur mask
        out[name] = (prev_best, next_best)
    return out


class _MaskCache:
    """Per-video LRU-ish cache of grayscale mask reads.

    Each PNG inside ``mask_dir`` is decoded the first time it is asked
    for, then served from the dict on subsequent requests.  With a
    standard one-second sampling, every mask is read three times (as
    next, as cur, as prev across three consecutive frames), so caching
    cuts cv2.imdecode work to one third.  The cache is bounded by the
    number of distinct masks in the video, which is acceptable on
    typical inputs (1000 grayscale frames at full HD weigh about 2 GB
    if every mask is decoded; in practice the cache is bounded by the
    number of distinct neighbour resolutions, which is at most N+2).

    Reset between videos by constructing a new instance.
    """

    def __init__(self, mask_dir: str) -> None:
        self.dir = mask_dir
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def get(self, image_name: str) -> Optional[np.ndarray]:
        if image_name in self._cache:
            return self._cache[image_name]
        m = _read_image_uint8(
            os.path.join(self.dir, image_name), gray=True,
        )
        self._cache[image_name] = m
        return m

    def __len__(self) -> int:
        return len(self._cache)


def compute_video_metrics_csv(
    video_dir: str,
    mask_dir_name: str,
    output_csv_path: str,
    metric_cols: Optional[List[str]] = None,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Walk every frame in ``video_dir/images/``, compute the 3-mask
    metric vector for each, and write the result to ``output_csv_path``.

    Behaviour notes (see module docstring for the full rationale):

    - Seconds are parsed as floats and frames are ordered by second.
    - When the operator samples at non-uniform cadence, the prev / next
      neighbours are picked by closest existing second to (cur - 1.0)
      and (cur + 1.0) respectively, excluding cur itself.
    - When the output CSV already exists and ``overwrite`` is False,
      ONLY frames whose ``image_name`` is missing from it are
      recomputed; existing rows are kept verbatim.
    - The per-video mask cache reads each PNG at most once.

    Returns the row count in the final CSV (existing + newly written).
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
    cols_out = list(metric_cols) if metric_cols else list(METRIC_COLS)

    # Parse seconds (float) for every PNG.
    pngs = [
        f for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(".png")
    ]
    ordered: List[Tuple[float, str]] = []
    n_unparseable = 0
    for png in pngs:
        sec = _parse_second(png, video_name)
        if sec is None:
            n_unparseable += 1
            continue
        ordered.append((sec, png))
    ordered.sort(key=lambda t: (t[0], t[1]))
    if n_unparseable:
        _log(
            f"  WARNING: {n_unparseable} frame(s) could not be parsed "
            f"with the FlatMorphAI naming convention; those frames are "
            f"excluded from the 3-mask window."
        )
    if not ordered:
        _log(f"  WARNING: no parseable frames in {images_dir}; "
             "metrics CSV will be empty.")
        return 0

    # Partial-update mode: read existing rows so they are kept and only
    # missing image_name entries are recomputed.  When the CSV is
    # absent or unreadable, treat as a clean run.
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

    # Resolve prev / next neighbour names ONCE up front so the inner
    # loop only sees cache lookups.
    neighbour_of = _resolve_neighbours(ordered)

    mask_cache = _MaskCache(masks_dir)
    rows = []
    n_skipped_existing = 0
    n_skipped_missing_mask = 0
    for sec, image_name in ordered:
        if image_name in existing_rows:
            rows.append(existing_rows[image_name])
            n_skipped_existing += 1
            continue
        cur_mask = mask_cache.get(image_name)
        if cur_mask is None:
            n_skipped_missing_mask += 1
            continue
        prev_name, next_name = neighbour_of[image_name]
        prev_mask = mask_cache.get(prev_name)
        if prev_mask is None:
            prev_mask = cur_mask
        next_mask = mask_cache.get(next_name)
        if next_mask is None:
            next_mask = cur_mask
        cur_gray = _read_image_uint8(
            os.path.join(images_dir, image_name), gray=True,
        )
        feat = compute_metric_vector(
            prev_mask, cur_mask, next_mask, cur_gray,
        )
        row = {"image_name": image_name, "second": float(sec)}
        for col in cols_out:
            row[col] = float(feat.get(col, np.nan))
        rows.append(row)

    if n_skipped_existing and not overwrite:
        _log(
            f"  Reused {n_skipped_existing} existing row(s); only "
            f"computed {len(rows) - n_skipped_existing} new row(s)."
        )
    if n_skipped_missing_mask:
        _log(
            f"  WARNING: {n_skipped_missing_mask} frame(s) had no "
            f"current mask under '{mask_dir_name}/'; those rows are "
            f"not in the CSV."
        )
    if not rows:
        _log(f"  No metric rows produced for {video_name}; "
             "check Step 2 output.")
        return 0

    # Build the output DataFrame.  Column order: image_name, second,
    # then the metric columns in the caller-specified order (so the
    # checkpoint's expected layout is honoured) followed by any extra
    # metric columns that survived from a previous partial run.
    df = pd.DataFrame(rows)
    front = ["image_name", "second"]
    metric_front = [c for c in cols_out if c in df.columns]
    other_metric = [
        c for c in df.columns
        if c not in front and c not in metric_front
    ]
    df = df[front + metric_front + other_metric]

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df.to_csv(output_csv_path, sep=";", index=False, encoding="utf-8")
    _log(
        f"  Wrote metrics CSV ({len(df)} row(s), {len(df.columns)} "
        f"cols, cache hits {len(mask_cache)} mask(s)): {output_csv_path}"
    )
    return len(df)
