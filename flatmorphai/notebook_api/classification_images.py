"""
Notebook 3 port: generate per-planarian classification images from
paired frame + mask PNGs.

This module reproduces the logic from
``3 - Generate images for classification.ipynb`` with progress and
interruption callbacks added so it can run inside a QThread worker.

Public functions
----------------
``resize_with_black_borders``
``compute_mask_image_planarian_reshaped``
``get_mask_folders``
``biggest_objects_in_the_binary_mask``
``generate_stereotypie_image_for_classification``
``build_processed_dir_name``

Reference
---------
  - Suzuki & Abe (1985), "Topological structural analysis of digitized
    binary images by border following", Comput. Vis. Graph. Image
    Process. 30(1), 32-46 - contour extraction used by cv2.findContours.
  - Zuiderweg & Zuiderweg (1994), "Contrast Limited Adaptive Histogram
    Equalization" (CLAHE), Graphics Gems IV, ISBN 978-0123361554.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


ProgressCallback = Callable[[int, int, str], None]
PreviewCallback = Callable[[np.ndarray, np.ndarray, np.ndarray, str], None]
InterruptCheck = Callable[[], bool]


def _noop_interrupt() -> bool:
    return False


# ---------------------------------------------------------------------------
# Top-level worker for ProcessPoolExecutor (must be picklable).
# ---------------------------------------------------------------------------
def _format_one_frame_worker(
    task: Tuple[str, str, str, List[str], str, int, float, bool, bool, int, int, bool],
) -> Tuple[str, int, int, List[str]]:
    """Read a frame + mask, run the formatter, write the output.

    Runs in a *separate process* so cv2 / numpy get true multi-core
    parallelism without contending for the GIL.  Because processes
    cannot share Python objects, the worker takes a flat picklable
    tuple and returns a similarly flat tuple.  Any log strings the
    worker wants to surface are returned in the last slot and are
    logged by the caller on the main thread.

    Reference for offloading CPU-bound cv2 work to a process pool
    instead of a thread pool: Beazley, D. M., "Understanding the
    Python GIL", PyCon 2010,
    https://www.dabeaz.com/python/UnderstandingGIL.pdf (section 4:
    GIL removal patterns via multiprocessing).

    ``task`` layout (must stay a tuple so the pickle payload is
    dense)::

        ( img_name,
          full_img_path,
          out_dir,
          mask_paths,          # ordered list of candidate mask files
          overwrite,
          apply_mask,
          black_and_white,
          image_size,
          margin_percent,
          number_of_planaria_on_image,
          number_rows_of_planaria_on_image,
          multi_planaria_skip_hint,
        )

    Returns ``(img_name, n_written, n_skipped, log_lines)``.
    """
    (
        img_name, full_img_path, out_dir, mask_paths,
        overwrite, apply_mask, black_and_white, image_size,
        margin_percent, number_of_planaria_on_image,
        number_rows_of_planaria_on_image, _multi_hint,
    ) = task

    log_lines: List[str] = []
    n_written = 0
    n_skipped = 0

    # Existing-output skip (single-planarian): probe BEFORE any
    # decode so a fully cached rerun stays near-instant.
    if (
        not overwrite
        and number_of_planaria_on_image == 1
        and os.path.isfile(os.path.join(out_dir, img_name))
    ):
        return img_name, 0, 1, log_lines

    mask_path = None
    for candidate in mask_paths:
        if os.path.isfile(candidate):
            mask_path = candidate
            break
    if mask_path is None:
        log_lines.append(f"  WARN: mask not found for {img_name}")
        return img_name, 0, 0, log_lines

    try:
        original_image = cv2.imdecode(
            np.fromfile(full_img_path, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        mask_image = cv2.imdecode(
            np.fromfile(mask_path, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(
            f"  WARN: decode failed for {img_name} "
            f"({type(exc).__name__}: {exc})"
        )
        return img_name, 0, 0, log_lines
    if mask_image is None or original_image is None:
        log_lines.append(f"  WARN: failed to read {img_name}")
        return img_name, 0, 0, log_lines

    if len(mask_image.shape) == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, mask_image = cv2.threshold(
        mask_image, 127, 255, cv2.THRESH_BINARY,
    )
    if mask_image.dtype != np.uint8:
        mask_image = mask_image.astype(np.uint8)
    if mask_image.shape != original_image.shape[:2]:
        mask_image = cv2.resize(
            mask_image,
            (original_image.shape[1], original_image.shape[0]),
        )

    masks_planaria = biggest_objects_in_the_binary_mask(
        mask_image, number_of_planaria_on_image,
        number_rows_of_planaria_on_image,
    )

    for i, mask_planaria in enumerate(masks_planaria):
        if mask_planaria is None:
            continue
        stereotypie_image = compute_mask_image_planarian_reshaped(
            original_image, mask_planaria, image_size,
            margin_percent, apply_mask=apply_mask,
            black_and_white=black_and_white,
        )
        img_name_parts = img_name.split("_")
        if (
            len(img_name_parts) >= 4
            and number_of_planaria_on_image > 1
        ):
            img_name_parts.insert(-4, f"planaria_{i + 1}")
        img_name_to_save = "_".join(img_name_parts)
        path_to_save = os.path.join(out_dir, img_name_to_save)
        if not overwrite and os.path.isfile(path_to_save):
            n_skipped += 1
            continue
        cv2.imwrite(path_to_save, stereotypie_image)
        n_written += 1
    return img_name, n_written, n_skipped, log_lines


# ---------------------------------------------------------------------------
# Geometry helpers (verbatim port from notebook 3, cell 2)
# ---------------------------------------------------------------------------
def resize_with_black_borders(original_image: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Resize ``original_image`` to a square canvas keeping the aspect ratio."""
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    original_height, original_width = original_image.shape[:2]
    square_size = max(original_width, original_height)
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    x_offset = (square_size - original_width) // 2
    y_offset = (square_size - original_height) // 2
    square_image[y_offset:y_offset + original_height,
                 x_offset:x_offset + original_width] = original_image
    return cv2.resize(square_image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def compute_mask_image_planarian_reshaped(
    base_image: np.ndarray,
    mask_planarian: np.ndarray,
    image_size: int = 256,
    margin_percent: float = 0.05,
    apply_mask: bool = True,
    black_and_white: bool = True,
    show: bool = False,
) -> np.ndarray:
    """Crop the planarian bounding box and normalise it to ``image_size``."""
    if black_and_white:
        gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bitwise_not(gray_image)
    else:
        gray_image = base_image.copy()

    if apply_mask:
        if np.all(mask_planarian == 0):
            masked_image = mask_planarian
        else:
            masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask_planarian)
        if black_and_white:
            masked_image = cv2.normalize(
                masked_image, None, 0, 255, cv2.NORM_MINMAX,
                dtype=cv2.CV_8U, mask=mask_planarian,
            )
    else:
        masked_image = gray_image.copy()
        masked_image = cv2.normalize(
            masked_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
        )

    if black_and_white:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        masked_image = clahe.apply(masked_image)

    contours, _ = cv2.findContours(mask_planarian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return base_image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if h != w and not apply_mask:
        if w > h:
            y -= round((w - h) / 2)
            h = w
            if y < 0:
                y = 0
            if y + h > mask_planarian.shape[0]:
                new_h = y + h - mask_planarian.shape[0]
                y -= new_h
        elif w < h:
            x -= round((h - w) / 2)
            w = h
            if x < 0:
                x = 0
            if x + w > mask_planarian.shape[1]:
                new_w = x + w - mask_planarian.shape[1]
                x -= new_w

    # Grow the bounding box by ``margin`` pixels on all four sides.
    # The pre-v1.2 code used ``h += margin; w += margin`` which cancelled
    # the shift on x and y, so the crop grew only on the top and left
    # edges. The fix doubles the extra width/height so that the final
    # box spans (x - margin, y - margin, w + 2*margin, h + 2*margin).
    # Symmetric padding is the convention used by classical crop-and-pad
    # augmentations, for example Buslaev, A. et al., "Albumentations:
    # Fast and Flexible Image Augmentations", Information, 2020, DOI:
    # 10.3390/info11020125.
    margin = round(margin_percent * image_size)
    x -= margin
    y -= margin
    h += 2 * margin
    w += 2 * margin

    # Clip to image bounds on the left/top.
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    # Clip to image bounds on the right/bottom.
    if x + w > mask_planarian.shape[1]:
        w = mask_planarian.shape[1] - x
    if y + h > mask_planarian.shape[0]:
        h = mask_planarian.shape[0] - y

    cropped_image = masked_image[y:y + h, x:x + w]
    cropped_image = resize_with_black_borders(cropped_image, target_size=image_size)
    return cropped_image


# ---------------------------------------------------------------------------
# Mask folder helpers
# ---------------------------------------------------------------------------
def get_mask_folders(directory_path: str) -> List[str]:
    """Return subdirectory names starting with ``mask`` inside ``directory_path``."""
    try:
        entries = os.listdir(directory_path)
    except (FileNotFoundError, NotADirectoryError):
        return []
    return [
        entry for entry in entries
        if os.path.isdir(os.path.join(directory_path, entry)) and entry.startswith("mask")
    ]


# ---------------------------------------------------------------------------
# Biggest-object helpers (port from notebook 3, cell 4)
# ---------------------------------------------------------------------------
def _find_objects_in_zones(height, width, contours, num_rows, num_columns):
    zone_height = height / num_rows
    zone_width = width / num_columns
    center_points = [
        (int(zone_width * (j + 0.5)), int(zone_height * (i + 0.5)))
        for i in range(num_rows)
        for j in range(num_columns)
    ]
    selected_objects = []
    for center in center_points:
        cx, cy = center
        zone_x1, zone_y1 = int(cx - zone_width / 2), int(cy - zone_height / 2)
        zone_x2, zone_y2 = int(cx + zone_width / 2), int(cy + zone_height / 2)
        zone_rect = np.array([zone_x1, zone_y1, zone_x2, zone_y2])
        max_zone_overlap = 0
        selected_contour = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_rect = np.array([x, y, x + w, y + h])
            ix1 = max(zone_rect[0], contour_rect[0])
            iy1 = max(zone_rect[1], contour_rect[1])
            ix2 = min(zone_rect[2], contour_rect[2])
            iy2 = min(zone_rect[3], contour_rect[3])
            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                contour_area = w * h
                overlap_percentage = intersection_area / contour_area if contour_area else 0
                if overlap_percentage >= 0.51 and intersection_area > max_zone_overlap:
                    max_zone_overlap = intersection_area
                    selected_contour = contour
        selected_objects.append(selected_contour)
    return selected_objects


def biggest_objects_in_the_binary_mask(
    mask: np.ndarray,
    num_objects_to_keep: int,
    num_objects_by_row: int,
) -> List[Optional[np.ndarray]]:
    """Split the binary mask into up to ``num_objects_to_keep`` per-planarian masks.

    Holes inside a worm silhouette (for example the central background
    pixels of a C-shape body curled into a closed ring) are PRESERVED.
    The previous implementation called cv2.findContours with
    cv2.RETR_EXTERNAL, which drops every interior contour from the
    hierarchy, and then redrew the outer contour with
    cv2.drawContours(..., thickness=cv2.FILLED).  Drawing a filled outer
    contour implicitly closes every hole inside it, which is equivalent
    to a binary fill-holes / morphological closing on the mask.  That
    behaviour was never requested and corrupts ring-shaped worms.

    The fix below selects the largest connected component for
    ``num_objects_to_keep == 1`` and intersects the chosen component with
    the ORIGINAL binary mask, so any background pixel that was inside
    the component in the input stays as background in the output.  For
    multi-object zoning the per-zone contour is still rasterised with
    cv2.FILLED, but the result is bit-AND-ed against the input mask for
    the same hole-preservation guarantee.

    References:
    - Suzuki, S. and Abe, K., 'Topological structural analysis of
      digitized binary images by border following', Computer Vision,
      Graphics, and Image Processing 30(1), 32-46, 1985, DOI:
      10.1016/0734-189X(85)90016-7 (algorithm behind cv2.findContours;
      explains why RETR_EXTERNAL drops the inner border list).
    - Rosenfeld, A. and Pfaltz, J.L., 'Sequential operations in digital
      picture processing', Journal of the ACM 13(4), 471-494, 1966,
      DOI: 10.1145/321356.321357 (connected-component labelling used
      by cv2.connectedComponentsWithStats).
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if num_objects_to_keep == 1:
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        if n_lbl <= 1:
            return [None]
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        component_mask = np.where(labels == best, 255, 0).astype(np.uint8)
        keep = cv2.bitwise_and(component_mask, binary)
        return [keep]

    height, width = binary.shape[:2]
    num_rows = math.ceil(num_objects_to_keep / num_objects_by_row)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    largest_contours = _find_objects_in_zones(
        height, width, contours, num_rows, num_objects_by_row,
    )

    individual_masks: List[Optional[np.ndarray]] = []
    for contour in largest_contours:
        if contour is not None:
            outer = np.zeros_like(binary, dtype=np.uint8)
            cv2.drawContours(outer, [contour], -1, 255, thickness=cv2.FILLED)
            individual_masks.append(cv2.bitwise_and(outer, binary))
        else:
            individual_masks.append(None)
    return individual_masks


# ---------------------------------------------------------------------------
# Directory name helper used by Steps 3 and 4
# ---------------------------------------------------------------------------
def build_processed_dir_name(
    name_stereotypie_dir: str,
    image_size: int,
    margin_percent: float,
    black_and_white: bool,
    apply_mask: bool,
) -> str:
    """Return the directory name that Step 3 creates for a given set of parameters."""
    return (
        f"{name_stereotypie_dir}"
        f"_image-size({image_size})"
        f"_margin({margin_percent})"
        f"_b&w({black_and_white})"
        f"_apply-mask({apply_mask})"
    )


def build_temporal_dir_name(
    name_stereotypie_dir: str,
    image_size: int,
    margin_percent: float,
    apply_mask: bool,
) -> str:
    """Return the directory name Step 3 creates for TEMPORAL outputs.

    Distinguished from the standard formatted directory by the
    ``_temporal_`` token so the Step 4 resolver picks the right one
    for fusion + TSM checkpoints (which need the 3-frame masked-
    grayscale stack) versus single-frame CE-head / fusion checkpoints
    (which need the standard masked colour image).  The size, margin,
    and apply-mask tokens are kept so the auto-size align in Pipeline
    A continues to work unchanged.
    """
    return (
        f"{name_stereotypie_dir}"
        f"_temporal_image-size({image_size})"
        f"_margin({margin_percent})"
        f"_apply-mask({apply_mask})"
    )


# ---------------------------------------------------------------------------
# Temporal-image generator (Step 3 for TSM models)
# ---------------------------------------------------------------------------
def _parse_app_second(image_name: str) -> Optional[float]:
    """Lightweight FlatMorphAI ``<stem>_<second>s.png`` -> second
    parser used by the temporal-image walker.  Returns ``None`` when
    the filename does not match the convention so the frame is
    excluded from the dyn_window mapping (a per-folder WARNING is
    logged once, not per frame).
    """
    import re as _re
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    m = _re.search(r"_(?P<sec>[0-9]+(?:\.[0-9]+)?)s?$", stem)
    if not m:
        return None
    try:
        return float(m.group("sec"))
    except ValueError:
        return None


def _dyn_window(n: int, tot: int) -> List[int]:
    """The 3-frame window the TSM model was trained with.  Verbatim
    rule from ``fusion_tsm/nomenclature.py``; duplicated here so this
    module has no import-time dependency on the torch-bearing TSM
    subpackage.
    """
    if tot <= 1:
        return [n, n, n]
    if tot == 2:
        return [1, 1, 2] if n >= 2 else [1, 2, 2]
    if n <= 1:
        return [1, 2, 3]
    if n >= tot:
        return [tot - 2, tot - 1, tot]
    return [n - 1, n, n + 1]


def generate_temporal_images_for_classification(
    path_dataset: str = "Processed_videos_Frame_by_frame",
    name_stereotypie_dir: str = "Formated",
    name_mask_dir: Optional[List[str]] = None,
    image_size: int = 384,
    margin_percent: float = 0.0,
    apply_mask: bool = True,
    folders_stereotypie_to_process: Optional[List[str]] = None,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    is_interrupted: InterruptCheck = _noop_interrupt,
) -> int:
    """Step 3 for fusion + TSM classifiers.

    For each labelled frame, resolve the (T-1, T0, T+1) acquisition
    window from the FlatMorphAI naming convention
    (``<video_stem>_<second>s.png``, in ascending second order), read
    the three masked grayscale frames, and emit a single 384 x 384
    BGR image where channel 0 is T-1, channel 1 is T0, channel 2 is
    T+1.  The output is what the TSM model expects at inference time
    (its image branch is a temporal 9-channel tensor built from these
    3 channels at runtime).

    Directory: ``<video>/<temporal_dir>/<image_name>`` where
    ``temporal_dir`` is produced by ``build_temporal_dir_name``.

    Skip-existing semantics mirror the standard generator: when the
    output PNG is already on disk and ``overwrite`` is False the row
    is skipped without reading masks or rebuilding the temporal
    image.  Reference for skip-existing: Feldman, S.I., "Make: a
    program for maintaining computer programs", Software: Practice
    and Experience 9(4), 255-265, 1979, DOI:10.1002/spe.4380090402.
    """
    _log = log if log is not None else (lambda _msg: None)
    folders = folders_stereotypie_to_process or os.listdir(path_dataset)
    folders = [
        f for f in folders
        if os.path.isdir(os.path.join(path_dataset, f))
    ]

    out_dir_name = build_temporal_dir_name(
        name_stereotypie_dir, image_size, margin_percent, apply_mask,
    )

    total_frames = 0
    per_video_pngs: List[List[Tuple[float, str]]] = []
    for folder in folders:
        images_dir = os.path.join(path_dataset, folder, "images")
        if not os.path.isdir(images_dir):
            per_video_pngs.append([])
            continue
        rows = []
        n_skip = 0
        for fn in sorted(os.listdir(images_dir)):
            if not fn.lower().endswith(".png"):
                continue
            sec = _parse_app_second(fn)
            if sec is None:
                n_skip += 1
                continue
            rows.append((float(sec), fn))
        rows.sort(key=lambda t: (t[0], t[1]))
        per_video_pngs.append(rows)
        total_frames += len(rows)
        if n_skip:
            _log(
                f"  '{folder}': {n_skip} frame(s) skipped "
                f"(filename does not match the v1.2 naming convention)."
            )
    _log(
        f"[Step 3 temporal] Found {len(folders)} video(s), "
        f"{total_frames} parseable frame(s) total."
    )

    total_written = 0
    frames_done = 0
    for folder, rows in zip(folders, per_video_pngs):
        if is_interrupted():
            break
        if not rows:
            continue
        video_dir = os.path.join(path_dataset, folder)
        # Resolve the mask sub-folder per video; the caller may pin
        # one via name_mask_dir, otherwise we auto-detect the first
        # ``mask*/`` directory under the video.
        if name_mask_dir:
            mask_dirs = list(name_mask_dir)
        else:
            mask_dirs = get_mask_folders(video_dir)
        if not mask_dirs:
            _log(
                f"  '{folder}': no mask sub-folder found, skipping "
                "temporal-image generation for this video."
            )
            frames_done += len(rows)
            if progress_cb is not None:
                progress_cb(frames_done, total_frames, folder)
            continue
        mask_dir = mask_dirs[0]
        images_dir = os.path.join(video_dir, "images")
        masks_dir = os.path.join(video_dir, mask_dir)
        out_full_dir = os.path.join(video_dir, out_dir_name)
        os.makedirs(out_full_dir, exist_ok=True)

        # Per-video read cache so each PNG is decoded at most once
        # (each frame appears as cur once and as prev/next of its two
        # neighbours, so the cache cuts cv2.imdecode calls by ~3x).
        gray_cache: dict = {}
        mask_cache: dict = {}

        def _g(name: str):
            if name not in gray_cache:
                p = os.path.join(images_dir, name)
                if not os.path.isfile(p):
                    gray_cache[name] = None
                else:
                    try:
                        gray_cache[name] = cv2.imdecode(
                            np.fromfile(p, dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE,
                        )
                    except Exception:  # noqa: BLE001
                        gray_cache[name] = None
            return gray_cache[name]

        def _m(name: str):
            if name not in mask_cache:
                p = os.path.join(masks_dir, name)
                if not os.path.isfile(p):
                    mask_cache[name] = None
                else:
                    try:
                        mask_cache[name] = cv2.imdecode(
                            np.fromfile(p, dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE,
                        )
                    except Exception:  # noqa: BLE001
                        mask_cache[name] = None
            return mask_cache[name]

        # Pre-warm the shared decode caches single-threaded so the
        # worker pool below only READS the dicts (no concurrent
        # writes to the same key -> no race on cv2.imdecode).  Each
        # frame consults its dyn_window neighbours (prev, cur,
        # next), so we walk every planned window once here.
        n_skipped_existing = 0
        n_skipped_missing_mask = 0
        todo: List[Tuple[int, str, str, str, str]] = []
        for idx, (_sec, image_name) in enumerate(rows):
            if is_interrupted():
                break
            out_path = os.path.join(out_full_dir, image_name)
            if (not overwrite) and os.path.isfile(out_path):
                n_skipped_existing += 1
                frames_done += 1
                if progress_cb is not None:
                    progress_cb(frames_done, total_frames, folder)
                continue
            n = idx + 1
            ks = _dyn_window(n, len(rows))
            prev_name = rows[ks[0] - 1][1]
            cur_name  = rows[ks[1] - 1][1]
            next_name = rows[ks[2] - 1][1]
            for _nm in (prev_name, cur_name, next_name):
                _g(_nm)
                _m(_nm)
            todo.append((idx, image_name, prev_name, cur_name, next_name))

        # Lazy-import build_temporal_image once outside the pool so
        # the pool workers never race on the module cache.
        try:
            from .fusion_tsm.temporal_format import build_temporal_image
        except Exception as exc:  # noqa: BLE001
            _log(
                f"  ERROR: cannot import build_temporal_image "
                f"({type(exc).__name__}: {exc}); aborting "
                "temporal-image generation."
            )
            return total_written

        # Same rule as before (one thread per core, minus one for the
        # interface) but through the shared reading, so a run restricted to
        # part of the machine sizes itself on what it may actually use
        # instead of on the host's core count.
        from ..cpu_topology import pool_size
        max_workers = pool_size(reserve=1)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _process_one(
            spec: Tuple[int, str, str, str, str],
        ) -> Tuple[str, int, int]:
            _, img_name, prev_name, cur_name, next_name = spec
            gp = gray_cache.get(prev_name)
            gc = gray_cache.get(cur_name)
            gn = gray_cache.get(next_name)
            mp = mask_cache.get(prev_name)
            mc = mask_cache.get(cur_name)
            mn = mask_cache.get(next_name)
            if gc is None or mc is None:
                return img_name, 0, 1  # missing mask -> skip
            temporal_bgr = build_temporal_image(
                grays=(gp, gc, gn),
                masks=(mp, mc, mn),
                image_size=image_size,
            )
            if temporal_bgr is None:
                return img_name, 0, 1
            out_path_local = os.path.join(out_full_dir, img_name)
            ok = cv2.imwrite(out_path_local, temporal_bgr)
            return img_name, (1 if ok else 0), 0

        if todo:
            pending = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for spec in todo:
                    if is_interrupted():
                        break
                    pending.append(pool.submit(_process_one, spec))
                for fut in as_completed(pending):
                    if is_interrupted():
                        for p in pending:
                            p.cancel()
                        break
                    try:
                        _img, n_w, n_s = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        if log is not None:
                            log(
                                "  WARN: temporal-image worker "
                                f"failed ({type(exc).__name__}: {exc})"
                            )
                        n_w = 0
                        n_s = 1
                    total_written += n_w
                    n_skipped_missing_mask += n_s
                    frames_done += 1
                    if progress_cb is not None:
                        progress_cb(frames_done, total_frames, folder)

        if n_skipped_existing and not overwrite:
            _log(
                f"  '{folder}': skipped {n_skipped_existing} existing "
                "temporal image(s) (set overwrite=True to regenerate)."
            )
        if n_skipped_missing_mask:
            _log(
                f"  '{folder}': skipped {n_skipped_missing_mask} "
                "frame(s) (missing mask / unreadable image)."
            )

    if log is not None:
        log(
            f"[Step 3 temporal] finished: {total_written} temporal "
            f"image(s) written under '<video>/{out_dir_name}/'."
        )
    return total_written


# ---------------------------------------------------------------------------
# Main generator (port from notebook 3, cell 4)
# ---------------------------------------------------------------------------
def generate_stereotypie_image_for_classification(
    path_dataset: str = "Processed_videos_Frame_by_frame",
    name_stereotypie_dir: str = "Formated",
    name_mask_dir: Optional[List[str]] = None,
    number_of_planaria_on_image: int = 1,
    image_size: int = 224,
    margin_percent: float = 0.0,
    number_rows_of_planaria_on_image: int = 1,
    folders_stereotypie_to_process: Optional[List[str]] = None,
    apply_mask: bool = True,
    black_and_white: bool = False,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    preview_cb: Optional[PreviewCallback] = None,
    is_interrupted: InterruptCheck = _noop_interrupt,
    max_workers: Optional[int] = None,
) -> int:
    """Generate cropped per-planarian images for every video in ``path_dataset``.

    Returns the number of classification images successfully written.

    When ``overwrite`` is False (the default), every frame whose output
    file is already on disk is SKIPPED -- no mask read, no model crop,
    no imwrite.  The per-folder log line reports ``skipped=N`` so the
    operator sees how much work was avoided.  Set ``overwrite=True`` to
    force regeneration (for example after switching the segmentation
    model and wanting to refresh every formatted frame).

    Rationale for the skip-existing default: file-timestamp invalidation
    is the canonical build-system rule (Feldman, S.I., "Make: a program
    for maintaining computer programs", Software: Practice and
    Experience 9(4), 255-265, 1979, DOI:10.1002/spe.4380090402).
    Surfacing the per-folder skip count follows rule 4 of Wilson, G. et
    al., "Good Enough Practices in Scientific Computing", PLOS Comp
    Biol 13(6):e1005510, 2017, DOI:10.1371/journal.pcbi.1005510.
    """
    folders = folders_stereotypie_to_process or os.listdir(path_dataset)
    folders = [f for f in folders if os.path.isdir(os.path.join(path_dataset, f))]
    total_written = 0

    # Total number of frames across every video, for a single progress bar.
    total_frames = 0
    per_video_images: List[List[str]] = []
    for folder in folders:
        images_dir = os.path.join(path_dataset, folder, "images")
        try:
            pngs = sorted(p for p in os.listdir(images_dir) if p.lower().endswith(".png"))
        except FileNotFoundError:
            pngs = []
        per_video_images.append(pngs)
        total_frames += len(pngs)
    if log is not None:
        log(f"Found {len(folders)} video(s), {total_frames} total frame(s)")

    # Worker-pool sizing.  We use ``ProcessPoolExecutor`` here (not
    # ``ThreadPoolExecutor``) because the per-frame work runs
    # significant Python glue between cv2 calls -- contour
    # extraction, mask splitting, bbox clipping, resize-with-borders
    # -- and threads contend for the GIL on every one of those.
    # Processes give true multi-core parallelism: each worker gets
    # its own interpreter and its own cv2 initialisation.  The
    # pickle payload per task is small (a flat tuple of str/int/bool
    # values) so the round-trip is dominated by the useful work.
    #
    # We leave ONE core free for the GUI event loop and reserve a
    # SECOND core for the parent process's coordination work
    # (submit / drain / progress signals) so the visible progress
    # bar stays responsive during a heavy rebuild.  Reference for
    # the "GIL-free CPU-bound work belongs in processes" pattern:
    # Beazley, D. M., "Understanding the Python GIL", PyCon 2010,
    # https://www.dabeaz.com/python/UnderstandingGIL.pdf.
    # An explicit ``max_workers`` from the Settings page wins; 0 or
    # None means "auto" and keeps the cores-minus-two heuristic -- now read
    # through the shared helper (affinity included) rather than from the
    # host's logical count.
    from ..cpu_topology import pool_size
    _explicit = bool(max_workers and int(max_workers) > 0)
    if _explicit:
        max_workers = max(1, int(max_workers))
    else:
        # No fixed ceiling any more: the former cap of 16 had no measurement
        # behind it, and ``_auto_workers`` below already bounds the pool by the
        # number of frames left to write (~4 per worker) and by the RAM
        # available at that moment. On a bigger machine the pool now follows
        # the cores; on this one nothing changes.
        max_workers = pool_size(reserve=2)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    frames_done = 0

    def _auto_workers(n_todo):
        """Adaptive pool size (auto mode): never more processes than frames
        to write (~4 frames each, a worker costs ~1 s to start), and no more
        than the RAM available now allows (~180 MB per worker, keeping the
        same reserve as the Step-4 planner). Step 3 runs per folder DURING
        the segmentation and the descriptor workers, so RAM is shared."""
        n = min(max_workers, max(1, -(-n_todo // 4)))
        try:
            mi = {}
            if os.name == "nt":      # native Windows: no /proc
                from ..platform_compat import memory_status_mb
                mi = memory_status_mb()
            else:
                with open("/proc/meminfo", encoding="ascii") as fh:
                    for ln in fh:
                        k, v = ln.split(":", 1)
                        mi[k] = float(v.split()[0]) / 1024.0
            avail, total = mi.get("MemAvailable", 0.0), mi.get("MemTotal", 0.0)
            if avail > 0:
                reserve = max(600.0, 0.07 * total)
                n = min(n, max(1, int((avail - reserve) // 180.0)))
        except Exception:  # noqa: BLE001 - not Linux: CPU rule only
            pass
        return max(1, n)

    # The pool is created only when a frame really has to be written (a
    # fully cached folder costs no process start at all), sized on the work.
    _pool_box = {"pool": None}

    def _get_pool(n_todo):
        if _pool_box["pool"] is None:
            nw = max_workers if _explicit else _auto_workers(n_todo)
            try:
                _pool_box["pool"] = ProcessPoolExecutor(max_workers=nw, max_tasks_per_child=2000)
            except TypeError:  # pragma: no cover -- Python < 3.11
                _pool_box["pool"] = ProcessPoolExecutor(max_workers=nw)
            if log is not None:
                log(f"Formatted-image pool: {nw} worker process(es)"
                    + ("" if _explicit else " (adaptive: frames to write, RAM available)"))
        return _pool_box["pool"]

    try:
        for folder, pngs in zip(folders, per_video_images):
            if is_interrupted():
                break
            mask_dirs: List[str]
            if not name_mask_dir:
                mask_dirs = get_mask_folders(
                    os.path.join(path_dataset, folder)
                )
            else:
                mask_dirs = list(name_mask_dir)

            path_stereotypie_dir = os.path.join(
                path_dataset, folder,
                build_processed_dir_name(
                    name_stereotypie_dir, image_size, margin_percent,
                    black_and_white, apply_mask,
                ),
            )
            os.makedirs(path_stereotypie_dir, exist_ok=True)
            path_images_dir = os.path.join(
                path_dataset, folder, "images",
            )

            if log is not None:
                log(
                    f"Processing video '{folder}' with {len(pngs)} "
                    f"frame(s)"
                )

            skipped_existing = 0
            # Pre-build the flat picklable task tuples so the pool
            # submission is a tight loop with no per-iteration
            # closure work.  ``mask_paths`` is the ordered list of
            # candidate mask file paths the worker probes in order.
            tasks = []
            for img_name in pngs:
                if is_interrupted():
                    break
                full_img_path = os.path.join(path_images_dir, img_name)
                mask_paths = [
                    os.path.join(
                        path_dataset, folder, candidate, img_name,
                    )
                    for candidate in mask_dirs
                ]
                tasks.append(
                    (
                        img_name,
                        full_img_path,
                        path_stereotypie_dir,
                        mask_paths,
                        bool(overwrite),
                        bool(apply_mask),
                        bool(black_and_white),
                        int(image_size),
                        float(margin_percent),
                        int(number_of_planaria_on_image),
                        int(number_rows_of_planaria_on_image),
                        False,  # reserved for a future hint slot
                    )
                )

            # ``as_completed`` order-independent progress: each
            # future increments ``frames_done`` exactly once and
            # reports via progress_cb.
            # Already-written outputs: same test as the worker's first line
            # (single planarian, no overwrite, file present) -> no submission.
            todo = []
            for task in tasks:
                if (not overwrite and int(number_of_planaria_on_image) == 1
                        and os.path.isfile(os.path.join(task[2], task[0]))):
                    skipped_existing += 1
                    frames_done += 1
                    if progress_cb is not None:
                        progress_cb(frames_done, total_frames, folder)
                else:
                    todo.append(task)
            tasks = todo
            pending_futures = []
            if tasks:
                _n_left = len(tasks) + sum(len(p) for p in per_video_images[folders.index(folder) + 1:])
                _pool = _get_pool(_n_left)
                pending_futures = [
                    _pool.submit(_format_one_frame_worker, task)
                    for task in tasks
                ]
            for fut in as_completed(pending_futures):
                if is_interrupted():
                    for pending in pending_futures:
                        pending.cancel()
                    break
                try:
                    _img, n_w, n_s, log_lines = fut.result()
                except Exception as exc:  # noqa: BLE001
                    if log is not None:
                        log(
                            f"  WARN: worker failed "
                            f"({type(exc).__name__}: {exc})"
                        )
                    n_w = 0
                    n_s = 0
                    log_lines = []
                # Drain any log messages the worker produced.  Runs
                # on the caller's thread (Step 3 Worker), which is
                # already off the main Qt thread, so ``log`` (a
                # bound signal emitter) is thread-safe.
                if log is not None and log_lines:
                    for line in log_lines:
                        log(line)
                total_written += n_w
                skipped_existing += n_s
                frames_done += 1
                if progress_cb is not None:
                    progress_cb(frames_done, total_frames, folder)
                # Preview: intentionally NOT emitted from
                # ProcessPool workers -- pickling numpy arrays back
                # to the parent for every 10th frame would negate
                # the parallel win.  A future refactor could open
                # the just-written PNG for the preview instead.
                if (
                    preview_cb is not None
                    and (frames_done % 10 == 0)
                ):
                    try:
                        # Best-effort: reload the freshly written
                        # frame from disk for the preview panel.
                        preview_path = os.path.join(
                            path_stereotypie_dir, _img,
                        )
                        if os.path.isfile(preview_path):
                            preview_img = cv2.imdecode(
                                np.fromfile(
                                    preview_path, dtype=np.uint8,
                                ),
                                cv2.IMREAD_UNCHANGED,
                            )
                            if preview_img is not None:
                                preview_cb(
                                    preview_img, preview_img,
                                    preview_img, _img,
                                )
                    except Exception:
                        pass

            # End-of-folder summary: report how many existing outputs
            # were skipped so the user sees that the existing-file
            # check is active and roughly how much work was avoided.
            if log is not None and skipped_existing:
                log(
                    f"  '{folder}': skipped {skipped_existing} existing "
                    f"output file(s) (set overwrite=True to regenerate)."
                )
    finally:
        if _pool_box["pool"] is not None:
            _pool_box["pool"].shutdown(wait=True, cancel_futures=True)

    if log is not None:
        log(f"Generation finished: {total_written} image(s) written")
    return total_written
