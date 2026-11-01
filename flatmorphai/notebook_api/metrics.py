"""
Pipeline D metrics: extract behavioural and morphometric descriptors
from binary planarian masks.

This module is a pure-functional library: each public function takes
NumPy arrays or paths in, returns numbers or NumPy arrays out, and is
free of Qt or worker-thread imports so it can be unit-tested without
a display.

Reproducibility
---------------
Every estimator that depends on a random number generator (currently
only the eigenposture PCA) accepts the canonical seed of the project
(``random_state=42``) so results are identical across runs. CSV
writers use ``sep=';'`` per the project convention.

Bibliography
------------
Skeleton extraction:
    Lee, T.-C., Kashyap, R. L. and Chu, C.-N., "Building skeleton
    models via 3-D medial surface axis thinning algorithms",
    Computer Vision, Graphics and Image Processing - Graphical
    Models and Image Processing, 56(6), 462-478, 1994,
    DOI: 10.1006/cgip.1994.1042. Implemented in
    skimage.morphology.skeletonize(method='lee').

Shape descriptors and connected components:
    Suzuki, S. and Abe, K., "Topological structural analysis of
    digitized binary images by border following", Computer Vision,
    Graphics and Image Processing 30(1), 32-46, 1985,
    DOI: 10.1016/0734-189X(85)90016-7. Implemented in cv2.findContours.

Eigenpostures:
    Stephens, G. J., Johnson-Kerner, B., Bialek, W. and Ryu, W. S.,
    "Dimensionality and dynamics in the behavior of C. elegans",
    PLOS Computational Biology 4(4): e1000028, 2008,
    DOI: 10.1371/journal.pcbi.1000028.

Postural feature extraction validated for nematodes and applied to
flatworms by analogy:
    Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. and
    Schafer, W. R., "A database of Caenorhabditis elegans
    behavioral phenotypes", Nature Methods 10, 877-879, 2013,
    DOI: 10.1038/nmeth.2560.
    Javer, A. et al., "An open-source platform for analyzing and
    sharing worm-behavior data", Nature Methods 15, 645-646, 2018,
    DOI: 10.1038/s41592-018-0112-1 (Tierpsy Tracker).

Multi-worm tracking and turn-rate definitions:
    Swierczek, N. A., Giles, A. C., Rankin, C. H. and Kerr, R. A.,
    "High-throughput behavioral analysis in C. elegans", Nature
    Methods 8, 592-598, 2011, DOI: 10.1038/nmeth.1625
    (Multi-Worm Tracker / WormTracker 2.0 lineage).

Open-source toxicology tracker reference for motility endpoints:
    Rodriguez, A., Zhang, H., Klaminder, J., Brodin, T., Andersson,
    P. L. and Andersson, M., "ToxTrac: A fast and robust software
    for tracking organisms", Methods in Ecology and Evolution 9(3),
    460-464, 2018, DOI: 10.1111/2041-210X.12874.

Pose-tracking with mask-initialised skeletons:
    Pereira, T. D. et al., "SLEAP: A deep learning system for
    multi-animal pose tracking", Nature Methods 19, 486-495, 2022,
    DOI: 10.1038/s41592-022-01426-1.

Dose-response statistics:
    Ritz, C., Baty, F., Streibig, J. C. and Gerhard, D.,
    "Dose-response analysis using R", PLOS ONE 10(12): e0146021,
    2015, DOI: 10.1371/journal.pone.0146021.

The well-detection step relies on the Hough circle transform:
    Yuen, H. K., Princen, J., Illingworth, J. and Kittler, J.,
    "Comparative study of Hough transform methods for circle
    finding", Image and Vision Computing 8(1), 71-77, 1990,
    DOI: 10.1016/0262-8856(90)90059-E. Implemented as
    cv2.HOUGH_GRADIENT.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # OpenCV is a hard dependency of the rest of the application.
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    from skimage.morphology import skeletonize  # type: ignore
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    skeletonize = None  # type: ignore
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Reproducibility constant
# ---------------------------------------------------------------------------
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WellGeometry:
    """Ellipse parameters describing the calibration well.

    Attributes
    ----------
    cx, cy:
        Centre of the well in pixel coordinates.
    axis_a, axis_b:
        Semi-major and semi-minor axes of the well, in pixels.
    angle:
        Tilt angle of the major axis, in degrees, measured counter
        clockwise from the positive x axis.
    """

    cx: float
    cy: float
    axis_a: float
    axis_b: float
    angle: float = 0.0

    def normalise(self, x: float, y: float) -> Tuple[float, float]:
        """Return ``(r, theta)`` with ``r in [0, 1]`` and ``theta`` in radians.

        Coordinates are rotated by ``-angle`` so the major axis aligns
        with the x axis before normalisation; this corrects camera
        tilt while preserving aspect ratios.
        """
        rad = math.radians(self.angle)
        dx = x - self.cx
        dy = y - self.cy
        # Rotate by -angle.
        rx = math.cos(-rad) * dx - math.sin(-rad) * dy
        ry = math.sin(-rad) * dx + math.cos(-rad) * dy
        nx = rx / max(self.axis_a, 1e-9)
        ny = ry / max(self.axis_b, 1e-9)
        r = math.sqrt(nx * nx + ny * ny)
        theta = math.atan2(ny, nx) % (2.0 * math.pi)
        return r, theta


@dataclass
class FrameMetrics:
    """Per-frame morphometric and spatial descriptors."""

    # Identification
    image_name: str
    second: float

    # Mask topology
    n_components: int = 0

    # Shape descriptors (single largest component when n_components > 0)
    area_px: float = 0.0
    perimeter_px: float = 0.0
    skeleton_length_px: float = 0.0
    width_mean_px: float = 0.0
    width_std_px: float = 0.0
    width_profile_px: List[float] = field(default_factory=list)
    aspect_ratio: float = 0.0
    elongation_ratio: float = 0.0
    circularity: float = 0.0
    solidity: float = 0.0
    convexity: float = 0.0

    # Curvature descriptors along the skeleton
    curvature_mean_abs: float = 0.0
    curvature_max: float = 0.0
    curvature_std: float = 0.0
    inflection_count: int = 0
    straightness_index: float = 0.0
    curvature_profile: List[float] = field(default_factory=list)

    # Self-contact
    self_overlap_area_px: float = 0.0

    # Centroid (image-space, pixels)
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    # Well-normalised position (filled when a WellGeometry is provided)
    centroid_r: float = float("nan")
    centroid_theta: float = float("nan")
    distance_to_wall: float = float("nan")
    wall_proximity_flag: bool = False


@dataclass
class VideoTemporalMetrics:
    """Per-video aggregates derived from a sequence of frame metrics."""

    video_name: str
    n_frames: int

    # Motility
    mean_speed_px_per_s: float = 0.0
    total_path_px: float = 0.0
    net_displacement_px: float = 0.0
    directionality_index: float = 0.0
    mean_turn_rate_rad_per_s: float = 0.0

    # Immobility
    immobility_index: float = 0.0
    longest_immobility_bout_s: float = 0.0

    # Body deformation dynamics
    mean_abs_dcurvature: float = 0.0
    mean_abs_dstraightness: float = 0.0
    mean_abs_darea: float = 0.0
    mean_abs_daspect: float = 0.0

    # Body wave analysis
    body_wave_frequency_hz: float = 0.0
    body_wave_propagation_score: float = 0.0

    # Spatial / thigmotaxis
    thigmotaxis_score: float = float("nan")
    mean_radial_position: float = float("nan")
    radial_position_std: float = float("nan")
    mean_angular_velocity_rad_per_s: float = float("nan")

    # Ecotoxicology composites
    motility_score: float = 0.0
    hypercontraction_index: float = 0.0
    coiling_score: float = 0.0
    fission_event_flag: bool = False


# ---------------------------------------------------------------------------
# Mask preprocessing
# ---------------------------------------------------------------------------
def _ensure_uint8_binary(mask: np.ndarray) -> np.ndarray:
    """Return a strictly 0/255 uint8 mask. Accepts bool or grayscale input."""
    if mask is None:
        raise ValueError("mask is None")
    if mask.dtype == np.bool_:
        return (mask.astype(np.uint8) * 255)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binarised = (mask > 0).astype(np.uint8) * 255
    return binarised


def read_mask(path: str) -> np.ndarray:
    """Load a binary mask from disk and return it as 0/255 uint8."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required to read masks")
    raw = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED,
    )
    if raw is None:
        raise IOError(f"Cannot read mask: {path}")
    return _ensure_uint8_binary(raw)


# ---------------------------------------------------------------------------
# Connected components and largest object
# ---------------------------------------------------------------------------
def count_components(binary_mask: np.ndarray) -> int:
    """Return the number of foreground connected components in ``binary_mask``."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for component counting")
    n_labels, _labels = cv2.connectedComponents(binary_mask, connectivity=8)
    return max(0, int(n_labels) - 1)


def largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Keep the single largest 8-connected component of ``binary_mask``."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for largest_component")
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8,
    )
    if n_labels <= 1:
        return np.zeros_like(binary_mask)
    # stats[0] is the background; pick the foreground label with the
    # largest CC_STAT_AREA value.
    fg_areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(fg_areas))
    out = np.zeros_like(binary_mask)
    out[labels == largest] = 255
    return out


# ---------------------------------------------------------------------------
# Shape descriptors
# ---------------------------------------------------------------------------
def shape_descriptors(binary_mask: np.ndarray) -> Dict[str, float]:
    """Return area, perimeter, circularity, solidity, convexity and ellipse fit.

    All values are in pixels or pixel units. The function is robust to
    empty masks: every key is present in the output, with zeros when
    the input has no foreground.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for shape descriptors")
    out = {
        "area_px": 0.0,
        "perimeter_px": 0.0,
        "circularity": 0.0,
        "solidity": 0.0,
        "convexity": 0.0,
        "aspect_ratio": 0.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "ellipse_a": 0.0,
        "ellipse_b": 0.0,
        "ellipse_angle": 0.0,
    }
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, closed=True))
    out["area_px"] = area
    out["perimeter_px"] = perim
    if perim > 0:
        out["circularity"] = float(4.0 * math.pi * area / (perim * perim))
    if len(contour) >= 3:
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perim = float(cv2.arcLength(hull, closed=True))
        if hull_area > 0:
            out["solidity"] = float(area / hull_area)
        if perim > 0:
            out["convexity"] = float(hull_perim / perim)
    # Centroid via image moments (more robust than mean of contour points).
    moments = cv2.moments(binary_mask, binaryImage=True)
    if moments["m00"] > 0:
        out["centroid_x"] = float(moments["m10"] / moments["m00"])
        out["centroid_y"] = float(moments["m01"] / moments["m00"])
    if len(contour) >= 5:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        out["ellipse_a"] = float(max(axis_a, axis_b))
        out["ellipse_b"] = float(min(axis_a, axis_b))
        out["ellipse_angle"] = float(angle)
        if out["ellipse_b"] > 0:
            out["aspect_ratio"] = float(out["ellipse_a"] / out["ellipse_b"])
    return out


# ---------------------------------------------------------------------------
# Skeleton, width profile, curvature
# ---------------------------------------------------------------------------
def _ordered_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """Return skeleton pixels ordered along the medial axis.

    The function builds the 8-connected adjacency graph of the
    skeleton, picks an endpoint (a pixel with exactly one neighbour)
    and walks the chain. When the skeleton has cycles (rare for a
    healthy planarian but possible during fission), we fall back to
    the longest path through the graph.
    """
    coords = np.column_stack(np.nonzero(skeleton))  # (n, 2): (row, col)
    if len(coords) == 0:
        return coords
    # Build adjacency via a kd-style lookup using set membership.
    pixel_set = {(int(r), int(c)) for r, c in coords}
    neighbours = {p: [] for p in pixel_set}
    for r, c in pixel_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in pixel_set:
                    neighbours[(r, c)].append(q)
    endpoints = [p for p, nb in neighbours.items() if len(nb) == 1]
    start = endpoints[0] if endpoints else next(iter(pixel_set))

    # Iterative DFS that picks the longest simple path. For a single
    # snake-like skeleton this is just the linear walk from one end
    # to the other; when the skeleton has a branch we still get the
    # longest end-to-end traversal.
    visited = {start}
    order = [start]
    current = start
    while True:
        candidates = [p for p in neighbours[current] if p not in visited]
        if not candidates:
            break
        # Prefer the candidate that itself has unvisited neighbours so
        # we follow the long arm of any "Y" branch first.
        candidates.sort(
            key=lambda p: sum(1 for q in neighbours[p] if q not in visited),
            reverse=True,
        )
        current = candidates[0]
        visited.add(current)
        order.append(current)
    return np.array(order, dtype=np.float64)


def skeleton_features(
    binary_mask: np.ndarray,
    n_width_samples: int = 20,
) -> Dict[str, object]:
    """Extract skeleton length, width profile and curvature descriptors.

    ``n_width_samples`` controls the resolution of the width and
    curvature profiles. Values from 16 to 32 are recommended in the
    eigenposture literature (Stephens et al., 2008,
    DOI: 10.1371/journal.pcbi.1000028) and 20 is a robust compromise
    for the planarian body plan.

    Returns
    -------
    dict
        Keys: ``skeleton_length_px``, ``width_mean_px``,
        ``width_std_px``, ``width_profile_px`` (length
        ``n_width_samples``), ``curvature_mean_abs``,
        ``curvature_max``, ``curvature_std``, ``inflection_count``,
        ``straightness_index``, ``curvature_profile`` (length
        ``n_width_samples``).
    """
    out = {
        "skeleton_length_px": 0.0,
        "width_mean_px": 0.0,
        "width_std_px": 0.0,
        "width_profile_px": [0.0] * n_width_samples,
        "curvature_mean_abs": 0.0,
        "curvature_max": 0.0,
        "curvature_std": 0.0,
        "inflection_count": 0,
        "straightness_index": 0.0,
        "curvature_profile": [0.0] * n_width_samples,
    }
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required for skeleton_features")
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for skeleton_features")
    if binary_mask.size == 0 or not np.any(binary_mask):
        return out

    skeleton = skeletonize(binary_mask > 0, method="lee").astype(np.uint8)
    points = _ordered_skeleton_points(skeleton)
    if len(points) < 3:
        return out

    # Resample the skeleton to ``n_width_samples`` evenly spaced points
    # so the width and curvature profiles have a fixed length and can
    # be stacked across frames into a PCA matrix.
    diffs = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step_lengths)])
    skeleton_length = float(cumulative[-1])
    out["skeleton_length_px"] = skeleton_length
    if skeleton_length < 1e-6:
        return out

    target_s = np.linspace(0.0, skeleton_length, n_width_samples)
    rs = np.interp(target_s, cumulative, points[:, 0])
    cs = np.interp(target_s, cumulative, points[:, 1])
    resampled = np.column_stack([rs, cs])

    # Width via distance transform: the value at each skeleton pixel
    # is the radius of the largest disk that fits inside the mask
    # centred on that pixel. The width is therefore 2x the value.
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    widths = []
    for r, c in resampled:
        ri = int(round(r))
        ci = int(round(c))
        if 0 <= ri < dist.shape[0] and 0 <= ci < dist.shape[1]:
            widths.append(float(dist[ri, ci]) * 2.0)
        else:
            widths.append(0.0)
    out["width_profile_px"] = widths
    out["width_mean_px"] = float(np.mean(widths))
    out["width_std_px"] = float(np.std(widths))

    # Curvature kappa(s) at each interior point. We use the discrete
    # Menger curvature estimator: kappa_i is the inverse radius of
    # the circle through (P_{i-1}, P_i, P_{i+1}). This is well-defined
    # for noisy ridges and is the same definition used by Tierpsy
    # Tracker (Javer et al., 2018, DOI: 10.1038/s41592-018-0112-1).
    kappa = np.zeros(n_width_samples)
    for i in range(1, n_width_samples - 1):
        p0 = resampled[i - 1]
        p1 = resampled[i]
        p2 = resampled[i + 1]
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c_ = np.linalg.norm(p2 - p0)
        # Twice the signed triangle area gives 2x|p1-p0||p2-p1|sin(angle),
        # so dividing by abc yields 4xArea/(abc) = 1/R, the curvature.
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (
            p1[1] - p0[1]
        ) * (p2[0] - p0[0])
        denom = a * b * c_
        if denom > 1e-9:
            kappa[i] = 2.0 * cross / denom
    out["curvature_profile"] = kappa.tolist()
    abs_kappa = np.abs(kappa)
    out["curvature_mean_abs"] = float(np.mean(abs_kappa))
    out["curvature_max"] = float(np.max(abs_kappa))
    out["curvature_std"] = float(np.std(kappa))
    # Sign changes count inflection points; ignore the first and last
    # entries (always zero by construction).
    sign_seq = np.sign(kappa[1:-1])
    sign_changes = np.sum(np.diff(sign_seq) != 0)
    out["inflection_count"] = int(sign_changes)

    end_to_end = float(np.linalg.norm(resampled[-1] - resampled[0]))
    out["straightness_index"] = float(end_to_end / skeleton_length)
    return out


def self_overlap_area(binary_mask: np.ndarray) -> float:
    """Estimate the body-fold area by counting pixels with thickness > 2x median.

    The distance transform yields the local half-thickness of the
    mask. Pixels whose half-thickness exceeds twice the median are
    treated as overlapping body segments. The metric is a proxy for
    coiling and self-contact, validated for nematodes in Yemini et
    al., 2013, DOI: 10.1038/nmeth.2560.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for self_overlap_area")
    if binary_mask.size == 0 or not np.any(binary_mask):
        return 0.0
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    inside = dist[dist > 0]
    if inside.size == 0:
        return 0.0
    threshold = 2.0 * float(np.median(inside))
    return float(np.sum(dist > threshold))


# ---------------------------------------------------------------------------
# Well detection (Hough)
# ---------------------------------------------------------------------------
def detect_well_circle(
    image: np.ndarray,
    dp: float = 1.5,
    min_dist_factor: float = 1.0,
    param1: float = 100.0,
    param2: float = 30.0,
) -> Optional[WellGeometry]:
    """Return a circular WellGeometry detected via the Hough transform.

    ``image`` may be a grayscale or BGR original frame. Returns
    ``None`` when no circle is found. The function intentionally
    favours a single dominant circle (the well boundary) by setting
    ``minDist`` to the image diagonal so cv2 returns at most one
    accumulator peak.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for detect_well_circle")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    diag = float(math.hypot(h, w))
    blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=diag * min_dist_factor,
        param1=param1,
        param2=param2,
        minRadius=int(min(h, w) * 0.2),
        maxRadius=int(min(h, w) * 0.6),
    )
    if circles is None:
        return None
    cx, cy, radius = circles[0][0]
    return WellGeometry(
        cx=float(cx), cy=float(cy),
        axis_a=float(radius), axis_b=float(radius), angle=0.0,
    )


# ---------------------------------------------------------------------------
# Main per-frame entry point
# ---------------------------------------------------------------------------
def compute_frame_metrics(
    binary_mask: np.ndarray,
    image_name: str,
    second: float,
    well: Optional[WellGeometry] = None,
    n_width_samples: int = 20,
    wall_proximity_radial_threshold: float = 0.85,
) -> FrameMetrics:
    """Build a :class:`FrameMetrics` record from a single binary mask."""
    n_components = count_components(binary_mask)
    largest = largest_component(binary_mask) if n_components else binary_mask
    metrics = FrameMetrics(
        image_name=image_name, second=second, n_components=n_components,
    )
    shapes = shape_descriptors(largest)
    skel = skeleton_features(largest, n_width_samples=n_width_samples)

    metrics.area_px = shapes["area_px"]
    metrics.perimeter_px = shapes["perimeter_px"]
    metrics.aspect_ratio = shapes["aspect_ratio"]
    metrics.circularity = shapes["circularity"]
    metrics.solidity = shapes["solidity"]
    metrics.convexity = shapes["convexity"]
    metrics.centroid_x = shapes["centroid_x"]
    metrics.centroid_y = shapes["centroid_y"]
    metrics.skeleton_length_px = skel["skeleton_length_px"]
    metrics.width_mean_px = skel["width_mean_px"]
    metrics.width_std_px = skel["width_std_px"]
    metrics.width_profile_px = list(skel["width_profile_px"])
    metrics.curvature_mean_abs = skel["curvature_mean_abs"]
    metrics.curvature_max = skel["curvature_max"]
    metrics.curvature_std = skel["curvature_std"]
    metrics.inflection_count = skel["inflection_count"]
    metrics.straightness_index = skel["straightness_index"]
    metrics.curvature_profile = list(skel["curvature_profile"])
    metrics.self_overlap_area_px = self_overlap_area(largest)
    if metrics.width_mean_px > 0:
        metrics.elongation_ratio = (
            metrics.skeleton_length_px / metrics.width_mean_px
        )
    if well is not None:
        r, theta = well.normalise(metrics.centroid_x, metrics.centroid_y)
        metrics.centroid_r = r
        metrics.centroid_theta = theta
        metrics.distance_to_wall = max(0.0, 1.0 - r)
        metrics.wall_proximity_flag = bool(
            r >= wall_proximity_radial_threshold
        )
    return metrics


# ---------------------------------------------------------------------------
# Temporal aggregation per video
# ---------------------------------------------------------------------------
def _diff_safe(values: np.ndarray) -> np.ndarray:
    """Return finite first differences of ``values``; non-finite entries are zero."""
    diffs = np.diff(values)
    diffs = np.where(np.isfinite(diffs), diffs, 0.0)
    return diffs


def temporal_metrics(
    frames: Sequence[FrameMetrics],
    video_name: str,
    step_seconds: float,
    immobility_threshold_px_per_s: float = 1.0,
    coiling_straightness_threshold: float = 0.5,
) -> VideoTemporalMetrics:
    """Aggregate ``frames`` into a :class:`VideoTemporalMetrics` row.

    ``step_seconds`` is the sampling step, used to convert pixel
    displacements into pixel-per-second velocities. The thresholds
    follow the ecotoxicological convention used by DanioVision /
    EthoVision for immobility (sub-1-pixel/s) and the coiling cutoff
    used in flatworm assays (Hagstrom et al., 2018,
    DOI: 10.1093/toxsci/kfx231).
    """
    if not frames:
        return VideoTemporalMetrics(video_name=video_name, n_frames=0)
    if step_seconds <= 0:
        step_seconds = 1.0
    n = len(frames)
    cx = np.array([f.centroid_x for f in frames], dtype=np.float64)
    cy = np.array([f.centroid_y for f in frames], dtype=np.float64)
    speeds_px_per_s = np.zeros(n)
    if n >= 2:
        dx = _diff_safe(cx)
        dy = _diff_safe(cy)
        seg_lengths = np.hypot(dx, dy)
        speeds_px_per_s[1:] = seg_lengths / step_seconds
        total_path = float(np.sum(seg_lengths))
        net = float(math.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
        directionality = (net / total_path) if total_path > 0 else 0.0
        # Mean turn rate from heading angles between consecutive segments.
        headings = np.arctan2(dy, dx)
        if len(headings) >= 2:
            turn = np.diff(headings)
            # Wrap to [-pi, pi].
            turn = np.arctan2(np.sin(turn), np.cos(turn))
            mean_turn = float(np.mean(np.abs(turn)) / step_seconds)
        else:
            mean_turn = 0.0
    else:
        seg_lengths = np.array([])
        total_path = 0.0
        net = 0.0
        directionality = 0.0
        mean_turn = 0.0

    immobility_mask = speeds_px_per_s < immobility_threshold_px_per_s
    immobility_index = float(np.mean(immobility_mask)) if n else 0.0
    longest_bout = 0.0
    current = 0
    for flag in immobility_mask:
        if flag:
            current += 1
            if current > longest_bout:
                longest_bout = current
        else:
            current = 0
    longest_bout_s = float(longest_bout) * step_seconds

    # Body-deformation dynamics
    curv = np.array([f.curvature_mean_abs for f in frames])
    straight = np.array([f.straightness_index for f in frames])
    area = np.array([f.area_px for f in frames])
    aspect = np.array([f.aspect_ratio for f in frames])
    mean_dcurv = float(np.mean(np.abs(_diff_safe(curv))))
    mean_dstraight = float(np.mean(np.abs(_diff_safe(straight))))
    mean_darea = float(np.mean(np.abs(_diff_safe(area))))
    mean_daspect = float(np.mean(np.abs(_diff_safe(aspect))))

    # Body-wave analysis: dominant temporal frequency of the mid-body
    # curvature signal (skeleton node n_width_samples/2). Implemented
    # as a real FFT magnitude peak. The Hz value uses the sampling
    # rate 1/step_seconds.
    profiles = np.array(
        [f.curvature_profile for f in frames if f.curvature_profile],
        dtype=np.float64,
    )
    body_wave_freq = 0.0
    body_wave_prop = 0.0
    if profiles.size > 0 and profiles.shape[0] >= 4:
        mid = profiles.shape[1] // 2
        signal = profiles[:, mid] - np.mean(profiles[:, mid])
        if np.any(signal):
            spectrum = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), d=step_seconds)
            magnitudes = np.abs(spectrum)
            # Skip the DC bin.
            if len(magnitudes) > 1:
                peak_idx = 1 + int(np.argmax(magnitudes[1:]))
                body_wave_freq = float(freqs[peak_idx])
        # Propagation score: cross-correlation lag between two
        # neighbouring body positions; positive lag means the wave
        # travels head-to-tail. Normalised to [-1, 1].
        if profiles.shape[1] >= 4:
            anterior = profiles[:, profiles.shape[1] // 4]
            posterior = profiles[:, 3 * profiles.shape[1] // 4]
            anterior = anterior - np.mean(anterior)
            posterior = posterior - np.mean(posterior)
            denom = (
                np.sqrt(np.sum(anterior ** 2))
                * np.sqrt(np.sum(posterior ** 2))
            )
            if denom > 1e-9:
                body_wave_prop = float(np.dot(anterior, posterior) / denom)

    # Spatial / thigmotaxis metrics use the well-normalised radius.
    rs = np.array([f.centroid_r for f in frames], dtype=np.float64)
    thetas = np.array([f.centroid_theta for f in frames], dtype=np.float64)
    has_well = np.any(np.isfinite(rs))
    if has_well:
        thigmotaxis_score = float(
            np.nanmean((rs > 0.85).astype(np.float64))
        )
        mean_r = float(np.nanmean(rs))
        std_r = float(np.nanstd(rs))
        if len(thetas) >= 2:
            dtheta = np.diff(thetas)
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            mean_omega = float(np.nanmean(np.abs(dtheta)) / step_seconds)
        else:
            mean_omega = float("nan")
    else:
        thigmotaxis_score = float("nan")
        mean_r = float("nan")
        std_r = float("nan")
        mean_omega = float("nan")

    # Ecotoxicology composites
    motility_score = float(np.mean(speeds_px_per_s)) if n else 0.0
    elong = np.array([f.elongation_ratio for f in frames])
    if elong.size:
        baseline = float(np.mean(elong))
        sd = float(np.std(elong))
        threshold = baseline - 2.0 * sd
        hyper = float(np.mean(elong < threshold))
    else:
        hyper = 0.0
    coiling_score = float(np.mean(straight < coiling_straightness_threshold))
    fission_flag = any(f.n_components > 1 for f in frames)

    return VideoTemporalMetrics(
        video_name=video_name,
        n_frames=n,
        mean_speed_px_per_s=float(np.mean(speeds_px_per_s)),
        total_path_px=total_path,
        net_displacement_px=net,
        directionality_index=float(directionality),
        mean_turn_rate_rad_per_s=mean_turn,
        immobility_index=immobility_index,
        longest_immobility_bout_s=longest_bout_s,
        mean_abs_dcurvature=mean_dcurv,
        mean_abs_dstraightness=mean_dstraight,
        mean_abs_darea=mean_darea,
        mean_abs_daspect=mean_daspect,
        body_wave_frequency_hz=body_wave_freq,
        body_wave_propagation_score=body_wave_prop,
        thigmotaxis_score=thigmotaxis_score,
        mean_radial_position=mean_r,
        radial_position_std=std_r,
        mean_angular_velocity_rad_per_s=mean_omega,
        motility_score=motility_score,
        hypercontraction_index=hyper,
        coiling_score=coiling_score,
        fission_event_flag=fission_flag,
    )


# ---------------------------------------------------------------------------
# Eigenposture decomposition
# ---------------------------------------------------------------------------
def eigenposture_decomposition(
    curvature_matrix: np.ndarray,
    n_components: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on the per-frame curvature profile matrix.

    Parameters
    ----------
    curvature_matrix:
        Array of shape ``(n_frames, n_skeleton_nodes)``. Each row is a
        single-frame curvature profile produced by
        :func:`skeleton_features`.
    n_components:
        Number of eigenpostures to keep. Stephens et al., 2008
        (DOI: 10.1371/journal.pcbi.1000028) reported that 4 to 6
        components capture more than 95% of the postural variance for
        nematode locomotion, and we adopt 6 by default for flatworms.

    Returns
    -------
    tuple of arrays ``(projections, components, explained_variance)``
        ``projections`` is shape ``(n_frames, n_components)``,
        ``components`` is shape ``(n_components, n_skeleton_nodes)``,
        ``explained_variance`` is shape ``(n_components,)``.
    """
    if curvature_matrix.size == 0:
        return (
            np.zeros((0, n_components)),
            np.zeros((n_components, 0)),
            np.zeros(n_components),
        )
    from sklearn.decomposition import PCA  # local import to avoid a heavy global cost

    k = int(min(n_components, curvature_matrix.shape[0], curvature_matrix.shape[1]))
    if k <= 0:
        return (
            np.zeros((0, n_components)),
            np.zeros((n_components, 0)),
            np.zeros(n_components),
        )
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    projections = pca.fit_transform(curvature_matrix)
    components = pca.components_
    explained = pca.explained_variance_ratio_
    return projections, components, explained


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_per_frame_csv(
    frames: Sequence[FrameMetrics],
    output_path: str,
    extra_columns: Optional[Dict[str, Sequence[object]]] = None,
) -> None:
    """Write per-frame metrics to ``output_path`` with ``sep=';'``."""
    import pandas as pd  # local import

    if not frames:
        pd.DataFrame().to_csv(
            output_path, sep=";", index=False, encoding="latin1",
        )
        return
    rows = []
    for f in frames:
        row = {
            "image_name": f.image_name,
            "second_value": f.second,
            "n_components": f.n_components,
            "area_px": f.area_px,
            "perimeter_px": f.perimeter_px,
            "skeleton_length_px": f.skeleton_length_px,
            "width_mean_px": f.width_mean_px,
            "width_std_px": f.width_std_px,
            "aspect_ratio": f.aspect_ratio,
            "elongation_ratio": f.elongation_ratio,
            "circularity": f.circularity,
            "solidity": f.solidity,
            "convexity": f.convexity,
            "curvature_mean_abs": f.curvature_mean_abs,
            "curvature_max": f.curvature_max,
            "curvature_std": f.curvature_std,
            "inflection_count": f.inflection_count,
            "straightness_index": f.straightness_index,
            "self_overlap_area_px": f.self_overlap_area_px,
            "centroid_x": f.centroid_x,
            "centroid_y": f.centroid_y,
            "centroid_r": f.centroid_r,
            "centroid_theta": f.centroid_theta,
            "distance_to_wall": f.distance_to_wall,
            "wall_proximity_flag": f.wall_proximity_flag,
        }
        for k, value in enumerate(f.width_profile_px):
            row[f"width_node_{k:02d}"] = value
        for k, value in enumerate(f.curvature_profile):
            row[f"curvature_node_{k:02d}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    if extra_columns:
        for name, seq in extra_columns.items():
            df[name] = list(seq)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep=";", index=False, encoding="latin1")


def write_per_video_csv(
    rows: Sequence[VideoTemporalMetrics],
    output_path: str,
) -> None:
    """Write the per-video aggregates to ``output_path`` with ``sep=';'``."""
    import pandas as pd

    if not rows:
        pd.DataFrame().to_csv(
            output_path, sep=";", index=False, encoding="latin1",
        )
        return
    serialised = []
    for row in rows:
        serialised.append({
            "video_name": row.video_name,
            "n_frames": row.n_frames,
            "mean_speed_px_per_s": row.mean_speed_px_per_s,
            "total_path_px": row.total_path_px,
            "net_displacement_px": row.net_displacement_px,
            "directionality_index": row.directionality_index,
            "mean_turn_rate_rad_per_s": row.mean_turn_rate_rad_per_s,
            "immobility_index": row.immobility_index,
            "longest_immobility_bout_s": row.longest_immobility_bout_s,
            "mean_abs_dcurvature": row.mean_abs_dcurvature,
            "mean_abs_dstraightness": row.mean_abs_dstraightness,
            "mean_abs_darea": row.mean_abs_darea,
            "mean_abs_daspect": row.mean_abs_daspect,
            "body_wave_frequency_hz": row.body_wave_frequency_hz,
            "body_wave_propagation_score": row.body_wave_propagation_score,
            "thigmotaxis_score": row.thigmotaxis_score,
            "mean_radial_position": row.mean_radial_position,
            "radial_position_std": row.radial_position_std,
            "mean_angular_velocity_rad_per_s": row.mean_angular_velocity_rad_per_s,
            "motility_score": row.motility_score,
            "hypercontraction_index": row.hypercontraction_index,
            "coiling_score": row.coiling_score,
            "fission_event_flag": row.fission_event_flag,
        })
    df = pd.DataFrame(serialised)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep=";", index=False, encoding="latin1")


__all__ = [
    "FrameMetrics",
    "VideoTemporalMetrics",
    "WellGeometry",
    "compute_frame_metrics",
    "count_components",
    "detect_well_circle",
    "eigenposture_decomposition",
    "largest_component",
    "read_mask",
    "self_overlap_area",
    "shape_descriptors",
    "skeleton_features",
    "temporal_metrics",
    "write_per_frame_csv",
    "write_per_video_csv",
    "RANDOM_STATE",
]
