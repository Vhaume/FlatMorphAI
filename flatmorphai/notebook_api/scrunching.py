"""
Scrunching versus gliding detection from a sequence of binary masks.

Background and method
---------------------
Freshwater planarians of the genera Schmidtea and Dugesia normally
glide forward through ciliary propulsion on their ventral side. Under
noxious stimuli (low pH, high osmolarity, near-UV light, mechanical
disturbance, certain chemicals) they switch to a musculature-driven
gait called "scrunching" that consists of periodic axial elongations
and contractions of the body, with the head temporarily anchored to
the substrate. Cochet-Escartin, Mickolajczyk and Collins (2015)
formalised the quantitative discrimination between gliding and
scrunching using four parameters of the body-length time series
``L(t)`` recovered from the worm mask:

  1. Oscillation frequency ``f`` (number of elongation/contraction
     cycles per second), typically 0.2 to 0.8 Hz across species.
  2. Maximum elongation ``A = (L_max - L_min) / L_max`` measured
     within a sliding window.
  3. Relative speed ``A * f`` (body lengths per second).
  4. Asymmetry, computed as the fraction of frames where the worm is
     elongating (``dL/dt > 0``).

Scrunching is identified when the dominant frequency of ``L(t)`` lies
in [0.2, 0.8] Hz with elongation amplitude above ~0.20 and low net
forward displacement (the head is anchored). Gliding is identified by
near-constant body length with a directional centroid drift. The
2022 review by Reho, Lelievre and Cadiou (DOI:
10.3389/fnmol.2022.935918) confirms this as the standard
quantitative readout for planarian nociception assays.

This module implements the algorithm with a per-frame sliding window
so the pipeline can label every frame with its current behavioural
state ("gliding", "scrunching", "peristalsis", "immobile", or
"mixed") and a continuous scrunching intensity in the same
sampling-rate as the input masks.

References
----------
- Cochet-Escartin, O., Mickolajczyk, K. J. and Collins, E. M.-S.,
  "Scrunching: a novel escape gait in planarians", Physical Biology,
  2015, DOI: 10.1088/1478-3975/12/5/056010.
- Sabry, Z. et al., "Planarian Scrunching as a Quantitative
  Behavioral Readout for Noxious Stimuli Sensing", Journal of
  Visualized Experiments, 161: e61549, 2020,
  DOI: 10.3791/61549.
- Reho, G., Lelievre, V. and Cadiou, H., "Planarian nociception:
  Lessons from a scrunching flatworm", Frontiers in Molecular
  Neuroscience, 2022, DOI: 10.3389/fnmol.2022.935918 (review).
- Stephens, G. J., et al., "Dimensionality and dynamics in the
  behavior of C. elegans", PLOS Computational Biology, 2008,
  DOI: 10.1371/journal.pcbi.1000028 (sliding-window posture
  decomposition that inspired the windowed FFT approach used here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility constant kept consistent with the rest of the project.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Default scrunching frequency band, in hertz, taken verbatim from
# Cochet-Escartin et al., 2015, DOI: 10.1088/1478-3975/12/5/056010.
SCRUNCHING_FREQ_BAND_HZ: Tuple[float, float] = (0.2, 0.8)

# Default amplitude thresholds. The 0.20 cut for scrunching is the
# value reported in Cochet-Escartin et al., 2015 (figure 4); the 0.05
# cut for gliding is the upper bound on the residual length variation
# observed during pure ciliary propulsion in the same paper.
SCRUNCHING_AMPLITUDE_THRESHOLD: float = 0.20
GLIDING_AMPLITUDE_THRESHOLD: float = 0.05

# Defaults for the sliding-window analysis. Five seconds covers two
# to four full scrunching cycles at the typical 0.2 to 0.8 Hz, which
# is the recommended minimum for a stable FFT magnitude estimate
# (Bracewell, R. N., "The Fourier Transform and Its Applications",
# 3rd ed., McGraw-Hill, 1999, ISBN 978-0-07-303938-1, chapter 4).
DEFAULT_WINDOW_SECONDS: float = 5.0

# A frame whose centroid moves less than 1 pixel/s in the surrounding
# window is treated as immobile. This is the threshold used by
# DanioVision and EthoVision XT for non-active-state detection
# (Noldus user manuals, Wageningen, 2023, no DOI).
IMMOBILE_SPEED_PX_PER_S: float = 1.0

# Directionality cut-off for "gliding": at least 70 percent of the
# total path is forward progress. Validated in Cochet-Escartin et al.,
# 2015, where ciliary gliding produced near-monotonic centroid tracks.
GLIDING_DIRECTIONALITY: float = 0.7


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FrameScrunchingState:
    """Per-frame behavioural label and intensity."""

    image_name: str
    second: float

    # Raw measurements (on the frame itself)
    body_length_px: float = 0.0
    area_px: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    # First derivative of the body length, in pixels per second.
    body_length_velocity_px_per_s: float = 0.0

    # Window-level features used to classify the state.
    window_amplitude: float = 0.0
    window_dominant_frequency_hz: float = 0.0
    window_in_band_power_fraction: float = 0.0
    window_relative_speed: float = 0.0
    window_asymmetry: float = 0.0
    window_centroid_speed_px_per_s: float = 0.0
    window_directionality: float = 0.0

    # Final classification.
    state: str = "unknown"
    scrunching_intensity: float = 0.0


@dataclass
class VideoScrunchingSummary:
    """Per-video aggregates used in the metrics CSV."""

    video_name: str
    n_frames: int
    step_seconds: float

    fraction_gliding: float = 0.0
    fraction_scrunching: float = 0.0
    fraction_peristalsis: float = 0.0
    fraction_immobile: float = 0.0
    fraction_mixed: float = 0.0
    fraction_unknown: float = 0.0

    mean_body_length_px: float = 0.0
    body_length_cv: float = 0.0
    mean_window_amplitude: float = 0.0
    mean_dominant_frequency_hz: float = 0.0
    mean_relative_speed: float = 0.0
    mean_scrunching_intensity: float = 0.0
    n_scrunching_cycles: float = 0.0
    n_state_transitions: int = 0
    longest_scrunching_bout_s: float = 0.0


# ---------------------------------------------------------------------------
# Sampling-rate sanity
# ---------------------------------------------------------------------------
def recommend_step_seconds(
    freq_band: Tuple[float, float] = SCRUNCHING_FREQ_BAND_HZ,
    safety_factor: float = 2.5,
) -> float:
    """Return the largest step that still resolves ``freq_band``.

    Per Shannon's sampling theorem (Shannon, C. E., "Communication in
    the presence of noise", Proc. IRE 37, 10-21, 1949,
    DOI: 10.1109/JRPROC.1949.232969), the step must be smaller than
    ``1 / (2 * f_max)``. The ``safety_factor`` gives extra margin for
    spectral leakage and noise, so the practical recommendation is
    ``1 / (safety_factor * f_max)``. With the default 0.8 Hz upper
    band edge and a safety factor of 2.5, the recommended step is
    0.5 s.
    """
    return 1.0 / (safety_factor * max(freq_band[1], 1e-9))


def step_seconds_warning(
    step_seconds: float,
    freq_band: Tuple[float, float] = SCRUNCHING_FREQ_BAND_HZ,
) -> Optional[str]:
    """Return a warning string when the step is too coarse, else ``None``."""
    nyquist = 1.0 / (2.0 * max(freq_band[1], 1e-9))
    if step_seconds > nyquist:
        return (
            f"step_seconds={step_seconds:.3f}s is above the Nyquist "
            f"limit {nyquist:.3f}s for the {freq_band[0]:.1f}-{freq_band[1]:.1f} Hz "
            "scrunching band; oscillations will alias. Use Pipeline D's "
            "regeneration option to rebuild masks at a finer step "
            f"(recommended <= {recommend_step_seconds(freq_band):.3f}s)."
        )
    if step_seconds > recommend_step_seconds(freq_band):
        return (
            f"step_seconds={step_seconds:.3f}s is above the recommended "
            f"safety margin for the {freq_band[0]:.1f}-{freq_band[1]:.1f} Hz "
            "scrunching band; the FFT estimate of dominant frequency may "
            "be biased."
        )
    return None


# ---------------------------------------------------------------------------
# Body-length series construction
# ---------------------------------------------------------------------------
def smooth_body_length(
    body_length: Sequence[float],
    window_points: int = 3,
) -> np.ndarray:
    """Median-smooth ``body_length`` with a small odd window.

    Median smoothing is used instead of a Savitzky-Golay filter
    because it preserves the sharp transitions between elongation and
    contraction phases without introducing ringing
    (Tukey, J. W., "Exploratory Data Analysis", Addison-Wesley, 1977,
    ISBN 0-201-07616-0, chapter 7).
    """
    arr = np.asarray(body_length, dtype=np.float64)
    if window_points <= 1 or arr.size < 3:
        return arr.copy()
    if window_points % 2 == 0:
        window_points += 1
    half = window_points // 2
    out = arr.copy()
    for i in range(arr.size):
        s = max(0, i - half)
        e = min(arr.size, i + half + 1)
        out[i] = float(np.median(arr[s:e]))
    return out


# ---------------------------------------------------------------------------
# Sliding-window FFT
# ---------------------------------------------------------------------------
def _window_features(
    body_length: np.ndarray,
    centroid: np.ndarray,
    step_seconds: float,
    freq_band: Tuple[float, float],
) -> Dict[str, float]:
    """Compute amplitude, dominant frequency, asymmetry, etc. on a window."""
    n = body_length.size
    if n < 4:
        return {
            "amplitude": 0.0,
            "dominant_frequency_hz": 0.0,
            "in_band_power_fraction": 0.0,
            "relative_speed": 0.0,
            "asymmetry": 0.0,
            "centroid_speed_px_per_s": 0.0,
            "directionality": 0.0,
        }
    # Detrend so a slow drift of the body length (for example the
    # animal slowly stretching while gliding) does not create a
    # spurious DC-adjacent peak in the FFT.
    t_axis = np.arange(n, dtype=np.float64) * step_seconds
    a, b = np.polyfit(t_axis, body_length, 1)
    detrended = body_length - (a * t_axis + b)
    # FFT magnitudes; rfft drops the negative-frequency mirror.
    spectrum = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, d=step_seconds)
    magnitudes = np.abs(spectrum)
    # Skip the DC bin to find the dominant frequency.
    if magnitudes.size > 1:
        non_dc_total = float(np.sum(magnitudes[1:]) + 1e-9)
        in_band = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        if np.any(in_band):
            band_mags = magnitudes[in_band]
            band_freqs = freqs[in_band]
            peak_idx = int(np.argmax(band_mags))
            dom_freq = float(band_freqs[peak_idx])
            dom_power = float(band_mags[peak_idx])
        else:
            dom_freq = 0.0
            dom_power = 0.0
        in_band_power_fraction = (
            float(np.sum(magnitudes[in_band])) / non_dc_total
            if np.any(in_band) else 0.0
        )
    else:
        dom_freq = 0.0
        dom_power = 0.0
        in_band_power_fraction = 0.0

    # Amplitude is computed on the raw body length so the value is
    # always interpretable as an elongation fraction. The Collins lab
    # convention is (max - min) / max within the window.
    L_max = float(np.max(body_length))
    L_min = float(np.min(body_length))
    amplitude = (L_max - L_min) / L_max if L_max > 0 else 0.0
    relative_speed = amplitude * dom_freq

    # Asymmetry: the fraction of frames in the window where the body
    # was elongating, i.e. dL/dt > 0. A symmetric scrunch has a value
    # close to 0.5; head-anchored scrunching biases the value because
    # the elongation phase is slower than the snap-back contraction.
    deltas = np.diff(body_length)
    if deltas.size > 0:
        asymmetry = float(np.mean(deltas > 0))
    else:
        asymmetry = 0.5

    # Centroid kinematics in the same window.
    if centroid.shape[0] >= 2:
        d = np.diff(centroid, axis=0)
        seg = np.linalg.norm(d, axis=1)
        path = float(np.sum(seg))
        net = float(np.linalg.norm(centroid[-1] - centroid[0]))
        window_seconds = (n - 1) * step_seconds
        speed = path / window_seconds if window_seconds > 0 else 0.0
        directionality = (net / path) if path > 0 else 0.0
    else:
        speed = 0.0
        directionality = 0.0

    return {
        "amplitude": amplitude,
        "dominant_frequency_hz": dom_freq,
        "in_band_power_fraction": in_band_power_fraction,
        "relative_speed": relative_speed,
        "asymmetry": asymmetry,
        "centroid_speed_px_per_s": speed,
        "directionality": directionality,
    }


# ---------------------------------------------------------------------------
# Per-frame state classification
# ---------------------------------------------------------------------------
def classify_state(
    amplitude: float,
    dom_freq: float,
    in_band_power_fraction: float,
    centroid_speed: float,
    directionality: float,
    freq_band: Tuple[float, float] = SCRUNCHING_FREQ_BAND_HZ,
    scrunching_amplitude_threshold: float = SCRUNCHING_AMPLITUDE_THRESHOLD,
    gliding_amplitude_threshold: float = GLIDING_AMPLITUDE_THRESHOLD,
    immobile_speed_threshold: float = IMMOBILE_SPEED_PX_PER_S,
    gliding_directionality_threshold: float = GLIDING_DIRECTIONALITY,
) -> str:
    """Return one of "scrunching" / "gliding" / "peristalsis" / "immobile" / "mixed"."""
    # Immobile dominates when both the body and the centroid are
    # nearly still. This is the EthoVision / DanioVision rule: low
    # speed and low body deformation.
    if (
        centroid_speed < immobile_speed_threshold
        and amplitude < gliding_amplitude_threshold
    ):
        return "immobile"
    # Scrunching: large oscillation amplitude in the right frequency
    # band, low net forward displacement (head-anchored). Cochet-
    # Escartin et al., 2015 found that scrunching has directionality
    # below 0.5 because the centroid pivots around the anchored head.
    in_band = freq_band[0] <= dom_freq <= freq_band[1]
    if (
        amplitude >= scrunching_amplitude_threshold
        and in_band
        and directionality < 0.5
    ):
        return "scrunching"
    # Peristalsis: body-length oscillation in a higher frequency
    # range than scrunching (typically > freq_band[1]) with sustained
    # forward displacement. Reported in Cochet-Escartin et al., 2015
    # and reviewed in Reho et al., 2022 (DOI: 10.3389/fnmol.2022.935918).
    if (
        amplitude >= scrunching_amplitude_threshold
        and dom_freq > freq_band[1]
        and directionality >= 0.5
    ):
        return "peristalsis"
    # Gliding: small oscillation, near-monotonic centroid drift.
    if (
        amplitude < gliding_amplitude_threshold
        and directionality >= gliding_directionality_threshold
        and centroid_speed >= immobile_speed_threshold
    ):
        return "gliding"
    return "mixed"


def scrunching_intensity(
    amplitude: float,
    dom_freq: float,
    in_band_power_fraction: float,
    state: str,
    freq_band: Tuple[float, float] = SCRUNCHING_FREQ_BAND_HZ,
) -> float:
    """Continuous intensity for the per-frame scrunching curve.

    Intensity is the dimensionless product
    ``amplitude * dominant_frequency * in_band_power_fraction``,
    clipped to the scrunching state. The product is sometimes called
    the "scrunching power" by Sabry et al., 2020 (JoVE 61549,
    DOI: 10.3791/61549) and varies between 0 (no scrunching) and a
    typical maximum of ~0.5 for vigorous mid-body scrunching at
    0.6-0.8 Hz.
    """
    if state != "scrunching":
        return 0.0
    return float(amplitude * dom_freq * in_band_power_fraction)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def analyse_video(
    image_names: Sequence[str],
    seconds: Sequence[float],
    body_lengths: Sequence[float],
    centroids: Sequence[Tuple[float, float]],
    areas: Sequence[float],
    step_seconds: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    freq_band: Tuple[float, float] = SCRUNCHING_FREQ_BAND_HZ,
    scrunching_amplitude_threshold: float = SCRUNCHING_AMPLITUDE_THRESHOLD,
    gliding_amplitude_threshold: float = GLIDING_AMPLITUDE_THRESHOLD,
    immobile_speed_threshold: float = IMMOBILE_SPEED_PX_PER_S,
    gliding_directionality_threshold: float = GLIDING_DIRECTIONALITY,
    smoothing_window_points: int = 3,
) -> List[FrameScrunchingState]:
    """Label every frame with its behavioural state.

    The function is a single pass over the time series. For each frame
    it constructs a centred window of width ``window_seconds`` (in
    frames: ``window_points = round(window_seconds / step_seconds)``),
    computes :func:`_window_features` over it and classifies the state
    with :func:`classify_state`. The first and last frames whose
    window would fall outside the recording are still labelled but
    use a truncated window. Body-length values are median-smoothed
    upstream so single-frame segmentation glitches do not pollute the
    FFT.
    """
    n = len(image_names)
    if n == 0:
        return []
    body_arr = smooth_body_length(body_lengths, window_points=smoothing_window_points)
    centroid_arr = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    seconds_arr = np.asarray(seconds, dtype=np.float64)
    if step_seconds <= 0:
        step_seconds = 1.0
    window_points = max(4, int(round(window_seconds / step_seconds)))
    half = window_points // 2

    # First derivative dL/dt in pixel per second (central differences
    # with forward/backward at the boundaries).
    if n >= 2:
        velocity = np.gradient(body_arr, step_seconds)
    else:
        velocity = np.zeros(n)

    out: List[FrameScrunchingState] = []
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        L_win = body_arr[s:e]
        c_win = centroid_arr[s:e]
        feats = _window_features(
            L_win, c_win, step_seconds, freq_band,
        )
        state = classify_state(
            amplitude=feats["amplitude"],
            dom_freq=feats["dominant_frequency_hz"],
            in_band_power_fraction=feats["in_band_power_fraction"],
            centroid_speed=feats["centroid_speed_px_per_s"],
            directionality=feats["directionality"],
            freq_band=freq_band,
            scrunching_amplitude_threshold=scrunching_amplitude_threshold,
            gliding_amplitude_threshold=gliding_amplitude_threshold,
            immobile_speed_threshold=immobile_speed_threshold,
            gliding_directionality_threshold=gliding_directionality_threshold,
        )
        intensity = scrunching_intensity(
            amplitude=feats["amplitude"],
            dom_freq=feats["dominant_frequency_hz"],
            in_band_power_fraction=feats["in_band_power_fraction"],
            state=state,
            freq_band=freq_band,
        )
        cx = float(centroid_arr[i, 0]) if i < centroid_arr.shape[0] else 0.0
        cy = float(centroid_arr[i, 1]) if i < centroid_arr.shape[0] else 0.0
        out.append(
            FrameScrunchingState(
                image_name=str(image_names[i]),
                second=float(seconds_arr[i]) if i < seconds_arr.size else 0.0,
                body_length_px=float(body_arr[i]),
                area_px=float(areas[i]) if i < len(areas) else 0.0,
                centroid_x=cx,
                centroid_y=cy,
                body_length_velocity_px_per_s=float(velocity[i]),
                window_amplitude=float(feats["amplitude"]),
                window_dominant_frequency_hz=float(
                    feats["dominant_frequency_hz"]
                ),
                window_in_band_power_fraction=float(
                    feats["in_band_power_fraction"]
                ),
                window_relative_speed=float(feats["relative_speed"]),
                window_asymmetry=float(feats["asymmetry"]),
                window_centroid_speed_px_per_s=float(
                    feats["centroid_speed_px_per_s"]
                ),
                window_directionality=float(feats["directionality"]),
                state=state,
                scrunching_intensity=intensity,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-video aggregates
# ---------------------------------------------------------------------------
def summarise_video(
    states: Sequence[FrameScrunchingState],
    video_name: str,
    step_seconds: float,
) -> VideoScrunchingSummary:
    """Aggregate a list of :class:`FrameScrunchingState` into a single row."""
    n = len(states)
    if n == 0:
        return VideoScrunchingSummary(
            video_name=video_name, n_frames=0, step_seconds=step_seconds,
        )
    counts: Dict[str, int] = {
        "gliding": 0, "scrunching": 0, "peristalsis": 0,
        "immobile": 0, "mixed": 0, "unknown": 0,
    }
    for s in states:
        counts[s.state] = counts.get(s.state, 0) + 1
    total = float(n)
    summary = VideoScrunchingSummary(
        video_name=video_name,
        n_frames=n,
        step_seconds=step_seconds,
        fraction_gliding=counts["gliding"] / total,
        fraction_scrunching=counts["scrunching"] / total,
        fraction_peristalsis=counts["peristalsis"] / total,
        fraction_immobile=counts["immobile"] / total,
        fraction_mixed=counts["mixed"] / total,
        fraction_unknown=counts["unknown"] / total,
    )
    body = np.array([s.body_length_px for s in states])
    if body.size and float(np.mean(body)) > 0:
        summary.mean_body_length_px = float(np.mean(body))
        summary.body_length_cv = float(np.std(body) / np.mean(body))
    summary.mean_window_amplitude = float(
        np.mean([s.window_amplitude for s in states])
    )
    summary.mean_dominant_frequency_hz = float(
        np.mean([s.window_dominant_frequency_hz for s in states])
    )
    summary.mean_relative_speed = float(
        np.mean([s.window_relative_speed for s in states])
    )
    intensities = np.array(
        [s.scrunching_intensity for s in states], dtype=np.float64,
    )
    summary.mean_scrunching_intensity = float(np.mean(intensities))
    # The total number of scrunching cycles across the recording is
    # the integral of the dominant frequency over the time spent in
    # the "scrunching" state, computed as a Riemann sum with the
    # sampling step. This matches the definition used by Sabry et al.,
    # 2020 (DOI: 10.3791/61549).
    cycles = 0.0
    for s in states:
        if s.state == "scrunching":
            cycles += s.window_dominant_frequency_hz * step_seconds
    summary.n_scrunching_cycles = float(cycles)
    # State transitions and longest scrunching end.
    transitions = 0
    prev = None
    longest = 0
    current = 0
    for s in states:
        if prev is not None and s.state != prev:
            transitions += 1
        prev = s.state
        if s.state == "scrunching":
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    summary.n_state_transitions = int(transitions)
    summary.longest_scrunching_bout_s = float(longest) * step_seconds
    return summary


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_per_frame_csv(
    states: Sequence[FrameScrunchingState],
    output_path: str,
    extra_columns: Optional[Dict[str, Sequence[object]]] = None,
) -> None:
    """Write the per-frame state series with ``sep=';'`` and latin1 encoding."""
    import os
    import pandas as pd

    rows = []
    for s in states:
        rows.append({
            "image_name": s.image_name,
            "second_value": s.second,
            "body_length_px": s.body_length_px,
            "area_px": s.area_px,
            "centroid_x": s.centroid_x,
            "centroid_y": s.centroid_y,
            "body_length_velocity_px_per_s": s.body_length_velocity_px_per_s,
            "window_amplitude": s.window_amplitude,
            "window_dominant_frequency_hz": s.window_dominant_frequency_hz,
            "window_in_band_power_fraction": s.window_in_band_power_fraction,
            "window_relative_speed": s.window_relative_speed,
            "window_asymmetry": s.window_asymmetry,
            "window_centroid_speed_px_per_s": s.window_centroid_speed_px_per_s,
            "window_directionality": s.window_directionality,
            "state": s.state,
            "scrunching_intensity": s.scrunching_intensity,
        })
    df = pd.DataFrame(rows)
    if extra_columns:
        for name, seq in extra_columns.items():
            df[name] = list(seq)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep=";", index=False, encoding="latin1")


def write_per_video_csv(
    rows: Sequence[VideoScrunchingSummary],
    output_path: str,
) -> None:
    """Write the per-video scrunching aggregates with ``sep=';'``."""
    import os
    import pandas as pd

    serialised = []
    for row in rows:
        serialised.append({
            "video_name": row.video_name,
            "n_frames": row.n_frames,
            "step_seconds": row.step_seconds,
            "fraction_gliding": row.fraction_gliding,
            "fraction_scrunching": row.fraction_scrunching,
            "fraction_peristalsis": row.fraction_peristalsis,
            "fraction_immobile": row.fraction_immobile,
            "fraction_mixed": row.fraction_mixed,
            "fraction_unknown": row.fraction_unknown,
            "mean_body_length_px": row.mean_body_length_px,
            "body_length_cv": row.body_length_cv,
            "mean_window_amplitude": row.mean_window_amplitude,
            "mean_dominant_frequency_hz": row.mean_dominant_frequency_hz,
            "mean_relative_speed": row.mean_relative_speed,
            "mean_scrunching_intensity": row.mean_scrunching_intensity,
            "n_scrunching_cycles": row.n_scrunching_cycles,
            "n_state_transitions": row.n_state_transitions,
            "longest_scrunching_bout_s": row.longest_scrunching_bout_s,
        })
    import pandas as pd
    df = pd.DataFrame(serialised)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep=";", index=False, encoding="latin1")


__all__ = [
    "FrameScrunchingState",
    "VideoScrunchingSummary",
    "SCRUNCHING_FREQ_BAND_HZ",
    "SCRUNCHING_AMPLITUDE_THRESHOLD",
    "GLIDING_AMPLITUDE_THRESHOLD",
    "DEFAULT_WINDOW_SECONDS",
    "RANDOM_STATE",
    "analyse_video",
    "classify_state",
    "recommend_step_seconds",
    "scrunching_intensity",
    "smooth_body_length",
    "step_seconds_warning",
    "summarise_video",
    "write_per_frame_csv",
    "write_per_video_csv",
]
