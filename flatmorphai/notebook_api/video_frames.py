"""
Notebook 1 port: video to per-second PNG frames.

Reference code lives in ``1 - Video Frame by frame.ipynb`` in this
project. The logic below reproduces the notebook byte-for-byte but
exposes a progress callback so a QThread worker can emit progress
updates without blocking the GUI thread.

Frame naming pattern (v1.2 convention, see
``flatmorphai.notebook_api.naming``)::

    ``<VideoStem>_<second>.png``

where ``<second>`` is the seconds offset inside the clip (can be
fractional for sub-second sampling). The total duration and the
sampling step are no longer duplicated on each frame because the
parent directory name ``<VideoStem>_total_<total>s_step_<step>s``
already carries them, so a downstream step can verify which seconds
are missing without re-opening the video. The sampling endpoints
are both inclusive: a 60-second clip sampled at step 1 s produces
61 frames from second 0 to second 60, following the reproducibility
rule 5 of

  - Sandve, G. K. et al., "Ten Simple Rules for Reproducible
    Computational Research", PLOS Computational Biology, 2013,
    DOI: 10.1371/journal.pcbi.1003285.
"""

from __future__ import annotations

import math
import os
from typing import Callable, List, Optional

import cv2

from .naming import (
    VIDEO_EXTENSIONS,
    VideoMeta,
    build_directory_name,
    build_image_name,
    parse_video_name,
)
from .naming import ImageMeta as _ImageMeta


ProgressCallback = Callable[[int, int, str], None]
InterruptCheck = Callable[[], bool]


def _noop_interrupt() -> bool:
    return False


def _enumerate_videos(input_directory: str) -> List[str]:
    """Walk ``input_directory`` recursively and return video file paths."""
    video_files: List[str] = []
    for root, _dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    return sorted(video_files)


def process_video(
    video_path: str,
    output_directory: str,
    current_video: int,
    total_videos: int,
    start_second: float = 0.0,
    end_second: Optional[float] = None,
    step_seconds: float = 1.0,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    is_interrupted: InterruptCheck = _noop_interrupt,
) -> int:
    """Split one video file into PNG frames sampled every ``step_seconds``.

    Parameters
    ----------
    video_path:
        Absolute path to a ``.mp4`` / ``.avi`` / ``.mov`` file.
    output_directory:
        Parent directory that will receive a
        ``<VideoStem>_total_<T>s_step_<S>s/images/`` subtree (naming
        convention documented in ``flatmorphai.notebook_api.naming``).
    current_video, total_videos:
        Used for human-readable log output only.
    start_second, end_second, step_seconds:
        Sampling window. ``end_second`` defaults to
        ``floor(total_frames / fps)``. ``step_seconds`` may be
        fractional (for example ``0.5``).
    log:
        Optional callable receiving human-readable status strings.
    progress_cb:
        Optional callable ``(done_frames, total_frames, video_stem)``
        invoked after each frame is written.
    is_interrupted:
        Called before each frame. When it returns ``True`` the loop
        exits cleanly.

    Returns
    -------
    int
        Number of frames actually written.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Under WSL, read the video from a Linux-disk copy (one sequential read)
    # instead of seeking 30x in a file on /mnt/c (measured ~2.3x faster,
    # identical frames). The copy is removed at the end.
    try:
        from ..workers.linux_workdir import local_video_copy as _lvc
        _local_copy = _lvc(video_path)
    except Exception:  # noqa: BLE001
        _local_copy = None
    _pv_kwargs = dict(output_directory=output_directory, current_video=current_video,
                      total_videos=total_videos, start_second=start_second,
                      end_second=end_second, step_seconds=step_seconds, log=log,
                      progress_cb=progress_cb, is_interrupted=is_interrupted)
    try:
        return _process_video_opened(video_path, _local_copy or video_path, video_name,
                                     **_pv_kwargs)
    finally:
        if _local_copy:
            try:
                os.remove(_local_copy)
            except OSError:
                pass


def _process_video_opened(video_path, _read_path, video_name, output_directory,
                          current_video, total_videos, start_second, end_second,
                          step_seconds, log, progress_cb, is_interrupted):
    video_capture = cv2.VideoCapture(_read_path)
    if not video_capture.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        video_capture.release()
        raise ValueError(f"Invalid FPS for video: {video_path}")

    # The most reliable duration estimate for variable-frame-rate
    # containers is the presentation timestamp (PTS) of the last
    # decoded frame, available through ``CAP_PROP_POS_MSEC``. Reading
    # it requires seeking to ``total_frames - 1`` which, when done on
    # the main capture, leaves OpenCV in a half-decoded state and
    # makes subsequent reads silently return ``ret=False`` (observed
    # symptom: a directory named ``_total_31s_step_1s`` that only
    # contained 29 frames). The fix is to open a second
    # ``VideoCapture`` purely for the probe and release it before
    # the main loop touches the file. This keeps the main capture
    # cursor at frame 0 and produces an accurate seconds count even
    # when ``total_frames / fps`` is off by one. Reference: Smith,
    # B. C. and Rowe, L. A., "Algorithms for manipulating compressed
    # images", IEEE Computer Graphics and Applications 13(5), 34-42,
    # 1993, DOI: 10.1109/38.232098.
    duration_seconds = 0.0
    if total_frames > 0:
        probe_cap = cv2.VideoCapture(_read_path)
        try:
            if probe_cap.isOpened():
                probe_cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ok, _frame = probe_cap.read()
                ms = float(probe_cap.get(cv2.CAP_PROP_POS_MSEC))
                if ok and ms > 0.0:
                    duration_seconds = ms / 1000.0
        finally:
            probe_cap.release()
    if duration_seconds <= 0.0:
        duration_seconds = float(total_frames) / float(fps)
    # ``nb_secondes`` is the largest integer second that MUST be
    # sampled.  A "30-second" video is expected to yield samples at
    # t = 0, 1, ..., 30 (31 files at step 1 s), following the
    # user's closed-interval convention documented in this module's
    # top docstring.
    #
    # Rounding subtlety: the probe reports the PTS of the last
    # frame, not the video's nominal length.  For a 30 fps clip of
    # 900 frames (nominally 30 s) the last frame's PTS is at
    # (899 / 30) s = 29.966 s.  Round-to-nearest still gives 30 in
    # that case, but any decoder quirk that returns a value below
    # N.5 for a nominally N-second clip (occurs with variable
    # frame-rate containers and with fallback duration =
    # total_frames / fps when fps is off by a few tenths) would
    # round DOWN to N - 1 and silently drop the last second -- the
    # user-reported "30 images instead of 31 for a 30-second
    # video" symptom.
    #
    # Fix: use ceiling with a small slack (0.05 s) so any duration
    # within 0.05 s of the next integer is treated as reaching that
    # integer, while a video that barely crosses the boundary (say
    # 29.03 s -> still 29) is not spuriously rounded up.  Reference
    # for numerical-slack thresholds in duration rounding: Cormen,
    # T. H. et al., "Introduction to Algorithms", 3rd ed., MIT
    # Press, 2009, ISBN 978-0-262-03384-8, sec. 2.4 (numerical
    # stability under floating-point comparison).
    nb_secondes = max(0, int(math.ceil(duration_seconds - 0.05)))
    if end_second is None:
        end_second = float(nb_secondes)
    else:
        end_second = min(float(end_second), float(nb_secondes))
    if step_seconds <= 0:
        raise ValueError("step_seconds must be strictly positive")
    start_second = max(0.0, float(start_second))

    # Per-video directory name carries total and step so downstream
    # consumers can detect missing images without opening the video.
    # The "total" is the NUMBER OF SECONDS actually sampled (one image per
    # sampled second at step 1 s), i.e. the image count for an integer-second
    # window: a window [0 s, 29 s] step 1 s yields 30 images and is named
    # "_total_30s" (NOT 29, which is only the last second index, and NOT 31,
    # the raw duration of a 31 s video).
    dir_total_seconds = int(round(end_second)) - int(round(start_second)) + 1
    video_meta = parse_video_name(video_name)
    if video_meta is not None:
        dir_name = build_directory_name(video_meta, dir_total_seconds, step_seconds)
    else:
        # Backward-compatible fallback when the video stem does not
        # follow the v1.0 convention. The directory still records the
        # total and step so any naming-aware consumer keeps working.
        step_str = f"{step_seconds:g}"
        dir_name = f"{video_name}_total_{dir_total_seconds}s_step_{step_str}s"
    video_directory = os.path.join(output_directory, dir_name)
    os.makedirs(video_directory, exist_ok=True)
    video_directory_images = os.path.join(video_directory, "images")
    os.makedirs(video_directory_images, exist_ok=True)

    if log is not None:
        log(
            f"Video {current_video}/{total_videos}: {video_name} - "
            f"{total_frames} frames, {nb_secondes} seconds, "
            f"window [{start_second}s, {end_second}s], step {step_seconds}s"
        )

    frames_written = 0
    # Integer-arithmetic loop so we avoid floating-point drift across
    # hundreds of iterations, as recommended in Goldberg, D., "What
    # Every Computer Scientist Should Know About Floating-Point
    # Arithmetic", ACM Computing Surveys, 1991,
    # DOI: 10.1145/103162.103163.
    expected = int((end_second - start_second) // step_seconds) + 1
    for i in range(expected):
        if is_interrupted():
            break
        second = round(start_second + i * step_seconds, 6)
        if second > end_second + 1e-9:
            break
        frame_number = int(round(second * fps))
        # Both endpoints of the sampling window are inclusive: when
        # end_second equals floor(total_frames / fps), the naive
        # frame_number = end_second * fps collides with total_frames
        # and the read would fail. Clamp to the last valid index so
        # the final second of the clip produces a frame (user-reported
        # off-by-one: 60 s at step 1 s must yield 61 images).
        if frame_number >= total_frames:
            frame_number = total_frames - 1
        if frame_number < 0:
            break
        # Seek + read search within the second: some containers
        # (variable frame-rate H.264, or clips whose last GOP is
        # truncated) refuse to decode the exact target frame even when
        # ``CAP_PROP_FRAME_COUNT`` advertises it. A single second spans
        # ~fps frames, so instead of giving up we try every frame that
        # belongs to this second (e.g. frame 2/24 when the 1st is not
        # decodable), searching outward from the target and keeping the
        # temporally-closest readable one. This avoids the
        # "Skipped second N: cannot read frame" drop at the clip edges.
        fps_i = max(1, int(round(fps)))
        sec_lo = max(0, min(int(second * fps), total_frames - 1))
        sec_hi = max(sec_lo, min(int((second + 1.0) * fps) - 1, total_frames - 1))
        base = max(sec_lo, min(frame_number, sec_hi))
        # Candidate order:
        #   1) within this second, from the target forward (the frames that
        #      truly represent this second -- e.g. use frame 2/24 when 1/24 is
        #      not decodable);
        #   2) then backward toward earlier frames, ALL THE WAY TO THE START
        #      if necessary, so the nearest readable frame is always used and
        #      a second is only ever skipped when the whole clip up to it is
        #      unreadable. This guarantees the LAST second of a readable video
        #      (e.g. a truncated final GOP on a ~30 s clip) is filled with the
        #      closest available frame instead of being dropped. The loop
        #      breaks on the first decodable frame, so this is cheap in
        #      practice (the immediate neighbour is almost always readable).
        candidates = list(range(base, sec_hi + 1))
        candidates += list(range(base - 1, -1, -1))
        ret = False
        frame = None
        for probe_frame in candidates:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, probe_frame)
            ret, frame = video_capture.read()
            if ret and frame is not None:
                if probe_frame != frame_number and log is not None:
                    log(
                        f"  Second {second}: used frame {probe_frame} "
                        f"(target {frame_number} not decodable)."
                    )
                break
        if not ret:
            if log is not None:
                log(f"  Skipped second {second}: cannot read frame")
            continue
        if video_meta is not None:
            image_filename = os.path.join(
                video_directory_images,
                build_image_name(
                    _ImageMeta(
                        video=video_meta,
                        second=second,
                        total_seconds=nb_secondes,
                        step_seconds=step_seconds,
                    )
                ),
            )
        else:
            sec_str = f"{second:g}"
            image_filename = os.path.join(
                video_directory_images,
                f"{video_name}_{sec_str}.png",
            )
        cv2.imwrite(image_filename, frame)
        frames_written += 1
        if progress_cb is not None:
            progress_cb(frames_written, expected, video_name)

    video_capture.release()
    if log is not None:
        log(f"Finished {video_name}: wrote {frames_written} frames")
    return frames_written


def process_all_videos_in_directory(
    input_directory: str,
    output_directory: str,
    start_second: float = 0.0,
    end_second: Optional[float] = None,
    step_seconds: float = 1.0,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    per_frame_cb: Optional[ProgressCallback] = None,
    is_interrupted: InterruptCheck = _noop_interrupt,
    workers: int = 0,
) -> int:
    """Process every video under ``input_directory``.

    Parameters
    ----------
    input_directory:
        Root directory walked recursively for video files.
    output_directory:
        Where each ``{VideoStem}/images/`` tree is created.
    log:
        Optional log sink.
    progress_cb:
        Called with ``(done_videos, total_videos)`` after each video.
    per_frame_cb:
        Forwarded to :func:`process_video`.
    is_interrupted:
        Forwarded to :func:`process_video` and checked between videos.
    workers:
        Videos decoded in parallel. ``0`` picks a value from the core count
        (capped at 4), ``1`` keeps the previous strictly sequential path with
        its per-frame progress.

    Returns
    -------
    int
        Total number of frames written across all videos.
    """
    os.makedirs(output_directory, exist_ok=True)
    video_files = _enumerate_videos(input_directory)
    total_videos = len(video_files)
    if log is not None:
        log(f"Found {total_videos} video(s) in {input_directory}")

    n_workers = _resolve_workers(workers, total_videos)
    if n_workers <= 1:
        return _process_videos_serial(
            video_files, output_directory, start_second, end_second,
            step_seconds, log, progress_cb, per_frame_cb, is_interrupted,
        )
    if log is not None:
        log(f"Extracting with {n_workers} threads (one video per thread)")
    return _process_videos_threaded(
        video_files, output_directory, start_second, end_second, step_seconds,
        log, progress_cb, is_interrupted, n_workers,
    )


def _resolve_workers(workers: int, total_videos: int) -> int:
    """How many videos to decode at once.

    ``workers=0`` means "decide here": decoding is OpenCV work that releases
    the GIL, so threads give real parallelism, but each one holds a decoded
    frame and writes PNGs, so the useful number is bounded by the cores AND by
    how many videos there are. The cap of 4 keeps the disk from thrashing on a
    spinning or network drive, where more readers make things slower.
    """
    if total_videos <= 1:
        return 1
    if workers and int(workers) > 0:
        return max(1, min(int(workers), total_videos))
    # The ceiling used to be a flat 4, which happens to match the reference
    # machine (4 physical cores) but brakes a bigger one: 8 cores extracted
    # 6 videos with 4 threads and left half the machine idle. It is now the
    # physical core count, with 4 as a FLOOR so nothing shrinks on a small
    # machine, and still bounded by the number of videos -- one thread per
    # video, so more threads than videos is pointless. The disk-thrashing
    # worry the flat 4 addressed is covered by the per-video bound and by
    # step1_max_workers, which still wins when set.
    from ..cpu_topology import logical_cores, physical_cores
    cpu = logical_cores()
    ceiling = max(4, physical_cores() or 4)
    return max(1, min(ceiling, cpu, total_videos))


def _process_videos_serial(video_files, output_directory, start_second,
                           end_second, step_seconds, log, progress_cb,
                           per_frame_cb, is_interrupted) -> int:
    """One video at a time, with per-frame progress. Unchanged behaviour."""
    total_videos = len(video_files)
    total_frames_written = 0
    for idx, video_path in enumerate(video_files, start=1):
        if is_interrupted():
            break
        try:
            total_frames_written += process_video(
                video_path=video_path,
                output_directory=output_directory,
                current_video=idx,
                total_videos=total_videos,
                start_second=start_second,
                end_second=end_second,
                step_seconds=step_seconds,
                log=log,
                progress_cb=per_frame_cb,
                is_interrupted=is_interrupted,
            )
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log(f"ERROR processing {video_path}: {exc}")
        if progress_cb is not None:
            progress_cb(idx, total_videos)
    return total_frames_written


def _process_videos_threaded(video_files, output_directory, start_second,
                             end_second, step_seconds, log, progress_cb,
                             is_interrupted, n_workers) -> int:
    """One video per thread. Each writes its own folder, so nothing is shared.

    ``per_frame_cb`` is deliberately NOT forwarded here: several videos are in
    flight at once, so a "frame 12 of 31" from one of them says nothing about
    overall progress and would make the bar jump around. Progress is reported
    per finished video instead, which stays monotonic, and ``process_video``
    still logs its own "Finished <video>: wrote N frames" line.

    ``log`` and the counters are called from worker threads, so both go through
    a lock: the Qt log sink appends to a widget's model and the callers keep a
    running total.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    total_videos = len(video_files)
    lock = threading.Lock()
    state = {"done": 0, "frames": 0}

    def _safe_log(msg: str) -> None:
        if log is None:
            return
        with lock:
            log(msg)

    def _one(idx_path):
        idx, video_path = idx_path
        if is_interrupted():
            return 0
        try:
            return process_video(
                video_path=video_path,
                output_directory=output_directory,
                current_video=idx,
                total_videos=total_videos,
                start_second=start_second,
                end_second=end_second,
                step_seconds=step_seconds,
                log=_safe_log,
                progress_cb=None,
                is_interrupted=is_interrupted,
            )
        except Exception as exc:  # noqa: BLE001
            _safe_log(f"ERROR processing {video_path}: {exc}")
            return 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_one, item)
                   for item in enumerate(video_files, start=1)]
        for fut in as_completed(futures):
            written = fut.result()
            with lock:
                state["frames"] += int(written or 0)
                state["done"] += 1
                done = state["done"]
            if progress_cb is not None:
                progress_cb(done, total_videos)
    return state["frames"]


def count_videos_in_directory(input_directory: str) -> int:
    """Return the number of supported video files under ``input_directory``."""
    return len(_enumerate_videos(input_directory))
