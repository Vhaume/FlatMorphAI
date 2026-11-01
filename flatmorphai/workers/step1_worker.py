"""
Worker for Step 1: convert a directory of videos into per-second frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..base_worker import BaseWorker
from ..notebook_api.video_frames import (
    count_videos_in_directory,
    process_all_videos_in_directory,
)


@dataclass
class Step1Params:
    input_directory: str
    output_directory: str
    # Sampling window expressed in seconds. ``end_second`` is ``None``
    # to mean "floor(duration / 1) s", i.e. the last full second of the
    # clip. ``step_seconds`` matches the notebook convention of one
    # frame per second by default.
    start_second: float = 0.0
    end_second: Optional[float] = None
    step_seconds: float = 1.0
    # Videos decoded in parallel. 0 lets video_frames pick from the core
    # count (capped at 4); 1 forces the sequential path.
    max_workers: int = 0


class Step1Worker(BaseWorker):
    """Extracts frames from every video under ``input_directory``."""

    def __init__(self, params: Step1Params) -> None:
        super().__init__()
        self.params = params
        self._total_videos = 0
        self._done_videos = 0
        self._current_frames = 0
        self._current_total_frames = 0

    def _on_video_progress(self, done_videos: int, total_videos: int) -> None:
        self._done_videos = done_videos
        self._total_videos = total_videos
        self._update_progress()

    def _on_frame_progress(self, done_frames: int, total_frames: int, video_name: str) -> None:
        self._current_frames = done_frames
        self._current_total_frames = total_frames
        self._update_progress()
        if total_frames and done_frames == total_frames:
            self.log(f"Finished {video_name} ({done_frames} frames)")

    def _update_progress(self) -> None:
        if self._total_videos <= 0:
            return
        fractional_video = 0.0
        if self._current_total_frames > 0:
            fractional_video = self._current_frames / self._current_total_frames
        completed = (self._done_videos + fractional_video) / max(1, self._total_videos)
        self.progress(int(completed * 100))

    def run(self) -> None:
        params = self.params
        if not os.path.isdir(params.input_directory):
            self.error_signal.emit(
                f"Input directory does not exist: {params.input_directory}",
            )
            return

        total_videos = count_videos_in_directory(params.input_directory)
        if total_videos == 0:
            self.error_signal.emit(
                f"No .mp4/.avi/.mov files under {params.input_directory}",
            )
            return
        self._total_videos = total_videos

        self.log(
            f"Starting Step 1: {total_videos} video(s) -> {params.output_directory}"
        )
        total_frames_written = process_all_videos_in_directory(
            input_directory=params.input_directory,
            output_directory=params.output_directory,
            start_second=params.start_second,
            end_second=params.end_second,
            step_seconds=params.step_seconds,
            log=self.log,
            progress_cb=self._on_video_progress,
            per_frame_cb=self._on_frame_progress,
            is_interrupted=self.is_interrupted,
        )
        self.log(
            f"Sampling window: start={params.start_second}s, "
            f"end={params.end_second if params.end_second is not None else 'auto'}, "
            f"step={params.step_seconds}s"
        )
        self.progress(100)
        self.log(f"Step 1 done: {total_frames_written} frame(s) written")
        self.result_signal.emit(
            {
                "total_videos": self._total_videos,
                "total_frames_written": total_frames_written,
                "output_directory": params.output_directory,
            }
        )
