"""
Pipeline worker that chains several step workers sequentially.

The worker receives a list of step parameter payloads and runs them in
order on a dedicated :class:`QThread`. Progress is reported both
per-step and cumulatively. If a step fails the pipeline stops and emits
:attr:`error_signal`.

References
----------
  - The six-step notebook workflow documented in FlatMorphAI_PySide6_Prompt_v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Signal

from ..base_worker import BaseWorker


StepRunner = Callable[[Any, Callable[[str], None], Callable[[int], None],
                        Callable[[], bool]], Dict[str, Any]]


@dataclass
class PipelineStep:
    """Describe one pipeline step.

    - ``name``: user-visible step label (``"Step 1"``, etc.)
    - ``runner``: callable that executes the step synchronously on the
      worker thread. Receives ``(params, log, progress, is_interrupted)``
      and returns a result ``dict``.
    - ``params``: opaque parameter object forwarded to ``runner``.
    """

    name: str
    runner: StepRunner
    params: Any


@dataclass
class PipelineParams:
    steps: List[PipelineStep] = field(default_factory=list)


class PipelineWorker(BaseWorker):
    """Sequentially run a list of :class:`PipelineStep` instances."""

    # Emitted at every step transition with (index, step_name).  The
    # index is 0-based; pipeline pages connect this to
    # ``timeline.set_current(...)`` so the "Current step:" label
    # updates in lock-step with the work, mirroring Nielsen J., "10
    # Usability Heuristics for User Interface Design", 1994
    # (https://www.nngroup.com/articles/ten-usability-heuristics/),
    # heuristic 1 ("visibility of system status").  When the run
    # finishes or is cancelled the worker emits index=-1 so the page
    # can revert the label.
    current_step_changed_signal = Signal(int, str)

    def __init__(self, params: PipelineParams) -> None:
        super().__init__()
        self.params = params
        self._current_step_index = 0
        self._step_count = len(params.steps)

    def _scoped_progress(self, step_index: int) -> Callable[[int], None]:
        """Return a ``progress_cb`` that maps 0-100 inside a step onto the pipeline bar."""
        def _cb(pct: int) -> None:
            if self._step_count <= 0:
                return
            span = 100.0 / self._step_count
            base = step_index * span
            value = int(base + span * max(0, min(100, pct)) / 100.0)
            self.progress(value)
        return _cb

    def run(self) -> None:
        results: Dict[str, Dict[str, Any]] = {}
        if not self.require_packages("stereotype"):
            return
        self.log(f"Pipeline starting: {self._step_count} step(s)")
        for index, step in enumerate(self.params.steps):
            if self.is_interrupted():
                self.log("Pipeline cancelled by user")
                self.current_step_changed_signal.emit(-1, "cancelled")
                break
            self._current_step_index = index
            self.log(f"--- {step.name} ({index + 1}/{self._step_count}) ---")
            # Announce the active step BEFORE running so the page's
            # "Current step: ..." label reflects the work in progress.
            self.current_step_changed_signal.emit(index, step.name)
            try:
                res = step.runner(
                    step.params,
                    self.log,
                    self._scoped_progress(index),
                    self.is_interrupted,
                )
            except Exception as exc:  # noqa: BLE001
                self.error_signal.emit(
                    f"{step.name} failed: {type(exc).__name__}: {exc}"
                )
                self.current_step_changed_signal.emit(-1, "error")
                return
            results[step.name] = res or {}
            self.log(f"{step.name} done")
        self.progress(100)
        self.log("Pipeline finished")
        # Revert the page label to "idle" now that all steps are done.
        self.current_step_changed_signal.emit(-1, "done")
        self.result_signal.emit(results)
