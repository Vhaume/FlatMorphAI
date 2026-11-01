"""
Common base class for every QThread worker in the application.

Defines the four mandatory signals from the v2 specification, seeds the
relevant RNGs (``torch.manual_seed(42)``, ``numpy.random.seed(42)``,
``random.seed(42)``), and exposes interruption helpers so subclasses
only need to implement :meth:`run`.
"""

from __future__ import annotations

import datetime
import faulthandler
import os
import random
import sys
import traceback
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot


# Guarantee a Python traceback on fatal signals even when the app was
# launched outside ``main.py`` (for example a standalone worker
# unit-test). ``faulthandler.enable`` is idempotent.
try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:  # pragma: no cover
    pass

try:  # torch is a hard dependency for Steps 2 and 4 but may be absent.
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional at import time
    torch = None
    _HAS_TORCH = False

from .constants import RANDOM_SEED


def _format_timestamp(ts: Optional[datetime.datetime] = None) -> str:
    """Return an ``[HH:MM:SS]`` prefix for log messages."""
    if ts is None:
        ts = datetime.datetime.now()
    return ts.strftime("[%H:%M:%S]")


class BaseWorker(QObject):
    """Abstract worker object to move into a :class:`QThread`.

    Signals
    -------
    progress_signal(int)
        Emits values in ``[0, 100]`` for the progress bar.
    log_signal(str)
        Emits a timestamped human-readable message.
    result_signal(object)
        Emits the final result on successful completion.
    error_signal(str)
        Emits a human-readable error string on failure.
    finished_signal()
        Emitted exactly once when the worker has terminated (success or failure).
    """

    progress_signal = Signal(int)
    log_signal = Signal(str)
    result_signal = Signal(object)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._interrupted = False
        # Seed every RNG that can affect reproducibility (spec section 6.4).
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        if _HAS_TORCH:
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Interruption helpers
    # ------------------------------------------------------------------
    def request_interruption(self) -> None:
        """Mark the worker as interrupted. Polled via :meth:`is_interrupted`."""
        self._interrupted = True

    def is_interrupted(self) -> bool:
        """Return True when the worker has been asked to stop."""
        return self._interrupted

    def force_terminate(self) -> None:
        """Stop the worker as hard as possible (called on application quit).

        The base implementation only requests interruption. Subclasses that
        own an external resource -- notably the pipelines that drive a child
        process -- override this to KILL that process immediately, so no
        subprocess is left running (and holding GPU memory) after the window
        closes. See the regeneration / eyespots / Pipeline C workers.
        """
        self.request_interruption()

    # ------------------------------------------------------------------
    # Convenience emitters
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        """Emit a timestamped log message, minus the tuning trace.

        Every worker in the application logs through here, so this is the one
        place where the measurement lines -- block durations, CPU seconds,
        cache hit rates, the RAM planner's arithmetic, the GPU self-test --
        are held back. They were the instrument of the optimisation work and
        they remain how a slowdown is diagnosed, but during a normal run they
        are most of what scrolls past, and none of it is about the worms.
        What describes the WORK still goes through untouched: the step
        boundaries and their durations, the videos processed, the frames
        written, the predicted classes, where the results were copied.
        Nothing disappears silently: the hidden lines are counted and the
        count is stated when the run ends; FLATMORPHAI_VERBOSE=1 restores
        every line.
        """
        try:
            from .workers import log_noise as _noise
            if _noise.diagnostic(message):
                self._hidden_lines = getattr(self, "_hidden_lines", 0) + 1
                return
            # "DONE <n>" ends a regeneration / eyespots run, "Pipeline
            # finished" a stereotype run: the count has to be stated
            # whichever chain was used, or the hidden lines vanish silently.
            if message.startswith(("Pipeline finished", "Pipeline cancelled",
                                   "DONE")):
                _n = getattr(self, "_hidden_lines", 0)
                if _n:
                    self._hidden_lines = 0
                    self.log_signal.emit(
                        "%s   (%d measurement line(s) hidden: block durations, "
                        "caches, memory planner; FLATMORPHAI_VERBOSE=1 shows "
                        "them)" % (_format_timestamp(), _n))
        except Exception:  # noqa: BLE001 - logging must never break a run
            pass
        self.log_signal.emit(f"{_format_timestamp()} {message}")

    def progress(self, value: int) -> None:
        """Emit a progress percentage clamped to [0, 100]."""
        value = max(0, min(100, int(value)))
        self.progress_signal.emit(value)

    # ------------------------------------------------------------------
    # Pre-run dependency gate
    # ------------------------------------------------------------------
    def require_packages(self, pipeline: str) -> bool:
        """Check the packages this processing chain needs, before running.

        The heavy work happens in a child process, so a missing third-party
        package (``skan`` for the regeneration axis, ``segmentation_models_
        pytorch`` for the detectors, ...) would otherwise surface as a raw
        ImportError traceback in the log, or as a child that dies with no
        usable message. Probing here - with ``importlib.util.find_spec``, so
        nothing heavy is imported in the GUI process - turns that into one
        actionable line naming each package and its install command.

        Returns True when the run may proceed; otherwise emits the error and
        returns False.
        """
        try:
            from .dependency_check import (
                format_descriptor_module_error, format_pipeline_error,
                missing_descriptor_modules, missing_for,
            )
        except Exception:  # noqa: BLE001 - never block a run on this gate
            return True
        try:
            missing = missing_for(pipeline)
        except Exception:  # noqa: BLE001
            missing = []
        if missing:
            self.error_signal.emit(format_pipeline_error(pipeline, missing))
            return False
        # Bundled descriptor SOURCE modules (not pip packages): a file missing
        # from metrics_lib/ would otherwise crash the run with a raw
        # ModuleNotFoundError. Report it as one clear line instead.
        try:
            mods = missing_descriptor_modules(pipeline)
        except Exception:  # noqa: BLE001
            mods = []
        if mods:
            self.error_signal.emit(format_descriptor_module_error(pipeline, mods))
            return False
        return True

    # ------------------------------------------------------------------
    # Device resolution (spec section 6.3)
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_device():
        """Return ``torch.device('cuda')`` when available, else ``'cpu'``."""
        if _HAS_TORCH:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return None

    @staticmethod
    def devices_equivalent(left, right) -> bool:
        """Return ``True`` when two device-like values point to the same device.

        ``torch.device('cuda')`` and ``torch.device('cuda:0')`` compare as
        unequal because the former leaves the index unset, but in
        single-GPU configurations they refer to the same physical device
        and re-issuing ``model.to(device)`` is a no-op that still pushes
        a synchronisation through the CUDA driver. This helper treats
        any two ``cuda`` devices as equivalent (we do not target multi-
        GPU dispatch yet) and otherwise falls back to a strict
        equality check.
        """
        if _HAS_TORCH and torch is not None:
            try:
                left_dev = (
                    left if isinstance(left, torch.device) else torch.device(left)
                )
                right_dev = (
                    right if isinstance(right, torch.device) else torch.device(right)
                )
            except Exception:
                return left == right
            if left_dev.type != right_dev.type:
                return False
            if left_dev.type == "cpu":
                return True
            # For cuda treat an unset index as "the current device". On
            # a single-GPU workstation that is always device 0.
            l_idx = left_dev.index if left_dev.index is not None else 0
            r_idx = right_dev.index if right_dev.index is not None else 0
            return l_idx == r_idx
        return left == right

    # ------------------------------------------------------------------
    # Entry point - subclasses override :meth:`run`.
    # ------------------------------------------------------------------
    @staticmethod
    def _configure_thread_safety() -> None:
        """Reduce native thread-pool contention inside the worker.

        OpenCV spawns its own thread pool on first call and several
        BLAS backends (OpenBLAS, MKL) do the same. When a PySide6
        QThread runs heavy Torch inference on top of those pools the
        total number of OS threads can exceed the kernel's ulimit and
        trigger SIGSEGV during allocation. Pinning each pool to a
        single thread keeps the worker predictable and is the
        recommended setting for latency-sensitive GPU inference in the
        PyTorch performance tuning guide
        (pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
        """
        try:
            import cv2  # type: ignore
            cv2.setNumThreads(0)
        except Exception:
            pass
        try:
            import torch  # type: ignore
            torch.set_num_threads(1)
        except Exception:
            pass
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(var, "1")

    @Slot()
    def start(self) -> None:
        """Default slot connected to ``QThread.started``.

        On failure we emit a short one-line summary on
        :attr:`error_signal` and the full traceback on
        :attr:`log_signal`. Surfacing the stack trace matters for
        long-running workers (Step 2 segmentation, Step 4
        classification) where the root cause may be a CUDA/Torch error
        that is not obvious from ``type(exc).__name__`` alone. This
        matches the diagnostic guidance in Wilson, G. et al., "Good
        Enough Practices in Scientific Computing", PLOS Computational
        Biology, 2017, DOI: 10.1371/journal.pcbi.1005510, rule 4 (make
        failures loud and informative).
        """
        BaseWorker._configure_thread_safety()
        try:
            self.run()
        except BaseException as exc:  # pragma: no cover - reported via signal
            # ``BaseException`` is intentionally wide: we also want to
            # intercept ``KeyboardInterrupt`` and ``SystemExit`` so the
            # UI gets a message instead of a silent thread death. Log
            # the full stack trace first so the UI's log panel shows
            # the real root cause, then emit a concise summary that
            # fits the error banner.
            tb = traceback.format_exc()
            self.log_signal.emit(f"{_format_timestamp()} TRACEBACK:\n{tb}")
            self.error_signal.emit(
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            # Force the CUDA caching allocator to release blocks that
            # are no longer referenced. Doing this on the worker thread
            # (where the GPU work happened) avoids deferring CUDA-side
            # cleanup to the GUI thread, which has been observed to
            # SIGSEGV inside ``libcuda.so`` on WSL2 when the GUI thread
            # later touches the driver. Reference: PyTorch CUDA
            # semantics note, "Memory management" section
            # (pytorch.org/docs/stable/notes/cuda.html), which states
            # that ``empty_cache`` returns memory to the device only
            # from the calling thread's allocator state.
            if _HAS_TORCH and torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            self.finished_signal.emit()

    def run(self) -> None:
        """Override me in subclasses."""
        raise NotImplementedError
