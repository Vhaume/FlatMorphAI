"""
Singleton tracking active QThreads and worker objects.

Used by :class:`~flatmorphai.main_window.MainWindow` on close events to
terminate any remaining worker before shutting down.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread


class WorkerRegistry:
    """Register of all currently alive worker/thread pairs."""

    _instance: Optional["WorkerRegistry"] = None
    _lock = Lock()

    def __new__(cls) -> "WorkerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._threads = {}
        return cls._instance

    def __init__(self) -> None:
        self._threads: Dict[int, Dict[str, object]]

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------
    def register(self, worker: QObject, thread: QThread) -> int:
        """Register a worker/thread pair and return the tracking id."""
        with self._lock:
            tid = id(thread)
            self._threads[tid] = {"worker": worker, "thread": thread}
        return tid

    def unregister(self, tracking_id: int) -> None:
        """Remove a tracked worker/thread pair."""
        with self._lock:
            self._threads.pop(tracking_id, None)

    def active(self) -> List[Dict[str, object]]:
        """Return all currently tracked worker/thread pairs."""
        with self._lock:
            return list(self._threads.values())

    def stop_all(self, wait_ms: int = 3000) -> None:
        """Interrupt and join every registered worker."""
        entries: List[Dict[str, object]] = self.active()
        for entry in entries:
            worker = entry.get("worker")
            thread = entry.get("thread")
            if isinstance(worker, QObject):
                try:
                    # Prefer force_terminate(): BaseWorker requests
                    # interruption, and the subprocess pipelines override it to
                    # KILL their child process so none is left running (holding
                    # GPU memory) after the window closes. Killing the child
                    # also closes its stdout pipe, which unblocks the worker's
                    # read loop so the thread below can actually finish.
                    stop = (
                        getattr(worker, "force_terminate", None)
                        or getattr(worker, "request_interruption", None)
                    )
                    if callable(stop):
                        stop()
                except Exception:
                    pass
            if isinstance(thread, QThread) and thread.isRunning():
                thread.requestInterruption()
                thread.quit()
                thread.wait(wait_ms)
