"""
Common base class shared by every step / pipeline page.

The page knows how to launch a :class:`~flatmorphai.base_worker.BaseWorker`
on a :class:`QThread`, register it with the :class:`WorkerRegistry`,
and expose ``started`` / ``finished`` events for the main window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QWidget

from ..base_worker import BaseWorker
from ..worker_registry import WorkerRegistry


@dataclass
class RunningContext:
    worker: BaseWorker
    thread: QThread
    tracking_id: int


def create_worker_thread(worker: BaseWorker) -> Tuple[QThread, int]:
    """Move ``worker`` to a new :class:`QThread` and register it.

    Returns ``(thread, tracking_id)``. The caller is responsible for
    starting the thread and for calling :func:`WorkerRegistry.unregister`
    when it finishes.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start)
    tid = WorkerRegistry().register(worker, thread)
    return thread, tid


class BasePage(QWidget):
    """Abstract page wrapping a single step worker."""

    run_state_changed = Signal(str, bool)  # (page_name, is_running)
    page_status_changed = Signal(str)      # 'idle' / 'running' / 'done' / 'error'
    navigate_requested = Signal(str)       # sidebar key to switch to (on success)

    def __init__(self, name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name = name
        self._context: Optional[RunningContext] = None

    @property
    def name(self) -> str:
        return self._name

    def is_running(self) -> bool:
        return self._context is not None

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _launch_worker(self, worker: BaseWorker) -> None:
        """Move ``worker`` to a thread and wire up clean teardown.

        The clean-up must never call ``QThread.wait`` from inside the
        worker thread itself (this raises the Qt warning ``QThread::wait:
        Thread tried to wait on itself`` and has been observed to trigger
        a segmentation fault during Step 2 on some platforms). We keep
        the pattern that Qt recommends: the worker's
        :attr:`finished_signal` is routed to ``thread.quit`` through a
        queued connection, and the final teardown is driven by
        ``thread.finished`` which fires on the thread that owns the
        :class:`QThread` object (the main GUI thread). References:
        Summerfield, M., "Rapid GUI Programming with Python and Qt",
        Prentice Hall, 2007, ISBN 0-13-235418-9, chapter 19.
        """
        thread, tid = create_worker_thread(worker)
        ctx = RunningContext(worker=worker, thread=thread, tracking_id=tid)
        self._context = ctx

        # Ask the worker's thread to quit once the worker is done. Use a
        # queued connection so the quit slot is invoked on the receiver's
        # own event loop (the worker thread) without any blocking wait.
        worker.finished_signal.connect(thread.quit, Qt.QueuedConnection)

        def _on_thread_finished() -> None:
            # Runs on the main thread (QThread.finished is delivered to
            # the thread that owns the QThread object). Safe to call
            # deleteLater on both objects here.
            WorkerRegistry().unregister(tid)
            worker.deleteLater()
            thread.deleteLater()
            self._context = None
            self.run_state_changed.emit(self._name, False)

        thread.finished.connect(_on_thread_finished)
        worker.error_signal.connect(
            lambda _msg: self.page_status_changed.emit("error")
        )
        worker.result_signal.connect(
            lambda _r: self.page_status_changed.emit("done")
        )
        thread.start()
        self.run_state_changed.emit(self._name, True)
        self.page_status_changed.emit("running")

    def cancel_current(self) -> None:
        """Request interruption of the currently running worker."""
        if self._context is None:
            return
        self._context.worker.request_interruption()
        self._context.thread.quit()
        self._context.thread.wait(3000)
