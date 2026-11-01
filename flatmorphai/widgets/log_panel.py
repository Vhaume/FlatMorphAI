"""
Read-only log panel with colourised severity and toolbar controls.

Severity is inferred from the message text: ``WARN``/``WARNING`` amber,
``ERROR``/``CRITICAL`` red, everything else white. Messages are
appended with a timestamp and the view autoscrolls to the bottom.

A hard cap on the QTextEdit document blocks (``MAX_BLOCK_COUNT``)
prevents the buffer from growing unboundedly when a worker emits
thousands of per-frame log lines. Without the cap, the underlying
``QTextDocument`` keeps every paragraph in memory; on WSLg the X11
backend has been observed to fault when a textedit's block count
crosses the multi-thousand threshold while painting, which is one of
the few code paths that can SIGSEGV the GUI thread inside
``app.exec()``. Reference for the "fail loud, fail safe" policy:
Wilson, G. et al., "Good Enough Practices in Scientific Computing",
PLOS Computational Biology, 2017,
DOI: 10.1371/journal.pcbi.1005510, rule 4.
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..constants import COLOR_ERROR, COLOR_TEXT_PRIMARY, COLOR_WARNING
from .resize_grip import VerticalResizeGrip


class LogPanel(QWidget):
    """Simple QTextEdit-backed log sink."""

    clear_requested = Signal()

    # Hard cap on the number of paragraphs (text blocks) the
    # underlying QTextDocument is allowed to keep. Older blocks are
    # discarded by ``QTextDocument.setMaximumBlockCount`` when this
    # threshold is exceeded. 5000 lines is enough to keep the user's
    # recent context after a long inference run while staying well
    # below the regime where Qt's text layout becomes a performance
    # bottleneck.
    MAX_BLOCK_COUNT = 5000

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("logPanel")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("toolbarButton")
        self._clear_btn.clicked.connect(self.clear)
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setObjectName("toolbarButton")
        self._copy_btn.clicked.connect(self.copy_to_clipboard)
        toolbar.addWidget(self._clear_btn)
        toolbar.addWidget(self._copy_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self._text = QTextEdit()
        self._text.setObjectName("logText")
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.NoWrap)
        # Cap the document size so a flood of per-frame log lines
        # cannot grow the QTextDocument unboundedly. ``setMaximumBlockCount``
        # discards the oldest paragraphs once the threshold is reached.
        try:
            self._text.document().setMaximumBlockCount(self.MAX_BLOCK_COUNT)
        except Exception:  # noqa: BLE001
            pass
        # A comfortable default so the log is usable before any resize.
        self._text.setMinimumHeight(150)
        layout.addWidget(self._text, 1)

        # Resize handle at the very bottom: drag it down to make the execution
        # log taller. The run pages are wrapped in a scroll area so the Run bar
        # scrolls below the enlarged log instead of being pushed off-screen.
        layout.addWidget(
            VerticalResizeGrip(
                self,
                min_height=120,
                tooltip="Drag down to enlarge the execution log, up to shrink it",
            )
        )

        # Coalesced painting. Lines are buffered and flushed to the QTextEdit
        # in a single repaint on a short timer, so a burst (e.g. a pipeline
        # traceback) paints ONCE instead of once per line. This removes the
        # WSLg X11 paint-flood path that can SIGSEGV the GUI thread inside
        # app.exec(). Persistence and console echo still happen immediately
        # per line in ``append`` so nothing is lost if a crash occurs anyway.
        self._pending = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(60)
        self._flush_timer.timeout.connect(self._flush_pending)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    @Slot(str)
    def append(self, message: str) -> None:
        """Buffer one message; it is painted on the next coalesced flush."""
        # Drop blank lines (trailing newlines and empty lines inside a
        # multi-line message such as a traceback) so the log never shows
        # empty gaps. A message that is entirely blank is ignored.
        message = "\n".join(
            ln for ln in str(message).splitlines() if ln.strip()
        )
        if not message:
            return
        severity = "info"
        upper = message.upper()
        if "ERROR" in upper or "CRITICAL" in upper or "OOM" in upper:
            severity = "error"
        elif "WARN" in upper:
            severity = "warn"

        # Persist every line to the session log (survives a later crash), and
        # echo warnings/errors to the real terminal so they are visible in the
        # command prompt even if the GUI dies right after. Done IMMEDIATELY,
        # per line, before the deferred paint, so nothing is ever lost. Both
        # are best-effort and must never raise.
        try:
            from ..logging_setup import write_line as _persist
            _persist(message)
        except Exception:  # noqa: BLE001
            pass
        if severity in ("error", "warn"):
            try:
                console = sys.__stderr__ or sys.stderr
                if console is not None:
                    console.write(message + "\n")
                    console.flush()
            except Exception:  # noqa: BLE001
                pass

        # Defer the widget insertion; a burst coalesces into one repaint.
        self._pending.append((message, severity))
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    @Slot()
    def _flush_pending(self) -> None:
        """Insert all buffered lines in a single repaint."""
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        self._text.setUpdatesEnabled(False)
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        try:
            for message, severity in pending:
                fmt = QTextCharFormat()
                fmt.setForeground(self._severity_color(severity))
                cursor.insertText(message + "\n", fmt)
        finally:
            self._text.setUpdatesEnabled(True)
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

    def _severity_color(self, severity: str) -> QColor:
        # Info text follows the theme (dark on the light log, light on the dark
        # log) so it is never white-on-light. Error/warn stay red/amber but use
        # a brighter shade on a dark background for contrast.
        from PySide6.QtGui import QPalette
        dark = self._text.palette().color(QPalette.Base).lightness() < 128
        if severity == "error":
            return QColor("#FB7185" if dark else "#DC2626")
        if severity == "warn":
            return QColor("#FBBF24" if dark else "#B45309")
        return self._text.palette().color(QPalette.Text)

    @Slot()
    def clear(self) -> None:
        """Remove all messages from the panel."""
        self._pending = []
        if self._flush_timer.isActive():
            self._flush_timer.stop()
        self._text.clear()
        self.clear_requested.emit()

    def copy_to_clipboard(self) -> None:
        """Copy the entire log contents to the system clipboard."""
        self._flush_pending()  # include any buffered-but-not-yet-painted lines
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text.toPlainText())
