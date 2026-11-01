"""
Reusable vertical resize handle.

A thin draggable bar placed at the bottom of a widget; dragging it down
grows that widget's height, dragging it up shrinks it. Used by the
execution-log panel and the detected-videos list so both can be enlarged
the same way. The host must be able to accommodate the extra height
(e.g. live inside a scroll area) for downward growth to be visible.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class VerticalResizeGrip(QWidget):
    """Drag-to-resize bar for the widget passed as ``target``."""

    def __init__(
        self,
        target: QWidget,
        min_height: int = 120,
        tooltip: str = "Drag down to enlarge, up to shrink",
        parent: Optional[QWidget] = None,
        on_resized: "Optional[Callable[[int], None]]" = None,
    ) -> None:
        """``on_resized`` is called with the final height when a drag ends.

        Callers use it to persist the height the operator chose, so the page
        comes back with the same layout at the next launch.
        """
        super().__init__(parent)
        self._target = target
        self._min_height = int(min_height)
        self._on_resized = on_resized
        self.setFixedHeight(16)
        self.setCursor(Qt.SizeVerCursor)
        self.setObjectName("resizeGrip")
        self.setToolTip(tooltip)
        self._dragging = False
        self._start_y = 0
        self._start_h = 0

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start_y = event.globalPosition().toPoint().y()
            self._start_h = self._target.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._dragging:
            dy = event.globalPosition().toPoint().y() - self._start_y
            new_h = max(self._min_height, int(self._start_h + dy))
            # Pin the height so the surrounding layout cannot shrink it back.
            self._target.setMinimumHeight(new_h)
            self._target.setMaximumHeight(new_h)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        was_dragging = self._dragging
        self._dragging = False
        if was_dragging and self._on_resized is not None:
            try:
                self._on_resized(int(self._target.height()))
            except Exception:  # noqa: BLE001 - persisting must never break the UI
                pass
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        c = self.palette().color(self.foregroundRole())
        cx = self.width() // 2
        y = self.height() // 2
        pill = QColor(c)
        pill.setAlpha(28)
        painter.setPen(Qt.NoPen)
        painter.setBrush(pill)
        painter.drawRoundedRect(cx - 26, y - 2, 52, 4, 2, 2)
        dot = QColor(c)
        dot.setAlpha(150)
        painter.setPen(QPen(dot, 1.8))
        for off in (-12, -4, 4, 12):
            painter.drawPoint(cx + off, y)
        painter.end()
