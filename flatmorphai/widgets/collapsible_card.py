"""
CollapsibleCard: a single tile whose clickable header folds its content.

Unlike :class:`CollapsibleSection` (a bare header + a separate widget), the
header here lives *inside* the card frame, directly above the content and with
no gap, so the whole thing reads as one tile. The fold indicator is a large
chevron (sized to the title text) and the header has a hover highlight and a
divider above the content, so the section boundary is obvious.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


def _chevron_pixmap(expanded: bool, color: QColor, size: int = 18) -> QPixmap:
    """Draw a down (expanded) or right (collapsed) chevron."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(color, max(2.0, size * 0.16))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    m = size * 0.26
    if expanded:  # v
        pts = [(m, size * 0.36), (size * 0.5, size * 0.66), (size - m, size * 0.36)]
    else:  # >
        pts = [(size * 0.40, m), (size * 0.70, size * 0.5), (size * 0.40, size - m)]
    p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
    p.end()
    return pm


class _CardHeader(QWidget):
    """Clickable header row with a large chevron and a bold title."""

    clicked = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("collapsibleCardHeader")
        self.setAttribute(Qt.WA_StyledBackground, True)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(10)
        self._chevron = QLabel()
        self._chevron.setObjectName("collapsibleChevron")
        self._chevron.setFixedSize(QSize(20, 20))
        h.addWidget(self._chevron, 0, Qt.AlignVCenter)
        self._title = QLabel(title)
        self._title.setObjectName("collapsibleCardTitle")
        h.addWidget(self._title, 1, Qt.AlignVCenter)
        self._expanded = True

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self._refresh_chevron()

    def _refresh_chevron(self) -> None:
        color = self.palette().color(QPalette.WindowText)
        self._chevron.setPixmap(_chevron_pixmap(self._expanded, color, size=18))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Palette is reliable once shown, so paint the chevron in the theme's
        # text colour (dark on light, light on dark).
        self._refresh_chevron()


class CollapsibleCard(QFrame):
    """One tile: a clickable header folding the content inside the same frame."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        expanded: bool = True,
        on_toggle: Optional[Callable[[bool], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._content = content
        self._on_toggle = on_toggle
        self._expanded = bool(expanded)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._header = _CardHeader(title)
        self._header.clicked.connect(self._toggle)
        v.addWidget(self._header)

        self._divider = QFrame()
        self._divider.setObjectName("collapsibleDivider")
        self._divider.setFrameShape(QFrame.HLine)
        v.addWidget(self._divider)

        v.addWidget(self._content)

        self._apply_state()

    def _toggle(self) -> None:
        self.set_expanded(not self._expanded, notify=True)

    def set_expanded(self, expanded: bool, notify: bool = False) -> None:
        self._expanded = bool(expanded)
        self._apply_state()
        if notify and self._on_toggle is not None:
            self._on_toggle(self._expanded)

    def _apply_state(self) -> None:
        self._content.setVisible(self._expanded)
        self._divider.setVisible(self._expanded)
        self._header.set_expanded(self._expanded)

    def is_expanded(self) -> bool:
        return self._expanded
