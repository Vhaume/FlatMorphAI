"""Small painted icons shared across pages (validation check, blank spacer)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF


def check_icon(size: int = 16, color: str = "#16A34A") -> QIcon:
    """A green check mark used to flag a validated/corrected list item."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(color), max(1.8, size * 0.14))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    s = size
    p.drawPolyline(QPolygonF([
        QPointF(s * 0.22, s * 0.54),
        QPointF(s * 0.42, s * 0.74),
        QPointF(s * 0.80, s * 0.28),
    ]))
    p.end()
    return QIcon(pm)


def blank_icon(size: int = 16) -> QIcon:
    """A transparent icon of ``size`` so rows without a check keep the column."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    return QIcon(pm)
