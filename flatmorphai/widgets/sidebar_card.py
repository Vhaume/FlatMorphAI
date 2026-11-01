"""
Sidebar card used to represent each step / pipeline.

Clicking the card emits ``step_selected(int)``. The ``status`` property
controls the colour of the right-hand dot (idle, running, done, error).
The active card is drawn with a 3px accent-blue border.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..constants import (
    COLOR_ACCENT,
    COLOR_STATUS_DONE,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_IDLE,
    COLOR_STATUS_RUNNING,
)


class _StatusDot(QWidget):
    """Small 10x10 px coloured circle drawn with :class:`QPainter`."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor(COLOR_STATUS_IDLE)

    def set_color(self, hex_color: str) -> None:
        self._color = QColor(hex_color)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(self._color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(1, 1, 10, 10)


class SidebarCard(QWidget):
    """One clickable card in the left sidebar.

    Emits :pysignal:`step_selected` with the card's ``step_index`` when
    clicked.
    """

    step_selected = Signal(int)

    STATUS_COLORS = {
        "idle": COLOR_STATUS_IDLE,
        "running": COLOR_STATUS_RUNNING,
        "done": COLOR_STATUS_DONE,
        "error": COLOR_STATUS_ERROR,
    }

    def __init__(
        self,
        step_index: int,
        step_number: str,
        step_name: str,
        icon_color: str = COLOR_ACCENT,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.step_index = step_index
        self._active = False
        self._icon_color = icon_color
        self.setObjectName("sidebarCard")
        self.setProperty("active", False)
        self._build_layout(step_number, step_name)

    @staticmethod
    def _letter_for(title: str) -> str:
        """Single glyph for the tile icon, matching the dashboard design."""
        t = (title or "").strip()
        if t.lower().startswith("pipeline ") and len(t) > 9:
            return t[9].upper()
        return (t[0].upper() if t else "?")

    @staticmethod
    def _tint(hex_color: str, alpha: float) -> str:
        """``rgba(...)`` string of ``hex_color`` at ``alpha`` for a soft fill."""
        c = QColor(hex_color)
        return "rgba(%d, %d, %d, %.2f)" % (c.red(), c.green(), c.blue(), alpha)

    def _build_layout(self, step_number: str, step_name: str) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Coloured rounded-square tile with a letter, like the dashboard nav.
        self._icon = QLabel(self._letter_for(step_number))
        self._icon.setFixedSize(30, 30)
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setStyleSheet(
            "background-color: %s; color: %s; border-radius: 9px; "
            "font-weight: 800; font-size: 11pt;"
            % (self._tint(self._icon_color, 0.16), self._icon_color)
        )
        layout.addWidget(self._icon, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)
        self._number_label = QLabel(step_number)
        self._number_label.setObjectName("sidebarNumber")
        # Word-wrap so the label fully displays whatever the active UI
        # zoom level is: users can push the font past Ctrl+++++ and the
        # card height must grow instead of clipping the text. See
        # "Responsive text wrapping for dynamic font scaling", Nielsen J.,
        # Usability Engineering, 1993, ISBN: 0-12-518406-9, chapter 5.
        self._number_label.setWordWrap(True)
        self._number_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Minimum
        )
        self._name_label = QLabel(step_name)
        self._name_label.setObjectName("sidebarName")
        self._name_label.setWordWrap(True)
        self._name_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Minimum
        )
        text_layout.addWidget(self._number_label)
        text_layout.addWidget(self._name_label)
        layout.addLayout(text_layout, 1)

        self._dot = _StatusDot(self)
        layout.addWidget(self._dot, 0, Qt.AlignVCenter)
        # The card itself must grow vertically if the labels wrap to two
        # lines at large font sizes. Without this policy the sidebar
        # clips the second line when the user zooms with Ctrl+wheel.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def natural_width(self) -> int:
        """Width needed to show the text without clipping mid-word.

        Word-wrapped labels report a tiny ``sizeHint().width()`` (they can wrap
        to anything), which makes an auto-fit based on the size hint collapse
        the sidebar to a sliver at large font sizes. This instead measures the
        real text: the icon, plus room for the longest WORD of the title and
        the full subtitle (titles may wrap across words, but never inside one),
        plus the status dot and margins.
        """
        tfm = self._number_label.fontMetrics()
        sfm = self._name_label.fontMetrics()
        longest_title_word = max(
            (tfm.horizontalAdvance(w) for w in self._number_label.text().split()),
            default=0,
        )
        subtitle_w = max(
            (sfm.horizontalAdvance(w) for w in self._name_label.text().split()),
            default=0,
        )
        text_w = max(longest_title_word, subtitle_w)
        # icon (30) + layout spacing (10) + text + dot (12) + inner margins (24).
        return 30 + 10 + text_w + 12 + 24

    def set_collapsed(self, collapsed: bool) -> None:
        """Hide the text (and status dot), leaving only the icon tile."""
        self._number_label.setVisible(not collapsed)
        self._name_label.setVisible(not collapsed)
        self._dot.setVisible(not collapsed)

    def set_active(self, active: bool) -> None:
        """Toggle the active border. The stylesheet reads the ``active`` property."""
        if self._active == active:
            return
        self._active = active
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_status(self, status: str) -> None:
        """Set the status dot colour. ``status`` in {idle, running, done, error}."""
        self._dot.set_color(self.STATUS_COLORS.get(status, COLOR_STATUS_IDLE))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.step_selected.emit(self.step_index)
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._active:
            painter = QPainter(self)
            pen = QPen(QColor(COLOR_ACCENT))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(0, 2, 0, self.height() - 2)
