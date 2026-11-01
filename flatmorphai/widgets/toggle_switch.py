"""
Sliding toggle switch and inline numeric stepper.

Both reproduce the dashboard's HTML/CSS controls:

* :class:`ToggleSwitch` is a ``QCheckBox`` painted as a pill with a sliding
  knob (grey track off, brand track on). Being a ``QCheckBox`` subclass it keeps
  the full ``isChecked`` / ``setChecked`` / ``toggled`` API, so it is a drop-in
  replacement for the plain check boxes.
* :class:`Stepper` is the ``[- N +]`` inline control used for the correction
  value. It mirrors enough of the ``QSpinBox`` API (``value`` / ``setValue`` /
  ``setRange`` / ``valueChanged`` / ``blockSignals``) for the editor to use it
  in place of a spin box.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

# Palette fallbacks; the real colours come from the widget palette / QSS where
# available, so these only matter if a lookup fails.
_TRACK_OFF = QColor("#cbd5e1")
_TRACK_ON = QColor("#0d9488")
_KNOB = QColor("#ffffff")


class ToggleSwitch(QCheckBox):
    """A check box drawn as an iOS-style sliding switch."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        # The pill is painted in paintEvent; the label text is drawn by us too
        # so it sits to the right of the pill with the right spacing.
        self._track_w = 36
        self._track_h = 20
        self._gap = 8
        self._on_color = QColor("#0d9488")  # brand teal

    def set_on_color(self, hex_color: str) -> None:
        """Override the 'on' track colour (e.g. for the dark theme)."""
        self._on_color = QColor(hex_color)
        self.update()

    # -- geometry ------------------------------------------------------
    def sizeHint(self) -> QSize:  # noqa: N802
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text())
        w = self._track_w + (self._gap + text_w if self.text() else 0)
        h = max(self._track_h, fm.height())
        return QSize(w + 2, h + 2)

    def hitButton(self, pos) -> bool:  # noqa: N802
        # The whole widget toggles (pill + label).
        return self.rect().contains(pos)

    # -- painting ------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        h = self.height()
        track_y = (h - self._track_h) / 2.0
        track = QRectF(1.0, track_y, self._track_w, self._track_h)

        on = self.isChecked()
        track_color = self._on_color if on else _TRACK_OFF
        painter.setPen(Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, self._track_h / 2.0, self._track_h / 2.0)

        # Knob
        d = self._track_h - 4
        kx = (track.right() - d - 2) if on else (track.left() + 2)
        painter.setBrush(_KNOB)
        painter.drawEllipse(QRectF(kx, track_y + 2, d, d))

        # Label text
        if self.text():
            painter.setPen(self.palette().color(QPalette.WindowText))
            tx = int(self._track_w + 1 + self._gap)
            painter.drawText(tx, 0, self.width() - tx, h,
                             Qt.AlignVCenter | Qt.AlignLeft, self.text())
        painter.end()


class Stepper(QWidget):
    """An inline ``[- value +]`` control with a minimal QSpinBox-like API."""

    valueChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepper")
        # A plain QWidget only paints its QSS background/border when told to.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._min = 0
        self._max = 99
        self._value = 0

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(0)
        self._minus = QToolButton()
        self._minus.setText("−")
        self._minus.setObjectName("stepperBtn")
        self._minus.clicked.connect(lambda: self.setValue(self._value - 1, emit=True))
        self._val = QLabel("0")
        self._val.setObjectName("stepperValue")
        self._val.setAlignment(Qt.AlignCenter)
        self._val.setMinimumWidth(26)
        self._plus = QToolButton()
        self._plus.setText("+")
        self._plus.setObjectName("stepperBtn")
        self._plus.clicked.connect(lambda: self.setValue(self._value + 1, emit=True))
        lay.addWidget(self._minus)
        lay.addWidget(self._val)
        lay.addWidget(self._plus)

    # -- QSpinBox-like API --------------------------------------------
    def setRange(self, lo: int, hi: int) -> None:  # noqa: N802
        self._min, self._max = int(lo), int(hi)
        self.setValue(self._value)

    def value(self) -> int:
        return self._value

    def setValue(self, v: int, emit: bool = False) -> None:  # noqa: N802
        v = max(self._min, min(self._max, int(v)))
        changed = v != self._value
        self._value = v
        self._val.setText(str(v))
        self._minus.setEnabled(v > self._min)
        self._plus.setEnabled(v < self._max)
        if changed and emit and not self.signalsBlocked():
            self.valueChanged.emit(v)
