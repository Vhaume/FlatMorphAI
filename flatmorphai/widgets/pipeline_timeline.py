"""
Horizontal step-timeline widget used by the Pipeline A and Pipeline B
pages.

Shows one circular node per step, with an arrow between them. Each
node can be set to one of four statuses (idle, running, done, error)
that control its fill colour.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..constants import (
    COLOR_BORDER,
    COLOR_STATUS_DONE,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_IDLE,
    COLOR_STATUS_RUNNING,
)


class _StepCircle(QWidget):
    """Single coloured disc used as the status indicator inside a step tile."""

    STATUS_COLORS = {
        "idle": COLOR_STATUS_IDLE,
        "running": COLOR_STATUS_RUNNING,
        "done": COLOR_STATUS_DONE,
        "error": COLOR_STATUS_ERROR,
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._status = "idle"
        self.setFixedSize(40, 40)

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor(self.STATUS_COLORS.get(self._status, COLOR_STATUS_IDLE))
        painter.setBrush(color)
        pen = QPen(QColor(COLOR_BORDER))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(2, 2, self.width() - 4, self.height() - 4)


class _StepNode(QWidget):
    """Stacked coloured disc + text label for a single pipeline step.

    The former implementation drew the text with :func:`QPainter.drawText`
    inside a widget of fixed size 80x60, which clipped the label as soon
    as the user pushed the global UI scale past a few Ctrl+ steps. A
    :class:`QLabel` below a dedicated :class:`_StepCircle` lets Qt size
    the text row with the current font metrics, matching the behaviour
    recommended in Nielsen J., Usability Engineering, 1993, ISBN:
    0-12-518406-9, chapter 5, for zoom-aware text layouts.
    """

    def __init__(self, step_label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._circle = _StepCircle(self)
        self._label = QLabel(step_label, self)
        self._label.setObjectName("timelineStepLabel")
        self._label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self._label.setWordWrap(True)
        self._label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self._circle, 0, Qt.AlignHCenter)
        layout.addWidget(self._label, 1, Qt.AlignHCenter | Qt.AlignTop)
        # Allow vertical growth so the tile below the circle always
        # fits the rendered text, regardless of the active UI zoom.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def set_status(self, status: str) -> None:
        self._circle.set_status(status)


class _Arrow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setMinimumWidth(30)
        # Expanding width so the arrows stretch to fill the page and the whole
        # timeline spreads across its full length.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(COLOR_BORDER))
        pen.setWidth(2)
        painter.setPen(pen)
        y = 20
        painter.drawLine(4, y, self.width() - 10, y)
        painter.drawLine(self.width() - 10, y, self.width() - 14, y - 4)
        painter.drawLine(self.width() - 10, y, self.width() - 14, y + 4)


class PipelineTimeline(QWidget):
    """Stacked step nodes connected by arrows.

    A dedicated "Current step:" label sits below the node row and
    displays the name of the step that is actively running, mirroring
    Nielsen J., "10 Usability Heuristics for User Interface Design",
    1994 (https://www.nngroup.com/articles/ten-usability-heuristics/),
    heuristic 1 ("visibility of system status").  Pipeline pages drive
    this label via ``set_current(index, name)`` connected to the
    PipelineWorker ``current_step_changed_signal`` so the label
    updates without having to parse log lines.
    """

    def __init__(
        self,
        step_labels: List[str],
        step_descriptions: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Pipeline progress")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)
        row = QHBoxLayout()
        row.setSpacing(4)
        outer.addLayout(row)
        self._nodes: List[_StepNode] = []
        self._labels: List[str] = list(step_labels)
        # Per-step descriptions used by the "Current step:" label.
        # When the caller does not provide them, fall back to the
        # short labels.  Pages SHOULD pass descriptive strings
        # (e.g. "Frame extraction", "Mask2Former segmentation",
        # "Per-frame 3-mask metrics + classification") so the
        # operator can tell what work is happening without scanning
        # the log panel.  Reference: Nielsen, J., "10 Usability
        # Heuristics for User Interface Design", 1994,
        # https://www.nngroup.com/articles/ten-usability-heuristics/,
        # heuristic 2 (match between system and real world).
        if step_descriptions and len(step_descriptions) == len(step_labels):
            self._descriptions: List[str] = list(step_descriptions)
        else:
            self._descriptions = list(step_labels)
        # No end stretches: the expanding arrows make the row fill the full
        # page width with equal spacing, so the steps spread across the whole
        # length and stay balanced (no left bias / trailing gap).
        for i, label in enumerate(step_labels):
            node = _StepNode(label)
            self._nodes.append(node)
            row.addWidget(node)
            if i != len(step_labels) - 1:
                row.addWidget(_Arrow(), 1)
        # Dedicated "Current step:" line, updated by set_current().
        # The label stays visible at all times (with "Idle" between
        # runs) so the user can confirm which step is running without
        # scanning the log panel.
        self._current_label = QLabel("Current step: idle")
        self._current_label.setObjectName("currentStepLabel")
        self._current_label.setWordWrap(True)
        outer.addWidget(self._current_label)

    def set_status(self, index: int, status: str) -> None:
        if 0 <= index < len(self._nodes):
            self._nodes[index].set_status(status)

    def set_current(self, index: int, name: Optional[str] = None) -> None:
        """Update the "Current step:" label.

        When ``index`` is in range, the label reads
        ``Current step: <short> <i+1>/<N> - <description>`` where
        ``<short>`` is the timeline node label (e.g. ``"Step 1"``)
        and ``<description>`` is the per-step description supplied at
        construction time (e.g. ``"Frame extraction"``).  When the
        caller did not provide descriptions, the bare label is used.
        When ``index`` is negative (-1 by convention), the label
        reverts to ``idle``, which signals the end of a run or a
        cancel.
        """
        if index < 0 or index >= len(self._nodes):
            self._current_label.setText("Current step: idle")
            return
        if name is None:
            name = self._labels[index]
        n = len(self._nodes)
        description = self._descriptions[index]
        if description and description != name:
            self._current_label.setText(
                f"Current step: {name} ({index + 1}/{n}) - {description}"
            )
        else:
            self._current_label.setText(
                f"Current step: {name} ({index + 1}/{n})"
            )

    def reset(self) -> None:
        for node in self._nodes:
            node.set_status("idle")
        self.set_current(-1)
