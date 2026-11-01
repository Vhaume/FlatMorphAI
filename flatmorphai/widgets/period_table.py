"""
Period configuration table used by Step 6.

Each row pairs a video directory (chosen from the processing root)
with a second-range and a planaria index. The period label no longer
appears as an editable column. It is derived at runtime from the
v1.1 video directory name through
:func:`flatmorphai.notebook_api.naming.parse_directory_name`, falling
back to :func:`parse_video_name` and finally to the raw video stem.

Reference
---------
  - Sandve G. K., Nekrutenko A., Taylor J., Hovig E. (2013),
    "Ten Simple Rules for Reproducible Computational Research",
    PLOS Computational Biology, DOI: 10.1371/journal.pcbi.1003285,
    rule 1 (track how every result was produced) and rule 6 (avoid
    manual data manipulation steps). Deriving the period from the
    directory name avoids asking the user to re-type a token that
    is already encoded in the filename.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QLabel,
    QSpinBox,
    QTableWidget,
    QWidget,
)

from ..notebook_api.naming import parse_directory_name, parse_video_name


def _derive_period_from_video(video_name: str) -> str:
    """Return the period token for ``video_name`` or an empty string.

    Prefers :func:`parse_directory_name` which understands the v1.1
    directory name ``<stem>_(<total>)s_step_(<step>)s``. Falls back to
    :func:`parse_video_name` on the stem alone.
    """
    if not video_name:
        return ""
    parsed_dir = parse_directory_name(video_name)
    if parsed_dir is not None:
        return parsed_dir[0].period
    meta = parse_video_name(video_name)
    if meta is not None:
        return meta.period
    return ""


class PeriodTableWidget(QTableWidget):
    """Flat table: one row per (video, range, planaria index).

    The period label is auto-derived from the selected video and shown
    read-only in the leftmost column so the user can verify the group
    each row will land in after fusion.
    """

    COLS = ("Period (auto)", "Video name", "Start second", "End second", "Planaria #")

    def __init__(self, video_names: Optional[List[str]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.AllEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._video_names: List[str] = list(video_names or [])

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def set_video_names(self, names: List[str]) -> None:
        self._video_names = list(names)
        # Refresh combos in existing rows and rewrite the derived period.
        for row in range(self.rowCount()):
            combo = self.cellWidget(row, 1)
            if isinstance(combo, QComboBox):
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self._video_names)
                if current and current in self._video_names:
                    combo.setCurrentText(current)
                combo.blockSignals(False)
                self._refresh_period_label(row)

    def _make_period_label(self, video_name: str) -> QLabel:
        """Return a right-padded QLabel that displays the derived period."""
        period = _derive_period_from_video(video_name)
        label = QLabel(period or "?")
        label.setAlignment(Qt.AlignCenter)
        label.setObjectName("periodDerivedLabel")
        label.setToolTip(
            "Period derived from the video directory name "
            "(compound_concentration_plate_position_period)."
        )
        return label

    def _refresh_period_label(self, row: int) -> None:
        combo = self.cellWidget(row, 1)
        if not isinstance(combo, QComboBox):
            return
        video = combo.currentText().strip()
        self.setCellWidget(row, 0, self._make_period_label(video))

    def add_row(
        self,
        video_name: str = "",
        start: int = 1,
        end: int = 30,
        planaria_index: int = 1,
        period_name: str = "",
    ) -> None:
        """Add a single video/range sub-row.

        The ``period_name`` argument is accepted for API compatibility
        with the previous version but is ignored: the displayed period
        is always derived from the chosen video.
        """
        del period_name  # Deprecated, kept for backward-compatible callers.
        row = self.rowCount()
        self.insertRow(row)

        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self._video_names)
        if video_name:
            combo.setCurrentText(video_name)
        self.setCellWidget(row, 1, combo)

        period_label = self._make_period_label(combo.currentText())
        self.setCellWidget(row, 0, period_label)
        combo.currentTextChanged.connect(lambda _text, r=row: self._refresh_period_label(r))

        start_spin = QSpinBox()
        start_spin.setRange(1, 10 ** 6)
        start_spin.setValue(start)
        self.setCellWidget(row, 2, start_spin)

        end_spin = QSpinBox()
        end_spin.setRange(1, 10 ** 6)
        end_spin.setValue(end)
        self.setCellWidget(row, 3, end_spin)

        pi_spin = QSpinBox()
        pi_spin.setRange(1, 100)
        pi_spin.setValue(planaria_index)
        self.setCellWidget(row, 4, pi_spin)

    def add_period(self, period_name: str = "") -> None:
        """Convenience wrapper: add a single row pre-populated with the first video.

        ``period_name`` is kept for backward compatibility and ignored.
        """
        del period_name
        default_video = self._video_names[0] if self._video_names else ""
        self.add_row(video_name=default_video)

    def remove_selected(self) -> None:
        """Remove the rows currently selected by the user."""
        rows = sorted({idx.row() for idx in self.selectedIndexes()}, reverse=True)
        for row in rows:
            self.removeRow(row)

    def remove_period(self, row: int) -> None:
        if 0 <= row < self.rowCount():
            self.removeRow(row)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_config(self) -> Tuple[List[str], List[List[List[int]]], List[List[str]], List[List[int]]]:
        """Return ``(periods_name, seconds_by_periods, videos_to_fuse, planaria_indexes)``.

        Rows are grouped by the period token derived from each row's
        video directory name. Rows whose video cannot be parsed are
        grouped under the literal ``"unknown"`` label so Step 6 still
        produces a complete CSV instead of silently dropping them.
        """
        grouped = {}
        order: List[str] = []
        for row in range(self.rowCount()):
            video_widget = self.cellWidget(row, 1)
            start_widget = self.cellWidget(row, 2)
            end_widget = self.cellWidget(row, 3)
            pi_widget = self.cellWidget(row, 4)
            if None in (video_widget, start_widget, end_widget, pi_widget):
                continue
            video_name = video_widget.currentText().strip()
            period_name = _derive_period_from_video(video_name) or "unknown"
            if period_name not in grouped:
                grouped[period_name] = ([], [], [])
                order.append(period_name)
            videos, ranges, planarias = grouped[period_name]
            videos.append(video_name)
            ranges.append([int(start_widget.value()), int(end_widget.value())])
            planarias.append(int(pi_widget.value()))

        periods_name: List[str] = []
        videos_to_fuse: List[List[str]] = []
        seconds_by_periods: List[List[List[int]]] = []
        planaria_indexes: List[List[int]] = []
        for name in order:
            videos, ranges, planarias = grouped[name]
            periods_name.append(name)
            videos_to_fuse.append(videos)
            seconds_by_periods.append(ranges)
            planaria_indexes.append(planarias)
        return periods_name, seconds_by_periods, videos_to_fuse, planaria_indexes

    def validate_ranges(self) -> List[str]:
        """Return a list of error messages for overlapping ranges per derived period."""
        errors: List[str] = []
        groups: dict[str, List[Tuple[int, int]]] = {}
        for row in range(self.rowCount()):
            combo = self.cellWidget(row, 1)
            video_name = combo.currentText().strip() if isinstance(combo, QComboBox) else ""
            name = _derive_period_from_video(video_name) or "unknown"
            start = int(self.cellWidget(row, 2).value())
            end = int(self.cellWidget(row, 3).value())
            if end < start:
                errors.append(f"Row {row + 1}: end ({end}) before start ({start})")
            groups.setdefault(name, []).append((start, end))
        for name, ranges in groups.items():
            ranges_sorted = sorted(ranges)
            for (s1, e1), (s2, e2) in zip(ranges_sorted, ranges_sorted[1:]):
                if s2 <= e1:
                    errors.append(
                        f"Period '{name}' has overlap between [{s1},{e1}] and [{s2},{e2}]"
                    )
        return errors
