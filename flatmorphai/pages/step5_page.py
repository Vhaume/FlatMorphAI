"""
Step 5 page: Stereotype Editor.

The page lets a user load a classification CSV (either a per-video
``results_Stereotype_[<model>].csv`` or the merged ``all_videos.csv``
produced by Step 6), browse every frame in a left side panel, view
the original "FRAME" alongside the Step-3 "FORMATED" image, and
correct the predicted stereotype class. Corrections are persisted in
place to the loaded CSV.

Design notes
------------
The Frames panel groups rows by ``video_name`` in a collapsible tree
so the user can fold the videos that have already been reviewed. A
row is considered "corrected" as soon as its ``corrected_*_name``
column is non-empty, regardless of whether the corrected value
matches the forecast. This is the count surfaced in the status line.
Corrected frames are coloured ``#008000`` so the eye scans the tree
for green entries.

Click-to-uncheck on the corrected radios uses a manual two-state
machine because :class:`QButtonGroup` (in exclusive mode) does not
emit a deselection event when the user clicks the already-checked
radio. We intercept the click before the toggle, compare the new id
to the previously committed id and, if they match, clear the row's
correction instead of re-writing it. This pattern is documented in
the Qt for Python "Forms and dialogs" guide
(doc.qt.io/qtforpython, no DOI).

Auto-advance and Validate are convenience features for long manual
review sessions. References for the rapid-review interaction model:
Tullis, T. & Albert, B., "Measuring the User Experience", 2nd ed.,
Morgan Kaufmann, 2013, ISBN 978-0-12-415781-1, chapter 4 (task-time
metrics for repeated-action interfaces).
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import (
    CLASSES_NAMES_STEREOTYPE,
    CSV_ENCODING_LATIN1,
    CSV_SEPARATOR,
)
from ..notebook_api.csv_editor import modify_column_of_row_in_csv
from ..widgets import ImagePairViewer, ToggleSwitch, blank_icon, check_icon
from .base_page import BasePage
from ..widgets.file_dialogs import (
    get_existing_directory,
    get_open_file_name,
    question as _msg_question,
)


# Per-video directory naming convention (v1.2.1):
# ``<video_stem>_total_<T>s_step_<S>s``. The Step 5 reader matches
# every directory whose name starts with the video stem followed by
# the literal ``_total_`` token so we tolerate any total/step values.
_DIR_TOTAL_INFIX = "_total_"


# Step 3 output directory prefix (v1.2.1). Earlier versions of the
# project shipped the ``Regeneration_process_of_Planaria_*`` prefix;
# the reader keeps it as a backward-compatible alias.
_FORMATED_DIR_PREFIXES = ("Formated", "Regeneration_process_of_Planaria")


# Colour used to highlight corrected frame names in the tree. Pure
# green per the user request: any row whose ``corrected_*_name``
# column is non-empty turns green so the eye can locate the
# remaining work at a glance. Reference for the "use of colour as a
# secondary cue" rule: World Wide Web Consortium, "Web Content
# Accessibility Guidelines 2.1", Recommendation 2018-06-05, success
# criterion 1.4.1 "Use of Color", normative document, no DOI.
_CORRECTED_FG = QColor("#008000")

# Colour dot per stereotype class, keyed by CLASSES_NAMES_STEREOTYPE order
# (C-shape, Screw-like, Snake-like, Walnut, Wildtype).
_STEREOTYPE_COLORS = ["#38BDF8", "#A78BFA", "#2DD4BF", "#FBBF24", "#4ADE80"]


# High-contrast stylesheet shared by every QRadioButton in the
# editor. The font-size is intentionally absent so the global
# stylesheet's ``QRadioButton { font-size: 11pt; }`` rule applies and
# is scaled by ThemeManager when the user zooms with Ctrl+wheel
# (W3C WCAG 2.1, Recommendation 2018-06-05, success criterion 1.4.4
# "Resize text", normative document, no DOI). The 2px border + filled
# inner circle on selection brings the contrast above the WCAG 2.1
# AA threshold of 3:1 for non-text UI components (success criterion
# 1.4.11 "Non-text Contrast").
_HIGH_CONTRAST_RADIO_QSS = (
    "QRadioButton { padding: 4px 2px; } "
    "QRadioButton::indicator { width: 18px; height: 18px; } "
    "QRadioButton::indicator:unchecked { "
    "  border: 2px solid #1f2933; border-radius: 10px; "
    "  background: #ffffff; } "
    "QRadioButton::indicator:unchecked:hover { "
    "  border: 2px solid #0b63c4; } "
    "QRadioButton::indicator:checked { "
    "  border: 2px solid #0b63c4; border-radius: 10px; "
    "  background: #0b63c4; } "
    # ``:disabled`` is a pseudo-state that would otherwise override
    # ``:checked`` (Qt Style Sheets specificity, doc.qt.io/qt-6/
    # stylesheet-syntax.html). Without the compound selector below
    # the read-only Forecasted radio paints as an empty light grey
    # circle even when its underlying button is checked, so the user
    # cannot tell which class was forecasted. Adding the
    # ``:checked:disabled`` rule restores the accent-fill on a
    # disabled radio. Reference for the "show state, do not hide
    # affordance" rule: Norman, D. A., "The Design of Everyday
    # Things", revised edition, Basic Books, 2013,
    # ISBN 978-0-465-05065-9, chapter 4.
    "QRadioButton::indicator:checked:disabled { "
    "  border: 2px solid #0b63c4; border-radius: 10px; "
    "  background: #0b63c4; } "
    "QRadioButton::indicator:unchecked:disabled { "
    "  border: 2px solid #52606d; background: #d9e2ec; } "
    "QRadioButton:disabled { color: #1f2933; }"
)


# Roles attached to QTreeWidget items so we can recover the row index
# of the underlying dataframe in O(1) on selection.
_ROW_INDEX_ROLE = Qt.UserRole + 1
_VIDEO_ROOT_ROLE = Qt.UserRole + 2  # marks top-level (per-video) nodes


def _looks_like_video_dir(entry: str, video_stem: str) -> bool:
    return entry == video_stem or entry.startswith(video_stem + _DIR_TOTAL_INFIX)


def _is_formated_dir(entry: str) -> bool:
    return any(entry.startswith(p) for p in _FORMATED_DIR_PREFIXES)


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class _CorrectableRadioButton(QRadioButton):
    """Radio button that lets the user un-check it by clicking again.

    The default Qt behaviour for a radio inside an exclusive
    :class:`QButtonGroup` is "click-to-check, no deselection". The
    Step 5 editor needs the user to be able to cancel a correction by
    clicking the already-active class, so we override
    :meth:`mousePressEvent` and route the cancellation through a
    parent-page callback. The override only fires when the radio is
    already checked; otherwise it defers to the base class so the
    normal selection logic is unchanged.
    """

    def __init__(self, *args, on_cancel=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_cancel = on_cancel

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.LeftButton
            and self.isChecked()
            and self._on_cancel is not None
        ):
            try:
                self._on_cancel(self)
            except Exception:  # noqa: BLE001
                pass
            event.accept()
            return
        super().mousePressEvent(event)


class Step5Page(BasePage):
    """Manual correction of forecasted stereotype classes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Stereotype Editor", parent)
        self._settings = AppSettings()
        self._current_csv_path: Optional[str] = None
        self._current_df: Optional[pd.DataFrame] = None
        self._current_row_index: int = -1
        self._forecast_idx_col: Optional[str] = None
        self._forecast_name_col: Optional[str] = None
        self._correction_idx_col: Optional[str] = None
        self._correction_name_col: Optional[str] = None
        # Cache mapping ``video_stem`` -> per-video directory (resolved
        # once per CSV load) so frame look-up is O(1) per row.
        self._video_dir_cache: Dict[str, str] = {}
        # Map dataframe-row -> tree leaf so a saved correction can
        # update its colour without re-walking the tree.
        self._row_items: Dict[int, QTreeWidgetItem] = {}
        self._build_ui()
        self._restore_from_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- header: title + subtitle + count chips (like the Regeneration
        #     states editor) ---
        header = QHBoxLayout()
        header.setSpacing(12)
        title = QLabel("Stereotype editor")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        subtitle = QLabel("Review and correct the predicted stereotypes")
        subtitle.setObjectName("pageSubtitle")
        header.addWidget(subtitle, 0, Qt.AlignVCenter)
        header.addStretch(1)
        self._chip_frames = QLabel("")
        self._chip_frames.setObjectName("chip")
        self._chip_videos = QLabel("")
        self._chip_videos.setObjectName("chip")
        self._chip_corrected = QLabel("")
        self._chip_corrected.setObjectName("chipBrand")
        for chip in (self._chip_frames, self._chip_videos, self._chip_corrected):
            chip.setVisible(False)
            header.addWidget(chip, 0, Qt.AlignVCenter)
        root.addLayout(header)

        # --- source card (same single-row layout as the Regeneration states
        #     editor): caption + field + Browse for the processing root, then
        #     caption + field + Browse + Load + Regenerate for the results CSV.
        src_card = QFrame()
        src_card.setObjectName("card")
        sl = QHBoxLayout(src_card)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(10)

        root_cap = QLabel("Processing root")
        root_cap.setObjectName("caption")
        sl.addWidget(root_cap, 0, Qt.AlignVCenter)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Processed_videos_Frame_by_frame/")
        self._input_edit.textChanged.connect(self._on_input_changed)
        sl.addWidget(self._input_edit, 3)
        browse_root = QPushButton("Browse")
        browse_root.clicked.connect(self._on_browse_input)
        sl.addWidget(browse_root)
        sl.addSpacing(14)

        csv_cap = QLabel("Results CSV")
        csv_cap.setObjectName("caption")
        sl.addWidget(csv_cap, 0, Qt.AlignVCenter)
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText(
            "Processed_videos_Frame_by_frame/all_videos.csv"
        )
        sl.addWidget(self._csv_edit, 3)
        browse_csv = QPushButton("Browse")
        browse_csv.clicked.connect(self._on_browse_csv)
        sl.addWidget(browse_csv)
        load_btn = QPushButton("Load")
        load_btn.setObjectName("primaryButton")
        load_btn.clicked.connect(self._on_load)
        sl.addWidget(load_btn)
        root.addWidget(src_card)

        # Live counter, kept as an object but NO LONGER SHOWN: the three chips
        # at the top of the page already carry the same three numbers
        # ("<n> frames", "<n> videos", "corrected: <n> / <n>"), so the line
        # "Total: 180 | Correct (Manual click): 0 / 180" only repeated them a
        # second time, lower down. The widget stays so the two places that
        # write to it keep working; it is simply never added to a layout.
        self._counter_label = QLabel("No CSV loaded")
        self._counter_label.setObjectName("mutedLabel")
        self._counter_label.setVisible(False)

        # Main 30/70 splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left: Frames tree grouped by video, inside a card tile.
        left = QFrame()
        left.setObjectName("card")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        # Header "Frames   n / N" like the Regeneration states editor.
        tree_head = QHBoxLayout()
        left_header = QLabel("Frames")
        left_header.setObjectName("sectionTitle")
        tree_head.addWidget(left_header)
        tree_head.addStretch(1)
        self._row_info_short = QLabel("")
        self._row_info_short.setObjectName("mutedLabel")
        tree_head.addWidget(self._row_info_short, 0, Qt.AlignVCenter)
        left_layout.addLayout(tree_head)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        # Enlarge the expand/collapse chevron so it is easy to hit
        # with the mouse; Qt draws the branch marker inside a
        # ``branch:has-children`` region whose default width is only
        # 12 px.  The stylesheet below reserves 28 px per level of
        # indentation and pushes the drawn triangle to a matching
        # size via ``padding``.  Reference for interactive-control
        # sizing: Nielsen, J. and Norman, D. A., "The Design of
        # Everyday Things", MIT Press, 2013, ISBN 978-0-262-52567-7,
        # ch. 4 (affordance and target-size guidance).
        self._tree.setIndentation(28)
        # Small icon column so a validation check fits just before each name.
        self._tree.setIconSize(QSize(16, 16))
        # Same compact item style as the Regeneration states editor so the
        # hover / selected / corrected row backgrounds match exactly.
        self._tree.setStyleSheet("QTreeWidget::item { padding: 2px 2px; }")
        self._tree.currentItemChanged.connect(self._on_tree_selected)
        left_layout.addWidget(self._tree, 1)

        # Prev / Next under the list so they are reachable without
        # leaving the editor pane.
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self._select_previous)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._select_next)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        left_layout.addLayout(nav_row)
        main_splitter.addWidget(left)

        # Right: vertical splitter with viewer on top and classes
        # below. The viewer titles match the user request: FRAME for
        # the original image, FORMATED for the Step-3 cropped image.
        right_splitter = QSplitter(Qt.Vertical)
        self._viewer = ImagePairViewer(left_title="FRAME", right_title="FORMATED")
        # "3s display" toggle lives in the FORMATED toolbar, just
        # before its Open button.  Restore the saved state BEFORE
        # connecting so loading the preference does not itself
        # trigger a redraw of a page that is not populated yet.
        self._viewer.right_toggle.setChecked(
            bool(self._settings.get("step5_3s_display", False))
        )
        self._viewer.right_toggle.toggled.connect(
            self._on_three_s_toggled
        )
        # Remember the FRAME/FORMATED divider position across sessions.
        self._viewer.bind_persistence(self._settings, "step5_viewer_split")
        # Viewer card: the row indication ("Row X / Y: (period) image.jpg") and
        # the FRAME/FORMATED viewer live inside one white tile, like the
        # Regeneration states editor's viewers card.
        viewer_wrap = QFrame()
        viewer_wrap.setObjectName("card")
        vw = QVBoxLayout(viewer_wrap)
        vw.setContentsMargins(12, 10, 12, 12)
        vw.setSpacing(6)
        self._row_info = QLabel("")
        self._row_info.setObjectName("mutedLabel")
        vw.addWidget(self._row_info)
        vw.addWidget(self._viewer, 1)
        right_splitter.addWidget(viewer_wrap)

        # --- class selection as a single table card: one row per stereotype
        #     with a colour dot + name, the FORECASTED (read-only) radio and
        #     the CORRECT (manual click) radio with its "Pick" number key. ---
        self._forecast_group = QButtonGroup(self)
        self._forecast_group.setExclusive(True)
        self._forecast_radios: List[QRadioButton] = []
        self._correction_group = QButtonGroup(self)
        self._correction_group.setExclusive(True)
        self._correction_radios: List[_CorrectableRadioButton] = []
        self._class_name_labels: List[QLabel] = []

        classes_card = QFrame()
        classes_card.setObjectName("card")
        cvl = QVBoxLayout(classes_card)
        cvl.setContentsMargins(16, 12, 16, 14)
        cvl.setSpacing(8)
        grid_w = QWidget()
        grid_w.setStyleSheet("background: transparent;")
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)

        # Header cell: "STEREOTYPE" caption with the state chip inline next to
        # it ("Forecasted" indigo until corrected, then "Corrected" teal), so
        # the chip is visible without wasting an empty row.
        hdr_w = QWidget()
        hdr_w.setStyleSheet("background: transparent;")
        hdr_h = QHBoxLayout(hdr_w)
        hdr_h.setContentsMargins(0, 0, 0, 0)
        hdr_h.setSpacing(10)
        hdr = QLabel("STEREOTYPE")
        hdr.setObjectName("caption")
        hdr_h.addWidget(hdr, 0, Qt.AlignVCenter)
        self._state_tag = QLabel("Forecasted")
        self._state_tag.setObjectName("chipForecast")
        hdr_h.addWidget(self._state_tag, 0, Qt.AlignVCenter)
        hdr_h.addStretch(1)
        grid.addWidget(hdr_w, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(
            self._column_header("FORECASTED", "read-only"), 0, 1, Qt.AlignCenter
        )
        grid.addWidget(
            self._column_header("CORRECT", "manual click"), 0, 2,
            Qt.AlignLeft | Qt.AlignVCenter,
        )

        for i, cls in enumerate(CLASSES_NAMES_STEREOTYPE):
            r = i + 1
            # col 0: colour dot + class name.
            name_w = QWidget()
            name_w.setStyleSheet("background: transparent;")
            nh = QHBoxLayout(name_w)
            nh.setContentsMargins(0, 2, 0, 2)
            nh.setSpacing(10)
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(
                "background-color: %s; border-radius: 6px;" % _STEREOTYPE_COLORS[i]
            )
            nh.addWidget(dot, 0, Qt.AlignVCenter)
            name_lbl = QLabel(cls)
            name_lbl.setObjectName("className")
            self._class_name_labels.append(name_lbl)
            nh.addWidget(name_lbl, 0, Qt.AlignVCenter)
            nh.addStretch(1)
            grid.addWidget(name_w, r, 0)

            # col 1: read-only forecast radio (centred under FORECASTED).
            fr = QRadioButton()
            fr.setEnabled(False)
            fr.setStyleSheet(_HIGH_CONTRAST_RADIO_QSS)
            self._forecast_group.addButton(fr, i)
            self._forecast_radios.append(fr)
            grid.addWidget(fr, r, 1, Qt.AlignCenter)

            # col 2: correct radio + "Pick" + number-key box.
            corr_w = QWidget()
            corr_w.setStyleSheet("background: transparent;")
            ch = QHBoxLayout(corr_w)
            ch.setContentsMargins(0, 2, 0, 2)
            ch.setSpacing(8)
            cr = _CorrectableRadioButton(on_cancel=self._on_correction_cancelled)
            cr.setStyleSheet(_HIGH_CONTRAST_RADIO_QSS)
            self._correction_group.addButton(cr, i)
            self._correction_radios.append(cr)
            ch.addWidget(cr, 0, Qt.AlignVCenter)
            pick = QLabel("Pick")
            pick.setObjectName("mutedLabel")
            ch.addWidget(pick, 0, Qt.AlignVCenter)
            numbox = QLabel(str(i + 1))
            numbox.setObjectName("mnemonicBox")
            numbox.setAlignment(Qt.AlignCenter)
            ch.addWidget(numbox, 0, Qt.AlignVCenter)
            ch.addStretch(1)
            grid.addWidget(corr_w, r, 2)

        grid.setColumnStretch(0, 1)
        cvl.addWidget(grid_w)
        self._correction_group.idToggled.connect(self._on_correction_chosen)

        # Number keys 1..N pick the matching correction class (replacing the
        # old "&<digit>" radio-label mnemonics now that the radios are blank).
        self._class_shortcuts: List[QShortcut] = []
        for i in range(len(CLASSES_NAMES_STEREOTYPE)):
            sc = QShortcut(QKeySequence(str(i + 1)), self)
            sc.activated.connect(lambda idx=i: self._pick_class(idx))
            self._class_shortcuts.append(sc)

        # --- CORRECTION (clear) + VALIDATE (accept forecast) in a white card ---
        actions_card = QFrame()
        actions_card.setObjectName("card")
        av = QVBoxLayout(actions_card)
        av.setContentsMargins(14, 12, 14, 12)
        av.setSpacing(10)
        av.addLayout(self._build_correction_row())
        av.addLayout(self._build_validate_row())

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(12)
        controls_layout.addWidget(actions_card)
        controls_layout.addWidget(classes_card)
        controls_layout.addStretch(1)

        right_splitter.addWidget(controls_widget)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        # Give the frame list a wider default and allow it to be dragged much
        # wider (the names are long), instead of being capped narrow.
        main_splitter.setSizes([460, 1040])
        root.addWidget(main_splitter, 1)
        # The frame-list width and the viewer height the operator dragged
        # to are remembered, like the regeneration editor's.
        self._main_splitter = main_splitter
        self._right_splitter = right_splitter
        self._restore_editor_splitters()
        main_splitter.splitterMoved.connect(
            lambda *_: self._save_editor_splitters())
        right_splitter.splitterMoved.connect(
            lambda *_: self._save_editor_splitters())

        # Keyboard navigation: arrow keys move through the frames.
        # Strong Python references for the QShortcut wrappers so the
        # shiboken garbage collector does not drop them while the
        # parent C++ side is still alive (Qt for Python bug
        # PYSIDE-2131).
        self._shortcuts: List[QShortcut] = []
        for sequence, handler in (
            (Qt.Key_Up, self._select_previous),
            (Qt.Key_Down, self._select_next),
            (Qt.Key_Right, self._select_next),
            (Qt.Key_Left, self._select_previous),
            # Enter validates the current frame (same as the Validate button).
            (Qt.Key_Return, self._on_enter_validate),
            (Qt.Key_Enter, self._on_enter_validate),
        ):
            sc = QShortcut(QKeySequence(sequence), self)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

        # The row/status line lives only at the top of the viewer (self._row_info),
        # like the Regeneration states editor; the redundant bottom label is
        # removed. Status messages (cleared/validated counts, errors) are routed
        # to that same top label.
        self._status_label = self._row_info

    def _on_enter_validate(self) -> None:
        """Enter validates the current frame, unless a text field is focused."""
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            return
        self._on_validate()

    def _pick_class(self, idx: int) -> None:
        """Select the correction class ``idx`` via its number key.

        Ignored while a text field or spin box has focus so typing a digit in
        the path/CSV fields does not change a correction.
        """
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            return
        if 0 <= idx < len(self._correction_radios):
            self._correction_radios[idx].click()

    @staticmethod
    def _column_header(title: str, subtitle: str) -> QWidget:
        """A two-line centred column header (bold title + muted subtitle)."""
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        t = QLabel(title)
        t.setObjectName("colHeader")
        t.setAlignment(Qt.AlignCenter)
        s = QLabel(subtitle)
        s.setObjectName("mutedLabel")
        s.setAlignment(Qt.AlignCenter)
        v.addWidget(t)
        v.addWidget(s)
        return w

    # ------------------------------------------------------------------
    # Correction / Validate option rows (above the class radios)
    # ------------------------------------------------------------------
    def _build_correction_row(self) -> QHBoxLayout:
        """Clear buttons: current frame, current video, whole file."""
        row = QHBoxLayout()
        row.setSpacing(10)
        caption = QLabel("Correction")
        caption.setObjectName("caption")
        row.addWidget(caption)
        self._clear_btn = QPushButton("Clear current correction")
        self._clear_btn.setToolTip(
            "Remove the manual correction of the current frame (same as "
            "clicking its already-selected class)."
        )
        self._clear_btn.clicked.connect(self._on_clear_current)
        row.addWidget(self._clear_btn)
        self._clear_video_btn = QPushButton("Clear all current video corrections")
        self._clear_video_btn.setToolTip(
            "Remove every manual correction of the current video only."
        )
        self._clear_video_btn.clicked.connect(self._on_clear_video)
        row.addWidget(self._clear_video_btn)
        self._clear_all_btn = QPushButton("Clear all file corrections")
        self._clear_all_btn.setObjectName("dangerButton")
        self._clear_all_btn.setToolTip(
            "Remove every manual correction from the whole CSV (all videos)."
        )
        self._clear_all_btn.clicked.connect(self._on_clear_all)
        row.addWidget(self._clear_all_btn)
        row.addStretch(1)
        return row

    def _build_validate_row(self) -> QHBoxLayout:
        """Validate buttons: accept the forecast as the correction."""
        row = QHBoxLayout()
        row.setSpacing(10)
        caption = QLabel("Validate")
        caption.setObjectName("caption")
        row.addWidget(caption)
        self._validate_btn = QPushButton("Validate forecast, current image")
        self._validate_btn.setToolTip(
            "Copy the forecasted class to the corrected column for the current "
            "frame. With Auto-skip on, jumps to the next frame."
        )
        self._validate_btn.clicked.connect(self._on_validate)
        row.addWidget(self._validate_btn)
        self._auto_skip = ToggleSwitch("Auto-skip")
        self._auto_skip.setToolTip(
            "When on, validating or correcting a frame automatically moves to "
            "the next one."
        )
        self._auto_skip.setChecked(
            bool(self._settings.get("step5_auto_advance", True))
        )
        self._auto_skip.toggled.connect(
            lambda v: self._settings.set("step5_auto_advance", bool(v))
        )
        row.addWidget(self._auto_skip)
        self._validate_video_btn = QPushButton("Validate forecast, all current video")
        self._validate_video_btn.setToolTip(
            "Validate the forecast of every not-yet-corrected frame of the "
            "current video."
        )
        self._validate_video_btn.clicked.connect(self._on_validate_video)
        row.addWidget(self._validate_video_btn)
        self._validate_all_btn = QPushButton("Validate forecast, all file")
        self._validate_all_btn.setObjectName("validateAllButton")
        self._validate_all_btn.setToolTip(
            "Validate the forecast of every not-yet-corrected frame in the "
            "whole CSV (all videos)."
        )
        self._validate_all_btn.clicked.connect(self._on_validate_all)
        row.addWidget(self._validate_all_btn)
        row.addStretch(1)
        return row

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_editor_splitters(self) -> None:
        """Apply the saved frame-list width and viewer height."""
        try:
            tw = int(self._settings.get("step5_tree_width", 0) or 0)
            if tw > 120:
                total = sum(self._main_splitter.sizes()) or 1500
                self._main_splitter.setSizes([tw, max(200, total - tw)])
            vh = int(self._settings.get("step5_viewer_height", 0) or 0)
            if vh > 120:
                total = sum(self._right_splitter.sizes()) or 920
                self._right_splitter.setSizes([vh, max(120, total - vh)])
        except Exception:  # noqa: BLE001 - layout memory must never break the page
            pass

    def _save_editor_splitters(self) -> None:
        try:
            ms = self._main_splitter.sizes()
            if ms and ms[0] > 0:
                self._settings.set("step5_tree_width", int(ms[0]))
            rs = self._right_splitter.sizes()
            if rs and rs[0] > 0:
                self._settings.set("step5_viewer_height", int(rs[0]))
        except Exception:  # noqa: BLE001
            pass

    def _restore_from_settings(self) -> None:
        from ..constants import PROCESSING_ROOT_DEFAULT

        proc_root = (
            str(self._settings.get("step5_processing_root", "")).strip()
            or str(self._settings.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._input_edit.setText(proc_root)

        saved_csv = str(self._settings.get("step5_csv_path", "")).strip()
        if not saved_csv and proc_root:
            candidate = os.path.join(proc_root, "all_videos.csv")
            if os.path.isfile(candidate):
                saved_csv = candidate
        self._csv_edit.setText(saved_csv)

        # Auto-skip preference is restored where the toggle is built
        # (``_build_validate_row``); default is ON.

        # Auto-load the saved Results CSV silently when the page
        # opens (only if the file still exists on disk).  When
        # missing or absent the page stays empty, no error banner --
        # per user request, this is a convenience feature, not a
        # required load.  Reference for the "recognition rather than
        # recall" pattern that motivates auto-reloading the last
        # successful state: Nielsen, J., "10 Usability Heuristics
        # for User Interface Design", 1994,
        # https://www.nngroup.com/articles/ten-usability-heuristics/,
        # heuristic 6.
        if saved_csv and os.path.isfile(saved_csv):
            try:
                self._load_csv(saved_csv)
            except Exception:  # noqa: BLE001
                # Silent: the user explicitly asked for no error
                # dialog when the auto-load path is missing or
                # malformed; the manual Load button remains
                # available for explicit retry.
                pass

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _on_input_changed(self, text: str) -> None:
        # Per-page processing root, saved locally as soon as the field changes.
        self._settings.set("step5_processing_root", text)

    def showEvent(self, event) -> None:  # noqa: N802
        """Auto-load the latest results CSV when the page becomes visible.

        Pipelines A and B save the merged ``all_videos.csv`` path (and the
        dataset root) on completion; switching to this editor afterwards loads
        it without a manual Load click, when it is not already loaded.
        """
        super().showEvent(event)
        root = str(self._settings.get("step5_processing_root", "")).strip()
        if root and root != self._input_edit.text().strip():
            self._input_edit.setText(root)
        saved = str(self._settings.get("step5_csv_path", "")).strip()
        current = self._current_csv_path or ""
        if (
            saved
            and os.path.isfile(saved)
            and os.path.abspath(saved) != os.path.abspath(current)
        ):
            self._csv_edit.setText(saved)
            try:
                self._load_csv(saved)
            except Exception:  # noqa: BLE001
                pass

    def _on_browse_input(self) -> None:
        start = self._input_edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select processing root", start,
        )
        if path:
            self._input_edit.setText(path)

    def _on_browse_csv(self) -> None:
        start = self._csv_edit.text() or self._input_edit.text() or os.getcwd()
        if start and not os.path.isdir(start):
            start = os.path.dirname(start) or os.getcwd()
        path, _ = get_open_file_name(
            self, "Select results CSV", start, "CSV files (*.csv)",
        )
        if path:
            self._csv_edit.setText(path)

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------
    def _on_load(self) -> None:
        csv_path = self._csv_edit.text().strip()
        if not csv_path or not os.path.isfile(csv_path):
            _msg_critical(
                self, "Step 5",
                "Select an existing CSV file to load.",
            )
            return
        self._settings.set("step5_csv_path", csv_path)
        self._load_csv(csv_path)

    def _on_regenerate_merged(self) -> None:
        """Rebuild ``all_videos.csv`` from every per-video CSV under the root.

        The handler calls
        :func:`flatmorphai.notebook_api.periods.fuse_periods_by_subject`
        on the configured processing root, points the CSV line edit
        at the freshly written file and reloads the editor. The
        operation is pure pandas IO so we run it inline; if it ever
        becomes slow enough to block the GUI thread it can be moved
        to a worker without changing the surrounding logic.
        """
        from ..notebook_api.periods import fuse_periods_by_subject

        root = self._input_edit.text().strip()
        if not root or not os.path.isdir(root):
            _msg_critical(
                self, "Step 5",
                "Set a valid processing root before regenerating "
                "all_videos.csv.",
            )
            return
        new_index = int(
            self._settings.get("step6_new_planaria_index", 1)
        )
        try:
            df = fuse_periods_by_subject(
                processing_root=root,
                output_directory=root,
                output_csv_name="all_videos.csv",
                new_planaria_start_index=new_index,
                log=lambda msg: None,
            )
        except Exception as exc:  # noqa: BLE001
            _msg_critical(
                self, "Step 5",
                f"Regeneration failed: {type(exc).__name__}: {exc}",
            )
            return
        if df is None:
            _msg_warning(
                self, "Step 5",
                "No per-video results CSV was found under the "
                "processing root; all_videos.csv was not written.",
            )
            return
        merged_path = os.path.join(root, "all_videos.csv")
        self._csv_edit.setText(merged_path)
        self._settings.set("step5_csv_path", merged_path)
        self._load_csv(merged_path)
        self._status_label.setText(
            f"Regenerated all_videos.csv: {len(df)} row(s) across "
            f"{df['video_name'].nunique() if 'video_name' in df.columns else 0} video(s)."
        )

    def _load_csv(self, csv_path: str) -> None:
        # Per-video Step 4 outputs are written as utf-8 (so the unicode
        # micro letter in ``3000uM`` style concentrations survives
        # round-trips); Step 6's merged CSV is written as latin1 for
        # backward compatibility with notebook 6. Try utf-8 first
        # because reading a utf-8 file as latin1 silently mojibakes
        # the non-ASCII characters (latin1 accepts every byte
        # sequence), whereas reading a latin1 file as utf-8 raises
        # ``UnicodeDecodeError`` so the fallback always succeeds when
        # the file is genuinely latin1. Reference: Spolsky, J.,
        # "The Absolute Minimum Every Software Developer Must Know
        # About Unicode and Character Sets", 2003 (no DOI, identifier:
        # joelonsoftware.com/2003/10/08).
        df: Optional[pd.DataFrame] = None
        for encoding in ("utf-8", CSV_ENCODING_LATIN1):
            try:
                df = pd.read_csv(csv_path, sep=CSV_SEPARATOR, encoding=encoding)
                break
            except Exception:  # noqa: BLE001
                continue
        if df is None:
            _msg_critical(
                self, "Step 5",
                f"Cannot read {csv_path}. Verify it uses ';' as separator.",
            )
            return
        self._current_csv_path = csv_path
        self._current_df = df
        self._video_dir_cache.clear()
        self._row_items.clear()

        self._forecast_idx_col, self._forecast_name_col = (
            self._resolve_columns(df, prefix="forecasted")
        )
        self._correction_idx_col, self._correction_name_col = (
            self._resolve_columns(df, prefix="corrected")
        )

        # Step 4 always emits both forecasted and corrected columns. If
        # the loaded file is missing the corrected ones (very old run
        # or a hand-edited file), append empty columns and persist.
        added = False
        if self._correction_idx_col is None and self._forecast_idx_col is not None:
            new_idx = self._forecast_idx_col.replace(
                "forecasted", "corrected"
            )
            df[new_idx] = ""
            self._correction_idx_col = new_idx
            added = True
        if self._correction_name_col is None and self._forecast_name_col is not None:
            new_name = self._forecast_name_col.replace(
                "forecasted", "corrected"
            )
            df[new_name] = ""
            self._correction_name_col = new_name
            added = True
        if added:
            try:
                df.to_csv(
                    csv_path,
                    sep=CSV_SEPARATOR,
                    encoding=CSV_ENCODING_LATIN1,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
            except Exception as exc:  # noqa: BLE001
                _msg_warning(
                    self, "Step 5",
                    f"Could not write the new corrected columns to "
                    f"{csv_path}: {exc}",
                )

        self._populate_tree()
        self._update_counter()
        self._status_label.setText(
            f"Loaded {len(df)} row(s) from {os.path.basename(csv_path)}"
        )

    @staticmethod
    def _resolve_columns(
        df: pd.DataFrame,
        prefix: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(index_col, name_col)`` for ``forecasted`` or ``corrected``."""
        idx_col: Optional[str] = None
        name_col: Optional[str] = None
        for col in df.columns:
            lower = col.lower()
            if lower.startswith(prefix.lower()):
                if "index" in lower and idx_col is None:
                    idx_col = col
                elif "name" in lower and name_col is None:
                    name_col = col
        return idx_col, name_col

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------
    def _populate_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        self._row_items.clear()
        if self._current_df is None:
            self._tree.blockSignals(False)
            return

        # Group rows by ``video_name`` while preserving the original
        # ordering. Each top-level node is one video; its children are
        # the individual frames.
        per_video: Dict[str, List[int]] = {}
        for idx in range(len(self._current_df)):
            video = str(self._current_df.iloc[idx].get("video_name", "")).strip()
            if not video:
                # Fall back to the image-name minus the trailing
                # ``_<second>s`` token so frames without a video_name
                # still cluster correctly.
                image_name = str(
                    self._current_df.iloc[idx].get("image_name", "")
                ).strip()
                base, _ext = os.path.splitext(image_name)
                video = base.rsplit("_", 1)[0] or "(unknown)"
            per_video.setdefault(video, []).append(idx)

        for video, indices in per_video.items():
            corrected = sum(
                1 for i in indices
                if self._is_row_corrected(self._current_df.iloc[i])
            )
            top = QTreeWidgetItem(
                [self._video_label(video, len(indices), corrected)]
            )
            top.setData(0, _VIDEO_ROOT_ROLE, video)
            top.setFirstColumnSpanned(True)
            self._tree.addTopLevelItem(top)
            for row_idx in indices:
                series = self._current_df.iloc[row_idx]
                child = QTreeWidgetItem([self._row_label(series)])
                child.setData(0, _ROW_INDEX_ROLE, row_idx)
                self._apply_row_colour(child, series)
                top.addChild(child)
                self._row_items[row_idx] = child
            top.setExpanded(True)
        self._tree.blockSignals(False)
        # Auto-select the first leaf so the editor lands on a row.
        first_leaf = self._first_leaf()
        if first_leaf is not None:
            self._tree.setCurrentItem(first_leaf)

    def _video_label(self, video: str, total: int, corrected: int) -> str:
        return f"{video}  ({corrected}/{total} manually corrected)"

    def _row_label(self, series: pd.Series) -> str:
        image_name = str(series.get("image_name", ""))
        period = str(series.get("period", "")).strip()
        forecasted = self._cell(series, self._forecast_name_col)
        corrected = self._cell(series, self._correction_name_col)
        # Show ONE class tag before the filename: the corrected value
        # takes precedence over the forecast when present, otherwise
        # the forecast is displayed.  The row's green foreground still
        # signals "corrected" so no extra "-> corrected" arrow is
        # needed alongside the tag.  Reference for the "single source
        # of truth per row" pattern: Wilkinson, M.D. et al., "The FAIR
        # Guiding Principles for scientific data management and
        # stewardship", Sci Data 3, 160018, 2016, DOI:
        # 10.1038/sdata.2016.18 (rule F2 -- data are described with
        # rich metadata in-band, not duplicated across columns).
        effective = corrected or forecasted or "?"
        prefix = f"[{effective}] "
        period_token = f"  ({period})" if period else ""
        return f"{prefix}{image_name}{period_token}"

    def _apply_row_colour(
        self, item: QTreeWidgetItem, series: pd.Series,
    ) -> None:
        if self._is_row_corrected(series):
            item.setForeground(0, QBrush(_CORRECTED_FG))
            # A green check before the name flags the validated/corrected row.
            item.setIcon(0, check_icon(16))
        else:
            # ``QTreeWidgetItem.setForeground`` with a default brush
            # restores the palette colour, so a row that gets
            # un-corrected (cancelled) reverts to the theme default.
            item.setForeground(0, QBrush())
            # A transparent placeholder keeps the icon column aligned.
            item.setIcon(0, blank_icon(16))

    def _is_row_corrected(self, series: pd.Series) -> bool:
        return bool(self._cell(series, self._correction_name_col))

    @staticmethod
    def _cell(series: pd.Series, column: Optional[str]) -> str:
        if column is None or column not in series:
            return ""
        value = series[column]
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _first_leaf(self) -> Optional[QTreeWidgetItem]:
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            if top.childCount() > 0:
                return top.child(0)
        return None

    # ------------------------------------------------------------------
    # Selection / navigation
    # ------------------------------------------------------------------
    def _on_tree_selected(
        self,
        current: Optional[QTreeWidgetItem],
        _previous: Optional[QTreeWidgetItem],
    ) -> None:
        if current is None or self._current_df is None:
            return
        row = current.data(0, _ROW_INDEX_ROLE)
        if row is None:
            return
        self._current_row_index = int(row)
        series = self._current_df.iloc[self._current_row_index]
        self._refresh_preview(series)
        self._refresh_radios(series)
        image_name = str(series.get("image_name", ""))
        period = str(series.get("period", "")).strip()
        period_txt = ("(%s) " % period) if period and period.lower() != "nan" else ""
        row_line = (
            "Row %d / %d: %s%s"
            % (self._current_row_index + 1, len(self._current_df),
               period_txt, image_name)
        )
        self._row_info.setText(row_line)
        self._row_info_short.setText(
            "%d / %d" % (self._current_row_index + 1, len(self._current_df))
        )

    def _select_previous(self) -> None:
        leaf = self._sibling_leaf(-1)
        if leaf is not None:
            self._tree.setCurrentItem(leaf)

    def _select_next(self) -> None:
        leaf = self._sibling_leaf(+1)
        if leaf is not None:
            self._tree.setCurrentItem(leaf)

    def _sibling_leaf(self, direction: int) -> Optional[QTreeWidgetItem]:
        """Return the next or previous frame leaf in tree order."""
        # Flatten the tree once; keeping the cost constant per
        # navigation matters only for very large CSVs.
        leaves: List[QTreeWidgetItem] = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            for j in range(top.childCount()):
                leaves.append(top.child(j))
        if not leaves:
            return None
        current = self._tree.currentItem()
        if current is None or current.data(0, _ROW_INDEX_ROLE) is None:
            return leaves[0] if direction > 0 else leaves[-1]
        try:
            idx = leaves.index(current)
        except ValueError:
            return leaves[0]
        new_idx = idx + direction
        if 0 <= new_idx < len(leaves):
            return leaves[new_idx]
        return None

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _three_s_enabled(self) -> bool:
        """True when the FORMATED pane should show the 3s composite."""
        viewer = getattr(self, "_viewer", None)
        toggle = getattr(viewer, "right_toggle", None)
        if toggle is None:
            return False
        return bool(toggle.isChecked())

    def _on_three_s_toggled(self, checked: bool) -> None:
        """Persist the choice and redraw the current row."""
        self._settings.set("step5_3s_display", bool(checked))
        if (
            self._current_df is not None
            and 0 <= self._current_row_index < len(self._current_df)
        ):
            # Same frame, only the FORMATED rendering changed: keep zoom/pan.
            self._refresh_preview(
                self._current_df.iloc[self._current_row_index],
                preserve_view=True,
            )

    def _refresh_preview(self, series: pd.Series, preserve_view: bool = False) -> None:
        root = self._input_edit.text().strip()
        image_name = str(series.get("image_name", "")).strip()
        if not root or not image_name:
            self._viewer.clear()
            return
        video_dir = self._resolve_video_dir(root, series)
        if video_dir is None:
            self._viewer.clear()
            self._status_label.setText(
                f"Could not locate the per-video directory for "
                f"{image_name}. Check the processing root."
            )
            return
        frame_path = self._find_frame(video_dir, image_name)
        # Right pane: either the 3-second colour composite (when the
        # "3s display" toggle is on) or the single-frame crop.
        # ``_resolve_formated`` falls back to rebuilding the crop on
        # the fly (into ``temp_formated/``) when Step 3 never wrote
        # one -- including the case where the formatted folder does
        # not exist at all.  Whatever either branch returns is a real
        # file on disk, which is exactly what the viewer's Open
        # button needs in order to reveal the right entry in the file
        # explorer.
        if self._three_s_enabled():
            formated_path = self._build_three_second_image(
                video_dir, image_name,
            )
            if formated_path is None:
                # No usable neighbour / mask -- fall back to the
                # single-frame crop rather than showing nothing.
                formated_path = self._resolve_formated(
                    video_dir, image_name,
                )
        else:
            formated_path = self._resolve_formated(video_dir, image_name)

        # Why the Open button used to do nothing on FORMATED: the
        # toolbar handler is ``if path: reveal_in_explorer(path)``,
        # so a ``None`` source path makes the button a SILENT no-op.
        # ``_find_frame`` always returned an existing file, hence
        # FRAME worked; ``_find_formated`` returned ``None`` as soon
        # as Step 3 had not written that crop, hence FORMATED did
        # not.  ``_resolve_formated`` now always yields a real file,
        # which fixes the button.  Absolute paths below are belt and
        # braces: the processing root is often typed in relative
        # form, and an absolute path is what the WSL ``wslpath -w``
        # translation inside ``reveal_in_explorer`` expects.
        if frame_path:
            frame_path = os.path.abspath(frame_path)
        if formated_path:
            formated_path = os.path.abspath(formated_path)

        self._viewer.update_left(
            QPixmap(frame_path) if frame_path else QPixmap(),
            source_path=frame_path, preserve_view=preserve_view,
        )
        self._viewer.update_right(
            QPixmap(formated_path) if formated_path else QPixmap(),
            source_path=formated_path, preserve_view=preserve_view,
        )

    def _resolve_video_dir(
        self, root: str, series: pd.Series,
    ) -> Optional[str]:
        stem = str(series.get("video_name", "")).strip()
        if not stem:
            image_name = str(series.get("image_name", "")).strip()
            if image_name:
                base, _ext = os.path.splitext(image_name)
                stem = base.rsplit("_", 1)[0]
        if not stem:
            return None
        if stem in self._video_dir_cache:
            return self._video_dir_cache[stem]
        # Fallback when the loaded CSV is a per-video file rather than
        # the merged all_videos.csv: the per-video directory is the
        # CSV's own parent. We accept any entry under that parent that
        # contains the expected image_name on disk; if found we cache
        # the result keyed by the row's video_name so subsequent rows
        # in the same file resolve in O(1) without re-walking the tree.
        csv_path = self._current_csv_path or ""
        if csv_path:
            parent = os.path.dirname(csv_path)
            if os.path.isdir(parent):
                image_name = str(series.get("image_name", "")).strip()
                if image_name and os.path.isfile(
                    os.path.join(parent, "images", image_name)
                ):
                    self._video_dir_cache[stem] = parent
                    return parent
        if not os.path.isdir(root):
            return None
        for entry in os.listdir(root):
            full = os.path.join(root, entry)
            if not os.path.isdir(full):
                continue
            if _looks_like_video_dir(entry, stem):
                self._video_dir_cache[stem] = full
                return full
        return None

    _FRAME_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    @staticmethod
    def _find_frame(video_dir: str, image_name: str) -> Optional[str]:
        """Absolute path of the ORIGINAL frame, resolved robustly.

        The previous version only accepted the EXACT path
        ``<video_dir>/images/<image_name>`` and returned ``None`` otherwise,
        which made the FRAME pane's Open button a silent no-op whenever the
        file on disk differs from the CSV ``image_name`` by extension or case,
        or the originals are not under an ``images/`` sub-folder. The FORMATED
        pane never hit this because ``_find_formated`` scans the folders.
        We now mirror that robustness: exact match first, then same stem with
        any image extension in ``images/``, then the name directly under the
        video folder, then a bounded scan of the video folder's sub-dirs
        (skipping the formatted and mask folders so a mask is never revealed in
        place of the original). Returns ``None`` only when nothing matches."""
        def _stem_match(folder: str) -> Optional[str]:
            if not os.path.isdir(folder):
                return None
            exact = os.path.join(folder, image_name)
            if os.path.isfile(exact):
                return exact
            stem = os.path.splitext(image_name)[0]
            # same stem + a known image extension, then any case-insensitive hit
            for ext in Step5Page._FRAME_EXTS:
                c = os.path.join(folder, stem + ext)
                if os.path.isfile(c):
                    return c
            try:
                for e in os.listdir(folder):
                    if (os.path.splitext(e)[0].lower() == stem.lower()
                            and os.path.splitext(e)[1].lower() in Step5Page._FRAME_EXTS
                            and os.path.isfile(os.path.join(folder, e))):
                        return os.path.join(folder, e)
            except OSError:
                return None
            return None

        # 1. the canonical location, exact or by stem.
        hit = _stem_match(os.path.join(video_dir, "images"))
        if hit:
            return hit
        # 2. some layouts keep the originals at the video-folder root.
        hit = _stem_match(video_dir)
        if hit:
            return hit
        # 3. bounded scan of the other sub-dirs, skipping formatted/mask
        #    folders so the original wins over its crop or its mask.
        try:
            entries = sorted(os.listdir(video_dir))
        except OSError:
            return None
        for entry in entries:
            low = entry.lower()
            if _is_formated_dir(entry) or "mask" in low or "masque" in low:
                continue
            sub = os.path.join(video_dir, entry)
            if os.path.isdir(sub):
                hit = _stem_match(sub)
                if hit:
                    return hit
        return None

    # Folder that holds the crops rebuilt on the fly.  It lives ONCE
    # at the application root rather than inside each video folder,
    # so browsing a dataset never litters it with per-video cache
    # directories.
    TEMP_FORMATED_DIRNAME = "temp_formated"
    # Prefix for the cached 3-second composite so it never collides
    # with the single-frame crop of the same frame.
    THREE_S_PREFIX = "3s_"

    @staticmethod
    def _app_root() -> str:
        """Absolute path of the PySide application root.

        ``step5_page.py`` lives at ``<root>/flatmorphai/pages/``, so
        the root is two levels up from the package directory.
        """
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    def _temp_formated_path(
        self, video_dir: str, image_name: str, three_s: bool = False,
    ) -> str:
        """Return the cache path for one rebuilt crop.

        The file name is prefixed with the video folder's name so two
        videos that contain an identically-named frame (very common:
        the frame name repeats the video stem, but a renamed copy of
        a dataset would not) cannot overwrite each other's entry.
        """
        video_tag = os.path.basename(os.path.normpath(video_dir))
        prefix = self.THREE_S_PREFIX if three_s else ""
        return os.path.join(
            self._app_root(),
            self.TEMP_FORMATED_DIRNAME,
            f"{prefix}{video_tag}__{image_name}",
        )

    def _build_three_second_image(
        self, video_dir: str, image_name: str,
    ) -> Optional[str]:
        """Build the 3-second colour composite for ``image_name``.

        The current frame and its two temporal neighbours are packed
        into one RGB image sharing a single crop window:

            Red   = T-1s      Green = T0 (current)      Blue = T+1s

        A stationary worm therefore renders grey; any displacement
        shows up as coloured fringes on the leading and trailing
        edges.  This is the same construction the old temporal
        pipeline used (``fusion_tsm.temporal_format``): each frame is
        masked to the worm, the three masks are unioned into one
        square box, and all three are cropped with that single box so
        real motion is preserved rather than re-centred away.

        Note on channel order: the old pipeline stored the triple as
        OpenCV BGR ``[T-1, T0, T+1]``, which renders as Red = T+1 and
        Blue = T-1 -- the opposite of the mapping requested here.  We
        emit the requested Red = T-1 / Blue = T+1 order.

        At a clip boundary the missing neighbour falls back to T0
        (that channel becomes identical to green), so the first and
        last frames still render instead of failing.

        Returns the cached file path, or ``None`` when the frame or
        its mask cannot be read.
        """
        import cv2
        import numpy as np

        from ..notebook_api.classification_images import get_mask_folders
        from ..notebook_api.fusion_tsm.temporal_format import (
            build_temporal_image,
        )
        from ..notebook_api.fusion_tsm.nomenclature import app_second

        images_dir = os.path.join(video_dir, "images")
        if not os.path.isdir(images_dir):
            return None

        # Order the clip by acquisition second and locate the current
        # frame by RANK, so sub-second sampling works unchanged.
        frames: List[Tuple[float, str]] = []
        for fn in os.listdir(images_dir):
            if not fn.lower().endswith(".png"):
                continue
            sec = app_second(fn)
            if sec is not None:
                frames.append((float(sec), fn))
        if not frames:
            return None
        frames.sort(key=lambda t: (t[0], t[1]))
        names = [fn for _s, fn in frames]
        try:
            idx = names.index(image_name)
        except ValueError:
            return None
        prev_name = names[idx - 1] if idx > 0 else None
        next_name = names[idx + 1] if idx + 1 < len(names) else None

        # Resolve the mask folder once.
        mask_dirs = get_mask_folders(video_dir)
        mask_dirs.sort(reverse=True)
        if not mask_dirs:
            return None

        def _imread(path: str):
            try:
                return cv2.imdecode(
                    np.fromfile(path, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
            except Exception:  # noqa: BLE001
                return None

        def _pair(fn: Optional[str]):
            """Return (gray, mask) for one frame, or (None, None)."""
            if fn is None:
                return None, None
            gray = _imread(os.path.join(images_dir, fn))
            if gray is None:
                return None, None
            for entry in mask_dirs:
                mp = os.path.join(video_dir, entry, fn)
                if os.path.isfile(mp):
                    mask = _imread(mp)
                    if mask is not None:
                        return gray, mask
            return None, None

        g_cur, m_cur = _pair(image_name)
        if g_cur is None or m_cur is None:
            return None
        g_prev, m_prev = _pair(prev_name)
        g_next, m_next = _pair(next_name)

        try:
            composite = build_temporal_image(
                grays=(g_prev, g_cur, g_next),
                masks=(m_prev, m_cur, m_next),
                image_size=int(
                    self._settings.get("step3_image_size", 384)
                ),
            )
        except Exception:  # noqa: BLE001
            return None
        if composite is None:
            return None

        # ``build_temporal_image`` returns BGR = [T-1, T0, T+1], i.e.
        # Red = T+1.  Swap the outer channels to get the requested
        # Red = T-1 / Green = T0 / Blue = T+1.
        composite = composite[:, :, ::-1]

        out_path = self._temp_formated_path(
            video_dir, image_name, three_s=True,
        )
        out_dir = os.path.dirname(out_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            return None
        # Same single-slot cache policy as the plain crop: moving to
        # another frame replaces the stored image.
        try:
            for stale in os.listdir(out_dir):
                if stale != os.path.basename(out_path):
                    try:
                        os.remove(os.path.join(out_dir, stale))
                    except OSError:
                        pass
        except OSError:
            pass
        try:
            ok, buf = cv2.imencode(".png", composite)
            if not ok:
                return None
            buf.tofile(out_path)
        except Exception:  # noqa: BLE001
            return None
        return out_path if os.path.isfile(out_path) else None

    def _resolve_formated(
        self, video_dir: str, image_name: str,
    ) -> Optional[str]:
        """Return a displayable FORMATED path for ``image_name``.

        Resolution order:

        1. A crop left on disk by an older run (any ``Formated*`` /
           ``Regeneration_process_of_Planaria*`` folder).  No
           pipeline writes these any more, but existing datasets
           still carry them.
        2. A crop previously rebuilt into the application-root
           ``temp_formated/`` cache.
        3. A crop built right now from the raw frame plus its Step 2
           mask, written into that same cache.

        Returns ``None`` only when even the on-demand build is
        impossible (no frame, no mask, or an unreadable pair).  A
        missing formatted folder is NOT an error -- it simply falls
        through to the on-demand branch.
        """
        stored = self._find_formated(video_dir, image_name)
        if stored:
            return stored
        temp_path = self._temp_formated_path(video_dir, image_name)
        if os.path.isfile(temp_path):
            return temp_path
        return self._build_formated_on_demand(
            video_dir, image_name, temp_path,
        )

    def _build_formated_on_demand(
        self, video_dir: str, image_name: str, out_path: str,
    ) -> Optional[str]:
        """Crop ``image_name`` the way Step 3 would and cache it.

        Uses the SAME formatter Step 3 calls
        (:func:`compute_mask_image_planarian_reshaped`) with the
        operator's current Step 3 settings, so an on-demand crop is
        pixel-identical to a stored one.

        The result is written to ``temp_formated/<image_name>``.  The
        folder is pruned first so it never grows past a single frame:
        the operator asked that moving to another frame replace the
        cached image rather than accumulate one file per visited
        frame.
        """
        import cv2
        import numpy as np

        from ..notebook_api.classification_images import (
            biggest_objects_in_the_binary_mask,
            compute_mask_image_planarian_reshaped,
            get_mask_folders,
        )

        frame_path = self._find_frame(video_dir, image_name)
        if not frame_path:
            return None

        # Locate the matching Step 2 mask.  Mask folders are named
        # ``masks_<segmentation model>``; take the most recent one
        # that actually holds this frame.
        mask_path: Optional[str] = None
        mask_dirs = get_mask_folders(video_dir)
        mask_dirs.sort(reverse=True)
        for entry in mask_dirs:
            candidate = os.path.join(video_dir, entry, image_name)
            if os.path.isfile(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            return None

        def _imread(path: str, flag: int):
            # cv2.imread chokes on non-ASCII paths on Windows; go
            # through numpy so accented folder names keep working.
            try:
                return cv2.imdecode(
                    np.fromfile(path, dtype=np.uint8), flag,
                )
            except Exception:  # noqa: BLE001
                return None

        original = _imread(frame_path, cv2.IMREAD_UNCHANGED)
        mask = _imread(mask_path, cv2.IMREAD_UNCHANGED)
        if original is None or mask is None:
            return None

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(
                mask, (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        try:
            planaria_masks = biggest_objects_in_the_binary_mask(mask, 1, 1)
        except Exception:  # noqa: BLE001
            return None
        if not planaria_masks or planaria_masks[0] is None:
            return None

        s = self._settings
        try:
            crop = compute_mask_image_planarian_reshaped(
                original,
                planaria_masks[0],
                int(s.get("step3_image_size", 384)),
                float(s.get("step3_margin_percent", 0.0)),
                apply_mask=bool(s.get("step3_apply_mask", True)),
                black_and_white=bool(
                    s.get("step3_black_and_white", False)
                ),
            )
        except Exception:  # noqa: BLE001
            return None
        if crop is None:
            return None

        out_dir = os.path.dirname(out_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            return None
        # Keep exactly one cached crop: drop whatever was rebuilt for
        # the previously displayed frame.
        try:
            for stale in os.listdir(out_dir):
                if stale != os.path.basename(out_path):
                    try:
                        os.remove(os.path.join(out_dir, stale))
                    except OSError:
                        pass
        except OSError:
            pass
        try:
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                return None
            buf.tofile(out_path)
        except Exception:  # noqa: BLE001
            return None
        return out_path if os.path.isfile(out_path) else None

    @staticmethod
    def _find_formated(video_dir: str, image_name: str) -> Optional[str]:
        try:
            entries = os.listdir(video_dir)
        except (FileNotFoundError, NotADirectoryError):
            return None
        candidates: List[str] = []
        for entry in entries:
            if _is_formated_dir(entry):
                candidate = os.path.join(video_dir, entry, image_name)
                if os.path.isfile(candidate):
                    candidates.append(candidate)
        if not candidates:
            return None
        # Prefer the temporal folder over the standard one when both
        # exist (Pipeline A + FlatMorphAI bundle always writes the
        # temporal form; a stale ``Formated_image-size(...)_...`` dir
        # from a prior non-temporal run would otherwise win by
        # inode order and Open would reveal the wrong image).
        temporal = [c for c in candidates if "_temporal_" in os.path.basename(
            os.path.dirname(c)
        )]
        if temporal:
            candidates = temporal
        # Among the surviving candidates, pick the most recently
        # modified so a fresh Pipeline A rerun's output always wins
        # over any stale prior-run directory.  Reference for the
        # "freshness-wins" rule when duplicate artefacts exist:
        # Feldman, S.I., "Make: a program for maintaining computer
        # programs", Software: Practice and Experience 9(4),
        # 255-265, 1979, DOI:10.1002/spe.4380090402.
        try:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        except OSError:
            pass
        return candidates[0]

    # ------------------------------------------------------------------
    # Radios
    # ------------------------------------------------------------------
    def _refresh_radios(self, series: pd.Series) -> None:
        forecast_value = self._cell(series, self._forecast_name_col)
        if forecast_value not in CLASSES_NAMES_STEREOTYPE:
            forecast_value = ""
            forecast_idx = self._cell(series, self._forecast_idx_col)
            if forecast_idx.isdigit():
                idx = int(forecast_idx)
                if 0 <= idx < len(CLASSES_NAMES_STEREOTYPE):
                    forecast_value = CLASSES_NAMES_STEREOTYPE[idx]

        self._forecast_group.setExclusive(False)
        for radio in self._forecast_radios:
            radio.setChecked(False)
        self._forecast_group.setExclusive(True)
        fidx = -1
        if forecast_value:
            fidx = CLASSES_NAMES_STEREOTYPE.index(forecast_value)
            self._forecast_radios[fidx].setChecked(True)

        corrected_value = self._cell(series, self._correction_name_col)
        if corrected_value not in CLASSES_NAMES_STEREOTYPE:
            corrected_value = ""

        # State chip: Corrected (teal) when a manual value is set, else
        # Forecasted (indigo).
        if corrected_value:
            self._state_tag.setText("Corrected")
            self._state_tag.setObjectName("chipBrand")
        else:
            self._state_tag.setText("Forecasted")
            self._state_tag.setObjectName("chipForecast")
        self._state_tag.style().unpolish(self._state_tag)
        self._state_tag.style().polish(self._state_tag)

        # Emphasise the effective class name: the corrected one when set,
        # otherwise the forecasted one. Use a medium weight (DemiBold) so it
        # reads as emphasised without the heavy look of full bold.
        active_idx = (
            CLASSES_NAMES_STEREOTYPE.index(corrected_value)
            if corrected_value else fidx
        )
        for j, lbl in enumerate(self._class_name_labels):
            f = lbl.font()
            f.setBold(False)
            f.setWeight(QFont.DemiBold if j == active_idx else QFont.Normal)
            lbl.setFont(f)
        self._correction_group.blockSignals(True)
        self._correction_group.setExclusive(False)
        for radio in self._correction_radios:
            radio.setChecked(False)
        self._correction_group.setExclusive(True)
        if corrected_value:
            self._correction_radios[
                CLASSES_NAMES_STEREOTYPE.index(corrected_value)
            ].setChecked(True)
        self._correction_group.blockSignals(False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _on_correction_chosen(self, button_id: int, checked: bool) -> None:
        if not checked or button_id < 0:
            return
        if (
            self._current_df is None
            or self._current_csv_path is None
            or self._correction_name_col is None
            or self._current_row_index < 0
        ):
            return
        new_name = CLASSES_NAMES_STEREOTYPE[button_id]
        new_index = button_id
        self._save_correction(self._current_row_index, new_name, new_index)
        self._maybe_advance()

    def _on_correction_cancelled(self, _radio: _CorrectableRadioButton) -> None:
        """Clear the corrected value for the current row.

        Triggered when the user clicks the already-checked corrected
        radio. We blank both the ``corrected_*_name`` and the
        ``corrected_*_index`` columns and persist the change.
        """
        if (
            self._current_df is None
            or self._current_csv_path is None
            or self._correction_name_col is None
            or self._current_row_index < 0
        ):
            return
        self._save_correction(self._current_row_index, "", "")
        # Visually clear the correction radios.
        self._correction_group.blockSignals(True)
        self._correction_group.setExclusive(False)
        for radio in self._correction_radios:
            radio.setChecked(False)
        self._correction_group.setExclusive(True)
        self._correction_group.blockSignals(False)

    def _on_validate(self) -> None:
        """Copy the forecasted class to the corrected column for the current row."""
        if (
            self._current_df is None
            or self._forecast_name_col is None
            or self._current_row_index < 0
        ):
            return
        series = self._current_df.iloc[self._current_row_index]
        forecast_name = self._cell(series, self._forecast_name_col)
        forecast_idx_value = self._cell(series, self._forecast_idx_col)
        if not forecast_name:
            self._status_label.setText(
                "Validate: this row has no forecasted class to copy."
            )
            return
        try:
            forecast_idx = int(forecast_idx_value)
        except ValueError:
            forecast_idx = (
                CLASSES_NAMES_STEREOTYPE.index(forecast_name)
                if forecast_name in CLASSES_NAMES_STEREOTYPE
                else ""
            )
        self._save_correction(
            self._current_row_index, forecast_name, forecast_idx,
        )
        # Reflect the new state on the corrected radios.
        if forecast_name in CLASSES_NAMES_STEREOTYPE:
            self._correction_group.blockSignals(True)
            self._correction_group.setExclusive(False)
            for radio in self._correction_radios:
                radio.setChecked(False)
            self._correction_group.setExclusive(True)
            self._correction_radios[
                CLASSES_NAMES_STEREOTYPE.index(forecast_name)
            ].setChecked(True)
            self._correction_group.blockSignals(False)
        self._maybe_advance()

    def _save_correction(
        self, row: int, name_value, idx_value,
    ) -> None:
        """Persist ``name_value`` and ``idx_value`` to the CSV row.

        Either argument may be the empty string to clear the cell.
        Updates the in-memory dataframe first, then rewrites the disk
        cell, then refreshes the row label, the parent video header,
        the row colour and the global counter.
        """
        if self._current_df is None or self._current_csv_path is None:
            return
        if self._correction_name_col is None:
            return
        # In-memory update first so the UI stays consistent if the
        # disk write fails below.
        self._current_df.at[row, self._correction_name_col] = (
            "" if name_value in ("", None) else str(name_value)
        )
        if self._correction_idx_col is not None:
            self._current_df.at[row, self._correction_idx_col] = (
                "" if idx_value in ("", None) else int(idx_value)
            )

        try:
            modify_column_of_row_in_csv(
                filename=self._current_csv_path,
                row_index=row,
                column_name=self._correction_name_col,
                new_value=name_value,
                separator=CSV_SEPARATOR,
                encoding=CSV_ENCODING_LATIN1,
            )
            if self._correction_idx_col is not None:
                modify_column_of_row_in_csv(
                    filename=self._current_csv_path,
                    row_index=row,
                    column_name=self._correction_idx_col,
                    new_value=idx_value,
                    separator=CSV_SEPARATOR,
                    encoding=CSV_ENCODING_LATIN1,
                )
        except PermissionError:
            _msg_critical(
                self, "Step 5",
                "CSV file is locked by another process. Close it and "
                "try again.",
            )
            return
        except Exception as exc:  # noqa: BLE001
            _msg_critical(
                self, "Step 5",
                f"Failed to save correction: {exc}",
            )
            return

        series = self._current_df.iloc[row]
        item = self._row_items.get(row)
        if item is not None:
            item.setText(0, self._row_label(series))
            self._apply_row_colour(item, series)
            parent = item.parent()
            if parent is not None:
                video = str(parent.data(0, _VIDEO_ROOT_ROLE) or "")
                total = parent.childCount()
                corrected = sum(
                    1 for j in range(total)
                    if self._cell(
                        self._current_df.iloc[
                            int(parent.child(j).data(0, _ROW_INDEX_ROLE))
                        ],
                        self._correction_name_col,
                    )
                )
                parent.setText(0, self._video_label(video, total, corrected))
        self._update_counter()

        if name_value:
            self._status_label.setText(
                f"Row {row + 1} / {len(self._current_df)} saved: {name_value}"
            )
        else:
            self._status_label.setText(
                f"Row {row + 1} / {len(self._current_df)} cleared"
            )

    def _maybe_advance(self) -> None:
        if self._auto_skip.isChecked():
            self._select_next()

    # ------------------------------------------------------------------
    # Clear / batch-validate handlers (CORRECTION and VALIDATE rows)
    # ------------------------------------------------------------------
    def _rows_in_current_video(self) -> List[int]:
        """Row indices of every frame under the current frame's video node."""
        item = self._tree.currentItem()
        if item is None:
            return []
        parent = item.parent() if item.parent() is not None else item
        rows: List[int] = []
        for j in range(parent.childCount()):
            r = parent.child(j).data(0, _ROW_INDEX_ROLE)
            if r is not None:
                rows.append(int(r))
        return rows

    def _on_clear_current(self) -> None:
        if self._current_df is None or self._current_row_index < 0:
            return
        self._save_correction(self._current_row_index, "", "")
        self._refresh_radios(self._current_df.iloc[self._current_row_index])

    def _on_clear_video(self) -> None:
        self._clear_rows(self._rows_in_current_video(), "this video")

    def _on_clear_all(self) -> None:
        if self._current_df is None:
            return
        self._clear_rows(list(range(len(self._current_df))), "every video in this CSV")

    def _clear_rows(self, candidate_rows: List[int], scope: str) -> None:
        if self._current_df is None or self._current_csv_path is None:
            return
        rows = [
            int(r) for r in candidate_rows
            if self._is_row_corrected(self._current_df.iloc[int(r)])
        ]
        if not rows:
            self._status_label.setText("No corrections to clear in %s." % scope)
            return
        answer = _msg_question(
            self, "Clear corrections",
            "Remove all %d correction(s) from %s?" % (len(rows), scope),
        )
        if answer != QMessageBox.Yes:
            return
        for r in rows:
            self._save_correction(r, "", "")
        if 0 <= self._current_row_index < len(self._current_df):
            self._refresh_radios(self._current_df.iloc[self._current_row_index])
        self._status_label.setText("Cleared %d correction(s) in %s." % (len(rows), scope))

    def _on_validate_video(self) -> None:
        self._validate_rows(self._rows_in_current_video(), "this video")

    def _on_validate_all(self) -> None:
        if self._current_df is None:
            return
        self._validate_rows(
            list(range(len(self._current_df))), "every video in this CSV",
        )

    def _validate_rows(self, candidate_rows: List[int], scope: str) -> None:
        """Copy the forecast to the correction for not-yet-corrected frames."""
        if (
            self._current_df is None
            or self._current_csv_path is None
            or self._forecast_name_col is None
        ):
            return
        targets: List[int] = []
        for r in candidate_rows:
            r = int(r)
            s = self._current_df.iloc[r]
            if self._is_row_corrected(s):
                continue  # keep manual corrections untouched
            if not self._cell(s, self._forecast_name_col):
                continue  # nothing forecasted to accept
            targets.append(r)
        if not targets:
            self._status_label.setText(
                "No not-yet-corrected frame to validate in %s." % scope
            )
            return
        answer = _msg_question(
            self, "Validate forecasts",
            "Validate the forecast of %d not-yet-corrected frame(s) in %s?"
            % (len(targets), scope),
        )
        if answer != QMessageBox.Yes:
            return
        done = 0
        for r in targets:
            s = self._current_df.iloc[r]
            name = self._cell(s, self._forecast_name_col)
            idx_value = self._cell(s, self._forecast_idx_col)
            try:
                idx = int(idx_value)
            except ValueError:
                idx = (
                    CLASSES_NAMES_STEREOTYPE.index(name)
                    if name in CLASSES_NAMES_STEREOTYPE else ""
                )
            self._save_correction(r, name, idx)
            done += 1
        if 0 <= self._current_row_index < len(self._current_df):
            self._refresh_radios(self._current_df.iloc[self._current_row_index])
        self._status_label.setText("Validated %d frame(s) in %s." % (done, scope))

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------
    def _update_counter(self) -> None:
        if self._current_df is None or self._correction_name_col is None:
            self._counter_label.setText("No CSV loaded")
            return
        total = len(self._current_df)
        # "Corrected" means the row carries an explicit, non-default
        # corrected value, defined per the user spec as "cell not
        # empty and value above -1 in the csv". The previous logic
        # coerced NaN to the literal string "nan" via .astype(str),
        # so empty cells were misclassified as corrected and the
        # counter always read N/N. We now drive the count from the
        # corrected index column (numeric, easy to test for >= 0)
        # when present, and fall back to the name column with an
        # explicit ``notna`` filter otherwise.
        idx_col = self._correction_idx_col
        if idx_col is not None and idx_col in self._current_df.columns:
            numeric = pd.to_numeric(
                self._current_df[idx_col], errors="coerce",
            )
            corrected = int((numeric >= 0).fillna(False).sum())
        else:
            name_series = self._current_df[self._correction_name_col]
            non_null = name_series.notna()
            stripped = name_series.astype(str).str.strip()
            corrected = int((non_null & stripped.ne("") & stripped.str.lower().ne("nan")).sum())
        self._counter_label.setText(
            f"Total: {total}  |  Correct (Manual click): {corrected} / {total}"
        )
        n_videos = 0
        if "video_name" in self._current_df.columns:
            n_videos = int(self._current_df["video_name"].nunique())
        self._chip_frames.setText("%d frames" % total)
        self._chip_videos.setText("%d videos" % n_videos)
        self._chip_corrected.setText("corrected: %d / %d" % (corrected, total))
        for chip in (self._chip_frames, self._chip_videos, self._chip_corrected):
            chip.setVisible(True)
