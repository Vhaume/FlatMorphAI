"""
Regeneration States Editor.

A sibling of the Stereotype Editor, but built around the regeneration-state
prediction file written by the "Forecast regeneration states" pipeline
(``regeneration_states.csv``, saved next to each ``images/`` folder).  By
default the page looks, under the processing root, for the first ``images/``
folder that has a sibling ``regeneration_states.csv`` and loads it.

Left panel
----------
The frames are grouped by ``index_planaria`` (one planaria = one plate +
position).  A group is labelled ``N(<index>)_Plate(<plate>)_Position(<pos>)``
and every frame reads ``State(<state>)_Day(<day>)_N(<index>)_Plate(<plate>)_
Position(<pos>)_Date(<date>)``.  ``State`` is the rounded regeneration state,
or the corrected value when one is set.

Below the tree, replacing the stereotype radios, the editor shows the current
state and two collapsible, interactive graphs for the selected planaria:

* States evolution -- the rounded/corrected state (accent) and the decimal
  state predicted by the model (grey), against ``Day``.
* Eyespots number evolution -- the number of eyespots detected at both
  extremities (balanced threshold), against ``Day``.

Both graphs mark the current image's Day with a dashed line, highlight the
pointed Day on hover and jump to that Day's image when a column is clicked.
Their expanded/collapsed state is remembered between sessions.

Right panel
-----------
The dynamic preview shows the raw FRAME and a DETECTION view with two optional
overlays, toggled by the "Show blastema" and "Show eyespots" checkboxes:

* Show blastema -- blends the blastema mask (``blasteme_masks/<frame>.png``,
  written when the pipeline's "Save blastema masks" option was on) over the
  image.
* Show eyespots -- draws a red cross on every detected eye cup, using the (x, y)
  positions stored in ``eyespots_detection.csv`` for the balanced threshold.
"""

from __future__ import annotations

import ast
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import CSV_SEPARATOR, PROCESSING_ROOT_DEFAULT
from ..notebook_api.csv_editor import modify_column_of_row_in_csv
from ..widgets import (
    CollapsibleSection,
    EvolutionGraph,
    ImagePairViewer,
    Stepper,
    ToggleSwitch,
    blank_icon,
    check_icon,
)
from .base_page import BasePage
from ..widgets.file_dialogs import (
    critical as _msg_critical,
    get_existing_directory,
    get_open_file_name,
    question as _msg_question,
)

_ROW_INDEX_ROLE = Qt.UserRole + 1
_GROUP_ROLE = Qt.UserRole + 2

_CORRECTED_FG = QColor("#008000")

# Minimum and maximum regeneration state (the pipeline clamps to 1..6).
_STATE_MIN = 1
_STATE_MAX = 6

# Badge colour per state 1..6 (matches the dashboard design's scale).
_STATE_COLORS = ["#94A3B8", "#38BDF8", "#2DD4BF", "#A3E635", "#FBBF24", "#FB7185"]

# Output sub-folders written by the regeneration pipeline next to images/.
_BLASTEME_DIR = "blasteme_masks"
_MASKS_DIR = "masks"
_EYESPOTS_CSV = "eyespots_detection.csv"
_HEAD_CSV = "head_position.csv"


class RegenerationEditorPage(BasePage):
    """Browse and correct predicted regeneration states with live graphs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Regeneration States Editor", parent)
        self._settings = AppSettings()
        self._csv_path: Optional[str] = None
        self._video_dir: Optional[str] = None
        self._df: Optional[pd.DataFrame] = None
        self._row_index: int = -1
        self._cols: Dict[str, Optional[str]] = {}
        # index_planaria -> rows sorted by Day; row -> index_planaria.
        self._groups: Dict[str, List[int]] = {}
        self._row_group: Dict[int, str] = {}
        self._row_items: Dict[int, QTreeWidgetItem] = {}
        # image_name -> {"bal": int, "bal_pts": [(x, y)], "pos": int}
        self._eyespots: Dict[str, dict] = {}
        # image_name -> head position ("up" / "down") for formated orientation
        self._head: Dict[str, str] = {}
        self._build_ui()
        self._restore_from_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- header: title + subtitle + count chips ----------------
        header = QHBoxLayout()
        header.setSpacing(12)
        title = QLabel("Regeneration states editor")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        subtitle = QLabel("Review and correct the predicted regeneration states")
        subtitle.setObjectName("pageSubtitle")
        header.addWidget(subtitle, 0, Qt.AlignVCenter)
        header.addStretch(1)
        self._chip_frames = QLabel("")
        self._chip_frames.setObjectName("chip")
        self._chip_planaria = QLabel("")
        self._chip_planaria.setObjectName("chip")
        self._chip_corrected = QLabel("")
        self._chip_corrected.setObjectName("chipBrand")
        for chip in (self._chip_frames, self._chip_planaria, self._chip_corrected):
            chip.setVisible(False)
            header.addWidget(chip, 0, Qt.AlignVCenter)
        root.addLayout(header)

        # --- source card -------------------------------------------
        src_card = QFrame()
        src_card.setObjectName("card")
        sl = QHBoxLayout(src_card)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(10)

        root_cap = QLabel("Processing root")
        root_cap.setObjectName("caption")
        sl.addWidget(root_cap, 0, Qt.AlignVCenter)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText(PROCESSING_ROOT_DEFAULT)
        self._root_edit.textChanged.connect(
            lambda t: self._settings.set("regedit_processing_root", t)
        )
        sl.addWidget(self._root_edit, 3)
        browse_root = QPushButton("Browse")
        browse_root.clicked.connect(self._on_browse_root)
        sl.addWidget(browse_root)
        sl.addSpacing(14)

        csv_cap = QLabel("States CSV")
        csv_cap.setObjectName("caption")
        sl.addWidget(csv_cap, 0, Qt.AlignVCenter)
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText(
            "…/<video>/regeneration_states.csv (found automatically)"
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

        # Internal status label the counter still writes to (counts are shown as
        # the header chips; this is kept for programmatic text / tests).
        self._counter_label = QLabel("No CSV loaded")
        self._counter_label.setObjectName("mutedLabel")
        self._counter_label.setVisible(False)

        # --- workspace: tree card (left) + content column (right) in a
        #     splitter so the image-list width is user-adjustable. ---
        workspace = QSplitter(Qt.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(10)

        # LEFT: planaria tree card + navigation
        tree_card = QFrame()
        tree_card.setObjectName("card")
        tree_card.setMinimumWidth(200)
        tcl = QVBoxLayout(tree_card)
        tcl.setContentsMargins(10, 10, 10, 10)
        tcl.setSpacing(8)
        tree_head = QHBoxLayout()
        left_header = QLabel("Planaria / frames")
        left_header.setObjectName("sectionTitle")
        tree_head.addWidget(left_header)
        tree_head.addStretch(1)
        self._row_info_short = QLabel("")
        self._row_info_short.setObjectName("mutedLabel")
        tree_head.addWidget(self._row_info_short, 0, Qt.AlignVCenter)
        tcl.addLayout(tree_head)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        # Same compact layout as the Stereotype editor's frame list: tighter
        # rows, a wider branch target, and a small icon column so a validation
        # check fits just before each name.
        self._tree.setIndentation(28)
        self._tree.setIconSize(QSize(16, 16))
        self._tree.setStyleSheet("QTreeWidget::item { padding: 2px 2px; }")
        self._tree.currentItemChanged.connect(self._on_tree_selected)
        tcl.addWidget(self._tree, 1)
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._select_previous)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._select_next)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        tcl.addLayout(nav)
        workspace.addWidget(tree_card)

        # RIGHT: scrollable column of cards (viewers / state / charts). The
        # content keeps the canvas (grey) background so the gaps between the
        # white cards read as clear tile separations.
        content = QWidget()
        content.setObjectName("canvasArea")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)

        # --- viewers card ---
        viewers_card = QFrame()
        viewers_card.setObjectName("card")
        pv = QVBoxLayout(viewers_card)
        pv.setContentsMargins(12, 10, 12, 12)
        pv.setSpacing(8)
        overlay_row = QHBoxLayout()
        overlay_row.setContentsMargins(0, 0, 0, 0)
        self._status_label = QLabel("")
        self._status_label.setObjectName("mutedLabel")
        overlay_row.addWidget(self._status_label, 0, Qt.AlignVCenter)
        self._show_mask = ToggleSwitch("Mask")
        self._show_mask.setToolTip(
            "Formated view: cross the frame with the body mask so only the worm "
            "shows on a black background. Use 'Head-up' to straighten it. "
            "Reuses thumbnails_all_body when available. "
            "Requires the 'masks' folder (keep 'Save body masks' on)."
        )
        self._show_mask.toggled.connect(self._on_overlay_toggled)
        # Head-up is its OWN control: the rotation used to ride on the Mask
        # toggle, so there was no way to see the masked worm in the frame's
        # own orientation, nor to straighten the raw frame without masking it.
        # It works with or without the mask -- with one the crop follows the
        # mask's bounding box, without one a box centred on the head-tail
        # midpoint. Needs head_position.csv; with nothing known the image is
        # left as it is.
        self._show_headup = ToggleSwitch("Head-up")
        self._show_headup.setToolTip(
            "Rotate the worm so its head points up, then crop to it, using "
            "the head-to-tail axis of head_position.csv. Works with or "
            "without the Mask overlay. Without a known head position the "
            "image is shown unchanged."
        )
        self._show_headup.toggled.connect(self._on_overlay_toggled)
        self._show_blasteme = ToggleSwitch("Blastema")
        self._show_blasteme.setToolTip(
            "Blend the blastema mask over the image. Requires the "
            "'blasteme_masks' folder written by the regeneration pipeline "
            "(its 'Save blastema masks' option)."
        )
        self._show_blasteme.toggled.connect(self._on_overlay_toggled)
        self._show_eyespots = ToggleSwitch("Eyespots")
        self._show_eyespots.setToolTip(
            "Draw a red cross on every detected eyespot, using the (x, y) "
            "positions of the balanced threshold stored in "
            "eyespots_detection.csv."
        )
        self._show_eyespots.toggled.connect(self._on_overlay_toggled)
        # These overlays affect the DETECTION (right) pane, so sit them above
        # it: a leading stretch pushes them over the right half of the viewer.
        overlay_row.setSpacing(16)
        overlay_row.addStretch(1)
        overlay_row.addWidget(self._show_mask)
        overlay_row.addWidget(self._show_headup)
        overlay_row.addWidget(self._show_blasteme)
        overlay_row.addWidget(self._show_eyespots)
        pv.addLayout(overlay_row)
        self._viewer = ImagePairViewer(left_title="FRAME", right_title="DETECTION")
        # This editor has no "3s display"; hide the inherited toggle.
        self._viewer.right_toggle.setVisible(False)
        self._viewer.setMinimumHeight(320)
        # Remember the FRAME/DETECTION divider position across sessions.
        self._viewer.bind_persistence(self._settings, "regedit_viewer_split")
        pv.addWidget(self._viewer, 1)
        # viewers_card is added to a vertical splitter below (resizable height).

        # --- state + correction card -------------------------------
        state_card = QFrame()
        state_card.setObjectName("card")
        bl = QVBoxLayout(state_card)
        bl.setContentsMargins(14, 12, 14, 12)
        bl.setSpacing(10)

        # Row 1: read-outs (current effective state badge + raw prediction).
        info_row = QHBoxLayout()
        info_row.setSpacing(10)
        self._state_badge = QLabel("-")
        self._state_badge.setObjectName("stateBadge")
        info_row.addWidget(self._state_badge, 0, Qt.AlignVCenter)
        state_caption = QLabel("Current\nregeneration state")
        state_caption.setObjectName("caption")
        info_row.addWidget(state_caption, 0, Qt.AlignVCenter)
        self._corrected_tag = QLabel("Corrected")
        self._corrected_tag.setObjectName("chipBrand")
        self._corrected_tag.setVisible(False)
        info_row.addWidget(self._corrected_tag, 0, Qt.AlignVCenter)
        info_row.addSpacing(28)
        # Raw decimal state predicted by the model for the current image, before
        # rounding or any correction (e.g. "4.03").
        pred_caption = QLabel("Prediction")
        pred_caption.setObjectName("caption")
        info_row.addWidget(pred_caption, 0, Qt.AlignVCenter)
        self._predict_value = QLabel("-")
        self._predict_value.setObjectName("predictValue")
        self._predict_value.setToolTip(
            "The model's raw decimal regeneration-state output for this image, "
            "before rounding and before any correction."
        )
        info_row.addWidget(self._predict_value, 0, Qt.AlignVCenter)
        info_row.addStretch(1)
        bl.addLayout(info_row)

        # Separator between the current-state read-out and the action rows.
        state_divider = QFrame()
        state_divider.setObjectName("collapsibleDivider")
        state_divider.setFrameShape(QFrame.HLine)
        bl.addWidget(state_divider)

        # Row 2: correction controls (stepper + propagate + clear buttons).
        corr_row = QHBoxLayout()
        corr_row.setSpacing(10)
        corr_caption = QLabel("Correction")
        corr_caption.setObjectName("caption")
        corr_row.addWidget(corr_caption)
        self._state_spin = Stepper()
        self._state_spin.setRange(_STATE_MIN, _STATE_MAX)
        self._state_spin.setToolTip(
            "Override the rounded state for this image. Saved to the "
            "'state_corrected' column; the graph's main curve follows it."
        )
        # Each +/- click commits the new value.
        self._state_spin.valueChanged.connect(self._on_correction_committed)
        corr_row.addWidget(self._state_spin)
        self._propagate_chk = ToggleSwitch("Propagate")
        self._propagate_chk.setToolTip(
            "When on, a correction is propagated along the planaria so the "
            "state never decreases over time: later days below the corrected "
            "value are raised to it, and earlier days above it are lowered to "
            "it. Every row changed this way is also marked as corrected (green)."
        )
        self._propagate_chk.setChecked(
            bool(self._settings.get("regedit_propagate_correction", False))
        )
        self._propagate_chk.toggled.connect(
            lambda v: self._settings.set("regedit_propagate_correction", bool(v))
        )
        corr_row.addWidget(self._propagate_chk)
        self._clear_btn = QPushButton("Clear current correction")
        self._clear_btn.clicked.connect(self._on_clear_correction)
        corr_row.addWidget(self._clear_btn)
        self._clear_worm_btn = QPushButton("Clear all worm corrections")
        self._clear_worm_btn.setToolTip(
            "Remove every manual correction of the current planaria only."
        )
        self._clear_worm_btn.clicked.connect(self._on_clear_worm_corrections)
        corr_row.addWidget(self._clear_worm_btn)
        self._clear_all_btn = QPushButton("Clear all file corrections")
        self._clear_all_btn.setObjectName("dangerButton")
        self._clear_all_btn.setToolTip(
            "Remove every manual correction from the whole CSV (all planaria)."
        )
        self._clear_all_btn.clicked.connect(self._on_clear_all_corrections)
        corr_row.addWidget(self._clear_all_btn)
        corr_row.addStretch(1)
        bl.addLayout(corr_row)

        # Row 3: validation (accept the predicted state as correct), mirroring
        # the Stereotype Editor's Validate. Its "auto-skip" toggle replaces the
        # correction row's "propagate": when on, validating jumps straight to
        # the next image for fast sequential review.
        val_row = QHBoxLayout()
        val_row.setSpacing(10)
        val_caption = QLabel("Validate")
        val_caption.setObjectName("caption")
        val_row.addWidget(val_caption)
        # Current image: plain neutral button (same look as Next).
        self._validate_btn = QPushButton("Validate state, current image")
        self._validate_btn.setToolTip(
            "Accept the current predicted (rounded) state as the correction "
            "for this image, marking it corrected (green). With Auto-skip on, "
            "jumps to the next image so you can validate a whole worm quickly."
        )
        self._validate_btn.clicked.connect(self._on_validate)
        val_row.addWidget(self._validate_btn)
        self._validate_autoskip = ToggleSwitch("Auto-skip")
        self._validate_autoskip.setToolTip(
            "When on, validating the current image automatically moves to the "
            "next one."
        )
        self._validate_autoskip.setChecked(
            bool(self._settings.get("regedit_validate_autoskip", True))
        )
        self._validate_autoskip.toggled.connect(
            lambda v: self._settings.set("regedit_validate_autoskip", bool(v))
        )
        val_row.addWidget(self._validate_autoskip)
        # Batch validators (after Auto-skip). These only touch images that are
        # NOT already corrected, so manual corrections are never overwritten.
        self._validate_worm_btn = QPushButton("Validate state, all current worm")
        self._validate_worm_btn.setToolTip(
            "Validate the predicted state of every not-yet-corrected image of "
            "the current planaria."
        )
        self._validate_worm_btn.clicked.connect(self._on_validate_worm)
        val_row.addWidget(self._validate_worm_btn)
        self._validate_all_btn = QPushButton("Validate state, all file")
        self._validate_all_btn.setObjectName("validateAllButton")
        self._validate_all_btn.setToolTip(
            "Validate the predicted state of every not-yet-corrected image in "
            "the whole CSV (all planaria)."
        )
        self._validate_all_btn.clicked.connect(self._on_validate_all_file)
        val_row.addWidget(self._validate_all_btn)
        val_row.addStretch(1)
        bl.addLayout(val_row)
        cl.addWidget(state_card)

        # --- charts card (two charts side by side) -----------------
        charts_card = QFrame()
        charts_card.setObjectName("card")
        ccl = QVBoxLayout(charts_card)
        ccl.setContentsMargins(12, 10, 12, 12)

        # States evolution graph (two curves: corrected state + model forecast).
        self._states_graph = EvolutionGraph(
            primary_name="Corrected", secondary_name="Forecast",
            show_curve_toggles=True, export_basename="states_evolution",
        )
        self._states_graph.daySelected.connect(self._on_day_selected)
        self._states_section = CollapsibleSection(
            "States evolution",
            self._states_graph,
            expanded=bool(self._settings.get("regedit_graph_states_expanded", True)),
            on_toggle=lambda e: self._settings.set("regedit_graph_states_expanded", bool(e)),
        )
        # Eyespots number evolution graph (single curve).
        self._eyespots_graph = EvolutionGraph(
            primary_name="Eyespots", show_curve_toggles=False,
            export_basename="eyespots_evolution",
        )
        self._eyespots_graph.daySelected.connect(self._on_day_selected)
        self._eyespots_section = CollapsibleSection(
            "Eyespots number evolution",
            self._eyespots_graph,
            expanded=bool(self._settings.get("regedit_graph_eyespots_expanded", True)),
            on_toggle=lambda e: self._settings.set("regedit_graph_eyespots_expanded", bool(e)),
        )
        charts_row = QHBoxLayout()
        charts_row.setSpacing(20)
        charts_row.addWidget(self._states_section, 1)
        charts_row.addWidget(self._eyespots_section, 1)
        ccl.addLayout(charts_row)
        cl.addWidget(charts_card)
        cl.addStretch(1)

        # The content column scrolls vertically so tall graphs never clip.
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QScrollArea.NoFrame)
        # AsNeeded (not AlwaysOff) so that at very large UI font scales the
        # viewers/toolbars can be scrolled to instead of being clipped.
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setWidget(content)
        content_scroll.setStyleSheet("QScrollArea { background: transparent; }")

        # The viewers sit on top of a vertical splitter so their height is
        # user-adjustable (drag the divider); the state + charts scroll below.
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(10)
        right_splitter.addWidget(viewers_card)
        right_splitter.addWidget(content_scroll)
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([460, 460])
        workspace.addWidget(right_splitter)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([260, 1100])
        root.addWidget(workspace, 1)

        # Persist the image-list width and the viewer height across sessions.
        self._workspace_splitter = workspace
        self._right_splitter = right_splitter
        self._restore_editor_splitters()
        workspace.splitterMoved.connect(lambda *_: self._save_editor_splitters())
        right_splitter.splitterMoved.connect(lambda *_: self._save_editor_splitters())

        self._shortcuts: List[QShortcut] = []
        for seq, handler in (
            (Qt.Key_Up, self._select_previous),
            (Qt.Key_Down, self._select_next),
            # Enter validates the current image (same as the Validate button).
            (Qt.Key_Return, self._on_enter_validate),
            (Qt.Key_Enter, self._on_enter_validate),
            # Overlay toggles: m = Mask, h = Head-up, b = Blastema, e = Eyespots.
            (Qt.Key_M, self._show_mask.toggle),
            (Qt.Key_H, self._show_headup.toggle),
            (Qt.Key_B, self._show_blasteme.toggle),
            (Qt.Key_E, self._show_eyespots.toggle),
            # Correction stepper: + / - do the same as the +/- buttons.
            (Qt.Key_Plus, self._shortcut_correction_up),
            (Qt.Key_Equal, self._shortcut_correction_up),  # '+' without Shift
            (Qt.Key_Minus, self._shortcut_correction_down),
        ):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)
        # self._status_label is the file-line label in the viewers card header.

    def _shortcut_correction_up(self) -> None:
        """'+' key: same as clicking the correction stepper's + button."""
        try:
            self._state_spin._plus.click()
        except Exception:  # noqa: BLE001
            pass

    def _shortcut_correction_down(self) -> None:
        """'-' key: same as clicking the correction stepper's - button."""
        try:
            self._state_spin._minus.click()
        except Exception:  # noqa: BLE001
            pass

    def _restore_editor_splitters(self) -> None:
        """Apply the saved image-list width and viewer height."""
        try:
            tw = int(self._settings.get("regedit_tree_width", 0) or 0)
            if tw > 120:
                total = sum(self._workspace_splitter.sizes()) or 1360
                self._workspace_splitter.setSizes([tw, max(200, total - tw)])
            vh = int(self._settings.get("regedit_viewer_height", 0) or 0)
            if vh > 120:
                total = sum(self._right_splitter.sizes()) or 920
                self._right_splitter.setSizes([vh, max(200, total - vh)])
        except Exception:  # noqa: BLE001
            pass

    def _save_editor_splitters(self) -> None:
        try:
            ws = self._workspace_splitter.sizes()
            if ws:
                self._settings.set("regedit_tree_width", int(ws[0]))
            rs = self._right_splitter.sizes()
            if rs:
                self._settings.set("regedit_viewer_height", int(rs[0]))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_from_settings(self) -> None:
        proc_root = (
            str(self._settings.get("regedit_processing_root", "")).strip()
            or str(self._settings.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._root_edit.setText(proc_root)

        self._restoring = True
        self._show_mask.setChecked(
            bool(self._settings.get("regedit_show_mask", False))
        )
        self._show_headup.setChecked(
            bool(self._settings.get("regedit_show_headup", True))
        )
        self._show_blasteme.setChecked(
            bool(self._settings.get("regedit_show_blasteme", False))
        )
        self._show_eyespots.setChecked(
            bool(self._settings.get("regedit_show_eyespots", False))
        )
        self._restoring = False

        saved_csv = str(self._settings.get("regedit_csv_path", "")).strip()
        if not saved_csv or not os.path.isfile(saved_csv):
            saved_csv = self._find_default_csv(proc_root) or saved_csv
        self._csv_edit.setText(saved_csv)
        if saved_csv and os.path.isfile(saved_csv):
            try:
                self._load_csv(saved_csv)
            except Exception:  # noqa: BLE001
                pass

    def showEvent(self, event) -> None:  # noqa: N802
        """Auto-load the latest regeneration_states.csv when shown.

        The Forecast regeneration states pipeline saves the generated CSV path
        (and its processing root) on completion; switching to this editor
        afterwards loads it without a manual Load click, when not already loaded.
        """
        super().showEvent(event)
        root = str(self._settings.get("regedit_processing_root", "")).strip()
        if root and root != self._root_edit.text().strip():
            self._root_edit.setText(root)
        saved = str(self._settings.get("regedit_csv_path", "")).strip()
        current = self._csv_path or ""
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

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _on_browse_root(self) -> None:
        start = self._root_edit.text() or os.getcwd()
        path = get_existing_directory(self, "Select processing root", start)
        if path:
            self._root_edit.setText(path)
            found = self._find_default_csv(path)
            if found:
                self._csv_edit.setText(found)

    def _on_browse_csv(self) -> None:
        start = self._csv_edit.text() or self._root_edit.text() or os.getcwd()
        if start and not os.path.isdir(start):
            start = os.path.dirname(start) or os.getcwd()
        path, _ = get_open_file_name(
            self, "Select regeneration_states.csv", start, "CSV files (*.csv)"
        )
        if path:
            self._csv_edit.setText(path)

    @staticmethod
    def _find_default_csv(root: str) -> Optional[str]:
        """First ``regeneration_states.csv`` that sits next to an ``images/``."""
        if not root or not os.path.isdir(root):
            return None
        skip_prefixes = ("masks", "thumbnails_", "segmented_", "temp_")
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath).lower()
            if base.startswith(skip_prefixes) or base == "eyespots heatmaps":
                dirnames[:] = []
                continue
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.lower().startswith(skip_prefixes)
                and d.lower() != "eyespots heatmaps"
            )
            if "images" in dirnames and "regeneration_states.csv" in filenames:
                return os.path.join(dirpath, "regeneration_states.csv")
        return None

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------
    def _on_load(self) -> None:
        csv_path = self._csv_edit.text().strip()
        if not csv_path or not os.path.isfile(csv_path):
            _msg_critical(
                self, "Regeneration editor",
                "Select an existing regeneration_states.csv to load.",
            )
            return
        self._settings.set("regedit_csv_path", csv_path)
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str) -> None:
        df: Optional[pd.DataFrame] = None
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(csv_path, sep=CSV_SEPARATOR, encoding=encoding)
                break
            except Exception:  # noqa: BLE001
                continue
        if df is None:
            _msg_critical(
                self, "Regeneration editor",
                "Cannot read %s. Verify it uses ';' as separator." % csv_path,
            )
            return
        self._csv_path = csv_path
        self._video_dir = os.path.dirname(csv_path)
        self._df = df
        self._cols = self._resolve_columns(df)
        self._settings.set("regedit_csv_path", csv_path)
        self._load_eyespots_map()
        self._load_head_map()
        self._build_groups()
        self._populate_tree()
        self._status_label.setText(
            "Loaded %d row(s) from %s" % (len(df), os.path.basename(csv_path))
        )

    @staticmethod
    def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        lower = {c.lower(): c for c in df.columns}

        def pick(*cands: str) -> Optional[str]:
            for c in cands:
                if c.lower() in lower:
                    return lower[c.lower()]
            return None

        return {
            "image_name": pick("image_name", "image"),
            "index": pick("index_planaria", "planaria_index"),
            "plate": pick("plate"),
            "position": pick("position"),
            "species": pick("species"),
            "compound": pick("compound"),
            "concentration": pick("concentration"),
            "date": pick("date"),
            "day": pick("Day", "day"),
            "round": pick("state_round"),
            "decimal": pick("state_decimal"),
            "round_corrected": pick("state_round_corrected"),
            "corrected": pick("state_corrected"),
        }

    def _load_eyespots_map(self) -> None:
        """Index eyespots_detection.csv by image_name (balanced count/positions)."""
        self._eyespots = {}
        if not self._video_dir:
            return
        path = os.path.join(self._video_dir, _EYESPOTS_CSV)
        if not os.path.isfile(path):
            return
        df: Optional[pd.DataFrame] = None
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(path, sep=CSV_SEPARATOR, encoding=encoding)
                break
            except Exception:  # noqa: BLE001
                continue
        if df is None:
            return
        name_col = next((c for c in df.columns if c.lower() in ("image_name", "image")), None)
        # Prefer the monotonic-corrected balanced count when the pipeline wrote
        # it ("..._corrected"); fall back to the raw balanced count otherwise.
        bal_corr = next(
            (c for c in df.columns
             if c.startswith("n_eyespots_strong_balanced") and c.endswith("_corrected")),
            None,
        )
        bal_raw = next(
            (c for c in df.columns
             if c.startswith("n_eyespots_strong_balanced") and not c.endswith("_corrected")),
            None,
        )
        bal_cnt = bal_corr or bal_raw
        bal_pos = next((c for c in df.columns if c.startswith("eyespots_positions_balanced")), None)
        # "positive" since 30/10/2026; "positifs" was the French spelling of
        # the same column, kept here so a CSV written before that date still
        # opens in this editor.
        pos_cnt = next((c for c in df.columns
                        if c.startswith(("n_eyespots_strong_positive",
                                         "n_eyespots_strong_positifs"))), None)
        if name_col is None:
            return
        for _i, rec in df.iterrows():
            name = str(rec.get(name_col, "")).strip()
            if not name:
                continue
            self._eyespots[name] = {
                "bal": self._to_int(rec.get(bal_cnt)) if bal_cnt else 0,
                "bal_pts": self._parse_points(rec.get(bal_pos)) if bal_pos else [],
                "pos": self._to_int(rec.get(pos_cnt)) if pos_cnt else 0,
            }

    def _load_head_map(self) -> None:
        """Index head_position.csv by image_name.

        Stores, per image, which extremity is the head ('up'/'down') and the two
        extremity points in original-frame coordinates, so the formated view can
        rotate the worm head-up.
        """
        self._head = {}
        if not self._video_dir:
            return
        path = os.path.join(self._video_dir, _HEAD_CSV)
        if not os.path.isfile(path):
            return
        df: Optional[pd.DataFrame] = None
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(path, sep=CSV_SEPARATOR, encoding=encoding)
                break
            except Exception:  # noqa: BLE001
                continue
        if df is None:
            return
        name_col = next((c for c in df.columns if c.lower() in ("image_name", "image")), None)
        head_col = next((c for c in df.columns if c.lower() == "head_position"), None)
        up_col = next((c for c in df.columns if c.lower() == "up_position"), None)
        bot_col = next((c for c in df.columns if c.lower() == "bottom_position"), None)
        # No tail column: the file holds the two tips of the medial axis and
        # says which one is the head, so the tail is the other one.
        if name_col is None or head_col is None:
            return
        for _i, rec in df.iterrows():
            name = str(rec.get(name_col, "")).strip()
            if not name:
                continue
            up_pt = self._parse_point(rec.get(up_col)) if up_col else None
            bot_pt = self._parse_point(rec.get(bot_col)) if bot_col else None
            raw = str(rec.get(head_col, "")).strip()
            # ``head_position`` holds the head's COORDINATES, not the word
            # "up" or "down": the file writes "(951.7, 736.9)", which is a
            # copy of either up_position or bottom_position. Reading it as a
            # word gave a value that matched neither, so the view never knew
            # which end was the head and never rotated -- the toggle looked
            # dead while the information was there all along. The side is
            # recovered by seeing which extremity the point coincides with;
            # a file that really does carry the word is still accepted.
            side = raw.lower() if raw.lower() in ("up", "down") else ""
            if not side:
                head_pt = self._parse_point(raw)
                if head_pt is not None and (up_pt is not None or bot_pt is not None):
                    d_up = (self._dist2(head_pt, up_pt)
                            if up_pt is not None else float("inf"))
                    d_bot = (self._dist2(head_pt, bot_pt)
                             if bot_pt is not None else float("inf"))
                    side = "up" if d_up <= d_bot else "down"
            head_pt = self._parse_point(raw)
            tail_pt = None
            if head_pt is not None and up_pt is not None and bot_pt is not None:
                # Whichever tip is not the head. Compared by distance rather
                # than by equality: the two are written rounded to a tenth of
                # a pixel, and an exact match is not something to depend on.
                tail_pt = bot_pt if self._dist2(head_pt, up_pt) <= self._dist2(head_pt, bot_pt) else up_pt
            self._head[name] = {
                "head": side,
                "head_pt": head_pt,
                "tail_pt": tail_pt,
                "up": up_pt,
                "bottom": bot_pt,
            }

    @staticmethod
    def _dist2(a, b) -> float:
        """Squared distance between two points (no square root needed to
        decide which of two extremities a third point sits on)."""
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    @staticmethod
    def _parse_point(value) -> Optional[Tuple[float, float]]:
        """Parse a single ``(x, y)`` string into a float tuple, or None."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text or text == "nan":
            return None
        try:
            data = ast.literal_eval(text)
            x, y = data
            return (float(x), float(y))
        except (ValueError, SyntaxError, TypeError):
            return None

    @staticmethod
    def _parse_points(value) -> List[Tuple[float, float]]:
        """Parse ``[(x, y), (x, y)]`` into a list of float tuples."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        text = str(value).strip()
        if not text or text in ("[]", "nan"):
            return []
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        out: List[Tuple[float, float]] = []
        for item in data or []:
            try:
                x, y = item
                out.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _to_int(value) -> int:
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return 0
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Grouping / tree
    # ------------------------------------------------------------------
    def _worm_key(self, s: pd.Series) -> str:
        """Composite identity of one worm.

        A worm is one (plate, position, species, compound, concentration): the
        same plate+position can hold two different treatments (e.g. Caf 500 and
        Ctrl 0) photographed on the same dates, which are two DIFFERENT worms.
        Grouping on index_planaria alone merged them when an older CSV assigned
        the same index to both, producing "duplicate" days and zig-zag graphs;
        this composite key keeps them apart whatever the stored index.
        """
        parts = [
            self._cell(s, "plate"), self._cell(s, "position"),
            self._cell(s, "species"), self._cell(s, "compound"),
            self._cell(s, "concentration"),
        ]
        key = "|".join(parts)
        return key if key.strip("|") else (str(self._cell(s, "index")) or "?")

    def _build_groups(self) -> None:
        self._groups = {}
        self._row_group = {}
        if self._df is None:
            return
        for row in range(len(self._df)):
            s = self._df.iloc[row]
            key = self._worm_key(s)
            self._groups.setdefault(key, []).append(row)
            self._row_group[row] = key
        # Order each group by Day so the graph and the tree read in time order.
        for key in self._groups:
            self._groups[key].sort(key=lambda r: self._day_of(r))

    def _populate_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        self._row_items.clear()
        if self._df is None:
            self._tree.blockSignals(False)
            return
        # Groups ordered numerically by planaria index when possible.
        def _grp_key(k: str):
            try:
                return (0, int(float(k)))
            except (TypeError, ValueError):
                return (1, k)

        for key in sorted(self._groups, key=_grp_key):
            rows = self._groups[key]
            s0 = self._df.iloc[rows[0]]
            top = QTreeWidgetItem([self._group_label(s0, key)])
            top.setData(0, _GROUP_ROLE, key)
            top.setFirstColumnSpanned(True)
            self._tree.addTopLevelItem(top)
            for row in rows:
                series = self._df.iloc[row]
                child = QTreeWidgetItem([self._row_label(series, key)])
                child.setData(0, _ROW_INDEX_ROLE, row)
                if self._corrected_of(series) is not None:
                    child.setForeground(0, QBrush(_CORRECTED_FG))
                    child.setIcon(0, check_icon(16))
                else:
                    child.setIcon(0, blank_icon(16))
                top.addChild(child)
                self._row_items[row] = child
            top.setExpanded(True)
        self._tree.blockSignals(False)
        first = self._first_leaf()
        if first is not None:
            self._tree.setCurrentItem(first)
        self._update_counter()

    def _group_label(self, series: pd.Series, key: str) -> str:
        # ``key`` is the internal composite identity; the label shows the
        # stored planaria index plus the treatment so two worms sharing a
        # plate/position (e.g. Caf 500 vs Ctrl 0) are told apart at a glance.
        label = "N(%s)_Plate(%s)_Position(%s)" % (
            self._cell(series, "index") or "?",
            self._cell(series, "plate"), self._cell(series, "position"),
        )
        compound = self._cell(series, "compound")
        concentration = self._cell(series, "concentration")
        species = self._cell(series, "species")
        extra = []
        if species:
            extra.append("Species(%s)" % species)
        if compound:
            extra.append("Compound(%s)" % compound)
        if concentration:
            extra.append("Concentration(%s)" % concentration)
        if extra:
            label += "_" + "_".join(extra)
        return label

    def _row_label(self, series: pd.Series, key: str) -> str:
        state = self._effective_state(series)
        state_txt = "%d" % state if state is not None else "?"
        return "[State(%s)]_Day(%s)_N(%s)_Date(%s)" % (
            state_txt, self._cell(series, "day"), key,
            self._cell(series, "date"),
        )

    def _first_leaf(self) -> Optional[QTreeWidgetItem]:
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            if top.childCount() > 0:
                return top.child(0)
        return None

    # ------------------------------------------------------------------
    # Cell helpers / state values
    # ------------------------------------------------------------------
    def _cell(self, series: pd.Series, role: str) -> str:
        col = self._cols.get(role)
        if col is None or col not in series:
            return ""
        value = series[col]
        if pd.isna(value):
            return ""
        text = str(value).strip()
        # Drop a trailing ".0" so an integer read as float prints cleanly.
        if text.endswith(".0"):
            text = text[:-2]
        return text

    def _day_of(self, row: int) -> float:
        if self._df is None:
            return 0.0
        try:
            return float(self._cell(self._df.iloc[row], "day") or 0)
        except ValueError:
            return 0.0

    def _rounded_of(self, series: pd.Series) -> Optional[int]:
        # The "rounded" state, after the pipeline's monotonic auto-correction
        # when present, else the raw rounded prediction.
        for role in ("round_corrected", "round"):
            text = self._cell(series, role)
            if text:
                try:
                    return int(round(float(text)))
                except ValueError:
                    continue
        return None

    def _corrected_of(self, series: pd.Series) -> Optional[int]:
        text = self._cell(series, "corrected")
        if not text:
            return None
        try:
            return int(round(float(text)))
        except ValueError:
            return None

    def _decimal_of(self, series: pd.Series) -> Optional[float]:
        text = self._cell(series, "decimal")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _effective_state(self, series: pd.Series) -> Optional[int]:
        corrected = self._corrected_of(series)
        return corrected if corrected is not None else self._rounded_of(series)

    # ------------------------------------------------------------------
    # Selection / navigation
    # ------------------------------------------------------------------
    def _on_tree_selected(self, current, _previous) -> None:
        if current is None or self._df is None:
            return
        row = current.data(0, _ROW_INDEX_ROLE)
        if row is None:
            return
        self._row_index = int(row)
        self._refresh_all()

    def _select_previous(self) -> None:
        leaf = self._sibling_leaf(-1)
        if leaf is not None:
            self._tree.setCurrentItem(leaf)

    def _select_next(self) -> None:
        leaf = self._sibling_leaf(+1)
        if leaf is not None:
            self._tree.setCurrentItem(leaf)

    def _sibling_leaf(self, direction: int) -> Optional[QTreeWidgetItem]:
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

    def _on_day_selected(self, day: int) -> None:
        """Jump to the image of ``day`` within the current planaria."""
        if self._df is None or self._row_index < 0:
            return
        key = self._row_group.get(self._row_index)
        if key is None:
            return
        for row in self._groups.get(key, []):
            if int(round(self._day_of(row))) == int(day):
                item = self._row_items.get(row)
                if item is not None:
                    self._tree.setCurrentItem(item)
                return

    # ------------------------------------------------------------------
    # Refresh (image, state, graphs) -- called on every selection change
    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        if self._df is None or not (0 <= self._row_index < len(self._df)):
            return
        series = self._df.iloc[self._row_index]
        self._refresh_preview(series)
        self._refresh_state_controls(series)
        self._refresh_graphs()
        self._status_label.setText(
            "Row %d / %d: (Day %s) %s"
            % (self._row_index + 1, len(self._df),
               self._cell(series, "day") or "?", self._cell(series, "image_name"))
        )
        self._row_info_short.setText("%d / %d" % (self._row_index + 1, len(self._df)))

    def _refresh_clear_buttons(self) -> None:
        """Enable the clear-all buttons only when there is something to clear."""
        col = self._cols.get("corrected")
        if self._df is None or col is None or col not in self._df.columns:
            self._clear_worm_btn.setEnabled(False)
            self._clear_all_btn.setEnabled(False)
            return
        numeric = pd.to_numeric(self._df[col], errors="coerce")
        self._clear_all_btn.setEnabled(bool(numeric.notna().any()))
        worm_has = False
        key = self._row_group.get(self._row_index)
        if key is not None:
            worm_has = any(
                pd.notna(numeric.iloc[r]) for r in self._groups.get(key, [])
            )
        self._clear_worm_btn.setEnabled(worm_has)

    def _refresh_state_controls(self, series: pd.Series) -> None:
        eff = self._effective_state(series)
        self._state_badge.setText("-" if eff is None else str(eff))
        self._state_badge.setStyleSheet(self._badge_style(eff))
        dec = self._decimal_of(series)
        self._predict_value.setText("-" if dec is None else ("%.2f" % dec))
        self._refresh_clear_buttons()
        corrected = self._corrected_of(series)
        self._corrected_tag.setVisible(corrected is not None)
        self._state_spin.blockSignals(True)
        self._state_spin.setValue(
            corrected if corrected is not None else (eff if eff is not None else _STATE_MIN)
        )
        self._state_spin.blockSignals(False)
        self._clear_btn.setEnabled(corrected is not None)

    @staticmethod
    def _badge_style(state: Optional[int]) -> str:
        """Background colour for the state badge, keyed by state 1..6."""
        if state is None or not (1 <= int(state) <= 6):
            return "background-color: #94A3B8;"
        return "background-color: %s;" % _STATE_COLORS[int(state) - 1]

    # ------------------------------------------------------------------
    # Preview + overlays
    # ------------------------------------------------------------------
    def _on_overlay_toggled(self, _checked: bool) -> None:
        # Restoring the four toggles fires this handler once per toggle, and
        # each call wrote ALL FOUR settings back -- including the ones not
        # restored yet, which were still at their widget default. Restoring
        # "Mask" therefore saved Head-up as unchecked a moment before Head-up
        # read its own saved value, so the view came back unrotated and
        # uncropped whatever the user had chosen. Nothing is saved while the
        # page is still loading its state.
        if getattr(self, "_restoring", False):
            return
        self._settings.set("regedit_show_mask", bool(self._show_mask.isChecked()))
        self._settings.set("regedit_show_headup", bool(self._show_headup.isChecked()))
        self._settings.set("regedit_show_blasteme", bool(self._show_blasteme.isChecked()))
        self._settings.set("regedit_show_eyespots", bool(self._show_eyespots.isChecked()))
        if self._df is not None and 0 <= self._row_index < len(self._df):
            # Same frame, only the overlay changed: keep the user's zoom/pan.
            self._refresh_preview(self._df.iloc[self._row_index], preserve_view=True)

    def _refresh_preview(self, series: pd.Series, preserve_view: bool = False) -> None:
        image_name = self._cell(series, "image_name")
        if not self._video_dir or not image_name:
            self._viewer.clear()
            return
        frame_path = os.path.join(self._video_dir, "images", image_name)
        if not os.path.isfile(frame_path):
            self._viewer.clear()
            self._status_label.setText("Frame not found: %s" % image_name)
            return
        frame_path = os.path.abspath(frame_path)
        self._viewer.update_left(
            QPixmap(frame_path), source_path=frame_path, preserve_view=preserve_view,
        )
        det, det_source = self._compose_detection(frame_path, image_name)
        # The DETECTION Open button reveals ``det_source``: the saved modified
        # image when an overlay is on, else the original frame.
        self._viewer.update_right(
            det, source_path=det_source, preserve_view=preserve_view,
        )

    # Folder (at the application root) that holds the modified DETECTION images
    # the Open button reveals, mirroring the stereotype editor's temp_formated.
    _TEMP_FORMATED_DIRNAME = "temp_formated"

    @staticmethod
    def _app_root() -> str:
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    def _temp_formated_path(self, image_name: str) -> str:
        video_tag = os.path.basename(os.path.normpath(self._video_dir or ""))
        stem, _ext = os.path.splitext(image_name)
        return os.path.join(
            self._app_root(), self._TEMP_FORMATED_DIRNAME,
            "%s__%s_detection.png" % (video_tag, stem),
        )

    def _orient_formated(self, out, body_mask, image_name: str):
        """Rotate the view so the worm stands upright, head at the top, then
        crop to it.

        The axis comes from the WORM, not from the CSV. ``up_position`` and
        ``bottom_position`` are not the two ends: measured on the 52 images of
        the reference set, they sit 12 % of the worm's length apart, every
        time. Using them as a head-to-tail axis meant rotating along a vector
        that is mostly noise, which is why one image came out at a wrong
        angle. The principal axis of the body mask (SVD of its pixel
        coordinates) is the real direction of the worm -- the same quantity
        the production composite uses -- and ``head_position`` is then used
        for the one thing it does say: which of the two ends is the head, so
        the image is flipped by 180 degrees when the head lands at the bottom.

        The canvas is sized to hold the rotated worm instead of keeping the
        frame's own size, so a worm near an edge is no longer cut by the
        rotation. Overlays already drawn on ``out`` rotate with it.
        """
        import math

        import cv2
        import numpy as np

        h, w = out.shape[:2]
        info = self._head.get(image_name) or {}
        head_pt = info.get("head_pt")
        if head_pt is None:
            side = info.get("head")
            head_pt = info.get("up") if side == "up" else info.get("bottom")

        mask = None
        if body_mask is not None:
            mask = (body_mask > 127)
            if not mask.any():
                mask = None

        # --- direction of the worm -------------------------------------
        # The axis is anchored on the HEAD recorded in head_position.csv. The
        # other end is NOT in that file: despite their names, up_position and
        # bottom_position are not the two ends of the worm -- the pipeline
        # writes the head's TIP in one and the head framing's CENTRE in the
        # other (regen_stage_v20_pipeline, "tip" and "centre"), which is
        # why they sit 0.12 of the worm's length apart on all 52 reference
        # images. So the far end is taken from the body itself: the mask pixel
        # furthest from the head. Head at the top, tail at the bottom, which
        # is the alignment asked for; the head point stays the authority on
        # which end is which.
        theta = None
        center = None
        # The tail now has its own column in head_position.csv, written by the
        # pipeline (and added to older files by maj_tail_position.py). When it
        # is there the axis comes straight from the file; otherwise it is
        # measured here the same way, so an older file still works.
        tail_pt = info.get("tail_pt")
        if tail_pt is None and mask is not None and head_pt is not None:
            from ..workers.mask_refine import tail_from_mask
            tail_pt = tail_from_mask(mask, head_pt)
        if tail_pt is not None and head_pt is not None:
            dx, dy = head_pt[0] - tail_pt[0], head_pt[1] - tail_pt[1]
            if dx or dy:
                theta = math.degrees(math.atan2(dy, dx)) + 90.0
                center = ((head_pt[0] + tail_pt[0]) / 2.0,
                          (head_pt[1] + tail_pt[1]) / 2.0)
        if theta is None and mask is not None:
            # No head recorded: fall back on the body's own principal axis,
            # which gives an upright worm but cannot tell head from tail.
            ys, xs = np.nonzero(mask)
            pts = np.stack((xs.astype(np.float64), ys.astype(np.float64)))
            mean = pts.mean(axis=1)
            try:
                vals, vecs = np.linalg.eigh(np.cov(pts - mean[:, None]))
                ax = vecs[:, int(np.argmax(vals))]
                theta = math.degrees(math.atan2(float(ax[1]), float(ax[0]))) + 90.0
                center = (float(mean[0]), float(mean[1]))
            except np.linalg.LinAlgError:
                theta = None
        if theta is None:
            up, bot = info.get("up"), info.get("bottom")
            if up is not None and bot is not None and up != bot:
                dx, dy = up[0] - bot[0], up[1] - bot[1]
                theta = math.degrees(math.atan2(dy, dx)) + 90.0
                center = ((up[0] + bot[0]) / 2.0, (up[1] + bot[1]) / 2.0)
        if theta is None or center is None:
            return out

        def _rotated(angle):
            """(matrix, output size) for a rotation that clips nothing."""
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
            r = (M[:, :2] @ corners.T).T + M[:, 2]
            x0, y0 = r.min(axis=0)
            x1, y1 = r.max(axis=0)
            M = M.copy()
            M[0, 2] -= x0
            M[1, 2] -= y0
            return M, (int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0)))

        def _apply(point, M):
            return (M[0, 0] * point[0] + M[0, 1] * point[1] + M[0, 2],
                    M[1, 0] * point[0] + M[1, 1] * point[1] + M[1, 2])

        # --- head at the top -------------------------------------------
        M, size = _rotated(theta)
        if head_pt is not None:
            ref = tail_pt if tail_pt is not None else center
            if _apply(head_pt, M)[1] > _apply(ref, M)[1]:
                M, size = _rotated(theta + 180.0)   # head below: turn over

        rot = cv2.warpAffine(out, M, size, flags=cv2.INTER_LINEAR,
                             borderValue=(0, 0, 0))
        rot_mask = None
        if mask is not None:
            rot_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, M, size,
                                      flags=cv2.INTER_NEAREST)

        # --- crop to the worm ------------------------------------------
        pad = 24
        if rot_mask is not None:
            ys, xs = np.nonzero(rot_mask > 127)
            if xs.size and ys.size:
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(size[0], int(xs.max()) + pad)
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(size[1], int(ys.max()) + pad)
                if x1 > x0 and y1 > y0:
                    rot = rot[y0:y1, x0:x1]
        else:
            cx, cy = _apply(center, M)
            half = int(max(size) * 0.25)
            x0, x1 = max(0, int(cx) - half), min(size[0], int(cx) + half)
            y0, y1 = max(0, int(cy) - half), min(size[1], int(cy) + half)
            if x1 > x0 and y1 > y0:
                rot = rot[y0:y1, x0:x1]
        return rot

    @staticmethod
    def _mask_path_for(video_dir: str, stem: str) -> str:
        """Path of the body mask for ``stem``, searching ``masks*`` folders.

        The pipelines write the body masks to ``masks/`` (older runs
        used a bare ``masks/``), so look in any sibling folder whose name
        starts with ``masks`` and return the first that holds this frame; the
        bare ``masks/`` is checked first for backward compatibility.
        """
        if not video_dir:
            return ""
        candidates = [_MASKS_DIR]
        try:
            candidates += sorted(
                d for d in os.listdir(video_dir)
                if d != _MASKS_DIR and d.lower().startswith("masks")
                and os.path.isdir(os.path.join(video_dir, d))
            )
        except OSError:
            pass
        for folder in candidates:
            path = os.path.join(video_dir, folder, stem + ".png")
            if os.path.isfile(path):
                return path
        # None found: return the conventional path so callers can report it.
        return os.path.join(video_dir, _MASKS_DIR, stem + ".png")

    def _compose_detection(self, frame_path: str, image_name: str):
        """Build the DETECTION pixmap and the path its Open button should reveal.

        Returns ``(pixmap, source_path)``. With no overlay the detection is the
        raw frame, so the source is the original frame. With an overlay the
        modified image is saved under ``temp_formated/`` and that saved file is
        the source, so Open reveals exactly what is shown.
        """
        show_mask = self._show_mask.isChecked()
        show_bl = self._show_blasteme.isChecked()
        show_cup = self._show_eyespots.isChecked()
        show_up = self._show_headup.isChecked()
        # Head-up alone is a view too: the frame straightened and cropped to
        # the worm. It used to be missing from this test, so ticking only
        # Head-up returned the raw frame and the button looked dead.
        if not (show_mask or show_bl or show_cup or show_up):
            return QPixmap(frame_path), frame_path
        try:
            import cv2
            import numpy as np
        except Exception:  # noqa: BLE001
            return QPixmap(frame_path), frame_path

        def _imread(path, flag):
            try:
                return cv2.imdecode(np.fromfile(path, np.uint8), flag)
            except Exception:  # noqa: BLE001
                return None

        stem = os.path.splitext(image_name)[0]
        vdir = self._video_dir or ""
        mask_path = self._mask_path_for(vdir, stem)
        mask_exists = bool(mask_path) and os.path.isfile(mask_path)

        # The formated view is always rebuilt here (frame -> rotate head-up ->
        # overlays -> crop) rather than reusing thumbnails_all_body, because the
        # pipeline thumbnail cannot carry the overlays and its orientation is not
        # always reliable. The black background needs the body mask; without it
        # the oriented crop keeps the original background. A one-off note tells
        # the user how to get the black background.
        if show_mask and not mask_exists:
            self._status_label.setText(
                "Showing the oriented worm; keep 'Save body masks' on and re-run "
                "to get the worm isolated on a black background."
            )

        bgr = _imread(frame_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return QPixmap(frame_path), frame_path
        out = bgr.copy()

        # Formated view: keep only the worm (body mask) on a black background.
        # Done first so the overlays below land on the masked image.
        body_mask = None
        # The mask is also read when only Head-up is on: it is not painted on
        # the image then, but its bounding box gives a far tighter crop than
        # the fallback box built from the head-tail distance.
        if show_mask or show_up:
            body = _imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_exists else None
            if body is not None:
                if body.shape[:2] != out.shape[:2]:
                    body = cv2.resize(body, (out.shape[1], out.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                if show_mask:
                    out[body <= 127] = 0
                body_mask = body
            elif show_mask:
                self._status_label.setText(
                    "No body mask for %s (run the pipeline with "
                    "'Save body masks')." % image_name
                )

        if self._show_blasteme.isChecked() and self._video_dir:
            mpath = os.path.join(self._video_dir, _BLASTEME_DIR, stem + ".png")
            if os.path.isfile(mpath):
                mask = _imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != out.shape[:2]:
                        mask = cv2.resize(mask, (out.shape[1], out.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                    idx = mask > 127
                    if idx.any():
                        tint = np.array([255, 255, 0], np.float32)  # cyan (BGR)
                        out[idx] = (0.55 * out[idx].astype(np.float32)
                                    + 0.45 * tint).astype(np.uint8)
            else:
                self._status_label.setText(
                    "No blastema mask for %s (run the pipeline with "
                    "'Save blastema masks')." % image_name
                )

        if self._show_eyespots.isChecked():
            info = self._eyespots.get(image_name, {})
            # Scale the cross with the image so it stays visible on a full-res
            # plate as well as on a tight formated crop.
            ms = max(16, int(min(out.shape[:2]) * 0.02))
            th = max(2, ms // 7)
            for x, y in info.get("bal_pts", []):
                cv2.drawMarker(out, (int(round(x)), int(round(y))), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=ms, thickness=th)

        # Orient the formated view head-up and crop to the worm. The overlays
        # were drawn in original coordinates, so they rotate with the image and
        # stay aligned (this is why they cannot simply be pasted onto the
        # pre-rotated thumbnail).
        if show_up:
            out = self._orient_formated(out, body_mask, image_name)

        rgb = np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        # Save the modified image so the DETECTION Open button reveals it.
        # Keep EXACTLY ONE image in temp_formated (shared with the stereotype
        # editor): prune every other file first, so visiting many frames in
        # either editor never accumulates one file per frame -- only the frame
        # currently shown stays on disk.
        source_path = frame_path
        try:
            save_path = self._temp_formated_path(image_name)
            out_dir = os.path.dirname(save_path)
            os.makedirs(out_dir, exist_ok=True)
            try:
                _keep = os.path.basename(save_path)
                for _stale in os.listdir(out_dir):
                    if _stale != _keep:
                        try:
                            os.remove(os.path.join(out_dir, _stale))
                        except OSError:
                            pass
            except OSError:
                pass
            ok, buf = cv2.imencode(".png", out)
            if ok:
                buf.tofile(save_path)
                source_path = os.path.abspath(save_path)
        except Exception:  # noqa: BLE001
            source_path = frame_path

        return QPixmap.fromImage(qimg), source_path

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------
    def _refresh_graphs(self) -> None:
        if self._df is None or self._row_index < 0:
            return
        key = self._row_group.get(self._row_index)
        rows = self._groups.get(key, []) if key is not None else []
        current_day = int(round(self._day_of(self._row_index)))

        # States: primary = effective (rounded/corrected), secondary = decimal.
        primary: List[Tuple[int, float]] = []
        secondary: List[Tuple[int, float]] = []
        cup_points: List[Tuple[int, float]] = []
        for row in rows:
            s = self._df.iloc[row]
            day = int(round(self._day_of(row)))
            eff = self._effective_state(s)
            if eff is not None:
                primary.append((day, float(eff)))
            dec = self._decimal_of(s)
            if dec is not None:
                secondary.append((day, dec))
            name = self._cell(s, "image_name")
            cup = self._eyespots.get(name)
            if cup is not None:
                cup_points.append((day, float(cup.get("bal", 0))))

        # Y axis locked to the integer state range 1..6.
        self._states_graph.set_data(
            primary, secondary, current_day,
            y_min=_STATE_MIN, y_max=_STATE_MAX, y_integer=True,
        )
        cur_series = self._df.iloc[self._row_index]
        eff = self._effective_state(cur_series)
        self._states_section.set_value_text("-" if eff is None else str(eff))

        cup_max = max((v for _, v in cup_points), default=0)
        self._eyespots_graph.set_data(
            cup_points, None, current_day,
            y_min=0, y_max=max(2.0, cup_max + 1.0), y_integer=True,
        )
        cur_cup = self._eyespots.get(self._cell(cur_series, "image_name"), {})
        self._eyespots_section.set_value_text(str(int(cur_cup.get("bal", 0))))

    # ------------------------------------------------------------------
    # Correction persistence
    # ------------------------------------------------------------------
    def _on_correction_value_changed(self, _value: int) -> None:
        # Commit only when the change comes from the spin arrows/keys while the
        # control has focus (avoids committing during programmatic updates,
        # which are guarded by blockSignals).
        if self._state_spin.hasFocus():
            self._on_correction_committed()

    def _on_correction_committed(self) -> None:
        if self._df is None or self._row_index < 0:
            return
        self._save_corrected(self._row_index, int(self._state_spin.value()))

    def _on_clear_correction(self) -> None:
        if self._df is None or self._row_index < 0:
            return
        self._save_corrected(self._row_index, None)

    def _on_enter_validate(self) -> None:
        """Enter validates the current image, unless a text field is focused."""
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            return
        self._on_validate()

    def _on_validate(self) -> None:
        """Accept the current predicted (rounded/effective) state as correct.

        Writes the effective state into ``state_corrected`` (so the row turns
        green and the graph's main curve follows it), then - when Auto-skip is
        on - advances to the next image, mirroring the Stereotype Editor's
        Validate + auto-advance flow.
        """
        if self._df is None or self._row_index < 0:
            return
        eff = self._effective_state(self._df.iloc[self._row_index])
        if eff is None:
            self._status_label.setText(
                "Validate: this image has no predicted state to accept."
            )
            return
        if self._write_corrected_cell(self._row_index, int(eff)):
            self._refresh_row_item(self._row_index)
            self._refresh_state_controls(self._df.iloc[self._row_index])
            self._refresh_graphs()
            self._update_counter()
            self._status_label.setText("Validated state %d." % int(eff))
            if self._validate_autoskip.isChecked():
                self._select_next()

    def _on_validate_worm(self) -> None:
        """Validate every not-yet-corrected image of the current planaria."""
        if self._df is None or self._row_index < 0:
            return
        key = self._row_group.get(self._row_index)
        if key is None:
            return
        self._validate_rows(self._groups.get(key, []), scope="this planaria")

    def _on_validate_all_file(self) -> None:
        """Validate every not-yet-corrected image in the whole CSV."""
        if self._df is None:
            return
        self._validate_rows(
            list(range(len(self._df))), scope="every planaria in this CSV",
        )

    def _validate_rows(self, candidate_rows, scope: str) -> None:
        """Write the effective state as the correction for not-corrected rows.

        Rows that already carry a manual correction are left untouched so a
        batch validate never overwrites deliberate edits.
        """
        if self._df is None or self._csv_path is None:
            return
        targets = []
        for r in candidate_rows:
            r = int(r)
            s = self._df.iloc[r]
            if self._corrected_of(s) is not None:
                continue  # already corrected -> keep as is
            if self._effective_state(s) is None:
                continue  # nothing predicted to accept
            targets.append(r)
        if not targets:
            self._status_label.setText(
                "No not-yet-corrected image to validate in %s." % scope
            )
            return
        answer = _msg_question(
            self, "Validate states",
            "Validate the predicted state of %d not-yet-corrected image(s) "
            "in %s?" % (len(targets), scope),
        )
        if answer != QMessageBox.Yes:
            return
        done = 0
        for r in targets:
            eff = self._effective_state(self._df.iloc[r])
            if eff is not None and self._write_corrected_cell(r, int(eff)):
                self._refresh_row_item(r)
                done += 1
        if 0 <= self._row_index < len(self._df):
            self._refresh_state_controls(self._df.iloc[self._row_index])
        self._refresh_graphs()
        self._update_counter()
        self._status_label.setText("Validated %d image(s) in %s." % (done, scope))

    def _on_clear_worm_corrections(self) -> None:
        """Remove every manual correction of the current planaria only."""
        if self._df is None or self._row_index < 0:
            return
        key = self._row_group.get(self._row_index)
        if key is None:
            return
        self._clear_corrections_rows(
            self._groups.get(key, []), scope="this planaria",
        )

    def _on_clear_all_corrections(self) -> None:
        """Remove every manual correction from the whole CSV."""
        if self._df is None:
            return
        self._clear_corrections_rows(
            list(range(len(self._df))), scope="every planaria in this CSV",
        )

    def _clear_corrections_rows(self, candidate_rows, scope: str) -> None:
        col = self._cols.get("corrected")
        if self._df is None or self._csv_path is None or col is None:
            return
        numeric = pd.to_numeric(self._df[col], errors="coerce")
        rows = [int(r) for r in candidate_rows if pd.notna(numeric.iloc[int(r)])]
        if not rows:
            self._status_label.setText("No corrections to clear.")
            return
        answer = _msg_question(
            self, "Clear corrections",
            "Remove all %d correction(s) from %s?" % (len(rows), scope),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        ok = True
        for row in rows:
            if not self._write_corrected_cell(row, None):
                ok = False
                break
        for row in rows:
            self._refresh_row_item(row)
        if self._df is not None and 0 <= self._row_index < len(self._df):
            self._refresh_state_controls(self._df.iloc[self._row_index])
        self._refresh_graphs()
        self._update_counter()
        if ok:
            self._status_label.setText("Cleared %d correction(s)." % len(rows))

    def _write_corrected_cell(self, row: int, value: Optional[int]) -> bool:
        """Write one state_corrected cell to the DataFrame and the CSV."""
        col = self._cols.get("corrected")
        if self._df is None or self._csv_path is None or col is None:
            if col is None:
                self._status_label.setText(
                    "This CSV has no 'state_corrected' column; cannot correct."
                )
            return False
        cell = "" if value is None else int(value)
        self._df.at[row, col] = cell
        try:
            modify_column_of_row_in_csv(
                filename=self._csv_path,
                row_index=row,
                column_name=col,
                new_value=cell,
                separator=CSV_SEPARATOR,
                encoding="utf-8",
            )
        except PermissionError:
            _msg_critical(
                self, "Regeneration editor",
                "CSV file is locked by another process. Close it and retry.",
            )
            return False
        except Exception as exc:  # noqa: BLE001
            _msg_critical(
                self, "Regeneration editor", "Failed to save: %s" % exc
            )
            return False
        return True

    def _refresh_row_item(self, row: int) -> None:
        """Refresh one tree row's label and corrected (green) colour."""
        if self._df is None:
            return
        series = self._df.iloc[row]
        item = self._row_items.get(row)
        if item is not None:
            key = self._row_group.get(row, "?")
            item.setText(0, self._row_label(series, key))
            corrected = self._corrected_of(series) is not None
            item.setForeground(
                0, QBrush(_CORRECTED_FG) if corrected else QBrush()
            )
            item.setIcon(0, check_icon(16) if corrected else blank_icon(16))

    def _propagated_changes(self, row: int, value: int) -> Dict[int, int]:
        """Rows whose corrected value must change to keep the state monotonic.

        With the edited ``row`` pinned at ``value`` inside its planaria (ordered
        by Day), later days below ``value`` are raised to the running maximum
        and earlier days above ``value`` are lowered to the running minimum, so
        the state never decreases over time. Returns ``{row: new_value}`` for
        the rows that actually change (the edited row excluded).
        """
        changes: Dict[int, int] = {}
        key = self._row_group.get(row)
        if key is None:
            return changes
        group = self._groups.get(key, [])
        try:
            pos = group.index(row)
        except ValueError:
            return changes

        def eff(r: int) -> Optional[int]:
            if r == row:
                return value
            return self._effective_state(self._df.iloc[r])

        # Forward: later days must be >= the running maximum.
        running = value
        for r in group[pos + 1:]:
            cur = eff(r)
            if cur is None:
                continue
            if cur < running:
                changes[r] = running
            else:
                running = cur
        # Backward: earlier days must be <= the running minimum.
        running = value
        for r in reversed(group[:pos]):
            cur = eff(r)
            if cur is None:
                continue
            if cur > running:
                changes[r] = running
            else:
                running = cur
        return changes

    def _save_corrected(self, row: int, value: Optional[int]) -> None:
        if not self._write_corrected_cell(row, value):
            return
        touched = [row]
        # Propagate the correction along the planaria when requested (only for a
        # real value, not when clearing).
        if (
            value is not None
            and getattr(self, "_propagate_chk", None) is not None
            and self._propagate_chk.isChecked()
        ):
            for r, v in self._propagated_changes(row, int(value)).items():
                if self._write_corrected_cell(r, v):
                    touched.append(r)
        for r in touched:
            self._refresh_row_item(r)
        self._refresh_state_controls(self._df.iloc[row])
        self._refresh_graphs()
        self._update_counter()

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------
    def _update_counter(self) -> None:
        if self._df is None:
            self._counter_label.setText("No CSV loaded")
            return
        total = len(self._df)
        n_planaria = len(self._groups)
        col = self._cols.get("corrected")
        corrected = 0
        if col is not None and col in self._df.columns:
            numeric = pd.to_numeric(self._df[col], errors="coerce")
            corrected = int(numeric.notna().sum())
        self._counter_label.setText(
            "Total: %d frame(s)  |  %d planaria  |  corrected: %d / %d"
            % (total, n_planaria, corrected, total)
        )
        self._chip_frames.setText("%d frames" % total)
        self._chip_planaria.setText("%d planaria" % n_planaria)
        self._chip_corrected.setText("corrected: %d / %d" % (corrected, total))
        for chip in (self._chip_frames, self._chip_planaria, self._chip_corrected):
            chip.setVisible(True)
        self._refresh_clear_buttons()
