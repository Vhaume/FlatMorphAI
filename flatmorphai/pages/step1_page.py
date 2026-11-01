"""
Step 1 page: video directory picker + frame extraction launcher.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QDir, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import COLOR_ERROR, COLOR_SUCCESS
from ..notebook_api.naming import VIDEO_EXTENSIONS, parse_video_name
from ..notebook_api.video_frames import count_videos_in_directory
from ..widgets import LogPanel
from ..workers.step1_worker import Step1Params, Step1Worker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class Step1Page(BasePage):
    """Select an input directory and run frame extraction."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Step 1 - Video to frames", parent)
        self._settings = AppSettings()
        self._build_ui()
        self._restore_from_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Step 1 - Video to frames")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        source_box = QGroupBox("Videos directory (default: Videos_to_process)")
        source_box.setObjectName("sectionBox")
        box_layout = QVBoxLayout(source_box)
        box_layout.setContentsMargins(12, 16, 12, 12)
        box_layout.setSpacing(8)

        path_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Videos_to_process")
        self._input_edit.textChanged.connect(self._on_input_changed)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_input)
        path_row.addWidget(self._input_edit, 1)
        path_row.addWidget(browse_btn)
        box_layout.addLayout(path_row)

        path_row2 = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Processed_videos_Frame_by_frame")
        self._output_edit.textChanged.connect(
            lambda t: self._settings.set("processing_root", t)
        )
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._on_browse_output)
        path_row2.addWidget(self._output_edit, 1)
        path_row2.addWidget(out_browse_btn)
        box_layout.addLayout(path_row2)

        self._count_label = QLabel("No directory selected")
        self._count_label.setObjectName("mutedLabel")
        box_layout.addWidget(self._count_label)

        # Per-video compliance list. Each entry shows a video filename
        # found in the input directory and flags, in red, any name that
        # does not satisfy the v1.2 convention. Validating filenames at
        # the UI level before any heavy processing runs is recommended
        # by Wilson, G. et al., "Good Enough Practices in Scientific
        # Computing", PLOS Computational Biology, 2017, DOI:
        # 10.1371/journal.pcbi.1005510, rule 4 (make incremental
        # changes visible and reviewable).
        # The compliance list is wrapped in a vertical splitter so the
        # user can drag its bottom border to grow the list when there
        # are many videos to audit. The default height is large enough
        # to show ~12 rows without scrolling, and the size policy is
        # ``Expanding`` so the widget claims new space when the parent
        # group box is enlarged. The same pattern is recommended in
        # Nielsen, J., "Usability Engineering", Academic Press, 1993,
        # ISBN 0-12-518406-9, chapter 5 (resizable panels for
        # information that benefits from a wide range of densities).
        self._videos_list = QListWidget()
        self._videos_list.setObjectName("step1VideosList")
        self._videos_list.setAlternatingRowColors(True)
        self._videos_list.setMinimumHeight(280)
        self._videos_list.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding,
        )

        self._convention_summary = QLabel("")
        self._convention_summary.setObjectName("mutedLabel")
        self._convention_summary.setWordWrap(True)

        # The compliance splitter holds only the list and the
        # per-folder summary; the verbose naming-convention notice
        # lives below the splitter so its multi-line wrapped text is
        # never clipped by the splitter's bottom pane. This addresses
        # the "convention text is cut off" report from batch 9.
        below_widget = QWidget()
        below_layout = QVBoxLayout(below_widget)
        below_layout.setContentsMargins(0, 4, 0, 0)
        below_layout.setSpacing(6)
        below_layout.addWidget(self._convention_summary)

        compliance_splitter = QSplitter(Qt.Vertical)
        compliance_splitter.setChildrenCollapsible(False)
        compliance_splitter.setHandleWidth(6)
        compliance_splitter.addWidget(self._videos_list)
        compliance_splitter.addWidget(below_widget)
        compliance_splitter.setStretchFactor(0, 5)
        compliance_splitter.setStretchFactor(1, 1)
        compliance_splitter.setSizes([320, 60])
        box_layout.addWidget(compliance_splitter, 1)

        # Naming-convention notice. Pulled out of the splitter so the
        # wrapped text is never truncated by the splitter pane size,
        # and rewritten for v1.2.2 with the optional ``comments`` slot
        # and the four accepted video container extensions. Reference:
        # Sansone, S.-A. et al., "FAIRsharing as a community approach
        # to standards, repositories and policies", Nature
        # Biotechnology, 2019, DOI: 10.1038/s41587-019-0080-8.
        convention_label = QLabel(
            "Video naming convention: "
            "<code>species_compound_concentration_plate_position_period[_comments].&lt;ext&gt;</code>. "
            "Accepted extensions: <code>.mp4</code>, <code>.avi</code>, "
            "<code>.mov</code>, <code>.mkv</code>. The <code>comments</code> "
            "token is optional and may be used to reference an "
            "acquisition date or notebook entry (letters, digits and "
            "the dash; no underscores). Examples: "
            "<code>Smed_nicotine_10um_P1_B4_T0h.mp4</code>, "
            "<code>Smed_Acetylcholine_3000uM_P1_R3(2)_T3h.avi</code>, "
            "<code>Smed_DMSO_0.1pct_P2_C5_T24h_20251225.mov</code>. "
            "Generated images reuse the video stem and append "
            "<code>_&lt;second&gt;s</code> (total and step are on the "
            "parent directory)."
        )
        convention_label.setObjectName("mutedLabel")
        convention_label.setTextFormat(Qt.RichText)
        convention_label.setWordWrap(True)
        convention_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding,
        )
        box_layout.addWidget(convention_label)

        root.addWidget(source_box)

        # Video sampling window. Same three parameters as Image segmentation:
        # start second, end second (auto/manual) and step between frames
        # in seconds. Defaults mirror the v1.0 defaults used by
        # :func:`flatmorphai.notebook_api.video_frames.process_video`
        # ([0 s, duration], 1 s step), which was the behaviour of
        # every pipeline before v1.0. Rationale for exposing the
        # window on Step 1 matches Image segmentation and follows Shneiderman
        # B. et al., "Designing the User Interface: Strategies for
        # Effective Human-Computer Interaction", 6th edition, 2016,
        # ISBN: 978-0-13-438038-3, rule 1 (strive for consistency
        # across the UI) and rule 3 (offer informative feedback for
        # every action).
        window_box = QGroupBox("Video sampling window")
        window_box.setObjectName("sectionBox")
        window_form = QFormLayout(window_box)
        window_form.setContentsMargins(12, 16, 12, 12)

        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 100000.0)
        self._start_spin.setDecimals(3)
        self._start_spin.setSingleStep(1.0)
        self._start_spin.setValue(0.0)
        self._start_spin.setSuffix(" s")
        self._start_spin.valueChanged.connect(
            lambda v: self._settings.set("step1_start_second", float(v))
        )
        window_form.addRow("Start second:", self._start_spin)

        self._end_auto = QCheckBox("Use maximum available second per video")
        self._end_auto.setChecked(True)
        self._end_auto.toggled.connect(self._on_end_auto_toggled)
        window_form.addRow("End mode:", self._end_auto)

        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0.0, 100000.0)
        self._end_spin.setDecimals(3)
        self._end_spin.setSingleStep(1.0)
        self._end_spin.setValue(60.0)
        self._end_spin.setSuffix(" s")
        self._end_spin.setEnabled(False)
        self._end_spin.valueChanged.connect(
            lambda v: self._settings.set("step1_end_second_manual", float(v))
        )
        window_form.addRow("End second (manual):", self._end_spin)

        self._step_spin = QDoubleSpinBox()
        # Constrained to (0, 1] s -- same rule as pipeline_a_page.py
        # (ConvNeXt-V2's 1-second metric window).
        self._step_spin.setRange(0.001, 1.0)
        self._step_spin.setDecimals(3)
        self._step_spin.setSingleStep(0.1)
        self._step_spin.setValue(1.0)
        self._step_spin.setSuffix(" s")
        self._step_spin.setToolTip(
            "Time between sampled frames.  Must be in (0, 1] s.  A "
            "step > 1 s would put the one-second metric window "
            "(T-1 s, T0, T+1 s) outside its training context."
        )
        self._step_spin.valueChanged.connect(
            lambda v: self._settings.set("step1_step_seconds", float(v))
        )
        window_form.addRow("Step between frames:", self._step_spin)

        root.addWidget(window_box)

        # Tree view of processed output
        tree_box = QGroupBox("Processed videos")
        tree_box.setObjectName("sectionBox")
        tree_layout = QVBoxLayout(tree_box)
        tree_layout.setContentsMargins(12, 16, 12, 12)
        self._fs_model = QFileSystemModel()
        self._fs_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self._tree_view = QTreeView()
        self._tree_view.setModel(self._fs_model)
        self._tree_view.setHeaderHidden(False)
        self._tree_view.setRootIsDecorated(True)
        self._tree_view.setAlternatingRowColors(True)
        for col in (1, 2, 3):
            self._tree_view.hideColumn(col)
        tree_layout.addWidget(self._tree_view, 1)
        root.addWidget(tree_box, 1)

        # Action row
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self._run_btn = QPushButton("Run")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_current)
        action_row.addWidget(self._run_btn)
        action_row.addWidget(self._cancel_btn)
        action_row.addStretch(1)
        self._progress = QProgressBar()
        self._progress.setValue(0)
        action_row.addWidget(self._progress, 2)
        root.addLayout(action_row)

        # Log
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        self.run_state_changed.connect(self._on_running_changed)

    # ------------------------------------------------------------------
    # Settings wiring
    # ------------------------------------------------------------------
    def _restore_from_settings(self) -> None:
        # Default locations match the naming convention documented in the
        # Source group: "Videos_to_process" for the raw input folder and
        # "Processed_videos_Frame_by_frame" for the extracted-frame
        # output folder. The defaults are only applied on the very first
        # launch, subsequent runs reuse the values persisted through
        # :class:`AppSettings`.
        self._input_edit.setText(
            self._settings.get("input_video_directory", "Videos_to_process")
        )
        self._output_edit.setText(
            self._settings.get("processing_root", "Processed_videos_Frame_by_frame")
        )
        self._start_spin.setValue(
            float(self._settings.get("step1_start_second", 0.0))
        )
        end_auto = bool(self._settings.get("step1_end_second_auto", True))
        self._end_auto.setChecked(end_auto)
        self._end_spin.setValue(
            float(self._settings.get("step1_end_second_manual", 60.0))
        )
        self._end_spin.setEnabled(not end_auto)
        self._step_spin.setValue(
            float(self._settings.get("step1_step_seconds", 1.0))
        )
        self._refresh_tree_root()
        self._update_count_label(self._input_edit.text())
        self._refresh_videos_list(self._input_edit.text())

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_browse_input(self) -> None:
        start = self._input_edit.text() or os.getcwd()
        path = get_existing_directory(self, "Select video directory", start)
        if path:
            self._input_edit.setText(path)

    def _on_browse_output(self) -> None:
        start = self._output_edit.text() or os.getcwd()
        path = get_existing_directory(self, "Select output directory", start)
        if path:
            self._output_edit.setText(path)

    def _on_input_changed(self, text: str) -> None:
        self._settings.set("input_video_directory", text)
        self._update_count_label(text)
        self._refresh_videos_list(text)

    def _update_count_label(self, text: str) -> None:
        if text and os.path.isdir(text):
            count = count_videos_in_directory(text)
            self._count_label.setText(f"{count} video file(s) found under {text}")
        else:
            self._count_label.setText("No directory selected")

    def _refresh_videos_list(self, text: str) -> None:
        """Populate the per-video compliance list.

        Each item shows a filename and a short status token. Names
        that fail :func:`parse_video_name` are flagged in red so the
        user can fix them before launching Step 1. Early validation
        follows the principle of "fail fast, explicit errors" from
        Ousterhout, J., "A Philosophy of Software Design", 2nd edition,
        2021, ISBN: 978-1732102217, chapter 10 (define errors out of
        existence by surfacing them at the input stage).
        """
        self._videos_list.clear()
        if not text or not os.path.isdir(text):
            self._convention_summary.setText("")
            return
        try:
            entries = sorted(os.listdir(text))
        except OSError:
            self._convention_summary.setText("")
            return
        total = 0
        bad = 0
        bad_color = QBrush(QColor(COLOR_ERROR))
        ok_color = QBrush(QColor(COLOR_SUCCESS))
        for name in entries:
            full = os.path.join(text, name)
            if not os.path.isfile(full):
                continue
            if not name.lower().endswith(VIDEO_EXTENSIONS):
                continue
            total += 1
            meta = parse_video_name(name)
            if meta is None:
                bad += 1
                item = QListWidgetItem(f"{name}  [INVALID NAME]")
                item.setForeground(bad_color)
            else:
                item = QListWidgetItem(f"{name}  [OK]")
                item.setForeground(ok_color)
            self._videos_list.addItem(item)
        if total == 0:
            self._convention_summary.setText(
                "No video files detected in this directory."
            )
            return
        if bad == 0:
            self._convention_summary.setText(
                f"<span style='color: {COLOR_SUCCESS};'>"
                f"All {total} video file(s) respect the naming convention."
                f"</span>"
            )
        else:
            self._convention_summary.setText(
                f"<span style='color: {COLOR_ERROR};'>"
                f"{bad} of {total} video file(s) do not respect the "
                f"naming convention. Rename them before running Step 1."
                f"</span>"
            )
        self._convention_summary.setTextFormat(Qt.RichText)

    def _refresh_tree_root(self) -> None:
        root = self._output_edit.text() or "."
        if os.path.isdir(root):
            self._fs_model.setRootPath(root)
            self._tree_view.setRootIndex(self._fs_model.index(root))

    def _on_end_auto_toggled(self, checked: bool) -> None:
        self._end_spin.setEnabled(not checked)
        self._settings.set("step1_end_second_auto", bool(checked))

    def _on_run(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip() or "Processed_videos_Frame_by_frame"
        if not input_dir or not os.path.isdir(input_dir):
            _msg_warning(self, "Step 1", "Select a valid input video directory.")
            return
        if count_videos_in_directory(input_dir) == 0:
            _msg_warning(self, "Step 1", "No .mp4/.avi/.mov files found.")
            return

        # Defensive guard on the sampling step (see pipeline_a_page.py
        # for the full ConvNeXt-V2 1-second-window rationale).
        step_seconds_now = float(self._step_spin.value())
        if not (0.0 < step_seconds_now <= 1.0):
            _msg_critical(
                self, "Step 1",
                f"'Step between frames' is {step_seconds_now} s but "
                "must be in (0, 1] s.",
            )
            return

        end_second = None if self._end_auto.isChecked() else float(self._end_spin.value())
        self._progress.setValue(0)
        self._log_panel.clear()
        worker = Step1Worker(
            Step1Params(
                input_directory=input_dir,
                output_directory=output_dir,
                start_second=float(self._start_spin.value()),
                end_second=end_second,
                step_seconds=float(self._step_spin.value()),
            )
        )
        worker.progress_signal.connect(self._progress.setValue)
        worker.log_signal.connect(self._log_panel.append)
        worker.error_signal.connect(lambda msg: self._log_panel.append(f"ERROR: {msg}"))
        worker.result_signal.connect(self._on_done)
        self._launch_worker(worker)

    def _on_done(self, result: object) -> None:
        if isinstance(result, dict):
            self._log_panel.append(
                f"Finished. Total frames written: {result.get('total_frames_written')}"
            )
        self._refresh_tree_root()

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
