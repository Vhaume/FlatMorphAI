"""
Mask metrics page: extract behavioural and morphometric metrics from
binary planarian masks.

The page accepts either an input video directory (used to regenerate
frames and masks when the requested sampling rate differs from what
the existing per-video directories encode) or a processing root that
already contains per-video sub-directories with a ``masks_*`` folder.
The output is two CSVs written to the processing root,
``metrics_per_frame.csv`` and ``metrics_per_video.csv``, both with
``sep=';'``.

Reference for the metric design: Yemini, E. et al., "A database of
Caenorhabditis elegans behavioral phenotypes", Nature Methods 10,
877-879, 2013, DOI: 10.1038/nmeth.2560 (validated for nematodes;
applied here by analogy to flatworms).
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import (
    CHECKPOINTS_SEGMENTATION_DIR,
    PROCESSING_ROOT_DEFAULT,
)
from ..widgets import LogPanel
from ..workers.pipeline_d_worker import PipelineDParams, PipelineDWorker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class PipelineDPage(BasePage):
    """Run end-to-end metric extraction over an existing or regenerated tree."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Mask metrics - Mask metrics", parent)
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

        title = QLabel("Mask metrics - Mask metrics")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        intro = QLabel(
            "Mask metrics extracts per-frame morphometrics, well-normalised "
            "spatial descriptors, temporal motility, body-deformation "
            "dynamics, body-wave analysis, eigenpostures (PCA, "
            "<code>random_state=42</code>) and ecotoxicology composite "
            "endpoints from the binary masks under the processing root. "
            "When a per-video directory does not yet exist for the "
            "requested sampling step, the worker falls back to Step 1 + "
            "Step 2 to regenerate frames and masks at that step."
        )
        intro.setObjectName("mutedLabel")
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Inputs
        inputs_box = QGroupBox("Inputs")
        inputs_box.setObjectName("sectionBox")
        inputs_form = QFormLayout(inputs_box)
        inputs_form.setContentsMargins(12, 16, 12, 12)

        self._videos_edit = QLineEdit()
        self._videos_edit.setPlaceholderText(
            "Directory of input .mp4 files (used only when regeneration "
            "is required)"
        )
        videos_row = QHBoxLayout()
        videos_row.addWidget(self._videos_edit, 1)
        videos_browse = QPushButton("Browse...")
        videos_browse.clicked.connect(
            lambda: self._browse_directory(self._videos_edit)
        )
        videos_row.addWidget(videos_browse)
        videos_w = QWidget()
        videos_w.setLayout(videos_row)
        inputs_form.addRow("Input videos directory:", videos_w)

        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText(
            "Processed_videos_Frame_by_frame/"
        )
        self._root_edit.textChanged.connect(self._on_root_changed)
        root_row = QHBoxLayout()
        root_row.addWidget(self._root_edit, 1)
        root_browse = QPushButton("Browse...")
        root_browse.clicked.connect(
            lambda: self._browse_directory(self._root_edit)
        )
        root_row.addWidget(root_browse)
        root_w = QWidget()
        root_w.setLayout(root_row)
        inputs_form.addRow("Processing root:", root_w)

        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(
            "Defaults to the processing root"
        )
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_edit, 1)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(
            lambda: self._browse_directory(self._out_edit)
        )
        out_row.addWidget(out_browse)
        out_w = QWidget()
        out_w.setLayout(out_row)
        inputs_form.addRow("Output directory:", out_w)

        root.addWidget(inputs_box)

        # Parameters
        params_box = QGroupBox("Parameters")
        params_box.setObjectName("sectionBox")
        params_form = QFormLayout(params_box)
        params_form.setContentsMargins(12, 16, 12, 12)

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.05, 30.0)
        self._step_spin.setSingleStep(0.5)
        self._step_spin.setDecimals(2)
        params_form.addRow("Target step (s):", self._step_spin)

        self._eigen_spin = QSpinBox()
        self._eigen_spin.setRange(2, 20)
        params_form.addRow("Eigenposture components:", self._eigen_spin)

        self._well_cb = QCheckBox(
            "Detect the well by Hough transform on the first frame and "
            "express centroid position in normalised polar coordinates"
        )
        self._well_cb.setChecked(True)
        params_form.addRow("", self._well_cb)

        self._seg_model_edit = QLineEdit()
        self._seg_model_edit.setPlaceholderText(
            "Same as Step 2 (used only when masks must be regenerated)"
        )
        params_form.addRow("Segmentation model name:", self._seg_model_edit)

        root.addWidget(params_box)

        # Log + actions
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        # Auto-open the output directory on successful finish.
        self._open_dir_on_done = QCheckBox(
            "Open output directory when the pipeline finishes"
        )
        self._open_dir_on_done.setChecked(True)
        root.addWidget(self._open_dir_on_done)

        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_current)
        self._open_results_btn = QPushButton("Open results dir")
        self._open_results_btn.setToolTip(
            "Open the Output dir folder in the OS file explorer. "
            "The folder is created if it does not exist yet."
        )
        self._open_results_btn.clicked.connect(
            self._on_open_results_dir
        )
        action_row.addWidget(self._run_btn)
        action_row.addWidget(self._cancel_btn)
        action_row.addWidget(self._open_results_btn)
        action_row.addStretch(1)
        self._progress = QProgressBar()
        self._progress.setValue(0)
        action_row.addWidget(self._progress, 2)
        root.addLayout(action_row)

        self.run_state_changed.connect(self._on_running_changed)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_from_settings(self) -> None:
        s = self._settings
        self._videos_edit.setText(
            s.get("input_video_directory", "Videos_to_process")
        )
        proc_root = (
            str(s.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._root_edit.setText(proc_root)
        out_default = (
            str(s.get("pipelined_output_directory", "")).strip()
            or proc_root
        )
        self._out_edit.setText(out_default)
        self._step_spin.setValue(float(s.get("step1_step_seconds", 1.0)))
        self._eigen_spin.setValue(int(s.get("pipelined_eigen_components", 6)))
        self._well_cb.setChecked(bool(s.get("pipelined_detect_well", True)))
        self._seg_model_edit.setText(
            str(s.get("step2_checkpoint_model", ""))
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _browse_directory(self, edit: QLineEdit) -> None:
        start = edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select directory", start,
        )
        if path:
            edit.setText(path)

    def _on_root_changed(self, text: str) -> None:
        # Mirror the processing root into the output directory only
        # when the user has not customised it.
        current_out = self._out_edit.text().strip()
        if not current_out or current_out == self._settings.get(
            "processing_root", ""
        ):
            self._out_edit.setText(text)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        proc_root = self._root_edit.text().strip()
        if not proc_root:
            _msg_critical(
                self, "Mask metrics",
                "Select a processing root directory.",
            )
            return
        os.makedirs(proc_root, exist_ok=True)
        out_dir = self._out_edit.text().strip() or proc_root
        os.makedirs(out_dir, exist_ok=True)

        self._settings.set("processing_root", proc_root)
        self._settings.set("pipelined_output_directory", out_dir)
        self._settings.set(
            "pipelined_eigen_components", int(self._eigen_spin.value())
        )
        self._settings.set(
            "pipelined_detect_well", bool(self._well_cb.isChecked())
        )

        params = PipelineDParams(
            processing_root=proc_root,
            input_videos_directory=self._videos_edit.text().strip(),
            output_directory=out_dir,
            step_seconds=float(self._step_spin.value()),
            seg_checkpoint_root=str(
                self._settings.get(
                    "step2_checkpoint_dir", CHECKPOINTS_SEGMENTATION_DIR,
                )
            ),
            seg_model_name=self._seg_model_edit.text().strip(),
            detect_well=bool(self._well_cb.isChecked()),
            eigenposture_components=int(self._eigen_spin.value()),
        )

        self._progress.setValue(0)
        self._log_panel.clear()
        worker = PipelineDWorker(params)
        worker.progress_signal.connect(self._progress.setValue)
        worker.log_signal.connect(self._log_panel.append)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append(f"ERROR: {msg}")
        )
        worker.result_signal.connect(self._on_done)
        self._launch_worker(worker)

    def _on_done(self, result: object) -> None:
        if isinstance(result, dict):
            self._log_panel.append(
                "Mask metrics finished: "
                f"{result.get('n_videos', 0)} video(s), "
                f"{result.get('n_frames', 0)} frame(s). "
                f"Wrote {result.get('per_frame_csv', '')} and "
                f"{result.get('per_video_csv', '')}."
            )
        if (
            hasattr(self, "_open_dir_on_done")
            and self._open_dir_on_done.isChecked()
        ):
            target = self._root_edit.text().strip() or "."
            if os.path.isdir(target):
                from ..widgets.image_pair_viewer import open_folder
                if open_folder(target):
                    self._log_panel.append(
                        f"Opened output directory: {target}"
                    )
                else:
                    self._log_panel.append(
                        f"Cannot open output directory "
                        f"(open_folder returned False): {target}"
                    )
            else:
                self._log_panel.append(
                    f"Cannot open output directory (path not found): "
                    f"{target}"
                )

    def _on_open_results_dir(self) -> None:
        target = self._root_edit.text().strip()
        if not target:
            self._log_panel.append(
                "Cannot open results directory: the Output dir "
                "field is empty."
            )
            return
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as exc:
            self._log_panel.append(
                f"Cannot create results directory {target}: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        from ..widgets.image_pair_viewer import open_folder
        if open_folder(target):
            self._log_panel.append(
                f"Opened results directory: {target}"
            )
        else:
            self._log_panel.append(
                f"Failed to open results directory: {target}"
            )

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
