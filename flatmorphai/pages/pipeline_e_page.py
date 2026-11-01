"""
Scrunching detector page: per-frame scrunching versus gliding state.

The page exposes the inputs that drive the algorithm in
:mod:`flatmorphai.notebook_api.scrunching`: processing root,
sampling step, sliding window, scrunching frequency band and the
amplitude / directionality thresholds. It also offers a Hough-based
mask regeneration option that reuses Step 1 + Step 2 when the
existing per-video directories were sampled at a different step.

References (full citations in
:mod:`flatmorphai.notebook_api.scrunching`):
- Cochet-Escartin et al., 2015, DOI: 10.1088/1478-3975/12/5/056010.
- Sabry et al., 2020, DOI: 10.3791/61549.
- Reho, Lelievre and Cadiou, 2022, DOI: 10.3389/fnmol.2022.935918.
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
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import (
    CHECKPOINTS_SEGMENTATION_DIR,
    PROCESSING_ROOT_DEFAULT,
)
from ..notebook_api.scrunching import (
    DEFAULT_WINDOW_SECONDS,
    GLIDING_AMPLITUDE_THRESHOLD,
    GLIDING_DIRECTIONALITY,
    IMMOBILE_SPEED_PX_PER_S,
    SCRUNCHING_AMPLITUDE_THRESHOLD,
    SCRUNCHING_FREQ_BAND_HZ,
    recommend_step_seconds,
    step_seconds_warning,
)
from ..widgets import LogPanel
from ..workers.pipeline_e_worker import PipelineEParams, PipelineEWorker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class PipelineEPage(BasePage):
    """Run the scrunching / gliding classifier over a processing root."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Scrunching detector - Scrunching detector", parent)
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

        title = QLabel("Scrunching detector - Scrunching detector")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        intro = QLabel(
            "Scrunching detector reconstructs the body length L(t) from the masks, "
            "runs a sliding-window FFT over it and labels every frame as "
            "<b>scrunching</b>, <b>gliding</b>, <b>peristalsis</b>, "
            "<b>immobile</b> or <b>mixed</b>. Scrunching is the "
            "musculature-driven escape gait described in Cochet-Escartin "
            "et al., 2015, DOI: 10.1088/1478-3975/12/5/056010, and "
            "reviewed in Reho, Lelievre and Cadiou, 2022, DOI: "
            "10.3389/fnmol.2022.935918. The default 0.2-0.8 Hz band and "
            "0.20 amplitude cut-off match those papers. The classifier "
            "also reports a continuous <i>scrunching intensity</i> that "
            "you can plot against time."
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
            "Directory of input .mp4 files (only used when masks must "
            "be regenerated at a finer step)"
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

        self._seg_model_edit = QLineEdit()
        self._seg_model_edit.setPlaceholderText(
            "Same as Step 2 (only used during regeneration)"
        )
        inputs_form.addRow("Segmentation model name:", self._seg_model_edit)

        root.addWidget(inputs_box)

        # Algorithm parameters
        params_box = QGroupBox("Detector parameters")
        params_box.setObjectName("sectionBox")
        params_form = QFormLayout(params_box)
        params_form.setContentsMargins(12, 16, 12, 12)

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.05, 5.0)
        self._step_spin.setSingleStep(0.05)
        self._step_spin.setDecimals(3)
        self._step_spin.setValue(0.25)
        self._step_spin.valueChanged.connect(self._refresh_step_warning)
        params_form.addRow("Sampling step (s):", self._step_spin)

        self._step_warning = QLabel("")
        self._step_warning.setWordWrap(True)
        self._step_warning.setObjectName("mutedLabel")
        params_form.addRow("", self._step_warning)

        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(1.0, 60.0)
        self._window_spin.setSingleStep(0.5)
        self._window_spin.setDecimals(2)
        self._window_spin.setValue(DEFAULT_WINDOW_SECONDS)
        params_form.addRow("Sliding window (s):", self._window_spin)

        self._freq_lo = QDoubleSpinBox()
        self._freq_lo.setRange(0.05, 5.0)
        self._freq_lo.setSingleStep(0.05)
        self._freq_lo.setDecimals(2)
        self._freq_lo.setValue(SCRUNCHING_FREQ_BAND_HZ[0])
        self._freq_hi = QDoubleSpinBox()
        self._freq_hi.setRange(0.05, 5.0)
        self._freq_hi.setSingleStep(0.05)
        self._freq_hi.setDecimals(2)
        self._freq_hi.setValue(SCRUNCHING_FREQ_BAND_HZ[1])
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("from"))
        band_row.addWidget(self._freq_lo)
        band_row.addWidget(QLabel("Hz to"))
        band_row.addWidget(self._freq_hi)
        band_row.addWidget(QLabel("Hz"))
        band_w = QWidget()
        band_w.setLayout(band_row)
        params_form.addRow("Scrunching frequency band:", band_w)

        self._amp_scrunch = QDoubleSpinBox()
        self._amp_scrunch.setRange(0.0, 1.0)
        self._amp_scrunch.setSingleStep(0.01)
        self._amp_scrunch.setDecimals(3)
        self._amp_scrunch.setValue(SCRUNCHING_AMPLITUDE_THRESHOLD)
        params_form.addRow(
            "Scrunching amplitude threshold:", self._amp_scrunch,
        )

        self._amp_glide = QDoubleSpinBox()
        self._amp_glide.setRange(0.0, 1.0)
        self._amp_glide.setSingleStep(0.01)
        self._amp_glide.setDecimals(3)
        self._amp_glide.setValue(GLIDING_AMPLITUDE_THRESHOLD)
        params_form.addRow(
            "Gliding amplitude threshold:", self._amp_glide,
        )

        self._immobile_spin = QDoubleSpinBox()
        self._immobile_spin.setRange(0.0, 1000.0)
        self._immobile_spin.setSingleStep(0.5)
        self._immobile_spin.setDecimals(2)
        self._immobile_spin.setValue(IMMOBILE_SPEED_PX_PER_S)
        params_form.addRow(
            "Immobile speed threshold (px / s):", self._immobile_spin,
        )

        self._dir_spin = QDoubleSpinBox()
        self._dir_spin.setRange(0.0, 1.0)
        self._dir_spin.setSingleStep(0.05)
        self._dir_spin.setDecimals(2)
        self._dir_spin.setValue(GLIDING_DIRECTIONALITY)
        params_form.addRow(
            "Gliding directionality threshold:", self._dir_spin,
        )

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
            str(s.get("pipelinee_output_directory", "")).strip()
            or proc_root
        )
        self._out_edit.setText(out_default)
        self._seg_model_edit.setText(
            str(s.get("step2_checkpoint_model", ""))
        )
        self._step_spin.setValue(
            float(s.get("pipelinee_step_seconds", 0.25))
        )
        self._window_spin.setValue(
            float(s.get("pipelinee_window_seconds", DEFAULT_WINDOW_SECONDS))
        )
        self._refresh_step_warning(self._step_spin.value())

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
        current_out = self._out_edit.text().strip()
        if not current_out or current_out == self._settings.get(
            "processing_root", ""
        ):
            self._out_edit.setText(text)

    def _refresh_step_warning(self, value: float) -> None:
        warning = step_seconds_warning(
            float(value),
            freq_band=(self._freq_lo.value(), self._freq_hi.value()),
        )
        if warning:
            self._step_warning.setText("WARNING: " + warning)
        else:
            self._step_warning.setText(
                "Recommended maximum step for the configured "
                f"frequency band: {recommend_step_seconds((self._freq_lo.value(), self._freq_hi.value())):.3f} s."
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        proc_root = self._root_edit.text().strip()
        if not proc_root:
            _msg_critical(
                self, "Scrunching detector",
                "Select a processing root directory.",
            )
            return
        os.makedirs(proc_root, exist_ok=True)
        out_dir = self._out_edit.text().strip() or proc_root
        os.makedirs(out_dir, exist_ok=True)

        self._settings.set("processing_root", proc_root)
        self._settings.set("pipelinee_output_directory", out_dir)
        self._settings.set(
            "pipelinee_step_seconds", float(self._step_spin.value())
        )
        self._settings.set(
            "pipelinee_window_seconds", float(self._window_spin.value())
        )

        params = PipelineEParams(
            processing_root=proc_root,
            input_videos_directory=self._videos_edit.text().strip(),
            output_directory=out_dir,
            step_seconds=float(self._step_spin.value()),
            window_seconds=float(self._window_spin.value()),
            freq_band_hz=(
                float(self._freq_lo.value()),
                float(self._freq_hi.value()),
            ),
            scrunching_amplitude_threshold=float(self._amp_scrunch.value()),
            gliding_amplitude_threshold=float(self._amp_glide.value()),
            immobile_speed_threshold=float(self._immobile_spin.value()),
            gliding_directionality_threshold=float(self._dir_spin.value()),
            seg_checkpoint_root=str(
                self._settings.get(
                    "step2_checkpoint_dir", CHECKPOINTS_SEGMENTATION_DIR,
                )
            ),
            seg_model_name=self._seg_model_edit.text().strip(),
        )

        self._progress.setValue(0)
        self._log_panel.clear()
        worker = PipelineEWorker(params)
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
                "Scrunching detector finished: "
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
