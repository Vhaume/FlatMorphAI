"""
Step 2 page: Mask2Former segmentation UI.

The per-run parameters (batch size, two-pass, merge strategy, keep
largest component, overwrite, checkpoint) live in the Settings page
only. Keeping a single source of truth for hyper-parameters is a
recommendation of Wilson, G. et al., "Good Enough Practices in
Scientific Computing", PLOS Computational Biology, 2017,
DOI: 10.1371/journal.pcbi.1005510, rule 4 (make everything you do
reproducible from one configuration).
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from ..config import AppSettings
from ..constants import CHECKPOINTS_SEGMENTATION_DIR
from ..widgets import ImagePairViewer, LogPanel
from ..notebook_api.segmentation import (
    resolve_most_recent_checkpoint as _resolve_most_recent_seg_checkpoint,
)
from ..workers.step2_worker import Step2Params, Step2Worker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class Step2Page(BasePage):
    """Frames-to-masks page.

    All tunable parameters (checkpoint name, batch size, two-pass,
    merge strategy, keep-largest, overwrite) are read from
    :class:`AppSettings` at run time. To change them, open the
    Settings page.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Step 2 - Frames to masks", parent)
        self._settings = AppSettings()
        self._build_ui()
        self._restore_from_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Step 2 - Frames to Semantic Masks")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        hint = QLabel(
            "Model and parameters are configured in the Settings page."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Input dir
        input_box = QGroupBox("Input")
        input_box.setObjectName("sectionBox")
        input_layout = QHBoxLayout(input_box)
        input_layout.setContentsMargins(12, 16, 12, 12)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Processed_videos_Frame_by_frame/")
        self._input_edit.textChanged.connect(
            lambda t: self._settings.set("processing_root", t)
        )
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_input)
        input_layout.addWidget(self._input_edit, 1)
        input_layout.addWidget(browse_btn)
        root.addWidget(input_box)

        # Log + preview splitter
        splitter = QSplitter(Qt.Horizontal)
        self._log_panel = LogPanel()
        self._preview = ImagePairViewer(
            left_title="Frame", right_title="Predicted mask"
        )
        splitter.addWidget(self._log_panel)
        splitter.addWidget(self._preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # Actions
        action_row = QHBoxLayout()
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

        self.run_state_changed.connect(self._on_running_changed)

    def _restore_from_settings(self) -> None:
        self._input_edit.setText(self._settings.get("processing_root", ""))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_browse_input(self) -> None:
        start = self._input_edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select input directory", start
        )
        if path:
            self._input_edit.setText(path)

    def _on_run(self) -> None:
        input_dir = (
            self._input_edit.text().strip()
            or "Processed_videos_Frame_by_frame"
        )
        if not os.path.isdir(input_dir):
            _msg_critical(
                self, "Step 2", "Processing root does not exist."
            )
            return
        seg_dir = str(
            self._settings.get(
                "step2_checkpoint_dir", CHECKPOINTS_SEGMENTATION_DIR
            )
        )
        model_name = str(self._settings.get("step2_checkpoint_model", ""))
        if not model_name:
            # Auto-resolve the most recent checkpoint sub-folder when the
            # user has not pinned one explicitly. Defaulting to the newest
            # trained model matches the "recognition rather than recall"
            # heuristic from Nielsen, J. and Molich, R., "Heuristic
            # evaluation of user interfaces", CHI 1990, DOI:
            # 10.1145/97243.97281.
            model_name = _resolve_most_recent_seg_checkpoint(seg_dir) or ""
        if not model_name:
            _msg_critical(
                self,
                "Step 2",
                "No segmentation checkpoint found under "
                f"{seg_dir}. Please configure the checkpoints folder "
                "in the Settings page.",
            )
            return
        try:
            import torch
            if not torch.cuda.is_available():
                _msg_critical(
                    self, "Step 2",
                    "No CUDA device detected. Step 2 requires a GPU.",
                )
                return
        except ImportError:
            _msg_critical(self, "Step 2", "PyTorch is not installed.")
            return

        self._progress.setValue(0)
        self._log_panel.clear()
        params = Step2Params(
            checkpoint_root=seg_dir,
            model_name=model_name,
            processing_root=input_dir,
            batch_size=int(self._settings.get("step2_batch_size", 8)),
            two_pass=bool(self._settings.get("step2_two_pass", True)),
            two_pass_merge=str(
                self._settings.get("step2_merge_strategy", "skeleton_guided")
            ),
            keep_largest_component=bool(
                self._settings.get("step2_keep_largest", True)
            ),
            overwrite=bool(self._settings.get("step2_overwrite", False)),
        )
        worker = Step2Worker(params)
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
                f"Wrote masks to {result.get('mask_directory_name')}"
            )

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
