"""
Step 3 page: classification-ready image generation UI.

All tunable parameters (stereotype directory name, image size,
margin %, grayscale, mask application, number of planaria per
image) live in the Settings page. The planaria-rows parameter was
removed because the position is already encoded in the ``position``
field of the v1.2 naming convention.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..notebook_api.classification_images import get_mask_folders
from ..widgets import ImageTripleViewer, LogPanel
from ..workers.step3_worker import Step3Params, Step3Worker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class Step3Page(BasePage):
    """Generate per-planarian classification-ready images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Step 3 - Classification images", parent)
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

        title = QLabel("Step 3 - Generate classification-ready images")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        hint = QLabel(
            "Processing parameters are configured in the Settings page."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        input_box = QGroupBox("Input")
        input_box.setObjectName("sectionBox")
        input_layout = QHBoxLayout(input_box)
        input_layout.setContentsMargins(12, 16, 12, 12)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Processed_videos_Frame_by_frame/")
        self._input_edit.textChanged.connect(self._on_input_changed)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_input)
        input_layout.addWidget(self._input_edit, 1)
        input_layout.addWidget(browse_btn)
        root.addWidget(input_box)

        # Mask-directory selector (runtime choice depending on Step 2 output).
        mask_box = QGroupBox("Mask directory")
        mask_box.setObjectName("sectionBox")
        mask_row = QHBoxLayout(mask_box)
        mask_row.setContentsMargins(12, 16, 12, 12)
        self._mask_combo = QComboBox()
        self._mask_combo.setEditable(True)
        self._mask_combo.currentTextChanged.connect(
            lambda t: self._settings.set("step3_mask_dir", t)
        )
        refresh_mask_btn = QPushButton("Refresh")
        refresh_mask_btn.clicked.connect(self._refresh_mask_dirs)
        mask_row.addWidget(self._mask_combo, 1)
        mask_row.addWidget(refresh_mask_btn)
        root.addWidget(mask_box)

        # Splitter: folder list on left, preview + log on right
        splitter = QSplitter(Qt.Horizontal)

        folders_box = QGroupBox("Video folders")
        folders_box.setObjectName("sectionBox")
        folders_layout = QVBoxLayout(folders_box)
        folders_layout.setContentsMargins(12, 16, 12, 12)
        self._folders_list = QListWidget()
        self._folders_list.setSelectionMode(QListWidget.ExtendedSelection)
        folders_layout.addWidget(self._folders_list, 1)
        refresh_folders_btn = QPushButton("Refresh")
        refresh_folders_btn.clicked.connect(self._refresh_folders)
        folders_layout.addWidget(refresh_folders_btn)
        splitter.addWidget(folders_box)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        self._preview = ImageTripleViewer(
            titles=("Original", "Mask", "Output"),
        )
        right_layout.addWidget(self._preview, 2)
        self._log_panel = LogPanel()
        right_layout.addWidget(self._log_panel, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
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

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_from_settings(self) -> None:
        self._input_edit.setText(self._settings.get("processing_root", ""))
        self._refresh_folders()
        self._refresh_mask_dirs()
        saved_mask = self._settings.get("step3_mask_dir", "")
        if saved_mask:
            self._mask_combo.setCurrentText(saved_mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_input_changed(self, text: str) -> None:
        self._settings.set("processing_root", text)
        self._refresh_folders()
        self._refresh_mask_dirs()

    def _on_browse_input(self) -> None:
        start = self._input_edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select input directory", start
        )
        if path:
            self._input_edit.setText(path)

    def _refresh_folders(self) -> None:
        root = self._input_edit.text().strip()
        self._folders_list.clear()
        if not root or not os.path.isdir(root):
            return
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name)):
                self._folders_list.addItem(name)

    def _refresh_mask_dirs(self) -> None:
        """Aggregate mask folder names across all video folders."""
        root = self._input_edit.text().strip()
        self._mask_combo.clear()
        if not root or not os.path.isdir(root):
            return
        seen = set()
        for folder in sorted(os.listdir(root)):
            full = os.path.join(root, folder)
            if not os.path.isdir(full):
                continue
            for mask_name in get_mask_folders(full):
                if mask_name not in seen:
                    seen.add(mask_name)
                    self._mask_combo.addItem(mask_name)

    def _selected_folders(self) -> List[str]:
        return [item.text() for item in self._folders_list.selectedItems()]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        root = self._input_edit.text().strip()
        if not root or not os.path.isdir(root):
            _msg_warning(
                self, "Step 3", "Select a valid processing directory."
            )
            return
        mask_dir = self._mask_combo.currentText().strip() or None

        self._progress.setValue(0)
        self._log_panel.clear()
        params = Step3Params(
            processing_root=root,
            name_stereotypie_dir=str(
                self._settings.get(
                    "step3_stereotypie_dir",
                    "Formated",
                )
            ),
            image_size=int(self._settings.get("step3_image_size", 224)),
            margin_percent=float(
                self._settings.get("step3_margin_percent", 0.0)
            ),
            black_and_white=bool(
                self._settings.get("step3_black_and_white", False)
            ),
            apply_mask=bool(self._settings.get("step3_apply_mask", True)),
            number_of_planaria_on_image=int(
                self._settings.get("step3_num_planaria", 1)
            ),
            mask_dir=mask_dir,
            folders_stereotypie_to_process=self._selected_folders(),
        )
        worker = Step3Worker(params)
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
                f"Wrote {result.get('total_written')} image(s) into "
                f"{result.get('processed_dir_name')}"
            )

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
