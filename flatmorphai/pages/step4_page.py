"""
Step 4 page: stereotype forecasting UI.

All tunable parameters (classification checkpoint, prototype pickle,
batch size, number of worker processes) live in the Settings page.
Each page exposes only the inputs that depend on the current dataset
(processing root and selected videos), keeping a single source of
truth for hyper-parameters as recommended by Wilson, G. et al.,
"Good Enough Practices in Scientific Computing", PLOS Computational
Biology, 2017, DOI: 10.1371/journal.pcbi.1005510.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import (
    CLASSES_NAMES_STEREOTYPE,
)
from ..widgets import LogPanel
from ..workers.step4_worker import Step4Params, Step4Worker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class Step4Page(BasePage):
    """Forecast stereotype classes with the LightGBM bundle.

    (Until October 2026 this page ran a ConvNeXt-V2 model with a Mahalanobis
    prototype; the LightGBM bundle replaced it and is now the only option.)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Step 4 - Forecast", parent)
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

        title = QLabel("Step 4 - Forecast stereotype classes")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        hint = QLabel(
            "Model, prototype file and processing parameters are "
            "configured in the Settings page."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Input directory (runtime choice).
        input_box = QGroupBox("Input")
        input_box.setObjectName("sectionBox")
        input_row = QHBoxLayout(input_box)
        input_row.setContentsMargins(12, 16, 12, 12)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Processed_videos_Frame_by_frame/")
        self._input_edit.textChanged.connect(self._on_input_changed)
        browse_input = QPushButton("Browse...")
        browse_input.clicked.connect(self._on_browse_input)
        input_row.addWidget(self._input_edit, 1)
        input_row.addWidget(browse_input)
        root.addWidget(input_box)

        # Splitter: video list on left, results table + log on right
        splitter = QSplitter(Qt.Horizontal)

        videos_box = QGroupBox("Videos to process")
        videos_box.setObjectName("sectionBox")
        videos_layout = QVBoxLayout(videos_box)
        videos_layout.setContentsMargins(12, 16, 12, 12)
        self._videos_list = QListWidget()
        self._videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        videos_layout.addWidget(self._videos_list, 1)
        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh_videos)
        videos_layout.addWidget(refresh_btn)
        splitter.addWidget(videos_box)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self._results_table = QTableWidget(0, 2)
        self._results_table.setHorizontalHeaderLabels(["Class", "Count"])
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._results_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self._results_table, 2)

        self._log_panel = LogPanel()
        right_layout.addWidget(self._log_panel, 1)
        splitter.addWidget(right)

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

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_from_settings(self) -> None:
        self._input_edit.setText(self._settings.get("processing_root", ""))
        self._refresh_videos()
        self._init_results_table()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_results_table(self) -> None:
        self._results_table.setRowCount(len(CLASSES_NAMES_STEREOTYPE))
        for i, cls in enumerate(CLASSES_NAMES_STEREOTYPE):
            self._results_table.setItem(i, 0, QTableWidgetItem(cls))
            self._results_table.setItem(i, 1, QTableWidgetItem("0"))

    def _on_input_changed(self, text: str) -> None:
        self._settings.set("processing_root", text)
        self._refresh_videos()

    def _on_browse_input(self) -> None:
        start = self._input_edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select input directory", start
        )
        if path:
            self._input_edit.setText(path)

    def _refresh_videos(self) -> None:
        root = self._input_edit.text().strip()
        self._videos_list.clear()
        if not root or not os.path.isdir(root):
            return
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name)):
                self._videos_list.addItem(name)

    def _selected_videos(self) -> List[str]:
        return [item.text() for item in self._videos_list.selectedItems()]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        root = self._input_edit.text().strip()
        if not root or not os.path.isdir(root):
            _msg_critical(
                self, "Step 4", "Processing root does not exist."
            )
            return
        # Classifier: the LightGBM bundle is the only option.
        from ..notebook_api.lightgbm_bundle import (
            bundle_available as _bundle_available,
        )
        bundle_dir = str(self._settings.get("step4_bundle_dir", ""))
        if not _bundle_available(bundle_dir or None):
            _msg_critical(
                self,
                "Step 4",
                "LightGBM classifier not found"
                + (
                    f" at: {bundle_dir}"
                    if bundle_dir
                    else " in the default 'lightgbm_stereotype_classifier' "
                         "folder"
                )
                + ".\n\nSet 'LightGBM classifier folder' in the Step 4 "
                "tab of the Settings page.",
            )
            return

        self._progress.setValue(0)
        self._log_panel.clear()
        self._init_results_table()
        params = Step4Params(
            processing_root=root,
            bundle_dir=bundle_dir,
            num_workers=int(self._settings.get("step4_num_workers", 4)),
            max_workers=int(self._settings.get("step3_max_workers", 0)),
            videos_to_process=self._selected_videos(),
        )
        worker = Step4Worker(params)
        worker.progress_signal.connect(self._progress.setValue)
        worker.log_signal.connect(self._log_panel.append)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append(f"ERROR: {msg}")
        )
        worker.result_signal.connect(self._on_done)
        self._launch_worker(worker)

    def _on_done(self, result: object) -> None:
        if isinstance(result, dict):
            videos = result.get("videos", []) or []
            self._log_panel.append(
                f"Finished: processed {len(videos)} video(s) with model "
                f"{result.get('model_name')}"
            )
            counts = result.get("class_counts")
            if isinstance(counts, dict):
                for i, cls in enumerate(CLASSES_NAMES_STEREOTYPE):
                    self._results_table.setItem(
                        i, 1, QTableWidgetItem(str(counts.get(cls, 0))),
                    )

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
