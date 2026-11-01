"""
Step 6 page: merge per-video CSVs into one ``all_videos.csv`` grouped
by subject (the first five underscore-separated tokens of each
per-video directory name). The page is intentionally minimal: only
the processing root, the output directory, the renumbering anchor
and the "lock output to processing root" toggle are exposed. The
period table that used to live here was removed because the v1.2.1
fusion derives every grouping field from the directory names and
never consulted the table at run time.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..widgets import LogPanel
from ..workers.step6_worker import Step6Params, Step6Worker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class Step6Page(BasePage):
    """Combine per-video stereotype CSVs into ``all_videos.csv``."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Step 6 - Period fusion", parent)
        self._settings = AppSettings()
        # Suppression flag used while we programmatically synchronise
        # the output line edit to the input line edit (so the change
        # does not register as a manual override).
        self._suspend_out_sync = False
        self._build_ui()
        self._restore_from_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Step 6 - Period CSV fusion")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        paths_box = QGroupBox("Paths")
        paths_box.setObjectName("sectionBox")
        paths_form = QFormLayout(paths_box)
        paths_form.setContentsMargins(12, 16, 12, 12)

        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Processed_videos_Frame_by_frame/")
        self._input_edit.textChanged.connect(self._on_input_changed)
        in_row = QHBoxLayout()
        in_row.addWidget(self._input_edit, 1)
        browse_in = QPushButton("Browse...")
        browse_in.clicked.connect(self._on_browse_input)
        in_row.addWidget(browse_in)
        in_w = QWidget()
        in_w.setLayout(in_row)
        paths_form.addRow("Processing root:", in_w)

        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(
            "Processed_videos_Frame_by_frame/ (all_videos.csv)"
        )
        self._out_edit.textChanged.connect(self._on_out_changed)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_edit, 1)
        browse_out = QPushButton("Browse...")
        browse_out.clicked.connect(self._on_browse_output)
        out_row.addWidget(browse_out)
        out_w = QWidget()
        out_w.setLayout(out_row)
        paths_form.addRow("Output directory:", out_w)

        # When checked, the output directory mirrors the processing
        # root automatically and the line edit is read-only. Default
        # is checked because the most common deployment writes
        # ``all_videos.csv`` next to the per-video folders.
        self._lock_out_cb = QCheckBox(
            "Use the processing root as the output directory"
        )
        self._lock_out_cb.setChecked(True)
        self._lock_out_cb.toggled.connect(self._on_lock_toggled)
        paths_form.addRow("", self._lock_out_cb)

        # Renumbering anchor stays exposed because it is the only
        # value the user might want to change between runs (e.g. when
        # appending a new batch and continuing the index sequence).
        self._start_idx_spin = QSpinBox()
        self._start_idx_spin.setRange(1, 99999)
        self._start_idx_spin.valueChanged.connect(
            lambda v: self._settings.set("step6_new_planaria_index", int(v))
        )
        paths_form.addRow("New planaria start index:", self._start_idx_spin)

        # Descriptive hint kept verbatim from the previous Metadata
        # section so the user sees what Step 6 will do without
        # opening the help.
        hint = QLabel(
            "Step 6 walks the processing root, groups per-video "
            "directories by the first five underscore-separated tokens "
            "of their name "
            "(<code>species_compound_concentration_plate_position</code>) "
            "and writes one merged <code>all_videos.csv</code> in the "
            "output directory. Each subject (one physical planarian) "
            "receives a unique <code>planaria_index</code> starting "
            "from the value above; rows from the same subject across "
            "different periods share that index."
        )
        hint.setObjectName("mutedLabel")
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        paths_form.addRow("", hint)

        root.addWidget(paths_box)

        # Output preview table, kept because it is informative once
        # the run has completed.
        preview_box = QGroupBox("Output preview")
        preview_box.setObjectName("sectionBox")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(12, 16, 12, 12)
        self._preview_table = QTableWidget(0, 0)
        self._preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        preview_layout.addWidget(self._preview_table, 1)
        root.addWidget(preview_box, 1)

        # Log panel
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        # Run / cancel actions
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
        from ..constants import PROCESSING_ROOT_DEFAULT

        proc_root = (
            str(self._settings.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._input_edit.setText(proc_root)

        # Restore the lock-state. When the saved output equals the
        # input, the lock stays on; otherwise the user has previously
        # picked a custom output directory and the lock is released.
        saved_out = (
            str(self._settings.get("step6_output_directory", "")).strip()
            or proc_root
        )
        lock_default = bool(
            self._settings.get("step6_lock_out_to_root", True)
        )
        if saved_out and saved_out != proc_root:
            lock_default = False
        self._lock_out_cb.setChecked(lock_default)
        if lock_default:
            saved_out = proc_root
        self._suspend_out_sync = True
        self._out_edit.setText(saved_out)
        self._suspend_out_sync = False
        self._out_edit.setReadOnly(lock_default)
        self._settings.set("step6_output_directory", saved_out)
        self._settings.set("step6_lock_out_to_root", lock_default)

        self._start_idx_spin.setValue(
            int(self._settings.get("step6_new_planaria_index", 1))
        )

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _on_input_changed(self, text: str) -> None:
        self._settings.set("processing_root", text)
        if self._lock_out_cb.isChecked():
            self._suspend_out_sync = True
            self._out_edit.setText(text)
            self._suspend_out_sync = False
            self._settings.set("step6_output_directory", text)

    def _on_out_changed(self, text: str) -> None:
        if self._suspend_out_sync:
            return
        self._settings.set("step6_output_directory", text)
        # Manual edit while the lock is on means the user wants a
        # different output directory; release the lock so the value
        # is no longer overwritten by input changes.
        if self._lock_out_cb.isChecked() and text != self._input_edit.text():
            self._lock_out_cb.setChecked(False)

    def _on_lock_toggled(self, checked: bool) -> None:
        self._settings.set("step6_lock_out_to_root", bool(checked))
        self._out_edit.setReadOnly(bool(checked))
        if checked:
            self._suspend_out_sync = True
            self._out_edit.setText(self._input_edit.text())
            self._suspend_out_sync = False
            self._settings.set(
                "step6_output_directory", self._input_edit.text()
            )

    def _on_browse_input(self) -> None:
        start = self._input_edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select input directory", start,
        )
        if path:
            self._input_edit.setText(path)

    def _on_browse_output(self) -> None:
        if self._lock_out_cb.isChecked():
            self._lock_out_cb.setChecked(False)
        start = self._out_edit.text() or os.getcwd()
        path = get_existing_directory(
            self, "Select output directory", start,
        )
        if path:
            self._out_edit.setText(path)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        root = self._input_edit.text().strip()
        out_dir = self._out_edit.text().strip()
        if not root or not os.path.isdir(root):
            _msg_critical(
                self, "Step 6", "Processing root does not exist.",
            )
            return
        if not out_dir:
            out_dir = root
            self._suspend_out_sync = True
            self._out_edit.setText(out_dir)
            self._suspend_out_sync = False
        os.makedirs(out_dir, exist_ok=True)

        self._progress.setValue(0)
        self._log_panel.clear()
        params = Step6Params(
            processing_root=root,
            output_directory=out_dir,
            new_planaria_index=int(self._start_idx_spin.value()),
        )
        worker = Step6Worker(params)
        worker.progress_signal.connect(self._progress.setValue)
        worker.log_signal.connect(self._log_panel.append)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append(f"ERROR: {msg}")
        )
        worker.result_signal.connect(self._on_done)
        self._launch_worker(worker)

    def _on_done(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        merged = result.get("periods_csv", "all_videos.csv")
        grouped = result.get("grouped_csv", "")
        if grouped:
            self._log_panel.append(f"Wrote {merged} and {grouped}")
        else:
            self._log_panel.append(f"Wrote {merged}")
        out_dir = self._out_edit.text().strip()
        merged_csv = os.path.join(out_dir, merged)
        self._preview_csv(merged_csv, max_rows=10)

    def _preview_csv(self, csv_path: str, max_rows: int = 10) -> None:
        if not csv_path or not os.path.isfile(csv_path):
            return
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, sep=";", encoding="latin1")
        except Exception as exc:  # noqa: BLE001
            self._log_panel.append(f"Preview failed: {exc}")
            return
        preview = df.head(max_rows)
        self._preview_table.setColumnCount(len(preview.columns))
        self._preview_table.setRowCount(len(preview))
        self._preview_table.setHorizontalHeaderLabels(
            [str(c) for c in preview.columns]
        )
        for r, (_, row) in enumerate(preview.iterrows()):
            for c, val in enumerate(row):
                self._preview_table.setItem(r, c, QTableWidgetItem(str(val)))

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
