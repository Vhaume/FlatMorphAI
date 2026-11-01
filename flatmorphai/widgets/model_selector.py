"""
Reusable widget to choose an AI-model checkpoint.

Layout: ``QComboBox`` (auto-populated by scanning the checkpoint
directory), ``QLineEdit`` for manual override, ``Browse`` button, and
``Validate`` button. Used in Steps 2 and 4.
"""

from __future__ import annotations

import os
from .file_dialogs import get_existing_directory
from typing import Callable, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ModelSelectorWidget(QWidget):
    """Small helper widget that resolves a model checkpoint name."""

    model_selected = Signal(str)   # emits the resolved checkpoint name
    validate_requested = Signal(str)  # emits the name when Validate is pressed

    def __init__(
        self,
        checkpoint_root: str,
        label: str = "Checkpoint",
        list_subdirs: Optional[Callable[[str], List[str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._checkpoint_root = checkpoint_root
        self._list_subdirs = list_subdirs or self._default_list_subdirs
        self._build_ui(label)
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self, label: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(220)
        self._combo.currentTextChanged.connect(self._on_combo_change)
        form.addRow(QLabel(f"{label}:"), self._combo)

        self._manual_path = QLineEdit()
        self._manual_path.setPlaceholderText("Manual override (path or name)")
        self._manual_path.editingFinished.connect(self._on_manual_change)
        form.addRow(QLabel("Manual:"), self._manual_path)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        self._validate_btn = QPushButton("Validate")
        self._validate_btn.setObjectName("primaryButton")
        self._validate_btn.clicked.connect(self._on_validate)

        buttons.addWidget(self._browse_btn)
        buttons.addWidget(self._refresh_btn)
        buttons.addStretch(1)
        buttons.addWidget(self._validate_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_list_subdirs(self, root: str) -> List[str]:
        if not os.path.isdir(root):
            return []
        entries = [
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]
        entries.sort(
            key=lambda n: os.path.getmtime(os.path.join(root, n)),
            reverse=True,
        )
        return entries

    def refresh(self) -> None:
        """Rescan the checkpoint directory."""
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        options = self._list_subdirs(self._checkpoint_root)
        self._combo.addItems(options)
        if current in options:
            self._combo.setCurrentText(current)
        elif options:
            self._combo.setCurrentIndex(0)
        self._combo.blockSignals(False)
        if self._combo.count() > 0:
            self.model_selected.emit(self._combo.currentText())

    def current_model_name(self) -> str:
        """Return the currently selected model name (manual path wins)."""
        manual = self._manual_path.text().strip()
        if manual:
            return manual
        return self._combo.currentText()

    def set_current(self, name: str) -> None:
        """Preselect a specific entry if available."""
        if not name:
            return
        idx = self._combo.findText(name)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        else:
            self._manual_path.setText(name)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_combo_change(self, name: str) -> None:
        if name:
            self.model_selected.emit(name)

    def _on_manual_change(self) -> None:
        name = self._manual_path.text().strip()
        if name:
            self.model_selected.emit(name)

    def _on_browse(self) -> None:
        start_dir = self._checkpoint_root if os.path.isdir(self._checkpoint_root) else ""
        path = get_existing_directory(self, "Select checkpoint folder", start_dir)
        if path:
            self._manual_path.setText(path)
            self.model_selected.emit(path)

    def _on_validate(self) -> None:
        self.validate_requested.emit(self.current_model_name())
