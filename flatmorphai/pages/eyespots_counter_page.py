"""
Eyespots counter page.

From a processing root (``Processed_videos_Frame_by_frame/`` by default) the
page finds every ``images/`` folder and, for each frame: generates the
Mask2Former mask when it is missing from ``masks/`` (existing masks are
reused), straightens the worm, locates its two extremities, builds the eyespot
detection thumbnail of EACH extremity, predicts the eyespot heat map, then counts
the eyespots of BOTH extremities together.

For every image it writes one row of ``eyespots_detection.csv`` next to
``images/`` with, per threshold, the eyespot count and the list of (x, y) eyespot
positions in the original image frame.  Two thresholds are used, following the
application convention: balanced (0.80) and positive (0.60); each gets its own
count column and its own positions column.

Two checkboxes control the optional image outputs: the thumbnail heat maps
(folder ``Eyespots heatmaps``) and the detection thumbnails (folder
``thumbnails_eyespots``).  Both are off by default and saved for both extremities
when enabled.

The heavy work runs in a separate process (see
:mod:`flatmorphai.workers.eyespots_counter_worker`) so this page only builds the
parameters, streams the log and reports progress.  The eyespot detector follows
Ronneberger, O., Fischer, P., Brox, T., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI, 2015, DOI: 10.1007/978-3-319-24574-4_28.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
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
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import CHECKPOINTS_SEGMENTATION_DIR, PROCESSING_ROOT_DEFAULT
from ..notebook_api.segmentation import (
    resolve_most_recent_checkpoint as _resolve_most_recent_seg_checkpoint,
)
from ..widgets import CollapsibleCard, LogPanel
from ..workers.eyespots_counter_worker import (
    EyespotsCounterParams,
    EyespotsCounterWorker,
)
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class EyespotsCounterPage(BasePage):
    """Count the eyespots of every worm under a processing root."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Eyespots counter", parent)
        self._settings = AppSettings()
        self._build_ui()
        self._restore_from_settings()

    def _collapsible(self, title, content, key: str, default: bool = True):
        """Wrap ``content`` in a CollapsibleCard whose state is persisted."""
        return CollapsibleCard(
            title, content,
            expanded=bool(self._settings.get(key, default)),
            on_toggle=lambda e, k=key: self._settings.set(k, bool(e)),
        )

    #: Defaults of THIS page. Its own settings keys on purpose: the keys
    #: ``eyespots_balanced_threshold`` / ``eyespots_positive_threshold`` are
    #: SHARED with the Forecast regeneration states page (they name its CSV
    #: columns) and are edited in Settings. Changing the threshold here must
    #: never move that page, so nothing below touches them.
    THR_STRICT_DEFAULT = 0.80
    THR_PERMISSIVE_DEFAULT = 0.60
    THR_MODE_DEFAULT = "strict"

    def _build_thresholds_card(self):
        """Collapsible 'Eyespot detection threshold' card.

        At each extremity the detector returns a heat map whose value is, for
        every pixel, the probability of being part of an eyespot. A local
        maximum becomes a counted eyespot once it reaches the threshold, so the
        threshold IS the strictness of the count. ONE of the two thresholds is
        used per run -- they are two readings of the same images, not two
        measurements to mix -- and the chosen one names the CSV columns.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QButtonGroup, QDoubleSpinBox, QGridLayout,
                                       QHBoxLayout, QLabel, QPushButton,
                                       QRadioButton)

        body = QWidget()
        body.setObjectName("collapsibleBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(10)

        intro = QLabel(
            "The detector gives each extremity a map of the <i>probability of "
            "being an eyespot</i>, between 0 and 1. A peak of that map is "
            "counted as an eyespot once it reaches the threshold: "
            "<b>raise it to count only what is unambiguous, lower it to also "
            "catch what is faint</b>. Choose the reading that matches your "
            "question - one threshold is used per run, and its value is written "
            "into the names of the CSV columns."
        )
        intro.setObjectName("mutedLabel")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)

        self._thr_group = QButtonGroup(body)
        self._thr_group.setExclusive(True)

        def _block(key, label, setting, default, definition):
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(2)

            radio = QRadioButton(label)
            self._thr_group.addButton(radio)
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(0.05, 0.99)
            spin.setSingleStep(0.05)
            spin.setValue(float(self._settings.get(setting, default)))
            spin.setMinimumWidth(90)

            note = QLabel(definition)
            note.setObjectName("mutedLabel")
            note.setWordWrap(True)
            note.setTextFormat(Qt.RichText)
            note.setContentsMargins(24, 0, 0, 0)

            grid.addWidget(radio, 0, 0, Qt.AlignVCenter)
            grid.addWidget(spin, 0, 1, Qt.AlignVCenter)
            grid.addWidget(note, 1, 0, 1, 3)
            grid.setColumnStretch(2, 1)
            lay.addLayout(grid)

            radio.toggled.connect(
                lambda v, k=key: self._on_threshold_mode(k) if v else None)
            spin.valueChanged.connect(
                lambda v, st=setting: (self._settings.set(st, round(float(v), 2)),
                                       self._refresh_threshold_hint()))
            return radio, spin

        self._mode_strict, self._thr_strict = _block(
            "strict", "Strict threshold", "eyespots_counter_strict",
            self.THR_STRICT_DEFAULT,
            "Counts only the peaks the detector is confident about: a dark, "
            "well-formed eyespot. The reading to use for comparing animals or "
            "for scoring a head that has finished regenerating.")
        self._mode_permissive, self._thr_permissive = _block(
            "permissive", "Permissive threshold", "eyespots_counter_permissive",
            self.THR_PERMISSIVE_DEFAULT,
            "Also counts the weak responses: an eyespot that is only starting "
            "to pigment, or one half-hidden by the edge of the mask. The "
            "reading for the first days of regeneration, at the price of "
            "occasional false positives on dark tissue.")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._thr_reset = QPushButton("Restore defaults")
        self._thr_reset.setToolTip(
            "Back to the values in service: the strict threshold at 0.80, "
            "selected, and the permissive one at 0.60.")
        self._thr_reset.clicked.connect(self._reset_thresholds)
        row.addWidget(self._thr_reset)
        row.addStretch(1)
        lay.addLayout(row)

        self._thr_hint = QLabel()
        self._thr_hint.setObjectName("mutedLabel")
        self._thr_hint.setWordWrap(True)
        self._thr_hint.setTextFormat(Qt.RichText)
        lay.addWidget(self._thr_hint)

        return self._collapsible(
            "Eyespot detection threshold", body,
            "eyespots_sec_thresholds_expanded", default=True,
        )

    def _threshold_mode(self):
        """"strict" or "permissive" -- the reading selected right now."""
        return "permissive" if self._mode_permissive.isChecked() else "strict"

    def _on_threshold_mode(self, key):
        self._settings.set("eyespots_counter_mode", key)
        self._refresh_threshold_hint()

    def _reset_thresholds(self):
        """Back to the defaults: both values and the selected reading."""
        self._thr_strict.setValue(self.THR_STRICT_DEFAULT)
        self._thr_permissive.setValue(self.THR_PERMISSIVE_DEFAULT)
        self._mode_permissive.setChecked(False)   # explicit: do not rely on the group
        self._mode_strict.setChecked(True)
        self._settings.set("eyespots_counter_strict", self.THR_STRICT_DEFAULT)
        self._settings.set("eyespots_counter_permissive", self.THR_PERMISSIVE_DEFAULT)
        self._settings.set("eyespots_counter_mode", self.THR_MODE_DEFAULT)
        self._refresh_threshold_hint()

    def _refresh_threshold_hint(self):
        """Grey out the unused value and name the columns that will be written."""
        try:
            strict = self._threshold_mode() == "strict"
            self._thr_strict.setEnabled(strict)
            self._thr_permissive.setEnabled(not strict)
            from ..workers import eyespot_shared
            a, _ap, b, _bp = eyespot_shared.eyespot_columns(
                float(self._thr_strict.value()), float(self._thr_permissive.value()))
            self._thr_hint.setText(
                "This run writes <code>%s</code> and its list of (x, y) "
                "positions. The threshold is part of the column name, so "
                "changing it renames the columns and an editor set on the "
                "previous names has to be pointed at the new ones.<br>"
                "These two values belong to this page only: the thresholds of "
                "the <i>Forecast regeneration states</i> page live in Settings "
                "and are not touched here." % (a if strict else b))
        except Exception:  # noqa: BLE001
            self._thr_hint.setText("")

    def _build_mask_refinement_card(self):
        """Collapsible 'Mask refinement' card for the eyespots segmentation."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox

        body = QWidget()
        body.setObjectName("collapsibleBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(8)

        self._bl_refine = QCheckBox("Full body mask refined with the blastema mask")
        self._bl_refine.setToolTip(
            "Before counting eyespots, grow the full body mask with the blastema "
            "prediction (same refinement as the other pipelines). The blastema "
            "mask always keeps only its biggest detected component. Off by "
            "default."
        )
        self._bl_refine.toggled.connect(
            lambda v: self._settings.set("eyespots_blasteme_refined", bool(v))
        )
        lay.addWidget(self._bl_refine)

        return self._collapsible(
            "Mask refinement", body,
            "eyespots_sec_refine_expanded", default=False,
        )

    def _build_autocorrect_card(self):
        """Corrections applied to the counts, each one optional.

        The page had no such section: every correction lived on the
        regeneration page, which counts with its own (strict) threshold. The
        rule below belongs to both, so it is offered here too, with its own
        setting -- the two chains already use different thresholds, and one
        page must not silently change the other's results.
        """
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 12, 16, 14)
        self._pair_rule = QCheckBox(
            "Accept the other eyespots at the permissive threshold when one is "
            "already certain"
        )
        self._pair_rule.setToolTip(
            "Eyespots come in pairs: once one of them passes the strict "
            "threshold, a second peak is plausible and is accepted at the "
            "permissive threshold.\n"
            "The guard is the entry condition -- with no certain eyespot "
            "nothing is admitted, so a noisy image stays protected by the "
            "strict threshold.\n"
            "Measured on the 52 reference images: one image changes (a second "
            "eyespot at 0.612 that missed 0.80), none goes above two, and the "
            "19 images without a certain eyespot peak at 0.314, far below the "
            "permissive threshold.\n"
            "Needs both thresholds to be meaningful: the strict one decides "
            "entry, the permissive one what is then admitted."
        )
        self._pair_rule.toggled.connect(
            lambda v: self._settings.set("eyespots_counter_pair_rule", bool(v))
        )
        lay.addWidget(self._pair_rule)
        hint = QLabel(
            "No other correction is applied here: the count is the number of "
            "eyespots detected on the image, with no comparison to the other "
            "dates."
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return self._collapsible(
            "Automatic data correction", body,
            "eyespots_sec_autocorrect_expanded", default=True,
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Eyespots counter")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        notice = QLabel(
            "Give the processing root (default "
            "<code>Processed_videos_Frame_by_frame/</code>). Every "
            "<code>images/</code> folder found under it is processed: a "
            "Mask2Former mask is generated when it is missing (existing masks in "
            "<code>masks/</code> are reused), the worm is "
            "straightened, its two extremities are located, and the eyespot "
            "detector predicts the eyespot heat map on each extremity. The "
            "eyespots of BOTH extremities are counted together and written to "
            "<code>eyespots_detection.csv</code> next to <code>images/</code>, "
            "with, for each of the two thresholds set below, the count and the "
            "list of (x, y) positions.<br>"
            "<b>Expected image name:</b> "
            "<code><b style='color:#0D9488;'>PLATE_POSITION_DATE_Species_Compound_"
            "Concentration[_Comment].&lt;ext&gt;</b></code> &nbsp;—&nbsp; e.g. "
            "<code>R1_Q1_2024-06-05_Djap_Caf_500.png</code> "
            "(same convention as Forecast regeneration states; any image format; "
            "Species / Compound / Concentration go to their own columns, Comment "
            "is optional). Names that do not match are listed in the log and "
            "skipped for the metadata."
        )
        notice.setObjectName("mutedLabel")
        notice.setTextFormat(Qt.RichText)
        notice.setWordWrap(True)
        root.addWidget(notice)

        # --- processing root (single-row card: caption + field + Browse) ---
        src_box = QFrame()
        src_box.setObjectName("card")
        sl = QHBoxLayout(src_box)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(10)
        src_cap = QLabel("Processing root")
        src_cap.setObjectName("caption")
        sl.addWidget(src_cap, 0, Qt.AlignVCenter)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText(PROCESSING_ROOT_DEFAULT)
        # Per-page processing root, saved locally the moment the field changes.
        self._root_edit.textChanged.connect(
            lambda t: self._settings.set("eyespots_processing_root", t)
        )
        sl.addWidget(self._root_edit, 3)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_dir(self._root_edit))
        sl.addWidget(browse)
        root.addWidget(src_box)

        # --- optional image outputs -----------------------------------
        # Both are off by default and saved per extremity (top/bottom) when
        # enabled; the CSV is always written. The choices are persisted so
        # the next run remembers them.
        out_box = QWidget()
        out_box.setObjectName("collapsibleBody")
        olay = QVBoxLayout(out_box)
        olay.setContentsMargins(16, 12, 16, 14)

        self._save_heatmaps = QCheckBox("Save eyespots heatmaps")
        self._save_heatmaps.setToolTip(
            "Save the predicted eyespot heat map of each extremity (top and "
            "bottom) to the 'Eyespots heatmaps' folder next to images/."
        )
        self._save_heatmaps.toggled.connect(
            lambda v: self._settings.set("eyespots_save_heatmaps", bool(v))
        )
        olay.addWidget(self._save_heatmaps)

        self._save_thumbnails = QCheckBox("Save eyespots thumbnails")
        self._save_thumbnails.setToolTip(
            "Save the detection thumbnail (vignette) of each extremity (top and "
            "bottom) to the 'thumbnails_eyespots' folder next to images/."
        )
        self._save_thumbnails.toggled.connect(
            lambda v: self._settings.set("eyespots_save_thumbnails", bool(v))
        )
        olay.addWidget(self._save_thumbnails)

        root.addWidget(self._build_thresholds_card())
        root.addWidget(self._build_autocorrect_card())
        root.addWidget(self._build_mask_refinement_card())

        root.addWidget(self._collapsible(
            "Optional outputs (the CSV is always written)", out_box,
            "eyespots_sec_outputs_expanded", default=True,
        ))

        # --- log ------------------------------------------------------
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        self._open_dir_on_done = QCheckBox(
            "Open the processing root when the run finishes"
        )
        self._open_dir_on_done.setChecked(True)
        root.addWidget(self._open_dir_on_done)

        # --- actions --------------------------------------------------
        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Count eyespots")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_current)
        self._open_results_btn = QPushButton("Open results dir")
        self._open_results_btn.setToolTip(
            "Open the processing root in the OS file explorer."
        )
        self._open_results_btn.clicked.connect(self._on_open_results_dir)
        action_row.addWidget(self._run_btn)
        action_row.addWidget(self._cancel_btn)
        action_row.addWidget(self._open_results_btn)
        self._status_label = QLabel("")
        self._status_label.setObjectName("mutedLabel")
        self._status_label.setWordWrap(False)
        action_row.addSpacing(12)
        action_row.addWidget(self._status_label, 3)
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
        self._root_edit.setText(
            str(s.get("eyespots_processing_root", "")).strip()
            or str(s.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._save_heatmaps.setChecked(bool(s.get("eyespots_save_heatmaps", False)))
        self._save_thumbnails.setChecked(bool(s.get("eyespots_save_thumbnails", False)))
        self._bl_refine.setChecked(bool(s.get("eyespots_blasteme_refined", False)))
        self._pair_rule.setChecked(bool(s.get("eyespots_counter_pair_rule", False)))
        self._thr_strict.setValue(
            float(s.get("eyespots_counter_strict", self.THR_STRICT_DEFAULT)))
        self._thr_permissive.setValue(
            float(s.get("eyespots_counter_permissive", self.THR_PERMISSIVE_DEFAULT)))
        _mode = str(s.get("eyespots_counter_mode", self.THR_MODE_DEFAULT))
        self._mode_permissive.setChecked(_mode == "permissive")
        self._mode_strict.setChecked(_mode != "permissive")
        self._refresh_threshold_hint()

    def _browse_dir(self, edit: QLineEdit) -> None:
        start = edit.text() or os.getcwd()
        path = get_existing_directory(self, "Select directory", start)
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        root = self._root_edit.text().strip() or PROCESSING_ROOT_DEFAULT
        if not os.path.isdir(root):
            _msg_critical(
                self, "Eyespots counter",
                "Processing root does not exist: " + root,
            )
            return

        # Resolve the segmentation checkpoint exactly as the rest of the
        # application does, so generated masks match Pipelines A and B.
        seg_dir = str(
            self._settings.get("step2_checkpoint_dir", CHECKPOINTS_SEGMENTATION_DIR)
        )
        seg_model = str(self._settings.get("step2_checkpoint_model", ""))
        if not seg_model:
            seg_model = _resolve_most_recent_seg_checkpoint(seg_dir) or ""
        if not seg_model:
            _msg_critical(
                self, "Eyespots counter",
                "No segmentation checkpoint found under %s. Please configure "
                "the checkpoints folder in the Settings page." % seg_dir,
            )
            return
        try:
            import torch
            if not torch.cuda.is_available():
                _msg_critical(
                    self, "Eyespots counter",
                    "No CUDA device detected. The eyespots counter requires a GPU.",
                )
                return
        except ImportError:
            _msg_critical(self, "Eyespots counter", "PyTorch is not installed.")
            return

        self._settings.set("eyespots_processing_root", root)
        self._progress.setValue(0)
        self._log_panel.clear()
        self._status_label.setText("")

        params = EyespotsCounterParams(
            processing_root=root,
            seg_checkpoint=os.path.join(seg_dir, seg_model),
            seg_model=seg_model,
            save_heatmaps=self._save_heatmaps.isChecked(),
            save_thumbnails=self._save_thumbnails.isChecked(),
            threshold_balanced=round(float(self._thr_strict.value()), 2),
            threshold_positive=round(float(self._thr_permissive.value()), 2),
            use_balanced=self._threshold_mode() == "strict",
            use_positive=self._threshold_mode() == "permissive",
            blasteme_refined=self._bl_refine.isChecked(),
            eyespots_pair_rule=self._pair_rule.isChecked(),
        )
        worker = EyespotsCounterWorker(params)
        worker.progress_signal.connect(self._progress.setValue)
        worker.log_signal.connect(self._log_panel.append)
        worker.log_signal.connect(self._on_log_status)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append("ERROR: " + msg)
        )
        worker.result_signal.connect(self._on_done)
        self._launch_worker(worker)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_log_status(self, line: str) -> None:
        text = line
        if text.startswith("[") and "] " in text:
            text = text.split("] ", 1)[1]
        if len(text) > 140:
            text = text[:137] + "..."
        self._status_label.setText(text)

    def _on_done(self, _result: object) -> None:
        self._log_panel.append("Eyespots counter finished successfully")
        if self._open_dir_on_done.isChecked():
            target = self._root_edit.text().strip() or "."
            if os.path.isdir(target):
                from ..widgets.image_pair_viewer import open_folder
                if not open_folder(target):
                    self._log_panel.append("Cannot open output directory: " + target)

    def _on_open_results_dir(self) -> None:
        target = self._root_edit.text().strip()
        if not target:
            self._log_panel.append("Cannot open results directory: the Root field is empty.")
            return
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as exc:
            self._log_panel.append(
                "Cannot create results directory %s: %s" % (target, exc)
            )
            return
        from ..widgets.image_pair_viewer import open_folder
        if not open_folder(target):
            self._log_panel.append("Failed to open results directory: " + target)

    def _on_running_changed(self, _name: str, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
