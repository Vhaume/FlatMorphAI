"""
Forecast regeneration states page.

From a processing root (``Processed_videos_Frame_by_frame/`` by default) the
page finds every ``images/`` folder, segments the frames with the bundled
Mask2Former (raw input, as the model was trained), then runs the head cascade of
``Export_head_detection`` (eye cups, blastema, body darkness), builds the state
composite and predicts the regeneration state of each image. It writes, next to
each ``images/`` folder, ``eyespots_detection.csv``, ``head_position.csv`` and
``regeneration_states.csv``, and on request the blastema CSV, the blastema
masks, the eyespot heat maps and the four thumbnail folders.

The heavy work runs in a separate process (see
:mod:`flatmorphai.workers.regeneration_worker`) so this page only builds the
parameters, streams the log and reports progress.
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import PROCESSING_ROOT_DEFAULT
from ..widgets import CollapsibleCard, LogPanel, PipelineTimeline
from ..workers.regeneration_worker import RegenerationParams, RegenerationWorker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class RegenerationPage(BasePage):
    """Predict the regeneration state of every image under a processing root."""

    # The three visible steps of the pipeline, mirrored by the STEP markers
    # emitted by the driver (0 segmentation, 1 head detection, 2 classification).
    STEPS = ["Step 1", "Step 2", "Step 3"]
    STEP_DESCRIPTIONS = [
        "Mask2Former segmentation (per-frame worm masks)",
        "Head detection (eye cups / blastema / darkness) and composite",
        "Regeneration state classification",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Forecast regeneration states", parent)
        self._settings = AppSettings()
        self._build_ui()
        self._restore_from_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _collapsible(self, title, content, key: str, default: bool = True):
        """Wrap ``content`` in a CollapsibleCard whose state is persisted.

        The content is a plain QWidget (no border of its own) so the card frame
        is the only tile boundary, with the header sitting inside it.
        """
        content.setObjectName("collapsibleBody")
        return CollapsibleCard(
            title, content,
            expanded=bool(self._settings.get(key, default)),
            on_toggle=lambda e, k=key: self._settings.set(k, bool(e)),
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Forecast regeneration states")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        notice = QLabel(
            "Give the processing root (default "
            "<code>Processed_videos_Frame_by_frame/</code>). Every "
            "<code>images/</code> folder found under it is processed: the "
            "frames are segmented (Mask2Former), then the head cascade, the "
            "eyespot and blastema detectors and the state classifier run on "
            "each image. The results (eyespots_detection.csv, head_position.csv, "
            "regeneration_states.csv) are written next to each "
            "<code>images/</code> folder.<br>"
            "<b>Expected image name:</b> "
            "<code><b style='color:#0D9488;'>Species_Compound_Concentration_"
            "Plate_Position_Date[_Comment].&lt;ext&gt;</b></code> &nbsp;—&nbsp; "
            "e.g. <code>Djap_Caf_+500_R1_Q1_2024-06-05.jpg</code> "
            "(any image format; the six fields are mandatory and Comment is "
            "optional). Names that do not match are listed in the log and "
            "skipped for the metadata."
        )
        notice.setObjectName("mutedLabel")
        notice.setTextFormat(Qt.RichText)
        notice.setWordWrap(True)
        root.addWidget(notice)

        # Step timeline (segmentation -> head detection -> classification),
        # like the other pipelines. Driven by the worker's step_signal.
        self._timeline = PipelineTimeline(
            self.STEPS, step_descriptions=self.STEP_DESCRIPTIONS
        )
        root.addWidget(self._timeline)

        # scroll area so the option groups never crowd the log
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        clay = QVBoxLayout(content)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(12)

        # --- processing root (single-row card: caption + field + Browse) ---
        src_card = QFrame()
        src_card.setObjectName("card")
        sl = QHBoxLayout(src_card)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(10)
        src_cap = QLabel("Processing root")
        src_cap.setObjectName("caption")
        sl.addWidget(src_cap, 0, Qt.AlignVCenter)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText(PROCESSING_ROOT_DEFAULT)
        # Per-page processing root, saved locally the moment the field changes.
        self._root_edit.textChanged.connect(
            lambda t: self._settings.set("regen_processing_root", t)
        )
        sl.addWidget(self._root_edit, 3)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_dir(self._root_edit))
        sl.addWidget(browse)
        clay.addWidget(src_card)

        # --- automatic data correction ---
        opt_box = QWidget()
        olay = QVBoxLayout(opt_box)
        olay.setContentsMargins(16, 12, 16, 14)
        self._auto_state = QCheckBox(
            "Automatic state increase correction (per planaria, non-decreasing over time)"
        )
        self._auto_state.toggled.connect(
            lambda v: self._settings.set("regen_auto_state_increase_correction", bool(v))
        )
        olay.addWidget(self._auto_state)
        self._auto_eyespots = QCheckBox(
            "Automatic eyespots count correction (per planaria, non-decreasing over time)"
        )
        self._auto_eyespots.setToolTip(
            "An eyespot, once regenerated, does not disappear: the balanced "
            "eyespot count must not drop over time within a worm. A transient "
            "under-count is lifted to the running maximum. The raw count is "
            "preserved; the corrected value drives the editor's eyespot curve."
        )
        self._auto_eyespots.toggled.connect(
            lambda v: self._settings.set("regen_auto_eyespots_increase_correction", bool(v))
        )
        olay.addWidget(self._auto_eyespots)
        self._eyespots_pair = QCheckBox(
            "Accept a second eyespot at the permissive threshold when one is "
            "already certain"
        )
        self._eyespots_pair.setToolTip(
            "Eyespots come in pairs: once one of them passes the strict "
            "threshold, a second peak is plausible and is accepted at the "
            "permissive threshold.\n"
            "The guard is the entry condition -- with no certain eyespot "
            "nothing is admitted, so a noisy image stays protected by the "
            "strict threshold.\n"
            "Measured on the 52 reference images: one image changes (a second "
            "eyespot at 0.612 that missed 0.80), none goes above two, and the "
            "19 images without a certain eyespot peak at 0.314, far below the "
            "permissive threshold."
        )
        self._eyespots_pair.toggled.connect(
            lambda v: self._settings.set("regen_eyespots_pair_rule", bool(v))
        )
        olay.addWidget(self._eyespots_pair)

        # The eyespot detection thresholds now live ONLY in Settings (the
        # "Eyespot detection thresholds" section). A hint points there.
        cup_hint = QLabel(
            "Eyespot detection thresholds are configured in the Settings page "
            "(Eyespot detection thresholds)."
        )
        cup_hint.setObjectName("mutedLabel")
        cup_hint.setWordWrap(True)
        olay.addWidget(cup_hint)
        card_autocorrect = self._collapsible(
            "Automatic data correction", opt_box,
            "regen_sec_autocorrect_expanded", default=True,
        )

        # --- mask refinement (same options as Image segmentation) ---
        refine_box = QWidget()
        rlay = QVBoxLayout(refine_box)
        rlay.setContentsMargins(16, 12, 16, 14)
        self._blasteme = QCheckBox("Full body mask refined with the blastema mask")
        self._blasteme.setToolTip(
            "Grow the full body (Mask2Former) mask with the blastema prediction "
            "(union) at both extremities, so a pale blastema the segmenter cut "
            "off is recovered. Off by default."
        )
        self._blasteme.toggled.connect(
            lambda v: self._settings.set("regen_blasteme_refined", bool(v))
        )
        rlay.addWidget(self._blasteme)
        # The blastema mask always keeps only its biggest detected component
        # (spurious smaller detections such as a reflection are dropped). This
        # is enforced unconditionally in the pipeline, so there is no longer a
        # user toggle for it.
        card_maskrefine = self._collapsible(
            "Mask refinement", refine_box,
            "regen_sec_maskrefine_expanded", default=False,
        )

        # --- optional outputs ---
        out_box = QWidget()
        oul = QVBoxLayout(out_box)
        oul.setContentsMargins(16, 12, 16, 14)
        self._save_masks = QCheckBox("Save body masks (folder 'masks')")
        self._save_masks.setToolTip(
            "Keep the Mask2Former body masks in the 'masks' folder "
            "(same naming as the stereotype and segmentation pages). They are always computed "
            "and used during the run; when this is off the folder is deleted "
            "afterwards. The editor's 'Show mask' view needs these masks, so "
            "keep them on to use it."
        )
        self._save_masks.toggled.connect(
            lambda v: self._settings.set("regen_save_body_masks", bool(v))
        )
        oul.addWidget(self._save_masks)
        self._save_heat = QCheckBox(
            "Save eyespots heatmaps (folder 'Eyespots heatmaps')"
        )
        self._save_heat.toggled.connect(
            lambda v: self._settings.set("regen_save_eyespots_heatmaps", bool(v))
        )
        oul.addWidget(self._save_heat)
        self._save_bl_mask = QCheckBox(
            "Save blastema masks (folder 'blasteme_masks')"
        )
        self._save_bl_mask.toggled.connect(
            lambda v: self._settings.set("regen_save_blasteme_masks", bool(v))
        )
        oul.addWidget(self._save_bl_mask)
        self._save_bl_csv = QCheckBox(
            "Save blastema CSV (surface_area_up, surface_area_down)"
        )
        self._save_bl_csv.toggled.connect(
            lambda v: self._settings.set("regen_save_blasteme_csv", bool(v))
        )
        oul.addWidget(self._save_bl_csv)
        card_outputs = self._collapsible(
            "Optional outputs", out_box,
            "regen_sec_outputs_expanded", default=False,
        )

        # --- thumbnails ---
        th_box = QWidget()
        thl = QVBoxLayout(th_box)
        thl.setContentsMargins(16, 12, 16, 14)
        self._th_all = QCheckBox("Save images: thumbnails_all_body (whole body, state channel 0)")
        self._th_head = QCheckBox("Save images: thumbnails_head_zoomed (head zoom, state channel 1)")
        self._th_regen = QCheckBox("Save images: thumbnails_regeneration (state composite fed to the model)")
        self._th_blast = QCheckBox("Save images: thumbnails_blastemes (blastema thumbnail of the head end)")
        self._th_cup = QCheckBox("Save images: thumbnails_eyespots (eye-cup thumbnail of the head end)")
        for cb, key in (
            (self._th_all, "regen_save_thumb_all_body"),
            (self._th_head, "regen_save_thumb_head_zoomed"),
            (self._th_regen, "regen_save_thumb_regeneration"),
            (self._th_blast, "regen_save_thumb_blastemes"),
            (self._th_cup, "regen_save_thumb_eyespots"),
        ):
            cb.toggled.connect(lambda v, k=key: self._settings.set(k, bool(v)))
            thl.addWidget(cb)
        card_thumbs = self._collapsible(
            "Save thumbnails (none by default; folders removed when unchecked)",
            th_box, "regen_sec_thumbnails_expanded", default=False,
        )

        # One option card per row (stacked single column).
        clay.addWidget(card_autocorrect)
        clay.addWidget(card_maskrefine)
        clay.addWidget(card_outputs)
        clay.addWidget(card_thumbs)
        clay.addStretch(1)

        scroll.setWidget(content)
        # Give the options area the larger share of the height so more of the
        # (now collapsible) settings are visible without scrolling.
        root.addWidget(scroll, 3)

        # --- log ---
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        self._open_dir_on_done = QCheckBox(
            "Open the processing root when the run finishes"
        )
        self._open_dir_on_done.setChecked(True)
        root.addWidget(self._open_dir_on_done)

        # --- actions ---
        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run forecast")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_current)
        self._open_results_btn = QPushButton("Open results dir")
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
            str(s.get("regen_processing_root", "")).strip()
            or str(s.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._auto_state.setChecked(
            bool(s.get("regen_auto_state_increase_correction", True))
        )
        self._auto_eyespots.setChecked(
            bool(s.get("regen_auto_eyespots_increase_correction", True))
        )
        self._eyespots_pair.setChecked(
            bool(s.get("regen_eyespots_pair_rule", False))
        )
        self._save_masks.setChecked(bool(s.get("regen_save_body_masks", True)))
        self._save_heat.setChecked(bool(s.get("regen_save_eyespots_heatmaps", False)))
        self._save_bl_mask.setChecked(bool(s.get("regen_save_blasteme_masks", False)))
        self._save_bl_csv.setChecked(bool(s.get("regen_save_blasteme_csv", False)))
        self._th_all.setChecked(bool(s.get("regen_save_thumb_all_body", False)))
        self._th_head.setChecked(bool(s.get("regen_save_thumb_head_zoomed", False)))
        self._th_regen.setChecked(bool(s.get("regen_save_thumb_regeneration", False)))
        self._th_blast.setChecked(bool(s.get("regen_save_thumb_blastemes", False)))
        self._th_cup.setChecked(bool(s.get("regen_save_thumb_eyespots", False)))
        self._blasteme.setChecked(bool(s.get("regen_blasteme_refined", False)))

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
                self, "Forecast regeneration states",
                "Processing root does not exist: " + root,
            )
            return
        self._settings.set("regen_processing_root", root)
        self._progress.setValue(0)
        self._log_panel.clear()
        self._status_label.setText("")
        self._timeline.reset()

        params = RegenerationParams(
            processing_root=root,
            save_body_masks=self._save_masks.isChecked(),
            save_eyespots_heatmaps=self._save_heat.isChecked(),
            save_blasteme_masks=self._save_bl_mask.isChecked(),
            save_blasteme_csv=self._save_bl_csv.isChecked(),
            save_thumb_all_body=self._th_all.isChecked(),
            save_thumb_head_zoomed=self._th_head.isChecked(),
            save_thumb_regeneration=self._th_regen.isChecked(),
            save_thumb_blastemes=self._th_blast.isChecked(),
            save_thumb_eyespots=self._th_cup.isChecked(),
            auto_state_increase_correction=self._auto_state.isChecked(),
            auto_eyespots_increase_correction=self._auto_eyespots.isChecked(),
            eyespots_pair_rule=self._eyespots_pair.isChecked(),
            eyespots_balanced_threshold=float(
                self._settings.get("eyespots_balanced_threshold", 0.80)
            ),
            eyespots_positive_threshold=float(
                self._settings.get("eyespots_positive_threshold", 0.60)
            ),
            blasteme_refined=self._blasteme.isChecked(),
            # Always keep only the biggest blastema component.
            blasteme_keep_biggest=True,
        )
        worker = RegenerationWorker(params)
        worker.step_signal.connect(self._on_step)
        worker.progress_signal.connect(self._progress.setValue)
        worker.log_signal.connect(self._log_panel.append)
        worker.log_signal.connect(self._on_log_status)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append("ERROR: " + msg)
        )
        worker.result_signal.connect(self._on_done)
        self._launch_worker(worker)

    def _on_step(self, index: int) -> None:
        """Advance the timeline to the step the driver just entered."""
        for j in range(len(self.STEPS)):
            if j < index:
                self._timeline.set_status(j, "done")
            elif j == index:
                self._timeline.set_status(j, "running")
        self._timeline.set_current(index)

    def _on_log_status(self, line: str) -> None:
        text = line
        if text.startswith("[") and "] " in text:
            text = text.split("] ", 1)[1]
        if len(text) > 140:
            text = text[:137] + "..."
        self._status_label.setText(text)

    def _on_done(self, _result: object) -> None:
        for j in range(len(self.STEPS)):
            self._timeline.set_status(j, "done")
        self._timeline.set_current(-1)
        self._log_panel.append("Forecast finished successfully")
        # Remember the generated regeneration_states.csv and its parent
        # processing root so the Regeneration states editor auto-loads them
        # when the user switches to it.
        self._remember_editor_csv(self._root_edit.text().strip())
        # Switch straight to the Regeneration states editor now it succeeded.
        self.navigate_requested.emit("regeneration_editor")
        if self._open_dir_on_done.isChecked():
            target = self._root_edit.text().strip() or "."
            if os.path.isdir(target):
                from ..widgets.image_pair_viewer import open_folder
                if not open_folder(target):
                    self._log_panel.append("Cannot open output directory: " + target)

    def _remember_editor_csv(self, root: str) -> None:
        """Persist the newest regeneration_states.csv and its root for the editor.

        The Regeneration states editor reads ``regedit_csv_path`` /
        ``regedit_processing_root`` and auto-loads them on open and when the
        user switches to it.
        """
        if not root:
            return
        self._settings.set("regedit_processing_root", root)
        newest, newest_mtime = "", -1.0
        skip = ("masks", "thumbnails_", "segmented_", "temp_")
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath).lower()
            if base.startswith(skip) or base == "eyespots heatmaps":
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if not d.lower().startswith(skip) and d.lower() != "eyespots heatmaps"
            ]
            if "regeneration_states.csv" in filenames:
                path = os.path.join(dirpath, "regeneration_states.csv")
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest, newest_mtime = path, mtime
        if newest:
            self._settings.set("regedit_csv_path", newest)

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
