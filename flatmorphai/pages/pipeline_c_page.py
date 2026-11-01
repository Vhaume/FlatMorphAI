"""
Image segmentation page: segmentation-only workflow on the processing root.

Image segmentation takes exactly the same input as Stereotype classification from images: the processing
root (``Processed_videos_Frame_by_frame/`` by default) that contains
one ``<video>_total_<T>s_step_<S>s/images/`` tree per video.  It runs
the Mask2Former segmentation on every video folder and writes the masks
into ``masks/`` next to ``images/``, exactly where Pipelines A
and B expect them.  No classification is performed.

Its purpose is quality control: an operator can inspect the masks of a
new acquisition before committing it to the full classification chain,
or produce masks for a use outside the application.  The mask-first
preview workflow is recommended for bulk phenotype scoring by
Carpenter, A. E. et al., "CellProfiler: image analysis software for
identifying and quantifying cell phenotypes", Genome Biology, 2006,
DOI: 10.1186/gb-2006-7-10-r100.

Three extra options, saved locally (keys ``pipeline_c_*`` in the
application config file) and restored when the page opens, refine the
masks.  They run in a dedicated subprocess that reuses the bundled
``Export_head_detection`` segmenter (see
:mod:`flatmorphai.workers.segmentation_c_pipeline`):

* Blasteme refined - after Mask2Former, the worm skeleton and its two
  extremities are computed and the blastema segmenter is run on both
  ends (exactly like the blastema pipeline).  The blastema prediction is
  combined (union) with the Mask2Former mask so a pale blastema that the
  segmenter cut off is recovered.  Off by default.

The base checkpoint and the "overwrite existing masks" behaviour are
configured in the Settings page (Step 2), following Wilson, G. et al.,
"Good Enough Practices in Scientific Computing", PLOS Computational
Biology, 2017, DOI: 10.1371/journal.pcbi.1005510.
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import CHECKPOINTS_SEGMENTATION_DIR
from ..constants import PROCESSING_ROOT_DEFAULT
from ..notebook_api.segmentation import (
    resolve_most_recent_checkpoint as _resolve_most_recent_seg_checkpoint,
)
from ..widgets import CollapsibleCard, LogPanel
from ..workers.segmentation_c_worker import SegmentationCParams, SegmentationCWorker
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class PipelineCPage(BasePage):
    """Segmentation-only pipeline over an already-extracted frame tree."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Image segmentation", parent)
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

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Image segmentation - Images to masks")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        notice = QLabel(
            "Give the processing root (default "
            "<code>Processed_videos_Frame_by_frame/</code>). Image segmentation "
            "searches every <code>images/</code> folder under it, segments its "
            "frames with Mask2Former and writes the masks to "
            "<code>masks/</code> next to <code>images/</code>, "
            "with the same file names as the images. No classification is "
            "performed. The base checkpoint and the 'overwrite existing masks' "
            "option are configured in the Settings page (Step 2); the options "
            "below refine the masks and choose the outputs, and are saved for "
            "the next run.<br>"
            "<b>Expected input:</b> image files inside an "
            "<code><b style='color:#0D9488;'>images/</b></code> sub-folder "
            "(e.g. <code>&lt;video&gt;/images/*.png</code>)."
        )
        notice.setObjectName("mutedLabel")
        notice.setTextFormat(Qt.RichText)
        notice.setWordWrap(True)
        root.addWidget(notice)

        # --- processing root (single-row card: caption + field + Browse) ---
        common_box = QFrame()
        common_box.setObjectName("card")
        cv = QVBoxLayout(common_box)
        cv.setContentsMargins(14, 12, 14, 12)
        cv.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(10)
        root_cap = QLabel("Processing root")
        root_cap.setObjectName("caption")
        row.addWidget(root_cap, 0, Qt.AlignVCenter)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText(PROCESSING_ROOT_DEFAULT)
        # Each page remembers its OWN processing root (saved locally the moment
        # the field changes - typing or picking a folder with the browse button
        # on the right), independent from the other pages.
        self._root_edit.textChanged.connect(
            lambda t: self._settings.set("pipeline_c_processing_root", t)
        )
        row.addWidget(self._root_edit, 3)
        browse_root = QPushButton("Browse")
        browse_root.clicked.connect(lambda: self._browse_dir(self._root_edit))
        row.addWidget(browse_root)
        cv.addLayout(row)
        convention_label = QLabel(
            "Every <code>…/images/*.png</code> folder found under the root is "
            "segmented and receives a sibling <code>masks/</code> "
            "folder. Existing masks are kept unless 'Overwrite existing masks' "
            "is enabled in Settings, Step 2."
        )
        convention_label.setObjectName("mutedLabel")
        convention_label.setTextFormat(Qt.RichText)
        convention_label.setWordWrap(True)
        cv.addWidget(convention_label)
        root.addWidget(common_box)

        # --- Mask refinement ------------------------------------------
        refine_box = QWidget()
        refine_box.setObjectName("collapsibleBody")
        rlay = QVBoxLayout(refine_box)
        rlay.setContentsMargins(16, 12, 16, 14)

        self._blasteme = QCheckBox("Full body mask refined with the blastema mask")
        self._blasteme.setToolTip(
            "After Mask2Former, run the blastema segmenter on both extremities "
            "and union its prediction with the full body mask, recovering a "
            "pale blastema the segmenter cut off. Off by default."
        )
        self._blasteme.toggled.connect(
            lambda v: self._settings.set("pipeline_c_blasteme_refined", bool(v))
        )
        rlay.addWidget(self._blasteme)

        # The blastema mask always keeps only its biggest detected component
        # (spurious smaller detections such as a reflection are dropped). This
        # is enforced unconditionally in the pipeline, so there is no longer a
        # user toggle for it.

        root.addWidget(self._collapsible(
            "Mask refinement", refine_box,
            "pipec_sec_maskrefine_expanded", default=True,
        ))

        # --- Optional outputs -----------------------------------------
        out_box = QWidget()
        out_box.setObjectName("collapsibleBody")
        oul = QVBoxLayout(out_box)
        oul.setContentsMargins(16, 12, 16, 14)
        self._save_body = QCheckBox("Save full body masks (folder 'masks')")
        self._save_body.setToolTip(
            "Keep the Mask2Former full body masks in 'masks'. This is "
            "the main output of this page; turn it off only if you want the "
            "blastema masks alone."
        )
        self._save_body.toggled.connect(
            lambda v: self._settings.set("pipeline_c_save_body_masks", bool(v))
        )
        oul.addWidget(self._save_body)
        self._save_bl_mask = QCheckBox("Save blastema masks (folder 'blasteme_masks')")
        self._save_bl_mask.setToolTip(
            "Save the detected blastema binary mask of each frame to a "
            "'blasteme_masks' folder next to images/. Runs the blastema "
            "detector on every frame."
        )
        self._save_bl_mask.toggled.connect(
            lambda v: self._settings.set("pipeline_c_save_blasteme_masks", bool(v))
        )
        oul.addWidget(self._save_bl_mask)
        self._save_head_csv = QCheckBox(
            "Save head position CSV (head_position.csv)"
        )
        self._save_head_csv.setToolTip(
            "Locate the worm head and write, per frame, the two extremity "
            "positions and which one is the head, to 'head_position.csv' next "
            "to images/. Runs the head detector on every frame."
        )
        self._save_head_csv.toggled.connect(
            lambda v: self._settings.set("pipeline_c_save_head_csv", bool(v))
        )
        oul.addWidget(self._save_head_csv)
        out_hint = QLabel(
            "You can save only the blastema masks by turning off 'Save full "
            "body masks'. At least one of the mask outputs must stay on."
        )
        out_hint.setObjectName("mutedLabel")
        out_hint.setWordWrap(True)
        oul.addWidget(out_hint)
        root.addWidget(self._collapsible(
            "Optional outputs", out_box,
            "pipec_sec_outputs_expanded", default=False,
        ))

        # --- Save thumbnails ------------------------------------------
        th_box = QWidget()
        th_box.setObjectName("collapsibleBody")
        thl = QVBoxLayout(th_box)
        thl.setContentsMargins(16, 12, 16, 14)
        self._save_body_thumb = QCheckBox(
            "Save full body thumbnails (folder 'thumbnails_all_body')"
        )
        self._save_body_thumb.setToolTip(
            "Save a thumbnail of the whole worm (the frame masked by its body "
            "mask, cropped to the worm) to a 'thumbnails_all_body' folder next "
            "to images/, like the other pages."
        )
        self._save_body_thumb.toggled.connect(
            lambda v: self._settings.set(
                "pipeline_c_save_body_thumbnails", bool(v)
            )
        )
        thl.addWidget(self._save_body_thumb)
        self._save_bl_thumb = QCheckBox(
            "Save blastema thumbnails (folder 'thumbnails_blastemes')"
        )
        self._save_bl_thumb.setToolTip(
            "Keep the two end (top/bottom) blastema thumbnails per frame "
            "instead of discarding them. Requires the blastema detector to "
            "run (any blastema option above enables it)."
        )
        self._save_bl_thumb.toggled.connect(
            lambda v: self._settings.set(
                "pipeline_c_save_blasteme_thumbnails", bool(v)
            )
        )
        thl.addWidget(self._save_bl_thumb)
        root.addWidget(self._collapsible(
            "Save thumbnails", th_box,
            "pipec_sec_thumbnails_expanded", default=False,
        ))

        # --- log ------------------------------------------------------
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        self._open_dir_on_done = QCheckBox(
            "Open output directory when the pipeline finishes"
        )
        self._open_dir_on_done.setChecked(True)
        root.addWidget(self._open_dir_on_done)

        # --- actions --------------------------------------------------
        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run segmentation")
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
        # Per-page processing root. Fall back to the shared key (for a first
        # launch after the per-page split) then to the legacy source key.
        saved = str(s.get("pipeline_c_processing_root", "")).strip()
        if not saved:
            saved = str(s.get("processing_root", "")).strip()
        if not saved:
            saved = str(s.get("pipeline_c_source", "")).strip()
        self._root_edit.setText(saved or PROCESSING_ROOT_DEFAULT)

        self._blasteme.setChecked(bool(s.get("pipeline_c_blasteme_refined", False)))
        self._save_body.setChecked(bool(s.get("pipeline_c_save_body_masks", True)))
        self._save_bl_mask.setChecked(
            bool(s.get("pipeline_c_save_blasteme_masks", False))
        )
        self._save_head_csv.setChecked(
            bool(s.get("pipeline_c_save_head_csv", False))
        )
        self._save_body_thumb.setChecked(
            bool(s.get("pipeline_c_save_body_thumbnails", False))
        )
        self._save_bl_thumb.setChecked(
            bool(s.get("pipeline_c_save_blasteme_thumbnails", False))
        )

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
                self, "Image segmentation",
                "Processing root does not exist: " + root,
            )
            return

        # Resolve the segmentation checkpoint exactly as the rest of the
        # application does, so Image segmentation masks match Pipelines A and B.
        seg_dir = str(
            self._settings.get("step2_checkpoint_dir", CHECKPOINTS_SEGMENTATION_DIR)
        )
        seg_model = str(self._settings.get("step2_checkpoint_model", ""))
        if not seg_model:
            seg_model = _resolve_most_recent_seg_checkpoint(seg_dir) or ""
        if not seg_model:
            _msg_critical(
                self, "Image segmentation",
                "No segmentation checkpoint found under %s. Please configure "
                "the checkpoints folder in the Settings page." % seg_dir,
            )
            return
        try:
            import torch
            if not torch.cuda.is_available():
                _msg_critical(
                    self, "Image segmentation",
                    "No CUDA device detected. Image segmentation requires a GPU.",
                )
                return
        except ImportError:
            _msg_critical(self, "Image segmentation", "PyTorch is not installed.")
            return

        if not any((
            self._save_body.isChecked(), self._save_bl_mask.isChecked(),
            self._save_head_csv.isChecked(), self._save_body_thumb.isChecked(),
            self._save_bl_thumb.isChecked(),
        )):
            _msg_critical(
                self, "Image segmentation",
                "Select at least one output (a mask, the head position CSV "
                "or a thumbnail).",
            )
            return

        self._settings.set("pipeline_c_processing_root", root)
        self._progress.setValue(0)
        self._log_panel.clear()
        self._status_label.setText("")

        params = SegmentationCParams(
            processing_root=root,
            seg_checkpoint=os.path.join(seg_dir, seg_model),
            seg_model=seg_model,
            blasteme_refined=self._blasteme.isChecked(),
            # Always keep only the biggest blastema component.
            blasteme_keep_biggest=True,
            overwrite=bool(self._settings.get("step2_overwrite", False)),
            save_body_masks=self._save_body.isChecked(),
            save_blasteme_masks=self._save_bl_mask.isChecked(),
            save_blasteme_thumbnails=self._save_bl_thumb.isChecked(),
            save_head_csv=self._save_head_csv.isChecked(),
            save_body_thumbnails=self._save_body_thumb.isChecked(),
        )
        worker = SegmentationCWorker(params)
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
        self._log_panel.append("Image segmentation finished successfully")
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
