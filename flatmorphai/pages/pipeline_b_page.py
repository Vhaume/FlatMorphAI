"""
Stereotype classification from images page: run Steps 2 -> 3 -> 4 -> 6 sequentially.

This pipeline assumes Step 1 (frame extraction) has already been
executed and a ``Processed_videos_Frame_by_frame`` tree already
exists. Step 5 is intentionally excluded.

All tunable hyper-parameters (segmentation checkpoint, classification
checkpoint, prototype file, batch sizes, merge strategy, etc.) live
in the Settings page. Each pipeline step tile exposes only the
per-run inputs, keeping a single source of truth for hyper-parameters
as recommended by Wilson, G. et al., "Good Enough Practices in
Scientific Computing", PLOS Computational Biology, 2017, DOI:
10.1371/journal.pcbi.1005510.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..widgets.collapsible_card import CollapsibleCard
from ..constants import (
    CHECKPOINTS_SEGMENTATION_DIR,
)
from ..notebook_api.segmentation import (
    resolve_most_recent_checkpoint as _resolve_most_recent_seg_checkpoint,
)
from ..widgets import LogPanel, PipelineTimeline
from ..workers.pipeline_runners import (
    run_step2,
    run_step3,
    run_step4,
    run_step6,
)
from ..workers.pipeline_worker import PipelineParams, PipelineStep, PipelineWorker
from ..workers.step2_worker import Step2Params
from ..workers.step3_worker import Step3Params
from ..workers.step4_worker import Step4Params
from ..workers.step6_worker import Step6Params
from .base_page import BasePage
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class PipelineBPage(BasePage):
    """Pipeline starting from already-extracted frames."""

    # Displayed labels are numbered CONSECUTIVELY (1..3).  Pipeline
    # B skips frame extraction because it starts from an
    # already-extracted directory, and the formatted-image stage no
    # longer exists anywhere.  Its three automated stages are
    # segmentation, classification and period fusion -- the last of
    # which is "Step 5" in the global numbering (the Stereotype
    # Editor is a standalone Tools page, not a numbered step).
    STEPS = ["Step 1", "Step 2", "Step 3", "Step 4"]
    # Stereotype classification from images skips frame extraction (it starts from an
    # already-extracted Processed_videos_Frame_by_frame directory).
    # Its automated stages are segmentation, classification-image
    # formatting, classification and period fusion. The formatting
    # stage builds the masked, cropped and resized images that the
    # LightGBM descriptor extractor reads (process_videos resolves a
    # ``_image-size(<N>)`` sub-folder and fails without it).
    STEP_DESCRIPTIONS = [
        "Mask2Former segmentation (per-frame worm masks)",
        "Classification-image formatting (mask-applied, cropped and "
        "resized to the classifier input size, 384 px)",
        "Classification (LightGBM; computes the descriptor table "
        "from the formatted images and masks, then predicts one of "
        "five stereotypes per frame)",
        "Period fusion (concatenates per-video CSVs into "
        "all_videos.csv; also all_videos_metrics.csv when the "
        "run produced per-video descriptors)",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Stereotype classification from images", parent)
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

        title = QLabel("Stereotype classification from images - Images to stereotype")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        hint = QLabel(
            "Hyper-parameters (segmentation model, classification model, "
            "prototype, batch sizes, merge strategy, etc.) are configured "
            "in the Settings page. This pipeline tile exposes only the "
            "processing root; the period fusion writes all_videos.csv into "
            "that root automatically."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._timeline = PipelineTimeline(
            self.STEPS, step_descriptions=self.STEP_DESCRIPTIONS,
        )
        root.addWidget(self._timeline)

        # Processing root selector: same bordered card as Image segmentation,
        # with the caption + field + Browse row layout of the Regeneration
        # states editor's source card.
        common_box = QFrame()
        common_box.setObjectName("card")
        cv = QVBoxLayout(common_box)
        cv.setContentsMargins(14, 12, 14, 12)
        cv.setSpacing(8)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("Processed_videos_Frame_by_frame")
        # Per-page processing root, saved locally the moment the field changes.
        self._root_edit.textChanged.connect(
            lambda t: self._settings.set("pipeline_b_processing_root", t)
        )
        browse_root = QPushButton("Browse")
        browse_root.clicked.connect(lambda: self._browse_dir(self._root_edit))
        row = QHBoxLayout()
        row.setSpacing(10)
        root_cap = QLabel("Processing root")
        root_cap.setObjectName("caption")
        row.addWidget(root_cap, 0, Qt.AlignVCenter)
        row.addWidget(self._root_edit, 3)
        row.addWidget(browse_root)
        cv.addLayout(row)

        convention_label = QLabel(
            "<b>Expected per-video folder name:</b> "
            "<code><b style='color:#0D9488;'>species_compound_concentration_"
            "plate_position_period</b></code> &nbsp;—&nbsp; e.g. "
            "<code>Smed_nicotine_10um_P1_B4_T0h</code> (each holding an "
            "<code>images/</code> sub-folder of frames).<br>"
            "<b>Expected image name</b> inside each <code>images/</code> "
            "sub-folder: "
            "<code><b style='color:#0D9488;'>&lt;folder_name&gt;_&lt;second&gt;"
            "s.png</b></code> &nbsp;—&nbsp; e.g. "
            "<code>Smed_nicotine_10um_P1_B4_T0h_0s.png</code>, "
            "<code>Smed_nicotine_10um_P1_B4_T0h_1s.png</code>, … "
            "(the second carries an <code>s</code> suffix, dropped decimals "
            "when integer, e.g. <code>_0.5s.png</code> at step 0.5 s) — exactly "
            "the names produced by Videos → Stereotype."
        )
        convention_label.setObjectName("mutedLabel")
        convention_label.setTextFormat(Qt.RichText)
        convention_label.setWordWrap(True)
        cv.addWidget(convention_label)

        root.addWidget(common_box)
        root.addWidget(self._build_mask_refinement_card())

        # Period fusion no longer has its own "Step 5" section: it is handled
        # automatically and always writes all_videos.csv into the processing
        # root. ``_s6_out`` is kept as a hidden field (synced to the root) so
        # the run logic and the saved ``step6_output_directory`` setting keep
        # working without an extra directory picker for the user to fill in.
        self._s6_out = QLineEdit()
        self._s6_out.setVisible(False)

        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        # Auto-open the output directory when the pipeline finishes
        # (same pattern as Stereotype classification; see comment there).
        self._open_dir_on_done = QCheckBox(
            "Open output directory when the pipeline finishes"
        )
        self._open_dir_on_done.setChecked(True)
        root.addWidget(self._open_dir_on_done)

        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run pipeline")
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
        # Lower-left status label mirroring the latest worker log line.
        # Placement follows Nielsen J., "10 Usability Heuristics for
        # User Interface Design", 1994, heuristic 1 (visibility of
        # system status).
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

    def _build_mask_refinement_card(self) -> QWidget:
        """Collapsible 'Mask refinement' card for the Images segmentation."""
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.setSpacing(8)

        self._bl_refine = QCheckBox("Full body mask refined with the blastema mask")
        self._bl_refine.setToolTip(
            "After segmentation, run the blastema segmenter on both extremities "
            "and union its prediction with the full body mask (same refinement "
            "as Image segmentation / Forecast regeneration states). The blastema "
            "mask always keeps only its biggest detected component. Off by "
            "default."
        )
        self._bl_refine.toggled.connect(
            lambda v: self._settings.set("stereotypie_blasteme_refined", bool(v))
        )
        lay.addWidget(self._bl_refine)

        # Folded/unfolded state saved and restored like every other card.
        return CollapsibleCard(
            "Mask refinement", body,
            expanded=bool(self._settings.get(
                "stereo_images_sec_refine_expanded", False)),
            on_toggle=lambda e: self._settings.set(
                "stereo_images_sec_refine_expanded", bool(e)),
        )

    def _restore_from_settings(self) -> None:
        # Stereotype classification from images IS Stereotype classification without the video-to-images step, so the
        # mask refinement is the SAME switch, not a second copy of it: this
        # page used to keep its own ``pipeline_b_blasteme_refined`` key, which
        # silently sent the two pages through two different segmentation
        # drivers (blastema-refined pins the model's 320x320 input and
        # announces each finished video; the plain two-pass one feeds 384x384
        # and announces nothing, so the overlap stayed idle and the masks were
        # not the ones Stereotype classification produced). The shared key is read below and
        # written by the toggle; the old per-page values are still honoured as
        # the starting point, A first, so nothing is lost on upgrade.
        self._bl_refine.setChecked(
            bool(self._settings.get(
                "stereotypie_blasteme_refined",
                self._settings.get(
                    "pipeline_a_blasteme_refined",
                    self._settings.get("pipeline_b_blasteme_refined", False),
                ),
            ))
        )
        self._root_edit.setText(
            str(self._settings.get("pipeline_b_processing_root", "")).strip()
            or str(self._settings.get("processing_root", "")).strip()
            or "Processed_videos_Frame_by_frame"
        )
        # Step 6 output directory defaults to the canonical processing
        # root so the fused ``all_videos.csv`` sits next to the per-video
        # Step 4 outputs. The fallback chain is:
        #   1. saved ``step6_output_directory`` if non-empty
        #   2. saved ``processing_root`` if non-empty
        #   3. the app-wide default ``PROCESSING_ROOT_DEFAULT``
        # The user can still override this at run time.
        from ..constants import PROCESSING_ROOT_DEFAULT
        saved_s6_out = (
            str(self._settings.get("step6_output_directory", "")).strip()
            or str(self._settings.get("processing_root", "")).strip()
            or PROCESSING_ROOT_DEFAULT
        )
        self._s6_out.setText(saved_s6_out)
        self._settings.set("step6_output_directory", saved_s6_out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _browse_dir(self, edit: QLineEdit) -> None:
        start = edit.text() or os.getcwd()
        path = get_existing_directory(self, "Select directory", start)
        if path:
            edit.setText(path)

    def _video_names(self) -> List[str]:
        root = self._root_edit.text().strip()
        if not root or not os.path.isdir(root):
            return []
        return sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )

    def _refresh_videos(self) -> None:
        # No-op: Step 6 (v1.2.1) enumerates per-video directories on
        # its own when the pipeline runs.
        return

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        root = self._root_edit.text().strip()
        if not root or not os.path.isdir(root):
            _msg_critical(
                self, "Stereotype classification from images", "Select a valid processing root."
            )
            return
        seg_dir = str(
            self._settings.get(
                "step2_checkpoint_dir", CHECKPOINTS_SEGMENTATION_DIR
            )
        )
        seg_model = str(self._settings.get("step2_checkpoint_model", ""))
        if not seg_model:
            seg_model = _resolve_most_recent_seg_checkpoint(seg_dir) or ""
        if not seg_model:
            _msg_critical(
                self, "Stereotype classification from images",
                f"No segmentation checkpoint found under {seg_dir}. "
                "Please configure the checkpoints folder in the Settings "
                "page.",
            )
            return

        # Classifier: the LightGBM bundle is the only option.
        from ..notebook_api.lightgbm_bundle import (
            bundle_available as _bundle_available,
        )
        _bundle_dir_saved = str(
            self._settings.get("step4_bundle_dir", "")
        )
        if not _bundle_available(_bundle_dir_saved or None):
            _msg_critical(
                self, "Stereotype classification from images",
                "LightGBM classifier not found"
                + (
                    f" at: {_bundle_dir_saved}"
                    if _bundle_dir_saved
                    else " in the default 'lightgbm_stereotype_classifier' "
                         "folder"
                )
                + ".\n\nSet 'LightGBM classifier folder' in the Step 4 "
                "tab of the Settings page.",
            )
            return
        try:
            import torch
            if not torch.cuda.is_available():
                _msg_critical(
                    self, "Stereotype classification from images",
                    "No CUDA device detected.  Step 2 requires a GPU "
                    "(LightGBM classification is CPU-only).",
                )
                return
        except ImportError:
            _msg_critical(self, "Stereotype classification from images", "PyTorch is not installed.")
            return

        # Period fusion always writes all_videos.csv into the processing root
        # (no separate output directory to pick): the fused CSV sits next to
        # the per-video Step 4 outputs it was built from. Rule 1 of Sandve
        # G. K. et al., "Ten Simple Rules for Reproducible Computational
        # Research", PLOS Comput. Biol. 2013, DOI: 10.1371/journal.pcbi.1003285.
        out_dir = root
        self._s6_out.setText(out_dir)
        self._settings.set("step6_output_directory", out_dir)
        os.makedirs(out_dir, exist_ok=True)

        self._progress.setValue(0)
        self._log_panel.clear()
        self._timeline.reset()

        # Stereotype classification from images is Stereotype classification without the video-to-image step: the images
        # are already there, so it starts at the segmentation. It therefore
        # deserves the same two accelerations, which until now only Stereotype classification
        # had because they were wired into its Step 1.
        #
        # 1. Linux work folder. Under WSL the root is on /mnt/c, where writing a
        #    full-resolution PNG costs ~0.30 s against ~0.07 s on the Linux
        #    disk; this run writes one mask and one formatted image per frame.
        #    The existing content is copied to the work folder first (reads are
        #    the cheap direction), everything is produced there, and the results
        #    are copied back while the next steps run.
        # 2. Overlap. The descriptors of a video start as soon as its masks are
        #    written, in a background pool, while the card segments the next
        #    one -- instead of waiting for every video to be segmented.
        # Both are the exact mechanisms of Stereotype classification, bit-identical outputs.
        # FLATMORPHAI_LINUX_WORKDIR=0 disables the first; the second is opt-in.
        from ..workers import linux_workdir as _LW
        _user_root = root
        _work = _LW.work_path(root)
        _same_out = os.path.abspath(out_dir) == os.path.abspath(root)
        if _work:
            root = _work
            if _same_out:
                out_dir = _work

        _stream = {"fs": None}
        _bundle_for_stream = _bundle_dir_saved

        def _start_stream(log):
            # ON by default: validated on this machine in the Stereotype classification runs
            # of 28/10 (segmentation 241 s for 6 videos while the descriptors
            # ran, Step 4 then finding 6/6 videos already computed, 9 min 52
            # total). Those runs used the blastema-refined driver, which pins
            # the fed resolution to the model's 320x320; the plain two-pass
            # driver feeds 384x384, and with it the card is saturated enough
            # that the overlap starves the segmentation (batches 11 s -> 91 s,
            # measured 29/10 on Stereotype classification from images). FLATMORPHAI_OVERLAP=0 disables it.
            if os.environ.get("FLATMORPHAI_OVERLAP", "1") not in ("1", "true", "True"):
                log("Overlap OFF: the steps run one after the other "
                    "(FLATMORPHAI_OVERLAP=1 to compute the descriptors during "
                    "the segmentation)")
                return None
            try:
                from ..notebook_api.lightgbm_bundle import (
                    FeatureStream, load_classifier, resolve_bundle_dir)
                bd = resolve_bundle_dir(_bundle_for_stream or None)
                clf, bd2 = load_classifier(bundle_dir=bd)
                fs = FeatureStream(log)
                fs.meta = dict(bundle_dir=bd2, base=list(clf.base_cols),
                               sig=list(clf.sig_cols))
                log("Overlap ON: descriptors of each video start as soon as it "
                    "is segmented")
                return fs
            except Exception as exc:  # noqa: BLE001
                log("Overlap not available (%s: %s); sequential steps"
                    % (type(exc).__name__, exc))
                return None

        def _on_folder_segmented(path, log, _root=root):
            fs = _stream["fs"]
            if fs is None:
                return
            try:
                from ..notebook_api.classification_images import (
                    generate_stereotypie_image_for_classification,
                    build_processed_dir_name)
                folder = os.path.basename(os.path.normpath(path))
                generate_stereotypie_image_for_classification(
                    path_dataset=_root, name_stereotypie_dir="Formated",
                    name_mask_dir=["masks"], number_of_planaria_on_image=1,
                    image_size=384, margin_percent=0.0,
                    number_rows_of_planaria_on_image=1,
                    folders_stereotypie_to_process=[folder],
                    apply_mask=True, black_and_white=False,
                    overwrite=bool(self._settings.get("step2_overwrite", False)),
                    log=None, max_workers=2)
                fmt = os.path.join(_root, folder,
                                   build_processed_dir_name("Formated", 384, 0.0, False, True))
                vd = os.path.join(_root, folder)
                masks = sorted((e for e in os.listdir(vd)
                                if e.lower().startswith("mask")
                                and os.path.isdir(os.path.join(vd, e))), reverse=True)
                if not masks or not os.path.isdir(fmt):
                    return
                fs.add(dict(video_dir=vd, mask_dir_name=masks[0],
                            output_csv_path=os.path.join(vd, "%s_features.csv" % folder),
                            bundle_dir=fs.meta["bundle_dir"], baseline_cols=fs.meta["base"],
                            specialist_cols=fs.meta["sig"], overwrite=False,
                            formatted_dir=fmt))
            except Exception as exc:  # noqa: BLE001
                log("  overlap: %s left to the descriptor step (%s: %s)"
                    % (path, type(exc).__name__, exc))

        def _seg_runner(params, log, progress, is_interrupted, _w=_work, _u=_user_root):
            if _w and not _LW.prepare(_w, _u, log):
                raise RuntimeError("could not prepare the Linux work folder %s" % _w)
            _stream["fs"] = _start_stream(log)

            def _log2(msg):
                log(msg)
                try:
                    txt = msg.split("] ", 1)[-1].strip()
                    if txt.startswith("Folder segmented: "):
                        _on_folder_segmented(txt[len("Folder segmented: "):].strip(), log)
                except Exception:  # noqa: BLE001
                    pass
            try:
                return run_step2(params, _log2, progress, is_interrupted)
            finally:
                if _stream["fs"] is not None:
                    _stream["fs"].close()

        def _descriptors_runner(params, log, progress, is_interrupted,
                                _w=_work, _u=_user_root):
            fs = _stream.get("fs")
            if fs is not None:
                if _w:
                    _LW.start_background_sync(_w, _u, log)
                if fs.th.is_alive():
                    log("Waiting for the descriptors computed during the "
                        "segmentation (%d video(s))" % fs.n)
                fs.join()
            elif _w:
                _LW.defer_background_sync(_w, _u, log)
            return run_step4(params, log, progress, is_interrupted)

        def _gather_runner(params, log, progress, is_interrupted,
                           _w=_work, _u=_user_root):
            try:
                return run_step6(params, log, progress, is_interrupted)
            finally:
                if _w and os.path.isdir(_w):
                    _LW.sync_back(_w, _u, log)

        steps = [
            PipelineStep(
                name="Step 1",
                runner=_seg_runner,
                params=Step2Params(
                    checkpoint_root=seg_dir,
                    model_name=seg_model,
                    processing_root=root,
                    batch_size=int(self._settings.get("step2_batch_size", 8)),
                    two_pass=bool(self._settings.get("step2_two_pass", True)),
                    two_pass_merge=str(
                        self._settings.get(
                            "step2_merge_strategy", "skeleton_guided"
                        )
                    ),
                    keep_largest_component=bool(
                        self._settings.get("step2_keep_largest", True)
                    ),
                    overwrite=bool(self._settings.get("step2_overwrite", False)),
                    blasteme_refined=self._bl_refine.isChecked(),
                ),
            ),
            PipelineStep(
                name="Step 2",
                runner=run_step3,
                # Build the masked / cropped / resized classification images
                # the LightGBM descriptor extractor needs. Parameters are
                # FIXED to what the classifier expects (see
                # notebook_api.classification.load_lightgbm_bundle:
                # input_shape [384, 384], apply_mask True, b&w False,
                # margin 0.0); process_videos resolves the folder by these
                # exact tokens, so they must match or classification finds
                # nothing.
                params=Step3Params(
                    processing_root=root,
                    image_size=384,
                    margin_percent=0.0,
                    black_and_white=False,
                    apply_mask=True,
                    number_of_planaria_on_image=1,
                    mask_dir="masks",
                    overwrite=bool(self._settings.get("step2_overwrite", False)),
                ),
            ),
            PipelineStep(
                name="Step 3",
                runner=_descriptors_runner,
                params=Step4Params(
                    processing_root=root,
                    bundle_dir=_bundle_dir_saved,
                    # Same knob as Stereotype classification (Settings page, default True):
                    # this page used to leave it out and take the dataclass
                    # default, so turning the smoothing off applied to A only.
                    correction_4s_enabled=bool(
                        self._settings.get(
                            "step4_4s_correction",
                            self._settings.get("step4_5s_correction", True),
                        )
                    ),
                    num_workers=int(
                        self._settings.get("step4_num_workers", 4)
                    ),
                    # Shared descriptor / booster thread count.
                    max_workers=int(
                        self._settings.get("step3_max_workers", 0)
                    ),
                ),
            ),
            PipelineStep(
                name="Step 4",
                runner=_gather_runner,
                # Step 6 (v1.2.1) walks the processing root, groups
                # per-video directories by the first five
                # underscore-separated tokens of their name and writes
                # one merged CSV. The period table is no longer
                # consulted because the period token is parsed from
                # each directory name.
                params=Step6Params(
                    processing_root=root,
                    output_directory=out_dir,
                    new_planaria_index=int(
                        self._settings.get("step6_new_planaria_index", 1)
                    ),
                ),
            ),
        ]

        worker = PipelineWorker(PipelineParams(steps=steps))
        worker.progress_signal.connect(self._progress.setValue)
        worker.progress_signal.connect(self._update_timeline_from_progress)
        worker.log_signal.connect(self._log_panel.append)
        worker.log_signal.connect(self._on_log_status)
        # Drive the "Current step:" label in PipelineTimeline (-1 at
        # end so it reverts to "idle"; see PipelineWorker.run).
        worker.current_step_changed_signal.connect(self._timeline.set_current)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append(f"ERROR: {msg}")
        )
        worker.result_signal.connect(self._on_done)
        self._status_label.setText("")
        self._launch_worker(worker)

    def _on_log_status(self, line: str) -> None:
        """Mirror the most recent log line in the lower-left status label."""
        text = line
        if text.startswith("[") and "] " in text:
            text = text.split("] ", 1)[1]
        if len(text) > 140:
            text = text[:137] + "..."
        self._status_label.setText(text)

    def _update_timeline_from_progress(self, pct: int) -> None:
        step_count = len(self.STEPS)
        if step_count == 0:
            return
        step_size = 100.0 / step_count
        current = (
            min(step_count - 1, int(pct // step_size))
            if pct < 100
            else step_count - 1
        )
        for i in range(step_count):
            if i < current:
                self._timeline.set_status(i, "done")
            elif i == current:
                self._timeline.set_status(i, "running")
            else:
                self._timeline.set_status(i, "idle")
        if pct >= 100:
            for i in range(step_count):
                self._timeline.set_status(i, "done")

    def _on_done(self, _result: object) -> None:
        for i in range(len(self.STEPS)):
            self._timeline.set_status(i, "done")
        self._log_panel.append("Stereotype classification from images finished successfully")
        # Remember the results CSV and the dataset root so the Stereotype
        # editor auto-loads them when the user switches to it.
        self._remember_stereotype_csv(
            self._s6_out.text().strip(), self._root_edit.text().strip()
        )
        # Switch straight to the Stereotype editor now the run succeeded.
        self.navigate_requested.emit("step5")
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

    def _remember_stereotype_csv(self, csv_dir: str, root_dir: str) -> None:
        """Persist the merged results CSV and dataset root for the editor.

        The Stereotype editor reads ``step5_csv_path`` / ``step5_processing_root``
        and auto-loads them on open and when the user switches to it.
        """
        if root_dir:
            self._settings.set("step5_processing_root", root_dir)
        csv_path = os.path.join(csv_dir, "all_videos.csv") if csv_dir else ""
        if csv_path and os.path.isfile(csv_path):
            self._settings.set("step5_csv_path", csv_path)

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
