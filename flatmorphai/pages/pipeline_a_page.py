"""
Stereotype classification page: run Steps 1 -> 2 -> 3 -> 4 -> 6 sequentially.

Step 5 (Stereotype Editor) is intentionally excluded from pipelines
because it is an interactive manual review step.

All tunable hyper-parameters (segmentation checkpoint, classification
checkpoint, prototype file, batch sizes, merge strategy, etc.) live in
the Settings page. Each pipeline step tile exposes only the inputs
that change per run (input / output directories, sampling window,
Step 6 period table), keeping a single source of truth for
hyper-parameters as recommended by Wilson, G. et al., "Good Enough
Practices in Scientific Computing", PLOS Computational Biology, 2017,
DOI: 10.1371/journal.pcbi.1005510, and by Sandve, G. K. et al., "Ten
Simple Rules for Reproducible Computational Research", PLOS
Computational Biology, 2013, DOI: 10.1371/journal.pcbi.1003285, rule
6 (avoid manual data manipulation steps).
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..constants import (
    CHECKPOINTS_SEGMENTATION_DIR,
    COLOR_ERROR,
    COLOR_SUCCESS,
)
from ..notebook_api.naming import VIDEO_EXTENSIONS, parse_video_name
from ..notebook_api.segmentation import (
    resolve_most_recent_checkpoint as _resolve_most_recent_seg_checkpoint,
)
from ..widgets import LogPanel, PipelineTimeline
from ..widgets.collapsible_card import CollapsibleCard
from ..widgets.resize_grip import VerticalResizeGrip
from ..workers.pipeline_runners import (
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    run_step6,
)
from ..workers.pipeline_worker import PipelineParams, PipelineStep, PipelineWorker
from ..workers.step1_worker import Step1Params
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

class PipelineAPage(BasePage):
    """End-to-end pipeline starting from raw videos."""

    # Displayed labels are numbered CONSECUTIVELY (1..5): frame
    # extraction, segmentation, classification-image formatting,
    # classification, period fusion. The formatting stage builds the
    # masked, cropped and resized classification images that the
    # LightGBM descriptor extractor reads (process_videos resolves a
    # ``_image-size(<N>)`` sub-folder and fails without it), so it is a
    # required step of the chain, not optional overhead.
    STEPS = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    # Per-step descriptions shown in the timeline's "Current step:"
    # label so the operator can tell what work is happening without
    # reading the log panel.  Kept in sync with the runner functions
    # in workers/pipeline_runners.py: run_step1, run_step2,
    # run_step4, run_step6.  Reference for the user-facing wording:
    # Nielsen, J., "10 Usability Heuristics for User Interface
    # Design", 1994, https://www.nngroup.com/articles/
    # ten-usability-heuristics/, heuristic 2 (match between system
    # and real world: use the operator's vocabulary, not internal
    # function names).
    STEP_DESCRIPTIONS = [
        "Frame extraction (videos -> 1 PNG / sampled second)",
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
        super().__init__("Stereotype classification", parent)
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

        title = QLabel("Stereotype classification - Videos to stereotypes")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        hint = QLabel(
            "Hyper-parameters (segmentation model, classification model, "
            "prototype, batch sizes, merge strategy, etc.) are configured "
            "in the Settings page. This pipeline tile exposes only the "
            "inputs that change per run."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._timeline = PipelineTimeline(
            self.STEPS, step_descriptions=self.STEP_DESCRIPTIONS,
        )
        root.addWidget(self._timeline)

        # Period fusion no longer has its own "Step 5" section: Stereotype classification
        # always writes all_videos.csv into the Step 1 output (the processing
        # root). ``_s6_out`` is kept as a hidden field (synced to that root) so
        # the run logic and the saved ``step6_output_directory`` setting keep
        # working without an extra directory picker to fill in.
        self._s6_out = QLineEdit()
        self._s6_out.setVisible(False)

        # The Step 1 options are laid out directly (no inner scroll area): the
        # whole Videos-directory-to-Detected-videos block is fixed in place,
        # matching the "Images -> Stereotypes" tab. The detected-videos list is
        # compact and drag-resizable, so this block never needs its own
        # scrollbar.
        root.addWidget(self._build_step1_group())

        # Log + actions
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel, 1)

        # The Bridge-like prototype toggle used to live here for the
        # retired Mahalanobis path.  With the LightGBM classifier
        # there are only 5 classes and no open-set surcharge, so
        # the checkbox has been removed.

        # Auto-open the output directory in the OS file explorer when
        # the pipeline finishes successfully.  Checked by default per
        # the operator request; QDesktopServices.openUrl dispatches to
        # Explorer on Windows, Finder on macOS, xdg-open on Linux.
        # Reference for the cross-platform pattern: Nielsen, J., "10
        # Usability Heuristics for User Interface Design", 1994,
        # https://www.nngroup.com/articles/ten-usability-heuristics/,
        # heuristic 7 ("flexibility and efficiency of use").
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
        # Lower-left status label: shows the most recent worker log line
        # so the user sees per-image progress without scrolling the log
        # panel. Placing live status next to the trigger action follows
        # Nielsen, J., "10 Usability Heuristics for User Interface
        # Design", 1994, published on https://www.nngroup.com/articles/
        # ten-usability-heuristics/, heuristic 1 (visibility of system
        # status).
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

    # --- Step groups -------------------------------------------------
    def _build_step1_group(self) -> QWidget:
        # Inside the "Videos -> Stereotype" collapsible card the "Step 1" title
        # is redundant, so this is a plain container rather than a titled group.
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Directory selectors use the same caption + field + Browse row layout
        # as the Regeneration states editor's source card.
        self._s1_input = QLineEdit()
        self._s1_input.setPlaceholderText("Videos_to_process")
        self._s1_input.textChanged.connect(self._on_s1_input_changed)
        self._s1_output = QLineEdit()
        self._s1_output.setPlaceholderText("Processed_videos_Frame_by_frame")
        # Per-page processing root (Stereotype classification output), saved locally the moment
        # the field changes, independent from the other pages.
        self._s1_output.textChanged.connect(
            lambda t: self._settings.set("pipeline_a_processing_root", t)
        )
        browse_in = QPushButton("Browse")
        browse_in.clicked.connect(lambda: self._browse_dir(self._s1_input))
        # Manual refresh is required because the per-video compliance
        # list only re-scans the input directory on textChanged events
        # of the path line edit. A user who drops new .mp4 files into
        # an already-selected directory would otherwise see a stale
        # list. The handler reuses the existing ``_refresh_s1_videos_list``
        # method so the audit logic stays in one place.
        refresh_in = QPushButton("Refresh")
        refresh_in.setToolTip(
            "Re-scan the input videos directory and rebuild the "
            "per-video compliance list. Use this after adding or "
            "renaming files in the selected folder."
        )
        refresh_in.clicked.connect(
            lambda: self._refresh_s1_videos_list(
                self._s1_input.text().strip()
            )
        )
        # Both directory selectors live in one bordered white card, matching
        # the "Images -> Stereotypes" tab's processing-root card. The captions
        # share a fixed width so the two line edits line up.
        dir_card = QFrame()
        dir_card.setObjectName("card")
        dir_cv = QVBoxLayout(dir_card)
        dir_cv.setContentsMargins(14, 12, 14, 12)
        dir_cv.setSpacing(8)

        row_in = QHBoxLayout()
        row_in.setSpacing(10)
        in_cap = QLabel("Videos directory")
        in_cap.setObjectName("caption")
        in_cap.setMinimumWidth(150)
        row_in.addWidget(in_cap, 0, Qt.AlignVCenter)
        row_in.addWidget(self._s1_input, 3)
        row_in.addWidget(browse_in)
        row_in.addWidget(refresh_in)
        dir_cv.addLayout(row_in)

        browse_out = QPushButton("Browse")
        browse_out.clicked.connect(lambda: self._browse_dir(self._s1_output))
        row_out = QHBoxLayout()
        row_out.setSpacing(10)
        out_cap = QLabel("Output (processing root)")
        out_cap.setObjectName("caption")
        out_cap.setMinimumWidth(150)
        row_out.addWidget(out_cap, 0, Qt.AlignVCenter)
        row_out.addWidget(self._s1_output, 3)
        row_out.addWidget(browse_out)
        dir_cv.addLayout(row_out)

        outer.addWidget(dir_card)

        # --- Video sampling window (collapsible card) -----------------------
        # The timing parameters fold into a collapsible card; the adaptive
        # frame-count message below the card stays visible at all times, even
        # when the card is collapsed, so the user always sees how many frames
        # the current settings produce.
        window_body = QWidget()
        wc = QVBoxLayout(window_body)
        wc.setContentsMargins(14, 10, 14, 12)
        wc.setSpacing(8)

        window_form = QFormLayout()
        window_form.setContentsMargins(0, 0, 0, 0)
        window_form.setSpacing(8)
        self._s1_start = QDoubleSpinBox()
        self._s1_start.setRange(0.0, 100000.0)
        self._s1_start.setDecimals(3)
        self._s1_start.setSingleStep(1.0)
        self._s1_start.setValue(0.0)
        self._s1_start.setSuffix(" s")
        self._s1_start.valueChanged.connect(
            lambda v: self._settings.set("step1_start_second", float(v))
        )
        self._s1_start.valueChanged.connect(lambda *_: self._update_frame_estimate())
        window_form.addRow("Start second:", self._s1_start)

        self._s1_end_auto = QCheckBox("Use maximum available second per video")
        self._s1_end_auto.setChecked(True)
        self._s1_end_auto.toggled.connect(self._on_s1_end_auto_toggled)
        window_form.addRow("End mode:", self._s1_end_auto)

        self._s1_end = QDoubleSpinBox()
        self._s1_end.setRange(0.0, 100000.0)
        self._s1_end.setDecimals(3)
        self._s1_end.setSingleStep(1.0)
        self._s1_end.setValue(60.0)
        self._s1_end.setSuffix(" s")
        self._s1_end.setEnabled(False)
        self._s1_end.valueChanged.connect(
            lambda v: self._settings.set("step1_end_second_manual", float(v))
        )
        self._s1_end.valueChanged.connect(lambda *_: self._update_frame_estimate())
        window_form.addRow("End second (manual):", self._s1_end)

        self._s1_step = QDoubleSpinBox()
        # The Step-1 sampling step MUST be in the half-open interval
        # (0, 1] seconds.  Two reasons: a step of 0 would re-sample the
        # SAME frame infinitely, and a step above 1 second breaks the
        # ConvNeXt-V2 3-mask window (the model was trained on a one-
        # second context T-1s/T0/T+1s; with step > 1s the closest-
        # neighbour fallback in metrics_csv would point at frames more
        # than one physical second away, which is outside the training
        # distribution).  Reference for the design-by-constraint rule:
        # Nielsen, J., "10 Usability Heuristics for User Interface
        # Design", 1994, https://www.nngroup.com/articles/
        # ten-usability-heuristics/, heuristic 5 (error prevention).
        self._s1_step.setRange(0.001, 1.0)
        self._s1_step.setDecimals(3)
        self._s1_step.setSingleStep(0.1)
        self._s1_step.setValue(1.0)
        self._s1_step.setSuffix(" s")
        self._s1_step.setToolTip(
            "Time between sampled frames.  Must be in (0, 1] s.  A "
            "step > 1 s would put the one-second metric window "
            "(T-1 s, T0, T+1 s) outside its training context."
        )
        self._s1_step.valueChanged.connect(
            lambda v: self._settings.set("step1_step_seconds", float(v))
        )
        self._s1_step.valueChanged.connect(lambda *_: self._update_frame_estimate())
        window_form.addRow("Step between frames:", self._s1_step)
        wc.addLayout(window_form)

        self._window_card = self._persisted_card(
            "Video sampling window", window_body,
            "stereo_videos_sec_window_expanded", default=True,
        )
        outer.addWidget(self._window_card)

        # Adaptive estimate of how many frames the current settings extract.
        # Kept OUTSIDE the collapsible card so it stays visible even when the
        # sampling window is folded.
        self._s1_frame_estimate = QLabel("")
        self._s1_frame_estimate.setObjectName("mutedLabel")
        self._s1_frame_estimate.setTextFormat(Qt.RichText)
        self._s1_frame_estimate.setWordWrap(True)
        self._s1_frame_estimate.setContentsMargins(4, 0, 4, 0)
        outer.addWidget(self._s1_frame_estimate)

        # Naming-convention notice, placed right next to the detected-videos
        # list so the expected format sits beside the files it flags.
        convention_label = QLabel(
            "<b>Expected video name:</b> "
            "<code><b style='color:#0D9488;'>species_compound_concentration_"
            "plate_position_period[_comments].&lt;ext&gt;</b></code><br>"
            "Accepted extensions: <code>.mp4</code>, <code>.avi</code>, "
            "<code>.mov</code>, <code>.mkv</code>. The <code>comments</code> "
            "token is optional (letters, digits and the dash; no "
            "underscores). Examples: "
            "<code>Smed_nicotine_10um_P1_B4_T0h.mp4</code>, "
            "<code>Smed_Acetylcholine_3000uM_P1_R3(2)_T3h.avi</code>, "
            "<code>Smed_DMSO_0.1pct_P2_C5_T24h_20251225.mov</code>."
        )
        convention_label.setObjectName("mutedLabel")
        convention_label.setTextFormat(Qt.RichText)
        convention_label.setWordWrap(True)
        convention_label.setContentsMargins(4, 4, 4, 0)
        outer.addWidget(convention_label)

        # --- Detected videos (collapsible card) -----------------------------
        # The list (compact, drag-to-enlarge) folds into a collapsible card;
        # the compliance summary below the card stays visible at all times.
        detected_body = QWidget()
        db = QVBoxLayout(detected_body)
        db.setContentsMargins(14, 10, 14, 12)
        db.setSpacing(6)

        self._s1_videos_list = QListWidget()
        self._s1_videos_list.setAlternatingRowColors(True)
        # Compact by default so the sampling parameters above keep the room;
        # the grip below lets the user drag the list taller to audit many files.
        # The height the operator dragged it to is remembered.
        try:
            _list_h = int(self._settings.get("stereo_videos_list_height", 0) or 0)
        except (TypeError, ValueError):
            _list_h = 0
        _list_h = _list_h if _list_h >= 90 else 120
        self._s1_videos_list.setMinimumHeight(_list_h)
        self._s1_videos_list.setMaximumHeight(_list_h)

        list_grip = VerticalResizeGrip(
            self._s1_videos_list,
            min_height=90,
            tooltip="Drag down to enlarge the video list, up to shrink it",
            on_resized=lambda h: self._settings.set(
                "stereo_videos_list_height", int(h)),
        )
        db.addWidget(self._s1_videos_list)
        db.addWidget(list_grip)

        self._detected_card = self._persisted_card(
            "Detected videos", detected_body,
            "stereo_videos_sec_detected_expanded", default=True,
        )
        outer.addWidget(self._detected_card)

        # Compliance summary kept OUTSIDE the collapsible card so it stays
        # visible even when the detected-videos list is folded.
        self._s1_convention_summary = QLabel("")
        self._s1_convention_summary.setObjectName("mutedLabel")
        self._s1_convention_summary.setWordWrap(True)
        self._s1_convention_summary.setContentsMargins(4, 0, 4, 0)
        outer.addWidget(self._s1_convention_summary)

        outer.addWidget(self._build_mask_refinement_card())

        self._update_frame_estimate()
        return box

    def _build_mask_refinement_card(self) -> QWidget:
        """Collapsible 'Mask refinement' card for the Videos segmentation."""
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

        return self._persisted_card(
            "Mask refinement", body,
            "stereo_videos_sec_refine_expanded", default=False,
        )

    def _persisted_card(self, title, body, key: str, default: bool):
        """A CollapsibleCard whose folded/unfolded state is saved under
        ``key`` and restored at the next launch, like every other card."""
        return CollapsibleCard(
            title, body,
            expanded=bool(self._settings.get(key, default)),
            on_toggle=lambda e, k=key: self._settings.set(k, bool(e)),
        )

    def _update_frame_estimate(self) -> None:
        """Refresh the adaptive 'frames with the current settings' message."""
        import math

        label = getattr(self, "_s1_frame_estimate", None)
        if label is None:
            return
        try:
            start = float(self._s1_start.value())
            step = float(self._s1_step.value())
        except Exception:  # noqa: BLE001
            return
        if step <= 0.0:
            label.setText("")
            return
        rate = 1.0 / step
        if self._s1_end_auto.isChecked():
            label.setText(
                "With the current settings: "
                f"<b>~{rate:g} frame(s) per second</b> of video "
                f"(1 frame every {step:g} s), starting at {start:g} s. "
                "The exact total per video depends on each video's length."
            )
            return
        end = float(self._s1_end.value())
        if end < start:
            label.setText(
                "End second is before start second — no frames would be "
                "sampled. Increase the end second."
            )
            return
        n = int(math.floor((end - start) / step + 1e-9)) + 1
        label.setText(
            "With the current settings: "
            f"<b>{n} frame(s)</b> per video "
            f"(1 frame every {step:g} s, from {start:g} s to {end:g} s)."
        )

    def _on_s1_end_auto_toggled(self, checked: bool) -> None:
        """Enable the manual end-second spinner only when auto is off."""
        self._s1_end.setEnabled(not checked)
        self._settings.set("step1_end_second_auto", bool(checked))
        self._update_frame_estimate()

    def _on_s1_input_changed(self, text: str) -> None:
        self._settings.set("input_video_directory", text)
        self._refresh_s1_videos_list(text)

    def _refresh_s1_videos_list(self, text: str) -> None:
        self._s1_videos_list.clear()
        if not text or not os.path.isdir(text):
            self._s1_convention_summary.setText("")
            return
        try:
            entries = sorted(os.listdir(text))
        except OSError:
            self._s1_convention_summary.setText("")
            return
        total = 0
        bad = 0
        bad_color = QBrush(QColor(COLOR_ERROR))
        ok_color = QBrush(QColor(COLOR_SUCCESS))
        for name in entries:
            full = os.path.join(text, name)
            if not os.path.isfile(full):
                continue
            if not name.lower().endswith(VIDEO_EXTENSIONS):
                continue
            total += 1
            meta = parse_video_name(name)
            if meta is None:
                bad += 1
                item = QListWidgetItem(f"{name}  [INVALID NAME]")
                item.setForeground(bad_color)
            else:
                item = QListWidgetItem(f"{name}  [OK]")
                item.setForeground(ok_color)
            self._s1_videos_list.addItem(item)
        if total == 0:
            self._s1_convention_summary.setText(
                "No video files detected in this directory."
            )
            return
        if bad == 0:
            self._s1_convention_summary.setText(
                f"<span style='color: {COLOR_SUCCESS};'>"
                f"All {total} video file(s) respect the naming convention."
                f"</span>"
            )
        else:
            self._s1_convention_summary.setText(
                f"<span style='color: {COLOR_ERROR};'>"
                f"{bad} of {total} video file(s) do not respect the "
                f"naming convention. Rename them before running the pipeline."
                f"</span>"
            )
        self._s1_convention_summary.setTextFormat(Qt.RichText)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_from_settings(self) -> None:
        self._bl_refine.setChecked(
            # Shared with Stereotype classification from images (same chain, one switch); the old
            # per-page key is still honoured as the starting point.
            bool(self._settings.get(
                "stereotypie_blasteme_refined",
                self._settings.get("pipeline_a_blasteme_refined", False),
            ))
        )
        self._s1_input.setText(
            self._settings.get("input_video_directory", "Videos_to_process")
        )
        self._s1_output.setText(
            str(self._settings.get("pipeline_a_processing_root", "")).strip()
            or str(self._settings.get("processing_root", "")).strip()
            or "Processed_videos_Frame_by_frame"
        )
        self._s1_start.setValue(
            float(self._settings.get("step1_start_second", 0.0))
        )
        s1_end_auto = bool(self._settings.get("step1_end_second_auto", True))
        self._s1_end_auto.setChecked(s1_end_auto)
        self._s1_end.setValue(
            float(self._settings.get("step1_end_second_manual", 60.0))
        )
        self._s1_end.setEnabled(not s1_end_auto)
        self._s1_step.setValue(
            float(self._settings.get("step1_step_seconds", 1.0))
        )
        self._refresh_s1_videos_list(self._s1_input.text())
        self._update_frame_estimate()

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
        root = self._s1_output.text().strip()
        if not root or not os.path.isdir(root):
            return []
        return sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )

    def _refresh_videos(self) -> None:
        # Kept as a no-op for backwards compatibility with any caller
        # that referenced the previous Step 6 table refresh button.
        # Step 6 in v1.2.1 enumerates the directories itself when the
        # pipeline runs, so the page does not need to maintain a list.
        return

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        input_dir = self._s1_input.text().strip()
        output_dir = (
            self._s1_output.text().strip()
            or "Processed_videos_Frame_by_frame"
        )
        if not input_dir or not os.path.isdir(input_dir):
            _msg_critical(
                self, "Stereotype classification", "Select a valid input video directory."
            )
            return

        # Defensive guard on the Step-1 sampling step.  The spinbox's
        # setRange((0.001, 1.0]) normally enforces this at the UI
        # level, but a settings-file edit could still load an out-of-
        # range value into the page; surface a clear modal in that
        # case so the user does not start a long pipeline that the
        # downstream ConvNeXt-V2 metrics window cannot handle.
        # Reference: Nielsen, J., "10 Usability Heuristics for User
        # Interface Design", 1994, https://www.nngroup.com/articles/
        # ten-usability-heuristics/, heuristic 5 (error prevention).
        step_seconds_now = float(self._s1_step.value())
        if not (0.0 < step_seconds_now <= 1.0):
            _msg_critical(
                self, "Stereotype classification",
                f"'Step between frames' is {step_seconds_now} s but "
                "must be in the half-open interval (0, 1] s.  Set it "
                "above 0 and at most 1 second, then run again.\n\n"
                "Why: the ConvNeXt-V2 classifier was trained on a "
                "one-second metric window (T-1s, T0, T+1s).  A step "
                "larger than 1 s would put the prev / next masks "
                "outside that window and degrade prediction quality.",
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
                self, "Stereotype classification",
                f"No segmentation checkpoint found under {seg_dir}. "
                "Please configure the checkpoints folder in the Settings "
                "page.",
            )
            return

        # Classifier resolution.  There is exactly ONE classifier --
        # the LightGBM bundle -- so this is a single existence check
        # rather than the former multi-root checkpoint scan.
        from ..notebook_api.lightgbm_bundle import (
            bundle_available as _bundle_available,
        )
        _bundle_dir_saved = str(
            self._settings.get("step4_bundle_dir", "")
        )
        _use_bundle = _bundle_available(_bundle_dir_saved or None)
        if not _use_bundle:
            _msg_critical(
                self, "Stereotype classification",
                "LightGBM classifier not found"
                + (
                    f" at: {_bundle_dir_saved}"
                    if _bundle_dir_saved
                    else " in the default 'lightgbm_stereotype_classifier' "
                         "folder"
                )
                + ".\n\nSet 'LightGBM classifier folder' in the Step 4 "
                "tab of the Settings page to a folder containing "
                "classifier.py, config/feature_columns.json and "
                "models/*.txt.",
            )
            return

        self._log_panel.append(
            "[Stereotype classification] Selected classifier:\n"
            f"   folder: {_bundle_dir_saved or '(default sibling)'}\n"
            "   model : LightGBM stereotype classifier\n"
            "   format: 5 classes (C-shape, Screw-like, Snake-like, "
            "Walnut, Wildtype); two boosters + Walnut vs C-shape "
            "arbitration"
        )

        # No prototype file to check anymore -- Step 4 runs LightGBM.
        try:
            import torch
            if not torch.cuda.is_available():
                _msg_critical(
                    self, "Stereotype classification",
                    "No CUDA device detected.  Step 2 requires a GPU "
                    "(LightGBM classification in Step 4 is CPU-only).",
                )
                return
        except ImportError:
            _msg_critical(self, "Stereotype classification", "PyTorch is not installed.")
            return

        # Period fusion always writes all_videos.csv into the Step 1 output
        # (the processing root) - no separate output directory to pick. Rule 1
        # of Sandve G. K. et al., "Ten Simple Rules for Reproducible
        # Computational Research", PLOS Comput. Biol. 2013,
        # DOI: 10.1371/journal.pcbi.1003285.
        out_dir = self._s1_output.text().strip() or self._settings.get(
            "processing_root", ""
        )
        if not out_dir:
            _msg_critical(
                self, "Stereotype classification",
                "Set the Step 1 output (processing root) first.",
            )
            return
        self._s6_out.setText(out_dir)
        self._settings.set("step6_output_directory", out_dir)
        os.makedirs(out_dir, exist_ok=True)

        self._progress.setValue(0)
        self._log_panel.clear()
        self._timeline.reset()

        # Under WSL, when the processing root is on a Windows drive (/mnt/c),
        # run every step on the Linux disk and copy the results back at the
        # end (PNG writes ~4.6x and video reads ~2.3x faster, measured; same
        # files). FLATMORPHAI_LINUX_WORKDIR=0 keeps working in place.
        from ..workers import linux_workdir as _LW
        _user_root = output_dir
        _work = _LW.work_path(output_dir)
        _same_out = os.path.abspath(out_dir) == os.path.abspath(output_dir)
        if _work:
            output_dir = _work
            if _same_out:
                out_dir_run = _work
            else:
                out_dir_run = out_dir
        else:
            out_dir_run = out_dir

        def _step1_runner(params, log, progress, is_interrupted, _w=_work, _u=_user_root):
            if _w and not _LW.prepare(_w, _u, log):
                raise RuntimeError("could not prepare the Linux work folder %s" % _w)
            return run_step1(params, log, progress, is_interrupted)

        # OVERLAP (default): each video's descriptors start as soon as its
        # masks are written (segmentation announces "Folder segmented: ..."),
        # in a background pool, while the GPU segments the next video. Step 3
        # is run for that one video first (formatted images). Step 4 then finds
        # every feature CSV complete and only predicts. Same computer, same
        # values (bit-identical whatever the split, checked). FLATMORPHAI_OVERLAP=0
        # restores the strictly sequential steps.
        _stream = {"fs": None}
        _bundle_for_stream = str(self._settings.get("step4_bundle_dir", ""))

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
                fs.meta = dict(bundle_dir=bd2, base=list(clf.base_cols), sig=list(clf.sig_cols))
                log("Overlap ON: descriptors of each video start as soon as it is segmented")
                return fs
            except Exception as exc:  # noqa: BLE001
                log("Overlap not available (%s: %s); sequential steps" % (type(exc).__name__, exc))
                return None

        def _on_folder_segmented(path, log, _root=output_dir):
            fs = _stream["fs"]
            if fs is None:
                return
            try:
                from ..notebook_api.classification_images import (
                    generate_stereotypie_image_for_classification, build_processed_dir_name)
                folder = os.path.basename(os.path.normpath(path))
                generate_stereotypie_image_for_classification(
                    path_dataset=_root, name_stereotypie_dir="Formated", name_mask_dir=["masks"],
                    number_of_planaria_on_image=1, image_size=384, margin_percent=0.0,
                    number_rows_of_planaria_on_image=1, folders_stereotypie_to_process=[folder],
                    apply_mask=True, black_and_white=False,
                    overwrite=bool(self._settings.get("step2_overwrite", False)),
                    log=None, max_workers=2)
                fmt = os.path.join(_root, folder, build_processed_dir_name("Formated", 384, 0.0, False, True))
                vd = os.path.join(_root, folder)
                masks = sorted((e for e in os.listdir(vd) if e.lower().startswith("mask")
                                and os.path.isdir(os.path.join(vd, e))), reverse=True)
                if not masks or not os.path.isdir(fmt):
                    return
                fs.add(dict(video_dir=vd, mask_dir_name=masks[0],
                            output_csv_path=os.path.join(vd, "%s_features.csv" % folder),
                            bundle_dir=fs.meta["bundle_dir"], baseline_cols=fs.meta["base"],
                            specialist_cols=fs.meta["sig"], overwrite=False, formatted_dir=fmt))
            except Exception as exc:  # noqa: BLE001
                log("  overlap: %s left to Step 4 (%s: %s)" % (path, type(exc).__name__, exc))

        def _step2_runner(params, log, progress, is_interrupted):
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

        def _step4_runner(params, log, progress, is_interrupted, _w=_work, _u=_user_root):
            fs = _stream.get("fs")
            if fs is not None:
                if _w:      # pool already up: copy frames / masks / formatted images now
                    _LW.start_background_sync(_w, _u, log)
                if fs.th.is_alive():
                    log("Waiting for the descriptors computed during the segmentation (%d video(s))" % fs.n)
                fs.join()
            elif _w:      # frames / masks / formatted images are final: copy them once
                _LW.defer_background_sync(_w, _u, log)   # the Step-4 workers are up
            return run_step4(params, log, progress, is_interrupted)

        def _step5_runner(params, log, progress, is_interrupted, _w=_work, _u=_user_root):
            try:
                return run_step6(params, log, progress, is_interrupted)
            finally:
                if _w and os.path.isdir(_w):
                    _LW.sync_back(_w, _u, log)

        steps = [
            PipelineStep(
                name="Step 1",
                runner=_step1_runner,
                params=Step1Params(
                    input_directory=input_dir,
                    output_directory=output_dir,
                    start_second=float(self._s1_start.value()),
                    end_second=(
                        None
                        if self._s1_end_auto.isChecked()
                        else float(self._s1_end.value())
                    ),
                    step_seconds=float(self._s1_step.value()),
                    max_workers=int(
                        self._settings.get("step1_max_workers", 0) or 0
                    ),
                ),
            ),
            PipelineStep(
                name="Step 2",
                runner=_step2_runner,
                params=Step2Params(
                    checkpoint_root=seg_dir,
                    model_name=seg_model,
                    processing_root=output_dir,
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
                name="Step 3",
                runner=run_step3,
                # Build the masked / cropped / resized classification images
                # the LightGBM descriptor extractor needs. The size and crop
                # parameters are FIXED to what the classifier expects (see
                # notebook_api.classification.load_lightgbm_bundle:
                # input_shape [384, 384], apply_mask True, b&w False,
                # margin 0.0); process_videos resolves the folder by these
                # exact tokens, so they must match or classification finds
                # nothing. Not driven by the Settings-page Step 3 size (224),
                # which is for the standalone Step 3 page.
                params=Step3Params(
                    processing_root=output_dir,
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
                name="Step 4",
                runner=_step4_runner,
                params=Step4Params(
                    processing_root=output_dir,
                    # LightGBM bundle is the default classifier now;
                    # when this path is empty the runner falls back
                    # to the sibling ``lightgbm_stereotype_classifier``
                    # folder.
                    bundle_dir=str(
                        self._settings.get("step4_bundle_dir", "")
                    ),
                    # 4-second isolated-class smoothing (Settings page
                    # exposes this knob, default True).
                    correction_4s_enabled=bool(
                        self._settings.get(
                            "step4_4s_correction",
                            # Migration: respect legacy 5s key when
                            # the new key has not been written yet.
                            self._settings.get(
                                "step4_5s_correction", True
                            ),
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
                name="Step 5",
                runner=_step5_runner,
                # Step 6 (v1.2.1) walks the processing root, groups
                # per-video directories by the first five
                # underscore-separated tokens of their name and writes
                # one merged CSV. The period table is no longer
                # consulted because the period token is parsed from
                # each directory name.
                params=Step6Params(
                    processing_root=output_dir,
                    output_directory=out_dir_run,
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
        # Drive the "Current step:" label in PipelineTimeline.  The
        # worker emits (-1, "<reason>") at completion / cancel / error
        # so the label reverts to "idle" without the page having to
        # parse log lines.
        worker.current_step_changed_signal.connect(self._timeline.set_current)
        worker.error_signal.connect(
            lambda msg: self._log_panel.append(f"ERROR: {msg}")
        )
        worker.result_signal.connect(self._on_done)
        self._status_label.setText("")
        self._launch_worker(worker)

    def _on_log_status(self, line: str) -> None:
        """Mirror the most recent log message in the status label.

        Log lines are timestamped by :class:`BaseWorker` as
        ``[HH:MM:SS] message``. The label drops the timestamp and
        truncates the remainder to a reasonable length so the row
        layout stays stable.
        """
        text = line
        if text.startswith("[") and "] " in text:
            text = text.split("] ", 1)[1]
        if len(text) > 140:
            text = text[:137] + "..."
        self._status_label.setText(text)

    def _update_timeline_from_progress(self, pct: int) -> None:
        """Set timeline status based on progress buckets."""
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
        self._log_panel.append("Stereotype classification finished successfully")
        # Remember the results CSV and the dataset root so the Stereotype
        # editor auto-loads them when the user switches to it.
        self._remember_stereotype_csv(
            self._s6_out.text().strip(), self._s1_output.text().strip()
        )
        # Switch straight to the Stereotype editor now the run succeeded.
        self.navigate_requested.emit("step5")
        if (
            hasattr(self, "_open_dir_on_done")
            and self._open_dir_on_done.isChecked()
        ):
            target = self._s1_output.text().strip() or "."
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
        """Open the pipeline's Output dir in the OS file explorer.

        The folder is created (``os.makedirs`` with ``exist_ok=True``)
        when it does not yet exist so the button is usable BEFORE
        the first run.  Cross-platform via
        :func:`widgets.image_pair_viewer.open_folder` (WSL-aware).
        """
        target = self._s1_output.text().strip()
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
