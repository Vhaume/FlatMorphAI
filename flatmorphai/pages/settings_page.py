"""
Settings page: central location for user preferences and default
parameters.

The page is organised as a :class:`QTabWidget` so each category fits on
a single screen at the default UI scale:

- **Appearance**: theme and UI scale factor.
- **Paths**: default input, processing and output directories.
- **Defaults: Step 1**: start second, end mode (auto or manual),
  manual end second, step between frames.
- **Defaults: Step 2**: Mask2Former batch size, two-pass, merge
  strategy, post-processing, overwrite policy.
- **Defaults: Step 3**: cropping size, margin percent, B&W,
  apply-mask, number of planaria per image, image-generation
  toggle and the descriptor worker-thread count.
- **Defaults: Step 4**: LightGBM classifier folder, forecasting
  batch size, worker count, 4-second correction.
- **Defaults: Step 5**: starting planaria index for the period
  fusion. Species, compound, concentration, plate, arena and
  period are parsed per video from the directory names.
  (The settings keys are still named ``step6_*`` for backwards
  compatibility with previously saved configs.)
- **Pipeline**: stop-on-error toggle.

Every widget reads and writes directly from :class:`AppSettings`,
which debounces disk writes. The Reset button restores every
preference to its factory default through
:py:meth:`AppSettings.reset_to_defaults`.

The layout pattern is inspired by Nielsen, J., Loranger, H. (2006).
*Prioritizing Web Usability*. New Riders (no DOI). The idea is to
let power users tune default parameters once and have every step
inherit them.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..widgets.file_dialogs import get_existing_directory, get_open_file_name
from ..constants import (
    AVAILABLE_THEMES,
    THEME_DARK,
    THEME_LIGHT,
    UI_SCALE_MAX,
    UI_SCALE_MIN,
    UI_SCALE_STEP,
)
from ..theme_manager import ThemeManager


from ..widgets.file_dialogs import (  # centered message boxes
    critical as _msg_critical,
    information as _msg_information,
    question as _msg_question,
    warning as _msg_warning,
)

class _PathRow(QWidget):
    """Line edit plus Browse button, for directory or file selection."""

    def __init__(
        self,
        placeholder: str = "",
        select_files: bool = False,
        file_filter: str = "All files (*)",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._browse = QPushButton("Browse...")
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._browse)
        self._select_files = select_files
        self._file_filter = file_filter
        self._browse.clicked.connect(self._on_browse)

    def _on_browse(self) -> None:
        if self._select_files:
            path, _ = get_open_file_name(
                self, "Select file", self._edit.text(), self._file_filter,
            )
        else:
            path = get_existing_directory(
                self, "Select directory", self._edit.text(),
            )
        if path:
            self._edit.setText(path)

    def text(self) -> str:
        return self._edit.text()

    def setText(self, text: str) -> None:  # noqa: N802
        self._edit.setText(text)

    @property
    def editingFinished(self):  # noqa: N802
        return self._edit.editingFinished

    @property
    def textChanged(self):  # noqa: N802
        return self._edit.textChanged


class SettingsPage(QWidget):
    """Dashboard page exposing theme, scale and per-step defaults."""

    page_status_changed = Signal(str)
    run_state_changed = Signal(str, bool)

    STEP2_MERGE_STRATEGIES = [
        "skeleton_guided",
        "adaptive_boundary",
        "spatially_adaptive",
        "constrained_union",
        "replace",
    ]

    def __init__(
        self,
        theme_manager: Optional[ThemeManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._name = "Settings"
        self._settings = AppSettings()
        self._theme_manager = theme_manager
        self._build_ui()
        self._load_from_settings()

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        subtitle = QLabel(
            "Preferences are saved to ~/.flatmorphai/config.json and "
            "reloaded on the next launch."
        )
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidgetResizable(True)

        # Each former tab is now a collapsible card titled with its real
        # section title (the inner group box's title) instead of a "Step N"
        # label, e.g. "Worm crop defaults (Stereotype editor)". The
        # expanded/collapsed state of each is persisted.
        from ..widgets import CollapsibleCard
        sections = [
            ("settings_sec_appearance", self._build_appearance_tab),
            ("settings_sec_paths", self._build_paths_tab),
            # "Video sampling window" lives on the Stereotype classification
            # page (Videos -> Stereotype), so it is not duplicated here.
            ("settings_sec_step2", self._build_step2_tab),
            ("settings_sec_step3", self._build_step3_tab),
            ("settings_sec_step4", self._build_step4_tab),
            # ``step6_*`` setting keys are kept so old configs keep loading.
            ("settings_sec_step6", self._build_step6_tab),
            ("settings_sec_eyespot", self._build_eyespot_tab),
            ("settings_sec_pipeline", self._build_pipeline_tab),
        ]
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(12)
        for i, (key, builder) in enumerate(sections):
            w = builder()
            gb = w.findChild(QGroupBox)
            card_title = gb.title() if gb is not None else key
            if gb is not None:
                # The card header carries the title now; neutralise the inner
                # group box so it is just a transparent container.
                gb.setTitle("")
                gb.setObjectName("collapsibleBody")
                gb.setFlat(True)
            w.setObjectName("collapsibleBody")
            vbox.addWidget(CollapsibleCard(
                card_title, w,
                expanded=bool(self._settings.get(key + "_expanded", i == 0)),
                on_toggle=lambda e, k=key: self._settings.set(
                    k + "_expanded", bool(e)
                ),
            ))
        vbox.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        actions = QHBoxLayout()
        self._reset_btn = QPushButton("Reset all to defaults")
        self._reset_btn.clicked.connect(self._on_reset)
        self._open_config_btn = QPushButton("Open config folder")
        self._open_config_btn.clicked.connect(self._on_open_config)
        actions.addWidget(self._reset_btn)
        actions.addWidget(self._open_config_btn)
        actions.addStretch(1)
        root.addLayout(actions)

    # -- tabs ----------------------------------------------------------
    def _build_appearance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        appearance_box = QGroupBox("Theme and scale")
        form = QFormLayout(appearance_box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._theme_combo = QComboBox()
        for theme in AVAILABLE_THEMES:
            label = {THEME_DARK: "Dark", THEME_LIGHT: "Light"}.get(theme, theme)
            self._theme_combo.addItem(label, userData=theme)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        form.addRow("Theme:", self._theme_combo)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(UI_SCALE_MIN, UI_SCALE_MAX)
        self._scale_spin.setSingleStep(UI_SCALE_STEP)
        self._scale_spin.setDecimals(2)
        self._scale_spin.setSuffix(" x")
        self._scale_spin.valueChanged.connect(self._on_scale_changed)
        form.addRow("UI scale:", self._scale_spin)

        self._cursor_mode = QComboBox()
        self._cursor_mode.addItem("System (Windows pointer)", userData="system")
        self._cursor_mode.addItem("Drawn by the application", userData="drawn")
        self._cursor_mode.setToolTip(
            "System: the pointer is the one Windows and Qt provide, untouched "
            "-- the same as everywhere else on the desktop.\n"
            "Drawn by the application: the pointer is drawn here, at the size "
            "and colour below. It is then the same everywhere INSIDE the "
            "window, but still differs from the Windows pointer on the window "
            "frame and outside the application.\n"
            "Note: inside the window, the system pointer comes from the X "
            "cursor theme; this distribution has none installed, which is why "
            "it is small there. Installing one (for example "
            "'sudo apt install adwaita-icon-theme') gives a proper pointer at "
            "the size of XCURSOR_SIZE, with the system's own shapes."
        )
        self._cursor_mode.currentIndexChanged.connect(self._on_cursor_mode)
        form.addRow("Pointer:", self._cursor_mode)

        self._cursor_spin = QSpinBox()
        self._cursor_spin.setRange(16, 128)
        self._cursor_spin.setSingleStep(4)
        self._cursor_spin.setSuffix(" px")
        self._cursor_spin.setToolTip(
            "Size of the mouse pointer inside the application.\n"
            "The shapes and their animations are unchanged -- the arrow, the "
            "open and closed hands on the viewers, the text beam -- they are "
            "simply drawn larger by the cursor theme.\n"
            "Applies immediately."
        )
        self._cursor_spin.valueChanged.connect(self._on_cursor_size)
        form.addRow("Pointer size:", self._cursor_spin)

        self._cursor_color_btn = QPushButton()
        self._cursor_color_btn.setFixedWidth(120)
        self._cursor_color_btn.setToolTip(
            "Body colour of the pointer. The outline is chosen against it -- "
            "black on a light colour, white on a dark one -- so the pointer "
            "stays visible on a black image as on a white panel.\n"
            "Applies immediately."
        )
        self._cursor_color_btn.clicked.connect(self._on_cursor_color)
        form.addRow("Pointer colour:", self._cursor_color_btn)

        hint = QLabel(
            "Tip: hold Ctrl and scroll the mouse wheel anywhere to zoom. "
            "Ctrl + / Ctrl - change scale in steps of 0.1. Ctrl 0 resets to 1.0x."
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        form.addRow(hint)

        layout.addWidget(appearance_box)
        layout.addStretch(1)
        return tab

    def _build_paths_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Default directories")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._input_dir = _PathRow(placeholder="Videos_to_process/")
        self._input_dir.textChanged.connect(
            lambda t: self._settings.set("input_video_directory", t)
        )
        form.addRow("Input videos:", self._input_dir)

        self._processing_dir = _PathRow(placeholder="Processed_videos_Frame_by_frame/")
        self._processing_dir.textChanged.connect(
            lambda t: self._settings.set("processing_root", t)
        )
        form.addRow("Processing root:", self._processing_dir)

        self._output_dir = _PathRow(placeholder="Output directory")
        self._output_dir.textChanged.connect(
            lambda t: self._settings.set("output_directory", t)
        )
        form.addRow("Output directory:", self._output_dir)

        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _build_step1_tab(self) -> QWidget:
        # Default sampling window used by Step 1, Stereotype classification and
        # Image segmentation when no per-run override is given. Exposing the
        # values in Settings (rather than burying them in code) follows
        # Nielsen, J., "Usability Engineering", 1993, ISBN
        # 0-12-518406-9, heuristic 6 (recognition rather than recall),
        # and matches the reproducibility argument of Sandve, G. K. et
        # al., "Ten Simple Rules for Reproducible Computational
        # Research", PLOS Computational Biology, 2013, DOI:
        # 10.1371/journal.pcbi.1003285, rule 6.
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Video sampling window (Step 1)")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._s1_start = QDoubleSpinBox()
        self._s1_start.setRange(0.0, 100000.0)
        self._s1_start.setDecimals(3)
        self._s1_start.setSingleStep(1.0)
        self._s1_start.setSuffix(" s")
        self._s1_start.valueChanged.connect(
            lambda v: self._settings.set("step1_start_second", float(v))
        )
        form.addRow("Start second:", self._s1_start)

        self._s1_end_auto = QCheckBox(
            "Use the maximum available second per video"
        )
        self._s1_end_auto.toggled.connect(self._on_s1_end_auto_toggled)
        form.addRow("End mode:", self._s1_end_auto)

        self._s1_end_manual = QDoubleSpinBox()
        self._s1_end_manual.setRange(0.0, 100000.0)
        self._s1_end_manual.setDecimals(3)
        self._s1_end_manual.setSingleStep(1.0)
        self._s1_end_manual.setSuffix(" s")
        self._s1_end_manual.valueChanged.connect(
            lambda v: self._settings.set("step1_end_second_manual", float(v))
        )
        form.addRow("End second (manual):", self._s1_end_manual)

        self._s1_step = QDoubleSpinBox()
        self._s1_step.setRange(0.001, 3600.0)
        self._s1_step.setDecimals(3)
        self._s1_step.setSingleStep(0.5)
        self._s1_step.setSuffix(" s")
        self._s1_step.valueChanged.connect(
            lambda v: self._settings.set("step1_step_seconds", float(v))
        )
        form.addRow("Step between frames:", self._s1_step)

        hint = QLabel(
            "The sampling window is used by Step 1, Stereotype classification and "
            "Image segmentation. Stereotype classification from images starts from already-extracted "
            "frames, so these defaults do not apply to it."
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        form.addRow(hint)

        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _on_s1_end_auto_toggled(self, checked: bool) -> None:
        """Enable the manual end-second spinner only when auto is off."""
        self._s1_end_manual.setEnabled(not checked)
        self._settings.set("step1_end_second_auto", bool(checked))

    def _build_step2_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Mask2Former defaults")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # Directory that holds the segmentation checkpoint sub-folders.
        # Exposing the path (rather than hard-coding it) lets the user
        # relocate trained weights without editing the source, matching
        # the Accessible principle of the FAIR guidelines (Wilkinson
        # et al., Scientific Data, 2016, DOI: 10.1038/sdata.2016.18).
        self._s2_checkpoint_dir = _PathRow(
            placeholder="checkpoints_segmentation_model/"
        )
        self._s2_checkpoint_dir.textChanged.connect(
            lambda t: self._settings.set("step2_checkpoint_dir", t)
        )
        form.addRow("Checkpoints folder:", self._s2_checkpoint_dir)

        self._s2_batch = QSpinBox()
        self._s2_batch.setRange(1, 64)
        self._s2_batch.valueChanged.connect(
            lambda v: self._settings.set("step2_batch_size", int(v))
        )
        form.addRow("Batch size:", self._s2_batch)

        self._s2_two_pass = QCheckBox("Run two-pass inference by default")
        self._s2_two_pass.toggled.connect(
            lambda v: self._settings.set("step2_two_pass", bool(v))
        )
        form.addRow(self._s2_two_pass)

        self._s2_merge = QComboBox()
        self._s2_merge.addItems(self.STEP2_MERGE_STRATEGIES)
        self._s2_merge.currentTextChanged.connect(
            lambda t: self._settings.set("step2_merge_strategy", t)
        )
        form.addRow("Merge strategy:", self._s2_merge)

        self._s2_keep_largest = QCheckBox("Keep only the largest connected component")
        self._s2_keep_largest.toggled.connect(
            lambda v: self._settings.set("step2_keep_largest", bool(v))
        )
        form.addRow(self._s2_keep_largest)

        self._s2_overwrite = QCheckBox("Overwrite existing masks")
        self._s2_overwrite.toggled.connect(
            lambda v: self._settings.set("step2_overwrite", bool(v))
        )
        form.addRow(self._s2_overwrite)

        # Mask2Former sometimes leaves a stub hanging off the body by a thin
        # thread -- debris, or a shadow caught by the second pass. The rule is
        # general rather than aimed at one image: open the mask at this radius
        # (anything thinner than it is severed), keep the largest piece, then
        # give that piece back its exact original outline. Detached debris goes
        # the same way. Shared by all four segmentation pages, because they run
        # the same clean_seg_mask; off by default, because it changes the masks
        # and therefore the descriptors.
        self._s2_prune = QSpinBox()
        self._s2_prune.setRange(0, 30)
        self._s2_prune.setSpecialValueText("off")
        self._s2_prune.setSuffix(" px")
        self._s2_prune.setToolTip(
            "Cut every connection thinner than this radius, keep the largest "
            "object, and restore its exact outline.\n"
            "On by default at 4 px; 0 turns it off. Measured on both sets of "
            "masks, since this setting is shared:\n"
            "  52 regeneration masks -- 4 px removes the growth of "
            "Djap_Ctrl_0_R1_Q1_2024-06-12 (1.8 % of that worm) and touches no "
            "other image, up to 8 px;\n"
            "  180 stereotype masks -- 4 and 5 px change nothing at all, 6 px "
            "starts eating into two worms, 8 px into eight.\n"
            "So 4 to 5 is the window, and 4 is one step below the first radius "
            "that touches real tissue. Nothing is removed at all when the "
            "result would lose more than a third of the mask.\n"
            "Check a value with test_excroissance.py before keeping it: it "
            "changes the masks, so it changes the descriptors."
        )
        self._s2_prune.valueChanged.connect(self._on_seg_prune)

        layout.addWidget(box)

        # ---- Mask refinement -------------------------------------------
        # The two steps that CHANGE a mask rather than just produce it. Both
        # are shared by the four segmentation pages, because they all run the
        # same clean_seg_mask.
        box_mr = QGroupBox("Mask refinement")
        form_mr = QFormLayout(box_mr)
        form_mr.setHorizontalSpacing(12)
        form_mr.setVerticalSpacing(8)
        form_mr.addRow("Cut thin links, keep largest:", self._s2_prune)

        self._s2_weld = QCheckBox("Weld a gap only when the image says it is tissue")
        self._s2_weld.setToolTip(
            "A closing does not know what it fills: it plugs a notch because "
            "the notch is narrow, whether it is missing tissue or the plate "
            "seen between two arcs of a coiled worm.\n"
            "This step closes at radius 5, then asks KNN matting (Chen, Li & "
            "Tang, CVPR 2012 / IEEE TPAMI 2013) whether each proposed bridge "
            "belongs to the worm, and keeps only those that do.\n"
            "Measured on the 180 stereotype masks: 28 bridges proposed, the "
            "two that are missing tissue score 0.99 and 1.00, the 26 real "
            "holes at most 0.76. One image corrected, none degraded; the 52 "
            "regeneration masks are untouched.\n"
            "Cost about 3 s per 180-image run -- only the images that have a "
            "bridge are processed.\n"
            "Needs the pymatting package (python -m pip install --no-deps "
            "pymatting); without it the step does nothing.\n"
            "Off by default: it changes a mask, hence the descriptors. Replay "
            "the measurement with test_matting.py before turning it on."
        )
        self._s2_weld.toggled.connect(self._on_seg_weld)
        form_mr.addRow(self._s2_weld)

        layout.addWidget(box_mr)
        layout.addStretch(1)
        return tab

    def _build_step3_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Worm crop defaults (Stereotype editor)")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._s3_stereo_dir = QLineEdit()
        self._s3_stereo_dir.textChanged.connect(
            lambda t: self._settings.set("step3_stereotypie_dir", t)
        )
        form.addRow("Stereotype dir name:", self._s3_stereo_dir)

        self._s3_image_size = QSpinBox()
        self._s3_image_size.setRange(64, 1024)
        self._s3_image_size.setSingleStep(16)
        self._s3_image_size.valueChanged.connect(
            lambda v: self._settings.set("step3_image_size", int(v))
        )
        form.addRow("Image size (px):", self._s3_image_size)

        self._s3_margin = QDoubleSpinBox()
        self._s3_margin.setRange(0.0, 0.5)
        self._s3_margin.setSingleStep(0.01)
        self._s3_margin.setDecimals(3)
        self._s3_margin.valueChanged.connect(
            lambda v: self._settings.set("step3_margin_percent", float(v))
        )
        form.addRow("Margin (percent of bbox):", self._s3_margin)

        self._s3_bw = QCheckBox("Convert crops to grayscale")
        self._s3_bw.toggled.connect(
            lambda v: self._settings.set("step3_black_and_white", bool(v))
        )
        form.addRow(self._s3_bw)

        self._s3_apply_mask = QCheckBox("Apply mask on crops")
        self._s3_apply_mask.toggled.connect(
            lambda v: self._settings.set("step3_apply_mask", bool(v))
        )
        form.addRow(self._s3_apply_mask)

        self._s3_num_planaria = QSpinBox()
        self._s3_num_planaria.setRange(1, 32)
        self._s3_num_planaria.valueChanged.connect(
            lambda v: self._settings.set("step3_num_planaria", int(v))
        )
        form.addRow("Planaria per image:", self._s3_num_planaria)

        # The former "Rows of planaria" setting has been removed: the
        # arena position is encoded in the video filename (v1.2 naming
        # convention), so a single row is always correct.

        # The "Generate formatted images" toggle was removed along
        # with the formatting stage itself: no pipeline writes crops
        # any more.  The values above now describe how the
        # Stereotype editor builds a crop ON DEMAND when you select
        # a frame, so they still matter -- just at display time
        # rather than during a run.

        # Descriptor / image worker threads.  The LightGBM feature
        # pipeline is pure CPU (OpenCV + scikit-image + numpy), all
        # of which release the GIL for the bulk of their work, so a
        # thread pool scales nearly linearly with physical cores.
        # 0 means "auto": cpu_count() - 1, leaving one core for the
        # GUI event loop.  Reference for the GIL-releasing native
        # call + thread pool pattern: Beazley, D. M., "Python
        # Essential Reference", 4th ed., Addison-Wesley, 2009,
        # ISBN 978-0-672-32978-4, sec. 20.6.
        self._s3_workers = QSpinBox()
        self._s3_workers.setRange(0, 64)
        self._s3_workers.setSpecialValueText("auto (cores - 1)")
        self._s3_workers.setToolTip(
            "Worker threads used for image formatting and for the\n"
            "LightGBM descriptor computation in Step 4.\n"
            "0 = auto, which uses all detected cores minus one so\n"
            "the interface stays responsive.  Raise it on a machine\n"
            "with fast storage, lower it if the GUI stutters."
        )
        self._s3_workers.valueChanged.connect(
            lambda v: self._settings.set("step3_max_workers", int(v))
        )
        form.addRow("Worker threads:", self._s3_workers)

        self._s1_workers = QSpinBox()
        self._s1_workers.setRange(0, 16)
        self._s1_workers.setSpecialValueText("auto (cores, max 4)")
        self._s1_workers.setToolTip(
            "Videos decoded in parallel when extracting frames.\n"
            "Decoding releases the interpreter lock, so several videos\n"
            "really do run at once; each thread writes its own folder.\n"
            "0 = auto (core count, capped at 4). Use 1 on a spinning or\n"
            "network drive, where more readers make things slower, or to\n"
            "get the per-frame progress of the sequential path back."
        )
        self._s1_workers.valueChanged.connect(
            lambda v: self._settings.set("step1_max_workers", int(v))
        )
        form.addRow("Videos extracted in parallel:", self._s1_workers)

        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _build_step4_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Forecasting defaults")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # LightGBM stereotype classifier folder -- the directory that
        # carries ``classifier.py``, ``config/feature_columns.json``
        # and ``models/*.txt`` (the two boosters plus the arbitration
        # threshold).  Browsing to any copy of that folder switches
        # the classifier the application uses, so a retrained bundle
        # can be dropped in without touching the code.  Reference:
        # lightgbm_stereotype_classifier/README.md.
        self._s4_bundle_dir = _PathRow(
            placeholder="lightgbm_stereotype_classifier/"
        )
        self._s4_bundle_dir.setToolTip(
            "Folder of the LightGBM stereotype classifier (October\n"
            "2026 bundle: 285 baseline + 29 Walnut-expert descriptors,\n"
            "301 distinct, arbitration 0.85).  It must contain:\n"
            "  classifier.py\n"
            "  config/feature_columns.json\n"
            "  models/baseline_multiclass_lgbm.txt\n"
            "  models/specialist_binary_lgbm.txt\n"
            "  models/arbitration_threshold.txt\n"
            "Browse to any copy of that folder to switch classifier;\n"
            "leave empty to use the sibling\n"
            "'lightgbm_stereotype_classifier' folder shipped with the\n"
            "application."
        )
        self._s4_bundle_dir.textChanged.connect(
            lambda t: self._settings.set("step4_bundle_dir", t)
        )
        form.addRow("LightGBM classifier folder:", self._s4_bundle_dir)

        # The legacy classifier checkpoint folder and the batch-size
        # spinner were removed with the switch to LightGBM: there is
        # exactly one classifier now, it loads from the folder above,
        # and gradient-boosted trees score a whole table at once so
        # there is no mini-batch to size.

        self._s4_workers = QSpinBox()
        self._s4_workers.setRange(0, 16)
        self._s4_workers.valueChanged.connect(
            lambda v: self._settings.set("step4_num_workers", int(v))
        )
        form.addRow("Num workers:", self._s4_workers)

        # 4-second isolated-class smoothing.  When checked (default),
        # Step 4 walks every per-video results CSV after writing it
        # and inspects every sliding 4-frame window (positions i,
        # i+1, i+2, i+3 ordered by ``second_value``).  Two patterns
        # rewrite a frame's class to the majority class of the
        # remaining three frames in the same window:
        #   Pattern A: slots {i, i+2, i+3} agree, slot i+1 differs
        #              -> rewrite slot i+1.
        #   Pattern B: slots {i, i+1, i+3} agree, slot i+2 differs
        #              -> rewrite slot i+2.
        # Targets the single-frame "blip" pattern common on
        # transition frames.  Temporal-coherence assumption from
        # Tian, Y. et al., "What Should Not Be Contrastive in
        # Contrastive Learning", ICLR 2021, arXiv:2008.05659, sec. 2
        # (single-frame label noise in video classifiers).
        self._s4_4s_correction = QCheckBox(
            "4-second timespan correction (default ON)"
        )
        self._s4_4s_correction.setToolTip(
            "After every per-video forecast CSV is written, slide a\n"
            "4-frame window over the rows in acquisition order.  In\n"
            "each window of frames {1, 2, 3, 4}:\n"
            " - if frames 1, 3, 4 share the same class and frame 2\n"
            "   differs, rewrite frame 2 to the majority class;\n"
            " - if frames 1, 2, 4 share the same class and frame 3\n"
            "   differs, rewrite frame 3 to the majority class.\n"
            "Targets the single-frame label blip emitted on\n"
            "transition frames.  Turn off when one-frame events are\n"
            "signal rather than noise."
        )
        self._s4_4s_correction.toggled.connect(
            lambda v: self._settings.set("step4_4s_correction", bool(v))
        )
        form.addRow("Post-processing:", self._s4_4s_correction)

        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _build_step6_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Period fusion defaults")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # Species, compound, concentration, plate and arena are all derived
        # per video by Step 6 from the v1.1 directory names
        # (species_compound_concentration_plate_position_period), following
        # Sandve, G. K. et al., "Ten Simple Rules for Reproducible
        # Computational Research", PLOS Computational Biology, 2013,
        # DOI: 10.1371/journal.pcbi.1003285, rule 6 (avoid manual
        # data manipulation steps). The settings tab therefore only
        # stores what is NOT encoded in filenames.
        self._s6_start_index = QSpinBox()
        self._s6_start_index.setRange(1, 999)
        self._s6_start_index.valueChanged.connect(
            lambda v: self._settings.set("step6_new_planaria_index", int(v))
        )
        form.addRow("Starting planaria index:", self._s6_start_index)

        hint = QLabel(
            "Period names are derived from the v1.1 video directory "
            "names (<code>compound_concentration_plate_position_period</code>), "
            "so no list is stored in the application settings."
        )
        hint.setObjectName("mutedLabel")
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        form.addRow(hint)

        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    # Factory defaults for the eyespot thresholds (match config.py). The
    # "Reset to defaults" button in the Eyespot tab restores these two.
    EYESPOT_BALANCED_DEFAULT = 0.80
    EYESPOT_POSITIVE_DEFAULT = 0.60

    def _build_eyespot_tab(self) -> QWidget:
        """Single home for every eyespot detection threshold.

        These two thresholds are shared by every page that counts eyespots
        (Forecast regeneration states, Eyespots counter, Regeneration states
        editor); they are edited ONLY here so there is one source of truth.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Eyespot detection thresholds")
        form = QFormLayout(box)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(4)

        # Both thresholds are a confidence level on the eyespot heat map: a
        # peak counts as an eyespot only where the heat map exceeds it. A
        # higher value is stricter (fewer, more confident eyespots); a lower
        # value is more permissive (catches fainter eyespots, but more false
        # positives such as a reflection).
        self._cup_balanced = QDoubleSpinBox()
        self._cup_balanced.setRange(0.0, 1.0)
        self._cup_balanced.setSingleStep(0.05)
        self._cup_balanced.setDecimals(2)
        self._cup_balanced.setToolTip(
            "Confidence threshold for the balanced (stricter) eyespot count. "
            "A peak on the eyespot heat map counts only above this value. "
            "Higher = fewer, more confident eyespots. This is also the "
            "threshold used to draw the red eyespot crosses in the editors."
        )
        self._cup_balanced.valueChanged.connect(
            lambda v: self._settings.set("eyespots_balanced_threshold", float(v))
        )
        form.addRow("Balanced threshold:", self._cup_balanced)
        bal_hint = QLabel(
            "Balanced: the stricter confidence level. Use it for a conservative "
            "eyespot count and for the red crosses shown in the editors."
        )
        bal_hint.setObjectName("mutedLabel")
        bal_hint.setWordWrap(True)
        form.addRow(bal_hint)

        self._cup_positive = QDoubleSpinBox()
        self._cup_positive.setRange(0.0, 1.0)
        self._cup_positive.setSingleStep(0.05)
        self._cup_positive.setDecimals(2)
        self._cup_positive.setToolTip(
            "Confidence threshold for the positive (more permissive) eyespot "
            "count, the lower of the two. It is used as the detection "
            "threshold when the heat map is computed, so it catches fainter "
            "eyespots than the balanced one."
        )
        self._cup_positive.valueChanged.connect(
            lambda v: self._settings.set("eyespots_positive_threshold", float(v))
        )
        form.addRow("Positive threshold:", self._cup_positive)
        pos_hint = QLabel(
            "Positive: the more permissive (lower) confidence level, used as "
            "the counting threshold. It detects fainter eyespots but is more "
            "exposed to false positives."
        )
        pos_hint.setObjectName("mutedLabel")
        pos_hint.setWordWrap(True)
        form.addRow(pos_hint)
        layout.addWidget(box)

        reset_row = QHBoxLayout()
        reset_btn = QPushButton("Reset eyespot thresholds to defaults")
        reset_btn.setToolTip(
            "Restore the two eyespot thresholds to their factory values "
            "(balanced %.2f, positive %.2f)."
            % (self.EYESPOT_BALANCED_DEFAULT, self.EYESPOT_POSITIVE_DEFAULT)
        )
        reset_btn.clicked.connect(self._on_reset_eyespot)
        reset_row.addWidget(reset_btn)
        reset_row.addStretch(1)
        layout.addLayout(reset_row)

        hint = QLabel(
            "These thresholds apply everywhere eyespots are counted "
            "(Forecast regeneration states, Eyespots counter, Regeneration "
            "states editor). They are configured only here."
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return tab

    def _on_reset_eyespot(self) -> None:
        """Restore the two eyespot thresholds to their factory defaults.

        Setting the spin values emits ``valueChanged``, which persists them.
        """
        self._cup_balanced.setValue(self.EYESPOT_BALANCED_DEFAULT)
        self._cup_positive.setValue(self.EYESPOT_POSITIVE_DEFAULT)
        # Setting a spin box to a value it already holds emits no signal, so
        # write both keys explicitly to guarantee persistence.
        self._settings.set("eyespots_balanced_threshold", self.EYESPOT_BALANCED_DEFAULT)
        self._settings.set("eyespots_positive_threshold", self.EYESPOT_POSITIVE_DEFAULT)

    def _build_pipeline_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = QGroupBox("Pipeline behaviour")
        form = QFormLayout(box)
        self._pipeline_stop = QCheckBox("Stop the pipeline on the first error")
        self._pipeline_stop.toggled.connect(
            lambda v: self._settings.set("pipeline_stop_on_error", bool(v))
        )
        form.addRow(self._pipeline_stop)

        # Detailed log. The measurement trace -- block durations, CPU seconds,
        # cache hit rates per worker, the memory planner's arithmetic, the GPU
        # self-test -- is hidden during a normal run: it describes the machine,
        # not the worms, and it is the bulk of what scrolls past. This box
        # brings it back when a run feels slow. It is written to the
        # environment, so the segmentation and descriptor subprocesses started
        # afterwards inherit it.
        self._verbose_log = QCheckBox(
            "Detailed log (diagnostics: durations, caches, memory planner)")
        self._verbose_log.setToolTip(
            "Off: the log shows what was processed and how long each step "
            "took.\nOn: every measurement line is shown as well -- useful to "
            "find out why a run was slow.\nTakes effect on the next run.")
        self._verbose_log.toggled.connect(self._on_verbose_log)
        form.addRow(self._verbose_log)
        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _on_cursor_mode(self, _index: int) -> None:
        mode = str(self._cursor_mode.currentData() or "system")
        self._settings.set("cursor_mode", mode)
        drawn = mode == "drawn"
        self._cursor_spin.setEnabled(drawn)
        self._cursor_color_btn.setEnabled(drawn)
        try:
            from .. import cursors as _cursors
            _cursors.apply(on=drawn)
            err = _cursors.last_error() if drawn else ""
            if err:
                # Silence here would look like the setting does nothing.
                self._cursor_mode.setToolTip(
                    "The drawn pointer could not be installed: %s" % err)
                _msg_warning(
                    self, "Pointer",
                    "The pointer drawn by the application could not be "
                    "installed:\n\n%s\n\nThe system pointer is kept." % err)
                self._cursor_mode.blockSignals(True)
                self._cursor_mode.setCurrentIndex(0)
                self._cursor_mode.blockSignals(False)
                self._settings.set("cursor_mode", "system")
                self._cursor_spin.setEnabled(False)
                self._cursor_color_btn.setEnabled(False)
        except Exception:  # noqa: BLE001
            pass

    def _on_cursor_size(self, value: int) -> None:
        self._settings.set("cursor_size", int(value))
        try:
            from .. import cursors as _cursors
            _cursors.apply(new_size=int(value))
        except Exception:  # noqa: BLE001
            pass

    def _on_cursor_color(self) -> None:
        from PySide6.QtWidgets import QColorDialog
        from .. import cursors as _cursors
        cur = QColor(str(self._settings.get("cursor_color", _cursors.DEFAULT_COLOR)))
        picked = QColorDialog.getColor(
            cur if cur.isValid() else QColor(_cursors.DEFAULT_COLOR), self,
            "Pointer colour")
        if not picked.isValid():
            return
        self._settings.set("cursor_color", picked.name())
        _cursors.apply(new_color=picked.name())
        self._paint_cursor_swatch(picked.name())

    def _paint_cursor_swatch(self, hexcolor: str) -> None:
        """Show the chosen colour on the button, with readable text on it."""
        c = QColor(hexcolor)
        if not c.isValid():
            c = QColor("#FFFFFF")
        lum = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
        self._cursor_color_btn.setText(c.name().upper())
        self._cursor_color_btn.setStyleSheet(
            "background-color: %s; color: %s;"
            % (c.name(), "#000000" if lum > 0.45 else "#FFFFFF"))

    def _on_seg_prune(self, value: int) -> None:
        self._settings.set("seg_prune_radius", int(value))
        try:
            from ..config import apply_seg_prune
            apply_seg_prune(int(value))
        except Exception:  # noqa: BLE001
            pass

    def _on_seg_weld(self, checked: bool) -> None:
        self._settings.set("seg_weld_enabled", bool(checked))
        try:
            from ..config import apply_seg_weld
            apply_seg_weld(bool(checked))
        except Exception:  # noqa: BLE001
            pass

    def _on_verbose_log(self, checked: bool) -> None:
        self._settings.set("verbose_log", bool(checked))
        try:
            from ..config import apply_verbose_log
            apply_verbose_log(bool(checked))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------
    def _load_from_settings(self) -> None:
        s = self._settings

        theme = s.get("theme", THEME_DARK)
        for idx in range(self._theme_combo.count()):
            if self._theme_combo.itemData(idx) == theme:
                self._theme_combo.setCurrentIndex(idx)
                break
        self._scale_spin.blockSignals(True)
        self._scale_spin.setValue(float(s.get("ui_scale", 1.0) or 1.0))
        self._scale_spin.blockSignals(False)
        self._cursor_spin.blockSignals(True)
        self._cursor_spin.setValue(int(s.get("cursor_size", 48) or 48))
        self._cursor_spin.blockSignals(False)
        self._paint_cursor_swatch(str(s.get("cursor_color", "#FFFFFF")))
        _mode = str(s.get("cursor_mode", "system"))
        self._cursor_mode.blockSignals(True)
        self._cursor_mode.setCurrentIndex(1 if _mode == "drawn" else 0)
        self._cursor_mode.blockSignals(False)
        self._cursor_spin.setEnabled(_mode == "drawn")
        self._cursor_color_btn.setEnabled(_mode == "drawn")

        self._input_dir.setText(s.get("input_video_directory", ""))
        self._processing_dir.setText(s.get("processing_root", ""))
        self._output_dir.setText(s.get("output_directory", ""))

        # The video sampling window (Step 1) now lives on the Stereotype
        # classification page, so it is no longer loaded here.

        self._s2_checkpoint_dir.setText(
            s.get("step2_checkpoint_dir", "checkpoints_segmentation_model")
        )
        self._s2_batch.setValue(int(s.get("step2_batch_size", 8)))
        self._s2_two_pass.setChecked(bool(s.get("step2_two_pass", True)))
        self._s2_merge.setCurrentText(s.get("step2_merge_strategy", "skeleton_guided"))
        self._s2_keep_largest.setChecked(bool(s.get("step2_keep_largest", True)))
        self._s2_overwrite.setChecked(bool(s.get("step2_overwrite", False)))
        self._s2_prune.blockSignals(True)
        self._s2_prune.setValue(int(s.get("seg_prune_radius", 0) or 0))
        self._s2_weld.setChecked(bool(s.get("seg_weld_enabled", False)))
        self._s2_prune.blockSignals(False)

        self._s3_stereo_dir.setText(s.get("step3_stereotypie_dir", ""))
        self._s3_image_size.setValue(int(s.get("step3_image_size", 384)))
        self._s3_margin.setValue(float(s.get("step3_margin_percent", 0.0)))
        self._s3_bw.setChecked(bool(s.get("step3_black_and_white", False)))
        self._s3_apply_mask.setChecked(bool(s.get("step3_apply_mask", True)))
        self._s3_num_planaria.setValue(int(s.get("step3_num_planaria", 1)))
        self._s3_workers.setValue(int(s.get("step3_max_workers", 0)))
        self._s1_workers.setValue(int(s.get("step1_max_workers", 0)))

        self._s4_bundle_dir.setText(s.get("step4_bundle_dir", ""))
        self._s4_workers.setValue(int(s.get("step4_num_workers", 4)))
        self._s4_4s_correction.setChecked(
            bool(
                s.get(
                    "step4_4s_correction",
                    # Migration: respect the legacy 5-second key on
                    # first launch after the rename so an operator
                    # who turned the previous correction OFF keeps
                    # their choice.
                    s.get("step4_5s_correction", True),
                )
            )
        )

        self._s6_start_index.setValue(int(s.get("step6_new_planaria_index", 1)))

        self._cup_balanced.setValue(float(
            s.get("eyespots_balanced_threshold", self.EYESPOT_BALANCED_DEFAULT)
        ))
        self._cup_positive.setValue(float(
            s.get("eyespots_positive_threshold", self.EYESPOT_POSITIVE_DEFAULT)
        ))

        self._pipeline_stop.setChecked(bool(s.get("pipeline_stop_on_error", False)))
        self._verbose_log.setChecked(bool(s.get("verbose_log", False)))

    def reload_from_settings(self) -> None:
        """Public helper used after a settings reset."""
        self._load_from_settings()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_theme_changed(self) -> None:
        theme = self._theme_combo.currentData()
        if not theme:
            return
        if self._theme_manager is not None:
            self._theme_manager.set_theme(theme)
        else:
            self._settings.set("theme", theme)

    def _on_scale_changed(self, value: float) -> None:
        if self._theme_manager is not None:
            self._theme_manager.set_scale(float(value))
        else:
            self._settings.set("ui_scale", float(value))

    def _on_reset(self) -> None:
        from ..widgets.file_dialogs import question as _msg_question
        reply = _msg_question(
            self, "Reset settings",
            "Restore every preference to its factory default?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._settings.reset_to_defaults()
        self.reload_from_settings()
        if self._theme_manager is not None:
            self._theme_manager.set_theme(self._settings.get("theme", THEME_DARK))
            self._theme_manager.set_scale(float(self._settings.get("ui_scale", 1.0)))
        _msg_information(self, "Reset settings", "Defaults restored.")

    def _on_open_config(self) -> None:
        from pathlib import Path
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        config_dir = Path.home() / ".flatmorphai"
        config_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_dir)))
