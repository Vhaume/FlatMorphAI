"""
Persistent application settings.

Stores configuration in JSON at ``~/.flatmorphai/config.json`` and writes
with a 500 ms debounced QTimer on every parameter change. The
``AppSettings`` singleton exposes a dict-like ``get`` / ``set`` API and
emits ``changed`` whenever any value is updated so that widgets bound
to the same key stay in sync.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from .constants import (
    CHECKPOINTS_SEGMENTATION_DIR,
    CONFIG_DIR_NAME,
    CONFIG_FILE_NAME,
    PROCESSING_ROOT_DEFAULT,
    INPUT_VIDEOS_ROOT_DEFAULT,
    THEME_LIGHT,
    UI_SCALE_DEFAULT,
)


# ----------------------------------------------------------------------
# Portable folder paths
# ----------------------------------------------------------------------
# The folders the pages remember are often inside the application folder
# (Processed_videos_Frame_by_frame, Processed_regeneration_images, the
# checkpoints, ...). Stored as absolute paths, they would tie the settings to
# one copy of the application: a second copy (a clone, the GitHub export, a
# moved folder) reading the same ~/.flatmorphai/config.json would open the
# folders of the first one. So, on disk, a path inside the application folder
# is written relative to it, behind the prefix "@app/", and read back against
# the folder of the copy that is running. Paths outside the application folder
# are kept as they are. Only the file on disk changes: in memory every value is
# what the page set, so nothing else in the application sees the difference.
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_ROOT_REAL = os.path.realpath(_APP_ROOT)
_PORTABLE_PREFIX = "@app/"
#: Written once the absolute paths of a configuration written before portable
#: paths existed have been converted (see ``_from_disk``).
_PORTABLE_MARKER = "portable_paths_v1"
#: The copy of the application that last wrote the file. A relative path that
#: does not exist in the running copy is looked for there, so that a folder
#: only one copy holds (a dataset kept outside the repository, say) is still
#: found by the other.
_WRITER_KEY = "portable_paths_writer"


def _owning_app_root(real_path: str) -> Optional[str]:
    """The nearest folder above ``real_path`` (itself included) that holds a
    copy of the application, or None."""
    cur = real_path
    while True:
        if os.path.isfile(os.path.join(cur, "flatmorphai", "main_window.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _to_disk_value(value: Any) -> Any:
    """Absolute path inside THIS copy of the application -> "@app/<relative>"."""
    if isinstance(value, str):
        if value and os.path.isabs(value):
            real = os.path.realpath(value)
            if _owning_app_root(real) == _APP_ROOT_REAL:
                rel = os.path.relpath(real, _APP_ROOT_REAL).replace(os.sep, "/")
                return _PORTABLE_PREFIX + rel
        return value
    if isinstance(value, list):
        return [_to_disk_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_disk_value(v) for k, v in value.items()}
    return value


def _from_disk_value(value: Any, legacy: bool, writer: Optional[str] = None) -> Any:
    """"@app/<relative>" -> absolute path inside the running copy.

    ``legacy``: the configuration predates portable paths. An absolute path
    inside ANOTHER copy of the application is then moved to the same place in
    this copy, when that place exists -- once: afterwards, an absolute path is
    always the operator's explicit choice and is kept verbatim.
    """
    if isinstance(value, str):
        if value.startswith(_PORTABLE_PREFIX):
            rel = value[len(_PORTABLE_PREFIX):].split("/")
            here = os.path.normpath(os.path.join(_APP_ROOT, *rel))
            if not os.path.exists(here) and writer:
                there = os.path.normpath(os.path.join(writer, *rel))
                if os.path.exists(there):
                    return there
            return here
        if legacy and value and os.path.isabs(value):
            real = os.path.realpath(value)
            owner = _owning_app_root(real)
            if owner is not None and owner != _APP_ROOT_REAL:
                cand = os.path.normpath(
                    os.path.join(_APP_ROOT, os.path.relpath(real, owner)))
                if os.path.exists(cand):
                    return cand
        return value
    if isinstance(value, list):
        return [_from_disk_value(v, legacy, writer) for v in value]
    if isinstance(value, dict):
        return {k: _from_disk_value(v, legacy, writer) for k, v in value.items()}
    return value


def _default_settings() -> Dict[str, Any]:
    """Return the default configuration dictionary."""
    return {
        # Global paths. ``processing_root`` stays the app-wide default
        # (edited on the Settings page and used by the step-by-step
        # workflow). Each stand-alone page instead remembers its OWN last
        # processing root under a dedicated key, saved locally the moment
        # the field changes, so changing it on one page never affects the
        # others. Each page falls back to ``processing_root`` when its own
        # key is still empty.
        "input_video_directory": INPUT_VIDEOS_ROOT_DEFAULT,
        "processing_root": PROCESSING_ROOT_DEFAULT,
        "output_directory": PROCESSING_ROOT_DEFAULT,
        "pipeline_a_processing_root": "",
        "pipeline_b_processing_root": "",
        "pipeline_c_processing_root": "",
        "regen_processing_root": "",
        "eyespots_processing_root": "",
        "regedit_processing_root": "",
        "step5_processing_root": "",

        # UI preferences
        # Sidebar key of the last visited page; restored on launch.
        "last_page": "",
        "theme": THEME_LIGHT,
        "ui_scale": UI_SCALE_DEFAULT,

        # Step 1 parameters. ``step1_end_second_auto`` = True means the
        # frame extractor samples up to the last integer second of each
        # video, regardless of clip length. Manual mode uses
        # ``step1_end_second_manual`` instead. Rationale for exposing a
        # persisted default follows Sandve, G. K. et al., "Ten Simple
        # Rules for Reproducible Computational Research", PLOS
        # Computational Biology, 2013, DOI:
        # 10.1371/journal.pcbi.1003285, rule 6 (avoid manual data
        # manipulation steps).
        "step1_start_second": 0.0,
        "step1_end_second_auto": True,
        "step1_end_second_manual": 60.0,
        "step1_step_seconds": 1.0,

        # Step 2 parameters. ``step2_checkpoint_dir`` is the folder that
        # contains the Mask2Former checkpoint sub-folders produced by the
        # training notebook. The default points at the
        # checkpoints_segmentation_model/ directory shipped alongside
        # the application, which is how the BTSB notebooks organise
        # trained weights (see section 2 of
        # "2 - Generate image mask.ipynb"). Exposing it as a user
        # preference matches the FAIR Accessible principle from
        # Wilkinson, M. D. et al., "The FAIR Guiding Principles for
        # scientific data management and stewardship", Scientific
        # Data, Vol 3, March 2016, DOI: 10.1038/sdata.2016.18.
        "step2_checkpoint_dir": CHECKPOINTS_SEGMENTATION_DIR,
        "step2_checkpoint_model": "",
        "step2_batch_size": 8,
        "step2_two_pass": True,
        "step2_merge_strategy": "skeleton_guided",
        "step2_keep_largest": True,
        "step2_overwrite": False,

        # Step 3 parameters. Key names match the ones used by Step 3,
        # Pipeline A and Pipeline B so the Settings tab and each page
        # share a single source of truth.
        # v1.2.1 renames the per-video processed-image directory to
        # ``Formated_image-size(S)_margin(M)_b&w(B)_apply-mask(A)`` so
        # the prefix names what the directory contains rather than the
        # research project that motivated it. The Step 5 reader still
        # accepts the legacy ``Regeneration_process_of_Planaria_*``
        # form for backward compatibility with previously processed
        # videos.
        "step3_stereotypie_dir": "Formated",
        "step3_image_size": 384,
        # Step 3 margin must match the value the classification model
        # was trained with. Notebook 4 hardcodes _margin(0.03) in the
        # expected directory name but the deployed prototype in this
        # workspace was actually built at margin=0.0 (the user verified
        # this by passing the same images the Bridge-like prototype was
        # constructed from and observing the misclassification persists
        # with margin=0.03). The default is therefore reverted to 0.0;
        # the resolver still matches on the directory's _margin(M)
        # token so a re-trained model with a different margin can be
        # picked up by simply rerunning Step 3 at the new value.
        # Train/test preprocessing-parity rule reference:
        # Wilkinson, M. D. et al., "The FAIR Guiding Principles for
        # scientific data management and stewardship", Scientific
        # Data, 2016, DOI: 10.1038/sdata.2016.18.
        "step3_margin_percent": 0.0,
        "step3_black_and_white": False,
        "step3_apply_mask": True,
        "step3_num_planaria": 1,
        # step3_planaria_rows was dropped with the v1.2 naming convention:
        # the arena position is encoded in the filename so a single row
        # is always correct.
        "step3_mask_dir": "",
        # Worker threads for image formatting AND for the LightGBM
        # descriptor computation.  0 means "auto" = cpu_count() - 1,
        # leaving one core for the Qt event loop.
        # Videos decoded in parallel by step 1 (0 = from the core count,
        # capped at 4; 1 = sequential).
        "step1_max_workers": 0,
        "step3_max_workers": 0,

        # Step 4 parameters.  The ONLY classifier is the LightGBM
        # bundle; its folder is ``step4_bundle_dir`` (empty = the
        # sibling ``lightgbm_stereotype_classifier`` shipped with the
        # application).  The former ``step4_checkpoint_dir`` /
        # ``step4_checkpoint_model`` / ``step4_batch_size`` keys were
        # removed with the legacy neural classifier: boosted trees
        # score a whole feature table in one call, so there is no
        # mini-batch to size.
        # Empty = use the sibling ``lightgbm_stereotype_classifier``
        # folder shipped with the application; set it from the Step 4
        # Settings tab to point at any other copy of that bundle.
        "step4_bundle_dir": "",
        "step4_num_workers": 4,

        # Step 6 parameters. Species, compound, concentration, plate,
        # arena and period are all derived per video from the v1.2
        # directory names, so none of them need a persisted default.
        # The default output directory is the same processing root so
        # the fused "all_videos.csv" sits next to the per-video CSVs
        # written by Step 4.
        "step6_planaria_species": "",
        "step6_new_planaria_index": 1,
        "step6_output_directory": PROCESSING_ROOT_DEFAULT,
        "step6_fused_csv_name": "all_videos.csv",

        # Pipeline C uses the shared ``processing_root`` (same input as
        # Pipeline B); no dedicated key is needed any more.

        # Pipeline options
        "pipeline_stop_on_error": False,
        # Detailed log: show the measurement trace (durations, caches, memory
        # planner) on top of what the run actually did. Off by default.
        "verbose_log": False,
        # Mouse pointer size in pixels. Read at start-up, BEFORE Qt builds its
        # cursors, and handed to the X cursor theme through XCURSOR_SIZE: the
        # theme then draws every shape at that size -- arrow, open and closed
        # hand on the viewers, text beam, busy -- so the shapes and their
        # animations are the system's, only bigger. Qt cannot resize a cursor
        # after the fact, hence the restart.
        "cursor_size": 48,
        # Pointer body colour. The outline is chosen against it (black on a
        # light body, white on a dark one), so the pointer stays visible on
        # any page whatever colour is picked.
        "cursor_color": "#FFFFFF",
        # "system": the pointer is the one Qt and Windows provide, untouched.
        # "drawn": the application draws its own, at the size and colour
        # above. System by default: the drawn pointer is consistent inside
        # the window but differs from the one Windows shows on the frame and
        # on the rest of the desktop.
        "cursor_mode": "system",
        # Cut connections thinner than this radius and keep the largest
        # object (px, 0 = off). Shared by every segmentation page, because
        # they all run the same clean_seg_mask.
        # ON by default at 4 px since 30/10/2026, measured on BOTH sets of
        # masks -- the setting is shared, so the stereotype chain decides as
        # much as the regeneration one:
        #   * 52 regeneration masks: every radius from 4 to 8 removes the same
        #     growth on Djap_Ctrl_0_R1_Q1_2024-06-12 (1.8 % of that worm, one
        #     piece, no pixel added, worm still in one part) and leaves the 51
        #     others untouched;
        #   * 180 stereotype masks: 4 and 5 px change NOTHING at all, 6 px
        #     starts eating into two worms and 8 px into eight, up to 7.2 % of
        #     one of them.
        # The window where the growth goes and no real tissue does is 4 to 5.
        # 4 is taken rather than 5 because the failure mode is on the upper
        # side: one step below the first radius that touches a real worm.
        "seg_prune_radius": 4,

        # Image-guided weld (mask refinement): fill a gap proposed by a
        # closing only when KNN matting says the gap is tissue rather than
        # the plate seen between two arcs of a coiled worm. OFF by default --
        # it changes a mask, hence the descriptors. Needs the pymatting
        # package; without it the step does nothing.
        "seg_weld_enabled": False,

        # Forecast regeneration states page. Default input is the
        # shared processing root (the pipeline finds every ``images``
        # folder under it). The monotonic state-increase correction is on
        # by default; no thumbnail or heat map is saved by default. The two
        # eyespot thresholds give the two "strong eyespot" counts whose
        # column names carry the threshold value. (CLAHE before segmentation
        # has been removed application-wide.)
        "regen_save_body_masks": True,
        "regen_save_eyespots_heatmaps": False,
        "regen_save_blasteme_masks": False,
        "regen_save_blasteme_csv": False,
        "regen_save_thumb_all_body": False,
        "regen_save_thumb_head_zoomed": False,
        "regen_save_thumb_regeneration": False,
        "regen_save_thumb_blastemes": False,
        "regen_save_thumb_eyespots": False,
        "regen_auto_state_increase_correction": True,
        "regen_auto_eyespots_increase_correction": True,
        # A certain eyespot (strict threshold) opens the door to the others
        # at the permissive one. One key per page: the two chains already use
        # different thresholds, the regeneration one staying strict.
        "regen_eyespots_pair_rule": False,
        "eyespots_counter_pair_rule": False,
        # Eyespot thresholds are shared (keys ``eyespots_balanced_threshold`` /
        # ``eyespots_positive_threshold`` below) and edited only in
        # Settings > Eyespot; the old per-page ``regen_eyespots_*`` keys are gone.
        # Optional mask refinement: grow the mask with the blastema prediction
        # (off by default).
        "regen_blasteme_refined": False,

        # Pipeline C (segmentation only) options. The blastema refinement is
        # OFF by default. (CLAHE and the Gaussian edge smoothing have been
        # removed application-wide.)
        "pipeline_c_blasteme_refined": False,

        # Eyespots counter page. Counts the eyespots at BOTH
        # extremities of every worm and writes eyespots_detection.csv with,
        # per threshold, a count and the (x, y) positions. Masks are
        # generated only when missing. Neither the thumbnail heat maps nor
        # the detection thumbnails are saved by default. The two counting
        # thresholds reuse the application convention (0.80 balanced, 0.60
        # positive), matching the regeneration pipeline.
        "eyespots_save_heatmaps": False,
        "eyespots_save_thumbnails": False,
        "eyespots_balanced_threshold": 0.80,
        "eyespots_positive_threshold": 0.60,

        # Regeneration states editor. Remembers the last loaded
        # regeneration_states.csv, the two preview overlay toggles and
        # whether each of the two evolution graphs is expanded. Both
        # graphs start expanded; the overlays start off.
        "regedit_csv_path": "",
        "regedit_show_blasteme": False,
        "regedit_show_eyespots": False,
        "regedit_graph_states_expanded": True,
        "regedit_graph_eyespots_expanded": True,

        # Pipeline D (mask metrics) parameters. The output directory
        # defaults to the processing root so the metric CSVs sit next
        # to the per-video folders. The eigenposture component count
        # follows Stephens, G. J., et al., "Dimensionality and dynamics
        # in the behavior of C. elegans", PLOS Computational Biology,
        # 2008, DOI: 10.1371/journal.pcbi.1000028, where 4 to 6
        # components capture more than 95 percent of postural
        # variance.
        "pipelined_output_directory": "",
        "pipelined_eigen_components": 6,
        "pipelined_detect_well": True,

        # Pipeline E (scrunching detector) parameters. The defaults
        # mirror the four scrunching parameters reported in
        # Cochet-Escartin, O., Mickolajczyk, K. J. and Collins,
        # E. M.-S., "Scrunching: a novel escape gait in planarians",
        # Physical Biology, 2015,
        # DOI: 10.1088/1478-3975/12/5/056010, refined in Sabry, Z.
        # et al., 2020, DOI: 10.3791/61549, and reviewed in Reho, G.,
        # Lelievre, V. and Cadiou, H., "Planarian nociception:
        # Lessons from a scrunching flatworm", Frontiers in Molecular
        # Neuroscience, 2022, DOI: 10.3389/fnmol.2022.935918.
        "pipelinee_output_directory": "",
        "pipelinee_step_seconds": 0.25,
        "pipelinee_window_seconds": 5.0,
    }


def cursor_size_from_disk(default: int = 48) -> int:
    """The saved pointer size, read WITHOUT building the settings object.

    Called from ``configure_environment`` before ``QApplication`` exists, so
    it cannot go through :class:`AppSettings` (a QObject with a timer). Plain
    JSON, and the default on any problem: a pointer that is the wrong size
    must never stop the application from starting.
    """
    try:
        import json as _json
        from pathlib import Path as _Path
        p = _Path.home() / CONFIG_DIR_NAME / CONFIG_FILE_NAME
        with open(p, "r", encoding="utf-8") as fh:
            v = int((_json.load(fh) or {}).get("cursor_size", default))
        return v if 8 <= v <= 256 else default
    except Exception:  # noqa: BLE001
        return default


def apply_seg_prune(radius: int) -> None:
    """Put the "cut thin connections, keep the largest object" radius in the
    environment.

    ``clean_seg_mask`` (shared by the four segmentation pages) reads
    FLATMORPHAI_SEG_KEEPMAIN: it opens the mask at this radius, which severs
    anything hanging by a thread thinner than it, keeps the largest piece, and
    gives that piece back its exact original outline. A general rule rather
    than one aimed at a particular image: it also drops detached debris. 0
    disables it, which is the default -- it changes the masks, hence the
    descriptors, so it is the operator's decision.
    """
    try:
        r = max(0, int(radius))
    except (TypeError, ValueError):
        r = 0
    os.environ["FLATMORPHAI_SEG_KEEPMAIN"] = str(r)


#: Alpha above which a gap proposed by the closing is welded. Measured on the
#: 180 stereotype masks: the two bridges that are missing tissue score 0.99 and
#: 1.00, the 26 real holes at most 0.76 (half resolution) or 0.68 (full), so
#: 0.87 sits in an empty interval in both modes, about 0.11 away from each
#: side. See test_matting.py, which replays the measurement.
SEG_WELD_THRESHOLD = 87


def apply_seg_weld(enabled: bool) -> None:
    """Switch the image-guided weld of :func:`mask_refine.clean_seg_mask`.

    OFF by default: it changes a mask, hence the descriptors. On the reference
    data it corrects exactly one image out of 180 (the worm whose Mask2Former
    mask is incomplete) and leaves the 179 others and the 52 regeneration
    masks bit for bit identical.
    """
    os.environ["FLATMORPHAI_SEG_WELD"] = (
        str(int(SEG_WELD_THRESHOLD)) if enabled else "0"
    )


def apply_verbose_log(enabled: bool) -> None:
    """Put the detailed-log choice in the environment.

    The log filter lives in ``flatmorphai.workers.log_noise`` and is read by
    the main process AND inherited by every subprocess started afterwards
    (they are launched with a copy of this environment), so one variable
    covers the whole application.
    """
    os.environ["FLATMORPHAI_VERBOSE"] = "1" if enabled else "0"


class AppSettings(QObject):
    """Dict-like configuration store with debounced JSON persistence."""

    changed = Signal(str, object)  # key, value

    _instance: Optional["AppSettings"] = None

    def __new__(cls) -> "AppSettings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Prevent re-initialisation of the singleton.
        if getattr(self, "_initialised", False):
            return
        super().__init__()
        self._initialised = True
        self._data: Dict[str, Any] = _default_settings()
        self._path: Path = self._resolve_config_path()
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)  # 500 ms debounce
        self._save_timer.timeout.connect(self._flush_to_disk)
        self._load()
        # First-run hardware calibration: pick worker-count defaults
        # from the detected CPU count.  Runs exactly once per
        # profile version so a manual override the user set later
        # is preserved on subsequent launches.
        self.apply_hardware_defaults()
        # One-shot migration for the legacy ``step3_stereotypie_dir``
        # value.  Very early builds shipped the prefix
        # ``Regeneration_process_of_Planaria`` which is verbose and
        # confuses downstream folder resolvers.  Rewriting it here
        # ensures new Step 3 runs land under the canonical
        # ``Formated_*`` prefix.  Existing folders on disk keep the
        # legacy prefix and stay resolvable because the reader-side
        # helpers accept both prefixes.
        self._migrate_legacy_stereotypie_dir()
        # The detailed-log choice is an environment variable, because the
        # subprocesses (segmentation, descriptors) have to see it too. Applied
        # here so it holds from the first run of the session.
        apply_verbose_log(bool(self._data.get("verbose_log", False)))
        self.apply_cleanup_default()
        if not self._data.get(self._HEADUP_MARKER):
            self._data["regedit_show_headup"] = True
            self._data[self._HEADUP_MARKER] = True
            self.flush()
        apply_seg_prune(self._data.get("seg_prune_radius", 0))
        apply_seg_weld(bool(self._data.get("seg_weld_enabled", False)))

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_config_path() -> Path:
        home = Path.home()
        directory = home / CONFIG_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        return directory / CONFIG_FILE_NAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return the value associated with ``key`` or ``default``."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` and schedule a disk write."""
        if self._data.get(key) == value:
            return
        self._data[key] = value
        self.changed.emit(key, value)
        self._save_timer.start()

    def all(self) -> Dict[str, Any]:
        """Return a shallow copy of the full settings dictionary."""
        return dict(self._data)

    def flush(self) -> None:
        """Force an immediate write to disk (bypasses the debounce)."""
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._flush_to_disk()

    def reset_to_defaults(self) -> None:
        """Replace every stored value with the factory default and flush."""
        defaults = _default_settings()
        changed_keys: list[str] = []
        for key, value in defaults.items():
            if self._data.get(key) != value:
                self._data[key] = value
                changed_keys.append(key)
        for key in changed_keys:
            self.changed.emit(key, self._data[key])
        # Force the hardware calibration to re-apply on next launch
        # so a "Reset to defaults" click actually returns worker
        # settings to the CPU-derived values, not the compile-time
        # factory constants.
        self._data.pop("hardware_defaults_applied_v1", None)
        self.flush()
        self.apply_hardware_defaults()

    # ------------------------------------------------------------------
    # Legacy setting migrations
    # ------------------------------------------------------------------
    def _migrate_legacy_stereotypie_dir(self) -> None:
        """Rewrite the legacy ``Regeneration_process_of_Planaria``
        value of ``step3_stereotypie_dir`` to the canonical
        ``Formated``.  Silent no-op when the setting was never
        touched or was already migrated.  Any user-set custom value
        (anything OTHER than the legacy string) is preserved
        verbatim.
        """
        current = str(self._data.get("step3_stereotypie_dir", ""))
        if current == "Regeneration_process_of_Planaria":
            self._data["step3_stereotypie_dir"] = "Formated"
            self.changed.emit("step3_stereotypie_dir", "Formated")
            self.flush()

    # ------------------------------------------------------------------
    # Hardware calibration
    # ------------------------------------------------------------------
    _HARDWARE_MARKER = "hardware_defaults_applied_v1"
    #: Written once the mask cleanup has been switched on for an existing
    #: configuration. Without it, a profile created before 30/10/2026 (which
    #: holds an explicit seg_prune_radius = 0, the old default) would keep the
    #: cleanup off for ever, since a stored value always wins over a default.
    _CLEANUP_MARKER = "seg_cleanup_default_applied_v1"
    #: The regeneration editor's Head-up toggle saved itself as unchecked on
    #: every launch for a few hours on 30/10/2026 (the four overlay toggles
    #: wrote all four settings while they were still being restored, so the
    #: first one to be restored overwrote the ones that had not been read
    #: yet). Any configuration touched in that window holds a False that the
    #: user never chose, so it is reset once.
    _HEADUP_MARKER = "regedit_headup_fixed_v1"

    def apply_hardware_defaults(self) -> None:
        """Initialise worker-count settings from the detected CPU count.

        Runs on first launch (and after :meth:`reset_to_defaults`).
        The marker key ``hardware_defaults_applied_v1`` is written
        once the calibration completes so subsequent launches keep
        any manual override the user set later.  Only the settings
        that meaningfully scale with CPU cores are touched; the GPU-
        bound batch size (``step2_batch_size``) stays at its
        model-derived default and remains user-editable.

        Reference for the "detect once, respect user overrides
        afterwards" pattern: Nielsen, J., "10 Usability Heuristics
        for User Interface Design", 1994,
        https://www.nngroup.com/articles/ten-usability-heuristics/,
        heuristic 3 (user control and freedom).
        """
        if self._data.get(self._HARDWARE_MARKER):
            return
        from .cpu_topology import logical_cores, physical_cores
        cpu = logical_cores()
        # Leave ONE core for the GUI event loop and reserve a second
        # for the CUDA context / segmentation model threads.  Cap at
        # 12 to avoid over-subscribing DataLoader shuffling threads
        # on 32-core boxes (the parallel gain flattens after that
        # for our per-frame OpenCV routines).
        recommended = max(1, min(12, cpu - 2))
        _phys = physical_cores()
        changes = {
            # Step 4 DataLoader workers -- surfaced in Settings.
            # Note: process_videos currently forces this to 0 at
            # runtime because the CUDA fork start-method segfaults
            # in the worker thread; the value stored here is what
            # the Settings UI shows and what future non-CUDA
            # data-loading paths will honour.
            "step4_num_workers": recommended,
        }
        applied: Dict[str, Any] = {}
        for key, value in changes.items():
            self._data[key] = value
            applied[key] = value
            self.changed.emit(key, value)
        self._data[self._HARDWARE_MARKER] = {
            "cpu_count": cpu,
            "physical_cores": _phys,
            "recommended": recommended,
            "applied": applied,
        }
        self.flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def apply_cleanup_default(self) -> None:
        """Switch the mask cleanup on for a configuration written before it
        existed -- once, and never again.

        The value is then the operator's: setting it back to 0 sticks,
        because the marker stays written.
        """
        if self._data.get(self._CLEANUP_MARKER):
            return
        if int(self._data.get("seg_prune_radius", 0) or 0) <= 0:
            self._data["seg_prune_radius"] = 4
            self.changed.emit("seg_prune_radius", 4)
        self._data[self._CLEANUP_MARKER] = True
        self.flush()

    def _load(self) -> None:
        if not self._path.exists():
            self._flush_to_disk()
            return
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                legacy = not stored.get(_PORTABLE_MARKER)
                writer = stored.pop(_WRITER_KEY, None)
                writer = writer if isinstance(writer, str) else None
                stored = {k: _from_disk_value(v, legacy, writer)
                          for k, v in stored.items()}
                stored[_PORTABLE_MARKER] = True
                self._data.update(stored)
                if legacy:
                    self._flush_to_disk()
        except (OSError, json.JSONDecodeError):
            # Corrupt or unreadable file: keep defaults and overwrite.
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        try:
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                on_disk = {k: _to_disk_value(v) for k, v in self._data.items()}
                on_disk[_PORTABLE_MARKER] = True
                on_disk[_WRITER_KEY] = _APP_ROOT
                json.dump(on_disk, handle, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            # Never raise from the debounce timer.
            pass
