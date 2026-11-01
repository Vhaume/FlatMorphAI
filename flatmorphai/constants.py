"""
Global constants for the FlatMorphAI Dashboard.

All magic strings, class lists, colours and directory conventions
are gathered here so that the rest of the code does not duplicate
them. Values reflect the v2 Notebook-Verified Specification.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stereotype class list.  The LightGBM classifier bundle emits exactly
# five classes (see ``lightgbm_stereotype_classifier/config/
# feature_columns.json``).  The historical Bridge-like class was tied
# to the retired Mahalanobis prototype path and is no longer part of
# the app.  The order below matches the LightGBM ``classes`` list
# exactly, so ``pred_idx`` returned by the classifier maps 1-to-1 to
# this list.  Reference for the "one identifier per entity" rule:
# Sandve, G. K. et al., "Ten Simple Rules for Reproducible
# Computational Research", PLOS Comput. Biol., 2013,
# DOI: 10.1371/journal.pcbi.1003285.
CLASSES_NAMES_STEREOTYPE = [
    "C-shape",       # index 0
    "Screw-like",    # index 1
    "Snake-like",    # index 2
    "Walnut",        # index 3
    "Wildtype",      # index 4
]

NUM_STEREOTYPE_CLASSES = len(CLASSES_NAMES_STEREOTYPE)

# ---------------------------------------------------------------------------
# Directory names used throughout the pipeline.
# ---------------------------------------------------------------------------
PROCESSING_ROOT_DEFAULT = "Processed_videos_Frame_by_frame"
INPUT_VIDEOS_ROOT_DEFAULT = "Videos_to_process"
CHECKPOINTS_SEGMENTATION_DIR = "checkpoints_segmentation_model"
PROTOTYPE_FILE_DEFAULT = "models/stage_mahal_allk1_best.pkl"
# Step 6 also writes a per-video metrics fusion next to all_videos.csv
# when at least one video produced a *_metrics.csv (i.e. a fusion
# inference ran).  Convention: same naming scheme as the result fusion
# so downstream code can ingest both files identically.
ALL_VIDEOS_METRICS_CSV = "all_videos_metrics.csv"
# Step 6 default fusion target. One consolidated CSV gathers every
# per-video result file produced by Step 4 into a single table, which
# simplifies downstream statistics as recommended by Wickham, H.,
# "Tidy Data", Journal of Statistical Software, 2014,
# DOI: 10.18637/jss.v059.i10.
ALL_VIDEOS_FUSION_CSV = "all_videos.csv"

# ---------------------------------------------------------------------------
# Design system colours (v2 spec section 1.5).
# ---------------------------------------------------------------------------
COLOR_ACCENT = "#2563EB"
COLOR_SUCCESS = "#16A34A"
COLOR_WARNING = "#D97706"
COLOR_ERROR = "#DC2626"
COLOR_SIDEBAR_BG = "#0F172A"
COLOR_MAIN_BG = "#1E293B"
COLOR_CARD_BG = "#334155"
COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_BORDER = "#475569"

# Status dot colours
COLOR_STATUS_IDLE = "#64748B"
COLOR_STATUS_RUNNING = COLOR_WARNING
COLOR_STATUS_DONE = COLOR_SUCCESS
COLOR_STATUS_ERROR = COLOR_ERROR

# ---------------------------------------------------------------------------
# Config file location.
# ---------------------------------------------------------------------------
CONFIG_DIR_NAME = ".flatmorphai"
CONFIG_FILE_NAME = "config.json"

# ---------------------------------------------------------------------------
# CSV I/O conventions.
# ---------------------------------------------------------------------------
CSV_SEPARATOR = ";"
CSV_ENCODING_UTF8 = "utf-8-sig"
CSV_ENCODING_LATIN1 = "latin1"

# ---------------------------------------------------------------------------
# Random seed applied in every Worker.__init__ (spec section 6.4).
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Theme and UI scale limits used by ThemeManager.
# ---------------------------------------------------------------------------
THEME_DARK = "dark"
THEME_LIGHT = "light"
AVAILABLE_THEMES = (THEME_DARK, THEME_LIGHT)

UI_SCALE_MIN = 0.8
UI_SCALE_MAX = 4.0
UI_SCALE_STEP = 0.1
UI_SCALE_DEFAULT = 1.15

# Stylesheet file names relative to the package parent directory.
STYLESHEET_FILES = {
    THEME_DARK: "flatmorphai_style_dark.qss",
    THEME_LIGHT: "flatmorphai_style_light.qss",
}

# The ConvNeXt-V2 checkpoint roots (stereotype and regeneration) were
# removed on 31 October 2026 with that classifier: the stereotype chain uses
# the LightGBM bundle, and CHECKPOINTS_REGENERATION_DIR was never read.
