"""
Wrapper for the LightGBM stereotype classifier bundle.

The bundle lives at
``<root>/lightgbm_stereotype_classifier/`` and ships:

  * ``classifier.py`` -- ``StereotypeClassifier`` with the two
    LightGBM boosters (baseline + Walnut-vs-C-shape specialist) and
    the deterministic arbitration rule.
  * ``config/feature_columns.json`` -- 285 baseline descriptors + 29
    Walnut-expert descriptors (301 distinct; 13 shared), arbitration
    threshold (0.85) applied to the expert probability only (the
    former ``hole_frac`` rule was dropped in the October 2026 bundle).
  * ``models/*.txt`` -- native LightGBM text models (baseline trained
    after Borderline-SMOTE, binary Walnut expert).
  * ``feature_extraction/`` -- every descriptor generator (mask3
    base metrics, geodesic width, size residuals, luminosity,
    new_metrics, expert image features, patchint, curv, nsharp) and
    ``assemble_features.py``.  The generators locate the dataset
    through the ``FLATMORPH_DATASET`` / ``FLATMORPH_PRESET`` /
    ``FLATMORPH_SPLITS`` environment variables; this wrapper points
    them at a minimal stub (see :func:`_ensure_dataset_stub`) so the
    per-family functions import without the training tree.

The bundle's modules use top-level unqualified imports
(``import classifier``, ``from feature_extraction import ...``), so
this wrapper prepends the bundle folder to ``sys.path`` on first
use.

References
----------
  * ``lightgbm_stereotype_classifier/README.md`` -- API + feature
    pipeline description (test balanced accuracy 96.0%).
  * Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree", NeurIPS 2017.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:
    import numpy as np  # noqa: F401 -- imported for callers
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


# ---------------------------------------------------------------------
# Bundle path resolution
# ---------------------------------------------------------------------
def _default_bundle_dir() -> str:
    """Return the default LightGBM bundle folder path.

    Layout expected::

        <root>/
          flatmorphai/
            notebook_api/
              lightgbm_bundle.py  <-- this file
          lightgbm_stereotype_classifier/  <-- sibling of the package
    """
    here = Path(__file__).resolve()
    return str(here.parent.parent.parent / "lightgbm_stereotype_classifier")


def _ensure_dataset_stub(bundle_dir: str) -> str:
    """Create the minimal ``dataset_training_stereotype`` tree that
    ``compute_new_metrics.find_dataset`` requires at import time and
    export the ``FLATMORPH_*`` variables that the bundle's generators
    read.  The stub lives under the application's config folder so the
    bundle folder itself is never written to.  Idempotent.
    """
    preset = "flatmorphai_app"
    root = os.path.join(
        os.path.expanduser("~"), ".flatmorphai", "lightgbm_dataset_stub",
        "dataset_training_stereotype",
    )
    try:
        os.makedirs(os.path.join(root, "mask_model_Mask2Former_20260324_0451"), exist_ok=True)
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        lst = os.path.join(root, "dataset_images_list", preset)
        os.makedirs(lst, exist_ok=True)
        for split in ("train", "predict"):
            for suffix in ("", "_mask3_metrics"):
                fp = os.path.join(lst, f"{preset}_{split}{suffix}.csv")
                if not os.path.isfile(fp):
                    with open(fp, "w", encoding="utf-8") as fh:
                        fh.write("row_index;class;image_path\n" if suffix == "" else "image\n")
    except OSError:
        pass
    os.environ.setdefault("FLATMORPH_DATASET", root)
    os.environ.setdefault("FLATMORPH_PRESET", preset)
    os.environ.setdefault("FLATMORPH_SPLITS", "train")
    return root


def _register_bundle_on_sys_path(bundle_dir: str) -> None:
    """Prepend ``bundle_dir`` and ``bundle_dir/feature_extraction`` to
    ``sys.path`` so the bundle's top-level modules resolve
    (``import classifier``, ``from feature_extraction import ...``).
    Idempotent.
    """
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(
            "LightGBM bundle folder not found: "
            f"{bundle_dir}. Set 'step4_bundle_dir' in Settings to "
            "point at the 'lightgbm_stereotype_classifier' folder."
        )
    _ensure_dataset_stub(bundle_dir)
    for p in (bundle_dir, os.path.join(bundle_dir, "feature_extraction")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def resolve_bundle_dir(
    bundle_dir: Optional[str] = None,
) -> str:
    """Return the LightGBM bundle folder (validated on disk).

    A user-provided path is honoured verbatim.  When empty the
    default sibling ``lightgbm_stereotype_classifier`` folder is
    used.
    """
    if bundle_dir:
        bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
    else:
        bundle_dir = _default_bundle_dir()
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(
            f"LightGBM bundle folder not found: {bundle_dir}."
        )
    required = [
        os.path.join(bundle_dir, "classifier.py"),
        os.path.join(bundle_dir, "config", "feature_columns.json"),
        os.path.join(bundle_dir, "models", "baseline_multiclass_lgbm.txt"),
        os.path.join(bundle_dir, "models", "specialist_binary_lgbm.txt"),
        os.path.join(bundle_dir, "models", "arbitration_threshold.txt"),
    ]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "LightGBM bundle is incomplete; missing files:\n  - "
            + "\n  - ".join(missing)
        )
    return bundle_dir


def bundle_available(bundle_dir: Optional[str] = None) -> bool:
    """Return True when :func:`resolve_bundle_dir` would succeed."""
    try:
        resolve_bundle_dir(bundle_dir)
    except FileNotFoundError:
        return False
    return True


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_classifier(
    bundle_dir: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    num_threads: Optional[int] = None,
) -> Tuple[Any, str]:
    """Instantiate ``StereotypeClassifier`` from the bundle.

    Returns ``(classifier, absolute_bundle_dir)``.  Silence the
    bundle's own imports so the terminal stays clean; a single
    "loaded" line is emitted by :func:`load_lightgbm_bundle` in
    ``classification.py``.

    ``num_threads`` sets the booster-level thread count used during
    ``predict``.  LightGBM defaults to one thread per *logical*
    core, which over-subscribes on hyper-threaded CPUs and is
    measurably slower than one thread per physical core; passing an
    explicit value avoids that.  ``None`` / ``0`` means auto
    (``cpu_count() - 1``, leaving a core for the GUI).  Reference:
    LightGBM docs, "Parameters -> num_threads" ("for the best speed,
    set this to the number of real CPU cores, not the number of
    threads").
    """
    import contextlib
    import io

    bundle_dir = resolve_bundle_dir(bundle_dir)
    _register_bundle_on_sys_path(bundle_dir)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        import classifier as _clf_module  # noqa: E402 -- sys.path setup
        clf = _clf_module.StereotypeClassifier(root=bundle_dir)

    # Pin the booster thread count on both models.
    if num_threads is None or int(num_threads) <= 0:
        try:
            num_threads = max(1, (os.cpu_count() or 4) - 1)
        except Exception:  # noqa: BLE001
            num_threads = 4
    else:
        num_threads = max(1, int(num_threads))
    for _booster_attr in ("baseline", "specialist"):
        _b = getattr(clf, _booster_attr, None)
        if _b is None:
            continue
        # IMPORTANT: do NOT call ``Booster.reset_parameter`` here.  On a
        # Booster loaded from a saved model file, ``reset_parameter`` calls
        # the native ``LGBM_BoosterResetParameter`` on a config that is not
        # fully initialised for a reset, and in the pipeline process (where
        # torch's OpenMP runtime is already loaded) this reproducibly
        # SEGFAULTS the whole application -- a C-level crash a ``try/except``
        # cannot catch (observed on lightgbm with torch 2.5.1 / CUDA 12.4).
        # Setting the Python ``params`` dict, which ``predict`` consults, is
        # a pure-Python no-crash way to request the thread count.
        try:
            if getattr(_b, "params", None) is None:
                _b.params = {}
            _b.params["num_threads"] = int(num_threads)
        except Exception:  # noqa: BLE001
            pass
    return clf, bundle_dir


def bundle_config(bundle_dir: str) -> dict:
    """Load ``config/feature_columns.json`` as a plain dict."""
    import json
    with open(
        os.path.join(bundle_dir, "config", "feature_columns.json"),
        "r", encoding="utf-8",
    ) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------

def _quiet_log(log):
    """The log function to use, decided ONCE per call site.

    FLATMORPHAI_QUIET=1 drops the explanatory lines of Step 4 (RAM planner,
    adaptive pool, per-worker summary) for a clean timing run. The name is then
    bound to a function that does nothing, so no test is left on the calling
    path and the quiet mode cannot itself cost anything. Measured: an emitted
    line costs 0.44 us and a dropped one 0.29 us, for about 200 lines in a whole
    run -- the logs were never a performance question, only a readability one.
    """
    if log is None or os.environ.get("FLATMORPHAI_QUIET", "0") in ("1", "true", "True"):
        return lambda _m: None
    return log

def _load_bundle_extractors(bundle_dir: str) -> dict:
    """Import the bundle's per-family generators.

    Returns a dict with the callables the app needs to build a
    per-video feature DataFrame:

      * ``compute_new_metrics``  -- (mask, gray) -> dict of 18 new
        metrics (sym, hole, twist, temporal, ...).
      * ``lum_features``         -- (mask, gray) -> dict of 5
        luminosity descriptors.
      * ``patch_features``       -- (image_path, mask_path) -> tuple
        of 4 (pdist, psolid, pfrac, pcontrast) patch descriptors.
      * ``frame_curv``           -- (mask_path) -> (solidity, concav)
        per-frame contour descriptors (worm-level aggregation is done
        upstream).
      * ``frame_n_sharp``        -- (mask_path) -> int number of
        sharp contour bends per frame.
    """
    import contextlib
    import io

    _register_bundle_on_sys_path(bundle_dir)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        # ``compute_new_metrics`` is the module.  Import via the
        # package alias.
        from feature_extraction import compute_new_metrics as _cnm  # noqa: E402
        from feature_extraction import (  # noqa: E402
            centerline_luminosity_features as _lum,
            patch_dorsal_features as _patch,
            worm_curvature_features as _curv,
            worm_sharpcurve_features as _nsharp,
        )
    return {
        "sym_feats": _cnm.sym_feats,
        "hole_feats": _cnm.hole_feats,
        "twist_osc_feats": _cnm.twist_osc_feats,
        "load_pair": _cnm.load_pair,
        "sym_cols": _cnm.SYM,
        "hole_cols": _cnm.HOLE,
        "twist_cols": _cnm.TW,
        "temporal_cols": _cnm.TEMP,
        "lum_features": _lum.lum_features,
        "lum_cols": _lum.CLX,
        "patch_features": _patch.patch_features,
        "frame_curv": _curv._frame_curv,
        "frame_n_sharp": _nsharp.frame_n_sharp,
        **_optional_extractors(),
    }


def _load_funcs_from_source(module_name: str, want) -> dict:
    """Return named callables/constants from a bundle module WITHOUT running its
    top-level script body.

    Some bundle feature scripts execute a dataset batch at import time (reading
    the training CSVs), which raises on any machine without that dataset. We
    locate the module file via ``find_spec`` (which does NOT execute it), parse
    it, keep only imports / assignments / function / class definitions, drop
    every other top-level statement (the batch loop, ``if __name__`` block,
    bare calls and prints), then exec that reduced tree. The wanted functions
    and the module constants they rely on are defined in the kept nodes.
    """
    import ast
    import importlib.util
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return {}
    with open(spec.origin, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    tree.body = [n for n in tree.body if isinstance(
        n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
            ast.ClassDef, ast.Assign, ast.AnnAssign))]
    ast.fix_missing_locations(tree)
    ns = {"__name__": "_frozen_%s" % module_name, "__file__": spec.origin}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        exec(compile(tree, spec.origin, "exec"), ns)  # noqa: S102
    return {w: ns[w] for w in want if w in ns}


def _optional_extractors() -> dict:
    """Descriptor families that were previously never computed (so their
    columns came out NaN): the trunk-evolution (bodyevo) family and the
    bilateral-asymmetry (shape-lum) family. Imported best-effort -- a missing
    one simply leaves its columns NaN as before, never breaks the run."""
    out: dict = {}
    try:
        from .fusion_tsm.metrics_lib.body_evolution import body_evolution as _be
        out["body_evolution"] = _be
    except Exception:  # noqa: BLE001
        pass
    # compute_shape_lum_metrics.py runs a dataset batch at IMPORT time
    # (``for sp in ...: run(sp)`` reads the training CSVs), so a plain import
    # raises on any machine without that dataset and the functions never load
    # -- which is why bilat_asym_mean stayed NaN. Load only its function/const
    # definitions, skipping the top-level batch, from the file's own source.
    try:
        _sl = _load_funcs_from_source(
            "compute_shape_lum_metrics", ("bilateral", "brightpeak"))
        if "bilateral" in _sl:
            out["bilateral"] = _sl["bilateral"]
        if "brightpeak" in _sl:
            out["brightpeak"] = _sl["brightpeak"]
    except Exception:  # noqa: BLE001
        pass
    # Expert image families (central_*, pig_localstd, glcm_*, sat_*, valley_*):
    # the assemble-time 'sig' families the per-frame path never computed.
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            from feature_extraction import (  # noqa: E402
                compute_expert_image_features as _eif,
            )
        out["expert_pigment"] = _eif.pigment
        out["expert_saturation"] = _eif.saturation
        out["expert_central"] = _eif.central
        out["expert_valley"] = _eif.valley
    except Exception:  # noqa: BLE001
        pass
    return out


# Motion image (registered prev/cur/next) builder, copied verbatim from the
# bundle's make_motion_dataset.py so the 'motion' / 'motioncl' metric groups
# (rm_*, mot_*, ip_*, bt_*) are fed EXACTLY the image they were trained on:
# B = aligned previous, G = current, R = aligned next, each masked to the worm.
# Built here from the formatted (384 px, mask-applied) grays the app already
# reads, which is the folder make_motion_dataset registers from.
def _motion_pose(gray):
    import cv2
    import numpy as _np
    m = cv2.moments((gray > 8).astype("uint8"), binaryImage=True)
    if m['m00'] < 50:
        return None
    cx = m['m10'] / m['m00']; cy = m['m01'] / m['m00']
    mu20 = m['mu20'] / m['m00']; mu02 = m['mu02'] / m['m00']; mu11 = m['mu11'] / m['m00']
    theta = 0.5 * _np.arctan2(2 * mu11, mu20 - mu02)
    ecc = ((mu20 - mu02) ** 2 + 4 * mu11 ** 2) ** 0.5 / (mu20 + mu02 + 1e-9)
    return cx, cy, float(theta), float(ecc)


def _motion_align_to(src_gray, src_pose, dst_pose, dst_gray):
    import cv2
    import numpy as _np
    H, W = src_gray.shape
    scx, scy, sth, sec = src_pose; dcx, dcy, dth, dec = dst_pose
    angles = [0.0] if min(sec, dec) < 0.15 else [dth - sth, dth - sth + _np.pi]
    cand = []
    for a in angles:
        M = cv2.getRotationMatrix2D((scx, scy), _np.degrees(a), 1.0)
        M[0, 2] += (dcx - scx); M[1, 2] += (dcy - scy)
        cand.append(cv2.warpAffine(src_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0))
    if len(cand) == 1:
        return cand[0]
    dm = dst_gray > 8
    sads = [float(_np.abs((c - dst_gray)[dm]).mean()) if dm.any() else _np.abs(c).mean() for c in cand]
    return cand[int(_np.argmin(sads))]


def _build_motion_image(prev_bgr, cur_bgr, next_bgr):
    """Registered temporal-RGB for one frame, or None when the current frame
    has no usable worm. prev_bgr/next_bgr may be None (video ends): the missing
    side falls back to the current frame, exactly as make_motion_dataset does."""
    import cv2
    import numpy as _np
    if cur_bgr is None:
        return None
    gt = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY).astype(_np.float32) if cur_bgr.ndim == 3 else cur_bgr.astype(_np.float32)
    pt = _motion_pose(gt)

    def prep(nb):
        if nb is None or pt is None:
            return gt
        gn = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY).astype(_np.float32) if nb.ndim == 3 else nb.astype(_np.float32)
        pn = _motion_pose(gn)
        if pn is None:
            return gt
        return _motion_align_to(gn, pn, pt, gt)

    worm = (gt > 8)
    R = prep(prev_bgr); B = prep(next_bgr)
    out = _np.zeros((gt.shape[0], gt.shape[1], 3), _np.uint8)
    out[..., 0] = (_np.clip(R, 0, 255) * worm).astype(_np.uint8)   # B = aligned previous
    out[..., 1] = (_np.clip(gt, 0, 255) * worm).astype(_np.uint8)  # G = current
    out[..., 2] = (_np.clip(B, 0, 255) * worm).astype(_np.uint8)   # R = aligned next
    return out


# ---------------------------------------------------------------------
# Per-video feature CSV writer (LightGBM format)
# ---------------------------------------------------------------------
def compute_lightgbm_features_csv(
    video_dir: str,
    mask_dir_name: str,
    output_csv_path: str,
    bundle_dir: str,
    baseline_cols: Sequence[str],
    specialist_cols: Sequence[str],
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    max_workers: Optional[int] = None,
    formatted_dir: Optional[str] = None,
    _return_computer: bool = False,
):
    """Compute the per-frame LightGBM feature table for one video.

    The output CSV carries the columns LightGBM's ``StereotypeClassifier``
    needs to predict, keyed by ``image`` (basename) so
    ``feature_columns.json`` order can be applied downstream.

    Columns are gathered from three sources:

      1. Base mask3 metrics -- reuses ``compute_required_metrics``
         from ``fusion_tsm.metrics_tsm``, filtered to the LightGBM
         ``baseline_feature_order`` list (drops ~50 obsolete metrics
         the old ConvNeXt-V2 pipeline used but LightGBM does not).
      2. New families (sym, hole, twist, temporal, luminosity, patch)
         computed by the bundled LightGBM extractors:
         ``compute_new_metrics``, ``centerline_luminosity_features``,
         ``patch_dorsal_features``.
      3. Worm-level aggregates (``curv``, ``nsharp``) computed once
         across every frame of the video then replicated to each
         row so the classifier's per-frame call sees the same worm-
         level value on every row of that worm.

    Skip-existing semantics: when ``output_csv_path`` already exists
    and ``overwrite`` is False, every row whose ``image`` is already
    on disk is reused as-is.
    """
    _log = _quiet_log(log)
    import contextlib
    import io
    import pandas as pd
    import cv2

    # GIL-free, bit-identical MCP_Geometric (must run BEFORE the bundle / v4j
    # programs are imported or exec'd, so they bind the patched class).
    _install_mcp_numba(_log)

    # Register bundle so its extractors are importable, then pull
    # the callables we need.
    ext = _load_bundle_extractors(bundle_dir)
    # Exact numba twin of the round() ray walks (tm_ / be_ / bilat_ families):
    # FLATMORPHAI_RAY_NUMBA=0 disables. Self-validates; any doubt -> Python.
    _ray_patch(ext, _log)
    # v4j-only descriptor families, computed by reusing the v4j chain's own
    # per-frame functions (see v4j_stereotype_descriptors). Optional: if the
    # module or the chain folder is absent, those columns are simply left out.
    try:
        from .v4j_stereotype_descriptors import compute_v4j_extra as _v4j_extra
    except Exception:  # noqa: BLE001
        _v4j_extra = None

    images_dir = os.path.join(video_dir, "images")
    masks_dir = os.path.join(video_dir, mask_dir_name)
    if not os.path.isdir(images_dir):
        _log(
            f"  WARNING: no images/ sub-folder under {video_dir}; "
            "skipping LightGBM feature CSV."
        )
        return 0
    if not os.path.isdir(masks_dir):
        _log(
            f"  WARNING: no '{mask_dir_name}/' under {video_dir}; "
            "skipping LightGBM feature CSV."
        )
        return 0

    # ---- 1. Enumerate frames in acquisition order ----------------
    from .fusion_tsm.nomenclature import app_second
    frames: List[Tuple[float, str]] = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(".png"):
            continue
        sec = app_second(fn)
        if sec is None:
            continue
        frames.append((float(sec), fn))
    frames.sort(key=lambda t: (t[0], t[1]))
    if not frames:
        _log(
            f"  WARNING: no parseable frames in {images_dir}; "
            "LightGBM feature CSV will be empty."
        )
        return 0

    n = len(frames)
    if progress_cb is not None:
        try:
            progress_cb(0, n, os.path.basename(video_dir))
        except Exception:  # noqa: BLE001
            pass

    # ---- 2. Skip-existing partial-update -------------------------
    existing_rows: dict = {}
    if (not overwrite) and os.path.isfile(output_csv_path):
        try:
            df_prev = pd.read_csv(
                output_csv_path, sep=";", encoding="utf-8",
            )
            for _, row in df_prev.iterrows():
                k = str(row.get("image", ""))
                if k:
                    existing_rows[k] = row.to_dict()
            _log(
                f"  LightGBM feature CSV partial-update: "
                f"{len(existing_rows)} row(s) already on disk."
            )
        except Exception:  # noqa: BLE001
            existing_rows = {}

    # ---- 3. Base mask3 metrics via the app's TSM helper ----------
    # We reuse the app's compute_required_metrics logic (which
    # produces the 308-metric table) and downstream filter to the
    # LightGBM baseline_feature_order list.  Unnecessary metrics
    # are dropped at write time -- the compute step runs the same
    # per-frame body so a second global sweep is not needed.
    from .fusion_tsm.metrics_tsm import compute_required_metrics

    # Match the reference golden descriptor pipeline: v3c replaces skeletonize
    # with the single longest medial axis (axe_raster) in the axis-based metrics
    # and the worm centroid with centre_corps for the central expert. Without
    # this the app's branchy skeleton makes ax_straightness / ax_helix_* / ts_* /
    # gw_* / twist_* / central_* drift from the trained features. Best-effort;
    # any patch that doesn't match its source is skipped, never raised.
    try:
        from . import v3c_descriptor_patches as _v3c
        _v3c.apply_patches(ext, log=_log)
    except Exception as _pe:  # noqa: BLE001
        _log("  v3c descriptor patches unavailable (%s)" % _pe)

    # Golden texture chain (14 texture descriptors via the b&w/CLAHE chain +
    # size residual) and the shape/twist size residuals.
    try:
        from .texture_v3c import TextureV3c as _TextureV3c
        _tex_v3c = _TextureV3c()
    except Exception as _te:  # noqa: BLE001
        _tex_v3c = None
        _log("  v3c texture chain unavailable (%s)" % _te)

    # Decode ON DEMAND, without a per-video cache. The previous version cached
    # every frame's decoded mask / gray / color arrays for the whole video,
    # which on full-resolution frames held the entire clip in RAM at once and
    # could OOM-kill the process (observed as "Killed" on low-memory hosts).
    # Decoding inside each frame's worker keeps at most ``max_workers`` frames
    # resident simultaneously instead of all of them, with no change to the
    # computed feature values. The small per-frame re-decode cost is hidden by
    # the thread pool.
    def _mask(name: str):
        p = os.path.join(masks_dir, name)
        return (
            cv2.imdecode(_readbytes(p), cv2.IMREAD_GRAYSCALE)
            if os.path.isfile(p) else None
        )

    def _gray(name: str):
        p = os.path.join(images_dir, name)
        return (
            cv2.imdecode(_readbytes(p), cv2.IMREAD_GRAYSCALE)
            if os.path.isfile(p) else None
        )

    def _color(name: str):
        p = os.path.join(images_dir, name)
        return (
            cv2.imdecode(_readbytes(p), cv2.IMREAD_COLOR)
            if os.path.isfile(p) else None
        )

    def _formatted(name: str):
        # The formatted-image metric groups (shape, tunif, centerline, texture,
        # curated) are meant to run on the SMALL masked crop Step 3 produced
        # (e.g. 384x384), not the full-resolution raw frame. Running them on a
        # 1200x1600 raw frame was ~20-40 s PER GROUP PER FRAME and also wrong
        # (the background was never masked out). Read the formatted crop by the
        # same file name; fall back to the raw colour only when it is missing.
        if formatted_dir:
            fp = os.path.join(formatted_dir, name)
            if os.path.isfile(fp):
                im = cv2.imdecode(_readbytes(fp), cv2.IMREAD_COLOR)
                if im is not None:
                    return im
        return _color(name)

    def _formatted_gray3(name: str):
        # The reference reads the formatted crop with IMREAD_GRAYSCALE for the
        # shape / basic / tunif / centerline / terminal groups, which differs by
        # +/-1 from cvtColor(IMREAD_COLOR) and shifts the curvature descriptors.
        # Read it the same way, then replicate to 3 channels so the downstream
        # cvtColor(BGR2GRAY) returns exactly those IMREAD_GRAYSCALE values
        # (R=G=B -> luma = the grey) and colour_metrics still receives 3 channels.
        g = None
        if formatted_dir:
            fp = os.path.join(formatted_dir, name)
            if os.path.isfile(fp):
                g = cv2.imdecode(_readbytes(fp), cv2.IMREAD_GRAYSCALE)
        if g is None:
            c = _color(name)
            if c is None:
                return None
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # Resolve the worker count ONCE: an explicit value from the
    # Settings page wins, 0/None means "auto" = cpu_count() - 1 so
    # one core stays free for the Qt event loop.  cv2, numpy and
    # scikit-image release the GIL for the bulk of their work, so a
    # thread pool scales nearly linearly here.  Reference: Beazley,
    # D. M., "Python Essential Reference", 4th ed., Addison-Wesley,
    # 2009, ISBN 978-0-672-32978-4, sec. 20.6.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        _cpu = os.cpu_count() or 4
    except Exception:  # noqa: BLE001
        _cpu = 4
    if max_workers is None or int(max_workers) <= 0:
        max_workers = max(1, _cpu - 1)
    else:
        max_workers = max(1, int(max_workers))

    # No pre-warm pass: decoding is done on demand inside each frame worker
    # (see _mask / _gray / _color above) so the whole clip is never resident
    # in RAM at once.

    # ---- 4. Worm-level aggregates (curv + nsharp) ---------------
    # Both read one mask per frame and aggregate over the whole
    # video.  The two bundle helpers take a mask PATH (they decode
    # internally), so this pass is independent of the caches above
    # -- but it is embarrassingly parallel, and running it serially
    # was the single slowest part of the feature stage.
    frame_curv_fn = ext["frame_curv"]
    frame_nsharp_fn = ext["frame_n_sharp"]

    def _worm_frame_stats(name: str):
        """Return (solidity, concavity, n_sharp) for one frame."""
        mp = os.path.join(masks_dir, name)
        if not os.path.isfile(mp):
            return None
        sol = con = nsh = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = frame_curv_fn(mp)
            if r is not None:
                sol, con = float(r[0]), float(r[1])
        except Exception:  # noqa: BLE001
            pass
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                v = frame_nsharp_fn(mp)
            if v is not None:
                nsh = float(v)
        except Exception:  # noqa: BLE001
            pass
        return sol, con, nsh

    sols: List[float] = []
    concs: List[float] = []
    nsharps: List[float] = []
    with ThreadPoolExecutor(max_workers=max_workers) as _worm_pool:
        for res in _worm_pool.map(
            _worm_frame_stats, [nm for _s, nm in frames],
        ):
            if res is None:
                continue
            sol, con, nsh = res
            if sol is not None and con is not None:
                sols.append(sol)
                concs.append(con)
            if nsh is not None:
                nsharps.append(nsh)
    if len(sols) >= 2:
        _worm_concav_max = float(max(concs)) if concs else float("nan")
        _worm_concav_range = (
            float(max(concs) - min(concs)) if concs else float("nan")
        )
        _worm_sol_min = float(min(sols))
    else:
        _worm_concav_max = float("nan")
        _worm_concav_range = float("nan")
        _worm_sol_min = float("nan")
    if len(nsharps) >= 2:
        import numpy as _np
        _worm_nsharp_max = float(max(nsharps))
        _worm_nsharp_med = float(_np.median(nsharps))
    else:
        _worm_nsharp_max = float("nan")
        _worm_nsharp_med = float("nan")

    # ---- 5. Per-frame feature rows ------------------------------
    # ``ThreadPoolExecutor`` / ``as_completed`` and ``max_workers``
    # were already resolved above, before the cache-warming pass.
    import numpy as _np

    keep_cols_set = set(baseline_cols) | set(specialist_cols)
    keep_cols_set.add("image")

    # v4j axis/temporal descriptor chain: a single V4jChain, configured once for
    # this video, reused by every frame worker (its per-frame ``calcule`` calls
    # only read images, so the shared namespaces are safe across threads). Needs
    # the 384 formatted crop folder; without it only the self-contained wn_/tt_
    # families are produced. Each worker passes its frame name so the chain picks
    # that frame's T-1/T+1 neighbours in acquisition order.
    _v4j_chain = None
    if _v4j_extra is not None and formatted_dir and os.path.isdir(formatted_dir):
        try:
            from .v4j_stereotype_descriptors import V4jChain as _V4jChain
            _v4j_chain = _V4jChain()
            _v4j_chain.set_video(
                images_dir, masks_dir, formatted_dir,
                [nm for _s, nm in frames])
            if not getattr(_v4j_chain, "_ok", False):
                _v4j_chain = None
        except Exception:  # noqa: BLE001
            _v4j_chain = None

    # ---- Old-model descriptors are excluded from the code, not just dropped ---
    # The classifier now uses the v4j feature set. The base-metric groups and
    # expert / new-metric families below produce ONLY columns the OLD model used;
    # none appear in the v4j model (``keep_cols_set``), so they are never computed
    # -- the group list passed to compute_required_metrics omits them and the
    # per-frame path no longer calls those ext families at all. This is a static
    # replacement for the current model, not a runtime check.
    _V4J_SKIP_GROUPS = ("neckext", "pigment")
    # NOTE: texture / curated are computed here (full base groups) AND then
    # overridden by the v3c b&w chain (recompute_formatted). Skipping them as a
    # speed optimisation was tried but reverted -- correctness first: the base
    # groups stay exactly as the pre-optimisation pipeline computed them.
    _fallback_groups: List[str] = []   # no skip, so no fallback recompute needed
    _base_groups = None            # None => all groups (compute_required default)
    try:
        from .fusion_tsm.metrics_tsm import required_groups as _required_groups
        _base_groups = [g for g in _required_groups()
                        if g not in _V4J_SKIP_GROUPS]
    except Exception:  # noqa: BLE001
        _base_groups = None
    # Skip the base texture/curated groups that recompute_formatted overrides
    # (they were computed twice and the first result thrown away), with an
    # exact per-column fallback whenever the b&w chain does not return them.
    # ON by default: validated bit-identical on the full per-frame compute
    # (1960 values) incl. simulated partial/total failures of the b&w chain.
    # FLATMORPHAI_SKIP_DUP_GROUPS=none to compute them twice as before.
    _dup_env = os.environ.get("FLATMORPHAI_SKIP_DUP_GROUPS", "texture,curated")
    _dup_groups = tuple(g.strip() for g in _dup_env.split(",")
                        if g.strip() and g.strip().lower() != "none")
    if _dup_groups and _base_groups is not None and _tex_v3c is not None:
        _base_groups = [g for g in _base_groups if g not in _dup_groups]
        _fallback_groups = list(_dup_groups)
    _dup_state = {"cols": None}   # model columns of the skipped groups (learned)

    # Diagnostics: why a frame ends up with NaN descriptors. Each entry is
    # (frame, stage, error); summarised once at the end so a silent failure
    # stops being silent. Thread-safe because the workers run in a pool.
    import threading as _threading
    _diag_lock = _threading.Lock()
    # "expert" was missing: any exception in an expert family then raised a
    # KeyError inside the error handler and the WHOLE frame was lost.
    _diag = {"no_mask": [], "base": [], "ext": [], "lum": [], "expert": [],
             "patch": []}

    def _note(kind, frame, err=""):
        # setdefault: a diagnostic must never be what kills a frame.
        with _diag_lock:
            _diag.setdefault(kind, []).append((frame, str(err)[:120]))

    def _compute_one_frame(spec: Tuple[int, float, str]):
        idx, sec, name = spec
        if name in existing_rows:
            return idx, existing_rows[name]
        if _FRAME_MEMO_PASSES[0] < 2:
            # exact per-image memo of the shared skeleton / axis / shape helpers;
            # pass 2 (next frame) catches the modules imported lazily by frame 1
            _FRAME_MEMO_PASSES[0] += 1
            try:
                from .memo_cache import patch_frame_memo as _pfm
                _pfm(_log if _FRAME_MEMO_PASSES[0] == 2 else None)
            except Exception:  # noqa: BLE001
                pass
            try:   # skeleton / EDT / distanceTransform on the object's bounding box
                from .crop_exact import install as _cei
                _cei(_log if _FRAME_MEMO_PASSES[0] == 1 else None)
            except Exception:  # noqa: BLE001
                pass
        mask = _mask(name)
        gray = _gray(name)
        if mask is None:
            # The dominant NaN source: no mask file for this frame, so every
            # mask-based descriptor below is undefined. Almost always a
            # segmentation gap or a name mismatch between images/ and masks/.
            _note("no_mask", name)
        row: dict = {"image": name, "second": float(sec)}
        # NOTE: the previous version cropped the mask to the worm bounding box
        # to halve the single-frame cost. That is incompatible with the
        # three-frame window below -- the neighbours must share the full-frame
        # coordinate system for the motion/temporal groups to register them --
        # so the full-resolution mask is used. The single-frame groups are
        # unchanged by this (they are translation-invariant).
        # (a) base metrics. The three-frame (T-1, T0, T+1) and temporal groups
        # need the REAL neighbouring frames, not the current one repeated three
        # times -- that repetition is why every motion / temporal / flow
        # descriptor (ax_flow_*, ax_helix_rot, body_lat_*, ...) came out NaN.
        # The neighbours are the frames just before and after this one in the
        # second-ordered list, clamped at the two ends of the video.
        _prev_name = frames[idx - 1][1] if idx > 0 else name
        _next_name = frames[idx + 1][1] if idx + 1 < len(frames) else name
        _mp, _mn = _mask(_prev_name), _mask(_next_name)
        _gp, _gn = _gray(_prev_name), _gray(_next_name)
        _cp, _cc, _cn = _color(_prev_name), _color(name), _color(_next_name)
        # Registered motion image (motion / motioncl groups: rm_*, mot_*, ip_*,
        # bt_*), built from the formatted 384 px grays like make_motion_dataset.
        _motion = _build_motion_image(
            _formatted(_prev_name), _formatted(name), _formatted(_next_name))
        # +/-5 and +/-2 mask windows for the wide-horizon (hxw_*) and C-shape
        # persistence (sper_*) groups, in temporal order, missing frames dropped.
        _wide = [_mask(frames[j][1])
                 for j in range(max(0, idx - 5), min(len(frames), idx + 6))]
        _wide = [w for w in _wide if w is not None]
        _shape = [_mask(frames[j][1])
                  for j in range(max(0, idx - 2), min(len(frames), idx + 3))]
        _shape = [w for w in _shape if w is not None]
        try:
            feat_base = compute_required_metrics(
                masks3=(_mp, mask, _mn),
                grays3=(_gp, gray, _gn),
                formatted_t0_bgr=_formatted_gray3(name),
                raw_color3=((_cp, _cc, _cn)
                            if (_cp is not None and _cc is not None
                                and _cn is not None) else None),
                motion_image=_motion,
                wide_masks=(_wide or None),
                shape_masks=(_shape or None),
                groups=_base_groups,
            )
        except Exception as _e:  # noqa: BLE001
            feat_base = {}
            _note("base", name, _e)
        # (a') trunk-evolution (bodyevo) family -- a three-frame window the base
        # writer does not cover; computed directly when available.
        if "body_evolution" in ext \
                and mask is not None and _mp is not None and _mn is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    feat_base.update(ext["body_evolution"](_mp, mask, _mn))
            except Exception as _e:  # noqa: BLE001
                _note("base", name, _e)
        # (a'') bilateral-asymmetry + bright-peak (shape-lum) family. The bundle
        # crops the mask and gray to the worm bounding box (+8 px margin) before
        # the measure (compute_shape_lum_metrics.features), so do the same here
        # for values identical to the training table.
        if "bilateral" in ext and mask is not None and gray is not None:
            try:
                _m = _np.asarray(mask); _g = _np.asarray(gray)
                if _g.shape != _m.shape:
                    _g = cv2.resize(_g, (_m.shape[1], _m.shape[0]))
                _ys, _xs = _np.where(_m > 0)
                if len(_ys) >= 50:
                    _mr = 8
                    _y0, _y1 = max(0, _ys.min() - _mr), _ys.max() + _mr
                    _x0, _x1 = max(0, _xs.min() - _mr), _xs.max() + _mr
                    _mc = _m[_y0:_y1, _x0:_x1]
                    with contextlib.redirect_stdout(io.StringIO()):
                        feat_base.update(ext["bilateral"](_mc) or {})
            except Exception as _e:  # noqa: BLE001
                _note("base", name, _e)
        for col, val in feat_base.items():
            if col in keep_cols_set:
                row[col] = float(val) if val is not None else _np.nan
        # (b) new_metrics families
        if mask is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    row.update(ext["sym_feats"](mask, gray))
                    row.update(ext["twist_osc_feats"](mask, gray))
            except Exception as _e:  # noqa: BLE001
                _note("ext", name, _e)
            # (c') expert image families the v4j model reads (central_*,
            # pig_localstd, glcm_*). They read the RAW frame + mask by path.
            # expert_saturation (sat_*) and expert_valley (valley_*) produce only
            # old-model columns and are intentionally not computed.
            _ip = os.path.join(images_dir, name)
            _mpth = os.path.join(masks_dir, name)
            for _ek in ("expert_pigment", "expert_central"):
                if _ek in ext:
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            _ev = ext[_ek](_ip, _mpth)
                        if _ev:
                            row.update(_ev)
                    except Exception as _e:  # noqa: BLE001
                        _note("expert", name, _e)
        # (d) patch (dorsal dark patch) requires the RGB image AND
        # the mask path (patch_features re-reads via cv2.imread).
        if "patch_features" in ext:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pd_, ps_, pf_, pc_ = ext["patch_features"](
                        os.path.join(images_dir, name),
                        os.path.join(masks_dir, name),
                    )
                row["pdist"] = pd_
                row["psolid"] = ps_
                row["pfrac"] = pf_
                row["pcontrast"] = pc_
            except Exception as _e:  # noqa: BLE001
                _note("patch", name, _e)
                row.setdefault("pdist", _np.nan)
                row.setdefault("psolid", _np.nan)
                row.setdefault("pfrac", _np.nan)
                row.setdefault("pcontrast", _np.nan)
        # (e) worm-level aggregates -- same value for every frame
        row["worm_concav_max"] = _worm_concav_max
        row["worm_concav_range"] = _worm_concav_range
        row["worm_sol_min"] = _worm_sol_min
        row["worm_nsharp_max"] = _worm_nsharp_max
        row["worm_nsharp_med"] = _worm_nsharp_med
        # (e2) golden texture chain + size residuals. The 14 texture descriptors
        # (tgb_*, tb_gabor_*, tw_*, tk_bhat5_*, ti_kurt, te_lap_var) are computed
        # by v3c on a CLEANED mask + b&w/CLAHE 384 image (not the colour one the
        # classifier uses), with a size residual -- verified to reproduce the
        # golden exactly. The shape (sh_m*) and tm_twist_max descriptors get the
        # original-coefs size residual on their normal-pipeline value.
        _tf = {}
        _tf_called = False
        if _tex_v3c is not None and mask is not None and _cc is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _tf = _tex_v3c.recompute_formatted(_cc, mask)
                for col, val in _tf.items():
                    if col in keep_cols_set:
                        row[col] = float(val)
                _tf_called = True
            except Exception as _e:  # noqa: BLE001
                _note("ext", name, _e)
                _tf = {}
        # Exact fallback for the base groups skipped because recompute_formatted
        # overrides them (texture / curated). Column by column: any model column
        # those groups produce that the b&w chain did NOT return on this frame
        # (total OR partial failure) gets the base (colour-image) value, exactly
        # what the pre-optimisation pipeline kept. The column set is learned on
        # the first frame (one extra base computation per video). Runs BEFORE
        # the size residuals, i.e. in the original order.
        if _fallback_groups and mask is not None:
            _need = _dup_state["cols"]
            if (not _tf) or (_need is None) or any(c not in _tf for c in _need):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _fb = compute_required_metrics(
                            masks3=(_mp, mask, _mn),
                            grays3=(_gp, gray, _gn),
                            formatted_t0_bgr=_formatted_gray3(name),
                            raw_color3=((_cp, _cc, _cn)
                                        if (_cp is not None and _cc is not None
                                            and _cn is not None) else None),
                            motion_image=_motion,
                            wide_masks=(_wide or None),
                            shape_masks=(_shape or None),
                            groups=_fallback_groups,
                        )
                    if _dup_state["cols"] is None:
                        _dup_state["cols"] = {c for c in _fb if c in keep_cols_set}
                    for col, val in _fb.items():
                        if col in keep_cols_set and col not in _tf:
                            row[col] = float(val) if val is not None else _np.nan
                except Exception as _e:  # noqa: BLE001
                    _note("base", name, _e)
        if _tf_called:
            try:
                _tex_v3c.shape_twist_residuals(row, mask)
            except Exception as _e:  # noqa: BLE001
                _note("ext", name, _e)
        # (f) v4j-only descriptor families (wn_, and more as wired): computed by
        # reusing the v4j chain's own per-frame functions on the 384 formatted
        # image, so they match the chain by construction. Only the ones the
        # current model reads survive the keep filter above.
        if _v4j_extra is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _ex = _v4j_extra(formatted_bgr=_formatted(name),
                                     motion_bgr=_motion,
                                     frame_name=name, chain=_v4j_chain)
                for col, val in (_ex or {}).items():
                    if col in keep_cols_set:
                        row[col] = float(val) if val is not None else _np.nan
            except Exception as _e:  # noqa: BLE001
                _note("v4j", name, _e)
        return idx, row

    rows: List[Optional[dict]] = [None] * n
    work = [(i, sec, name) for i, (sec, name) in enumerate(frames)]

    def _finalize_rows(rows):
        # Per-cause NaN summary: tells the operator exactly why descriptors are
        # missing (which the generic "left as NaN" count below cannot). A gap in
        # segmentation (no_mask) is the usual culprit and is actionable; a
        # family raising on every frame points at a code or data problem.
        _total_f = len(work)
        for _kind, _label in (("no_mask", "no mask file (segmentation gap)"),
                              ("base", "base metric suite failed"),
                              ("ext", "shape/hole/twist family failed"),
                              ("lum", "luminosity family failed"),
                              ("expert", "expert image family failed"),
                              ("patch", "dorsal-patch family failed")):
            _items = _diag.get(_kind) or []
            if _items:
                _ex = ", ".join(sorted({f for f, _ in _items})[:5])
                _err = next((e for _, e in _items if e), "")
                _log("  NaN cause: %d/%d frame(s) - %s (e.g. %s)%s"
                     % (len(_items), _total_f, _label, _ex,
                        (" | " + _err) if _err else ""))

        valid_rows = [r for r in rows if r is not None]
        if not valid_rows:
            _log("  LightGBM feature CSV: no rows produced "
                 "(every frame failed; see the NaN causes above).")
            return 0

        # Assemble the DataFrame -- keep every column the models care about;
        # missing columns are left as NaN (LightGBM handles them natively).
        import pandas as pd
        df = pd.DataFrame(valid_rows)
        _needed = [c for c in keep_cols_set if c != "image"]
        _missing = sorted(c for c in _needed if c not in df.columns or df[c].isna().all())
        if _missing:
            _log(
                f"  LightGBM feature CSV: {len(_needed) - len(_missing)}/{len(_needed)} "
                f"descriptors computed; {len(_missing)} left as NaN "
                f"(first ones: {', '.join(_missing[:8])})"
            )
        df.to_csv(output_csv_path, sep=";", index=False, encoding="utf-8")
        _log(
            f"  LightGBM feature CSV: wrote {len(df)} row(s) to "
            f"{os.path.basename(output_csv_path)}"
        )
        return len(df)

    # Expose the per-frame computer so a caller can run EVERY video's frames in
    # ONE shared thread pool (see compute_lightgbm_features_for_videos) instead
    # of a fresh pool per video: one thread per image, images spread across all
    # videos. The caller fills ``rows`` by frame index, then calls ``finalize``.
    if _return_computer:
        return {
            "video_dir": video_dir, "output_csv_path": output_csv_path,
            "work": work, "n": n, "compute_one_frame": _compute_one_frame,
            "finalize": _finalize_rows, "log": _log,
        }

    done = 0
    if work:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_compute_one_frame, w) for w in work]
            for fut in as_completed(futures):
                try:
                    idx, row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    _log(
                        "  LightGBM feature worker failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                rows[idx] = row
                done += 1
                if progress_cb is not None and (done % 5 == 0 or done == n):
                    try:
                        progress_cb(done, n, os.path.basename(video_dir))
                    except Exception:  # noqa: BLE001
                        pass
    try:
        from .geodesic_gpu import status as _geod_status
        _log("  " + _geod_status())
    except Exception:  # noqa: BLE001
        pass
    return _finalize_rows(rows)


_FRAME_MEMO_PASSES = [0]


def _ray_patch(ext=None, _log=None):
    try:
        if os.environ.get("FLATMORPHAI_RAY_NUMBA", "1") not in ("1", "true", "True"):
            return
        from . import ray_numba as _RN
        _RN.patch_modules(_log)
        try:
            from . import vec_twins as _VT
            _VT.patch_modules(_log)
        except Exception:  # noqa: BLE001
            pass
        if ext is not None and callable(ext.get("bilateral")):
            ext["bilateral"] = _RN.patch_bilateral(ext["bilateral"])
    except Exception:  # noqa: BLE001
        pass


def _install_mcp_numba(log=None) -> bool:
    """Swap skimage's MCP_Geometric for the GIL-free Numba twin (mcp_numba).

    skimage's Cython MCP holds the GIL, so the descriptor threads could not
    parallelise the dominant geodesics of the v4j chain. The Numba version is
    bit-identical (self-validated against skimage at install: same cost field,
    same traceback; validated on the user's masks: 560 v4j descriptors, 0
    difference) and releases the GIL. ON by default, opt out with
    FLATMORPHAI_MCP_NUMBA=0; without numba, or on any validation mismatch, it
    stays on skimage. Idempotent (logs once per process)."""
    # Bounded, copy-on-hit memo of the pure repeated calls (skeletonize,
    # Hessian, Farneback, imdecode): validated bit-identical on the full
    # per-frame compute (1960 values, 1 and 2 threads) and in the app (2 full
    # runs, identical classes). ON by default; FLATMORPHAI_DESC_MEMO=0 to disable.
    if os.environ.get("FLATMORPHAI_DESC_MEMO", "1") in ("1", "true", "True"):
        try:
            from .memo_cache import install as _memo_install
            _memo_install(log)
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log("  descriptor memo: OFF (%s: %s)" % (type(exc).__name__, exc))
    if os.environ.get("FLATMORPHAI_MCP_NUMBA", "1") not in ("1", "true", "True"):
        return False
    try:
        from .mcp_numba import install as _install
        return bool(_install(log))
    except Exception as exc:  # noqa: BLE001 - never block the run on this
        if log is not None:
            log("  MCP numba: OFF (%s: %s)" % (type(exc).__name__, exc))
        return False


# ---------------------------------------------------------------------
# Optional PROCESS-pool path for the descriptor stage
# ---------------------------------------------------------------------
# The v4j golden chain is pure-Python and GIL-bound (measured: 1->7 threads
# gives ~x1.05, no real parallelism), so the thread pool can only overlap the
# numpy/cv2 families. True parallelism needs separate PROCESSES. This is opt-in
# via FLATMORPHAI_LGBM_PROCESSES=N (N>=2); the default stays the thread pool so
# behaviour is unchanged unless asked. Processes are isolated memory, so the
# computed values are BIT-IDENTICAL to the serial path (no shared state to drift,
# unlike threads). 'spawn' is used so a fresh interpreter initialises CUDA
# cleanly (the Gabor bank) without the fork+CUDA segfault that forced DataLoader
# num_workers to 0. Each worker rebuilds the per-video computers once from the
# (picklable) specs; frames are distributed across processes independently of
# which video they belong to, as requested.
_PROC_SPECS = None          # per-process: the specs list (set by the initializer)
_PROC_CTXS: dict = {}       # per-process: spec_index -> computer dict (lazy)


def _proc_init(specs):
    """Process-pool initializer (runs once per worker process)."""
    global _PROC_SPECS, _PROC_CTXS
    _PROC_SPECS = dict(enumerate(specs or []))
    _PROC_CTXS = {}
    # Keep each process single-threaded in its native libs so N processes do not
    # oversubscribe the cores (the chain is GIL-bound Python; BLAS threads add
    # contention, not speed, here).
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import sys as _sys_t
        if "torch" in _sys_t.modules:     # never import torch just for this
            _sys_t.modules["torch"].set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _proc_ctx(spec_index):
    """Build (once) and return this process's computer for one video spec."""
    c = _PROC_CTXS.get(spec_index)
    if c is None:
        _sp = dict(_PROC_SPECS[spec_index])
        _sp.setdefault("max_workers", 1)
        c = compute_lightgbm_features_csv(_return_computer=True, log=None, **_sp)
        _PROC_CTXS[spec_index] = c
    return c


def _proc_worker(task):
    """Compute ONE frame in a worker process. task = (ctx_index, spec_index, w).
    Returns (ctx_index, frame_index, row_or_None)."""
    ci, si, w = task
    try:
        c = _proc_ctx(si)
        if not isinstance(c, dict):
            return (ci, None, None)
        idx, row = c["compute_one_frame"](w)
        return (ci, idx, row)
    except Exception:  # noqa: BLE001
        return (ci, None, None)


def _proc_chunk(task):
    """Compute a CONTIGUOUS block of frames (possibly spanning two videos) in a
    worker process. ``task`` = [(spec_index, [frame_index, ...]), ...]. Each
    video's computer is built once per process (_proc_ctx) and reused for every
    later block of the same video. Returns {"rows": [(si, idx, row)], "skip":
    [si...], "sec": elapsed}."""
    import time as _t
    t0 = _t.time()
    c0 = _t.process_time()
    rows, skip = [], []
    for _it in task:            # a streamed video: its spec travels with the block
        if len(_it) > 2 and _it[0] not in _PROC_SPECS:
            _PROC_SPECS[_it[0]] = _it[2]
    task = [(_it[0], _it[1]) for _it in task]
    try:
        _first = min(si for si, _i in task)
        for _old in [k for k in list(_PROC_CTXS) if k < _first]:
            _PROC_CTXS.pop(_old, None)
    except Exception:  # noqa: BLE001
        pass
    for si, idxs in task:
        try:
            c = _proc_ctx(si)
        except Exception:  # noqa: BLE001
            c = None
        if not isinstance(c, dict):
            skip.append(si)          # nothing to compute (cached / no frames)
            continue
        work = c["work"]
        for i in idxs:
            if 0 <= i < len(work):
                try:
                    idx, row = c["compute_one_frame"](work[i])
                    rows.append((si, idx, row))
                except Exception:  # noqa: BLE001
                    rows.append((si, i, None))
    summ = ""
    try:
        import sys as _s
        _mc = _s.modules.get("flatmorphai.notebook_api.memo_cache")
        _mn = _s.modules.get("flatmorphai.notebook_api.mcp_numba")
        parts = []
        if _mc is not None and _mc._installed:
            parts.append("memo " + _mc.summary())
        _gg = _s.modules.get("gabor_gpu")
        _gst = getattr(_gg, "GABOR_GPU_STATE", None) if _gg is not None else None
        if _gst:
            parts.append("gabor GPU%s %s (%d calls, validation error %.1e%s)" % (
                " (server)" if _gst.get("server") else "",
                "active" if _gst.get("ok") else ("CPU FALLBACK" if _gst.get("ok") is False else "not requested"),
                _gst.get("calls", 0), _gst.get("err", 0.0),
                (", " + _gst["error"]) if _gst.get("error") else ""))
        if _mn is not None and _mn._state.get("validated"):
            _pst = getattr(_mn, "PAD_STATE", None) or {}
            parts.append("MCP numba %d accelerated/%d delegated%s" % (
                _mn.STATS["numba"], _mn.STATS["delegated"],
                (", border Dijkstra %s" % ("active" if _pst.get("ok") else
                                            ("FALLBACK (%s)" % _pst.get("error")) if _pst.get("ok") is False
                                            else "not validated")) if _pst else ""))
        try:
            _gsm = _s.modules.get("flatmorphai.notebook_api.gpu_server")
            _gss = _gsm.shm_summary() if _gsm is not None else ""
            if _gss:
                parts.append(_gss)
        except Exception:  # noqa: BLE001
            pass
        summ = " | ".join(parts)
    except Exception:  # noqa: BLE001
        pass
    pss = _pss_mb()
    # Keep the heap for the next block (re-faulting freed pages costs time);
    # give it back to the OS only when RAM is getting tight.
    try:
        _av = _meminfo_mb().get("MemAvailable", 1e9)
    except Exception:  # noqa: BLE001
        _av = 1e9
    if _av < float(os.environ.get("FLATMORPHAI_TRIM_BELOW_MB", "1500") or 1500):
        _trim_heap()
    return {"rows": rows, "skip": skip, "sec": _t.time() - t0,
            "pid": os.getpid(), "summary": summ, "rss_mb": pss or _peak_rss_mb(),
            "pss_after_trim": _pss_mb(), "cpu": _t.process_time() - c0}


def _proc_video(spec):
    """Compute+write ONE whole video's feature CSV in a worker process.

    VIDEO-LEVEL distribution: each process builds exactly ONE video's chain and
    writes its CSV, so there is NO rebuild duplication (the frame-level version
    rebuilt every video in every process, and that overhead swamped the win).
    'spawn' + single-threaded native libs so N processes do not oversubscribe.
    Bit-identical to the serial path (isolated process). Returns (csv_path, n)."""
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import sys as _sys_t
        if "torch" in _sys_t.modules:     # never import torch just for this
            _sys_t.modules["torch"].set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    import time as _t
    t0 = _t.time()
    try:
        n = compute_lightgbm_features_csv(log=None, max_workers=1, **spec)
    except Exception as exc:  # noqa: BLE001
        n = "ERR:%s: %s" % (type(exc).__name__, exc)
    extra = ""
    try:
        import sys as _s
        _mc = _s.modules.get("flatmorphai.notebook_api.memo_cache")
        if _mc is not None and _mc._installed:
            extra = _mc.summary()
    except Exception:  # noqa: BLE001
        pass
    return (spec.get("output_csv_path"), n, _t.time() - t0, extra)


_RAM_CALIB = os.path.join(os.path.expanduser("~"), ".flatmorphai", "step4_memory.json")
_MEMO_TARGET_MB = 160.0      # per process: keeps every Farneback/imdecode repeat
#: Ceiling for the automatic growth of the per-process memo. The budget used to
#: be a flat 160 MB whatever the machine: on 8 GB it is already shaved to
#: 110-159 MB to fit one more process, but on a 64 GB machine it stayed at 160
#: while the rest of the RAM sat unused -- and the logs show that budget is the
#: binding one (Farneback 0/240 reuses, i.e. the optical flow recomputes every
#: repeat). Past this point the caches hold the whole working set of a block,
#: so more would only reserve RAM for nothing.
_MEMO_MAX_MB = 512.0
_PROC_BASE_MB_DEFAULT = 1100.0   # PSS of one worker without memo (torch+CUDA ctx incl.)


def _win():
    """flatmorphai.platform_compat under native Windows, else None. The Linux
    readings above are never replaced: this is only their fallback."""
    if os.name != "nt":
        return None
    try:
        from .. import platform_compat as _pc
        return _pc
    except Exception:  # noqa: BLE001
        try:
            import flatmorphai.platform_compat as _pc  # type: ignore
            return _pc
        except Exception:  # noqa: BLE001
            return None


def _meminfo_mb():
    """{'MemAvailable': MB, 'MemTotal': MB, ...} from /proc/meminfo ({} if absent)."""
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                k, v = ln.split(":", 1)
                out[k.strip()] = float(v.split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    if not out and _win() is not None:
        out = _win().memory_status_mb()
    return out


def _pss_mb():
    """Proportional set size (shared libs split between the processes that map
    them): the real RAM one more worker costs. 0.0 if unavailable."""
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for ln in fh:
                if ln.startswith("Pss:"):
                    return float(ln.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    if _win() is not None:
        return _win().process_private_mb() or 0.0
    return 0.0


def _trim_heap():
    """Give freed heap pages back to the OS (glibc keeps ~200 MB of fragments
    per worker otherwise). Exact: no data is touched."""
    try:
        import gc
        import ctypes
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        if _win() is not None:
            return _win().peak_working_set_mb() or 0.0
        return 0.0


def _physical_cores():
    """Physical cores (hyper-threads excluded), or None when unknown.

    The reading itself now lives in ``flatmorphai.cpu_topology``, shared with
    the regeneration chain and the four pools that used to ask ``cpu_count()``
    instead: one answer per machine, affinity included. Same value as the
    local copy this replaces (4 on the reference machine).
    """
    try:
        from ..cpu_topology import physical_cores as _pc
        return _pc()
    except Exception:  # noqa: BLE001
        return None


def _machine_sig():
    """What the saved calibration is worth: it describes ONE machine.

    ``step4_memory.json`` holds a measured RAM-per-process and a best process
    count. Moving the folder (or the application) to another machine used to
    silently reuse both, so the first run there started from the old machine's
    figures -- the in-run controller corrects itself, but the start was wrong.
    The signature is deliberately coarse (physical cores, total RAM rounded to
    the GB): a run with a browser open must NOT invalidate the calibration,
    only a real change of machine.
    """
    try:
        from ..cpu_topology import physical_cores, logical_cores
        total = _meminfo_mb().get("MemTotal", 0.0)
        return {"phys": physical_cores() or 0, "logical": logical_cores(),
                "mem_total_gb": int(round(total / 1024.0))}
    except Exception:  # noqa: BLE001
        return {}


def _calib_doc():
    """The saved calibration if it was measured on THIS machine, else {}."""
    try:
        import json
        with open(_RAM_CALIB, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(d, dict):
        return {}
    saved = d.get("machine")
    if saved is None:
        return d            # written before the signature existed: trusted once
    now = _machine_sig()
    if not now:
        return d
    if (int(saved.get("phys") or 0) != int(now.get("phys") or 0)
            or abs(int(saved.get("mem_total_gb") or 0)
                   - int(now.get("mem_total_gb") or 0)) > 1):
        return {}           # another machine: measure again rather than guess
    return d


def _best_procs_load():
    try:
        v = int(_calib_doc().get("best_processes", 0))
        return v if v > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _best_procs_save(n):
    if os.environ.get("FLATMORPHAI_LGBM_PROCESSES", "").strip():
        return            # a forced count (tests, manual runs) is not a measurement
    try:
        import json
        d = {}
        try:
            with open(_RAM_CALIB, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:  # noqa: BLE001
            d = {}
        d["best_processes"] = int(n)
        os.makedirs(os.path.dirname(_RAM_CALIB), exist_ok=True)
        with open(_RAM_CALIB, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except Exception:  # noqa: BLE001
        pass


def _ram_calib_load():
    try:
        v = float(_calib_doc().get("proc_pss_mean_mb", 0.0))
        return v if 300.0 <= v <= 16000.0 else None
    except Exception:  # noqa: BLE001
        return None


def _ram_calib_save(base_mb, peak_mb, memo_mb, nproc):
    try:
        import json
        import time as _t
        os.makedirs(os.path.dirname(_RAM_CALIB), exist_ok=True)
        _bp = _best_procs_load()
        with open(_RAM_CALIB, "w", encoding="utf-8") as fh:
            json.dump({"best_processes": _bp or 0,
                       "proc_pss_mean_mb": round(base_mb, 1), "pss_mean_mb": round(peak_mb, 1),
                       "memo_mb": round(memo_mb, 1), "processes": int(nproc),
                       "machine": _machine_sig(),
                       "when": _t.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    except Exception:  # noqa: BLE001
        pass


def _plan_ram(nproc_cpu, forced, _log, memo_req=None):
    """(nproc, memo_mb_per_process) fitting the RAM available now.
    need/process = measured base RSS (last run, else 1000 MB) * 1.10 + memo;
    keep a reserve for the GUI / main process / OS (max(700 MB, 8 % total))."""
    mi = _meminfo_mb()
    avail, total = mi.get("MemAvailable", 0.0), mi.get("MemTotal", 0.0)
    memo = max(64.0, min(512.0, float(memo_req) if memo_req else _MEMO_TARGET_MB))
    calib = _ram_calib_load()
    base = calib if calib is not None else _PROC_BASE_MB_DEFAULT
    need = base * 1.05 + memo
    if not avail:
        _log("  RAM planner: /proc/meminfo unavailable -> %d process(es), memo %.0f MB"
             % (nproc_cpu, memo))
        return nproc_cpu, memo
    reserve = max(600.0, 0.07 * total)
    fit = int((avail - reserve) // need)
    if forced:
        nproc = nproc_cpu
        if fit < nproc:
            # forced count: shrink the memo first (down to 64 MB) to fit
            memo = max(64.0, min(memo, (avail - reserve) / max(1, nproc) - base * 1.10))
    else:
        nproc = max(1, min(nproc_cpu, fit))
        if fit < 1:
            memo = 64.0
        if calib is None:
            nproc = 1                     # slow start, ramps up on live memory
        else:
            _bp = _best_procs_load() or _physical_cores()
            if _bp:
                nproc = max(1, min(nproc, int(_bp)))
            # One more process is worth more than cache: when the measured best
            # count does not fit with the full memo, shrink the per-process memo
            # (never below 64 MB) until it does. The caches only save repeated
            # work; a missing process loses a whole share of the CPU.
            # target = the CPU limit (the pool's maximum): the "best count" of a
            # previous run may itself have been capped by RAM (5 on 8 GB with
            # the full memo), so it must not cap the trade; the in-run
            # throughput probe still steps down when an extra process gains nothing
            _want = max(1, int(nproc_cpu))
            for _n in range(_want, nproc, -1):          # most processes that fit
                _m = (avail - reserve) / _n - base * 1.05
                if _m >= 64.0:
                    if _m < memo:
                        _log("  RAM planner: memo reduced %.0f -> %.0f MB per process so that %d process(es) "
                             "fit instead of %d" % (memo, _m, _n, nproc))
                    else:
                        _log("  RAM planner: %d process(es) fit with the full memo (start count of the "
                             "last run was %d)" % (_n, nproc))
                    memo = min(memo, _m)
                    nproc = _n
                    break
            # The other direction, which did not exist: when every process the
            # CPU allows already fits, the RAM left over is worth MORE cache
            # rather than nothing. Only in automatic mode (an explicit
            # FLATMORPHAI_DESC_MEMO_MB is the operator's decision) and only up
            # to _MEMO_MAX_MB. Values are unchanged: the caches are keyed on
            # the exact bytes of each image and hand back copies, so a larger
            # budget only avoids recomputing -- it never changes a result.
            if memo_req is None and nproc >= max(1, int(nproc_cpu)):
                _room = (avail - reserve) / max(1, nproc) - base * 1.05
                _grown = min(_MEMO_MAX_MB, _room)
                if _grown >= memo + 16.0:
                    _log("  RAM planner: memo raised %.0f -> %.0f MB per process "
                         "(%d process(es) already fit; the spare RAM goes to the "
                         "caches)" % (memo, _grown, nproc))
                    memo = _grown
    _log("  RAM planner: %.1f GB available (%.1f GB total, %.1f GB reserve, swap used %.1f GB), "
         "~%.0f MB/process (%s) + memo %.0f MB -> %d process(es)%s"
         % (avail / 1024.0, total / 1024.0, reserve / 1024.0,
            (mi.get("SwapTotal", 0.0) - mi.get("SwapFree", 0.0)) / 1024.0, base,
            "measured last run" if calib is not None else "no measure yet: slow start",
            memo, nproc,
            (" (CPU would allow %d)" % nproc_cpu) if nproc < nproc_cpu else ""))
    return nproc, memo


def _run_balanced_chunks(specs, nproc, _mp, ProcessPoolExecutor, _log, progress_cb,
                         nproc_max=None, feed=None):
    """Frames of ALL videos, in order, split into ~2*nproc CONTIGUOUS blocks of
    equal size and handed dynamically to the worker processes: the load is
    balanced whatever the number of frames per video (and 1-2 videos can use
    every process), while each process only builds the videos of its blocks
    and keeps neighbouring frames together (cache locality). Rows are put back
    in frame order and each video's CSV is written exactly like _finalize_rows
    (same DataFrame, same to_csv call). Values are bit-identical (same per-frame
    computer, isolated processes)."""
    import math
    import time as _t
    import pandas as pd
    from concurrent.futures import as_completed
    from .fusion_tsm.nomenclature import app_second
    counts = []
    for sp in specs:
        n = 0
        d = os.path.join(sp.get("video_dir", ""), "images")
        try:
            n = sum(1 for fn in os.listdir(d)
                    if fn.lower().endswith(".png") and app_second(fn) is not None)
        except OSError:
            n = 0
        counts.append(n)
    specs = list(specs)
    items = [(si, i) for si, n in enumerate(counts) for i in range(n)]
    total = len(items)
    if total == 0 and feed is None:
        return {}
    nchunks = max(1, nproc, min(max(1, total), 4 * max(int(nproc), int(nproc_max or nproc))))
    try:   # optional override (tests / tuning)
        _nb = int(os.environ.get("FLATMORPHAI_LGBM_BLOCKS", "0") or "0")
        if _nb > 0:
            nchunks = max(1, min(total, _nb))
    except ValueError:
        pass
    size = max(1, int(math.ceil(total / float(nchunks))))
    if feed is not None:      # streamed videos: fixed block size
        try:
            size = max(1, int(os.environ.get("FLATMORPHAI_LGBM_BLOCK_FRAMES", "8") or 8))
        except ValueError:
            size = 8
    tasks = []
    for k in range(0, total, size):
        block = items[k:k + size]
        grouped = []
        for si, i in block:
            if grouped and grouped[-1][0] == si:
                grouped[-1][1].append(i)
            else:
                grouped.append((si, [i]))
        tasks.append(grouped)
    _log("  CPU: %s physical core(s), %d logical available; best process count known: %s"
         % (_physical_cores() or "?", (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 0),
            _best_procs_load() or "not yet (measured by throughput during this run)"))
    _log("LightGBM: BALANCED frame pool: %d frame(s) of %d video(s) in %d contiguous "
         "block(s) of ~%d frame(s), ADAPTIVE: starts with %d process(es), up to %d "
         "(spawn); values bit-identical." % (total, len(specs), len(tasks), size, nproc,
                                             max(int(nproc), int(nproc_max or nproc))))
    rows_by = {si: [None] * n for si, n in enumerate(counts)}
    last_summary = {}               # pid -> cumulative summary of that worker
    skip_by = {si: 0 for si in range(len(specs))}
    blocks_by = {si: sum(1 for t in tasks for (s2, _i) in t if s2 == si)
                 for si in range(len(specs))}
    done = 0
    peak_rss = {}
    cpu_blocks = []          # CPU seconds of each block (calibration only from real work)
    t0 = _t.time()
    ctx = _mp.get_context("spawn")

    def _take(res):
        nonlocal done
        for si, idx, row in res["rows"]:
            if idx is not None and 0 <= idx < len(rows_by[si]):
                rows_by[si][idx] = row
        for si in res["skip"]:
            skip_by[si] += 1
        if res.get("summary"):
            last_summary[res.get("pid")] = res["summary"]
        if res.get("rss_mb"):
            peak_rss[res.get("pid")] = max(peak_rss.get(res.get("pid"), 0.0), res["rss_mb"])
        cpu_blocks.append(float(res.get("cpu", 0.0) or 0.0))
        done += 1
        _mi = _meminfo_mb()
        _log("  block %d/%d done in %.0f s, cpu %.0f s (%.0f s elapsed) | RAM free %.1f GB, swap used %.1f GB"
             % (done, len(tasks), res["sec"], res.get("cpu", 0.0), _t.time() - t0,
                _mi.get("MemAvailable", 0.0) / 1024.0,
                (_mi.get("SwapTotal", 0.0) - _mi.get("SwapFree", 0.0)) / 1024.0))
        if progress_cb is not None:
            try:
                progress_cb(done, len(tasks), "blocks")
            except Exception:  # noqa: BLE001
                pass

    # ADAPTIVE pool: one single-process executor per worker "slot", so a worker
    # can be added or retired at any time (ProcessPoolExecutor cannot shrink).
    # After every block (and every 5 s) the controller reads MemAvailable, the
    # swap-in rate and the measured PSS of the workers:
    #   * grow   by 1 when there is room for one more full worker (+15 %), the
    #            newest worker has finished a block (its size is known) and
    #            there are more pending blocks than workers;
    #   * shrink by 1 (an idle worker is retired, freeing its RAM) when the
    #            free RAM falls under the reserve or the system swaps in;
    #   * critical (free RAM < 250 MB): the youngest busy worker is stopped and
    #            its block re-queued, before the kernel OOM-kills a random one.
    # A worker that dies anyway only loses its current block (re-queued).
    # Values are unchanged: same per-frame computer, rows put back in order.
    import itertools
    from concurrent.futures import wait as _wait, FIRST_COMPLETED

    nmax = max(1, int(nproc_max or nproc))
    try:
        ncpu_av = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        ncpu_av = os.cpu_count() or 4
    target = max(1, min(int(nproc), nmax))
    _mi0 = _meminfo_mb()
    reserve = max(600.0, 0.07 * _mi0.get("MemTotal", 0.0))
    critical = 250.0
    try:
        memo_mb = float(os.environ.get("FLATMORPHAI_DESC_MEMO_MB", "160") or 160.0)
    except ValueError:
        memo_mb = 160.0
    _cal = _ram_calib_load()
    est_mb = (_cal if _cal is not None else _PROC_BASE_MB_DEFAULT) * 1.05 + memo_mb
    pending = list(range(len(tasks)))
    slots = {}
    sid_gen = itertools.count(1)
    stats = {"max": 0, "grow": 0, "shrink": 0, "kill": 0, "lost": 0}
    last_change = [_t.time()]
    fails = {}

    def _swapin():
        try:
            with open("/proc/vmstat") as fh:
                for ln in fh:
                    if ln.startswith("pswpin "):
                        return float(ln.split()[1])
        except Exception:  # noqa: BLE001
            pass
        return 0.0
    sw = [_swapin(), _t.time()]

    def _new_slot():
        sid = next(sid_gen)
        slots[sid] = {"ex": ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_proc_init,
                                                initargs=(specs,)),
                      "fut": None, "k": None, "t": 0.0, "blocks": 0, "pss": 0.0,
                      "born": _t.time(), "live": 0.0}
        stats["max"] = max(stats["max"], len(slots))
        return sid

    def _retire(sid, kill=False):
        sl = slots.pop(sid, None)
        if sl is None:
            return
        if kill:
            try:
                for pr in list(getattr(sl["ex"], "_processes", {}).values()):
                    pr.terminate()
            except Exception:  # noqa: BLE001
                pass
        try:
            sl["ex"].shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass

    def _live_pss(sl):
        tot = 0.0
        if _win() is not None:
            for pid in list(getattr(sl["ex"], "_processes", {}) or {}):
                tot += _win().process_private_mb(int(pid)) or 0.0
            return tot
        try:
            for pid in list(getattr(sl["ex"], "_processes", {}) or {}):
                with open("/proc/%d/smaps_rollup" % int(pid)) as fh:
                    for ln in fh:
                        if ln.startswith("Pss:"):
                            tot += float(ln.split()[1]) / 1024.0
                            break
        except Exception:  # noqa: BLE001
            pass
        return tot

    def _worker_mb():
        """Size of one worker: the largest of the block-end PSS, the live PSS of
        workers running for >= 60 s, and the calibrated estimate if nothing
        has been measured yet."""
        now_ = _t.time()
        known = [sl["pss"] for sl in slots.values() if sl["pss"] > 0]
        known += [sl["live"] for sl in slots.values() if sl.get("live", 0) > 0 and now_ - sl["born"] >= 60.0]
        return max(known) if known else est_mb

    def _growth_left():
        """RAM the running workers will still take: a worker's PSS keeps rising
        while its caches fill (measured: ~900 MB after a few blocks -> ~1 GB at
        the end), up to the calibrated end-of-run size ``est_mb``."""
        left = 0.0
        for sl in slots.values():
            cur = max(sl.get("pss", 0.0) or 0.0, sl.get("live", 0.0) or 0.0)
            left += max(0.0, est_mb - cur)
        return left

    probe = {}                      # throughput check after a growth
    cap = [nmax]

    def _thr(since=None):
        """Aggregate frames/s = sum of the last block rate of each worker
        (only blocks started after ``since`` when given)."""
        tot, n = 0.0, 0
        for sl in slots.values():
            if sl.get("rate") and (since is None or sl.get("rate_t0", 0.0) >= since):
                tot += sl["rate"]; n += 1
        return tot, n

    def _note(msg):
        _mi = _meminfo_mb()
        _log("  adaptive pool: %s -> %d process(es) | RAM free %.1f GB, ~%.0f MB/worker"
             % (msg, target, _mi.get("MemAvailable", 0.0) / 1024.0, _worker_mb()))

    feed_state = {"closed": feed is None, "burst": False}
    # While the segmentation still runs (feed open) it is the critical path:
    # keep (physical cores - 1) workers so it is not starved of CPU / RAM;
    # when the last video is queued, jump straight to the best known count.
    _phys_c = _physical_cores() or max(1, ncpu_av // 2)
    seg_cap = max(1, _phys_c - 1) if feed is not None else nmax
    if feed is not None and target > seg_cap:
        target = seg_cap

    def _count_frames(sp):
        d = os.path.join(sp.get("video_dir", ""), "images")
        try:
            return sum(1 for fn in os.listdir(d)
                       if fn.lower().endswith(".png") and app_second(fn) is not None)
        except OSError:
            return 0

    def _add_spec(sp):
        si = len(specs)
        specs.append(sp)
        n = _count_frames(sp)
        rows_by[si] = [None] * n
        skip_by[si] = 0
        nb = 0
        for a in range(0, n, size):
            tasks.append([(si, list(range(a, min(n, a + size))))])
            pending.append(len(tasks) - 1)
            nb += 1
        blocks_by[si] = nb
        _log("  stream: %s added (%d frame(s), %d block(s))"
             % (os.path.basename(sp.get("video_dir", "?")), n, nb))

    def _drain_feed(block=False):
        import queue as _q
        while not feed_state["closed"]:
            try:
                sp = feed.get(timeout=2.0) if block else feed.get_nowait()
            except _q.Empty:
                return
            block = False
            if sp is None:
                feed_state["closed"] = True
                feed_state["burst"] = True
            else:
                _add_spec(sp)

    def _submit(k):
        return [(si, ix, specs[si]) for si, ix in tasks[k]]

    for _i in range(target):
        _new_slot()
    try:
        while (pending or any(sl["fut"] is not None for sl in slots.values())
               or not feed_state["closed"]):
            if not feed_state["closed"]:
                _idle = not pending and not any(sl["fut"] is not None for sl in slots.values())
                _drain_feed(block=_idle)
                if not pending and not any(sl["fut"] is not None for sl in slots.values()):
                    continue
            # hand blocks to idle workers (frame order kept: lowest block first)
            busy = sum(1 for sl in slots.values() if sl["fut"] is not None)
            for sid, sl in list(slots.items()):
                if sl["fut"] is None and pending and busy < target:
                    k = pending.pop(0)
                    sl["fut"], sl["k"], sl["t"] = sl["ex"].submit(_proc_chunk, _submit(k)), k, _t.time()
                    sl["t_start"] = sl["t"]
                    busy += 1
            # retire idle workers above the target (frees their RAM now)
            for sid, sl in list(slots.items()):
                if len(slots) > target and sl["fut"] is None:
                    _retire(sid)
                    last_change[0] = _t.time()       # let the freed RAM show up
            if not slots:
                _new_slot()
                continue
            futs = {sl["fut"]: sid for sid, sl in slots.items() if sl["fut"] is not None}
            if not futs:
                if pending and len(slots) < target:
                    _new_slot()
                continue
            done_set, _nd = _wait(list(futs), timeout=5.0, return_when=FIRST_COMPLETED)
            for fut in done_set:
                sid = futs[fut]
                sl = slots.get(sid)
                if sl is None:
                    continue
                k = sl["k"]
                sl["fut"], sl["k"] = None, None
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 -- this worker died
                    stats["lost"] += 1
                    fails[k] = fails.get(k, 0) + 1
                    if fails[k] >= 3:
                        raise
                    pending.insert(0, k)
                    pending.sort()
                    _retire(sid, kill=True)
                    target = max(1, min(target, len(slots)) if slots else 1)
                    last_change[0] = _t.time()
                    _note("worker lost (%s), block %d re-queued" % (type(exc).__name__, k + 1))
                    continue
                sl["blocks"] += 1
                if not stats.get("kicked"):
                    stats["kicked"] = True
                    try:   # workers loaded: the deferred copy-back may start now
                        from ..workers import linux_workdir as _LWk
                        _LWk.kick_pending()
                    except Exception:  # noqa: BLE001
                        pass
                sl["pss"] = max(sl["pss"], float(res.get("rss_mb") or 0.0))
                try:
                    _nf = sum(len(ix) for _si, ix in tasks[k])
                    sl["rate"] = _nf / max(1e-3, float(res.get("sec") or 0.0))
                    sl["rate_t0"] = sl.get("t_start", 0.0)
                except Exception:  # noqa: BLE001
                    pass
                _take(res)
            # ---- adapt to RAM ------------------------------------------------
            now = _t.time()
            _tw = stats.setdefault("tw", {})
            if feed_state["closed"]:   # the capped segmentation phase is not a measurement
                _tw[len(slots)] = _tw.get(len(slots), 0.0) + (now - stats.get("tlast", now))
            stats["tlast"] = now
            mi = _meminfo_mb()
            avail = mi.get("MemAvailable", 0.0)
            if not avail:
                continue
            si_now = _swapin()
            rate = (si_now - sw[0]) / max(1e-3, now - sw[1])       # pages/s
            sw[0], sw[1] = si_now, now
            busy_slots = sorted(((sl["t"], sid) for sid, sl in slots.items() if sl["fut"] is not None),
                                reverse=True)
            if avail < critical and len(busy_slots) > 1:
                _t0b, sid = busy_slots[0]                     # youngest block
                k = slots[sid]["k"]
                pending.insert(0, k)
                pending.sort()
                _retire(sid, kill=True)
                stats["kill"] += 1
                target = max(1, len(slots))
                last_change[0] = now
                _note("RAM critical, youngest worker stopped, block %d re-queued" % (k + 1))
            elif ((avail < reserve or (rate > 2000.0 and avail < 2.0 * reserve))
                  and target > 1 and len(slots) == target
                  and now - last_change[0] > 20.0):
                target -= 1
                stats["shrink"] += 1
                last_change[0] = now
                _note("RAM low (%s)" % ("swap-in %.0f pages/s" % rate if rate > 2000.0 else "under reserve"))
            for sl in slots.values():
                lp = _live_pss(sl)
                if lp > 0:
                    sl["live"] = max(sl.get("live", 0.0), lp)
            try:
                load1 = os.getloadavg()[0]
            except Exception:  # noqa: BLE001
                load1 = 0.0
            newest = max((sl["born"] for sl in slots.values()), default=0.0)
            if probe and len(slots) == target:
                thr_after, n_after = _thr(since=probe["t"])
                if n_after >= len(slots):
                    gain = thr_after / max(1e-9, probe["thr"])
                    if gain < 1.04 and target > 1:
                        target -= 1
                        cap[0] = target
                        stats["shrink"] += 1
                        last_change[0] = now
                        _best_procs_save(target)
                        _note("throughput %.3f -> %.3f frames/s (x%.2f): no gain, staying at %d"
                              % (probe["thr"], thr_after, gain, target))
                    else:
                        _best_procs_save(target)
                        _note("throughput %.3f -> %.3f frames/s (x%.2f): kept"
                              % (probe["thr"], thr_after, gain))
                    probe.clear()
            if avail < critical and len(busy_slots) > 1:
                pass                                   # handled above
            if feed_state.get("burst"):
                feed_state["burst"] = False
                _want = nmax          # measured best on this CPU (6 > 5 > 4 > 8, test_pool_threads)
                _fit = int(max(0.0, avail - reserve - _growth_left())
                           // max(1.0, max(_worker_mb(), est_mb) * 1.05))
                _new_t = max(target, min(_want, len(slots) + _fit))
                if _new_t > target:
                    for _j in range(_new_t - target):
                        _new_slot()
                    stats["grow"] += _new_t - target
                    target = _new_t
                    last_change[0] = now
                    _note("segmentation finished: straight to the CPU limit (RAM permitting)")
            _live_cap = nmax if feed_state["closed"] else min(nmax, seg_cap)
            if False:
                pass
            elif (not probe and target < min(_live_cap, cap[0]) and len(pending) >= 2 and len(slots) == target
                  and now - newest >= 45.0
                  and avail - reserve - _growth_left() > max(_worker_mb(), est_mb) * 1.15
                  and rate < 200.0
                  and load1 <= ncpu_av - 0.5
                  and now - last_change[0] > 30.0):
                _thr_b, _n_b = _thr()
                if _n_b >= len(slots) and _thr_b > 0:
                    probe.update(thr=_thr_b, t=now)
                target += 1
                stats["grow"] += 1
                last_change[0] = now
                _new_slot()
                _note("RAM + CPU allow one more worker (load %.1f / %d CPU)" % (load1, ncpu_av))
    finally:
        for sid in list(slots):
            _retire(sid)
    _log("  adaptive pool: up to %d process(es) used (%d grow, %d shrink, %d stopped, %d lost)"
         % (stats["max"], stats["grow"], stats["shrink"], stats["kill"], stats["lost"]))
    try:   # the count used most of the time becomes the next run's start
        _tw = stats.get("tw") or {}
        if _tw:
            _mode = max(_tw, key=lambda k: _tw[k])
            # only RAISES the start count: a count limited by RAM in this run
            # (not by the CPU) is not a CPU optimum; lowering it is the job of
            # the throughput probe ("no gain") above
            if _mode >= 1 and int(_mode) > int(_best_procs_load() or 0):
                _best_procs_save(int(_mode))
                _log("  adaptive pool: %d process(es) used most of the time -> next run starts there"
                     % _mode)
    except Exception:  # noqa: BLE001
        pass
    for k, (pid, summ) in enumerate(sorted(last_summary.items()), 1):
        _log("  worker %d (pid %s) : %s" % (k, pid, summ))
    if peak_rss:
        try:
            memo_mb = float(os.environ.get("FLATMORPHAI_DESC_MEMO_MB", "0") or 0.0)
        except ValueError:
            memo_mb = 0.0
        pk = max(peak_rss.values())
        pm = sum(peak_rss.values()) / len(peak_rss)
        # A run that reused already computed rows (blocks of 0-2 s CPU) never
        # loads the descriptor working set: its tiny PSS must not become the
        # next run's estimate (it made the planner assume 300 MB/process).
        _real = len(cpu_blocks) >= 2 and sorted(cpu_blocks)[len(cpu_blocks) // 2] >= 20.0
        if _real:
            _ram_calib_save(max(300.0, pm - memo_mb), pm, memo_mb, nproc)
        _mi = _meminfo_mb()
        _log("  RAM: PSS per worker at block end %s MB (max %.0f, mean %.0f MB, memo %.0f MB) | %.1f GB "
             "available now; saved for the next run's planner."
             % ("/".join("%.0f" % v for _p, v in sorted(peak_rss.items())), pk, pm, memo_mb,
                _mi.get("MemAvailable", 0.0) / 1024.0))
    out = {}
    for si, sp in enumerate(specs):
        path = sp.get("output_csv_path")
        if skip_by[si] and skip_by[si] >= blocks_by[si]:
            continue                 # nothing computed here (already on disk)
        valid = [r for r in rows_by[si] if r is not None]
        if not valid:
            out[path] = 0
            continue
        df = pd.DataFrame(valid)
        df.to_csv(path, sep=";", index=False, encoding="utf-8")
        out[path] = len(df)
        _log("  LightGBM feature CSV: wrote %d row(s) to %s"
             % (len(df), os.path.basename(path)))
    return out


def _csv_complete(sp):
    """True when the spec's feature CSV already holds one row per frame (it
    was computed while the segmentation ran): nothing left to compute."""
    try:
        import pandas as pd
        from .fusion_tsm.nomenclature import app_second
        p = sp.get("output_csv_path")
        if sp.get("overwrite") or not p or not os.path.isfile(p):
            return False
        d = os.path.join(sp.get("video_dir", ""), "images")
        names = {fn for fn in os.listdir(d)
                 if fn.lower().endswith(".png") and app_second(fn) is not None}
        have = set(pd.read_csv(p, sep=";", usecols=["image"])["image"].astype(str))
        return bool(names) and names <= have
    except Exception:  # noqa: BLE001
        return False


class FeatureStream:
    """Background Step-4 descriptor pool fed video by video while the
    segmentation is still running (Pipeline A overlap). add(spec) as soon as a
    video's masks + formatted images exist, close() when all are queued,
    join() before Step 4 (which then finds the CSVs complete and only predicts).
    """

    def __init__(self, log=None):
        import queue
        import threading
        self.q = queue.Queue()
        self.log = log or (lambda _m: None)
        self.error = None
        self.n = 0
        self.th = threading.Thread(target=self._run, name="flatmorphai-feature-stream", daemon=True)
        self.th.start()

    def _run(self):
        try:
            compute_lightgbm_features_for_videos([], log=self.log, feed=self.q)
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.log("  stream: descriptor pool failed (%s: %s); Step 4 will compute them"
                     % (type(exc).__name__, exc))

    def add(self, spec):
        self.n += 1
        self.q.put(spec)

    def close(self):
        self.q.put(None)

    def join(self):
        self.th.join()


def compute_lightgbm_features_for_videos(specs, max_workers=None, log=None,
                                         progress_cb=None, feed=None):
    """Compute the feature CSVs of MANY videos in ONE shared thread pool.

    ``specs`` is a list of kwargs dicts for ``compute_lightgbm_features_csv``
    (video_dir, mask_dir_name, output_csv_path, bundle_dir, baseline_cols,
    specialist_cols, overwrite, formatted_dir). Each video is prepared once
    (its ``_return_computer`` handle), then EVERY frame of EVERY video is
    submitted to a single ThreadPoolExecutor -- so the worker threads spread
    across all images regardless of which video they belong to, exactly as
    asked, instead of draining and refilling a fresh pool per video. Each
    video's CSV is written by its own finalize once its rows are in.

    Returns ``{output_csv_path: n_rows}`` for the videos that produced a CSV.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _log = _quiet_log(log)
    # Installed here too so the operator's log shows the decision once (the
    # per-video path and the process workers install it on their own).
    _install_mcp_numba(_log)
    try:
        _cpu = os.cpu_count() or 4
    except Exception:  # noqa: BLE001
        _cpu = 4
    workers = max(1, _cpu - 1) if (max_workers is None or int(max_workers) <= 0) \
        else max(1, int(max_workers))

    # Prepare every video (frames, per-frame computer, finalize) without running
    # any pool yet. Videos with nothing to do (all rows cached, etc.) just
    # return an int and are written immediately.
    # VIDEO-LEVEL process pool (multi-video batches): each video
    # computed in its own process, true parallelism for the GIL-bound v4j chain,
    # bit-identical (isolated). One chain build per process (no duplication).
    # Default: AUTO for multi-video batches -- validated in the app (6 videos:
    # 18.2 -> 15.3 min, identical classes). One process per video, capped at
    # cpu_count-2 and 6 (VRAM: one small Gabor CUDA context per process).
    # FLATMORPHAI_LGBM_PROCESSES=1 (or 0) forces the thread pool; =N forces N.
    _env_np = os.environ.get("FLATMORPHAI_LGBM_PROCESSES", "").strip()
    if _env_np:
        try:
            _nproc0 = int(_env_np)
        except ValueError:
            _nproc0 = 0
    else:
        try:
            _cpu_av = len(os.sched_getaffinity(0))
        except Exception:  # noqa: BLE001
            _cpu_av = _cpu
        _nproc0 = max(1, _cpu_av - 2)
    _distrib = os.environ.get("FLATMORPHAI_LGBM_DISTRIB", "frames").strip().lower()
    if _distrib == "video":
        _nproc0 = min(_nproc0, len(specs))   # one process per video at most
    if feed is None and specs:
        # videos whose descriptors were already computed (e.g. during the
        # segmentation) are skipped: no pool start-up for nothing
        _todo = [sp for sp in specs if not _csv_complete(sp)]
        if len(_todo) < len(specs):
            _log("  LightGBM: %d/%d video(s) already have all their descriptors (computed "
                 "during the segmentation)" % (len(specs) - len(_todo), len(specs)))
        if not _todo:
            return {sp.get("output_csv_path"): -1 for sp in specs}
        specs = _todo
    if feed is not None:
        _distrib = "frames"
    if _nproc0 >= 2 and (len(specs) >= 2 or _distrib != "video" or feed is not None):
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        nproc = _nproc0 if _distrib != "video" else min(_nproc0, len(specs))
        if _distrib == "video":
            _log("LightGBM: VIDEO-level PROCESS pool (FLATMORPHAI_LGBM_PROCESSES=%d, "
                 "spawn): %d video(s) across %d process(es), true parallelism for "
                 "the GIL-bound v4j chain; values bit-identical (isolated)."
                 % (_nproc0, len(specs), nproc))
        out = {}
        # NOTE: capping native BLAS threads per process was tried and REVERTED --
        # it made the run SLOWER (31 vs 20 min), i.e. numpy's multi-threaded BLAS
        # helps each process more than the oversubscription hurts. Left at the
        # library defaults on purpose.
        # RAM planner: process count AND per-process memo budget are chosen
        # together from the RAM available right now, so the pool never runs
        # the machine out of memory (WSL kills a worker -> BrokenProcessPool)
        # nor starves the memo (107 MB/process gave 0 % Farneback hits; 160 MB
        # keeps every repeat: 8.7 -> 7.0 s/frame for ~+60 MB). Explicit
        # FLATMORPHAI_LGBM_PROCESSES / FLATMORPHAI_DESC_MEMO_MB still win.
        _memo_saved = os.environ.get("FLATMORPHAI_DESC_MEMO_MB")
        _memo_req = None
        if _memo_saved:          # user value = TOTAL budget, split per process
            try:
                _memo_req = float(_memo_saved) / max(1, nproc)
            except ValueError:
                _memo_req = None
        _nproc_cpu = nproc
        nproc, _per = _plan_ram(nproc, bool(_env_np), _log, _memo_req)
        os.environ["FLATMORPHAI_DESC_MEMO_MB"] = str(_per)
        _gs_started = False
        try:
            from . import gpu_server as _GS
            _gs_started = _GS.start(_log)
        except Exception:  # noqa: BLE001
            _gs_started = False
        if _distrib != "video":
            try:
                _res_b = _run_balanced_chunks(specs, nproc, _mp, ProcessPoolExecutor,
                                              _log, progress_cb,
                                              nproc_max=(nproc if _env_np else _nproc_cpu),
                                              feed=feed)
                if _gs_started:
                    try:
                        _GS.stop()
                    except Exception:  # noqa: BLE001
                        pass
                return _res_b
            except Exception as exc:  # noqa: BLE001
                _log("  balanced frame pool failed (%s: %s); falling back to one "
                     "process per video." % (type(exc).__name__, exc))
        try:
            _ctx = _mp.get_context("spawn")
            done_v = 0
            with ProcessPoolExecutor(max_workers=nproc, mp_context=_ctx) as pool:
                for res in pool.map(_proc_video, specs):
                    csv_path, n = res[0], res[1]
                    out[csv_path] = n
                    done_v += 1
                    _vid = os.path.basename(os.path.dirname(csv_path or "?"))
                    if isinstance(n, str) and n.startswith("ERR:"):
                        _log("  video failed (%s): %s" % (_vid, n))
                    elif len(res) >= 3:
                        _log("  video %d/%d done in %.0f s: %s%s"
                             % (done_v, len(specs), res[2], _vid,
                                (" | memo " + res[3]) if len(res) > 3 and res[3] else ""))
                    if progress_cb is not None:
                        try:
                            progress_cb(done_v, len(specs), "videos")
                        except Exception:  # noqa: BLE001
                            pass
            return out
        except Exception as exc:  # noqa: BLE001
            _log("  VIDEO process pool failed (%s: %s); falling back to the thread "
                 "pool." % (type(exc).__name__, exc))
        finally:
            if _memo_saved is None:
                os.environ.pop("FLATMORPHAI_DESC_MEMO_MB", None)
            else:
                os.environ["FLATMORPHAI_DESC_MEMO_MB"] = _memo_saved
            if _gs_started:
                try:
                    _GS.stop()
                except Exception:  # noqa: BLE001
                    pass

    ctxs = []
    for _si, sp in enumerate(specs):
        try:
            c = compute_lightgbm_features_csv(_return_computer=True, **sp)
        except Exception as exc:  # noqa: BLE001
            _log("  LightGBM pre-pass: prepare failed on '%s' (%s)"
                 % (os.path.basename(sp.get("video_dir", "")), exc))
            continue
        if isinstance(c, dict):
            c["rows"] = [None] * c["n"]
            c["_spec_index"] = _si
            ctxs.append(c)

    # One global task list: (ctx_index, frame_work_item).
    tasks = [(ci, w) for ci, c in enumerate(ctxs) for w in c["work"]]
    total = len(tasks)
    _log("LightGBM: computing descriptors for %d image(s) across %d video(s) "
         "in one pool (%d worker(s))." % (total, len(ctxs), workers))
    # Up-front GPU visibility probe, so the GPU geodesic situation is known
    # immediately rather than only after the whole pool. Purely informational.
    _gpu_geod_on = os.environ.get(
        "FLATMORPH_V4J_GPU_GEOD", "0") in ("1", "true", "True")
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _log("  CUDA visible to descriptor stage: %s (%d device(s)); "
                 "GPU geodesic %s."
                 % (_torch.cuda.get_device_name(0), _torch.cuda.device_count(),
                    "ON via FLATMORPH_V4J_GPU_GEOD=1 (float32, approximate)"
                    if _gpu_geod_on else
                    "OFF (exact CPU path; float32 GPU can change results)"))
            # The heavy convolution battery DOES run on the GPU: Gabor (batched,
            # one conv2d/frame) and the Frangi/Sato/Meijering/Hessian vesselness,
            # in float32, each self-validated against the CPU reference on the
            # first call (auto CPU fallback on any mismatch). Only the geodesic
            # axis stays on CPU (above). The actual GPU-vs-CPU decision is logged
            # once by gabor_gpu/hessian_gpu ("[FlatMorphAI] GPU: ...") on the
            # first frame.
            _hess_gpu = os.environ.get(
                "FLATMORPH_V4J_GPU_CONV", "0") in ("1", "true", "True")
            _log("  GPU convolutions: Gabor (batched) on GPU float32, "
                 "auto-validated (CPU fallback on mismatch). Hessian/vesselness "
                 "%s. CPU-bound parts (geodesic axis, GLCM/LBP, shape) stay on "
                 "CPU, spread across the worker threads."
                 % ("on GPU float32 via FLATMORPH_V4J_GPU_CONV=1 (self-validated)"
                    if _hess_gpu else
                    "on CPU (skimage; default -- GPU drop-in exists but its many "
                    "tiny convolutions contend badly under the thread pool)"))
        else:
            _log("  CUDA NOT visible to descriptor stage "
                 "(torch.cuda.is_available() is False) -- geodesic on CPU.")
    except Exception as _te:  # noqa: BLE001
        _log("  torch not importable in descriptor stage (%s: %s) -- "
             "geodesic on CPU." % (type(_te).__name__, _te))
    # Opt-in PROCESS pool (FLATMORPHAI_LGBM_PROCESSES=N, N>=2): true parallelism
    # for the GIL-bound v4j chain. Frames are spread across processes regardless
    # of video. Bit-identical to the serial path (isolated memory). Default (unset
    # or 1) keeps the thread pool below.
    _nproc = 0
    try:
        _nproc = int(os.environ.get("FLATMORPHAI_LGBM_PROCESSES", "0") or "0")
    except ValueError:
        _nproc = 0
    done = 0
    if tasks and _nproc >= 2:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        _nproc = min(_nproc, len(tasks))
        _log("LightGBM: PROCESS pool enabled (FLATMORPHAI_LGBM_PROCESSES=%d, spawn): "
             "true parallelism for the GIL-bound v4j chain; values bit-identical "
             "(isolated processes)." % _nproc)
        ptasks = [(ci, c["_spec_index"], w) for ci, c in enumerate(ctxs) for w in c["work"]]
        try:
            _ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=_nproc, mp_context=_ctx,
                                     initializer=_proc_init, initargs=(specs,)) as pool:
                for ci, idx, row in pool.map(_proc_worker, ptasks):
                    if idx is not None:
                        ctxs[ci]["rows"][idx] = row
                    done += 1
                    if progress_cb is not None and (done % 10 == 0 or done == total):
                        try:
                            progress_cb(done, total, "features")
                        except Exception:  # noqa: BLE001
                            pass
        except Exception as exc:  # noqa: BLE001
            _log("  PROCESS pool failed (%s: %s); falling back to the thread pool."
                 % (type(exc).__name__, exc))
            _nproc = 0
            done = 0
    if tasks and _nproc < 2:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(ctxs[ci]["compute_one_frame"], w): ci
                    for ci, w in tasks}
            for fut in as_completed(futs):
                ci = futs[fut]
                try:
                    idx, row = fut.result()
                    ctxs[ci]["rows"][idx] = row
                except Exception as exc:  # noqa: BLE001
                    _log("  LightGBM feature worker failed: %s: %s"
                         % (type(exc).__name__, exc))
                done += 1
                if progress_cb is not None and (done % 10 == 0 or done == total):
                    try:
                        progress_cb(done, total, "features")
                    except Exception:  # noqa: BLE001
                        pass

    try:
        from .geodesic_gpu import status as _geod_status
        _log("  " + _geod_status())
    except Exception:  # noqa: BLE001
        pass
    try:
        import sys as _s2
        _mn = _s2.modules.get("flatmorphai.notebook_api.mcp_numba")
        if _mn is not None and _mn._state.get("validated"):
            _log("  MCP numba this run: %d geodesic(s) accelerated, %d delegated to "
                 "skimage" % (_mn.STATS["numba"], _mn.STATS["delegated"]))
        _mc = _s2.modules.get("flatmorphai.notebook_api.memo_cache")
        if _mc is not None and _mc._installed:
            _log("  descriptor memo this run: " + _mc.summary())
    except Exception:  # noqa: BLE001
        pass

    # Actual GPU-vs-CPU decision taken by the convolution drop-ins during this
    # run (read from their module flags, set on the first frame). Confirms, in
    # the log the operator sees, that Gabor + Hessian/vesselness really ran on
    # the GPU -- the heavy battery that was offloaded -- vs fell back to CPU.
    try:
        import sys as _sys
        _gg = _sys.modules.get("gabor_gpu")
        _hg = _sys.modules.get("hessian_gpu")

        def _st(flag):
            return ("GPU" if flag is True
                    else "CPU (fallback)" if flag is False else "not requested")
        if _gg is not None and getattr(_gg, "_FFT_OK", None) is True:
            _log("  Gabor this run: CPU FFT float64 (validated vs skimage) | "
                 "Hessian/vesselness %s" % _st(getattr(_hg, "_GPU_OK", None)))
        elif _gg is not None or _hg is not None:
            _log("  GPU actually used this run: Gabor %s | batched Gabor %s | "
                 "Hessian/vesselness %s"
                 % (_st(getattr(_gg, "_GPU_OK", None)),
                    _st(getattr(_gg, "_MANY_OK", None)),
                    _st(getattr(_hg, "_GPU_OK", None))))
    except Exception:  # noqa: BLE001
        pass

    out = {}
    for c in ctxs:
        try:
            out[c["output_csv_path"]] = c["finalize"](c["rows"])
        except Exception as exc:  # noqa: BLE001
            _log("  LightGBM pre-pass: finalize failed on '%s' (%s)"
                 % (os.path.basename(c["output_csv_path"]), exc))
    return out


def _readbytes(path: str):
    """cv2-safe read that survives non-ASCII paths."""
    import numpy as _np
    return _np.fromfile(path, dtype=_np.uint8)


# ---------------------------------------------------------------------
# Batched prediction over a feature DataFrame
# ---------------------------------------------------------------------
def predict_dataframe(
    classifier: Any,
    features_df: Any,
    apply_specialist: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], List[int]]:
    """Run inference over an assembled feature DataFrame.

    ``apply_specialist`` toggles the Walnut-vs-C-shape arbitration
    (expert probability >= threshold on a baseline C-shape).  Since
    the October 2026 bundle the arbitration needs no extra column,
    missing descriptors are NaN and handled natively by the trees.

    Returns ``(pred_names, pred_indices)`` where the indices are
    into ``classifier.classes`` (the five base classes -- Bridge-like
    was removed).
    """
    _log = _quiet_log(log)
    if features_df is None or len(features_df) == 0:
        return [], []
    use_specialist = bool(apply_specialist)
    names = classifier.predict(
        features_df,
        apply_specialist=use_specialist,
        return_labels=True,
    )
    class_to_idx = {c: i for i, c in enumerate(classifier.classes)}
    pred_idx = [int(class_to_idx.get(str(n), -1)) for n in names]
    return [str(n) for n in names], pred_idx
