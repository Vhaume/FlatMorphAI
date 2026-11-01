# -*- coding: utf-8 -*-
"""Make the app's stereotype descriptors match the reference golden exactly.

The reference golden descriptors were produced by
``calcul_descripteurs_stereotypies_v3c.py``, which does NOT use the descriptor
modules as shipped: at run time it replaces the Zhang-Suen ``skeletonize`` with
``axe_raster`` (the single longest medial geodesic path, one pixel thick) in the
modules whose metrics only use the main body axis, replaces the worm centroid
with ``centre_corps`` for the central-pigment expert, and tweaks
``patch_features``. Without these replacements the app computes a BRANCHY
skeleton, so every axis-based descriptor (``ax_straightness``, ``ax_helix_*``,
``ts_*``/``tl_*``/``to_*`` texture, ``gw_*``, ``twist_*``/``waist_*``/``screw_*``,
central_*) drifts from the reference -- verified: ``ax_straightness`` went from
0.398 (app) to 0.939 (= reference) once ``axe_raster`` is used.

This module reproduces v3c's replacements EXACTLY and applies them to the app's
module objects, so the app's feature table matches the reference the model was
trained on. Idempotent; any individual patch that does not find its expected
source is skipped (logged) rather than raising, so it can never break the run.
"""
from __future__ import annotations

import re

import numpy as np
import cv2
from skimage.morphology import skeletonize as _skeletonize
from skimage.graph import MCP_Geometric


# --------------------------------------------------------------------------
# v3c body axis: the single longest medial geodesic path (verbatim from
# calcul_descripteurs_stereotypies_v3c.py).
# --------------------------------------------------------------------------
def _plus_grande_composante(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    assert n > 1, "masque vide"
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def axe_chemin(mask):
    m = _plus_grande_composante(np.asarray(mask) > 0)
    if m.sum() == 1:
        y, x = np.argwhere(m)[0]
        return [(int(y), int(x))]
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    mcp = MCP_Geometric(np.where(m > 0, 1.0, np.inf), fully_connected=True)
    s = tuple(int(v) for v in np.unravel_index(int(np.argmax(dt)), dt.shape))
    cum, _ = mcp.find_costs([s]); cum = np.where((m > 0) & np.isfinite(cum), cum, -1.0)
    a = tuple(int(v) for v in np.unravel_index(int(np.argmax(cum)), cum.shape))
    cum, _ = mcp.find_costs([a]); cum = np.where((m > 0) & np.isfinite(cum), cum, -1.0)
    b = tuple(int(v) for v in np.unravel_index(int(np.argmax(cum)), cum.shape))
    mcp2 = MCP_Geometric(np.where(m > 0, 1.0 / np.maximum(dt, 1.0), np.inf),
                         fully_connected=True)
    mcp2.find_costs([a], [b])
    return [(int(y), int(x)) for y, x in mcp2.traceback(b)]


def axe_raster(mask):
    r = np.zeros(np.asarray(mask).shape, bool)
    P = np.array(axe_chemin(mask))
    r[P[:, 0], P[:, 1]] = True
    return _skeletonize(r)


def centre_corps(mm):
    P = axe_chemin(mm)
    return float(P[len(P) // 2][0]), float(P[len(P) // 2][1])


def _remplace(ns, texte, fname, ancien, nouveau, nom):
    """Replace ``ancien`` by ``nouveau`` inside function ``fname`` only, then
    exec the new definition into namespace ``ns`` (v3c's exact mechanism)."""
    i = texte.index("def %s(" % fname)
    mo = re.search(r"\n(?=\S)", texte[i + 1:])
    j = i + 1 + mo.start() if mo else len(texte)
    src = texte[i:j]
    assert src.count(ancien) == 1, ("replacement impossible", nom, fname)
    exec(compile(src.replace(ancien, nouveau), nom + ":" + fname, "exec"), ns)  # noqa: S102


_MODULES_PATCHED = False


def _src(mod):
    import inspect
    return inspect.getsource(mod)


def _patch_global_skeletonize(modname, log):
    """TX/GW/TMM/CLF style: replace the module-global skeletonize."""
    try:
        mod = __import__(modname)
        if hasattr(mod, "skeletonize"):
            mod.skeletonize = axe_raster
            return True
    except Exception as exc:  # noqa: BLE001
        log("  v3c patch: %s skeletonize not patched (%s)" % (modname, exc))
    return False


def apply_module_patches(log=None):
    """Patch the fusion_tsm / feature_extraction modules ONCE (global state)."""
    global _MODULES_PATCHED
    _log = log or (lambda _m: None)
    if _MODULES_PATCHED:
        return
    # modules where EVERY skeleton use is the body axis -> global replace
    for name in ("texture_features", "geodesic_width", "twist_motion_masks",
                 "centerline_luminosity_features"):
        _patch_global_skeletonize(name, _log)
    # fold_twist_metrics: ONLY axial_twist_features uses the axis; self_prox
    # (dense_maps) keeps the real skeleton, exactly as v3c.
    try:
        import fold_twist_metrics as FTM
        FTM._AXE_RASTER = axe_raster
        _remplace(FTM.__dict__, _src(FTM), "axial_twist_features",
                  "sk=skeletonize(worm); P=_ordered_centerline(sk)",
                  "sk=_AXE_RASTER(worm); P=_ordered_centerline(sk)",
                  "fold_twist_metrics")
    except Exception as exc:  # noqa: BLE001
        _log("  v3c patch: fold_twist axial_twist_features not patched (%s)" % exc)
    _MODULES_PATCHED = True
    _log("  v3c descriptor patches applied (axe_raster / centre_corps).")


def apply_ext_patches(ext, log=None):
    """Patch the bundle extractor modules and rebind the ``ext`` entries.

    Called per video (``ext`` is rebuilt each call); the underlying modules are
    only re-defined once but the ext dict must be re-pointed at the patched
    callables every time.
    """
    _log = log or (lambda _m: None)
    # compute_new_metrics.twist_osc_feats: axis via axe_raster
    try:
        # Imported under its BARE name here, so the module may be executed a
        # second time (it is already in sys.modules as
        # feature_extraction.compute_new_metrics). Its module body prints
        # "python = / DATASET = / indexed masks:", which is what made those
        # three lines appear once per worker process and at every page load.
        import contextlib as _ctx
        import io as _io
        with _ctx.redirect_stdout(_io.StringIO()), _ctx.redirect_stderr(_io.StringIO()):
            import compute_new_metrics as CNM
        if not getattr(CNM, "_V3C_PATCHED", False):
            CNM._AXE_RASTER = axe_raster
            _remplace(CNM.__dict__, _src(CNM), "twist_osc_feats",
                      "sk = skeletonize(b > 0); path = longest_path(sk)",
                      "sk = _AXE_RASTER(b > 0); path = longest_path(sk)",
                      "compute_new_metrics")
            CNM._V3C_PATCHED = True
        if ext is not None and "twist_osc_feats" in ext:
            ext["twist_osc_feats"] = CNM.twist_osc_feats
    except Exception as exc:  # noqa: BLE001
        _log("  v3c patch: compute_new_metrics not patched (%s)" % exc)
    # compute_expert_image_features.central: centroid -> centre_corps
    try:
        import compute_expert_image_features as CEI
        if not getattr(CEI, "_V3C_PATCHED", False):
            CEI._CENTRE_CORPS = centre_corps
            _remplace(CEI.__dict__, _src(CEI), "central",
                      "yc, xc = Ys.mean(), Xs.mean()",
                      "yc, xc = _CENTRE_CORPS(mm)", "compute_expert_image_features")
            if hasattr(CEI, "FUNCS"):
                CEI.FUNCS["central"] = CEI.central
            CEI._V3C_PATCHED = True
        if ext is not None and "expert_central" in ext:
            ext["expert_central"] = CEI.central
    except Exception as exc:  # noqa: BLE001
        _log("  v3c patch: compute_expert_image_features not patched (%s)" % exc)


def apply_patches(ext=None, log=None):
    apply_module_patches(log=log)
    apply_ext_patches(ext, log=log)
