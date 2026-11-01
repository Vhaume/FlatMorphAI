"""step3_metrics.py -- compute ONLY the metrics the shipped model needs, skipping the rest.

The model was trained on 308 metric columns (config/metrics_required.txt). Those 308 span 16 of the 31 metric GROUPS
(config/metrics_groups_required.txt). This module computes exactly those 16 groups and skips the other 15 (the static
shape group is cheap, the skipped ones include the heavy 3-mask deformation, peristalsis, topology and trajectory
panels), so a frame is scored at a fraction of the full-precompute cost. ALL metric code is still shipped under
metrics_lib/, nothing is deleted, the config file is the single switch that selects what runs.

The group->function dispatch below is a faithful copy of compute_feat in precompute_mask3_metrics.py, restricted to the
required groups. Groups that need an input the app supplies separately (the registered motion image for 'motion' and
'motioncl', the wider mask window for 'widehead') are computed when that input is passed and otherwise skipped, in
which case the model falls back to the training-median value for those columns, exactly as in training.

Inputs (per labelled frame T0):
  masks3   : (prev, cur, next) worm masks at native resolution, binary or 0/255.
  grays3   : (prev, cur, next) raw grayscale frames at native resolution.
  formatted_t0_bgr : the 384 masked+reshaped colour image of T0 (classification_images.compute_mask_image_planarian_reshaped).
  raw_color3 (optional) : (prev, cur, next) raw COLOUR frames, for the aligned temporal-colour metrics.
  motion_image (optional): the registered prev/cur/next motion image, for the 'motion' / 'motioncl' groups.
  wide_masks (optional)  : a list of >=4 binary masks over a +/-5 frame window, for the 'widehead' group.

Dependencies: numpy, opencv, scikit-image, plus the metric modules in metrics_lib/.
"""
import os
import sys
import numpy as np
import cv2

_HERE = os.path.dirname(__file__)
# The original export_model_and_code layout placed metrics_lib/ as a
# sibling of pipeline/; in the FlatMorphAI tree the metric modules
# live INSIDE this subpackage at ``fusion_tsm/metrics_lib/`` (and the
# bundled ``flatmorphai_api/`` reference copies are NOT used at
# runtime -- we always defer to the live app modules).  Point
# sys.path at the local metrics_lib so the bundle's bare imports keep
# resolving without further patching.
sys.path.insert(0, os.path.join(_HERE, 'metrics_lib'))

import core_shape_metrics as _core
import shape_metrics as _shp
import texture_uniformity as _tu
import twist_border_evolution as _tbe
import texture_features as _txt
import fold_twist_metrics as _ftm
import regional_motion as _rm
import motion_metrics as _mm2
import twist_border_evolution_temporal as _tbet
import twist_motion_masks as _tmm
import nonrigid_motion as _nrm
import sota_motion_candidates as _sota
import head_extension as _he
import neck_extent as _nke
import head_extension_wide as _hxw
# Families the first port of this module dropped, which the LightGBM model still
# reads (their columns came out NaN at inference). Added back here, called
# EXACTLY as the bundle's precompute_mask3_metrics.compute_feat calls them, so
# the per-frame values match the training tables. Resolved from the bundle on
# sys.path, like the imports above.
import geodesic_width as _gw
import terminal_shape as _tsh
import axial_pigment as _apg
import peristalsis as _per
import topology_motion as _tpm
import axial_texture_flow as _txf
import centerline_texture as _ctex
import end_bend as _eb
import shape_persistence as _sper
import tail_mobility as _tmob

_CFG = os.path.join(_HERE, 'config')


def required_metrics():
    return [l.strip() for l in open(os.path.join(_CFG, 'metrics_required.txt'), encoding='utf-8') if l.strip()]


def required_groups():
    return [l.strip() for l in open(os.path.join(_CFG, 'metrics_groups_required.txt'), encoding='utf-8') if l.strip()]


def _b(m):
    return (np.asarray(m) > 0).astype(np.uint8)


def compute_required_metrics(masks3, grays3, formatted_t0_bgr, raw_color3=None,
                             motion_image=None, wide_masks=None, shape_masks=None,
                             groups=None):
    """Return {metric_name: value} for the required columns. Missing inputs simply leave their columns out (the model
    median-fills them). Mirrors compute_feat for the required groups."""
    groups = set(groups or required_groups())
    mp, mc, mn = [_b(x) if x is not None else None for x in masks3]
    gp, gc, gn = grays3
    feat = {}

    # ---- single-frame, current mask ----
    if mc is not None:
        Bc = _core.shape_desc(mc)
        if 'basic' in groups:
            try:
                feat.update(_core.static_features(Bc))
                if gc is not None:
                    feat.update(_core.new_metrics(mc, np.asarray(gc)))
                feat.update(_core.curv_metrics_T0(mc))
                feat.update(_core.circ_metrics_T0(mc))
            except Exception:
                pass
        if 'neckext' in groups:
            try: feat.update(_nke.neck_extent(mc))
            except Exception: pass
        if 'terminal' in groups:
            try: feat.update(_tsh.terminal_shape(mc))
            except Exception: pass
        if 'geodesic' in groups:
            try: feat.update(_gw.geodesic_width(mc))
            except Exception: pass
        if 'pigment' in groups and gc is not None:
            try: feat.update(_apg.axial_pigment(mc, np.asarray(gc)))
            except Exception: pass
    else:
        Bc = None

    # ---- formatted (reshaped, masked) T0 image: shape / tunif / centerline / texture / curated ----
    if formatted_t0_bgr is not None and (groups & {'shape', 'tunif', 'centerline', 'texture', 'curated'}):
        ff = formatted_t0_bgr
        gg = cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY) if ff.ndim == 3 else ff
        fm = (gg > 8).astype(np.uint8)
        if 'shape' in groups:
            try: feat.update(_shp.shape_T0(fm))
            except Exception: pass
        if 'tunif' in groups:
            try: feat.update(_tu.texture_uniformity_features(gg, fm))
            except Exception: pass
        if 'centerline' in groups:
            try: feat.update(_tbe.border_evolution(gg))
            except Exception: pass
        if 'ctex' in groups:
            try: feat.update(_ctex.centerline_texture(gg))
            except Exception: pass
        if 'endbend' in groups:
            try: feat.update(_eb.end_bend(gg))
            except Exception: pass
        if 'texture' in groups:
            try: feat.update(_txt.all_texture_features(gg, ff, fm))
            except Exception: pass
        if 'curated' in groups:
            try:
                feat.update(_ftm.curated_metrics(gg, (gg > 0)))
                feat.update(_ftm.color_metrics(ff, (cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY) > 0) if ff.ndim == 3 else (ff > 0)))
            except Exception: pass

    # ---- registered motion image (app-provided): motion / motioncl ----
    if motion_image is not None and (groups & {'motion', 'motioncl'}):
        try:
            if 'motion' in groups:
                feat.update(_rm.regional_motion_features(motion_image)); feat.update(_mm2.motion_metrics(motion_image))
            if 'motioncl' in groups:
                feat.update(_tbet.twist_border_evolution_temporal(motion_image))
        except Exception: pass

    # ---- 3-mask groups ----
    md3 = [mp, mc, mn]
    if all(m is not None for m in md3):
        if 'motionmask' in groups:
            try: feat.update(_tmm.twist_motion_masks(mp * 255, mc * 255, mn * 255))
            except Exception: pass
        if 'nonrigid' in groups:
            try: feat.update(_nrm.nonrigid_motion(mp * 255, mc * 255, mn * 255))
            except Exception: pass
        if 'sota' in groups:
            try:
                gs = [np.asarray(g) if g is not None else None for g in (gp, gc, gn)]
                gs = [cv2.resize(g, (mc.shape[1], mc.shape[0])) if (g is not None and g.shape != mc.shape) else g for g in gs]
                feat.update(_sota.sota_motion(mp * 255, mc * 255, mn * 255, gs[0], gs[1], gs[2]))
            except Exception: pass
        if 'headext' in groups:
            try: feat.update(_he.head_extension(mp, mc, mn))
            except Exception: pass
        if 'tailmob' in groups:
            try: feat.update(_tmob.tail_mobility(mp, mc, mn))
            except Exception: pass
        if 'peristalsis' in groups:
            try: feat.update(_per.peristalsis(mp, mc, mn))
            except Exception: pass
        if 'topology' in groups:
            try: feat.update(_tpm.topology_motion(mp * 255, mc * 255, mn * 255))
            except Exception: pass
        if 'texflow' in groups:
            try:
                # Texture optical flow on the MASKED raw grayscale (the glide the
                # silhouette hides), exactly as compute_feat's 'texflow' group:
                # raw grays resized to the mask, zeroed outside the worm.
                gtf = [np.asarray(g) if g is not None else None for g in (gp, gc, gn)]
                gtf = [cv2.resize(g, (mc.shape[1], mc.shape[0])) if (g is not None and g.shape != mc.shape) else g for g in gtf]
                if all(g is not None for g in gtf):
                    mgf = [(gtf[i].astype(np.float32) * (md3[i] > 0)).astype(np.uint8) for i in range(3)]
                    feat.update(_txf.texture_flow(mgf[0], mgf[1], mgf[2], mc))
            except Exception: pass
        if 'dyn' in groups:
            try:
                d3 = [_core.shape_desc(x) for x in md3]
                if all(d is not None for d in d3):
                    feat.update(_core.dyn_features(d3, Bc if Bc is not None else d3[1]))
                feat.update(_core.curv_dynamic(mp, mc, mn)); feat.update(_core.circ_dynamic(mp, mc, mn))
                _nd = feat.get('net_disp', np.nan); _rl = feat.get('rel_len_chg', np.nan)
                _ec = feat.get('ecc_chg', np.nan); _cr = feat.get('circ_resid_chg', np.nan)
                if _nd == _nd:
                    feat['rigid_translation'] = _nd / ((0 if _rl != _rl else _rl) + (0 if _ec != _ec else _ec) + 0.05)
                if _nd == _nd and _cr == _cr:
                    feat['moved_kept_arc'] = _nd / (_cr + 0.05)
            except Exception: pass
        if 'temporal' in groups and raw_color3 is not None:
            try: feat.update(_ftm.temporal_metrics_aligned(list(raw_color3), md3))
            except Exception: pass

    # ---- wide-horizon scrunch (app supplies the +/-5 frame mask window) ----
    if 'widehead' in groups and wide_masks is not None and len(wide_masks) >= 4:
        try: feat.update(_hxw.head_extension_wide([_b(w) for w in wide_masks]))
        except Exception: pass

    # ---- C-shape persistence (app supplies the +/-2 frame mask window) ----
    if 'shapeper' in groups and shape_masks is not None and len(shape_masks) >= 3:
        try: feat.update(_sper.shape_persistence([_b(w) for w in shape_masks]))
        except Exception: pass

    keep = set(required_metrics())
    return {k: v for k, v in feat.items() if k in keep}
