# =============================================================================
# evaluate_test_images_v3.py
# GPU-batched evaluation pipeline for biomedical image segmentation (v3)
#
# Architecture
# ------------
# Phase 1 (batch, GPU):  Pass-1 inference on batches of batch_size images.
# Phase 2 (CPU):         Connected-component extraction; crop queue built.
# Phase 3 (batch, GPU):  Pass-2 inference on all component crops batched.
# Phase 4 (CPU):         Mask reconstruction, fusion, post-processing.
# Phase 5:               Metric computation, summary statistics, CDF plots.
#
# GPU optimisation
# ----------------
# All component crops across a batch of images are gathered into a flat list
# then forwarded through the model in sub-batches of batch_size.  This
# maximises GPU throughput while keeping memory bounded.
#
# Correctness guarantee
# ---------------------
# The ROI computation, reflect-padding, mask resizing, and pasting logic
# replicate _run_single_component_pass2 from predict_one_image_augmented_v9
# exactly.  The skeleton_guided distortion gate is replicated verbatim from
# the fusion block in predict_one_image_augmented (v9).
#
# Import note
# -----------
# This file imports from predict_one_image_augmented_v9.  Adjust the module
# name at the top of the import block if your file is named differently.
#
# References
# ----------
# Dice L.R., Ecology, 26(3), 297-302, 1945. DOI: 10.2307/1932409
# Cheng B. et al., Boundary IoU, CVPR 2021. arXiv: 2103.16562
# Shit S. et al., clDice, CVPR 2021. DOI: 10.1109/CVPR46437.2021.01629
# Huttenlocher D.P. et al., IEEE TPAMI, 15(9), 850-863, 1993.
#   DOI: 10.1109/34.232073
# Maier-Hein L., Reinke A. et al., Nature Methods, 21(2), 195-212, 2024.
#   DOI: 10.1038/s41592-023-02151-z
# Isensee F. et al., nnU-Net, Nature Methods, 18, 203-211, 2021.
#   DOI: 10.1038/s41592-020-01008-z
# Ke L. et al., HQ-SAM, NeurIPS 2023. arXiv: 2306.01567
# Filatov A., Shiryaev E., ISSPA 2010. DOI: 10.1109/ISSPA.2010.5605426
# Maurer C.R. et al., IEEE TPAMI, 25(2), 265-270, 2003.
#   DOI: 10.1109/TPAMI.2003.1177156
# =============================================================================

import io
import os
import contextlib
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from scipy import ndimage as sp_ndimage
from skimage.morphology import skeletonize as sk_skeletonize

# ---------------------------------------------------------------------------
# Import metric and fusion helpers from the v9 inference module.
# Adjust the module name if your file is named differently.
# ---------------------------------------------------------------------------
# =============================================================================
# predict_one_image_augmented - context-aware two-pass inference
# + four-metric evaluation suite (v9)
#
# Changelog v9 vs v8
# -------------------
# Confirmed failure of all radius-based gates on thin-branch images:
#   spatially_adaptive (v8): P2_blocked=772px (6.7%), HD95=68.34px.
#   The 772 blocked pixels were TRUE POSITIVES: thin-branch extensions
#   that P2 correctly recovered but that were ~68px from P1's endpoint,
#   further than r_outer_thin=10px.  No radius-based gate can solve this
#   because branch length is unbounded relative to feature width.
#
# Root cause of all previous failures:
#   Every gate strategy uses distance from P1 as the acceptance criterion.
#   A missed thin-branch extension can be arbitrarily far from P1 (depends
#   on how much of the branch P1 missed), so any fixed or proportional
#   radius will block genuine recoveries on long thin branches.
#
# Fix -- _skeleton_guided_fusion (new in v9, recommended):
#   Uses TOPOLOGICAL CONNECTIVITY as the acceptance criterion instead of
#   distance.
#   Key insight:
#     - Missed thin-branch extensions: their P2 skeleton is always
#       CONNECTED to P1 (the branch grows from P1's endpoint).
#     - Reflection/hallucination blobs: their P2 skeleton is ISOLATED
#       from P1 (the reflection is spatially disjoint from the worm).
#   Strategy:
#     1. Skeletonise P2.
#     2. Find P2 skeleton pixels within r_connect of P1 (connected).
#     3. Grow a tube of radius r_tube around connected P2 skeleton pixels.
#     4. Gate = tube (for connected branch recovery)
#            OR tight dilation of P1 (for surface micro-corrections).
#     5. final = P1 OR (P2 AND gate).
#   Parameters (all auto from P1 EDT + skeleton):
#     r_connect = max(round(typical_radius), 4)   skeleton attachment tol.
#     r_tube    = max(round(typical_radius), 4)   tube width (= branch width)
#     r_tight   = max(2, round(0.25*typical_radius))  surface micro-gate
#   Effect for thin-branch case (typical_r=10px):
#     r_connect=10px -> P2 skeleton at tip of P1 is "connected".
#     r_tube=10px    -> tube covers the full thin branch (width=~10px).
#     Branch 68px long: fully captured regardless of length.
#     Reflection skeleton 50px from P1: NOT connected -> blocked.
#   References:
#     Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
#     Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
#     Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
#     Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
#
# All v8 functions are unchanged.
# New parameter value: two_pass_merge="skeleton_guided"  (recommended)
#
# Changelog v8 vs v7
# -------------------
# Observed bimodal failure of all previous single-radius fusion strategies:
#   "adaptive_boundary" (v7): good Dice on thick worm segments (tight gate
#     blocks P2 deformations), but clDice drops on thin parts because
#     r_outer~2px is too tight and rejects P2 pixels on genuinely missed
#     thin branches.
#   "replace" (v5): good clDice on thin parts (no gate, all P2 accepted),
#     but Dice drops on thick segments because P2 boundary hallucinations
#     and reflection pixels are accepted without filtering.
#   No single global radius solves both regimes simultaneously.
#
# Fix -- _spatially_adaptive_fusion (new, recommended):
#   Builds a SPATIALLY VARYING gate from P1's distance transform.
#   At each location the gate width is proportional to the local feature
#   width, so thick regions get a tight gate and thin regions a wide one.
#     thin_mask  = P1 pixels where EDT <= thin_threshold
#     thick_mask = P1 pixels where EDT >  thin_threshold
#     thin_threshold = 0.75 * typical_radius  (auto)
#     r_outer_thin   = max(4, round(typical_radius))   wide: recovers branches
#     r_outer_thick  = max(2, round(0.25*typical_radius)) tight: blocks deformation
#     gate  = dilate(thin_mask,  r_outer_thin)
#          OR dilate(thick_mask, r_outer_thick)
#     final = P1 OR (P2 AND gate)     (P1 always preserved by construction)
#   References:
#     Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
#     Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
#     Kamnitsas K. et al., Medical Image Analysis 2017,
#       DOI: 10.1016/j.media.2016.10.004
#     Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
#     Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
#
# All v7 functions and the Pass-2 raw-crop path are unchanged.
# New parameter value: two_pass_merge="spatially_adaptive"  (recommended)
#
# Changelog v7 vs v6
# -------------------
# Problem summary (confirmed from debug logs across v5/v6/v7-wrong):
#
#   v5 ("replace"):
#     Raw Pass-2 union. High clDice (thin branches recovered), but P2 mask
#     deforms the boundary because the unconstrained union accepts all P2
#     pixels including hallucinations.
#
#   v6 ("constrained_union"):
#     Fixed-fraction dilation radius r = max(3, round(0.05*min(H,W))) = 54px.
#     For the observed crop (1080x1440 image, component ROI 320x973), 54px
#     covered the entire P2 mask: P2_blocked=0.0%.  The gate was useless.
#     Fusion was algebraically identical to v5 (P1 OR P2 = no filtering).
#
#   v7-wrong ("zoom-resize"):
#     Sent the ROI crop resized to (W, H) via INTER_LINEAR.  This amplified
#     the reflection (which lives near the worm in the image) to fill the
#     full frame, causing the model -- which uses background context to
#     separate worm from reflection -- to predict the reflection as foreground.
#     Result: worse than v5.
#
# Root cause of v5/v6 deformation (revised understanding):
#   The Pass-2 crop (raw, with padding_ratio=0.50) is sent directly to
#   predict_fn.  The AutoImageProcessor resizes it to the model's square
#   input, which introduces geometric distortion on elongated crops.
#   However, the padded ROI retains enough background context (worm + water
#   surface + surroundings) for the model to suppress the reflection.
#   Removing that context (zoom-resize) makes things worse.
#   The deformation therefore comes from the unconstrained P1 OR P2 union,
#   NOT from the crop geometry.
#
# v7 fix (this file):
#   Keep the v5 raw-crop Pass-2 path exactly as it was (crop -> reflect-pad
#   -> predict_fn).  Replace the fusion from a fixed-radius gate to an
#   adaptive gate whose radii are derived automatically from P1's Euclidean
#   distance transform evaluated at skeleton pixels.
#
#   typical_radius = median(EDT(P1) at skeleton pixels)  [feature half-width]
#   r_outer = max(2, round(0.25 * typical_radius))
#     For thin worm (~6px half-width): r_outer = 2px  (vs 54px in v6)
#   r_core  = max(0, round(0.75 * typical_radius - 1))
#   final   = core OR (P2 AND dilate(P1, r_outer))
#
#   With r_outer=2px on a thin elongated worm:
#     - P2 pixels that genuinely lie on missed thin branches (within 2px of
#       the P1 contour) are accepted.
#     - P2 pixels that are reflections or boundary hallucinations (further
#       away from P1) are blocked.
#   Expected: P2_blocked >> 0% (meaningful filtering).
#
# Metrics
# -------
# dice            : volumetric Sorensen-Dice overlap
#                   Dice L.R., Ecology 1945, DOI: 10.2307/1932409
#
# boundary_dice   : Dice on boundary strips of width d pixels
#                   Cheng B. et al., CVPR 2021, pp. 15334-15342.
#                   arXiv: 2103.16562
#
# cl_dice         : centerline Dice on morphological skeleta
#                   Shit S. et al., CVPR 2021, pp. 16560-16569.
#                   DOI: 10.1109/CVPR46437.2021.01629  arXiv: 2003.07311
#
# hd95            : 95th-percentile Hausdorff distance (in pixels)
#                   Huttenlocher D.P. et al.
#                   IEEE TPAMI, 15(9), 850-863, 1993.
#                   DOI: 10.1109/34.232073
#
# Metric selection rationale:
#   Maier-Hein L., Reinke A. et al.
#   "Metrics reloaded: recommendations for image analysis validation."
#   Nature Methods, 21(2), 195-212, 2024. DOI: 10.1038/s41592-023-02151-z
#
# Two-pass strategy
# -----------------
# Pass 1: predict() on full image -> binary mask -> connected components.
#   Filatov & Shiryaev, ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
# Pass 2 (v5 raw-crop path, unchanged): for each component, padded ROI crop
#   -> reflect-pad to model_size -> predict_fn.  Context is preserved.
#   He et al., Mask R-CNN, ICCV 2017, DOI: 10.1109/ICCV.2017.322
# Fusion (v7 new): adaptive_boundary -- auto-param tight gate from P1 EDT.
#   Isensee F. et al. Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
#   Ke L. et al. NeurIPS 2023, arXiv: 2306.01567
#   Kamnitsas K. et al. Medical Image Analysis 2017,
#     DOI: 10.1016/j.media.2016.10.004
#   Maurer C.R. et al. IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
#   Shit S. et al. CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
#
# References (full list)
# ----------------------
# Dice L.R. Ecology 1945, DOI: 10.2307/1932409
# Cheng B. et al. CVPR 2021, arXiv: 2103.16562
# Shit S. et al. CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629,
#   arXiv: 2003.07311
# Huttenlocher D.P. et al. IEEE TPAMI 1993, DOI: 10.1109/34.232073
# Maier-Hein L., Reinke A. et al. Nature Methods 2024,
#   DOI: 10.1038/s41592-023-02151-z
# Isensee F. et al. Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
# Ke L. et al. NeurIPS 2023, arXiv: 2306.01567
# Kamnitsas K. et al. Medical Image Analysis 2017,
#   DOI: 10.1016/j.media.2016.10.004
# Maurer C.R. et al. IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
# Ronneberger O. et al. U-Net MICCAI 2015,
#   DOI: 10.1007/978-3-319-24574-4_28
# He K. et al. Mask R-CNN ICCV 2017, DOI: 10.1109/ICCV.2017.322
# Ren S. et al. Faster R-CNN NeurIPS 2015, DOI: 10.1109/TPAMI.2016.2577031
# Liu W. et al. SSD ECCV 2016, DOI: 10.1007/978-3-319-46448-0_2
# Long J. et al. FCN CVPR 2015, DOI: 10.1109/CVPR.2015.7298965
# Innamorati C. et al. BMVC 2018, arXiv: 1805.03106
# Filatov A., Shiryaev E. ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
# Reinke A. et al. arXiv: 2206.01653
# =============================================================================

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as sp_ndimage
from skimage.morphology import skeletonize as sk_skeletonize


# =============================================================================
# dice
# Reference: Dice L.R., Ecology, 26(3), 297-302, 1945. DOI: 10.2307/1932409
# =============================================================================
def dice(pred, target, k=255):
    """
    Sorensen-Dice coefficient between two uint8 masks.

    Parameters
    ----------
    pred   : np.ndarray (H x W), predicted mask, uint8
    target : np.ndarray (H x W), ground-truth mask, uint8
    k      : int, foreground pixel value (default 255)

    Returns
    -------
    float in [0, 1]
    """
    pred_bin   = (pred   == k).astype(np.float32)
    target_bin = (target == k).astype(np.float32)
    intersection = np.sum(pred_bin * target_bin)
    denominator  = np.sum(pred_bin) + np.sum(target_bin)
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


# =============================================================================
# boundary_dice  (degeneration-safe)
#
# References:
#   Cheng B. et al. CVPR 2021, pp. 15334-15342. arXiv: 2103.16562
#   Maier-Hein L., Reinke A. et al. Nature Methods 2024.
#   DOI: 10.1038/s41592-023-02151-z
#   Dice L.R. Ecology, 26(3), 297-302, 1945. DOI: 10.2307/1932409
# =============================================================================
def boundary_dice(pred, target, k=255, dilation_ratio=0.02):
    """
    Boundary Dice (F1) coefficient -- degeneration-safe version.

    Computes Dice between the boundary strips of pred and target.
    d = min(max(1, round(dilation_ratio * sqrt(H^2+W^2))),
            max(1, min_bbox_dim // 2))

    Parameters
    ----------
    pred           : np.ndarray (H x W), uint8
    target         : np.ndarray (H x W), uint8
    k              : int, foreground pixel value (default 255)
    dilation_ratio : float (default 0.02)

    Returns
    -------
    float in [0, 1]
    """
    pred_bin   = (pred   == k).astype(np.uint8)
    target_bin = (target == k).astype(np.uint8)

    pred_empty   = pred_bin.sum()   == 0
    target_empty = target_bin.sum() == 0
    if pred_empty and target_empty:
        return 1.0
    if pred_empty or target_empty:
        return 0.0

    H, W     = pred_bin.shape[:2]
    img_diag = np.sqrt(float(H) ** 2 + float(W) ** 2)
    d_img    = max(1, int(round(dilation_ratio * img_diag)))

    def _min_bbox_dim(mask_bin):
        ys, xs = np.where(mask_bin)
        if len(ys) == 0:
            return 1
        return min(int(ys.max() - ys.min()) + 1,
                   int(xs.max() - xs.min()) + 1)

    min_bbox = min(_min_bbox_dim(pred_bin), _min_bbox_dim(target_bin))
    d_cap    = max(1, min_bbox // 2)
    d        = min(d_img, d_cap)

    if d < d_img:
        print(
            f"DEBUG boundary_dice: d capped {d_img} -> {d} px "
            f"(min_bbox_dim={min_bbox}, image_diag={img_diag:.1f})."
        )

    kernel = np.ones((3, 3), dtype=np.uint8)

    def _mask_to_boundary(mask_bin):
        padded = cv2.copyMakeBorder(
            mask_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        eroded = cv2.erode(padded, kernel, iterations=d)
        eroded = eroded[1:H + 1, 1:W + 1]
        return mask_bin - eroded

    boundary_pred   = _mask_to_boundary(pred_bin)
    boundary_target = _mask_to_boundary(target_bin)

    if boundary_pred.sum() == 0 or boundary_target.sum() == 0:
        print(
            f"DEBUG boundary_dice: boundary strip empty after erosion "
            f"(d={d}). Falling back to standard Dice."
        )
        return dice(pred, target, k=k)

    intersection = float(np.logical_and(boundary_pred > 0,
                                        boundary_target > 0).sum())
    denominator  = float(boundary_pred.sum() + boundary_target.sum())
    if denominator == 0:
        return dice(pred, target, k=k)
    return 2.0 * intersection / denominator


# =============================================================================
# cl_dice  (centerline Dice)
#
# References:
#   Shit S. et al. CVPR 2021, pp. 16560-16569.
#   DOI: 10.1109/CVPR46437.2021.01629  arXiv: 2003.07311
#   Maier-Hein L., Reinke A. et al. Nature Methods 2024.
#   DOI: 10.1038/s41592-023-02151-z
# =============================================================================
def cl_dice(pred, target, k=255):
    """
    centerline Dice (clDice) coefficient.

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    where Tprec = |S_P cap V_L| / |S_P|
          Tsens = |S_L cap V_P| / |S_L|

    Parameters
    ----------
    pred   : np.ndarray (H x W), uint8
    target : np.ndarray (H x W), uint8
    k      : int, foreground pixel value (default 255)

    Returns
    -------
    float in [0, 1]
    """
    pred_bin   = (pred   == k).astype(bool)
    target_bin = (target == k).astype(bool)

    pred_empty   = not pred_bin.any()
    target_empty = not target_bin.any()
    if pred_empty and target_empty:
        return 1.0
    if pred_empty or target_empty:
        return 0.0

    skel_pred   = sk_skeletonize(pred_bin)
    skel_target = sk_skeletonize(target_bin)

    n_skel_pred = skel_pred.sum()
    tprec = (
        float(np.logical_and(skel_pred, target_bin).sum()) / float(n_skel_pred)
        if n_skel_pred > 0 else 0.0
    )

    n_skel_target = skel_target.sum()
    tsens = (
        float(np.logical_and(skel_target, pred_bin).sum()) / float(n_skel_target)
        if n_skel_target > 0 else 0.0
    )

    denominator = tprec + tsens
    if denominator == 0.0:
        return 0.0
    return float(2.0 * tprec * tsens / denominator)


# =============================================================================
# hd95  (95th-percentile Hausdorff distance)
#
# References:
#   Huttenlocher D.P. et al. IEEE TPAMI, 15(9), 850-863, 1993.
#   DOI: 10.1109/34.232073
#   Maier-Hein L., Reinke A. et al. Nature Methods 2024.
#   DOI: 10.1038/s41592-023-02151-z
#   Isensee F. et al. Nature Methods 2021.
#   DOI: 10.1038/s41592-020-01008-z
# =============================================================================
def hd95(pred, target, k=255, percentile=95):
    """
    95th-percentile Hausdorff distance between two binary uint8 masks.

    Parameters
    ----------
    pred       : np.ndarray (H x W), predicted mask, uint8
    target     : np.ndarray (H x W), ground-truth mask, uint8
    k          : int, foreground pixel value (default 255)
    percentile : float in (0, 100] (default 95)

    Returns
    -------
    float >= 0 in pixels, or np.inf when one mask is empty.
    """
    pred_bin   = (pred   == k).astype(np.uint8)
    target_bin = (target == k).astype(np.uint8)

    pred_empty   = pred_bin.sum()   == 0
    target_empty = target_bin.sum() == 0
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return np.inf

    kernel = np.ones((3, 3), dtype=np.uint8)
    H, W   = pred_bin.shape[:2]

    def _get_boundary(mask_bin):
        padded = cv2.copyMakeBorder(
            mask_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        eroded = cv2.erode(padded, kernel, iterations=1)
        eroded = eroded[1:H + 1, 1:W + 1]
        return (mask_bin - eroded).astype(bool)

    boundary_pred   = _get_boundary(pred_bin)
    boundary_target = _get_boundary(target_bin)

    dist_to_boundary_target = sp_ndimage.distance_transform_edt(~boundary_target)
    dist_to_boundary_pred   = sp_ndimage.distance_transform_edt(~boundary_pred)

    d_pred_to_target = dist_to_boundary_target[boundary_pred]
    d_target_to_pred = dist_to_boundary_pred[boundary_target]

    return max(
        float(np.percentile(d_pred_to_target, percentile)),
        float(np.percentile(d_target_to_pred, percentile)),
    )


# =============================================================================
# Helper 1 - _expand_roi_to_min_size
# Reference: Ronneberger et al. U-Net MICCAI 2015,
#   DOI: 10.1007/978-3-319-24574-4_28
# =============================================================================
def _expand_roi_to_min_size(image_shape, y1, y2, x1, x2, min_h, min_w):
    """Expand ROI so the window is >= (min_h, min_w) using real image space."""
    H, W = image_shape[:2]
    deficit = min_h - (y2 - y1)
    if deficit > 0:
        et = deficit // 2;  eb = deficit - et
        y1 = max(0, y1 - et);  y2 = min(H, y2 + eb)
        remaining = min_h - (y2 - y1)
        if remaining > 0:
            if y1 == 0: y2 = min(H, y2 + remaining)
            else:        y1 = max(0, y1 - remaining)
    deficit = min_w - (x2 - x1)
    if deficit > 0:
        el = deficit // 2;  er = deficit - el
        x1 = max(0, x1 - el);  x2 = min(W, x2 + er)
        remaining = min_w - (x2 - x1)
        if remaining > 0:
            if x1 == 0: x2 = min(W, x2 + remaining)
            else:        x1 = max(0, x1 - remaining)
    return y1, y2, x1, x2



# =============================================================================
# Helper 2 - _reflect_pad_to_model_size
# Reference: Innamorati et al. BMVC 2018, arXiv: 1805.03106
# =============================================================================
def _reflect_pad_to_model_size(crop, model_size):
    """Reflect-pad crop to at least (model_size, model_size)."""
    ch, cw = crop.shape[:2]
    pt = max(0, model_size - ch) // 2;  pb = max(0, model_size - ch) - pt
    pl = max(0, model_size - cw) // 2;  pr = max(0, model_size - cw) - pl
    if pt > 0 or pb > 0 or pl > 0 or pr > 0:
        pw = ((pt, pb), (pl, pr), (0, 0)) if crop.ndim == 3 else ((pt, pb), (pl, pr))
        crop = np.pad(crop, pw, mode="reflect")
    return crop, pt, pl


# =============================================================================
# Helper 3 - _expand_roi_to_min_image_fraction
# Reference: Ren et al. Faster R-CNN NeurIPS 2015,
#   DOI: 10.1109/TPAMI.2016.2577031
# =============================================================================
def _expand_roi_to_min_image_fraction(image_shape, y1, y2, x1, x2,
                                       min_roi_fraction):
    """Expand ROI to cover at least min_roi_fraction of the image axes."""
    H, W  = image_shape[:2]
    min_h = int(np.ceil(min_roi_fraction * H))
    min_w = int(np.ceil(min_roi_fraction * W))
    if (y2 - y1) < min_h:
        centre_y = (y1 + y2) // 2;  half = min_h // 2
        y1       = max(0, centre_y - half);  y2 = min(H, y1 + min_h)
        y1       = max(0, y2 - min_h)
    if (x2 - x1) < min_w:
        centre_x = (x1 + x2) // 2;  half = min_w // 2
        x1       = max(0, centre_x - half);  x2 = min(W, x1 + min_w)
        x1       = max(0, x2 - min_w)
    return y1, y2, x1, x2


# =============================================================================
# Helper 4 - _run_single_component_pass2  (v5 raw-crop path, unchanged)
#
# IMPORTANT: this is the ORIGINAL v5 path.
# The crop is sent directly to predict_fn after reflect-padding to model_size.
# The padded ROI (padding_ratio=0.50) retains enough background context for
# the model to suppress reflections.  Do NOT replace with zoom-resize.
#
# References:
#   Liu et al. SSD ECCV 2016, DOI: 10.1007/978-3-319-46448-0_2
#   Ren et al. Faster R-CNN NeurIPS 2015, DOI: 10.1109/TPAMI.2016.2577031
#   He et al. Mask R-CNN ICCV 2017, DOI: 10.1109/ICCV.2017.322
#   Long et al. FCN CVPR 2015, DOI: 10.1109/CVPR.2015.7298965
# =============================================================================
def _run_single_component_pass2(
    image, component_coords, padding_ratio, model_size,
    predict_fn, model, processor, device,
    min_pad_pixels=32, min_roi_fraction=0.0,
):
    """
    Run Pass-2 inference for one connected component (v5 raw-crop path).

    The padded ROI crop is sent directly to predict_fn after reflect-padding
    to at least (model_size, model_size).  The large padding (default 0.50)
    preserves background context that allows the model to distinguish the
    target object from reflections or similar-looking neighbours.

    Parameters
    ----------
    image            : np.ndarray (H x W x 3), uint8, full image
    component_coords : tuple (row_indices, col_indices) from np.where()
    padding_ratio    : float, fractional bbox padding on each side.
                       Keep >= 0.50 to include enough background context.
    model_size       : int, minimum crop size (reflect-padded if smaller)
    predict_fn       : callable(model, processor, device, image) -> np.ndarray
    model            : segmentation model
    processor        : HuggingFace AutoImageProcessor
    device           : torch.device
    min_pad_pixels   : int, absolute minimum padding per side in pixels
    min_roi_fraction : float, minimum ROI as fraction of the image axes

    Returns
    -------
    component_full : np.ndarray (H x W), uint8
    debug_info     : dict
    """
    H, W   = image.shape[:2]
    y_min  = int(component_coords[0].min())
    y_max  = int(component_coords[0].max())
    x_min  = int(component_coords[1].min())
    x_max  = int(component_coords[1].max())
    box_h  = max(1, y_max - y_min)
    box_w  = max(1, x_max - x_min)

    pad_h  = max(int(box_h * padding_ratio), min_pad_pixels)
    pad_w  = max(int(box_w * padding_ratio), min_pad_pixels)
    roi_y1 = max(0, y_min - pad_h);  roi_y2 = min(H, y_max + pad_h)
    roi_x1 = max(0, x_min - pad_w);  roi_x2 = min(W, x_max + pad_w)

    roi_y1, roi_y2, roi_x1, roi_x2 = _expand_roi_to_min_image_fraction(
        image.shape, roi_y1, roi_y2, roi_x1, roi_x2, min_roi_fraction
    )
    roi_h_before = roi_y2 - roi_y1;  roi_w_before = roi_x2 - roi_x1

    roi_y1, roi_y2, roi_x1, roi_x2 = _expand_roi_to_min_size(
        image.shape, roi_y1, roi_y2, roi_x1, roi_x2, model_size, model_size
    )
    roi_h_after = roi_y2 - roi_y1;  roi_w_after = roi_x2 - roi_x1
    context_fill_triggered = (
        (roi_h_after != roi_h_before) or (roi_w_after != roi_w_before)
    )

    roi_crop = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    roi_crop, pad_top, pad_left = _reflect_pad_to_model_size(roi_crop, model_size)

    comp_crop_mask = predict_fn(model, processor, device, roi_crop)

    expected_h, expected_w = roi_crop.shape[:2]
    if comp_crop_mask.shape[:2] != (expected_h, expected_w):
        # Our own resize: INTER_LINEAR + threshold.
        # Handles the case where predict_fn returns a mask at a different
        # spatial size than the crop (some processor configurations).
        # Reference: Ronneberger O. et al., U-Net, MICCAI 2015.
        # DOI: 10.1007/978-3-319-24574-4_28
        comp_crop_mask_f = cv2.resize(
            comp_crop_mask.astype(np.float32),
            (expected_w, expected_h),
            interpolation=cv2.INTER_LINEAR,
        )
        comp_crop_mask = (comp_crop_mask_f >= 127.0).astype(np.uint8) * 255


    component_full = np.zeros((H, W), dtype=np.uint8)
    offset_y = roi_y1 - pad_top;  offset_x = roi_x1 - pad_left
    dst_y1 = max(0, offset_y);  dst_y2 = min(H, offset_y + comp_crop_mask.shape[0])
    dst_x1 = max(0, offset_x);  dst_x2 = min(W, offset_x + comp_crop_mask.shape[1])
    src_y1 = dst_y1 - offset_y;  src_y2 = dst_y2 - offset_y
    src_x1 = dst_x1 - offset_x;  src_x2 = dst_x2 - offset_x
    component_full[dst_y1:dst_y2, dst_x1:dst_x2] = (
        comp_crop_mask[src_y1:src_y2, src_x1:src_x2]
    )

    debug_info = {
        "bbox_pass1":             (y_min, y_max, x_min, x_max),
        "padded_roi":             (roi_y1, roi_y2, roi_x1, roi_x2),
        "roi_size_before_expand": (roi_h_before, roi_w_before),
        "roi_size_after_expand":  (roi_h_after,  roi_w_after),
        "context_fill_triggered": context_fill_triggered,
        "crop_shape":             roi_crop.shape,
        "crop_mask_shape":        comp_crop_mask.shape,
        "filled_crop_image":      roi_crop,
        "second_pass_crop_mask":  comp_crop_mask,
    }
    return component_full, debug_info


# =============================================================================
# _per_component_two_pass_predict  (unchanged from v5)
#
# References:
#   Filatov & Shiryaev ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
#   He et al. Mask R-CNN ICCV 2017, DOI: 10.1109/ICCV.2017.322
# =============================================================================
def _per_component_two_pass_predict(
    model, processor, device, image,
    padding_ratio=0.50, model_size=320, predict_fn=None,
    ground_truth_mask=None, min_component_pixels=10,
    min_pad_pixels=32, min_roi_fraction=0.0,
):
    """Per-component context-aware two-pass inference (v5 raw-crop path)."""
    if predict_fn is None:
        raise ValueError("predict_fn must be provided.")

    H, W            = image.shape[:2]
    first_pass_mask = predict_fn(model, processor, device, image)
    binary          = (first_pass_mask > 127).astype(np.uint8)

    if binary.sum() == 0:
        return first_pass_mask, {
            "note":            "no ROI detected in first pass",
            "first_pass_mask": first_pass_mask,
            "num_components":  0,
            "components":      [],
        }

    # Filatov & Shiryaev, ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
    labeled_array, num_features = sp_ndimage.label(binary)
    print(f"DEBUG two-pass: {num_features} component(s) detected in Pass-1.")

    second_pass_full = np.zeros((H, W), dtype=np.uint8)
    components_debug = []

    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_array == comp_id).astype(np.uint8)
        n_pixels  = int(comp_mask.sum())
        if n_pixels < min_component_pixels:
            print(
                f"DEBUG two-pass: component {comp_id} skipped "
                f"({n_pixels} px < {min_component_pixels} threshold)."
            )
            continue

        comp_coords = np.where(comp_mask > 0)
        comp_full, comp_debug = _run_single_component_pass2(
            image=image,
            component_coords=comp_coords,
            padding_ratio=padding_ratio,
            model_size=model_size,
            predict_fn=predict_fn,
            model=model,
            processor=processor,
            device=device,
            min_pad_pixels=min_pad_pixels,
            min_roi_fraction=min_roi_fraction,
        )
        comp_debug["component_id"]     = comp_id
        comp_debug["component_pixels"] = n_pixels
        components_debug.append(comp_debug)
        second_pass_full = np.maximum(second_pass_full, comp_full)

        print(
            f"DEBUG two-pass: component {comp_id}/{num_features} | "
            f"pixels={n_pixels} | "
            f"roi_before={comp_debug['roi_size_before_expand']} | "
            f"roi_after={comp_debug['roi_size_after_expand']} | "
            f"context_fill={comp_debug['context_fill_triggered']} | "
            f"crop_shape={comp_debug['crop_shape']} | "
            f"crop_mask_shape={comp_debug['crop_mask_shape']}"
        )

    if second_pass_full.sum() == 0 and first_pass_mask.sum() > 0:
        print("DEBUG two-pass: Pass-2 empty. Falling back to Pass-1 mask.")
        second_pass_full = first_pass_mask.copy()

    gt_crop = None
    if ground_truth_mask is not None and len(components_debug) > 0:
        all_coords = np.where(binary > 0)
        gy1 = int(all_coords[0].min());  gy2 = int(all_coords[0].max())
        gx1 = int(all_coords[1].min());  gx2 = int(all_coords[1].max())
        gt_crop = ground_truth_mask[gy1:gy2, gx1:gx2].copy()

    debug = {
        "first_pass_mask":               first_pass_mask,
        "num_components":                num_features,
        "components":                    components_debug,
        "filled_crop_image":             components_debug[0]["filled_crop_image"]
                                         if components_debug else None,
        "second_pass_crop_mask":         components_debug[0]["second_pass_crop_mask"]
                                         if components_debug else None,
        "filled_crop_ground_truth_mask": gt_crop,
        "context_fill_triggered":        any(
                                             c["context_fill_triggered"]
                                             for c in components_debug
                                         ),
        "roi_size_before_expand":        components_debug[0]["roi_size_before_expand"]
                                         if components_debug else None,
        "roi_size_after_expand":         components_debug[0]["roi_size_after_expand"]
                                         if components_debug else None,
    }
    return second_pass_full, debug


# =============================================================================
# _keep_largest_component
#
# References:
#   Isensee F. et al. Nature Methods 18, 203-211, 2021.
#   DOI: 10.1038/s41592-020-01008-z
#   Filatov A., Shiryaev E. ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
# =============================================================================
def _keep_largest_component(mask, k=255):
    """
    Keep only the largest connected component in a binary uint8 mask.
    Modified in-place.  Returns (mask, n_removed).
    """
    binary = (mask == k).astype(np.uint8)
    if binary.sum() == 0:
        return mask, 0

    labeled_array, num_features = sp_ndimage.label(binary)
    if num_features <= 1:
        return mask, 0

    label_sizes    = np.bincount(labeled_array.ravel())
    label_sizes[0] = 0
    largest_label  = int(label_sizes.argmax())
    mask[labeled_array != largest_label] = 0

    n_removed = num_features - 1
    print(
        f"DEBUG keep_largest_component: {num_features} component(s) found. "
        f"Kept label {largest_label} "
        f"({int(label_sizes[largest_label])} px). "
        f"Removed {n_removed} component(s)."
    )
    return mask, n_removed


# =============================================================================
# _constrained_union_fusion  (v6, kept for backward compatibility)
#
# DEPRECATED -- use _adaptive_boundary_zone_fusion.
# Limitation confirmed: fixed radius r=54px covered entire P2 mask
# (P2_blocked=0%), making the gate useless on the tested 1080x1440 image
# with a thin elongated worm component.
#
# References:
#   Isensee F. et al. Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
#   Ke L. et al. NeurIPS 2023, arXiv: 2306.01567
#   Kamnitsas K. et al. Medical Image Analysis 2017,
#     DOI: 10.1016/j.media.2016.10.004
# =============================================================================
def _constrained_union_fusion(
    pass1_mask, pass2_full,
    dilation_radius=None, dilation_fraction=0.05, k=255,
):
    """
    [DEPRECATED in v7 -- use _adaptive_boundary_zone_fusion]
    final = P1 UNION (P2 AND dilate(P1, r))
    """
    p1 = (pass1_mask == k).astype(np.uint8)
    p2 = (pass2_full == k).astype(np.uint8)

    if p1.sum() == 0:
        print("DEBUG constrained_union: P1 empty, returning P2 unchanged.")
        return pass2_full.copy()

    H, W = p1.shape[:2]
    if dilation_radius is None:
        dilation_radius = max(3, int(round(dilation_fraction * min(H, W))))

    ksize  = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    gate   = cv2.dilate(p1, kernel, iterations=1)

    p2_c       = np.logical_and(p2 > 0, gate > 0).astype(np.uint8)
    final      = np.logical_or(p1 > 0, p2_c > 0).astype(np.uint8)
    final      = (final * k).astype(np.uint8)
    p2_blocked = int(p2.sum()) - int(p2_c.sum())
    added_px   = int(p2_c.sum()) - int(np.logical_and(p2_c > 0, p1 > 0).sum())

    print(
        f"DEBUG constrained_union: r={dilation_radius}px | "
        f"P1={int(p1.sum())}px | P2_raw={int(p2.sum())}px | "
        f"P2_blocked={p2_blocked}px "
        f"({100.0 * p2_blocked / max(1, int(p2.sum())):.1f}%) | "
        f"net_added={added_px}px | final={int((final==k).sum())}px"
    )
    return final


# =============================================================================
# _adaptive_boundary_zone_fusion  (v7 -- replaces constrained_union_fusion)
#
# Why _constrained_union_fusion failed
# -------------------------------------
# Fixed radius r = max(3, round(dilation_fraction * min(H,W))).
# For the tested image (1080x1440) with dilation_fraction=0.05:
#   r = max(3, round(0.05 * 1080)) = 54px.
# The elongated worm is ~6px wide (typical_radius ~ 3px).
# A 54px dilation gate covers a ~108px wide band around P1, which is 18x the
# worm width.  Every P2 pixel was inside the gate: P2_blocked=0%.
# The fusion was therefore equivalent to raw P1 OR P2 -- identical to v5.
#
# Adaptive strategy
# -----------------
# All gate radii are derived from P1's structure so no tuning is needed.
#
# Step 1: Euclidean distance transform of P1.
#   DT(x) = distance from pixel x to the nearest background pixel.
#   Reference:
#     Maurer C.R., Qi R., Raghavan V.
#     "A linear time algorithm for computing exact Euclidean distance
#     transforms of binary images in arbitrary dimensions."
#     IEEE TPAMI 25(2), 265-270, 2003. DOI: 10.1109/TPAMI.2003.1177156
#
# Step 2: Medial axis (skeleton) of P1.
#   Skeleton pixels carry the locally maximum DT value = feature half-width.
#   Reference: Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
#
# Step 3: typical_radius = median(DT at skeleton pixels).
#   For the thin worm: typical_radius ~ 3-6px (vs 54px fixed in v6).
#
# Step 4: auto-derived gate radii.
#   r_outer = max(2, round(0.25 * typical_radius))
#     Thin worm (typical_radius=6): r_outer=2px.  Gate width=4px total.
#     Thick organ (typical_radius=20): r_outer=5px.
#     This is tight enough to block reflection hallucinations (which are
#     typically 10-50px from the true worm boundary) while still accepting
#     1-2px thin branches that P1 missed.
#
#   r_core = max(0, round(0.75 * typical_radius - 1))
#     Thin worm: r_core=3px.  Pixels at DT > 3px are frozen (always in
#     the output regardless of P2), preserving the P1 interior exactly.
#     Thick organ: r_core=14px.
#
# Step 5: final = core OR (P2 AND dilate(P1, r_outer))
#   core: erode(P1, r_core) -- frozen interior.
#   P2 gate: accept only P2 pixels within r_outer of the P1 boundary.
#
# Thin-feature safety: if erosion destroys core entirely (r_core >= half-width),
# fall back to skeleton dilation (1px) clipped to P1 to guarantee the medial
# axis is always preserved.
#
# Expected outcome for the worm case:
#   typical_radius~3-6px -> r_outer=2px -> gate covers a 4px-wide band around P1.
#   Reflection is ~15-50px from the worm boundary -> BLOCKED.
#   Missed thin branch pixels along the worm edge -> ACCEPTED.
#   P2_blocked >> 0% (meaningful filtering, unlike v6's 0%).
#
# References:
#   Isensee F. et al., Nature Methods 18, 203-211, 2021.
#   DOI: 10.1038/s41592-020-01008-z
#   Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
#   Kamnitsas K. et al., Medical Image Analysis 36, 61-78, 2017.
#   DOI: 10.1016/j.media.2016.10.004
#   Maurer C.R. et al., IEEE TPAMI 25(2), 265-270, 2003.
#   DOI: 10.1109/TPAMI.2003.1177156
#   Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
# =============================================================================
def _adaptive_boundary_zone_fusion(pass1_mask, pass2_full,
                                    boundary_dilation_ratio=0.02, k=255):
    """
    Adaptive boundary zone fusion with automatic gate radii (v7).

    Derives r_outer and r_core from P1's Euclidean distance transform
    evaluated at skeleton pixels.  For a thin elongated worm (~6px wide),
    r_outer=2px gives a 4px-wide gate that blocks reflections (>15px away)
    while accepting missed thin-branch pixels along the worm edge.

    final = core OR (P2 AND dilate(P1, r_outer))
    where core = erode(P1, r_core)  [or skeleton fallback for thin features]

    Parameters
    ----------
    pass1_mask             : np.ndarray (H x W), uint8, Pass-1 mask
    pass2_full             : np.ndarray (H x W), uint8, Pass-2 union mask
    boundary_dilation_ratio: float (default 0.02), printed for diagnostics
                             only; not used in gate computation.
    k                      : int, foreground pixel value (default 255)

    Returns
    -------
    np.ndarray (H x W), uint8

    References
    ----------
    Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
    Kamnitsas K. et al., Medical Image Analysis 2017,
      DOI: 10.1016/j.media.2016.10.004
    Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    """
    p1 = (pass1_mask == k).astype(np.uint8)
    p2 = (pass2_full == k).astype(np.uint8)

    if p1.sum() == 0:
        print("DEBUG adaptive_boundary_fusion: P1 empty, returning P2 unchanged.")
        return pass2_full.copy()

    # Step 1 -- EDT of P1.
    # Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    dist_transform = cv2.distanceTransform(p1, cv2.DIST_L2, 5)

    # Step 2 -- Medial axis.
    # Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    skel = sk_skeletonize(p1.astype(bool))

    # Step 3 -- Typical feature half-width.
    if skel.any():
        typical_radius = float(np.median(dist_transform[skel]))
    else:
        dist_vals      = dist_transform[p1 > 0]
        typical_radius = float(np.median(dist_vals)) if len(dist_vals) > 0 else 3.0
    typical_radius = max(1.0, typical_radius)

    # Step 4 -- Gate radii.
    # Ke L. et al., HQ-SAM, NeurIPS 2023. arXiv: 2306.01567
    # Kamnitsas K. et al., Medical Image Analysis 2017,
    #   DOI: 10.1016/j.media.2016.10.004
    r_outer = max(2, int(round(0.25 * typical_radius)))
    r_core  = max(0, int(round(0.75 * typical_radius - 1.0)))

    # Step 5a -- Outer gate (elliptical kernel avoids diagonal bias).
    # Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    ksize_out = 2 * r_outer + 1
    kern_out  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_out, ksize_out))
    gate      = cv2.dilate(p1, kern_out, iterations=1)

    # Step 5b -- Frozen core.
    if r_core > 0:
        ksize_in = 2 * r_core + 1
        kern_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_in, ksize_in))
        core     = cv2.erode(p1, kern_in, iterations=1)
    else:
        core = p1.copy()

    # Thin-feature safety: if erosion destroyed core, use skeleton dilation.
    # Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    if core.sum() == 0:
        skel_u8 = skel.astype(np.uint8)
        if skel_u8.sum() > 0:
            kern_skel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            core      = cv2.dilate(skel_u8, kern_skel, iterations=1)
            core      = np.minimum(core, p1)
        else:
            core = p1.copy()

    # Step 5c -- Final union.
    p2_c  = np.logical_and(p2 > 0, gate > 0).astype(np.uint8)
    final = np.logical_or(core > 0, p2_c > 0).astype(np.uint8)
    final = (final * k).astype(np.uint8)

    p1_px      = int(p1.sum())
    p2_raw     = int(p2.sum())
    p2_valid   = int(p2_c.sum())
    p2_blocked = p2_raw - p2_valid
    added_px   = p2_valid - int(np.logical_and(p2_c > 0, p1 > 0).sum())
    final_px   = int((final == k).sum())

    print(
        f"DEBUG adaptive_boundary_fusion: "
        f"typical_r={typical_radius:.1f}px | "
        f"r_outer={r_outer}px | r_core={r_core}px | "
        f"P1={p1_px}px | P2_raw={p2_raw}px | "
        f"P2_blocked={p2_blocked}px "
        f"({100.0 * p2_blocked / max(1, p2_raw):.1f}%) | "
        f"net_added={added_px}px | final={final_px}px"
    )
    return final


# =============================================================================
# _spatially_adaptive_fusion  (v8 -- replaces single-radius strategies)
#
# Motivation
# ----------
# Single-radius fusion strategies fail on worms that contain both thick and
# thin regions:
#   Tight gate (r~2px, adaptive_boundary):
#     Blocks reflection hallucinations on thick parts -- good Dice.
#     Also blocks genuine thin-branch P2 recoveries -- bad clDice.
#   No gate (replace):
#     Accepts all P2 thin-branch additions -- good clDice.
#     Also accepts P2 reflection hallucinations on thick parts -- bad Dice.
#
# Solution: spatially varying gate width derived from P1's distance transform
# ---------------------------------------------------------------------------
# The Euclidean distance transform (EDT) of P1 gives, at each foreground
# pixel, the distance to the nearest background pixel.  This equals the
# local feature half-width.  Pixels with small EDT are in thin regions;
# pixels with large EDT are in thick regions.  We split P1 at a threshold
# derived from the structure's own statistics, then dilate each part with
# an appropriate radius:
#
#   thin_mask      = P1 pixels where EDT <= thin_threshold
#   thick_mask     = P1 pixels where EDT >  thin_threshold
#   thin_threshold = 0.75 * typical_radius  (auto, from skeleton median EDT)
#
#   r_outer_thin   = max(4, round(typical_radius))
#     Wide gate for thin regions: allows P2 to add entire missed thin branches
#     (which are typically 1 * typical_radius wide) while still blocking
#     reflections that are typically >> typical_radius away.
#
#   r_outer_thick  = max(2, round(0.25 * typical_radius))
#     Tight gate for thick regions: same as _adaptive_boundary_zone_fusion.
#     Blocks P2 boundary hallucinations and reflection pixels (>15px away)
#     while accepting 1-2px surface corrections.
#
#   gate  = dilate(thin_mask,  ellipse(r_outer_thin))
#        OR dilate(thick_mask, ellipse(r_outer_thick))
#   final = P1 OR (P2 AND gate)
#
# P1 is always preserved by construction (OR operation, no erosion of P1).
#
# Example for typical thin worm (typical_radius=6px):
#   thin_threshold = 4.5px   -> pixels narrower than 4.5px half-width are thin
#   r_outer_thin   = 6px     -> gate covers a 12px band around thin parts
#   r_outer_thick  = 2px     -> gate covers a  4px band around thick parts
#   Reflection pixel 20px from thick boundary: BLOCKED
#   Missed thin branch pixel 5px from thin boundary: ACCEPTED
#
# References
# ----------
# Isensee F., Jaeger P.F., Kohl S.A.A., Petersen J., Maier-Hein K.H.
#   "nnU-Net: a self-configuring method for deep learning-based biomedical
#   image segmentation." Nature Methods 18, 203-211, 2021.
#   DOI: 10.1038/s41592-020-01008-z
#
# Ke L., Ye M., Danelljan M., Liu Y., Tai Y.W., Tang C.K., Yu F.
#   "Segment Anything in High Quality."
#   Advances in Neural Information Processing Systems 36 (NeurIPS 2023).
#   arXiv: 2306.01567
#
# Kamnitsas K., Ledig C., Newcombe V.F.J., Simpson J.P., Kane A.D.,
#   Menon D.K., Rueckert D., Glocker B.
#   "Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate
#   Brain Lesion Segmentation."
#   Medical Image Analysis 36, 61-78, 2017. DOI: 10.1016/j.media.2016.10.004
#
# Maurer C.R., Qi R., Raghavan V.
#   "A linear time algorithm for computing exact Euclidean distance transforms
#   of binary images in arbitrary dimensions."
#   IEEE TPAMI 25(2), 265-270, 2003. DOI: 10.1109/TPAMI.2003.1177156
#
# Shit S., Paetzold J.C., Sekuboyina A., Ezhov I., Unger A., Zhylka A.,
#   Pluim J.P.W., Bauer U., Menze B.H.
#   "clDice - A Novel Topology-Preserving Loss Function for Tubular Structure
#   Segmentation." CVPR 2021, pp. 16560-16569.
#   DOI: 10.1109/CVPR46437.2021.01629
# =============================================================================
def _spatially_adaptive_fusion(pass1_mask, pass2_full,
                                boundary_dilation_ratio=0.02, k=255):
    """
    Spatially varying gate fusion (v8 -- recommended).

    Splits P1 into thin and thick regions using P1's Euclidean distance
    transform, then dilates each region with a proportional gate radius.
    Thick regions receive a tight gate (blocks P2 deformations and
    reflections); thin regions receive a wide gate (recovers missed branches).

    final = P1 OR (P2 AND gate)

    where gate = dilate(thin_mask, r_outer_thin)
              OR dilate(thick_mask, r_outer_thick)

    All parameters are derived automatically from P1's structure.
    No manual tuning is required.

    Parameters
    ----------
    pass1_mask             : np.ndarray (H x W), uint8, Pass-1 mask
    pass2_full             : np.ndarray (H x W), uint8, Pass-2 union mask
    boundary_dilation_ratio: float (default 0.02), printed for diagnostics
                             only; not used in gate computation
    k                      : int, foreground pixel value (default 255)

    Returns
    -------
    np.ndarray (H x W), uint8

    References
    ----------
    Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
    Kamnitsas K. et al., Medical Image Analysis 2017,
      DOI: 10.1016/j.media.2016.10.004
    Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    """
    p1 = (pass1_mask == k).astype(np.uint8)
    p2 = (pass2_full == k).astype(np.uint8)

    if p1.sum() == 0:
        print("DEBUG spatially_adaptive_fusion: P1 empty, returning P2 unchanged.")
        return pass2_full.copy()

    # ------------------------------------------------------------------
    # Step 1 -- Euclidean distance transform of P1.
    # EDT(x) = distance to nearest background pixel = local feature half-width.
    # Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    # ------------------------------------------------------------------
    dist_transform = cv2.distanceTransform(p1, cv2.DIST_L2, 5)

    # ------------------------------------------------------------------
    # Step 2 -- Medial axis of P1 (skeleton).
    # Skeleton pixels carry locally maximum EDT values.
    # Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    # ------------------------------------------------------------------
    skel = sk_skeletonize(p1.astype(bool))

    # ------------------------------------------------------------------
    # Step 3 -- Typical feature half-width from skeleton EDT.
    # ------------------------------------------------------------------
    if skel.any():
        typical_radius = float(np.median(dist_transform[skel]))
    else:
        dist_vals      = dist_transform[p1 > 0]
        typical_radius = float(np.median(dist_vals)) if len(dist_vals) > 0 else 3.0
    typical_radius = max(1.0, typical_radius)

    # ------------------------------------------------------------------
    # Step 4 -- Thin/thick split threshold and gate radii.
    #
    # thin_threshold: P1 pixels with EDT <= threshold are "thin region".
    #   Set to 0.75 * typical_radius so the split adapts to the structure.
    #   For typical_radius=6px: thin_threshold=4.5px.
    #
    # r_outer_thin: wide gate for thin regions.
    #   Set to typical_radius (full half-width) so the gate covers an entire
    #   missed branch width on each side.
    #   Minimum 4px to ensure at least a small recovery zone.
    #
    # r_outer_thick: tight gate for thick regions.
    #   Set to 0.25 * typical_radius (same as adaptive_boundary, v7).
    #   For typical_radius=6px: r_outer_thick=2px.
    #
    # Ke L. et al., HQ-SAM, NeurIPS 2023. arXiv: 2306.01567
    # Kamnitsas K. et al., Medical Image Analysis 2017,
    #   DOI: 10.1016/j.media.2016.10.004
    # ------------------------------------------------------------------
    thin_threshold = 0.75 * typical_radius
    r_outer_thin   = max(4, int(round(typical_radius)))
    r_outer_thick  = max(2, int(round(0.25 * typical_radius)))

    # ------------------------------------------------------------------
    # Step 5 -- Build thin and thick masks from P1.
    # ------------------------------------------------------------------
    thin_mask  = ((p1 > 0) & (dist_transform <= thin_threshold)).astype(np.uint8)
    thick_mask = ((p1 > 0) & (dist_transform  > thin_threshold)).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 6 -- Build spatially varying gate.
    # Each part dilated independently with its own elliptical kernel.
    # Elliptical kernel avoids diagonal bias of square kernels.
    # Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    # ------------------------------------------------------------------
    ksize_thin  = 2 * r_outer_thin  + 1
    ksize_thick = 2 * r_outer_thick + 1
    kern_thin   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (ksize_thin,  ksize_thin))
    kern_thick  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (ksize_thick, ksize_thick))

    gate_thin   = cv2.dilate(thin_mask,  kern_thin,  iterations=1)
    gate_thick  = cv2.dilate(thick_mask, kern_thick, iterations=1)
    gate        = np.logical_or(gate_thin > 0, gate_thick > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 7 -- Final: P1 always preserved, P2 accepted only within gate.
    # final = P1 OR (P2 AND gate)
    # P1 is never eroded here (no core step), so no thin-feature safety
    # branch is needed: the OR guarantees all P1 pixels survive.
    # ------------------------------------------------------------------
    p2_c  = np.logical_and(p2 > 0, gate > 0).astype(np.uint8)
    final = np.logical_or(p1 > 0, p2_c > 0).astype(np.uint8)
    final = (final * k).astype(np.uint8)

    # Debug statistics
    p1_px         = int(p1.sum())
    p2_raw        = int(p2.sum())
    p2_valid      = int(p2_c.sum())
    p2_blocked    = p2_raw - p2_valid
    added_px      = p2_valid - int(np.logical_and(p2_c > 0, p1 > 0).sum())
    final_px      = int((final == k).sum())
    n_thin_px     = int(thin_mask.sum())
    n_thick_px    = int(thick_mask.sum())

    print(
        f"DEBUG spatially_adaptive_fusion: "
        f"typical_r={typical_radius:.1f}px | "
        f"thin_thr={thin_threshold:.1f}px | "
        f"r_outer_thin={r_outer_thin}px | r_outer_thick={r_outer_thick}px | "
        f"thin_px={n_thin_px} | thick_px={n_thick_px} | "
        f"P1={p1_px}px | P2_raw={p2_raw}px | "
        f"P2_blocked={p2_blocked}px "
        f"({100.0 * p2_blocked / max(1, p2_raw):.1f}%) | "
        f"net_added={added_px}px | final={final_px}px"
    )
    return final


# =============================================================================
# _skeleton_guided_fusion  (v9 -- replaces all radius-based strategies)
#
# Why radius-based gates fail on thin branches
# --------------------------------------------
# All previous strategies (constrained_union, adaptive_boundary,
# spatially_adaptive) use "distance from P1" as the acceptance criterion
# for P2 pixels.  For a missed thin-branch extension, this fails because:
#   - The branch can be arbitrarily long relative to its width.
#   - If P1 missed the last 68px of a 10px-wide branch, the gate must be
#     r >= 68px to accept those pixels.  But r >= 68px also accepts
#     reflection pixels 50px from P1.  No single radius works.
#
# Topological connectivity criterion (v9)
# ----------------------------------------
# The correct distinction is not DISTANCE but CONNECTIVITY:
#
#   Missed thin-branch extension:
#     P2 skeleton connects to P1 at the branch root (near P1's endpoint).
#     Even if the branch extends 68px, its skeleton passes through a pixel
#     adjacent to P1's tip, so it is "connected" to P1.
#
#   Reflection / hallucination:
#     P2 skeleton of the reflection is spatially disjoint from P1.
#     It does NOT pass near P1 at any point.
#
# This is a binary, topology-aware criterion with no free radius for branch
# length.
#
# Algorithm
# ---------
# Step 1: EDT and skeleton of P1 -> typical_radius (auto).
#
# Step 2: Skeleton of P2.
#
# Step 3: Find "connected" P2 skeleton pixels.
#   Dilate P1 by r_connect to create an attachment zone.
#   P2 skeleton pixels inside the attachment zone are "connected" to P1.
#   r_connect = max(round(typical_radius), 4)
#   This tolerance handles the 1-2px gap that can appear between P1's
#   endpoint and the first P2 skeleton pixel.
#
# Step 4: Grow a tube around connected P2 skeleton pixels.
#   r_tube = max(round(typical_radius), 4)
#   The tube covers the full width of the branch, so all P2 pixels that
#   belong to the branch are accepted, regardless of how far the branch
#   extends from P1.
#
# Step 5: Tight surface gate on P1.
#   r_tight = max(2, round(0.25 * typical_radius))
#   Accepts P2 pixels within r_tight of P1's surface: small boundary
#   micro-corrections on thick parts (same as adaptive_boundary, v7).
#
# Step 6: final = P1 OR (P2 AND (tube OR gate_tight))
#   P1 is always preserved (OR, no erosion).
#
# Edge cases
# ----------
# No connected P2 skeleton: only gate_tight is used (safe fallback).
# P2 is empty: P1 is returned unchanged.
# P1 is empty: P2 is returned unchanged (no anchor available).
#
# Expected outcome for the observed cases
# ----------------------------------------
# Thin-branch image (typical_r=10px, branch=68px):
#   r_connect=10px, r_tube=10px.
#   P2 skeleton at branch tip connects to P1 -> tube grows to cover branch.
#   Branch 68px long: accepted regardless of length.
#   Dice should recover toward replace-level (0.9038).
#   clDice should recover toward replace-level (0.9959).
#   HD95 should recover toward replace-level (4.24px).
#
# Thick-worm image with reflection (typical_r >> 10px):
#   P2 skeleton of reflection is 50px from P1 -> NOT connected.
#   r_tight=2-5px surface gate handles boundary micro-corrections only.
#   Reflection remains blocked.
#   Dice stays at adaptive_boundary-level (high).
#
# References
# ----------
# Shit S., Paetzold J.C., Sekuboyina A., Ezhov I., Unger A., Zhylka A.,
#   Pluim J.P.W., Bauer U., Menze B.H.
#   "clDice - A Novel Topology-Preserving Loss Function for Tubular
#   Structure Segmentation." CVPR 2021, pp. 16560-16569.
#   DOI: 10.1109/CVPR46437.2021.01629
#
# Maurer C.R., Qi R., Raghavan V.
#   "A linear time algorithm for computing exact Euclidean distance
#   transforms of binary images in arbitrary dimensions."
#   IEEE TPAMI 25(2), 265-270, 2003. DOI: 10.1109/TPAMI.2003.1177156
#
# Isensee F., Jaeger P.F., Kohl S.A.A., Petersen J., Maier-Hein K.H.
#   "nnU-Net: a self-configuring method for deep learning-based biomedical
#   image segmentation." Nature Methods 18, 203-211, 2021.
#   DOI: 10.1038/s41592-020-01008-z
#
# Ke L., Ye M., Danelljan M., Liu Y., Tai Y.W., Tang C.K., Yu F.
#   "Segment Anything in High Quality."
#   NeurIPS 2023. arXiv: 2306.01567
# =============================================================================
def _skeleton_guided_fusion(pass1_mask, pass2_full,
                             boundary_dilation_ratio=0.02, k=255):
    """
    Skeleton-guided fusion: reflection-aware component filter (v9 final).

    Design contract
    ----------------
    This function solves ONE problem: P2 may contain isolated spurious
    components (reflections, hallucinations) whose skeleton does NOT touch P1.
    Those components are filtered out.

    It does NOT attempt to merge P1 and P2 or recover P1 pixels that P2
    missed.  That problem is irreducible with two binary masks and no GT:
    "P1 pixel that P2 missed (TP)" is indistinguishable from "P1 pixel that
    P2 correctly rejected (FP)" without ground truth.  All attempted recovery
    strategies (P1 OR, surface-bounded P1 recovery) consistently degraded Dice
    on at least one configuration.

    Behaviour by case
    -----------------
    Single-component P2 (mask_cc=1, mask_accepted=1):
      final = P2  (identical to 'replace')

    Multi-component P2 (mask_cc > 1):
      Accept P2 mask components whose skeleton touches dilate(P1, r_connect).
      Reject isolated components (reflections, hallucinations).
      final = union of accepted P2 components

    In both cases, the boundary is P2's own smooth boundary.
    No tube, no staircase, no P1 pixels re-introduced.

    Parameters
    ----------
    pass1_mask             : np.ndarray (H x W), uint8, Pass-1 mask
    pass2_full             : np.ndarray (H x W), uint8, Pass-2 union mask
    boundary_dilation_ratio: float (default 0.02), diagnostic only
    k                      : int, foreground pixel value (default 255)

    Returns
    -------
    np.ndarray (H x W), uint8

    References
    ----------
    Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    Filatov A., Shiryaev E., ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
    Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
    """
    p1 = (pass1_mask == k).astype(np.uint8)
    p2 = (pass2_full == k).astype(np.uint8)

    if p1.sum() == 0:
        print("DEBUG skeleton_guided_fusion: P1 empty, returning P2 unchanged.")
        return pass2_full.copy()

    if p2.sum() == 0:
        print("DEBUG skeleton_guided_fusion: P2 empty, returning P1 unchanged.")
        return pass1_mask.copy()

    # ------------------------------------------------------------------
    # Step 1 -- EDT and skeleton of P1 -> typical_radius -> r_connect.
    # Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    # Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
    # ------------------------------------------------------------------
    dist_transform = cv2.distanceTransform(p1, cv2.DIST_L2, 5)
    skel_p1        = sk_skeletonize(p1.astype(bool))

    if skel_p1.any():
        typical_radius = float(np.median(dist_transform[skel_p1]))
    else:
        dist_vals      = dist_transform[p1 > 0]
        typical_radius = float(np.median(dist_vals)) if len(dist_vals) > 0 else 3.0
    typical_radius = max(1.0, typical_radius)

    r_connect = max(int(round(typical_radius)), 4)

    # ------------------------------------------------------------------
    # Step 2 -- Fast path: single P2 component always accepted in full.
    # For single-component images skeleton_guided == replace.
    # Filatov A., Shiryaev E., ISSPA 2010, DOI: 10.1109/ISSPA.2010.5605426
    # ------------------------------------------------------------------
    struct8 = np.ones((3, 3), dtype=np.int32)
    _, n_mask_cc = sp_ndimage.label(p2, structure=struct8)

    if n_mask_cc <= 1:
        final = (p2 > 0).astype(np.uint8) * k
        print(
            f"DEBUG skeleton_guided_fusion: "
            f"typical_r={typical_radius:.1f}px | r_connect={r_connect}px | "
            f"mask_cc=1 -> single component, returning P2 unchanged | "
            f"P1={int(p1.sum())}px | P2_raw={int(p2.sum())}px | "
            f"final={int((final==k).sum())}px"
        )
        return final

    # ------------------------------------------------------------------
    # Step 3 -- Multi-component P2: filter isolated components.
    # Accept P2 MASK components whose skeleton touches dilate(P1, r_connect).
    # ------------------------------------------------------------------
    skel_p2 = sk_skeletonize(p2.astype(bool)).astype(np.uint8)

    kern_conn = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r_connect + 1, 2 * r_connect + 1)
    )
    p1_attach = cv2.dilate(p1, kern_conn, iterations=1)

    labeled_skel, n_skel_cc = sp_ndimage.label(skel_p2, structure=struct8)
    accepted_skel = np.zeros_like(skel_p2)
    n_skel_accepted = 0

    for comp_id in range(1, n_skel_cc + 1):
        comp_mask = (labeled_skel == comp_id)
        if np.logical_and(comp_mask, p1_attach > 0).any():
            accepted_skel[comp_mask] = 1
            n_skel_accepted += 1

    labeled_p2_mask, _ = sp_ndimage.label(p2, structure=struct8)
    accepted_p2 = np.zeros_like(p2)
    n_mask_accepted = 0
    n_mask_rejected = 0

    for comp_id in range(1, n_mask_cc + 1):
        comp_mask = (labeled_p2_mask == comp_id)
        if np.logical_and(comp_mask, accepted_skel > 0).any():
            accepted_p2[comp_mask] = 1
            n_mask_accepted += 1
        else:
            n_mask_rejected += 1

    final = (accepted_p2 > 0).astype(np.uint8) * k

    p1_px      = int(p1.sum())
    p2_raw     = int(p2.sum())
    p2_acc     = int(accepted_p2.sum())
    p2_blocked = p2_raw - p2_acc
    added_px   = int((final == k).sum()) - p1_px
    final_px   = int((final == k).sum())

    print(
        f"DEBUG skeleton_guided_fusion: "
        f"typical_r={typical_radius:.1f}px | r_connect={r_connect}px | "
        f"mask_cc={n_mask_cc} "
        f"(accepted={n_mask_accepted} rejected={n_mask_rejected}) | "
        f"P1={p1_px}px | P2_raw={p2_raw}px | "
        f"P2_blocked={p2_blocked}px "
        f"({100.0 * p2_blocked / max(1, p2_raw):.1f}%) | "
        f"net_added={added_px}px | final={final_px}px"
    )
    return final


# =============================================================================
# Main function (v9)
# =============================================================================
def predict_one_image_augmented(
    path_dataset,
    path_stereotypie,
    name_directory_dataset,
    model,
    processor,
    device,
    process_windows=False,
    tfms=None,
    save_mask=False,
    two_pass=False,
    two_pass_padding_ratio=0.50,
    two_pass_merge="replace",
    two_pass_context_fill=True,
    two_pass_model_size=320,
    export_results=False,
    export_dir="inference_outputs",
    min_component_pixels=10,
    min_pad_pixels=32,
    min_roi_fraction=0.0,
    boundary_dilation_ratio=0.02,
    keep_largest_component=False,
    # v6 parameters (backward compat, used only with merge="constrained_union")
    two_pass_merge_dilation_radius=None,
    two_pass_merge_dilation_fraction=0.05,
):
    """
    Predict segmentation mask with optional per-component two-pass inference.

    v9 recommended call
    --------------------
    predict_one_image_augmented(
        ...,
        two_pass=True,
        two_pass_merge="skeleton_guided",   # SOTA v9: connectivity-based gate
        two_pass_padding_ratio=0.50,
    )

    Pass-2 uses the v5 raw-crop path: the padded ROI crop is sent directly
    to predict_fn after reflect-padding to model_size.  The large padding
    (0.50) preserves background context that allows the model to distinguish
    the worm from its water-surface reflection.

    The fusion step ("adaptive_boundary") derives gate radii automatically
    from P1's distance transform and skeleton, giving a tight gate (~2px for
    a thin worm) that blocks reflection hallucinations (>15px from boundary)
    while still accepting missed thin-branch pixels.

    Parameters
    ----------
    path_dataset           : str, root directory with images/ and masks/
    path_stereotypie       : str, relative path to the image file
    name_directory_dataset : str, dataset split directory name
    model                  : segmentation model (torch.nn.Module)
    processor              : HuggingFace AutoImageProcessor
    device                 : torch.device
    process_windows        : bool, unused
    tfms                   : albumentations transform or None
    save_mask              : bool, unused (see export_results)
    two_pass               : bool, enable per-component two-pass inference
    two_pass_padding_ratio : float (default 0.50)
        Fractional bbox padding on each side.  Keep >= 0.50 to include
        enough background for reflection suppression.
    two_pass_merge         : str
        "replace"             -- raw Pass-2 union (v5, no gate)
        "constrained_union"   -- v6, deprecated (fixed r=54px, gate useless)
        "adaptive_boundary"   -- v7: tight auto-param gate from P1 EDT
        "spatially_adaptive"  -- v8: per-region varying gate
        "skeleton_guided"     -- v9 RECOMMENDED: connectivity-based gate;
                                  wide for connected branches (any length),
                                  tight for isolated hallucinations
    two_pass_context_fill  : bool, enable focused ROI cropping in Pass-2
    two_pass_model_size    : int, model input size for reflect-padding
                             (default 320)
    export_results         : bool
    export_dir             : str
    min_component_pixels   : int, minimum Pass-1 component size for Pass-2
    min_pad_pixels         : int, absolute minimum padding per side (pixels)
    min_roi_fraction       : float, minimum ROI as fraction of image axes
    boundary_dilation_ratio: float (default 0.02), used in boundary_dice
    keep_largest_component : bool (default False)
        Isensee F. et al., Nature Methods 18, 203-211, 2021.
        DOI: 10.1038/s41592-020-01008-z
    two_pass_merge_dilation_radius   : int or None (v6 compat)
    two_pass_merge_dilation_fraction : float (default 0.05) (v6 compat)

    Export layout (flat, no sub-folder)
    ------------------------------------
    <stem>_1_original_image.png
    <stem>_2_ground_truth_mask.png              (if GT exists)
    <stem>_3_final_mask.png
    When two_pass=True:
      <stem>_tp1_first_pass_mask.png
      <stem>_tp2_filled_crop_image.png
      <stem>_tp3_second_pass_crop_mask.png
      <stem>_tp4_filled_crop_ground_truth_mask.png (if GT exists)
      <stem>_tp_comp<N>_crop_image.png
      <stem>_tp_comp<N>_crop_mask.png
    """

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    img_path  = os.path.join(path_dataset, "images", path_stereotypie)
    mask_path = os.path.join(path_dataset, "masks",  path_stereotypie)

    image_raw = Image.open(img_path).convert("RGB")
    image     = np.array(image_raw)

    try:
        mask = cv2.imdecode(
            np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if mask is not None and len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except Exception:
        mask = None

    if tfms is not None:
        transformed = tfms(image=image, mask=mask if mask is not None else image)
        image = transformed["image"]
        if mask is not None:
            mask = transformed["mask"]

    # ------------------------------------------------------------------
    # 2. Dataset Split Check
    # ------------------------------------------------------------------
    image_status = "NOT in"
    base_split_path = os.path.join(
        "..", "..", "Dataset_stereotype_regeneration",
        "dataset_training_segmentation", "dataset_images_list",
        name_directory_dataset,
    )
    for split_label, file_name in {
        "train":      "train_dataset.txt",
        "validation": "val_dataset.txt",
        "test":       "test_dataset.txt",
    }.items():
        txtfile_path = os.path.join(base_split_path, file_name)
        if os.path.exists(txtfile_path):
            with open(txtfile_path, "r", encoding="utf-8") as f:
                if path_stereotypie in [l.rstrip("\n") for l in f]:
                    image_status = f"in {split_label}"
                    break

    print(f"DEBUG: Image {image_status} dataset of {name_directory_dataset}")

    # ------------------------------------------------------------------
    # 3. Normalisation
    # ------------------------------------------------------------------
    if image.dtype != np.uint8:
        image = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1.0
            else image.astype(np.uint8)
        )
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    debug = None

    if two_pass:
        if two_pass_context_fill:
            try:
                # Per-component two-pass (v5 raw-crop path).
                # He et al., Mask R-CNN, ICCV 2017, DOI: 10.1109/ICCV.2017.322
                forecasted_image, debug = _per_component_two_pass_predict(
                    model=model,
                    processor=processor,
                    device=device,
                    image=image,
                    padding_ratio=two_pass_padding_ratio,
                    model_size=two_pass_model_size,
                    predict_fn=predict,
                    ground_truth_mask=mask,
                    min_component_pixels=min_component_pixels,
                    min_pad_pixels=min_pad_pixels,
                    min_roi_fraction=min_roi_fraction,
                )
                print(
                    f"DEBUG two-pass summary: "
                    f"num_components={debug.get('num_components')}, "
                    f"context_fill={debug.get('context_fill_triggered')}"
                )
            except Exception as e:
                print(f"Two-pass failed: {e}. Falling back to single pass.")
                forecasted_image = predict(model, processor, device, image)
        else:
            print("Context fill disabled. Single pass.")
            forecasted_image = predict(model, processor, device, image)
    else:
        forecasted_image = predict(model, processor, device, image)

    # ------------------------------------------------------------------
    # 4b. Fusion strategy
    #
    # "replace" (v5 default):
    #   raw Pass-2 union.  No gate, reflection hallucinations propagate.
    #
    # "constrained_union" (v6, deprecated):
    #   P1 OR (P2 AND dilate(P1, r_fixed)).
    #   r_fixed=54px > worm width -> gate covers all of P2 -> P2_blocked=0%.
    #   Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    #   Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
    #
    # "adaptive_boundary" (v7, RECOMMENDED):
    #   core OR (P2 AND dilate(P1, r_outer)).
    #   r_outer ~ 2px for thin worm -> blocks reflection (15-50px away).
    #   r_core  ~ 3px -> freezes P1 interior.
    #   Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    #   Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
    #   Kamnitsas K. et al., Medical Image Analysis 2017,
    #     DOI: 10.1016/j.media.2016.10.004
    #   Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
    # ------------------------------------------------------------------
    if two_pass and debug is not None:
        pass1_anchor = debug.get("first_pass_mask")

        if two_pass_merge == "constrained_union" and pass1_anchor is not None:
            forecasted_image = _constrained_union_fusion(
                pass1_mask=pass1_anchor,
                pass2_full=forecasted_image,
                dilation_radius=two_pass_merge_dilation_radius,
                dilation_fraction=two_pass_merge_dilation_fraction,
                k=255,
            )

        elif two_pass_merge == "adaptive_boundary" and pass1_anchor is not None:
            forecasted_image = _adaptive_boundary_zone_fusion(
                pass1_mask=pass1_anchor,
                pass2_full=forecasted_image,
                boundary_dilation_ratio=boundary_dilation_ratio,
                k=255,
            )

        elif two_pass_merge == "spatially_adaptive" and pass1_anchor is not None:
            # v8: spatially varying gate (wide on thin, tight on thick).
            # Fails on thin branches longer than r_outer_thin from P1 endpoint.
            # Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
            # Ke L. et al., NeurIPS 2023. arXiv: 2306.01567
            # Kamnitsas K. et al., Medical Image Analysis 2017,
            #   DOI: 10.1016/j.media.2016.10.004
            # Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
            forecasted_image = _spatially_adaptive_fusion(
                pass1_mask=pass1_anchor,
                pass2_full=forecasted_image,
                boundary_dilation_ratio=boundary_dilation_ratio,
                k=255,
            )

        elif two_pass_merge == "skeleton_guided" and pass1_anchor is not None:
            # v9 RECOMMENDED: component-filter fusion.
            # Single-component P2: returns P2 directly (= replace).
            # Multi-component P2: filters isolated blobs (reflections).
            # Special case: high-aspect-ratio crop AND P2 < P1.
            #   The AutoImageProcessor compresses the long axis of the crop
            #   (e.g. 973/320 = 3.04x), producing a distorted P2 that
            #   under-segments the worm.  P1 was predicted on the full image
            #   (no distortion) and is more reliable in this case.
            #   P2 still contributes pixels that P1 missed (thin branches in
            #   the compressed direction), so P1 OR P2 union is the best
            #   recoverable result without ground truth.
            #   Trigger: max(crop_H, crop_W) / min(crop_H, crop_W) > 2.0
            #            AND P2_raw < P1_px (P2 under-segments vs P1).
            #   For undistorted crops (aspect < 2.0) or when P2 > P1,
            #   do not apply -- P2 is reliable and should not be diluted.
            # Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
            # Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629

            # Compute crop aspect ratio from debug info.
            _crop_shape  = None
            _components  = debug.get("components", []) if debug else []
            if _components:
                _crop_shape = _components[0].get("crop_shape")
            _crop_aspect = 1.0
            if _crop_shape is not None and len(_crop_shape) >= 2:
                _ch, _cw = _crop_shape[0], _crop_shape[1]
                _crop_aspect = max(_ch, _cw) / max(1, min(_ch, _cw))

            _p1_px = int((pass1_anchor == 255).sum())
            _p2_px = int((forecasted_image == 255).sum())
            
            # --- UNIVERSAL DISTORTION GATE ---
            # Calculate the percentage of pixels P2 lost compared to P1.
            _loss_ratio = (_p1_px - _p2_px) / max(1, _p1_px)
            
            # Use an asymptotic curve to calculate tolerance based on aspect ratio.
            # Max tolerance asymptotes at 15% for infinitely thin crops.
            _adaptive_loss_tolerance = 0.15 * (1.0 - (1.0 / max(1.0, _crop_aspect)))

            if _crop_aspect > 1.5 and _p2_px < _p1_px and _loss_ratio <= _adaptive_loss_tolerance:
                # Loss is within the mathematical limit for boundary degradation -> Recover it
                _union = np.logical_or(
                    pass1_anchor == 255, forecasted_image == 255
                ).astype(np.uint8) * 255
                print(
                    f"DEBUG skeleton_guided: boundary recovery "
                    f"(aspect={_crop_aspect:.2f}, P2={_p2_px} < P1={_p1_px}, "
                    f"loss={_loss_ratio:.1%} <= tol={_adaptive_loss_tolerance:.1%}) -> P1 OR P2 union"
                )
                forecasted_image = _union
            else:
                if _crop_aspect > 1.5 and _p2_px < _p1_px:
                    # Loss exceeds the asymptotic limit -> P2 correctly rejected a hallucination
                    print(
                        f"DEBUG skeleton_guided: hallucination rejection confirmed "
                        f"(loss={_loss_ratio:.1%} > tol={_adaptive_loss_tolerance:.1%})."
                    )
                # Standard topological filtering
                forecasted_image = _skeleton_guided_fusion(
                    pass1_mask=pass1_anchor,
                    pass2_full=forecasted_image,
                    boundary_dilation_ratio=boundary_dilation_ratio,
                    k=255,
                )

        elif two_pass_merge not in ("replace", "constrained_union",
                                    "adaptive_boundary", "spatially_adaptive",
                                    "skeleton_guided"):
            print(
                f"WARNING: unknown two_pass_merge='{two_pass_merge}'. "
                f"Falling back to 'replace'. "
                f"Valid values: 'replace', 'constrained_union', "
                f"'adaptive_boundary', 'spatially_adaptive', 'skeleton_guided'."
            )

    # ------------------------------------------------------------------
    # 4c. Post-processing: keep only the largest connected component.
    #
    # Isensee F. et al., Nature Methods 18, 203-211, 2021.
    # DOI: 10.1038/s41592-020-01008-z
    # ------------------------------------------------------------------
    n_components_removed = 0
    if keep_largest_component:
        forecasted_image, n_components_removed = _keep_largest_component(
            forecasted_image, k=255
        )

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    dice_score          = None
    boundary_dice_score = None
    cl_dice_score       = None
    hd95_score          = None

    if mask is not None:
        # Volumetric overlap.
        # Dice L.R., Ecology 1945, DOI: 10.2307/1932409
        dice_score = dice(forecasted_image, mask, k=255)

        # Boundary quality (degeneration-safe).
        # Cheng B. et al., CVPR 2021, arXiv: 2103.16562
        boundary_dice_score = boundary_dice(
            forecasted_image, mask, k=255,
            dilation_ratio=boundary_dilation_ratio,
        )

        # Topological connectivity.
        # Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
        cl_dice_score = cl_dice(forecasted_image, mask, k=255)

        # Worst-case boundary misalignment.
        # Huttenlocher D.P. et al., IEEE TPAMI 1993, DOI: 10.1109/34.232073
        hd95_score = hd95(forecasted_image, mask, k=255, percentile=95)

        hd95_str = (
            f"{hd95_score:.2f} px"
            if np.isfinite(hd95_score)
            else "inf (one mask empty)"
        )

        print(f"RESULT: Dice score:          {dice_score:.4f}")
        print(
            f"RESULT: Boundary Dice score: {boundary_dice_score:.4f}  "
            f"(d = min(max(1, round({boundary_dilation_ratio} * diag)), "
            f"max(1, min_bbox // 2)))"
        )
        print(f"RESULT: clDice score:        {cl_dice_score:.4f}")
        print(f"RESULT: HD95 score:          {hd95_str}")

    # ------------------------------------------------------------------
    # 6. Export (flat, no sub-folder)
    # ------------------------------------------------------------------
    if export_results:
        os.makedirs(export_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path_stereotypie))[0]

        cv2.imwrite(
            os.path.join(export_dir, f"{stem}_1_original_image.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        if mask is not None:
            cv2.imwrite(
                os.path.join(export_dir, f"{stem}_2_ground_truth_mask.png"),
                mask,
            )
        cv2.imwrite(
            os.path.join(export_dir, f"{stem}_3_final_mask.png"),
            forecasted_image,
        )

        if two_pass and debug is not None:
            fpm   = debug.get("first_pass_mask")
            fcgtm = debug.get("filled_crop_ground_truth_mask")
            if fpm is not None:
                cv2.imwrite(
                    os.path.join(export_dir, f"{stem}_tp1_first_pass_mask.png"),
                    fpm,
                )
                print(f"DEBUG export: tp1 first_pass_mask {fpm.shape}")
            if fcgtm is not None:
                cv2.imwrite(
                    os.path.join(
                        export_dir,
                        f"{stem}_tp4_filled_crop_ground_truth_mask.png",
                    ),
                    fcgtm,
                )
                print(f"DEBUG export: tp4 filled_crop_ground_truth_mask "
                      f"{fcgtm.shape}")
            for idx, comp in enumerate(debug.get("components", [])):
                comp_id  = comp.get("component_id", idx + 1)
                fci      = comp.get("filled_crop_image")
                spcm     = comp.get("second_pass_crop_mask")
                tag_img  = (
                    f"{stem}_tp2_filled_crop_image.png" if idx == 0
                    else f"{stem}_tp_comp{comp_id}_crop_image.png"
                )
                tag_mask = (
                    f"{stem}_tp3_second_pass_crop_mask.png" if idx == 0
                    else f"{stem}_tp_comp{comp_id}_crop_mask.png"
                )
                if fci is not None:
                    cv2.imwrite(
                        os.path.join(export_dir, tag_img),
                        cv2.cvtColor(fci, cv2.COLOR_RGB2BGR),
                    )
                    print(f"DEBUG export: comp {comp_id} crop_image {fci.shape}")
                if spcm is not None:
                    cv2.imwrite(os.path.join(export_dir, tag_mask), spcm)
                    print(
                        f"DEBUG export: comp {comp_id} crop_mask {spcm.shape}"
                    )

    # ------------------------------------------------------------------
    # 7. Visualisation
    # ------------------------------------------------------------------
    n_subplots = 3 if mask is not None else 2
    fig, axes  = plt.subplots(1, n_subplots, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original ({image_status})")
    axes[0].axis("off")

    if mask is not None:
        hd95_disp = (
            f"{hd95_score:.2f} px"
            if hd95_score is not None and np.isfinite(hd95_score)
            else ("inf" if hd95_score is not None else "N/A")
        )
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(
            f"Ground Truth\n"
            f"Dice: {dice_score:.4f}  "
            f"BDice: {boundary_dice_score:.4f}\n"
            f"clDice: {cl_dice_score:.4f}  "
            f"HD95: {hd95_disp}"
        )
        axes[1].axis("off")

    n_comp_label = (
        f"{debug.get('num_components', 'N/A')} component(s) detected"
        if debug else "single pass"
    )
    lcc_label   = (
        f", {n_components_removed} removed by LCC"
        if keep_largest_component else ""
    )
    merge_label = f", merge={two_pass_merge}" if two_pass else ""
    axes[n_subplots - 1].imshow(forecasted_image, cmap="gray")
    axes[n_subplots - 1].set_title(
        f"Forecasted Mask\n({n_comp_label}{lcc_label}{merge_label})"
    )
    axes[n_subplots - 1].axis("off")

    plt.tight_layout()
    plt.show()

    # Overlay visualisation
    image_bgr    = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result_image = image_bgr.copy()
    result_image[forecasted_image == 255] = np.array([125, 80, 25])
    darkened     = (image_bgr * 0.05).astype(np.uint8)
    overlay      = cv2.addWeighted(darkened, 0.5, result_image, 1.0, 0.5)

    d_str  = f"{dice_score:.4f}"          if dice_score          is not None else "N/A"
    bd_str = f"{boundary_dice_score:.4f}" if boundary_dice_score is not None else "N/A"
    cl_str = f"{cl_dice_score:.4f}"       if cl_dice_score       is not None else "N/A"
    hd_str = (
        f"{hd95_score:.2f} px"
        if hd95_score is not None and np.isfinite(hd95_score)
        else ("inf" if hd95_score is not None else "N/A")
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Overlay | Dice: {d_str}  BDice: {bd_str}  "
        f"clDice: {cl_str}  HD95: {hd_str}"
        + (f"  [LCC: -{n_components_removed}]" if keep_largest_component else "")
        + (f"  [{two_pass_merge}]" if two_pass else "")
    )
    plt.axis("off")
    plt.show()


test_tfms = None

# =============================================================================
# I/O helpers
# =============================================================================

@contextlib.contextmanager
def _maybe_suppress_stdout(verbose: bool):
    """Redirect stdout to /dev/null when verbose is False."""
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def _load_image(path: str) -> np.ndarray:
    """Load an image as uint8 RGB numpy array (H x W x 3)."""
    img = PILImage.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_mask(path) -> np.ndarray:
    """
    Load a ground-truth mask as uint8 grayscale (H x W).
    Returns None if path is None or the file does not exist.
    """
    if path is None or not os.path.exists(str(path)):
        return None
    raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return raw.astype(np.uint8)


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 RGB (H x W x 3)."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


# =============================================================================
# GPU-batched inference
#
# Produces binary uint8 masks (0 or 255) at the original image spatial
# dimensions.  Internally:
#   1. HuggingFace processor preprocesses the batch (resize + normalise).
#   2. Model forward pass (batched, fully on `device`).
#   3. Semantic-segmentation post-processing at model_size x model_size
#      (matches the fixed 320x320 output of the original single-image predict).
#   4. Bilinear resize back to each image's original dimensions,
#      followed by threshold at 127 -- identical to the resize path in
#      _run_single_component_pass2 (v9).
#
# References:
#   Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
#   Ronneberger O. et al., U-Net, MICCAI 2015,
#     DOI: 10.1007/978-3-319-24574-4_28
# =============================================================================

def _predict_batch_gpu(
    model,
    processor,
    device: torch.device,
    images: list,
    model_size: int = 320,
) -> list:
    """
    GPU-batched inference on a list of numpy images.

    Parameters
    ----------
    model      : HuggingFace segmentation model.  Must already reside on
                 `device` (verified externally by predict_images_test).
    processor  : HuggingFace AutoImageProcessor.
    device     : torch.device.
    images     : list of np.ndarray (H x W x 3), uint8.  Mixed sizes allowed.
    model_size : int (default 320).  Kept for API compatibility.  No longer
                 used as an intermediate resolution; post-processing is done
                 directly at each image's original (H, W).

    Returns
    -------
    list of np.ndarray (H x W), uint8 (0 or 255).
    Each output has the same spatial dimensions as the corresponding input.

    Implementation note
    -------------------
    The original single-image predict() calls
      processor.post_process_semantic_segmentation(outputs,
                                                   target_sizes=[(H, W)])
    and returns a mask at (H, W).  Routing through an intermediate
    (model_size, model_size) crop introduces one extra bilinear
    downsample + upsample cycle that shifts boundary pixels and degrades
    clDice by ~0.003-0.005 on thin structures.

    Fix (this version): target_sizes = original_sizes, so each image is
    decoded directly to its own dimensions -- bit-identical to single-image
    predict().  The fallback resize block below is retained as a safety
    net for processor configurations that ignore target_sizes, but it
    should never trigger for standard HuggingFace Mask2Former configs.

    References:
      Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
      Ronneberger O. et al., U-Net, MICCAI 2015,
        DOI: 10.1007/978-3-319-24574-4_28
    """
    if not images:
        return []

    original_sizes = [(img.shape[0], img.shape[1]) for img in images]
    pil_images = [PILImage.fromarray(img) for img in images]

    # Explicit device placement for all input tensors.
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process directly at each image's original spatial dimensions.
    # This is the critical fix: using original_sizes instead of a fixed
    # (model_size, model_size) avoids the intermediate downsample/upsample
    # that caused clDice degradation on thin structures.
    target_sizes = original_sizes
    pred_maps = processor.post_process_semantic_segmentation(
        outputs, target_sizes=target_sizes
    )

    results = []
    for pred_map, (orig_h, orig_w) in zip(pred_maps, original_sizes):
        # Foreground = any label > 0 (background label = 0).
        mask_out = (pred_map.cpu().numpy() > 0).astype(np.uint8) * 255

        # Safety fallback: some processor variants ignore target_sizes and
        # return a fixed-resolution map.  Resize to original if needed,
        # using INTER_LINEAR + threshold-127 to match the resize path in
        # _run_single_component_pass2 (v9).
        # Ronneberger O. et al., U-Net, MICCAI 2015,
        #   DOI: 10.1007/978-3-319-24574-4_28
        if mask_out.shape != (orig_h, orig_w):
            mask_f = cv2.resize(
                mask_out.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_out = (mask_f >= 127.0).astype(np.uint8) * 255

        results.append(mask_out)

    return results


# =============================================================================
# Two-pass: Pass-2 crop extraction (CPU)
#
# Reproduces the ROI computation in _run_single_component_pass2 (v9) without
# running inference.  The resulting crops are queued for Phase-3 batch GPU
# inference.
#
# References:
#   Filatov A., Shiryaev E., ISSPA 2010. DOI: 10.1109/ISSPA.2010.5605426
#   He K. et al., Mask R-CNN, ICCV 2017. DOI: 10.1109/ICCV.2017.322
#   Ren S. et al., Faster R-CNN, NeurIPS 2015.
#     DOI: 10.1109/TPAMI.2016.2577031
# =============================================================================

def _extract_p2_jobs(
    image: np.ndarray,
    p1_mask: np.ndarray,
    image_idx: int,
    padding_ratio: float,
    model_size: int,
    min_component_pixels: int,
    min_pad_pixels: int,
    min_roi_fraction: float,
) -> list:
    """
    Extract Pass-2 component crops from a single image (CPU).

    Returns a list of job dicts with keys:
      image_idx   int      index in the current batch of test images
      comp_id     int      1-based component label
      n_pixels    int      component area in pixels
      roi_y1      int      top-left y of the padded ROI (full-image coords)
      roi_x1      int      top-left x of the padded ROI (full-image coords)
      pad_top     int      reflect-padding rows added at the top of the crop
      pad_left    int      reflect-padding cols added at the left of the crop
      crop        ndarray  reflect-padded crop (H' x W' x 3), uint8
    """
    H, W = image.shape[:2]
    binary = (p1_mask > 127).astype(np.uint8)

    if binary.sum() == 0:
        return []

    # Connected-component labelling.
    # Filatov A., Shiryaev E., ISSPA 2010. DOI: 10.1109/ISSPA.2010.5605426
    labeled_array, num_features = sp_ndimage.label(binary)
    jobs = []

    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_array == comp_id).astype(np.uint8)
        n_pixels = int(comp_mask.sum())

        if n_pixels < min_component_pixels:
            continue

        comp_coords = np.where(comp_mask > 0)
        y_min = int(comp_coords[0].min());  y_max = int(comp_coords[0].max())
        x_min = int(comp_coords[1].min());  x_max = int(comp_coords[1].max())
        box_h = max(1, y_max - y_min);      box_w = max(1, x_max - x_min)

        pad_h  = max(int(box_h * padding_ratio), min_pad_pixels)
        pad_w  = max(int(box_w * padding_ratio), min_pad_pixels)
        roi_y1 = max(0, y_min - pad_h);  roi_y2 = min(H, y_max + pad_h)
        roi_x1 = max(0, x_min - pad_w);  roi_x2 = min(W, x_max + pad_w)

        # Minimum image-fraction and absolute-size constraints.
        # Ren S. et al., Faster R-CNN, DOI: 10.1109/TPAMI.2016.2577031
        roi_y1, roi_y2, roi_x1, roi_x2 = _expand_roi_to_min_image_fraction(
            image.shape, roi_y1, roi_y2, roi_x1, roi_x2, min_roi_fraction
        )
        roi_y1, roi_y2, roi_x1, roi_x2 = _expand_roi_to_min_size(
            image.shape, roi_y1, roi_y2, roi_x1, roi_x2, model_size, model_size
        )

        # Reflect-pad to at least model_size x model_size.
        # Innamorati et al., BMVC 2018, arXiv: 1805.03106
        roi_crop = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        roi_crop_padded, pad_top, pad_left = _reflect_pad_to_model_size(
            roi_crop, model_size
        )

        jobs.append({
            "image_idx": image_idx,
            "comp_id":   comp_id,
            "n_pixels":  n_pixels,
            "roi_y1":    roi_y1,
            "roi_x1":    roi_x1,
            "pad_top":   pad_top,
            "pad_left":  pad_left,
            "crop":      roi_crop_padded,
        })

    return jobs


# =============================================================================
# Two-pass: P2 crop mask pasting (CPU)
#
# Pastes a P2 component crop mask onto a full-image canvas using the same
# offset arithmetic as _run_single_component_pass2 (v9).
# =============================================================================

def _paste_p2_result(
    comp_crop_mask: np.ndarray,
    canvas: np.ndarray,
    job: dict,
) -> None:
    """
    Paste a Pass-2 component crop mask onto a full-image canvas in-place.

    comp_crop_mask has already been resized to the padded-crop dimensions by
    _predict_batch_gpu (matching the INTER_LINEAR + threshold-127 path in v9).
    """
    H, W     = canvas.shape[:2]
    roi_y1   = job["roi_y1"];   roi_x1   = job["roi_x1"]
    pad_top  = job["pad_top"];  pad_left = job["pad_left"]

    offset_y = roi_y1 - pad_top
    offset_x = roi_x1 - pad_left

    dst_y1 = max(0, offset_y)
    dst_y2 = min(H, offset_y + comp_crop_mask.shape[0])
    dst_x1 = max(0, offset_x)
    dst_x2 = min(W, offset_x + comp_crop_mask.shape[1])

    src_y1 = dst_y1 - offset_y;  src_y2 = dst_y2 - offset_y
    src_x1 = dst_x1 - offset_x;  src_x2 = dst_x2 - offset_x

    # Element-wise maximum preserves foreground pixels from earlier components.
    np.maximum(
        canvas[dst_y1:dst_y2, dst_x1:dst_x2],
        comp_crop_mask[src_y1:src_y2, src_x1:src_x2],
        out=canvas[dst_y1:dst_y2, dst_x1:dst_x2],
    )


# =============================================================================
# Fusion dispatcher
#
# Replicates the Step-4b fusion and distortion-gate logic from
# predict_one_image_augmented (v9), including the skeleton_guided universal
# distortion gate for high-aspect-ratio crops.
#
# References:
#   Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
#   Shit S. et al., CVPR 2021, DOI: 10.1109/CVPR46437.2021.01629
#   Ke L. et al., NeurIPS 2023, arXiv: 2306.01567
#   Maurer C.R. et al., IEEE TPAMI 2003, DOI: 10.1109/TPAMI.2003.1177156
# =============================================================================

def _apply_fusion(
    p1_mask: np.ndarray,
    p2_full: np.ndarray,
    two_pass_merge: str,
    boundary_dilation_ratio: float,
    crop_shape,
    verbose: bool,
    two_pass_merge_dilation_radius,
    two_pass_merge_dilation_fraction: float,
) -> np.ndarray:
    """
    Apply the selected fusion strategy and return the fused mask.

    Replicates the full Step-4b block from predict_one_image_augmented (v9),
    including the skeleton_guided universal distortion gate.
    """
    with _maybe_suppress_stdout(verbose):
        if two_pass_merge == "constrained_union":
            return _constrained_union_fusion(
                p1_mask, p2_full,
                dilation_radius=two_pass_merge_dilation_radius,
                dilation_fraction=two_pass_merge_dilation_fraction,
                k=255,
            )

        if two_pass_merge == "adaptive_boundary":
            return _adaptive_boundary_zone_fusion(
                p1_mask, p2_full,
                boundary_dilation_ratio=boundary_dilation_ratio,
                k=255,
            )

        if two_pass_merge == "spatially_adaptive":
            return _spatially_adaptive_fusion(
                p1_mask, p2_full,
                boundary_dilation_ratio=boundary_dilation_ratio,
                k=255,
            )

        if two_pass_merge == "skeleton_guided":
            # Universal distortion gate -- identical to v9.
            # High-aspect-ratio crops are geometrically compressed by the
            # processor's fixed-size resize.  If P2 under-segments P1 by
            # less than the asymptotic boundary-loss tolerance, recover by
            # P1 OR P2 union (boundary degradation, not hallucination).
            # Isensee F. et al., Nature Methods 2021,
            #   DOI: 10.1038/s41592-020-01008-z
            _crop_aspect = 1.0
            if crop_shape is not None and len(crop_shape) >= 2:
                _ch, _cw = int(crop_shape[0]), int(crop_shape[1])
                _crop_aspect = max(_ch, _cw) / max(1, min(_ch, _cw))

            _p1_px = int((p1_mask == 255).sum())
            _p2_px = int((p2_full == 255).sum())
            _loss_ratio = (_p1_px - _p2_px) / max(1, _p1_px)
            _adaptive_loss_tol = 0.15 * (1.0 - (1.0 / max(1.0, _crop_aspect)))

            if (
                _crop_aspect > 1.5
                and _p2_px < _p1_px
                and _loss_ratio <= _adaptive_loss_tol
            ):
                _union = np.logical_or(
                    p1_mask == 255, p2_full == 255
                ).astype(np.uint8) * 255
                if verbose:
                    print(
                        f"  skeleton_guided: boundary recovery "
                        f"(aspect={_crop_aspect:.2f}, "
                        f"loss={_loss_ratio:.1%} <= tol={_adaptive_loss_tol:.1%})"
                        f" -> P1 OR P2 union"
                    )
                return _union

            if verbose and _crop_aspect > 1.5 and _p2_px < _p1_px:
                print(
                    f"  skeleton_guided: hallucination rejection "
                    f"(loss={_loss_ratio:.1%} > tol={_adaptive_loss_tol:.1%})"
                )
            return _skeleton_guided_fusion(
                p1_mask, p2_full,
                boundary_dilation_ratio=boundary_dilation_ratio,
                k=255,
            )

        # "replace" (default) -- raw P2, no gate.
        if two_pass_merge not in ("replace",):
            print(
                f"WARNING: unknown two_pass_merge='{two_pass_merge}'. "
                f"Falling back to 'replace'. Valid values: "
                f"replace, constrained_union, adaptive_boundary, "
                f"spatially_adaptive, skeleton_guided."
            )
        return p2_full.copy()


# =============================================================================
# Summary statistics
# =============================================================================

def _print_metric_summary(
    name: str,
    values: list,
    filenames: list,
    draw_line: float,
    higher_is_better: bool = True,
    n_extremes: int = 2,
):
    """
    Print summary statistics for a single metric, mirroring the format of
    the example output in the evaluation notebook.

    Parameters
    ----------
    name             : metric label for the header line.
    values           : list of float metric scores.
    filenames        : list of image filenames (same order as values).
    draw_line        : pass threshold (count elements >= or <= draw_line).
    higher_is_better : True for Dice/clDice, False for HD95.
    n_extremes       : number of best/worst image names to display.
    """
    arr = np.array(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v  = float(np.std(arr))
    min_v  = float(np.min(arr))
    max_v  = float(np.max(arr))
    n_total = len(arr)

    if higher_is_better:
        n_pass = int(np.sum(arr >= draw_line))
        pass_label = f">= {draw_line}"
        worst_idx = list(np.argsort(arr)[:n_extremes])
        best_idx  = list(np.argsort(arr)[-n_extremes:][::-1])
    else:
        n_pass = int(np.sum(arr <= draw_line))
        pass_label = f"<= {draw_line}"
        best_idx  = list(np.argsort(arr)[:n_extremes])
        worst_idx = list(np.argsort(arr)[-n_extremes:][::-1])

    print(f"###### {name}")
    print(f"Mean: {mean_v}")
    print(f"Min:  {min_v}")
    for rank, idx in enumerate(worst_idx, 1):
        print(f"    img {rank}: {filenames[idx]}")
    print(f"Max:  {max_v}")
    for rank, idx in enumerate(best_idx, 1):
        print(f"    img {rank}: {filenames[idx]}")
    print(f"Std:  {std_v}")
    print(
        f"Number of elements {pass_label}: "
        f"{n_pass}/{n_total} ({100.0 * n_pass / max(1, n_total):.1f}%)"
    )
    print()


# =============================================================================
# CDF plotting
#
# References:
#   Maier-Hein L., Reinke A. et al., Nature Methods 2024.
#   DOI: 10.1038/s41592-023-02151-z
# =============================================================================

def _plot_cdf(
    metrics_list: list,
    draw_line: float = 0.94,
    hd95_cdf_line: float = 0.99,
    show: bool = True,
):
    """
    Plot empirical CDFs for Dice, clDice, and HD95 in one figure.

    Dice and clDice (higher is better):
      X-axis = metric value; vertical line at draw_line (pass threshold).

    HD95 (lower is better):
      X-axis = HD95 in pixels; horizontal line at hd95_cdf_line (0.99)
      marking the 99th-percentile HD95; vertical dashed line at that value.

    Metric-selection rationale:
      Maier-Hein L., Reinke A. et al., Nature Methods, 21(2), 195-212, 2024.
      DOI: 10.1038/s41592-023-02151-z
    """
    valid = [m for m in metrics_list if not np.isnan(m["dice"])]
    if not valid:
        print("_plot_cdf: no valid metrics to plot.")
        return

    dice_vals   = np.array([m["dice"]   for m in valid], dtype=np.float64)
    cldice_vals = np.array([m["cldice"] for m in valid], dtype=np.float64)
    hd95_finite = np.array(
        [m["hd95"] for m in valid if np.isfinite(m["hd95"])],
        dtype=np.float64,
    )
    n = len(dice_vals)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Empirical Cumulative Distribution Functions -- Segmentation Metrics",
        fontsize=13,
    )

    # --- Dice CDF ---
    ax = axes[0]
    sorted_dice = np.sort(dice_vals)
    cdf = np.arange(1, n + 1) / float(n)
    ax.plot(sorted_dice, cdf, color="steelblue", linewidth=2, label="Dice CDF")
    ax.axvline(
        draw_line, color="crimson", linestyle="--", linewidth=1.5,
        label=f"Pass threshold = {draw_line}",
    )
    n_pass_dice = int(np.sum(dice_vals >= draw_line))
    ax.set_xlabel("Dice coefficient")
    ax.set_ylabel("Cumulative fraction of images")
    ax.set_title(
        f"Dice  (mean={np.mean(dice_vals):.4f},"
        f" pass={n_pass_dice}/{n})"
    )
    ax.set_xlim([max(0.0, float(sorted_dice.min()) - 0.02), 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- clDice CDF ---
    ax = axes[1]
    sorted_cl = np.sort(cldice_vals)
    ax.plot(sorted_cl, cdf, color="darkorange", linewidth=2, label="clDice CDF")
    ax.axvline(
        draw_line, color="crimson", linestyle="--", linewidth=1.5,
        label=f"Pass threshold = {draw_line}",
    )
    n_pass_cl = int(np.sum(cldice_vals >= draw_line))
    ax.set_xlabel("clDice coefficient")
    ax.set_ylabel("Cumulative fraction of images")
    ax.set_title(
        f"clDice  (mean={np.mean(cldice_vals):.4f},"
        f" pass={n_pass_cl}/{n})"
    )
    ax.set_xlim([max(0.0, float(sorted_cl.min()) - 0.02), 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- HD95 CDF ---
    ax = axes[2]
    if len(hd95_finite) > 0:
        sorted_hd = np.sort(hd95_finite)
        n_hd = len(sorted_hd)
        cdf_hd = np.arange(1, n_hd + 1) / float(n_hd)

        ax.plot(sorted_hd, cdf_hd, color="seagreen", linewidth=2,
                label="HD95 CDF")

        # Horizontal line at hd95_cdf_line (e.g. 0.99).
        ax.axhline(
            hd95_cdf_line, color="crimson", linestyle="--", linewidth=1.5,
            label=f"{int(hd95_cdf_line * 100)}th-pct CDF line",
        )

        # Vertical line at the HD95 value corresponding to that CDF fraction.
        idx_pct = min(
            int(np.ceil(hd95_cdf_line * n_hd)) - 1, n_hd - 1
        )
        hd_at_pct = float(sorted_hd[idx_pct])
        ax.axvline(
            hd_at_pct, color="crimson", linestyle=":", linewidth=1.2,
            label=f"HD95 @ p{int(hd95_cdf_line*100)} = {hd_at_pct:.1f} px",
        )

        ax.set_xlabel("HD95 (pixels)")
        ax.set_ylabel("Cumulative fraction of images")
        ax.set_title(
            f"HD95  (mean={np.mean(hd95_finite):.2f} px,"
            f" p99={np.percentile(hd95_finite, 99):.2f} px)"
        )
        ax.set_xlim([0.0, float(sorted_hd.max()) * 1.08])
        ax.set_ylim([0.0, 1.05])
    else:
        ax.text(
            0.5, 0.5, "No finite HD95 values",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


# =============================================================================
# Main evaluation function
# =============================================================================

def predict_images_test(
    test_images: list,
    test_masks: list,
    model,
    processor,
    device: torch.device,
    tfms=None,
    batch_size: int = 8,
    draw_line: float = 0.94,
    show: bool = True,
    # Two-pass parameters
    two_pass: bool = False,
    two_pass_merge: str = "skeleton_guided",
    two_pass_padding_ratio: float = 0.50,
    two_pass_model_size: int = 320,
    two_pass_context_fill: bool = True,
    min_component_pixels: int = 10,
    min_pad_pixels: int = 32,
    min_roi_fraction: float = 0.0,
    two_pass_verbose: bool = False,
    keep_largest_component: bool = True,
    boundary_dilation_ratio: float = 0.02,
    # v6 backward-compat parameters (used only with merge="constrained_union")
    two_pass_merge_dilation_radius=None,
    two_pass_merge_dilation_fraction: float = 0.05,
) -> list:
    """
    Evaluate a segmentation model on a test set using GPU-batched inference.

    GPU optimisation (two-pass)
    ---------------------------
    Phase 1 (batch GPU): Pass-1 on batch_size images at once.
    Phase 2 (CPU):       Connected-component detection + crop extraction,
                         batched across all images before any GPU work.
    Phase 3 (batch GPU): Pass-2 on all component crops in sub-batches.
    Phase 4 (CPU):       Mask reconstruction, fusion, post-processing.

    This grouping maximises GPU utilisation: for N images with K total
    components, Pass-2 requires ceil(K / batch_size) GPU calls instead of K.

    Parameters
    ----------
    test_images    : list of str, absolute image file paths.
    test_masks     : list of str or None, absolute ground-truth mask paths.
    model          : HuggingFace segmentation model.
    processor      : HuggingFace AutoImageProcessor.
    device         : torch.device.
    tfms           : albumentations.Compose or None.
    batch_size     : int (default 8), GPU batch size for both passes.
    draw_line      : float (default 0.94), pass threshold for Dice and clDice.
    show           : bool (default True), display CDF plot.
    two_pass       : bool (default False), enable per-component two-pass.
    two_pass_merge : str, fusion strategy.
        "replace"           -- raw P2 union, no gate (v5).
        "constrained_union" -- fixed-radius gate (v6, deprecated).
        "adaptive_boundary" -- EDT-based tight gate (v7).
        "spatially_adaptive"-- per-region varying gate (v8).
        "skeleton_guided"   -- topological connectivity filter (v9, default).
    two_pass_padding_ratio  : float (default 0.50), ROI fractional padding.
    two_pass_model_size     : int (default 320), reflect-pad target size.
    two_pass_context_fill   : bool (default True), enable ROI cropping in P2.
                              If False, single-pass inference is used.
    min_component_pixels    : int (default 10), min P1 component size for P2.
    min_pad_pixels          : int (default 32), min absolute padding (pixels).
    min_roi_fraction        : float (default 0.0), min ROI / image fraction.
    two_pass_verbose        : bool (default False), print fusion debug lines.
    keep_largest_component  : bool (default True), post-processing LCC filter.
      Isensee F. et al., Nature Methods 2021, DOI: 10.1038/s41592-020-01008-z
    boundary_dilation_ratio : float (default 0.02), for boundary_dice.
    two_pass_merge_dilation_radius   : int or None (v6 compat).
    two_pass_merge_dilation_fraction : float (default 0.05) (v6 compat).

    Returns
    -------
    list of dicts, one per test image.
    Keys: filename (str), dice (float), cldice (float),
          hd95 (float, np.inf when one mask is empty),
          boundary_dice (float).

    References
    ----------
    Dice L.R., Ecology 1945. DOI: 10.2307/1932409
    Shit S. et al., CVPR 2021. DOI: 10.1109/CVPR46437.2021.01629
    Huttenlocher D.P. et al., IEEE TPAMI 1993. DOI: 10.1109/34.232073
    Maier-Hein L. et al., Nature Methods 2024. DOI: 10.1038/s41592-023-02151-z
    Isensee F. et al., Nature Methods 2021. DOI: 10.1038/s41592-020-01008-z
    Filatov A., Shiryaev E., ISSPA 2010. DOI: 10.1109/ISSPA.2010.5605426
    He K. et al., Mask R-CNN, ICCV 2017. DOI: 10.1109/ICCV.2017.322
    """
    # ------------------------------------------------------------------
    # Guard: ensure model is on the correct device.
    # This is the most common source of silent CPU fallbacks.
    # ------------------------------------------------------------------
    model.eval()
    try:
        model_device = next(model.parameters()).device
        if str(model_device) != str(device):
            print(
                f"WARNING: model is on {model_device} but device={device}. "
                f"Moving model to {device}."
            )
            model = model.to(device)
    except StopIteration:
        pass

    n_total    = len(test_images)
    n_batches  = (n_total + batch_size - 1) // batch_size
    all_metrics: list = []

    print(f"N images: {n_total}")
    print(f"#### Pred on all masks #####")
    t_start = time.time()

    # ------------------------------------------------------------------
    # Main loop: process test images in batches of batch_size.
    # ------------------------------------------------------------------
    for batch_idx in range(n_batches):
        b_start = batch_idx * batch_size
        b_end   = min(b_start + batch_size, n_total)
        batch_img_paths  = test_images[b_start:b_end]
        batch_mask_paths = test_masks[b_start:b_end]
        n_b = len(batch_img_paths)

        # --------------------------------------------------------------
        # Load images and masks.
        # --------------------------------------------------------------
        images_np = []
        masks_np  = []
        for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
            img = _ensure_uint8_rgb(_load_image(str(img_path)))
            msk = _load_mask(mask_path)
            if tfms is not None:
                aug_kwargs = {"image": img}
                if msk is not None:
                    aug_kwargs["mask"] = msk
                aug_out = tfms(**aug_kwargs)
                img = aug_out["image"]
                if msk is not None:
                    msk = aug_out.get("mask", msk)
            images_np.append(img)
            masks_np.append(msk)

        # --------------------------------------------------------------
        # Phase 1: Pass-1 batch inference (GPU).
        # Isensee F. et al., Nature Methods 2021,
        #   DOI: 10.1038/s41592-020-01008-z
        # --------------------------------------------------------------
        p1_masks = _predict_batch_gpu(
            model, processor, device, images_np,
            model_size=two_pass_model_size,
        )

        use_two_pass = two_pass and two_pass_context_fill

        if not use_two_pass:
            final_masks = p1_masks
        else:
            # -----------------------------------------------------------
            # Phase 2: Extract Pass-2 crop jobs for all images (CPU).
            # Crops are collected across all images before any GPU work.
            # Filatov A., Shiryaev E., ISSPA 2010,
            #   DOI: 10.1109/ISSPA.2010.5605426
            # -----------------------------------------------------------
            all_jobs       = []
            jobs_per_image = [[] for _ in range(n_b)]

            for i in range(n_b):
                jobs_i = _extract_p2_jobs(
                    image=images_np[i],
                    p1_mask=p1_masks[i],
                    image_idx=i,
                    padding_ratio=two_pass_padding_ratio,
                    model_size=two_pass_model_size,
                    min_component_pixels=min_component_pixels,
                    min_pad_pixels=min_pad_pixels,
                    min_roi_fraction=min_roi_fraction,
                )
                all_jobs.extend(jobs_i)
                jobs_per_image[i] = jobs_i

            # -----------------------------------------------------------
            # Phase 3: Pass-2 batch inference on all crops (GPU).
            # All component crops collected in Phase 2 are forwarded
            # through the model in sub-batches of batch_size.
            # He K. et al., Mask R-CNN, ICCV 2017,
            #   DOI: 10.1109/ICCV.2017.322
            # -----------------------------------------------------------
            p2_crop_results = []
            if all_jobs:
                crops = [j["crop"] for j in all_jobs]
                for sub_s in range(0, len(crops), batch_size):
                    sub_crops = crops[sub_s:sub_s + batch_size]
                    sub_masks = _predict_batch_gpu(
                        model, processor, device, sub_crops,
                        model_size=two_pass_model_size,
                    )
                    p2_crop_results.extend(sub_masks)

            # -----------------------------------------------------------
            # Phase 4a: Reconstruct full-image P2 masks from crop results.
            # -----------------------------------------------------------
            p2_full_masks = [
                np.zeros(images_np[i].shape[:2], dtype=np.uint8)
                for i in range(n_b)
            ]
            for job, crop_mask in zip(all_jobs, p2_crop_results):
                _paste_p2_result(crop_mask, p2_full_masks[job["image_idx"]], job)

            # Fall back to P1 for images where all P2 components were
            # empty (e.g. very small components skipped in Phase 2).
            for i in range(n_b):
                if p2_full_masks[i].sum() == 0 and p1_masks[i].sum() > 0:
                    p2_full_masks[i] = p1_masks[i].copy()

            # -----------------------------------------------------------
            # Phase 4b: Fusion.
            # Crop shape for the distortion-gate heuristic in
            # skeleton_guided comes from the first component's crop.
            # -----------------------------------------------------------
            final_masks = []
            for i in range(n_b):
                crop_shape = None
                if jobs_per_image[i]:
                    crop_shape = jobs_per_image[i][0]["crop"].shape

                fused = _apply_fusion(
                    p1_mask=p1_masks[i],
                    p2_full=p2_full_masks[i],
                    two_pass_merge=two_pass_merge,
                    boundary_dilation_ratio=boundary_dilation_ratio,
                    crop_shape=crop_shape,
                    verbose=two_pass_verbose,
                    two_pass_merge_dilation_radius=two_pass_merge_dilation_radius,
                    two_pass_merge_dilation_fraction=two_pass_merge_dilation_fraction,
                )
                final_masks.append(fused)

        # --------------------------------------------------------------
        # Post-processing: largest connected component (LCC).
        # Isensee F. et al., Nature Methods 2021,
        #   DOI: 10.1038/s41592-020-01008-z
        # --------------------------------------------------------------
        if keep_largest_component:
            for i in range(n_b):
                with _maybe_suppress_stdout(two_pass_verbose):
                    final_masks[i], _ = _keep_largest_component(
                        final_masks[i], k=255
                    )

        # --------------------------------------------------------------
        # Metric computation per image.
        # Dice L.R., Ecology 1945. DOI: 10.2307/1932409
        # Shit S. et al., CVPR 2021. DOI: 10.1109/CVPR46437.2021.01629
        # Huttenlocher D.P. et al., IEEE TPAMI 1993. DOI: 10.1109/34.232073
        # Cheng B. et al., CVPR 2021. arXiv: 2103.16562
        # --------------------------------------------------------------
        for i, (img_path, mask_gt) in enumerate(
            zip(batch_img_paths, masks_np)
        ):
            filename = os.path.basename(str(img_path))
            pred     = final_masks[i]

            if mask_gt is None:
                all_metrics.append({
                    "filename":      filename,
                    "dice":          float("nan"),
                    "cldice":        float("nan"),
                    "hd95":          float("nan"),
                    "boundary_dice": float("nan"),
                })
                continue

            with _maybe_suppress_stdout(two_pass_verbose):
                d   = dice(pred, mask_gt, k=255)
                bd  = boundary_dice(
                    pred, mask_gt, k=255,
                    dilation_ratio=boundary_dilation_ratio,
                )
                cld = cl_dice(pred, mask_gt, k=255)
                h95 = hd95(pred, mask_gt, k=255, percentile=95)

            all_metrics.append({
                "filename":      filename,
                "dice":          d,
                "cldice":        cld,
                "hd95":          h95,
                "boundary_dice": bd,
            })

        print(
            f"  Batch {batch_idx + 1}/{n_batches} done "
            f"({b_end}/{n_total} images)"
        )

    elapsed = time.time() - t_start
    print(
        f"\nTotal inference time: {elapsed:.1f}s "
        f"({elapsed / max(1, n_total):.2f}s/image)"
    )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    valid = [m for m in all_metrics if not np.isnan(m["dice"])]
    n_valid = len(valid)

    if n_valid == 0:
        print("No valid metric values -- no ground-truth masks found.")
        return all_metrics

    dice_vals   = [m["dice"]          for m in valid]
    cldice_vals = [m["cldice"]        for m in valid]
    hd95_vals   = [m["hd95"]          for m in valid]
    bdice_vals  = [m["boundary_dice"] for m in valid]
    filenames   = [m["filename"]       for m in valid]

    print()
    _print_metric_summary(
        "Dice coefficient (all masks)",
        dice_vals, filenames,
        draw_line=draw_line, higher_is_better=True,
    )
    _print_metric_summary(
        "clDice coefficient (all masks)",
        cldice_vals, filenames,
        draw_line=0.99, higher_is_better=True,
    )

    # For HD95, lower is better.
    # Use 10 px as the pass threshold (typical clinical/research criterion).
    hd95_finite  = [v for v in hd95_vals if np.isfinite(v)]
    fn_hd95_fin  = [fn for fn, v in zip(filenames, hd95_vals) if np.isfinite(v)]
    _print_metric_summary(
        "HD95 (pixels, all masks)",
        hd95_finite if hd95_finite else [0.0],
        fn_hd95_fin if fn_hd95_fin else ["N/A"],
        draw_line=10.0,
        higher_is_better=False,
    )

    _print_metric_summary(
        "Boundary Dice (all masks)",
        bdice_vals, filenames,
        draw_line=draw_line, higher_is_better=True,
    )

    # ------------------------------------------------------------------
    # CDF plots
    # Maier-Hein L. et al., Nature Methods 2024.
    # DOI: 10.1038/s41592-023-02151-z
    # ------------------------------------------------------------------
    if show:
        _plot_cdf(valid, draw_line=draw_line, hd95_cdf_line=0.99, show=True)

    return all_metrics