# -*- coding: utf-8 -*-
"""Reproduce the golden's 14 texture descriptors and the size residualisation.

The reference golden table (``lightgbm_stereotype_v3t`` + v3z..v4e columns, the
table the v4j model is trained/evaluated on) does NOT compute these 14 texture
descriptors on the colour formatted image the app uses. ``calcul_descripteurs_
stereotypies_v3c`` computes them with a dedicated chain (verified to reproduce
features_golden to float32 precision):

  1. mask: the full-res Mask2Former mask;
  2. clean: largest 8-connected component, then holes FILLED (floodFill from the
     corner) -- the opposite of the classifier's formatted image which keeps
     holes;
  3. grey image: compute_mask_image_planarian_reshaped(raw, clean*255, 384, 0,
     apply_mask=True, black_and_white=True) -- grey, inverted, min-max stretched
     in the mask, CLAHE (2.0, 8x8), then the same crop/square to 384;
  4. mask_384 of the cleaned mask;
  5. texture_features.all_texture_features(grey, image, mask_384);
  6. size residualisation for the columns in size_residual_coefs_v3.csv:
     value - (b0 + b1 * ln(full-res mask area + 1)).

A separate size residualisation (size_residual_coefs.csv) applies to the shape
descriptors sh_mbend_energy_T0 / sh_mcurv_absmean_T0 / sh_mtotturn_T0 and to
tm_twist_max, which are computed by the normal pipeline then corrected.
"""
from __future__ import annotations

import os

import numpy as np
import cv2

# The texture descriptors the v4j model reads that need the b&w chain
# (all_texture_features on the cleaned-mask b&w/CLAHE image). tm_twist_max is one
# of them -- same chain as the 14 texture descriptors, then the v3 size residual.
TEX14 = ["tgb_aniso1", "tgb_aniso2", "tgb_hl_ratio", "tb_gabor_aniso",
         "tb_gabor_mean", "tw_E5R5", "tw_L5E5", "tw_L5R5", "tw_L5S5", "tw_R5R5",
         "tk_bhat5_max", "tk_bhat5_mean", "ti_kurt", "te_lap_var", "tm_twist_max"]

# Shape descriptors residualised by the ORIGINAL coefs. They are computed by the
# normal pipeline on the formatted-grey image (correct already); only the size
# residual is missing, applied here to the row's value.
SHAPE_TWIST_RESID = ["sh_mbend_energy_T0", "sh_mcurv_absmean_T0",
                     "sh_mtotturn_T0"]


def _clean(mk):
    """precompute_mask3_metrics._clean: biggest 8-conn component, holes filled."""
    m = (mk > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return m
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = (m * 255).copy()
    h, w = ff.shape
    fl = ff.copy()
    cv2.floodFill(fl, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return (m | (cv2.bitwise_not(fl) > 0)).astype(np.uint8)


def _masque_384(cm, CIM):
    cnts = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    r = CIM.resize_with_black_borders(
        (cm[y:y + h, x:x + w] > 0).astype(np.uint8) * 255, target_size=384)
    return (cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) >= 128).astype(np.uint8)


def _load_coefs(path):
    import pandas as pd
    if path and os.path.isfile(path):
        return pd.read_csv(path, sep=";").set_index("col")
    return None


class TextureV3c:
    """Computes the 14 b&w-chain texture descriptors and the shape/twist size
    residuals, matching the golden. Best-effort: on any failure returns {} so
    the caller keeps the normal-pipeline values."""

    def __init__(self, coefs_v3_path=None, coefs_orig_path=None):
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))  # app root
        cfg = os.path.join(root, "lightgbm_stereotype_classifier", "config")
        fe = os.path.join(root, "lightgbm_stereotype_classifier",
                          "feature_extraction")
        self._co_v3 = _load_coefs(
            coefs_v3_path or os.path.join(cfg, "size_residual_coefs_v3.csv"))
        self._co_orig = _load_coefs(
            coefs_orig_path or os.path.join(fe, "size_residual_coefs.csv"))
        self._tx = None
        self._cim = None
        self._ok = True
        self.last_error = None   # last per-frame failure (diagnostics)
        self.n_fail = 0          # count of frames that fell back to {}

    def _load(self):
        if self._tx is None:
            import texture_features as TX   # on the metrics_lib sys.path
            from . import classification_images as CIM
            self._tx = TX
            self._cim = CIM

    def _fold_twist(self):
        import fold_twist_metrics as FTM
        return FTM

    def recompute_formatted(self, raw_bgr, mask_full):
        """The texture + curated + colour descriptors, computed the v3c way:
        cleaned mask, b&w/CLAHE 384 image for texture/curated, colour 384 image
        for colour metrics, m384 mask, then the v3 size residual on the affected
        columns. Returns {col: value} for every column these groups output (the
        caller keeps only those the model reads). {} on a PER-FRAME failure
        (does NOT disable future frames); only a module-load failure disables."""
        if not self._ok or raw_bgr is None or mask_full is None:
            return {}
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - module import failed: disable
            self._ok = False
            self.last_error = "load: %s: %s" % (type(exc).__name__, exc)
            return {}
        try:
            FTM = self._fold_twist()
            cm = _clean(mask_full)
            m384 = _masque_384(cm, self._cim)
            m8 = (cm * 255).astype(np.uint8)
            ffb = self._cim.compute_mask_image_planarian_reshaped(
                raw_bgr, m8, 384, 0.0, apply_mask=True, black_and_white=True)
            ggb = cv2.cvtColor(ffb, cv2.COLOR_BGR2GRAY) if ffb.ndim == 3 else ffb
            ffc = self._cim.compute_mask_image_planarian_reshaped(
                raw_bgr, m8, 384, 0.0, apply_mask=True, black_and_white=False)
            out = {}
            try:
                out.update(self._tx.all_texture_features(ggb, ffb, m384))
            except Exception:  # noqa: BLE001
                pass
            try:
                out.update(FTM.curated_metrics(
                    ggb.astype(np.float32), m384.astype(bool)))
            except Exception:  # noqa: BLE001
                pass
            try:
                out.update(FTM.color_metrics(ffc, m384.astype(bool)))
            except Exception:  # noqa: BLE001
                pass
            la = np.log(float((mask_full > 0).sum()) + 1.0)
            if self._co_v3 is not None:
                for c in list(out.keys()):
                    if c in self._co_v3.index:
                        out[c] = float(out[c]) - (
                            self._co_v3.loc[c, "b0"]
                            + self._co_v3.loc[c, "b1"] * la)
            return {k: float(v) for k, v in out.items()}
        except Exception as exc:  # noqa: BLE001 - PER-FRAME failure: do NOT
            # disable; this frame falls back to the pipeline value, the next
            # frame tries again. (Previously self._ok=False here silently
            # disabled the b&w chain for the rest of the video.)
            import traceback
            self.n_fail += 1
            self.last_error = "%s: %s | %s" % (
                type(exc).__name__, exc, traceback.format_exc().splitlines()[-2:])
            return {}

    def shape_twist_residuals(self, row, mask_full):
        """Subtract the ORIGINAL-coefs size residual from the shape/twist
        descriptors already in ``row`` (in place). Returns the changed keys."""
        if self._co_orig is None or mask_full is None:
            return []
        try:
            la = np.log(float((mask_full > 0).sum()) + 1.0)
        except Exception:  # noqa: BLE001
            return []
        changed = []
        for c in SHAPE_TWIST_RESID:
            if c in row and c in self._co_orig.index and row[c] is not None:
                try:
                    row[c] = float(row[c]) - (
                        self._co_orig.loc[c, "b0"]
                        + self._co_orig.loc[c, "b1"] * la)
                    changed.append(c)
                except Exception:  # noqa: BLE001
                    pass
        return changed
