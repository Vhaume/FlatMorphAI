# -*- coding: utf-8 -*-
"""Step 2. Refinement of the worm mask by the blastema segmenter (mask ..._forme_raffine_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2).

Mask2Former often misses the translucent blastema. The functions fill_holes, contact, lisser_local, enveloppe, ouvrir,
raffiner and ajouter_annotation are read from production/raffiner_masques_blasteme_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2.py
and executed unchanged. Path of raffiner (nothing of the Mask2Former mask is ever removed):
  1. worm mask = largest component of the mask >= 128 (CS.masque_ver), axis by skeleton + skan + Dijkstra with angular
     prolongation (cadrage_squelette_skan2), two ends (tip and thumbnail centre at 0.12 L), length L,
  2. at each end, thumbnail of side 0.90 L, tip up, background outside the worm dilated by DIL_BL = 0.30 L set to 0,
     blastema U-Net v5_s1neg, probability p, binary p >= SEUIL_P (0.40), Gaussian blur sigma SIGMA_VIGNETTE (2 px)
     re-thresholded at 0.5, back-projection into the image (maximum of the two ends),
  3. candidate pixels inside the band and outside the worm, components touching the worm kept, holes filled, contact
     rule (pixels touching the mask / sqrt(area) >= SEUIL_CONTACT = 2.5),
  4. smoothing 2: Gaussian blur sigma SIGMA_IMAGE x L of the mask in the added zone, threshold 0.5,
  5. convex hull of each added piece, restricted to the band and to pixels where the back-projected probability is
     >= SEUIL_GARDE (0.10),
  6. morphological opening by a disc of radius RAYON_OUV x L (0.03 L), applied to the added blastema only.
Expert annotation (optional, as in production for the 114 annotated images of cyan_blasteme): when an annotated copy of the
image is given, the cyan strokes absent from the original (opening 3 x 3, at least MIN_ANNOT px) REPLACE the model output:
final mask = worm mask + annotated components (>= MIN_COMP_ANNOT px) touching it (annotfoi rule).

References (verified). Ronneberger O., Fischer P., Brox T. (2015). U-Net. MICCAI 2015, arXiv 1505.04597. Tan M., Le Q.
(2019). EfficientNet. ICML 2019, arXiv 1905.11946. Nunez-Iglesias J., Blanch A. J., Looker O., Dixon M. W., Tilley L.
(2018). A new Python library to analyse skeleton images confirms malaria parasite remodelling of the red blood cell
membrane skeleton. PeerJ 6, e4312, DOI 10.7717/peerj.4312 (skan). Dijkstra E. W. (1959). A note on two problems in
connexion with graphs. Numerische Mathematik 1, 269-271, DOI 10.1007/BF01386390. Graham R. L. (1972). An efficient
algorithm for determining the convex hull of a finite planar set. Information Processing Letters 1(4), 132-133,
DOI 10.1016/0020-0190(72)90045-2. Serra J. (1982). Image Analysis and Mathematical Morphology. Academic Press,
ISBN 978-0-12-637240-3. Lindeberg T. (1994). Scale-space theory. Journal of Applied Statistics 21(1-2), 225-270,
DOI 10.1080/757582976.
"""
import numpy as np, cv2, torch
from . import _env
from ._env import AC, A3b, CS, read, model_path, exec_slice, apply_parameters, show_device, check_on_device, jpeg_roundtrip, jpeg_enabled

PROD_FILE = "raffiner_masques_blasteme_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2.py"


def load_unets():
    """Loads (once) the blastema and eye U-Nets of the chain through the cascade loader, paths from models.csv."""
    A3b.CK_BL3 = model_path("blastema_segmenter"); A3b.CK_EY3 = model_path("eye_detector")
    mbl, mey = A3b.charger(verbeux=False)
    check_on_device(mbl, "blastema U-Net"); check_on_device(mey, "eye U-Net")
    return mbl, mey


class Refiner:
    """ref = Refiner(); final, info = ref.refine(bgr, mask_m2f, annotated_bgr=None)."""

    def __init__(self, verbose=True):
        show_device()
        apply_parameters("axis", {"": CS})
        self.P = read("refinement")
        self.mdl_bl, _ = load_unets()
        ck = torch.load(A3b.CK_BL3, map_location="cpu", weights_only=False)
        self.DIL_BL = float(ck["dil"])
        assert abs(self.DIL_BL - 0.30) < 1e-9, "unexpected band of the blastema segmenter (%s)" % self.DIL_BL
        ns = dict(np=np, cv2=cv2, CS=CS, AC=AC, mdl_bl=self.mdl_bl, DIL_BL=self.DIL_BL, CYAN={})
        for k in ("SEUIL_P", "SIGMA_VIGNETTE", "SIGMA_IMAGE", "SEUIL_GARDE", "RAYON_OUV", "SEUIL_CONTACT",
                  "MIN_COMP_ANNOT", "MIN_ANNOT"):
            ns[k] = self.P[k]
        exec_slice(PROD_FILE, "def fill_holes(r):", "# In[ ]:", ns)
        self.ns = ns
        if verbose:
            print("refinement: blastema U-Net %s (band %.2f L, Dice val %.3f) | threshold %.2f | opening %.2f L"
                  % (A3b.CK_BL3, self.DIL_BL, float(ck.get("val_dice", np.nan)), self.P["SEUIL_P"], self.P["RAYON_OUV"]))

    def annotation(self, bgr, annotated_bgr):
        """Expert blastema annotation (cyan strokes of the annotated copy absent from the original), or None."""
        if annotated_bgr is None or annotated_bgr.shape != bgr.shape: return None
        a = CS.cyan_strict(annotated_bgr) & ~CS.cyan_strict(bgr)
        a = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        return a if a.sum() >= self.P["MIN_ANNOT"] else None

    def refine(self, bgr, mraw, annotated_bgr=None):
        """bgr: original image, mraw: Mask2Former mask (grey 0..255). Returns (final mask uint8 0/255 as read by the
        next steps, info). A mask of less than 300 px is returned unchanged, as in production."""
        res = self.ns["raffiner"](bgr, mraw)
        if res is None:
            return mraw.copy(), dict(refined=0, annotated=0)
        wm, r, L, n_g, n_r, a_r = res
        info = dict(refined=1, L=float(L), area_m2f=int(wm.sum()), area_added_model=int(r.sum() - wm.sum()),
                    pieces_kept=n_g, pieces_removed_contact=n_r, area_removed_contact=int(a_r), annotated=0)
        fin = r
        a = self.annotation(bgr, annotated_bgr)
        if a is not None:
            fin = self.ns["ajouter_annotation"](wm, (wm > 0), a)   # annotation alone (annotfoi), model output discarded
            info.update(annotated=1, area_annotation=int(a.sum()))
        info["added_fraction"] = float((fin.sum() - wm.sum()) / max(wm.sum(), 1))
        out = fin.astype(np.uint8) * 255
        if self.P["JPEG_ROUNDTRIP"] and jpeg_enabled(): out = jpeg_roundtrip(out, quality=100)
        return out, info
