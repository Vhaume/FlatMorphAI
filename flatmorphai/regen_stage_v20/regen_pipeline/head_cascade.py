# -*- coding: utf-8 -*-
"""Steps 3 and 4. Worm axis (skeleton correction) and head detection by the cascade with tip correction by the border radius.

The code in service is production/ancrage_cascade_v3c13_rayon_blasteme_v5s1neg_s040.py, imported unchanged (module A3b), with the
axis of production/cadrage_squelette_skan2.py and the shared functions of production/ancrage_cascade.py. This wrapper
only sets the model paths (parameters/models.csv) and the thresholds (parameters/cascade.csv, parameters/axis.csv)
on the module attributes, then calls A3b.ancrer(bgr, refined_mask).

Step 3, axis (skeleton correction, cadrage_squelette_skan2.squelette and extremites). Morphological skeleton of the worm
mask (skimage skeletonize, no pruning), longest path of the skeleton graph (skan, then Dijkstra distances,
scipy.sparse.csgraph.dijkstra, between end points), prolongation of each end up to the contour along the direction of
the last pixels (angle of the path, no global principal line), smoothing and resampling to N_AXE = 200 points
(axe_largeur.lisser), length L. A compact worm without a usable skeleton falls back on the principal axis (PCA). The two
ends give the tip and the thumbnail centre, placed at POS_CENTRE = 0.12 L from the tip along the axis.

Step 4, cascade v3c13 (20 October 2026, v3c12 with the blastema threshold 0.40, itself v3c8 with only the tip correction changed):
  1. blastema: thumbnail of side 0.90 L at each end (band 0.30 L), U-Net v5_s1neg, objects p >= SEUIL_BL (0.40) of at
     least A_MIN px. Kept object (objet_blasteme): the ADMITTED object closest to the tip, admitted if its extent at
     p >= 0.50 contains the tip or if one of its pixels is at most CONTACT_PX = 10 px from the worm contour,
  2. tip correction by the border radius: for the end closest to a blastema whose tip is not inside the blastema object,
     on the half of the axis on the tip side, r(i) = distance from P(i) to the worm border (cv2.distanceTransform) and
     g(i) = distance from P(i) to the blastema object. P(i) is in connection when g(i) <= r(i) (its inscribed circle
     reaches the blastema), and the point kept is the point in connection farthest from the blastema centre, with no
     threshold (fallback when none: largest r(i) - g(i)). The corrected tip is the last worm pixel (blastema included)
     on the half line from that point through the blastema centre, the thumbnail centre is moved back by 0.12 L,
  3. eyes: eye U-Net v5_jeu3E on a thumbnail of side 0.60 L at each candidate. The eye step decides when a maximum
     response reaches PEAK_ANCRAGE (0.85): among the candidates carrying a blastema (YEUX_SUR_BLASTEME), the one with
     the most eyes >= SEUIL_YEUX_TETE (0.85), ties broken by the mean score,
  4. otherwise blastema: relative margin of the blastema areas >= MARGE_BL (0.84), largest area,
  5. otherwise darkness: Otsu index on NB_COUPES = 60 perpendicular cuts over 0 to 0.30 of the axis from each end, the
     darkest end.
Returned point: barycentre of the two best eye peaks (>= AC.PEAK_THR 0.60) when eyes are seen, otherwise the tip.
Accuracy: v3c13 0.9953 on 2777 worms (13 errors) against v3c12 0.9950 (14 errors), test_ancrage_cascade_v3c13_s040 (non-inferior).

References (verified). Nunez-Iglesias J. et al. (2018). PeerJ 6, e4312, DOI 10.7717/peerj.4312 (skan). Dijkstra E. W.
(1959). Numerische Mathematik 1, 269-271, DOI 10.1007/BF01386390. Zhang T. Y., Suen C. Y. (1984). A fast parallel
algorithm for thinning digital patterns. Communications of the ACM 27(3), 236-239, DOI 10.1145/357994.358023
(skeletonize). Otsu N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems,
Man and Cybernetics 9(1), 62-66, DOI 10.1109/TSMC.1979.4310076. Borgefors G. (1986). Distance transformations in digital
images. CVGIP 34(3), 344-371, DOI 10.1016/S0734-189X(86)80047-0. Blum H. (1967). A transformation for extracting new
descriptors of shape. MIT Press, 362-380 (no DOI). Ronneberger O. et al. (2015). arXiv 1505.04597.
"""
import numpy as np
from . import _env
from ._env import AC, A3b, CS, apply_parameters, show_device
from .refinement import load_unets


class HeadCascade:
    """hc = HeadCascade(); r = hc.anchor(bgr, refined_mask). r: dict of A3b.ancrer (x, y, stage, confidence, tip,
    centre, L, axis_kind, n_blastemes, corrected, source, candidates, diag) or None if the mask is under 300 px."""

    def __init__(self, verbose=True):
        show_device()
        apply_parameters("axis", {"": CS})
        self.P = apply_parameters("cascade", {"A3b": A3b, "AC": AC})
        self.mbl, self.mey = load_unets()
        if verbose:
            print("head cascade v3c13 (tip by border radius) | blastema %.2f | eyes count %.2f decision %.2f | margin %.2f | contact %d px"
                  % (A3b.SEUIL_BL, A3b.PEAK_THR, A3b.PEAK_ANCRAGE, A3b.MARGE_BL, A3b.CONTACT_PX))

    def anchor(self, bgr, mask):
        return A3b.ancrer(bgr, mask)

    @staticmethod
    def axis(mask):
        """Smoothed axis (N_AXE x 2, y x), length L, nature (skeleton or fallback), and the two (tip, centre) ends."""
        wm = CS.masque_ver(mask)
        P, L, nature = CS.squelette(wm)
        ends, L2, meth = CS.extremites(wm)
        return dict(axis=P, L=float(L2), nature=nature, ends=ends, method=meth)

    @staticmethod
    def summary(r):
        """Flat record of a cascade result, same columns as the production journal ancres_composite_v20.csv."""
        if r is None: return dict(stage="masque_vide")
        return dict(stage=r["stage"], confidence=round(float(r["confidence"]), 4), x_tete=float(r["x"]), y_tete=float(r["y"]),
                    tip_y=round(float(r["tip"][0]), 1), tip_x=round(float(r["tip"][1]), 1),
                    centre_y=round(float(r["centre"][0]), 1), centre_x=round(float(r["centre"][1]), 1),
                    L=round(float(r["L"]), 1), axis_kind=r["axis_kind"], n_blastemes=r["n_blastemes"],
                    tip_corrected=int(r["corrected"]), source=r["source"])
