# -*- coding: utf-8 -*-
"""Step 6. The 128 morphometric descriptors of one worm, among which the 86 read by the state classifier.

Same mechanism as production/enrich_metrics_v20.py, of which this module executes the code unchanged:
  * block "computation functions of the original programs": the descriptor functions are read from the eight original
    programs (precompute_regeneration_metrics_v2, eye_inference_and_features, generate_composite_and_features,
    enrich_metrics_blasteme, enrich_metrics_blasteme_forme, couleur_locale_blasteme, enrich_metrics_yeux_unet_v5,
    enrich_metrics_pupille, all in production/) and executed as they are. The only text changes, checked by assert in
    enrich_metrics_v20, are the blastema threshold 0.5 -> 0.40 (forme_feats, couleur_locale) and the head end imposed by
    the chain in enrich_metrics_blasteme,
  * functions pics, vers_image, chaine, masque_384, tete_haute_384, haut_dans_repere and descripteurs.
Two replacements are made here, checked to occur once: descripteurs reads the image and the refined mask from memory
(dictionary ENTREE) instead of the dataset folders, and the head (tip, thumbnail centre, L) comes from the cascade of
this pipeline, rounded to 0.1 px as in the production journal ancres_composite_v20.csv.
Inputs of every descriptor family: refined worm mask, head tip and centre of the head cascade, blastema U-Net v5_s1neg
on the thumbnail of side 0.90 L (band 0.30 L) with the object rule A3b.objet_blasteme at threshold 0.40, eye
U-Net v5_jeu3E on the thumbnail of side 0.60 L, counting threshold 0.61, peaks at least 7 px apart.
blast_seg_prop_yr (normalisation of blast_seg_prop by acquisition year with the statistics of the training table) is
not computed for an isolated image: it is not one of the 86 descriptors of the model (metriques_finales_v22.csv).
Families (prefix, number of columns, list in parameters/descriptor_columns.csv): shp shape (11), pig pigmentation (8),
pigp head pigmentation profile (13), tex texture (9), end ends (5), cl and clh light cap and CLAHE (10 + 10), eye eye
rings (19), eyeb eye surroundings (5), blast blastema without network and segmented blastema (22), loc local colour
(2), eyeun eye detector scores (7), pup eyespot maturation (7). Each original program documents its columns in its
header.

References (verified) of the descriptor formulas are in the headers of the original programs, among which Hu M.-K.
(1962). Visual pattern recognition by moment invariants. IRE Transactions on Information Theory 8(2), 179-187,
DOI 10.1109/TIT.1962.1057692. Haralick R. M., Shanmugam K., Dinstein I. (1973). Textural features for image
classification. IEEE Transactions on Systems, Man and Cybernetics SMC-3(6), 610-621, DOI 10.1109/TSMC.1973.4309314.
Otsu N. (1979). DOI 10.1109/TSMC.1979.4310076.
"""
import os
import threading

import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from . import _env
from ._env import AC, A3b, CS, PROD, read, exec_slice, apply_parameters, show_device
from .refinement import load_unets

PROD_FILE = "enrich_metrics_v22.py"
SELECTION = os.path.join(PROD, "metriques_finales_v22.csv")


class DescriptorComputer:
    def __init__(self, verbose=True):
        show_device()
        apply_parameters("axis", {"": CS})
        self.P = read("descriptors")
        self._mbl, self._mey = load_unets()
        assert abs(A3b.DIL_BL - float(self.P["DIL_BL"])) < 1e-9, "DIL_BL differs from the band of the blastema model"
        self._tl = threading.local()
        self._claim = threading.Lock()
        self._claimed = False
        ns = self._build_ns()
        self.ns = ns
        self.columns_all = [c for c in pd.read_csv(os.path.join(_env.EXPORT_ROOT, "parameters", "descriptor_columns.csv"),
                                                   sep=";")["descriptor"]]
        self.selected = list(pd.read_csv(SELECTION, sep=";")["descripteur"])
        if verbose:
            print("descriptors: %d columns, %d read by the state model | blastema threshold %.2f | eye threshold %.2f"
                  % (len(self.columns_all), len(self.selected), ns["SEUIL_BL"], ns["THR_EYE"]))


    def _build_ns(self):
        """One namespace: the descriptor functions of the original programs, read
        from their source and executed unchanged."""
        mbl, mey = self._mbl, self._mey
        ns = dict(os=os, np=np, pd=pd, cv2=cv2, torch=torch, F=F, AC=AC, A3b=A3b, CS=CS, PROD=PROD, mbl=mbl, mey=mey,
                  SEUIL_BL=float(self.P["SEUIL_BL"]), F_BL=float(self.P["F_BL"]), F_EYE=float(self.P["F_EYE"]),
                  DIL_BL=float(self.P["DIL_BL"]), THR_EYE=float(self.P["THR_EYE"]), MD_EYE=int(self.P["MD_EYE"]))
        exec_slice(PROD_FILE, "# ---- computation functions of the original programs", "# ---- chain inputs ----", ns,
                   replacements=[('def _src(f): return open(f, encoding="utf-8").read()',
                                  'def _src(f): return open(os.path.join(PROD, f), encoding="utf-8").read()')])
        ns.update(J=pd.DataFrame(columns=["tip_y", "tip_x", "centre_y", "centre_x", "L"]), IMGIDX={}, MKIDX={}, ENTREE={})
        exec_slice(PROD_FILE, "def pics(pr, thr, k):", "# ---- computation over the three tables", ns,
                   replacements=[('    ip, mp = IMGIDX.get(base), MKIDX.get(base)\n'
                                  '    if ip is None or mp is None: return None, "fichier absent"\n'
                                  '    bgr = cv2.imread(ip); mraw = cv2.imread(mp, 0)\n',
                                  '    bgr, mraw = ENTREE[base]\n')])
        return ns

    def _ns(self):
        """Namespace of THIS thread. ``descripteurs`` reads its image and its head
        from the module globals ENTREE / J, so a single shared namespace means two
        worms analysed at once overwrite each other's input -- which is what made
        the eye and blastema descriptors drift under concurrency. One namespace per
        thread removes the sharing; the code executed is the same text, so a given
        worm gives the same values whatever the number of workers."""
        ns = getattr(self._tl, "ns", None)
        if ns is None:
            with self._claim:
                take = not self._claimed
                self._claimed = True
            ns = self.ns if take else self._build_ns()   # the first thread reuses the one built at load
            self._tl.ns = ns
        return ns

    def compute(self, bgr, refined_mask, r3, name="image"):
        """bgr original image, refined_mask (grey 0..255, as read from the production file), r3 cascade result.
        Returns (dict of descriptors, status). The head is r3 rounded to 0.1 px like the production journal."""
        ns = self._ns(); base = os.path.basename(name)
        ns["ENTREE"] = {base: (bgr, refined_mask)}
        if r3 is not None:
            ns["J"] = pd.DataFrame([dict(tip_y=round(float(r3["tip"][0]), 1), tip_x=round(float(r3["tip"][1]), 1),
                                         centre_y=round(float(r3["centre"][0]), 1), centre_x=round(float(r3["centre"][1]), 1),
                                         L=round(float(r3["L"]), 1))], index=[base])
        else:
            ns["J"] = ns["J"].iloc[0:0]
        d, status = ns["descripteurs"](base)
        if d is None: return None, status
        d = dict(d); d.pop("_source_tete", None); d["blast_seg_prop_yr"] = np.nan
        return d, status
