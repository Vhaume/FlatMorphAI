# CHAIN v20 (20 October 2026): copy of raffiner_masques_blasteme_v5s1neg_s070_annotfoi_lisse_env_p10_ouv3_skan2.py (unmodified). Blastema threshold 0.40 (calibrer_seuil_objet_blasteme_20261020_023318,
# maximal IoU of the blastema object) and cascade ancrage_cascade_v3c13_rayon_blasteme_v5s1neg_s040 (v3c12 at threshold 0.40).
# SKAN VERSION 2 (17 October 2026): axis from cadrage_squelette_skan22 (angle extension, no principal line).
# Copy of the skan version, itself a copy of raffiner_masques_blasteme_v5s1neg_s070_annotfoi_lisse_env_p10_ouv3 (unmodified),
# the only difference being that the worm axis comes from cadrage_squelette_skan2 (skeleton without pruning, longest path by
# skan and Dijkstra's algorithm, scipy.sparse.csgraph.dijkstra), substituted for cadrage_squelette in every imported module.
# Output: masks ..._ouv3_skan2. References: skan, Nunez-Iglesias et al. 2018, PeerJ, DOI 10.7717/peerj.4312;
# Dijkstra 1959, Numerische Mathematik, DOI 10.1007/BF01386390.
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Opening variant (lisse_env_p10_ouv3): copy of raffiner_masques_blasteme_v5s1neg_s070_annotfoi_lisse_env_p10, which stays
# intact. Feedback of 17 October 2026: a thin outgrowth on 2023-01-06-J3-R2-DMSO1-Q2 (translucent filament stuck to the head,
# segmenter probability 0.95, hence not correctable by a threshold). Diagnosis over 114 annotated images: 3 images with
# notable overshoot, all of them a pale translucent object stuck to the worm (filament, blurred debris). Addition, after the
# kept hull: morphological opening of the mask by a disc of radius 0.03 L, kept only inside the added blastema (Mask2Former
# is never removed). Structures narrower than 0.06 L are deleted, the blastema (a wide cap) is preserved.
#   Rule fixed before computation (eval_ouverture.py): overshoot beyond 5 px upper bound <= +0.005 and inclusion lower bound
#   >= -0.005 on both samples, then the highest IoU as confirmation. Confirmation (54 images, 24 worms), minus
#   production: IoU +0.0345 [+0.0179, +0.0535], overshoot -0.0034 [-0.0095, +0.0001], inclusion +0.0047 [+0.0032, +0.0066].
#   Selection (60 images): IoU +0.0132 [-0.0020, +0.0282]. DMSO1: overshoot 0.136 (production) -> 0.011;
#   2023-04-17-J6-R2-B+E-Q5: 0.108 -> 0.044.
# Reference, verified, not clickable. Serra J. (1982). Image Analysis and Mathematical Morphology. Academic Press,
# ISBN 978-0-12-637240-3 (morphological opening).
# Kept-hull variant (lisse_env_p10): copy of raffiner_masques_blasteme_v5s1neg_s070_annotfoi_lisse_env, which stays
# intact. Feedback of 17 October 2026: on 2023-01-06-J3-R2-DMSO1-Q2 the raw hull fills a notch with a triangle outside
# the annotated blastema -- the very overshoot the 0.70 threshold was meant to limit. The pixels gained by the hull are kept
# only where the segmenter probability (maximum of the two back-projected thumbnails) is at least 0.10 (variant H_p10,
# eval_enveloppe_garde.py, rule fixed before computation: overshoot beyond 5 px upper bound <= +0.005 and inclusion lower
# bound >= -0.005 on both samples, then the highest blastema IoU as confirmation).
#   Confirmation (54 images, 24 worms), H_p10 minus production: IoU +0.0319 [+0.0159, +0.0494], overshoot beyond 5 px
#   -0.0004 [-0.0017, +0.0005], inclusion +0.0051 [+0.0034, +0.0071]. Selection (60 images): IoU +0.0147 [-0.0008, +0.0303],
#   overshoot +0.0002 [-0.0000, +0.0004]. DMSO1: overshoot 0.222 (raw hull) brought back to 0.105 (production 0.136).
# Hull variant (lisse_env): copy of raffiner_masques_blasteme_v5s1neg_s070_annotfoi_lisse, which stays intact with its
# output folder. Only addition, after the V3 smoothing: each added blastema piece (component of the mask outside Mask2Former)
# is replaced by its convex hull, restricted to the segmenter band (0.30 L), holes filled (variant H of
# 17 October 2026, eval_cavites.py and eval_enveloppe.py). Retained after the visual feedback (the contour follows the
# annotation better) and the IoU of the added blastema with the annotated blastema outside Mask2Former, a measure already used for the 0.70 threshold.
#   Confirmation sample (54 images, 24 worms distinct from the first sample), H minus production:
#   IoU +0.0271 [+0.0096, +0.0463], inclusion +0.0055 [+0.0037, +0.0077], overshoot beyond 5 px from the annotation
#   +0.0021 [+0.0000, +0.0059], 95 % Hausdorff distance -0.0033 L [-0.0071, +0.0002]. Raw overshoot +0.0100
#   [... , +0.0213], of which 58 % within 5 px of the trace. Choice made after reading both samples (to be declared).
#   Known limitation: a hull may fill a deep notch with a triangle (2023-01-06-J3-R2-DMSO1-Q2).
# Reference, verified, not clickable. Graham R. L. (1972). An efficient algorithm for determining the convex hull of a finite
# planar set. Information Processing Letters 1(4), 132-133. DOI 10.1016/0020-0190(72)90045-2.
# Smoothed variant (lisse): copy of raffiner_masques_blasteme_v5s1neg_s070_annotfoi, which stays intact with its output
# folder. Only change: the V3 variant retained by the pre-registered rule of 17 October 2026 (eval_lissage_image.py,
# 60 annotated validation and test images, 20 per state, 40 worms, bootstrap by worm, 2 000 draws, seed 42):
#   1. in each thumbnail, the blastema p >= 0.70 is blurred (Gaussian, sigma 2 px) then thresholded at 0.5 before the
#      back-projection (instead of back-projecting the raw probability);
#   2. after the fusion with Mask2Former (components touching the mask, holes filled, contact rule), the final mask
#      is blurred (Gaussian, sigma 0.01 L) and thresholded at 0.5 inside the added zone dilated by 3 sigma only. No
#      Mask2Former pixel is removed.
#   Result, V3 minus production: inclusion of the annotated blastema +0.0003 [-0.0001, +0.0006], overshoot -0.0006
#   [-0.0010, -0.0001], local contour roughness -0.0059 [-0.0092, -0.0033] (acceptable and smoother, retained).
#   Annotated images keep the annotation alone (annotfoi); the smoothing applies to the blastema predicted by the model.
# Smoothing references, verified, not clickable. Gonzalez R. C., Woods R. E. (2018). Digital Image Processing, 4th ed.,
# Pearson, ISBN 978-0-13-335672-4 (Gaussian filtering, chapter 3). Lindeberg T. (1994). Scale-space theory: a basic tool
# for analysing structures at different scales. Journal of Applied Statistics 21(1-2), 225-270. DOI 10.1080/757582976.
# Annotation-only variant (annotfoi): on the images you annotated, the blastema added to the mask is YOUR annotation
# alone; the model prediction that overshoots is discarded. Zero overshoot on the annotated images.
# NON-annotated images keep the v5_s1neg prediction at threshold 0.70. Copy of raffiner_masques_blasteme_v5s1neg_s070.
# Final refinement: v5_s1neg segmenter (halos corrected) at threshold 0.70 (overshoot tightened). Combines both
# corrections. Copy of raffiner_masques_blasteme_v3, distinct output folder, nothing existing modified.
# =============================================================================================
# Worm mask refinement, version 3: segmenter blasteme_unet_384_v5_s1neg + contact rule + expert annotations
#
# Copy of raffiner_masques_blasteme.py (v1, v3 segmenter) which stays intact, as do raffiner_masques_blasteme_v2_contact.py
# and their output folders. Three changes, fixed before the pass:
#   1. segmenter blasteme_unet_384_v5_s1neg, trained with a band of 0.30 L around the worm instead of 0.12 L
#      (train_blasteme_segmenter_384_v5_dil030_20261011_190119, pre-registered rule, v5 minus v4: test A Dice
#      +0.0087 [+0.0014, +0.0165], test B +0.0040 [-0.0042, +0.0162], overshoot -0.0008 [-0.0025, +0.0005]; v5 retained).
#      The thumbnails are built with the band recorded in the model (dil = 0.30), positioning unchanged.
#      The path and the band are changed in this notebook only (A3.CK_BL3, DIL_BL); ancrage_cascade_v3.py and
#      cadrage_squelette.py are not modified;
#   2. the v2 contact rule is integrated: each added piece (component of the refined mask outside the Mask2Former
#      mask) is kept only if (pixels of the piece adjacent to the mask) / sqrt(area of the piece) >= 2.5;
#   3. expert annotations: for an image traced in cyan_blasteme, the annotated blastema (annotation cyan minus
#      the cyan already present in the image, 3 x 3 opening) is added to the mask, components of at least 30 pixels that
#      touch the mask. No restriction to the dilated worm: an annotation may overflow the dilated mask when
#      Mask2Former missed the head (2023-04-18-J7-R3-B+E-Q6, 1119 pixels outside the dilated worm).
# Nothing is removed from the Mask2Former mask. Holes filled, a single component.
#
# Pre-registered evaluation (section 3), before the annotations are added (otherwise inclusion equals 1 by construction):
# positive images of the TEST split of v5 (blasteme_seg_split_v5_dil030.csv, identical to v4, worms never seen by the segmenter), share of the
# annotated blastema included in the mask, mask refined by the model minus Mask2Former mask, bootstrap by worm
# (2000 draws, seed 42). The refinement is retained if the lower bound of the 95 % interval is > 0 AND if the addition outside
# the annotated blastema (pixels added more than 5 px away from the annotated blastema, relative to the mask area) has a median
# <= 0.01 and a 90th percentile <= 0.03 (same thresholds as v1). Also reported: the same measure for the v2 masks (v3 segmenter
# + contact) over the same images, per state, and control sheets.
#
# Output: essai_raffine_v5s1neg_s070_annotfoi_masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme (a name outside the pattern read by the
# downstream notebooks, to be renamed only after visual validation), log Claude outputs/masques_raffines_v3/.
# Graphics card, about 15 minutes, automatic resume. Seed 42.
#
# Verified references, not clickable. Ronneberger O., Fischer P., Brox T. (2015). U-Net: Convolutional Networks for
# Biomedical Image Segmentation. MICCAI 2015, arXiv 1505.04597. Efron B. (1979). Bootstrap methods: another look at the
# jackknife. Annals of Statistics 7(1), 1-26. DOI 10.1214/aos/1176344552. Roberts D. R. et al. (2017).
# Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8),
# 913-929. DOI 10.1111/ecog.02881. Connected components: Wu K., Otoo E., Suzuki K. (2009). Optimizing two-pass
# connected-component labeling algorithms. Pattern Analysis and Applications 12, 117-135. DOI 10.1007/s10044-008-0109-y.
# =============================================================================================
import os, sys, glob, time, datetime, numpy as np, pandas as pd, cv2, torch
import cadrage_squelette_skan2 as _CSK
sys.modules["cadrage_squelette"] = _CSK   # skan + Dijkstra axis for every module imported afterwards
GRAINE = 42; np.random.seed(GRAINE); torch.manual_seed(GRAINE); torch.cuda.manual_seed_all(GRAINE)
HORO = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class _Double(object):
    def __init__(self, flux, chemin): self.flux = flux; self.fic = open(chemin, "w", encoding="utf-8")
    def write(self, t): self.flux.write(t); self.fic.write(t); self.fic.flush()
    def flush(self): self.flux.flush(); self.fic.flush()


if not isinstance(sys.stdout, _Double): sys.stdout = _Double(sys.stdout, "raffiner_masques_blasteme_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2_%s.log" % HORO)
REPO = os.getcwd()
if REPO not in sys.path: sys.path.insert(0, REPO)
if not torch.cuda.is_available(): raise SystemExit("graphics processor unavailable, run aborted.")
print("graphics processor:", torch.cuda.get_device_name(0))
import ancrage_cascade as AC
import ancrage_cascade_v3 as A3
import cadrage_squelette as CS
assert CS is _CSK and CS.squelette is _CSK.squelette, "the axis must come from cadrage_squelette_skan2"
print("worm axis: cadrage_squelette_skan2 (skan + Dijkstra, angle extension, no principal line)")
A3.CK_BL3 = os.path.join("checkpoints_regeneration", "blasteme_unet_384_v5_s1neg", "blasteme_unet_384_best.pth")   # this notebook only
if not os.path.exists(A3.CK_BL3): raise SystemExit("v5 segmenter not found: %s" % A3.CK_BL3)
_ck = torch.load(A3.CK_BL3, map_location="cpu", weights_only=False)
assert _ck.get("apprentissage_termine", False), "the v5 segmenter is not complete"
DIL_BL = float(_ck["dil"]); assert abs(DIL_BL - 0.30) < 1e-9, "unexpected segmenter band"
mdl_bl, _ = A3.charger(True)
assert next(mdl_bl.parameters()).device.type == "cuda", "the blastema segmenter is not on the graphics card"
print("blastema segmenter:", A3.CK_BL3, "| epoch", _ck.get("epoque"), "| Dice val %.3f" % _ck["val_dice"], "| band %.2f L" % DIL_BL)

REGEN = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_regeneration")
IMG = os.path.join(REGEN, "images")
MK = os.path.join(REGEN, "masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme")
V2 = os.path.join(REGEN, "essai_raffine_v2_masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme")
CB = os.path.join(REGEN, "cyan_blasteme")
OUT = os.path.join(REGEN, "masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme_raffine_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2")
SORTIE = os.path.join("Claude outputs", "masques_raffines_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2"); os.makedirs(SORTIE, exist_ok=True)
JOURNAL = os.path.join(SORTIE, "raffinement_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2.csv")
SEUIL_P = 0.40; SIGMA_VIGNETTE = 2.0; SIGMA_IMAGE = 0.01; SEUIL_GARDE = 0.10; RAYON_OUV = 0.03; SEUIL_CONTACT = 2.5; MIN_COMP_ANNOT = 30; MIN_ANNOT = 150
SIGNALEES = ["2023-04-18-J7-R3-B+E-Q6.jpg", "2023-02-16-T12-J10-R1-PET-Q4.jpg", "2023-02-17-T12-J11-R3-HDPE-Q3.jpg",
             "2023-04-18-J7-R2-B-Q1.jpg", "2023-02-12-T12-J6-R3-Ctrl-Q4.jpg", "2023-02-16-T12-J10-R1-HDPE-Q3.jpg"]
CYAN = {os.path.basename(p): p for p in glob.glob(os.path.join(CB, "State_*", "*.jpg"))}
for c in (MK, V2, CB): assert os.path.isdir(c), "not found: %s" % c
print("input masks:", os.path.basename(MK), "| cyan annotations:", len(CYAN))


def fill_holes(r):
    H, W = r.shape; ff = (~r).astype(np.uint8); msk = np.zeros((H + 2, W + 2), np.uint8)
    for y, x in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        if ff[y, x]: cv2.floodFill(ff, msk, (x, y), 0)
    return CS.masque_ver(((r | (ff > 0)).astype(np.uint8)) * 255) > 0


def contact(wm, r):
    """Contact rule (v2): keep the added pieces whose contact with the mask is clear-cut."""
    voisin = cv2.dilate(wm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    k, lab, s, _ = cv2.connectedComponentsWithStats((r & (wm == 0)).astype(np.uint8), 8)
    garde = wm > 0; n_g = n_r = 0; a_r = 0
    for i in range(1, k):
        comp = lab == i; A = float(s[i, 4])
        if float((comp & voisin).sum()) / np.sqrt(A) >= SEUIL_CONTACT: garde |= comp; n_g += 1
        else: n_r += 1; a_r += A
    return garde, n_g, n_r, a_r


def annotation_brute(nm, bgr):
    """Annotated blastema (without restriction to the dilated worm) or None."""
    if nm not in CYAN: return None
    an = cv2.imread(CYAN[nm])
    if an is None or an.shape != bgr.shape: return None
    a = CS.cyan_strict(an) & ~CS.cyan_strict(bgr)
    a = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return a if a.sum() >= MIN_ANNOT else None


def lisser_local(wm, f, sig):
    """Smoothing 2: Gaussian blur of the final mask thresholded at 0.5, inside the added zone dilated by 3 sigma. Mask2Former is never removed."""
    add = f & ~(wm > 0)
    if not add.any(): return f
    r = max(1, int(round(3 * sig))); Z = cv2.dilate(add.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    Rs = cv2.GaussianBlur(f.astype(np.float32), (0, 0), sig) >= 0.5
    return fill_holes((wm > 0) | (Rs & Z) | (f & ~Z))


def enveloppe(wm, f, wd, Bp):
    """Convex hull of each added piece outside Mask2Former, restricted to the segmenter band, holes filled."""
    add = (f & ~(wm > 0)).astype(np.uint8); n, lab, _, _ = cv2.connectedComponentsWithStats(add, 8); out = f.copy()
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 3: continue
        h = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)); hm = np.zeros(f.shape, np.uint8); cv2.fillPoly(hm, [h], 1)
        out |= (hm > 0) & (wd > 0) & (Bp >= SEUIL_GARDE)                                    # support of the segmenter
    return fill_holes((wm > 0) | out)


def ouvrir(wm, f, rad):
    """Opening by a disc of radius rad (px), applied to the added blastema only: removes the thin outgrowths."""
    r = max(1, int(round(rad))); o = cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return fill_holes((wm > 0) | (o & f))


def raffiner(bgr, mraw):
    """Mask2Former mask, mask refined by the model (contact applied), length, counters."""
    wm = CS.masque_ver(mraw)
    if wm.sum() < 300: return None
    ends, L, meth = CS.extremites(wm); H, W = wm.shape
    k = max(1, int(DIL_BL * L)); wd = cv2.dilate(wm.astype(np.uint8), np.ones((k, k), np.uint8))   # band of the v5 segmenter
    B = np.zeros((H, W), np.float32); Bp = np.zeros((H, W), np.float32)
    for tip, centre in ends:
        M = CS.affine(tip, centre, L); p = AC._pred(mdl_bl, CS.vignette(bgr, wd, M))
        b = cv2.GaussianBlur((p >= SEUIL_P).astype(np.float32), (0, 0), SIGMA_VIGNETTE) >= 0.5          # smoothing 1: inside the thumbnail
        B = np.maximum(B, CS.retroprojeter(b.astype(np.float32), M, (H, W)))
        Bp = np.maximum(Bp, CS.retroprojeter(p, M, (H, W)))                                        # probability, for the guard
    cand = ((B >= 0.5) & (wd > 0) & (wm == 0)).astype(np.uint8)                                         # back-projected mask, threshold 0.5
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    voisin = cv2.dilate(wm, np.ones((3, 3), np.uint8)) > 0
    garder = np.zeros(n, bool)
    for k in range(1, n): garder[k] = voisin[lab == k].any()
    r = fill_holes((wm > 0) | (garder[lab] & (cand > 0)))
    r, n_g, n_r, a_r = contact(wm, r)
    return wm > 0, ouvrir(wm, enveloppe(wm, lisser_local(wm, fill_holes(r), SIGMA_IMAGE * L), wd, Bp), RAYON_OUV * L), L, n_g, n_r, a_r


def ajouter_annotation(wm, r, a):
    """Union of the annotation components (>= MIN_COMP_ANNOT px) that touch or overlap the refined mask."""
    voisin = cv2.dilate(r.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    k, lab, s, _ = cv2.connectedComponentsWithStats(a.astype(np.uint8), 8); f = r.copy()
    for i in range(1, k):
        comp = lab == i
        if s[i, 4] >= MIN_COMP_ANNOT and (comp & voisin).any(): f |= comp
    return fill_holes(f)


# In[ ]:


# =============================================================================================
# 2. Pass over every image (resume: images already logged are skipped)
# =============================================================================================


# fixed columns: a non-annotated image has no evaluation columns; without a fixed list the batches appended to the log
# would not have the same number of columns (read error of the first pass of 11 October)
COLS = ["image", "classe", "L", "aire_masque", "aire_ajoutee_modele", "morceaux_gardes", "morceaux_retires_contact", "aire_retiree_contact",
        "annotee", "aire_annot", "inclus_m2f", "inclus_modele", "inclus_final", "ajout_hors_blasteme", "aire_ajoutee_annotation",
        "inclus_v2", "ajout_hors_blasteme_v2", "part_ajoutee"]
if os.path.exists(JOURNAL) and list(pd.read_csv(JOURNAL, sep=";", nrows=0).columns) != COLS:
    raise SystemExit("%s exists with different columns (previous pass); move it before running again." % JOURNAL)
faits = set(pd.read_csv(JOURNAL, sep=";").image) if os.path.exists(JOURNAL) else set()
rows, t0, n = [], time.time(), 0


def vider():
    global rows
    if rows: pd.DataFrame(rows).reindex(columns=COLS).to_csv(JOURNAL, sep=";", index=False, mode="a", header=not os.path.exists(JOURNAL)); rows = []


for st in ["State_%d" % k for k in range(1, 7)]:
    os.makedirs(os.path.join(OUT, st), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(IMG, st, "*.jpg"))):
        nm = os.path.basename(f)
        if nm in faits and os.path.exists(os.path.join(OUT, st, nm)): continue
        bgr = cv2.imread(f); mraw = cv2.imread(os.path.join(MK, st, nm), 0)
        if bgr is None or mraw is None or mraw.shape != bgr.shape[:2]:
            print("   skipped:", nm); continue
        res = raffiner(bgr, mraw)
        if res is None:
            cv2.imwrite(os.path.join(OUT, st, nm), mraw); continue
        wm, r, L, n_g, n_r, a_r = res
        d = dict(image=nm, classe=st, L=L, aire_masque=int(wm.sum()), aire_ajoutee_modele=int(r.sum() - wm.sum()),
                 morceaux_gardes=n_g, morceaux_retires_contact=n_r, aire_retiree_contact=int(a_r), annotee=0)
        a = annotation_brute(nm, bgr); fin = r
        if a is not None:
            fin = ajouter_annotation(wm, (wm > 0), a); proche = cv2.dilate(a.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0   # annotation only: base = worm body, the prediction that overshoots is discarded
            d.update(annotee=1, aire_annot=int(a.sum()), inclus_m2f=float((a & wm).sum() / a.sum()), inclus_modele=float((a & r).sum() / a.sum()),
                     inclus_final=float((a & fin).sum() / a.sum()), ajout_hors_blasteme=float((fin & ~wm & ~proche).sum() / wm.sum()),
                     aire_ajoutee_annotation=int(fin.sum() - r.sum()))
            r2p = os.path.join(V2, st, nm)
            if os.path.exists(r2p):
                r2 = cv2.imread(r2p, 0) >= 128
                if r2.shape == wm.shape: d.update(inclus_v2=float((a & r2).sum() / a.sum()), ajout_hors_blasteme_v2=float((r2 & ~wm & ~proche).sum() / wm.sum()))
        d["part_ajoutee"] = float((fin.sum() - wm.sum()) / max(wm.sum(), 1))
        cv2.imwrite(os.path.join(OUT, st, nm), fin.astype(np.uint8) * 255, [cv2.IMWRITE_JPEG_QUALITY, 100])
        rows.append(d); n += 1
        if n % 200 == 0: vider(); print("   %d images, %.0f s" % (n, time.time() - t0))
vider()
D = pd.read_csv(JOURNAL, sep=";").drop_duplicates("image", keep="last")
print("pass finished: %d images in this run, %.0f s | images with an addition: %d / %d | pieces removed by the contact rule: %d over %d images"
      % (n, time.time() - t0, int((D.part_ajoutee > 0).sum()), len(D), int(D.morceaux_retires_contact.sum()), int((D.morceaux_retires_contact > 0).sum())))
A_ = D[D.annotee == 1]
print("annotated images: %d | annotation added to the mask (pixels outside the model-refined mask): %d images > 0, %d images > 1000 px"
      % (len(A_), int((A_.aire_ajoutee_annotation > 0).sum()), int((A_.aire_ajoutee_annotation > 1000).sum())))


# In[ ]:


# =============================================================================================
# 3. Pre-registered evaluation: v5 test (worms never seen by the v5 segmenter), before the annotations are added
# =============================================================================================
SP = pd.read_csv(os.path.join(REGEN, "blasteme_seg_split_v5_dil030.csv"), sep=";")
TEST = set(SP[(SP.split == "test") & (SP.positive == 1)].group); VER = dict(zip(SP.group, SP.ver))
E = D[D.annotee == 1].copy(); E["test"] = E.image.isin(TEST).astype(int); E["ver"] = E.image.map(lambda g: VER.get(g, g))
E.to_csv(os.path.join(SORTIE, "evaluation_inclusion_blasteme_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2.csv"), sep=";", index=False)
T = E[E.test == 1].copy()


def boot(dif, vers):
    g = pd.DataFrame(dict(d=dif, v=vers)).groupby("v").d.agg(["size", "sum"]); rng = np.random.RandomState(GRAINE); bs = []
    for _ in range(2000):
        j = rng.randint(0, len(g), len(g)); bs.append(g["sum"].values[j].sum() / g["size"].values[j].sum())
    return np.percentile(bs, [2.5, 97.5])


T["d"] = T.inclus_modele - T.inclus_m2f; lo, hi = boot(T.d.values, T.ver.values)
med_aj, p90_aj = T.ajout_hors_blasteme.median(), T.ajout_hors_blasteme.quantile(0.9)
print("\nv5 TEST: %d annotated blastemas, %d worms" % (len(T), T.ver.nunique()))
print("inclusion of the annotated blastema: Mask2Former %.4f | v3 refined (model) %.4f | difference %+.4f [%+.4f, %+.4f]" % (T.inclus_m2f.mean(), T.inclus_modele.mean(), T.d.mean(), lo, hi))
print("addition outside the annotated blastema (share of the mask area): median %.4f, 90th percentile %.4f" % (med_aj, p90_aj))
ok = (lo > 0) and (med_aj <= 0.01) and (p90_aj <= 0.03)
print("VERDICT:", "REFINEMENT RETAINED" if ok else "REFINEMENT NOT RETAINED (pre-registered rule)")
T2 = T.dropna(subset=["inclus_v2"])
if len(T2):
    d2 = (T2.inclus_modele - T2.inclus_v2).values; lo2, hi2 = boot(d2, T2.ver.values)
    print("reported: v3 refined minus v2 refined (v3 segmenter + contact), %d images: %+.4f [%+.4f, %+.4f] | addition outside the v2 blastema: median %.4f, 90th percentile %.4f"
          % (len(T2), d2.mean(), lo2, hi2, T2.ajout_hors_blasteme_v2.median(), T2.ajout_hors_blasteme_v2.quantile(0.9)))
print("\nper state (test):"); print(T.groupby("classe")[["inclus_m2f", "inclus_v2", "inclus_modele", "ajout_hors_blasteme"]].mean().round(4).to_string())
print("\nall annotated images (optimistic, part of them served for training):")
print(E.groupby("classe")[["inclus_m2f", "inclus_v2", "inclus_modele", "inclus_final"]].mean().round(4).to_string())
print("blastemas included below 80 %% : Mask2Former %d, v2 %d, v3 model %d, v3 final %d over %d"
      % (int((E.inclus_m2f < 0.8).sum()), int((E.inclus_v2 < 0.8).sum()), int((E.inclus_modele < 0.8).sum()), int((E.inclus_final < 0.8).sum()), len(E)))


# In[ ]:


# =============================================================================================
# 4. Plates: red Mask2Former, green final v3 mask. Flagged images, 24 largest additions, 12 largest additions
#    coming from the annotations.
# =============================================================================================
def tuile(nm, st, titre):
    im = cv2.imread(os.path.join(IMG, st, nm)); fm = cv2.imread(os.path.join(OUT, st, nm), 0)
    if im is None or fm is None: return None
    for m, col in (((fm >= 128).astype(np.uint8), (0, 255, 0)), ((cv2.imread(os.path.join(MK, st, nm), 0) >= 128).astype(np.uint8), (0, 0, 255))):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, col, 3)
    ys, xs = np.nonzero(fm >= 128)
    if len(xs): im = im[max(0, ys.min() - 60):ys.max() + 60, max(0, xs.min() - 60):xs.max() + 60]
    t = cv2.resize(im, (360, 360)); cv2.putText(t, titre, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1); return t


def planche(liste, nom):
    tiles = [t for t in (tuile(nm, st, ti) for nm, st, ti in liste) if t is not None]
    if not tiles: return
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(SORTIE, nom), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("plate:", os.path.join(SORTIE, nom))


Di = D.set_index("image")
planche([(nm, Di.classe[nm], "%s +%.1f%%" % (nm[:30], 100 * Di.part_ajoutee[nm])) for nm in SIGNALEES if nm in Di.index], "planche_images_signalees_v3.jpg")
planche([(r.image, r.classe, "%s +%.1f%%" % (r.image[:30], 100 * r.part_ajoutee)) for r in D.sort_values("part_ajoutee", ascending=False).head(24).itertuples()], "planche_plus_gros_ajouts_v3.jpg")
planche([(r.image, r.classe, "%s annot +%d px" % (r.image[:28], r.aire_ajoutee_annotation)) for r in A_.sort_values("aire_ajoutee_annotation", ascending=False).head(12).itertuples()], "planche_ajouts_annotation_v3.jpg")
print("red: Mask2Former, green: final v3 mask | folder", os.path.basename(OUT), "to be renamed masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme_raffine_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2 after validation | done.")
