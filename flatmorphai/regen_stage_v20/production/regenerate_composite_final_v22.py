# CHAIN v22 (22 October 2026): copy of regenerate_composite_final_v20.py (unmodified). Cascade ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 (start point:
# largest border radius among the points in contact with the blastema), blastema threshold 0.40, s040 masks unchanged.
# CHAIN v20 (20 October 2026): copy of regenerate_composite_final_v18.py (unmodified). Blastema threshold 0.40 (calibrer_seuil_objet_blasteme_20261020_023318,
# maximal IoU of the blastema object) and cascade ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 (v3c12 at threshold 0.40).
# Composites v18 (19 October 2026): copy of regenerate_composite_final_v17 (unmodified), cascade ancrage_cascade_v3c8_aragad_v5s1neg_s070.
# Output: processed_composite_384_uni_final_v22, log ancres_composite_v22.csv.
# Composites v17 (18 October 2026): copy of regenerate_composite_final_v16 (unmodified), cascade
# ancrage_cascade_v3c8_aragad_v5s1neg_s070 (tip corrected when it is not inside the blastema). Output:
# processed_composite_384_uni_final_v22, log ancres_composite_v22.csv.
# Composites of the v16 state classifier (18 October 2026): copy of regenerate_composite_final_v15 (unmodified). The
# cascade uses the final ARAGAD method (ancrage_cascade_v3c8_aragad_v5s1neg_s070, stopping at the peak of the angle theta
# before it breaks), retained by the experimenter. The verdict of test_ancrage_cascade_aragad8_v5s1neg_s070 is displayed as a check.
# Output: processed_composite_384_uni_final_v22, log ancres_composite_v22.csv.
# Composites of the v15 state classifier (17 October 2026): copy of regenerate_composite_final_v14 (unmodified), the ARAGA
# cascade being replaced by ARAGAD (ancrage_cascade_v3c5_aragad, trial test_ancrage_cascade_aragad_v5s1neg_s070).
# Version v14: copy of regenerate_composite_final_v12 (unmodified). The only changes are
# (1) the worm axis: cadrage_squelette_skan2 (skan + Dijkstra, angle extension, no principal line),
# substituted for cadrage_squelette in every module, on the ..._ouv3_skan2 masks, only if
# test_ancrage_cascade_skan2_v5s1neg_s070 RETAINED it, and (2) the cascade: ARAGA (ancrage_cascade_v3c4_araga, retained mode)
# if test_ancrage_cascade_araga_v5s1neg_s070 RETAINED it, otherwise v3c3 if test_ancrage_cascade_v3c3_v5s1neg_s070 retained
# it, otherwise v3c2. Output: processed_composite_384_uni_final_v15.
# References: skan, Nunez-Iglesias et al. 2018, PeerJ, DOI 10.7717/peerj.4312; Dijkstra 1959, DOI 10.1007/BF01386390.
# Composites of the v12 state classifier (17 October 2026): copy of regenerate_composite_final_v11 (unmodified), with only two
# changes. (1) Masks: forme_raffine_v5s1neg_s070_annotfoi_lisse_env_p10_ouv3, the refinement retained on
# 17 October (smoothing, convex hull kept p >= 0.10, opening 0.03 L). (2) Cascade: ancrage_cascade_v3c2_v5s1neg_s070
# (blastema threshold 0.70, trial test_ancrage_cascade_v3c2_v5s1neg_s070 RETAINED) by default, or, with CASCADE_V3C3=1,
# ancrage_cascade_v3c3_v5s1neg_s070 (local tip correction) only if test_ancrage_cascade_v3c3_v5s1neg_s070
# RETAINED it. Fix: the output folders are created AFTER the cascade is chosen (in v11 the _s070 suffix was
# added after their creation, and cv2.imwrite failed silently in a missing folder).
# Composites of the v11 state classifier: super-zoom and whole worm oriented like the thumbnails of the head chain.
# Copy of regenerate_composite_final_v10 (unmodified). Only the framing of the classifier images changes:
#   - super-zoom (channel 2): cut by cadrage_squelette.affine on the tip and the centre of the candidate retained by
#     the cascade (tip corrected by the blastema when needed), centre at 12 % of L from the tip along the axis, tip at
#     the top, side 0.30 L (fraction 0.15), inside the golden mask refined by the blastema. Same centre and same
#     orientation as the blastema (0.90 L) and eyespot (0.60 L) thumbnails; only the size differs.
#   - channel 1 (not read by the classifier, channels (0, 2, 2)): same framing, side 0.60 L, kept for the composite structure.
#   - whole worm (channel 0): straightened by its principal axis as in v10, then rotated 180 degrees if the head is at
#     the bottom, so that the head is always at the top.
# In v10 the super-zoom was a square aligned on the worm's global principal axis (PCA), centred on the barycentre of the
# eyespots or on the tip, of side 0.30 times the vertical extent of the straightened worm, head up or down depending on the worm.
# Output: processed_composite_384_uni_final_v11. Nothing is overwritten. Graphics processor required. Seed 42.
import os, glob, sys, time, datetime, numpy as np, cv2, torch
import pandas as pd
from collections import Counter
import cadrage_squelette_skan2 as _CSK
sys.modules["cadrage_squelette"] = _CSK   # skan 2 axis for every module imported afterwards
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42)

REGEN = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_regeneration")
IMG = os.path.join(REGEN, "images")
MK = os.path.join(REGEN, "masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme_raffine_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2")
assert os.path.isdir(MK), "final masks not found: " + MK
classes = ["State_%d" % i for i in range(1, 7)]
SIZE = 384; MASK_THR = 128
JEUX = {"uni_final": os.path.join(REGEN, "processed_composite_384_uni_final_v22")}
FRAC_SZ = 0.15; FRAC_Z1 = 0.30
CK_EY_V5 = os.path.join("checkpoints_regeneration", "eye_unet_cmp_fondu_poisson_v5_jeu3E", "eye_unet_unified_best.pth")
CK_BL_V5 = os.path.join("checkpoints_regeneration", "blasteme_unet_384_stratifie", "blasteme_unet_384_best.pth")   # ou blasteme_unet_384_v2
PEAK_THR_V5 = 0.61        # counting threshold of the v5 detector (peak positions), decision unchanged at 0.85
JOURNAL = "ancres_composite_v22.csv"

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

import ancrage_cascade as AC
_VERD = os.path.join("Claude outputs", "essai_ancrage_cascade_v3c2_v5s1neg_s070", "verdict.txt")
if not os.path.exists(_VERD) or not open(_VERD, encoding="utf-8").read().startswith("RETENUE"):
    raise SystemExit("cascade v3c2 at threshold 0.70 not validated (%s)" % _VERD)
print("trial test_ancrage_cascade_v3c2_v5s1neg_s070:", open(_VERD, encoding="utf-8").read().strip().replace("\n", " | "))
# Cascade used: threshold 0.70 (retained). CASCADE_V3C3=1 uses the local tip correction, only if
# test_ancrage_cascade_v3c3_v5s1neg_s070 RETAINED it (output suffixed _v3c3).
_VSK = os.path.join("Claude outputs", "essai_ancrage_cascade_skan2_v5s1neg_s070", "verdict.txt")
if not os.path.exists(_VSK) or not open(_VSK, encoding="utf-8").read().startswith("RETENUE"):
    raise SystemExit("skan 2 axis not validated (%s): use regenerate_composite_final_v12" % _VSK)
print("trial test_ancrage_cascade_skan2_v5s1neg_s070:", open(_VSK, encoding="utf-8").read().strip().replace("\n", " | "))
_V3 = os.path.join("Claude outputs", "essai_ancrage_cascade_v3c3_v5s1neg_s070", "verdict.txt")
_VAR = os.path.join("Claude outputs", "essai_ancrage_cascade_v3c14_s040", "verdict.txt")
if not os.path.exists(_VAR):
    raise SystemExit("run test_ancrage_cascade_v3c14_s040 first: " + _VAR)
print("check of the v3c14 cascade:", open(_VAR, encoding="utf-8").read().strip().replace("\n", " | "))
if not open(_VAR, encoding="utf-8").read().startswith("RETENUE"): raise SystemExit("cascade v3c14 not retained: " + _VAR)
import ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 as A3b
print("final cascade v3c14 (point at the border radius, threshold 0.40)")
for d in JEUX.values():
    for c in classes: os.makedirs(os.path.join(d, c), exist_ok=True)
print("cascade:", A3b.__name__, "| output:", JEUX["uni_final"])
import cadrage_squelette as CS
assert CS is _CSK and A3b.CS is _CSK, "the axis must come from cadrage_squelette_skan2"
SORTIE_V9 = os.path.join("Claude outputs", "composites_final_v22"); os.makedirs(SORTIE_V9, exist_ok=True)
for _c in (A3b.CK_BL3, A3b.CK_EY3):
    assert os.path.exists(_c), "checkpoint missing: " + _c
print("device:", AC.DEVICE, "| cuda :", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("graphics processor unavailable, run aborted.")
print("   carte :", torch.cuda.get_device_name(0))
A3b.charger(verbeux=True)
assert "blasteme_unet_384_v5_s1neg" in A3b.CK_BL3 and abs(A3b.DIL_BL - 0.30) < 1e-9, "cascade v3c2: unexpected segmenter or band"
assert "fondu_poisson_v5_jeu3E" in A3b.CK_EY3 and abs(A3b.PEAK_THR - 0.61) < 1e-9 and A3b.PEAK_ANCRAGE == 0.85 and A3b.YEUX_SUR_BLASTEME and A3b.GARDER_PLUS_GROS and abs(A3b.SEUIL_YEUX_TETE - 0.85) < 1e-9, "cascade v3c2: unexpected settings"
print("masks:", os.path.basename(MK), "| cascade v3c2: blastema band %.2f L, eyespots threshold %.2f decision %.2f" % (A3b.DIL_BL, A3b.PEAK_THR, A3b.PEAK_ANCRAGE))

# Thumbnails already written are skipped below. Any correction to the cascade therefore requires
# setting the old sets aside before running again. REPRISE = 1 keeps what exists and allows an
# interrupted pass to be resumed. REPRISE = 0, the default, renames both folders by appending the
# date -- nothing is erased -- and starts from an empty set.
REPRISE = int(os.environ.get("REPRISE", "0"))
if not REPRISE:
    _suf = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in JEUX.values():
        if os.path.isdir(d) and any(glob.glob(os.path.join(d, c, "*.png")) for c in classes):
            _old = d + "_avant_" + _suf
            os.rename(d, _old)
            print("previous set set aside:", os.path.basename(_old))
    for f in (JOURNAL,):
        if os.path.exists(f):
            os.rename(f, f.replace(".csv", "_avant_%s.csv" % _suf))
            print("previous log set aside:", f)
else:
    print("REPRISE active, thumbnails already written are kept")

# Worms whose annotation carries cyan on BOTH sides of the middle, hence undecidable. There the
# oracle falls back on the cascade rather than on an average of two opposite marks.
AMB = set(pd.read_csv("annotations_deux_cotes.csv", sep=";")["image"]) if os.path.exists("annotations_deux_cotes.csv") else set()
# Worms whose single cyan mark falls between 0.35 and 0.65 of the length, hence in the middle of
# the body. It designates no end, so there too the oracle falls back on the cascade.
if os.path.exists("annotations_mediane_a_ecarter.csv"):
    AMB |= set(pd.read_csv("annotations_mediane_a_ecarter.csv", sep=";")["image"])
print("   worms with an undecidable annotation, oracle = cascade:", len(AMB))

SOURCES = ["cyan_position_tete", "cyan_eye", "cyan_blasteme"]
_INDEX = {}
for src in SOURCES:
    for f in glob.glob(os.path.join(REGEN, src, "*", "*")):
        _INDEX.setdefault(os.path.splitext(os.path.basename(f))[0], []).append((SOURCES.index(src), f))
for k in _INDEX: _INDEX[k].sort()
print("worms carrying at least one annotation:", len(_INDEX))


def masque_cyan(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    m = ((b > 120) & (g > 120) & (r < 110) & ((b - r) > 60) & ((g - r) > 40)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def ancre_experte(stem, M, H, W, ymid=None, span=None):
    """Barycentre of the cyan blobs of the most direct source, in the straightened frame. A cyan_blasteme annotation
    with blobs at both ends (beyond 0.25 of the length on either side of the middle) is skipped."""
    for isrc, f in _INDEX.get(stem, []):
        an = cv2.imread(f)
        if an is None: continue
        rc = cv2.warpAffine(masque_cyan(an) * 255, M, (W, H), flags=cv2.INTER_NEAREST) > 127
        n, lab, st, cen = cv2.connectedComponentsWithStats(rc.astype(np.uint8), 8)
        bl = [(float(st[i, cv2.CC_STAT_AREA]), float(cen[i][0]), float(cen[i][1])) for i in range(1, n)
              if st[i, cv2.CC_STAT_AREA] >= 3]
        if not bl: continue
        if SOURCES[isrc] == "cyan_blasteme" and ymid is not None and span:
            if any(q[2] < ymid - 0.25 * span for q in bl) and any(q[2] > ymid + 0.25 * span for q in bl):
                continue   # blastema annotated at both ends: does not designate the head
        a = sum(q[0] for q in bl)
        return sum(q[0] * q[1] for q in bl) / a, sum(q[0] * q[2] for q in bl) / a, os.path.basename(os.path.dirname(os.path.dirname(f)))
    return None

def worm_mask(m):
    m = (m >= MASK_THR).astype(np.uint8)
    n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return m if n <= 1 else (l == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def orient(bgr, mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ang = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
    H, W = mask.shape
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang - 90, 1.0)
    rb = cv2.warpAffine(bgr, M, (W, H))
    rm = cv2.warpAffine((mask * 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 127
    return rb, rm, M


def _sq(c):
    s = max(c.shape[:2]); sq = np.zeros((s, s), c.dtype)
    sq[(s - c.shape[0]) // 2:(s - c.shape[0]) // 2 + c.shape[0],
       (s - c.shape[1]) // 2:(s - c.shape[1]) // 2 + c.shape[1]] = c
    return cv2.resize(sq, (SIZE, SIZE))


def whole_channel(rg, rm):
    g = rg.copy(); g[~rm] = 0
    ys, xs = np.nonzero(rm)
    return _sq(g[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def _boite(shape, ax, ay, h):
    H, W = shape
    return (max(0, int(ay - h)), min(H, int(ay + h)), max(0, int(ax - h)), min(W, int(ax + h)))


def composite_v11(bgr, wm, r3, c0):
    """Channels 1 and 2 cut in the frame of the chain thumbnails: centre at 12 % of L from the retained tip,
    tip at the top. Channel 2: super-zoom of side 0.30 L, CLAHE, background outside the worm mask set to 0."""
    out = [c0]
    for frac, cl in ((FRAC_Z1, False), (FRAC_SZ, True)):
        M = CS.affine(r3["tip"], r3["centre"], r3["L"], frac=frac)
        v = CS.vignette(bgr, wm.astype(np.uint8), M)
        foot = cv2.warpAffine(wm.astype(np.uint8) * 255, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 127
        g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        if cl: g = _clahe.apply(g)
        g[~foot] = 0; out.append(g)
    return cv2.merge(out)


def tete_en_haut(rg, rm, M, r3):
    """Rotate the straightened worm by 180 degrees if the retained tip falls in its lower half."""
    ys, _ = np.nonzero(rm); ymid = 0.5 * (ys.min() + ys.max())
    py = (M @ np.array([r3["tip"][1], r3["tip"][0], 1.0]))[1]
    if py > ymid: return rg[::-1, ::-1].copy(), rm[::-1, ::-1].copy(), True
    return rg, rm, False

NB = int(os.environ.get("NB", "100000"))
etapes = Counter(); srcs = Counter(); journal = []; t0 = time.time(); n = 0

for cls in classes:
    for f in sorted(glob.glob(os.path.join(IMG, cls, "*.jpg"))):
        nm = os.path.basename(f); stem = os.path.splitext(nm)[0]; png = stem + ".png"
        if all(os.path.exists(os.path.join(d, cls, png)) for d in JEUX.values()) or n >= NB:
            continue
        b = cv2.imread(f); m = cv2.imread(os.path.join(MK, cls, nm), 0)
        if b is None or m is None:
            for d in JEUX.values(): cv2.imwrite(os.path.join(d, cls, png), np.zeros((SIZE, SIZE, 3), np.uint8))
            continue
        mask = worm_mask(m)
        if mask.sum() < 300:
            for d in JEUX.values(): cv2.imwrite(os.path.join(d, cls, png), np.zeros((SIZE, SIZE, 3), np.uint8))
            continue
        rbgr, rm, M = orient(b, mask)
        rg = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY)
        ys, _ = np.nonzero(rm); span = int(ys.max() - ys.min() + 1); ymid = 0.5 * (ys.min() + ys.max())
        r3 = A3b.ancrer(b, m)   # cascade v3c2 in the original image
        if r3 is None:
            for d in JEUX.values(): cv2.imwrite(os.path.join(d, cls, png), np.zeros((SIZE, SIZE, 3), np.uint8))
            continue
        stage, conf = r3["stage"], r3["confidence"]
        etapes[stage] += 1
        rg, rm, tourne = tete_en_haut(rg, rm, M, r3)
        c0 = whole_channel(rg, rm)
        wm = CS.masque_ver(m)
        cv2.imwrite(os.path.join(JEUX["uni_final"], cls, png), composite_v11(b, wm, r3, c0))
        Msz = CS.affine(r3["tip"], r3["centre"], r3["L"], frac=FRAC_SZ); hx, hy = (Msz @ np.array([r3["x"], r3["y"], 1.0])).tolist()
        # share of the expert-annotated blastema (cyan_blasteme) contained in the super-zoom, for the text of part 2.2.1
        part_bl = np.nan
        for isrc, fa in _INDEX.get(stem, []):
            if SOURCES[isrc] != "cyan_blasteme": continue
            an = cv2.imread(fa)
            if an is None or an.shape[:2] != b.shape[:2]: continue
            cy = (masque_cyan(an) > 0) & ~(masque_cyan(b) > 0)
            if cy.sum() >= 30:
                dedans = cv2.warpAffine(cy.astype(np.uint8) * 255, Msz, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 127
                # ratio of the areas brought to the same scale (the thumbnail magnifies by a constant factor)
                ech = (SIZE / (2.0 * FRAC_SZ * r3["L"])) ** 2
                part_bl = float(min(1.0, dedans.sum() / (cy.sum() * ech)))
            break
        journal.append(dict(image=nm, classe=cls, stage=stage, confidence=round(float(conf), 4), tourne_180=int(tourne),
                            point_tete_dans_superzoom=int(0 <= hx < SIZE and 0 <= hy < SIZE), part_blasteme_superzoom=part_bl,
                            tip_y=round(float(r3["tip"][0]), 1), tip_x=round(float(r3["tip"][1]), 1),
                            centre_y=round(float(r3["centre"][0]), 1), centre_x=round(float(r3["centre"][1]), 1), L=round(float(r3["L"]), 1)))
        n += 1
        if n % 300 == 0: print("  %d worms, %.0f s" % (n, time.time() - t0), flush=True)

J = pd.DataFrame(journal)
if len(J):
    J.to_csv(JOURNAL, sep=";", index=False, mode="a",
             header=not os.path.exists(JOURNAL))
print("\n%d worms processed in %.0f s" % (n, time.time() - t0))
print("deciding stage:", dict(etapes))
print("COMPLETE" if n == 0 else "run this cell again if you interrupted it")
for k, d in JEUX.items():
    print("   %-11s : %d images" % (k, sum(len(glob.glob(os.path.join(d, c, "*.png"))) for c in classes)))

if os.path.exists(JOURNAL):
    J = pd.read_csv(JOURNAL, sep=";").drop_duplicates("image")
    print("worms: %d | whole worm rotated 180 degrees to put the head up: %d" % (len(J), int(J.tourne_180.sum())))
    print(J.groupby("classe").stage.value_counts().unstack(fill_value=0).to_string())
    print("cascade head point (eyespot barycentre or tip) contained in the super-zoom: %d / %d" % (int(J.point_tete_dans_superzoom.sum()), len(J)))
    B_ = J[J.part_blasteme_superzoom.notna()]
    print("\nshare of the annotated blastema contained in the super-zoom, per state (annotated images):")
    for cls, g in B_.groupby("classe"):
        p = g.part_blasteme_superzoom
        print("   %s n %4d | median %.3f | >= 0.99: %.3f | >= 0.90: %.3f" % (cls, len(p), p.median(), (p >= 0.99).mean(), (p >= 0.90).mean()))
