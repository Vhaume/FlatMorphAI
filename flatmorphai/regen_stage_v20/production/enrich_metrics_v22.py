# CHAIN v22 (22 October 2026): copy of enrich_metrics_v21.py (unmodified). Cascade ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 (start point:
# largest border radius among the points in contact with the blastema), blastema threshold 0.40, s040 masks unchanged.
# CHAIN v21 (20 October 2026): copy of enrich_metrics_v20.py (unmodified). Complete descriptors, without NaN or imputation: eyeun_* at 0
# when there is no eyespot peak, every image of the computed lists, stops if a descriptor or a decimal state is missing.
# CHAIN v20 (20 October 2026): copy of enrich_metrics_v19.py (unmodified). Blastema threshold 0.40 (calibrer_seuil_objet_blasteme_20261020_023318,
# maximal IoU of the blastema object) and cascade ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 (v3c12 at threshold 0.40).
# -*- coding: utf-8 -*-
# Version v19 (19 October 2026): copy of enrich_metrics_v18 (unmodified). Inputs from ancrage_cascade_v3c8_cascade_v5s1neg_s070: composites
# processed_composite_384_uni_final_v18 and log ancres_composite_v22.csv. Outputs suffixed v19. The blastema descriptors are measured on
# the blastema object retained by the v3c8 rule (function objet_blasteme).
# Version v19 (18 October 2026): copy of enrich_metrics_v17 (unmodified). Inputs from ancrage_cascade_v3c8_cascade_v5s1neg_s070:
# composites processed_composite_384_uni_final_v18 and log ancres_composite_v22.csv. Outputs suffixed v19.
# Version v19 (18 October 2026): copy of enrich_metrics_v16 (unmodified). Only the inputs change, coming from the final cascade
# method (ancrage_cascade_v3c8_cascade_v5s1neg_s070): composites processed_composite_384_uni_final_v18 and log
# ancres_composite_v22.csv, and the outputs, suffixed v19. Nothing is overwritten.
# [V16] Recomputation of ALL the state classifier descriptors on the current pipeline (18 October 2026).
#
# The 128 descriptors were produced by a series of programs (precompute_regeneration_metrics_v2,
# eye_inference_and_features, generate_composite_and_features, enrich_metrics_blasteme, enrich_metrics_blasteme_forme,
# enrich_metrics_pupille, maj_couleur_locale, enrich_metrics_yeux_unet_v5), each with its own mask, its own detector and its
# own way of finding the head (principal axis, palest end or strongest response). This program recomputes
# them ALL on the same inputs, those of the retained chain:
#   - worm mask: Mask2Former mask refined by the blastema (..._ouv3_skan2);
#   - head: tip and thumbnail centre retained by the chain (cascade, skan and Dijkstra axis), read from the
#     log ancres_composite_v22.csv written by regenerate_composite_final_v22;
#   - blastema: retained segmenter (checkpoints_regeneration/blasteme_unet_384_v5_s1neg), thumbnail of side 0.90 L,
#     band of 0.15 L on each side, threshold 0.70 (the chain's);
#   - eyespots: retained detector (eye_unet_cmp_fondu_poisson_v5_jeu3E), thumbnail of side 0.60 L, counting threshold 0.61,
#     peaks at least 7 pixels apart.
# The descriptor FORMULAS do not change: the computation functions are read from the original programs and
# executed as they are (exec of their source); only their inputs are replaced. The only textual changes,
# checked by assert, are the blastema threshold (0.5 -> 0.70) in forme_feats and couleur_locale, and the head end
# imposed by the chain in enrich_metrics_blasteme (instead of the palest end).
# eyeun_conf2, with no known producing program and absent from the 97, is neither recomputed nor copied.
# blast_seg_prop_yr is renormalised per year on the training table alone (no leakage).
# The descriptor selection (mutual information >= 0.02 then single-link grouping at |r| >= 0.90, the most
# informative kept) is redone on the recomputed training table: metriques_finales_v22.csv.
# Output: metrics_regeneration/dataset_dedup_State_50valid_30test_v22 (new folder, nothing is overwritten).
# Graphics processor required, seed 42, separator ";". About 30 minutes.
import os, sys, re, glob, time, shutil, datetime, numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
import cadrage_squelette_skan2 as _CSK
sys.modules["cadrage_squelette"] = _CSK          # skan and Dijkstra axis for every module imported afterwards
import ancrage_cascade as AC
import ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 as A3b   # [V20] cascade v3c14 (fallback if an image is missing from the log)
import cadrage_squelette as CS
assert CS is _CSK and A3b.CS is _CSK, "the axis must come from cadrage_squelette_skan2"
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42)

REGEN = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_regeneration")
DN = "dataset_dedup_State_50valid_30test"
IMG = os.path.join(REGEN, "images")
MK = os.environ.get("MK_V16", os.path.join(REGEN, "masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme_raffine_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2"))
SRC_MET = os.path.join(REGEN, "metrics_regeneration", DN + "_v5")      # list of images and column order, read only
OUT_MET = os.path.join(REGEN, "metrics_regeneration", DN + "_v22")
JOURNAL_CHAINE = "ancres_composite_v22.csv"
SELECTION = "metriques_finales_v22.csv"
SEUIL_BL = 0.40; F_BL = 0.45; F_EYE = 0.30; DIL_BL = 0.30; THR_EYE = 0.61; MD_EYE = 7
NB = int(os.environ.get("NB", "100000"))         # NB=20 for a quick trial
for p in (IMG, MK, SRC_MET, JOURNAL_CHAINE): assert os.path.exists(p), "not found: " + p
assert not os.path.exists(OUT_MET) or NB < 100000 or os.environ.get("FORCER_V16") == "1", "folder already present: " + OUT_MET
if NB < 100000: OUT_MET = OUT_MET + "_essai"
os.makedirs(OUT_MET, exist_ok=True)

AC.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("CPU_OK") != "1":
    assert AC.DEVICE.type == "cuda", "graphics processor not found (CUDA/ROCm)"
mbl, mey = A3b.charger(verbeux=True)
assert next(mbl.parameters()).device.type == AC.DEVICE.type and next(mey.parameters()).device.type == AC.DEVICE.type
assert "v5_s1neg" in A3b.CK_BL3 and "fondu_poisson_v5_jeu3E" in A3b.CK_EY3
print("device:", AC.DEVICE, "| masks:", os.path.basename(MK))
print("output:", os.path.abspath(OUT_MET))

# ---- computation functions of the original programs, executed as-is ----
def _src(f): return open(f, encoding="utf-8").read()
def _tranche(s, debut, fin):
    i = s.index(debut); j = s.index(fin, i); return s[i:j]
def _remplace(s, a, b, n=1):
    assert s.count(a) == n, ("pattern expected %d times" % n, a); return s.replace(a, b)

# (1) precompute_regeneration_metrics_v2: shape, pigment, ends, texture, dark blobs, cap, axis, CLAHE (80)
S_PV = _src("precompute_regeneration_metrics_v2.py")
PV = {}
exec(_tranche(S_PV, "import numpy as np, cv2\nfrom skimage.feature", "import os, glob, numpy as np, pandas as pd, cv2\nnp.random.seed(42)"), PV)
EYE_PARAMS = {"k_frac": 0.06, "n_sigma": 2.2, "halo_k": 0.3, "dark_k": 0.4}      # production values (L566)
assert "EYE_PARAMS = " + repr(EYE_PARAMS).replace("'", '"') in S_PV

# (2) eye_inference_and_features: pupil rings, pale cap, tissue around the eyespots (8)
EIF = {"np": np}
exec(_tranche(_src("eye_inference_and_features.py"), "def eye_feats_at", "FCOLS="), EIF)

# (3) generate_composite_and_features: head profile (13)
GCF = {"np": np, "cv2": cv2}
_g = _src("generate_composite_and_features.py")
exec(_tranche(_g, "def orient(bgr,mask):", "def crop256"), GCF)
exec(_tranche(_g, "PCOLS=", "\n\n# In[ ]:\n\n# ---- composite + profile generation"), GCF)

# (4) enrich_metrics_blasteme: pale cap without a network (4), head end imposed by the chain
EMB = {"np": np, "cv2": cv2}
_b = _tranche(_src("enrich_metrics_blasteme.py"), "MASK_THR=128; T_PALE", "IMGIDX={}")
_b = _remplace(_b, "for top in (True,False):", "for top in (TETE_HAUT,):")
exec(_b, EMB)

# (5) enrich_metrics_blasteme_forme: dark border and shape of the segmented blastema (9), chain threshold
EMF = {"np": np, "cv2": cv2, "SEUIL_BL": SEUIL_BL}
_f = _tranche(_src("enrich_metrics_blasteme_forme.py"), "def darkrim", "def seg_feats")
_f = _remplace(_f, "m=(p>=0.5).astype(np.uint8)", "m=(p>=SEUIL_BL).astype(np.uint8)")
exec(_f, EMF)

# (6) couleur_locale_blasteme: blastema saturation relative to the body (2 retained), chain threshold
CL = {"np": np, "cv2": cv2, "SEUIL_BL": SEUIL_BL}
_c = _tranche(_src("couleur_locale_blasteme.py"), "COLS = [\"loc_sat_blast\"", "def mesure(")
_c = _remplace(_c, "B = (p >= 0.5) & tissu", "B = (p >= SEUIL_BL) & tissu")
exec(_c, CL)
NEUTRE_LOC = {"loc_sat_ecart": 0.0, "loc_sat_ratio": 1.0}      # neutral values of maj_couleur_locale (empty blastema)

# (7) enrich_metrics_yeux_unet_v5: detector scores at the peaks (7)
EYV = {"np": np, "cv2": cv2, "torch": torch, "F": F}
_e = _src("enrich_metrics_yeux_unet_v5.py")
assert "PEAK_THR=0.61" in _e and "MD=7" in _e and "ECH=0.35/0.30" in _e
EYV.update(S=384, FRAC=0.30, PEAK_THR=THR_EYE, MD=MD_EYE, ECH=0.35 / 0.30)
_e = _remplace(_e, "eyeun_scl_pup=np.nan,eyeun_pupil=np.nan,", "eyeun_scl_pup=0.0,eyeun_pupil=0.0,")      # [V21] no peak: zero contrasts
_e = _remplace(_e, "eyeun_tis_scl=np.nan,eyeun_grad_pig=np.nan,eyeun_white_scl=np.nan", "eyeun_tis_scl=0.0,eyeun_grad_pig=0.0,eyeun_white_scl=0.0")
exec(_tranche(_e, "def peaks(hm,thr):", "# index of image and mask files"), EYV)

# (8) enrich_metrics_pupille: eyespot maturation, radial profile (7)
PUP = {"np": np, "cv2": cv2, "torch": torch, "F": F}
_p = _src("enrich_metrics_pupille.py")
exec(_tranche(_p, "S=384; FRAC=0.30; PEAK_THR=0.30; MASK_THR=128", "def worm_mask"), PUP)
exec(_tranche(_p, "def peaks(hm,thr):", "IMGIDX={}"), PUP)
assert PUP["MD"] == MD_EYE and abs(EYV["ECH"] - PUP["ECH"]) < 1e-9

COLS_PV = ["shp_area_frac", "shp_solidity", "shp_circularity", "shp_extent", "shp_aspect_bbox", "shp_perim_norm", "shp_elong",
           "shp_eccentricity", "shp_hu1", "shp_hu2", "shp_hu3"]
print("descriptor functions loaded: precompute, eye_feats_at, head_features, coiffe, forme_feats, couleur_locale, eyeun, pupille")

# ---- chain inputs ----
J = pd.read_csv(JOURNAL_CHAINE, sep=";").drop_duplicates("image", keep="last").set_index("image")
IMGIDX, MKIDX = {}, {}
for s in range(1, 7):
    for p in glob.glob(os.path.join(IMG, "State_%d" % s, "*.jpg")): IMGIDX.setdefault(os.path.basename(p), p)
    for p in glob.glob(os.path.join(MK, "State_%d" % s, "*.jpg")): MKIDX.setdefault(os.path.basename(p), p)
print("images %d | masks %d | chain log %d" % (len(IMGIDX), len(MKIDX), len(J)))

def pics(pr, thr, k):
    """Local maxima of the map (max_pool2d of kernel 2 MD + 1), sorted by score, first k: (x, y, score)."""
    t = torch.from_numpy(pr)[None, None]; mx = F.max_pool2d(t, 2 * MD_EYE + 1, 1, MD_EYE)[0, 0].numpy()
    ys, xs = np.nonzero((pr == mx) & (pr >= thr))
    return sorted(zip(xs.tolist(), ys.tolist(), pr[ys, xs].tolist()), key=lambda z: -z[2])[:k]

def vers_image(M, x, y):
    Mi = cv2.invertAffineTransform(M); return Mi @ np.array([x, y, 1.0])

def chaine(base, bgr, wm):
    """Tip (y, x), centre (y, x) and length L retained by the chain. Fallback: cascade re-run on the refined mask."""
    if base in J.index and np.isfinite(J.loc[base, "tip_y"]):
        r = J.loc[base]; return (float(r.tip_y), float(r.tip_x)), (float(r.centre_y), float(r.centre_x)), float(r.L), "journal"
    r3 = A3b.ancrer(bgr, (wm * 255).astype(np.uint8))
    ends, L, _ = CS.extremites(wm)
    return tuple(map(float, r3["tip"])), tuple(map(float, r3["centre"])), float(L), "cascade"

def masque_384(bgr, wm):
    """384 x 384 masked image of the base metrics: worm box, long side at 384 (INTER_AREA), centred square, black background."""
    ys, xs = np.nonzero(wm); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = (bgr * wm[..., None].astype(bgr.dtype))[y0:y1, x0:x1]; h, w = c.shape[:2]; s = 384.0 / max(h, w)
    r = cv2.resize(c, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    o = np.zeros((384, 384, 3), np.uint8); oy = (384 - r.shape[0]) // 2; ox = (384 - r.shape[1]) // 2
    o[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
    return o, (x0, y0, s, ox, oy)

def tete_haute_384(img384, tr, tip):
    """head_high of compute_metrics: true if the chain tip falls on the t-near-1 side of the principal axis."""
    x0, y0, s, ox, oy = tr; px = (tip[1] - x0) * s + ox; py = (tip[0] - y0) * s + oy
    m = PV["_largest_mask"](cv2.cvtColor(img384, cv2.COLOR_BGR2GRAY))
    ys, xs = np.nonzero(m); pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False); t = (pts - c) @ vt[0]
    tp = (np.array([px, py]) - c) @ vt[0]
    return bool((tp - t.min()) / (np.ptp(t) + 1e-9) > 0.5)

def haut_dans_repere(M, wm, tip):
    """True if the tip falls in the upper half of the worm straightened by M (same test as tete_en_haut of the chain)."""
    H, W = wm.shape; rm = cv2.warpAffine(wm.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_NEAREST) > 127
    ys = np.nonzero(rm)[0]; ymid = 0.5 * (ys.min() + ys.max())
    return bool((M @ np.array([tip[1], tip[0], 1.0]))[1] < ymid)

def descripteurs(base):
    ip, mp = IMGIDX.get(base), MKIDX.get(base)
    if ip is None or mp is None: return None, "fichier absent"
    bgr = cv2.imread(ip); mraw = cv2.imread(mp, 0)
    if bgr is None or mraw is None: return None, "lecture impossible"
    wm = CS.masque_ver(mraw).astype(np.uint8)          # threshold 128, largest component
    if wm.sum() < 300: return None, "mask too small"
    tip, centre, L, source = chaine(base, bgr, wm)
    d = {}
    # (1) base metrics on the 384 masked image, head from the chain
    img384, tr = masque_384(bgr, wm)
    m = PV["compute_metrics"](img384, eye_params=EYE_PARAMS, head_high=tete_haute_384(img384, tr, tip))
    if m is None: return None, "compute_metrics vide"
    d.update(m)
    # eyespot thumbnail of the chain (side 0.60 L, refined mask without dilation) and retained detector
    Me = CS.affine(tip, centre, L, frac=F_EYE); ve = CS.vignette(bgr, wm, Me); pe = AC._pred(mey, ve)
    rgb_e = cv2.cvtColor(ve, cv2.COLOR_BGR2RGB)
    # (7) eyeun_* and (8) pup_*: the chain thumbnail replaces best_crop
    EYV["best_crop"] = lambda b, mm: (float(pe.max()), rgb_e, pe)
    PUP["best_crop"] = lambda b, mm: (float(pe.max()), rgb_e, pe)
    r = EYV["features"](bgr, mraw); d.update(r if r is not None else {})
    d.update(PUP["feats"](bgr, mraw))
    # (2) eye rings at most 3 peaks, brought back into the original image
    g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); ft = []
    for (x, y, _) in pics(pe, THR_EYE, 3):
        X, Y = vers_image(Me, x, y); f = EIF["eye_feats_at"](g0, int(round(X)), int(round(Y)))
        if f: ft.append(f)
    vals = [float(len(ft))] + ([float(v) for v in np.mean(ft, 0)] if ft else [0.0] * 7)
    if not ft: vals = [0.0] * 8
    d.update(dict(zip(["eye_n", "eye_pupil", "eye_sclera", "eye_tissue", "eye_sclera_tissue", "eye_sclera_pupil", "eye_tissue_pupil", "eye_white_frac"], vals)))
    # (3) head profile: worm straightened by its principal axis, head and eyespots from the chain
    rbgr, rm = GCF["orient"](bgr, wm); rg = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY); Mr = EMB["orient"](wm)
    reyes = []
    for (x, y, _) in pics(pe, THR_EYE, 2):
        X, Y = vers_image(Me, x, y); q = Mr @ np.array([X, Y, 1.0]); reyes.append((float(q[0]), float(q[1])))
    d.update(dict(zip(GCF["PCOLS"], GCF["head_features"](rg, rm, haut_dans_repere(Mr, wm, tip), reyes))))
    # (4) pale cap without a network, head end from the chain
    EMB["TETE_HAUT"] = haut_dans_repere(Mr, wm, tip)
    d.update(EMB["features"](bgr, wm * 255))
    # (5) and (6) segmented blastema: chain thumbnail (side 0.90 L, band of 0.15 L) and retained segmenter
    k = max(1, int(DIL_BL * L)); wd = cv2.dilate(wm, np.ones((k, k), np.uint8))
    Mb = CS.affine(tip, centre, L, frac=F_BL); vb = CS.vignette(bgr, wd, Mb); pb = AC._pred(mbl, vb)
    pb = A3b.objet_blasteme(pb, Mb, wm, tip)   # [V19] same blastema object as the v3c8 cascade (the admitted one closest to the tip)
    soft = float(pb.sum()) * (2.0 * F_BL * L / 384.0) ** 2
    d["blast_seg_prop"] = soft / float(wm.sum())
    d["blast_seg_rim"] = EMF["darkrim"](vb, (pb >= SEUIL_BL))
    d.update(EMF["forme_feats"](vb, pb))
    loc = CL["couleur_locale"](vb, pb)
    for c, v in NEUTRE_LOC.items(): d[c] = loc[c] if np.isfinite(loc[c]) else v
    d["_source_tete"] = source
    return d, "ok"

# ---- computation over the three tables, in the original row order ----
SRC = {sp: pd.read_csv(os.path.join(SRC_MET, "%s_%s_regeneration_metrics_eyespig.csv" % (DN, sp)), sep=";") for sp in ("train", "valid", "test")}
COLS_ORDRE = [c for c in SRC["train"].columns if c not in ("image", "class", "eyeun_conf2")]
# [V21] rows = images of the training, validation and test lists (the v5 tables did not have one each)
LISTES = os.path.join(REGEN, "dataset_images_list", DN)
SRC_L = {}
for _sp in ("train", "valid", "test"):
    _l = pd.read_csv(os.path.join(LISTES, "%s_%s.csv" % (DN, _sp)), sep=";")
    SRC_L[_sp] = pd.DataFrame({"image": _l["image_path"].astype(str).map(os.path.basename), "class": _l["State"].astype(str)})
    _abs = sorted(set(SRC_L[_sp].image) - set(SRC[_sp].image.astype(str).map(os.path.basename)))
    print("list %s: %d images, missing from the v5 tables: %s" % (_sp, len(SRC_L[_sp]), _abs))
# [V21] decimal state of each image of the lists, otherwise stop (no fallback state)
for sp, d in SRC_L.items():
    _dc = os.path.join(SRC_MET, "decimal_%s.csv" % sp); _dd = pd.read_csv(_dc, sep=";") if os.path.exists(_dc) else pd.DataFrame(columns=["image"])
    _sans = sorted(set(d.image) - set(_dd.image.astype(str)))
    _cp = "decimal_complements_v22.csv"
    if _sans and os.path.exists(_cp): _sans = sorted(set(_sans) - set(pd.read_csv(_cp, sep=";").image.astype(str)))
    assert not _sans, ("annotated decimal state missing, to be added to decimal_complements_v22.csv (image;class;stade_entier;stade_decimal)", sp, _sans)
print("descriptors to recompute:", len(COLS_ORDRE))
CACHE = {}; etats = {}; t0 = time.time(); n = 0
TABS = {}
for sp, d0 in SRC_L.items():
    rows = []
    for i, (im, cl) in enumerate(zip(d0["image"].astype(str), d0["class"])):
        if n >= NB: break
        base = os.path.basename(im)
        if base not in CACHE:
            CACHE[base] = descripteurs(base); n += 1
            if n % 100 == 0: print("  %d images | %.0f s" % (n, time.time() - t0))
        r, etat = CACHE[base]; etats[etat] = etats.get(etat, 0) + 1
        row = {"image": im, "class": cl}
        row.update({c: (r.get(c, np.nan) if r else np.nan) for c in COLS_ORDRE})
        row["_source_tete"] = r.get("_source_tete", "") if r else ""
        rows.append(row)
    TABS[sp] = pd.DataFrame(rows)
print("states:", etats, "| %.0f s" % (time.time() - t0))
manquants = set(COLS_ORDRE) - {"blast_seg_prop_yr"} - set(k for r, _ in CACHE.values() if r for k in r)
assert not manquants, ("descriptors not produced", sorted(manquants))
for sp in TABS: print(sp, "head read from the log:", int((TABS[sp]["_source_tete"] == "journal").sum()), "/", len(TABS[sp]))

# ---- blast_seg_prop_yr: normalisation per acquisition year, statistics from the training table alone ----
def annee(b):
    mo = re.match(r"(\d{4})-", os.path.basename(str(b))); return mo.group(1) if mo else "autre"
tr = TABS["train"].assign(_a=TABS["train"]["image"].map(annee)); g = tr.groupby("_a")["blast_seg_prop"]
mu = g.mean().to_dict(); sd = g.std(ddof=0).to_dict()
gm = float(tr["blast_seg_prop"].mean()); gs = float(tr["blast_seg_prop"].std(ddof=0)) or 1.0
for sp, d in TABS.items():
    a = d["image"].map(annee)
    m_ = a.map(lambda k: mu.get(k, gm)); s_ = a.map(lambda k: sd.get(k, gs) if sd.get(k, gs) and sd.get(k, gs) > 1e-9 else gs)
    d["blast_seg_prop_yr"] = (d["blast_seg_prop"] - m_) / s_
print("mean blast_seg_prop per year (training):", {k: round(v, 4) for k, v in mu.items()})
# [V21] check: no missing or infinite value in the 128 descriptors, otherwise stop
for sp, d in TABS.items():
    _x = d[COLS_ORDRE].apply(pd.to_numeric, errors="coerce"); _n = _x.isna().sum(); _n = _n[_n > 0]
    assert _n.empty and np.isfinite(_x.values).all(), ("missing descriptors in %s" % sp, _n.to_dict())
print("check: no missing or infinite value in the three tables")

# ---- writing: same file names as the previous folders, columns in the original order ----
for sp, d in TABS.items():
    d[["image", "class"] + COLS_ORDRE].to_csv(os.path.join(OUT_MET, "%s_%s_regeneration_metrics_eyespig.csv" % (DN, sp)), sep=";", index=False)
    _dc = os.path.join(SRC_MET, "decimal_%s.csv" % sp)
    _dd = pd.read_csv(_dc, sep=";")
    if os.path.exists("decimal_complements_v22.csv"):
        _cmp = pd.read_csv("decimal_complements_v22.csv", sep=";"); _cmp = _cmp[_cmp.image.isin(d.image) & ~_cmp.image.isin(_dd.image)]
        _dd = pd.concat([_dd, _cmp[_dd.columns]], ignore_index=True)
    _dd[_dd.image.isin(d.image)].to_csv(os.path.join(OUT_MET, "decimal_%s.csv" % sp), sep=";", index=False)
pd.concat([d.assign(split=sp) for sp, d in TABS.items()])[["image", "split", "_source_tete"]].to_csv(os.path.join(OUT_MET, "journal_enrich_v22.csv"), sep=";", index=False)
print("written:", OUT_MET)

# ---- check: agreement with the previous descriptors (Spearman correlation over the same images) ----
from scipy.stats import spearmanr
anc = pd.concat(SRC.values()).drop_duplicates("image").set_index("image"); neu = pd.concat(TABS.values()).drop_duplicates("image").set_index("image")
com = anc.index.intersection(neu.index); ctrl = []
for c in COLS_ORDRE:
    a, b = anc.loc[com, c].astype(float), neu.loc[com, c].astype(float); ok = a.notna() & b.notna()
    rho = spearmanr(a[ok], b[ok]).correlation if ok.sum() > 10 and a[ok].nunique() > 1 and b[ok].nunique() > 1 else np.nan
    ctrl.append((c, c.split("_")[0], rho, int(b.isna().sum())))
C = pd.DataFrame(ctrl, columns=["descripteur", "famille", "spearman_ancien_nouveau", "nan_nouveau"])
C.to_csv(os.path.join(OUT_MET, "controle_ancien_nouveau_v22.csv"), sep=";", index=False)
print(C.groupby("famille")["spearman_ancien_nouveau"].median().round(3).to_string())

# ---- descriptor selection on the recomputed training table (same rule as metriques_finales_IM002_r090) ----
# Relevance: mutual information with the 6 states >= 0.02 (k-nearest-neighbour estimator, Ross 2014, DOI
# 10.1371/journal.pone.0087357, k = 3, seed 42). Redundancy: single-linkage clustering (Sibson 1973, DOI
# 10.1093/comjnl/16.1.30); two descriptors are linked as soon as |Pearson r| >= 0.90, the most informative of each group kept.
from sklearn.feature_selection import mutual_info_classif
from scipy.sparse.csgraph import connected_components
X = TABS["train"][COLS_ORDRE].astype(float); assert not X.isna().any().any(); y = TABS["train"]["class"].astype(str).values
cols = [c for c in COLS_ORDRE if X[c].nunique() > 1]
im_ = pd.Series(mutual_info_classif(X[cols].values, y, n_neighbors=3, random_state=42), index=cols)
pert = [c for c in cols if im_[c] >= 0.02]
R = np.abs(np.corrcoef(X[pert].values, rowvar=False)); A_ = (R >= 0.90).astype(int)
ng, lab = connected_components(A_, directed=False)
garde = [max([pert[i] for i in np.flatnonzero(lab == k)], key=lambda c: im_[c]) for k in range(ng)]
garde = sorted(garde, key=lambda c: -im_[c])
pd.DataFrame({"descripteur": garde, "IM_6stades": [round(float(im_[c]), 4) for c in garde]}).to_csv(
    SELECTION if NB >= 100000 else SELECTION.replace(".csv", "_essai.csv"), sep=";", index=False)
anc_sel = pd.read_csv("metriques_finales_IM002_r090.csv", sep=";")["descripteur"].tolist()
print("retained descriptors: %d (relevant %d, grouped %d) | shared with the former list of %d: %d"
      % (len(garde), len(pert), ng, len(anc_sel), len(set(garde) & set(anc_sel))))
print("list written:", SELECTION)
