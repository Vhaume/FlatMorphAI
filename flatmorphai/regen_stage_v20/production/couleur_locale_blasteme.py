# -*- coding: utf-8 -*-
"""# Local blastema colour, referenced against the worm itself

Porting the stereotype metrics revealed a colour signal on the State 5 against State 6 boundary:
st_col_sat_std at 0.629 ROC area, the leading family for that boundary, but far from the best known
variable at 0.726. The eight ported colour variables are global, averaged over the whole worm. The
blastema occupies roughly one hundredth of the worm area, so a global average dilutes by a factor of ten
to a hundred the contrast that separates an unpigmented blastema from a repigmented one. This notebook
measures the colour where it carries the information -- inside the blastema mask predicted by the
segmenter -- and refers it to the pigmented body of the same image, which cancels the white balance and
the lighting specific to each shot without any per-year statistic.

Geometry strictly identical to the segmenter and to enrich_metrics_blasteme_seg_complet: orientation by
principal component analysis, zoom on both ends, strongest-response end retained, four-channel input.
Three regions are defined in the thumbnail: the blastema B where the probability exceeds 0.5, the front F
obtained by dilating B and subtracting B, and the reference body R made of tissue twelve dilations away
from B. The descriptors are written to Claude outputs; no metric table is modified.

The chrominance of a pixel is defined by equation 1, in the CIELAB space coded on eight bits where the
achromatic point is 128.

chroma = sqrt((a - 128) squared + (b - 128) squared)                              equation 1

The saturation contrast is defined by equation 2, positive when the blastema is paler than the body.

loc_sat_ecart = S_body - S_blastema                                               equation 2

The repigmented fraction is defined by equation 3: the share of blastema pixels whose chrominance already
reaches the low level of the pigmented body. This is the variable expected on the State 5 against State 6
boundary.

loc_repig_frac = card(pixels of B such that chroma >= q10(chroma over R)) / card(B)  equation 3

To be run from the FasterViT_for_Image_Classification root, on GPU. Semicolon separator, seed 42.

Verified references, not clickable.
Ronneberger O., Fischer P., Brox T. (2015). U-Net: Convolutional Networks for Biomedical Image
Segmentation. arXiv:1505.04597. DOI 10.1007/978-3-319-24574-4_28.
Tan M., Le Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
arXiv:1905.11946.
Sharma G., Wu W., Dalal E. N. (2005). The CIEDE2000 color-difference formula: implementation notes,
supplementary test data, and mathematical observations. Color Research and Application, 30(1), 21-30.
DOI 10.1002/col.20070.
Reddien P. W., Sanchez Alvarado A. (2004). Fundamentals of planarian regeneration. Annual Review of Cell
and Developmental Biology, 20, 725-757. DOI 10.1146/annurev.cellbio.20.010403.095114.
Hanley J. A., McNeil B. J. (1982). The meaning and use of the area under a receiver operating
characteristic (ROC) curve. Radiology, 143(1), 29-36. DOI 10.1148/radiology.143.1.7063747.
Varoquaux G., Raamana P. R., Engemann D. A., Hoyos-Idrobo A., Schwartz Y., Thirion B. (2017). Assessing
and tuning brain decoding cross-validation, caveats and guidelines. NeuroImage, 145, 166-179.
DOI 10.1016/j.neuroimage.2016.10.038.
Cameron A. C., Gelbach J. B., Miller D. L. (2008). Bootstrap-based improvements for inference with
clustered errors. The Review of Economics and Statistics, 90(3), 414-427. DOI 10.1162/rest.90.3.414.
"""

# %% CELL
# =============================================================================================
# 1. LOCAL COLOUR DESCRIPTORS, COMPUTED INSIDE THE SEGMENTER THUMBNAIL
# =============================================================================================
import os, glob, re, sys, time, datetime, traceback
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import segmentation_models_pytorch as smp
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42)

JOURNAL = os.path.join("Claude outputs", "couleur_locale_blasteme_%s.log"
                       % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs("Claude outputs", exist_ok=True)
class _Tee:
    def __init__(s, f): s.f = open(f, "w", encoding="utf-8"); s.t = sys.stdout
    def write(s, x): s.t.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.t.flush(); s.f.flush()
sys.stdout = _Tee(JOURNAL)
print("journal :", os.path.basename(JOURNAL))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| device:", device)
if device.type != "cuda":
    print("  WARNING: running on the CPU, inference will be about ten times slower.")
else:
    print("  GPU :", torch.cuda.get_device_name(0),
          "| memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

REGEN = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_regeneration")
IMG = os.path.join(REGEN, "images")
_mk = (sorted(glob.glob(os.path.join(REGEN, "masks_model_Mask2Former_*_2pass_clahe*")))
       or sorted(glob.glob(os.path.join(REGEN, "masks_model_Mask2Former_*_2pass*")))
       or sorted(glob.glob(os.path.join(REGEN, "masks_model_Mask2Former*"))))
MK = _mk[-1]; print("mask used:", os.path.basename(MK))
JEU = "dataset_dedup_State_50valid_30test"
METRIQUES = os.path.join(REGEN, "metrics_regeneration", JEU)
CK = os.path.join("checkpoints_regeneration", "blasteme_unet_384_complet", "blasteme_unet_384_best.pth")
SORTIE = os.path.join("Claude outputs", "couleur_locale_blasteme.csv")
S = 384; FRAC = 0.45; MASK_THR = 128
NM = [0.485, 0.456, 0.406, 0.20]; NS = [0.229, 0.224, 0.225, 0.15]
N_MAX = 100000

# ---- geometry taken identically from the segmenter ----
def worm_mask(m):
    m = (m >= MASK_THR).astype(np.uint8)
    n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return m if n <= 1 else (l == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

def orient(bgr, mask):
    ys, xs = np.nonzero(mask); pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ang = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
    H, W = mask.shape; M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang - 90, 1.0)
    rb = cv2.warpAffine(bgr, M, (W, H))
    rm = cv2.warpAffine((mask * 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 127
    return rb, rm

def ends(rm):
    ys, xs = np.nonzero(rm); y0, y1 = ys.min(), ys.max(); span = y1 - y0 + 1
    yt = y0 + int(0.12 * span); rt = rm[yt]
    xt = int(np.where(rt)[0].mean()) if rt.any() else rm.shape[1] // 2
    yb = y1 - int(0.12 * span); rb = rm[yb]
    xb = int(np.where(rb)[0].mean()) if rb.any() else rm.shape[1] // 2
    return (xt, yt), (xb, yb), span

def crop_box(rbgr, rm, cx, cy, span):
    h = int(FRAC * span); H, W = rm.shape
    y0 = max(0, int(cy) - h); y1 = min(H, int(cy) + h)
    x0 = max(0, int(cx) - h); x1 = min(W, int(cx) + h)
    k = max(1, int(0.12 * span))
    rmz = cv2.dilate(rm.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    g3 = rbgr.copy(); g3[~rmz] = 0; c = g3[y0:y1, x0:x1]
    if c.size == 0: return None
    ch, cw = c.shape[:2]; side = max(ch, cw)
    sq = np.zeros((side, side, 3), np.uint8); oy = (side - ch) // 2; ox = (side - cw) // 2
    sq[oy:oy + ch, ox:ox + cw] = c
    return cv2.resize(sq, (S, S)), (y0, y1, x0, x1, oy, ox, side)

def build_channels(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lm = cv2.blur(g, (21, 21)); var = cv2.blur(g * g, (21, 21)) - lm * lm
    ls = np.sqrt(np.maximum(var, 0.0)) + 1e-3
    z = (lm - g) / ls; zc = np.clip((z + 1.0) / 6.0, 0.0, 1.0)
    return torch.from_numpy(np.concatenate(
        [(im.astype(np.float32) / 255.0).transpose(2, 0, 1), zc[None]], 0).astype(np.float32))

class Backbone(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("m", torch.tensor(NM).view(1, 4, 1, 1))
        self.register_buffer("s", torch.tensor(NS).view(1, 4, 1, 1))
    def forward(self, x): return torch.sigmoid(self.net((x - self.m) / self.s))

net = smp.Unet(encoder_name="timm-efficientnet-b0", encoder_weights=None,
               in_channels=4, classes=1, activation=None)
model = Backbone(net).to(device)
_ck = torch.load(CK, map_location=device)
_sd = {k: v for k, v in _ck["state_dict"].items() if not k.startswith("_enc.")}
model.load_state_dict(_sd); model.eval()
print("segmenter loaded:", CK, "| recorded validation Dice:", round(float(_ck.get("val_dice", 0.0)), 3))
print("parameters on the device:", next(model.parameters()).device)

# ---- local colour descriptors ----
COLS = ["loc_sat_blast", "loc_sat_corps", "loc_sat_ecart", "loc_sat_ratio",
        "loc_chroma_blast", "loc_chroma_corps", "loc_chroma_ratio", "loc_chroma_p90_blast",
        "loc_a_ecart", "loc_b_ecart", "loc_L_ecart", "loc_L_cv_blast",
        "loc_sat_cv_blast", "loc_sat_front", "loc_front_contraste",
        "loc_repig_frac", "loc_repig_frac_q25", "loc_chroma_pente", "loc_front_pos",
        "loc_aire_blast"]

def _moy(x, sel):
    return float(np.mean(x[sel])) if int(sel.sum()) > 0 else np.nan

def couleur_locale(hcrop, p):
    """Colour inside the blastema referred to the pigmented body of the same thumbnail, equations 1 to 3."""
    tissue = hcrop.sum(2) > 0
    B = (p >= 0.5) & tissue
    if int(B.sum()) < 30 or int(tissue.sum()) < 500:
        return {c: np.nan for c in COLS}
    k3 = np.ones((3, 3), np.uint8); Bu = B.astype(np.uint8)
    F = cv2.dilate(Bu, k3, iterations=4).astype(bool) & (~B) & tissue
    R = tissue & (~cv2.dilate(Bu, k3, iterations=12).astype(bool))
    if int(R.sum()) < 300:
        return {c: np.nan for c in COLS}
    lab = cv2.cvtColor(hcrop, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(hcrop, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    sat = hsv[:, :, 1]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)          # equation 1
    d = {}
    d["loc_sat_blast"] = _moy(sat, B); d["loc_sat_corps"] = _moy(sat, R)
    d["loc_sat_ecart"] = d["loc_sat_corps"] - d["loc_sat_blast"]   # equation 2
    d["loc_sat_ratio"] = d["loc_sat_blast"] / (d["loc_sat_corps"] + 1e-6)
    d["loc_chroma_blast"] = _moy(chroma, B); d["loc_chroma_corps"] = _moy(chroma, R)
    d["loc_chroma_ratio"] = d["loc_chroma_blast"] / (d["loc_chroma_corps"] + 1e-6)
    d["loc_chroma_p90_blast"] = float(np.percentile(chroma[B], 90))
    d["loc_a_ecart"] = _moy(a, R) - _moy(a, B); d["loc_b_ecart"] = _moy(b, R) - _moy(b, B)
    d["loc_L_ecart"] = _moy(L, B) - _moy(L, R)
    d["loc_L_cv_blast"] = float(np.std(L[B]) / (np.mean(L[B]) + 1e-6))
    d["loc_sat_cv_blast"] = float(np.std(sat[B]) / (np.mean(sat[B]) + 1e-6))
    d["loc_sat_front"] = _moy(sat, F) if int(F.sum()) > 20 else np.nan
    d["loc_front_contraste"] = (d["loc_sat_front"] - d["loc_sat_blast"]
                                if np.isfinite(d["loc_sat_front"]) else np.nan)
    q10 = float(np.percentile(chroma[R], 10)); q25 = float(np.percentile(chroma[R], 25))
    d["loc_repig_frac"] = float((chroma[B] >= q10).mean())          # equation 3
    d["loc_repig_frac_q25"] = float((chroma[B] >= q25).mean())
    d["loc_aire_blast"] = float(B.sum()) / float(tissue.sum())
    # profile along the antero-posterior axis, oriented from the blastema end towards the body
    ys_b = np.nonzero(B)[0]; ys_t = np.nonzero(tissue)[0]
    sens = 1.0 if float(ys_b.mean()) < float(ys_t.mean()) else -1.0
    y0 = float(ys_b.min()) if sens > 0 else float(ys_b.max())
    portee = max(float(ys_t.max()) - float(ys_t.min()), 1.0)
    yy = np.arange(hcrop.shape[0], dtype=np.float32)[:, None] * np.ones((1, hcrop.shape[1]), np.float32)
    u = (yy - y0) * sens
    selp = (B | F)
    if int(selp.sum()) > 50 and float(np.std(u[selp])) > 1e-6:
        d["loc_chroma_pente"] = float(np.polyfit(u[selp], chroma[selp], 1)[0] * 100.0)
    else:
        d["loc_chroma_pente"] = np.nan
    pos = np.nan
    pas = max(int(0.02 * portee), 3)
    for deb in range(0, int(0.6 * portee), pas):
        sel = tissue & (u >= deb) & (u < deb + pas)
        if int(sel.sum()) < 30:
            continue
        if float(np.mean(chroma[sel])) >= q25:
            pos = float(deb + 0.5 * pas) / portee; break
    d["loc_front_pos"] = pos if np.isfinite(pos) else 0.6
    return d

def mesure(bgr, mimg):
    wm = worm_mask(mimg)
    if int(wm.sum()) < 50: return {c: np.nan for c in COLS}
    rbgr, rm = orient(bgr, wm); pa, pb, span = ends(rm); best = None
    for pt in (pa, pb):
        cb = crop_box(rbgr, rm, pt[0], pt[1], span)
        if cb is None: continue
        hcrop, hbox = cb
        with torch.no_grad():
            p = model(build_channels(cv2.cvtColor(hcrop, cv2.COLOR_BGR2RGB))[None].to(device)).cpu()[0, 0].numpy()
        soft = float(p.sum()) * (hbox[-1] / float(S)) ** 2
        if best is None or soft > best[2]: best = (hcrop, p, soft)
    if best is None: return {c: np.nan for c in COLS}
    return couleur_locale(best[0], best[1])

IMGIDX = {}; MKIDX = {}
for s in range(1, 7):
    for p in glob.glob(os.path.join(IMG, "State_%d" % s, "*.jpg")): IMGIDX.setdefault(os.path.basename(p), p)
    for p in glob.glob(os.path.join(MK, "State_%d" % s, "*.jpg")): MKIDX.setdefault(os.path.basename(p), p)
print("indexed images:", len(IMGIDX), "| indexed masks:", len(MKIDX))

liste = pd.concat([pd.read_csv(os.path.join(METRIQUES, "%s_%s_regeneration_metrics_eyespig.csv"
                                            % (JEU, sp)), sep=";")[["image", "class"]]
                   for sp in ("train", "valid", "test")], ignore_index=True)
liste["stade"] = liste["class"].astype(str).str.extract(r"(\d)").astype(int)
print("classifier images:", len(liste), "| found on disk:",
      int(liste.image.astype(str).isin(IMGIDX).sum()))

lignes = []; echecs = 0; t0 = time.time()
for i, r in enumerate(liste.itertuples(index=False), 1):
    if i > N_MAX: break
    ip = IMGIDX.get(str(r.image)); mp = MKIDX.get(str(r.image))
    if ip is None or mp is None: continue
    bgr = cv2.imread(ip); mm = cv2.imread(mp, 0)
    if bgr is None or mm is None: continue
    try:
        d = mesure(bgr, mm)
    except Exception:
        echecs += 1
        if echecs == 1:
            print("   premiere exception :"); print("   " + traceback.format_exc().strip().splitlines()[-1])
        d = {c: np.nan for c in COLS}
    d["image"] = str(r.image); d["stade"] = int(r.stade)
    lignes.append(d)
    if i % 200 == 0:
        print("   %d of %d | %.0f s elapsed | %.2f s per image"
              % (i, len(liste), time.time() - t0, (time.time() - t0) / i), flush=True)

dd = pd.DataFrame(lignes)[["image", "stade"] + COLS]
dd.to_csv(SORTIE, sep=";", index=False)
print("\nimages processed: %d | columns produced: %d | failures: %d | %.0f minutes"
      % (len(dd), len(COLS), echecs, (time.time() - t0) / 60.0))
print("written", SORTIE)
print("\n   %-24s %10s %12s" % ("variable", "NaN share", "std dev"))
for c in COLS:
    print("   %-24s %9.1f %% %12.4f" % (c, 100.0 * float(dd[c].isna().mean()), float(dd[c].std())))
print("\n   mean per state of the key variables")
print("   %-6s %8s %14s %14s %14s %14s"
      % ("stade", "images", "sat_ecart", "chroma_ratio", "repig_frac", "front_pos"))
for s in sorted(dd.stade.unique()):
    u = dd[dd.stade == s]
    print("   %-6d %8d %14.3f %14.3f %14.3f %14.3f"
          % (s, len(u), float(u.loc_sat_ecart.mean()), float(u.loc_chroma_ratio.mean()),
             float(u.loc_repig_frac.mean()), float(u.loc_front_pos.mean())))


# %% CELL
# =============================================================================================
# 2. SEPARATING POWER, BOUNDARY BY BOUNDARY, THEN GAIN BEYOND THE BEST KNOWN VARIABLE
# Protocol identical to metriques_stereotypie_pour_stade: threshold chosen on the training folds
# of a cross-validation grouped by worm and evaluated on the held-out fold, area under the ROC
# curve reported as a threshold-free measure. Grouping by worm follows Varoquaux et al. 2017, the area
# ROC follows Hanley and McNeil 1982; the interval follows the cluster bootstrap of Cameron et al. 2008.
# =============================================================================================
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FRONTIERES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
MEILLEURES = {(1, 2): ("blast_seg_prop", 0.874), (2, 3): ("blast_seg_prop", 0.854),
              (3, 4): ("eyeun_conf", 0.882), (4, 5): ("pup_depth", 0.778),
              (5, 6): ("eyeun_pupil", 0.726)}
GRILLE_Q = np.arange(0.05, 0.96, 0.025)
N_BOOT = 2000

def cle_ver(tige):
    mm = re.match(r"(\d{4})-(\d{2})-(\d{2})-.*?J(\d+)-?(.*)", tige)
    if not mm: return tige
    aa, mo, jj, jr, reste = mm.groups()
    try:
        return (datetime.date(int(aa), int(mo), int(jj))
                - datetime.timedelta(days=int(jr))).isoformat() + "|" + reste
    except Exception:
        return tige

dd = pd.read_csv(SORTIE, sep=";")
ref = pd.concat([pd.read_csv(os.path.join(METRIQUES, "%s_%s_regeneration_metrics_eyespig.csv"
                                          % (JEU, sp)), sep=";") for sp in ("train", "valid", "test")],
                ignore_index=True)
ref["image"] = ref["image"].astype(str).map(os.path.basename)
GARDE = [c for c in ("eyeun_pupil", "eyeun_conf", "pup_depth", "blast_seg_prop") if c in ref.columns]
dd = dd.merge(ref[["image"] + GARDE].drop_duplicates("image"), on="image", how="left")
dd["ver"] = dd["image"].astype(str).str.replace(r"\.(png|jpg|jpeg)$", "", regex=True).map(cle_ver)
LOCS = [c for c in COLS if dd[c].notna().sum() > 0.8 * len(dd) and float(dd[c].std()) > 0]
print("\nusable local variables: %d of %d | images %d | worms %d | known variables joined: %s"
      % (len(LOCS), len(COLS), len(dd), dd.ver.nunique(), ", ".join(GARDE)))

def bacc_seuil(x, y, t):
    p = x >= t
    n1 = max(int((y == 1).sum()), 1); n0 = max(int((y == 0).sum()), 1)
    return 0.5 * (float((p & (y == 1)).sum()) / n1 + float((~p & (y == 0)).sum()) / n0)

def pred_hors_pli(x, y, g):
    gk = GroupKFold(n_splits=min(5, len(np.unique(g)))); pred = np.zeros(len(y), bool)
    for iap, ite in gk.split(x, y, groups=g):
        gr = np.quantile(x[iap], GRILLE_Q)
        t = float(gr[int(np.argmax([bacc_seuil(x[iap], y[iap], u) for u in gr]))])
        pred[ite] = x[ite] >= t
    return pred

def bacc(pred, y):
    n1 = max(int((y == 1).sum()), 1); n0 = max(int((y == 0).sum()), 1)
    return 0.5 * (float((pred & (y == 1)).sum()) / n1 + float((~pred & (y == 0)).sum()) / n0)

def pred_hors_pli_multi(X, y, g):
    gk = GroupKFold(n_splits=min(5, len(np.unique(g)))); pred = np.zeros(len(y), bool)
    for iap, ite in gk.split(X, y, groups=g):
        mod = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=42))
        mod.fit(X[iap], y[iap])
        sap = mod.predict_proba(X[iap])[:, 1]
        gr = np.quantile(sap, GRILLE_Q)
        t = float(gr[int(np.argmax([bacc_seuil(sap, y[iap], u) for u in gr]))])
        pred[ite] = mod.predict_proba(X[ite])[:, 1] >= t
    return pred

lignes = []
for a, b in FRONTIERES:
    sub = dd[dd.stade.isin([a, b])].reset_index(drop=True)
    y = (sub.stade == b).values.astype(int); g = sub.ver.values
    ref_nom, ref_aire = MEILLEURES[(a, b)]
    res = []
    for c in LOCS:
        x = sub[c].values.astype(float); x = np.where(np.isnan(x), np.nanmedian(x), x)
        if np.std(x) == 0: continue
        au = roc_auc_score(y, x)
        res.append((max(au, 1.0 - au), au, c, x if au >= 0.5 else -x))
    res.sort(key=lambda z: -z[0])
    print("\n" + "=" * 92)
    print("STATE %d AGAINST STATE %d BOUNDARY, %d images, %d worms | best known, %s at %.3f"
          % (a, b, len(sub), len(np.unique(g)), ref_nom, ref_aire))
    print("=" * 92)
    print("   %-26s %10s %12s %14s" % ("variable", "ROC area", "direction", "out-of-fold bacc"))
    for aabs, au, c, xo in res:
        bh = bacc(pred_hors_pli(xo, y, g), y)
        mk = "  <-- BEATS the best known" if aabs > ref_aire else ""
        print("   %-26s %10.3f %12s %14.3f%s"
              % (c, aabs, "croissant" if au >= 0.5 else "decroissant", bh, mk))
        lignes.append(dict(frontiere="%d_contre_%d" % (a, b), variable=c, aire_roc=float(aabs),
                           sens="croissant" if au >= 0.5 else "decroissant", bacc_hors_pli=float(bh),
                           meilleure_connue=ref_nom, aire_meilleure_connue=ref_aire,
                           depasse=int(aabs > ref_aire)))
pd.DataFrame(lignes).to_csv(os.path.join("Claude outputs", "couleur_locale_frontieres.csv"),
                            sep=";", index=False)
print("\necrit Claude outputs/couleur_locale_frontieres.csv")

# ---- gain of the local colour over the best known variable, boundaries 4v5 and 5v6 ----
print("\n" + "=" * 92)
print("GAIN BEYOND THE BEST KNOWN VARIABLE, out-of-fold logistic regression grouped by worm")
print("=" * 92)
rng = np.random.default_rng(42)
for a, b in [(4, 5), (5, 6)]:
    ref_nom, _ = MEILLEURES[(a, b)]
    if ref_nom not in dd.columns:
        print("   boundary %d against %d: %s missing from the tables, test skipped" % (a, b, ref_nom)); continue
    sub = dd[dd.stade.isin([a, b])].reset_index(drop=True)
    y = (sub.stade == b).values.astype(int); g = sub.ver.values
    xr = sub[ref_nom].values.astype(float); xr = np.where(np.isnan(xr), np.nanmedian(xr), xr)
    meilleure = max(LOCS, key=lambda c: abs(roc_auc_score(
        y, np.where(np.isnan(sub[c].values.astype(float)),
                    np.nanmedian(sub[c].values.astype(float)), sub[c].values.astype(float))) - 0.5))
    xl = sub[meilleure].values.astype(float); xl = np.where(np.isnan(xl), np.nanmedian(xl), xl)
    p_seul = pred_hors_pli_multi(xr[:, None], y, g)
    p_duo = pred_hors_pli_multi(np.stack([xr, xl], 1), y, g)
    b_seul, b_duo = bacc(p_seul, y), bacc(p_duo, y)
    rho = float(np.corrcoef(xr, xl)[0, 1])
    vers = np.unique(g); ecarts = []
    for _ in range(N_BOOT):
        tir = rng.choice(vers, size=len(vers), replace=True)
        idx = np.concatenate([np.nonzero(g == v)[0] for v in tir])
        ecarts.append(bacc(p_duo[idx], y[idx]) - bacc(p_seul[idx], y[idx]))
    lo, hi = np.percentile(ecarts, [2.5, 97.5])
    print("\n   State %d against State %d boundary, %d images, %d worms" % (a, b, len(sub), len(vers)))
    print("   most informative local variable: %s | Pearson correlation with %s: %.3f"
          % (meilleure, ref_nom, rho))
    print("   out-of-fold bacc, %s alone       : %.4f" % (ref_nom, b_seul))
    print("   out-of-fold bacc, %s + %s : %.4f" % (ref_nom, meilleure, b_duo))
    print("   difference: %+.4f  95 %% interval by worm bootstrap [%+.4f, %+.4f]  %s"
          % (b_duo - b_seul, lo, hi, "SIGNIFICANT" if lo > 0 else "not significant"))

print("\nPRE-REGISTERED READING. A local variable that exceeds 0.726 on the State 5 against State 6")
print("boundary replaces the best known variable for that boundary. A variable that stays below 0.726")
print("but whose addition to eyeun_pupil yields a strictly positive bootstrap interval carries")
print("complementary information and deserves to enter the metric table. If both conditions fail, the")
print("blastema colour does not carry the separation between these two states and the lead is closed.")
print("\nend:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
