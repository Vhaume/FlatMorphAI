# -*- coding: utf-8 -*-
# calcul_descripteurs_commentaires_stereotypies_v3g (21 October 2026). New program; the v3c and v3f programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Finding (diagnostic of 21 October): the original 96% test score rested in part on the zero-motion artefact (the original
# models retrained with the real motion drop to 92.0% in test, 26 -> 22 correct Snake-like). The remaining errors
# (validation and test) match the cases described by the expert comments (analysis_out/class_signatures_from_comments.md,
# hardcase_comment_metric_analysis.md): elongated Snake-like read as Wildtype (body zigzagging over time, zones
# widening or thinning, abnormally long worm), Screw-like read as C-shape or Snake-like (hourglass neck, sharp
# turn, thin tail). The metrics written for these comments exist (twist_snake_metrics.py, twist_snake_temporal.py) but
# are not among the 301 descriptors: the temporal ones were reading the MOTION image of the artefact (neighbours copied, zero
# values). They are computed here for every frame with the real neighbours.
# Five groups added, all computed (no value guessed or imputed; the run stops if a value is not finite):
#   tt_ (6)  : twist_snake_temporal on the v3f MOTION image (T-1, T0, T+1 registered), zigzag of the median line over time,
#              width oscillation per zone, length and area oscillation (Rao and Schunck, 1991, CVGIP: Graphical Models
#              and Image Processing, DOI 10.1016/1049-9652(91)90059-S).
#   cm_ (19) : twist_snake_metrics on the T0 frame in 384 format (mask applied): hourglass neck, middle / ends ratio,
#              sharp turn of the contour, border undulation, zigzag of the median line, axial gradient (Mokhtarian and
#              Mackworth, 1992, IEEE TPAMI, DOI 10.1109/34.149591).
#   lp_ (10) : pigmentation profile along the body, Valentin's remark of 21 October: on the Snake-like read as Wildtype, one
#              part of the body is darker and the relaxed, stretched part is less dark and brighter. Body axis by
#              geodesic distance inside the mask from one extremity (MCP_Geometric of scikit-image, as in v3c), 16 slices
#              over 10 to 90% of the length, median luminance of the core of each slice (pixels at least half the
#              maximum border distance of the slice, to exclude the translucent border rim). Profile contrast, best
#              two-part split (dark / pale) and pale share, width / darkness correlation, ratio of the widths of the
#              pale part and the dark part, core luminance relative to the neighbouring background, and their variations over T-1, T0,
#              T+1 (stretching lightens the worm). Segmentation by optimal split into two segments: Bai and Perron, 1998,
#              Econometrica, DOI 10.2307/2998540.
#   tc_, te_, tn_ (11) : Valentin's remarks of 21 October on the test failures, measured on T0 and on T-1 and T+1 registered on
#              T0 (align_to of make_motion_dataset), geodesic axis and local width 2 x border distance (Blum, 1967, A
#              transformation for extracting new descriptors of shape, MIT Press) :
#              tc_ : a C-shape that lasts over time (Acetylcholine_10000uM_T1h_R2(1) F5, Wildtype read as C-shape): overlap (IoU,
#                    Jaccard, 1912, New Phytologist, DOI 10.1111/j.1469-8137.1912.tb05611.x) of the registered silhouettes, curvature
#                    global 1 - chord / axis length at each frame, its minimum and its range over the three frames.
#              te_ : change in the width of one extremity (Atropine_500uM_T24h_R2(5) F4, head down): width of the terminal 15%
#                    relative to the median width, variation over the three frames for each end (ends paired
#                    by proximity after registration), maximum, asymmetry between the ends and ratio of the two ends at T0.
#              tn_ : Screw-like very thin instead of the twist, width changing strongly at the same place between T-1, T0 and T+1
#                    (Methiothepin_5uM_T3h_R1(4) F2): minimum body width (outside the terminal 15%) relative to the median, and
#                    logarithmic difference of the local width of the neighbouring frames at the same place (mean and maximum).
#              Every quantity is dimensionless (ratios to the median width or to the worm length).
#   rs_ (12) : raw scale, read on the Mask2Former masks at the camera resolution (the 384 format normalises the size of
#              each worm and erases the scale): area, skeleton length (Zhang and Suen, 1984, Communications of the ACM,
#              DOI 10.1145/357994.358023), widths, variations over T-1, T0, T+1 and displacement of the centre of mass, in units of
#              sqrt(image height x width), so as to be independent of the resolution.
# Outputs: lightgbm_stereotype_v3g/data/features_{train,valid,test,golden}.csv (v3f tables + 58 columns), cache _cache_v3g.pkl.
# Central processor, one thread per process. Seed 42. Run from the FasterViT_for_Image_Classification root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
_src = open(os.path.join(ROOT, "calcul_mouvement_stereotypies_v3f.py"), encoding="utf-8").read()
assert 'if __name__ == "__main__":\n    main()' in _src
_G = {"__name__": "v3f_bibliotheque", "__file__": os.path.join(ROOT, "calcul_mouvement_stereotypies_v3f.py")}
exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_mouvement_stereotypies_v3f", "exec"), _G)
format_384, voisins, fixe, _lire, NS_MOT, CIM = (_G[k] for k in ("format_384", "voisins", "fixe", "_lire", "NS_MOT", "CIM"))
sys.path.insert(0, ROOT)
from twist_snake_temporal import twist_snake_temporal, TT_COLS
from twist_snake_metrics import twist_snake_metrics, CM_COLS
V3F = os.path.join(ROOT, "lightgbm_stereotype_v3f"); V3G = os.path.join(ROOT, "lightgbm_stereotype_v3g")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")
RS_COLS = ["rs_area", "rs_len", "rs_wmax", "rs_wmean", "rs_thin", "rs_area_rng", "rs_len_rng", "rs_len_max", "rs_area_max", "rs_disp", "rs_disp0", "rs_disp_len"]
LP0 = ["lp_range", "lp_step", "lp_light_frac", "lp_wi_corr", "lp_light_thin", "lp_dark_rel"]
LP_COLS = LP0 + ["lp_lum_rng", "lp_dark_rel_rng", "lp_range_rng", "lp_step_rng"]
EX_COLS = ["tc_iou_min", "tc_iou_mean", "tc_bend_t0", "tc_bend_min", "tc_bend_rng", "te_endw_chg_max", "te_endw_chg_asym", "te_endw_ratio",
           "tn_neck_thin", "tn_neck_chg", "tn_neck_chg_max"]
NEW = TT_COLS + CM_COLS + RS_COLS + LP_COLS + EX_COLS

def format_384_fond(nm):
    """Same crop as format_384, without applying the mask (background kept), in grey levels."""
    rp, mp = fixe(nm); raw = _lire(rp, cv2.IMREAD_COLOR); m = _lire(mp)
    obj = CIM.biggest_objects_in_the_binary_mask(m, num_objects_to_keep=1, num_objects_by_row=1)
    out = CIM.compute_mask_image_planarian_reshaped(raw, obj[0], image_size=384, margin_percent=0.0, apply_mask=False, black_and_white=False)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)

def _geod(m, depart):
    d, _ = MCP_Geometric(np.where(m, 1.0, np.inf)).find_costs([depart]); d = np.asarray(d, float); d[~m] = np.inf; return d

def profil(g, gf, NB=20):
    """Pigmentation profile along the geodesic axis (g: masked 384 frame, gf: same crop with the background)."""
    m = g > 8; k, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8); assert k >= 2, "masque vide"
    m = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); ys, xs = np.nonzero(m)
    d = _geod(m, (ys[0], xs[0])); a = np.unravel_index(int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape)
    d = _geod(m, a); L = float(d[np.isfinite(d)].max()); assert L > 10, "axis too short"
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); bb = np.full(m.shape, -1); bb[m] = np.minimum((d[m] / L * NB).astype(int), NB - 1)
    I = np.full(NB, np.nan); Wd = np.zeros(NB)
    for q in range(NB):
        sel = bb == q
        if sel.any(): I[q] = float(np.median(g[sel & (dt >= 0.5 * dt[sel].max())])); Wd[q] = sel.sum() / (L / NB)
    ok = np.isfinite(I); I = np.interp(np.arange(NB), np.where(ok)[0], I[ok])
    Is = gaussian_filter1d(I, 1.0)[2:NB - 2]; Ws = gaussian_filter1d(Wd, 1.0)[2:NB - 2]; n = len(Is); mu = float(np.median(Is))
    best, bk = -1.0, n // 2
    for kk in range(3, n - 2):
        c = abs(Is[:kk].mean() - Is[kk:].mean())
        if c > best: best, bk = c, kk
    clair_debut = Is[:bk].mean() > Is[bk:].mean()
    wl, wd = (Ws[:bk].mean(), Ws[bk:].mean()) if clair_debut else (Ws[bk:].mean(), Ws[:bk].mean())
    r = float(np.corrcoef(Ws, -Is)[0, 1]) if Ws.std() > 0 and Is.std() > 0 else 0.0
    anneau = (cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~(cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & (gf > 8)
    fond = float(np.median(gf[anneau])) if anneau.sum() > 50 else float(np.median(gf[~m]))
    coeur = float(np.median(g[m & (dt >= 0.5 * dt.max())]))
    return dict(lp_range=float(np.percentile(Is, 90) - np.percentile(Is, 10)) / mu, lp_step=best / mu, lp_light_frac=(bk / n if clair_debut else 1 - bk / n),
                lp_wi_corr=r, lp_light_thin=wl / max(wd, 1e-6), lp_dark_rel=coeur / max(fond, 1e-6), coeur=coeur)

def echelle(nm):
    """Quantities at the raw scale of the Mask2Former mask of frame nm (largest object)."""
    m = _lire(fixe(nm)[1]); h, w = m.shape[:2]; b = (m > 127).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(b, 8); assert k >= 2, "masque vide : " + nm
    b = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    s = cv2.resize(b.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) > 0
    L = 2.0 * float(skeletonize(s).sum()); dt = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(b); u = float(np.sqrt(h * w))
    return dict(area=b.sum() / float(h * w), L=L / u, wmax=2.0 * float(dt.max()) / u, wmean=(b.sum() / max(L, 1.0)) / u, c=np.array([ys.mean(), xs.mean()]) / u)

def _objet(g):
    m = g > 8; k, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8); assert k >= 2, "masque vide"
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))

def axe(m, NB=40):
    """Geodesic axis of mask m: extremities a and b, length L, medial points and local widths per slice."""
    ys, xs = np.nonzero(m); d = _geod(m, (ys[0], xs[0]))
    a = np.unravel_index(int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape); d = _geod(m, a)
    b = np.unravel_index(int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape); L = float(d[b]); assert L > 10, "axis too short"
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); q = np.full(m.shape, -1); q[m] = np.minimum((d[m] / L * NB).astype(int), NB - 1)
    P = np.zeros((NB, 2)); w = np.zeros(NB)
    for k in range(NB):
        yy, xx = np.nonzero(q == k)
        if len(yy): j = int(np.argmax(dt[yy, xx])); P[k] = (yy[j], xx[j]); w[k] = 2.0 * dt[yy[j], xx[j]]
        else: P[k] = P[k - 1] if k else a; w[k] = w[k - 1] if k else 1.0
    e = max(1, int(0.15 * NB)); wmed = float(np.median(w[e:NB - e])) + 1e-6
    return dict(a=np.array(a, float), b=np.array(b, float), L=L, P=P, w=w, wmed=wmed, dt=dt, e=e,
                bend=1.0 - float(np.hypot(*(np.array(a, float) - np.array(b, float)))) / L)

def expert(gt, gp, gn):
    """Descriptors tc_, te_, tn_ (T0 and neighbours already registered on T0)."""
    M = [_objet(g) for g in (gt, gp, gn)]; X = [axe(m) for m in M]; x0 = X[0]; e = x0["e"]
    iou = [float((M[0] & m).sum()) / float((M[0] | m).sum()) for m in M[1:]]
    bends = [x["bend"] for x in X]
    def bouts(x):
        return {"a": float(x["w"][:x["e"]].mean()) / x["wmed"], "b": float(x["w"][-x["e"]:].mean()) / x["wmed"]}
    B0 = bouts(x0); chg = {"a": [], "b": []}
    for x in X[1:]:
        Bx = bouts(x); direct = np.hypot(*(x["a"] - x0["a"])) + np.hypot(*(x["b"] - x0["b"])) <= np.hypot(*(x["a"] - x0["b"])) + np.hypot(*(x["b"] - x0["a"]))
        ba, bb = (Bx["a"], Bx["b"]) if direct else (Bx["b"], Bx["a"])
        chg["a"].append(abs(np.log(max(ba, 1e-3) / max(B0["a"], 1e-3)))); chg["b"].append(abs(np.log(max(bb, 1e-3) / max(B0["b"], 1e-3))))
    ca, cb = max(chg["a"]), max(chg["b"])
    k = e + int(np.argmin(x0["w"][e:len(x0["w"]) - e])); py, px = x0["P"][k].astype(int); w0 = max(float(x0["w"][k]), 1.0)
    r = max(2, int(round(x0["wmed"]))); ecarts = []
    for x in X[1:]:
        y0_, y1_, x0_, x1_ = max(0, py - r), py + r + 1, max(0, px - r), px + r + 1
        wl = max(2.0 * float(x["dt"][y0_:y1_, x0_:x1_].max()), 1.0); ecarts.append(abs(np.log(wl / w0)))
    return dict(tc_iou_min=min(iou), tc_iou_mean=float(np.mean(iou)), tc_bend_t0=bends[0], tc_bend_min=min(bends), tc_bend_rng=max(bends) - min(bends),
                te_endw_chg_max=max(ca, cb), te_endw_chg_asym=abs(ca - cb) / (ca + cb + 1e-6), te_endw_ratio=max(B0["a"], B0["b"]) / max(min(B0["a"], B0["b"]), 1e-3),
                tn_neck_thin=w0 / x0["wmed"], tn_neck_chg=float(np.mean(ecarts)), tn_neck_chg_max=float(max(ecarts)))

def calcule(arg):
    sp, bn = arg
    try:
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        pv, nx = voisins(bn)
        gp, gn = prep(pv), prep(nx)
        mi = NS_MOT["encode_motion"](gp, gt, gn, gt > 8)
        out = {}; out.update(twist_snake_temporal(mi)); out.update(twist_snake_metrics(gt)); out.update(expert(gt, gp, gn))
        pr = [profil(gt, format_384_fond(bn))] + [profil(format_384(x), format_384_fond(x)) for x in (pv, nx)]
        out.update({c: pr[0][c] for c in LP0})
        rng = lambda c: float(max(q[c] for q in pr) - min(q[c] for q in pr))
        out.update(lp_lum_rng=rng("coeur") / max(np.mean([q["coeur"] for q in pr]), 1e-6), lp_dark_rel_rng=rng("lp_dark_rel"), lp_range_rng=rng("lp_range"), lp_step_rng=rng("lp_step"))
        e0, e1, e2 = echelle(bn), echelle(pv), echelle(nx)
        A_ = np.array([e0["area"], e1["area"], e2["area"]]); L_ = np.array([e0["L"], e1["L"], e2["L"]])
        out.update(rs_area=e0["area"], rs_len=e0["L"], rs_wmax=e0["wmax"], rs_wmean=e0["wmean"], rs_thin=e0["wmean"] / max(e0["L"], 1e-6),
                   rs_area_rng=(A_.max() - A_.min()) / A_.mean(), rs_len_rng=(L_.max() - L_.min()) / L_.mean(), rs_len_max=L_.max(), rs_area_max=A_.max(),
                   rs_disp=float(np.linalg.norm(e1["c"] - e2["c"])), rs_disp0=float(max(np.linalg.norm(e0["c"] - e1["c"]), np.linalg.norm(e0["c"] - e2["c"]))),
                   rs_disp_len=float(np.linalg.norm(e1["c"] - e2["c"])) / max(e0["L"], 1e-6))
        return sp, bn, {k: float(out[k]) for k in NEW}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def main():
    t0 = time.time()
    print("v3g comment descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "| central processor, %d processes" % N_WORKERS, flush=True)
    assert not os.path.exists(os.path.join(V3G, "data")), "lightgbm_stereotype_v3g/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3F, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V3G, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V3G, "_cache_v3g.pkl"); RES = {}
    if os.path.exists(CACHE):
        c = pickle.load(open(CACHE, "rb")); RES = c["res"] if c.get("sig") == SIG else {}
    reste = [t for t in taches if t not in RES]; print("cache: %d, to compute: %d" % (len(RES), len(reste)), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, (s, bn, out) in enumerate(pool.imap_unordered(calcule, reste, chunksize=2), 1):
            RES[(s, bn)] = out
            if i % 300 == 0: pickle.dump(dict(sig=SIG, res=RES), open(CACHE, "wb")); print("  %d / %d | %.0f s" % (i, len(reste), time.time() - t0), flush=True)
    pickle.dump(dict(sig=SIG, res=RES), open(CACHE, "wb"))
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in NEW: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], NEW[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][NEW].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V3G, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v3g/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V3G, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V3G, "data", "features_%s.csv" % s), sep=";", index=False)
    print("\nmedians per class (training):")
    pd.set_option("display.width", 250)
    print(OUTT["train"].groupby("label")[["tt_midline_lat_osc", "tt_width_zone_osc", "cm_hourglass", "cm_max_turn", "rs_len", "rs_disp", "lp_step", "lp_dark_rel", "lp_lum_rng"]].median().round(4).to_string())
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d new descriptors of the 4 tables | %.0f s" % (len(NEW), time.time() - t0))
    print("v3g tables:", os.path.join(V3G, "data"), "| v3f tables kept:", os.path.join(V3F, "data"))

if __name__ == "__main__":
    main()
