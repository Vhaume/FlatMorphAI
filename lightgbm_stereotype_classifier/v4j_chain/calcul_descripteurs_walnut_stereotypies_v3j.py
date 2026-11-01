# -*- coding: utf-8 -*-
# calcul_descripteurs_walnut_stereotypies_v3j (21 October 2026). New program; the v3c to v3i programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Valentin's objective: Walnut classification at 100%. The hard cases are already commented (analysis_out/failed_images_to_review.csv,
# comment_correction_log.csv, ambiguous_by_comment.csv, failures_to_review_prev.csv) :
#   Walnut: worm almost motionless, compact and rounded, border in small waves, right border shaped like a '3', uniform texture
#            (Nicotine_50uM_T1h_R3(2) F26, Pilocarpine_50uM_T3h_R3(4) F27).
#   C-shape: worm on its back in a C that moves too much for a Walnut (Nicotine_500uM_T0_R3(3) F6), non-uniform texture with a
#            hill (Physostigmine_50uM_T24h_R1(5) F19), saturation at the centre indicating a C folded on itself
#            (Physostigmine_50uM_T24h_R1(1) F8, remark of 21 October).
#   Screw-like: border with sharp turns and scalloped, a Walnut would be more rounded (Physostigmine_50uM_T24h_R1(4) F19), width.
# Valentin's remark of 21 October (image with modified gamma): the Walnut has a saturation concentrated at its centre, less spread out than
# that of the folded C-shape. Probe over the 1,216 Walnut and C-shape images: among the motionless and compact worms (overlap between
# frames >= 0.93, solidity >= 0.97, 137 training Walnut and 142 training C-shape), the share of very dark body separates the two
# classes (AUC 0.70), whereas it does not separate them over all worms.
# Group wn_ (14 descriptors, dimensionless), frame T0 in 384 format in colour with the mask applied (same crop as v3f):
#   wn_dark03, wn_dark05, wn_dark08, wn_dark10 : share of the worm whose luminance is at most percentile 1 + 3, 5, 8 or 10% of
#                      the range between percentiles 1 and 99 of the worm (extent of the saturated zone).
#   wn_rouge_gamma_frac, wn_rouge_gamma_spread : red zone of the image stretched between percentiles 1 and 99 then corrected by a gamma
#                      of 1 / 2.2 (R > 1.25 G and R > 1.25 B), share of the worm and radius of gyration / sqrt(area).
#   wn_red_med, wn_red_p90 : redness (R - (G + B) / 2) / luminance, median and percentile 90 inside the worm.
#   wn_red_q80_spread, wn_red_q90_spread : radius of gyration of the reddest pixels (percentiles 80 and 90) / sqrt(area).
#   wn_solidite : area / area of the convex hull.   wn_circularite : 4 pi area / perimeter^2 (contour simplified to 1.5 pixel).
#   wn_droit, wn_courbure_cv : share of the contour that is nearly straight (turn < 2 degrees) and irregularity of the curvature (standard deviation /
#                      mean turn), contour resampled at 240 points and smoothed (Gaussian sigma = 2).
# Outputs: lightgbm_stereotype_v3j/data/features_{train,valid,test,golden}.csv (v3i tables + 14 columns), cache _cache_v3j.pkl.
# Zero tolerance on NaN. Central processor, one thread per process. Seed 42. Run from the FasterViT root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
_src = open(os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py"), encoding="utf-8").read()
assert 'if __name__ == "__main__":\n    main()' in _src
_L = {"__name__": "v3g_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py")}
exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_commentaires_stereotypies_v3g", "exec"), _L)
fixe, _lire, CIM = (_L[k] for k in ("fixe", "_lire", "CIM"))
V3I = os.path.join(ROOT, "lightgbm_stereotype_v3i"); V3J = os.path.join(ROOT, "lightgbm_stereotype_v3j")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")
WN_COLS = ["wn_dark03", "wn_dark05", "wn_dark08", "wn_dark10", "wn_rouge_gamma_frac", "wn_rouge_gamma_spread", "wn_red_med", "wn_red_p90", "wn_red_q80_spread", "wn_red_q90_spread", "wn_solidite", "wn_circularite", "wn_droit", "wn_courbure_cv"]

def couleur_384(nm):
    """Frame nm in 384 format, colour, mask applied (same crop as format_384 of v3f)."""
    rp, mp = fixe(nm); raw = _lire(rp, cv2.IMREAD_COLOR); m = _lire(mp)
    obj = CIM.biggest_objects_in_the_binary_mask(m, num_objects_to_keep=1, num_objects_by_row=1)
    assert obj and obj[0] is not None, "no object in the mask: " + nm
    return CIM.compute_mask_image_planarian_reshaped(raw, obj[0], image_size=384, margin_percent=0.0, apply_mask=True, black_and_white=False)

def walnut(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32); m = (g > 8).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); assert k >= 2, "masque vide"
    m = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); A = float(m.sum()); rA = np.sqrt(A)
    v = g[m]; p1, p99 = np.percentile(v, [1, 99]); rng = max(p99 - p1, 1.0); out = {}
    for t in (0.03, 0.05, 0.08, 0.10): out["wn_dark%02d" % int(round(t * 100))] = float((m & (g <= p1 + t * rng)).sum()) / A
    def rayon(z):
        ys, xs = np.nonzero(z)
        return float(np.sqrt(((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2).mean())) / rA if len(ys) > 5 else 0.0
    b, gg, r = (bgr[..., i].astype(np.float32) for i in range(3)); red = (r - (gg + b) / 2) / np.maximum(g, 1.0); rv = red[m]
    out["wn_red_med"] = float(np.median(rv)); out["wn_red_p90"] = float(np.percentile(rv, 90))
    for q in (80, 90): out["wn_red_q%d_spread" % q] = rayon(m & (red >= np.percentile(rv, q)))
    e = np.clip((bgr.astype(np.float32) - p1) / rng, 0, 1) ** (1 / 2.2) * 255
    rz = m & (e[..., 2] > 1.25 * e[..., 1]) & (e[..., 2] > 1.25 * e[..., 0])
    out["wn_rouge_gamma_frac"] = float(rz.sum()) / A; out["wn_rouge_gamma_spread"] = rayon(rz)
    mu = m.astype(np.uint8); c = max(cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    out["wn_solidite"] = A / max(float(cv2.contourArea(cv2.convexHull(c))), 1.0)
    cs = cv2.approxPolyDP(c, 1.5, True); out["wn_circularite"] = 4.0 * np.pi * A / max(float(cv2.arcLength(cs, True)), 1.0) ** 2
    P = c[:, 0, :].astype(float)[np.linspace(0, len(c) - 1, 240).astype(int)]
    P[:, 0] = gaussian_filter1d(P[:, 0], 2.0, mode="wrap"); P[:, 1] = gaussian_filter1d(P[:, 1], 2.0, mode="wrap")
    a_ = P - np.roll(P, 4, 0); b_ = np.roll(P, -4, 0) - P
    ang = np.degrees(np.arctan2(a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0], (a_ * b_).sum(1)))
    out["wn_droit"] = float((np.abs(ang) < 2).mean()); out["wn_courbure_cv"] = float(np.std(ang) / max(np.mean(np.abs(ang)), 1e-6))
    return out

def calcule(arg):
    sp, bn = arg
    try:
        out = walnut(couleur_384(bn)); return sp, bn, {k: float(out[k]) for k in WN_COLS}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def main():
    t0 = time.time()
    print("v3j Walnut descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "| central processor, %d processes" % N_WORKERS, flush=True)
    assert not os.path.exists(os.path.join(V3J, "data")), "lightgbm_stereotype_v3j/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3I, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V3J, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_walnut_stereotypies_v3j.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V3J, "_cache_v3j.pkl"); RES = {}
    if os.path.exists(CACHE):
        c = pickle.load(open(CACHE, "rb")); RES = c["res"] if c.get("sig") == SIG else {}
    reste = [t for t in taches if t not in RES]; print("cache: %d, to compute: %d" % (len(RES), len(reste)), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, (s, bn, out) in enumerate(pool.imap_unordered(calcule, reste, chunksize=4), 1):
            RES[(s, bn)] = out
            if i % 600 == 0: pickle.dump(dict(sig=SIG, res=RES), open(CACHE, "wb")); print("  %d / %d | %.0f s" % (i, len(reste), time.time() - t0), flush=True)
    pickle.dump(dict(sig=SIG, res=RES), open(CACHE, "wb"))
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in WN_COLS: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], WN_COLS[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][WN_COLS].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V3J, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v3j/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V3J, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V3J, "data", "features_%s.csv" % s), sep=";", index=False)
    pd.set_option("display.width", 250)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[WN_COLS].median().round(3).to_string())
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test")])
    print("\ncommented cases:\n" + T[T.image.str.contains("Physostigmine_50µM_T24h_R1|Nicotine_100µM_T1h_R3\\(5\\)_300323_Frame_10", regex=True)][["partition", "image", "label"] + WN_COLS].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors wn_ | %.0f s" % (len(WN_COLS), time.time() - t0))
    print("v3j tables:", os.path.join(V3J, "data"), "| v3i tables kept:", os.path.join(V3I, "data"))

if __name__ == "__main__":
    main()
