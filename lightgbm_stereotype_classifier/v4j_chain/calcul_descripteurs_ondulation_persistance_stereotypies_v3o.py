# -*- coding: utf-8 -*-
# calcul_descripteurs_ondulation_persistance_stereotypies_v3o (21 October 2026). New program; the v3c to v3n programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Valentin's remarks of 21 October (errors of the v3m and v3n test):
#   Atropine_500uM_T0_R2(1) F18 and Acetylcholine_10000uM_T1h_R2(1) F5 (Wildtype read as C-shape): LATERAL motion, the worm undulates.
#   Expert remark already recorded on Acetylcholine F5: a C is smoother and lasts longer than a sharp turn.
# v3n finding: the raw displacement of the ends (mv_) is more frequent in C-shape than in Wildtype (coiling is itself a
# motion). What separates a held C from an undulating worm is not the amplitude of the displacement but the SHAPE of the curvature over
# time: a C keeps a single curvature of the same sign over T-1, T0, T+1, whereas an undulating worm has curvatures of alternating signs along the
# body (a wave) that change from one frame to the next.
# Measures on the v3g geodesic axis (axe function, 40 slices, medial points, Blum, 1967, MIT Press), frame T0 in 384 format
# and neighbours T-1, T+1 registered on T0 (align_to, as in v3i and v3n), the neighbours' ends paired with those of T0 (oriente function).
# Signed curvature: medial points smoothed (Gaussian sigma = 1.5 slice), unwrapped tangent angle, signed angle increment per
# slice (Mokhtarian and Mackworth, 1992, IEEE TPAMI, DOI 10.1109/34.149591). Lateral deviation: signed distance of each point of
# the axis to the chord joining the two ends, divided by the length L of the frame. Everything is dimensionless.
#   cp_ (persistance du C, 5) :
#     cp_tour_min     : min over T-1, T0, T+1 of |total angle turned| / pi (a held C stays close to 1 on all three frames).
#     cp_tour_cv      : standard deviation / mean of |total angle turned| over the three frames (a held C varies little).
#     cp_tour_coh     : |sum of the signed total angles| / sum of the |total angles| (1 if the curvature keeps the same sign).
#     cp_unisigne_min : min over the three frames of |sum of the signed increments| / sum of the |increments| (1 for a single
#                       curvature of constant sign, low for an S or a wave).
#     cp_lisse_max    : max over the three frames of maximum curvature / mean curvature (sharp turn against smooth C).
#   ond_ (lateral undulation, 4):
#     ond_inflex_moy  : mean number of inflection points per frame (sign changes between curvature lobes whose cumulated
#                       angle exceeds 0.2 rad, so as to ignore contour noise).
#     ond_lat_chg     : max over the neighbours of the mean |lateral deviation of the neighbour - lateral deviation of T0| (a wave that moves).
#     ond_lat_inv     : max over the neighbours of the share of the lateral deviation whose sign flips between T0 and the neighbour
#                       (sum of the |deviations| at the sign-flipped positions / sum of the |deviations|, between 0 and 1).
#     ond_courb_dist  : max over the neighbours of the mean |signed increment of the neighbour - signed increment of T0| (rad per slice).
# Outputs: lightgbm_stereotype_v3o/data/features_{train,valid,test,golden}.csv (v3n tables + 9 columns), cache _cache_v3o.pkl.
# Zero tolerance on NaN (the run stops, no table written). Central processor, one thread per process. Seed 42.
# Run from the FasterViT_for_Image_Classification root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_O = ["cp_tour_min", "cp_tour_cv", "cp_tour_coh", "cp_unisigne_min", "cp_lisse_max",
          "ond_inflex_moy", "ond_lat_chg", "ond_lat_inv", "ond_courb_dist"]
SEUIL_LOBE = 0.2

def oriente(x0, x):
    """True if extremity a of x corresponds to extremity a of x0."""
    return np.hypot(*(x["a"] - x0["a"])) + np.hypot(*(x["b"] - x0["b"])) <= np.hypot(*(x["a"] - x0["b"])) + np.hypot(*(x["b"] - x0["a"]))

def profil(P, L):
    """Signed angle increments per slice, signed lateral deviation from the chord / L."""
    Ps = gaussian_filter1d(np.asarray(P, float), 1.5, axis=0, mode="nearest"); d = np.diff(Ps, axis=0)
    th = np.unwrap(np.arctan2(d[:, 0], d[:, 1])); dth = np.diff(th)
    c = Ps[-1] - Ps[0]; nc = float(np.hypot(*c))
    if nc < 1e-6: lat = np.hypot(*(Ps - Ps[0]).T) / L   # ends coincide: distance to the first point
    else: lat = ((Ps[:, 1] - Ps[0, 1]) * c[0] - (Ps[:, 0] - Ps[0, 0]) * c[1]) / nc / L
    return dth, lat

def inflexions(dth):
    """Number of sign changes between significant curvature lobes (cumulated angle >= SEUIL_LOBE)."""
    s = np.sign(dth); lobes, k = [], 0
    while k < len(dth):
        j = k
        while j + 1 < len(dth) and s[j + 1] == s[k]: j += 1
        aire = float(np.abs(dth[k:j + 1]).sum())
        if s[k] != 0 and aire >= SEUIL_LOBE: lobes.append(s[k])
        k = j + 1
    return float(sum(1 for u, v in zip(lobes, lobes[1:]) if u != v))

def descripteurs(X):
    """X: axes (output of axe) of T0, T-1, T+1, the neighbours already registered on T0."""
    x0 = X[0]; D, LAT = [], []
    for i, x in enumerate(X):
        P = x["P"] if (i == 0 or oriente(x0, x)) else x["P"][::-1]
        dth, lat = profil(P, x["L"]); D.append(dth); LAT.append(lat)
    tot = np.array([float(d.sum()) for d in D]); atot = np.abs(tot)
    uni = [abs(float(d.sum())) / max(float(np.abs(d).sum()), 1e-6) for d in D]
    lis = [float(np.abs(d).max()) / max(float(np.abs(d).mean()), 1e-6) for d in D]
    chg, inv, cd = [], [], []
    for d, lat in zip(D[1:], LAT[1:]):
        chg.append(float(np.abs(lat - LAT[0]).mean()))
        opp = np.sign(lat) * np.sign(LAT[0]) < 0; tot_l = float((np.abs(lat) + np.abs(LAT[0])).sum())
        inv.append(float((np.abs(lat) + np.abs(LAT[0]))[opp].sum()) / max(tot_l, 1e-6))
        cd.append(float(np.abs(d - D[0]).mean()))
    return dict(cp_tour_min=float(atot.min()) / np.pi, cp_tour_cv=float(atot.std()) / max(float(atot.mean()), 1e-6),
                cp_tour_coh=abs(float(tot.sum())) / max(float(atot.sum()), 1e-6), cp_unisigne_min=min(uni), cp_lisse_max=max(lis),
                ond_inflex_moy=float(np.mean([inflexions(d) for d in D])), ond_lat_chg=max(chg), ond_lat_inv=max(inv), ond_courb_dist=max(cd))

def _bibliotheque():
    _src = open(os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    L = {"__name__": "v3g_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_commentaires_stereotypies_v3g", "exec"), L)
    return L

if __name__ == "__main__":
    _L = _bibliotheque()
    format_384, voisins, NS_MOT, axe, _objet = (_L[k] for k in ("format_384", "voisins", "NS_MOT", "axe", "_objet"))
V3N = os.path.join(ROOT, "lightgbm_stereotype_v3n"); V3O = os.path.join(ROOT, "lightgbm_stereotype_v3o")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

def calcule(arg):
    sp, bn = arg
    try:
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        pv, nx = voisins(bn)
        out = descripteurs([axe(_objet(g)) for g in (gt, prep(pv), prep(nx))])
        return sp, bn, {k: float(out[k]) for k in COLS_O}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def main():
    t0 = time.time()
    print("v3o undulation and C-persistence descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "| central processor, %d processes" % N_WORKERS, flush=True)
    assert not os.path.exists(os.path.join(V3O, "data")), "lightgbm_stereotype_v3o/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3N, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V3O, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_ondulation_persistance_stereotypies_v3o.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V3O, "_cache_v3o.pkl"); RES = {}
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
        for c in COLS_O: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_O[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_O].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V3O, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v3o/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V3O, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V3O, "data", "features_%s.csv" % s), sep=";", index=False)
    pd.set_option("display.width", 250)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_O].median().round(3).to_string())
    print("\nmedians per class (validation):\n" + OUTT["valid"].groupby("label")[COLS_O].median().round(3).to_string())
    CAS = ["Atropine_500µM_T0_R2(1)_280223_Frame_18_", "Acetylcholine_10000µM_T1h_R2(1)_060423_Frame_5_", "Methiothepin_5µM_T3h_R1(4)_140323_Frame_2_"]
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test")])
    print("\ncommented cases (test):\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_O].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors cp_ and ond_ | %.0f s" % (len(COLS_O), time.time() - t0))
    print("v3o tables:", os.path.join(V3O, "data"), "| v3n tables kept:", os.path.join(V3N, "data"))

if __name__ == "__main__":
    main()
