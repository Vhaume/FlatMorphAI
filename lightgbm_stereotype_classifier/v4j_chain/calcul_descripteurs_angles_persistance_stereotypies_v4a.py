# -*- coding: utf-8 -*-
# calcul_descripteurs_angles_persistance_stereotypies_v4a (22 October 2026). New program; the v3c to v3z programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Valentin's remarks of 22 October on the failures of the final v3t model (Claude outputs/echecs_v3t):
#   Atropine_500uM_T24h_R1(2)_270223 F12 (Screw-like read as Wildtype): two sharp turns of opposite directions, angular,
#   held over the three frames, and at T+1 an hourglass twist at that spot.
#   The other failures are mostly false C-shapes: the worm moves a lot without holding a shape for long.
# Frames T-1, T0, T+1 only (v3f neighbours, 384 frames registered on T0 as in v3i to v3z). Axis and profiles from v3z
# (v3g geodesic axis, 40 slices, extremities oriented on T0), signed curvature = angle between successive tangents of the axis
# smoothed by a Gaussian of sigma 2 slices (5% of L, the per-slice medial axis being noisy to the pixel), terminal 10% excluded.
#
# Group ag_ (opposite sharp angles, held):
#   Outer contour of the mask (cv2.findContours), resampled to 160 points at constant arc step, smoothed (circular Gaussian
#   sigma 1.5 point), oriented counter-clockwise (signed area). Angle of a corner = sum of the turning angles over 5 points
#   (3% of the perimeter), positive for a salient corner, negative for a re-entrant corner. Facing corners: a salient corner and a
#   re-entrant corner at most 1.5 median worm widths apart (the two borders of the body at the same place, the angular fold described by
#   Valentin), strength = the smaller of the two angles in absolute value (radians).
#   ag_double_t0  : strength of the best pair of facing corners at T0.
#   ag_double_min : minimum over T-1, T0, T+1 of that strength (the angles hold over the three frames).
#   ag_pince_min  : minimum over the three frames of the distance between the two corners of the pair / median width (hourglass: low).
#   ag_franc_min  : minimum over the three frames of the share of the total axis turning concentrated on 3 slices (7.5% of L):
#                   sharp turn (near 1) against the regular arc of a C (low).
#   ag_opp_min    : minimum over the three frames of the strength of the pair of axis turns of opposite sign, 2 to 12 slices apart.
# Group pf_ (shape persistence, false C-shapes that move):
#   pf_c_min      : minimum over the three frames of |sum of the angles| / sum of the |angles| (1: curvature of a single sign, a C).
#   pf_c_rng      : maximum minus minimum of that quantity over the three frames (a C that forms and unforms: high).
#   pf_courb_corr : minimum over T-1 and T+1 of the Pearson correlation, with no shift, of the curvature profile with that of T0.
#   pf_courb_ecart: mean over T-1 and T+1 of the squared difference of the curvature profiles / squared deviation of the T0 profile.
#   pf_bend_rng   : maximum minus minimum of the compactness (1 - chord / L) over the three frames.
# Correlations: constant profile -> 0 (never NaN). Zero tolerance on NaN, nothing is imputed.
# Outputs: lightgbm_stereotype_v4a/data/features_{train,valid,test,golden}.csv (v3t tables + 10 columns), cache _cache_v4a.pkl.
# Trial: V4A_ESSAI=N computes N frames per validation and test class plus the v3t failure cases, writes
# lightgbm_stereotype_v4a/essai_v4a.csv and writes no table.
# Central processor, one thread per process (multiprocessing). Seed 42. Run from the FasterViT root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_A = ["ag_double_t0", "ag_double_min", "ag_pince_min", "ag_franc_min", "ag_opp_min", "pf_c_min", "pf_c_rng", "pf_courb_corr", "pf_courb_ecart", "pf_bend_rng"]
W3, DMIN, DMAX = 3, 2, 12

if __name__ == "__main__":
    _src = open(os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    _L = {"__name__": "v3n_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_mouvement_largeur_stereotypies_v3n", "exec"), _L)
    format_384, voisins, NS_MOT, _objet, oriente = (_L[k] for k in ("format_384", "voisins", "NS_MOT", "_objet", "oriente"))
    _z = open(os.path.join(ROOT, "calcul_descripteurs_torsion_onde_stereotypies_v3z.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _z
    _Z = {"__name__": "v3z_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_torsion_onde_stereotypies_v3z.py")}
    exec(compile(_z.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_torsion_onde_stereotypies_v3z", "exec"), _Z)
    _Z["oriente"] = oriente
    axe_profil, corr, NB = _Z["axe_profil"], _Z["corr"], _Z["NB"]
V3T = os.path.join(ROOT, "lightgbm_stereotype_v3t"); V4A = os.path.join(ROOT, "lightgbm_stereotype_v4a")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

SIG_AXE = 2.0
def courbure(x, inverse):
    """Signed curvature of the axis (angle between successive tangents), axis smoothed sigma SIG_AXE slices, terminal 10 % excluded."""
    from scipy.ndimage import gaussian_filter1d
    P = gaussian_filter1d(x["P"][::-1] if inverse else x["P"], SIG_AXE, axis=0, mode="nearest"); T = np.diff(P, axis=0)
    ang = np.arctan2(T[:, 0], T[:, 1]); d = np.diff(ang); d = (d + np.pi) % (2 * np.pi) - np.pi
    e = int(round(0.10 * NB)); return d[e:len(d) - e]

def paire_opposee(x, k):
    """Strongest pair of opposite-sign turns on the curvature profile k of axis x."""
    v = np.convolve(k, np.ones(W3), mode="same"); e = int(round(0.10 * NB)); best, bi, bj = 0.0, -1, -1
    for i in range(len(v)):
        for j in range(i + DMIN, min(len(v), i + DMAX + 1)):
            if v[i] * v[j] < 0:
                f = min(abs(v[i]), abs(v[j]))
                if f > best: best, bi, bj = f, i, j
    if bi < 0: return 0.0, 0.0, 1.0
    w = x["w"] / max(float(np.median(x["w"][e:NB - e])), 1e-6)
    a, b = bi + e + 1 - 1, bj + e + 1 + 1                    # axis slices covered by the pair, one slice of margin
    return float(best), (bj - bi) / NB, float(w[max(0, a):min(NB, b + 1)].min())

def coins_face(m, wmed, n=160, sig=1.5, win=5):
    """Facing salient / re-entrant corner pair on the outer contour of mask m: (strength, distance / median width)."""
    from scipy.ndimage import gaussian_filter1d
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    if len(c) < 20: return 0.0, 3.0
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T))]; t = np.linspace(0, seg[-1], n, endpoint=False)
    q = np.c_[np.interp(t, seg, np.r_[c[:, 0], c[0, 0]]), np.interp(t, seg, np.r_[c[:, 1], c[0, 1]])]
    q = gaussian_filter1d(q, sig, axis=0, mode="wrap")
    if 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) < 0: q = q[::-1]
    T = np.roll(q, -1, axis=0) - q; ang = np.arctan2(T[:, 1], T[:, 0]); d = np.diff(np.r_[ang, ang[0]]); d = (d + np.pi) % (2 * np.pi) - np.pi
    s_ = np.convolve(np.r_[d[-(win // 2):], d, d[:win // 2]], np.ones(win), mode="valid")       # angle of each corner, circular window
    conv = [i for i in range(n) if s_[i] > 0 and s_[i] >= s_[i - 1] and s_[i] >= s_[(i + 1) % n]]
    conc = [i for i in range(n) if s_[i] < 0 and s_[i] <= s_[i - 1] and s_[i] <= s_[(i + 1) % n]]
    best, dist = 0.0, 3.0
    for i in conv:
        for j in conc:
            dd = float(np.hypot(*(q[i] - q[j]))) / max(wmed, 1e-6)
            if dd <= 1.5:
                f = min(float(s_[i]), float(-s_[j]))
                if f > best: best, dist = f, dd
    return best, dist

def franc(k):
    v = np.abs(np.convolve(k, np.ones(W3), mode="same")); s_ = float(np.abs(k).sum()); return float(v.max()) / s_ if s_ > 1e-9 else 0.0

def c_unique(k):
    s = float(np.abs(k).sum()); return abs(float(k.sum())) / s if s > 1e-9 else 0.0

def calcule(arg):
    sp, bn = arg
    try:
        pv, nx = voisins(bn)
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        G = [gt, prep(pv), prep(nx)]; X = [axe_profil(_objet(g), g) for g in G]; x0 = X[0]
        K = [courbure(x0, False)] + [courbure(x, not oriente(x0, x)) for x in X[1:]]
        A = [paire_opposee(x, k) for x, k in zip(X, K)]
        M = [_objet(g) for g in G]; e = int(round(0.15 * NB))
        CF = [coins_face(mm, float(np.median(x["w"][e:NB - e]))) for mm, x in zip(M, X)]
        out = dict(ag_double_t0=CF[0][0], ag_double_min=min(c[0] for c in CF), ag_pince_min=min(c[1] for c in CF),
                   ag_franc_min=min(franc(k) for k in K), ag_opp_min=min(a[0] for a in A))
        C = [c_unique(k) for k in K]; out["pf_c_min"] = min(C); out["pf_c_rng"] = max(C) - min(C)
        n = min(len(k) for k in K); K = [k[:n] for k in K]; r0 = float(np.sqrt(np.mean(K[0] ** 2))) + 1e-6
        out["pf_courb_corr"] = min(corr(K[0], K[1]), corr(K[0], K[2]))
        out["pf_courb_ecart"] = float(np.mean([np.sqrt(np.mean((k - K[0]) ** 2)) / r0 for k in K[1:]]))
        B = [x["bend"] for x in X]; out["pf_bend_rng"] = max(B) - min(B)
        return sp, bn, {k: float(out[k]) for k in COLS_A}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def calcule_essai(arg):
    """Trial only: a frame whose neighbours are not visible from this machine is skipped and counted."""
    try: return calcule(arg)
    except RuntimeError as e: return arg[0], arg[1], str(e)

CAS = ["Atropine_500µM_T24h_R1(2)_270223_Frame_12_", "MK801_200µM_T3h_R2(2)_200423_Frame_17_", "MK801_200µM_T3h_R2(2)_200423_Frame_18_",
       "Atropine_500µM_T0_R2(1)_280223_Frame_18_", "Apomorphine_5µM_T1h_R3(2)_030523_Frame_2_", "Fluoxetine_20µM_T3h_R3(5)_260423_Frame_23_",
       "Apomorphine_10µM_T3h_R1(3)_030523_Frame_18_", "Apomorphine_10µM_T3h_R1(3)_030523_Frame_25_", "Apomorphine_10µM_T3h_R1(3)_030523_Frame_26_",
       "Atropine_500µM_T24h_R2(5)_280223_Frame_4_", "Methiothepin_10µM_T24h_R1(3)_140323_Frame_10_", "Halopéridol_1µM_T24h_R1(1)_230523_Frame_15_"]

def main():
    t0 = time.time(); ESSAI = int(os.environ.get("V4A_ESSAI", "0"))
    print("v4a opposite-angle and shape-persistence descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
          "| central processor, %d processes%s" % (N_WORKERS, " | TRIAL %d per class" % ESSAI if ESSAI else ""), flush=True)
    assert os.path.isdir(os.path.join(V3T, "data")), "v3t tables missing"
    if not ESSAI: assert not os.path.exists(os.path.join(V4A, "data")), "lightgbm_stereotype_v4a/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3T, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    if ESSAI:
        taches = []
        for s in ("valid", "test", "golden"):
            d = BASE[s]
            if s != "golden": taches += [(s, bn) for bn in d.groupby("label").head(ESSAI).image]
            taches += [(s, bn) for bn in d.image if any(c in bn for c in CAS) and (s, bn) not in taches]
    else: taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V4A, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_angles_persistance_stereotypies_v4a.py"), "rb").read()
                         + open(os.path.join(ROOT, "calcul_descripteurs_torsion_onde_stereotypies_v3z.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V4A, "_cache_v4a%s.pkl" % ("_essai" if ESSAI else "")); RES = {}; SAUT = []
    if os.path.exists(CACHE):
        c = pickle.load(open(CACHE, "rb")); RES = c["res"] if c.get("sig") == SIG else {}
    reste = [t for t in taches if t not in RES]; print("cache: %d, to compute: %d" % (len(RES), len(reste)), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, (s, bn, out) in enumerate(pool.imap_unordered(calcule_essai if ESSAI else calcule, reste, chunksize=2), 1):
            if isinstance(out, str): SAUT.append(out); continue
            RES[(s, bn)] = out
            if i % 300 == 0: pickle.dump(dict(sig=SIG, res=RES), open(CACHE, "wb")); print("  %d / %d | %.0f s" % (i, len(reste), time.time() - t0), flush=True)
    pickle.dump(dict(sig=SIG, res=RES), open(CACHE, "wb"))
    pd.set_option("display.width", 260)
    if ESSAI:
        if SAUT: print("trial: %d frames skipped (neighbours not visible from this machine), e.g. %s" % (len(SAUT), SAUT[0][:160]), flush=True)
        E = pd.DataFrame([dict(partition=s, image=bn, label=BASE[s].set_index("image").label[bn], **RES[(s, bn)]) for s, bn in taches if (s, bn) in RES])
        assert np.isfinite(E[COLS_A].to_numpy(float)).all(), "non-finite value in the trial"
        E.to_csv(os.path.join(V4A, "essai_v4a.csv"), sep=";", index=False)
        print("\nmeans per class (trial):\n" + E.groupby("label")[COLS_A].mean().round(3).to_string())
        print("\nv3t failure cases:\n" + E[E.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_A].round(3).to_string(index=False))
        print("TRIAL done, 0 non-finite value, no table written | %.0f s" % (time.time() - t0)); return
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in COLS_A: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_A[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_A].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V4A, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v4a/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V4A, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V4A, "data", "features_%s.csv" % s), sep=";", index=False)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_A].median().round(3).to_string())
    print("\nmeans per class (training):\n" + OUTT["train"].groupby("label")[COLS_A].mean().round(3).to_string())
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test", "golden")])
    print("\nv3t failure cases:\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["partition", "image", "label"] + COLS_A].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors ag_ and pf_ | %.0f s" % (len(COLS_A), time.time() - t0))
    print("v4a tables:", os.path.join(V4A, "data"), "| v3t tables kept:", os.path.join(V3T, "data"))

if __name__ == "__main__":
    main()
