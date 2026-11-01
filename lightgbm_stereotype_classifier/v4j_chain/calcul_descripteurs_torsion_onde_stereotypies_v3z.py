# -*- coding: utf-8 -*-
# calcul_descripteurs_torsion_onde_stereotypies_v3z (22 October 2026). New program; the v3c to v3t programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Origin: LightGBM contributions (pred_contrib) of the 13 test errors of the final v3t model (seed 42).
#   - 5 errors (MK801 F17 and F18 Screw-like, Atropine_500 T0 F18 Wildtype, Fluoxetine_20 F23 Snake-like, Atropine_500 T24h F4):
#     elong_T0 (contour elongation) pushes from +1.2 to +2.1 towards C-shape. The model reads "compact therefore C-shape" without knowing whether the
#     worm is curved in a plane, twisted on itself or coiled. Valentin's remark (Apomorphine_5 F3): the thin part is the
#     twisted part. Group ta_: what happens along the axis inside the silhouette.
#   - 4 Wildtype <-> Snake-like errors driven by the motion (hxw_reversals, tmob_mid, opt_strain). A Snake-like propagates a
#     curvature wave along the body; a Wildtype that extends changes length. Group ow_: shift of the curvature wave.
# Frames T-1, T0, T+1 only (v3f neighbours, 384 frames registered on T0 as in v3i to v3t).
# Axis: v3g geodesic axis (40 slices, medial points P, width w = 2 x border distance), extremities oriented on T0.
#
# Group ta_ (twist along the axis), computed on each frame then the median of the three frames, terminal 15% excluded:
#   width profile W(k) = w(k) / median width, intensity profile I(k) = median grey of the core pixels of slice k
#   (border distance >= half the slice maximum) / median of the profile, smoothed by a Gaussian of sigma 1 slice.
#   ta_saut_w    : largest difference |W(k+2) - W(k)| (abrupt narrowing or widening, 5% of L).
#   ta_saut_i    : largest difference |I(k+2) - I(k)| (abrupt change of shade along the axis).
#   ta_coloc     : max_k of (1 - W(k)) x |I(k+1) - I(k-1)| for W(k) < 1: narrow neck AND change of shade at the same place.
#   ta_bandes    : number of crossings of I(k) on either side of 1 +- 0.05 (successive pale and dark bands).
#   ta_coloc_x_compact : ta_coloc x compactness (1 - chord / L of T0, the v3g "bend" quantity): the twist is read only on
#                        compact worms, where elong_T0 decides alone today.
# Group ow_ (curvature wave), between T-1 and T0 then T0 and T+1:
#   signed curvature k(s) of the axis (angle between successive tangents, axis smoothed sigma 1 slice), ends aligned on T0.
#   For each pair, Pearson correlation of the two curvature profiles shifted from -8 to +8 slices (20% of L).
#   ow_decal   : mean of the |optimal shift| / 40 (fraction of L travelled by the wave).
#   ow_gain    : mean of (correlation at the optimal shift - correlation with no shift) (a propagating wave: high gain).
#   ow_sens    : 1 if both shifts have the same non-zero sign (continuous propagation in one direction), 0 otherwise.
#   ow_etir    : max |log(L neighbour / L T0)| (change of geodesic length).
#   ow_onde_etir : ow_decal / (ow_etir + 0.02) (wave moving without stretching: high; stretching without a wave: low).
# Correlations: constant profile -> correlation 0 (never NaN). Zero tolerance on NaN, nothing is imputed.
# Outputs: lightgbm_stereotype_v3z/data/features_{train,valid,test,golden}.csv (v3t tables + 10 columns), cache _cache_v3z.pkl.
# Trial: V3Z_ESSAI=N computes N frames per validation and test class plus the commented cases, writes
# lightgbm_stereotype_v3z/essai_v3z.csv and writes no table.
# Central processor, one thread per process (multiprocessing). Seed 42. Run from the FasterViT root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_Z = ["ta_saut_w", "ta_saut_i", "ta_coloc", "ta_bandes", "ta_coloc_x_compact", "ow_decal", "ow_gain", "ow_sens", "ow_etir", "ow_onde_etir"]
NB, LAG = 40, 8

if __name__ == "__main__":
    _src = open(os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    _L = {"__name__": "v3n_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_mouvement_largeur_stereotypies_v3n", "exec"), _L)
    format_384, voisins, NS_MOT, _objet, oriente = (_L[k] for k in ("format_384", "voisins", "NS_MOT", "_objet", "oriente"))
V3T = os.path.join(ROOT, "lightgbm_stereotype_v3t"); V3Z = os.path.join(ROOT, "lightgbm_stereotype_v3z")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

def axe_profil(m, g):
    """Geodesic axis (same construction as the v3g axis) plus the intensity profile of the core of each slice."""
    from skimage.graph import MCP_Geometric
    def geod(dep):
        d, _ = MCP_Geometric(np.where(m, 1.0, np.inf)).find_costs([dep]); d = np.asarray(d, float); d[~m] = np.inf; return d
    ys, xs = np.nonzero(m); d = geod((ys[0], xs[0]))
    a = np.unravel_index(int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape); d = geod(a)
    b = np.unravel_index(int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape); L = float(d[b]); assert L > 10, "axis too short"
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); q = np.full(m.shape, -1); q[m] = np.minimum((d[m] / L * NB).astype(int), NB - 1)
    P = np.zeros((NB, 2)); w = np.zeros(NB); I = np.zeros(NB); ok = np.zeros(NB, bool)
    for k in range(NB):
        yy, xx = np.nonzero(q == k)
        if len(yy):
            j = int(np.argmax(dt[yy, xx])); P[k] = (yy[j], xx[j]); w[k] = 2.0 * dt[yy[j], xx[j]]
            c = dt[yy, xx] >= 0.5 * dt[yy, xx].max(); I[k] = float(np.median(g[yy[c], xx[c]])); ok[k] = True
        else: P[k] = P[k - 1] if k else a; w[k] = w[k - 1] if k else 1.0
    assert ok.sum() >= NB // 2, "too many empty slices"
    I = np.interp(np.arange(NB), np.where(ok)[0], I[ok])       # empty slice (very short axis): value of the neighbouring measured slices
    return dict(a=np.array(a, float), b=np.array(b, float), L=L, P=P, w=w, I=I, bend=1.0 - float(np.hypot(*(np.array(a, float) - np.array(b, float)))) / L)

def torsion(x):
    e = int(round(0.15 * NB)); W = x["w"][e:NB - e]; I = x["I"][e:NB - e]
    W = gaussian_filter1d(W / max(float(np.median(W)), 1e-6), 1.0, mode="nearest"); I = gaussian_filter1d(I / max(float(np.median(I)), 1e-6), 1.0, mode="nearest")
    sw = float(np.abs(W[2:] - W[:-2]).max()); si = float(np.abs(I[2:] - I[:-2]).max())
    co = max([max(0.0, 1.0 - W[k]) * abs(I[k + 1] - I[k - 1]) for k in range(1, len(W) - 1)] + [0.0])
    etat = np.where(I > 1.05, 1, np.where(I < 0.95, -1, 0)); etat = etat[etat != 0]
    bandes = float((etat[1:] != etat[:-1]).sum()) if len(etat) > 1 else 0.0
    return dict(ta_saut_w=sw, ta_saut_i=si, ta_coloc=co, ta_bandes=bandes)

def courbure(x, inverse):
    P = gaussian_filter1d(x["P"][::-1] if inverse else x["P"], 1.0, axis=0, mode="nearest"); T = np.diff(P, axis=0)
    ang = np.arctan2(T[:, 0], T[:, 1]); d = np.diff(ang); d = (d + np.pi) % (2 * np.pi) - np.pi
    e = int(round(0.10 * NB)); return d[e:len(d) - e]

def corr(a, b):
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def onde(x0, xs):
    k0 = courbure(x0, False); dec, gain, sg, et = [], [], [], []
    for x in xs:
        k1 = courbure(x, not oriente(x0, x)); n = len(k0); best, bl = -2.0, 0
        for l in range(-LAG, LAG + 1):
            a = k0[max(0, l):n + min(0, l)]; b = k1[max(0, -l):n - max(0, l)]; c = corr(a, b)
            if c > best + 1e-12 or (abs(c - best) <= 1e-12 and abs(l) < abs(bl)): best, bl = c, l
        dec.append(abs(bl) / NB); gain.append(best - corr(k0, k1)); sg.append(np.sign(bl)); et.append(abs(np.log(max(x["L"], 1e-3) / max(x0["L"], 1e-3))))
    d = float(np.mean(dec)); e = float(max(et))
    return dict(ow_decal=d, ow_gain=float(np.mean(gain)), ow_sens=float(sg[0] != 0 and sg[0] == sg[1]), ow_etir=e, ow_onde_etir=d / (e + 0.02))

def calcule(arg):
    sp, bn = arg
    try:
        pv, nx = voisins(bn)
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        G = [gt, prep(pv), prep(nx)]; X = [axe_profil(_objet(g), g) for g in G]
        T = [torsion(x) for x in X]; out = {k: float(np.median([t[k] for t in T])) for k in T[0]}
        out["ta_coloc_x_compact"] = out["ta_coloc"] * X[0]["bend"]; out.update(onde(X[0], X[1:]))
        return sp, bn, {k: float(out[k]) for k in COLS_Z}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def calcule_essai(arg):
    """Trial only: a frame whose neighbours are not visible from this machine is skipped and counted."""
    try: return calcule(arg)
    except RuntimeError as e: return arg[0], arg[1], str(e)

CAS = ["MK801_200µM_T3h_R2(2)_200423_Frame_17_", "MK801_200µM_T3h_R2(2)_200423_Frame_18_", "Atropine_500µM_T0_R2(1)_280223_Frame_18_",
       "Fluoxetine_20µM_T3h_R3(5)_260423_Frame_23_", "Atropine_500µM_T24h_R2(5)_280223_Frame_4_", "Atropine_500µM_T24h_R2(5)_280223_Frame_2_",
       "Apomorphine_20µM_T0_R1(5)_030523_Frame_28_", "Apomorphine_0µM_T1h_R3(1)_030523_Frame_26_", "Halopéridol_1µM_T24h_R1(1)_230523_Frame_15_",
       "Methiothepin_10µM_T24h_R1(3)_140323_Frame_10_"]

def main():
    t0 = time.time(); ESSAI = int(os.environ.get("V3Z_ESSAI", "0"))
    print("v3z twist and wave descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "| central processor, %d processes%s"
          % (N_WORKERS, " | TRIAL %d per class" % ESSAI if ESSAI else ""), flush=True)
    assert os.path.isdir(os.path.join(V3T, "data")), "v3t tables missing"
    if not ESSAI: assert not os.path.exists(os.path.join(V3Z, "data")), "lightgbm_stereotype_v3z/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3T, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    if ESSAI:
        taches = []
        for s in ("valid", "test"):
            d = BASE[s]; taches += [(s, bn) for bn in d.groupby("label").head(ESSAI).image]
            taches += [(s, bn) for bn in d.image if any(c in bn for c in CAS) and (s, bn) not in taches]
    else: taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V3Z, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_torsion_onde_stereotypies_v3z.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V3Z, "_cache_v3z%s.pkl" % ("_essai" if ESSAI else "")); RES = {}; SAUT = []
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
    pd.set_option("display.width", 250)
    if ESSAI:
        if SAUT: print("trial: %d frames skipped (neighbours not visible from this machine), e.g. %s" % (len(SAUT), SAUT[0][:160]), flush=True)
        E = pd.DataFrame([dict(partition=s, image=bn, label=BASE[s].set_index("image").label[bn], **RES[(s, bn)]) for s, bn in taches if (s, bn) in RES])
        assert np.isfinite(E[COLS_Z].to_numpy(float)).all(), "non-finite value in the trial"
        E.to_csv(os.path.join(V3Z, "essai_v3z.csv"), sep=";", index=False)
        print("\nmeans per class (trial):\n" + E.groupby("label")[COLS_Z].mean().round(3).to_string())
        print("\nv3t error cases:\n" + E[E.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_Z].round(3).to_string(index=False))
        print("TRIAL done, 0 non-finite value, no table written | %.0f s" % (time.time() - t0)); return
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in COLS_Z: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_Z[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_Z].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V3Z, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v3z/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V3Z, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V3Z, "data", "features_%s.csv" % s), sep=";", index=False)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_Z].median().round(3).to_string())
    print("\nmeans per class (training):\n" + OUTT["train"].groupby("label")[COLS_Z].mean().round(3).to_string())
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test")])
    print("\nv3t error cases:\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["partition", "image", "label"] + COLS_Z].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors ta_ and ow_ | %.0f s" % (len(COLS_Z), time.time() - t0))
    print("v3z tables:", os.path.join(V3Z, "data"), "| v3t tables kept:", os.path.join(V3T, "data"))

if __name__ == "__main__":
    main()
