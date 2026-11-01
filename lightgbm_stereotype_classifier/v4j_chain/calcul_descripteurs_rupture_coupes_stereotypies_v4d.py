# -*- coding: utf-8 -*-
# calcul_descripteurs_rupture_coupes_stereotypies_v4d (22 October 2026). New program; the v3c to v4c programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Valentin's remark (cyan stroke on the corner of the L, Atropine_500uM_T24h_R1(2) F12): at the twist, the border breaks and the
# tissue changes sharply; comparing the cross sections along the axis makes the break immediately visible.
# Method: straightening the worm along its axis, as for C. elegans images (Peng, Long, Liu, Kim and Myers, 2008,
# Straightening Caenorhabditis elegans images, Bioinformatics 24(2):234-242, DOI 10.1093/bioinformatics/btm569) : en 60 points de
# the axis through the v4c tips (head tip to tail tip), cross section perpendicular to the axis sampled at 41 points
# (scipy.ndimage.map_coordinates, linear interpolation) on the grey-level image and on the mask. For each section: left
# and right borders (first exit from the mask starting at the axis), tissue profile (grey of the points inside the mask resampled to 21 points across
# the width, divided by the worm median). Then, at each point k of the axis, comparison of the 4 sections preceding k and the 4
# sections following it (windows of 6.7% of L, mean profiles, which rejects the noise of an isolated section), terminal 10% excluded, and
# search for the most marked break point along the axis (same principle as detecting a change point in a series,
# Killick, Fearnhead et Eckley, 2012, Optimal detection of changepoints with a linear computational cost, Journal of the American
# Statistical Association 107(500):1590-1598, arXiv:1101.1438, reduced here to a single break point by contrast with the median level).
#   d_tissu(k) = 1 - Pearson correlation of the mean tissue profile before k and the mean profile after k.
#   d_bord(k)  = (|jump of the mean left border| + |jump of the mean right border| between before and after k) / median worm width.
# Group ru_ (frames T-1, T0, T+1 registered on T0 as in v3i to v4c):
#   ru_tissu_t0      : max of d_tissu at T0 (sharpest tissue change along the axis).
#   ru_contraste_t0  : max of d_tissu / median of d_tissu at T0 (isolated break on an otherwise regular tissue).
#   ru_bord_deux_t0  : max over k of min(|left jump|, |right jump|) / median width at T0 (the border breaks on both sides).
#   ru_coloc_t0      : max over k of (d_tissu / its median) x (d_bord / its median) at T0 (tissue and border break at the same place).
#   ru_coloc_min     : minimum of ru_coloc over T-1, T0, T+1 (the break holds).
#   ru_contraste_min : minimum of ru_contraste over the three frames.
#   ru_int_saut_t0   : max of |mean grey after k - mean grey before k| / median worm grey at T0.
# Zero tolerance on NaN, nothing is imputed. Outputs: lightgbm_stereotype_v4d/data (v3t tables + 7 ru_ columns), cache
# _cache_v4d.pkl. Trial: V4D_ESSAI=N, with a plate of straightened worms (lightgbm_stereotype_v4d/essai_coupes_v4d.jpg).
# Central processor, one thread per process. Seed 42. Run from the FasterViT root (uses the v4c functions).
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from scipy.ndimage import map_coordinates, gaussian_filter1d
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_R = ["ru_tissu_t0", "ru_contraste_t0", "ru_bord_deux_t0", "ru_coloc_t0", "ru_coloc_min", "ru_contraste_min", "ru_int_saut_t0"]
NK, NL, NP, PAS, FEN = 60, 41, 21, 2, 4

if __name__ == "__main__":
    _src = open(os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    _C = {"__name__": "__main__", "__file__": os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_axe_pointes_stereotypies_v4c", "exec"), _C)
    format_384, voisins, NS_MOT, _objet, axe_pointes = (_C[k] for k in ("format_384", "voisins", "NS_MOT", "_objet", "axe_pointes"))
V3T = os.path.join(ROOT, "lightgbm_stereotype_v3t"); V4D = os.path.join(ROOT, "lightgbm_stereotype_v4d")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

def redresse(m, g):
    """Cross sections perpendicular to the tip-to-tip axis: grey (NK x NL), mask (NK x NL), sampling half-width."""
    Q, _ = axe_pointes(m)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; t = np.linspace(0, seg[-1], NK)
    A = np.c_[np.interp(t, seg, Q[:, 0]), np.interp(t, seg, Q[:, 1])]
    T = np.gradient(A, axis=0); T /= np.maximum(np.hypot(T[:, 0], T[:, 1])[:, None], 1e-6); N = np.c_[-T[:, 1], T[:, 0]]
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); R = 1.3 * float(dt.max()) + 2.0; s = np.linspace(-R, R, NL)
    Y = A[:, 0][:, None] + s[None, :] * N[:, 0][:, None]; X = A[:, 1][:, None] + s[None, :] * N[:, 1][:, None]
    G = map_coordinates(g.astype(np.float32), [Y, X], order=1, mode="constant", cval=0.0)
    M = map_coordinates(m.astype(np.float32), [Y, X], order=1, mode="constant", cval=0.0) > 0.5
    return G, M, s

def ruptures(G, M, s):
    c = NL // 2; gauche, droite, prof, moy = [], [], [], []
    for k in range(NK):
        row = M[k]
        if not row[c]:                                                             # axis outside the mask (smooth end): nearest point
            idx = np.nonzero(row)[0]; cc = int(idx[np.argmin(np.abs(idx - c))]) if len(idx) else c
        else: cc = c
        a = cc
        while a > 0 and row[a - 1]: a -= 1
        b = cc
        while b < NL - 1 and row[b + 1]: b += 1
        gauche.append(s[a]); droite.append(s[b])
        v = G[k, a:b + 1] if b > a else np.array([G[k, cc]] * 2)
        prof.append(np.interp(np.linspace(0, 1, NP), np.linspace(0, 1, len(v)), v)); moy.append(float(v.mean()))
    gauche, droite, prof, moy = np.array(gauche), np.array(droite), np.array(prof), np.array(moy)
    med = max(float(np.median(prof)), 1e-6); prof = prof / med; w = droite - gauche; wm = max(float(np.median(w)), 1e-6)
    e = int(round(0.10 * NK)); K = range(max(e, FEN), NK - max(e, FEN))
    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    av = lambda x, k: x[k - FEN:k].mean(0); ap = lambda x, k: x[k:k + FEN].mean(0)     # windows of FEN sections before and after k
    dti = np.array([1.0 - corr(av(prof, k), ap(prof, k)) for k in K])
    dgl = np.array([abs(av(gauche, k) - ap(gauche, k)) for k in K]) / wm; ddr = np.array([abs(av(droite, k) - ap(droite, k)) for k in K]) / wm
    dbo = dgl + ddr; din = np.array([abs(av(moy, k) - ap(moy, k)) for k in K]) / med
    mt = max(float(np.median(dti)), 1e-3); mb = max(float(np.median(dbo)), 1e-3)
    return dict(tissu=float(dti.max()), contraste=float(dti.max()) / mt, bord_deux=float(np.minimum(dgl, ddr).max()),
                coloc=float(((dti / mt) * (dbo / mb)).max()), int_saut=float(din.max()), courbe=dti, kmax=int(np.argmax((dti / mt) * (dbo / mb))) + max(e, FEN) - PAS // 2, k0=max(e, FEN))

def calcule(arg):
    sp, bn = arg
    try:
        pv, nx = voisins(bn)
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        R = [ruptures(*redresse(_objet(g), g)) for g in (gt, prep(pv), prep(nx))]
        out = dict(ru_tissu_t0=R[0]["tissu"], ru_contraste_t0=R[0]["contraste"], ru_bord_deux_t0=R[0]["bord_deux"], ru_coloc_t0=R[0]["coloc"],
                   ru_coloc_min=min(r["coloc"] for r in R), ru_contraste_min=min(r["contraste"] for r in R), ru_int_saut_t0=R[0]["int_saut"])
        return sp, bn, {k: float(out[k]) for k in COLS_R}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def calcule_essai(arg):
    try: return calcule(arg)
    except RuntimeError as e: return arg[0], arg[1], str(e)

CAS = ["Atropine_500µM_T24h_R1(2)_270223_Frame_12_", "NMDA_10000µM_T1h_R2(3)_190423_Frame_25_", "MK801_200µM_T3h_R2(2)_200423_Frame_17_",
       "MK801_200µM_T3h_R2(2)_200423_Frame_18_", "Apomorphine_5µM_T1h_R3(2)_030523_Frame_2_", "Fluoxetine_20µM_T3h_R3(5)_260423_Frame_23_",
       "Apomorphine_10µM_T3h_R1(3)_030523_Frame_18_", "Apomorphine_10µM_T3h_R1(3)_030523_Frame_25_", "Atropine_500µM_T0_R2(1)_280223_Frame_18_"]

def planche(noms, fichier):
    """Worm, axis and break section (cyan); on the right, straightened worm (sections as rows) and the d_tissu curve."""
    tu = []
    for nm in noms:
        try:
            g = format_384(nm); m = _objet(g); G, M, s = redresse(m, g); r = ruptures(G, M, s); Q, _ = axe_pointes(m)
        except Exception as e:
            print("plate:", nm, e); continue
        im = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.polylines(im, [Q[:, ::-1].astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; t = np.linspace(0, seg[-1], NK)
        A = np.c_[np.interp(t, seg, Q[:, 0]), np.interp(t, seg, Q[:, 1])]; k = r["kmax"] + PAS // 2
        Tg = A[min(NK - 1, k + 1)] - A[max(0, k - 1)]; Tg /= max(np.hypot(*Tg), 1e-6); Nn = np.array([-Tg[1], Tg[0]])
        p1, p2 = A[k] + s[0] * Nn, A[k] + s[-1] * Nn; cv2.line(im, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), (255, 255, 0), 2)
        cv2.putText(im, "coloc %.1f contrast %.1f edges %.2f" % (r["coloc"], r["contraste"], r["bord_deux"]), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(im, nm[:44].replace("µ", "u").replace("é", "e"), (4, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        st = np.clip(G * M, 0, 255).astype(np.uint8); st = cv2.resize(cv2.cvtColor(st, cv2.COLOR_GRAY2BGR), (120, 384), interpolation=cv2.INTER_NEAREST)
        y = int((k + 0.5) / NK * 384); cv2.line(st, (0, y), (119, y), (255, 255, 0), 1)
        cv = np.zeros((384, 80, 3), np.uint8); e = int(round(0.10 * NK)); d = r["courbe"]; dm = max(float(d.max()), 1e-6)
        pts = [(int(5 + 70 * v / dm), int((r['k0'] + i) / NK * 384)) for i, v in enumerate(d)]
        cv2.polylines(cv, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (0, 200, 255), 1)
        tu.append(np.hstack([im, st, cv]))
    while len(tu) % 2: tu.append(np.zeros_like(tu[0]))
    cv2.imwrite(fichier, np.vstack([np.hstack(tu[i:i + 2]) for i in range(0, len(tu), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])

def main():
    t0 = time.time(); ESSAI = int(os.environ.get("V4D_ESSAI", "0"))
    print("v4d between-section break descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
          "| central processor, %d processes%s" % (N_WORKERS, " | TRIAL %d per class" % ESSAI if ESSAI else ""), flush=True)
    assert os.path.isdir(os.path.join(V3T, "data")), "v3t tables missing"
    if not ESSAI: assert not os.path.exists(os.path.join(V4D, "data")), "lightgbm_stereotype_v4d/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3T, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    os.makedirs(V4D, exist_ok=True)
    if ESSAI:
        taches = []
        for s in ("valid", "test", "golden"):
            d = BASE[s]
            if s != "golden": taches += [(s, bn) for bn in d.groupby("label").head(ESSAI).image]
            taches += [(s, bn) for bn in d.image if any(c in bn for c in CAS) and (s, bn) not in taches]
        tr = BASE["train"]; ex = list(tr[tr.label == "C-shape"].image.iloc[[0, 300]]) + list(tr[tr.label == "Screw-like"].image.iloc[[10, 200]]) + list(tr[tr.label == "Wildtype"].image.iloc[[5]])
        planche([bn for s, bn in taches if any(c in bn for c in CAS)][:9] + ex, os.path.join(V4D, "essai_coupes_v4d.jpg"))
    else: taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_rupture_coupes_stereotypies_v4d.py"), "rb").read()
                         + open(os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V4D, "_cache_v4d%s.pkl" % ("_essai" if ESSAI else "")); RES = {}; SAUT = []
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
        assert np.isfinite(E[COLS_R].to_numpy(float)).all(), "non-finite value in the trial"
        E.to_csv(os.path.join(V4D, "essai_v4d.csv"), sep=";", index=False)
        print("\nmeans per class (trial):\n" + E.groupby("label")[COLS_R].mean().round(3).to_string())
        print("\nfailure cases:\n" + E[E.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_R].round(3).to_string(index=False))
        print("TRIAL done, 0 non-finite value, no table written | %.0f s" % (time.time() - t0)); return
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in COLS_R: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_R[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_R].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V4D, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v4d/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V4D, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V4D, "data", "features_%s.csv" % s), sep=";", index=False)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_R].median().round(3).to_string())
    print("\nmeans per class (training):\n" + OUTT["train"].groupby("label")[COLS_R].mean().round(3).to_string())
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test", "golden")])
    print("\nfailure cases:\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["partition", "image", "label"] + COLS_R].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors ru_ | %.0f s" % (len(COLS_R), time.time() - t0))
    print("v4d tables:", os.path.join(V4D, "data"), "| v3t tables kept:", os.path.join(V3T, "data"))

if __name__ == "__main__":
    main()
