# -*- coding: utf-8 -*-
# calcul_descripteurs_axe_pointes_stereotypies_v4c (22 October 2026). Copy of calcul_descripteurs_squelette_robuste_stereotypies_v4b
# (unmodified), the same 12 sq_ and l_ descriptors (definitions in the v4b header), the same frames T-1, T0, T+1. Only change: the axis.
# Valentin's remark on the v4b failures (MK801_200uM_T3h_R2(2) F16 to F19, Atropine_500uM_T0_R2(1) F18): the skeleton cuts or
# coils inside the body; the axis must run from the head tip to the tail tip following the body (cyan stroke).
# Tip-to-tip axis: tip candidates = the 8 most marked salient corners of the outer contour (200 points, smoothed sigma 2,
# angle over 7 points, at least 0.3 radian, 10 points apart at minimum); between each pair of candidates, a minimum-cost path
# inside the mask with a cost 1 / (border distance / maximum distance + 0.05)^2 that keeps the path in the middle of the body
# (skimage.graph.MCP_Geometric); the axis is the longest of those paths (the head and the tail are the two tips furthest apart
# along the body). 40 points at constant arc step, smoothing sigma 1.5 point. A hole in the mask (body touching or overlapping itself)
# is gone around by the path instead of being crossed.
# Angle of each border for the L shape: the most marked angle at +-6 contour points (out of 160) from the point reached by the normal,
# the centred axis rounding off the inner corner of an L.
# Limitation: an end lying ON the body (tail coiled over it, Atropine_500uM_T0 F18) has no contour of its own and cannot be
# followed from the mask alone.
# Outputs: lightgbm_stereotype_v4c/data (v3t tables + 12 columns), cache _cache_v4c.pkl. Trial: V4C_ESSAI=N.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_B = ["sq_boucle", "sq_c_min", "sq_c_rng", "sq_courb_corr", "sq_courb_ecart", "sq_franc_min", "sq_opp_min",
          "l_deux_cotes_t0", "l_deux_cotes_min", "l_symetrie_t0", "l_bras_min", "l_score_min"]
NB, W3, DMIN, DMAX = 40, 3, 2, 12

if __name__ == "__main__":
    _src = open(os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    _L = {"__name__": "v3n_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_mouvement_largeur_stereotypies_v3n", "exec"), _L)
    format_384, voisins, NS_MOT, _objet, oriente = (_L[k] for k in ("format_384", "voisins", "NS_MOT", "_objet", "oriente"))
V3T = os.path.join(ROOT, "lightgbm_stereotype_v3t"); V4C = os.path.join(ROOT, "lightgbm_stereotype_v4c")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

def pointes(m, n=200, sig=2.0, win=7, kmax=8, sep=10):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)[:, 0, ::-1].astype(float)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T))]; t = np.linspace(0, seg[-1], n, endpoint=False)
    q = np.c_[np.interp(t, seg, np.r_[c[:, 0], c[0, 0]]), np.interp(t, seg, np.r_[c[:, 1], c[0, 1]])]; q = gaussian_filter1d(q, sig, axis=0, mode="wrap")
    T = np.roll(q, -1, axis=0) - q; ang = np.arctan2(T[:, 0], T[:, 1]); d = np.diff(np.r_[ang, ang[0]]); d = (d + np.pi) % (2 * np.pi) - np.pi
    if d.sum() < 0: d = -d
    s = np.convolve(np.r_[d[-(win // 2):], d, d[:win // 2]], np.ones(win), mode="valid")
    order = np.argsort(-s); keep = []
    for i in order:
        if s[i] <= 0.3 or len(keep) >= kmax: break
        if all(min(abs(i - j), n - abs(i - j)) >= sep for j in keep): keep.append(int(i))
    return q[keep], s[keep]
def axe_pointes(m):
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); cost = np.where(m, 1.0 / (dt / max(dt.max(), 1e-6) + 0.05) ** 2, np.inf)
    P, S = pointes(m); H, W = m.shape
    def dedans(p):
        y, x = int(np.clip(round(p[0]), 0, H - 1)), int(np.clip(round(p[1]), 0, W - 1))
        if m[y, x]: return (y, x)
        ys, xs = np.nonzero(m); k = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2)); return (int(ys[k]), int(xs[k]))
    pts = [dedans(p) for p in P]; best = None
    for i in range(len(pts)):
        mcp = MCP_Geometric(cost); mcp.find_costs([pts[i]])
        for j in range(i + 1, len(pts)):
            path = np.array(mcp.traceback(pts[j]), float)
            L = float(np.sum(np.hypot(*np.diff(path, axis=0).T))) if len(path) > 1 else 0.0
            if best is None or L > best[0]: best = (L, path)
    assert best is not None and best[0] > 10, "no pair of tips"
    path = best[1]; seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]; tt = np.linspace(0, seg[-1], NB)
    Q = np.c_[np.interp(tt, seg, path[:, 0]), np.interp(tt, seg, path[:, 1])]; Q = gaussian_filter1d(Q, 1.5, axis=0, mode="nearest")
    return Q, P

def axe_robuste(m):
    """Axis from the head tip to the tail tip (see the header), in the format of the v3g to v4b programs."""
    Q, _ = axe_pointes(m); dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    cs, hi = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    boucle = hi is not None and any(hi[0][i][3] >= 0 and cv2.contourArea(cs[i]) >= 5 for i in range(len(cs)))
    w = 2.0 * dt[np.clip(np.rint(Q[:, 0]).astype(int), 0, m.shape[0] - 1), np.clip(np.rint(Q[:, 1]).astype(int), 0, m.shape[1] - 1)]
    Lg = float(np.sum(np.hypot(*np.diff(Q, axis=0).T)))
    return dict(a=Q[0].copy(), b=Q[-1].copy(), L=Lg, P=Q, w=w, boucle=float(boucle), bend=1.0 - float(np.hypot(*(Q[0] - Q[-1]))) / max(Lg, 1e-6))

def courbure(x, inverse):
    P = x["P"][::-1] if inverse else x["P"]; T = np.diff(P, axis=0); ang = np.arctan2(T[:, 0], T[:, 1]); d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi; e = int(round(0.10 * NB)); return d[e:len(d) - e]

def corr(a, b):
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def c_unique(k):
    s = float(np.abs(k).sum()); return abs(float(k.sum())) / s if s > 1e-9 else 0.0

def franc(k):
    v = np.abs(np.convolve(k, np.ones(W3), mode="same")); s = float(np.abs(k).sum()); return float(v.max()) / s if s > 1e-9 else 0.0

def opp(k):
    v = np.convolve(k, np.ones(W3), mode="same"); best = 0.0
    for i in range(len(v)):
        for j in range(i + DMIN, min(len(v), i + DMAX + 1)):
            if v[i] * v[j] < 0: best = max(best, min(abs(v[i]), abs(v[j])))
    return float(best)

def contour_angles(m, n=160, sig=1.5, win=5):
    """Outer contour points (160, counter-clockwise) and turning angle of each point (salient > 0, re-entrant < 0)."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)[:, 0, ::-1].astype(float)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T))]; t = np.linspace(0, seg[-1], n, endpoint=False)
    q = np.c_[np.interp(t, seg, np.r_[c[:, 0], c[0, 0]]), np.interp(t, seg, np.r_[c[:, 1], c[0, 1]])]; q = gaussian_filter1d(q, sig, axis=0, mode="wrap")
    T = np.roll(q, -1, axis=0) - q; ang = np.arctan2(T[:, 0], T[:, 1]); d = np.diff(np.r_[ang, ang[0]]); d = (d + np.pi) % (2 * np.pi) - np.pi
    if d.sum() < 0: d = -d                                     # traversal direction chosen so that the total turn is +2 pi (salient > 0)
    s = np.convolve(np.r_[d[-(win // 2):], d, d[:win // 2]], np.ones(win), mode="valid"); return q, s

def forme_L(m, x, k):
    """Contour angle on both sides of the body at the strongest turn of the axis, and straightness of the two arms."""
    e = int(round(0.10 * NB)); v = np.abs(np.convolve(k, np.ones(W3), mode="same")); i = int(np.argmax(v)); ka = i + e + 1   # point de l'axe
    P = x["P"]; Tg = P[min(NB - 1, ka + 1)] - P[max(0, ka - 1)]; Tg = Tg / max(float(np.hypot(*Tg)), 1e-6); Nn = np.array([-Tg[1], Tg[0]])
    q, s = contour_angles(m); bords = []
    for sg in (1.0, -1.0):
        tt = 0.0
        while True:
            y, xx = P[ka] + sg * (tt + 0.5) * Nn
            if not (0 <= int(round(y)) < m.shape[0] and 0 <= int(round(xx)) < m.shape[1]) or not m[int(round(y)), int(round(xx))]: break
            tt += 0.5
        pb = P[ka] + sg * tt * Nn; j = int(np.argmin(np.hypot(*(q - pb).T)))
        w_ = np.take(s, np.arange(j - 6, j + 7), mode="wrap"); bords.append(float(w_[int(np.argmax(np.abs(w_)))]))
    sal = max(bords); ren = min(bords)
    deux = min(abs(sal), abs(ren)) if (sal > 0 and ren < 0) else 0.0
    sym = (min(abs(sal), abs(ren)) / max(abs(sal), abs(ren), 1e-6)) if deux > 0 else 0.0
    def rect(Q):
        if len(Q) < 3: return 1.0
        arc = float(np.sum(np.hypot(*np.diff(Q, axis=0).T))); return float(np.hypot(*(Q[-1] - Q[0]))) / max(arc, 1e-6)
    bras = min(rect(P[:max(2, ka - 1)]), rect(P[min(NB - 2, ka + 2):]))
    return deux, sym, bras

def calcule(arg):
    sp, bn = arg
    try:
        pv, nx = voisins(bn)
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        M = [_objet(g) for g in (gt, prep(pv), prep(nx))]; X = [axe_robuste(mm) for mm in M]; x0 = X[0]
        K = [courbure(x0, False)] + [courbure(x, not oriente(x0, x)) for x in X[1:]]
        C = [c_unique(k) for k in K]; r0 = float(np.sqrt(np.mean(K[0] ** 2))) + 1e-6
        Ls = [forme_L(mm, x, k) for mm, x, k in zip(M, X, K)]
        out = dict(sq_boucle=max(x["boucle"] for x in X), sq_c_min=min(C), sq_c_rng=max(C) - min(C),
                   sq_courb_corr=min(corr(K[0], K[1]), corr(K[0], K[2])),
                   sq_courb_ecart=float(np.mean([np.sqrt(np.mean((k - K[0]) ** 2)) / r0 for k in K[1:]])),
                   sq_franc_min=min(franc(k) for k in K), sq_opp_min=min(opp(k) for k in K),
                   l_deux_cotes_t0=Ls[0][0], l_deux_cotes_min=min(l[0] for l in Ls), l_symetrie_t0=Ls[0][1],
                   l_bras_min=min(l[2] for l in Ls), l_score_min=min(l[0] * l[2] for l in Ls))
        return sp, bn, {k: float(out[k]) for k in COLS_B}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def calcule_essai(arg):
    try: return calcule(arg)
    except RuntimeError as e: return arg[0], arg[1], str(e)

CAS = ["Atropine_500µM_T24h_R1(2)_270223_Frame_12_", "MK801_200µM_T3h_R2(2)_200423_Frame_16_", "MK801_200µM_T3h_R2(2)_200423_Frame_17_", "NMDA_10000µM_T1h_R2(3)_190423_Frame_25_", "Apomorphine_10µM_T3h_R1(3)_030523_Frame_26_", "MK801_200µM_T3h_R2(2)_200423_Frame_18_",
       "Atropine_500µM_T0_R2(1)_280223_Frame_18_", "Apomorphine_5µM_T1h_R3(2)_030523_Frame_2_", "Fluoxetine_20µM_T3h_R3(5)_260423_Frame_23_",
       "Apomorphine_10µM_T3h_R1(3)_030523_Frame_18_", "Apomorphine_10µM_T3h_R1(3)_030523_Frame_25_", "Atropine_500µM_T24h_R2(5)_280223_Frame_4_",
       "Methiothepin_10µM_T24h_R1(3)_140323_Frame_10_", "Halopéridol_1µM_T24h_R1(1)_230523_Frame_15_"]

def planche(noms, fichier):
    """Control plate: contour (green), robust axis (red), strongest turn (yellow)."""
    tu = []
    for nm in noms:
        try:
            g = format_384(nm); m = _objet(g); x = axe_robuste(m); k = courbure(x, False)
        except Exception as e:
            print("plate:", nm, e); continue
        im = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (0, 200, 0), 1)
        cv2.polylines(im, [x["P"][:, ::-1].astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        ka = int(np.argmax(np.abs(np.convolve(k, np.ones(W3), mode="same")))) + int(round(0.10 * NB)) + 1
        cv2.circle(im, (int(x["P"][ka, 1]), int(x["P"][ka, 0])), 6, (0, 255, 255), 2)
        dc, sy, br = forme_L(m, x, k)
        cv2.putText(im, "loop %d  c %.2f  L %.2f  arm %.2f" % (x["boucle"], c_unique(k), dc, br), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, nm[:42].replace("µ", "u").replace("é", "e"), (4, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tu.append(im)
    while len(tu) % 4: tu.append(np.zeros_like(tu[0]))
    cv2.imwrite(fichier, np.vstack([np.hstack(tu[i:i + 4]) for i in range(0, len(tu), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])

def main():
    t0 = time.time(); ESSAI = int(os.environ.get("V4C_ESSAI", "0"))
    print("v4c tip-to-tip axis and L-shape descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
          "| central processor, %d processes%s" % (N_WORKERS, " | TRIAL %d per class" % ESSAI if ESSAI else ""), flush=True)
    assert os.path.isdir(os.path.join(V3T, "data")), "v3t tables missing"
    if not ESSAI: assert not os.path.exists(os.path.join(V4C, "data")), "lightgbm_stereotype_v4c/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3T, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    os.makedirs(V4C, exist_ok=True)
    if ESSAI:
        taches = []
        for s in ("valid", "test", "golden"):
            d = BASE[s]
            if s != "golden": taches += [(s, bn) for bn in d.groupby("label").head(ESSAI).image]
            taches += [(s, bn) for bn in d.image if any(c in bn for c in CAS) and (s, bn) not in taches]
        tr = BASE["train"]; ex = list(tr[tr.label == "C-shape"].image.iloc[[0, 50, 300, 500]]) + list(tr[tr.label == "Walnut"].image.iloc[[0, 40]])
        planche([bn for s, bn in taches if any(c in bn for c in CAS)][:10] + ex, os.path.join(V4C, "essai_axes_v4c.jpg"))
    else: taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V4C, "_cache_v4c%s.pkl" % ("_essai" if ESSAI else "")); RES = {}; SAUT = []
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
    pd.set_option("display.width", 280)
    if ESSAI:
        if SAUT: print("trial: %d frames skipped (neighbours not visible from this machine), e.g. %s" % (len(SAUT), SAUT[0][:160]), flush=True)
        E = pd.DataFrame([dict(partition=s, image=bn, label=BASE[s].set_index("image").label[bn], **RES[(s, bn)]) for s, bn in taches if (s, bn) in RES])
        assert np.isfinite(E[COLS_B].to_numpy(float)).all(), "non-finite value in the trial"
        E.to_csv(os.path.join(V4C, "essai_v4c.csv"), sep=";", index=False)
        print("\nmeans per class (trial):\n" + E.groupby("label")[COLS_B].mean().round(3).to_string())
        print("\nv3t failure cases:\n" + E[E.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_B].round(3).to_string(index=False))
        print("TRIAL done, 0 non-finite value, no table written | %.0f s" % (time.time() - t0)); return
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in COLS_B: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_B[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_B].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V4C, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v4c/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V4C, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V4C, "data", "features_%s.csv" % s), sep=";", index=False)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_B].median().round(3).to_string())
    print("\nmeans per class (training):\n" + OUTT["train"].groupby("label")[COLS_B].mean().round(3).to_string())
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test", "golden")])
    print("\nv3t failure cases:\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["partition", "image", "label"] + COLS_B].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors sq_ and l_ | %.0f s" % (len(COLS_B), time.time() - t0))
    print("v4c tables:", os.path.join(V4C, "data"), "| v3t tables kept:", os.path.join(V3T, "data"))

if __name__ == "__main__":
    main()
