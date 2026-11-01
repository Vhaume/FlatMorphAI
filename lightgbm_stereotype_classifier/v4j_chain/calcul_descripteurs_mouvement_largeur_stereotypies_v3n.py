# -*- coding: utf-8 -*-
# calcul_descripteurs_mouvement_largeur_stereotypies_v3n (21 October 2026). New program; the v3c to v3m programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Valentin's remarks of 21 October on the errors of the v3m test:
#   Methiothepin_5uM_T3h_R1(4) F2 (Screw-like read as Snake-like): look at how the worm's width evolves over time, part
#   upper part (thin neck at F1 and F2, wide part at F3).
#   Atropine_500uM_T0_R2(1) F18 and Acetylcholine_10000uM_T1h_R2(1) F5 (Wildtype read as C-shape): the body moves a lot over
#   time, motion metrics are missing.
# The existing motion descriptors (motion, motioncl, tt_) are measured on the frames REGISTERED on T0 (align_to of
# make_motion_dataset): the registration removes the overall displacement and the rotation, hence much of the motion of a worm
# that glides or sweeps its head. Two groups added, dimensionless, no value guessed (the run stops if one is not finite):
#   mv_ (8), UNREGISTERED motion, raw Mask2Former masks of T-1, T0, T+1 (same image / mask pair as v3c, fixe
#   function), at half resolution (a neighbour stored as a copy rotated by 90 degrees is put back in the orientation of T0, the orientation of the best
#   overlap), geodesic axis (axe function of v3g, 40 slices), distances relative to the worm length L at T0:
#     mv_iou_min, mv_iou_moy overlap (IoU, Jaccard, 1912, New Phytologist, DOI 10.1111/j.1469-8137.1912.tb05611.x) des
#                              raw silhouettes T0 / T-1 and T0 / T+1, without registration (a held C-shape keeps a high IoU).
#     mv_centre              : maximum displacement of the centre of mass / L.
#     mv_bout_max, mv_bout_asym : maximum displacement of one extremity (ends paired by proximity) / L, and asymmetry
#                              |d_a - d_b| / (d_a + d_b) between the two ends (a sweeping head).
#     mv_axe_moy, mv_axe_max : mean and maximum chamfer distance from the axis points of T0 to the axis of the neighbour / L
#                              (Barrow et al., 1977, Parametric correspondence and chamfer matching, IJCAI 1977).
#     mv_courbure_chg        : variation of the global curvature 1 - chord / length between T0 and the neighbours (maximum).
#   wz_ (5), evolution of the width along the body, frames in 384 format registered on T0 (local width 2 x border distance
#   / frame length L, 40 slices, ends paired):
#     wz_max        : max over the slices and the neighbours of |log(neighbour width / T0 width)|.
#     wz_bout_max   : the same, limited to the terminal third of each end (the upper part of the worm in the remark).
#     wz_moy        : mean over the slices of |log(ratio)|, maximum over the neighbours.
#     wz_col_t0     : minimum width outside the ends (15%) / median width at T0.
#     wz_col_chg    : |log| of the width variation in the neck slice of T0 between T0 and the neighbours (maximum).
# Outputs: lightgbm_stereotype_v3n/data/features_{train,valid,test,golden}.csv (v3l tables + 13 columns), cache _cache_v3n.pkl.
# Central processor, one thread per process. Seed 42. Run from the FasterViT root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
_src = open(os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py"), encoding="utf-8").read()
assert 'if __name__ == "__main__":\n    main()' in _src
_L = {"__name__": "v3g_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_commentaires_stereotypies_v3g.py")}
exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_commentaires_stereotypies_v3g", "exec"), _L)
format_384, voisins, NS_MOT, axe, _objet, fixe, _lire = (_L[k] for k in ("format_384", "voisins", "NS_MOT", "axe", "_objet", "fixe", "_lire"))
V3L = os.path.join(ROOT, "lightgbm_stereotype_v3l"); V3N = os.path.join(ROOT, "lightgbm_stereotype_v3n")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")
COLS_N = ["mv_iou_min", "mv_iou_moy", "mv_centre", "mv_bout_max", "mv_bout_asym", "mv_axe_moy", "mv_axe_max", "mv_courbure_chg",
          "wz_max", "wz_bout_max", "wz_moy", "wz_col_t0", "wz_col_chg"]
lg = lambda a, b: np.abs(np.log(np.maximum(a, 1e-3) / np.maximum(b, 1e-3)))

def masque_brut(nm):
    """Raw Mask2Former mask of frame nm at half resolution (largest object)."""
    m = _lire(fixe(nm)[1]); h, w = m.shape[:2]
    m = cv2.resize((m > 127).astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); assert k >= 2, "masque vide : " + nm
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))

def oriente(x0, x):
    """True if extremity a of x corresponds to extremity a of x0."""
    return np.hypot(*(x["a"] - x0["a"])) + np.hypot(*(x["b"] - x0["b"])) <= np.hypot(*(x["a"] - x0["b"])) + np.hypot(*(x["b"] - x0["a"]))

def mouvement(bn, pv, nx):
    M = [masque_brut(x) for x in (bn, pv, nx)]
    for i in (1, 2):   # copy rotated by 90 degrees (v3c image / mask pair): orientation chosen by the best overlap with T0
        if M[i].shape != M[0].shape:
            cand = [np.rot90(M[i], k) for k in (1, 3)]; assert all(c.shape == M[0].shape for c in cand), "incompatible mask sizes"
            M[i] = max(cand, key=lambda c: float((c & M[0]).sum()))
    X = [axe(m) for m in M]; x0 = X[0]; L = x0["L"]
    c = [np.array(np.nonzero(m)).mean(1) for m in M]
    iou = [float((M[0] & m).sum()) / float((M[0] | m).sum()) for m in M[1:]]
    bouts, asym, axm, axx, cch = [], [], [], [], []
    for x, m in zip(X[1:], M[1:]):
        d = oriente(x0, x); ea, eb = (x["a"], x["b"]) if d else (x["b"], x["a"])
        da, db = float(np.hypot(*(ea - x0["a"]))) / L, float(np.hypot(*(eb - x0["b"]))) / L
        bouts.append(max(da, db)); asym.append(abs(da - db) / (da + db + 1e-6))
        dd = np.sqrt(((x0["P"][:, None, :] - x["P"][None, :, :]) ** 2).sum(2)).min(1) / L
        axm.append(float(dd.mean())); axx.append(float(dd.max())); cch.append(abs(x["bend"] - x0["bend"]))
    return dict(mv_iou_min=min(iou), mv_iou_moy=float(np.mean(iou)), mv_centre=max(float(np.hypot(*(ci - c[0]))) for ci in c[1:]) / L,
                mv_bout_max=max(bouts), mv_bout_asym=asym[int(np.argmax(bouts))], mv_axe_moy=max(axm), mv_axe_max=max(axx), mv_courbure_chg=max(cch))

def largeur(gt, gp, gn):
    X = [axe(_objet(g)) for g in (gt, gp, gn)]; x0 = X[0]; w0 = x0["w"] / x0["L"]; n = len(w0); e = max(1, int(0.15 * n)); t = n // 3
    zmax, zbout, zmoy = [], [], []
    k = e + int(np.argmin(x0["w"][e:n - e])); col = []
    for x in X[1:]:
        w = x["w"] / x["L"]; w = w if oriente(x0, x) else w[::-1]; r = lg(w, w0)
        zmax.append(float(r.max())); zbout.append(float(max(r[:t].max(), r[-t:].max()))); zmoy.append(float(r.mean())); col.append(float(r[k]))
    return dict(wz_max=max(zmax), wz_bout_max=max(zbout), wz_moy=max(zmoy), wz_col_t0=float(x0["w"][k] / x0["wmed"]), wz_col_chg=max(col))

def calcule(arg):
    sp, bn = arg
    try:
        pv, nx = voisins(bn); out = mouvement(bn, pv, nx)
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        out.update(largeur(gt, prep(pv), prep(nx)))
        return sp, bn, {k: float(out[k]) for k in COLS_N}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def main():
    t0 = time.time()
    print("v3n motion and width descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "| central processor, %d processes" % N_WORKERS, flush=True)
    assert not os.path.exists(os.path.join(V3N, "data")), "lightgbm_stereotype_v3n/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3L, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V3N, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V3N, "_cache_v3n.pkl"); RES = {}
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
        for c in COLS_N: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_N[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_N].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V3N, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v3n/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V3N, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V3N, "data", "features_%s.csv" % s), sep=";", index=False)
    pd.set_option("display.width", 250)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_N].median().round(3).to_string())
    CAS = ["Methiothepin_5µM_T3h_R1(4)_140323_Frame_2_", "Atropine_500µM_T0_R2(1)_280223_Frame_18_", "Acetylcholine_10000µM_T1h_R2(1)_060423_Frame_5_"]
    T = OUTT["test"]
    print("\ncommented cases (test):\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_N].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors mv_ and wz_ | %.0f s" % (len(COLS_N), time.time() - t0))
    print("v3n tables:", os.path.join(V3N, "data"), "| v3l tables kept:", os.path.join(V3L, "data"))

if __name__ == "__main__":
    main()
