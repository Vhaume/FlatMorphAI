# -*- coding: utf-8 -*-
# calcul_descripteurs_ligne_pli_stereotypies_v4e (22 October 2026). New program; the v3c to v4d programs and the
# lightgbm_stereotype_classifier folder are not modified.
# Valentin's remark (cyan stroke at the corner of the L, Atropine_500uM_T24h_R1(2) F12): at the twist, a sharp line crosses the body, the
# border breaks and the tissue changes on either side. v4d (comparison of sections of the straightened worm) places that break badly in
# a sharp corner, where the perpendicular sections cross. Here the fold line is looked for directly in the image.
# Method, frame by frame:
#   1. Internal edges: Canny detector (Canny, 1986, A Computational Approach to Edge Detection, IEEE Transactions on Pattern
#      Analysis and Machine Intelligence 8(6):679-698, DOI 10.1109/TPAMI.1986.4767851 ; skimage.feature.canny, sigma 1.5 pixel, thresholds
#      by quantiles 0.80 and 0.95 of the gradient inside the worm) on the grey-level image, restricted to pixels more than 4 pixels from the
#      border (the pale rim of the outer contour is excluded).
#   2. Segments: connected components (8-neighbourhood) of at least 8 pixels. For each segment, the border distance of its pixels
#      (cv2.distanceTransform) relative to the local half-width of the worm (at the nearest point of the v4c tip-to-tip axis).
#      Penetration = maximum distance (how far the line enters the body), anchoring = minimum distance (the line starts at the border).
#   Segments whose nearest axis point falls in the terminal 15% are discarded (eyespots and head auricles).
#   3. Fold line: a segment that starts at the border (anchoring <= 0.5) and enters the body (penetration >= 0.4), like Valentin's cyan
#      stroke. Fold strength = penetration x (1 - anchoring). Contrast = mean gradient along the segment / median worm gradient.
# Group pl_ (frames T-1, T0, T+1 registered on T0 as in v3i to v4d):
#   pl_pli_t0         : strength of the most marked fold line at T0 (0 if there is none).
#   pl_pli_min        : minimum over the three frames (the line holds).
#   pl_penetration_t0 : penetration of that line at T0.
#   pl_contraste_t0   : contrast of that line at T0.
#   pl_n_t0           : number of fold lines at T0.
#   pl_long_lignes    : total length of the fold lines / axis length, median over the three frames.
# Zero tolerance on NaN, nothing is imputed. Outputs: lightgbm_stereotype_v4e/data (v3t tables + 6 pl_ columns), cache
# _cache_v4e.pkl. Trial: V4E_ESSAI=N, plate lightgbm_stereotype_v4e/essai_pli_v4e.jpg (internal borders in grey, fold lines
# en cyan, la plus marquee en vert, axe en rouge).
# Central processor, one thread per process. Seed 42. Run from the FasterViT root (uses the v4c functions).
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from skimage.feature import canny
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_P = ["pl_pli_t0", "pl_pli_min", "pl_penetration_t0", "pl_contraste_t0", "pl_n_t0", "pl_long_lignes"]
SIG, EROS, NMIN, PEN_MIN, ANC_MAX, BOUT = 1.5, 4, 8, 0.4, 0.5, 0.15

if __name__ == "__main__":
    _src = open(os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    _C = {"__name__": "__main__", "__file__": os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_axe_pointes_stereotypies_v4c", "exec"), _C)
    format_384, voisins, NS_MOT, _objet, axe_pointes = (_C[k] for k in ("format_384", "voisins", "NS_MOT", "_objet", "axe_pointes"))
V3T = os.path.join(ROOT, "lightgbm_stereotype_v3t"); V4E = os.path.join(ROOT, "lightgbm_stereotype_v4e")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

def lignes(m, g):
    """Internal border segments and their measures relative to the tip-to-tip axis."""
    Q, _ = axe_pointes(m); L = float(np.sum(np.hypot(*np.diff(Q, axis=0).T)))
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); inn = dt > EROS
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), SIG); gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1); gm = np.hypot(gx, gy)
    gmed = max(float(np.median(gm[inn])), 1e-6) if inn.any() else 1.0
    E = canny(g.astype(float), sigma=SIG, low_threshold=0.80, high_threshold=0.95, use_quantiles=True, mask=inn)
    k, lab = cv2.connectedComponents(E.astype(np.uint8), connectivity=8); seg = []
    T = np.gradient(Q, axis=0); T /= np.maximum(np.hypot(T[:, 0], T[:, 1])[:, None], 1e-6)
    for i in range(1, k):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < NMIN: continue
        P = np.c_[ys, xs].astype(float); c = P.mean(0); U, S, Vt = np.linalg.svd(P - c, full_matrices=False); d = Vt[0]
        j = int(np.argmin(np.hypot(*(Q - c).T))); t = T[j]; nrm = np.array([-t[1], t[0]])
        if not (BOUT * len(Q) <= j <= (1 - BOUT) * len(Q)): continue                 # head and tail excluded (eyespots, auricles)
        sin = abs(d[0] * t[1] - d[1] * t[0]); w = max(2.0 * float(dt[int(np.clip(round(Q[j, 0]), 0, m.shape[0] - 1)), int(np.clip(round(Q[j, 1]), 0, m.shape[1] - 1))]), 1.0)
        dd = dt[ys, xs] / (w / 2.0); pen = float(dd.max()); anc = float(dd.min())               # penetration and anchoring, in half-widths
        proj = (P - c) @ nrm; et = float(proj.max() - proj.min()) / w
        con = float(gm[ys, xs].mean()) / gmed
        seg.append(dict(sin=sin, etendue=et, penetration=pen, ancrage=anc, pli=pen * max(0.0, 1.0 - anc), contraste=con, n=len(ys), pix=(ys, xs)))
    tr = [s for s in seg if s["penetration"] >= PEN_MIN and s["ancrage"] <= ANC_MAX]     # lines that start at the border and enter the body
    best = max(tr, key=lambda s: s["pli"]) if tr else None
    return dict(etendue=best["penetration"] if best else 0.0, ancrage=best["ancrage"] if best else 1.0, contraste=best["contraste"] if best else 0.0,
                pli=best["pli"] if best else 0.0, n=float(len(tr)), long_trans=float(sum(s["n"] for s in tr)) / max(L, 1e-6), seg=seg, tr=tr, best=best, Q=Q, E=E)

def calcule(arg):
    sp, bn = arg
    try:
        pv, nx = voisins(bn)
        gt = format_384(bn); pt = NS_MOT["pose"](gt); assert pt is not None, "pose impossible"
        def prep(nm):
            gn = format_384(nm); pn = NS_MOT["pose"](gn); assert pn is not None, "pose impossible : " + nm
            return NS_MOT["align_to"](gn, pn, pt, gt)
        R = [lignes(_objet(g), g) for g in (gt, prep(pv), prep(nx))]
        out = dict(pl_pli_t0=R[0]["pli"], pl_pli_min=min(r["pli"] for r in R), pl_penetration_t0=R[0]["etendue"],
                   pl_contraste_t0=R[0]["contraste"], pl_n_t0=R[0]["n"], pl_long_lignes=float(np.median([r["long_trans"] for r in R])))
        return sp, bn, {k: float(out[k]) for k in COLS_P}
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def calcule_essai(arg):
    try: return calcule(arg)
    except RuntimeError as e: return arg[0], arg[1], str(e)

CAS = ["Atropine_500µM_T24h_R1(2)_270223_Frame_11_", "Atropine_500µM_T24h_R1(2)_270223_Frame_12_", "Atropine_500µM_T24h_R1(2)_270223_Frame_13_",
       "NMDA_10000µM_T1h_R2(3)_190423_Frame_25_", "MK801_200µM_T3h_R2(2)_200423_Frame_17_", "MK801_200µM_T3h_R2(2)_200423_Frame_18_",
       "Apomorphine_5µM_T1h_R3(2)_030523_Frame_2_", "Fluoxetine_20µM_T3h_R3(5)_260423_Frame_23_", "Apomorphine_10µM_T3h_R1(3)_030523_Frame_18_",
       "Apomorphine_10µM_T3h_R1(3)_030523_Frame_25_"]

def planche(noms, fichier):
    tu = []
    for nm in noms:
        try:
            g = format_384(nm); m = _objet(g); r = lignes(m, g)
        except Exception as e:
            print("plate:", nm, e); continue
        im = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        im[r["E"]] = (150, 150, 150)
        for s in r["tr"]: im[s["pix"]] = (255, 255, 0)
        if r["best"] is not None: im[r["best"]["pix"]] = (0, 255, 0)
        cv2.polylines(im, [r["Q"][:, ::-1].astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 1)
        cv2.putText(im, "fold %.2f penetr %.2f anchor %.2f n %d" % (r["pli"], r["etendue"], r["ancrage"], r["n"]), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(im, nm[:44].replace("µ", "u").replace("é", "e"), (4, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tu.append(im)
    while len(tu) % 4: tu.append(np.zeros_like(tu[0]))
    cv2.imwrite(fichier, np.vstack([np.hstack(tu[i:i + 4]) for i in range(0, len(tu), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])

def main():
    t0 = time.time(); ESSAI = int(os.environ.get("V4E_ESSAI", "0"))
    print("v4e fold-line descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
          "| central processor, %d processes%s" % (N_WORKERS, " | TRIAL %d per class" % ESSAI if ESSAI else ""), flush=True)
    assert os.path.isdir(os.path.join(V3T, "data")), "v3t tables missing"
    if not ESSAI: assert not os.path.exists(os.path.join(V4E, "data")), "lightgbm_stereotype_v4e/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3T, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    os.makedirs(V4E, exist_ok=True)
    if ESSAI:
        taches = []
        for s in ("valid", "test", "golden"):
            d = BASE[s]
            if s != "golden": taches += [(s, bn) for bn in d.groupby("label").head(ESSAI).image]
            taches += [(s, bn) for bn in d.image if any(c in bn for c in CAS) and (s, bn) not in taches]
        tr = BASE["train"]; ex = list(tr[tr.label == "C-shape"].image.iloc[[0, 300]]) + list(tr[tr.label == "Screw-like"].image.iloc[[10, 200, 600]]) + list(tr[tr.label == "Wildtype"].image.iloc[[5, 300]])
        noms = [c + "in_30.png" for c in CAS]
        planche(noms + ex, os.path.join(V4E, "essai_pli_v4e.jpg"))
    else: taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    SIG_ = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_ligne_pli_stereotypies_v4e.py"), "rb").read()
                          + open(os.path.join(ROOT, "calcul_descripteurs_axe_pointes_stereotypies_v4c.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V4E, "_cache_v4e%s.pkl" % ("_essai" if ESSAI else "")); RES = {}; SAUT = []
    if os.path.exists(CACHE):
        c = pickle.load(open(CACHE, "rb")); RES = c["res"] if c.get("sig") == SIG_ else {}
    reste = [t for t in taches if t not in RES]; print("cache: %d, to compute: %d" % (len(RES), len(reste)), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, (s, bn, out) in enumerate(pool.imap_unordered(calcule_essai if ESSAI else calcule, reste, chunksize=2), 1):
            if isinstance(out, str): SAUT.append(out); continue
            RES[(s, bn)] = out
            if i % 300 == 0: pickle.dump(dict(sig=SIG_, res=RES), open(CACHE, "wb")); print("  %d / %d | %.0f s" % (i, len(reste), time.time() - t0), flush=True)
    pickle.dump(dict(sig=SIG_, res=RES), open(CACHE, "wb"))
    pd.set_option("display.width", 260)
    if ESSAI:
        if SAUT: print("trial: %d frames skipped (neighbours not visible from this machine), e.g. %s" % (len(SAUT), SAUT[0][:160]), flush=True)
        E = pd.DataFrame([dict(partition=s, image=bn, label=BASE[s].set_index("image").label[bn], **RES[(s, bn)]) for s, bn in taches if (s, bn) in RES])
        assert np.isfinite(E[COLS_P].to_numpy(float)).all(), "non-finite value in the trial"
        E.to_csv(os.path.join(V4E, "essai_v4e.csv"), sep=";", index=False)
        print("\nmeans per class (trial):\n" + E.groupby("label")[COLS_P].mean().round(3).to_string())
        print("\nfailure cases:\n" + E[E.image.apply(lambda x: any(c in x for c in CAS))][["image", "label"] + COLS_P].round(3).to_string(index=False))
        print("TRIAL done, 0 non-finite value, no table written | %.0f s" % (time.time() - t0)); return
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in COLS_P: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_P[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_P].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V4E, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v4e/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V4E, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V4E, "data", "features_%s.csv" % s), sep=";", index=False)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_P].median().round(3).to_string())
    print("\nmeans per class (training):\n" + OUTT["train"].groupby("label")[COLS_P].mean().round(3).to_string())
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("valid", "test", "golden")])
    print("\nfailure cases:\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["partition", "image", "label"] + COLS_P].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors pl_ | %.0f s" % (len(COLS_P), time.time() - t0))
    print("v4e tables:", os.path.join(V4E, "data"), "| v3t tables kept:", os.path.join(V3T, "data"))

if __name__ == "__main__":
    main()
