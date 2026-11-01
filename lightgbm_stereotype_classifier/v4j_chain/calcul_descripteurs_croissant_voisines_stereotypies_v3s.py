# -*- coding: utf-8 -*-
# calcul_descripteurs_croissant_voisines_stereotypies_v3s (22 October 2026). New program; the v3c to v3r programs and the
# lightgbm_stereotype_classifier folder are not modified. Copy of v3r with a single difference, requested by Valentin:
# the temporal descriptors use only T-1, T0 and T+1 (v3f neighbours, 3 frames), never a wider window.
# v3r (window of +/- 5 frames) is abandoned.
# Valentin's remarks of 22 October on the Wildtype / C-shape confusions (Claude outputs/confusions_wildtype_cshape):
#   C-shape read as Wildtype (Haloperidol_100uM_T1h_R1(4), Haloperidol_10uM_T1h_R2(2)): C-shapes that hardly move and that have
#   a crescent-moon shape.
#   Apomorphine_0uM_T3h_R2(2) F6 and Acetylcholine_5000uM_T0_R1(4) F6 (Wildtype read as C-shape): non-constant width, markedly
#   wider at the centre, a single rounded side.
#   Acetylcholine_10000uM_T0_R1(3) F3 and Methiothepin_5uM_T3h_R1(2) F14 (Wildtype read as C-shape): zigzag shaped like an L, and
#   move more than a C-shape.
# Three groups, dimensionless, no value guessed (stop if not finite):
#   cr_ (crescent, 4), frame T0 in 384 format, outer contour resampled at 400 points and smoothed (Gaussian sigma = 3
#   points), points attached to the nearest point of the v3g geodesic axis (40 slices) and split into left side / right
#   side by the sign of the cross product with the axis tangent, ends excluded (terminal 15%). Each side is ordered along
#   the axis, and its sagitta f (maximum deviation from its chord c, measured along the normal to the axis chord, common to both
#   sides) gives its curvature 8 f / c^2 (approximation of a circular arc by its sagitta, a stable fit even for a straight side):
#     cr_courb_min  : min over the two sides of curvature x L (a crescent has both sides curved, a straight side gives 0).
#     cr_meme_sens  : signed ratio of the sagittae of the two sides, between -1 and 1 (1: both sides curve the same way and
#                     by the same amount, like the borders of a crescent; 0: one straight side; negative: opposite sides).
#     cr_resid_max  : max over the two sides of the root-mean-square deviation from the fitted parabola (numpy.polyfit, degree 2) / L
#                     (zigzag or L: border poorly described by a single arc).
#     cr_w_centre   : mean width of the central third / mean width of the slices from 15 to 33% and from 67 to 85% of the
#                     length (smoothed profile, sigma = 1 slice): body wider at the centre (> 1) or of constant width (1).
#   zl_ (zigzag and L, 2), T0 axis smoothed (Gaussian sigma = 1.5 slice):
#     zl_coin_max   : maximum angle turned by the axis over a window of 4 slices (10% of L), as a fraction of pi / 2 (1 = corner
#                     en L).
#     zl_coin_frac  : angle turned by the 20% most curved slices / max(total angle turned, pi / 2) (uniform curvature
#                     of a C near 0.3, concentrated corner near 1, straight worm near 0 thanks to the pi / 2 floor).
#   fw_ (neighbours T-1 and T+1, 4), raw Mask2Former masks (same camera, half resolution, a copy rotated by 90 degrees put back
#   in the orientation of T0 as in v3n), v3f neighbours (n - 1 and n + 1, or two neighbours on the same side at the start and end of a video),
#   medians over the two neighbours, displacement and shape change relative to the frame gap |k - n|:
#     fw_hu_med     : median of the shape distance between T0 and each frame (Hu invariants, cv2.matchShapes method I1, Hu,
#                     1962, IRE Trans Inf Theory, DOI 10.1109/TIT.1962.1057692).
#     fw_hu_pente   : median of that distance / |k - n| (rate of shape change).
#     fw_vit_med    : median of the displacement of the centre of mass / (L x |k - n|) (speed of the worm).
#     fw_iou_med    : median of the overlap (IoU) of the raw masks of T0 and of each frame (immobility).
# Outputs: lightgbm_stereotype_v3s/data/features_{train,valid,test,golden}.csv (v3q tables + 10 columns), cache _cache_v3s.pkl,
# frames_voisines_v3s.csv (frames used). Central processor, one thread per process. Seed 42. Run from the FasterViT root.
import os, sys, time, datetime, pickle, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(1); np.random.seed(42)
ROOT = os.getcwd()
COLS_R = ["cr_courb_min", "cr_meme_sens", "cr_resid_max", "cr_w_centre", "zl_coin_max", "zl_coin_frac", "fw_hu_med", "fw_hu_pente", "fw_vit_med", "fw_iou_med"]
NPTS = 400

def contour(m, n=NPTS):
    """Outer contour of mask m (y, x) resampled to n points, and the raw OpenCV contour."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)
    p = c[:, 0, ::-1].astype(float); p = np.vstack([p, p[:1]]); s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], c

def cote_arc(q, nrm):
    """Signed sagitta of one side (points q): maximum deviation from its chord along the common normal nrm, curvature 8 f / c^2,
    and root-mean-square deviation from the parabola fitted in the chord frame."""
    t = q[-1] - q[0]; c = float(np.hypot(*t)); t = t / max(c, 1e-9)
    u = (q - q[0]) @ t; v = (q - q[0]) @ nrm; v = v - (u / max(c, 1e-9)) * float(v[-1])   # deviation from the chord
    f = float(v[np.argmax(np.abs(v))]); co = np.polyfit(u, v, 2); res = float(np.sqrt(np.mean((np.polyval(co, u) - v) ** 2)))
    return f, 8.0 * abs(f) / max(c * c, 1e-9), res

def croissant(m, x):
    p = gaussian_filter1d(contour(m)[0], 3.0, axis=0, mode="wrap")
    P = gaussian_filter1d(x["P"], 1.5, axis=0, mode="nearest"); T = np.gradient(P, axis=0); nb = len(P)
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1])[:, None], 1e-6)
    j = ((p[:, None, 0] - P[None, :, 0]) ** 2 + (p[:, None, 1] - P[None, :, 1]) ** 2).argmin(1)
    cote = np.sign(T[j, 1] * (p[:, 0] - P[j, 0]) - T[j, 0] * (p[:, 1] - P[j, 1]))
    e = int(round(0.15 * nb)); corps = (j >= e) & (j < nb - e); L = x["L"]
    ch = P[nb - 1 - e] - P[e]; nrm = np.array([-ch[1], ch[0]]) / max(float(np.hypot(*ch)), 1e-9)   # common normal (chord of the axis)
    A = []
    for s in (-1, 1):
        sel = np.where(corps & (cote == s))[0]
        if len(sel) < 5: A.append((0.0, 0.0, 0.0)); continue
        q = p[sel][np.lexsort(((p[sel] - P[e]) @ (ch / max(float(np.hypot(*ch)), 1e-9)), j[sel]))]
        A.append(cote_arc(q, nrm))
    f1, f2 = A[0][0] / L, A[1][0] / L
    w = gaussian_filter1d(x["w"], 1.0, mode="nearest"); i = lambda f: int(round(f * nb))
    centre = float(w[i(1 / 3):i(2 / 3)].mean()); quarts = float(np.r_[w[i(0.15):i(0.33)], w[i(0.67):i(0.85)]].mean())
    return dict(cr_courb_min=min(A[0][1], A[1][1]) * L, cr_meme_sens=min(f1 * np.sign(f1 + f2), f2 * np.sign(f1 + f2)) / max(abs(f1), abs(f2), 1e-9),
                cr_resid_max=max(A[0][2], A[1][2]) / L, cr_w_centre=centre / max(quarts, 1e-6))

def zigzag(x):
    P = gaussian_filter1d(x["P"], 1.5, axis=0, mode="nearest"); k = 2
    u, v = P[k:-k] - P[:-2 * k], P[2 * k:] - P[k:-k]
    ang = np.arccos(np.clip((u * v).sum(1) / np.maximum(np.hypot(*u.T) * np.hypot(*v.T), 1e-9), -1.0, 1.0))
    d = np.diff(P, axis=0); dth = np.abs(np.diff(np.unwrap(np.arctan2(d[:, 0], d[:, 1])))); top = np.sort(dth)[::-1][:max(1, int(round(0.2 * len(dth))))]
    return dict(zl_coin_max=float(ang.max()) / (np.pi / 2), zl_coin_frac=float(top.sum()) / max(float(dth.sum()), np.pi / 2))

def fenetre(M0, n, frames, L):
    """M0: raw mask of T0, frames: list of (k, raw mask), L: length of T0 on the raw mask."""
    c0 = np.array(np.nonzero(M0)).mean(1); h0 = contour(M0)[1]; hu, hp, vit, iou = [], [], [], []
    for k, m in frames:
        dk = abs(k - n); d = float(cv2.matchShapes(h0, contour(m)[1], cv2.CONTOURS_MATCH_I1, 0.0))
        hu.append(d); hp.append(d / dk); vit.append(float(np.hypot(*(np.array(np.nonzero(m)).mean(1) - c0))) / (L * dk))
        iou.append(float((M0 & m).sum()) / float((M0 | m).sum()))
    return dict(fw_hu_med=float(np.median(hu)), fw_hu_pente=float(np.median(hp)), fw_vit_med=float(np.median(vit)), fw_iou_med=float(np.median(iou)))

if __name__ == "__main__":
    _src = open(os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__":\n    main()' in _src
    _L = {"__name__": "v3n_bibliotheque", "__file__": os.path.join(ROOT, "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py")}
    exec(compile(_src.replace('if __name__ == "__main__":\n    main()', ""), "calcul_descripteurs_mouvement_largeur_stereotypies_v3n", "exec"), _L)
    format_384, voisins, axe, _objet, masque_brut = (_L[k] for k in ("format_384", "voisins", "axe", "_objet", "masque_brut"))
    _G = voisins.__globals__; parse_frame, frame_name, NAME2PATH, MASK_ALL = (_G[k] for k in ("parse_frame", "frame_name", "NAME2PATH", "MASK_ALL"))
V3Q = os.path.join(ROOT, "lightgbm_stereotype_v3q"); V3S = os.path.join(ROOT, "lightgbm_stereotype_v3s")
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 4)))
SPL = ("train", "valid", "test", "golden")

def choix_frames(bn):
    """v3f neighbours (T-1 and T+1) only."""
    sid, n, tot = parse_frame(bn); pv, nx = voisins(bn)
    return n, sorted([(parse_frame(x)[1], x) for x in (pv, nx)])

def calcule(arg):
    sp, bn = arg
    try:
        n, fr = choix_frames(bn)
        M0 = masque_brut(bn); F = []
        for k, nm in fr:
            m = masque_brut(nm)
            if m.shape != M0.shape:   # copy rotated by 90 degrees: orientation chosen by the best overlap with T0 (as in v3n)
                cand = [np.rot90(m, r) for r in (1, 3)]; assert all(c.shape == M0.shape for c in cand), "incompatible mask sizes: " + nm
                m = max(cand, key=lambda c: float((c & M0).sum()))
            F.append((k, m))
        g0 = _objet(format_384(bn)); x0 = axe(g0)
        out = croissant(g0, x0); out.update(zigzag(x0)); out.update(fenetre(M0, n, F, axe(M0)["L"]))
        return sp, bn, {k: float(out[k]) for k in COLS_R}, "|".join(nm for _, nm in fr)
    except Exception as e:
        raise RuntimeError("%s | %s : %r" % (sp, bn, e)) from e

def main():
    t0 = time.time()
    print("v3s crescent, zigzag and neighbour descriptors |", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "| central processor, %d processes" % N_WORKERS, flush=True)
    assert not os.path.exists(os.path.join(V3S, "data")), "lightgbm_stereotype_v3s/data already exists, nothing is overwritten"
    BASE = {s: pd.read_csv(os.path.join(V3Q, "data", "features_%s.csv" % s), sep=";") for s in SPL}
    taches = [(s, bn) for s in SPL for bn in BASE[s].image]
    os.makedirs(V3S, exist_ok=True)
    SIG = hashlib.sha256(open(os.path.join(ROOT, "calcul_descripteurs_croissant_voisines_stereotypies_v3s.py"), "rb").read()).hexdigest()
    CACHE = os.path.join(V3S, "_cache_v3s.pkl"); RES, FR = {}, {}
    if os.path.exists(CACHE):
        c = pickle.load(open(CACHE, "rb"))
        if c.get("sig") == SIG: RES, FR = c["res"], c["fr"]
    reste = [t for t in taches if t not in RES]; print("cache: %d, to compute: %d" % (len(RES), len(reste)), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, (s, bn, out, fr) in enumerate(pool.imap_unordered(calcule, reste, chunksize=2), 1):
            RES[(s, bn)] = out; FR[(s, bn)] = fr
            if i % 300 == 0: pickle.dump(dict(sig=SIG, res=RES, fr=FR), open(CACHE, "wb")); print("  %d / %d | %.0f s" % (i, len(reste), time.time() - t0), flush=True)
    pickle.dump(dict(sig=SIG, res=RES, fr=FR), open(CACHE, "wb"))
    OUTT = {}
    for s in SPL:
        O = BASE[s].copy()
        for c in COLS_R: O[c] = [RES[(s, bn)][c] for bn in O.image]
        OUTT[s] = O
    fautes = [(s, OUTT[s].image.iloc[i], COLS_R[j]) for s in SPL for i, j in zip(*np.where(~np.isfinite(OUTT[s][COLS_R].to_numpy(float))))]
    if fautes:
        pd.DataFrame(fautes, columns=["partition", "image", "descripteur"]).to_csv(os.path.join(V3S, "valeurs_non_finies.csv"), sep=";", index=False)
        raise SystemExit("STOP: %d non-finite values, see lightgbm_stereotype_v3s/valeurs_non_finies.csv (no table written)" % len(fautes))
    os.makedirs(os.path.join(V3S, "data"))
    for s in SPL: OUTT[s].to_csv(os.path.join(V3S, "data", "features_%s.csv" % s), sep=";", index=False)
    pd.DataFrame([dict(partition=s, image=bn, frames=FR[(s, bn)]) for s, bn in taches]).to_csv(os.path.join(V3S, "frames_voisines_v3s.csv"), sep=";", index=False)
    pd.set_option("display.width", 250)
    print("\nmedians per class (training):\n" + OUTT["train"].groupby("label")[COLS_R].median().round(3).to_string())
    print("\nmedians per class (validation):\n" + OUTT["valid"].groupby("label")[COLS_R].median().round(3).to_string())
    CAS = ["Atropine_500µM_T0_R2(1)_280223_Frame_18_", "Acetylcholine_10000µM_T1h_R2(1)_060423_Frame_5_", "Haloperidol_100µM_T1h_R1(4)_230523_Frame_12_",
           "Apomorphine_0µM_T3h_R2(2)_030523_Frame_6_", "Acetylcholine_5000µM_T0_R1(4)_050423_Frame_6_", "Acetylcholine_10000µM_T0_R1(3)_050423_Frame_3_",
           "Methiothepin_5µM_T3h_R1(2)_140323_Frame_14_"]
    T = pd.concat([OUTT[s].assign(partition=s) for s in ("train", "valid", "test")])
    print("\ncommented cases:\n" + T[T.image.apply(lambda x: any(c in x for c in CAS))][["partition", "image", "label"] + COLS_R].round(3).to_string(index=False))
    print("ZERO TOLERANCE MET: 0 missing or infinite value in the %d descriptors cr_, zl_, fw_ | %.0f s" % (len(COLS_R), time.time() - t0))
    print("v3s tables:", os.path.join(V3S, "data"), "| v3q tables kept:", os.path.join(V3Q, "data"))

if __name__ == "__main__":
    main()
