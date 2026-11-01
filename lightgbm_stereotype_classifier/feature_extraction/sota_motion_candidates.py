# sota_motion_candidates.py
# SOTA shape/motion descriptors tested before committing them to precompute, each aimed at a confusion the hardcase
# analysis exposed and each computable inside the fixed 3-frame window (T-1,T0,T+1). All reuse the centerline and
# curvature profile already traced by twist_motion_masks._frame_state. No GPU, seed 42.
#   srvf_bend       elastic shape distance (square-root velocity function) between consecutive centerlines, rigid- and
#                   reparametrization-invariant non-rigid bending. Srivastava, Klassen, Joshi, Jermyn, "Shape Analysis
#                   of Elastic Curves in Euclidean Spaces", IEEE TPAMI 2011, 33(7):1415-1428, DOI 10.1109/TPAMI.2010.184
#   proc_resid      full Procrustes residual between centerline landmarks (translation/rotation/scale removed).
#                   Goodall, "Procrustes Methods in the Statistical Analysis of Shape", JRSS-B 1991, 53(2):285-339,
#                   DOI 10.1111/j.2517-6161.1991.tb01825.x
#   curv_wave_speed propagation lag of the body-curvature profile across the 3 frames (traveling wave = Snake vs
#                   in-place twist = Screw); curv_wave_coh its peak coherence. Fang-Yen et al., PNAS 2010,
#                   107(47):20323-20328, DOI 10.1073/pnas.1003016107; Stephens et al., PLoS Comput Biol 2008,
#                   4(4):e1000028, DOI 10.1371/journal.pcbi.1000028
#   tps_bend        thin-plate-spline bending energy to warp one frame's centerline onto the next, the pure
#                   non-affine (non-rigid) bending. Bookstein, "Principal Warps: Thin-Plate Splines and the
#                   Decomposition of Deformations", IEEE TPAMI 1989, 11(6):567-585, DOI 10.1109/34.24792
#   opt_strain      mean optical-strain magnitude (symmetric gradient of dense optical flow) over the body, the
#                   non-rigid deformation an appearance flow reveals. Shreve, Godavarthy, Goldgof, Sarkar, "Macro- and
#                   micro-expression spotting in long videos using spatio-temporal strain", IEEE FG 2011, pp 51-56,
#                   DOI 10.1109/FG.2011.5771451
import numpy as np, cv2
from twist_motion_masks import _frame_state, triplet

SOTA_COLS = ['srvf_bend', 'proc_resid', 'tps_bend', 'curv_wave_speed', 'curv_wave_coh', 'opt_strain']

def _arclen(P, n=60):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-6: return None
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, P[:, 0]), np.interp(u, d, P[:, 1])], 1)

def _opt_rot(A, B):   # rotation that best maps B onto A (2D Procrustes via SVD), returns B rotated
    H = A.T @ B; U, _, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt = Vt.copy(); Vt[-1] *= -1; R = Vt.T @ U.T
    return B @ R.T

def _srvf(P):
    v = np.diff(P, axis=0); sp = np.sqrt(np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    q = v / sp; q = q - q.mean(0); n = np.linalg.norm(q) + 1e-8
    return q / n   # unit-norm SRVF, lies on the sphere

def _srvf_dist(P1, P2, n=60):
    a, b = _arclen(P1, n), _arclen(P2, n)
    if a is None or b is None: return np.nan
    q1, q2 = _srvf(a), _srvf(b); q2 = _opt_rot(q1, q2)
    return float(np.arccos(np.clip((q1 * q2).sum(), -1, 1)))   # geodesic distance on the sphere

def _proc_resid(P1, P2, n=60):
    a, b = _arclen(P1, n), _arclen(P2, n)
    if a is None or b is None: return np.nan
    a = a - a.mean(0); b = b - b.mean(0)
    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)
    b = _opt_rot(a, b)
    return float(np.sum((a - b) ** 2))   # full Procrustes residual sum of squares

def _tps_bend(P1, P2, n=24):
    a, b = _arclen(P1, n), _arclen(P2, n)
    if a is None or b is None: return np.nan
    a = a - a.mean(0); sc = np.sqrt((a ** 2).sum() / len(a)) + 1e-8; a = a / sc; b = (b - b.mean(0)) / sc
    d2 = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1); K = d2 * np.log(d2 + 1e-12); np.fill_diagonal(K, 0.0)
    Pm = np.c_[np.ones(n), a]; L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K + 1e-8 * np.eye(n); L[:n, n:] = Pm; L[n:, :n] = Pm.T
    try: sol = np.linalg.solve(L, np.r_[b, np.zeros((3, 2))])
    except np.linalg.LinAlgError: return np.nan
    W = sol[:n]
    return float(np.trace(W.T @ K @ W))   # TPS non-affine bending energy

def _wave(cp, cn):
    a = cp - cp.mean(); b = cn - cn.mean(); na = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    L = len(a); best = (-2.0, 0)
    for lag in range(-L // 4, L // 4 + 1):
        bs = np.roll(b, lag); cc = float((a * bs).sum() / na)
        if cc > best[0]: best = (cc, lag)
    return abs(best[1]) / L, best[0]   # |normalized propagation lag|, peak coherence

def _strain(g0, g1, mask):
    if g0 is None or g1 is None or g0.shape != g1.shape: return np.nan
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = fl[..., 0], fl[..., 1]
    ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, 3); uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, 3)
    vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, 3); vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(ux ** 2 + vy ** 2 + 2 * (0.5 * (uy + vx)) ** 2)   # |symmetric strain tensor|, zero for rigid motion
    m = mask > 0
    return float(mag[m].mean()) if m.sum() > 50 else np.nan

def sota_motion(m_prev, m_cur, m_next, g_prev=None, g_cur=None, g_next=None):
    # array core: 3 worm masks (0/255) and optional grayscale frames -> the SOTA descriptor dict. Used by precompute.
    nan = {c: np.nan for c in SOTA_COLS}
    sp, sc, sn = (_frame_state(m) for m in (m_prev, m_cur, m_next))
    if sc is None: return nan
    out = dict(nan); ds, pr, tb = [], [], []
    for s in (sp, sn):
        if s is not None:
            ds.append(_srvf_dist(s['Po'], sc['Po'])); pr.append(_proc_resid(s['Po'], sc['Po'])); tb.append(_tps_bend(s['Po'], sc['Po']))
    if ds and not all(np.isnan(ds)): out['srvf_bend'] = float(np.nanmean(ds))
    if pr and not all(np.isnan(pr)): out['proc_resid'] = float(np.nanmean(pr))
    if tb and not all(np.isnan(tb)): out['tps_bend'] = float(np.nanmean(tb))
    if sp is not None and sn is not None:
        ws, wc = _wave(sp['cN'], sn['cN']); out['curv_wave_speed'] = ws; out['curv_wave_coh'] = wc
    st = []
    if g_cur is not None:
        if g_prev is not None: st.append(_strain(g_prev, g_cur, m_cur))
        if g_next is not None: st.append(_strain(g_cur, g_next, m_cur))
    st = [x for x in st if not np.isnan(x)]
    if st: out['opt_strain'] = float(np.mean(st))
    return out

def sota_from_masks(basename, mask_index, image_index=None):
    nan = {c: np.nan for c in SOTA_COLS}; tn = triplet(basename)
    if tn is None: return nan
    masks, grays = [], []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        masks.append((mk > 0).astype(np.uint8) * 255)
        grays.append(cv2.imread(image_index.get(nm), cv2.IMREAD_GRAYSCALE) if (image_index and image_index.get(nm)) else None)
    return sota_motion(masks[0], masks[1], masks[2], grays[0], grays[1], grays[2])
