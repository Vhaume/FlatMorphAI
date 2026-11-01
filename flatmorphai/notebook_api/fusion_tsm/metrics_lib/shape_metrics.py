"""shape_metrics.py -- a comprehensive, scale-free SOTA shape battery for the planarian stereotype classes,
computed from a single binary mask (T0) and across the 3-frame mask window (T-1, T0, T+1) as change/oscillation.

It targets shapes the existing metric set does not encode well: the rough, wavy, multi-turn Screw-like outline
versus the single smooth bend of C-shape, and analogous shape cues for Snake-like, Walnut and Wildtype. All
features are normalised by sqrt(area) or are dimensionless, so there is no pixel-to-millimetre dependence.

Backing papers (each function notes the one it implements):
  Mokhtarian F, Mackworth A K. A theory of multiscale, curvature-based shape representation for planar curves.
    IEEE TPAMI, 1992. DOI 10.1109/34.149591.                       (boundary curvature, zero-crossings)
  Zahn C T, Roskies R. Fourier Descriptors for Plane Closed Curves. IEEE Trans. Computers, 1972.
    DOI 10.1109/TC.1972.5008949.                                    (turning-function and signature waviness)
  Kuhl F P, Giardina C R. Elliptic Fourier features of a closed contour. Computer Graphics and Image
    Processing, 1982. DOI 10.1016/0146-664X(82)90034-X.             (elliptic Fourier descriptors)
  Teague M R. Image analysis via the general theory of moments. J. Opt. Soc. Am., 1980.
    DOI 10.1364/JOSA.70.000920.                                     (Zernike moments, via mahotas if present)
  Zhang T Y, Suen C Y. A fast parallel algorithm for thinning digital patterns. Commun. ACM, 1984.
    DOI 10.1145/357994.358023.                                      (skeleton / medial axis)
  Stephens G J, Johnson-Kerner B, Bialek W, Ryu W S. Dimensionality and Dynamics in the Behavior of
    C. elegans. PLoS Comput. Biol., 2008. DOI 10.1371/journal.pcbi.1000028.  (midline curvature for worms)
  Fornberg B. Generation of finite difference formulas on arbitrarily spaced grids. Mathematics of
    Computation, 1988. DOI 10.1090/S0025-5718-1988-0935077-0.       (centred/one-sided temporal differences)
"""
import numpy as np, cv2
from collections import deque

try:
    from skimage.morphology import skeletonize as _skeletonize
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False
try:
    import mahotas as _mahotas
    _HAVE_MH = True
except Exception:
    _HAVE_MH = False

N_BND = 256; N_RAD = 128; N_MID = 48; N_EFD = 6; N_ZERN = 9

# base feature names (no temporal suffix). Kept fixed so the column set never changes.
BASE = [
    'convexity', 'defdepth_mean',
    'bcurv_absmean', 'bcurv_std', 'bcurv_max', 'bcurv_p90', 'bcurv_entropy', 'bcss_zcross',
    'bturn_hf', 'bturn_mf', 'bturn_centroid',
    'rad_rough', 'rad_hf', 'rad_entropy',
    'efd_h2', 'efd_h3', 'efd_h4', 'efd_h5', 'efd_h6', 'efd_hf',
    'mcurv_absmean', 'mcurv_max', 'mbend_energy', 'mninflect', 'mtotturn', 'mtort',
    'tang_mode1_frac', 'tang_hf', 'tang_domfreq', 'tang_entropy',
    'width_entropy',
] + ['zern_%d' % i for i in range(N_ZERN)]
SHAPE_T0_COLS = ['sh_%s_T0' % k for k in BASE]
SHAPE_DYN_COLS = []   # temporal shape deltas were <=0.63 AUC for every class in the POC -> dropped
ALL_SHAPE_COLS = SHAPE_T0_COLS + SHAPE_DYN_COLS

# ---------- geometry helpers ----------
def _largest(m):
    n, lab = cv2.connectedComponents((m > 0).astype(np.uint8))
    if n <= 1: return (m > 0).astype(np.uint8)
    s = [(lab == i).sum() for i in range(1, n)]
    return (lab == (1 + int(np.argmax(s)))).astype(np.uint8)

def _smooth_periodic(a, sigma):
    n = len(a); r = max(1, int(3 * sigma)); k = np.arange(-r, r + 1)
    w = np.exp(-(k ** 2) / (2 * sigma ** 2)); w /= w.sum()
    return np.convolve(np.concatenate([a[-r:], a, a[:r]]), w, 'same')[r:r + n]

def _smooth_open(a, sigma):
    n = len(a); r = max(1, int(3 * sigma)); k = np.arange(-r, r + 1)
    w = np.exp(-(k ** 2) / (2 * sigma ** 2)); w /= w.sum()
    return np.convolve(np.concatenate([np.full(r, a[0]), a, np.full(r, a[-1])]), w, 'same')[r:r + n]

def _resample_closed(pts, n):
    p = pts.astype(np.float64); p = np.vstack([p, p[:1]])
    d = np.sqrt((np.diff(p, axis=0) ** 2).sum(1)); s = np.concatenate([[0], np.cumsum(d)]); L = s[-1]
    if L < 1e-6: return None
    u = np.linspace(0, L, n, endpoint=False)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)

def _curv_closed(p, sigma=3.0):
    x = _smooth_periodic(p[:, 0], sigma); y = _smooth_periodic(p[:, 1], sigma)
    dx = (np.roll(x, -1) - np.roll(x, 1)) / 2; dy = (np.roll(y, -1) - np.roll(y, 1)) / 2
    ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1); ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
    return (dx * ddy - dy * ddx) / ((dx * dx + dy * dy) ** 1.5 + 1e-9)

def _entropy(v, bins=16):
    v = v[np.isfinite(v)]
    if len(v) < 4: return np.nan
    h, _ = np.histogram(v, bins=bins); p = h / (h.sum() + 1e-12); p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _spec_bands(sig):
    # FFT energy bands of a periodic signal (Zahn & Roskies 1972 waviness; Fornberg 1988 differencing context)
    F = np.abs(np.fft.rfft(sig - np.mean(sig))) ** 2; tot = F[1:].sum() + 1e-12
    hf = float(F[8:].sum() / tot); mf = float(F[4:8].sum() / tot)
    ks = np.arange(len(F)); cen = float((ks[1:] * F[1:]).sum() / (F[1:].sum() + 1e-12))
    ent = _entropy(F[1:] / (F[1:].sum() + 1e-12) * 1000.0)
    return hf, mf, cen, ent

# ---------- skeleton path ----------
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def _skel_path(sk):
    pts = [tuple(p) for p in np.argwhere(sk)]; S = set(pts)
    if len(pts) < 6: return None, 0, 0
    adj = {p: [q for q in ((p[0] + dy, p[1] + dx) for dy, dx in _NB) if q in S] for p in pts}
    deg = {p: len(v) for p, v in adj.items()}
    nbranch = int(sum(1 for d in deg.values() if d >= 3)); nend = int(sum(1 for d in deg.values() if d == 1))
    def bfs(s):
        seen = {s: None}; dq = deque([s]); last = s
        while dq:
            u = dq.popleft(); last = u
            for v in adj[u]:
                if v not in seen: seen[v] = u; dq.append(v)
        return last, seen
    a, _ = bfs(next(iter(S))); b, par = bfs(a); path = []; cur = b
    while cur is not None: path.append(cur); cur = par[cur]
    return path[::-1], nbranch, nend

def _seg_cross(P):
    # count self-intersections of the midline polyline (helical screws self-cross; a single bend does not)
    def ccw(a, b, c): return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    n = len(P); c = 0
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j == n - 2: continue
            A, B, C, D = P[i], P[i + 1], P[j], P[j + 1]
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D): c += 1
    return c

# ---------- elliptic Fourier descriptors (Kuhl & Giardina 1982) ----------
def _efd(pts, n_harm=N_EFD):
    dxy = np.diff(np.vstack([pts, pts[:1]]), axis=0); dt = np.sqrt((dxy ** 2).sum(1)); dt[dt == 0] = 1e-9
    t = np.concatenate([[0], np.cumsum(dt)]); T = t[-1]
    amps = []
    for n in range(1, n_harm + 1):
        c = 2 * n * np.pi / T
        a = (T / (2 * n * n * np.pi ** 2)) * np.sum((dxy[:, 0] / dt) * (np.cos(c * t[1:]) - np.cos(c * t[:-1])))
        b = (T / (2 * n * n * np.pi ** 2)) * np.sum((dxy[:, 0] / dt) * (np.sin(c * t[1:]) - np.sin(c * t[:-1])))
        cc = (T / (2 * n * n * np.pi ** 2)) * np.sum((dxy[:, 1] / dt) * (np.cos(c * t[1:]) - np.cos(c * t[:-1])))
        d = (T / (2 * n * n * np.pi ** 2)) * np.sum((dxy[:, 1] / dt) * (np.sin(c * t[1:]) - np.sin(c * t[:-1])))
        amps.append(np.sqrt(a * a + b * b + cc * cc + d * d))
    amps = np.array(amps); e1 = amps[0] + 1e-12
    return amps / e1   # size-normalised harmonic amplitudes; amp[0]==1

# ---------- the battery on one mask ----------
def shape_features(mask):
    out = {k: np.nan for k in BASE}
    if mask is None: return out
    m = _largest(mask)
    if m.sum() < 50: return out
    area = float(m.sum()); scale = np.sqrt(area) + 1e-6
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return out
    c = max(cnts, key=cv2.contourArea); per = float(cv2.arcLength(c, True))
    if per < 20 or cv2.contourArea(c) < 50: return out
    hull = cv2.convexHull(c); hper = float(cv2.arcLength(hull, True))
    out['convexity'] = hper / (per + 1e-9)
    try:
        hi = cv2.convexHull(c, returnPoints=False)
        dfx = cv2.convexityDefects(c, hi) if (hi is not None and len(hi) > 3) else None
        if dfx is not None:
            dep = dfx.reshape(-1, 4)[:, 3] / 256.0; dep = dep[dep > 0.04 * scale]
            out['defdepth_mean'] = float(dep.mean() / scale) if len(dep) else 0.0
        else: out['defdepth_mean'] = 0.0
    except Exception: out['defdepth_mean'] = np.nan
    # boundary curvature + turning waviness (Mokhtarian-Mackworth 1992; Zahn-Roskies 1972)
    rp = _resample_closed(c[:, 0, :], N_BND)
    if rp is not None:
        k = _curv_closed(rp, 3.0); ak = np.abs(k)
        out['bcurv_absmean'] = float(ak.mean()); out['bcurv_std'] = float(ak.std())
        out['bcurv_max'] = float(ak.max()); out['bcurv_p90'] = float(np.percentile(ak, 90))
        out['bcurv_entropy'] = _entropy(ak)
        ks = np.sign(_smooth_periodic(k, 4.0)); out['bcss_zcross'] = float(np.sum(ks != np.roll(ks, 1)))
        hf, mf, cen, _ = _spec_bands(k)
        out['bturn_hf'] = hf; out['bturn_mf'] = mf; out['bturn_centroid'] = cen
        # elliptic Fourier descriptors (Kuhl & Giardina 1982)
        try:
            amp = _efd(rp, N_EFD)
            for i in range(2, N_EFD + 1): out['efd_h%d' % i] = float(amp[i - 1])
            out['efd_hf'] = float(amp[3:].sum())
        except Exception: pass
    # centroid-distance signature waviness (Zahn-Roskies 1972 framework)
    M = cv2.moments(m)
    if M['m00'] > 0 and rp is not None:
        cy, cx = M['m01'] / M['m00'], M['m10'] / M['m00']
        d = np.hypot(rp[:, 0] - cx, rp[:, 1] - cy)
        if len(d) >= N_RAD: d = np.interp(np.linspace(0, len(d), N_RAD, endpoint=False), np.arange(len(d)), d)
        mu = d.mean() + 1e-9; out['rad_rough'] = float(d.std() / mu)
        hf, mf, cen, ent = _spec_bands(d); out['rad_hf'] = hf; out['rad_entropy'] = ent
    # midline curvature + self-crossing (Stephens 2008; Zhang-Suen 1984)
    if _HAVE_SK:
        sk = _skeletonize(m > 0); path, nbr, nend = _skel_path(sk)
        if path is not None and len(path) >= 12:
            yy = np.array([p[0] for p in path], float); xx = np.array([p[1] for p in path], float)
            xx = _smooth_open(xx, 2.0); yy = _smooth_open(yy, 2.0)
            seg = np.hypot(np.diff(xx), np.diff(yy)); Larc = float(seg.sum()) + 1e-6
            dx = np.gradient(xx); dy = np.gradient(yy); ddx = np.gradient(dx); ddy = np.gradient(dy)
            kk = (dx * ddy - dy * ddx) / ((dx * dx + dy * dy) ** 1.5 + 1e-9); ak = np.abs(kk)
            out['mcurv_absmean'] = float(ak.mean()); out['mcurv_max'] = float(ak.max())
            out['mbend_energy'] = float((kk * kk).mean() * Larc)
            sg = np.sign(_smooth_open(kk, 2.0)); out['mninflect'] = float(np.sum(sg[1:] != sg[:-1]))
            th = np.unwrap(np.arctan2(dy, dx)); out['mtotturn'] = float(np.abs(np.diff(th)).sum())
            end = np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]); out['mtort'] = float(Larc / (end + 1e-6))
            prof = th - th.mean(); Pw = np.abs(np.fft.rfft(prof)) ** 2; tw = Pw[1:].sum() + 1e-12
            out['tang_mode1_frac'] = float(Pw[1] / tw); out['tang_hf'] = float(Pw[3:].sum() / tw)
            out['tang_domfreq'] = float(1 + int(np.argmax(Pw[1:]))); out['tang_entropy'] = _entropy(Pw[1:] / tw * 1000.0)
        # medial width distribution entropy (distance transform along the skeleton)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5); wv = dt[sk] if sk.sum() else dt[m > 0]
        out['width_entropy'] = _entropy(wv / (wv.mean() + 1e-9))
    # Zernike moments (Teague 1980) -- rotation-invariant region descriptor
    if _HAVE_MH:
        try:
            ys, xs = np.where(m); R = float(max(np.hypot(ys - ys.mean(), xs - xs.mean()).max(), 1.0))
            z = _mahotas.features.zernike_moments(m.astype(bool), R, degree=8)
            for i in range(min(N_ZERN, len(z))): out['zern_%d' % i] = float(z[i])
        except Exception: pass
    return out

# ---------- public API ----------
def shape_T0(mask):
    f = shape_features(mask); return {'sh_%s_T0' % k: f[k] for k in BASE}

def shape_dyn(mA, mB, mC):
    # temporal shape deltas were proven useless in the POC (<=0.63 AUC), so this is intentionally empty.
    return {}

def shape_dyn_from_feats(fA, fB, fC):
    # same as shape_dyn but when the per-frame base dicts are already computed (avoids recompute)
    out = {}
    for k in BASE:
        v = np.array([d.get(k, np.nan) for d in (fA, fB, fC)], float); v = v[np.isfinite(v)]
        out['sh_%s_chg' % k] = float(np.mean(np.abs(np.diff(v)))) if len(v) > 1 else np.nan
        out['sh_%s_osc' % k] = float(np.std(v)) if len(v) > 1 else np.nan
    return out
