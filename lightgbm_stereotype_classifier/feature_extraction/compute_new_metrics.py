# -*- coding: utf-8 -*-
# %% [markdown]
# Generation of the NEW METRICS derived from the expert analysis of the hard cases (Walnut, Screw-like).
# Four families, all computed from the segmentation mask and the grayscale image.
#
# 1. Reflection symmetry (PCA principal axis): shape and intensity. The folded Walnut is a bilaterally symmetric
#    heart, whereas the compact C-shape is a chiral spiral that is not symmetric about its principal axis.
#    Reference: Zabrodsky, Peleg, Avnir, Symmetry as a Continuous Feature, IEEE TPAMI 17(12) 1154-1166, 1995,
#    DOI 10.1109/34.476508.
# 2. Topology of the interior hole (Euler number by flood-fill). A Walnut never has a central hole, whereas the
#    C-shape spiral has one. Reference: Gonzalez, Woods, Digital Image Processing, 3e ed., Pearson, 2008.
# 3. Torsion (twist): overlap that is both dark AND wide, localized along the skeleton (the body crosses itself,
#    doubled optical density, Beer-Lambert law). Skeletonization Zhang, Suen, A fast parallel algorithm for
#    thinning digital patterns, CACM 27(3) 236-239, 1984, DOI 10.1145/357994.358023.
# 4. Oscillatory alternation (screw_osc): band-limited FFT power of the centerline luminance profile. The corkscrew
#    Screw shows a periodic light/dark alternation, one per twist.
# 5. Dark-and-narrow twist (twist_darknarrow): darkness contrast AT the constriction minus the shoulders, measured on
#    the CLAHE-enhanced image. A screw twist is a self-overlap, so at the narrowest point the optical path is
#    doubled and the pixel is DARKER than its shoulders. A Snake bend narrows but WITHOUT darkening.
#    This measure corrects the initial dark-AND-wide definition, which was unsuitable because the screw crossing is narrow.
#    Local contrast enhancement (CLAHE): Zuiderveld, Contrast Limited Adaptive Histogram Equalization,
#    Graphics Gems IV, Academic Press, 1994, DOI 10.1016/B978-0-12-336156-1.50061-6. Pizer et al., Adaptive
#    histogram equalization and its variations, Computer Vision Graphics and Image Processing 39(3) 355-368, 1987,
#    DOI 10.1016/S0734-189X(87)80186-X.
#
# DEVICE NOTE: this script uses neither torch nor a GPU, only numpy, OpenCV and scikit-image on CPU. There is
# therefore no risk of an erroneous CPU/GPU loading here. random_state 42, CSV separator ';'.

# %%
# Dependency bootstrap: automatically installs the missing packages in the current environment.
import os, sys, subprocess, glob, traceback, datetime, re
def _ensure(pip_name, import_name=None):
    try:
        __import__(import_name or pip_name)
    except Exception:
        print('installing', pip_name, '...', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip_name], check=False)
for _pk, _im in [('numpy', 'numpy'), ('pandas', 'pandas'), ('opencv-python', 'cv2'),
                 ('scikit-image', 'skimage'), ('scipy', 'scipy')]:
    _ensure(_pk, _im)
import numpy as np, pandas as pd, cv2
from collections import deque
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
np.random.seed(42)
print('python =', sys.executable, flush=True)

# %% [markdown]
# Path configuration. The dataset and analysis_out are resolved from known absolute paths, then by relative
# search (compatible with Windows and WSL). The index of images and masks is then built.

# %%
HERE = (os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals()
        else os.getcwd())      # notebook case without __file__: the current folder
MASK_SUBDIR = 'mask_model_Mask2Former_20260324_0451'
PRESET = os.environ.get('FLATMORPH_PRESET', 'dataset_repaired_wormsplit_relabel_v2_test30_chook_cdense8')   # name of the split preset (environment variable FLATMORPH_PRESET)
SPLIT_SPLIT = os.environ.get('FLATMORPH_SPLITS', 'train').split(',')[0]
SPLIT_SUB = os.path.join('dataset_images_list', PRESET, PRESET + '_' + SPLIT_SPLIT + '_mask3_metrics.csv')
def _valid_dataset(p):
    return os.path.isdir(os.path.join(p, MASK_SUBDIR)) and os.path.isfile(os.path.join(p, SPLIT_SUB))
def find_dataset(start):
    rel = os.path.join('Dataset_stereotype_regeneration', 'dataset_training_stereotype')
    candidates = [p for p in [os.environ.get('FLATMORPH_DATASET', '')] if p]
    cur = start
    for _ in range(8):
        candidates.append(os.path.join(cur, rel)); cur = os.path.dirname(cur)
    for c in candidates:
        c = os.path.normpath(c)
        if _valid_dataset(c):
            return c
    raise FileNotFoundError('Dataset not found. Edit find_dataset() with the path to '
                            'dataset_training_stereotype. Candidates tried: ' + ' | '.join(candidates))
DATASET = find_dataset(HERE)
MASK_DIR = os.path.join(DATASET, MASK_SUBDIR)
def _find_analysis(start):
    for c in [os.environ.get('FLATMORPH_ANALYSIS', ''),
              os.path.join(start, 'analysis_out'), os.path.join(start, '..', 'analysis_out')]:
        if not c:
            continue
        c = os.path.normpath(c)
        if os.path.isdir(c):
            return c
    return os.path.normpath(os.path.join(start, 'analysis_out'))
ANALYSIS_DIR = _find_analysis(HERE)
print('DATASET =', DATASET)
print('MASK_DIR =', MASK_DIR)
print('ANALYSIS_DIR =', ANALYSIS_DIR)
SPLIT_BASE = os.path.join(DATASET, 'dataset_images_list', PRESET, PRESET + '_')
OUT_DIR = os.path.join(HERE, 'new_metrics_out')
os.makedirs(OUT_DIR, exist_ok=True)
GOLDEN_CSV = os.path.join(ANALYSIS_DIR, 'golden_metrics.csv')

def build_index(folder, pattern='**/*.png'):
    idx = {}
    for p in glob.glob(os.path.join(folder, pattern), recursive=True):
        bn = os.path.basename(p)
        if bn not in idx:
            idx[bn] = p
    return idx

MASK = build_index(MASK_DIR, '*.png')
IMG = build_index(os.path.join(DATASET, 'images'), '**/*.png')
print('indexed masks:', len(MASK), ' indexed images:', len(IMG))

# %% [markdown]
# Loads a (mask, grayscale) pair cropped to the worm bounding box, downscaled to about 200 px for speed.

# %%
def load_pair(bn, target=200, margin=8):
    mp = MASK.get(bn); ip = IMG.get(bn)
    if mp is None:
        return None, None
    m = cv2.imdecode(np.fromfile(mp, np.uint8), 0)
    if m is None:
        return None, None
    g = None
    if ip is not None and os.path.isfile(ip):
        gg = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if gg is not None:
            g = cv2.resize(gg, (m.shape[1], m.shape[0])) if gg.shape != m.shape else gg
    ys, xs = np.where(m > 0)
    if len(ys) < 60:
        return None, None
    y0, y1 = max(0, ys.min() - margin), min(m.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(m.shape[1], xs.max() + margin)
    m = m[y0:y1, x0:x1]; g = None if g is None else g[y0:y1, x0:x1]
    h, w = m.shape; s = float(target) / max(h, w)
    if s < 1.0:
        m = cv2.resize(m, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST)
        if g is not None:
            g = cv2.resize(g, (m.shape[1], m.shape[0]))
    return m, g

# %% [markdown]
# 1. Reflection symmetry about the principal axis (shape and intensity).

# %%
SYM = ['sym_shape_major', 'sym_shape_minor', 'sym_int_major', 'sym_int_minor']
def sym_feats(m, g):
    out = {c: np.nan for c in SYM}
    b = (m > 0).astype(np.uint8)
    if b.sum() < 150:
        return out
    ys, xs = np.where(b); pts = np.c_[xs, ys].astype(np.float64); c = pts.mean(0); p = pts - c
    cov = np.cov(p.T); w, v = np.linalg.eigh(cov); v = v[:, np.argsort(w)[::-1]]
    pr = p @ v; S = 161; sc = 0.9 * (S / 2) / (np.abs(pr).max() + 1e-6)
    q = pr * sc + S / 2; xi = np.clip(q[:, 0].astype(int), 0, S - 1); yi = np.clip(q[:, 1].astype(int), 0, S - 1)
    shp = np.zeros((S, S), bool); shp[yi, xi] = True
    def iou(A, Af):
        u = (A | Af).sum(); return float((A & Af).sum() / u) if u else 0.0
    out['sym_shape_major'] = iou(shp, shp[::-1, :]); out['sym_shape_minor'] = iou(shp, shp[:, ::-1])
    if g is not None:
        gv = g[ys, xs].astype(np.float64); gv = (gv - gv.mean()) / (gv.std() + 1e-6)
        acc = np.zeros((S, S)); cnt = np.zeros((S, S)); np.add.at(acc, (yi, xi), gv); np.add.at(cnt, (yi, xi), 1.0)
        occ = cnt > 0; gi = np.zeros((S, S)); gi[occ] = acc[occ] / cnt[occ]
        def corr(A, Af, M):
            mm = M & (Af != 0) & (A != 0)
            if mm.sum() < 40 or A[mm].std() < 1e-6 or Af[mm].std() < 1e-6:
                return 0.0
            return float(np.corrcoef(A[mm], Af[mm])[0, 1])
        out['sym_int_major'] = corr(gi, gi[::-1, :], occ); out['sym_int_minor'] = corr(gi, gi[:, ::-1], occ)
    return out

# %% [markdown]
# 2. Topology of the interior hole (flood-fill of the background from the border) and convex hull deficit.

# %%
HOLE = ['hole_frac', 'cavity_frac']
def hole_feats(m):
    out = {c: np.nan for c in HOLE}
    b = (m > 0).astype(np.uint8)
    if b.sum() < 150:
        return out
    b2 = cv2.copyMakeBorder(b, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = b2.shape; ff = b2.copy(); mask = np.zeros((h + 2, w + 2), np.uint8); cv2.floodFill(ff, mask, (0, 0), 1)
    holes = int((ff == 0).sum()); area = int(b2.sum())
    cs, _ = cv2.findContours(b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c); hm = np.zeros_like(b2); cv2.drawContours(hm, [hull], -1, 1, -1)
    cavity = int(((hm > 0) & (b2 == 0)).sum())
    out['hole_frac'] = holes / area; out['cavity_frac'] = cavity / area
    return out

# %% [markdown]
# Skeleton: longest geodesic path (double BFS) to traverse the midline.

# %%
def longest_path(sk):
    ys, xs = np.where(sk); nodes = set(zip(ys.tolist(), xs.tolist()))
    if len(nodes) < 20:
        return []
    adj = {(y, x): [nn for nn in [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)]
                    if nn in nodes] for (y, x) in nodes}
    def far(s):
        d = {s: 0}; pr = {s: None}; q = deque([s]); last = s
        while q:
            u = q.popleft(); last = u
            for vv in adj[u]:
                if vv not in d:
                    d[vv] = d[u] + 1; pr[vv] = u; q.append(vv)
        return last, pr
    a, _ = far(next(iter(nodes))); b, par = far(a); path = []; cc = b
    while cc is not None:
        path.append(cc); cc = par[cc]
    return path

# %% [markdown]
# 3. Torsion (twist) and 4. Oscillatory alternation, from the centerline luminance profile.

# %%
TW = ['twist_peak', 'twist_frac', 'twist_conc', 'screw_osc', 'screw_nalt', 'waist_full', 'kink_full', 'waist_abrupt',
      'twist_darknarrow', 'twist_dn_clahe', 'twist_dn_raw', 'twist_ws', 'waist_ws', 'dark_ws',
      'c_net_turn', 'c_mono', 'c_cwind', 'c_curv_signchg']
def twist_osc_feats(m, g):
    out = {c: np.nan for c in TW}
    b = (m > 0).astype(np.uint8)
    if b.sum() < 150 or g is None:
        return out
    dt = cv2.distanceTransform(b, cv2.DIST_L2, 5); sk = skeletonize(b > 0); path = longest_path(sk)
    if len(path) < 25:
        return out
    L = len(path); gg = cv2.GaussianBlur(g, (0, 0), 1.1).astype(np.float32)
    v = gg[b > 0]; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    gn = np.clip((gg - lo) / (hi - lo + 1e-6), 0, 1)
    lum = np.array([gn[y, x] for (y, x) in path]); wid = np.array([dt[y, x] * 2 for (y, x) in path])
    k = max(2, L // 28); sm = np.convolve(lum, np.ones(k) / k, 'same'); wn = wid / (np.median(wid) + 1e-6)
    medl = np.median(lum); sdl = np.std(lum) + 1e-6
    prod = np.clip((medl - sm) / sdl, 0, None) * np.clip(wn - 1.1, 0, None)
    out['twist_peak'] = float(prod.max()); out['twist_frac'] = float((prod > 0.15).mean())
    out['twist_conc'] = float(prod.max() / (prod.mean() + 1e-6))
    t = np.arange(L); det = sm - np.polyval(np.polyfit(t, sm, 1), t); detn = det / (det.std() + 1e-6)
    f = np.abs(np.fft.rfft(detn * np.hanning(L))) ** 2; f = f / (f.sum() + 1e-9)
    pk, _ = find_peaks(detn, prominence=0.5, distance=max(3, L // 20))
    vl, _ = find_peaks(-detn, prominence=0.5, distance=max(3, L // 20))
    out['screw_osc'] = float(f[2:max(3, L // 8)].sum()); out['screw_nalt'] = int(len(pk) + len(vl))
    # waist_full (dark constriction = rotation) and kink_full (bend) on the FULL skeleton, ends included.
    # The Screw twist is often near the head, so the terminal portions are no longer excluded (lo=4% only).
    lo_f = 0.04; a = int(L * lo_f); z = int(L * (1 - lo_f))
    narrow = np.clip(1.0 - wn, 0, None); dark = np.clip((medl - sm) / sdl, 0, None)
    out['waist_full'] = float((dark * narrow)[a:z].max()) if z > a else 0.0
    P = np.array(path, float); kk = max(2, L // 15); ang = np.zeros(L)
    for i in range(L):
        u = max(0, i - kk); w2 = min(L - 1, i + kk); d = P[w2] - P[u]; ang[i] = np.arctan2(d[0], d[1])
    # C-SHAPE DESCRIPTOR (integrated curvature, expert lead): a tight C winds in ONE direction only, so its
    # NET winding (signed sum of the angle variations) is large and close to its TOTAL winding (monotonicity ~1),
    # even if the cumulative angle exceeds 180 degrees (closed C). A screw or an S changes direction (low monotonicity,
    # high signchg). c_net_turn in degrees, c_mono = net/total in [0,1], c_curv_signchg = number of curvature reversals.
    uang = np.unwrap(ang); dsum = np.diff(uang)
    c_total = float(np.sum(np.abs(dsum))) + 1e-6; c_net = float(abs(np.sum(dsum)))
    out['c_net_turn'] = c_net * 180.0 / np.pi; out['c_mono'] = c_net / c_total
    out['c_cwind'] = (out['c_net_turn'] / 180.0) * out['c_mono']  # COMBINATION: coherent net winding (a single metric)
    sm2 = np.convolve(dsum, np.ones(kk) / kk, 'same'); ss = np.sign(sm2); ss = ss[ss != 0]
    out['c_curv_signchg'] = int(np.sum(ss[1:] != ss[:-1])) if len(ss) > 1 else 0
    curv = np.zeros(L)
    for i in range(L):
        u = max(0, i - kk); w2 = min(L - 1, i + kk); da = ang[w2] - ang[u]; da = (da + np.pi) % (2 * np.pi) - np.pi; curv[i] = abs(da)
    cn = curv / (np.median(curv) + 1e-6)
    out['kink_full'] = float((narrow * cn)[a:z].max()) if z > a else 0.0
    # waist_abrupt: ABRUPT hourglass (width constriction that rises again on BOTH sides within a SHORT window) =
    # signature of the Screw twist (rotation), distinct from the PROGRESSIVE width reduction of the Snake (bend/taper).
    smw = np.convolve(wn, np.ones(max(2, L // 40)) / max(2, L // 40), 'same')
    winh = max(3, L // 12); mins, _ = find_peaks(-smw, prominence=0.05); wa = 0.0; wi_best = None
    for ii in mins:
        if ii < L * 0.06 or ii > L * 0.94:
            continue
        aa = max(0, ii - winh); zz = min(L - 1, ii + winh)
        lr = smw[aa:ii].max() - smw[ii] if ii > aa else 0.0
        rr = smw[ii + 1:zz + 1].max() - smw[ii] if zz > ii else 0.0
        val = min(lr, rr)
        if val > wa:
            wa = val; wi_best = ii
    out['waist_abrupt'] = float(wa)
    # twist_darknarrow: at the abrupt constriction, does the darkness stand out MORE than on the shoulders?
    # Positive = screw crossing (darker self-overlap). ~0 or negative = simple Snake thinning.
    # Measured on the raw image AND on the CLAHE-enhanced image (local contrast boosted for thick worms).
    def _dark_contrast(gray, wi):
        bgp = float(np.median(gray[b == 0])) if (b == 0).any() else float(np.median(gray))
        dk = np.clip((bgp - gray.astype(np.float32)) / (bgp + 1e-6), 0.0, 1.0)
        dprof = np.array([dk[y, x] for (y, x) in path])
        sdc = np.convolve(dprof, np.ones(k) / k, 'same')
        a2 = max(0, wi - 2); z2 = min(L, wi + 3); dwv = float(sdc[a2:z2].mean())
        off = winh
        ls = sdc[max(0, wi - 2 * off):max(1, wi - off)]; rs = sdc[min(L - 1, wi + off):min(L, wi + 2 * off)]
        sh = np.concatenate([x for x in [ls, rs] if len(x)])
        dsh = float(sh.mean()) if len(sh) else dwv
        return float(dwv - dsh)
    if wi_best is not None:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gc = clahe.apply(np.clip(g, 0, 255).astype(np.uint8))
        except Exception:
            gc = np.clip(g, 0, 255).astype(np.uint8)
        out['twist_dn_raw'] = _dark_contrast(g, wi_best)
        out['twist_dn_clahe'] = _dark_contrast(gc, wi_best)
        out['twist_darknarrow'] = out['twist_dn_clahe']
    else:
        out['twist_dn_raw'] = 0.0; out['twist_dn_clahe'] = 0.0; out['twist_darknarrow'] = 0.0
    # --- twist ADAPTED TO THE WORM SCALE: windows set on the body WIDTH, not on the length ---
    # A local twist on a long worm covers only a small fraction of the skeleton. Searching at the scale
    # of the width (about 1.5 body widths) avoids diluting it. We measure the local width dip (in
    # width units) and the local darkness excess (CLAHE) versus the shoulders, both scale-invariant.
    mw = float(np.median(wid)) + 1e-6
    winw = int(max(3, round(1.5 * mw))); kk2 = int(max(2, round(0.6 * mw)))
    try:
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)); gcl = clahe2.apply(np.clip(g, 0, 255).astype(np.uint8))
    except Exception:
        gcl = np.clip(g, 0, 255).astype(np.uint8)
    bgw = float(np.median(gcl[b == 0])) if (b == 0).any() else float(np.median(gcl))
    darkw = np.clip((bgw - gcl.astype(np.float32)) / (bgw + 1e-6), 0.0, 1.0)
    dwprof = np.array([darkw[y, x] for (y, x) in path])
    sww = np.convolve(wid, np.ones(kk2) / kk2, 'same'); sdw = np.convolve(dwprof, np.ones(kk2) / kk2, 'same')
    tw_ws = 0.0; waist_ws = 0.0; dark_ws = 0.0
    for i in range(winw, L - winw):
        wl = sww[i - winw:i].max(); wr = sww[i + 1:i + winw + 1].max()
        dip = (min(wl, wr) - sww[i]) / mw
        dl = sdw[i - winw:i].mean(); dr = sdw[i + 1:i + winw + 1].mean(); dexc = sdw[i] - 0.5 * (dl + dr)
        if dip > waist_ws: waist_ws = dip
        if dexc > dark_ws: dark_ws = dexc
        v = max(0.0, dip) * max(0.0, dexc)
        if v > tw_ws: tw_ws = v
    out['twist_ws'] = float(tw_ws); out['waist_ws'] = float(waist_ws); out['dark_ws'] = float(dark_ws)
    return out

# %% [markdown]
# 5. TEMPORAL metric (3-image window): elongation variation between the image and its immediate neighbours.
# A Snake UNDULATES, so its elongation (skeleton length / median width) VARIES from one image to the next, whereas
# a Wildtype glides more steadily. Weak signal, but oriented towards Snake-vs-Wildtype. Neighbours are looked up in
# the mask index (same worm, frame +/- 1). t_elong_var = relative amplitude, t_elong_std = standard deviation.

# %%
TEMP = ['t_elong_var', 't_elong_std']
def _elong_from_mask(bn):
    mp = MASK.get(bn)
    if mp is None:
        return None
    m = cv2.imdecode(np.fromfile(mp, np.uint8), 0)
    if m is None:
        return None
    ys, xs = np.where(m > 0)
    if len(ys) < 60:
        return None
    y0, y1 = max(0, ys.min() - 8), min(m.shape[0], ys.max() + 8)
    x0, x1 = max(0, xs.min() - 8), min(m.shape[1], xs.max() + 8)
    b = (m[y0:y1, x0:x1] > 0).astype(np.uint8)
    if max(b.shape) > 200:
        s = 200.0 / max(b.shape)
        b = cv2.resize(b, (max(1, int(b.shape[1] * s)), max(1, int(b.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(b, cv2.DIST_L2, 5); sk = skeletonize(b > 0); path = longest_path(sk)
    if len(path) < 15:
        return None
    wid = np.array([dt[y, x] * 2 for (y, x) in path])
    return len(path) / (np.median(wid) + 1e-6)
def temporal_feats(bn):
    out = {c: np.nan for c in TEMP}
    mm = re.match(r'(.+)_Frame_(\d+)_in', bn)
    if not mm:
        return out
    worm, fr = mm.group(1), int(mm.group(2))
    vals = []
    for f in (fr - 1, fr, fr + 1):
        e = _elong_from_mask('%s_Frame_%d_in_30.png' % (worm, f))
        if e is not None:
            vals.append(e)
    if len(vals) >= 2:
        vals = np.array(vals); mean = vals.mean() + 1e-6
        out['t_elong_var'] = float((vals.max() - vals.min()) / mean)
        out['t_elong_std'] = float(vals.std())
    return out

# %% [markdown]
# Main loop: for each split, the image column is read and a CSV of the new metrics is written.

# %%
ALL_COLS = SYM + HOLE + TW + TEMP
def compute_for_csv(in_csv, out_csv):
    d = pd.read_csv(in_csv, sep=';')
    col = 'image' if 'image' in d.columns else d.columns[0]
    names = [os.path.basename(str(x)) for x in d[col]]
    rows = []
    for i, bn in enumerate(names):
        m, g = load_pair(bn)
        r = {c: np.nan for c in ALL_COLS}
        if m is not None:
            r.update(sym_feats(m, g)); r.update(hole_feats(m)); r.update(twist_osc_feats(m, g))
        r.update(temporal_feats(bn))
        rows.append(r)
        if (i + 1) % 500 == 0:
            print('  ', i + 1, '/', len(names), flush=True)
    out = pd.DataFrame(rows); out.insert(0, 'image', names)
    out.to_csv(out_csv, sep=';', index=False)
    ok = int(out['twist_peak'].notna().sum())
    print('written', os.path.basename(out_csv), len(out), 'rows,', ok, 'with skeleton', flush=True)
    return out

# %%
def _run_all():
    log_path = os.path.join(OUT_DIR, 'run_log.txt')
    lines = ['run compute_new_metrics ' + datetime.datetime.now().isoformat(),
             'python = ' + sys.executable, 'HERE = ' + HERE, 'DATASET = ' + DATASET,
             'MASK_DIR = ' + MASK_DIR, 'masks = %d , images = %d' % (len(MASK), len(IMG)),
             'OUT_DIR = ' + OUT_DIR]
    try:
        for sp in [s for s in os.environ.get('FLATMORPH_SPLITS', 'train,valid,test').split(',') if s]:   # FLATMORPH_SPLITS, default train,valid,test
            in_csv = SPLIT_BASE + sp + '_mask3_metrics.csv'
            if os.path.isfile(in_csv):
                print('=== split', sp, '===', flush=True)
                out = compute_for_csv(in_csv, os.path.join(OUT_DIR, 'new_metrics_' + sp + '.csv'))
                lines.append('split %s : %d rows written' % (sp, len(out)))
            else:
                msg = 'CSV not found for %s : %s' % (sp, in_csv)
                print(msg, flush=True); lines.append(msg)
        if os.path.isfile(GOLDEN_CSV):
            print('=== golden ===', flush=True)
            out = compute_for_csv(GOLDEN_CSV, os.path.join(OUT_DIR, 'new_metrics_golden.csv'))
            lines.append('golden : %d rows written' % len(out))
        else:
            lines.append('golden CSV missing : ' + GOLDEN_CSV)
        lines.append('FINISHED OK')
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True); lines.append('ERROR :\n' + tb)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('log written:', log_path, flush=True)
    except Exception:
        pass

# In a notebook as in a script, __name__ equals '__main__', so execution is triggered in both cases.
if __name__ == '__main__':
    _run_all()
