# twist_border_evolution.py
# Border / body evolution measured along the worm's OWN centerline, not a straight axis, so it is correct for the
# curved worms seen in the masks. Skeletonize the mask, trace the longest centerline path (head to tail), and at
# each point take the true cross-section width from the distance transform and the midline turning angle. Tracks
# the pointed features: hourglass neck (width minimum), bigger middle (width ratio), sharp V / 90-degree turn (max
# midline curvature), border waving (curvature oscillation), rounds-on-itself (skeleton branch points), thin curved
# tail (thin + bent centerline end), periodic helical banding (FFT of width/curvature profile, Zahn & Roskies IEEE
# Trans Computers 1972 DOI 10.1109/TC.1972.5008949), left/right border asymmetry (signed radius split, medial axis
# Blum 1967), and border V-notch concavity (convexity defects). Prefix be_. Scale-free, no GPU.
# Basis: medial axis / distance transform (Blum 1967, no DOI), curvature-scale-space (Mokhtarian & Mackworth, IEEE
# TPAMI 1992, DOI 10.1109/34.149591), morphological skeleton (Zhang & Suen, CACM 1984, DOI 10.1145/357994.358023).
import numpy as np, cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.signal import find_peaks

BE_COLS = ['be_neck_depth', 'be_hourglass', 'be_neck_len', 'be_mid_end_ratio', 'be_width_cv',
           'be_max_turn', 'be_total_turn', 'be_wav', 'be_self_overlap', 'be_path_len',
           'be_axial_dark_grad', 'be_axial_tex_step', 'be_dark_core', 'be_darkwide',
           'be_tail_thin', 'be_tail_curve', 'be_tail_taper', 'be_width_fft', 'be_curv_fft',
           'be_n_necks', 'be_n_inflect', 'be_lr_asym', 'be_lr_bias', 'be_concave_depth', 'be_n_concave', 'be_curv_tail_rise', 'be_cross_std', 'be_cross_lrasym', 'be_tortuosity', 'be_bend_energy', 'be_cross_lr_flips', 'be_symmetry', 'be_roundness', 'be_border_wav', 'be_width_grad', 'be_uniform_run', 'be_n_sharp', 'be_head_curv', 'be_ellipse_fill']

def _longest_path(skel):
    ys, xs = np.where(skel)
    if len(ys) < 8: return None
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    nb = [[] for _ in range(len(ys))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                j = idx.get((y + dy, x + dx))
                if j is not None: nb[i].append(j)
    def bfs(src):
        dist = [-1] * len(nb); par = [-1] * len(nb); dist[src] = 0; q = [src]
        while q:
            nq = []
            for u in q:
                for v in nb[u]:
                    if dist[v] < 0: dist[v] = dist[u] + 1; par[v] = u; nq.append(v)
            q = nq
        far = int(np.argmax(dist)); return far, dist, par
    a, _, _ = bfs(0); b, dist, par = bfs(a)
    path = []; u = b
    while u != -1: path.append(u); u = par[u]
    return [(int(ys[i]), int(xs[i])) for i in path]

def _fft_peak(profile):
    # fraction of spectral power in the strongest non-DC low frequency band (periodic banding signature)
    p = profile - profile.mean()
    if p.std() < 1e-6 or len(p) < 16: return 0.0
    f = np.abs(np.fft.rfft(p * np.hanning(len(p)))) ** 2
    if f[1:].sum() < 1e-9: return 0.0
    return float(f[1:6].max() / (f[1:].sum() + 1e-9))

def _lr_radii(worm, P, k):
    # signed left/right distance from centerline to border along the local normal
    H, W = worm.shape; left = []; right = []
    for i in range(k, len(P) - k):
        t = P[i + k] - P[i - k]; n = np.array([-t[1], t[0]], np.float32); nn = np.linalg.norm(n)
        if nn < 1e-6: continue
        n /= nn; y0, x0 = P[i]
        d = [0.0, 0.0]
        for s, sign in ((0, 1.0), (1, -1.0)):
            for r in range(1, 60):
                yy = int(round(y0 + sign * n[0] * r)); xx = int(round(x0 + sign * n[1] * r))
                if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0: break
                d[s] = r
        left.append(d[0]); right.append(d[1])
    return np.array(left, np.float32), np.array(right, np.float32)

def _border_evolution_impl(gray, return_debug=False):
    out = {c: np.nan for c in BE_COLS}; _dbg = None
    if gray is None or gray.ndim != 2: return (out, _dbg) if return_debug else out
    mask = (gray > 8).astype(np.uint8)
    if mask.sum() < 200: return (out, _dbg) if return_debug else out
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: return (out, _dbg) if return_debug else out
    worm = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    edt = distance_transform_edt(worm)
    skel = skeletonize(worm > 0)
    sk = skel.astype(np.uint8); kern = np.ones((3, 3), np.uint8); ncount = cv2.filter2D(sk, -1, kern) - sk
    out['be_self_overlap'] = float(((sk > 0) & (ncount >= 3)).sum())
    # WALNUT cue: symmetrical compact round shape. be_symmetry = mean IoU of the centred, axis-aligned mask with its
    # reflection about each principal axis (round symmetric blob ~ 1). be_roundness = area / min-enclosing-circle area
    # (compact blob ~ 1, elongated worm low). Both scale-free.
    ys0, xs0 = np.where(worm); out['be_symmetry'] = np.nan; out['be_roundness'] = np.nan
    if len(ys0) > 30:
        yy0 = ys0 - ys0.mean(); xx0 = xs0 - xs0.mean()
        ev, evec = np.linalg.eigh(np.cov(np.stack([xx0, yy0]).astype(np.float64))); ax = evec[:, -1]; ang = np.arctan2(ax[1], ax[0])
        ca, sa = np.cos(-ang), np.sin(-ang); rx = ca * xx0 - sa * yy0; ry = sa * xx0 + ca * yy0
        sc = 58.0 / (max(np.abs(rx).max(), np.abs(ry).max()) + 1e-6)
        gx = np.clip((rx * sc + 64).astype(int), 0, 127); gy = np.clip((ry * sc + 64).astype(int), 0, 127)
        cv0 = np.zeros((128, 128), np.uint8); cv0[gy, gx] = 1
        def _iou(a, b): return float((a & b).sum()) / (float((a | b).sum()) + 1e-6)
        out['be_symmetry'] = 0.5 * (_iou(cv0, np.flipud(cv0)) + _iou(cv0, np.fliplr(cv0)))
        _, rad = cv2.minEnclosingCircle(np.stack([xs0, ys0], 1).astype(np.int32))
        out['be_roundness'] = float(worm.sum()) / (np.pi * rad * rad + 1e-6)
    # border concavity (sharp V-notch at the border) via convexity defects
    out['be_concave_depth'] = 0.0; out['be_n_concave'] = 0.0; out['be_border_wav'] = np.nan; out['be_ellipse_fill'] = np.nan
    cnts, _ = cv2.findContours(worm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5:
            try:
                (_ex, _ey), (_MA, _ma), _ea = cv2.fitEllipse(c)   # WALNUT = OVAL: mask fills its fitted ellipse (~1); a worm/curve does not
                out['be_ellipse_fill'] = float(worm.sum()) / (np.pi / 4.0 * _MA * _ma + 1e-6)
            except Exception: pass
            try:
                hull = cv2.convexHull(c, returnPoints=False)
                df = cv2.convexityDefects(c, hull) if (hull is not None and len(hull) > 3) else None
            except cv2.error:
                df = None
            if df is not None:
                depths = df.reshape(-1, 4)[:, 3] / 256.0; sig = depths[depths > 4.0]
                scale = np.sqrt(worm.sum()) + 1e-6
                out['be_concave_depth'] = float(depths.max() / scale); out['be_n_concave'] = float(len(sig))
            cc = c[:, 0, :].astype(np.float32)   # WAVY cue: convex/concave alternations along the external border
            if len(cc) >= 24:
                st = max(1, len(cc) // 120); cp = cc[::st]
                cps = np.stack([gaussian_filter1d(cp[:, 0], 2, mode='wrap'), gaussian_filter1d(cp[:, 1], 2, mode='wrap')], 1)
                kk = 3; sg = []
                for i in range(len(cps)):
                    a = cps[i] - cps[i - kk]; b = cps[(i + kk) % len(cps)] - cps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
                    if na > 1e-6: sg.append(np.sign(a[0] * b[1] - a[1] * b[0]) * np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1))))
                sg = np.array(sg); strong = sg[np.abs(sg) > 12]
                out['be_border_wav'] = float((np.diff(np.sign(strong)) != 0).sum()) if len(strong) > 1 else 0.0
    path = _longest_path(skel)
    if path is None or len(path) < 10: return (out, _dbg) if return_debug else out
    P = np.array(path, np.float32)
    width = np.array([2.0 * edt[y, x] for y, x in path], np.float32)
    L = len(width); out['be_path_len'] = float(L)
    arc = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))); chord = float(np.linalg.norm(P[-1] - P[0])) + 1e-6
    out['be_tortuosity'] = arc / chord
    sw = gaussian_filter1d(width, max(2, L // 25)); medw = np.median(sw) + 1e-6; wn = sw / medw
    out['be_neck_depth'] = float(1.0 - wn.min()); out['be_width_cv'] = float(np.std(wn))
    mini, mp = find_peaks(-wn, prominence=0.07, distance=5)
    out['be_hourglass'] = float(mp['prominences'].max()) if len(mini) else 0.0
    out['be_n_necks'] = float(len(mini))
    out['be_neck_len'] = float((wn < 0.7).mean())
    t = L // 3; out['be_mid_end_ratio'] = float(np.median(wn[t:2 * t]) / (np.median(np.concatenate([wn[:t], wn[2 * t:]])) + 1e-6))
    out['be_width_fft'] = _fft_peak(wn)
    # per-position signed midline curvature
    Ps = np.stack([gaussian_filter1d(P[:, 0], 2.0), gaussian_filter1d(P[:, 1], 2.0)], 1)
    k = max(3, L // 30); curv = np.zeros(L, np.float32); sgnc = np.zeros(L, np.float32)
    for i in range(k, L - k):
        a = Ps[i] - Ps[i - k]; b = Ps[i + k] - Ps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
        if na < 1e-6: continue
        th = np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1))); curv[i] = th
        sgnc[i] = np.sign(a[1] * b[0] - a[0] * b[1]) * th
    aa = curv[k:L - k]
    if len(aa):
        out['be_max_turn'] = float(aa.max()); out['be_total_turn'] = float(aa.sum() / L); out['be_bend_energy'] = float(np.mean(aa ** 2))
        strong = sgnc[np.abs(sgnc) > 15]
        out['be_wav'] = float((np.diff(np.sign(strong)) != 0).sum()) if len(strong) > 1 else 0.0
        out['be_n_inflect'] = float((np.diff(np.sign(sgnc[np.abs(sgnc) > 8])) != 0).sum()) if (np.abs(sgnc) > 8).sum() > 1 else 0.0
        out['be_curv_fft'] = _fft_peak(curv)
        third = L // 3
        out['be_curv_tail_rise'] = float(curv[2 * third:].mean() - curv[:third].mean())
    # thin curved tail: pick the thinner end, measure its width and the curvature concentrated there
    e = max(3, L // 6)
    end_a = float(wn[:e].mean()); end_b = float(wn[-e:].mean())
    if end_a <= end_b:
        tailw = end_a; tail_curv = curv[:e].sum(); taper = float(wn[0] - wn[e])
    else:
        tailw = end_b; tail_curv = curv[-e:].sum(); taper = float(wn[-1] - wn[-e])
    out['be_tail_thin'] = float(max(0.0, 1.0 - tailw)); out['be_tail_curve'] = float(tail_curv / e)
    out['be_tail_taper'] = float(-taper)  # positive when the tip is thinner than just inside it
    # left/right border asymmetry along the centerline
    lr_l, lr_r = _lr_radii(worm, Ps, k)
    if len(lr_l):
        tot = lr_l + lr_r + 1e-6
        out['be_lr_asym'] = float(np.mean(np.abs(lr_l - lr_r) / tot))
        out['be_lr_bias'] = float(abs(np.mean((lr_l - lr_r) / tot)))
    # across-body (cross-section, perpendicular to centerline) texture: validated on cyan twist annotations as the
    # direction that localizes the twist (along-axis intensity does not). cross_std = intensity spread across the
    # width, cross_lrasym = left vs right mean-intensity asymmetry (rolling-surface shading at a twist).
    H, W = gray.shape; offs = np.arange(-40, 41); cs = np.zeros(L); cla = np.zeros(L); gm = float(gray[worm > 0].mean()) + 1e-6
    for i in range(k, L - k):
        t = Ps[i + k] - Ps[i - k]; nv = np.array([-t[1], t[0]], np.float32); nn2 = np.linalg.norm(nv)
        if nn2 < 1e-6: continue
        nv /= nn2; yy = np.clip((Ps[i, 0] + offs * nv[0]).round().astype(int), 0, H - 1); xx = np.clip((Ps[i, 1] + offs * nv[1]).round().astype(int), 0, W - 1)
        seg = gray[yy, xx].astype(np.float32); inb = worm[yy, xx] > 0
        if inb.sum() > 4:
            cs[i] = seg[inb].std() / gm
            lh = seg[:40][inb[:40]]; rh = seg[41:][inb[41:]]
            if len(lh) > 1 and len(rh) > 1: cla[i] = (lh.mean() - rh.mean()) / (seg[inb].mean() + 1e-6)
    out['be_cross_std'] = float(gaussian_filter1d(cs, 2).max())
    cla_s = gaussian_filter1d(cla, 2); out['be_cross_lrasym'] = float(np.abs(cla_s).max())
    strong_lr = cla_s[np.abs(cla_s) > 0.05]; out['be_cross_lr_flips'] = float((np.diff(np.sign(strong_lr)) != 0).sum()) if len(strong_lr) > 1 else 0.0
    # centerline intensity texture
    # extra single-frame shape cues from the latest review:
    # width taper (bigger-head/thinner-tail = monotonic slope), longest near-constant-width run (uniform tube),
    # number of sharp midline turns (multiple ~90deg), head-end curvature concentration (sharp vs smooth head).
    out['be_width_grad'] = float(abs(np.polyfit(np.linspace(0, 1, len(wn)), wn, 1)[0]))   # |slope| of width head->tail
    run = best = 1; anchor_v = wn[0]    # longest stretch where width stays within a band of a CONSTANT value
    for i in range(1, len(wn)):
        if abs(wn[i] - anchor_v) < 0.12: run += 1
        else: run = 1; anchor_v = wn[i]
        best = max(best, run)
    out['be_uniform_run'] = float(best) / max(1, len(wn))
    try:
        from scipy.signal import find_peaks as _fp
        out['be_n_sharp'] = float(len(_fp(curv, height=50, distance=5)[0]))
    except Exception:
        out['be_n_sharp'] = float((curv > 50).sum() > 0)
    e3 = max(3, L // 3); head_third = curv[:e3]; tail_third = curv[-e3:]
    out['be_head_curv'] = float(max(head_third.max(), tail_third.max()))   # the sharper end's peak curvature (sharp head/tail)
    inten = gaussian_filter1d(np.array([float(gray[y, x]) for y, x in path], np.float32), max(2, L // 25))
    bm = inten.mean() + 1e-6
    out['be_axial_dark_grad'] = float(abs(np.polyfit(np.linspace(0, 1, L), inten, 1)[0]) / bm)
    out['be_axial_tex_step'] = float(np.max(np.abs(np.diff(inten))) / bm)
    out['be_dark_core'] = float(1.0 - inten.min() / bm)
    wz = (sw - sw.mean()) / (sw.std() + 1e-6); dz = (inten.mean() - inten) / (inten.std() + 1e-6)
    out['be_darkwide'] = float(np.max(wz * dz))
    _dbg = dict(path=path, width=width, wn=wn, ang=list(curv[k:L - k]), inten=inten, sgnc=sgnc, lr_l=lr_l, lr_r=lr_r)
    return (out, _dbg) if return_debug else out


import functools as _ftc_be
@_ftc_be.lru_cache(maxsize=64)
def _be_cached(_kb, _shp, _dt):
    import numpy as _np
    return _border_evolution_impl(_np.frombuffer(_kb, dtype=_dt).reshape(_shp).copy())
def border_evolution(gray, return_debug=False):
    # content-memoised: panels that call this on the SAME frame (twist-texture and window-extreme) now share one compute
    if return_debug: return _border_evolution_impl(gray, True)
    return _be_cached(gray.tobytes(), gray.shape, str(gray.dtype))
