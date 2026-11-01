# twist_motion_masks.py
# Centerline-based TEMPORAL motion from the REAL, UNALIGNED masks of T-1, T0, T+1 (not the registered MOTION cache,
# which removes translation). For each of the three frames it traces the centerline, orients head-to-tail (thinner
# end on the right) with a consistent left/right convention, and takes the width, midline-curvature, and signed
# left-border and right-border distance profiles. It then measures, over the three frames: speed of movement
# (centroid displacement, kept because the masks are unaligned), direction consistency of the movement, principal-
# axis turning, body length / width / area change (scrunch and width-height modification), centerline reshaping
# (curvature oscillation), and the motion of the LEFT and RIGHT borders separately plus their asymmetry (a rolling
# or twisting body moves one flank more than the other). Prefix tm_. random_state 42, no GPU.
# Basis: medial axis / distance transform (Blum 1967, no DOI), worm posture-dynamics over time (Stephens et al.,
# From Modes to Movement, PLoS ONE 2010, DOI 10.1371/journal.pone.0013914).
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize
from twist_border_evolution import _longest_path

TM_COLS = ['tm_speed', 'tm_speed_dircos', 'tm_axis_turn', 'tm_len_chg', 'tm_width_chg', 'tm_maxw_chg',
           'tm_area_chg', 'tm_curv_osc', 'tm_border_L', 'tm_border_R', 'tm_border_LR_asym',
           'tm_bow_swing', 'tm_bow_rev', 'tm_border_L_rev', 'tm_border_R_rev',
           'tm_width_L_net', 'tm_width_R_net', 'tm_widthsum_net', 'tm_height_net',
           'tm_bilateral_elong', 'tm_elong_asym', 'tm_end_motion_asym',
           'tm_aspect_chg', 'tm_span_chg', 'tm_silwidth_chg']
N = 80

def _frame_state_impl(gray):
    mask = (gray > 8).astype(np.uint8)
    if mask.sum() < 150: return None
    nn, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if nn <= 1: return None
    wi = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); worm = (lab == wi).astype(np.uint8)
    area = float(worm.sum()); cy, cx = float(cent[wi][1]), float(cent[wi][0])
    mo = cv2.moments(worm); ori = 0.5 * np.arctan2(2 * mo['mu11'], (mo['mu20'] - mo['mu02']) + 1e-9)
    edt = distance_transform_edt(worm); path = _longest_path(skeletonize(worm > 0))
    if path is None or len(path) < 12: return None
    P = np.array(path, np.float32); L = len(P); Ps = np.stack([gaussian_filter1d(P[:, 0], 2.0), gaussian_filter1d(P[:, 1], 2.0)], 1)
    k = max(2, L // 30); H, W = gray.shape
    w = np.array([2.0 * edt[int(y), int(x)] for y, x in P], np.float32)
    cur = np.zeros(L, np.float32); lrL = np.zeros(L, np.float32); lrR = np.zeros(L, np.float32)
    rr = np.arange(1, 60)
    for i in range(k, L - k):
        a = Ps[i] - Ps[i - k]; b = Ps[i + k] - Ps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
        if na > 1e-6: cur[i] = np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1)))
        t = Ps[i + k] - Ps[i - k]; nv = np.array([-t[1], t[0]], np.float32); nn2 = np.linalg.norm(nv)
        if nn2 < 1e-6: continue
        nv /= nn2
        for sgn, store in ((1.0, 'L'), (-1.0, 'R')):
            d = 0
            for r in rr:
                yy = int(round(Ps[i, 0] + sgn * nv[0] * r)); xx = int(round(Ps[i, 1] + sgn * nv[1] * r))
                if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0: break
                d = r
            if store == 'L': lrL[i] = d
            else: lrR[i] = d
    xi = np.linspace(0, L - 1, N)
    wN = np.interp(xi, np.arange(L), w); cN = np.interp(xi, np.arange(L), cur)
    lN = np.interp(xi, np.arange(L), lrL); rN = np.interp(xi, np.arange(L), lrR)
    medw = np.median(w) + 1e-6
    flipped = wN[:N // 3].mean() < wN[-N // 3:].mean()
    if flipped:  # thinner end (tail) on the right, swap L/R so the convention holds
        wN, cN = wN[::-1], cN[::-1]; lN, rN = rN[::-1], lN[::-1]
    Po = Ps[::-1] if flipped else Ps
    PoN = np.stack([np.interp(np.linspace(0, L - 1, N), np.arange(L), Po[:, 0]),
                    np.interp(np.linspace(0, L - 1, N), np.arange(L), Po[:, 1])], 1)  # centreline points for cross-frame alignment
    ysw, xsw = np.where(worm); _yy = ysw - cy; _xx = xsw - cx   # silhouette principal-axis span + perp width (for elongation)
    _ev, _evec = np.linalg.eigh(np.cov(np.stack([_xx, _yy]).astype(np.float64))); _ax = _evec[:, -1]
    _proj = _xx * _ax[0] + _yy * _ax[1]; _perp = _xx * _ax[1] - _yy * _ax[0]
    span = float(_proj.max() - _proj.min()); silw = float(_perp.max() - _perp.min())
    mlL_px = float(np.median(lN)); mlR_px = float(np.median(rN))   # raw px half-width on each border (post head-tail flip)
    return dict(area=area, c=np.array([cy, cx]), ori=ori, length=float(L), wN=wN / medw, cN=cN,
                lN=lN / medw, rN=rN / medw, maxw=float(w.max()) / medw, medw=medw, Po=PoN,
                mlL_px=mlL_px, mlR_px=mlR_px, span=span, silw=silw)

def _angdiff(a, b):
    d = abs(a - b) % np.pi; return float(min(d, np.pi - d))

def twist_motion_masks(g_prev, g_cur, g_next):
    out = {c: np.nan for c in TM_COLS}
    s0, s1, s2 = _frame_state(g_prev), _frame_state(g_cur), _frame_state(g_next)
    if s1 is None or s0 is None or s2 is None: return out
    sca = np.sqrt(s1['area']) + 1e-6
    v01 = s1['c'] - s0['c']; v12 = s2['c'] - s1['c']
    out['tm_speed'] = float((np.linalg.norm(v01) + np.linalg.norm(v12)) / 2 / sca)
    nv = np.linalg.norm(v01) * np.linalg.norm(v12)
    out['tm_speed_dircos'] = float(v01 @ v12 / nv) if nv > 1e-6 else 0.0
    out['tm_axis_turn'] = _angdiff(s0['ori'], s1['ori']) + _angdiff(s1['ori'], s2['ori'])
    out['tm_len_chg'] = float((abs(s1['length'] - s0['length']) + abs(s2['length'] - s1['length'])) / (s1['length'] + 1e-6))
    out['tm_area_chg'] = float((abs(s1['area'] - s0['area']) + abs(s2['area'] - s1['area'])) / (s1['area'] + 1e-6))
    out['tm_width_chg'] = float(abs(np.median(s1['wN']) - np.median(s0['wN'])) + abs(np.median(s2['wN']) - np.median(s1['wN'])))
    out['tm_maxw_chg'] = float(abs(s1['maxw'] - s0['maxw']) + abs(s2['maxw'] - s1['maxw']))
    out['tm_curv_osc'] = float(np.mean(np.abs(s1['cN'] - s0['cN'])) + np.mean(np.abs(s2['cN'] - s1['cN'])))
    bL = float(np.mean(np.abs(s1['lN'] - s0['lN'])) + np.mean(np.abs(s2['lN'] - s1['lN'])))
    bR = float(np.mean(np.abs(s1['rN'] - s0['rN'])) + np.mean(np.abs(s2['rN'] - s1['rN'])))
    out['tm_border_L'] = bL; out['tm_border_R'] = bR
    out['tm_border_LR_asym'] = float(abs(bL - bR) / (bL + bR + 1e-6))
    # zigzag OVER TIME: the body bows one way then back. swing = amplitude of the bow change, rev = reversal (the
    # sign of the bow change flips between the two steps, i.e. a back-and-forth wiggle, not a steady bend).
    def _align(Po, ref):
        a = np.linalg.norm(Po[0] - ref[0]) + np.linalg.norm(Po[-1] - ref[-1])
        b = np.linalg.norm(Po[0] - ref[-1]) + np.linalg.norm(Po[-1] - ref[0])
        return Po[::-1] if b < a else Po
    def _bow(Po):
        ch = Po[-1] - Po[0]; cl = np.linalg.norm(ch)
        if cl < 1e-6: return 0.0
        cn = ch / cl; perp = np.array([-cn[1], cn[0]], np.float32)
        return float(np.mean((Po - Po[0]) @ perp) / cl)
    P1 = s1['Po']; P0 = _align(s0['Po'], P1); P2 = _align(s2['Po'], P1)
    b0, b1, b2 = _bow(P0), _bow(P1), _bow(P2)
    out['tm_bow_swing'] = float(max(b0, b1, b2) - min(b0, b1, b2))
    d1 = b1 - b0; d2 = b2 - b1
    out['tm_bow_rev'] = float(min(abs(d1), abs(d2))) if d1 * d2 < 0 else 0.0
    # border zigzag over time: each side's mean distance goes out then in (reversal of the per-side change)
    mlL0, mlL1, mlL2 = np.median(s0['lN']), np.median(s1['lN']), np.median(s2['lN'])
    mlR0, mlR1, mlR2 = np.median(s0['rN']), np.median(s1['rN']), np.median(s2['rN'])
    eL1, eL2 = mlL1 - mlL0, mlL2 - mlL1; eR1, eR2 = mlR1 - mlR0, mlR2 - mlR1
    out['tm_border_L_rev'] = float(min(abs(eL1), abs(eL2))) if eL1 * eL2 < 0 else 0.0
    out['tm_border_R_rev'] = float(min(abs(eR1), abs(eR2))) if eR1 * eR2 < 0 else 0.0
    # WIDTH and HEIGHT evolution from each border (signed net change over T-1 -> T+1, scale-free):
    ref = np.sqrt(s1['area']) + 1e-6   # one fixed scale (T0) so width changes are NOT conflated with length changes
    out['tm_width_L_net'] = float((s2['mlL_px'] - s0['mlL_px']) / ref)    # left border half-width grows(+)/shrinks(-)
    out['tm_width_R_net'] = float((s2['mlR_px'] - s0['mlR_px']) / ref)    # right border half-width grows/shrinks
    out['tm_widthsum_net'] = float(((s2['mlL_px'] + s2['mlR_px']) - (s0['mlL_px'] + s0['mlR_px'])) / ref)   # overall thicken/thin
    out['tm_height_net'] = float((s2['length'] - s0['length']) / (s1['length'] + 1e-6))   # body length (height) elong/contract
    # elongation pattern over time, from the head and tail tips relative to the centroid (translation-invariant):
    sca = np.sqrt(s1['area']) + 1e-6
    hh = [float(np.linalg.norm(P[0] - st['c'])) / sca for P, st in ((P0, s0), (P1, s1), (P2, s2))]   # head half-length
    tl = [float(np.linalg.norm(P[-1] - st['c'])) / sca for P, st in ((P0, s0), (P1, s1), (P2, s2))]   # tail half-length
    head_ext = hh[2] - hh[0]; tail_ext = tl[2] - tl[0]
    out['tm_bilateral_elong'] = float(2 * min(head_ext, tail_ext)) if (head_ext > 0 and tail_ext > 0) else 0.0   # BOTH ends extend
    out['tm_elong_asym'] = float(abs(head_ext - tail_ext) / (abs(head_ext) + abs(tail_ext) + 1e-6))              # one end vs the other
    hmot = (np.linalg.norm((P1[0] - s1['c']) - (P0[0] - s0['c'])) + np.linalg.norm((P2[0] - s2['c']) - (P1[0] - s1['c']))) / sca
    tmot = (np.linalg.norm((P1[-1] - s1['c']) - (P0[-1] - s0['c'])) + np.linalg.norm((P2[-1] - s2['c']) - (P1[-1] - s1['c']))) / sca
    out['tm_end_motion_asym'] = float(abs(hmot - tmot) / (hmot + tmot + 1e-6))   # head moves more than tail (or vice versa)
    # ELONGATION (audit winners, silhouette-based, area-invariant): aspect-ratio change is the strongest Snake cue
    # (0.82 vs centerline-len 0.75), silhouette-width change best for Snake-vs-Wildtype (0.74). MAGNITUDE not sign.
    asp = [st['span'] / (st['silw'] + 1e-6) for st in (s0, s1, s2)]
    out['tm_aspect_chg'] = float(abs(asp[1] - asp[0]) + abs(asp[2] - asp[1]))
    out['tm_span_chg'] = float((abs(s1['span'] - s0['span']) + abs(s2['span'] - s1['span'])) / sca)
    out['tm_silwidth_chg'] = float((abs(s1['silw'] - s0['silw']) + abs(s2['silw'] - s1['silw'])) / sca)
    return out


import re as _re
_FRRE = _re.compile(r'^(.*)_[Ff]rame_(\d+)_in_(\d+)$')
def triplet(basename):
    # the three frame basenames (prev, cur, next) for a labelled frame, with the same one-sided boundary rule as
    # precompute dyn_window (first -> n,n+1,n+2 ; last -> tot-2,tot-1,tot), so the temporal span is two steps.
    stem = basename[:-4] if basename.lower().endswith('.png') else basename
    m = _FRRE.match(stem)
    if not m: return None
    sid, n, tot = m.group(1), int(m.group(2)), int(m.group(3))
    if tot <= 1: return None
    if tot == 2: ks = [1, 2, 2] if n <= 1 else [1, 1, 2]
    elif n <= 1: ks = [1, 2, 3]
    elif n >= tot: ks = [tot - 2, tot - 1, tot]
    else: ks = [n - 1, n, n + 1]
    return ['%s_Frame_%d_in_%d.png' % (sid, k, tot) for k in ks]

def tm_from_index(basename, index):
    # index maps frame basename -> processed-grayscale path; returns the tm_ dict (NaN if a neighbor is missing)
    nan = {c: np.nan for c in TM_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = index.get(nm)
        if p is None: return nan
        g = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None: return nan
        gs.append(g)
    return twist_motion_masks(gs[0], gs[1], gs[2])


def tm_from_masks(basename, mask_index):
    # like tm_from_index but reads SEGMENTATION MASKS (any nonzero pixel = worm) from the full Mask2Former mask set,
    # so the T-1/T+1 neighbours exist even when the labelled/processed set is sparse. Boundary frames (Frame_1,
    # Frame_30) use the one-sided window from triplet(), matching the precompute dyn_window rule.
    nan = {c: np.nan for c in TM_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return twist_motion_masks(gs[0], gs[1], gs[2])


import functools as _ftc_fs
@_ftc_fs.lru_cache(maxsize=64)
def _fs_cached(_kb, _shp, _dt):
    import numpy as _np
    return _frame_state_impl(_np.frombuffer(_kb, dtype=_dt).reshape(_shp).copy())
def _frame_state(gray):
    # content-memoised: snake_undulation, traj_motion, deform_window, motionmask share one centerline trace per mask
    return _fs_cached(gray.tobytes(), gray.shape, str(gray.dtype))
