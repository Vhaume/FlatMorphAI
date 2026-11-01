# motion_metrics.py
# Scalar motion descriptors computed from the CACHED motion image (make_motion_dataset), whose channels are
# B = aligned previous frame, G = current frame, R = aligned next frame (already registered). The earlier
# temporal metrics failed for two reasons: they used the silhouette and they were not registered, so they
# measured gross glide. These are computed on the registered texture motion of the worm interior, and they
# target the patterns the montage showed: interior texture churn (rolling Screw), axial travelling motion
# (undulating Snake), and near-flat motion (gliding Wildtype). Methodological basis: oriented-energy /
# structure-tensor anisotropy (Rao & Schunck, CVGIP GMIP 1991, DOI 10.1016/1049-9652(91)90059-S) and
# image-difference temporal decorrelation as a roll/rotation cue (analogous to C. elegans roll detection,
# PNAS 2018, DOI 10.1073/pnas.1706754115). All values are scale-free ratios or normalized intensities.
import numpy as np, cv2

MOTION_COLS = ['mot_mag', 'mot_p90', 'mot_active_frac', 'mot_interior_frac', 'mot_edge_frac',
               'mot_decorr', 'mot_aniso', 'mot_axis_align']

def _pearson(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.size < 30 or a.std() < 1e-6 or b.std() < 1e-6: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def motion_metrics(motion_bgr):
    """motion_bgr: the cached motion image read with cv2 (BGR -> B=prev, G=cur, R=next)."""
    out = {k: np.nan for k in MOTION_COLS}
    if motion_bgr is None or motion_bgr.ndim != 3: return out
    B = motion_bgr[..., 0].astype(np.float32); G = motion_bgr[..., 1].astype(np.float32); R = motion_bgr[..., 2].astype(np.float32)
    worm = G > 8
    if worm.sum() < 200: return out
    inter = cv2.erode(worm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    band = worm & (~inter)
    db = G - B; df = R - G; m = (np.abs(db) + np.abs(df)) * 0.5     # registered motion magnitude
    mw = m[worm]; tot = float(mw.sum()) + 1e-6
    out['mot_mag'] = float(mw.mean())
    out['mot_p90'] = float(np.percentile(mw, 90))
    out['mot_active_frac'] = float((mw > 8).mean())
    out['mot_interior_frac'] = float(m[inter].sum() / tot) if inter.any() else np.nan
    out['mot_edge_frac'] = float(m[band].sum() / tot) if band.any() else np.nan
    # temporal decorrelation on the interior: rolling churns surface texture -> low corr after alignment
    if inter.sum() > 50:
        cb = _pearson(G[inter], B[inter]); cr = _pearson(G[inter], R[inter])
        cs = [c for c in (cb, cr) if c == c]
        out['mot_decorr'] = float(1.0 - np.mean(cs)) if cs else np.nan
    # spatial distribution of motion energy: anisotropy and alignment with the body axis
    ys, xs = np.where(worm); w = m[worm].astype(np.float64) + 1e-9
    wx = float((w * xs).sum() / w.sum()); wy = float((w * ys).sum() / w.sum())
    dx = xs - wx; dy = ys - wy
    cxx = float((w * dx * dx).sum() / w.sum()); cyy = float((w * dy * dy).sum() / w.sum()); cxy = float((w * dx * dy).sum() / w.sum())
    tr = cxx + cyy; det = cxx * cyy - cxy * cxy
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    l1 = tr / 2 + disc; l2 = tr / 2 - disc
    out['mot_aniso'] = float((l1 - l2) / (l1 + l2 + 1e-9))
    motion_axis = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    # body axis from the worm shape
    mm = cv2.moments(worm.astype(np.uint8), binaryImage=True)
    if mm['m00'] > 0:
        bmu20 = mm['mu20'] / mm['m00']; bmu02 = mm['mu02'] / mm['m00']; bmu11 = mm['mu11'] / mm['m00']
        body_axis = 0.5 * np.arctan2(2 * bmu11, bmu20 - bmu02)
        out['mot_axis_align'] = float(abs(np.cos(2 * (motion_axis - body_axis))))  # 1 = motion elongated along body (undulation)
    return out


# ---- shared motion-image builder (same registration as make_motion_dataset), reused by the golden verify ----
def _pose(gray):
    m = cv2.moments((gray > 8).astype(np.uint8), binaryImage=True)
    if m['m00'] < 50: return None
    cx = m['m10'] / m['m00']; cy = m['m01'] / m['m00']
    mu20 = m['mu20'] / m['m00']; mu02 = m['mu02'] / m['m00']; mu11 = m['mu11'] / m['m00']
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    ecc = ((mu20 - mu02) ** 2 + 4 * mu11 ** 2) ** 0.5 / (mu20 + mu02 + 1e-9)
    return cx, cy, float(theta), float(ecc)

def _align_to(src_gray, src_pose, dst_pose, dst_gray):
    H, W = src_gray.shape
    scx, scy, sth, sec = src_pose; dcx, dcy, dth, dec = dst_pose
    angles = [0.0] if min(sec, dec) < 0.15 else [dth - sth, dth - sth + np.pi]
    cand = []
    for a in angles:
        M = cv2.getRotationMatrix2D((scx, scy), np.degrees(a), 1.0)
        M[0, 2] += (dcx - scx); M[1, 2] += (dcy - scy)
        cand.append(cv2.warpAffine(src_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0))
    if len(cand) == 1: return cand[0]
    dm = dst_gray > 8
    sads = [float(np.abs((c - dst_gray)[dm]).mean()) if dm.any() else np.abs(c).mean() for c in cand]
    return cand[int(np.argmin(sads))]

def build_motion_image(g_prev, g_cur, g_next, register=True):
    """grayscale prev/cur/next (float or uint8) -> registration-aware temporal-RGB BGR image (B=prev,G=cur,R=next)."""
    g_cur = g_cur.astype(np.float32)
    pt = _pose(g_cur)
    def prep(gn):
        if gn is None or pt is None: return g_cur
        gn = gn.astype(np.float32); pn = _pose(gn)
        if pn is None: return g_cur
        return _align_to(gn, pn, pt, g_cur) if register else gn
    R = prep(g_prev); B = prep(g_next); worm = (g_cur > 8)
    out = np.zeros((g_cur.shape[0], g_cur.shape[1], 3), np.uint8)
    out[..., 0] = (np.clip(R, 0, 255) * worm).astype(np.uint8)
    out[..., 1] = (np.clip(g_cur, 0, 255) * worm).astype(np.uint8)
    out[..., 2] = (np.clip(B, 0, 255) * worm).astype(np.uint8)
    return out
