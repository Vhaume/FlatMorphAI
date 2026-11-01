# twist_snake_temporal.py
# Temporal metrics from the MOTION cache (BGR = aligned previous, current, aligned next), for the over-time
# signatures Valentin described: the body zigzagging over T-1,T0,T+1 (Snake), and zones getting bigger or thinner
# over time (Snake scrunch and Screw spinning). The three frames are already registered, so comparing them
# measures residual shape change net of gross translation. Complements the existing regional-motion metrics.
# Scale-free, no GPU. Basis: structure from registered temporal differences (Rao & Schunck, CVGIP GMIP 1991,
# DOI 10.1016/1049-9652(91)90059-S).
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

TT_COLS = ['tt_midline_lat_osc', 'tt_midline_lat_max', 'tt_width_zone_osc', 'tt_width_zone_max', 'tt_len_osc', 'tt_area_osc']

def _profile(chan, M, shape, x0, x1):
    cr = cv2.warpAffine(chan, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    mk = (cr > 8)
    width = np.full(x1 - x0 + 1, np.nan); mid = np.full(x1 - x0 + 1, np.nan)
    for j, x in enumerate(range(x0, x1 + 1)):
        r = np.where(mk[:, x])[0]
        if len(r): width[j] = r.max() - r.min() + 1; mid[j] = r.mean()
    return width, mid, int(mk.sum())

def twist_snake_temporal(motion_bgr):
    out = {c: np.nan for c in TT_COLS}
    if motion_bgr is None or motion_bgr.ndim != 3: return out
    B = motion_bgr[..., 0].astype(np.float32); G = motion_bgr[..., 1].astype(np.float32); R = motion_bgr[..., 2].astype(np.float32)
    cur = (G > 8).astype(np.uint8)
    if cur.sum() < 200: return out
    ys, xs = np.where(cur > 0); pts = np.stack([xs, ys], 1).astype(np.float32); mean = pts.mean(0)
    c = pts - mean; cov = (c.T @ c) / len(c); ev, evec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    Mr = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), ang, 1.0)
    gr = cv2.warpAffine(G, Mr, (G.shape[1], G.shape[0]), flags=cv2.INTER_LINEAR)
    cols = np.where((gr > 8).sum(0) > 0)[0]
    if len(cols) < 12: return out
    x0, x1 = cols.min(), cols.max()
    wB, mB, aB = _profile(B, Mr, G.shape, x0, x1); wG, mG, aG = _profile(G, Mr, G.shape, x0, x1); wR, mR, aR = _profile(R, Mr, G.shape, x0, x1)
    span = x1 - x0 + 1; medw = np.nanmedian(wG) + 1e-6
    def sm(a): a = a.copy(); a[~np.isfinite(a)] = np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0; return gaussian_filter1d(a, 3.0)
    mBs, mGs, mRs = sm(mB), sm(mG), sm(mR); wBs, wGs, wRs = sm(wB), sm(wG), sm(wR)
    # lateral midline oscillation over time (zigzag): mean and max of |cur-prev| + |next-cur|, normalized by width
    lat = (np.abs(mGs - mBs) + np.abs(mRs - mGs)) / (medw + 1e-6)
    out['tt_midline_lat_osc'] = float(np.nanmean(lat)); out['tt_midline_lat_max'] = float(np.nanmax(lat))
    # per-zone width oscillation over time (zones get bigger/thinner)
    zo = (np.abs(wGs - wBs) + np.abs(wRs - wGs)) / (medw + 1e-6)
    out['tt_width_zone_osc'] = float(np.nanmean(zo)); out['tt_width_zone_max'] = float(np.nanmax(zo))
    # length and area oscillation over time (scrunch)
    def length(w):
        nz = np.where(np.isfinite(w) & (w > 0))[0]; return (nz.max() - nz.min() + 1) if len(nz) else 0
    lens = [length(wB), length(wG), length(wR)]
    out['tt_len_osc'] = float((abs(lens[1] - lens[0]) + abs(lens[2] - lens[1])) / (lens[1] + 1e-6))
    out['tt_area_osc'] = float((abs(aG - aB) + abs(aR - aG)) / (aG + 1e-6))
    return out
