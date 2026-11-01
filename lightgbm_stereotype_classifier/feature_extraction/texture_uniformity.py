# texture_uniformity.py
# Spatially-structured pigment-uniformity descriptors, the family Valentin asked for and that the 70 existing
# (global) texture metrics do not cover: how many distinct uniformity zones the worm has, how uniformity
# evolves along the body (head-to-tail) and across it, and how much a folded zone differs from a smooth one.
# A fold or twist creates a local zone whose intensity/uniformity differs from the rest, so the SPATIAL
# variation of local uniformity, not its global average, is the cue. Walnut is uniform (few zones, flat
# gradient); a folded Screw shows extra zones and an uneven uniformity map. All values are scale-free.
# Method basis: local variance maps (sliding-window second moment) and quantile zone labelling; structure
# along the body principal axis. Backing for region/zone texture: Haralick, Shanmugam, Dinstein, Textural
# Features for Image Classification, IEEE Trans. SMC 1973, DOI 10.1109/TSMC.1973.4309314.
import numpy as np, cv2

TUNIF_COLS = ['tu_nzones', 'tu_localstd_mean', 'tu_localstd_cv', 'tu_localstd_p95',
              'tu_axial_range', 'tu_axial_grad', 'tu_perp_range', 'tu_zone_contrast', 'tu_dark_offset']

def _largest(m):
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    if n <= 1: return m.astype(bool)
    s = [(lab == i).sum() for i in range(1, n)]
    return lab == (1 + int(np.argmax(s)))

def texture_uniformity_features(gray, mask):
    out = {k: np.nan for k in TUNIF_COLS}
    m = _largest(mask > 0)
    if m.sum() < 200: return out
    inter = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inter.sum() < 100: inter = m
    G = gray.astype(np.float32)
    mu_glob = float(G[inter].mean()) + 1e-6
    # local std map (sliding 9x9 second moment)
    mean = cv2.boxFilter(G, -1, (9, 9)); sq = cv2.boxFilter(G * G, -1, (9, 9))
    lstd = np.sqrt(np.clip(sq - mean * mean, 0, None))
    lv = lstd[inter]
    out['tu_localstd_mean'] = float(lv.mean() / mu_glob)
    out['tu_localstd_cv'] = float(lv.std() / (lv.mean() + 1e-6))            # how much uniformity VARIES across the worm
    out['tu_localstd_p95'] = float(np.percentile(lv, 95) / mu_glob)        # most non-uniform zone
    # number of uniformity zones: quantile intensity bands -> connected components above a min area
    vals = G[inter]; qs = np.quantile(vals, [0.25, 0.5, 0.75]); band = np.digitize(G, qs)
    min_area = max(40, int(0.01 * inter.sum())); nz = 0
    for b in range(4):
        bm = ((band == b) & inter).astype(np.uint8)
        n, lab = cv2.connectedComponents(bm)
        for i in range(1, n):
            if (lab == i).sum() > min_area: nz += 1
    out['tu_nzones'] = float(nz)
    # zone contrast: range of the local-MEAN map across the worm (a localized dark/light fold)
    bm_in = mean[inter]
    out['tu_zone_contrast'] = float((np.percentile(bm_in, 95) - np.percentile(bm_in, 5)) / mu_glob)
    # uniformity gradient along the body axis and across it
    ys, xs = np.where(inter); mm = cv2.moments(inter.astype(np.uint8), binaryImage=True)
    cx = mm['m10'] / mm['m00']; cy = mm['m01'] / mm['m00']
    mu20 = mm['mu20'] / mm['m00']; mu02 = mm['mu02'] / mm['m00']; mu11 = mm['mu11'] / mm['m00']
    th = 0.5 * np.arctan2(2 * mu11, mu20 - mu02); ct, st = np.cos(th), np.sin(th)
    t = (xs - cx) * ct + (ys - cy) * st; u = -(xs - cx) * st + (ys - cy) * ct
    lvi = lstd[inter]
    def grad_range(coord):
        cn = (coord - coord.min()) / (np.ptp(coord) + 1e-6); bk = np.clip((cn * 5).astype(int), 0, 4)
        pb = np.array([lvi[bk == k].mean() if (bk == k).any() else np.nan for k in range(5)])
        pb = pb[np.isfinite(pb)]
        if len(pb) < 2: return np.nan, np.nan
        mn = pb.mean() + 1e-6
        return float((pb.max() - pb.min()) / mn), float(abs(pb[-1] - pb[0]) / mn)
    out['tu_axial_range'], out['tu_axial_grad'] = grad_range(t)
    out['tu_perp_range'], _ = grad_range(u)
    # dark-weighted centroid offset from the geometric centre (a fold shifts the dark mass off-centre)
    med = float(np.median(G[inter])); w = np.clip(med - G, 0, None) * inter
    sw = float(w.sum())
    if sw > 1e-6:
        dcx = float((w * np.arange(G.shape[1])[None, :]).sum() / sw); dcy = float((w * np.arange(G.shape[0])[:, None]).sum() / sw)
        out['tu_dark_offset'] = float(np.hypot(dcx - cx, dcy - cy) / (np.sqrt(m.sum()) + 1e-6))
    return out
