# regional_motion.py
# Per-region motion descriptors from the cached MOTION image (channels B=aligned prev, G=current, R=aligned
# next, already registered by make_motion_dataset). Where motion_metrics.py measures GLOBAL registered motion,
# this splits the worm along its body axis into head / centre / tail thirds and measures motion per region,
# plus body width per region and frame-to-frame length oscillation. It targets Valentin's class signatures:
#   Snake-like : an elongated head that MOVES while the thicker central body stays put -> high end/centre ratio
#   Wildtype   : the whole body moves                                                  -> high motion everywhere
#   Screw-like : scrunching, the body pumps length while barely gliding                -> high length oscillation
#   C-shape / Walnut : little motion in any region
# All values are scale-free ratios. Registration basis: oriented-energy / structure-tensor anisotropy
# (Rao & Schunck, CVGIP GMIP 1991, DOI 10.1016/1049-9652(91)90059-S). The head/tail end is not identified, so
# end features use the more-mobile or the max/imbalance of the two ends, which is head/tail invariant.
import numpy as np, cv2

REGIONAL_COLS = ['rm_total', 'rm_center_mot', 'rm_end_mot', 'rm_endcenter_contrast', 'rm_end_imbalance',
                 'rm_aniso', 'rm_axial_grad', 'rm_center_thick', 'rm_len_osc', 'rm_area_osc', 'ip_intensity_cv', 'ip_edge_mean', 'ip_edge_p95']

def _largest(worm):
    n, lab = cv2.connectedComponents(worm.astype(np.uint8))
    if n <= 1: return worm
    s = [(lab == i).sum() for i in range(1, n)]
    return lab == (1 + int(np.argmax(s)))

def regional_motion_features(motion_bgr):
    out = {k: np.nan for k in REGIONAL_COLS}
    if motion_bgr is None or motion_bgr.ndim != 3: return out
    B = motion_bgr[..., 0].astype(np.float32); G = motion_bgr[..., 1].astype(np.float32); R = motion_bgr[..., 2].astype(np.float32)
    worm = _largest(G > 8)
    if worm.sum() < 200: return out
    m = (np.abs(G - B) + np.abs(R - G)) * 0.5                       # registered internal motion magnitude
    ys, xs = np.where(worm); mv = m[worm].astype(np.float64)
    # body axis from the worm shape -> head-tail coordinate by projecting body pixels onto it
    mm = cv2.moments(worm.astype(np.uint8), binaryImage=True)
    cx = mm['m10'] / mm['m00']; cy = mm['m01'] / mm['m00']
    mu20 = mm['mu20'] / mm['m00']; mu02 = mm['mu02'] / mm['m00']; mu11 = mm['mu11'] / mm['m00']
    th = 0.5 * np.arctan2(2 * mu11, mu20 - mu02); ct, st = np.cos(th), np.sin(th)
    t = (xs - cx) * ct + (ys - cy) * st
    tn = (t - t.min()) / (np.ptp(t) + 1e-6); reg = np.clip((tn * 3).astype(int), 0, 2)
    r = [mv[reg == k].mean() if (reg == k).any() else np.nan for k in range(3)]
    r0, r1, r2 = r; ends = [e for e in (r0, r2) if e == e]; tot = float(mv.mean()) + 1e-9
    out['rm_total'] = float(mv.mean())
    out['rm_center_mot'] = float(r1) if r1 == r1 else np.nan
    out['rm_end_mot'] = float(max(ends)) if ends else np.nan
    out['rm_endcenter_contrast'] = float((max(ends) - r1) / (max(ends) + r1 + 1e-6)) if (ends and r1 == r1) else np.nan
    out['rm_end_imbalance'] = float(abs(r0 - r2) / (r0 + r2 + 1e-6)) if (r0 == r0 and r2 == r2) else np.nan
    rr = np.array([x for x in r if x == x], float)
    out['rm_aniso'] = float(rr.std() / (rr.mean() + 1e-9)) if len(rr) > 1 else np.nan
    out['rm_axial_grad'] = float(abs(r2 - r0) / tot) if (r0 == r0 and r2 == r2) else np.nan
    # body width per region (distance transform), centre-thick ratio (Snake has a stacked thicker centre)
    dt = cv2.distanceTransform(worm.astype(np.uint8), cv2.DIST_L2, 5); dv = dt[worm]
    w = [dv[reg == k].mean() if (reg == k).any() else np.nan for k in range(3)]
    if w[1] == w[1] and w[0] == w[0] and w[2] == w[2]:
        out['rm_center_thick'] = float(w[1] / (0.5 * (w[0] + w[2]) + 1e-6))
    # scrunch: length + area oscillation across the 3 aligned frames
    def axext(ch):
        m2 = ch > 8
        if m2.sum() < 50: return np.nan
        yy, xx = np.where(m2); tt = (xx - cx) * ct + (yy - cy) * st; return float(np.ptp(tt))
    exts = [e for e in (axext(B), axext(G), axext(R)) if e == e]
    if len(exts) > 1: out['rm_len_osc'] = float(np.std(exts) / (np.mean(exts) + 1e-6))
    areas = [float((c > 8).sum()) for c in (B, G, R)]; areas = [a for a in areas if a > 0]
    if len(areas) > 1: out['rm_area_osc'] = float(np.std(areas) / (np.mean(areas) + 1e-6))
    # internal pigment: uniform interior (Walnut) vs internal light borders (head/body separation, twist onset)
    inter = cv2.erode(worm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    gi = G[inter] if inter.sum() > 50 else G[worm]
    mu = float(gi.mean()) + 1e-6
    out['ip_intensity_cv'] = float(gi.std() / mu)                  # low = uniform pigment (Walnut)
    gx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    gv = grad[inter] if inter.sum() > 50 else grad[worm]           # interior gradient = internal light-border strength
    out['ip_edge_mean'] = float(gv.mean() / mu)
    out['ip_edge_p95'] = float(np.percentile(gv, 95) / mu)
    return out
