# twist_snake_metrics.py
# Metrics built from Valentin's frame-by-frame comments on the model failures, made ROBUST on real (jagged) worm
# masks via resampling, Gaussian smoothing and prominence-based peak detection, then validated on the annotated
# ground-truth frames. Signatures targeted:
#   Screw-like : hourglass neck (prominent width minimum), bigger middle body (mid/end width ratio), a sharp V or
#                90-degree turn on the smoothed border, and body superposition (a region both wider and darker).
#   Snake-like : a zigzagging midline (prominent midline reversals) and a head-to-tail darkness gradient.
#   Segmentation artifacts: stray components, a bright reflection halo just outside the worm.
# Scale-free, computed from the masked grayscale frame (worm bright on black). No GPU.
# Basis: medial width profile and turning angle (Mokhtarian & Mackworth, IEEE TPAMI 1992, DOI 10.1109/34.149591);
# connected components (Rosenfeld & Pfaltz, JACM 1966, DOI 10.1145/321356.321357); peak prominence (standard).
import numpy as np, cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

CM_COLS = ['cm_neck_count', 'cm_neck_depth', 'cm_hourglass', 'cm_mid_end_ratio', 'cm_width_cv', 'cm_width_osc',
           'cm_max_turn', 'cm_n_sharp', 'cm_darkwide', 'cm_midline_inflect', 'cm_midline_amp',
           'cm_axial_dark_grad', 'cm_n_components', 'cm_stray_frac', 'cm_bright_outside',
           'cm_border_wav', 'cm_border_wav_amp', 'cm_axial_tex_step', 'cm_neck_len']

def _resample(a, N=120):
    a = np.asarray(a, np.float32); ok = np.isfinite(a)
    if ok.sum() < 4: return None
    xi = np.linspace(0, len(a) - 1, N); return np.interp(xi, np.where(ok)[0], a[ok])

def twist_snake_metrics(gray):
    out = {c: np.nan for c in CM_COLS}
    if gray is None or gray.ndim != 2: return out
    g = gray.astype(np.float32); mask = (g > 8).astype(np.uint8)
    if mask.sum() < 200: return out
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: return out
    areas = stats[1:, cv2.CC_STAT_AREA]; main = 1 + int(np.argmax(areas))
    out['cm_n_components'] = float(n - 1)
    out['cm_stray_frac'] = float(areas.sum() - areas.max()) / float(areas.sum() + 1e-6)
    worm = (lab == main).astype(np.uint8)
    ring = (cv2.dilate(worm, np.ones((9, 9), np.uint8)) - worm) > 0
    out['cm_bright_outside'] = float((g[ring] > 40).mean()) if ring.any() else 0.0
    # rotate so principal axis is horizontal
    ys, xs = np.where(worm > 0); pts = np.stack([xs, ys], 1).astype(np.float32); mean = pts.mean(0)
    c = pts - mean; cov = (c.T @ c) / len(c); ev, evec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    M = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), ang, 1.0)
    wr = cv2.warpAffine(worm, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST)
    gr = cv2.warpAffine(g, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)
    cols = np.where(wr.sum(0) > 0)[0]
    if len(cols) < 12: return out
    x0, x1 = cols.min(), cols.max(); width = []; mid = []; inten = []
    for x in range(x0, x1 + 1):
        r = np.where(wr[:, x] > 0)[0]
        if len(r) == 0: width.append(np.nan); mid.append(np.nan); inten.append(np.nan)
        else: width.append(float(r.max() - r.min() + 1)); mid.append(float(r.mean())); inten.append(float(gr[r, x].mean()))
    # ---- robust width profile: resample, smooth, prominence peaks ----
    wN = _resample(width, 120)
    if wN is None: return out
    wN = gaussian_filter1d(wN, 3.0); medw = np.median(wN) + 1e-6; wn = wN / medw
    PRO = 0.07                                              # min prominence as fraction of median width
    maxi, _ = find_peaks(wn, prominence=PRO, distance=6)
    mini, mp = find_peaks(-wn, prominence=PRO, distance=6)
    out['cm_neck_count'] = float(len(mini))
    out['cm_neck_depth'] = float(1.0 - wn.min())
    out['cm_width_cv'] = float(np.std(wn))
    out['cm_width_osc'] = float(len(maxi) + len(mini))
    out['cm_hourglass'] = float(mp['prominences'].max()) if len(mini) else 0.0   # deepest prominent neck
    t = len(wn) // 3
    out['cm_mid_end_ratio'] = float(np.median(wn[t:2 * t]) / (np.median(np.concatenate([wn[:t], wn[2 * t:]])) + 1e-6))
    out['cm_neck_len'] = float((wn < 0.7).mean())   # fraction of the body that is a thin neck
    # ---- superposition: wider AND darker ----
    iN = _resample(inten, 120)
    if iN is not None:
        iN = gaussian_filter1d(iN, 3.0); wz = (wN - wN.mean()) / (wN.std() + 1e-6); dz = (iN.mean() - iN) / (iN.std() + 1e-6)
        out['cm_darkwide'] = float(np.max(wz * dz))
        out['cm_axial_dark_grad'] = float(abs(np.polyfit(np.linspace(0, 1, len(iN)), iN, 1)[0]) / (iN.mean() + 1e-6))
        out['cm_axial_tex_step'] = float(np.max(np.abs(np.diff(gaussian_filter1d(iN, 2.0)))) / (iN.mean() + 1e-6))   # lighter texture step before a twist
    # ---- midline zigzag: prominent reversals on the smoothed midline ----
    mN = _resample(mid, 120)
    if mN is not None:
        mN = gaussian_filter1d(mN, 3.0); amp = (mN.max() - mN.min()); thr = max(2.0, 0.15 * amp)
        pk, _ = find_peaks(mN, prominence=thr); tr, _ = find_peaks(-mN, prominence=thr)
        out['cm_midline_inflect'] = float(len(pk) + len(tr))
        span = (x1 - x0 + 1); out['cm_midline_amp'] = float(amp / (span + 1e-6))
    # ---- sharp turn on the SMOOTHED border (body-scale turning angle) ----
    cnts, _ = cv2.findContours(worm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32)
        if len(cnt) > 40:
            idx = np.linspace(0, len(cnt) - 1, 160).astype(int); P = cnt[idx]
            P[:, 0] = gaussian_filter1d(P[:, 0], 2.0, mode='wrap'); P[:, 1] = gaussian_filter1d(P[:, 1], 2.0, mode='wrap')
            k = 5; ang_l = []; sgn_l = []
            for i in range(len(P)):
                a = P[i] - P[i - k]; b = P[(i + k) % len(P)] - P[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
                if na < 1e-6: continue
                cs = np.clip((a @ b) / na, -1, 1); ang = np.degrees(np.arccos(cs)); ang_l.append(ang)
                sgn_l.append(np.sign(a[0] * b[1] - a[1] * b[0]) * ang)        # signed turning (left/right)
            if ang_l:
                aa = np.array(ang_l); out['cm_max_turn'] = float(aa.max()); out['cm_n_sharp'] = float((aa > 50).sum())
                sg = np.array(sgn_l); strong = sg[np.abs(sg) > 12]
                out['cm_border_wav'] = float((np.diff(np.sign(strong)) != 0).sum()) if len(strong) > 1 else 0.0   # border undulations
                out['cm_border_wav_amp'] = float(np.std(sg))
    return out
