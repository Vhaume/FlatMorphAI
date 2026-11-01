# twist_border_evolution_temporal.py
# Temporal border evolution along the centerline, from the MOTION cache (B=aligned prev, G=current, R=aligned
# next), available per single frame (train and test) so usable at inference. For each registered frame it traces
# that frame's centerline and takes the width and midline-curvature profiles, resampled head-to-tail, then measures
# how they change over time: body reshaping / zigzag (curvature oscillation, mean / max / periodicity), zones
# thickening or thinning (width oscillation), scrunch (length and area oscillation), and whether the motion is at
# the extremities or whole-body (ends-vs-middle). Prefix bt_. random_state 42, no GPU.
# Basis: medial axis (Blum 1967, no DOI), worm posture-dynamics over time (Stephens et al., From Modes to Movement,
# PLoS ONE 2010, DOI 10.1371/journal.pone.0013914), Fourier descriptor of the oscillation (Zahn & Roskies, IEEE
# Trans Computers 1972, DOI 10.1109/TC.1972.5008949).
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize
from twist_border_evolution import _longest_path

BT_COLS = ['bt_width_osc', 'bt_curv_osc', 'bt_curv_osc_max', 'bt_len_osc', 'bt_area_osc', 'bt_osc_endmid', 'bt_curv_osc_fft']
N = 80

def _fft_peak(p):
    p = p - p.mean()
    if p.std() < 1e-6 or len(p) < 16: return 0.0
    f = np.abs(np.fft.rfft(p * np.hanning(len(p)))) ** 2
    return float(f[1:6].max() / (f[1:].sum() + 1e-9)) if f[1:].sum() > 1e-9 else 0.0

def _profiles(gray):
    mask = (gray > 8).astype(np.uint8)
    if mask.sum() < 150: return None
    nn, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if nn <= 1: return None
    worm = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    edt = distance_transform_edt(worm); path = _longest_path(skeletonize(worm > 0))
    if path is None or len(path) < 10: return None
    P = np.array(path, np.float32); w = np.array([2.0 * edt[y, x] for y, x in path], np.float32)
    Ps = np.stack([gaussian_filter1d(P[:, 0], 2.0), gaussian_filter1d(P[:, 1], 2.0)], 1)
    k = max(2, len(P) // 30); cur = []
    for i in range(len(P)):
        a = Ps[i] - Ps[max(0, i - k)]; b = Ps[min(len(P) - 1, i + k)] - Ps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
        cur.append(np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1))) if na > 1e-6 else 0.0)
    xi = np.linspace(0, len(P) - 1, N)
    wN = np.interp(xi, np.arange(len(P)), w); cN = np.interp(xi, np.arange(len(P)), np.array(cur))
    return wN, cN, float(len(P)), float(worm.sum())

def twist_border_evolution_temporal(motion_bgr, return_profiles=False):
    out = {c: np.nan for c in BT_COLS}; prof = None
    if motion_bgr is None or motion_bgr.ndim != 3: return (out, prof) if return_profiles else out
    pB = _profiles(motion_bgr[..., 0]); pG = _profiles(motion_bgr[..., 1]); pR = _profiles(motion_bgr[..., 2])
    if pG is None or not (pB and pR): return (out, prof) if return_profiles else out
    wB, cB, lB, aB = pB; wG, cG, lG, aG = pG; wR, cR, lR, aR = pR
    medw = np.median(wG) + 1e-6
    wosc = (np.abs(wG - wB) + np.abs(wR - wG)) / medw
    cosc = np.abs(cG - cB) + np.abs(cR - cG)
    out['bt_width_osc'] = float(np.mean(wosc)); out['bt_curv_osc'] = float(np.mean(cosc))
    out['bt_curv_osc_max'] = float(np.max(cosc)); out['bt_curv_osc_fft'] = _fft_peak(cosc)
    out['bt_len_osc'] = float((abs(lG - lB) + abs(lR - lG)) / (lG + 1e-6))
    out['bt_area_osc'] = float((abs(aG - aB) + abs(aR - aG)) / (aG + 1e-6))
    t = N // 3; ends = np.concatenate([cosc[:t], cosc[-t:]]).mean(); mid = cosc[t:-t].mean()
    out['bt_osc_endmid'] = float(ends / (ends + mid + 1e-6))  # bounded [0,1], high = motion at the extremities
    if return_profiles: prof = (wosc, cosc)
    return (out, prof) if return_profiles else out
