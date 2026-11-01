# snake_undulation.py
# CANDIDATE metrics for the Snake-over-time gap (hardcase audit: 23 frames, 2nd-largest theme; existing zigzag
# metrics are weak for Snake or conflated with twist). The defining Snake signature is a LATERAL travelling wave:
# the body displaces side-to-side with the bend moving along its length over T-1,T0,T+1. We measure the midline's
# offset from its own head-tail chord, which is translation- and rotation-invariant by construction, so a rigid
# glide reads ~0 and a 3D twist (rotation) does not drive it the way curvature oscillation does. Prefix snk_. No GPU.
#   snk_lat_dyn     RMS change of the chord-relative lateral profile across the window = lateral undulation OVER TIME
#                   (glide ~0, static C-bend ~0, undulating Snake high). The key Snake-over-time cue.
#   snk_lat_nwave   sign-changes of the lateral profile = serpentine waviness (single C-bend low, zigzag high)
#   snk_lat_amp     max chord-relative lateral offset over the window (overall bend magnitude)
#   snk_lat_travel  cross-correlation lag of the lateral profile between frames = travelling wave (vs standing)
#   snk_border_dyn  RMS change of the left-minus-right border-offset profile = wavy BORDER over time
# Basis: undulatory locomotion / eigenworm posture dynamics, Stephens, Johnson-Kerner, Bialek, Ryu, PLoS Comput
# Biol 2008, DOI 10.1371/journal.pcbi.1000028; gait waveforms, Fang-Yen et al., PNAS 2010, DOI 10.1073/pnas.1003016107.
import numpy as np
from scipy.ndimage import gaussian_filter1d
from twist_motion_masks import _frame_state, triplet
import cv2

SNK_COLS = ['snk_lat_dyn', 'snk_lat_nwave', 'snk_lat_amp', 'snk_lat_travel', 'snk_border_dyn']

def _lat_profile(Po):
    # signed perpendicular offset of each midline point from the head-tail chord, normalised by chord length
    h, t = Po[0], Po[-1]; axis = t - h; L = float(np.linalg.norm(axis)) + 1e-6; ax = axis / L
    nrm = np.array([-ax[1], ax[0]])                 # left-hand normal to the body axis
    return ((Po - h) @ nrm) / L                     # rigid-invariant lateral profile d(s), in [-~,~]

def _align_dyn(a, b):
    # RMS difference of two lateral profiles, robust to a head-tail orientation flip (try both, keep the smaller)
    d1 = np.sqrt(np.mean((b - a) ** 2)); d2 = np.sqrt(np.mean((-b[::-1] - a) ** 2))
    return float(min(d1, d2))

def _nwave(d):
    ds = gaussian_filter1d(d, 2.0); s = np.sign(ds); s = s[s != 0]
    return float(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0.0

def _travel(a, b):
    a = a - a.mean(); b = b - b.mean(); na = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8; L = len(a); best = (-2.0, 0)
    for lag in range(-L // 4, L // 4 + 1):
        cc = float((a * np.roll(b, lag)).sum() / na)
        if cc > best[0]: best = (cc, lag)
    return abs(best[1]) / L

def snake_undulation(m_prev, m_cur, m_next):
    nan = {c: np.nan for c in SNK_COLS}
    st = [_frame_state(m) for m in (m_prev, m_cur, m_next)]
    ok = [s for s in st if s is not None]
    if len(ok) < 2: return nan
    prof = [_lat_profile(s['Po']) for s in ok]
    basym = [s['lN'] - s['rN'] for s in ok]          # left-minus-right border offset profile (border waviness)
    out = dict(nan)
    out['snk_lat_amp'] = float(np.max([np.max(np.abs(p)) for p in prof]))
    out['snk_lat_nwave'] = float(np.mean([_nwave(p) for p in prof]))
    dyns = [_align_dyn(prof[i], prof[i + 1]) for i in range(len(prof) - 1)]
    out['snk_lat_dyn'] = float(np.mean(dyns)) if dyns else np.nan
    if len(prof) >= 2: out['snk_lat_travel'] = float(np.mean([_travel(prof[i], prof[i + 1]) for i in range(len(prof) - 1)]))
    bd = [np.sqrt(np.mean((basym[i + 1] - basym[i]) ** 2)) for i in range(len(basym) - 1)]
    out['snk_border_dyn'] = float(np.mean(bd)) if bd else np.nan
    return out

def snk_from_masks(basename, mask_index):
    nan = {c: np.nan for c in SNK_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return snake_undulation(gs[0], gs[1], gs[2])
