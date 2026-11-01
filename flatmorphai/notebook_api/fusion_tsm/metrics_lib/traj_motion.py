# traj_motion.py
# TRUE 3-frame trajectory features. Existing temporal metrics sum the two consecutive pairwise changes (total path),
# which conflates a DIRECTED evolution (stretch then stretch -> Snake elongating, twist developing -> Screw) with an
# OSCILLATION (stretch then contract -> Wildtype wobble / jitter), since both give the same total. These metrics use
# all three frames jointly: directedness = |net change over the window| / (total path), ~1 monotonic-directed, ~0
# oscillatory; and acceleration = the second difference (speeding up, slowing, reversing). Prefix trj_. masks only.
# Basis: posture-trajectory dynamics over time, Stephens et al. PLoS Comput Biol 2008, DOI 10.1371/journal.pcbi.1000028.
import numpy as np, cv2
from twist_motion_masks import _frame_state, triplet
from snake_undulation import _lat_profile

TRJ_COLS = ['trj_elong_direct', 'trj_width_direct', 'trj_len_direct', 'trj_curv_direct', 'trj_bow_direct',
            'trj_elong_accel', 'trj_curv_accel', 'trj_bow_rev']

def _direct(q0, q1, q2):
    tot = abs(q1 - q0) + abs(q2 - q1)
    return float(abs(q2 - q0) / (tot + 1e-6)) if tot > 1e-9 else 0.0   # 1 = monotonic/directed, 0 = oscillatory

def traj_motion(m0, m1, m2):
    nan = {c: np.nan for c in TRJ_COLS}
    s = [_frame_state(m) for m in (m0, m1, m2)]
    if any(x is None for x in s): return nan   # needs all 3 for a real trajectory
    s0, s1, s2 = s
    sca = np.sqrt(s1['area']) + 1e-6
    span = [x['span'] for x in s]; silw = [x['silw'] for x in s]; length = [x['length'] for x in s]
    curv = [float(np.mean(np.abs(x['cN']))) for x in s]
    bow = [float(np.mean(_lat_profile(x['Po']))) for x in s]   # signed chord-relative bend (left + / right -)
    out = {}
    out['trj_elong_direct'] = _direct(*span)
    out['trj_width_direct'] = _direct(*silw)
    out['trj_len_direct'] = _direct(*length)
    out['trj_curv_direct'] = _direct(*curv)
    out['trj_bow_direct'] = _direct(*bow)
    out['trj_elong_accel'] = float(((span[2] - span[1]) - (span[1] - span[0])) / sca)   # elongation acceleration
    out['trj_curv_accel'] = float((curv[2] - curv[1]) - (curv[1] - curv[0]))            # bending acceleration
    d1 = bow[1] - bow[0]; d2 = bow[2] - bow[1]
    out['trj_bow_rev'] = float(1.0 if d1 * d2 < 0 else 0.0)                             # body bend reverses direction (zigzag tick)
    return out

def trj_from_masks(basename, mask_index):
    nan = {c: np.nan for c in TRJ_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return traj_motion(gs[0], gs[1], gs[2])
