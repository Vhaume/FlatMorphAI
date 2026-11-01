# deform_window.py
# 3-MASK-JOINT deformation. The deformation panel (nrm_deform, tps_bend, srvf) only averages the two consecutive
# pairwise residuals, so it cannot tell a progressive deformation (a Screw twisting further over the window) from a
# deform-and-return wobble (Wildtype / jitter). These deform T-1 DIRECTLY onto T+1 (the 2-step net) and compare it to
# the pairwise total: directedness ~1 = the body kept deforming the same way, ~0 = it returned. Applies the winning
# deformation family (nrm, tps) with the joint 3-frame aggregation the scalar trajectory panel used. Prefix dfw_. masks only.
import numpy as np, cv2
from nonrigid_motion import _state, _pair
from twist_motion_masks import _frame_state
from sota_motion_candidates import _tps_bend, _srvf_dist
from twist_motion_masks import triplet

DFW_COLS = ['dfw_deform_net', 'dfw_deform_direct', 'dfw_tps_net', 'dfw_srvf_net', 'dfw_deform_accel', 'dfw_axisrot_net']

def deform_window(m0, m1, m2):
    nan = {c: np.nan for c in DFW_COLS}
    s0, s1, s2 = _state(m0), _state(m1), _state(m2)
    if s0 is None or s1 is None or s2 is None: return nan   # needs all 3
    d01, _, r01, _ = _pair(s0, s1); d12, _, r12, _ = _pair(s1, s2); d02, _, r02, _ = _pair(s0, s2)
    tot = d01 + d12
    out = {}
    out['dfw_deform_net'] = float(d02)                                  # non-rigid residual T-1 -> T+1 (2-step)
    out['dfw_deform_direct'] = float(d02 / (tot + 1e-6)) if tot > 1e-9 else 0.0   # net/total: 1 progressive, 0 returns
    out['dfw_deform_accel'] = float(abs(d12 - d01))                     # deformation speeding up / reversing
    out['dfw_axisrot_net'] = float(r02)                                 # accumulated principal-axis rotation over window
    f0, f2 = _frame_state(m0), _frame_state(m2)   # centerline Po lives here, not in nonrigid _state
    try: out['dfw_tps_net'] = float(_tps_bend(f0['Po'], f2['Po'])) if (f0 and f2) else np.nan      # TPS bending T-1 -> T+1
    except Exception: out['dfw_tps_net'] = np.nan
    try: out['dfw_srvf_net'] = float(_srvf_dist(f0['Po'], f2['Po'])) if (f0 and f2) else np.nan    # elastic-shape dist T-1 -> T+1
    except Exception: out['dfw_srvf_net'] = np.nan
    return out

def dfw_from_masks(basename, mask_index):
    nan = {c: np.nan for c in DFW_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return deform_window(gs[0], gs[1], gs[2])
