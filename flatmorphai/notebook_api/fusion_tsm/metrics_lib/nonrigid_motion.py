# nonrigid_motion.py
# CANDIDATE metrics built to fix the largest failure pattern found by hardcase_diagnosis: a gliding Snake or Screw
# reads as a Wildtype because raw motion-magnitude (mot_mag, mot_p90, mot_edge_frac) only answers "is it moving".
# The discriminator that is missing is whether the motion is RIGID (the whole body translates/glides, Wildtype) or
# NON-RIGID (the body deforms in place: Snake undulation, Screw twist). Within the fixed 3-frame window (T-1,T0,T+1)
# each neighbour mask is aligned to the current one by removing the rigid part (centroid translation + principal-axis
# rotation), and the residual symmetric-difference is the non-rigid body deformation. Prefix nrm_. No GPU, seed 42.
# Basis: rigid vs non-rigid decomposition of shape motion, worm posture dynamics over time (Stephens, Johnson-Kerner,
# Bialek, Ryu, "Dimensionality and Dynamics in the Behavior of C. elegans", PLoS Comput Biol 2008,
# DOI 10.1371/journal.pcbi.1000028; medial-axis shape, Blum 1967, no DOI).
import numpy as np, cv2
from twist_motion_masks import triplet

NRM_COLS = ['nrm_deform', 'nrm_rigid_trans', 'nrm_axis_rot', 'nrm_frac', 'nrm_deform_local']

def _state(mk):
    m = (mk > 0).astype(np.uint8)
    if m.sum() < 150: return None
    nn, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if nn <= 1: return None
    wi = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); w = (lab == wi).astype(np.uint8)
    M = cv2.moments(w, binaryImage=True); A = M['m00']
    if A < 150: return None
    cx, cy = M['m10'] / A, M['m01'] / A
    mu20, mu02, mu11 = M['mu20'] / A, M['mu02'] / A, M['mu11'] / A
    theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02) + 1e-9)   # principal-axis orientation
    return dict(w=w, A=float(A), c=np.array([cx, cy]), theta=float(theta), r=float(np.sqrt(A / np.pi)))

def _residual(prev, cur, dtheta):
    # align prev to cur by rotating prev about its centroid by dtheta then translating centroid onto cur, return
    # the non-rigid residual = symmetric-difference area / current area, and the residual map for locality
    h, wd = cur['w'].shape
    Mrot = cv2.getRotationMatrix2D((float(prev['c'][0]), float(prev['c'][1])), float(np.degrees(dtheta)), 1.0)
    Mrot[0, 2] += cur['c'][0] - prev['c'][0]; Mrot[1, 2] += cur['c'][1] - prev['c'][1]
    al = cv2.warpAffine(prev['w'], Mrot, (wd, h), flags=cv2.INTER_NEAREST)
    xor = cv2.bitwise_xor(al, cur['w'])
    return float(xor.sum()) / (cur['A'] + 1e-6), xor

def _pair(prev, cur):
    # rigid translation (normalized glide), best non-rigid residual over a small rotation search, and chosen rotation
    trans = float(np.linalg.norm(cur['c'] - prev['c'])) / (cur['r'] + 1e-6)
    d0 = cur['theta'] - prev['theta']
    d0 = (d0 + np.pi / 2) % np.pi - np.pi / 2   # wrap to [-pi/2, pi/2], consecutive frames turn little
    best = None
    for dt in (0.0, d0, -d0):
        res, xmap = _residual(prev, cur, dt)
        if best is None or res < best[0]: best = (res, abs(dt), xmap)
    deform, rot, xmap = best
    # locality: largest connected residual blob / total residual (a twist is localized, an undulation distributed)
    nb, lb, st, _ = cv2.connectedComponentsWithStats(xmap, 8)
    loc = (float(st[1:, cv2.CC_STAT_AREA].max()) / (xmap.sum() + 1e-6)) if nb > 1 else 0.0
    return deform, trans, rot, loc

def nonrigid_motion(m_prev, m_cur, m_next):
    nan = {c: np.nan for c in NRM_COLS}
    sp, sc, sn = _state(m_prev), _state(m_cur), _state(m_next)
    if sc is None: return nan
    out = []
    for s in (sp, sn):
        if s is not None: out.append(_pair(s, sc))
    if not out: return nan
    deform = float(np.mean([o[0] for o in out])); trans = float(np.mean([o[1] for o in out]))
    rot = float(np.mean([o[2] for o in out])); loc = float(np.mean([o[3] for o in out]))
    frac = deform / (deform + trans + 1e-6)   # fraction of motion that is body deformation vs rigid glide
    return {'nrm_deform': deform, 'nrm_rigid_trans': trans, 'nrm_axis_rot': rot, 'nrm_frac': frac, 'nrm_deform_local': loc}

def nrm_from_masks(basename, mask_index):
    nan = {c: np.nan for c in NRM_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return nonrigid_motion(gs[0], gs[1], gs[2])
