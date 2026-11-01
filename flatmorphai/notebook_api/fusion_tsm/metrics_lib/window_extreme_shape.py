# window_extreme_shape.py
# 37% of the TRUE hard cases say the discriminating trait is visible in a NEIGHBOUR frame (T-1, T+1, or T-2/T+2 at
# clip ends), not the labelled T0: "twist visible at T+2", "multiple twists at T-1 and T+1", "zigzag clearly at T+1".
# The strong single-frame shape cues (be_max_turn, be_self_overlap, be_hourglass, be_n_inflect, neck depth, solidity)
# are computed only on T0, and the image branch only sees T0, so a trait that peaks one frame away is missed. This
# panel recomputes those strong cues on T-1,T0,T+1 and keeps the extreme (max for turn/overlap/hourglass/inflect/
# concave, min for solidity), so an intermittent twist or zigzag is caught. Prefix wte_. masks only, no GPU.
import numpy as np, cv2
from twist_border_evolution import border_evolution
from topology_motion import _largest, _solidity
from twist_motion_masks import triplet

WTE_COLS = ['wte_maxturn_max', 'wte_selfoverlap_max', 'wte_hourglass_max', 'wte_ninflect_max', 'wte_neckdepth_max', 'wte_nconcave_max', 'wte_solidity_min']

def window_extreme_shape(m0, m1, m2):
    nan = {c: np.nan for c in WTE_COLS}
    masks = [m for m in (m0, m1, m2) if m is not None and (m > 0).sum() > 150]
    if len(masks) < 2: return nan
    bes, sols = [], []
    for m in masks:
        mm = (m > 0).astype(np.uint8) * 255
        try: bes.append(border_evolution(mm))
        except Exception: bes.append(None)
        w = _largest(mm); sols.append(_solidity(w) if w is not None else np.nan)
    bok = [b for b in bes if b is not None]
    if not bok: return nan
    def mx(k):
        v = [b.get(k, np.nan) for b in bok]; return float(np.nanmax(v)) if not all(np.isnan(v)) else np.nan
    out = {'wte_maxturn_max': mx('be_max_turn'), 'wte_selfoverlap_max': mx('be_self_overlap'),
           'wte_hourglass_max': mx('be_hourglass'), 'wte_ninflect_max': mx('be_n_inflect'),
           'wte_neckdepth_max': mx('be_neck_depth'), 'wte_nconcave_max': mx('be_n_concave'),
           'wte_solidity_min': float(np.nanmin(sols)) if sols and not all(np.isnan(sols)) else np.nan}
    return out

def wte_from_masks(basename, mask_index):
    nan = {c: np.nan for c in WTE_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return window_extreme_shape(gs[0], gs[1], gs[2])


def window_extreme_from_grays(g0, g1, g2):
    # masked-grayscale frames (worm brightness on black). Shape cues via border_evolution(gray) so the call is SHARED
    # (memoised) with twist_texture_temporal, halving the heavy centerline traces. Solidity from the implied mask.
    nan = {c: np.nan for c in WTE_COLS}
    grays = [g for g in (g0, g1, g2) if g is not None and (g > 8).sum() > 150]
    if len(grays) < 2: return nan
    bes, sols = [], []
    for g in grays:
        try: bes.append(border_evolution(g))
        except Exception: bes.append(None)
        w = _largest((g > 8).astype(np.uint8) * 255); sols.append(_solidity(w) if w is not None else np.nan)
    bok = [b for b in bes if b is not None]
    if not bok: return nan
    def mx(k):
        v = [b.get(k, np.nan) for b in bok]; return float(np.nanmax(v)) if not all(np.isnan(v)) else np.nan
    return {'wte_maxturn_max': mx('be_max_turn'), 'wte_selfoverlap_max': mx('be_self_overlap'),
            'wte_hourglass_max': mx('be_hourglass'), 'wte_ninflect_max': mx('be_n_inflect'),
            'wte_neckdepth_max': mx('be_neck_depth'), 'wte_nconcave_max': mx('be_n_concave'),
            'wte_solidity_min': float(np.nanmin(sols)) if sols and not all(np.isnan(sols)) else np.nan}
