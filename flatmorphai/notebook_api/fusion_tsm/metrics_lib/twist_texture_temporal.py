# twist_texture_temporal.py
# CANDIDATE metrics for the largest remaining Screw gap (hardcase audit): a 3D helical twist that lives in the
# grayscale SHADING, not the silhouette, and often only appears or develops across T-1,T0,T+1. The single-frame
# twist-texture cues (validated against the expert cyan-border annotations) run on T0 only and miss it. This panel
# extends them over the 3-frame window and adds a flow-rotation cue. Prefix twt_. No GPU, seed 42.
#   twt_crosslr_max   max over the window of the cross-section left/right shading asymmetry (rolling surface at a twist)
#   twt_crosslr_dyn   how much that L/R shading asymmetry CHANGES across the window (a twist rotating/developing)
#   twt_crossstd_max  max cross-section intensity spread (texture across the body width)
#   twt_texstep_max   max along-axis translucent intensity step (lighter region at/after the twist)
#   twt_darkwide_max  max wider-and-darker fold (self-overlap of a twisting body)
#   twt_flowcurl      mean |curl| of dense optical flow inside the worm: a twist is rotational, a glide translational
# Basis: across-body shading localises the twist (cyan-annotation result); rolling/twisting surface shading and worm
# roll, Bilbao et al., Roll maneuvers, PNAS 2018, DOI 10.1073/pnas.1706754115; Gabor/texture, Daugman 1985,
# DOI 10.1364/JOSAA.2.001160; optical strain/flow non-rigid motion, Shreve et al., IEEE FG 2011, DOI 10.1109/FG.2011.5771451.
import numpy as np, cv2
from twist_border_evolution import border_evolution

TWT_COLS = ['twt_crosslr_max', 'twt_crosslr_dyn', 'twt_crossstd_max', 'twt_texstep_max', 'twt_darkwide_max', 'twt_flowcurl']

def _masked_gray(raw, mask):
    if raw is None or mask is None: return None
    if raw.shape != mask.shape: raw = cv2.resize(raw, (mask.shape[1], mask.shape[0]))
    return np.where(mask > 0, raw, 0).astype(np.uint8)

def _flow_curl(g0, g1, mask):
    if g0 is None or g1 is None or g0.shape != g1.shape: return np.nan
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = fl[..., 0], fl[..., 1]
    curl = cv2.Sobel(v, cv2.CV_32F, 1, 0, 3) - cv2.Sobel(u, cv2.CV_32F, 0, 1, 3)   # dv/dx - du/dy = rotation
    m = mask > 0
    return float(np.abs(curl)[m].mean()) if m.sum() > 50 else np.nan

def twist_texture_temporal(mg_prev, mg_cur, mg_next):
    # masked grayscale frames (worm brightness on black) for T-1, T0, T+1
    nan = {c: np.nan for c in TWT_COLS}
    frames = [g for g in (mg_prev, mg_cur, mg_next) if g is not None and (g > 8).sum() > 150]
    if not frames: return nan
    lr, cs, ts, dw = [], [], [], []
    for g in frames:
        try: be = border_evolution(g)
        except Exception: continue
        lr.append(be.get('be_cross_lrasym', np.nan)); cs.append(be.get('be_cross_std', np.nan))
        ts.append(be.get('be_axial_tex_step', np.nan)); dw.append(be.get('be_darkwide', np.nan))
    out = dict(nan)
    if lr and not all(np.isnan(lr)):
        out['twt_crosslr_max'] = float(np.nanmax(lr))
        out['twt_crosslr_dyn'] = float(np.nansum(np.abs(np.diff(lr)))) if len(lr) > 1 else 0.0
    if cs and not all(np.isnan(cs)): out['twt_crossstd_max'] = float(np.nanmax(cs))
    if ts and not all(np.isnan(ts)): out['twt_texstep_max'] = float(np.nanmax(ts))
    if dw and not all(np.isnan(dw)): out['twt_darkwide_max'] = float(np.nanmax(dw))
    cur = mg_cur if mg_cur is not None else frames[len(frames) // 2]
    cu = []
    if mg_prev is not None: cu.append(_flow_curl(mg_prev, cur, cur))
    if mg_next is not None: cu.append(_flow_curl(cur, mg_next, cur))
    cu = [x for x in cu if not np.isnan(x)]
    if cu: out['twt_flowcurl'] = float(np.mean(cu))
    return out

_BE_CACHE = {}; _MG_CACHE = {}
def _mg(nm, name2path, mask_index):
    if nm not in _MG_CACHE:
        rp = name2path.get(nm); mp = mask_index.get(nm)
        if rp is None or mp is None: _MG_CACHE[nm] = None
        else:
            raw = cv2.imdecode(np.fromfile(rp, np.uint8), cv2.IMREAD_GRAYSCALE); mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            _MG_CACHE[nm] = _masked_gray(raw, mk)
    return _MG_CACHE[nm]
def _be_cached(nm, mg):
    if nm not in _BE_CACHE:
        try: _BE_CACHE[nm] = border_evolution(mg) if (mg is not None and (mg > 8).sum() > 150) else None
        except Exception: _BE_CACHE[nm] = None
    return _BE_CACHE[nm]
def twt_from_raw(basename, name2path, mask_index):
    # build the 3-frame window from raw frames (full coverage) + masks; per-frame border_evolution is CACHED so the
    # heavy centerline trace runs once per frame even though triplets overlap.
    from twist_motion_masks import triplet
    nan = {c: np.nan for c in TWT_COLS}; tn = triplet(basename)
    if tn is None: return nan
    mgs = [_mg(nm, name2path, mask_index) for nm in tn]
    bes = [_be_cached(nm, mg) for nm, mg in zip(tn, mgs)]
    bes_ok = [b for b in bes if b is not None]
    if not bes_ok: return nan
    lr = [b.get('be_cross_lrasym', np.nan) for b in bes_ok]; cs = [b.get('be_cross_std', np.nan) for b in bes_ok]
    ts = [b.get('be_axial_tex_step', np.nan) for b in bes_ok]; dw = [b.get('be_darkwide', np.nan) for b in bes_ok]
    out = dict(nan)
    if not all(np.isnan(lr)):
        out['twt_crosslr_max'] = float(np.nanmax(lr)); out['twt_crosslr_dyn'] = float(np.nansum(np.abs(np.diff(lr)))) if len(lr) > 1 else 0.0
    if not all(np.isnan(cs)): out['twt_crossstd_max'] = float(np.nanmax(cs))
    if not all(np.isnan(ts)): out['twt_texstep_max'] = float(np.nanmax(ts))
    if not all(np.isnan(dw)): out['twt_darkwide_max'] = float(np.nanmax(dw))
    cur = mgs[1] if mgs[1] is not None else next((g for g in mgs if g is not None), None)
    cu = []
    if mgs[0] is not None and cur is not None: cu.append(_flow_curl(mgs[0], cur, cur))
    if mgs[2] is not None and cur is not None: cu.append(_flow_curl(cur, mgs[2], cur))
    cu = [x for x in cu if not np.isnan(x)]
    if cu: out['twt_flowcurl'] = float(np.mean(cu))
    return out
