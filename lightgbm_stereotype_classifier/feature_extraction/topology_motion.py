# topology_motion.py
# Motion-oriented TOPOLOGY metrics for the Screw gap. A Screw that rounds on itself encloses background and forms a
# hole/loop, and a fold collapses solidity and deepens concave pockets. These are connectivity, not geometry, so no
# width/curvature metric captures them. Measured over T-1,T0,T+1 so the FORMING of a loop/fold over time is the cue
# (a Screw twists into a loop, a Snake undulates without one, a Wildtype glides). Prefix tpm_. masks only, no GPU.
#   tpm_holes_max     max enclosed-hole count over the window (a self-crossing twist encloses background)
#   tpm_holes_dyn     change in hole count over the window (a loop forming = twisting)
#   tpm_solidity_min  min area/convex-hull ratio (a fold makes the body non-convex)
#   tpm_solidity_dyn  change in solidity over the window (folding/unfolding over time)
#   tpm_defect_dyn    change in deepest convexity-defect depth over the window (a fold pocket deepening)
#   tpm_skelloop_max  max skeleton loop count (1 - Euler number of the skeleton; a folded worm's skeleton has cycles)
# Basis: Euler-number / Betti topology of shape (Serra, Image Analysis and Mathematical Morphology, 1982, no DOI);
# convexity defects and solidity for non-convex body folds; medial-axis skeleton, Blum 1967 (no DOI).
import numpy as np, cv2
from skimage.measure import euler_number
from skimage.morphology import skeletonize
from twist_motion_masks import triplet

TPM_COLS = ['tpm_holes_max', 'tpm_holes_dyn', 'tpm_solidity_min', 'tpm_solidity_dyn', 'tpm_defect_dyn', 'tpm_skelloop_max']

def _largest(m):
    mm = (m > 0).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
    if n <= 1: return None
    wi = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); return (lab == wi).astype(np.uint8)

def _holes(w):
    cnts, hier = cv2.findContours(w, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return 0
    return int(sum(1 for h in hier[0] if h[3] != -1))   # inner contours = enclosed holes

def _solidity(w):
    cnts, _ = cv2.findContours(w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.nan
    c = max(cnts, key=cv2.contourArea); a = cv2.contourArea(c); ha = cv2.contourArea(cv2.convexHull(c))
    return float(a / (ha + 1e-6))

def _defect(w):
    cnts, _ = cv2.findContours(w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.nan
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 4: return 0.0
    try: d = cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False))
    except cv2.error: return 0.0
    if d is None: return 0.0
    d = np.asarray(d).reshape(-1, 4)   # FIX: convexityDefects returns (N,1,4) or (N,4) depending on the cv2 version
    return float(d[:, 3].max()) / 256.0 / (np.sqrt(cv2.contourArea(c)) + 1e-6)   # deepest pocket, scale-normalised

def _skelloops(w):
    try: e = euler_number(skeletonize(w > 0), connectivity=2)
    except Exception: return 0.0
    return float(max(0, 1 - e))   # connected skeleton: loops = 1 - Euler

def topology_motion(m0, m1, m2):
    nan = {c: np.nan for c in TPM_COLS}
    ws = [w for w in (_largest(m0) if m0 is not None else None, _largest(m1) if m1 is not None else None, _largest(m2) if m2 is not None else None) if w is not None and w.sum() > 150]
    if len(ws) < 2: return nan
    holes = [_holes(w) for w in ws]; sol = [_solidity(w) for w in ws]; df = [_defect(w) for w in ws]; sl = [_skelloops(w) for w in ws]
    def dyn(a): return float(np.nansum([abs(a[i + 1] - a[i]) for i in range(len(a) - 1)]))
    return {'tpm_holes_max': float(np.max(holes)), 'tpm_holes_dyn': dyn(holes),
            'tpm_solidity_min': float(np.nanmin(sol)), 'tpm_solidity_dyn': dyn(sol),
            'tpm_defect_dyn': dyn(df), 'tpm_skelloop_max': float(np.max(sl))}

def tpm_from_masks(basename, mask_index):
    nan = {c: np.nan for c in TPM_COLS}; tn = triplet(basename)
    if tn is None: return nan
    gs = []
    for nm in tn:
        p = mask_index.get(nm)
        if p is None: return nan
        mk = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mk is None: return nan
        gs.append((mk > 0).astype(np.uint8) * 255)
    return topology_motion(gs[0], gs[1], gs[2])
