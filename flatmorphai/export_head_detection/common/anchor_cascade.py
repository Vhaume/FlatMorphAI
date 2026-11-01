# -*- coding: utf-8 -*-
"""Head-anchoring cascade, single source for the whole chain. EXPORT VERSION: constants read from parameters/.
Logic identical to the version in service in the project (anchor_cascade_original_fr.py, 4 October 2026).

    Three steps evaluated at BOTH ends of the straightened worm, ordered from the most specific index to
    the most general. Each step decides if its condition is met, otherwise it hands over.

        Step 1, eyes. Detection at both ends by the eye detector (models.csv: eye_detector) on a thumbnail
        of fraction EYES_FRACTION = 0.30. Compared quantity, mass of the peaks above PEAK_THR = 0.60.
        Condition, a maximum response >= PEAK_ANCHOR = 0.85 on at least one end.
        Step 2, blastema. Segmentation at both ends by the blastema segmenter (models.csv: blastema_segmenter),
        trained on 764 worms with a split stratified by state and grouped by worm, on a thumbnail of
        fraction BLASTEMA_FRACTION = 0.45 whose mask is dilated by 0.12 of the length. Compared quantity,
        soft area. Condition, relative margin >= BLASTEMA_MARGIN = 0.84.
        Step 3, body darkness. The width-aligned medial axis is resampled to N_AXIS = 200 points. At each
        point a perpendicular cut bounded by the mask reads the worm pixels. The N_CUTS = 60 cuts from
        abscissa 0.00 to 0.30 from one end feed the pixel set of that end, the 60 symmetric cuts feed the
        other. On each set, in grey levels, the Otsu threshold separates dark pixels from bright ones, and
        the darkness index is 1 minus the median of the dark pixels divided by 255. Decision, the end with
        the highest index, i.e. the darkest, is retained, without a threshold on the gap between the two
        indices (DARKNESS_GAP = 0.0, version of 4 October 2026).
        Fallback. If the medial axis cannot be computed the darkness index does not exist and the anchor
        falls back on the largest soft blastema area, step marked undetermined. Never happened on the
        2705 worms of the annotated corpus.

    Accuracy of the full cascade, 0.9952 on 2705 worms with decidable expert annotation, i.e. 13 errors,
    after removing the 47 worms carrying cyan on both sides and the 15 worms whose single mark falls
    between 0.35 and 0.65 of the length. Out-of-fold accuracy, folds grouped by worm, 0.9952. Decisions,
    eyes 1443 worms and 0 errors, blastema 1091 worms and 10 errors, darkness 171 worms and 3 errors.
    Measured on 4 October 2026.

    Verified references, not clickable. U-Net, O. Ronneberger, P. Fischer, T. Brox, MICCAI 2015,
    arXiv 1505.04597, DOI 10.1007/978-3-319-24574-4_28. EfficientNet, M. Tan, Q. Le, ICML 2019,
    arXiv 1905.11946. Exact distance transform, C. R. Maurer, R. Qi, V. Raghavan, IEEE TPAMI, 2003,
    DOI 10.1109/TPAMI.2003.1177156. Medial axis, H. Blum, Models for the Perception of Speech and Visual
    Form, 1967. scikit-image, S. van der Walt et al., PeerJ, 2014, DOI 10.7717/peerj.453.
"""
import os, sys, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "parameters")):
    if _p not in sys.path: sys.path.insert(0, _p)
from width_axis import medial_axis, smooth
from parameter_reader import read, model_path

_PC = read("cascade_parameters"); _PT = read("eye_thresholds"); _PV = read("thumbnails")
np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = int(_PV["SIZE"]); F_BL = float(_PV["BLASTEMA_FRACTION"]); F_EYE = float(_PV["EYES_FRACTION"])
PEAK_THR = float(_PT["position_threshold"]); MINDIST = int(_PT["min_peak_distance"])
N_PEAKS_POSITION = int(_PT["max_peaks_for_position"])
BLASTEMA_MARGIN = float(_PC["BLASTEMA_MARGIN"]); DARKNESS_GAP = float(_PC["DARKNESS_GAP"])
PEAK_ANCHOR = float(_PT["head_decision_threshold"])
NM = [0.485, 0.456, 0.406, 0.20]; NS = [0.229, 0.224, 0.225, 0.15]
N_AXIS = int(_PC["N_AXIS"]); N_CUTS = int(_PC["N_CUTS"])
DARK_WINDOW = tuple(float(v) for v in _PC["DARK_WINDOW"])
BAND_POS = float(_PC["TERMINAL_BAND_POSITION"]); DIL_BL = float(_PV["BLASTEMA_MASK_DILATION"])
CK_BL = model_path("blastema_segmenter")
CK_EY = model_path("eye_detector")


class _Bk(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("m", torch.tensor(NM).view(1, 4, 1, 1))
        self.register_buffer("s", torch.tensor(NS).view(1, 4, 1, 1))

    def forward(self, x): return torch.sigmoid(self.net((x - self.m) / self.s))


def _load(ck):
    c = torch.load(ck, map_location=DEVICE)
    m = _Bk(smp.Unet(encoder_name="timm-efficientnet-b0", encoder_weights=None,
                     in_channels=4, classes=1, activation=None)).to(DEVICE)
    m.load_state_dict({k: v for k, v in c["state_dict"].items() if not k.startswith("_enc.")})
    m.eval(); return m, c


_MDL = {}


def load_models(verbose=True):
    """Load the two networks once. Returns (blastema model, eye model)."""
    if not _MDL:
        mb, cb = _load(CK_BL); me, ce = _load(CK_EY)
        _MDL["bl"] = mb; _MDL["ey"] = me
        if verbose:
            print("cascade anchoring, device:", DEVICE)
            print("   step 1 eyes     :", os.path.basename(CK_EY), "| peak threshold", PEAK_THR, "| decision threshold", PEAK_ANCHOR)
            print("   step 2 blastema :", os.path.basename(CK_BL), "| validation Dice", round(float(cb.get("val_dice", 0.0)), 4))
            print("   step 3 darkness : width-aligned medial axis, no learned model")
    return _MDL["bl"], _MDL["ey"]


def _chans(bgr):
    """Four input channels: RGB / 255 and local contrast z = (local mean - grey) / local std, (z + 1) / 6 in [0, 1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lm = cv2.blur(g, (21, 21)); var = cv2.blur(g * g, (21, 21)) - lm * lm
    ls = np.sqrt(np.maximum(var, 0.0)) + 1e-3
    zc = np.clip(((lm - g) / ls + 1.0) / 6.0, 0.0, 1.0)
    x = np.concatenate([(rgb.astype(np.float32) / 255.0).transpose(2, 0, 1), zc[None]], 0)
    return torch.from_numpy(x.astype(np.float32))


def _pred(mdl, crop):
    with torch.no_grad():
        return mdl(_chans(crop)[None].to(DEVICE)).cpu()[0, 0].numpy()


def ends(rm):
    """Centres of the two terminal bands of the straightened worm, top then bottom. Returns (points, span)."""
    ys, xs = np.nonzero(rm); y0, y1 = ys.min(), ys.max(); span = int(y1 - y0 + 1)
    out = []
    for yc in (y0 + int(BAND_POS * span), y1 - int(BAND_POS * span)):
        row = rm[yc]
        out.append((int(np.where(row)[0].mean()) if row.any() else rm.shape[1] // 2, int(yc)))
    return out, span


def thumbnail(rbgr, msk, cx, cy, span, frac):
    """Square thumbnail of half-side frac x span around (cx, cy), background zeroed outside msk, padded to a
    square and resized to S. Returns (thumbnail, box) with box = (y0, x0, oy, ox, side), or None."""
    h = int(frac * span); H, W = msk.shape
    y0 = max(0, int(cy) - h); y1 = min(H, int(cy) + h)
    x0 = max(0, int(cx) - h); x1 = min(W, int(cx) + h)
    # crop FIRST, then zero the background (same pixels as copying and
    # masking the whole full-resolution frame, 10x less memory traffic)
    c = rbgr[y0:y1, x0:x1].copy(); c[~msk[y0:y1, x0:x1]] = 0
    if c.size == 0: return None
    ch, cw = c.shape[:2]; side = max(ch, cw); oy = (side - ch) // 2; ox = (side - cw) // 2
    sq = np.zeros((side, side, 3), np.uint8); sq[oy:oy + ch, ox:ox + cw] = c
    return cv2.resize(sq, (S, S)), (y0, x0, oy, ox, side)


def _peaks(hm):
    """Highest local maxima above PEAK_THR, at most N_PEAKS_POSITION (used for the head POSITION only)."""
    # max_pool2d(k, stride 1, pad k//2) == grey dilation with a k x k square,
    # out-of-image samples ignored: cv2.dilate (exact, max is exact) instead of
    # torch on the CPU (~5 ms per call measured)
    k = 2 * MINDIST + 1
    h32 = np.ascontiguousarray(hm, dtype=np.float32)
    mx = cv2.dilate(h32, np.ones((k, k), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=-np.inf) \
        if h32.dtype == hm.dtype else F.max_pool2d(torch.from_numpy(hm)[None, None], k, 1, MINDIST)[0, 0].numpy()
    ys, xs = np.nonzero((hm == mx) & (hm >= PEAK_THR))
    return sorted(zip(xs.tolist(), ys.tolist(), hm[ys, xs].tolist()), key=lambda q: -q[2])[:N_PEAKS_POSITION]


def to_straight_frame(px, py, box):
    """Thumbnail coordinates -> straightened-image coordinates."""
    y0, x0, oy, ox, side = box; sc = side / float(S)
    return px * sc - ox + x0, py * sc - oy + y0


_LEVELS = np.arange(256.0)


def _histograms(P, msk, gf8, half):
    """Grey-level histograms read by the N_CUTS cuts of each end.

    A cut is perpendicular to the medial axis and bounded by the worm mask, no outside pixel is read.
    Cuts of index 0 to N_CUTS - 1 cover abscissa 0 to 0.30 from end P[0], the last N_CUTS cover the same
    portion from end P[-1].
    """
    H, W = msk.shape; n = len(P)
    h0 = np.zeros(256, np.int64); h1 = np.zeros(256, np.int64)
    for i in range(n):
        if N_CUTS <= i < n - N_CUTS: continue
        j0, j1 = max(0, i - 2), min(n - 1, i + 2)
        ty, tx = P[j1] - P[j0]; nn = np.hypot(ty, tx) + 1e-9
        py, px = -tx / nn, ty / nn
        v = []
        for r in range(-half, half + 1):
            Y, X = int(round(P[i][0] + py * r)), int(round(P[i][1] + px * r))
            if 0 <= Y < H and 0 <= X < W and msk[Y, X] > 0: v.append(gf8[Y, X])
        if not v: continue
        hh = np.bincount(np.array(v, np.int64), minlength=256)
        if i < N_CUTS: h0 += hh
        else: h1 += hh
    return h0, h1


def _darkness_index(h):
    """Darkness index of one end, 1 minus the median of the dark pixels over 255.

    The Otsu threshold separates dark from bright pixels on the grey-level distribution of the end,
    the median is taken on the dark class only.
    """
    tot = h.sum()
    if tot < 10: return float("nan")
    p = h / float(tot); w0 = np.cumsum(p); w1 = 1.0 - w0
    m0 = np.cumsum(p * _LEVELS); mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (mt * w0 - m0) ** 2 / (w0 * w1)
    var[~np.isfinite(var)] = -1.0
    t = int(np.argmax(var))
    d = h[:t + 1]
    if d.sum() < 1: return float("nan")
    c = np.cumsum(d); med = int(np.searchsorted(c, d.sum() / 2.0))
    return 1.0 - med / 255.0


def darkness(rbgr, rm):
    """Step 3. Returns (index of the retained end in top/bottom order, gap between the two indices)."""
    pa = medial_axis(rm.astype(np.uint8))
    if pa is None: return None, 0.0
    P, L = smooth(pa, N_AXIS)
    if P is None or L < 20: return None, 0.0
    gf8 = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY)
    half = max(3, int(round(0.5 * np.sqrt(rm.sum()))))
    h0, h1 = _histograms(P, rm.astype(np.uint8), gf8, half)
    n0 = _darkness_index(h0); n1 = _darkness_index(h1)
    if not (np.isfinite(n0) and np.isfinite(n1)): return None, 0.0
    top_is_P0 = P[0][0] <= P[-1][0]      # is P[0] the top end
    winner_is_P0 = n0 > n1
    i = 0 if (winner_is_P0 == top_is_P0) else 1
    return i, abs(n0 - n1)


def anchor(rbgr, rm, verbose=False):
    """Full cascade on an already straightened worm.

    Returns (x, y, step, confidence, end): the point is the barycentre of the detector peaks when an eye is
    detected and the centre of the retained terminal band otherwise, end being 0 for the top end and 1 for
    the bottom end. step is one of "eyes", "blastema", "darkness", "undetermined".
    """
    mdl_bl, mdl_ey = load_models(verbose)
    pts, span = ends(rm)
    k = max(1, int(DIL_BL * span))
    rmz = cv2.dilate(rm.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    soft = []
    for cx, cy in pts:
        v = thumbnail(rbgr, rmz, cx, cy, span, F_BL)
        if v is None: soft.append(0.0); continue
        p = _pred(mdl_bl, v[0]); soft.append(float(p.sum()) * (v[1][4] / S) ** 2)
    margin = abs(soft[0] - soft[1]) / (soft[0] + soft[1] + 1e-9)
    eye = []
    for cx, cy in pts:
        v = thumbnail(rbgr, rm, cx, cy, span, F_EYE)
        if v is None: eye.append((0.0, 0.0, [], None)); continue
        p = _pred(mdl_ey, v[0])
        eye.append((float(p[p >= PEAK_THR].sum()), float(p.max()), _peaks(p), v[1]))
    rmax = max(e[1] for e in eye)
    gap = 0.0; j = None
    if rmax >= PEAK_ANCHOR:
        # Head = the end with the MOST eyespots above the peak threshold; on a
        # tie, the end whose eyespots have the higher MEAN probability. (The
        # previous rule compared the summed peak "mass", which could prefer a
        # single strong blob over an end carrying two genuine eyes.) ``eye[k]``
        # is (mass, max_response, peaks, box); ``peaks`` are the maxima above
        # PEAK_THR, each as (x, y, height).
        def _eye_key(e):
            pks = e[2]
            c = len(pks)
            return (c, (sum(q[2] for q in pks) / c) if c else 0.0)
        i = 0 if _eye_key(eye[0]) >= _eye_key(eye[1]) else 1; step = "eyes"; conf = rmax
    elif margin >= BLASTEMA_MARGIN:
        i = 0 if soft[0] > soft[1] else 1; step = "blastema"; conf = margin
    else:
        j, gap = darkness(rbgr, rm)
        if j is not None and gap >= DARKNESS_GAP:
            i = j; step = "darkness"; conf = gap
        else:
            i = j if j is not None else (0 if soft[0] > soft[1] else 1)
            step = "undetermined"; conf = max(margin, gap)
    mass, r, pk, box = eye[i]
    if pk and box is not None:
        p2 = [to_straight_frame(px, py, box) for px, py, _ in pk]
        x = float(np.mean([q[0] for q in p2])); y = float(np.mean([q[1] for q in p2]))
    else:
        x, y = float(pts[i][0]), float(pts[i][1])
    return x, y, step, float(conf), int(i)
