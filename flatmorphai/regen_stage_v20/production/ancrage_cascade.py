# -*- coding: utf-8 -*-
"""Head anchoring cascade, single source for the whole chain.

    Three stages evaluated at BOTH ends of the straightened worm, ordered from the most specific
    index to the most general. Each stage decides if its condition is met, otherwise it hands over.

        Stage 1, eyespots. Detection at both ends by eye_unet_unified_384_rgbz_hn_f030_sansrenfort on
        a thumbnail of fraction 0.30. Compared quantity: mass of the peaks above PEAK_THR = 0.60.
        Condition: a maximal response greater than or equal to PEAK_ANCRAGE = 0.85 on at least one end.
        Stage 2, blastema. Segmentation at both ends by blasteme_unet_384_stratifie, trained on
        764 worms with a split stratified by state and grouped by worm, on a thumbnail of fraction
        0.45 whose mask is dilated by 0.12 of the length. Compared quantity: soft area. Condition:
        relative margin greater than or equal to MARGE_BL = 0.84.
        Stage 3, body darkness. The median axis aligned on the width is resampled to N_AXE = 200
        points. At each point a perpendicular section bounded by the mask reads the worm pixels. The
        NB_COUPES = 60 sections from abscissa 0.00 to 0.30 from one end feed the pixel set of that
        end, the 60 symmetric sections feed the other one. On each set, converted to greyscale,
        Otsu's threshold separates the dark pixels from the pale ones, and the darkness index is 1
        minus the median of the dark pixels divided by 255. Decision: the end whose index is the
        highest -- that is, the darkest -- is retained, with no threshold on the gap between the two
        indices (ECART_D = 0.0, version of 4 October 2026). The 0.015 threshold used before did not
        change the orientation, it only labelled the reading, so it was removed.
        Fallback. If the median axis cannot be computed, the darkness index does not exist and the
        anchor falls back on the largest soft blastema area, a stage marked as undetermined. This
        fallback occurred on none of the 2705 worms of the annotated corpus.

    Accuracy of the complete cascade: 0.9952 over 2705 worms with a decidable expert annotation, that
    is 13 errors, after removing the 47 worms carrying cyan on both sides and the 15 worms whose
    single mark falls between 0.35 and 0.65 of the length (annotations_mediane_a_ecarter.csv).
    Out-of-fold accuracy, folds grouped by worm: 0.9952. Breakdown of the decisions: eyespots 1443
    worms and 0 errors, blastema 1091 worms and 10 errors, darkness 171 worms and 3 errors.
    Measurement of 4 October 2026.

    Documented choices. The eyespots-then-blastema order was preferred over the reverse, which leaves
    11 errors instead of 3 on states 2 to 6. The stratified segmenter replaced blasteme_unet_384,
    trained on 316 worms of which 41 were State_2 and none State_5 or State_6, which plateaued at
    0.9890 over the six states. On the 1330 worms seen by neither segmenter, the blastema index rises
    from 0.9806 to 0.9955. Four leads were measured and discarded: darkness promoted to the first
    stage, a minimal ratio of blastema area over the body, a positional reading of the blastema map
    restricted to the distal third, and a narrower thumbnail field for the eyespot detector. Stage 3
    was re-measured on 3 October 2026 over the 2720 worms, with the section histograms recorded once
    and the variants then compared offline. On the 178 worms the chain routes to that stage, Otsu's
    index over the 60 sections from 0 to 30 per cent gives 171 correct orientations, against 170 for
    a darkness-deficit index over the same window and 168 for the 0.20 to 0.50 window. On the complete
    chain, at identical fallback, Otsu's index leaves 18 errors against 21 for the deficit index read
    from 0.20 to 0.50.

    Verified references, not clickable. U-Net, O. Ronneberger, P. Fischer, T. Brox, MICCAI 2015,
    arXiv 1505.04597, DOI 10.1007/978-3-319-24574-4_28. EfficientNet, M. Tan, Q. Le, ICML 2019,
    arXiv 1905.11946. Exact distance transform, C. R. Maurer, R. Qi, V. Raghavan, IEEE TPAMI,
    2003, DOI 10.1109/TPAMI.2003.1177156. Median axis, H. Blum, Models for the Perception of Speech
    and Visual Form, 1967. scikit-image, S. van der Walt et al., PeerJ, 2014, DOI 10.7717/peerj.453.
"""
import os, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from axe_largeur import axe_median, lisser

np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = 384; F_BL = 0.45; F_EYE = 0.30; PEAK_THR = 0.60; MINDIST = 7
MARGE_BL = 0.84; ECART_D = 0.0   # darkest end, no threshold (4 October 2026)
# DECISION threshold of the eyespot stage for the ANCHORING, distinct from PEAK_THR which stays at
# 0.60 to select the peaks giving the POSITION of the point. Set on 2 October 2026 over 2082 worms
# with a decidable expert annotation, plateau 0.80 to 0.90, out-of-fold 0.9986 against 0.9976 at 0.60.
PEAK_ANCRAGE = 0.85
NM = [0.485, 0.456, 0.406, 0.20]; NS = [0.229, 0.224, 0.225, 0.15]
N_AXE = 200; NB_COUPES = 60; SEG_D = (0.00, 0.30); SEG_C = (0.00, 0.08); POIDS_C = 0.05
CK_BL = os.path.join("checkpoints_regeneration", "blasteme_unet_384_stratifie", "blasteme_unet_384_best.pth")
CK_EY = os.path.join("checkpoints_regeneration", "eye_unet_unified_384_rgbz_hn_f030_sansrenfort", "eye_unet_unified_best.pth")


class _Bk(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("m", torch.tensor(NM).view(1, 4, 1, 1))
        self.register_buffer("s", torch.tensor(NS).view(1, 4, 1, 1))

    def forward(self, x): return torch.sigmoid(self.net((x - self.m) / self.s))


def _charge(ck):
    c = torch.load(ck, map_location=DEVICE)
    m = _Bk(smp.Unet(encoder_name="timm-efficientnet-b0", encoder_weights=None,
                     in_channels=4, classes=1, activation=None)).to(DEVICE)
    m.load_state_dict({k: v for k, v in c["state_dict"].items() if not k.startswith("_enc.")})
    m.eval(); return m, c


_MDL = {}


def charger(verbeux=True):
    """Load both networks once. Returns (blastema model, eyespot model)."""
    if not _MDL:
        mb, cb = _charge(CK_BL); me, ce = _charge(CK_EY)
        _MDL["bl"] = mb; _MDL["ey"] = me
        if verbeux:
            print("cascade anchoring, device:", DEVICE)
            print("   stage 1 blastema:", os.path.basename(os.path.dirname(CK_BL)),
                  "| validation Dice", round(float(cb.get("val_dice", 0.0)), 4))
            print("   stage 2 eyespots:", os.path.basename(os.path.dirname(CK_EY)),
                  "| peak threshold", PEAK_THR)
            print("   stage 3 darkness: median axis aligned on the width, no learned model")
    return _MDL["bl"], _MDL["ey"]


def _chans(bgr):
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


def bouts(rm):
    """Centres of the two terminal bands of the straightened worm, top then bottom."""
    ys, xs = np.nonzero(rm); y0, y1 = ys.min(), ys.max(); span = int(y1 - y0 + 1)
    out = []
    for yc in (y0 + int(0.12 * span), y1 - int(0.12 * span)):
        row = rm[yc]
        out.append((int(np.where(row)[0].mean()) if row.any() else rm.shape[1] // 2, int(yc)))
    return out, span


def vignette(rbgr, msk, cx, cy, span, frac):
    h = int(frac * span); H, W = msk.shape
    y0 = max(0, int(cy) - h); y1 = min(H, int(cy) + h)
    x0 = max(0, int(cx) - h); x1 = min(W, int(cx) + h)
    g3 = rbgr.copy(); g3[~msk] = 0; c = g3[y0:y1, x0:x1]
    if c.size == 0: return None
    ch, cw = c.shape[:2]; side = max(ch, cw); oy = (side - ch) // 2; ox = (side - cw) // 2
    sq = np.zeros((side, side, 3), np.uint8); sq[oy:oy + ch, ox:ox + cw] = c
    return cv2.resize(sq, (S, S)), (y0, x0, oy, ox, side)


def _pics(hm):
    t = torch.from_numpy(hm)[None, None]
    mx = F.max_pool2d(t, 2 * MINDIST + 1, 1, MINDIST)[0, 0].numpy()
    ys, xs = np.nonzero((hm == mx) & (hm >= PEAK_THR))
    return sorted(zip(xs.tolist(), ys.tolist(), hm[ys, xs].tolist()), key=lambda q: -q[2])[:2]


def _to_rframe(px, py, box):
    y0, x0, oy, ox, side = box; sc = side / float(S)
    return px * sc - ox + x0, py * sc - oy + y0


def _coupes(P, msk, gf, half):
    H, W = msk.shape; n = len(P)
    mu = np.full(n, np.nan); mn = np.full(n, np.nan)
    for i in range(n):
        j0, j1 = max(0, i - 2), min(n - 1, i + 2)
        ty, tx = P[j1] - P[j0]; nn = np.hypot(ty, tx) + 1e-9
        py, px = -tx / nn, ty / nn
        v = []
        for r in range(-half, half + 1):
            Y, X = int(round(P[i][0] + py * r)), int(round(P[i][1] + px * r))
            if 0 <= Y < H and 0 <= X < W and msk[Y, X] > 0: v.append(gf[Y, X])
        if v:
            v = np.array(v); mu[i] = v.mean(); mn[i] = v.min()
    return mu, mn


def _indices_DC(mu, mn, rev):
    if rev: mu, mn = mu[::-1], mn[::-1]
    core = slice(int(0.30 * N_AXE), int(0.70 * N_AXE))
    b = float(np.nanmedian(mu[core])); s = float(np.nanstd(mu[core])) + 1e-9
    a0, a1 = int(SEG_D[0] * N_AXE), int(SEG_D[1] * N_AXE)
    c0, c1 = int(SEG_C[0] * N_AXE), int(SEG_C[1] * N_AXE)
    D = float(np.nansum(np.maximum(0.0, b - mn[a0:a1]))) / (max(1, a1 - a0) * s)
    C = float(np.nansum(np.maximum(0.0, mu[c0:c1] - b))) / (max(1, c1 - c0) * s)
    return D, C


_NIV = np.arange(256.0)


def _histogrammes(P, msk, gf8, half):
    """Grey-level histograms read by the NB_COUPES sections of each end.

    A section is perpendicular to the median axis and bounded by the worm mask; no pixel outside is
    read. The sections of index 0 to NB_COUPES - 1 cover abscissa 0 to 0.30 from the end P[0], the
    last NB_COUPES cover the same portion from the end P[-1].
    """
    H, W = msk.shape; n = len(P)
    h0 = np.zeros(256, np.int64); h1 = np.zeros(256, np.int64)
    for i in range(n):
        if NB_COUPES <= i < n - NB_COUPES: continue
        j0, j1 = max(0, i - 2), min(n - 1, i + 2)
        ty, tx = P[j1] - P[j0]; nn = np.hypot(ty, tx) + 1e-9
        py, px = -tx / nn, ty / nn
        v = []
        for r in range(-half, half + 1):
            Y, X = int(round(P[i][0] + py * r)), int(round(P[i][1] + px * r))
            if 0 <= Y < H and 0 <= X < W and msk[Y, X] > 0: v.append(gf8[Y, X])
        if not v: continue
        hh = np.bincount(np.array(v, np.int64), minlength=256)
        if i < NB_COUPES: h0 += hh
        else: h1 += hh
    return h0, h1


def _indice_noirceur(h):
    """Darkness index of an end: 1 minus the median of the dark pixels over 255.

    Otsu's threshold separates the dark pixels from the pale ones on the grey-level distribution of
    the end; the median is taken over the dark class alone.
    """
    tot = h.sum()
    if tot < 10: return float("nan")
    p = h / float(tot); w0 = np.cumsum(p); w1 = 1.0 - w0
    m0 = np.cumsum(p * _NIV); mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (mt * w0 - m0) ** 2 / (w0 * w1)
    var[~np.isfinite(var)] = -1.0
    t = int(np.argmax(var))
    d = h[:t + 1]
    if d.sum() < 1: return float("nan")
    c = np.cumsum(d); med = int(np.searchsorted(c, d.sum() / 2.0))
    return 1.0 - med / 255.0


def noirceur(rbgr, rm):
    """Stage 3. Returns (index of the retained end in top-bottom order, gap between the two indices)."""
    pa = axe_median(rm.astype(np.uint8))
    if pa is None: return None, 0.0
    P, L = lisser(pa, N_AXE)
    if P is None or L < 20: return None, 0.0
    gf8 = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY)
    half = max(3, int(round(0.5 * np.sqrt(rm.sum()))))
    h0, h1 = _histogrammes(P, rm.astype(np.uint8), gf8, half)
    n0 = _indice_noirceur(h0); n1 = _indice_noirceur(h1)
    if not (np.isfinite(n0) and np.isfinite(n1)): return None, 0.0
    haut_est_P0 = P[0][0] <= P[-1][0]      # is P[0] the top end
    gagnant_est_P0 = n0 > n1
    i = 0 if (gagnant_est_P0 == haut_est_P0) else 1
    return i, abs(n0 - n1)


def ancrer(rbgr, rm, verbeux=False):
    """Complete cascade on an already straightened worm.

    Returns (x, y, stage, confidence, end), the point being the barycentre of the detector peaks when
    an eyespot is detected and the centre of the retained terminal band otherwise; end is 0 for the
    top end and 1 for the bottom one.
    """
    mdl_bl, mdl_ey = charger(verbeux)
    pts, span = bouts(rm)
    k = max(1, int(0.12 * span))
    rmz = cv2.dilate(rm.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    soft = []
    for cx, cy in pts:
        v = vignette(rbgr, rmz, cx, cy, span, F_BL)
        if v is None: soft.append(0.0); continue
        p = _pred(mdl_bl, v[0]); soft.append(float(p.sum()) * (v[1][4] / S) ** 2)
    marge = abs(soft[0] - soft[1]) / (soft[0] + soft[1] + 1e-9)
    eye = []
    for cx, cy in pts:
        v = vignette(rbgr, rm, cx, cy, span, F_EYE)
        if v is None: eye.append((0.0, 0.0, [], None)); continue
        p = _pred(mdl_ey, v[0])
        eye.append((float(p[p >= PEAK_THR].sum()), float(p.max()), _pics(p), v[1]))
    rmax = max(e[1] for e in eye)
    ecart = 0.0; j = None
    if rmax >= PEAK_ANCRAGE:
        i = 0 if eye[0][0] > eye[1][0] else 1; stage = "yeux"; conf = rmax
    elif marge >= MARGE_BL:
        i = 0 if soft[0] > soft[1] else 1; stage = "blastema"; conf = marge
    else:
        j, ecart = noirceur(rbgr, rm)
        if j is not None and ecart >= ECART_D:
            i = j; stage = "noirceur"; conf = ecart
        else:
            i = j if j is not None else (0 if soft[0] > soft[1] else 1)
            stage = "indetermine"; conf = max(marge, ecart)
    masse, r, pk, box = eye[i]
    if pk and box is not None:
        p2 = [_to_rframe(px, py, box) for px, py, _ in pk]
        x = float(np.mean([q[0] for q in p2])); y = float(np.mean([q[1] for q in p2]))
    else:
        x, y = float(pts[i][0]), float(pts[i][1])
    return x, y, stage, float(conf), int(i)
