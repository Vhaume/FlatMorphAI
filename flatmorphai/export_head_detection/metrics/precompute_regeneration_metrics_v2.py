#!/usr/bin/env python
# coding: utf-8

# # Precomputing the regeneration metrics (texture, pigment, shape, eyes) - one image = one row
# 
# Goal. For every image of every split (train, valid, test) of the State set, compute a fixed vector of dimensionless descriptors, then write one CSV per split in a NEW folder specific to regeneration, separate from the stereotype one. The state trainer reads those CSVs without ever recomputing the metrics.
# 
# Choices guided by sweeping the images. State_1 and State_2 are rounded dark blobs (no head). From State_3 on, the body lengthens and a pale regrowth zone appears at one extremity. In State_4 and above all State_5 and State_6, two sharp dark spots appear (the eyes), surrounded by a pale zone, the transparency then decreasing. Twists and postures do not separate the states; they are NOT used.
# 
# Intensity alone, no colour. One batch of images is brown, another paler or even yellow. These hue variations come from the lighting and are not relevant: only the intensity CONTRAST matters. The pigment metrics are therefore expressed as a relative deviation from the body median, and no colour or saturation metric is computed.
# 
# Transparent regrowth zone (blastema), group blast_. This is the strongest marker of how far head regeneration has progressed. The pale AND smooth (low texture) cap at one tip, paler than the median body, is segmented, requiring that it be localised (not the whole worm, which rejects a uniformly pale worm) and in contrast with the body. From it come the area, the contrast, the length, the width, the shape (aspect, solidity) and the position. On the 384 sample this group separates clearly: blast_present is 0.83 in State_5 and State_6 against 0.17 before. The eyes are then looked for INSIDE that cap (group eyeb_), where a dark spot stands out against a smooth pale background, which gives a far cleaner eye signal than over the whole textured body. Eye detection over the whole body is still provided (group eye_) but is unreliable, the internal pigment producing dark spots everywhere.
# 
# Eye detection (dark spots), at two levels. Morphological black-hat, which responds to small DARK spots on a PALER background, hence to LOCAL contrast and not to absolute brightness. The threshold is computed INSIDE the body tips, which makes the detection more sensitive to low-contrast eyes. Each candidate is validated by a halo (the centre must be darker than its ring, the pale zone connected to the eye amplifying the contrast) and by an absolute-relative darkness. The search covers BOTH tips, head/tail orientation not being reliable. Two complementary signals come out. A sensitive raw count, eye_count not capped at 2 (detects mutants with more or fewer than 2 eyes) and eye_present. A very specific PRESENCE signal, eye_pair_found, true only if two eyes form a symmetric pair at one and the same tip, which intestinal speckle does not do. The continuous contrasts eye_max_contrast and eye_halo_contrast provide graded evidence even when a pale eye does not cross the discrete threshold. Validation at 384: eye_pair_found is clean (no false pair on State_1, State_2, State_4 in the sample) and fires on the sharp pairs of State_5 and State_6; the continuous contrasts rise from State_4 on. The calibration cell below allows sensitivity to be traded against specificity.
# 
# Profile along the centerline (group cl_). The medial axis is extracted (Zhang-Suen skeleton), a BAR perpendicular to the axis is sampled at every point on the head side, and the intensity and texture profile is analysed. Since the eyes and the returning pigment are LATERAL to the axis, the cross section (minimum and amplitude of the bar, lateral dark pigment) reveals their return where a regional mean dilutes it. This is the best metric signal found for separating State_5 (smooth cap) from State_6 (structure and returning pigment): the centerline alone reaches 0.63 on that pair against 0.47 for the silhouette descriptors. The overall gain stays modest and is better measured on the full training set than on a small sample.
# 
# Local histogram equalisation (group clh_). A contrast-limited adaptive equalisation (CLAHE) is computed, which normalises the local contrast and makes the PIGMENT RETURN and the fine structure of the head visible, which the raw intensity drowns. On that CLAHE image the entropy and the texture, the repigmented dark fraction, the pale/dark bimodality, the edge density and a repigmentation index are measured on the head side. On the full set this group brings a real, non-redundant gain: the 6-class accuracy goes from 0.572 to 0.583, formed head from 0.867 to 0.889, State_5 against State_6 from 0.717 to 0.750, and clh_head_entropy becomes the most important variable of the model. Reference: Zuiderveld, Contrast Limited Adaptive Histogram Equalization, Graphics Gems IV 1994, ISBN 0-12-336155-9.
# 
# Sources. Hu, Visual pattern recognition by moment invariants, IRE Transactions on Information Theory 1962, DOI 10.1109/TIT.1962.1057692. Haralick, Shanmugam, Dinstein, Textural Features for Image Classification, IEEE Transactions on Systems Man and Cybernetics 1973, DOI 10.1109/TSMC.1973.4309314. Ojala, Pietikainen, Maenpaa, Multiresolution gray-scale and rotation invariant texture classification with local binary patterns, IEEE Transactions on Pattern Analysis and Machine Intelligence 2002, DOI 10.1109/TPAMI.2002.1017623. Soille, Morphological Image Analysis, Springer 2004, DOI 10.1007/978-3-662-05088-0. van der Walt et al., scikit-image, PeerJ 2014, DOI 10.7717/peerj.453. random_state 42, separator ';'.

# ## Metric module (one formatted colour image as input)
# 
# compute_metrics takes the colour image ALREADY formatted (384, masked, black background), converts it to grey levels (only the intensity matters) and returns the metric vector. The eye parameters are grouped in eye_params, adjustable through the calibration cell.


# In[ ]:


# Single-image regeneration metrics (texture / pigment / shape / eyes).
# Intensity alone (no colour): the brown/yellow variations come from the lighting
# and are not relevant, only the intensity contrast matters. Dimensionless values,
# normalised by the square root of the area, pigment profiles expressed as a relative deviation from the body median.
import numpy as np, cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments_hu
from skimage.morphology import skeletonize

# ---------- eye detection (dark spots): local contrast + pale halo ----------
def _halo_ok(gray, cy, cx, r, body_med, body_std, halo_k=0.3, dark_k=0.4):
    """True if the centre is DARKER than its ring (pale halo connected to the eye)
    AND dark enough in absolute-relative terms (an eye is a heavily pigmented point). The halo
    amplifies the contrast; dark_k rejects dark-on-dark speckles. dark_k=0
    disables the absolute-darkness criterion."""
    H, W = gray.shape
    r = max(1, int(round(r)))
    y0, y1 = max(0, cy - 2 * r), min(H, cy + 2 * r + 1)
    x0, x1 = max(0, cx - 2 * r), min(W, cx + 2 * r + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    center = patch[d <= r]; ring = patch[(d > r) & (d <= 2 * r)]
    if center.size < 1 or ring.size < 3:
        return False, 0.0
    contrast = float(ring.mean() - center.mean())
    ok = (contrast > halo_k * body_std) and (center.mean() < body_med - dark_k * body_std)
    return ok, contrast

def detect_eyes(gray, mask, c, perp, region_mask, body_med, body_std, area,
                k_frac=0.06, n_sigma=2.0, halo_k=0.2, dark_k=0.2, ret_debug=False):
    """Morphological black-hat for dark spots on a pale background
    (LOCAL contrast, robust to brightness), threshold computed INSIDE the target region
    (the tips) to catch low-contrast eyes, then validation by halo and
    darkness. Soille, Morphological Image Analysis, Springer 2004, DOI 10.1007/978-3-662-05088-0.
    Returns tuples (area, centroid, blackhat_contrast, halo_contrast)."""
    s = np.sqrt(max(area, 1.0))
    k = max(3, int(round(k_frac * s)));  k += (k % 2 == 0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
    region = region_mask > 0
    if region.sum() < 20:
        region = mask > 0
    inw = bh[region]                                   # LOCAL threshold for the region, more sensitive to pale eyes
    thr = max(6.0, float(inw.mean() + n_sigma * inw.std()))
    cand = ((bh > thr) & region).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(cand, 8)
    eyes = []; amax = max(6.0, area * 0.02)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if not (2 <= a <= amax):
            continue
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / (min(w, h) + 1e-9) > 2.6:
            continue
        cx, cy = int(round(cent[i][0])), int(round(cent[i][1]))
        r = max(1.0, np.sqrt(a / np.pi))
        ok, halo = _halo_ok(gray, cy, cx, r, body_med, body_std, halo_k, dark_k)
        if not ok:
            continue
        eyes.append((int(a), cent[i], float(bh[lab == i].mean()), halo))
    eyes.sort(key=lambda z: z[2], reverse=True)
    if ret_debug:
        return eyes, bh, thr
    return eyes

def _best_symmetric_pair(eyes, c, perp, s):
    """Looks for the best pair of symmetric eyes at one and the same tip: opposite sides
    of the axis, same axial level, plausible separation, comparable sizes. This
    PAIR constraint gives a very specific presence signal (intestinal
    speckle is not paired and symmetric)."""
    axis = np.array([-perp[1], perp[0]])
    best = None
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            pi = np.array(eyes[i][1], float); pj = np.array(eyes[j][1], float)
            wi = (pi - c) @ perp; wj = (pj - c) @ perp
            if wi * wj >= 0:                       # must sit on either side of the axis
                continue
            ti = (pi - c) @ axis; tj = (pj - c) @ axis
            if abs(ti - tj) > 0.18 * s:            # same axial level (same tip)
                continue
            sep = float(np.linalg.norm(pi - pj))
            if not (0.03 * s < sep < 0.35 * s):    # plausible separation between two eyes
                continue
            if min(eyes[i][0], eyes[j][0]) / max(eyes[i][0], eyes[j][0]) < 0.25:
                continue
            score = eyes[i][2] + eyes[j][2]
            if best is None or score > best[0]:
                best = (score, i, j, sep)
    return best

def eye_metrics(gray, mask, c, perp, region_mask, body_med, body_std, area,
                k_frac=0.06, n_sigma=2.0, halo_k=0.2, dark_k=0.2):
    eyes = detect_eyes(gray, mask, c, perp, region_mask, body_med, body_std, area,
                       k_frac=k_frac, n_sigma=n_sigma, halo_k=halo_k, dark_k=dark_k)
    s = np.sqrt(max(area, 1.0)); out = {}; cnt = len(eyes)
    out["eye_count"] = float(min(cnt, 8))              # raw candidates, NOT capped at 2: mutant indicator
    out["eye_present"] = 1.0 if cnt >= 1 else 0.0
    out["eye_excess"] = float(max(0, cnt - 2))         # beyond 2 = possible mutation
    out["eye_area_frac"] = float(sum(e[0] for e in eyes) / area)
    out["eye_max_contrast"] = float(eyes[0][2] / (body_std + 1e-6)) if eyes else 0.0
    out["eye_mean_contrast"] = float(np.mean([e[2] for e in eyes]) / (body_std + 1e-6)) if eyes else 0.0
    out["eye_halo_contrast"] = float(np.mean([e[3] for e in eyes]) / (body_std + 1e-6)) if eyes else 0.0
    pair = _best_symmetric_pair(eyes, c, perp, s) if cnt >= 2 else None
    out["eye_pair_found"] = 1.0 if pair else 0.0       # very specific presence signal (symmetric pair at one tip)
    if pair:
        _, i, j, sep = pair
        out["eye_pair_dist"] = float(sep / s)
        out["eye_pair_symmetry"] = 1.0
        out["eye_pair_contrast_bal"] = float(min(eyes[i][2], eyes[j][2]) / (max(eyes[i][2], eyes[j][2]) + 1e-6))
    else:
        out["eye_pair_dist"] = 0.0; out["eye_pair_symmetry"] = 0.0; out["eye_pair_contrast_bal"] = 0.0
    return out

# ---------- masque et axe ----------
def _largest_mask(gray):
    m = (gray > 8).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)

def _axis_coords(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]; perp = vt[1]
    t = (pts - c) @ axis; w = (pts - c) @ perp
    t = (t - t.min()) / (np.ptp(t) + 1e-9)
    return c, axis, perp, t, w, ys, xs

def compute_metrics(bgr, eye_params=None, head_high=None):
    ep = eye_params or {}
    out = {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = _largest_mask(gray)
    area = int(mask.sum())
    if area < 60:
        return None
    s = np.sqrt(area)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt); harea = cv2.contourArea(hull) + 1e-9
    x, y, bw, bh = cv2.boundingRect(cnt)
    # ---- shape shp_ ----
    out["shp_area_frac"] = area / float(mask.size)
    out["shp_solidity"] = area / harea
    out["shp_circularity"] = 4 * np.pi * area / (peri * peri + 1e-9)
    out["shp_extent"] = area / float(bw * bh + 1e-9)
    out["shp_aspect_bbox"] = max(bw, bh) / float(min(bw, bh) + 1e-9)
    out["shp_perim_norm"] = peri / s
    (ex, ey), (MA, ma), ang = cv2.fitEllipse(cnt) if len(cnt) >= 5 else ((0, 0), (1, 1), 0)
    out["shp_elong"] = max(MA, ma) / (min(MA, ma) + 1e-9)
    out["shp_eccentricity"] = np.sqrt(max(0.0, 1 - (min(MA, ma) / (max(MA, ma) + 1e-9)) ** 2))
    hu = moments_hu((mask * 255).astype(np.float64))
    for i in (0, 1, 2):
        out["shp_hu%d" % (i + 1)] = float(-np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30))
    # ---- axis, width + intensity profiles ----
    c, axis, perp, t, wperp, ys, xs = _axis_coords(mask)
    g_in = gray[ys, xs].astype(np.float32)
    body_med = float(np.median(g_in)); body_std = float(g_in.std() + 1e-6)
    NB = 12
    b = np.clip((t * NB).astype(int), 0, NB - 1)
    width_prof = np.array([np.ptp(wperp[b == k]) if (b == k).any() else 0.0 for k in range(NB)])
    int_prof = np.array([g_in[b == k].mean() if (b == k).any() else body_med for k in range(NB)])
    width_prof_n = width_prof / (s + 1e-9)
    end0 = int_prof[:3].mean(); end1 = int_prof[-3:].mean()
    # head/tail orientation: cascade U-Net anchor if provided, otherwise fall back on paleness
    head_hi_eff = (end0 <= end1) if head_high is None else bool(head_high)
    if not head_hi_eff:
        int_prof = int_prof[::-1]; width_prof_n = width_prof_n[::-1]
    head = slice(NB - 4, NB); tail = slice(0, 4); mid = slice(NB // 2 - 2, NB // 2 + 2)
    # ---- antero-posterior pigment pig_ (relative to the body median, robust to the batch) ----
    out["pig_head_excess"] = (int_prof[head].mean() - body_med) / body_std
    out["pig_tail_excess"] = (int_prof[tail].mean() - body_med) / body_std
    out["pig_head_tail_diff"] = (int_prof[head].mean() - int_prof[tail].mean()) / body_std
    out["pig_axial_range"] = (int_prof.max() - int_prof.min()) / body_std
    out["pig_light_loc"] = float(np.argmax(int_prof)) / (NB - 1)
    out["pig_light_frac"] = float((int_prof > body_med + 0.5 * body_std).mean())
    out["pig_profile_skew"] = float(((int_prof - int_prof.mean()) ** 3).mean() / (int_prof.std() ** 3 + 1e-9))
    out["pig_dark_frac"] = float((g_in < body_med - 0.8 * body_std).mean())
    # ---- width and tapering end_ ----
    out["end_head_width"] = float(width_prof_n[head].mean())
    out["end_tail_width"] = float(width_prof_n[tail].mean())
    out["end_mid_width"] = float(width_prof_n[mid].mean())
    out["end_taper_asym"] = float(width_prof_n[head].mean() - width_prof_n[tail].mean())
    out["end_head_pointed"] = float(width_prof_n[mid].mean() - width_prof_n[head].mean())
    # ---- texture: co-occurrence (Haralick) + local binary patterns (LBP) ----
    sub = gray[y:y+bh, x:x+bw]; submask = mask[y:y+bh, x:x+bw].astype(bool)
    q = (sub.astype(np.float32) / 256 * 32).astype(np.uint8); q[~submask] = 0
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi/2], levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        out["tex_%s" % prop] = float(graycoprops(glcm, prop).mean())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out["tex_lapvar"] = float(lap[ys, xs].var() / (body_std ** 2 + 1e-9))
    lbp = local_binary_pattern(sub, P=8, R=1, method="uniform")
    lv = lbp[submask]
    hist, _ = np.histogram(lv, bins=10, range=(0, 10), density=True)
    out["tex_lbp_entropy"] = float(-(hist * np.log2(hist + 1e-9)).sum())
    out["tex_lbp_uniform"] = float(hist[:2].sum())
    head_t = t if head_hi_eff else (1 - t)
    head_pix = g_in[head_t > 0.75]; body_pix = g_in[(head_t > 0.25) & (head_t < 0.75)]
    out["tex_head_std"] = float(head_pix.std() / body_std) if head_pix.size else 0.0
    out["tex_head_body_std_ratio"] = float((head_pix.std() + 1e-6) / (body_pix.std() + 1e-6)) if (head_pix.size and body_pix.size) else 1.0
    # ---- eyes (dark spots): local contrast + halo, in the cephalic tip ----
    tip_mask = np.zeros_like(mask)
    _tip = (t < 0.20) | (t > 0.80)      # the eyes are at one extremity: BOTH tips are searched, head/tail orientation not being reliable
    tip_mask[ys[_tip], xs[_tip]] = 1
    out.update(eye_metrics(gray, mask, c, perp, tip_mask, body_med, body_std, area,
                           k_frac=ep.get("k_frac", 0.06), n_sigma=ep.get("n_sigma", 2.0),
                           halo_k=ep.get("halo_k", 0.2), dark_k=ep.get("dark_k", 0.2)))
    # ---- transparent regrowth zone (blastema) + eyes INSIDE the cap ----
    bmet, _ = blastema_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std)
    out.update(bmet)
    # ---- unpigmented cap (Otsu) at the end anchored by the cascade: separates State_2 from State_3 ----
    out.update(unpig_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std, head_hi_eff))
    # ---- profile along the centerline (perpendicular bar): structure and returning pigment on the head side ----
    out.update(centerline_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std, head_hi_eff))
    # ---- metrics on the CLAHE-equalised image (repigmentation / head structure) ----
    out.update(clahe_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std, head_hi_eff))
    return out

def blastema_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std):
    """Segments the transparent regrowth zone (pale AND smooth, as a cap at one tip,
    paler than the median body) and extracts shape metrics from it. This is the
    marker of how far head regeneration has progressed. The eyes are then looked for
    INSIDE that cap, where a dark spot stands out clearly against a smooth pale background."""
    out = {k: 0.0 for k in (
        "blast_frac","blast_contrast","blast_len","blast_width","blast_aspect",
        "blast_loc","blast_solidity","blast_present",
        "eyeb_count","eyeb_present","eyeb_excess","eyeb_pair","eyeb_max_contrast")}
    s = np.sqrt(max(area, 1.0))
    gf = gray.astype(np.float32)
    gi = gf[ys, xs]
    core = gi[(t > 0.35) & (t < 0.65)]
    base = float(np.median(core)) if core.size else float(np.median(gi))
    win = max(3, int(round(0.05 * s))); win += (win % 2 == 0)
    mu = cv2.blur(gf, (win, win)); lstd = np.sqrt(np.maximum(cv2.blur(gf * gf, (win, win)) - mu * mu, 0))
    med_lstd = float(np.median(lstd[mask > 0]))
    bz = (((gf > base + 0.35 * body_std) & (lstd < 0.9 * med_lstd)) & (mask > 0)).astype(np.uint8)
    bz = cv2.morphologyEx(bz, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bz = cv2.morphologyEx(bz, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(bz, 8)
    tmap = np.full(gray.shape, -1.0); tmap[ys, xs] = t
    best = None
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 0.01 * area or a > 0.5 * area:
            continue
        tv = tmap[lab == i]; tv = tv[tv >= 0]
        if tv.size == 0:
            continue
        tc = float(tv.mean())
        if not (tc < 0.30 or tc > 0.70):          # must be a cap at one tip
            continue
        contrast = (gf[lab == i].mean() - base) / (body_std + 1e-6)
        if contrast < 0.4:
            continue
        if best is None or a > best[1]:
            best = (i, a, tc, contrast)
    if best is None:
        return out, np.zeros_like(mask)
    bmask = (lab == best[0]).astype(np.uint8)
    ptsy, ptsx = np.nonzero(bmask)
    p = np.stack([ptsx, ptsy], 1).astype(np.float64) - c
    tt = p @ axis; ww = p @ perp
    cnt, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc = max(cnt, key=cv2.contourArea); hull = cv2.convexHull(cc)
    out["blast_frac"] = best[1] / area
    out["blast_contrast"] = float(best[3])
    out["blast_len"] = float(np.ptp(tt) / s)
    out["blast_width"] = float(np.ptp(ww) / s)
    out["blast_aspect"] = float((np.ptp(tt) + 1e-6) / (np.ptp(ww) + 1e-6))
    out["blast_loc"] = float(best[2])                        # axial position of the cap (0 or 1 = tip)
    out["blast_solidity"] = float(best[1] / (cv2.contourArea(hull) + 1e-6))
    out["blast_present"] = 1.0
    # --- eyes INSIDE the cap (dark spot on a smooth pale background) ---
    k = max(3, int(round(0.05 * s))); k += (k % 2 == 0)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    inb = bh[bmask > 0]
    thr = max(6.0, float(inb.mean() + 2.0 * inb.std()))
    dark = ((bh > thr) & (bmask > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    nn, ll, ss, cci = cv2.connectedComponentsWithStats(dark, 8)
    amax = max(6.0, area * 0.02); spots = []
    for i in range(1, nn):
        a = ss[i, cv2.CC_STAT_AREA]
        if not (2 <= a <= amax):
            continue
        w, h = ss[i, cv2.CC_STAT_WIDTH], ss[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / (min(w, h) + 1e-9) > 2.8:
            continue
        spots.append((int(a), cci[i], float(bh[ll == i].mean())))
    spots.sort(key=lambda z: z[2], reverse=True)
    out["eyeb_count"] = float(min(len(spots), 8))
    out["eyeb_present"] = 1.0 if spots else 0.0
    out["eyeb_excess"] = float(max(0, len(spots) - 2))
    out["eyeb_max_contrast"] = float(spots[0][2] / (body_std + 1e-6)) if spots else 0.0
    if len(spots) >= 2:
        p0 = np.array(spots[0][1]) - c; p1 = np.array(spots[1][1]) - c
        out["eyeb_pair"] = 1.0 if (p0 @ perp) * (p1 @ perp) < 0 else 0.0
    return out, bmask

# ---- unpigmented cap (transparent blastema) by Otsu ----

def _otsu(vals):
    hist, _ = np.histogram(vals, bins=256, range=(0, 256)); tot = hist.sum()
    if tot == 0: return 127
    sa = float(np.dot(np.arange(256), hist)); wB = 0.0; sB = 0.0; best = -1.0; thr = 127
    for i in range(256):
        wB += hist[i]
        if wB == 0: continue
        wF = tot - wB
        if wF == 0: break
        sB += i * hist[i]; mB = sB / wB; mF = (sa - sB) / wF; v = wB * wF * (mB - mF) ** 2
        if v > best: best = v; thr = i
    return int(thr)

def unpig_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std, head_hi):
    """UNPIGMENTED cap at the top of the head, segmented by the Otsu threshold (Otsu 1979,
    DOI 10.1109/TSMC.1979.4310076). The transparent blastema grows from State_2 to State_3 where
    the paleness stays flat. Three measures: length and area of the unpigmented cap, and its
    smoothness (low gradient). The head end is the one given by the cascade anchor head_hi."""
    out = {k: 0.0 for k in ("blast_unpig_len", "blast_unpig_area", "blast_head_smooth")}
    gi = gray[ys, xs].astype(np.float32)
    if gi.size < 50: return out
    thr = _otsu(gi.astype(np.uint8))
    pig = gi <= thr                                   # pixel pigmente (sombre)
    th = t if head_hi else (1.0 - t)                  # head at th close to 1
    NB = 40; b = np.clip((th * NB).astype(int), 0, NB - 1)
    L = 0; gap = 0
    for k in range(NB - 1, -1, -1):                   # from the head towards the tail
        sel = b == k
        if sel.sum() < 3:
            gap += 1
        else:
            fpig = float(pig[sel].mean())
            if fpig < 0.5: L += 1; gap = 0
            else: gap += 1
        if gap >= 3: break
    out["blast_unpig_len"] = L / float(NB)            # length of the unpigmented cap / body length
    headband = th > (2.0 / 3.0)
    out["blast_unpig_area"] = float(((~pig) & headband).sum()) / float(max(area, 1))   # unpigmented area of the head third / body area
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gyy * gyy)[ys, xs]
    cap = (~pig) & headband
    out["blast_head_smooth"] = float(grad[cap].mean() / (body_std + 1e-6)) if cap.sum() > 10 else 0.0   # smooth cap = low gradient (normalised)
    return out

# ---- centerline metrics ----

def _extend_path(path, mask, k=9, step=1.0):
    """Extends the median line at both its extremities, along the mean tangent of its last k
    points, as long as the point stays inside the mask. The morphological skeleton retracts
    at the ends of a convex shape; this extension makes it reach the tip of the head, the palest
    and most informative zone of the profile."""
    if path is None or len(path) < k + 2: return path
    P = path.astype(np.float64); H, W = mask.shape; out = [P]
    for end in (0, 1):
        seg = P[:k] if end == 0 else P[-k:]
        v = (seg[0] - seg[-1]) if end == 0 else (seg[-1] - seg[0])
        v = v / (np.hypot(v[0], v[1]) + 1e-9)
        p = P[0].copy() if end == 0 else P[-1].copy(); ext = []
        for _ in range(int(0.6 * max(H, W))):
            p = p + v * step
            yy, xx = int(round(p[0])), int(round(p[1]))
            if not (0 <= yy < H and 0 <= xx < W) or mask[yy, xx] == 0: break
            ext.append(p.copy())
        if ext:
            E = np.array(ext)
            out = [E[::-1]] + out if end == 0 else out + [E]
    return np.vstack(out)


def _order_path(sk):
    pts=np.argwhere(sk)  # (y,x)
    if len(pts)<5: return None
    idx={tuple(p):i for i,p in enumerate(pts)}
    nb=[[] for _ in pts]
    for i,(y,x) in enumerate(pts):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                j=idx.get((y+dy,x+dx))
                if j is not None: nb[i].append(j)
    deg=np.array([len(n) for n in nb])
    ends=np.where(deg==1)[0]
    if len(ends)<2:
        ends=[int(np.argmax(pts[:,0])),int(np.argmin(pts[:,0]))]
    # the 2 farthest extremities
    P=pts[ends]; D=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2)
    a,b=np.unravel_index(np.argmax(D),D.shape); start,goal=ends[a],ends[b]
    # BFS shortest path
    from collections import deque
    prev={start:-1}; q=deque([start])
    while q:
        u=q.popleft()
        if u==goal: break
        for v in nb[u]:
            if v not in prev: prev[v]=u; q.append(v)
    if goal not in prev: return pts[[start,goal]]
    path=[]; u=goal
    while u!=-1: path.append(u); u=prev[u]
    return pts[path[::-1]]

def centerline_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std, head_hi=None):
    out={k:0.0 for k in ("cl_head_intensity","cl_head_std","cl_head_min_depth","cl_head_nmin",
        "cl_cross_min_head","cl_cross_range_head","cl_cross_range_body","cl_cross_range_ratio",
        "cl_head_texture","cl_dark_lateral_head")}
    s=np.sqrt(max(area,1.0)); gf=gray.astype(np.float32); g_in=gf[ys,xs]
    base=float(np.median(g_in[(t>0.30)&(t<0.70)])) if ((t>0.30)&(t<0.70)).any() else float(np.median(g_in))
    sk=skeletonize(mask>0)
    path=_order_path(sk)
    path=_extend_path(path,mask)      # the median line must reach both tips of the worm
    if path is None or len(path)<8: return out
    # resample a N points
    N=40; idx=np.linspace(0,len(path)-1,N).astype(int); P=path[idx].astype(float)  # (N,2) y,x
    # local tangents/perpendiculars
    prof=[]; cross_min=[]; cross_rng=[]; lat_dark=[]
    hw=max(2,int(round(0.10*s)))  # half-width of the bar
    for k in range(N):
        y0,x0=P[k]
        k2=min(N-1,k+1); k1=max(0,k-1)
        ty,tx=P[k2]-P[k1]; nn=np.hypot(ty,tx)+1e-9; py,px=-tx/nn,ty/nn  # perpendicular
        vals=[]
        for r in range(-hw,hw+1):
            yy=int(round(y0+py*r)); xx=int(round(x0+px*r))
            if 0<=yy<gray.shape[0] and 0<=xx<gray.shape[1] and mask[yy,xx]>0:
                vals.append(gf[yy,xx])
        if not vals: vals=[base]
        vals=np.array(vals)
        prof.append(float(gf[int(round(y0)),int(round(x0))]) if mask[int(round(y0)),int(round(x0))]>0 else vals.mean())
        cross_min.append(float(vals.min())); cross_rng.append(float(vals.max()-vals.min()))
        lat_dark.append(float((vals<base-0.6*body_std).mean()))
    prof=np.array(prof); cross_min=np.array(cross_min); cross_rng=np.array(cross_rng); lat_dark=np.array(lat_dark)
    # orient: head at the end given by the cascade anchor if provided, otherwise the palest extremity
    if head_hi is None:
        flip_cl = prof[:5].mean()>prof[-5:].mean()
    else:
        e0=(np.array([P[0,1],P[0,0]])-c)@axis; e1=(np.array([P[-1,1],P[-1,0]])-c)@axis
        flip_cl = (bool(head_hi) != (e1>e0))
    if flip_cl:
        prof=prof[::-1]; cross_min=cross_min[::-1]; cross_rng=cross_rng[::-1]; lat_dark=lat_dark[::-1]
    H=slice(int(0.7*N),N); B=slice(int(0.3*N),int(0.7*N))
    ph=prof[H]
    out["cl_head_intensity"]=float((ph.mean()-base)/(body_std+1e-6))
    out["cl_head_std"]=float(ph.std()/(body_std+1e-6))
    out["cl_head_min_depth"]=float((np.median(ph)-ph.min())/(body_std+1e-6))
    out["cl_head_nmin"]=float(sum((ph[i]<ph[i-1] and ph[i]<ph[i+1]) for i in range(1,len(ph)-1)))
    out["cl_cross_min_head"]=float((base-cross_min[H].mean())/(body_std+1e-6))     # dark transverse minimum = eyespots/pigment
    out["cl_cross_range_head"]=float(cross_rng[H].mean()/(body_std+1e-6))
    out["cl_cross_range_body"]=float(cross_rng[B].mean()/(body_std+1e-6))
    out["cl_cross_range_ratio"]=float((cross_rng[H].mean()+1e-6)/(cross_rng[B].mean()+1e-6))
    out["cl_head_texture"]=float(np.mean(np.abs(np.diff(ph)))/(body_std+1e-6))
    out["cl_dark_lateral_head"]=float(lat_dark[H].mean())                          # dark lateral pigment on the head side
    return out

# ---- CLAHE metrics ----

def clahe_metrics(gray, mask, c, axis, perp, t, ys, xs, area, body_std, head_hi=None):
    K=["clh_head_std","clh_head_entropy","clh_head_grad","clh_head_dark_frac","clh_head_bimod",
       "clh_head_edge","clh_repig_index","clh_axial_grad","clh_head_range","clh_body_head_tex_ratio"]
    out={k:0.0 for k in K}
    s=np.sqrt(max(area,1.0))
    cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray).astype(np.float32)
    ci=cl[ys,xs]; base=float(np.median(ci)); std=float(ci.std()+1e-6)
    # head = palest extremity (on the raw image, for the orientation)
    gi=gray[ys,xs].astype(np.float32)
    tipA=(t<0.25); tipB=(t>0.75)
    mA=gi[tipA].mean() if tipA.any() else 0; mB=gi[tipB].mean() if tipB.any() else 0
    head_hi_eff=(mB>=mA) if head_hi is None else bool(head_hi)
    ht=t if head_hi_eff else (1-t)
    head=ht>0.72; body=(ht>0.25)&(ht<0.72)
    if head.sum()<15: return out
    ch=ci[head]
    # --- texture / heterogeneity of the head on the CLAHE image (structure coming back) ---
    out["clh_head_std"]=float(ch.std()/std)
    hist,_=np.histogram(ch,bins=16,range=(0,256),density=True); p=hist[hist>0]
    out["clh_head_entropy"]=float(-(p*np.log2(p)).sum())
    gx=cv2.Sobel(cl,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(cl,cv2.CV_32F,0,1,ksize=3); gm=np.sqrt(gx*gx+gy*gy)
    out["clh_head_grad"]=float(gm[ys,xs][head].mean()/std)
    out["clh_head_range"]=float((np.percentile(ch,90)-np.percentile(ch,10))/std)
    # --- pigment return: dark pixels in the head (CLAHE reveals the pigment) ---
    out["clh_head_dark_frac"]=float((ch<base-0.5*std).mean())
    # bimodality (pale translucent + dark pigment) through Otsu's normalised between-class variance
    h2,e=np.histogram(ch,bins=32); hm=h2/(h2.sum()+1e-9); mids=(e[:-1]+e[1:])/2
    cum=np.cumsum(hm); mu=np.cumsum(hm*mids); muT=mu[-1]
    with np.errstate(divide='ignore',invalid='ignore'):
        sig=(muT*cum-mu)**2/(cum*(1-cum)+1e-9)
    out["clh_head_bimod"]=float(np.nanmax(sig)/(ch.var()+1e-6))
    # edge density in the head (structure: eyes, folds, pigment)
    ed=cv2.Canny(cl.astype(np.uint8),40,120); hmask=np.zeros_like(mask); hmask[ys[head],xs[head]]=1
    out["clh_head_edge"]=float((ed[hmask>0]>0).mean())
    # combined repigmentation index: dark + texture in the head
    out["clh_repig_index"]=float(out["clh_head_dark_frac"]*out["clh_head_std"])
    # CLAHE axial gradient, head vs body
    NB=12; b=np.clip((ht*NB).astype(int),0,NB-1)
    prof=np.array([ci[b==k].mean() if (b==k).any() else base for k in range(NB)])
    out["clh_axial_grad"]=float((prof[-3:].mean()-prof[3:9].mean())/std)
    # body/head texture ratio (the head re-textures in State_6)
    if body.sum()>15:
        out["clh_body_head_tex_ratio"]=float((ci[head].std()+1e-6)/(ci[body].std()+1e-6))
    return out


# In[ ]:


import os, glob, numpy as np, pandas as pd, cv2
np.random.seed(42)

CLASS_TYPE = "regeneration"
REGEN_ROOT = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_" + CLASS_TYPE)
LIST_ROOT = os.path.join(REGEN_ROOT, "dataset_images_list")
DATASET_NAME = os.environ.get("REGEN_DATASET", "dataset_dedup_State_50valid_30test")

IMAGE_SIZE = 384
COLOR_DIR = os.path.join(REGEN_ROOT, "processed_images_full_size_384_2pass_clahe_apply-mask")
# masked images rebuilt on the two-pass Mask2Former masks with CLAHE enhancement
# and on the validated labels, by rebuild_processed_images_full_size.ipynb

# NEW output folder, separate from the stereotype metrics
# base metrics RE-ANCHORED (cascade anchor) + unpigmented cap. Same path as the base
# so that the downstream chain (eye_inference_and_features then generate_composite_and_features) propagates
# the new columns through to _eyespig.csv, read by the cross-validation.
METRICS_ROOT = os.path.join(REGEN_ROOT, "metrics_regeneration", DATASET_NAME)
os.makedirs(METRICS_ROOT, exist_ok=True)

# eye detection parameters (tuned at 384; adjust through the calibration cell further down)
#   n_sigma: local contrast threshold (black-hat), higher = stricter
#   halo_k  : minimum centre-ring contrast, in body standard deviations
#   dark_k  : minimum absolute-relative darkness of the centre (set to 0 to disable)
EYE_PARAMS = {"k_frac": 0.06, "n_sigma": 2.2, "halo_k": 0.3, "dark_k": 0.4}

def split_csv(sp):
    return next((x for x in glob.glob(os.path.join(LIST_ROOT, DATASET_NAME, "*")) if x.endswith("_%s.csv" % sp)), None)
def out_csv(sp):
    return os.path.join(METRICS_ROOT, "%s_%s_regeneration_metrics.csv" % (DATASET_NAME, sp))

print("colour images:", os.path.abspath(COLOR_DIR), "| exists:", os.path.isdir(COLOR_DIR))
print("metrics output:", os.path.abspath(METRICS_ROOT))
print("eye_params :", EYE_PARAMS)

# head anchor from the cascade U-Net anchor, precomputed by head_anchor_pass.ipynb
HEAD_CSV = os.path.join(REGEN_ROOT, "head_anchors_regeneration.csv")
HEAD = {}
if os.path.exists(HEAD_CSV):
    _hdf = pd.read_csv(HEAD_CSV, sep=";")
    for _, _r in _hdf.iterrows():
        v = _r.get("head_high")
        if pd.notna(v) and str(v) != "":
            HEAD[str(_r["image"])] = bool(int(v))
    print("head anchors loaded:", len(HEAD), "->", os.path.basename(HEAD_CSV))
else:
    print("WARNING head anchor missing, falling back on paleness:", HEAD_CSV)


# In[ ]:


def process_split(sp, eye_params=None, max_rows=None, verbose=True):
    f = split_csv(sp)
    if f is None:
        print("split %s absent, saute" % sp); return None
    sdf = pd.read_csv(f, sep=";"); kc = sdf.columns[-2]
    rows = []; ok = miss = fail = 0
    for i, r in sdf.iterrows():
        if max_rows and i >= max_rows: break
        rel = str(r["image_path"]); base = os.path.basename(rel); stem = os.path.splitext(base)[0]
        ip = os.path.join(COLOR_DIR, os.path.dirname(rel), stem + ".jpg")
        rec = {"image": base, "class": str(r[kc])}
        if not os.path.exists(ip):
            miss += 1; rows.append(rec); continue
        img = cv2.imread(ip)
        hh = HEAD.get(base)
        m = compute_metrics(img, eye_params=eye_params or EYE_PARAMS, head_high=hh) if img is not None else None
        if m is None: fail += 1
        else: rec.update(m); ok += 1
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv(sp), sep=";", index=False)
    if verbose:
        print("  %-6s rows=%4d ok=%4d image-missing=%3d failed=%3d -> %s"
              % (sp, len(df), ok, miss, fail, os.path.basename(out_csv(sp))))
    return df

# quick test on 8 rows (set max_rows=None for the full run)
_ = process_split("train", max_rows=8)


# In[ ]:


# ============================ EYESPOT CALIBRATION (at 384) ============================
# Sweeps n_sigma and dark_k and shows, per State, the share of images where at least one eye is
# detected and the mean count. Pick the setting where State_1..3 stay close to 0 and
# where State_4 and above rise towards 2. Report the retained value in EYE_PARAMS above,
# then relaunch the full run. Sampled to stay fast. random_state 42.
def calibrate_eyes(split="train", per_class=60, grid=None):
    grid = grid or [(2.2, 0.4), (2.2, 0.5), (2.0, 0.4), (2.4, 0.4)]
    f = split_csv(split); sdf = pd.read_csv(f, sep=";"); kc = sdf.columns[-2]
    rng = np.random.RandomState(42)
    by = {}
    for st, g in sdf.groupby(kc):
        g = g.sample(min(per_class, len(g)), random_state=42)
        by[st] = [os.path.join(COLOR_DIR, str(p).rsplit("/", 1)[0] if "/" in str(p) else st,
                               os.path.splitext(os.path.basename(str(p)))[0] + ".jpg") for p in g["image_path"]]
    for ns, dk in grid:
        ep = {"k_frac": 0.06, "n_sigma": ns, "halo_k": 0.3, "dark_k": dk}
        print("--- n_sigma %.1f dark_k %.1f ---" % (ns, dk))
        for st in sorted(by):
            cc = []
            for ip in by[st]:
                if not os.path.exists(ip): continue
                m = compute_metrics(cv2.imread(ip), eye_params=ep)
                if m: cc.append(m["eye_count"])
            cc = np.array(cc) if cc else np.array([0.0])
            print("   %-8s n=%3d  pct>=1:%3.0f%%  pct==2:%3.0f%%  pct>2:%3.0f%%  moy=%.2f"
                  % (st, len(cc), 100*(cc>=1).mean(), 100*(cc==2).mean(), 100*(cc>2).mean(), cc.mean()))

# uncomment to calibrate once the 384 images have been generated:
# calibrate_eyes("train", per_class=60)


# In[ ]:


# ---- full run of the three splits with the retained EYE_PARAMS ----
dfs = {}
for sp in ("train", "valid", "test"):
    dfs[sp] = process_split(sp, eye_params=EYE_PARAMS, max_rows=None)

if dfs.get("train") is not None:
    d = dfs["train"].dropna(axis=0, how="any")
    keys = [c for c in ["shp_elong","pig_head_tail_diff","pig_axial_range","tex_contrast","tex_lapvar",
                        "eye_present","eye_count","eye_max_contrast","eye_halo_contrast"] if c in d.columns]
    print("\nMean per class (train):")
    print(d.groupby("class")[keys].mean().round(2).to_string())


# In[ ]:


# ---- optional QA: annotated plate of the detected eyes (for a visual check) ----
def eye_overlay_montage(split="test", per_class=4, out_png="eye_overlay_regeneration.png"):
    from PIL import Image, ImageDraw
    f = split_csv(split); sdf = pd.read_csv(f, sep=";"); kc = sdf.columns[-2]
    cell, pad, labh = 200, 4, 16
    classes = sorted(sdf[kc].unique())
    picks = {c: sdf[sdf[kc] == c].sample(min(per_class, (sdf[kc]==c).sum()), random_state=42) for c in classes}
    W = pad + per_class * (cell + pad); H = len(classes) * (cell + pad + labh)
    cv0 = Image.new("RGB", (W, H), (15, 15, 15)); D = ImageDraw.Draw(cv0)
    for ri, c in enumerate(classes):
        y0 = ri * (cell + pad + labh); D.text((6, y0 + 2), c, fill=(255, 255, 255))
        for ci, (_, r) in enumerate(picks[c].iterrows()):
            rel = str(r["image_path"]); stem = os.path.splitext(os.path.basename(rel))[0]
            ip = os.path.join(COLOR_DIR, os.path.dirname(rel), stem + ".jpg")
            if not os.path.exists(ip): continue
            bgr = cv2.imread(ip); g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = _largest_mask(g); cc, ax, perp, t, wp, ys, xs = _axis_coords(mask)
            gi = g[ys, xs].astype(np.float32); bm = float(np.median(gi)); bs = float(gi.std() + 1e-6)
            reg = np.zeros_like(mask); sel = (t < 0.18) | (t > 0.82); reg[ys[sel], xs[sel]] = 1
            eyes = detect_eyes(g, mask, cc, perp, reg, bm, bs, int(mask.sum()), **{k: EYE_PARAMS[k] for k in ("k_frac","n_sigma","halo_k","dark_k")})
            vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
            for (a, cen, resp, halo) in eyes:
                cv2.circle(vis, (int(cen[0]), int(cen[1])), 9, (0, 255, 0), 2)
            im = Image.fromarray(vis).resize((cell, cell)); x = pad + ci * (cell + pad)
            cv0.paste(im, (x, y0 + labh)); D.text((x + 2, y0 + labh + 2), "n=%d" % len(eyes), fill=(0, 255, 120))
    cv0.save(out_png); print("montage:", out_png)

# eye_overlay_montage("test", per_class=4)

