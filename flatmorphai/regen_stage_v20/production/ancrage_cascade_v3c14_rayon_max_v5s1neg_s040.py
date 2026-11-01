# CHAIN v21 (22 October 2026): copy of ancrage_cascade_v3c13_rayon_blasteme_v5s1neg_s040.py (unmodified). Only the function
# _ancre changes, to follow the rule requested by the experimenter: among the points of the tip-side half of the axis whose circle of
# radius r(i) (border radius, cv2.distanceTransform) is in contact with the blastema zone, the retained point is the one whose
# RADIUS IS THE LARGEST (numpy.argmax of r(i)), and no longer the one furthest from the blastema centre (v3c12, v3c13). On a tie,
# the first one met from the tip. Fallback unchanged if no point is in contact. Blastema threshold 0.40 unchanged.
# CHAIN v20 (20 October 2026): copy of ancrage_cascade_v3c12_rayon_blasteme_v5s1neg_s070.py (unmodified). Blastema threshold 0.40 (calibrer_seuil_objet_blasteme_20261020_023318,
# maximal IoU of the blastema object) and cascade ancrage_cascade_v3c13_rayon_blasteme_v5s1neg_s040 (v3c12 at threshold 0.40).
# Version v3c12 (19 October 2026), the experimenter's rule for the start point of the tip correction: copy of
# ancrage_cascade_v3c8_aragad_v5s1neg_s070 (unmodified). The ARAGAD name is no longer used. Only the choice of the axis point
# from which the correction ray starts changes (function _ancre). Selection of the point whose border radius is in
# contact with the blastema zone and which is furthest among the points in contact: over the tip-side half of the axis,
# r(i) = distance from P(i) to the worm border (cv2.distanceTransform of the mask, DIST_L2, mask 5, Borgefors 1986,
# DOI 10.1016/S0734-189X(86)80047-0, radius of the inscribed disc of the median axis, Blum 1967). P(i) is in contact if its
# distance to the blastema object (same transform over the complement of the object) is at most r(i). The retained point is the
# point in contact furthest from the blastema centre (numpy.argmax). No threshold. If no point is in contact,
# fallback without threshold: the point where r(i) minus the distance to the object is the largest (the closest to contact), counted in
# _TRACE["n_fallback"]. Corrected tip, thumbnail centre, eyespots, darkness and rules unchanged.
# Version v3c8 (19 October 2026), blastema selection rule retained by the experimenter: copy of ancrage_cascade_v3c7_aragad_v5s1neg_s070
# (unmodified). Only the choice of the blastema object inside each end thumbnail (stage 1) changes: instead of the largest
# object (p >= 0.70), the ADMITTED object closest to the tip is kept. An object is admitted if it contains the tip (extent of
# the object at p >= 0.50, the segmenter's decision threshold) or if at least one of its pixels is CONTACT_PX = 10 pixels or less
# from the worm contour (cv2.distanceTransform of the mask). Objects internal to the worm, far from the contour and without the tip (halo,
# translucent area), are rejected. The function objet_blasteme is also used by enrich_metrics_v19 (blastema descriptors).
# Version v3c7 ARAGAD (18 October 2026), final correction condition retained by the experimenter: copy of
# ancrage_cascade_v3c6_aragad_v5s1neg_s070 (unmodified). Only the tip correction condition changes: the tip of
# the end closest to the blastema centre is corrected by ARAGAD if it is not inside the object of that blastema (the tip pixel
# read in the labels of cv2.connectedComponentsWithStats, without dilation), instead of the distance test
# blastema centre - tip > 0.15 L. ARAGAD itself (theta peak before the break) is unchanged.
# Version v3c6 ARAGAD (18 October 2026), final method retained by the experimenter: copy of ancrage_cascade_v3c5_aragad_v5s1neg_s070
# (unmodified). Only the angle and the stopping rule of the ARAGAD walk change. On the smoothed 200-point axis, the walk goes
# from the tip towards the middle of the worm. At each current point P[i]: theta(i), the angle at vertex P[i] between the direction of
# the rest of the axis (P[i+1] - P[i-1], estimated on the two neighbours) and the vector P[i] -> blastema centre, computed by
# numpy.arccos, and d(i), the distance from P[i] to the blastema centre. Rule: no stop while d decreases (the search starts
# at the minimum of d), then stop at the peak of theta before it breaks, that is at the point of maximal theta, close to 180 degrees
# when the axis points at the blastema centre. Bound: half of the axis. Corrected tip unchanged (ray from the stop point ->
# blastema centre, last worm pixel).
# Variant v3c5 ARAGAD (17 October 2026): copy of ancrage_cascade_v3c4_araga_v5s1neg_s070 (unmodified). Anchor of the
# correction ray by the experimenter's ARAGAD method (adjustment by break in the improvement of the angular and distance
# gradient). On the smoothed 200-point axis, the walk goes from the tip towards the middle of the worm. At each point P[i]:
# theta(i), the angle between the axis tangent (P[i-1] - P[i+1], oriented towards the tip) and the vector P[i] -> blastema
# centre, and d(i), the distance from P[i] to the blastema centre. The walk continues while theta OR d still decreases, and
# stops at the first point where neither decreases: that point is the anchor. Bound: half of the axis.
# Corrected tip and blastema centre unchanged (ray anchor -> blastema centre, last worm pixel).
# Variant v3c3 (17 October 2026): copy of ancrage_cascade_v3c2_v5s1neg_s070 (unmodified), blastema threshold 0.70.
# Only change: the tip correction by the blastema (stage 2). Before, the ray started from the innermost point
# of the WHOLE worm (global maximum of the distance transform), far from the head on a long or curved worm. Here
# (at the experimenter's request): the axis points of the terminal portion closest to the blastema are scanned, from 0 to
# FEN_AXE = 0.30 L from the tip (the same portion as the darkness sections), and the one furthest from the borders is kept
# (largest value of cv2.distanceTransform, DIST_L2, mask 5), as the local anchor of the ray. The blastema centre
# is unchanged: the centroid (mean of the pixel coordinates) of its connected component in the image, given by
# cv2.connectedComponentsWithStats (8-connectivity) on the back-projected map binarised at threshold SEUIL_BL. The corrected tip
# is the last worm pixel (blastema included) along the ray anchor -> blastema centre (_rayon_bord, 1 px step).
# For a stocky worm (principal line), the scanned axis is the segment between the two tips.
# Variant of ancrage_cascade_v3c2_v5s1neg (unmodified): the stage 1 blastema is binarised at threshold SEUIL_BL = 0.40,
# that of the refined masks, instead of 0.50. Everything else (segmenter, band, eyespots, rules, thresholds) is identical.
# Copy of ancrage_cascade_v3c2.py, identical except for the blastema segmenter: v5_s1neg (v5 recipe + annotated
# State_1 negatives) instead of v5. Same v3c2 rule. ancrage_cascade_v3c2.py unmodified.
# -*- coding: utf-8 -*-
"""Head anchoring cascade v3c2 = cascade v3c with a new TIE-BREAK AT THE EYESPOT STAGE (the experimenter's rule,
    13 October 2026). Copy of ancrage_cascade_v3c.py (unmodified). Only change, at stage 3:
      before (v3c): the candidate with the greatest "mass" (sum of the probabilities >= 0.61 in the thumbnail) was the head;
      a wide, weak response (pigmentation of a tail blastema) could beat two sharp eyespots
      (2023-02-16-T12-J10-R1-PP-Q2: peaks 0.96 and 0.95 at the head, 0.88 and 0.80 at the tail, mass 131 against 137).
      v3c2: at each candidate ALL the eyespots are counted (local maxima, minimum distance AC.MINDIST, without a cap,
      supernumerary eyespots included) at the cascade decision threshold SEUIL_YEUX_TETE = PEAK_ANCRAGE = 0.85, a threshold chosen to
      limit false positives (the counting threshold 0.61, balanced, and the 2 best peaks remain those of the
      state classification). The candidate with the most eyespots is the head; on an equal count, the mean score of the
      counted eyespots breaks the tie. The eyespot stage still only decides if a peak reaches 0.85; the rule "the blastema is
      authoritative" is kept. The returned head point is still the barycentre of the 2 best eyespots (>= 0.61) of the retained candidate.

    Description of v3c (unchanged): cascade v3b + LARGEST OBJECT PER THUMBNAIL for the blastema (the experimenter's rule,
    12 October 2026). Copy of ancrage_cascade_v3b.py (unmodified): at stage 1, on each end thumbnail, only the
    largest connected component of the segmenter prediction (probability >= 0.5, 8-connectivity) is kept before the
    return into the image; the small blobs (reflections, pieces of wall, the other end of a curved worm) are ignored. One end
    carries at most one blastema. Everything else is identical to v3b.

    Description of v3b (unchanged): cascade v3 with the v5 blastema segmenter (band 0.30 L). Copy of
    ancrage_cascade_v3.py (unmodified) of 11 October 2026: only stage 1 changes (checkpoint
    blasteme_unet_384_v5_s1neg, trained with a band of 0.30 L around the worm, retained by the pre-registered rule of
    train_blasteme_segmenter_384_v5_dil030_20261011_190119) and the band of the blastema thumbnails, read from the model
    (key "dil"), instead of cadrage_squelette.DIL = 0.12. Positioning, eyespots, thresholds and rules unchanged.

    Description of the v3 cascade (unchanged):

    Working copy; ancrage_cascade.py is not modified (it supplies the network loading, the peak detection
    and the darkness index, reused as they are). Rule defined with the experimenter on 10 October 2026:
    the skeleton does not find the head in 100 % of cases (stocky worms), but the blastema shows where it
    is. A worm may carry two blastemas (one at each end); only one of them carries the eyespots.

        Stage 0, framing. Worm axis by cadrage_squelette (longest path of the pruned skeleton, or principal
        line for a stocky worm), two ends, thumbnails of fraction 0.45 taken in the original image
        without prior straightening.
        Stage 1, blastema. Segmentation of both thumbnails (blasteme_unet_384_v3, trained on this framing),
        predictions brought back into the image and merged. The components of at least A_MIN thumbnail pixels are
        blastemas (two at most, the largest ones).
        Stage 2, head correction by the blastema. For each blastema, the candidate end is the
        nearest skeleton tip, unless the blastema centre is further than D_CORR times the
        length away: the tip is then moved onto the worm border, where the ray starting from the
        innermost point of the worm and passing through the blastema centre leaves the worm (blastema included), and the
        thumbnail is re-framed with the same proportions. The opposite skeleton end stays a candidate
        when a single blastema is seen (the head may already be pigmented at the other end).
        Stage 3, eyespots. Eyespot detector fondu_v5_jeu3E, the only detector of the chain, on each candidate (thumbnail of
        fraction 0.30, as in the cascade in service, counting threshold 0.61). If a maximal response reaches PEAK_ANCRAGE, the candidate with the greatest peak mass is
        the head (a single blastema carries the eyespots). With YEUX_SUR_BLASTEME, only the candidates carrying a blastema are
        separated by the eyespots when at least one blastema is seen.
        Stage 4, without eyespots. Blastema: relative margin of the candidates' blastema areas at least MARGE_BL,
        the largest one is the head. Otherwise darkness (Otsu index over the 60 sections from 0 to 30 % of each
        end, along the skeleton axis), the darkest end.

    The thresholds are those of the cascade in service (PEAK_ANCRAGE 0.85, MARGE_BL 0.84), PEAK_THR 0.61 of the v5 detector, plus
    D_CORR = 0.15 and A_MIN = 30, fixed before the test_ancrage_cascade_v3 trial.
    References. See ancrage_cascade.py and cadrage_squelette.py.
"""
import os, numpy as np, cv2, torch
import torch.nn.functional as _F
import ancrage_cascade as AC
import cadrage_squelette as CS

CK_BL3 = os.path.join("checkpoints_regeneration", "blasteme_unet_384_v5_s1neg", "blasteme_unet_384_best.pth")
DIL_BL = None   # read from the model at load time (0.30)
# Single eyespot detector of the chain: fondu_v5_jeu3E (adaptive Poisson blending v5, retained on
# 9 October 2026), counting threshold 0.61 chosen on the real validation. Decision threshold of the eyespot
# stage PEAK_ANCRAGE = 0.85, the value of the cascade in service, to be re-checked for this detector on the
# responses recorded by test_ancrage_cascade_v3.
CK_EY3 = os.path.join("checkpoints_regeneration", "eye_unet_cmp_fondu_poisson_v5_jeu3E", "eye_unet_unified_best.pth")
PEAK_THR = 0.61; PEAK_ANCRAGE = AC.PEAK_ANCRAGE; MARGE_BL = AC.MARGE_BL
F_EYE = AC.F_EYE; D_CORR = 0.15; A_MIN = 30
# Rule set by the experimenter (10 October 2026, after test_ancrage_cascade_v3_20261010_210935): when at
# least one blastema is seen, the eyespot stage only chooses between the ends that carry a blastema (a single
# blastema carries the eyespots, and pigmentation at the other end must not win). With no blastema seen,
# every candidate stays in play. Rule laid down after examining the errors, to be confirmed on new worms.
YEUX_SUR_BLASTEME = True
SEUIL_BL = 0.40
FEN_AXE = 0.30   # (unused by ARAGA) terminal portion of the axis scanned for the local anchor (fraction of L)   # blastema binarisation threshold in the cascade (0.50 in the version in service)
GARDER_PLUS_GROS = True   # rule of 12 October 2026: a single object (the largest) per thumbnail for the blastema
SEUIL_YEUX_TETE = PEAK_ANCRAGE   # rule of 13 October 2026: eyespots counted at the threshold that limits false positives (0.85)
_MDL = {}
_TRACE = {}   # v3c12: quantities of the last selection (r, distance to the object, distance to the centre, retained point) and number of fallbacks


def charger(verbeux=True):
    """Networks of the v3 cascade (blastema v3, eyespots). Paths CK_BL3 and CK_EY3 may be changed beforehand."""
    cle = (CK_BL3, CK_EY3)
    if _MDL.get("cle") != cle:
        global DIL_BL
        mb, cb = AC._charge(CK_BL3); me, ce = AC._charge(CK_EY3)
        DIL_BL = float(cb["dil"]); assert abs(DIL_BL - 0.30) < 1e-9, "unexpected blastema segmenter (band %s)" % cb.get("dil")
        _MDL.update(cle=cle, bl=mb, ey=me)
        if verbeux:
            print("cascade v3c12, start point taken at the border radius in contact with the blastema (blastema band %.2f L, admitted object closest to the tip per thumbnail, eyespots ranked by their count at threshold %.2f then by their score), device:" % (DIL_BL, SEUIL_YEUX_TETE), AC.DEVICE, "| blastema:", os.path.basename(os.path.dirname(CK_BL3)),
                  "| eyespots:", os.path.basename(os.path.dirname(CK_EY3)), "peak threshold", PEAK_THR, "decision", PEAK_ANCRAGE)
    return _MDL["bl"], _MDL["ey"]


CONTACT_PX = 10   # v3c8: maximum distance, in image pixels, between a pixel of the object and the worm contour


def objet_blasteme(p, M, wm, tip):
    """v3c8. Probability map p (thumbnail of transform M) reduced to the admitted blastema object closest to the
    tip (y, x), zero elsewhere. Objects of at least A_MIN thumbnail pixels. Admitted: the object at p >= 0.50 contains the tip, or a pixel of the object (p >= SEUIL_BL) is
    CONTACT_PX pixels or less from the contour of the mask wm. No admitted object: null map."""
    H, W = wm.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((p >= SEUIL_BL).astype(np.uint8), 8)
    if n <= 1: return np.zeros_like(p)
    n5, lab5 = cv2.connectedComponents((p >= 0.5).astype(np.uint8), 8)
    d_bord = cv2.distanceTransform((wm > 0).astype(np.uint8), cv2.DIST_L2, 5)
    ty = int(min(max(round(float(tip[0])), 0), H - 1)); tx = int(min(max(round(float(tip[1])), 0), W - 1))
    best, dbest = 0, np.inf
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < A_MIN: continue   # same minimum area as stage 1 (A_MIN thumbnail pixels)
        ob = CS.retroprojeter((lab == k).astype(np.float32), M, (H, W)) >= 0.5
        if not ob.any(): continue
        e5 = np.unique(lab5[lab == k]); e5 = e5[e5 > 0]
        ob5 = CS.retroprojeter(np.isin(lab5, e5).astype(np.float32), M, (H, W)) >= 0.5
        if not (ob5[ty, tx] or float(d_bord[ob].min()) <= CONTACT_PX): continue
        ys, xs = np.nonzero(ob); dk = float(np.min(np.hypot(ys - ty, xs - tx)))
        if dk < dbest: best, dbest = k, dk
    return np.where(lab == best, p, 0.0).astype(p.dtype) if best else np.zeros_like(p)


def _plus_gros(p):
    """Probability map reduced to the largest connected component of p >= 0.5 (zero elsewhere)."""
    b = (p >= SEUIL_BL).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if n <= 2: return p
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == k, p, 0.0).astype(p.dtype)


def _pics_tous(hm, seuil):
    """Every local maximum >= threshold (same neighbourhood as AC._pics, without the cap at the 2 best)."""
    tt = torch.from_numpy(np.ascontiguousarray(hm, dtype=np.float32))[None, None]
    mx = _F.max_pool2d(tt, 2 * AC.MINDIST + 1, 1, AC.MINDIST)[0, 0].numpy()
    ys, xs = np.nonzero((hm == mx) & (hm >= seuil))
    return sorted(zip(xs.tolist(), ys.tolist(), hm[ys, xs].tolist()), key=lambda q: -q[2])


def _cle_yeux(c):
    """Tie-break of the eyespot stage: number of eyespots >= SEUIL_YEUX_TETE, then the mean of their scores."""
    return (c["n_yeux_tete"], c["score_yeux_tete"])


def _vers_image(M, x, y):
    Mi = cv2.invertAffineTransform(M); return Mi @ np.array([x, y, 1.0])


def _rayon_bord(dedans, c, b):
    """Exit point of the mask 'dedans' along the ray starting at c (y, x) and passing through b (y, x)."""
    d = np.array(b, float) - np.array(c, float); n = np.hypot(*d)
    if n < 1e-6: return np.array(b, float), np.array([-1.0, 0.0])
    d /= n; H, W = dedans.shape; p = np.array(c, float); dern = p.copy()
    for _ in range(int(2 * max(H, W))):
        p = p + d; y, x = int(round(p[0])), int(round(p[1]))
        if not (0 <= y < H and 0 <= x < W): break
        if dedans[y, x]: dern = p.copy()
        elif np.hypot(*(p - np.array(b, float))) > 0 and np.dot(p - np.array(b, float), d) > 0: break
    return dern, d


def ancrer(bgr, mraw, verbeux=False):
    """v3 cascade on the original image (BGR) and the raw Mask2Former mask. Returns a dict: x, y (head point in
    the image, barycentre of the eyespots when seen, otherwise the retained tip), stage, confidence, tip and
    centre (y, x) of the head thumbnail, L, and the detail of the candidates. None if the mask is unusable."""
    mdl_bl, mdl_ey = charger(verbeux)
    wm = CS.masque_ver(mraw)
    if wm.sum() < 300: return None
    P, L, nature = CS.squelette(wm)
    ends, L, meth = CS.extremites(wm)
    k_ = max(1, int(DIL_BL * L)); wd = cv2.dilate(wm.astype(np.uint8), np.ones((k_, k_), np.uint8)); H, W = wm.shape   # bande du segmenteur v5
    # stage 1, blastema at both skeleton ends, brought back into the image
    Bimg = np.zeros((H, W), np.float32)
    ends_blasteme = []   # per-end blastema thumbnail + back-projected area, for the editor's optional outputs
    for tip, centre in ends:
        M = CS.affine(tip, centre, L); v = CS.vignette(bgr, wd, M); pv = AC._pred(mdl_bl, v)
        pv = objet_blasteme(pv, M, wm, tip)   # v3c8: admitted object closest to the tip (replaces _plus_gros)
        bp = CS.retroprojeter(pv, M, (H, W))
        Bimg = np.maximum(Bimg, bp)
        ends_blasteme.append(dict(thumbnail=v, aire=int((bp >= SEUIL_BL).sum())))
    ech = (2 * CS.FRAC * L / CS.S) ** 2
    n, lab, st, cen = cv2.connectedComponentsWithStats((Bimg >= SEUIL_BL).astype(np.uint8), 8)
    blast = sorted([(float(st[k, cv2.CC_STAT_AREA]), (float(cen[k][1]), float(cen[k][0])), k) for k in range(1, n)
                    if st[k, cv2.CC_STAT_AREA] >= A_MIN * ech], reverse=True)[:2]
    # stage 2, candidates: one per blastema (tip corrected if needed), plus the opposite end when only one
    dt = cv2.distanceTransform(wm, cv2.DIST_L2, 5)
    Pax = P if P is not None else np.linspace(np.array(ends[0][0], float), np.array(ends[1][0], float), CS.N_AXE)
    if np.hypot(*(np.array(Pax[0]) - np.array(ends[0][0], float))) > np.hypot(*(np.array(Pax[-1]) - np.array(ends[0][0], float))): Pax = Pax[::-1]
    n_fen = max(2, int(round(FEN_AXE * (len(Pax) - 1))))
    def _ancre(j, cb, kb):
        """v3c14. Axis point whose circle of border radius is in contact with the blastema zone (object kb of
        lab) and whose radius is the largest among the points in contact. Half of the axis on side j. No threshold."""
        Q = Pax if j == 0 else Pax[::-1]; nmax = (len(Q) - 1) // 2; cbv = np.array(cb, float)
        dob = cv2.distanceTransform((lab != kb).astype(np.uint8), cv2.DIST_L2, 5)
        I = np.arange(0, nmax + 1); yy = np.clip(np.round(Q[I, 0]).astype(int), 0, H - 1); xx = np.clip(np.round(Q[I, 1]).astype(int), 0, W - 1)
        r = dt[yy, xx]; g = dob[yy, xx]; dB = np.hypot(Q[I, 0] - cbv[0], Q[I, 1] - cbv[1]); ok = g <= r
        if ok.any(): k = int(I[ok][np.argmax(r[ok])])   # v3c14: largest radius among the points in contact
        else: k = int(np.argmax(r - g)); _TRACE["n_fallback"] = _TRACE.get("n_fallback", 0) + 1
        _TRACE.update(r=r, g=g, dB=dB, k=k, Q=Q, cb=cbv)
        return (float(Q[k, 0]), float(Q[k, 1]))
    dedans = (wm > 0) | (lab > 0)
    cand = []
    for aire, cb, k in blast:
        j = int(np.argmin([np.hypot(cb[0] - e[0][0], cb[1] - e[0][1]) for e in ends]))
        dist = float(np.hypot(cb[0] - ends[j][0][0], cb[1] - ends[j][0][1]))
        _py = int(min(max(round(float(ends[j][0][0])), 0), lab.shape[0] - 1)); _px = int(min(max(round(float(ends[j][0][1])), 0), lab.shape[1] - 1))
        if lab[_py, _px] == k:   # v3c7: tip already inside the blastema, no correction
            cand.append(dict(tip=ends[j][0], centre=ends[j][1], source="blastema", corrected=0, end=j, blastema_area=aire, dist_corr=dist / L))
        else:
            c0 = _ancre(j, cb, k); tip, u = _rayon_bord(dedans, c0, cb); centre = tip - CS.POS_CENTRE * L * u
            cand.append(dict(tip=tip, centre=centre, source="blastema", corrected=1, end=j, blastema_area=aire, dist_corr=dist / L, anchor=c0, blastema_centre=cb))
    if len(cand) == 0:
        cand = [dict(tip=e[0], centre=e[1], source="skeleton", corrected=0, end=j, blastema_area=0.0, dist_corr=np.nan) for j, e in enumerate(ends)]
    elif len(cand) == 1:
        jo = 1 - cand[0]["end"]; cand.append(dict(tip=ends[jo][0], centre=ends[jo][1], source="opposee", corrected=0, end=jo, blastema_area=0.0, dist_corr=np.nan))
    elif cand[0]["end"] == cand[1]["end"] and not (cand[0]["corrected"] or cand[1]["corrected"]):
        cand = cand[:1]; jo = 1 - cand[0]["end"]
        cand.append(dict(tip=ends[jo][0], centre=ends[jo][1], source="opposee", corrected=0, end=jo, blastema_area=0.0, dist_corr=np.nan))
    # stage 3, eyespots on each candidate (thumbnail of fraction 0.30, worm mask not dilated)
    for c in cand:
        M = CS.affine(c["tip"], c["centre"], L, frac=F_EYE); v = CS.vignette(bgr, wm, M); p = AC._pred(mdl_ey, v)
        c["eye_thumbnail"] = v; c["eye_heatmap"] = p; c["eye_M"] = M   # kept for the editor outputs AND the eyespots counter (count at any threshold, map peaks to the image)
        c["masse_yeux"] = float(p[p >= PEAK_THR].sum()); c["rmax"] = float(p.max())
        pk = AC._pics(p); c["yeux"] = [tuple(_vers_image(M, px, py)) for px, py, _ in pk]
        pt = _pics_tous(p, SEUIL_YEUX_TETE); c["n_yeux_tete"] = len(pt)
        c["score_yeux_tete"] = float(np.mean([s for _, _, s in pt])) if pt else 0.0
    rmax = max(c["rmax"] for c in cand)
    # quantities of every stage, computed for each worm (diagnostics, alternative decisions offline)
    a = sorted([c["blastema_area"] for c in cand], reverse=True)
    marge = (a[0] - a[1]) / (a[0] + a[1] + 1e-9) if len(a) > 1 else 1.0
    n0 = n1 = float("nan")
    if P is not None:
        gf8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); half = max(3, int(round(0.5 * np.sqrt(wm.sum()))))
        h0, h1 = AC._histogrammes(P, wm, gf8, half); n0, n1 = AC._indice_noirceur(h0), AC._indice_noirceur(h1)
    i_yeux = max(range(len(cand)), key=lambda k: _cle_yeux(cand[k])); i_bl = int(np.argmax([c["blastema_area"] for c in cand]))
    i_noir = None
    if np.isfinite(n0) and np.isfinite(n1):
        end = P[0] if n0 > n1 else P[-1]
        i_noir = int(np.argmin([np.hypot(c["tip"][0] - end[0], c["tip"][1] - end[1]) for c in cand]))
    diag = dict(rmax=rmax, marge_bl=marge, noir0=n0, noir1=n1, i_yeux=i_yeux, i_blasteme=i_bl, i_noirceur=i_noir)
    if rmax >= PEAK_ANCRAGE:
        idx = list(range(len(cand)))
        if YEUX_SUR_BLASTEME and any(c["blastema_area"] > 0 for c in cand):
            idx = [k for k in idx if cand[k]["blastema_area"] > 0]   # the blastema is authoritative: eyespots sought among the ends that carry one
        i = max(idx, key=lambda k: _cle_yeux(cand[k])); stage = "yeux"; conf = rmax
    else:
        if a[0] > 0 and marge >= MARGE_BL:
            i = int(np.argmax([c["blastema_area"] for c in cand])); stage = "blastema"; conf = marge
        else:
            # stage 4, darkness along the skeleton axis, assigned to the nearest candidate
            if i_noir is not None: i = i_noir; stage = "noirceur"; conf = abs(n0 - n1)
            else: i = i_bl; stage = "indetermine"; conf = marge
    t = cand[i]
    if t["yeux"]:
        x = float(np.mean([q[0] for q in t["yeux"]])); y = float(np.mean([q[1] for q in t["yeux"]]))
    else:
        x, y = float(t["tip"][1]), float(t["tip"][0])
    return dict(x=x, y=y, stage=stage, confidence=float(conf), tip=t["tip"], centre=t["centre"], L=float(L), axis_kind=nature,
                n_blastemes=len(blast), corrected=int(t["corrected"]), source=t["source"], candidates=cand, diag=diag,
                blasteme_mask=((Bimg >= SEUIL_BL).astype(np.uint8) * 255),   # detected blastema region (both ends), image frame
                i_head=int(i), ends_blasteme=ends_blasteme)
