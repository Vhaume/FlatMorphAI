# -*- coding: utf-8 -*-
"""Head anchoring cascade v3: blastema first, head correction by the blastema, then eyespots.

    Working copy; ancrage_cascade.py is not modified (it supplies the network loading, the peak detection and
    the darkness index, reused as they are). Rule defined with the experimenter on 10 October 2026: the
    skeleton does not find the head in 100 % of cases (stocky worms), but the blastema shows where it is. A
    worm may carry two blastemas (one at each end); only one of them carries the eyespots.

        Stage 0, framing. Worm axis by cadrage_squelette (longest path of the pruned skeleton, or principal
        line for a stocky worm), two ends, thumbnails of fraction 0.45 taken in the original image without
        prior straightening.
        Stage 1, blastema. Segmentation of both thumbnails (blasteme_unet_384_v3, trained on this framing),
        predictions brought back into the image and merged. The components of at least A_MIN thumbnail pixels
        are blastemas (two at most, the largest ones).
        Stage 2, head correction by the blastema. For each blastema, the candidate end is the nearest
        skeleton tip, unless the blastema centre is further than D_CORR times the length away: the tip is
        then moved onto the worm border, where the ray starting from the innermost point of the worm and
        passing through the blastema centre leaves the worm (blastema included), and the thumbnail is
        re-framed with the same proportions. The opposite skeleton end stays a candidate when a single
        blastema is seen (the head may already be pigmented at the other end).
        Stage 3, eyespots. Eyespot detector fondu_v5_jeu3E, the only detector of the chain, on each candidate
        (thumbnail of fraction 0.30, as in the cascade in service, counting threshold 0.61). If a maximal
        response reaches PEAK_ANCRAGE, the candidate with the greatest peak mass is the head (a single
        blastema carries the eyespots). With YEUX_SUR_BLASTEME, only the candidates carrying a blastema are
        separated by the eyespots when at least one blastema is seen.
        Stage 4, without eyespots. Blastema: relative margin of the candidates' blastema areas at least
        MARGE_BL, the largest one is the head. Otherwise darkness (Otsu index over the 60 sections from 0 to
        30 % of each end, along the skeleton axis), the darkest end.

    The thresholds are those of the cascade in service (PEAK_ANCRAGE 0.85, MARGE_BL 0.84), PEAK_THR 0.61 of
    the v5 detector, plus D_CORR = 0.15 and A_MIN = 30, fixed before the test_ancrage_cascade_v3 trial.
    References. See ancrage_cascade.py and cadrage_squelette.py.
"""
import os, numpy as np, cv2
import ancrage_cascade as AC
import cadrage_squelette as CS

CK_BL3 = os.path.join("checkpoints_regeneration", "blasteme_unet_384_v3", "blasteme_unet_384_best.pth")
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
_MDL = {}


def charger(verbeux=True):
    """Networks of the v3 cascade (blastema v3, eyespots). Paths CK_BL3 and CK_EY3 may be changed beforehand."""
    cle = (CK_BL3, CK_EY3)
    if _MDL.get("cle") != cle:
        mb, cb = AC._charge(CK_BL3); me, ce = AC._charge(CK_EY3)
        _MDL.update(cle=cle, bl=mb, ey=me)
        if verbeux:
            print("cascade v3, device:", AC.DEVICE, "| blastema:", os.path.basename(os.path.dirname(CK_BL3)),
                  "| eyespots:", os.path.basename(os.path.dirname(CK_EY3)), "peak threshold", PEAK_THR, "decision", PEAK_ANCRAGE)
    return _MDL["bl"], _MDL["ey"]


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
    wd = CS.masque_dilate(wm, L); H, W = wm.shape
    # stage 1, blastema at both skeleton ends, brought back into the image
    Bimg = np.zeros((H, W), np.float32)
    for tip, centre in ends:
        M = CS.affine(tip, centre, L); v = CS.vignette(bgr, wd, M)
        Bimg = np.maximum(Bimg, CS.retroprojeter(AC._pred(mdl_bl, v), M, (H, W)))
    ech = (2 * CS.FRAC * L / CS.S) ** 2
    n, lab, st, cen = cv2.connectedComponentsWithStats((Bimg >= 0.5).astype(np.uint8), 8)
    blast = sorted([(float(st[k, cv2.CC_STAT_AREA]), (float(cen[k][1]), float(cen[k][0])), k) for k in range(1, n)
                    if st[k, cv2.CC_STAT_AREA] >= A_MIN * ech], reverse=True)[:2]
    # stage 2, candidates: one per blastema (tip corrected if needed), plus the opposite end when only one
    dt = cv2.distanceTransform(wm, cv2.DIST_L2, 5); c0 = np.unravel_index(int(np.argmax(dt)), dt.shape)
    dedans = (wm > 0) | (lab > 0)
    cand = []
    for aire, cb, k in blast:
        j = int(np.argmin([np.hypot(cb[0] - e[0][0], cb[1] - e[0][1]) for e in ends]))
        dist = float(np.hypot(cb[0] - ends[j][0][0], cb[1] - ends[j][0][1]))
        if dist <= D_CORR * L:
            cand.append(dict(tip=ends[j][0], centre=ends[j][1], source="blastema", corrected=0, end=j, blastema_area=aire, dist_corr=dist / L))
        else:
            tip, u = _rayon_bord(dedans, c0, cb); centre = tip - CS.POS_CENTRE * L * u
            cand.append(dict(tip=tip, centre=centre, source="blastema", corrected=1, end=j, blastema_area=aire, dist_corr=dist / L))
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
        c["masse_yeux"] = float(p[p >= PEAK_THR].sum()); c["rmax"] = float(p.max())
        pk = AC._pics(p); c["yeux"] = [tuple(_vers_image(M, px, py)) for px, py, _ in pk]
    rmax = max(c["rmax"] for c in cand)
    # quantities of every stage, computed for each worm (diagnostics, alternative decisions offline)
    a = sorted([c["blastema_area"] for c in cand], reverse=True)
    marge = (a[0] - a[1]) / (a[0] + a[1] + 1e-9) if len(a) > 1 else 1.0
    n0 = n1 = float("nan")
    if P is not None:
        gf8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); half = max(3, int(round(0.5 * np.sqrt(wm.sum()))))
        h0, h1 = AC._histogrammes(P, wm, gf8, half); n0, n1 = AC._indice_noirceur(h0), AC._indice_noirceur(h1)
    i_yeux = int(np.argmax([c["masse_yeux"] for c in cand])); i_bl = int(np.argmax([c["blastema_area"] for c in cand]))
    i_noir = None
    if np.isfinite(n0) and np.isfinite(n1):
        end = P[0] if n0 > n1 else P[-1]
        i_noir = int(np.argmin([np.hypot(c["tip"][0] - end[0], c["tip"][1] - end[1]) for c in cand]))
    diag = dict(rmax=rmax, marge_bl=marge, noir0=n0, noir1=n1, i_yeux=i_yeux, i_blasteme=i_bl, i_noirceur=i_noir)
    if rmax >= PEAK_ANCRAGE:
        idx = list(range(len(cand)))
        if YEUX_SUR_BLASTEME and any(c["blastema_area"] > 0 for c in cand):
            idx = [k for k in idx if cand[k]["blastema_area"] > 0]   # the blastema is authoritative: eyespots sought among the ends that carry one
        i = max(idx, key=lambda k: cand[k]["masse_yeux"]); stage = "yeux"; conf = rmax
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
                n_blastemes=len(blast), corrected=int(t["corrected"]), source=t["source"], candidates=cand, diag=diag)
