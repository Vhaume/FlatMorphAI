# -*- coding: utf-8 -*-
"""SKAN VERSION 2 (17 October 2026), copy of cadrage_squelette_skan (unmodified): each end is extended to the
    contour point that maximises the angle (previous point at r(e) back, end, contour point), and there is no
    principal-line fallback for stocky worms any more. Checked on 2 777 images (ouv3 masks): tip within 0.35 L
    of the annotated head for 2 765 images, against 2 758 with cadrage_squelette_skan.
    SKAN VERSION of cadrage_squelette: same interface and same proportions, axis computed without pruning, as
    the longest path of the skeleton graph (skan, Nunez-Iglesias et al. 2018, PeerJ, DOI 10.7717/peerj.4312)
    and Dijkstra's algorithm (1959, DOI 10.1007/BF01386390, scipy.sparse.csgraph.dijkstra). The original module
    is left unchanged.

    Framing of the worm end thumbnails on the SKELETON (median axis), without prior straightening.

    For the v3 blastema segmenter this replaces the historical framing (orient + ends + crop_box from
    train_blasteme_segmenter_384.py, ends + thumbnail from ancrage_cascade.py), which had two defects
    observed on 10 October 2026 on the undetected blastemas:
      * the thumbnail centre was taken at 12 % of the length along the principal axis -- a straight line --
        and not at the skeleton end, so that for a curved, V-shaped or stocky worm the thumbnail showed a
        flank instead of the head;
      * the worm was straightened by a rotation inside a canvas the size of the image, then cropped, which
        cut off the worm end and its blastema whenever it overflowed the canvas.
    Here the end and the direction come from the worm axis (below), and the thumbnail is sampled in ONE single
    affine transform from the original image (rotation, translation and scale composed), so nothing is cut by
    an intermediate step. The proportions are those of the historical framing: centre at 12 % of the length
    from the tip, half-side 0.45 of the length, worm mask dilated by 0.12 of the length before blacking out
    the background. No border and no additional enlargement. The tip is placed at the top of the thumbnail.
    The axis is the longest path of the morphological skeleton, extended to the contour at each end: it follows
    the body through its curves and passes through the head lobe even when a rounded shoulder is further away
    as the crow flies (the hooked worm of 10 October 2026, where the two geodesically furthest points gave a
    shoulder instead of the head).

    References. Median axis: H. Blum 1967 (see axe_largeur.py); exact distance transform:
    Maurer, Qi and Raghavan 2003, DOI 10.1109/TPAMI.2003.1177156.
"""
import numpy as np, cv2
from axe_largeur import lisser
from skimage.morphology import skeletonize
from skimage.draw import line
from skan import Skeleton
from scipy.sparse.csgraph import dijkstra

S = 384; FRAC = 0.45; POS_CENTRE = 0.12; DIL = 0.12; N_AXE = 200; MARGE_BOITE = 4


def masque_ver(m, seuil=128):
    """Largest component of the Mask2Former mask (same rule as worm_mask)."""
    m = (m >= seuil).astype(np.uint8); n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return m if n <= 1 else (l == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _boite(wm):
    ys, xs = np.nonzero(wm)
    return (max(0, ys.min() - MARGE_BOITE), min(wm.shape[0], ys.max() + MARGE_BOITE + 1),
            max(0, xs.min() - MARGE_BOITE), min(wm.shape[1], xs.max() + MARGE_BOITE + 1))


def _prolonger(P, mk, sens):
    """Extend the path end (P[0] when sens is 0, P[-1] otherwise) by a ray drawn with skimage.draw.line, in the
    direction of its last 6 pixels, stopping at the last pixel inside the mask before the first one outside."""
    q, r = (P[0], P[min(6, len(P) - 1)]) if sens == 0 else (P[-1], P[max(0, len(P) - 7)])
    d = q - r; n = np.hypot(*d)
    if n < 1e-6: return q
    d = d / n; H, W = mk.shape; loin = q + d * 2.0 * max(H, W)
    rr, cc = line(int(round(q[0])), int(round(q[1])), int(round(loin[0])), int(round(loin[1])))
    p = q.copy()
    for y, x in zip(rr[1:], cc[1:]):
        if not (0 <= y < H and 0 <= x < W and mk[y, x]): break
        p = np.array([y, x], float)
    return p


def plus_long_chemin(sk):
    """Longest path of the skeleton: graph of the skeleton pixels built by skan.Skeleton (edges of length 1 or
    sqrt 2), shortest distances between endpoints by Dijkstra's algorithm (scipy.sparse.csgraph.dijkstra),
    then the furthest apart pair of endpoints. Returns the (y, x) pixels of the path."""
    g = Skeleton(sk); deg = np.asarray(g.degrees).ravel(); co = np.asarray(g.coordinates)
    ends = np.flatnonzero(deg == 1)
    if len(ends) < 2: return None
    dist, pred = dijkstra(g.graph, directed=False, indices=ends, return_predecessors=True)
    dist = np.where(np.isfinite(dist), dist, -1.0)[:, ends]
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    a, b = ends[i], ends[j]; path_px = [b]
    while path_px[-1] != a and pred[i, path_px[-1]] >= 0: path_px.append(pred[i, path_px[-1]])
    return co[np.array(path_px[::-1])].astype(float)


def _prolonger_angle(path, m, dt, sens, cont):
    """Extend the path end (P[0] when sens is 0, P[-1] otherwise) to the contour. End e, radius r(e) of the
    inscribed disc (cv2.distanceTransform). Previous point: the path point at least r(e) away from e walking
    back along the path. Candidates: contour pixels (cv2.findContours) within 3 r(e) + 3 pixels of e whose
    segment from e (skimage.draw.line) stays inside the worm. Tip: the candidate maximising the angle
    (previous point, e, candidate), i.e. the closest to alignment (180 degrees). Returns the tip (y, x), the
    angle in degrees and the previous point."""
    P = path if sens == 1 else path[::-1]; e = P[-1]; r = float(dt[int(round(e[0])), int(round(e[1]))])
    d = np.hypot(*(P - e).T); k = np.flatnonzero(d >= max(r, 2.0)); prev = P[k[-1]] if len(k) else P[0]
    u = (prev - e) / (np.hypot(*(prev - e)) + 1e-9)
    c = cont[np.hypot(*(cont - e).T) <= 3 * r + 3]
    if len(c) == 0: return e, 0.0, prev
    v = c - e; ang = np.degrees(np.arccos(np.clip((v @ u) / (np.hypot(*v.T) + 1e-9), -1, 1)))
    for i in np.argsort(-ang, kind="stable"):
        rr, cc = line(int(round(e[0])), int(round(e[1])), int(c[i][0]), int(c[i][1]))
        if m[rr, cc].mean() >= 0.95: return c[i].astype(float), float(ang[i]), prev
    return e, 0.0, prev


def squelette(wm):
    """Worm axis (N_AXE points y x, length, kind). Morphological skeleton (skimage.morphology.skeletonize),
    longest path between two endpoints (skan.Skeleton and scipy.sparse.csgraph.dijkstra), extended at each end
    to the contour point best aligned with the path (_prolonger_angle), resampled and smoothed (numpy.interp,
    numpy.convolve). No pruning, and no principal-line fallback for stocky worms."""
    from scipy import ndimage as ndi
    y0, y1, x0, x1 = _boite(wm)
    m = ndi.binary_fill_holes(wm[y0:y1, x0:x1] > 0).astype(np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if m.sum() < 60: return None, 0.0, "echec"
    sk = skeletonize(m > 0)
    if sk.sum() < 3: return None, 0.0, "echec"
    try:
        path = plus_long_chemin(sk)
        if path is None or len(path) < 3: return None, 0.0, "echec"
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        cont = np.vstack([q[:, 0, ::-1] for q in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]).astype(float)
        a_ = _prolonger_angle(path, m, dt, 0, cont)[0]; b_ = _prolonger_angle(path, m, dt, 1, cont)[0]
        P, L = lisser(np.vstack([a_[None], path, b_[None]]), N_AXE)
    except Exception: return None, 0.0, "echec"
    if P is None or L < 20: return None, 0.0, "echec"
    return P + np.array([y0, x0], float), L, "skeleton"


def _repli_acp(wm):
    """Fallback when the skeleton fails: ends along the principal axis (centre at 12 % of the extent)."""
    ys, xs = np.nonzero(wm); pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    u = np.linalg.svd(pts - c, full_matrices=False)[2][0]; t = (pts - c) @ u
    L = float(t.max() - t.min()); out = []
    for s in (t.min(), t.max()):
        tip = c + s * u; cen = c + (s - np.sign(s) * POS_CENTRE * L) * u
        out.append((tip[::-1], cen[::-1]))
    return out, L


def extremites(wm):
    """The two ends [(tip (y, x), thumbnail centre (y, x)), ...] along the worm axis, its length
    and the method ('skeleton', or 'acp' as a fallback)."""
    P, L, nature = squelette(wm)
    if P is None:
        e, L = _repli_acp(wm); return e, L, "acp"
    i = int(round(POS_CENTRE * (N_AXE - 1)))
    return [(P[0], P[i]), (P[-1], P[N_AXE - 1 - i])], L, nature


def affine(tip, centre, L, frac=None):
    """2 x 3 matrix from the original image (x, y) to the S x S thumbnail: centre in the middle, tip at the top.
    frac: half-side as a fraction of L (FRAC = 0.45 for the blastema, 0.30 for the eyespots)."""
    frac = FRAC if frac is None else frac
    d = np.array([tip[1] - centre[1], tip[0] - centre[0]], float)
    n = np.hypot(*d)
    u = d / n if n > 1e-6 else np.array([0.0, -1.0])
    R = np.array([[-u[1], u[0]], [-u[0], -u[1]]])            # R u = (0, -1): the tip points up
    s = S / (2.0 * frac * L)
    A = s * R; c = np.array([centre[1], centre[0]], float)
    t = np.array([S / 2.0, S / 2.0]) - A @ c
    return np.hstack([A, t[:, None]]).astype(np.float64)


def masque_dilate(wm, L):
    k = max(1, int(DIL * L))
    return cv2.dilate(wm.astype(np.uint8), np.ones((k, k), np.uint8))


def vignette(bgr, wm_dil, M):
    """S x S thumbnail: original image sampled through M, background outside the dilated mask set to 0."""
    v = cv2.warpAffine(bgr, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    md = cv2.warpAffine(wm_dil * 255, M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
    v[~md] = 0
    return v


def projeter(binaire, M):
    """Binary mask of the original image projected into the thumbnail (linear interpolation, threshold 0.5)."""
    p = cv2.warpAffine(binaire.astype(np.float32), M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return p >= 0.5


def retroprojeter(carte, M, shape):
    """Thumbnail map brought back into the original image (H, W)."""
    return cv2.warpAffine(carte.astype(np.float32), M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def cyan_strict(bgr):
    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    return (b > 110) & (g > 110) & ((b - r) > 60) & ((g - r) > 50)


def annotation_nettoyee(annot_bgr, img_bgr, wm_dil):
    """Annotation cyan minus the cyan already present in the original image (plate reflections), restricted to
    the dilated worm. Both files must have the same size."""
    a = cyan_strict(annot_bgr)
    if img_bgr is not None and img_bgr.shape == annot_bgr.shape: a &= ~cyan_strict(img_bgr)
    a &= wm_dil > 0
    a = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return a


def vignettes_ver(bgr, m_brut):
    """The two end thumbnails of an image. List of dicts (tip, centre, M, thumbnail), length, method
    ('skeleton' or 'acp'), worm mask and dilated mask in the original frame."""
    wm = masque_ver(m_brut)
    if wm.sum() < 300: return None
    ends, L, meth = extremites(wm)
    wd = masque_dilate(wm, L)
    out = []
    for tip, centre in ends:
        M = affine(tip, centre, L)
        out.append(dict(tip=tip, centre=centre, M=M, vignette=vignette(bgr, wd, M)))
    return dict(bouts=out, L=L, methode=meth, wm=wm, wm_dil=wd)
