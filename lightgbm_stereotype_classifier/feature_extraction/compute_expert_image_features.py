# -*- coding: utf-8 -*-
# Image descriptors of the Walnut expert only (LightGBM #2), computed on the raw frame and its binary mask:
#   table pigment_<split>.csv : pig_localstd, pig_std, glcm_homogeneity, glcm_contrast
#   table sat_<split>.csv     : sat_std, sat_localstd, R_localstd, sat_iqr
#   table central_<split>.csv : central_bright, central_max, central_vs_ring
#   table valley_<split>.csv  : valley_len, valley_energy
# Only 7 columns are used by the model (pig_localstd, glcm_homogeneity, sat_std, sat_localstd, central_max,
# central_bright, valley_energy). The others are written to keep the format of the original tables.
# This script reproduces exactly the code that produced the training tables (verified on the validation set, see
# README section 3.3). All values are ratios or normalized quantities, independent of size.
#
# Common steps: cropping to the bounding box of the mask (+6 px), body interior obtained by elliptical erosion
# with a radius proportional to sqrt(area), local CLAHE equalization (Zuiderveld, Contrast Limited Adaptive
# Histogram Equalization, Graphics Gems IV, 1994, DOI 10.1016/B978-0-12-336156-1.50061-6).
#  - pigment: local (5x5 window) and global standard deviation of the intensity relative to the mean, homogeneity and
#    contrast of the gray-level co-occurrence matrix on 32 levels, distances 1 and 2, angles 0 and 90 degrees
#    (Haralick, Shanmugam, Dinstein, Textural Features for Image Classification, IEEE Trans. SMC, 1973,
#    DOI 10.1109/TSMC.1973.4309314), via scikit-image graycomatrix / graycoprops.
#  - saturation: S channel of the HSV space, global standard deviation, local 5x5 standard deviation and interquartile
#    range relative to the mean, local standard deviation of the red channel.
#  - central luminance: central disk (distance to the centroid < 0.35 equivalent radius) versus peripheral ring
#    (> 0.6 equivalent radius), 90th percentile and maximum of the disk minus the body median, divided by the
#    1-99 percentile range of the body intensity.
#  - dorsal valley: Frangi vesselness filter on dark ridges, scales 1 to 5 (Frangi et al., Multiscale
#    vessel enhancement filtering, MICCAI 1998, DOI 10.1007/BFb0056195), mean energy in the body interior
#    (x1000) and length of the longest path of the Zhang-Suen skeleton of the ridge zone (Zhang, Suen, A fast
#    parallel algorithm for thinning digital patterns, Commun. ACM, 1984, DOI 10.1145/357994.358023) divided by the
#    equivalent diameter.
#
# Inputs: folder of raw RGB frames (recursive search), folder of binary masks (same file name),
#         list of frames (semicolon CSV with an image or image_path column).
# Usage: python compute_expert_image_features.py --images <dir> --masks <dir> --list <csv> --split test --out <dir>
# Outputs: <out>/pigment_<split>.csv, sat_<split>.csv, central_<split>.csv, valley_<split>.csv (semicolon separator),
#          read by assemble_features.py (--sigdir <out> --split <split>).
# Reproducibility: deterministic (no random draws), CPU only, semicolon CSV separator.
import os, glob, argparse
from collections import deque
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import frangi
from skimage.morphology import skeletonize

NAN = float('nan')
COLS = {'pigment': ['pig_localstd', 'pig_std', 'glcm_homogeneity', 'glcm_contrast'],
        'sat': ['sat_std', 'sat_localstd', 'R_localstd', 'sat_iqr'],
        'central': ['central_bright', 'central_max', 'central_vs_ring'],
        'valley': ['valley_len', 'valley_energy']}
CLAHE_CLIP = {'pigment': 3.0, 'central': 4.0, 'valley': 5.0}


def _load(ip, mp, color=False):
    im = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    m = cv2.imdecode(np.fromfile(mp, np.uint8), 0)
    if im is None or m is None:
        return None
    if im.shape[:2] != m.shape:
        im = cv2.resize(im, (m.shape[1], m.shape[0]))
    ys, xs = np.where(m > 0)
    if len(ys) < 50:
        return None
    mr = 6
    sl = (slice(max(0, ys.min() - mr), ys.max() + mr), slice(max(0, xs.min() - mr), xs.max() + mr))
    return im[sl], m[sl] > 0


def _inner(mm, frac):
    rad = int(max(3 if frac > 0.1 else 2, frac * np.sqrt(mm.sum())))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    inner = cv2.erode(mm.astype(np.uint8), ker).astype(bool)
    return inner if inner.sum() >= 30 else mm


def _localstd(a):
    b = cv2.blur(a, (5, 5)); b2 = cv2.blur(a * a, (5, 5))
    return np.sqrt(np.clip(b2 - b * b, 0, None))


def pigment(ip, mp):
    r = _load(ip, mp)
    if r is None:
        return None
    g, mm = r
    gi = cv2.createCLAHE(clipLimit=CLAHE_CLIP['pigment'], tileGridSize=(8, 8)).apply(g).astype(float)
    inner = _inner(mm, 0.12); vals = gi[inner]
    q = np.zeros_like(gi, np.uint8); lo, hi = vals.min(), vals.max()
    q[inner] = np.clip((gi[inner] - lo) / (hi - lo + 1e-6) * 31, 0, 31).astype(np.uint8)
    gl = graycomatrix(q, [1, 2], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    return dict(pig_localstd=float(_localstd(gi)[inner].mean() / (vals.mean() + 1e-6)),
                pig_std=float(vals.std() / (vals.mean() + 1e-6)),
                glcm_homogeneity=float(graycoprops(gl, 'homogeneity').mean()),
                glcm_contrast=float(graycoprops(gl, 'contrast').mean()))


def saturation(ip, mp):
    r = _load(ip, mp, color=True)
    if r is None:
        return None
    im, mm = r
    S = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].astype(float); R = im[:, :, 2].astype(float)
    inner = _inner(mm, 0.12); v = S[inner]; mu = v.mean() + 1e-6
    return dict(sat_std=float(v.std() / mu), sat_localstd=float(_localstd(S)[inner].mean() / mu),
                R_localstd=float(_localstd(R)[inner].mean() / (R[inner].mean() + 1e-6)),
                sat_iqr=float((np.percentile(v, 75) - np.percentile(v, 25)) / mu))


def central(ip, mp):
    r = _load(ip, mp)
    if r is None:
        return None
    g, mm = r
    gi = cv2.createCLAHE(clipLimit=CLAHE_CLIP['central'], tileGridSize=(8, 8)).apply(g).astype(float)
    Ys, Xs = np.where(mm); yc, xc = Ys.mean(), Xs.mean(); Req = np.sqrt(mm.sum() / np.pi)
    idx = np.indices(mm.shape); dist = np.sqrt((idx[0] - yc) ** 2 + (idx[1] - xc) ** 2)
    body = gi[mm]; med = np.median(body); span = np.percentile(body, 99) - np.percentile(body, 1) + 1e-6
    core = mm & (dist < 0.35 * Req); ring = mm & (dist > 0.6 * Req)
    if core.sum() < 8:
        return None
    return dict(central_bright=float((np.percentile(gi[core], 90) - med) / span),
                central_max=float((gi[core].max() - med) / span),
                central_vs_ring=float((gi[core].mean() - gi[ring].mean()) / span) if ring.sum() > 8 else NAN)


def _longest_path(binimg):
    pts = set(map(tuple, np.argwhere(skeletonize(binimg > 0))))
    if len(pts) < 3:
        return 0

    def nb(y, x):
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in pts]

    def bfs(s):
        seen = {s: 0}; dq = deque([s]); far = s
        while dq:
            c = dq.popleft()
            for n in nb(*c):
                if n not in seen:
                    seen[n] = seen[c] + 1; dq.append(n)
                if seen.get(n, 0) > seen[far]:
                    far = n
        return far, seen
    a, _ = bfs(next(iter(pts))); b, seen = bfs(a)
    return seen[b]


def valley(ip, mp):
    r = _load(ip, mp)
    if r is None:
        return None
    g, mm = r
    gi = cv2.createCLAHE(clipLimit=CLAHE_CLIP['valley'], tileGridSize=(8, 8)).apply(g).astype(float) / 255.0
    diam = 2 * np.sqrt(mm.sum() / np.pi)
    inner = _inner(mm, 0.08)
    fr = frangi(gi, sigmas=range(1, 6), black_ridges=True); fr[~inner] = 0
    thr = np.percentile(fr[inner], 88) if inner.sum() > 0 else 0
    vln = (fr > max(thr, 1e-6)).astype(np.uint8)
    vln = cv2.morphologyEx(vln, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return dict(valley_len=float(_longest_path(vln) / (diam + 1e-6)), valley_energy=float(fr[inner].mean() * 1000))


FUNCS = {'pigment': pigment, 'sat': saturation, 'central': central, 'valley': valley}


def compute(images_dir, masks_dir, names, tables=('pigment', 'sat', 'central', 'valley')):
    img = {}
    for p in glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True):
        img.setdefault(os.path.basename(p), p)
    msk = {}
    for p in glob.glob(os.path.join(masks_dir, '**', '*.png'), recursive=True):
        msk.setdefault(os.path.basename(p), p)
    out = {}
    for t in tables:
        rows = []
        for bn in names:
            ip, mp = img.get(bn), msk.get(bn)
            v = FUNCS[t](ip, mp) if (ip and mp) else None
            rows.append({'image': bn, **(v or {c: NAN for c in COLS[t]})})
        out[t] = pd.DataFrame(rows, columns=['image'] + COLS[t])
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--images', required=True); a.add_argument('--masks', required=True)
    a.add_argument('--list', required=True); a.add_argument('--split', required=True); a.add_argument('--out', required=True)
    x = a.parse_args()
    lst = pd.read_csv(x.list, sep=';')
    col = 'image' if 'image' in lst.columns else 'image_path'
    names = [os.path.basename(str(v)) for v in lst[col]]
    os.makedirs(x.out, exist_ok=True)
    for t, df in compute(x.images, x.masks, names).items():
        p = os.path.join(x.out, '%s_%s.csv' % (t, x.split))
        df.to_csv(p, sep=';', index=False)
        print('written %s (%d rows, coverage %.2f)' % (p, len(df), df[COLS[t][0]].notna().mean()))
