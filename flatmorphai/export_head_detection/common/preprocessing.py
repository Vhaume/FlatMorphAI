# -*- coding: utf-8 -*-
"""Preprocessing shared by all detections: worm mask, straightening, file reading.

Functions taken verbatim from the training chain (train_eye_detector_unified_384_*, regenerate_composite_final,
rebuild_processed_images_full_size):
  * worm_mask : binarisation at MASK_THR then largest connected component ;
  * straighten : rotation aligning the main axis of the mask (SVD of the pixels) with the vertical ;
  * load_image / load_mask : BGR reading (OpenCV) and grey-level mask.
All detections (eyes, blastema, darkness, head) work on the STRAIGHTENED image and its straightened mask,
and return their coordinates in that straightened frame. The matrix M brings them back to the original
image frame (see to_original).
"""
import os, sys, numpy as np, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "..", "parameters") not in sys.path: sys.path.insert(0, os.path.join(_HERE, "..", "parameters"))
from parameter_reader import read
_PC = read("cascade_parameters")
MASK_THR = int(_PC["MASK_THR"]); MIN_AREA = int(_PC["MIN_MASK_AREA"])


def load_image(path):
    """BGR uint8 colour image (OpenCV convention), Windows paths with accents accepted."""
    d = np.fromfile(path, dtype=np.uint8); im = cv2.imdecode(d, cv2.IMREAD_COLOR)
    if im is None: raise IOError("unreadable image: %s" % path)
    return im


def load_mask(path):
    d = np.fromfile(path, dtype=np.uint8); m = cv2.imdecode(d, cv2.IMREAD_GRAYSCALE)
    if m is None: raise IOError("unreadable mask: %s" % path)
    return m


def worm_mask(m):
    """0/1 mask: binarisation at MASK_THR then largest connected component (8-connectivity)."""
    if m.dtype == bool: mk = m.astype(np.uint8)
    else: mk = (m >= MASK_THR).astype(np.uint8) if m.max() > 1 else m.astype(np.uint8)
    n, l, st, _ = cv2.connectedComponentsWithStats(mk, 8)
    return mk if n <= 1 else (l == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def straighten(bgr, mask):
    """Align the main axis of the mask with the vertical. Returns (straightened BGR image, bool mask, M)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ang = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
    H, W = mask.shape
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang - 90, 1.0)
    rb = cv2.warpAffine(bgr, M, (W, H))
    rm = cv2.warpAffine((mask * 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 127
    return rb, rm, M


def to_original(x, y, M):
    """Bring a point of the straightened frame back to the original image frame (inverse of M)."""
    Mi = cv2.invertAffineTransform(M); p = Mi @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


def prepare(bgr, raw_mask):
    """Full chain: worm mask, area check, straightening.
    Returns dict(mask, rbgr, rm, M, span, ok). ok = False when the mask is too small."""
    mask = worm_mask(raw_mask)
    if int(mask.sum()) < MIN_AREA: return dict(mask=mask, ok=False, reason="mask too small (%d px)" % int(mask.sum()))
    rbgr, rm, M = straighten(bgr, mask)
    ys, _ = np.nonzero(rm); span = int(ys.max() - ys.min() + 1)
    return dict(mask=mask, rbgr=rbgr, rm=rm, M=M, span=span, ok=True)


def project_map(amap, box, shape):
    """Put a 384 x 384 map produced on a thumbnail back into the straightened frame (H, W), background 0.
    box = (y0, x0, oy, ox, side) as returned by anchor_cascade.thumbnail."""
    y0, x0, oy, ox, side = box; H, W = shape
    g = cv2.resize(amap.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    c = g[oy:oy + (min(H, y0 + side) - y0), ox:ox + (min(W, x0 + side) - x0)]
    out = np.zeros((H, W), np.float32)
    h = min(c.shape[0], H - y0); w = min(c.shape[1], W - x0)
    out[y0:y0 + h, x0:x0 + w] = c[:h, :w]
    return out
