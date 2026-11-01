"""step2_temporal_format.py -- build the 384x384 3-channel TEMPORAL image the classifier consumes.

Each output image packs the labelled frame's 3-frame window into the colour channels: channel 0 = T-1, channel 1 = T0
(the labelled frame), channel 2 = T+1, each a masked grayscale frame. The three frames are cropped with ONE common box,
the exact union of their three worm masks expanded to a square, so the worm's real motion is preserved at full
resolution and undeformed. A missing neighbour falls back to T0 (that channel is then static) so a boundary frame is
never dropped. This reproduces gen_temporal_dataset.build_temporal exactly. No GPU.

Reference for the masked, motion-preserving temporal stacking: this mirrors the dynamic-window construction used to
train the model and the optical-flow-free temporal modelling of the Temporal Shift Module (Lin, Gan, Han, ICCV 2019,
arXiv 1811.08383), where time is carried in the tensor rather than computed by flow.
"""
import numpy as np
import cv2

IMAGE_SIZE = 384
SQUARE = True


def _as_binary(mask):
    return (np.asarray(mask) > 0).astype(np.uint8)


def _masked_gray(gray, mask):
    g = np.asarray(gray)
    b = _as_binary(mask)
    if b.shape != g.shape:
        b = cv2.resize(b, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (g * b).astype(np.uint8), b


def _union_box(masks, H, W):
    ys, xs = [], []
    for b in masks:
        yy, xx = np.where(b > 0)
        if len(yy):
            ys += [yy.min(), yy.max()]
            xs += [xx.min(), xx.max()]
    if not ys:
        return None
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    if SQUARE:
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        half = max(y1 - y0, x1 - x0) / 2.0
        y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
    return int(max(0, y0)), int(min(H, y1)), int(max(0, x0)), int(min(W, x1))


def build_temporal_image(grays, masks, image_size=IMAGE_SIZE):
    """grays, masks: 3-tuples (T-1, T0, T+1) at native resolution. T0 (index 1) is required, neighbours may be None and
    then fall back to T0. Returns a (image_size, image_size, 3) uint8 image, channel 0/1/2 = T-1/T0/T+1, or None."""
    g_prev, g_cur, g_next = grays
    m_prev, m_cur, m_next = masks
    if g_cur is None or m_cur is None:
        return None
    mg_cur, b_cur = _masked_gray(g_cur, m_cur)
    out_g, out_b = [None, mg_cur, None], [None, b_cur, None]
    for j, (gj, mj) in ((0, (g_prev, m_prev)), (2, (g_next, m_next))):
        if gj is not None and mj is not None:
            out_g[j], out_b[j] = _masked_gray(gj, mj)
        else:
            out_g[j], out_b[j] = mg_cur, b_cur                  # graceful fallback to T0
    rh, rw = out_g[1].shape
    for j in range(3):                                          # bring all frames into T0's coordinate system
        if out_g[j].shape != (rh, rw):
            out_g[j] = cv2.resize(out_g[j], (rw, rh), interpolation=cv2.INTER_AREA)
            out_b[j] = cv2.resize(out_b[j], (rw, rh), interpolation=cv2.INTER_NEAREST)
    box = _union_box(out_b, rh, rw)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    chans = []
    for g in out_g:
        crop = g[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        chans.append(cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA))
    return cv2.merge([chans[0], chans[1], chans[2]])            # BGR order = [T-1, T0, T+1]
