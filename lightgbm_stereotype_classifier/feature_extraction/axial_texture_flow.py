#!/usr/bin/env python
# coding: utf-8
"""axial_texture_flow.py -- TEXTURE-based motion: how the worm's internal grayscale texture flows between T-1, T0, T+1.

Motivation (from expert review of the Snake-like test failures): a gliding or scrunching Snake-like worm barely changes
its SILHOUETTE, so the mask-based motion metrics see nothing and the frame measures like a static Wildtype. What moves is
the internal TEXTURE, the pigment/shading pattern sliding along the body, with the head sliding faster than the tail. This
module computes dense optical flow on the masked GRAYSCALE (not the binary mask), projects it onto the body's principal
axis, and summarizes the gliding, the head-vs-tail differential, the lateral undulation and the directional coherence.

Columns:
  txf_axial_mag     mean |axial texture flow| inside the worm (gliding magnitude)
  txf_headtail_diff |mean axial flow in the head half - tail half| (scrunch / head-faster-than-tail)
  txf_coherence     |mean axial flow vector| / mean |axial flow| (1 = one coherent glide, 0 = incoherent)
  txf_lateral_mag   mean |lateral texture flow| (zigzag / undulation off the axis)
  txf_forward_frac  fraction of body pixels whose axial flow shares the dominant sign (directed glide)
  txf_score         composite glide score = txf_axial_mag * txf_coherence

References: Farneback, Two-Frame Motion Estimation Based on Polynomial Expansion, SCIA 2003, DOI 10.1007/3-540-45103-X_50
(dense optical flow). Cohen, Statistical Power Analysis for the Behavioral Sciences, 1988, ISBN 0-8058-0283-5 (effect
size used when ranking the columns). random_state 42, masks only plus grayscale, no GPU.
"""
import numpy as np, cv2
np.random.seed(42)

TXF_COLS = ['txf_axial_mag', 'txf_headtail_diff', 'txf_coherence', 'txf_lateral_mag', 'txf_forward_frac', 'txf_score']


def _axis(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return None
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float32)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    ax = vt[0]                                    # unit principal axis (x, y)
    return ax / (np.linalg.norm(ax) + 1e-9)


def _flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def texture_flow(g_prev, g_cur, g_next, mask_cur):
    out = {k: np.nan for k in TXF_COLS}
    if g_cur is None or mask_cur is None:
        return out
    m = (mask_cur > 0)
    if int(m.sum()) < 30:
        return out
    # both flows run FORWARD in time (T-1->T0 and T0->T+1) so consistent motion REINFORCES instead of cancelling
    def _al(g):
        return g if (g is not None and g.shape == g_cur.shape) else (cv2.resize(g, (g_cur.shape[1], g_cur.shape[0])) if g is not None else None)
    gp, gn = _al(g_prev), _al(g_next)
    flows = []
    if gp is not None:
        flows.append(_flow(gp, g_cur))            # T-1 -> T0
    if gn is not None:
        flows.append(_flow(g_cur, gn))            # T0 -> T+1
    if not flows:
        return out
    F = np.mean(flows, 0)                          # H x W x 2  (dx, dy) forward texture displacement
    ax = _axis(mask_cur)
    if ax is None:
        return out
    perp = np.array([-ax[1], ax[0]], np.float32)
    fx = F[:, :, 0][m]; fy = F[:, :, 1][m]
    axial = fx * ax[0] + fy * ax[1]               # signed flow along the body axis
    lateral = fx * perp[0] + fy * perp[1]         # flow perpendicular to the axis
    # head/tail split along the axis coordinate
    ys, xs = np.where(m)
    coord = (xs - xs.mean()) * ax[0] + (ys - ys.mean()) * ax[1]
    head = coord >= np.median(coord); tail = ~head
    amag = float(np.mean(np.abs(axial)))
    out['txf_axial_mag'] = amag
    out['txf_headtail_diff'] = float(abs(axial[head].mean() - axial[tail].mean())) if head.any() and tail.any() else np.nan
    out['txf_coherence'] = float(abs(axial.mean()) / (amag + 1e-9))
    out['txf_lateral_mag'] = float(np.mean(np.abs(lateral)))
    dom = np.sign(axial.mean() + 1e-9)
    out['txf_forward_frac'] = float(np.mean(np.sign(axial) == dom))
    out['txf_score'] = float(amag * out['txf_coherence'])
    return out
