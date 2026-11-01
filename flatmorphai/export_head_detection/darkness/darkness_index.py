# -*- coding: utf-8 -*-
"""Darkness index of the two ends, step 3 of the cascade, no learned model.

Method (anchor_cascade, version of 4 October 2026):
  1. width-aligned medial axis (width_axis.py: ridge of the distance transform, minimum-cost path),
     resampled to N_AXIS = 200 points ;
  2. at each point, a perpendicular cut bounded by the mask reads the worm pixels (half-length
     max(3, 0.5 x square root of the mask area)); the first N_CUTS = 60 cuts (abscissa 0 to 0.30) feed the
     grey-level histogram of one end, the last 60 feed the other ;
  3. on each histogram, Otsu threshold (Otsu 1979, DOI 10.1109/TSMC.1979.4310076) between dark and bright
     pixels, index = 1 - median of the dark class / 255 ;
  4. the end with the highest index (the darkest) is the head, without a gap threshold (DARKNESS_GAP = 0).
The fallback (axis not computable) is reported by ok = False.

Command line: python darkness/darkness_index.py image.jpg mask.png
"""
import os, sys, json, argparse, numpy as np, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "common"), os.path.join(_HERE, "..", "parameters")):
    if _p not in sys.path: sys.path.insert(0, _p)
import anchor_cascade as AC
from width_axis import medial_axis, smooth
from preprocessing import prepare, load_image, load_mask


def darkness_indices(rbgr, rm):
    """Indices of the two ends in (top, bottom) order of the straightened worm. ok = False when the axis is not found."""
    pa = medial_axis(rm.astype(np.uint8))
    if pa is None: return dict(ok=False, reason="medial axis not computable")
    P, L = smooth(pa, AC.N_AXIS)
    if P is None or L < 20: return dict(ok=False, reason="axis too short")
    gf8 = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY)
    half = max(3, int(round(0.5 * np.sqrt(rm.sum()))))
    h0, h1 = AC._histograms(P, rm.astype(np.uint8), gf8, half)
    n0, n1 = AC._darkness_index(h0), AC._darkness_index(h1)
    if not (np.isfinite(n0) and np.isfinite(n1)): return dict(ok=False, reason="insufficient histogram")
    top_is_P0 = P[0][0] <= P[-1][0]
    top, bottom = (n0, n1) if top_is_P0 else (n1, n0)
    return dict(ok=True, index_top=float(top), index_bottom=float(bottom), gap=float(abs(n0 - n1)),
                darkest_end=0 if top > bottom else 1, histograms=(h0, h1) if top_is_P0 else (h1, h0), axis=P, half_width=half)


def compute_darkness(bgr, raw_mask):
    pr = prepare(bgr, raw_mask)
    if not pr["ok"]: return dict(ok=False, reason=pr["reason"])
    r = darkness_indices(pr["rbgr"], pr["rm"]); r["M"] = pr["M"]; return r


def _cli():
    ap = argparse.ArgumentParser(description="Darkness index of the two ends"); ap.add_argument("image"); ap.add_argument("mask"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); r = compute_darkness(load_image(a.image), load_mask(a.mask))
    if not r["ok"]: print("failure:", r["reason"]); return
    print("index top %.4f | index bottom %.4f | gap %.4f | darkest end: %s" % (r["index_top"], r["index_bottom"], r["gap"], "top" if r["darkest_end"] == 0 else "bottom"))
    if a.out:
        os.makedirs(a.out, exist_ok=True); stem = os.path.splitext(os.path.basename(a.image))[0]
        with open(os.path.join(a.out, stem + "_darkness.json"), "w", encoding="utf-8") as f:
            json.dump(dict(image=a.image, index_top=r["index_top"], index_bottom=r["index_bottom"], gap=r["gap"], darkest_end=r["darkest_end"],
                           histogram_top=[int(v) for v in r["histograms"][0]], histogram_bottom=[int(v) for v in r["histograms"][1]]), f)
        print("written to", a.out)


if __name__ == "__main__": _cli()
