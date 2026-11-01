# -*- coding: utf-8 -*-
"""Head detection (full cascade) and construction of the state-classifier composite.

Cascade (common/anchor_cascade.py, accuracy 0.9952 on 2705 worms with expert annotation):
  step 1 eyes : eye detector at both ends, decides when the maximum response >= head_decision_threshold (0.85),
                end = largest mass of peaks >= position_threshold (0.60), point = barycentre of the peaks ;
  step 2 blastema : segmenter at both ends, decides when the relative margin of the soft areas >= BLASTEMA_MARGIN (0.84) ;
  step 3 darkness : Otsu index of the ends, darkest end (DARKNESS_GAP = 0) ;
  fallback : largest soft blastema area, step "undetermined".
State-classifier composite (regenerate_composite_final): three 384 x 384 grey channels, c0 whole straightened
  worm, c1 head zoom (half-side STAGE_C1_FRACTION x length), c2 head super-zoom (STAGE_C2_FRACTION) with CLAHE
  (STAGE_C2_CLAHE_CLIP, STAGE_C2_CLAHE_TILE) and background zeroed.

Command line: python head/head_detection.py image.jpg mask.png [--out folder]
Module: from head_detection import detect_head ; r = detect_head(bgr, mask)
"""
import os, sys, json, argparse, numpy as np, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "common"), os.path.join(_HERE, "..", "parameters")):
    if _p not in sys.path: sys.path.insert(0, _p)
import anchor_cascade as AC
from preprocessing import prepare, load_image, load_mask, to_original
from parameter_reader import read
_PV = read("thumbnails"); SIZE = int(_PV["SIZE"])
F_C1 = float(_PV["STAGE_C1_FRACTION"]); F_C2 = float(_PV["STAGE_C2_FRACTION"])
_clahe = cv2.createCLAHE(clipLimit=float(_PV["STAGE_C2_CLAHE_CLIP"]), tileGridSize=(int(_PV["STAGE_C2_CLAHE_TILE"]),) * 2)


def _sq(c):
    s = max(c.shape[:2]); sq = np.zeros((s, s), c.dtype)
    sq[(s - c.shape[0]) // 2:(s - c.shape[0]) // 2 + c.shape[0], (s - c.shape[1]) // 2:(s - c.shape[1]) // 2 + c.shape[1]] = c
    return cv2.resize(sq, (SIZE, SIZE))


def whole_channel(rg, rm):
    g = rg.copy(); g[~rm] = 0; ys, xs = np.nonzero(rm)
    return _sq(g[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def _box(shape, ax, ay, h):
    H, W = shape; return (max(0, int(ay - h)), min(H, int(ay + h)), max(0, int(ax - h)), min(W, int(ax + h)))


def composite(rg, rm, ax, ay, span):
    """BGR composite (c0, c1, c2), identical to regenerate_composite_final.composite."""
    c0 = whole_channel(rg, rm)
    y0, y1, x0, x1 = _box(rm.shape, ax, ay, int(F_C1 * span)); g = rg.copy(); g[~rm] = 0
    c1 = _sq(g[y0:y1, x0:x1]) if g[y0:y1, x0:x1].size else np.zeros((SIZE, SIZE), np.uint8)
    y0, y1, x0, x1 = _box(rm.shape, ax, ay, int(F_C2 * span)); cg = rg[y0:y1, x0:x1]; cm = rm[y0:y1, x0:x1].astype(np.uint8) * 255
    if cg.size == 0: c2 = np.zeros((SIZE, SIZE), np.uint8)
    else:
        znat = _sq(cg); foot = _sq(cm) > 127; c2 = _clahe.apply(znat); c2[~foot] = 0
    return cv2.merge([c0, c1, c2])


def detect_head(bgr, raw_mask, with_composite=True):
    """Returns dict: ok, x, y (straightened frame), x_original, y_original, step, confidence, end (0 top, 1 bottom),
    composite (BGR 384 x 384, channels c0 c1 c2) when requested, rbgr, rm, M, span."""
    pr = prepare(bgr, raw_mask)
    if not pr["ok"]: return dict(ok=False, reason=pr["reason"])
    rbgr, rm, M, span = pr["rbgr"], pr["rm"], pr["M"], pr["span"]
    x, y, step, conf, end = AC.anchor(rbgr, rm)
    xo, yo = to_original(x, y, M)
    r = dict(ok=True, x=x, y=y, x_original=xo, y_original=yo, step=step, confidence=conf, end=end, rbgr=rbgr, rm=rm, M=M, span=span)
    if with_composite: r["composite"] = composite(cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY), rm, x, y, span)
    return r


def _cli():
    ap = argparse.ArgumentParser(description="Head detection by the cascade and state composite")
    ap.add_argument("image"); ap.add_argument("mask"); ap.add_argument("--out", default=None); a = ap.parse_args()
    r = detect_head(load_image(a.image), load_mask(a.mask))
    if not r["ok"]: print("failure:", r["reason"]); return
    print("head: end %s | step %s | confidence %.3f | straightened point (%.1f, %.1f) | original point (%.1f, %.1f)"
          % ("top" if r["end"] == 0 else "bottom", r["step"], r["confidence"], r["x"], r["y"], r["x_original"], r["y_original"]))
    if a.out:
        os.makedirs(a.out, exist_ok=True); stem = os.path.splitext(os.path.basename(a.image))[0]
        cv2.imwrite(os.path.join(a.out, stem + "_composite.png"), r["composite"])
        cv2.imwrite(os.path.join(a.out, stem + "_straightened.png"), r["rbgr"])
        with open(os.path.join(a.out, stem + "_head.json"), "w", encoding="utf-8") as f:
            json.dump(dict(image=a.image, end=r["end"], step=r["step"], confidence=r["confidence"], x=r["x"], y=r["y"],
                           x_original=r["x_original"], y_original=r["y_original"], M=[list(map(float, l)) for l in r["M"]]), f, indent=1)
        print("written to", a.out)


if __name__ == "__main__": _cli()
