# -*- coding: utf-8 -*-
"""Full chain on ONE image: mask (optional), head, eye cups, blastema, darkness, state composite.

This is the entry point to reproduce in the PySide interface: every step is an importable function, the
parameters come from parameters/*.csv and the models from parameters/models.csv.

    python single_image_pipeline.py image.jpg [--mask mask.png] --out folder
Without --mask, the Mask2Former segmenter is loaded (GPU).
State prediction is not chained here: it needs the 97 descriptors of the metrics/ chain (see README,
section 7); stage/stage_prediction.py shows the call once those descriptors are computed.
"""
import os, sys, json, argparse, numpy as np, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("common", "parameters", "eyes", "blastema", "darkness", "head", "segmentation"):
    p = os.path.join(_HERE, d)
    if p not in sys.path: sys.path.insert(0, p)
from preprocessing import load_image, load_mask
from head_detection import detect_head
from eye_detection import eye_maps
from blastema_detection import blastema_maps
from darkness_index import darkness_indices


def analyse(bgr, mask=None, segmenter=None):
    """Returns a complete dict. mask: uint8 0/255 Mask2Former mask; otherwise segmenter.segment(bgr) is called."""
    if mask is None:
        if segmenter is None:
            from segment_mask import Segmenter; segmenter = Segmenter()
        mask = segmenter.segment(bgr)
    h = detect_head(bgr, mask, with_composite=True)
    if not h["ok"]: return dict(ok=False, reason=h["reason"], mask=mask)
    rbgr, rm, span = h["rbgr"], h["rm"], h["span"]
    ey = eye_maps(rbgr, rm, span); bl = blastema_maps(rbgr, rm, span); dk = darkness_indices(rbgr, rm)
    e = ey[h["end"]]
    return dict(ok=True, mask=mask, head=dict(end=h["end"], step=h["step"], confidence=h["confidence"], x=h["x"], y=h["y"], x_original=h["x_original"], y_original=h["y_original"]),
                eyes=dict(count=len(e["peaks"]), peaks_straight=e["peaks_straight"], max_response=[c["max_response"] for c in ey], mass=[c["mass"] for c in ey]),
                blastema=dict(soft_area=[c["soft_area"] for c in bl], binary_area_px=[c["binary_area_px"] for c in bl],
                              margin=abs(bl[0]["soft_area"] - bl[1]["soft_area"]) / (bl[0]["soft_area"] + bl[1]["soft_area"] + 1e-9)),
                darkness=(dict(index_top=dk["index_top"], index_bottom=dk["index_bottom"], gap=dk["gap"]) if dk["ok"] else dict(ok=False, reason=dk["reason"])),
                composite=h["composite"], eye_maps=ey, blastema_maps=bl, M=h["M"])


def _cli():
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--mask", default=None); ap.add_argument("--out", required=True); a = ap.parse_args()
    bgr = load_image(a.image); m = load_mask(a.mask) if a.mask else None
    r = analyse(bgr, m); os.makedirs(a.out, exist_ok=True); stem = os.path.splitext(os.path.basename(a.image))[0]
    cv2.imwrite(os.path.join(a.out, stem + "_mask.png"), r["mask"])
    if not r["ok"]: print("failure:", r["reason"]); return
    cv2.imwrite(os.path.join(a.out, stem + "_composite.png"), r["composite"])
    res = {k: r[k] for k in ("head", "eyes", "blastema", "darkness")}; res["image"] = a.image
    with open(os.path.join(a.out, stem + "_analysis.json"), "w", encoding="utf-8") as f: json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__": _cli()
