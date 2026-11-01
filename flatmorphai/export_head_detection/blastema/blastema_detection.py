# -*- coding: utf-8 -*-
"""Blastema detection from the original image and the Mask2Former mask.

Chain (identical to step 2 of the cascade):
  1. worm mask and straightening (common/preprocessing.py) ;
  2. centres of the two terminal bands ;
  3. the mask is DILATED by a square of side BLASTEMA_MASK_DILATION x length (0.12), because Mask2Former
     often cuts the pale blastema, then background zeroed ;
  4. for each end, a 384 square thumbnail of half-side BLASTEMA_FRACTION x length (0.45) ;
  5. four channels RGB + local contrast z, U-Net EfficientNet-B0 (models.csv: blastema_segmenter),
     output = blastema probability per pixel ;
  6. soft area = sum of the probabilities, brought to the straightened-image scale by (side / 384)^2 ;
  7. relative margin = |a0 - a1| / (a0 + a1). In the cascade, the blastema step decides when margin >= BLASTEMA_MARGIN.
Also returns the binary blastema mask (probability >= BLASTEMA_PROB_MIN, default 0.4) put back into the original image frame, for
display or for the proportion measure (blastema area / worm area).

Replaceable model: put the new checkpoint in parameters/models.csv (blastema_segmenter). Expected format:
dict with key state_dict of a smp.Unet(timm-efficientnet-b0, in_channels=4, classes=1); if the architecture
changes, adapt anchor_cascade._load and _chans.

Command line: python blastema/blastema_detection.py image.jpg mask.png [--out folder]
"""
import os, sys, json, argparse, numpy as np, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "common"), os.path.join(_HERE, "..", "parameters")):
    if _p not in sys.path: sys.path.insert(0, _p)
import anchor_cascade as AC
from preprocessing import prepare, load_image, load_mask, project_map
from parameter_reader import read
_PC = read("cascade_parameters"); BLASTEMA_MARGIN = float(_PC["BLASTEMA_MARGIN"])
# Minimum per-pixel blastema probability to classify a pixel as blastema in the
# binary mask (the displayed/saved "blastema mask"). 0.4, not 0.5: the
# U-Net probability of the pale blastema rarely exceeds 0.5, so 0.5 dropped it
# entirely on some worms. Overridable with FLATMORPHAI_BLASTEMA_PMIN.
import os as _os_bl
try:
    BLASTEMA_PROB_MIN = float(_os_bl.environ.get("FLATMORPHAI_BLASTEMA_PMIN", "0.4"))
except Exception:
    BLASTEMA_PROB_MIN = 0.4


def blastema_maps(rbgr, rm, span):
    """Blastema probability at the two ends of the straightened worm. List of two dicts (top, bottom):
    centre, thumbnail, probmap (384 x 384 in [0, 1]), box, soft_area, binary_area_px (prob >= BLASTEMA_PROB_MIN, straightened scale)."""
    mdl, _ = AC.load_models(verbose=False)
    pts, _ = AC.ends(rm)
    k = max(1, int(AC.DIL_BL * span))
    rmz = cv2.dilate(rm.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    out = []
    for cx, cy in pts:
        v = AC.thumbnail(rbgr, rmz, cx, cy, span, AC.F_BL)
        if v is None: out.append(dict(centre=(cx, cy), thumbnail=None, probmap=None, box=None, soft_area=0.0, binary_area_px=0)); continue
        p = AC._pred(mdl, v[0]); sc = (v[1][4] / AC.S) ** 2
        out.append(dict(centre=(int(cx), int(cy)), thumbnail=v[0], probmap=p, box=v[1],
                        soft_area=float(p.sum()) * sc, binary_area_px=int(round(float((p >= BLASTEMA_PROB_MIN).sum()) * sc))))
    return out


def detect_blastema(bgr, raw_mask):
    """Full detection. Returns dict: ok, ends, margin, blastema_end (0 top, 1 bottom), decides (margin >= BLASTEMA_MARGIN),
    proportion (binary area of the retained end / worm area), blastema_mask_original (uint8 0/255, original frame),
    probmap_original (float32, probability in the original frame), M."""
    pr = prepare(bgr, raw_mask)
    if not pr["ok"]: return dict(ok=False, reason=pr["reason"])
    rbgr, rm, M, span = pr["rbgr"], pr["rm"], pr["M"], pr["span"]
    ends = blastema_maps(rbgr, rm, span)
    a0, a1 = ends[0]["soft_area"], ends[1]["soft_area"]
    margin = abs(a0 - a1) / (a0 + a1 + 1e-9); i = 0 if a0 > a1 else 1; e = ends[i]
    H, W = rm.shape; Mi = cv2.invertAffineTransform(M)
    map_straight = project_map(e["probmap"], e["box"], rm.shape) if e["probmap"] is not None else np.zeros(rm.shape, np.float32)
    map_orig = cv2.warpAffine(map_straight, Mi, (W, H))
    mask_orig = ((map_orig >= BLASTEMA_PROB_MIN) & (pr["mask"] > 0)).astype(np.uint8) * 255
    return dict(ok=True, ends=ends, margin=float(margin), blastema_end=int(i), decides=bool(margin >= BLASTEMA_MARGIN),
                proportion=float((mask_orig > 0).sum()) / max(1.0, float(pr["mask"].sum())),
                blastema_mask_original=mask_orig, probmap_original=map_orig, M=M)


def _cli():
    ap = argparse.ArgumentParser(description="Blastema detection at both ends of the worm")
    ap.add_argument("image"); ap.add_argument("mask"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = detect_blastema(load_image(a.image), load_mask(a.mask))
    if not r["ok"]: print("failure:", r["reason"]); return
    print("soft areas top %.1f bottom %.1f | margin %.3f (%s BLASTEMA_MARGIN %.2f) | blastema end: %s | worm proportion %.4f"
          % (r["ends"][0]["soft_area"], r["ends"][1]["soft_area"], r["margin"], ">=" if r["decides"] else "<", BLASTEMA_MARGIN,
             "top" if r["blastema_end"] == 0 else "bottom", r["proportion"]))
    if a.out:
        os.makedirs(a.out, exist_ok=True); stem = os.path.splitext(os.path.basename(a.image))[0]
        cv2.imwrite(os.path.join(a.out, stem + "_blastema_mask.png"), r["blastema_mask_original"])
        cv2.imwrite(os.path.join(a.out, stem + "_blastema_probmap.png"), (255 * np.clip(r["probmap_original"], 0, 1)).astype(np.uint8))
        for k, e in enumerate(r["ends"]):
            if e["probmap"] is not None:
                nm = "top" if k == 0 else "bottom"
                cv2.imwrite(os.path.join(a.out, "%s_blastema_thumbnail_%s.png" % (stem, nm)), e["thumbnail"])
                cv2.imwrite(os.path.join(a.out, "%s_blastema_probmap_%s.png" % (stem, nm)), (255 * e["probmap"]).astype(np.uint8))
        with open(os.path.join(a.out, stem + "_blastema.json"), "w", encoding="utf-8") as f:
            json.dump(dict(image=a.image, soft_area=[e["soft_area"] for e in r["ends"]], binary_area_px=[e["binary_area_px"] for e in r["ends"]],
                           margin=r["margin"], decides=r["decides"], blastema_end=r["blastema_end"], proportion=r["proportion"]), f, indent=1)
        print("written to", a.out)


if __name__ == "__main__": _cli()
