# -*- coding: utf-8 -*-
"""Eye-cup (ocellus) detection: heat map and count, from the original image and the mask.

Chain (identical to training and to the cascade):
  1. worm mask and straightening (common/preprocessing.py) ;
  2. centres of the two terminal bands (TERMINAL_BAND_POSITION x length from each tip) ;
  3. for each end, a 384 square thumbnail of half-side EYES_FRACTION x length, background zeroed ;
  4. four channels: RGB / 255 and local contrast z = (local mean - grey) / local std on a 21 px window,
     brought to [0, 1] by (z + 1) / 6 ;
  5. U-Net EfficientNet-B0 (models.csv: eye_detector), sigmoid output = heat map ;
  6. peaks = local maxima (window 2 x min_peak_distance + 1) of height >= counting_threshold ;
  7. the head end is the one chosen by the cascade (head/head_detection.py) when requested, otherwise the
     end with the highest maximum response (deployment convention of the metrics chain).
The count is not capped: 1 eye cup (asymmetric or emerging) and 3 (ectopic) are legitimate.

Thresholds (parameters/eye_thresholds.csv): counting_threshold (minimum counting error on the frozen
validation split) to count, head_decision_threshold for the cascade to decide the orientation. Both are
editable through the corrected_value column.

References: U-Net, O. Ronneberger, P. Fischer, T. Brox, MICCAI 2015, DOI 10.1007/978-3-319-24574-4_28 ;
EfficientNet, M. Tan, Q. Le, ICML 2019, arXiv 1905.11946.

Command line:
    python eyes/eye_detection.py image.jpg mask.png [--out folder] [--head]
Module:
    from eye_detection import detect_eyes
    r = detect_eyes(bgr, mask)   # bgr uint8 OpenCV, mask 0/255 or 0/1
"""
import os, sys, json, argparse, numpy as np, cv2, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "common"), os.path.join(_HERE, "..", "parameters")):
    if _p not in sys.path: sys.path.insert(0, _p)
import anchor_cascade as AC
from preprocessing import prepare, load_image, load_mask, project_map, to_original
from parameter_reader import read

_PT = read("eye_thresholds")
COUNTING_THRESHOLD = float(_PT["counting_threshold"]); MINDIST = int(_PT["min_peak_distance"])


def _peaks(hm, thr):
    """Local maxima of the map above the threshold, sorted by decreasing height, no cap."""
    t = torch.from_numpy(hm)[None, None]
    mx = torch.nn.functional.max_pool2d(t, 2 * MINDIST + 1, 1, MINDIST)[0, 0].numpy()
    ys, xs = np.nonzero((hm == mx) & (hm >= thr))
    return sorted(zip(xs.tolist(), ys.tolist(), hm[ys, xs].tolist()), key=lambda q: -q[2])


def eye_maps(rbgr, rm, span, threshold=None):
    """Heat maps of the two ends of the straightened worm. Returns a list of two dicts (top, bottom):
    centre (cx, cy), thumbnail, heatmap (384 x 384 in [0, 1]), box, peaks [(x, y, height)] in the thumbnail,
    peaks_straight [(x, y, height)] in the straightened frame, max_response, mass (sum of the values >= threshold)."""
    thr = COUNTING_THRESHOLD if threshold is None else float(threshold)
    _, mdl = AC.load_models(verbose=False)
    pts, _ = AC.ends(rm); out = []
    for cx, cy in pts:
        v = AC.thumbnail(rbgr, rm, cx, cy, span, AC.F_EYE)
        if v is None: out.append(dict(centre=(cx, cy), thumbnail=None, heatmap=None, box=None, peaks=[], peaks_straight=[], max_response=0.0, mass=0.0)); continue
        p = AC._pred(mdl, v[0]); pk = _peaks(p, thr)
        ps = [AC.to_straight_frame(px, py, v[1]) + (h,) for px, py, h in pk]
        out.append(dict(centre=(int(cx), int(cy)), thumbnail=v[0], heatmap=p, box=v[1], peaks=pk, peaks_straight=ps,
                        max_response=float(p.max()), mass=float(p[p >= thr].sum())))
    return out


def detect_eyes(bgr, raw_mask, threshold=None, use_cascade=False):
    """Full detection on an original image and its Mask2Former mask.

    Returns dict:
      ok, reason ; ends : the two dicts of eye_maps ; head_end : 0 (top) or 1 (bottom) ; end_criterion :
      "cascade" or "max_response" ; eye_count : number of peaks of the head end ; eyes : [(x, y, height)] in the
      ORIGINAL IMAGE frame ; heatmap_original : heat map of the head end put back into the original image frame
      (float32 [0, 1]) ; M : straightening matrix ; threshold.
    """
    pr = prepare(bgr, raw_mask)
    if not pr["ok"]: return dict(ok=False, reason=pr["reason"], eye_count=0, eyes=[])
    rbgr, rm, M, span = pr["rbgr"], pr["rm"], pr["M"], pr["span"]
    ends = eye_maps(rbgr, rm, span, threshold)
    if use_cascade:
        _, _, step, conf, i = AC.anchor(rbgr, rm); criterion = "cascade (%s, confidence %.3f)" % (step, conf)
    else:
        i = 0 if ends[0]["max_response"] >= ends[1]["max_response"] else 1; criterion = "max_response"
    e = ends[i]
    map_straight = project_map(e["heatmap"], e["box"], rm.shape) if e["heatmap"] is not None else np.zeros(rm.shape, np.float32)
    Mi = cv2.invertAffineTransform(M); H, W = rm.shape
    map_orig = cv2.warpAffine(map_straight, Mi, (W, H))
    eyes = [to_original(x, y, M) + (h,) for x, y, h in e["peaks_straight"]]
    return dict(ok=True, ends=ends, head_end=int(i), end_criterion=criterion, eye_count=len(e["peaks"]),
                eyes=eyes, heatmap_original=map_orig, M=M, threshold=COUNTING_THRESHOLD if threshold is None else float(threshold))


def _cli():
    ap = argparse.ArgumentParser(description="Eye-cup heat map and count")
    ap.add_argument("image"); ap.add_argument("mask"); ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="overrides counting_threshold of eye_thresholds.csv")
    ap.add_argument("--head", action="store_true", help="choose the head end with the full cascade")
    a = ap.parse_args()
    r = detect_eyes(load_image(a.image), load_mask(a.mask), a.threshold, a.head)
    if not r["ok"]: print("failure:", r["reason"]); return
    print("head end: %s (%s) | eye cups counted: %d | threshold %.2f" % ("top" if r["head_end"] == 0 else "bottom", r["end_criterion"], r["eye_count"], r["threshold"]))
    for x, y, h in r["eyes"]: print("   eye cup x %.1f y %.1f height %.3f" % (x, y, h))
    if a.out:
        os.makedirs(a.out, exist_ok=True); stem = os.path.splitext(os.path.basename(a.image))[0]
        cv2.imwrite(os.path.join(a.out, stem + "_eyes_heatmap.png"), (255 * np.clip(r["heatmap_original"], 0, 1)).astype(np.uint8))
        for k, e in enumerate(r["ends"]):
            if e["heatmap"] is not None:
                nm = "top" if k == 0 else "bottom"
                cv2.imwrite(os.path.join(a.out, "%s_eyes_thumbnail_%s.png" % (stem, nm)), e["thumbnail"])
                cv2.imwrite(os.path.join(a.out, "%s_eyes_heatmap_%s.png" % (stem, nm)), (255 * e["heatmap"]).astype(np.uint8))
        with open(os.path.join(a.out, stem + "_eyes.json"), "w", encoding="utf-8") as f:
            json.dump(dict(image=a.image, head_end=r["head_end"], criterion=r["end_criterion"], threshold=r["threshold"], eye_count=r["eye_count"],
                           eyes=[dict(x=x, y=y, height=h) for x, y, h in r["eyes"]], max_response=[e["max_response"] for e in r["ends"]]), f, indent=1)
        print("written to", a.out)


if __name__ == "__main__": _cli()
