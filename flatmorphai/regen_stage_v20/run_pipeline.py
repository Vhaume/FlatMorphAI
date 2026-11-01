# -*- coding: utf-8 -*-
"""Entry point: regeneration state of one image or of a folder of images, from the original photograph.

    python run_pipeline.py image.jpg --out results
    python run_pipeline.py folder_of_images --out results
    python run_pipeline.py folder_of_images --masks folder_of_mask2former_masks --out results   (skips Mask2Former)
    python run_pipeline.py folder_of_images --annotations folder_of_cyan_annotated_copies --out results

For every image, written in --out: <stem>_mask_m2f.png (step 1), <stem>_mask_refined.png (step 2), <stem>_composite.png
(step 5), <stem>_head.jpg (control overlay: Mask2Former contour blue, refined contour green, tip and thumbnail centre
red, head point yellow, deciding step and state), <stem>.json (all records). For the whole run: summary_<date>.csv
(one line per image: state, class, uncertainty, cascade step, tip, tip correction, worm choice flag a_verifier,
refinement counters) and descriptors_<date>.csv (the 128 descriptors), separator ";".
Masks and annotations are matched by file name (same name as the image, any of .jpg .jpeg .png).
A GPU is required (CUDA or ROCm), REGEN_ALLOW_CPU=1 allows a slow CPU run. Seed 42.
"""
import os, sys, glob, argparse, datetime, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _match(folder, name):
    if not folder: return None
    st = os.path.splitext(name)[0]
    for e in ("", ".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = os.path.join(folder, name if e == "" else st + e)
        if os.path.isfile(p): return p
    hits = glob.glob(os.path.join(folder, "**", st + ".*"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="Planarian regeneration state, chain v20")
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default=None, help="folder of Mask2Former masks (grey 0..255), skips step 1")
    ap.add_argument("--annotations", default=None, help="folder of expert copies with the blastema drawn in cyan")
    a = ap.parse_args()
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if f.lower().endswith(EXT))
        else: files.append(p)
    from regen_pipeline import RegenerationPipeline
    pipe = RegenerationPipeline(segmentation=a.masks is None)
    os.makedirs(a.out, exist_ok=True); rows, drows = [], []; t0 = time.time()
    for i, f in enumerate(files, 1):
        nm = os.path.basename(f)
        try:
            res = pipe.analyse_file(f, out_dir=a.out, mask_path=_match(a.masks, nm), annotation_path=_match(a.annotations, nm))
        except Exception as e:
            print("  ERROR %s: %s" % (nm, e)); rows.append(dict(image=nm, status="error: %s" % e)); continue
        r = dict(image=nm, status=res.get("status"))
        r.update({k: v for k, v in (res.get("stage") or {}).items()})
        r.update({k: v for k, v in (res.get("head") or {}).items()})
        r.update({"worm_" + k: v for k, v in (res.get("segmentation") or {}).items() if k in ("n_comp", "n_cand", "accord", "rapport_circ", "a_verifier", "change")})
        r.update({"refine_" + k: v for k, v in (res.get("refinement") or {}).items()})
        r.update({k: v for k, v in (res.get("composite_info") or {}).items()})
        rows.append(r)
        if res.get("descriptors"): drows.append(dict(image=nm, **res["descriptors"]))
        sg = res.get("stage") or {}
        print("[%d/%d] %s | %s | step %s | state %s" % (i, len(files), nm, res.get("status"), (res.get("head") or {}).get("stage"),
              "%.2f -> %d" % (sg["stage_decimal"], sg["stage_class"]) if sg else "-"))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "summary_%s.csv" % stamp), sep=";", index=False)
    if drows: pd.DataFrame(drows).to_csv(os.path.join(a.out, "descriptors_%s.csv" % stamp), sep=";", index=False)
    print("%d images in %.0f s, results in %s" % (len(files), time.time() - t0, os.path.abspath(a.out)))


if __name__ == "__main__":
    main()
