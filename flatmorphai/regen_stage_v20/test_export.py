# -*- coding: utf-8 -*-
"""Check of the export against the production outputs of the v20 chain, on the 14 images of test_data/ (2 to 3 per state,
every deciding step of the cascade represented, none annotated in cyan_blasteme so the model path of the refinement is used).

Levels, each one isolating a step by starting from the production input of that step:
  L1 state       production composite + production descriptors (v20 table)  -> P, compared with
                 predictions_modele_production_v20.csv (same checkpoint, so differences are numerical only)
  L2 head inputs production refined mask + production head (journal ancres_composite_v20.csv) -> composite and
                 descriptors, compared with the production composite PNG and the v20 descriptor table
  L3 cascade     production refined mask -> head cascade, compared with the journal (step, tip, centre, L)
  L4 refinement  production Mask2Former mask -> refined mask, compared with the production refined mask (IoU)
  L5 segmentation original image -> Mask2Former mask, compared with the production mask (IoU)   [--segmentation]
  L6r steps 3-7  production refined mask -> cascade, composite, descriptors, state, compared with the production prediction
  L6 end to end  from the image (--segmentation) or from the production Mask2Former mask -> state, compared with the
                 production prediction
Tolerances (fixed before the run, they absorb the GPU versus CPU and driver differences of the networks near a
threshold): L1 |dP| <= 0.01, L2 pixels of the composite differing by more than 10 grey levels <= 0.5 % and 95th
percentile of the relative descriptor difference <= 1 %, L3 same step and tip distance <= 0.02 L, L4 IoU >= 0.99,
L5 IoU >= 0.98, L6r |dP| <= 0.15, L6 same class and |dP| <= 0.15.
Known sensitivity (cloud check of 19 October 2026, CPU): the eye U-Net response on a faint eye can move from 0.06 to
0.55 when the image grey levels change by plus or minus 1, and a few tens of blastema pixels added or not by the
refinement (probability close to 0.40, contact rule) can move the end of the skeleton. On a GPU with the production
software versions these levels are expected to agree, a CHECK on a CPU or with other library versions points to
these borderline images, not to a code difference (L1 and L3 isolate the code). Results: test_export_results/test_export_<date>.csv (separator ";").

Usage:  python test_export.py                 (levels 1 to 4 and 6 from the production mask)
        python test_export.py --segmentation  (adds level 5 and runs level 6 from the image, needs Mask2Former)
A GPU is required (CUDA or ROCm), REGEN_ALLOW_CPU=1 allows a slow CPU run. Seed 42.
"""
import os, sys, argparse, datetime, time
import numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
from regen_pipeline import RegenerationPipeline
from regen_pipeline._env import imread_color, imread_gray, DEVICE

TD = os.path.join(HERE, "test_data")
TOL = dict(L1_dP=0.01, L2_px=0.005, L2_rel95=0.01, L3_tip=0.02, L4_iou=0.99, L5_iou=0.98, L6_dP=0.15)


def iou(a, b):
    a = a >= 128; b = b >= 128; u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def rel_diffs(d, ref, cols):
    out = []
    for k in cols:
        x, y = d.get(k, np.nan), ref[k]
        if x is None: x = np.nan
        if pd.isna(x) and pd.isna(y): continue
        if pd.isna(x) or pd.isna(y): out.append(np.inf); continue
        out.append(abs(float(x) - float(y)) / max(abs(float(y)), 1e-6))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--segmentation", action="store_true"); a = ap.parse_args()
    pipe = RegenerationPipeline(segmentation=a.segmentation)
    REF = pd.read_csv(os.path.join(TD, "reference_production.csv"), sep=";").set_index("image")
    XD = pd.read_csv(os.path.join(TD, "reference_descriptors_v20.csv"), sep=";").set_index("image")
    cols = list(XD.columns); sel = pipe.stage.cols
    rows = []; t0 = time.time()
    for nm, R in REF.iterrows():
        st = os.path.splitext(nm)[0]; row = dict(image=nm, classe=R.classe, prod_etape=R.prod_etape)
        bgr = imread_color(os.path.join(TD, "images", nm))
        m2f_p = imread_gray(os.path.join(TD, "masks_m2f_production", nm))
        ref_p = imread_gray(os.path.join(TD, "masks_refined_production", nm))
        comp_p = cv2.imread(os.path.join(TD, "composites_production", st + ".png"))
        # L1 state on production inputs
        s1 = pipe.stage.predict(comp_p, XD.loc[nm].to_dict())
        row.update(L1_P=s1["stage_decimal"], prod_P=R.prod_prediction, L1_dP=abs(s1["stage_decimal"] - R.prod_prediction))
        # L2 composite and descriptors with the production head
        r3p = pipe.cascade.anchor(bgr, ref_p)
        r3j = dict(r3p); r3j.update(tip=(R.prod_pointe_y, R.prod_pointe_x), centre=(R.prod_centre_y, R.prod_centre_x), L=R.prod_L)
        comp2, _ = pipe.builder.build(bgr, ref_p, r3j)
        dd = np.abs(comp2.astype(int) - comp_p.astype(int))[:, :, (0, 2)]
        d2, _ = pipe.descr.compute(bgr, ref_p, r3j, nm)
        rd = rel_diffs(d2, XD.loc[nm], cols); rds = rel_diffs(d2, XD.loc[nm], sel)
        row.update(L2_px_diff=float((dd > 10).mean()), L2_rel_median=float(np.median(rd)), L2_rel95=float(np.percentile(rd, 95)),
                   L2_rel95_selected=float(np.percentile(rds, 95)), L2_n_exact=int((rd < 1e-6).sum()), L2_n_cols=len(rd))
        # L3 cascade
        row.update(L3_etape=r3p["stage"] if r3p else "none",
                   L3_tip_dist_L=float(np.hypot(r3p["tip"][0] - R.prod_pointe_y, r3p["tip"][1] - R.prod_pointe_x) / R.prod_L) if r3p else np.nan,
                   L3_dL=float(r3p["L"] - R.prod_L) if r3p else np.nan)
        # L6r end to end from the production refined mask (steps 3 to 7 only)
        rr = pipe.analyse(bgr, mask_refined=ref_p, name=nm); sr = rr.get("stage") or {}
        row.update(L6r_P=sr.get("stage_decimal", np.nan), L6r_dP=abs(sr.get("stage_decimal", np.nan) - R.prod_prediction))
        # L4 refinement
        ref4, inf4 = pipe.refiner.refine(bgr, m2f_p)
        row.update(L4_iou=iou(ref4, ref_p), L4_px_diff=int(((ref4 >= 128) ^ (ref_p >= 128)).sum()))
        # L5 segmentation and L6 end to end
        if a.segmentation:
            m5, inf5 = pipe.segmenter.segment(imread_color(os.path.join(TD, "images", nm), segmentation=True))
            row.update(L5_iou=iou(m5, m2f_p), L5_a_verifier=inf5.get("a_verifier"))
            res = pipe.analyse(bgr, mask_m2f=m5, name=nm)
        else:
            res = pipe.analyse(bgr, mask_m2f=m2f_p, name=nm)
        sg = res.get("stage") or {}
        row.update(L6_P=sg.get("stage_decimal", np.nan), L6_class=sg.get("stage_class", -1), prod_class=int(R.prod_classe_predite),
                   L6_dP=abs(sg.get("stage_decimal", np.nan) - R.prod_prediction), true_stage_decimal=R.prod_stade_decimal_vrai)
        rows.append(row)
        print("%-40s L1 dP %.4f | L2 px %.4f rel95 %.4f | L3 %s/%s tip %.4f L | L4 IoU %.4f | L6r P %.2f | L6 P %.2f (prod %.2f)" % (
            nm, row["L1_dP"], row["L2_px_diff"], row["L2_rel95"], row["L3_etape"], R.prod_etape, row["L3_tip_dist_L"],
            row["L4_iou"], row["L6r_P"], row["L6_P"], R.prod_prediction) + (" | L5 IoU %.4f" % row["L5_iou"] if a.segmentation else ""))
    D = pd.DataFrame(rows)
    ok = {"L1": bool((D.L1_dP <= TOL["L1_dP"]).all()),
          "L2": bool((D.L2_px_diff <= TOL["L2_px"]).all() and (D.L2_rel95 <= TOL["L2_rel95"]).all()),
          "L3": bool((D.L3_etape == D.prod_etape).all() and (D.L3_tip_dist_L <= TOL["L3_tip"]).all()),
          "L4": bool((D.L4_iou >= TOL["L4_iou"]).all()),
          "L6r": bool((D.L6r_dP <= TOL["L6_dP"]).all()),
          "L6": bool((D.L6_class == D.prod_class).all() and (D.L6_dP <= TOL["L6_dP"]).all())}
    if a.segmentation: ok["L5"] = bool((D.L5_iou >= TOL["L5_iou"]).all())
    out = os.path.join(HERE, "test_export_results"); os.makedirs(out, exist_ok=True)
    f = os.path.join(out, "test_export_%s_%s.csv" % (DEVICE.type, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    D.to_csv(f, sep=";", index=False)
    print("\ndevice %s | %d images | %.0f s | results %s" % (DEVICE, len(D), time.time() - t0, f))
    for k, v in ok.items(): print("  %s %s" % (k, "PASS" if v else "CHECK (see the csv, tolerance %s)" % {kk: vv for kk, vv in TOL.items() if kk.startswith(k)}))
    print("EXPORT %s" % ("VALIDATED" if all(ok.values()) else "TO CHECK"))
    return D, ok


if __name__ == "__main__":
    main()
