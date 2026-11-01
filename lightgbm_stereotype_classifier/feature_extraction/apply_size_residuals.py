# -*- coding: utf-8 -*-
# Applies to a NEW base metrics table (<split>_mask3_metrics.csv) the size normalization used during training.
# For a few columns related to worm size, the value was residualized against log(mask area + 1) with coefficients
# fitted on the TRAIN split (size_normalize_metrics.py). This script applies exactly the coefficients stored in
# size_residual_coefs.csv, without re-estimating them:
#     corrected_value = value - (b0 + b1 * log(area + 1))
# Inputs: the metrics table (column image = file name of the frame) and the folder of binary masks.
# Output: the same table, with the affected columns corrected (semicolon separator).
# Usage: python apply_size_residuals.py metrics.csv masks_dir out.csv
# Reproducibility: deterministic, semicolon CSV separator, CPU.
import os, sys, glob
import numpy as np
import pandas as pd
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def apply(df, mask_dir, coefs_path=os.path.join(HERE, 'size_residual_coefs.csv')):
    coefs = pd.read_csv(coefs_path, sep=';')
    idx = {os.path.basename(p): p for p in glob.glob(os.path.join(mask_dir, '**', '*.png'), recursive=True)}
    ls = []
    for im in df['image'].astype(str):
        p = idx.get(os.path.basename(im))
        m = cv2.imdecode(np.fromfile(p, np.uint8), 0) if p else None
        ls.append(np.log(float((m > 0).sum()) + 1) if m is not None else np.nan)
    ls = np.array(ls)
    for _, r in coefs.iterrows():
        c = r['col']
        if c in df.columns:
            pred = r['b0'] + r['b1'] * ls
            df[c] = df[c].values - np.where(np.isnan(pred), 0.0, pred)
    return df


if __name__ == '__main__':
    src, masks, out = sys.argv[1:4]
    d = apply(pd.read_csv(src, sep=';'), masks)
    d.to_csv(out, sep=';', index=False)
    print('written', out)
