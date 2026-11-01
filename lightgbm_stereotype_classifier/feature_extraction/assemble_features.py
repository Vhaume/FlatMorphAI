# -*- coding: utf-8 -*-
# Assembles the descriptor table expected by classifier.py from the files produced by the computation steps.
# One row per frame (key = image file name). Only the 301 descriptors in use are kept, in the order given by
# config/descriptor_provenance.csv.
#
# Inputs (semicolon separator, column image):
#   --mask3      <split>_mask3_metrics.csv       (precompute_mask3_metrics.py + precompute_geodesic_width.py,
#                                                 then apply_size_residuals.py)
#   --lum        luminosity_<split>.csv          (precompute_luminosity.py)
#   --newm       new_metrics_<split>.csv         (compute_new_metrics.py)
#   --sigdir     folder of the signature families: bilat_ (or shape_lum_extra_), patchint_, curv_, nsharp_, pigment_,
#                sat_, central_, valley_<split>.csv (feature_extraction/new_metrics_out by default in the scripts)
#   --split      split name used in the file names of --sigdir
#   --out        output table
# Usage: python assemble_features.py --mask3 m.csv --lum l.csv --newm n.csv --sigdir new_metrics_out --split test --out f.csv
# Reproducibility: deterministic, semicolon CSV separator.
import os, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROV = os.path.join(HERE, '..', 'config', 'descriptor_provenance.csv')


def _read(path):
    d = pd.read_csv(path, sep=';')
    d['image'] = [os.path.basename(str(x)) for x in d['image']]
    return d.drop_duplicates('image')


def assemble(mask3, lum, newm, sigdir, split):
    prov = pd.read_csv(PROV, sep=';')
    cols = prov['descriptor'].tolist()
    d = _read(mask3)
    extra = [lum, newm] + [os.path.join(sigdir, '%s_%s.csv' % (f, split))
                           for f in ('bilat', 'shape_lum_extra', 'patchint', 'curv', 'nsharp', 'pigment', 'sat', 'central', 'valley')]
    for p in extra:
        if p and os.path.isfile(p):
            e = _read(p)
            keep = [c for c in e.columns if c in cols and c not in d.columns]
            d = d.merge(e[['image'] + keep], on='image', how='left')
        elif not os.path.basename(p).startswith(('bilat_', 'shape_lum_extra_')):
            print('missing:', p)
    missing = [c for c in cols if c not in d.columns]
    if missing:
        print('WARNING, missing descriptors (NaN):', missing)
    for c in missing:
        d[c] = float('nan')
    front = [c for c in ('image', 'label', 'class') if c in d.columns]
    return d[front + cols]


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    for k in ('mask3', 'lum', 'newm', 'sigdir', 'split', 'out'):
        a.add_argument('--' + k, required=k in ('mask3', 'out'))
    x = a.parse_args()
    t = assemble(x.mask3, x.lum, x.newm, x.sigdir or '.', x.split or '')
    t.to_csv(x.out, sep=';', index=False)
    print('written', x.out, t.shape)
