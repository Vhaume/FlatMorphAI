# -*- coding: utf-8 -*-
# Single orchestrator for the signature metrics added to the final model.
#
# These three families are not part of the base metrics and are computed from the worm images and masks.
# They are all reproducible (random_state 42, semicolon CSV separator) and write into new_metrics_out:
#   patch_dorsal_features.py     -> patchint_<split>.csv   (dark dorsal patch, per frame)
#   worm_curvature_features.py   -> curv_<split>.csv       (curvature evolution, at the worm level)
#   worm_sharpcurve_features.py  -> nsharp_<split>.csv     (number of sharp bends, at the worm level)
#
# The temporal family (ecc_osc, ecc_chg, circ_chg, wl_osc, elong_osc, fold_chg) is already produced by the base
# metrics pipeline and nothing needs to be recomputed here.
#
# Important reminder. The curv and nsharp families are at the WORM level, not the frame level: they aggregate over the
# whole frame sequence of a given worm. Classifying a worm in production therefore requires its complete sequence, not
# a single isolated image.
#
# Usage:
#   python generate_signature_features.py                 # all splits (train valid test golden)
#   python generate_signature_features.py valid test      # a subset of splits
import os, sys
import patch_dorsal_features as patch
import worm_curvature_features as curv
import worm_sharpcurve_features as nsharp


def main(splits):
    print('[1/3] dorsal patch (patchint) ...')
    img_index = patch._index(patch.IMAGES)
    msk_index = patch._index(patch.MASKS)
    for sp in splits:
        patch.run_split(sp, img_index, msk_index)

    print('[2/3] worm-level curvature (curv) ...')
    mask_index = curv._index(curv.MASKS)
    worm_frames = curv.build_worm_frames(mask_index)
    cache = {}
    for sp in splits:
        curv.run_split(sp, worm_frames, mask_index, cache)

    print('[3/3] number of sharp bends (nsharp) ...')
    mask_index = nsharp._index(nsharp.MASKS)
    worm_frames = nsharp.build_worm_frames(mask_index)
    cache = {}
    for sp in splits:
        nsharp.run_split(sp, worm_frames, mask_index, cache)

    print('Done. The signature CSV files are up to date in new_metrics_out.')


if __name__ == '__main__':
    splits = sys.argv[1:] or [s for s in os.environ.get('FLATMORPH_SPLITS', 'train,valid,test').split(',') if s]
    main(splits)
