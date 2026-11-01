# -*- coding: utf-8 -*-
# Inference of the planarian stereotype classifier (final model: LightGBM #1 trained after Borderline-SMOTE,
# Walnut expert LightGBM #2 and arbitration).
#
# This module is self-contained for the CLASSIFICATION step. It loads the two trained LightGBM models
# (multiclass baseline + binary Walnut-versus-C-shape specialist), the arbitration threshold and the column
# configuration, then predicts the stereotype from a table of precomputed descriptors.
#
# Full pipeline: image + mask  ->  descriptors (see feature_extraction/ and the README)  ->  this module.
#
# The classifier works at the WORM level. Each row of the input table is a frame described by its
# descriptors. The dynamics descriptors (temporal, curv, nsharp families) are already worm-level aggregates,
# so they are identical for all frames of the same worm. The prediction is returned per frame.
#
# Final model: unified filter (MI >= 0.02 and Pearson |r| < 0.90) over all descriptors of both models, multiclass
# baseline on 285 descriptors, Walnut-versus-C-shape expert on 29 descriptors (13 shared with the baseline, 16 specific),
# arbitration threshold 0.85 chosen on the validation set, applied only to the expert probability. Hyperparameters chosen
# by a sweep on the validation set (tools/sweep_lgbm_cv.ipynb). Class imbalance compensated by Borderline-SMOTE
# on the training set (tools/modele_final_borderline_lgbm.ipynb). Deterministic training, seed 42.
# No GPU dependency. CSV separator ';'.
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))


class StereotypeClassifier:
    """Loads the models and predicts the stereotype (C-shape, Screw-like, Snake-like, Walnut, Wildtype)."""

    def __init__(self, root=None):
        root = root or HERE
        self.cfg = json.load(open(os.path.join(root, 'config', 'feature_columns.json'), encoding='utf-8'))
        self.classes = self.cfg['classes']
        self.W = self.cfg['walnut_index']
        self.Csh = self.cfg['cshape_index']
        self.base_cols = self.cfg['baseline_feature_order']
        self.sig_cols = self.cfg['specialist_feature_order']
        self.thr = float(self.cfg['arbitration']['threshold'])
        self.baseline = lgb.Booster(model_file=os.path.join(root, 'models', 'baseline_multiclass_lgbm.txt'))
        self.specialist = lgb.Booster(model_file=os.path.join(root, 'models', 'specialist_binary_lgbm.txt'))

    def _matrix(self, df, cols):
        # missing columns -> NaN, handled natively by LightGBM
        return pd.DataFrame({c: (df[c] if c in df.columns else np.nan) for c in cols}).values.astype(np.float32)

    def predict_proba_baseline(self, df):
        p = self.baseline.predict(self._matrix(df, self.base_cols))
        return p if p.ndim == 2 else p.reshape(len(df), -1)

    def predict_proba_specialist(self, df):
        p = self.specialist.predict(self._matrix(df, self.sig_cols))
        return p.ravel()  # probability of the Walnut class

    def _arbitrate(self, base_pred, spec_walnut_proba):
        # a worm predicted C-shape by the baseline becomes Walnut if the expert Walnut probability reaches the threshold
        out = base_pred.copy()
        out[(base_pred == self.Csh) & (spec_walnut_proba >= self.thr)] = self.W
        return out

    def predict(self, df, apply_specialist=True, return_labels=True):
        """df: DataFrame of descriptors (one row per frame).
        Returns the predictions (text labels if return_labels, otherwise indices)."""
        base_proba = self.predict_proba_baseline(df)
        base_pred = base_proba.argmax(1)
        if apply_specialist:
            pred = self._arbitrate(base_pred, self.predict_proba_specialist(df))
        else:
            pred = base_pred
        if return_labels:
            return np.array([self.classes[i] for i in pred])
        return pred

    def predict_from_csv(self, features_csv, sep=';'):
        df = pd.read_csv(features_csv, sep=sep)
        return self.predict(df)


if __name__ == '__main__':
    import sys
    clf = StereotypeClassifier()
    print('Classes            :', clf.classes)
    print('Baseline features  :', len(clf.base_cols))
    print('Signature features :', len(clf.sig_cols))
    print('Arbitration thresh :', clf.thr, '(chosen on the validation set only)')
    if len(sys.argv) > 1:
        preds = clf.predict_from_csv(sys.argv[1])
        print('Predictions        :', preds[:20], '...' if len(preds) > 20 else '')
