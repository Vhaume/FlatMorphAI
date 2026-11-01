# LightGBM planarian stereotype classifier (final model: Borderline-SMOTE + Walnut expert)

> **Status in the FlatMorphAI repository.** This is the internal reference documentation of the stereotype
> classifier, kept for its description of every descriptor and of the model. The application uses this folder
> through `flatmorphai/notebook_api/lightgbm_bundle.py`; you do not run anything here yourself. Install only the
> root `requirements.txt` of the repository. The development notebooks (`tools/`) and the retraining folder
> (`retraining/`) mentioned below are not part of the repository.

Self-contained package that classifies the locomotor stereotype of a segmented planarian frame into one of five classes:
**C-shape, Screw-like, Snake-like, Walnut, Wildtype**.

This README is written so that a person or a program can find, for every file, what it contains, how the input
descriptors are computed, and how the model is called. Machine-readable metadata lives in `config/`.

| Item | Value |
|---|---|
| Model | LightGBM #1 (multiclass, 285 descriptors, trained after Borderline-SMOTE over-sampling) + LightGBM #2 (binary Walnut vs C-shape expert, 29 descriptors) + arbitration |
| Descriptors used | 301 distinct (285 base, 29 expert, 13 shared) |
| Balanced accuracy (seed 42) | validation 94.8 %, external test 96.0 % |
| Library | `lightgbm` 4.7.0 (pin it), `numpy`, `pandas` |
| Hardware | CPU only for classification. Segmentation (upstream) uses a GPU |
| CSV separator | `;` everywhere |

---

## 1. File map

```
lightgbm_stereotype_classifier/
  README.md                          this file
  requirements.txt                   Python packages (lightgbm pinned to 4.7.0)
  classifier.py                      inference API: StereotypeClassifier (load models, predict from a descriptor table)
  config/
    feature_columns.json             classes, descriptor order of each model, hyperparameters, arbitration rule, filter
    descriptor_provenance.csv        one row per descriptor (301): family, model(s), source table, script, inputs, status
    descriptor_families.csv          one row per descriptor: family used in the article figure (shape, motion, ...)
    descriptor_reference.csv         one row per descriptor (301): description, temporal scope, method reference and DOI
    descriptor_reference.md          the same reference as a readable document, grouped by family
  models/
    baseline_multiclass_lgbm.txt     LightGBM #1, native text format, 350 trees per class (trained after Borderline-SMOTE)
    specialist_binary_lgbm.txt       LightGBM #2, native text format, 400 trees
    arbitration_threshold.txt        arbitration_threshold=0.85
    metric_keep.txt                  the 285 descriptors of LightGBM #1, one per line
  data/
    features_train.csv               4247 frames   (columns: image, label, then the 301 descriptors)
    features_valid.csv               250 frames
    features_test.csv                150 frames (external test)
    features_train_borderline_synthetic_seed42.csv   2328 synthetic training examples of the deployed model (Borderline-SMOTE,
                                     seed 42; columns synthetic_id, label, then the 285 descriptors of LightGBM #1)
  feature_extraction/                scripts that compute the 301 descriptors (section 3), only the ones still used
  retraining/                        self-contained folder to retrain both LightGBM models (section 9): data, config,
                                     train_lightgbm.ipynb and output/ (the current model)
  tools/
    sweep_lgbm_cv.ipynb              hyperparameter sweep of both models, selection on validation
    modele_final_e4_lgbm.ipynb       descriptor filter (301 descriptors) and previous class-weighted model
    modele_final_borderline_lgbm.ipynb   development version of the final training (project paths), see retraining/ for the
                                     self-contained version
    oversampling_multiseed_comparison.ipynb  comparison of over-sampling methods over 6 seeds (model choice)
    sweep_borderline_lgbm.ipynb      extended random search (300 configurations) after Borderline-SMOTE: no change retained
```

Key names in `config/feature_columns.json`:
`classes`, `baseline_feature_order` (285 names, column order expected by LightGBM #1), `specialist_feature_order`
(29 names, order expected by LightGBM #2), `arbitration.threshold` (0.85), `baseline_lgbm_params`,
`specialist_lgbm_params`, `unified_filter` (how the descriptors were filtered).

Columns of `config/descriptor_provenance.csv`:
`descriptor; family; base_model; walnut_expert; source_table; script; modules; inputs; size_residualised; status`.

---

## 2. How the model decides

```
descriptor table (one row per frame)
   |-- 285 columns --> LightGBM #1 (multiclass) --> provisional class (argmax)
   |-- 29 columns  --> LightGBM #2 (binary)     --> P(Walnut)
   v
if provisional class == C-shape and P(Walnut) >= 0.85 : final class = Walnut
else                                                   : final class = provisional class
```

- Missing values (NaN) are allowed and handled natively by the trees.
- No normalisation is applied before the models. Descriptors are used raw.
- LightGBM #1: 31 leaves, depth 8, learning rate 0.05, L2 3.0, at least 40 samples per leaf, column subsample 0.9,
  trained without class weights on the training partition over-sampled by Borderline-SMOTE (6575 examples),
  350 trees (number of trees selected on validation among 100 to 800).
- LightGBM #2: 15 leaves, depth 4, learning rate 0.03, L2 5.0, 400 trees, balanced classes, trained on Walnut and
  C-shape frames only.
- Hyperparameters of both models were selected by the sweep in `tools/sweep_lgbm_cv.ipynb` on the validation set (with class
  weighting), then kept for the training after Borderline-SMOTE, whose number of trees (350) and arbitration threshold
  (0.85) were selected again on the validation set.

---

## 3. Computing the descriptors

The 301 descriptors come from several source tables. `config/descriptor_provenance.csv` gives, for each descriptor,
the table and the script that produces it. Every descriptor now has a generator script in `feature_extraction/`.

### 3.0 Installation and folder layout

```
pip install -r requirements.txt          # + FlatMorphAI (segmentation) and a CUDA or ROCm build of torch
```

The extraction scripts work on a dataset folder with this layout (same as the training environment):

```
<FLATMORPH_DATASET>/                         a folder named dataset_training_stereotype
  images/<any sub-folder>/<video>_Frame_<n>_in_<N>.png        raw RGB frames (all frames of each worm)
  mask_model_Mask2Former_20260324_0451/<same file name>.png   binary masks (written by step 0 or step 3)
  dataset_images_list/<PRESET>/<PRESET>_<split>.csv           frame list of each split, separator ;
                                                              columns: row_index;class;image_path
```

For new data, `class` can be any placeholder (for example Undefined), it is not used by the descriptors.
The scripts read these environment variables (defaults = training environment):

| Variable | Meaning | Default |
|---|---|---|
| `FLATMORPH_DATASET` | the `dataset_training_stereotype` folder above | training path |
| `FLATMORPH_PRESET` | name of the split preset in `dataset_images_list/` | `dataset_repaired_wormsplit_relabel_v2_test30_chook_cdense8` |
| `FLATMORPH_SPLITS` | comma-separated split names to process | `train,valid,test` |
| `FLATMORPH_RAW_FRAMES` | extra root searched for raw frames and their neighbours | training path |
| `FLATMORPH_SEG_CKPT` | folder that contains the checkpoint `model_Mask2Former_20260324_0451` | `checkpoints_segmentation_model` |

Example for a new video decomposed into a split called `predict`:

```
set FLATMORPH_DATASET=D:\data\dataset_training_stereotype      (Linux/WSL: export FLATMORPH_DATASET=...)
set FLATMORPH_PRESET=my_new_videos
set FLATMORPH_SPLITS=predict
```

Run the scripts from the `feature_extraction/` folder. Tables of steps 9 and 10 are written to
`feature_extraction/new_metrics_out/`, luminosity tables to `analysis_out/` of the working folder.

### 3.1 Inputs needed per frame

A frame is identified by its file name `<video>_Frame_<n>_in_<N>.png`. The descriptors need:

| Input | Produced by |
|---|---|
| raw RGB frame (T0), its neighbours T-1 and T+1, and a wider window (+/-2) for a few groups | video decomposition |
| binary worm mask of every frame | Mask2Former segmentation (FlatMorphAI application, `flatmorphai.notebook_api.segmentation`, checkpoint `model_Mask2Former_20260324_0451`) |
| mask-constrained 384 px image | built on the fly by `precompute_mask3_metrics.py` (FlatMorphAI `classification_images`) and by `format_dataset_size.py` |
| temporal composite 384 px (T-1, T0, T+1 in the channels) | `gen_temporal_dataset.py` |
| registered motion image (aligned previous, current, next) | `make_motion_dataset.py` |
| all frames of the same worm (worm-level descriptors) | the same video |

Frame window rule: interior frames use (n-1, n, n+1). The first frame uses (n, n+1, n+2) and the last frame
(n-2, n-1, n), so that the window always spans three real frames.

### 3.2 Steps, in order

| Step | Script (in `feature_extraction/`) | Output | Descriptors |
|---|---|---|---|
| 0 | Mask2Former segmentation (FlatMorphAI) | masks | none |
| 1 | `gen_temporal_dataset.py` | temporal composites | fallback input of the motion groups when a registered motion image is missing |
| 2 | `make_motion_dataset.py` (uses `format_dataset_size.py`) | motion images | inputs of 12 descriptors |
| 3 | `precompute_mask3_metrics.py` (export version, only the 26 metric groups still used) | `<split>_mask3_metrics.csv` | 260 |
| 4 | `precompute_geodesic_width.py` (uses `geodesic_width.py`) | adds `gw_*` to the same table | 4 |
| 5 | `apply_size_residuals.py` with `size_residual_coefs.csv` | size-corrected mask3 table | corrects 14 of them |
| 6 | `precompute_luminosity.py` (uses `centerline_luminosity_features.py`) | `luminosity_<split>.csv` | 5 |
| 7 | `compute_new_metrics.py` | `new_metrics_<split>.csv` | 16 |
| 8 | `compute_shape_lum_metrics.py` | `new_metrics_out/shape_lum_extra_<split>.csv` (read directly by step 11) | 1 |
| 9 | `compute_expert_image_features.py --images <FLATMORPH_DATASET>/images --masks <FLATMORPH_DATASET>/mask_model_Mask2Former_20260324_0451 --list <PRESET>_<split>.csv --split <split> --out new_metrics_out` | `pigment_`, `sat_`, `central_`, `valley_<split>.csv` | 7 |
| 10 | `generate_signature_features.py` (runs `patch_dorsal_features.py`, `worm_curvature_features.py`, `worm_sharpcurve_features.py`, which read the frame list of `valley_<split>.csv`, so step 9 must run first) | `patchint_`, `curv_`, `nsharp_<split>.csv` | 8 |
| 11 | `assemble_features.py` | the final table: image, label, 301 descriptors | all |

Step 5 is required for new data: 14 texture and curvature descriptors were residualised against log(mask area + 1)
with coefficients fitted on the training set (`size_normalize_metrics.py`, kept for reference). Apply the saved
coefficients, never re-fit them:
`python apply_size_residuals.py <split>_mask3_metrics.csv <mask_dir> <split>_mask3_metrics_sized.csv`.

Step 11:
```
python assemble_features.py --mask3 <split>_mask3_metrics_sized.csv --lum analysis_out/luminosity_<split>.csv \
       --newm new_metrics_out/new_metrics_<split>.csv --sigdir new_metrics_out --split <split> \
       --out features_<split>.csv
```
Check: running step 11 on the training tables reproduces `data/features_*.csv`, and the classifier then gives
94.8 % on validation and 96.0 % on test. Step 11 prints the name of every descriptor it could not find (left as NaN).

### 3.3 Note on the 7 image descriptors of the Walnut expert (step 9)

`pig_localstd`, `glcm_homogeneity`, `sat_std`, `sat_localstd`, `central_max`, `central_bright` and `valley_energy` are
used by the Walnut expert only. Their generator was lost and has been rewritten in
`compute_expert_image_features.py` from the exact code that produced the training tables. Check on 60 validation frames:
values identical to `data/features_valid.csv` (maximum difference 4e-16) for every frame whose image was available at
training time. In `data/`, 3 to 6 % of these values are NaN because the raw frame was missing when the tables were built
(frames recovered later), the rewritten script computes them now. The models handle NaN natively.
End-to-end check of steps 9 and 10 with the environment variables of 3.0, on 10 test frames: the 15 expert columns
produced (pigment, saturation, central luminance, valley, dorsal patch, curvature evolution, sharp bends) are identical
to `data/features_test.csv`, except one frame whose raw image was missing at training time.

### 3.4 Order of execution and devices

- Steps 0 and 3 segment the worm with Mask2Former and need a GPU: `precompute_mask3_metrics.py` stops with an explicit
  error if CUDA/ROCm is not visible to torch (`REQUIRE_GPU = True`), instead of silently running on the CPU. It prints
  `SEG DEVICE: cuda` when the GPU is used. Masks already on disk are reused.
- All other steps and the classifier run on the CPU (LightGBM has no GPU code path here).
- Worm-level descriptors (`worm_*`, step 10, and the sequence terms of step 7) need every frame of the worm in the split.
- The golden-set blocks still present in some scripts are optional and skipped when their files are absent.

---

## 4. Using the model

```python
import pandas as pd
from classifier import StereotypeClassifier

clf = StereotypeClassifier()                                      # loads models, threshold and column orders
table = pd.read_csv("data/features_test.csv", sep=";")           # or the output of assemble_features.py
labels = clf.predict(table)                                       # one label per row (frame)
base_only = clf.predict(table, apply_specialist=False)            # multiclass model alone
proba = clf.predict_proba_baseline(table)                         # (n, 5) probabilities, order = clf.classes
p_walnut = clf.predict_proba_specialist(table)                    # (n,) expert probability of Walnut
```

- Columns are matched by name. Missing columns are created as NaN.
- Command line self-check: `python classifier.py data/features_test.csv`.
- Worm-level descriptors (`worm_*`, and the sequence terms of `compute_new_metrics.py`) need all frames of the worm to be
  processed before assembly.

---

## 5. Descriptor filter (why 301 descriptors)

On the training set, the descriptors of both models were filtered together: multi-class mutual information of at least
0.02, then single-linkage clustering of pairs with an absolute Pearson correlation of 0.90 or more, keeping the most
informative member of each cluster in both models. Three absolute gray-level run-length descriptors (`tr_RLN`,
`tr_GLN`, `tr_LRE`) were excluded beforehand because they depend on worm size and image resolution. Result: 301 distinct
descriptors, minimum mutual information 0.022, maximum absolute Pearson correlation 0.899. The filter code is in
`tools/modele_final_e4_lgbm.ipynb`. The clusters are the connected components of the graph |r| >= 0.90 (union-find),
which is exactly single-linkage hierarchical clustering on the distance 1 - |r| cut at 0.10 (checked against
`scipy.cluster.hierarchy.linkage(method='single')` + `fcluster(t=0.10, criterion='distance')`: same 301 clusters).
Mutual information: `sklearn.feature_selection.mutual_info_classif(n_neighbors=3, random_state=42)` after median
imputation. Pearson: `numpy.corrcoef`.

---

## 6. Performance (final model)

| split       | balanced accuracy (seed 42) | mean over 6 seeds |
|-------------|-----------------------------|-------------------|
| validation  | 94.8 %                      | 93.9 % +/- 0.7    |
| test (ext.) | 96.0 %                      | 95.1 % +/- 1.2    |

Test, per class: C-shape 30/30, Screw-like 28/30, Snake-like 27/30, Walnut 30/30, Wildtype 29/30.
Standard deviations use the n - 1 denominator. Previous class-weighted model (seed 42): validation 93.2 %, test 95.3 %.

---

## 7. Training data and class imbalance

- Training frames (`data/features_train.csv`, 4247 frames of 1384 worms): C-shape 880, Screw-like 1148, Snake-like
  1315, Walnut 176, Wildtype 728. Validation 250 frames (181 worms, 50 per class), test 150 frames (117 worms, 30 per
  class). No worm is shared between the three partitions.
- The training partition was densified: additional frames of videos of worms already in the training partition were
  added for the hard sub-types (tightly hooked C-shape, extended and immobile translucent Snake-like). They carry the
  label of their video, and the validation and test partitions keep their worms.
- Imbalance is handled by Borderline-SMOTE in LightGBM #1 (Han, Wang and Mao, ICIC 2005, DOI 10.1007/11538059_91,
  imbalanced-learn `BorderlineSMOTE`, kind borderline-1, k_neighbors 5, m_neighbors 10, random_state 42): every minority
  class is brought to the size of the largest class, synthetic examples are generated on the training partition only
  from median-imputed descriptors, real examples keep their missing values. LightGBM #2 keeps balanced class weights.
- Choice: ROS, SMOTE, Borderline-SMOTE, ADASYN, SMOTE restricted to Walnut and C-shape and SMOTE in the expert were
  compared with class weighting over 6 seeds (`tools/oversampling_multiseed_comparison.ipynb`). Borderline-SMOTE was
  retained because its validation gain was real (mean of the paired differences over six seeds positive and larger
  than their standard deviation) and it did not lower the
  validation or test balanced accuracy for any seed. The previous class-weighted models are kept in the project folder
  `analysis_out/previous_class_weighted_model/`.

---

## 8. Reproducibility

Training is deterministic (`deterministic=True`, `force_row_wise=True`, seed 42) and gives identical models across
machines with the exact versions of `requirements.txt` (reference environment: Python 3.10.12, numpy 2.2.6, scipy 1.15.3,
scikit-learn 1.7.2, imbalanced-learn 0.14.2, lightgbm 4.7.0, pandas 2.3.3). The Borderline-SMOTE examples depend on the
numpy and scipy versions (nearest neighbours): with Python 3.12, numpy 2.3.5, scipy 1.16.3 and lightgbm 4.6.0 they differ
and the retrained model gives 94.4 % / 95.3 % instead of 94.8 % / 96.0 %. The synthetic examples of the deployed model are
therefore provided in `data/features_train_borderline_synthetic_seed42.csv` and used by default, so that retraining does
not depend on the environment (`REGENERATE_SYNTH=1` regenerates them). Checked with `check_reproducibility_borderline.ipynb`
(project root): in the Python 3.12 / numpy 2.3.5 / scipy 1.16.3 / lightgbm 4.6.0 environment, retraining from this file
gives exactly 94.8 % / 96.0 % with 350 trees, so only the generation of the synthetic examples depends on the versions. To retrain the final model, run
`tools/modele_final_borderline_lgbm.ipynb`: it over-samples `data/features_train.csv` with Borderline-SMOTE, trains both
models, selects the number of trees and the threshold on validation, writes `models/` and `config/feature_columns.json`,
and checks that `classifier.py` gives 94.8 % on validation and 96.0 % on test (CPU only). Borderline-SMOTE: Han, Wang,
Mao, ICIC 2005 (DOI 10.1007/11538059_91). imbalanced-learn: Lemaitre, Nogueira, Aridas, JMLR 2017 (arXiv 1609.06570). Sources: LightGBM, Ke et al., NeurIPS 2017 (DOI 10.5555/3294996.3295074). mRMR selection,
Peng, Long, Ding, IEEE TPAMI 2005 (DOI 10.1109/TPAMI.2005.159). Mask2Former, Cheng et al., CVPR 2022 (arXiv 2112.01527).

---

## 9. Retraining the models (folder `retraining/`)

`retraining/` lets a user retrain the two LightGBM models without any other file of the project:

```
cd retraining
jupyter nbconvert --to notebook --execute train_lightgbm.ipynb
```

- `retraining/data/`: the descriptor tables (training, validation, test) and the 2328 Borderline-SMOTE examples of the
  deployed model. Metrics only, no images.
- `retraining/config/training_config.json`: descriptor order of each model, hyperparameters, Borderline-SMOTE settings,
  arbitration rule and seeds.
- `retraining/train_lightgbm.ipynb`: over-sampling, training of LightGBM #1 and LightGBM #2, selection of the number of
  trees and of the arbitration threshold on the validation set, evaluation, writing of the model.
- `retraining/output/`: the files written by the notebook. As shipped, they are the files of the current model, and a
  run with the reference data reproduces them exactly (validation 94.8 %, test 96.0 %, 350 trees, threshold 0.85).
  The notebook checks the new model against `models/` and can replace it (`DEPLOY_TO_PACKAGE = True`).

Details are in `retraining/README.md`. The descriptors are documented in `config/descriptor_reference.csv`.
