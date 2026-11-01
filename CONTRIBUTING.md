# Contributing to FlatMorphAI

FlatMorphAI is a scientific instrument before it is a program: a change is
acceptable only if it leaves the published numbers where they are, or moves them
on purpose. This file says how to check that, and which parts of the code react
to edits that look harmless.

## The test suite is the two comparison scripts

There is no unit-test suite that could replace a full recomputation. The
reference tables ship in `reference_csv/`, and two scripts compare a fresh run
with them, value by value:

```bash
python scripts/test_csv_stereotypies.py compare     # 14 tables: descriptors, classes, periods
python scripts/test_csv_regeneration.py compare     # 4 tables: states, head, eyespots, blastema
```

Before a change, take a reference of your own; after it, run the chain again and
compare:

```bash
python scripts/test_csv_stereotypies.py redo        # saves the reference AND deletes the tables,
                                            # so the next run really recomputes them
# ... run Stereotype classification ...
python scripts/test_csv_stereotypies.py compare
```

`keep` saves a reference without deleting anything; use `redo` for the
stereotype chain, otherwise the next run finds the tables already there and the
comparison only compares files with themselves. The stereotype script refuses to
overwrite a reference with a smaller one (`--force` to override), because a
shrunken reference makes the next comparison look stronger than it is.

A change is finished when both comparisons are green, or when every difference
they report is understood and intended.

## Code that reacts to harmless-looking edits

Three mechanisms run code by **reading source text**, not by importing it.
Renaming a variable, reflowing a comment or "tidying" a block in the files they
read can silently change which lines execute — no error, different numbers.

* **`exec_slice`** (`flatmorphai/regen_stage_v20/regen_pipeline/_env.py`). The
  regeneration chain executes slices of the programs in
  `flatmorphai/regen_stage_v20/production/`, each slice delimited by two
  **string markers** — some of them comments. Change a marker and the slice
  boundary moves or the chain stops starting. Some slices are also patched with
  `_remplace`, which asserts that the text it replaces appears exactly *n* times.
* **`v3c_descriptor_patches.py`** (`flatmorphai/notebook_api/`) patches functions
  of `lightgbm_stereotype_classifier/feature_extraction/` by exact source-text
  replacement.
* **`_load_funcs_from_source`** and **`_load_script`**
  (`flatmorphai/notebook_api/lightgbm_bundle.py`,
  `flatmorphai/notebook_api/regeneration/descriptors.py`) execute the imports,
  definitions and **top-level assignments** of a module while skipping its other
  statements. An assignment that calls a function therefore still runs.

So, in `flatmorphai/regen_stage_v20/` and `lightgbm_stereotype_classifier/`:
do not rename identifiers, do not reformat, do not rewrite comments that sit on
slice boundaries, and replay both comparisons after any change. Some internal
names there are in French (`charger`, `squelette`, `extremites`, …), and two CSV
columns are named `stade` and `ver`; they are kept deliberately, because renaming
them breaks the data contract and the slices.

## Calculation-sensitive dependencies

`opencv-python`, `scikit-image`, `mahotas`, `numpy` and `torch` change descriptor
values from one version to the next. Do not change their pins in
`requirements.txt` without replaying both comparisons. `tools/make_environment.py`
rebuilds `requirements.txt` from the code and the installed versions;
`python tools/make_environment.py --check` reports any drift without writing.
Without `--check` it overwrites `requirements.txt` and `environment.yml`
with what it finds installed, dropping their hand-written notes: run it only
in the validated environment, and review the diff before committing.

## Static checks

```bash
python tools/scopecheck.py $(git ls-files '*.py')
python -m pyflakes flatmorphai lightgbm_stereotype_classifier
```

`scopecheck.py` takes the files to check as arguments; called without any, it
checks nothing and exits 0. Both tools currently report a handful of names in
code the application does not call — a notebook evaluation driver
(`predict_one_image_augmented` in the three copies of
`evaluate_test_images_v2.py`), the loader of a classifier the application no
longer uses (`load_model_name_and_class_to_change` in `datasets.py`), a dataset
helper (`custom_dataset.py`), and names that the slicing mechanisms above inject
at run time. None is reached by the shipped chains. A new report, on the other
hand, deserves attention.

## Publishing new weights

The weights travel as GitHub release assets, not in the repository:

1. `python tools/make_release_assets.py build` writes the four archives to
   `release_assets/` and prints their SHA-256;
2. create the release and upload the four files;
3. put the repository, the tag and the digests in the constants at the top of
   `scripts/download_models.py` — until they are filled in, it refuses to download
   anything it cannot verify;
4. `python scripts/download_models.py` on a machine without the weights.

## Style

English only, in code, comments, logs and interface. The operator sees the
chains by their names — *Stereotype classification*, *Image segmentation*,
*Forecast regeneration states*, *Eyespots counter* — never by an internal letter.
