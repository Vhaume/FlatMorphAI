"""
format_dataset_size.py  (NEW FILE - does not modify retro-compatible code)

Ensure a processed-image dataset exists at a chosen input size, generated with
the SAME formatter the pipeline uses (FlatMorphAI classification_images, the
function compute_mask_image_planarian_reshaped). It reuses the existing per-image
binary masks and the original full-size images, so no Mask2Former rerun is needed,
and writes to the same root as the 224 set with the canonical folder name:

    processed_images_full_size_image-size(N)_margin(0.00)_b&w(False)_apply-mask(True)_dataset

Only images referenced by the dataset CSVs are generated, mirroring their
<class>/<file>.png layout. Already-present images are skipped.
"""
import os
import glob
import numpy as np
import cv2
import pandas as pd


def folder_name(image_size, margin=0.00, apply_mask=True, bw=False):
    return ("processed_images_full_size_image-size(%d)_margin(%.2f)_b&w(%s)_apply-mask(%s)_dataset"
            % (image_size, margin, str(bw), str(apply_mask)))


def _find(roots, rel):
    bn = os.path.basename(str(rel))
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    for r in roots:                                  # fallback: search by basename
        if not os.path.isdir(r):
            continue
        hits = glob.glob(os.path.join(r, "**", bn), recursive=True)
        if hits:
            return hits[0]
    return None


def ensure_processed_dataset(training_root, csv_dir, dataset_name, image_size,
                             margin=0.00, apply_mask=True, bw=False,
                             images_roots=None, mask_roots=None, verbose=True):
    """Return the processed-images root for image_size, generating any missing
    images with the same formatter. images_roots / mask_roots can be overridden."""
    out_root = os.path.join(training_root, folder_name(image_size, margin, apply_mask, bw))
    rels = set()
    for f in glob.glob(os.path.join(csv_dir, dataset_name, "*")):
        if f.endswith(".csv"):
            df = pd.read_csv(f, sep=";")
            if "image_path" not in df.columns:
                continue          # skip non-split CSVs (e.g. the precomputed mask3-metric tables)
            rels.update(df["image_path"].astype(str).tolist())
    missing = [r for r in rels if not os.path.exists(os.path.join(out_root, r))]
    if verbose:
        print("size %d -> %s" % (image_size, out_root))
        print("  referenced %d images, missing %d" % (len(rels), len(missing)))
    if not missing:
        if verbose:
            print("  dataset already complete, nothing to generate")
        return out_root, []

    from flatmorphai.notebook_api import classification_images as cim_api
    if images_roots is None:
        images_roots = [os.path.join(training_root, "images")]
    if mask_roots is None:
        mask_roots = ([os.path.join(training_root, "masks"), os.path.join(training_root, "mask")]
                      + sorted(glob.glob(os.path.join(training_root, "mask_*"))))
    made, skipped = 0, 0
    for rel in missing:
        src = _find(images_roots, rel)
        msk = _find(mask_roots, rel)
        if src is None or msk is None:
            skipped += 1
            if verbose and skipped <= 10:
                print("  no original/mask for", rel)
            continue
        img = cv2.imread(src)
        mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skipped += 1
            continue
        objs = cim_api.biggest_objects_in_the_binary_mask(mask, num_objects_to_keep=1, num_objects_by_row=1)
        if not objs or objs[0] is None:
            skipped += 1
            continue
        out = cim_api.compute_mask_image_planarian_reshaped(
            img, objs[0], image_size=image_size, margin_percent=margin,
            apply_mask=apply_mask, black_and_white=bw)
        if out is None:
            skipped += 1
            continue
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, out)
        made += 1
    # re-check what is still missing after generation
    still = [r for r in rels if not os.path.exists(os.path.join(out_root, r))]
    if verbose:
        print("  generated %d images, skipped %d, still missing %d" % (made, skipped, len(still)))
        if still:
            from collections import Counter
            byc = Counter(str(r).split("/")[0].split(os.sep)[0] for r in still)
            print("  still-missing by class:", dict(byc))
            print("  -> point images_roots / mask_roots at the drive that holds them (often F:)")
    return out_root, still
