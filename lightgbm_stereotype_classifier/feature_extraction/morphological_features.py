"""
morphological_features.py

Precompute deterministic shape descriptors for masked planarian images and
write them to a CSV (separator ';'). The descriptors encode the cues used to
separate Snake-like (elongation, thin body or big-head/small-body) from
Screw-like (a transverse border that cuts the body, local texture change of a
twisting body).

The pipeline is FULLY DETERMINISTIC (no randomness), so it is reproducible.
The masked training images already place the worm on a black background, so the
binary worm mask is recovered by thresholding (no separate mask file needed).

Feature columns (raw, NOT standardized; standardize at fusion time with the
TRAIN-set mean/std):
    elongation       major_axis / minor_axis           (high for Snake-like)
    eccentricity     regionprops eccentricity           (->1 for elongated)
    solidity         area / convex_area
    extent           area / bbox_area
    thinness         4*pi*area / perimeter^2            (low for elongated/irregular)
    head_body_ratio  max_width / median_width           (big-head / small-body)
    width_cv         std(width)/mean(width)
    pinch_count      narrow necks along the body        (twist proxy, Screw-like)
    edge_density     interior Canny edge pixels / area  (texture change, Screw-like)
    holes            number of internal holes           (fold-over proxy)

References for the descriptors:
    van der Walt et al., "scikit-image: image processing in Python", 2014,
    PeerJ 2:e453, DOI 10.7717/peerj.453

Usage (run on the machine where the dataset lives):
    python morphological_features.py \
        --train_csv  ../../Dataset_stereotype_regeneration/dataset_training_stereotype/dataset_images_list/<dataset>/<...>_train.csv \
        --valid_csv  ../../Dataset_stereotype_regeneration/dataset_training_stereotype/dataset_images_list/<dataset>/<...>_valid.csv \
        --image_dir  ../../Dataset_stereotype_regeneration/dataset_training_stereotype/processed_images_full_size_image-size(224)_margin(0.00)_b&w(False)_apply-mask(True)_dataset \
        --out        morphological_features.csv
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as cc_label, regionprops

FEATURE_NAMES = [
    "elongation", "eccentricity", "solidity", "extent", "thinness",
    "head_body_ratio", "width_cv", "pinch_count", "edge_density", "holes",
]

# Background threshold on the grayscale masked image. Pixels above this are worm.
_BG_THRESH = 10


def _largest_component(binary):
    """Return a boolean mask with only the largest connected component."""
    lbl = cc_label(binary, connectivity=2)
    if lbl.max() == 0:
        return None
    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # ignore background
    keep = counts.argmax()
    return lbl == keep


def _width_profile(mask_bool, orientation_rad):
    """Rotate the mask so the major axis is horizontal, return per-column worm
    width (number of worm pixels per column) for columns that contain the worm."""
    # skimage orientation is the angle (rad) between the x-axis and the major
    # axis. Rotate by +orientation (degrees) to make the major axis horizontal.
    angle_deg = np.degrees(orientation_rad)
    rotated = ndi.rotate(
        mask_bool.astype(np.uint8), angle=angle_deg, reshape=True, order=0
    )
    col_widths = rotated.sum(axis=0)
    col_widths = col_widths[col_widths > 0]
    return col_widths.astype(np.float64)


def compute_features_from_image(image_bgr_or_gray):
    """Compute the morphometric feature vector from one masked image.

    Returns a dict of FEATURE_NAMES -> float. On failure returns NaNs.
    """
    feats = {k: float("nan") for k in FEATURE_NAMES}
    try:
        img = image_bgr_or_gray
        if img is None:
            return feats
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        binary = gray > _BG_THRESH
        comp = _largest_component(binary)
        if comp is None or comp.sum() < 20:
            return feats

        props = regionprops(comp.astype(int))[0]
        area = float(props.area)
        perim = float(props.perimeter) if props.perimeter > 0 else float("nan")
        L = float(props.major_axis_length)
        W = float(props.minor_axis_length)
        eps = 1e-6

        feats["elongation"] = L / (W + eps)
        feats["eccentricity"] = float(props.eccentricity)
        feats["solidity"] = float(props.solidity)
        feats["extent"] = float(props.extent)
        feats["thinness"] = (4.0 * np.pi * area) / (perim * perim + eps)
        feats["holes"] = float(max(0, 1 - int(props.euler_number)))

        widths = _width_profile(comp, props.orientation)
        if widths.size >= 5:
            med = float(np.median(widths))
            mean_w = float(np.mean(widths))
            std_w = float(np.std(widths))
            feats["head_body_ratio"] = float(np.max(widths)) / (med + eps)
            feats["width_cv"] = std_w / (mean_w + eps)
            # Smooth the width profile and count narrow necks (local minima that
            # drop below 0.6 * median) as a twist/pinch proxy for Screw-like.
            k = max(3, (widths.size // 20) | 1)  # odd kernel
            kernel = np.ones(k) / k
            smooth = np.convolve(widths, kernel, mode="same")
            thr = 0.6 * med
            pinches = 0
            for j in range(1, smooth.size - 1):
                if smooth[j] < thr and smooth[j] <= smooth[j - 1] and smooth[j] <= smooth[j + 1]:
                    pinches += 1
            feats["pinch_count"] = float(pinches)
        else:
            feats["head_body_ratio"] = 1.0
            feats["width_cv"] = 0.0
            feats["pinch_count"] = 0.0

        # Interior edge density: Canny edges restricted to the worm interior,
        # normalized by area. A twisting Screw-like body shows an internal
        # border, raising this value.
        eroded = ndi.binary_erosion(comp, iterations=2)
        edges = cv2.Canny(gray, 40, 120) > 0
        interior_edges = np.logical_and(edges, eroded)
        feats["edge_density"] = float(interior_edges.sum()) / (area + eps)

    except Exception as exc:  # robust: never crash the whole run on one image
        print(f"[WARN] feature extraction failed: {exc}")
    return feats


def _read_image(path):
    if not os.path.exists(path):
        print(f"[WARN] missing image: {path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def build_csv(train_csv, valid_csv, image_dir, out_path):
    rows = []
    for split, csv_path in [("train", train_csv), ("valid", valid_csv)]:
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path, sep=";")
        key_class = df.keys()[-2]
        for _, r in df.iterrows():
            rel = r["image_path"]
            label = r[key_class]
            full = os.path.join(image_dir, rel)
            img = _read_image(full)
            feats = compute_features_from_image(img)
            row = {"image_path": rel, "split": split, "class": label}
            row.update(feats)
            rows.append(row)
    out = pd.DataFrame(rows, columns=["image_path", "split", "class"] + FEATURE_NAMES)
    out.to_csv(out_path, sep=";", index=False)
    print(f"[OK] wrote {len(out)} rows to {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--valid_csv", default=None)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", default="morphological_features.csv")
    args = ap.parse_args()
    build_csv(args.train_csv, args.valid_csv, args.image_dir, args.out)
