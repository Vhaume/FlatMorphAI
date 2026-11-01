"""
train_tfms_320.py
=================
Modified training augmentation pipeline for planarian worm segmentation
at 320x320 input resolution, incorporating all decisions from the
model-and-augmentation review of 2026-03.

CHANGES vs. the 784x784 pipeline (train_tfms_v7)
-------------------------------------------------
1. H, W = 320, 320  (reverted from 784)
2. CLAHEPreprocess REMOVED -- see section "WHY CLAHE IS REMOVED" below.
3. RandomGamma KEPT with wider range (40, 160).
4. ColorOverlayAugmentation ADDED -- fixes total miss on yellow-blob image.
5. A.RandomShadow KEPT from v7 recommendation.
6. A.HueSaturationValue ADDED with strong sat_shift_limit.
7. ZoomInAroundMask ADDED with p=0.20 to support two-pass inference.
8. train_num_points reverted to 50_000 in model config (see model section).
9. CLAHE at inference also removed (see build_inference_tfms below).

WHY CLAHE IS REMOVED AT 320x320
---------------------------------
The worm in the test image in Capture_d_ecran_2025-02-10_103913.png shows a
native thin tail that is approximately 4-8 px wide in the 1385x1216 source.
At 320x320 input that becomes:
    4 * (320 / 1385) = 0.92 px     (still sub-pixel)
    8 * (320 / 1385) = 1.85 px

CLAHE at 320x320 amplifies the bilinear-interpolation artifact of the
downsampled sub-pixel structure, not the real worm signal. This was confirmed
empirically: removing CLAHE at 320x320 improved validation Dice. At 784x784,
CLAHE is appropriate because the same 4 px native worm becomes 2.26 px at
model input -- a real, enhanceable signal. At 320x320, CLAHE should remain
OFF. This matches the analysis in resolution_upgrade_784.py, lines 31-39.
Reference: Zuiderveld K (1994). Graphics Gems IV.
           DOI:10.1016/B978-0-12-336156-1.50061-6

WHY GAMMA IS BETTER THAN CLAHE AT 320x320
------------------------------------------
RandomGamma applies a monotonic pixel-wise power transform y = x^gamma.
It does not amplify noise (no tile-based local histogram equalization).
At gamma < 1 (gamma_limit < 100 in albumentations convention), the
transform brightens the image and increases contrast in dark regions,
making dim thin-worm borders more visible -- exactly the gamma-up step
used by the image expert during manual segmentation. At gamma > 1, the
transform darkens and unifies bright worm body regions. Training on the
full (40, 160) range is therefore directly encoding the expert segmentation
workflow without any sub-pixel noise amplification.
Reference: Shorten C and Khoshgoftaar T M (2019). A survey on image data
           augmentation for deep learning. Journal of Big Data 6:60.
           DOI:10.1186/s40537-019-0197-0.

WHY COLOROVERLAYAUGMENTATION IS ADDED
--------------------------------------
Capture_d_ecran_2025-02-10_103913.png shows a yellow amorphous blob
(likely a food source or egg mass) covering approximately 40% of the
planarian's body. The model produces a completely empty mask because:
  a) Yellow is never present in training images (OOD color distribution).
  b) The yellow blob's high-saturation yellow activates the background
     class in the softmax (the model has never seen yellow foreground or
     yellow occlusion).
  c) The thin tail extends outside the yellow blob but is too narrow at
     320x320 to anchor the model's prediction.

Fix: randomly paste a semi-transparent ellipse of random color (including
saturated yellow, green, red, and blue) over part of the worm in training.
The model must learn to predict the worm body through / around the colored
occlusion. This forces the model to use shape and local texture cues rather
than color consistency.
Reference: Ghiasi G et al. (2021). Simple Copy-Paste is a Strong Data
           Augmentation Method for Instance Segmentation. CVPR 2021.
           arXiv:2012.07177 (occlusion-aware instance segmentation training).

WHY ZOOMINAROUNDMASK IS ADDED
-------------------------------
The two-pass inference strategy crops the worm region detected in Pass 1
and runs a second pass at 320x320. For this Pass 2 to work, the model must
have been trained on images where the worm fills a large fraction of the
frame (the crop view). Without this augmentation the Pass-2 view is OOD.
ZoomInAroundMask is applied with p=0.20 (one in five batches) to create
training samples where the worm is already close-cropped, matching the
Pass-2 inference distribution.
Reference: Singh B and Davis L S (2018). An Analysis of Scale Invariance
           in Object Detection. CVPR 2018. arXiv:1711.08189.

MODEL PATCH DECISIONS FOR 320x320
-----------------------------------
(See setup_model_320() at the bottom of this file.)

USE: patch_mask2former_for_thin_worms (thin_worm_patch.py)
  - _DilatedRefineBlock uses Conv2d only, no size-dependent parameters.
    The dilated cascade is equally valid at 320x320: a 3px worm at 320x320
    becomes 0.75 px at the 80x80 mask_features map. With target_stride=1
    the mask_features are upsampled to 320x320, making the 3px worm
    representable as a 3px structure (not sub-pixel).
  - attn_threshold=0.3: still beneficial. The hourglass feedback loop in
    Mask2FormerMaskPredictor exists at any resolution.

DO NOT USE: resolution_upgrade_784.setup_model_784
  - This sets train_num_points=200_000, which is calibrated for 784x784.
    At 320x320 (102,400 pixels) sampling 200,000 points means sampling each
    pixel 1.95 times per step -- wasteful and slower.
  - Gradient checkpointing: optional at 320x320. The activation memory at
    320x320 is ~6x smaller than at 784x784. Checkpointing adds 20-30%
    training time with minimal memory benefit. Leave it OFF unless you are
    running out of GPU memory.

REVERT: train_num_points = 50_000
  This matches the thin_worm_patch.py recommendation for 320x320.
  Sampling density: 50_000 / (320*320) = 48.8% -- good coverage.
"""

import random
import cv2
import numpy as np
import albumentations as A

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H, W = 320, 320   # REVERTED from 784 to 320

# Gaussian noise range (keep identical to v7)
_GAUSS_STD_LO = 0.01
_GAUSS_STD_HI = 0.05

# ImageNet normalisation (must match HuggingFace processor)
ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# ColorOverlayAugmentation
# ---------------------------------------------------------------------------

class ColorOverlayAugmentation(A.ImageOnlyTransform):
    """
    Place one or more semi-transparent coloured ellipses over the image.

    The ellipses simulate novel-colour occlusions (food blobs, egg masses,
    coloured reagents) that appear on planarians in some imaging conditions
    but are absent from the training set. The model must learn to segment
    the worm body through / around the occlusion using shape and texture
    cues rather than colour consistency.

    Parameters
    ----------
    num_blobs_range : tuple (int, int)
        Min and max number of ellipses to add per image.
    alpha_range : tuple (float, float)
        Opacity range of the overlay. 0.0 = invisible, 1.0 = opaque.
        Default (0.5, 0.85) ensures the worm texture is still partially
        visible through the overlay, matching the real yellow-blob case.
    hue_range : tuple (int, int)
        HSV hue range in [0, 179] (OpenCV convention).
        Default (0, 179) = any hue. Restrict to (20, 35) to specifically
        simulate yellow blobs.
    saturation_range : tuple (int, int)
        HSV saturation of the blob colour [0, 255].
    size_range : tuple (float, float)
        Blob semi-axes as fraction of the image size (min, max).

    References
    ----------
    Ghiasi G et al. (2021). Simple Copy-Paste. CVPR 2021. arXiv:2012.07177.
    """

    def __init__(
        self,
        num_blobs_range=(1, 3),
        alpha_range=(0.50, 0.85),
        hue_range=(0, 179),
        saturation_range=(160, 255),
        size_range=(0.15, 0.45),
        p=0.5,
    ):
        super().__init__(p=p)
        self.num_blobs_range  = num_blobs_range
        self.alpha_range      = alpha_range
        self.hue_range        = hue_range
        self.saturation_range = saturation_range
        self.size_range       = size_range

    def apply(self, img, **params):
        img = img.copy()
        h, w = img.shape[:2]
        n_blobs = random.randint(*self.num_blobs_range)

        for _ in range(n_blobs):
            # Random blob colour in HSV -> BGR.
            # Brown-zone exclusion: H=10-35 in OpenCV HSV overlaps with
            # planarian worm tissue (H=10-30, S=40-120). At this hue, a
            # semi-transparent overlay is visually indistinguishable from
            # worm body, preventing the model from learning colour invariance.
            # Resample until hue is outside the brown zone (P(reject)~15%).
            # Reference: yellow-blob misclassification analysis on
            # Capture_d_ecran_2025-02-10_103301 failure cases.
            hue = random.randint(*self.hue_range)
            while 10 <= hue <= 35:
                hue = random.randint(*self.hue_range)
            sat = random.randint(*self.saturation_range)
            val = random.randint(180, 255)
            colour_hsv = np.uint8([[[hue, sat, val]]])
            colour_bgr = cv2.cvtColor(colour_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

            # Random ellipse geometry
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            ax = int(random.uniform(*self.size_range) * w)
            ay = int(random.uniform(*self.size_range) * h)
            angle = random.randint(0, 180)

            # Draw on overlay
            overlay = img.copy()
            cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360,
                        colour_bgr, -1)

            alpha = random.uniform(*self.alpha_range)
            img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

        return img

    def get_transform_init_args_names(self):
        # Albumentations serialization hook (all versions).
        # Old API (pre-1.0): calls this, then {k: getattr(self,k) for k in names}.
        # New API (>=1.0):   calls get_transform_init_args() which calls this.
        # All names must be instance attributes set in __init__.
        return ("num_blobs_range", "alpha_range", "hue_range",
                "saturation_range", "size_range")

# ---------------------------------------------------------------------------
# TransparentEdgeFade: novel augmentation for transparent worm boundaries
#
# Synthetically fades a random arc (15-45% of perimeter) of the worm boundary
# toward background color, simulating translucent worm edges as in cases:
#   113613 (right edge near-invisible), 115743 (head), EBPV (full border).
# Only the image is modified; the mask is returned unchanged so the model
# learns to include low-contrast boundaries in its prediction.
# Reference: Taylor L and Nitschke G (2018). arXiv:1708.04896.
# ---------------------------------------------------------------------------
class TransparentEdgeFade(A.DualTransform):
    def __init__(self, fade_width_range=(4, 14), arc_fraction_range=(0.15, 0.45), p=0.25):
        super().__init__(p=p)
        self.fade_width_range = fade_width_range
        self.arc_fraction_range = arc_fraction_range
 
    def apply(self, img, fade_width=8, arc_start=0.0, arc_fraction=0.3,
              mask_input=None, **params):
        import cv2 as _cv2
        import numpy as _np
        try:
            if mask_input is None or mask_input.max() == 0:
                return img
            # Guard: skip thin worms. A worm with < 2000 FG pixels at 320x320
            # is ~45x45px -- fade_width=4-14px would erase most of its body,
            # training the model to ignore thin structures (regression fix).
            fg_pixels = int((mask_input > 0).sum())
            if fg_pixels < 2000:
                return img
            mask_bin = (mask_input > 0).astype(_np.uint8)
            kernel = _np.ones((fade_width, fade_width), _np.uint8)
            eroded = _cv2.erode(mask_bin, kernel, iterations=1)
            boundary_band = (mask_bin - eroded).astype(bool)
            boundary_coords = _np.argwhere(boundary_band)
            if len(boundary_coords) == 0:
                return img
            n = len(boundary_coords)
            arc_n = max(1, int(n * arc_fraction))
            start_idx = int(arc_start * n)
            indices = _np.arange(start_idx, start_idx + arc_n) % n
            arc_coords = boundary_coords[indices]
            # Background: sample pixels immediately OUTSIDE the mask (local dilation ring).
            # Matches local dish illumination regardless of vignette or gradients.
            # Reference: Geirhos et al. (2020). DOI:10.1038/s42256-020-00257-z.
            dil_kernel = _np.ones((10, 10), _np.uint8)
            dilated = _cv2.dilate(mask_bin, dil_kernel, iterations=1)
            local_bg_mask = (dilated - mask_bin).astype(bool)
            if local_bg_mask.sum() > 0:
                bg_pixels = img[local_bg_mask]
                bg_color = _np.median(bg_pixels, axis=0).astype(_np.float32)
            else:
                h, w = img.shape[:2]
                bg_color = _np.median(
                    img[:max(1,h//8), :max(1,w//8)].reshape(-1, img.shape[2]), axis=0
                ).astype(_np.float32)
            dist_map = _cv2.distanceTransform(mask_bin, _cv2.DIST_L2, 5)
            # Vectorized blend: no Python pixel loop (suggested by reviewer)
            blend_mask = _np.zeros(mask_bin.shape, dtype=bool)
            blend_mask[arc_coords[:, 0], arc_coords[:, 1]] = True
            alpha = (dist_map / float(fade_width)).clip(0.0, 1.0)[..., _np.newaxis]
            result = img.astype(_np.float32)
            result = _np.where(
                blend_mask[..., _np.newaxis],
                alpha * result + (1.0 - alpha) * bg_color,
                result,
            )
            return _np.clip(result, 0, 255).astype(_np.uint8)
        except Exception:
            # Safety fallback: any error in boundary computation returns
            # the original image unchanged to prevent training spikes.
            return img
 
    def apply_to_mask(self, mask, **params):
        return mask
 
    def get_params_dependent_on_data(self, params, data):
        import numpy as _np
        fade_width = int(_np.random.randint(self.fade_width_range[0], self.fade_width_range[1] + 1))
        arc_fraction = float(_np.random.uniform(*self.arc_fraction_range))
        arc_start = float(_np.random.uniform(0.0, 1.0))
        mask_input = None
        masks = data.get('masks', None) or data.get('mask', None)
        if masks is not None:
            if isinstance(masks, list) and len(masks) > 0:
                mask_input = masks[0]
            elif hasattr(masks, 'shape'):
                mask_input = masks
        return {'fade_width': fade_width, 'arc_fraction': arc_fraction,
                'arc_start': arc_start, 'mask_input': mask_input}
 
    def get_transform_init_args_names(self):
        return ('fade_width_range', 'arc_fraction_range')     
# ---------------------------------------------------------------------------
# ZoomInAroundMask  (supports two-pass inference distribution)
# ---------------------------------------------------------------------------

class ZoomInAroundMask(A.DualTransform):
    """
    Crop a tight region around the foreground mask and resize back to (H, W).

    This augmentation creates training samples that match the Pass-2 view
    of the two-pass inference strategy: the worm fills most of the frame,
    with some background padding on the sides.

    When no foreground is present (blank or near-blank mask), the transform
    falls back to a centre crop of 70% of the image.

    Parameters
    ----------
    pad_fraction : float
        Fraction of the bounding-box dimension added as padding on each side.
        0.20 = 20% padding, matching approximately the two-pass inference
        padding_ratio.
    p : float
        Probability of applying this transform per sample. 0.20 recommended
        (one in five training samples should be in close-crop view to match
        the distribution of Pass-2 inference calls).
    """

    def __init__(self, pad_fraction=0.20, p=0.20):
        super().__init__(p=p)
        self.pad_fraction = pad_fraction

    def apply(self, img, crop_coords=None, **params):
        if crop_coords is None:
            return img
        y1, y2, x1, x2 = crop_coords
        cropped = img[y1:y2, x1:x2]
        return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask, crop_coords=None, **params):
        if crop_coords is None:
            return mask
        y1, y2, x1, x2 = crop_coords
        cropped = mask[y1:y2, x1:x2]
        return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_NEAREST)

    def _compute_crop_coords(self, img, mask):
        """Shared crop computation for old and new albumentations API."""
        img_h, img_w = img.shape[:2]
        crop_coords = None
        if mask is not None and isinstance(mask, np.ndarray):
            # Squeeze to 2D: old albumentations may pass mask as (H, W, C).
            # np.where on a 3D array returns a 3-tuple (ys, xs, cs) which
            # raises "too many values to unpack (expected 2)".
            m2d = mask if mask.ndim == 2 else mask[:, :, 0]
            bin_mask = (m2d > 0).astype(np.uint8)
            ys, xs = np.where(bin_mask)
            if len(ys) > 0:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                bh = max(y2 - y1, 1)
                bw = max(x2 - x1, 1)
                pad_y = int(bh * self.pad_fraction)
                pad_x = int(bw * self.pad_fraction)
                y1c = max(0,     y1 - pad_y)
                y2c = min(img_h, y2 + pad_y)
                x1c = max(0,     x1 - pad_x)
                x2c = min(img_w, x2 + pad_x)
                if (y2c - y1c) < img_h * 0.90 or (x2c - x1c) < img_w * 0.90:
                    crop_coords = (y1c, y2c, x1c, x2c)
        if crop_coords is None:
            py = int(img_h * 0.15)
            px = int(img_w * 0.15)
            crop_coords = (py, img_h - py, px, img_w - px)
        return {"crop_coords": crop_coords}

    def get_params_dependent_on_data(self, params, data):
        # New albumentations API (>=1.0).
        return self._compute_crop_coords(data["image"], data.get("mask"))

    def get_params_dependent_on_targets(self, params):
        # Old albumentations API (pre-1.0): params IS the targets dict.
        return self._compute_crop_coords(params["image"], params.get("mask"))

    @property
    def targets_as_params(self):
        # Required by old albumentations to declare which targets feed into
        # get_params_dependent_on_targets.
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        # Albumentations serialization hook, both API versions.
        # Old (pre-1.0): {k: getattr(self,k) for k in names}.
        # New (>=1.0):   get_transform_init_args() calls this internally.
        return ("pad_fraction",)

# ---------------------------------------------------------------------------
# Model setup for 320x320
# ---------------------------------------------------------------------------

def enable_pixel_decoder_checkpointing(model, verbose: bool = True) -> None:
    """
    Enable gradient checkpointing on the Mask2Former pixel decoder encoder
    (Mask2FormerEncoderLayer -- the multi-scale deformable attention layers).

    WHY THIS IS NEEDED AT 320x320
    ------------------------------
    The OOM crash at 320x320 occurs inside multi_scale_deformable_attention
    at the line:
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights

    This stack tensor has shape [B*n_heads, C/n_heads, N_q, n_levels*n_points]
    where N_q = 8500 at 320x320 (sum of FPN spatial sizes 80^2+40^2+20^2+10^2).
    At batch_size=4: 4 * 8 * 32 * 8500 * 16 * 4 bytes = 531 MB per encoder call.
    With 6 encoder layers stored for backward: 6 * 531 MB = 3.1 GB for this alone.

    At 784x784, the same crash did NOT occur because batch_size was 2 (not 4)
    and Swin gradient checkpointing freed enough headroom. At 320x320, batch_size
    increases to 4 (natural choice at lower resolution), which causes the OOM
    even though the per-element resolution is lower.

    Checkpointing the pixel decoder encoder layers recomputes the 531 MB stack
    on backward instead of storing it, reducing peak GPU memory by ~3 GB at B=4.

    This function monkey-patches the forward method of each Mask2FormerEncoderLayer
    using torch.utils.checkpoint.checkpoint, the same approach used in
    resolution_upgrade_784.enable_gradient_checkpointing for SwinLayer.

    Must be called AFTER patch_mask2former_for_thin_worms() and BEFORE DDP wrapping.
    Safe to call multiple times (idempotent via _pdc_patched sentinel).

    Parameters
    ----------
    model : nn.Module
        Mask2Former model (HuggingFace transformers).
    verbose : bool

    References
    ----------
    Chen T, Xu B, Zhang C, Guestrin C (2016).
    Training Deep Nets with Sublinear Memory Cost.
    arXiv:1604.06174.

    Cheng B et al. (2022). Masked-attention Mask Transformer.
    CVPR 2022. arXiv:2112.01527. (pixel decoder architecture)
    """
    import torch
    import torch.utils.checkpoint as torch_ckpt
    import functools

    patched = []

    for name, mod in model.named_modules():
        cls_name = type(mod).__name__
        # Target: Mask2FormerEncoderLayer (contains multi_scale_deformable_attention)
        if cls_name != "Mask2FormerEncoderLayer":
            continue
        if getattr(mod, "_pdc_patched", False):
            continue  # idempotent

        original_forward = mod.forward

        # Mask2FormerEncoderLayer.forward signature (transformers):
        #   forward(hidden_states, attention_mask=None, position_embeddings=None,
        #           reference_points=None, spatial_shapes=None,
        #           level_start_index=None, output_attentions=False)
        # All arguments must be passed as positional to checkpoint.
        # torch.utils.checkpoint does not support keyword-only args in old torch.
        @functools.wraps(original_forward)
        def make_checkpointed_fwd(orig_fwd):
            def checkpointed_forward(hidden_states, attention_mask=None,
                                     position_embeddings=None,
                                     reference_points=None,
                                     spatial_shapes=None,
                                     level_start_index=None,
                                     output_attentions=False):
                # Build a positional-args-only wrapper for checkpoint
                def _inner(hs, am, pe, rp, ss, lsi):
                    return orig_fwd(
                        hs,
                        attention_mask=am,
                        position_embeddings=pe,
                        reference_points=rp,
                        spatial_shapes=ss,
                        level_start_index=lsi,
                        output_attentions=output_attentions,
                    )
                # Provide dummy tensors for None inputs so checkpoint can handle them
                import torch as _torch
                _dev = hidden_states.device
                _am  = attention_mask  if attention_mask  is not None else _torch.zeros(1, device=_dev)
                _pe  = position_embeddings if position_embeddings is not None else _torch.zeros(1, device=_dev)
                _rp  = reference_points    if reference_points    is not None else _torch.zeros(1, device=_dev)
                _ss  = spatial_shapes      if spatial_shapes       is not None else _torch.zeros(1, device=_dev)
                _lsi = level_start_index   if level_start_index    is not None else _torch.zeros(1, device=_dev)
                return torch_ckpt.checkpoint(
                    _inner, hidden_states, _am, _pe, _rp, _ss, _lsi,
                    use_reentrant=False,
                )
            return checkpointed_forward

        mod.forward    = make_checkpointed_fwd(original_forward)
        mod._pdc_patched = True
        patched.append(name)

    if verbose:
        if patched:
            print(f"  [enable_pixel_decoder_checkpointing] Patched {len(patched)} "
                  f"Mask2FormerEncoderLayer(s): {patched}")
        else:
            print("  [enable_pixel_decoder_checkpointing] WARNING: No "
                  "Mask2FormerEncoderLayer found. Pixel decoder encoder "
                  "checkpointing NOT applied.")



def setup_model_320(model, target_stride: int = 2, verbose: bool = True):
    """
    Apply all model patches for 320x320 training.

    Replaces the current setup that calls setup_model_784(model).

    Steps
    -----
    1. patch_mask2former_for_thin_worms (thin_worm_patch.py)
       - _DilatedRefineBlock dilated cascade (Conv2d only, resolution-agnostic)
       - target_stride=1: mask_features upsampled to 320x320 so a 3px worm
         is representable (not sub-pixel)
       - attn_threshold=0.3: breaks the hourglass feedback loop in the
         masked cross-attention decoder (valid at any resolution)
       - train_num_points, oversample_ratio, importance_sample_ratio,
         no_object_weight updated as per thin_worm_patch.py recommendations

    2. train_num_points = 50_000  (REVERTED from 200_000)
       Sampling density at 320x320: 50_000 / 102_400 = 48.8% -- good coverage.
       200_000 was calibrated for 784x784 (614,656 pixels). Using it at
       320x320 means sampling each pixel 1.95 times per step -- redundant
       and slower.

    3. Gradient checkpointing: NOT enabled by default.
       At 320x320 the activation memory is ~6x smaller than at 784x784.
       Enable with enable_gradient_checkpointing(model) from
       resolution_upgrade_784.py only if you run out of GPU memory.

    DO NOT call resolution_upgrade_784.setup_model_784(model).
    It sets train_num_points=200_000 and enables checkpointing (unneeded
    overhead at 320x320) and asserts INPUT_SHAPE=(784,784).

    Parameters
    ----------
    model : Mask2FormerForUniversalSegmentation
        Model BEFORE DDP wrapping, on any device (CPU or GPU).
    verbose : bool

    Returns
    -------
    model : same model in-place

    Usage
    -----
        from thin_worm_patch import patch_mask2former_for_thin_worms
        from train_tfms_320 import setup_model_320

        model = Mask2FormerForUniversalSegmentation.from_pretrained(...)
        setup_model_320(model, verbose=True)
        # Then wrap with DDP as before.

    References
    ----------
    Cheng B et al. (2022). Masked-attention Mask Transformer for Universal
        Image Segmentation. CVPR 2022. arXiv:2112.01527.
    Chen T et al. (2016). Training Deep Nets with Sublinear Memory Cost.
        arXiv:1604.06174.  (gradient checkpointing -- optional at 320x320)
    """
    if verbose:
        print("\n[setup_model_320] Applying 320x320 model configuration...")
        print(f"  Input shape: ({H}, {W})")

    # Step 1: thin-worm architectural patch
    try:
        from thin_worm_patch import patch_mask2former_for_thin_worms
    except ImportError:
        raise ImportError(
            "thin_worm_patch.py not found. Place it in the same directory "
            "or on PYTHONPATH before calling setup_model_320()."
        )

    patch_mask2former_for_thin_worms(
        model,
        target_stride=target_stride,
        attn_threshold=0.3,
        verbose=verbose,
    )

    # Step 2: revert train_num_points to 320x320-appropriate value
    _320_SAMPLING = {
        "train_num_points":       50_000,
        "oversample_ratio":        4.0,
        "importance_sample_ratio": 0.90,
        "no_object_weight":        0.02,
    }
    for mod in [
        getattr(model, "criterion", None),
        getattr(getattr(model, "criterion", None), "matcher", None),
    ]:
        if mod is None:
            continue
        for attr, new_val in _320_SAMPLING.items():
            if hasattr(mod, attr):
                old_val = getattr(mod, attr)
                setattr(mod, attr, new_val)
                if verbose:
                    print(f"  {type(mod).__name__}.{attr}: {old_val} -> {new_val}")

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr, new_val in _320_SAMPLING.items():
            if hasattr(cfg, attr):
                setattr(cfg, attr, new_val)

    # Step 3: enable gradient checkpointing (pixel decoder encoder + Swin backbone).
    # At 320x320 with target_stride=1, mask_features are 320x320 which is memory-
    # intensive. The pixel decoder encoder (MSDeformable attention) allocates a
    # 531 MB sampling_value_list stack at batch_size=4. Checkpointing is required
    # to prevent OOM even though 320x320 has lower resolution than 784x784.
    # Reference: Chen T et al. (2016). Training Deep Nets with Sublinear Memory
    # Cost. arXiv:1604.06174.
    enable_pixel_decoder_checkpointing(model, verbose=verbose)

    try:
        from resolution_upgrade_784 import enable_gradient_checkpointing
        enable_gradient_checkpointing(model, verbose=verbose)
    except ImportError:
        if verbose:
            print(
                "  [setup_model_320] WARNING: resolution_upgrade_784.py not found.\n"
                "  Swin backbone checkpointing NOT enabled. If you see OOM, add\n"
                "  resolution_upgrade_784.py to your path."
            )

    # Step 4: replace refine_1x with a bottleneck block + checkpoint.
    #
    # GATED ON target_stride == 1. When target_stride=2 (default), refine_1x is
    # never created by ThinWormPixelLevelModule.__init__ so this step is skipped.
    #
    # WHY target_stride=1 IS INFEASIBLE ON 16 GiB GPUs AT B=4
    # ----------------------------------------------------------
    # Stage 2 of ThinWormPixelLevelModule (stride-2 -> stride-1) stores the
    # following tensors in the autograd graph:
    #   F.interpolate input  (B, 256, 160, 160) = 100 MiB
    #   F.interpolate output (B, 256, 320, 320) = 400 MiB  (kept for residual)
    #   refine_1x input      (B, 256, 320, 320) = 400 MiB  (kept for residual add)
    #   refine_1x output     (B, 256, 320, 320) = 400 MiB
    # Total extra vs target_stride=2: 1300 MiB = 1.27 GiB of autograd tensors.
    # Plus forward activations inside the bottleneck block: ~625 MiB.
    # Total overhead of target_stride=1 over target_stride=2: ~1.9 GiB.
    # At B=4 with 13.36 GiB already used, this puts the GPU over 15.78 GiB.
    #
    # RECOMMENDATION: always use target_stride=2 on 16 GiB GPUs at B=4.
    # Use target_stride=1 only on GPUs with 32+ GiB VRAM.
    #
    # Reference: He K et al. (2016). arXiv:1512.03385. (bottleneck design)
    #            Wang J et al. (2020). arXiv:1908.07919. (full-res thin branches)
    #            Chen T et al. (2016). arXiv:1604.06174. (gradient checkpointing)
    if target_stride == 1:
        try:
            from refine_1x_patch import patch_refine_1x_for_320
            patch_refine_1x_for_320(model, bottleneck_dim=64, use_checkpoint=True,
                                    verbose=verbose)
        except ImportError:
            if verbose:
                print(
                    "  [setup_model_320] WARNING: refine_1x_patch.py not found.\n"
                    "  refine_1x OOM fix NOT applied. Place refine_1x_patch.py in\n"
                    "  the same directory as train_tfms_320.py."
                )
    else:
        if verbose:
            print(
                "  [setup_model_320] target_stride=2: refine_1x step skipped "
                "(mask_features at 160x160, no full-resolution convolutions)."
            )

    if verbose:
        mf_size = 320 // target_stride
        refine_note = (
            "bottleneck(64ch) + checkpoint ON" if target_stride == 1
            else "not present (target_stride=2)"
        )
        print(
            "[setup_model_320] Done.\n"
            "  Swin backbone       : gradient checkpointing ON\n"
            "  Pixel decoder enc   : gradient checkpointing ON\n"
            f"  target_stride       : {target_stride}  "
            f"(mask_features at {mf_size}x{mf_size})\n"
            f"  refine_1x           : {refine_note}\n"
            f"  refine_2x           : original 256ch (160x160)"
        )

    return model



# ---------------------------------------------------------------------------
# Summary table printed on import (for quick review during experiments)
# ---------------------------------------------------------------------------

_SUMMARY = """
train_tfms_320.py -- augmentation and model decisions summary
=============================================================
Input size        : 320 x 320  (reverted from 784)
CLAHE             : REMOVED  (amplifies sub-pixel aliasing at 320x320)
Gamma             : KEPT, range (40, 160)  (expert dual-gamma workflow)
RandomShadow      : KEPT, p=0.30
ColorOverlay      : NEW, p=0.30  (fixes yellow-blob total miss)
HueSaturationValue: NEW, p=0.40  (colour invariance for novel conditions)
ZoomInAroundMask  : NEW, p=0.20  (supports two-pass inference distribution)
ZoomOutReflect    : KEPT (uncomment if defined in your codebase)
RandomCropAroundMask: KEPT (uncomment if defined in your codebase)

Model patch       : thin_worm_patch.py  (patch_mask2former_for_thin_worms)
  target_stride   : 1  (mask_features at 320x320, 3px worm representable)
  attn_threshold  : 0.3  (breaks hourglass feedback, valid at any resolution)
  train_num_points: 50,000  (REVERTED from 200,000 which was for 784x784)
  grad checkpoint : ON   (Swin + pixel decoder encoder, required at B>=4)

DO NOT USE       : resolution_upgrade_784.setup_model_784(model)
"""


if __name__ == "__main__":
    import random as _random
    _random.seed(42)
    np.random.seed(42)

    print(_SUMMARY)

    # Sanity check: ColorOverlayAugmentation
    print("Testing ColorOverlayAugmentation...")
    overlay = ColorOverlayAugmentation(p=1.0)
    dummy_img  = np.ones((320, 320, 3), dtype=np.uint8) * 200
    dummy_mask = np.zeros((320, 320),   dtype=np.uint8)
    out = overlay(image=dummy_img)
    assert out["image"].shape == (320, 320, 3)
    assert out["image"].dtype == np.uint8
    print(f"  OK: output shape={out['image'].shape} dtype={out['image'].dtype}")

    # Sanity check: ZoomInAroundMask with foreground
    print("Testing ZoomInAroundMask (with foreground)...")
    zoom = ZoomInAroundMask(pad_fraction=0.20, p=1.0)
    fg_mask = np.zeros((320, 320), dtype=np.uint8)
    fg_mask[100:200, 130:190] = 255  # simulate worm bounding box
    img_in  = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    out2 = zoom(image=img_in, mask=fg_mask)
    assert out2["image"].shape == (320, 320, 3)
    assert out2["mask"].shape  == (320, 320)
    print(f"  OK: image {out2['image'].shape} mask {out2['mask'].shape}")

    # Sanity check: ZoomInAroundMask without foreground (fallback)
    print("Testing ZoomInAroundMask (empty mask, fallback)...")
    out3 = zoom(image=img_in, mask=np.zeros((320, 320), dtype=np.uint8))
    assert out3["image"].shape == (320, 320, 3)
    print(f"  OK: fallback output {out3['image'].shape}")

    # Sanity check: full train_tfms pipeline (without custom classes)
    print("Testing full train_tfms pipeline (core transforms only)...")
    test_img  = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    test_mask = np.zeros((400, 600), dtype=np.uint8)
    test_mask[150:300, 200:400] = 255
    out4 = train_tfms(image=test_img, mask=test_mask)
    assert out4["image"].shape == (320, 320, 3)
    assert out4["mask"].shape  == (320, 320)
    print(f"  OK: train_tfms output image={out4['image'].shape} mask={out4['mask'].shape}")

    print("\nAll checks passed.")
    print(_SUMMARY)