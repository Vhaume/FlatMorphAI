# -*- coding: utf-8 -*-
"""Step 1. Worm segmentation: Mask2Former, two passes, CLAHE, choice of the worm by its shape.

The functions predict_prob_batch, enhance, choisir_ver and two_pass_batch are read from
production/regenerate_masks_2pass_forme.py and executed unchanged (exec of their source), the constants of the two
passes come from parameters/segmentation.csv. Path, identical to the production masks
masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme:
  1. CLAHE on the L channel (Lab), clip 2, tile 16 (ENHANCE = clahe_2_t16),
  2. first pass on the whole image, worm probability rebuilt from the logits, threshold WORM_THR = 0.50,
  3. second pass on a crop of each component (MODEL_SIZE 320, PAD_RATIO 0.50), skeleton-guided fusion,
  4. choice of the worm among the components (choisir_ver): candidates of area >= AIRE_MIN and >= REL_MIN x the
     largest, the most circular object (C = 4 pi A / P^2) among the objects not cut by the image frame is kept,
     the contrast K (median grey of the object minus that of a surrounding ring) is only a control: when the most
     circular object is not the darkest one, info["a_verifier"] = 1 asks for a visual check.
Channel order: the production program passes the BGR array of cv2.imdecode unchanged to the model (CHANNEL_ORDER =
BGR reproduces the dataset masks, RGB is offered for a new calibration only).

References (verified). Cheng B., Misra I., Schwing A. G., Kirillov A., Girdhar R. (2022). Masked-attention Mask
Transformer for Universal Image Segmentation. CVPR 2022, arXiv 2112.01527. Zuiderveld K. (1994). Contrast Limited
Adaptive Histogram Equalization. Graphics Gems IV, 474-485, DOI 10.1016/B978-0-12-336156-1.50061-6. Suzuki S., Abe K.
(1985). Topological structural analysis of digitized binary images by border following. CVGIP 30(1), 32-46,
DOI 10.1016/0734-189X(85)90016-7.
"""
import os
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image as PILImage
from . import _env
from ._env import DEVICE, read, model_path, exec_slice, show_device, check_on_device, jpeg_roundtrip, jpeg_enabled

PROD_FILE = "regenerate_masks_2pass_forme.py"


def _hf_cache():
    """Uses models/hf_cache as Hugging Face cache when it exists (filled once by segmentation/prepare_hf_cache.py)."""
    c = os.path.join(_env.EXPORT_ROOT, "models", "hf_cache")
    hub = os.path.join(c, "hub")
    if os.path.isdir(hub) and os.listdir(hub) and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = c
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Segmenter:
    """seg = Segmenter(); mask, info = seg.segment(bgr). mask: uint8 0/255, image size."""

    def __init__(self, verbose=True):
        show_device(); _hf_cache()
        from mask2former_api import load_model_for_segmentation
        from evaluate_test_images_v2 import (_extract_p2_jobs, _paste_p2_result, _apply_fusion, _ensure_uint8_rgb,
                                             _maybe_suppress_stdout)
        self.P = read("segmentation")
        folder = model_path("mask2former_segmentation").rstrip("/\\"); name = os.path.basename(folder)
        self.model, self.processor = load_model_for_segmentation(name, DEVICE, checkpoint_root=os.path.dirname(folder))
        self.model.eval(); check_on_device(self.model, "Mask2Former")
        ns = dict(np=np, cv2=cv2, torch=torch, F=F, PILImage=PILImage, model=self.model, processor=self.processor,
                  device=DEVICE, _extract_p2_jobs=_extract_p2_jobs, _paste_p2_result=_paste_p2_result,
                  _apply_fusion=_apply_fusion, _ensure_uint8_rgb=_ensure_uint8_rgb,
                  _maybe_suppress_stdout=_maybe_suppress_stdout)
        exec_slice(PROD_FILE, "def predict_prob_batch(images", "REGEN = os.path.join(", ns)
        for k in ("BATCH", "MODEL_SIZE", "PAD_RATIO", "MERGE", "MIN_COMP", "MIN_PAD", "MIN_ROI", "BND_DIL",
                  "MERGE_DIL_F", "AIRE_MIN", "REL_MIN"):
            ns[k] = self.P[k]
        self.ns = ns; self._rgb = _ensure_uint8_rgb
        if verbose:
            print("segmenter %s | enhancement %s | worm threshold %.2f | channel order %s | worm choice by circularity"
                  % (name, self.P["ENHANCE"], float(self.P["WORM_THR"]), self.P["CHANNEL_ORDER"]))

    def segment_batch(self, bgr_list):
        """List of BGR images (as read by cv2) -> list of (mask uint8 0/255, info dict of choisir_ver)."""
        if str(self.P["CHANNEL_ORDER"]).upper() == "RGB":
            imgs = [self._rgb(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in bgr_list]
        else:
            imgs = [self._rgb(b) for b in bgr_list]
        out = []
        for k in range(0, len(imgs), int(self.P["BATCH"])):
            out.extend(self.ns["two_pass_batch"](imgs[k:k + int(self.P["BATCH"])], str(self.P["ENHANCE"]),
                                                 float(self.P["WORM_THR"])))
        res = []
        for mk, inf in out:
            if self.P["JPEG_ROUNDTRIP"] and jpeg_enabled():
                mk = jpeg_roundtrip(mk, pil=True)   # what the refinement read in production
            res.append((mk, inf))
        return res

    def segment(self, bgr):
        return self.segment_batch([bgr])[0]
