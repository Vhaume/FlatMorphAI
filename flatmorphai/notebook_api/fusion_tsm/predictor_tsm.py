"""step4_classify.py -- run the trained TSM fusion model on one prepared sample and return the class + probabilities.

Inputs per labelled frame (produced by the earlier pipeline steps):
  - temporal image: the 384x384 3-channel PNG built by step2_temporal_format (channel 0 = T-1, channel 1 = T0,
    channel 2 = T+1, each a masked grayscale frame). Pass either the path or the decoded BGR/3-channel array.
  - metric vector: a dict {metric_name: value} for this frame, produced by step3_metrics. Only the columns the model
    was trained on are used, missing ones fall back to the training median, exactly as in training.

CRITICAL preprocessing note: the TSM model was trained with a CUSTOM channel normalization, not the timm transform.
Each of the three temporal channels is scaled to [0, 1] then standardized as (g - 0.45) / 0.225 and replicated to 3
channels, giving a 9-channel tensor (3 frames x 3 channels) that the model reshapes into a clip. This file reproduces
that normalization byte-for-byte, so do not substitute a timm transform here.

Dependencies: torch, numpy, opencv-python, pandas. CPU or CUDA.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import torch

# Local relative import: model_fusion_tsm sits alongside this file
# inside the flatmorphai/notebook_api/fusion_tsm/ subpackage, so the
# original ``sys.path.insert(0, "../model")`` + bare import from the
# export_model_and_code bundle is replaced by a clean relative import.
from .model_fusion_tsm import load_fusion_tsm   # noqa: E402

IMAGE_SIZE_DEFAULT = 384
_NORM_MEAN, _NORM_STD = 0.45, 0.225            # the exact training channel normalization (FusionDS)


class TSMPredictor:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.info = load_fusion_tsm(ckpt_path, self.device)
        self.classes = self.info["classes"]
        self.cols = list(self.info["metric_cols"])
        self.size = int(self.info.get("image_size", IMAGE_SIZE_DEFAULT))
        self.mean = pd.Series(self.info["metric_mean"]).reindex(self.cols)
        self.std = pd.Series(self.info["metric_std"]).reindex(self.cols).replace(0, 1.0)
        self.med = pd.Series(self.info["metric_median"]).reindex(self.cols)

    # ---- image branch: reproduce the training temporal preprocessing exactly ----
    def image_tensor(self, temporal_image):
        if isinstance(temporal_image, str):
            img = cv2.imdecode(np.fromfile(temporal_image, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = temporal_image
        if img is None:
            raise ValueError("temporal image could not be read")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        chans = []
        for t in (0, 1, 2):                                  # channel t = the t-th temporal frame (T-1, T0, T+1)
            g = img[:, :, t].astype(np.float32) / 255.0
            g = (g - _NORM_MEAN) / _NORM_STD
            chans += [g, g, g]                               # replicate each frame to 3 channels
        x = torch.from_numpy(np.stack(chans, axis=0)).unsqueeze(0)   # (1, 9, H, W)
        return x.to(self.device)

    # ---- metric branch: standardize with the checkpoint's training statistics ----
    def metric_tensor(self, metric_dict):
        if metric_dict:
            v = pd.Series({k: metric_dict.get(k, np.nan) for k in self.cols}, dtype=float)
        else:
            v = pd.Series({k: np.nan for k in self.cols}, dtype=float)   # no metrics -> training-median vector
        v = v.reindex(self.cols).astype(float).fillna(self.med)
        v = ((v - self.mean) / self.std).to_numpy(dtype=np.float32)
        return torch.from_numpy(v).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, temporal_image, metric_dict=None):
        x = self.image_tensor(temporal_image)
        m = self.metric_tensor(metric_dict)
        logits = self.model(x, m)
        prob = torch.softmax(logits.float(), 1).cpu().numpy()[0]
        i = int(prob.argmax())
        return self.classes[i], {c: float(p) for c, p in zip(self.classes, prob)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="classify one prepared temporal frame")
    ap.add_argument("temporal_image")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "..", "model", "checkpoints",
                    "ConvNeXtV2_tsm_3f_v2_size384", "ConvNeXtV2_tsm_3f_v2_size384_20260623_225807.pth"))
    ap.add_argument("--metrics_csv", default=None, help="optional ';'-separated metric CSV with an 'image' column")
    a = ap.parse_args()
    pred = TSMPredictor(a.ckpt)
    md = None
    if a.metrics_csv and os.path.exists(a.metrics_csv):
        df = pd.read_csv(a.metrics_csv, sep=';')
        base = os.path.basename(a.temporal_image)
        row = df[df['image'].map(os.path.basename) == base]
        if len(row):
            md = {c: row.iloc[0][c] for c in df.columns if c not in ('row_index', 'image', 'class')}
    label, probs = pred.predict(a.temporal_image, md)
    print("predicted:", label)
    print("probs:", {k: round(v, 4) for k, v in probs.items()})
