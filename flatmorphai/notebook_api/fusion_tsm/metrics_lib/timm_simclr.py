"""
timm_simclr.py  (NEW FILE - does not modify the retro-compatible code)

A generic timm backbone wrapped with the SAME SimCLR v2 projection head design
used by model_wrapper.FasterViTWithProjection (3-layer head feat->feat->feat->proj)
plus a linear classifier. The interface mirrors the original wrapper:

    logits            = model(x)
    logits, proj      = model(x, return_features=True)   # proj is L2-normalized

so the 128-d projection output can be used downstream for prototype-based
classification (including a new class). This file is self-contained and imports
nothing from model_wrapper, so the FasterViT pipeline is untouched.

References:
    Chen et al., "Big Self-Supervised Models are Strong Semi-Supervised Learners"
    (SimCLR v2), NeurIPS 2020, arXiv 2006.10029 (the 3-layer projection head).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def simclr_v2_head(feature_dim, proj_dim=128):
    """SimCLR v2 style 3-layer projection head, sized to the backbone."""
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
        nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
        nn.Linear(feature_dim, proj_dim),
    )


class TimmWithProjection(nn.Module):
    """timm backbone (global-avg pooled features) + SimCLR v2 head + classifier."""

    def __init__(self, backbone_name, num_classes=5, proj_dim=128,
                 pretrained=True, img_size=None, projection_version="v2", drop_path_rate=0.0):
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
        if drop_path_rate and drop_path_rate > 0:
            kwargs["drop_path_rate"] = float(drop_path_rate)
        self.backbone = None
        if img_size is not None:
            try:
                self.backbone = timm.create_model(backbone_name, img_size=img_size,
                                                  dynamic_img_size=True, **kwargs)
            except TypeError:
                self.backbone = None
        if self.backbone is None:
            self.backbone = timm.create_model(backbone_name, **kwargs)
        self.feature_dim = int(self.backbone.num_features)
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        if projection_version == "v2":
            self.projector = simclr_v2_head(self.feature_dim, proj_dim)
        elif projection_version == "v1":
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, proj_dim))
        else:
            raise ValueError("projection_version must be v1 or v2")

    def extract(self, x):
        """Return the pooled backbone feature [B, feature_dim]."""
        return self.backbone(x)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)                 # [B, feature_dim]
        logits = self.classifier(feats)
        if return_features:
            proj = F.normalize(self.projector(feats), p=2, dim=1)
            return logits, proj
        return logits
