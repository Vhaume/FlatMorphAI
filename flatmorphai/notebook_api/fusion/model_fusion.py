"""model_fusion.py -- ConvNeXt-V2 image + 3-mask metric fusion model (v2 architecture).
Dependencies: torch, timm. load_fusion_model(ckpt) returns (model, info)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def simclr_v2_head(feature_dim, proj_dim=128):
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
        nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
        nn.Linear(feature_dim, proj_dim))


class FusionNetV2(nn.Module):
    def __init__(self, backbone_name, num_classes, metric_dim, proj_dim=128, mdim=128):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        C = int(self.backbone.num_features)
        self.vis_norm = nn.LayerNorm(C)
        self.metric_mlp = nn.Sequential(
            nn.Linear(metric_dim, mdim), nn.BatchNorm1d(mdim), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(mdim, mdim), nn.BatchNorm1d(mdim), nn.ReLU(inplace=True))
        self.metric_aux = nn.Linear(mdim, num_classes)
        self.classifier = nn.Linear(C + mdim, num_classes)
        self.projector = simclr_v2_head(C + mdim, proj_dim)

    def forward(self, x, m):
        f = self.vis_norm(self.backbone(x))
        g = self.metric_mlp(m)
        return self.classifier(torch.cat([f, g], dim=1))


def load_fusion_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    model = FusionNetV2(ck["backbone"], len(ck["classes"]), len(ck["metric_cols"]), proj_dim=ck.get("proj_dim", 128))
    model.load_state_dict(ck["state_dict"], strict=False)
    model.to(device).eval()
    info = dict(classes=ck["classes"], metric_cols=ck["metric_cols"],
                metric_mean=ck["metric_mean"], metric_std=ck["metric_std"], metric_median=ck["metric_median"],
                backbone=ck["backbone"], image_size=ck.get("image_size", 384))
    return model, info
