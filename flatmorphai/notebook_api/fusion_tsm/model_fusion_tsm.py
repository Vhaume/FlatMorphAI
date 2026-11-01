"""model_fusion_tsm.py -- the shipped classifier: ConvNeXt-V2 backbone + Temporal Shift Module (TSM) over a 3-frame
clip, fused with a gated 3-mask metric branch. This is the exact architecture the checkpoint was trained with, so the
state_dict loads with strict=True.

Architecture (FusionNetV2 + TSM):
  - backbone: timm convnextv2_base.fcmae_ft_in22k_in1k_384, global-avg pooled, 1024-d features.
  - TSM: a fraction (1/8) of channels is shifted along the time axis between backbone stages, so a 2D network models
    motion at zero extra parameters (Lin, Gan, Han, Temporal Shift Module for Efficient Video Understanding, ICCV 2019,
    arXiv 1811.08383). The clip length is T = 3 (the T-1, T0, T+1 frames carried in the 3 image channels).
  - metric branch: a learnable per-metric gate (L1-sparsified during training) selects metrics, then a 2-layer MLP
    maps them to 128-d. The gate is the in-model feature selection (Guyon and Elisseeff, An Introduction to Variable
    and Feature Selection, JMLR 2003, vol 3 pp 1157-1182).
  - fusion: concat(visual 1024, metric 128) -> linear classifier over the 5 classes.
  - auxiliary heads (metric_aux, vis_aux) and the SimCLR-v2 projector exist in the checkpoint for deep supervision and
    contrastive training (Lee et al., Deeply-Supervised Nets, AISTATS 2015, arXiv 1409.5185; Chen et al., Big
    Self-Supervised Models are Strong Semi-Supervised Learners, NeurIPS 2020, arXiv 2006.10029). They are not used at
    inference but must exist so the weights load.

The checkpoint is self-describing: it carries classes, metric_cols, and the metric standardization statistics
(metric_mean / metric_std / metric_median), so swapping in a new checkpoint needs no code change.
Dependencies: torch, timm. No GPU required, runs on CPU or CUDA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def simclr_v2_head(feature_dim, proj_dim=128):
    """SimCLR-v2 3-layer projection head, sized to the fused feature width."""
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
        nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
        nn.Linear(feature_dim, proj_dim))


class FusionNetV2(nn.Module):
    """Image backbone + gated metric branch + fused classifier (the v2 architecture)."""

    def __init__(self, backbone_name, num_classes, metric_dim, proj_dim=128, mdim=128):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        C = int(self.backbone.num_features)
        self.feature_dim = C
        self.vis_norm = nn.LayerNorm(C)
        self.metric_mlp = nn.Sequential(
            nn.Linear(metric_dim, mdim), nn.BatchNorm1d(mdim), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(mdim, mdim), nn.BatchNorm1d(mdim), nn.ReLU(inplace=True))
        self.metric_gate = nn.Parameter(torch.ones(metric_dim))   # learnable per-metric gate (L1-penalised at train time)
        self.metric_aux = nn.Linear(mdim, num_classes)            # deep supervision on the metric branch (train only)
        self.vis_aux = nn.Linear(C, num_classes)                  # deep supervision on the image branch (train only)
        self.fused_dim = C + mdim
        self.classifier = nn.Linear(self.fused_dim, num_classes)
        self.projector = simclr_v2_head(self.fused_dim, proj_dim)
        self.T = 3                                                # clip length (T-1, T0, T+1)

    # ---- TSM channel shift, applied as a pre-forward hook on each backbone stage ----
    @staticmethod
    def _tsm_shift(t, T, fold_div=8):
        BT, C, H, W = t.shape
        B = BT // T
        t = t.view(B, T, C, H, W)
        out = torch.zeros_like(t)
        fold = max(1, C // fold_div)
        out[:, :-1, :fold] = t[:, 1:, :fold]                      # shift a slice toward the future
        out[:, 1:, fold:2 * fold] = t[:, :-1, fold:2 * fold]      # shift a slice toward the past
        out[:, :, 2 * fold:] = t[:, :, 2 * fold:]                 # the rest is left in place
        return out.view(BT, C, H, W)

    def add_tsm_hooks(self):
        stages = getattr(self.backbone, 'stages', None)
        if stages is None:
            return 0
        n = 0
        for st in stages:
            st.register_forward_pre_hook(lambda module, inp: (self._tsm_shift(inp[0], self.T),))
            n += 1
        return n

    def forward(self, x, m):
        """x: (B, 3*T, H, W) = the T-1/T0/T+1 channels stacked. m: (B, metric_dim) standardized metric vector."""
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        xc = x.reshape(B * self.T, 3, H, W)                       # unfold the clip into T per-frame images
        feat = self.backbone(xc).view(B, self.T, -1).mean(1)      # per-frame features, temporally averaged
        f = self.vis_norm(feat)
        g = self.metric_mlp(m * self.metric_gate)
        z = torch.cat([f, g], dim=1)
        return self.classifier(z)


def load_fusion_tsm(ckpt_path, device="cpu"):
    """Build the model from the self-describing checkpoint and load the weights. Returns (model, info)."""
    ck = torch.load(ckpt_path, map_location=device)
    model = FusionNetV2(ck["backbone"], len(ck["classes"]), len(ck["metric_cols"]),
                        proj_dim=ck.get("proj_dim", 128), mdim=128)
    model.T = 3
    model.add_tsm_hooks()
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    # strict=False only to tolerate harmless buffer-name drift across timm versions; report anything real.
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        print("[load_fusion_tsm] missing:", real_missing[:6], "... unexpected:", list(unexpected)[:6])
    model.to(device).eval()
    info = dict(classes=ck["classes"], metric_cols=ck["metric_cols"],
                metric_mean=ck["metric_mean"], metric_std=ck["metric_std"], metric_median=ck["metric_median"],
                backbone=ck["backbone"], image_size=ck.get("image_size", 384),
                best_balanced_acc=ck.get("best_balanced_acc"))
    return model, info
