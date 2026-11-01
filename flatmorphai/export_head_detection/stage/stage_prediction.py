# -*- coding: utf-8 -*-
"""Regeneration-state prediction (1 to 6, decimal value then class) from the head composite and the metrics.

Model (training/train_modele_production_stade.py, checkpoint parameters/models.csv: stage_classifier):
  * image: 384 x 384 three-channel composite (head/head_detection.py), of which the model stacks the channels
    given by ch_map of the checkpoint (whole+superzoom mode: c0, c2, c2), ImageNet normalisation, spatial
    transformer, ConvNeXt V2 base (timm convnextv2_base.fcmae_ft_in22k_in1k_384) ;
  * metrics: vector of the descriptors metric_cols of the checkpoint (97 columns of
    metrics/metriques_finales_IM002_r090.csv), missing values replaced by metric_median, then
    (v - metric_mean) / metric_std, learned gate, MLP 128 ;
  * output: decimal state = 1 + 5 x sigmoid(fusion head), uncertainty log-variance, and two auxiliary outputs
    (metrics only, image only). Class = rounding bounded to [1, 6].
The checkpoint carries everything needed for inference (backbone, ch_map, metric_cols, normalisation).

MODEL REPLACEMENT: drop the new .pth and change stage_classifier in models.csv. If the new model keeps the same keys
(state_dict, backbone, ch_map, metric_cols, metric_mean, metric_std, metric_median, stage_min, stage_max,
proj_metric_hid, drop_path), nothing else changes. Otherwise adapt the Net class below.

The metrics themselves are computed by the metrics/ chain (see README, section 7).
References: ConvNeXt V2, S. Woo et al., CVPR 2023, arXiv 2301.00808 ; spatial transformer, M. Jaderberg et al.,
NeurIPS 2015, arXiv 1506.02025 ; learned aleatoric uncertainty, A. Kendall, Y. Gal, NeurIPS 2017, arXiv 1703.04977.

Command line: python stage/stage_prediction.py composite.png metrics.csv   (";" CSV with one row, columns = descriptors)
"""
import os, sys, argparse, numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "..", "parameters") not in sys.path: sys.path.insert(0, os.path.join(_HERE, "..", "parameters"))
from parameter_reader import model_path
np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _STN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(nn.Conv2d(3, 16, 7, stride=2, padding=3), nn.ReLU(True), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(True), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(32 * 4 * 4, 64), nn.ReLU(True), nn.Linear(64, 6))
    def forward(self, x):
        th = self.fc(self.loc(x).flatten(1)).view(-1, 2, 3)
        g = F.affine_grid(th, x.size(), align_corners=False); return F.grid_sample(x, g, align_corners=False)


class Net(nn.Module):
    def __init__(self, backbone, metric_dim, proj=128, drop_path=0.1, stage_min=1.0, stage_max=6.0):
        super().__init__(); import timm
        self.stn = _STN(); self.smin, self.smax = stage_min, stage_max
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg", drop_path_rate=drop_path)
        C = int(self.backbone.num_features)
        self.vis_norm = nn.LayerNorm(C); self.metric_gate = nn.Parameter(torch.ones(metric_dim))
        self.metric_mlp = nn.Sequential(nn.Linear(metric_dim, proj), nn.BatchNorm1d(proj), nn.ReLU(True), nn.Dropout(0.3),
                                        nn.Linear(proj, proj), nn.BatchNorm1d(proj), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(C + proj, 64), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(64, 1))
        self.logvar_head = nn.Sequential(nn.Linear(C + proj, 64), nn.ReLU(True), nn.Linear(64, 1))
        self.metric_aux = nn.Linear(proj, 1); self.vis_aux = nn.Linear(C, 1)
    def _sq(self, t): return self.smin + (self.smax - self.smin) * torch.sigmoid(t)
    def forward(self, x, m):
        x = self.stn(x); f = self.vis_norm(self.backbone(x)); g = self.metric_mlp(m * self.metric_gate); z = torch.cat([f, g], 1)
        return (self._sq(self.head(z).squeeze(1)), self.logvar_head(z).squeeze(1), self._sq(self.metric_aux(g).squeeze(1)), self._sq(self.vis_aux(f).squeeze(1)))


class StageClassifier:
    def __init__(self, path=None, verbose=True):
        import timm
        ck = torch.load(path or model_path("stage_classifier"), map_location=DEVICE, weights_only=False)
        self.ck = ck; self.cols = list(ck["metric_cols"])
        self.mean = np.asarray(ck["metric_mean"], np.float32); self.std = np.asarray(ck["metric_std"], np.float32); self.med = np.asarray(ck["metric_median"], np.float32)
        self.ch_map = tuple(ck.get("ch_map", (0, 2, 2))); self.size = int(ck.get("image_size", 384))
        self.net = Net(ck["backbone"], len(self.cols), int(ck.get("proj_metric_hid", 128)), float(ck.get("drop_path", 0.1)),
                       float(ck.get("stage_min", 1.0)), float(ck.get("stage_max", 6.0))).to(DEVICE)
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        cfg = timm.data.resolve_data_config({}, model=self.net.backbone)
        self.mt = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(3, 1, 1); self.st = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(3, 1, 1)
        if verbose: print("state classifier on %s | %s | %d descriptors | channels %s | trained on %s" % (DEVICE, ck["backbone"], len(self.cols), self.ch_map, ck.get("date", "?")))

    def metric_vector(self, metrics):
        """dict or Series {descriptor: value} -> normalised vector (missing = training median). Returns (vector, missing names)."""
        def _f(c):
            try: x = metrics.get(c, np.nan) if hasattr(metrics, "get") else np.nan; return float(x) if x is not None else np.nan
            except (TypeError, ValueError): return np.nan
        v = np.array([_f(c) for c in self.cols], np.float32)
        missing = [c for c, x in zip(self.cols, v) if np.isnan(x)]
        v = np.where(np.isnan(v), self.med, v); return ((v - self.mean) / self.std).astype(np.float32), missing

    def predict(self, composite_bgr, metrics):
        """composite_bgr: 384 x 384 x 3 image (c0, c1, c2) from head/head_detection.py. Returns dict."""
        im = composite_bgr
        if im.shape[:2] != (self.size, self.size): im = cv2.resize(im, (self.size, self.size))
        i0, i1, i2 = self.ch_map; x = np.stack([im[:, :, i0], im[:, :, i1], im[:, :, i2]], 2)
        x = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32) / 255.0); x = ((x - self.mt) / self.st)[None].to(DEVICE)
        vm, missing = self.metric_vector(metrics); m = torch.from_numpy(vm)[None].to(DEVICE)
        with torch.no_grad(): mu, lv, rm, rv = self.net(x, m)
        s = float(mu.item())
        return dict(decimal_stage=s, stage_class=int(min(6, max(1, round(s)))), log_variance=float(lv.item()), std=float(np.exp(0.5 * float(lv.item()))),
                    stage_metrics_only=float(rm.item()), stage_image_only=float(rv.item()), missing_descriptors=missing)


def _cli():
    ap = argparse.ArgumentParser(description="Regeneration-state prediction"); ap.add_argument("composite"); ap.add_argument("metrics_csv"); a = ap.parse_args()
    c = StageClassifier(); im = cv2.imdecode(np.fromfile(a.composite, np.uint8), cv2.IMREAD_COLOR)
    d = pd.read_csv(a.metrics_csv, sep=";").iloc[0].to_dict(); r = c.predict(im, d)
    print("decimal state %.3f | class State_%d | std %.3f | metrics only %.2f | image only %.2f | missing descriptors %d" % (
        r["decimal_stage"], r["stage_class"], r["std"], r["stage_metrics_only"], r["stage_image_only"], len(r["missing_descriptors"])))


if __name__ == "__main__": _cli()
