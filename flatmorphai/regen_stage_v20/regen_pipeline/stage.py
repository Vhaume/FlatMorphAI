# -*- coding: utf-8 -*-
"""Step 7. Regeneration state classifier v20 (two branches: image head and metric head).

The classes _STN and Net are read from production/train_modele_production_stade_v20.py and executed unchanged.
Architecture: spatial transformer (STN, initialised to the identity) on the composite, backbone
convnextv2_base.fcmae_ft_in22k_in1k_384 (timm, global average pooling, drop path 0.1) followed by a LayerNorm (image
branch), metric gate (one learned weight per descriptor) then MLP 86 -> 128 -> 128 with BatchNorm, ReLU and dropout 0.3
(metric branch), concatenation, head 64 -> 1 whose sigmoid is rescaled to [1, 6] (decimal state P), head of the log
variance (aleatoric uncertainty), auxiliary heads of each branch. Input image: composite channels (0, 2, 2) (whole
worm, super-zoom, super-zoom), scaled to [0, 1] and normalised by the timm mean and std of the backbone. Input metrics:
the 86 descriptors of metric_cols, NaN replaced by metric_median, standardised by metric_mean and metric_std (all stored
in the checkpoint). Class = clip(floor(P + 0.5), 1, 6) (CLASS_RULE floor_half, as evaluation_pli_test_stade_v22).
Measured (19 October 2026): frozen test split, 399 images of 48 worms, exact 0.744, balanced 0.698, within 0.5 state
0.852, within 1 state 0.965. Worm-grouped 5-fold cross-validation, 2196 images: exact 0.7427, balanced 0.7054, within
0.5 state 0.8365, within 1 state 0.959. The checkpoint in models/ is trained on all 2196 images.

References (verified). Woo S., Debnath S., Hu R., Chen X., Liu Z., Kweon I. S., Xie S. (2023). ConvNeXt V2: Co-designing
and Scaling ConvNets with Masked Autoencoders. CVPR 2023, arXiv 2301.00808. Jaderberg M., Simonyan K., Zisserman A.,
Kavukcuoglu K. (2015). Spatial Transformer Networks. NeurIPS 2015, arXiv 1506.02025. Kendall A., Gal Y. (2017). What
Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NeurIPS 2017, arXiv 1703.04977. Ba J. L.,
Kiros J. R., Hinton G. E. (2016). Layer Normalization. arXiv 1607.06450. Ioffe S., Szegedy C. (2015). Batch
Normalization. ICML 2015, arXiv 1502.03167. Huang G., Sun Y., Liu Z., Sedra D., Weinberger K. (2016). Deep Networks with
Stochastic Depth. ECCV 2016, arXiv 1603.09382 (drop path).
"""
import numpy as np, cv2, torch, timm
import torch.nn as nn
import torch.nn.functional as F
from . import _env
from ._env import DEVICE, read, model_path, exec_slice, show_device, check_on_device

PROD_FILE = "train_modele_production_stade_v22.py"


class StageClassifier:
    def __init__(self, verbose=True):
        show_device()
        self.P = read("stage")
        path = model_path("stage_classifier")
        c = torch.load(path, map_location="cpu", weights_only=False)   # numpy arrays inside: full load, own file
        self.ck = c
        ns = dict(np=np, torch=torch, nn=nn, F=F, timm=timm, BACKBONE=c["backbone"], DROP_PATH=float(c["drop_path"]),
                  METRIC_DIM=int(c["metric_dim"]), PROJ_METRIC_HID=int(c["proj_metric_hid"]),
                  STAGE_MIN=float(c["stage_min"]), STAGE_MAX=float(c["stage_max"]))
        exec_slice(PROD_FILE, "class _STN(nn.Module):", "huber = nn.SmoothL1Loss", ns)
        self.net = ns["Net"](pretrained=False).to(DEVICE)
        self.net.load_state_dict(c["state_dict"]); self.net.eval(); check_on_device(self.net, "state classifier")
        cfg = timm.data.resolve_data_config({}, model=self.net.backbone)
        self.mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(3, 1, 1)
        self.std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(3, 1, 1)
        self.ch_map = tuple(c["ch_map"]); self.size = int(c["image_size"]); self.cols = list(c["metric_cols"])
        self.m_mean = np.asarray(c["metric_mean"], np.float64); self.m_std = np.asarray(c["metric_std"], np.float64)
        self.m_med = np.asarray(c["metric_median"], np.float64)
        if verbose:
            print("state classifier %s | %s | channels %s | %d descriptors | trained %s on %s images"
                  % (path, c["backbone"], self.ch_map, len(self.cols), c.get("date"), c.get("n_images_entrainement")))

    def image_tensor(self, composite_bgr):
        """Composite as written by cv2 (or read by cv2.imdecode) -> normalised tensor 3 x 384 x 384 (load_comp of training)."""
        im = composite_bgr
        if im.shape[:2] != (self.size, self.size): im = cv2.resize(im, (self.size, self.size))
        i0, i1, i2 = self.ch_map
        im = np.stack([im[:, :, i0], im[:, :, i1], im[:, :, i2]], axis=2)
        x = torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return (x - self.mean) / self.std

    def metric_tensor(self, descriptors):
        v = np.array([float(descriptors.get(k, np.nan)) if descriptors.get(k) is not None else np.nan for k in self.cols], np.float64)
        n_missing = int(np.isnan(v).sum())
        v = np.where(np.isnan(v), self.m_med, v)
        s = self.m_std.copy(); s[s == 0] = 1.0
        return torch.from_numpy(((v - self.m_mean) / s).astype(np.float32)), n_missing

    def to_class(self, P):
        if str(self.P["CLASS_RULE"]) == "rint": return int(np.clip(np.rint(P), 1, 6))
        return int(np.clip(np.floor(P + 0.5), 1, 6))

    def predict(self, composite_bgr, descriptors):
        """Returns dict: stage_decimal P, stage_class, sigma (exp(logvar / 2)), image-branch and metric-branch stages,
        number of missing descriptors (replaced by the training median)."""
        x = self.image_tensor(composite_bgr)[None].to(DEVICE)
        m, n_missing = self.metric_tensor(descriptors); m = m[None].to(DEVICE)
        with torch.no_grad():
            mu, lv, aux_m, aux_v = self.net(x, m)
        P = float(mu.cpu()[0])
        return dict(stage_decimal=P, stage_class=self.to_class(P), sigma=float(torch.exp(0.5 * lv).cpu()[0]),
                    stage_metric_branch=float(aux_m.cpu()[0]), stage_image_branch=float(aux_v.cpu()[0]),
                    missing_descriptors=n_missing)
