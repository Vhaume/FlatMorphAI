"""
The 97 descriptors of the regeneration-state classifier, image by image.

The export's ``metrics/`` scripts are notebook exports: every one of them
defines per-image functions and then runs a dataset-level loop on
hard-coded paths.  Rewriting those functions would create a second source
of truth, so this module loads each script through the AST, executes only
its imports, class and function definitions and literal constants, and
injects the few globals the functions expect (``device``, the model
instances, the peak thresholds).  The per-image functions are then called
exactly as the training chain did.

Descriptor families and their origin (export README, section 7):

====================  =======================================  ==============================
family prefix          per-image function                       script
====================  =======================================  ==============================
shp_ pig_ end_ tex_    ``compute_metrics(vig384, EYE_PARAMS,     precompute_regeneration_metrics_v2
eye_(present, pair,    head_high)`` on the 384 masked image
max_contrast, area)
eyeb_ blast_(forme)
cl_ clh_ loc_
eye_n eye_pupil ...    ``infer_image`` (first eye U-Net, 3 ch)  eye_inference_and_features
pigp_                  ``head_features`` (+ first eye U-Net)    generate_composite_and_features
blast_prop/rim/...     ``features`` (no model)                  enrich_metrics_blasteme
blast_seg_*            ``seg_feats`` (metrics blastema U-Net)   enrich_metrics_blasteme_seg
eyeun_                 ``features`` (eye detector in service)   enrich_metrics_yeux_unet_v3
pup_                   ``feats`` (eye detector in service)      enrich_metrics_pupille
====================  =======================================  ==============================

``blast_seg_prop_yr`` is the blastema proportion centred and reduced per
acquisition year with statistics of the training table, which is not part
of the export.  It is therefore left missing and the classifier imputes
its training median, as its ``metric_vector`` is designed to do.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import export_bridge as EB

# Substrings that mark a top-level assignment as dataset-level (never run).
_UNSAFE = (
    "glob(", "os.path.join", "torch.load", "read_csv", "open(", "listdir",
    "smp.Unet", ".to(device)", "environ", "load_state_dict", "datetime.",
    "makedirs", "sys.argv", "concat(", "DataFrame(",
)


def _load_script(path: str, inject: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute the imports, defs and literal constants of a notebook export."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    ns: Dict[str, Any] = {"__name__": "_regen_" + os.path.splitext(os.path.basename(path))[0]}
    if inject:
        ns.update(inject)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            try:
                exec(compile(ast.Module([node], []), path, "exec"), ns)
            except Exception:  # noqa: BLE001  (optional imports such as sklearn)
                pass
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(src, node) or ""
            if any(u in seg for u in _UNSAFE):
                continue
            try:
                exec(compile(ast.Module([node], []), path, "exec"), ns)
            except Exception:  # noqa: BLE001
                pass
    if inject:
        ns.update(inject)  # injected values win over script defaults
    return ns


class DescriptorEngine:
    """Loads the metric scripts and their models once, then computes the
    descriptor dictionary of one image from the original BGR image, the
    raw Mask2Former mask and the head point found by the cascade."""

    def __init__(self, log: Optional[Callable[[str], None]] = None):
        self._log = log or (lambda _m: None)
        root = EB.activate()
        import torch  # noqa: WPS433
        import segmentation_models_pytorch as smp  # noqa: WPS433

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        met = os.path.join(root, "metrics")
        AC = EB.module("anchor_cascade")
        self.AC = AC
        self.pre = EB.module("preprocessing")
        mdl_bl_service, mdl_eye_service = AC.load_models(verbose=False)

        # ---- 1. base metrics (no model) -----------------------------------
        self.ns_base = _load_script(os.path.join(met, "precompute_regeneration_metrics_v2.py"))
        self.EYE_PARAMS = self.ns_base.get("EYE_PARAMS", {"k_frac": 0.06, "n_sigma": 2.2, "halo_k": 0.3, "dark_k": 0.4})

        # ---- 2. first eye U-Net (3 channels, 256) shared by two scripts -----
        inj = {"device": self.device, "torch": torch}
        self.ns_eyeinf = _load_script(os.path.join(met, "eye_inference_and_features.py"), inj)
        ck_path = EB.model_path("metrics_eye_unet_v1")
        ck = torch.load(ck_path, map_location=self.device)
        unet_v1 = self.ns_eyeinf["UNet"]().to(self.device)
        unet_v1.load_state_dict(ck["state_dict"]); unet_v1.eval()
        v1_globals = {
            "model": unet_v1, "device": self.device,
            "PEAK_THR": float(ck.get("peak_thr", 0.30)), "MINDIST": int(ck.get("peak_mindist", 10)),
        }
        self.ns_eyeinf.update(v1_globals)
        self.ns_comp = _load_script(os.path.join(met, "generate_composite_and_features.py"), dict(inj, **v1_globals))
        self.ns_comp.update(v1_globals)
        import cv2  # noqa: WPS433
        self.cv2 = cv2
        self.ns_comp["_clahe"] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ---- 3. classical blastema (no model) -------------------------------
        self.ns_blast = _load_script(os.path.join(met, "enrich_metrics_blasteme.py"))

        # ---- 4. metrics blastema segmenter (its own checkpoint) ------------
        self.ns_bseg = _load_script(os.path.join(met, "enrich_metrics_blasteme_seg.py"), inj)
        net = smp.Unet(encoder_name="timm-efficientnet-b0", encoder_weights=None, in_channels=4, classes=1, activation=None)
        bseg = self.ns_bseg["Backbone"](net).to(self.device)
        ckb = torch.load(EB.model_path("metrics_blastema_unet_384"), map_location=self.device)
        bseg.load_state_dict({k: v for k, v in ckb["state_dict"].items() if not k.startswith("_enc.")}); bseg.eval()
        self.ns_bseg.update({"model": bseg, "device": self.device})

        # ---- 5 and 6. eye detector in service (eyeun_, pup_) ----------------
        self.ns_v3 = _load_script(os.path.join(met, "enrich_metrics_yeux_unet_v3.py"), inj)
        self.ns_v3.update({"mdl": mdl_eye_service, "device": self.device})
        self.ns_pup = _load_script(os.path.join(met, "enrich_metrics_pupille.py"), inj)
        self.ns_pup.update({"mdl": mdl_eye_service, "device": self.device})

        self.columns: List[str] = []
        try:
            import pandas as pd  # noqa: WPS433
            self.columns = list(pd.read_csv(os.path.join(met, "metriques_finales_IM002_r090.csv"), sep=";")["descripteur"])
        except Exception:  # noqa: BLE001
            pass
        self._log(f"descriptor engine ready on {self.device} ({len(self.columns)} target descriptors)")

    # ------------------------------------------------------------------
    @staticmethod
    def rendu(bgr: np.ndarray, mk01: np.ndarray, size: int = 384) -> np.ndarray:
        """384 masked image of rebuild_processed_images_full_size (background 0, bbox, centred square)."""
        import cv2
        out = bgr.copy(); out[mk01 == 0] = 0
        ys, xs = np.nonzero(mk01)
        c = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = c.shape[:2]; s = max(h, w)
        sq = np.zeros((s, s, 3), np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
        return cv2.resize(sq, (size, size))

    @staticmethod
    def head_high(vig: np.ndarray, mk01: np.ndarray, hx: float, hy: float) -> Optional[bool]:
        """head_high flag of head_anchor_pass: head point projected on the main
        axis of the 384 masked image, True when it falls in the upper half."""
        import cv2
        ys, xs = np.nonzero(mk01)
        if ys.size == 0:
            return None
        y0, x0 = int(ys.min()), int(xs.min())
        h = int(ys.max() - y0 + 1); w = int(xs.max() - x0 + 1)
        s = max(h, w); oy = (s - h) // 2; ox = (s - w) // 2; k = vig.shape[0] / float(s)
        vx, vy = (hx - x0 + ox) * k, (hy - y0 + oy) * k
        g = cv2.cvtColor(vig, cv2.COLOR_BGR2GRAY)
        m = (g > 8).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        yy, xx = np.nonzero(m)
        if yy.size < 10:
            return None
        pts = np.stack([xx, yy], 1).astype(np.float64); c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False); axis = vt[0]
        proj = (pts - c) @ axis
        p = (((np.array([vx, vy]) - c) @ axis) - proj.min()) / (np.ptp(proj) + 1e-9)
        return bool(p > 0.5)

    # ------------------------------------------------------------------
    def compute(self, bgr: np.ndarray, raw_mask: np.ndarray,
                head_xy_original: Optional[tuple] = None) -> Dict[str, float]:
        """All descriptor families for one image.  ``raw_mask`` is the
        Mask2Former mask (0/255).  Missing values are left as NaN."""
        cv2 = self.cv2
        out: Dict[str, float] = {}
        mk01 = self.pre.worm_mask(raw_mask)
        if int(mk01.sum()) < 60:
            return out
        mimg = (mk01 * 255).astype(np.uint8)

        # 1. base metrics on the 384 masked image, anchored by the cascade
        vig = self.rendu(bgr, mk01)
        hh = None
        if head_xy_original is not None:
            hh = self.head_high(vig, mk01, float(head_xy_original[0]), float(head_xy_original[1]))
        try:
            m = self.ns_base["compute_metrics"](vig, eye_params=self.EYE_PARAMS, head_high=hh)
            if m:
                out.update({k: float(v) for k, v in m.items()})
        except Exception as exc:  # noqa: BLE001
            self._log(f"  base metrics failed: {type(exc).__name__}: {exc}")

        # 2. eye_* (first U-Net), same body as infer_image without file I/O
        try:
            ns = self.ns_eyeinf; torch = self.torch
            oc = ns["orient_crop"](bgr, mimg)
            vals = [0.0] * 8
            if oc is not None:
                rimg, crop256, tr = oc; g = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
                x = torch.from_numpy(cv2.cvtColor(crop256, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(self.device)
                with torch.no_grad():
                    hm = ns["model"](x).cpu()
                feats = []
                for (px, py) in ns["peaks"](hm[0]):
                    rx, ry = ns["to_rframe"](px, py, tr); f = ns["eye_feats_at"](g, rx, ry)
                    if f:
                        feats.append(f)
                if feats:
                    mean = np.mean(feats, 0); vals = [float(len(feats))] + [float(v) for v in mean]
            out.update(dict(zip(ns["FCOLS"], vals)))
        except Exception as exc:  # noqa: BLE001
            self._log(f"  eye_* failed: {type(exc).__name__}: {exc}")

        # 3. pigp_* (composite script), same body as its dataset loop
        try:
            ns = self.ns_comp
            if int(mk01.sum()) >= 300:
                rbgr, rm = ns["orient"](bgr, mk01); rg = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY)
                c256, tr = ns["crop256"](rbgr, rm); eyes = ns["detect_eyes"](c256)
                reyes = [ns["eye_to_rframe"](px, py, tr) for (px, py) in eyes]
                hx, hy, span = ns["head_tip"](rg, rm)
                ax_, ay_ = (np.mean([e[0] for e in reyes]), np.mean([e[1] for e in reyes])) if reyes else (hx, hy)
                ys, _ = np.nonzero(rm); ymid = (ys.min() + ys.max()) / 2; head_top = ay_ < ymid
                prof = ns["head_features"](rg, rm, head_top, reyes)
            else:
                prof = [0.0] * len(ns["PCOLS"])
            out.update(dict(zip(ns["PCOLS"], [float(v) for v in prof])))
        except Exception as exc:  # noqa: BLE001
            self._log(f"  pigp_* failed: {type(exc).__name__}: {exc}")

        # 4. classical blastema proxies
        try:
            out.update({k: float(v) for k, v in self.ns_blast["features"](bgr, mimg).items()})
        except Exception as exc:  # noqa: BLE001
            self._log(f"  blast_* failed: {type(exc).__name__}: {exc}")

        # 5. blastema segmenter of the metrics chain
        try:
            out.update({k: float(v) for k, v in self.ns_bseg["seg_feats"](bgr, mimg).items()})
            out["blast_seg_prop_yr"] = float("nan")  # year statistics not exported, imputed by the classifier
        except Exception as exc:  # noqa: BLE001
            self._log(f"  blast_seg_* failed: {type(exc).__name__}: {exc}")

        # 6. eyeun_* and 7. pup_* with the eye detector in service
        try:
            r = self.ns_v3["features"](bgr, mimg)
            if r:
                out.update({k: float(v) for k, v in r.items()})
        except Exception as exc:  # noqa: BLE001
            self._log(f"  eyeun_* failed: {type(exc).__name__}: {exc}")
        try:
            out.update({k: float(v) for k, v in self.ns_pup["feats"](bgr, mimg).items()})
        except Exception as exc:  # noqa: BLE001
            self._log(f"  pup_* failed: {type(exc).__name__}: {exc}")
        return out
