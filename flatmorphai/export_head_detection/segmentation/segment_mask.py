# -*- coding: utf-8 -*-
"""Worm segmentation with Mask2Former, two passes with CLAHE enhancement (parameters/clahe_segmentation.csv).

Reproduces exactly the path of regenerate_masks_2pass (reference notebook, copied in this folder):
  * CLAHE enhancement on the L channel (Lab) when ENABLE_CLAHE, clipLimit CLAHE_CLIP, tile CLAHE_TILE ;
  * first pass on the whole image, worm-class probability rebuilt from the logits (sum over the queries of
    softmax(classes) x sigmoid(masks)), threshold WORM_THRESHOLD (0.50 = argmax) ;
  * second pass on a zoomed crop of each component (MODEL_SIZE, PAD_RATIO, ...), skeleton-guided fusion
    (evaluate_test_images_v2), largest connected component.
The model is loaded by mask2former_api.load_model_for_segmentation (standalone copy of
flatmorphai.notebook_api.segmentation) from parameters/models.csv (mask2former_segmentation).

References: Cheng et al., Mask2Former, CVPR 2022, arXiv 2112.01527 ; Zuiderveld, CLAHE, Graphics Gems IV,
1994, DOI 10.1016/B978-0-12-336156-1.50061-6.

Command line: python segmentation/segment_mask.py image.jpg [image2.jpg ...] --out folder
Module: seg = Segmenter() ; mask = seg.segment(bgr)   # uint8 0/255 mask, same size as the image
A GPU is required in practice.
"""
import os, sys, argparse, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image as PILImage
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "parameters")):
    if _p not in sys.path: sys.path.insert(0, _p)
from parameter_reader import read, model_path
from evaluate_test_images_v2 import (_extract_p2_jobs, _paste_p2_result, _apply_fusion, _keep_largest_component, _maybe_suppress_stdout)
np.random.seed(42); torch.manual_seed(42)


import contextlib as _ctxlib


@_ctxlib.contextmanager
def _silenced(on=True):
    """Hide the model-building chatter (loss / ThinWorm / setup banners), which
    is printed with plain print() and C-level writes: redirect fd 1 and 2."""
    if not on:
        yield
        return
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    nul = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(nul, 1); os.dup2(nul, 2)
        yield
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for f in saved + [nul]:
            os.close(f)


class Segmenter:
    def __init__(self, device=None, verbose=True):
        from mask2former_api import load_model_for_segmentation
        self.P = read("clahe_segmentation")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder = model_path("mask2former_segmentation"); name = os.path.basename(folder.rstrip("/\\"))
        with _silenced(not verbose):
            self.model, self.processor = load_model_for_segmentation(name, self.device, checkpoint_root=os.path.dirname(folder))
        self.model.eval()
        if not verbose:   # no "FIRST FORWARD PASS" debug banner either
            for _m in self.model.modules():
                try:
                    _m._first_forward_done = True
                except Exception:
                    pass
        if verbose:
            print("segmenter %s on %s | CLAHE %s (clip %s, tile %s) | worm threshold %.2f | two passes %s" % (
                name, next(self.model.parameters()).device, self.P["ENABLE_CLAHE"], self.P["CLAHE_CLIP"], self.P["CLAHE_TILE"], float(self.P["WORM_THRESHOLD"]), self.P["TWO_PASSES"]))
            if self.device.type != "cuda": print("   WARNING: CPU, segmentation is very slow")

    def enhance(self, rgb):
        if not self.P["ENABLE_CLAHE"]: return rgb
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=float(self.P["CLAHE_CLIP"]), tileGridSize=(int(self.P["CLAHE_TILE"]),) * 2).apply(L)
        out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2RGB)
        if self.P["SATURATION"]:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32); hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out

    def _prob_batch(self, images, worm_thr):
        if not images: return []
        sizes = [(im.shape[0], im.shape[1]) for im in images]
        inp = self.processor(images=[PILImage.fromarray(im) for im in images], return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad(): outputs = self.model(**inp)
        mql = outputs.masks_queries_logits if hasattr(outputs, "masks_queries_logits") else outputs["masks_queries_logits"]
        cql = outputs.class_queries_logits if hasattr(outputs, "class_queries_logits") else outputs["class_queries_logits"]
        masks = mql.sigmoid(); classes = cql.softmax(-1)[..., :-1]
        seg = torch.einsum("bqc,bqhw->bchw", classes, masks); out = []
        for i, (H, W) in enumerate(sizes):
            s = F.interpolate(seg[i:i + 1].float(), size=(H, W), mode="bilinear", align_corners=False)[0]
            p = s / (s.sum(0, keepdim=True) + 1e-6); fg = p[1:].sum(0)
            out.append((fg >= worm_thr).cpu().numpy().astype(np.uint8) * 255)
        return out

    def segment_batch(self, bgr_list):
        """List of BGR images -> list of uint8 0/255 masks."""
        P = self.P; thr = float(P["WORM_THRESHOLD"]); batch = int(P["BATCH"])
        rgb = [self.enhance(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in bgr_list]; n = len(rgb)
        p1 = self._prob_batch(rgb, thr)
        if not P["TWO_PASSES"]:
            return [(_keep_largest_component(m, k=255)[0] if P["KEEP_LARGEST"] else m) for m in p1]
        all_jobs = []; jobs_per = [[] for _ in range(n)]
        for i in range(n):
            j = _extract_p2_jobs(rgb[i], p1[i], i, float(P["PAD_RATIO"]), int(P["MODEL_SIZE"]), int(P["MIN_COMP"]), int(P["MIN_PAD"]), float(P["MIN_ROI"]))
            all_jobs.extend(j); jobs_per[i] = j
        p2res = []
        if all_jobs:
            crops = [j["crop"] for j in all_jobs]
            for s in range(0, len(crops), batch): p2res.extend(self._prob_batch(crops[s:s + batch], thr))
        p2full = [np.zeros(rgb[i].shape[:2], np.uint8) for i in range(n)]
        for job, cm in zip(all_jobs, p2res): _paste_p2_result(cm, p2full[job["image_idx"]], job)
        for i in range(n):
            if p2full[i].sum() == 0 and p1[i].sum() > 0: p2full[i] = p1[i].copy()
        out = []
        for i in range(n):
            cs = jobs_per[i][0]["crop"].shape if jobs_per[i] else None
            with _maybe_suppress_stdout(False):
                fused = _apply_fusion(p1[i], p2full[i], str(P["MERGE"]), float(P["BND_DIL"]), cs, False, None, float(P["MERGE_DIL_F"]))
            if P["KEEP_LARGEST"]:
                with _maybe_suppress_stdout(False): fused, _ = _keep_largest_component(fused, k=255)
            out.append(fused)
        return out

    def segment(self, bgr): return self.segment_batch([bgr])[0]


def _cli():
    ap = argparse.ArgumentParser(description="Mask2Former two-pass masks with CLAHE"); ap.add_argument("images", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); seg = Segmenter(); os.makedirs(a.out, exist_ok=True)
    for i in range(0, len(a.images), int(seg.P["BATCH"])):
        names = a.images[i:i + int(seg.P["BATCH"])]
        ims = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in names]
        for p, m in zip(names, seg.segment_batch(ims)):
            cv2.imwrite(os.path.join(a.out, os.path.basename(p)), m); print("mask written:", os.path.basename(p), "| area", int((m > 0).sum()))


if __name__ == "__main__": _cli()
