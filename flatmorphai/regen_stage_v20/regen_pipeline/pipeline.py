# -*- coding: utf-8 -*-
"""Complete chain, from the original image to the regeneration state.

    from regen_pipeline import RegenerationPipeline
    pipe = RegenerationPipeline()                       # loads the models once (segmentation=False to skip Mask2Former)
    res = pipe.analyse(bgr)                             # or pipe.analyse(bgr, mask_m2f=mask) to start from a mask
    res = pipe.analyse_file("image.jpg", out_dir="out") # also writes the intermediate images and a JSON record

Order (each step reads what the production chain read, see README section 4):
  1 Segmenter.segment          Mask2Former two passes, CLAHE, worm chosen by circularity      -> mask_m2f
  2 Refiner.refine             blastema U-Net, smoothing, hull, opening (+ expert annotation)  -> mask_refined
  3-4 HeadCascade.anchor       skan axis, blastema object, tip by radius, eyes, darkness       -> head (r3)
  5 CompositeBuilder.build     whole worm, zoom, super-zoom CLAHE                              -> composite
  6 DescriptorComputer.compute 128 descriptors (86 used)                                        -> descriptors
  7 StageClassifier.predict    ConvNeXt V2 image branch + metric branch                        -> state
"""
import os, json, time
import numpy as np, cv2
from . import _env
from ._env import show_device, imread_color, imread_gray, CS

# Per-step wall-clock accumulators (diagnostic only, never change outputs). The
# worker reads these after a run to log where the per-image time goes (refine,
# cascade+skan, composite, descriptors, state). Summed across all images.
_STEP_TIMES: "dict[str, float]" = {}
_STEP_COUNT: "dict[str, int]" = {}


def _acc_step(name: str, dt: float) -> None:
    _STEP_TIMES[name] = _STEP_TIMES.get(name, 0.0) + float(dt)
    _STEP_COUNT[name] = _STEP_COUNT.get(name, 0) + 1


def step_stats() -> "dict[str, tuple]":
    """{step: (total_seconds, n_calls)} since the last reset."""
    return {k: (round(_STEP_TIMES[k], 2), _STEP_COUNT.get(k, 0)) for k in _STEP_TIMES}


def reset_step_stats() -> None:
    _STEP_TIMES.clear()
    _STEP_COUNT.clear()


class RegenerationPipeline:
    def __init__(self, segmentation=True, verbose=True):
        show_device()
        from .refinement import Refiner
        from .head_cascade import HeadCascade
        from .composite import CompositeBuilder
        from .descriptors import DescriptorComputer
        from .stage import StageClassifier
        self.segmenter = None
        if segmentation:
            from .segmentation import Segmenter
            self.segmenter = Segmenter(verbose)
        self.refiner = Refiner(verbose); self.cascade = HeadCascade(verbose); self.builder = CompositeBuilder()
        self.descr = DescriptorComputer(verbose); self.stage = StageClassifier(verbose)
        # Inference fast path for the U-Nets (inference_mode, repeated-crop
        # cache, cudnn autotuning). Outputs are unchanged; see fastpath.py.
        from .fastpath import install as _install_fastpath
        if _install_fastpath() and verbose:
            import torch as _t
            _mp = (os.environ.get("FLATMORPHAI_MAXPOOL_GPU", "1") == "1" and _t.cuda.is_available())
            print("inference fast path installed (shared U-Net calls cached%s)"
                  % (", peak-detection max-pool on GPU, bit-identical" if _mp else ""))

    def analyse(self, bgr, mask_m2f=None, annotated_bgr=None, name="image", mask_refined=None,
                full_composite=True):
        """bgr: original image (cv2). mask_m2f: optional Mask2Former mask (grey), computed when None.
        mask_refined: optional refined mask, skips steps 1 and 2 (used by the test against production).
        full_composite: False builds only the channels the state classifier reads (see CompositeBuilder.build);
        keep True when the composite PNG is written to disk.
        annotated_bgr: optional expert copy of the image with the blastema drawn in cyan.
        Returns a dict with the arrays (mask_m2f, mask_refined, composite) and the records of every step."""
        t0 = time.time(); out = dict(image=name, bgr=bgr)
        if mask_refined is None:
            if mask_m2f is None:
                if self.segmenter is None: raise ValueError("no mask given and segmentation disabled")
                _t = time.time()
                mask_m2f, seg_info = self.segmenter.segment(bgr); out["segmentation"] = seg_info
                _acc_step("segmentation", time.time() - _t)
            out["mask_m2f"] = mask_m2f
            _t = time.time()
            mask_refined, ref_info = self.refiner.refine(bgr, mask_m2f, annotated_bgr); out["refinement"] = ref_info
            _acc_step("refine (blastema U-Net)", time.time() - _t)
        out["mask_refined"] = mask_refined
        wm = CS.masque_ver(mask_refined)
        if wm.sum() < 300:
            out.update(status="mask under 300 px", composite=np.zeros((384, 384, 3), np.uint8)); return out
        _t = time.time()
        r3 = self.cascade.anchor(bgr, mask_refined); out["r3"] = r3; out["head"] = self.cascade.summary(r3)
        # The TWO ends of the worm, as the cascade itself sees them: each is a
        # (tip, framing centre) pair along the skeleton axis, and the cascade
        # keeps only the one it decided is the head. The two tips are what
        # head_position.csv is meant to carry, so they are passed up here --
        # otherwise the file could only show the head end twice over.
        try:
            _ends = self.cascade.axis(mask_refined).get("ends") or []
            # The tips the cascade itself kept, which are the CORRECTED ones.
            # When a blastema sits further from the skeleton's end than D_CORR
            # of the worm's length, the cascade recomputes that end on the
            # blastema instead of trusting the skeleton (``corrected`` = 1).
            # Reading the raw skeleton ends would throw that correction away
            # and put the tip back where the skeleton stopped.
            _tips = {}
            for _c in (r3 or {}).get("candidates", []) or []:
                _b = int(_c.get("end", -1))
                if _b in (0, 1) and (_b not in _tips or int(_c.get("corrected", 0))):
                    _tips[_b] = _c["tip"]
            out["ends"] = [(_tips.get(_j, _ends[_j][0] if _j < len(_ends) else None),
                            _ends[_j][1] if _j < len(_ends) else None)
                           for _j in (0, 1)]
            if any(e[0] is None for e in out["ends"]):
                out["ends"] = _ends or None
        except Exception:  # noqa: BLE001 - the ends are a bonus, not the run
            out["ends"] = None
        _acc_step("cascade (skan axis + eye U-Net)", time.time() - _t)
        _t = time.time()
        comp, cinfo = self.builder.build(bgr, mask_refined, r3, full=full_composite)
        _acc_step("composite", time.time() - _t)
        out["composite"] = comp; out["composite_info"] = cinfo
        _t = time.time()
        d, status = self.descr.compute(bgr, mask_refined, r3, name)
        _acc_step("descriptors", time.time() - _t)
        out["descriptors"] = d; out["descriptor_status"] = status
        if d is None:
            out["status"] = "descriptors: " + status; return out
        _t = time.time()
        out["stage"] = self.stage.predict(comp, d)
        _acc_step("state (ConvNeXt)", time.time() - _t)
        out["status"] = "ok"; out["seconds"] = round(time.time() - t0, 2)
        return out

    def analyse_file(self, path, out_dir=None, mask_path=None, annotation_path=None):
        """Reads the files as production did: cv2.imdecode for the segmentation, cv2.imread for steps 2 to 7."""
        bgr = imread_color(path)
        if bgr is None: raise FileNotFoundError(path)
        mm = imread_gray(mask_path) if mask_path else None
        an = imread_color(annotation_path) if annotation_path else None
        name = os.path.basename(path); seg_info = None
        if mm is None and self.segmenter is not None:
            mm, seg_info = self.segmenter.segment(imread_color(path, segmentation=True))
        res = self.analyse(bgr, mask_m2f=mm, annotated_bgr=an, name=name)
        if seg_info is not None: res["segmentation"] = seg_info
        if out_dir: self.write(res, out_dir)
        return res

    @staticmethod
    def overlay(res):
        """Control image: Mask2Former contour (blue), refined contour (green), tip and thumbnail centre retained by the
        cascade (red), head point (yellow), deciding step and state."""
        im = res["bgr"].copy(); t = max(2, int(round(max(im.shape[:2]) / 600)))
        for m, col in ((res.get("mask_m2f"), (255, 0, 0)), (res.get("mask_refined"), (0, 200, 0))):
            if m is None: continue
            cs, _ = cv2.findContours(((m >= 128) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, col, t)
        r3 = res.get("r3")
        if r3 is not None:
            py, px = map(int, np.round(r3["tip"])); cy, cx = map(int, np.round(r3["centre"]))
            cv2.circle(im, (px, py), 4 * t, (0, 0, 255), t); cv2.line(im, (px, py), (cx, cy), (0, 0, 255), t)
            cv2.drawMarker(im, (int(r3["x"]), int(r3["y"])), (0, 255, 255), cv2.MARKER_CROSS, 12 * t, t)
            txt = "%s" % r3["stage"]
            if res.get("stage"): txt += " | state %.2f -> %d" % (res["stage"]["stage_decimal"], res["stage"]["stage_class"])
            cv2.putText(im, txt, (10, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (0, 255, 255), t)
        return im

    @staticmethod
    def record(res):
        """JSON-serialisable record (no arrays)."""
        rec = dict(image=res["image"], status=res.get("status"))
        for k in ("segmentation", "refinement", "head", "composite_info", "stage"):
            if k in res and res[k] is not None:
                rec[k] = {a: (float(b) if isinstance(b, (np.floating,)) else int(b) if isinstance(b, (np.integer,)) else b)
                          for a, b in res[k].items()}
        if res.get("descriptors"):
            rec["descriptors"] = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v))
                                  for k, v in res["descriptors"].items()}
        return rec

    def write(self, res, out_dir):
        """Writes <stem>_mask_m2f.png, <stem>_mask_refined.png, <stem>_composite.png, <stem>_head.jpg (overlay) and
        <stem>.json. PNG is lossless: the masks written are exactly those used by the next steps."""
        os.makedirs(out_dir, exist_ok=True); stem = os.path.splitext(res["image"])[0]
        if res.get("mask_m2f") is not None: cv2.imwrite(os.path.join(out_dir, stem + "_mask_m2f.png"), res["mask_m2f"])
        if res.get("mask_refined") is not None: cv2.imwrite(os.path.join(out_dir, stem + "_mask_refined.png"), res["mask_refined"])
        if res.get("composite") is not None: cv2.imwrite(os.path.join(out_dir, stem + "_composite.png"), res["composite"])
        if res.get("bgr") is not None: cv2.imwrite(os.path.join(out_dir, stem + "_head.jpg"), self.overlay(res))
        with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(self.record(res), f, indent=1, ensure_ascii=False, default=str)
