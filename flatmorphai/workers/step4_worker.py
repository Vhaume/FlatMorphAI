"""
Worker for Step 4: forecast a stereotype class for each planarian image.

The default path routes inference through the consolidated
FlatMorphAI deployment bundle
(``FlatMorphAI_deployment/bundle``), which replaces the previous
ConvNeXt-V2 loaders.  A user-provided
``bundle_dir`` from Settings takes precedence over the default
sibling layout; when the bundle is unavailable the worker falls
back to the legacy checkpoint-based loader for backwards
compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..base_worker import BaseWorker
from ..model_cache import ModelCache
from ..notebook_api.csv_metadata import augment_all_forecast_csvs
from ..notebook_api.forecast import process_videos


@dataclass
class Step4Params:
    processing_root: str
    # LightGBM bundle folder (see
    # ``lightgbm_stereotype_classifier``).  Empty means "use the
    # sibling folder shipped with the application".  This is the
    # ONLY classifier: the legacy neural checkpoint path, its
    # prototype pickle and its mini-batch size were all removed.
    bundle_dir: str = ""
    num_workers: int = 4
    # Worker threads for the LightGBM descriptor computation AND the
    # booster-level ``num_threads``.  0 means "auto" =
    # cpu_count() - 1.  Shared with Step 3 through the single
    # ``step3_max_workers`` setting so the operator tunes one knob.
    max_workers: int = 0
    # 4-second isolated-class smoothing.  When True (default),
    # process_videos rewrites the 2nd / 3rd slot of every sliding
    # 4-frame window in the per-video results CSV when the three
    # remaining frames in that window unanimously agree on a single
    # class.  The Settings page exposes ``step4_4s_correction`` so
    # the operator can turn it off when the labelling task tolerates
    # single-frame blips.
    correction_4s_enabled: bool = True
    videos_to_process: List[str] = field(default_factory=list)


class Step4Worker(BaseWorker):
    """Run classification inference and write per-video stereotype CSVs."""

    def __init__(self, params: Step4Params) -> None:
        super().__init__()
        self.params = params

    def _progress_cb(self, done: int, total: int, label: str = "") -> None:
        if total > 0:
            self.progress(int(done / total * 100))

    def run(self) -> None:
        # LightGBM is CPU-only; a missing CUDA is no longer fatal.
        device = self.resolve_device()
        if device is None:
            try:
                import torch as _torch
                device = _torch.device("cpu")
            except Exception:  # noqa: BLE001
                device = None

        params = self.params
        from ..notebook_api.lightgbm_bundle import resolve_bundle_dir

        videos = params.videos_to_process
        if not videos:
            if not os.path.isdir(params.processing_root):
                self.error_signal.emit(
                    f"Processing root does not exist: {params.processing_root}"
                )
                return
            videos = sorted(
                name for name in os.listdir(params.processing_root)
                if os.path.isdir(os.path.join(params.processing_root, name))
            )

        # LightGBM is the only classifier.  A missing / incomplete
        # bundle is a hard error: there is nothing to fall back to.
        from ..notebook_api.classification import load_lightgbm_bundle
        try:
            bundle_dir_resolved = resolve_bundle_dir(
                params.bundle_dir or None,
            )
        except FileNotFoundError as exc:
            self.error_signal.emit(str(exc))
            return
        model_name = os.path.basename(
            os.path.normpath(bundle_dir_resolved)
        ) or "LightGBM_bundle"
        cache_key = f"lightgbm::{bundle_dir_resolved}"
        cache = ModelCache()
        self.log(
            "Step 4 starting with LightGBM bundle "
            f"'{bundle_dir_resolved}' on {device}"
        )
        if cache.has(cache_key):
            self.log("Using cached LightGBM bundle")
            model, proto, _classes, data_model, data_parameters = cache.get(cache_key)
        else:
            self.log("Loading LightGBM bundle...")
            model, proto, data_model, data_parameters = load_lightgbm_bundle(
                device=device,
                bundle_dir=bundle_dir_resolved,
                num_threads=getattr(params, "max_workers", 0) or None,
            )
            cache.set(
                cache_key,
                (
                    model, proto,
                    list(data_model.get("dataset_classes", [])),
                    data_model, data_parameters,
                ),
            )

        try:
            from datasets import get_orig_transform, get_valid_transform  # type: ignore
        except ImportError as exc:
            self.error_signal.emit(
                f"Cannot import datasets module: {exc}"
            )
            return

        input_shape = data_model.get("input_shape", [224, 224])
        orig_tfms = get_orig_transform(input_shape)
        test_tfms = get_valid_transform(input_shape)

        try:
            forecast_summary = process_videos(
                videos_to_process=videos,
                path_videos_frame_by_frame=params.processing_root,
                model=model,
                proto=proto,
                data_model=data_model,
                data_parameters=data_parameters,
                model_name=model_name,
                class_type="Stereotype",
                orig_tfms=orig_tfms,
                test_tfms=test_tfms,
                device=device,
                num_workers=params.num_workers,
                correction_4s_enabled=getattr(
                    params, "correction_4s_enabled", True,
                ),
                max_workers=getattr(params, "max_workers", 0) or None,
                log=self.log,
                progress_cb=self._progress_cb,
            )
        except Exception as exc:  # noqa: BLE001
            try:
                import torch
                if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
                    self.error_signal.emit("CUDA OOM: reduce batch size")
                    return
            except Exception:
                pass
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")
            return

        # Enrich every per-video CSV with the position / compound /
        # concentration / period / second / total_seconds / step_seconds
        # columns recovered from the image filenames. See
        # ``flatmorphai.notebook_api.csv_metadata`` for the rationale
        # and the supporting references (FAIR Principles 2016, Ten
        # Simple Rules 2013).
        try:
            rewritten = augment_all_forecast_csvs(
                processing_root=params.processing_root,
                separator=";",
            )
            if rewritten:
                self.log(
                    f"Enriched {len(rewritten)} CSV(s) with naming-convention columns"
                )
        except Exception as exc:  # noqa: BLE001
            self.log(f"CSV enrichment skipped ({type(exc).__name__}: {exc})")

        self.progress(100)
        self.log("Step 4 finished")

        # ----------------------------------------------------------
        # Unload the classifier from the ModelCache + free GPU RAM.
        # The classifier is not needed after Step 4 emits results
        # (Step 5 / 6 do not reload it), so releasing it here frees
        # up to ~2 GB of VRAM for any downstream work in the same
        # session.  Reference for the "release once no consumer
        # remains" rule: Wilson G. et al., "Good Enough Practices
        # in Scientific Computing", PLOS Comp Biol 13(6):e1005510,
        # 2017, DOI:10.1371/journal.pcbi.1005510, rule 5 (release
        # resources deterministically).
        try:
            cache.clear(cache_key)
        except Exception:  # noqa: BLE001
            pass
        try:
            del model, proto, data_model, data_parameters
        except Exception:  # noqa: BLE001
            pass
        try:
            import gc
            gc.collect()
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
        self.log(
            "Classifier unloaded and GPU cache emptied."
        )

        class_counts = {}
        if isinstance(forecast_summary, dict):
            class_counts = dict(forecast_summary.get("class_counts", {}))
        self.result_signal.emit(
            {
                "model_name": model_name,
                "videos": videos,
                "class_counts": class_counts,
                "csv_name_template": f"results_Stereotype_[{model_name}].csv",
            }
        )
