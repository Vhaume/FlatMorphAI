"""
Synchronous runners used by :class:`PipelineWorker`.

Each function here mirrors a step worker's ``run`` logic but takes
``log``, ``progress`` and ``is_interrupted`` callables so it can be
composed inside the pipeline worker. Heavy imports and CUDA checks are
performed exactly once per step.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict

from ..constants import (
    CHECKPOINTS_SEGMENTATION_DIR,
    CLASSES_NAMES_STEREOTYPE,
)
from ..model_cache import ModelCache
from ..notebook_api.classification_images import (
    build_processed_dir_name,
    generate_stereotypie_image_for_classification,
)
from ..notebook_api.forecast import process_videos
from ..notebook_api.periods import (
    compute_grouped_statistics,
    fuse_periods_by_subject,
)
# Step 2 segmentation now runs in a subprocess (segmentation_a_pipeline.py), so
# the heavy segmentation module and its CUDA/bundle imports are NOT pulled into
# the GUI process here.
from ..notebook_api.video_frames import (
    count_videos_in_directory,
    process_all_videos_in_directory,
)
from .step1_worker import Step1Params
from .step2_worker import Step2Params
from .step3_worker import Step3Params
from .step4_worker import Step4Params
from .step6_worker import Step6Params


Log = Callable[[str], None]
Progress = Callable[[int], None]
Interrupt = Callable[[], bool]


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
def run_step1(
    params: Step1Params,
    log: Log,
    progress: Progress,
    is_interrupted: Interrupt,
) -> Dict[str, Any]:
    if not os.path.isdir(params.input_directory):
        raise FileNotFoundError(
            f"Input directory does not exist: {params.input_directory}"
        )
    total_videos = count_videos_in_directory(params.input_directory)
    if total_videos == 0:
        raise FileNotFoundError(
            f"No videos under {params.input_directory}"
        )
    log(f"Step 1: {total_videos} video(s) -> {params.output_directory}")

    state = {"done": 0, "total": total_videos, "frames": 0, "total_frames": 0}

    def _on_video_done(done: int, total: int) -> None:
        state["done"] = done
        state["total"] = total
        _emit()

    def _on_frame(done: int, total: int, video_name: str) -> None:
        state["frames"] = done
        state["total_frames"] = total
        _emit()
        if total and done == total:
            log(f"Finished {video_name} ({done} frames)")

    def _emit() -> None:
        total = state["total"]
        if total <= 0:
            return
        fractional = 0.0
        if state["total_frames"] > 0:
            fractional = state["frames"] / state["total_frames"]
        completed = (state["done"] + fractional) / max(1, total)
        progress(int(completed * 100))

    total_frames_written = process_all_videos_in_directory(
        input_directory=params.input_directory,
        output_directory=params.output_directory,
        start_second=params.start_second,
        end_second=params.end_second,
        step_seconds=params.step_seconds,
        log=log,
        progress_cb=_on_video_done,
        per_frame_cb=_on_frame,
        is_interrupted=is_interrupted,
        workers=int(getattr(params, "max_workers", 0) or 0),
    )
    progress(100)
    return {
        "total_videos": total_videos,
        "total_frames_written": total_frames_written,
        "output_directory": params.output_directory,
    }


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
def run_step2(
    params: Step2Params,
    log: Log,
    progress: Progress,
    is_interrupted: Interrupt,
) -> Dict[str, Any]:
    """Run Mask2Former segmentation in a DEDICATED SUBPROCESS.

    Segmentation used to run in-process on the pipeline worker thread; doing so
    put a CUDA context and the ``evaluate_test_images_v2`` bundle inside the Qt
    GUI process, which crashed the application with a segmentation fault (the
    GUI event loop sharing a CUDA context with Mask2Former on a secondary
    thread). The heavy work now runs in a fresh subprocess - exactly like the
    Image segmentation, regeneration and blastema chains - so the GUI never holds a CUDA
    context. The child runs the SAME ``forecast_mask_with_a_mask2former``, so the
    masks are identical; only the isolation changed.
    """
    import json as _json
    import subprocess
    import sys
    import tempfile

    model_name = params.model_name
    if not model_name:
        raise ValueError("Step 2: model_name is required")

    folders = list(params.folders_stereotypie)
    if not folders:
        if not os.path.isdir(params.processing_root):
            raise FileNotFoundError(
                f"Processing root does not exist: {params.processing_root}"
            )
        folders = sorted(
            name for name in os.listdir(params.processing_root)
            if os.path.isdir(os.path.join(params.processing_root, name, "images"))
        )
    if not folders:
        raise FileNotFoundError(
            f"No video folders with images/ under {params.processing_root}"
        )

    workers_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(workers_dir))

    cancel_file = os.path.join(
        tempfile.gettempdir(), "flatmorphai_step2_cancel_%d" % os.getpid()
    )
    try:
        if os.path.exists(cancel_file):
            os.remove(cancel_file)
    except OSError:
        pass

    blasteme_refined = bool(getattr(params, "blasteme_refined", False))

    if blasteme_refined:
        # Refinement with the blastema mask is produced by the proven Image segmentation
        # subprocess, which runs the same two-pass Mask2Former and then grows the
        # full-body mask with the blastema prediction, writing the result to the
        # ``masks/`` folder exactly where the stereotype classifier reads it.
        try:
            from ..export_bundle import resolve_export_root
            export_root = resolve_export_root()
        except Exception:  # noqa: BLE001
            export_root = os.path.join(app_root, "Export_head_detection")
        if not os.path.isdir(export_root):
            raise FileNotFoundError(
                "Blastema refinement needs the Export_head_detection bundle: "
                + export_root
            )
        driver = os.path.join(workers_dir, "segmentation_c_pipeline.py")
        seg_checkpoint = (
            os.path.join(os.path.abspath(params.checkpoint_root), model_name)
            if params.checkpoint_root else ""
        )
        cfg = {
            "root": os.path.abspath(params.processing_root),
            "export_root": os.path.abspath(export_root),
            "cancel_file": cancel_file,
            "seg_checkpoint": seg_checkpoint,
            "seg_model": model_name,
            "blasteme_refined": True,
            "blasteme_keep_biggest": True,
            "overwrite": bool(params.overwrite),
            "save_body_masks": True,
            "save_blasteme_masks": False,
            "save_blasteme_thumbnails": False,
            "save_head_csv": False,
            "save_body_thumbnails": False,
        }
    else:
        driver = os.path.join(workers_dir, "segmentation_a_pipeline.py")
        cfg = {
            "app_root": app_root,
            "cancel_file": cancel_file,
            "processing_root": os.path.abspath(params.processing_root),
            "checkpoint_root": (
                os.path.abspath(params.checkpoint_root)
                if params.checkpoint_root else ""
            ),
            "model_name": model_name,
            "folders": folders,
            "batch_size": int(params.batch_size),
            "two_pass": bool(params.two_pass),
            "two_pass_context_fill": bool(params.two_pass_context_fill),
            "two_pass_model_size": int(params.two_pass_model_size),
            "two_pass_merge": str(params.two_pass_merge),
            "keep_largest_component": bool(params.keep_largest_component),
            "overwrite": bool(params.overwrite),
        }
    cfg_fh = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    _json.dump(cfg, cfg_fh)
    cfg_fh.close()

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log("Starting segmentation (separate process)")

    # The segmentation_a driver imports ``flatmorphai`` from the app root; the
    # segmentation_c driver uses bare imports from its own (workers) directory.
    proc = subprocess.Popen(
        [sys.executable, driver, cfg_fh.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(driver) if blasteme_refined else app_root,
        env=env,
        text=True,
        bufsize=1,
    )

    total = 0
    error_msg = None
    done = False
    # The child prints ~90 lines of model-construction trace on every run.
    # Most of it is silenced at the source (the driver wraps the model build
    # in suppressed_output); the few lines printed during the run are dropped
    # here and counted, so the operator is told how many were hidden.
    from . import log_noise as _noise
    hidden = [0]

    def _cancel() -> None:
        try:
            open(cancel_file, "w").close()
        except OSError:
            pass

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            if is_interrupted():
                _cancel()
            if line.startswith("PROGRESS "):
                try:
                    _, d, t = line.split()
                    progress(int(100.0 * float(d) / float(t)) if float(t) else 0)
                except ValueError:
                    pass
            elif line.startswith("LOG "):
                if _noise.drop(line[4:]):
                    hidden[0] += 1
                else:
                    log(line[4:])
            elif line.startswith("DONE"):
                done = True
                parts = line.split()
                if len(parts) > 1:
                    try:
                        total = int(parts[1])
                    except ValueError:
                        total = 0
            elif line.startswith("ERROR "):
                error_msg = line[6:]
                log("ERROR: " + error_msg)
            elif _noise.drop(line):
                hidden[0] += 1
            else:
                log(line)
    finally:
        if hidden[0]:
            log(_noise.summary(hidden[0]))
        # If the user cancelled and the child is still alive (e.g. stuck in a
        # long model load that has not reached a cancel checkpoint), terminate
        # it so no orphan GPU process survives.
        try:
            if is_interrupted() and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    proc.kill()
        except Exception:  # noqa: BLE001
            pass
        proc.wait()
        for path in (cfg_fh.name, cancel_file):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    if is_interrupted():
        log("Segmentation cancelled by the user")
        return {
            "mask_directory_name": "masks",
            "model_name": model_name,
            "total_processed": total,
        }
    if error_msg:
        raise RuntimeError("Step 2 segmentation failed: " + error_msg)
    if proc.returncode != 0 and not done:
        raise RuntimeError(
            "Step 2 segmentation subprocess exited with code %s"
            % (proc.returncode,)
        )

    progress(100)
    return {
        "mask_directory_name": "masks",
        "model_name": model_name,
        "total_processed": total,
    }


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def run_step3(
    params: Step3Params,
    log: Log,
    progress: Progress,
    is_interrupted: Interrupt,
) -> Dict[str, Any]:
    mask_dirs = [params.mask_dir] if params.mask_dir else None
    # Two Step 3 modes:
    #   * temporal_mode = True (set automatically by Stereotype classification when
    #     the selected classifier is the fusion + TSM model): write
    #     the 3-channel TEMPORAL image (channel 0 = T-1 masked
    #     grayscale, channel 1 = T0, channel 2 = T+1) per labelled
    #     frame.  The output directory carries the ``_temporal_``
    #     token so the Step 4 resolver can find it.
    #   * temporal_mode = False (legacy path): write the standard
    #     single-frame mask-applied colour image.
    # Without this branch the runner used by Stereotype classification would always
    # call the standard generator regardless of the page-level
    # ``temporal_mode`` flag -- that was the operator's reported bug
    # ("formatted colored image" still on disk even when TSM was
    # selected).  Reference for derive-don't-duplicate (single
    # source of truth for the temporal flag): Sandve, G.K. et al.,
    # "Ten Simple Rules for Reproducible Computational Research",
    # PLOS Comp Biol, 2013, DOI:10.1371/journal.pcbi.1003285, rule 1.
    progress_cb = lambda done, total, folder: progress(
        int(done / total * 100) if total else 0
    )
    if getattr(params, "temporal_mode", False):
        from ..notebook_api.classification_images import (
            build_temporal_dir_name,
            generate_temporal_images_for_classification,
        )
        log(
            f"Step 3 (temporal mode) running: image_size={params.image_size}, "
            f"margin={params.margin_percent}, "
            f"apply_mask={params.apply_mask}, "
            f"overwrite={getattr(params, 'overwrite', False)}"
        )
        total_written = generate_temporal_images_for_classification(
            path_dataset=params.processing_root,
            name_stereotypie_dir=params.name_stereotypie_dir,
            name_mask_dir=mask_dirs,
            image_size=params.image_size,
            margin_percent=params.margin_percent,
            apply_mask=params.apply_mask,
            folders_stereotypie_to_process=(
                params.folders_stereotypie_to_process or None
            ),
            overwrite=getattr(params, "overwrite", False),
            log=log,
            progress_cb=progress_cb,
            is_interrupted=is_interrupted,
        )
        processed_dir_name = build_temporal_dir_name(
            params.name_stereotypie_dir,
            params.image_size,
            params.margin_percent,
            params.apply_mask,
        )
    else:
        total_written = generate_stereotypie_image_for_classification(
            path_dataset=params.processing_root,
            name_stereotypie_dir=params.name_stereotypie_dir,
            name_mask_dir=mask_dirs,
            number_of_planaria_on_image=params.number_of_planaria_on_image,
            image_size=params.image_size,
            margin_percent=params.margin_percent,
            # Fixed to 1: the planaria-rows parameter was dropped with
            # the v1.2 naming convention because the arena position is
            # already encoded in the filename (see Step3Params docstring).
            number_rows_of_planaria_on_image=1,
            folders_stereotypie_to_process=(
                params.folders_stereotypie_to_process or None
            ),
            apply_mask=params.apply_mask,
            black_and_white=params.black_and_white,
            overwrite=getattr(params, "overwrite", False),
            log=log,
            progress_cb=progress_cb,
            is_interrupted=is_interrupted,
            max_workers=getattr(params, "max_workers", 0) or None,
        )
        processed_dir_name = build_processed_dir_name(
            params.name_stereotypie_dir,
            params.image_size,
            params.margin_percent,
            params.black_and_white,
            params.apply_mask,
        )
    progress(100)
    return {
        "processed_dir_name": processed_dir_name,
        "total_written": total_written,
    }


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------
def run_step4(
    params: Step4Params,
    log: Log,
    progress: Progress,
    is_interrupted: Interrupt,
) -> Dict[str, Any]:
    # LightGBM is CPU-only, so Step 4 no longer requires CUDA.
    # ``device`` is kept for API parity with the legacy loaders that
    # still branch on ``torch.device('cuda')`` when available.
    import torch
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    from ..notebook_api.lightgbm_bundle import resolve_bundle_dir
    from ..notebook_api.classification import load_lightgbm_bundle

    # LightGBM is the only classifier.  A missing or incomplete
    # bundle raises: there is no legacy model to fall back to.
    bundle_dir_resolved = resolve_bundle_dir(
        params.bundle_dir or None,
    )
    model_name = os.path.basename(
        os.path.normpath(bundle_dir_resolved)
    ) or "LightGBM_bundle"
    cache_key = f"lightgbm::{bundle_dir_resolved}"
    cache = ModelCache()
    if cache.has(cache_key):
        log("Using cached LightGBM bundle")
        model, proto, _classes, data_model, data_parameters = cache.get(cache_key)
    else:
        log(
            "Loading LightGBM bundle "
            f"from {bundle_dir_resolved}"
        )
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

    from datasets import get_orig_transform, get_valid_transform  # type: ignore
    input_shape = data_model.get("input_shape", [224, 224])
    orig_tfms = get_orig_transform(input_shape)
    test_tfms = get_valid_transform(input_shape)

    videos = params.videos_to_process
    if not videos:
        if not os.path.isdir(params.processing_root):
            raise FileNotFoundError(
                f"Processing root does not exist: {params.processing_root}"
            )
        videos = sorted(
            name for name in os.listdir(params.processing_root)
            if os.path.isdir(os.path.join(params.processing_root, name))
        )

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
        # 4-second isolated-class smoothing (Settings page knob,
        # default True).  See classification._apply_4s_isolation_correction
        # for the rule.
        correction_4s_enabled=getattr(
            params, "correction_4s_enabled", True,
        ),
        max_workers=getattr(params, "max_workers", 0) or None,
        log=log,
        # ``done/total`` drives the percentage bar; the per-batch
        # "Batch N/M" log line is emitted by classification.process_videos.
        progress_cb=lambda done, total, label="": progress(
            int(done / total * 100) if total else 0
        ),
    )
    progress(100)

    # Unload the classifier once Step 4 completes -- Step 6 does not
    # reload it, so releasing the ~2 GB VRAM footprint here keeps
    # the pipeline light for downstream work.  Same rationale as in
    # step4_worker.Step4Worker.run().
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    log("Classifier unloaded and GPU cache emptied.")

    class_counts = {}
    if isinstance(forecast_summary, dict):
        class_counts = dict(forecast_summary.get("class_counts", {}))
    return {
        "model_name": model_name,
        "videos": videos,
        "class_counts": class_counts,
    }


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------
def run_step6(
    params: Step6Params,
    log: Log,
    progress: Progress,
    is_interrupted: Interrupt,
) -> Dict[str, Any]:
    out_dir = params.output_directory or params.processing_root
    output_csv_name = params.output_csv_name
    grouped_csv_name = params.grouped_csv_name

    df = fuse_periods_by_subject(
        processing_root=params.processing_root,
        output_directory=out_dir,
        output_csv_name=output_csv_name,
        new_planaria_start_index=params.new_planaria_index,
        log=log,
        progress_cb=lambda done, total: progress(
            int(done / total * 50) if total else 0
        ),
    )
    if df is None:
        raise RuntimeError(
            "Step 6 found no per-video CSVs under "
            f"{params.processing_root}.  This usually means Step 4 "
            "wrote nothing, most often because Step 3's image_size did "
            "not match the active classifier's expected input shape "
            "(the Step 4 resolver looks for a sub-folder containing "
            "'_image-size(<model_input>)').  Check the per-video "
            "WARNING lines emitted by Step 4 in the log above to see "
            "which path was probed, then re-run Step 3 with the right "
            "size or pick a classifier whose input shape matches the "
            "existing Step 3 output."
        )

    periods_csv_path = os.path.join(out_dir, output_csv_name)
    try:
        compute_grouped_statistics(
            periods_csv_path=periods_csv_path,
            class_names=CLASSES_NAMES_STEREOTYPE,
            class_type="Stereotype",
            output_directory=out_dir,
            output_csv_name=grouped_csv_name,
            log=log,
        )
    except Exception as exc:  # noqa: BLE001
        # The grouped summary is optional; do not fail the pipeline if
        # the merged CSV column layout differs from the legacy one.
        log(
            f"Grouped statistics skipped ({type(exc).__name__}: {exc})"
        )
        grouped_csv_name = ""
    # Aggregate per-video metrics CSVs (ConvNeXt-V2 sidecar files)
    # into a single all_videos_metrics.csv next to all_videos.csv.
    # The per-video file is named ``<video_name>_metrics.csv`` and is
    # produced by classification.process_videos when the active
    # classifier is a fusion model.  When no video produced a metrics
    # CSV (e.g. a ConvNeXt-V2-only run) this step quietly
    # writes nothing.  Reference for the "tidy long" layout (one row
    # per frame, one column per metric): Wickham, H., "Tidy Data",
    # Journal of Statistical Software 59(10), 2014, DOI:
    # 10.18637/jss.v059.i10.
    metrics_out_name = ""
    try:
        import pandas as _pd
        _frames = []
        for entry in sorted(os.listdir(params.processing_root)):
            full = os.path.join(params.processing_root, entry)
            if not os.path.isdir(full):
                continue
            mp = os.path.join(full, f"{entry}_metrics.csv")
            if not os.path.isfile(mp):
                continue
            try:
                df_m = _pd.read_csv(mp, sep=";", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                log(
                    f"  Could not read {mp}: {type(exc).__name__}: {exc}"
                )
                continue
            df_m.insert(0, "video_name", entry)
            _frames.append(df_m)
        if _frames:
            out_metrics_path = os.path.join(out_dir, "all_videos_metrics.csv")
            df_all_metrics = _pd.concat(_frames, ignore_index=True)
            df_all_metrics.to_csv(
                out_metrics_path, sep=";", index=False, encoding="utf-8",
            )
            log(
                f"Wrote {out_metrics_path} "
                f"({len(df_all_metrics)} row(s) from "
                f"{len(_frames)} video(s))"
            )
            metrics_out_name = "all_videos_metrics.csv"
        # No message when there is nothing to aggregate. The per-video
        # *_metrics.csv files belonged to the fusion / TSM deployment bundle,
        # which the LightGBM classifier replaced; the stereotype chain never
        # writes them any more, so the former notice fired at every single run
        # and named a classifier that is no longer part of the application.
    except Exception as exc:  # noqa: BLE001
        log(
            f"  all_videos_metrics.csv aggregation skipped "
            f"({type(exc).__name__}: {exc})"
        )

    progress(100)
    return {
        "periods_csv": output_csv_name,
        "grouped_csv": grouped_csv_name,
        "metrics_csv": metrics_out_name,
        "rows": int(len(df)),
    }
