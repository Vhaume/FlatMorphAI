# -*- coding: utf-8 -*-
"""Driver for Stereotype classification / B - Step 2 segmentation (runs in a subprocess).

Why a subprocess.  The Mask2Former segmentation imports the self-contained
``evaluate_test_images_v2`` bundle and creates a CUDA context.  Running that
inside the Qt GUI process -- on a secondary QThread, alongside the event loop
and whatever CUDA state the app already holds -- has been observed to crash the
application with a segmentation fault (the bundle's bare, path-based imports can
also pick up the wrong module from a populated ``sys.modules``).  A fresh
subprocess gets an empty ``sys.modules`` and its own CUDA context, exactly like
the Image segmentation / regeneration / blastema drivers, so the GUI never shares a CUDA
context with the model.  The worker cancels the run by creating ``cancel_file``;
this process polls it between batches and terminates cleanly.

Protocol (stdout, line-buffered), matching the other subprocess drivers:
  ``PROGRESS <done> <total>``   -- progress tick
  ``LOG <message>``             -- a log line to surface in the GUI panel
  ``DONE <total_processed>``    -- success, with the frame count
  ``ERROR <message>``           -- failure
It runs the SAME ``forecast_mask_with_a_mask2former`` the application used
in-process, so the masks it writes are bit-for-bit what Stereotype classification/B expect.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# Imported FIRST, before transformers / albumentations / requests load: it
# turns off their non-fatal notices (update check, RequestsDependencyWarning,
# the huggingface_hub "Error importing hf_api" advisory) and provides
# suppressed_output() for the model build below. The blastema-refined driver
# has done this from the start; this one printed them all.
try:
    from . import _quiet_env                      # imported as a package
except ImportError:                               # started as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _quiet_env                             # type: ignore

# Silence Albumentations' online update check before the bundle imports it.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

# PyTorch allocator. ``expandable_segments:True`` reduces fragmentation and is
# what this project's production scripts use, but it routes allocations through
# the CUDA virtual-memory API, which is where a failure surfaces as the
# driver-level "CUDA driver error: out of memory" instead of the allocator's
# recoverable "CUDA out of memory". That path has been seen to fail under WSL,
# so it is NOT forced here: set FLATMORPHAI_CUDA_ALLOC_CONF (for example
# "expandable_segments:True", or "max_split_size_mb:128") to choose it, and the
# plain PyTorch default applies otherwise.
_alloc_conf = os.environ.get("FLATMORPHAI_CUDA_ALLOC_CONF", "").strip()
if _alloc_conf:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc_conf)


def _emit(prefix: str, msg: str = "") -> None:
    try:
        sys.stdout.write(prefix + msg + "\n")
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    if len(sys.argv) < 2:
        _emit("ERROR ", "no config file given")
        return 1
    try:
        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        _emit("ERROR ", "cannot read config: %r" % (exc,))
        return 1

    app_root = cfg.get("app_root", "")
    if app_root and app_root not in sys.path:
        sys.path.insert(0, app_root)

    cancel_file = cfg.get("cancel_file", "")

    def _log(message) -> None:
        _emit("LOG ", str(message))

    def _progress(done, total, _video="") -> None:
        try:
            _emit("PROGRESS ", "%d %d" % (int(done), int(total)))
        except Exception:  # noqa: BLE001
            pass

    def _interrupted() -> bool:
        return bool(cancel_file and os.path.exists(cancel_file))

    try:
        import torch

        if not torch.cuda.is_available():
            _emit("ERROR ", "No CUDA device detected. Step 2 requires a GPU.")
            return 1
        device = torch.device("cuda")

        from flatmorphai.notebook_api.segmentation import (
            forecast_mask_with_a_mask2former,
            load_model_for_segmentation,
        )

        model_name = cfg["model_name"]
        _log("Loading segmentation model '%s' (separate process)" % model_name)
        load_kwargs = {"device": device}
        if cfg.get("checkpoint_root"):
            load_kwargs["checkpoint_root"] = cfg["checkpoint_root"]
        # The construction of the patched Mask2Former prints its loss banner,
        # its ThinWorm report and its 320x320 setup -- about eighty lines,
        # identical at every run, with plain print() and C-level writes. They
        # are silenced at the file-descriptor level for the duration of the
        # build only: this driver's own LOG lines are emitted outside the
        # block, and the descriptors are restored before any exception
        # propagates, so a real load failure is still reported.
        # FLATMORPHAI_VERBOSE=1 keeps everything.
        if os.environ.get("FLATMORPHAI_VERBOSE", "0") in ("1", "true", "True"):
            model, processor = load_model_for_segmentation(model_name, **load_kwargs)
        else:
            with _quiet_env.suppressed_output():
                model, processor = load_model_for_segmentation(model_name, **load_kwargs)

        total = forecast_mask_with_a_mask2former(
            directory_all_images=cfg["processing_root"],
            model=model,
            processor=processor,
            device=device,
            model_name=model_name,
            folders_stereotypie=cfg.get("folders") or None,
            batch_size=int(cfg.get("batch_size", 8)),
            two_pass=bool(cfg.get("two_pass", True)),
            two_pass_context_fill=bool(cfg.get("two_pass_context_fill", True)),
            two_pass_model_size=int(cfg.get("two_pass_model_size", 320)),
            two_pass_merge=str(cfg.get("two_pass_merge", "skeleton_guided")),
            keep_largest_component=bool(cfg.get("keep_largest_component", True)),
            overwrite=bool(cfg.get("overwrite", False)),
            log=_log,
            progress_cb=_progress,
            is_interrupted=_interrupted,
        )
        # Free the CUDA context before the process exits.
        try:
            del model, processor
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        _emit("DONE ", str(int(total)))
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit("ERROR ", "%r" % (exc,))
        traceback.print_exc()
        # The success path above already drops the model and empties the
        # cache; do the same after a failure, so an out-of-memory error does
        # not leave its footprint on the card for the next attempt. Same
        # release the GUI performs when the operator cancels a run.
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import mask_refine

            mask_refine.release_gpu_on_crash(
                log=lambda m: _emit("LOG ", m))
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
