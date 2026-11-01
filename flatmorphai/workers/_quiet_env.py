# -*- coding: utf-8 -*-
"""Silence noisy third-party import notices in the pipeline subprocesses.

The regeneration / eyespots / segmentation subprocesses pull in transformers,
huggingface_hub, requests and albumentations, which emit non-fatal notices that
clutter the pipeline log and alarm the operator, e.g.:
  * albumentations "A new version is available" update check;
  * requests ``RequestsDependencyWarning`` (mismatched charset_normalizer);
  * huggingface_hub "Error importing huggingface_hub.hf_api: ..." advisory.
None of these affect the results, so this module turns them off. It is imported
FIRST by each driver (before transformers / albumentations load) and uses only
the standard library. It applies ONLY inside the inference subprocesses.

Why the previous, simpler version did not work (and this one does):
  * setting a logger to ``ERROR`` does NOT hide ERROR-level records (ERROR is
    allowed at that level) -- the huggingface_hub notice is logged at ERROR, so
    it still printed.  Here we disable ERROR-and-below globally via
    ``logging.disable`` and also set ``HF_HUB_VERBOSITY=critical`` ;
  * ``warnings.filterwarnings("ignore")`` is undone when transformers /
    albumentations reset the warnings filters on import, so we ALSO replace
    ``warnings.showwarning`` with a no-op, which silences every warning's
    OUTPUT regardless of how the filters are later changed.
"""
import contextlib
import logging
import os
import warnings

# Env toggles must be set before the libraries import.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")      # no online version check
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "critical")       # hide hf_api import ERROR log
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# 1) Ignore all warnings now AND make the output path a no-op, so a library
#    resetting the filters later cannot bring the messages back.
warnings.filterwarnings("ignore")
try:
    warnings.showwarning = lambda *args, **kwargs: None  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    pass

# 2) Disable the logging module for everything up to and including ERROR
#    (only CRITICAL still passes). This is a global switch checked at emit
#    time, so it holds even if a library re-sets its own logger level during
#    import. The pipeline talks to the GUI through its own stdout protocol
#    (plain prints), not the logging module, so nothing of ours is lost.
logging.disable(logging.ERROR)

# 3) Belt and braces: push the known-chatty loggers above CRITICAL too.
for _name in (
    "huggingface_hub", "transformers", "albumentations", "requests",
    "urllib3", "timm", "filelock", "datasets",
):
    try:
        logging.getLogger(_name).setLevel(logging.CRITICAL + 1)
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def suppressed_output():
    """Silence stdout AND stderr at the file-descriptor level for a block.

    The model-loading code in the Export bundle (``thin_worm_patch``,
    ``setup_model_320``, ``loss_initialization``, the huggingface_hub
    "Error importing hf_api" line, ...) writes with plain ``print()`` and even
    C-level writes, which neither the logging nor the warnings machinery can
    intercept. Redirecting fds 1 and 2 to ``os.devnull`` around the load phase
    removes all of it. The driver's own ``LOG`` / ``PROGRESS`` / ``ERROR``
    protocol lines are emitted OUTSIDE this block, so they are unaffected; and
    the fds are restored in ``finally`` before any exception propagates, so a
    real load-time failure is still reported normally.
    """
    import sys
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except Exception:  # noqa: BLE001
        # If we cannot open devnull, run without suppression rather than fail.
        yield
        return
    saved1, saved2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        for _fd in (devnull, saved1, saved2):
            try:
                os.close(_fd)
            except Exception:  # noqa: BLE001
                pass
