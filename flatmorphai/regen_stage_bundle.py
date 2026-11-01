"""
Local vendoring of the ``Export_regeneration_stage_v20`` bundle.

The upgraded regeneration chain (skan2 skeleton axis, head cascade v3c13 with
border-radius tip correction, blastema refinement, the two-branch state
classifier reading 86 descriptors) is shipped as a self-contained bundle,
``Export_regeneration_stage_v20``, imported through
``sys.path.insert(0, <root>); from regen_pipeline import RegenerationPipeline``.

As with the head-detection bundle, the application keeps its OWN local copy
inside the package (``flatmorphai/regen_stage_v20``) so the chain travels with
the code and does not depend on the external folder.
:func:`resolve_regen_stage_root` returns the local copy when complete, else the
sibling folder; :func:`vendor_regen_stage_bundle` performs the copy (run it
from ``vendor_regen_stage_bundle.py`` at the application root).
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))          # .../flatmorphai
_APP_ROOT = os.path.dirname(_PKG_DIR)                          # .../<root>

#: Local, in-package copy that travels with the code on export.
LOCAL_BUNDLE_DIR = os.path.join(_PKG_DIR, "regen_stage_v20")
#: Default external copy, a sibling of the ``flatmorphai`` package.
SIBLING_BUNDLE_DIR = os.path.join(_APP_ROOT, "Export_regeneration_stage_v20")

#: Code sub-packages the chain imports (regen_pipeline + the production sources
#: and parameters its _env.py puts on sys.path).
CODE_ENTRIES = ("regen_pipeline", "production", "segmentation", "parameters")
#: Model-weights folder (~1.2 GiB). Needed at runtime, copied only on request.
MODELS_ENTRY = "models"
#: Small top-level files kept with the vendored copy.
TOPLEVEL_FILES = ("README.md", "requirements.txt", "run_pipeline.py", "test_export.py")


def _has_code(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "regen_pipeline", "pipeline.py"))


def _has_models(root: str) -> bool:
    md = os.path.join(root, MODELS_ENTRY)
    try:
        return os.path.isdir(md) and any(os.scandir(md))
    except OSError:
        return False


def is_complete(root: str) -> bool:
    """True when ``root`` carries both the code and the model weights."""
    return _has_code(root) and _has_models(root)


def resolve_regen_stage_root() -> str:
    """Return the bundle root to use.

    Prefers the local in-package copy when it is COMPLETE (code + weights), so a
    code-only backup never shadows a working external bundle. Falls back to the
    sibling ``Export_regeneration_stage_v20`` otherwise.
    """
    if is_complete(LOCAL_BUNDLE_DIR):
        return LOCAL_BUNDLE_DIR
    return SIBLING_BUNDLE_DIR


def available(root: Optional[str] = None) -> bool:
    """True when a usable bundle (code at least) is resolvable."""
    return _has_code(root or resolve_regen_stage_root())


def _ignore(_dir, names):
    return [n for n in names if n in ("__pycache__", ".ipynb_checkpoints")]


def vendor_regen_stage_bundle(
    source: Optional[str] = None,
    dest: Optional[str] = None,
    include_models: bool = False,
    log: Callable[[str], None] = print,
) -> str:
    """Copy the needed ``Export_regeneration_stage_v20`` files into a local folder.

    ``include_models`` also copies the (large) ``models/`` weights, making the
    local copy a complete, self-contained bundle usable for export.
    """
    source = os.path.abspath(source or SIBLING_BUNDLE_DIR)
    dest = os.path.abspath(dest or LOCAL_BUNDLE_DIR)
    if not _has_code(source):
        raise FileNotFoundError(
            "Source Export_regeneration_stage_v20 bundle not found or incomplete: "
            + source
        )
    if os.path.abspath(source) == os.path.abspath(dest):
        raise ValueError("Source and destination are the same folder.")
    os.makedirs(dest, exist_ok=True)

    for entry in CODE_ENTRIES:
        src = os.path.join(source, entry)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, entry),
                            dirs_exist_ok=True, ignore=_ignore)
            log("  copied %s/" % entry)
    for fn in TOPLEVEL_FILES:
        src = os.path.join(source, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, fn))
    if include_models:
        src = os.path.join(source, MODELS_ENTRY)
        if os.path.isdir(src):
            log("  copying models/ (this is large, ~1.2 GiB, please wait)...")
            shutil.copytree(src, os.path.join(dest, MODELS_ENTRY),
                            dirs_exist_ok=True, ignore=_ignore)
            log("  copied models/")
    log("Vendored Export_regeneration_stage_v20 into: %s" % dest)
    if not include_models:
        log("  (code only — run with the models option for a complete, "
            "exportable bundle)")
    return dest
