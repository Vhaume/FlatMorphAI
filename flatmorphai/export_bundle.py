"""
Local vendoring of the ``Export_head_detection`` bundle.

The regeneration, eyespots-counter, Pipeline C and stereotype blastema-refine
pipelines all rely on the ``Export_head_detection`` bundle (head cascade, eye
and blastema detectors, the Mask2Former ``Segmenter`` and their parameters and
model weights). By default that bundle sits next to the application
(``<root>/Export_head_detection``), OUTSIDE the ``flatmorphai``
package, so it does not travel with the code when the program is exported.

This module lets the program keep its OWN local copy inside the package
(``flatmorphai/export_head_detection``). :func:`resolve_export_root` returns
that local copy when it is present and complete, otherwise the sibling folder,
so nothing breaks before vendoring and the app becomes self-contained after it.
:func:`vendor_export_bundle` performs the copy (run it from
``vendor_export_bundle.py`` at the application root).
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))          # .../flatmorphai
_APP_ROOT = os.path.dirname(_PKG_DIR)                          # .../<root>

#: Local, in-package copy that travels with the code on export.
LOCAL_BUNDLE_DIR = os.path.join(_PKG_DIR, "export_head_detection")
#: Default external copy, a sibling of the ``flatmorphai`` package.
SIBLING_BUNDLE_DIR = os.path.join(_APP_ROOT, "Export_head_detection")

#: Runtime sub-packages the pipelines import (added to ``sys.path`` by the
#: subprocess drivers) plus the parameter and metric folders they read.
CODE_ENTRIES = (
    "common", "parameters", "eyes", "blastema", "darkness",
    "head", "segmentation", "stage", "metrics",
)
#: The model-weights folder (large, ~1 GiB). Needed at runtime but copied only
#: on request so a quick code backup stays small.
MODELS_ENTRY = "models"
#: Small top-level files worth keeping with the vendored copy.
TOPLEVEL_FILES = ("README.md", "requirements.txt", "single_image_pipeline.py")


def _has_code(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "common", "anchor_cascade.py"))


def _has_models(root: str) -> bool:
    md = os.path.join(root, MODELS_ENTRY)
    try:
        return os.path.isdir(md) and any(os.scandir(md))
    except OSError:
        return False


def is_complete(root: str) -> bool:
    """True when ``root`` carries both the code and the model weights."""
    return _has_code(root) and _has_models(root)


def resolve_export_root() -> str:
    """Return the bundle root to use.

    Prefers the local in-package copy when it is COMPLETE (code + weights), so a
    partial/code-only backup never shadows a working external bundle. Falls back
    to the sibling ``Export_head_detection`` otherwise.
    """
    if is_complete(LOCAL_BUNDLE_DIR):
        return LOCAL_BUNDLE_DIR
    return SIBLING_BUNDLE_DIR


def _ignore_pycache(_dir, names):
    return [n for n in names if n == "__pycache__"]


def vendor_export_bundle(
    source: Optional[str] = None,
    dest: Optional[str] = None,
    include_models: bool = False,
    log: Callable[[str], None] = print,
) -> str:
    """Copy the needed ``Export_head_detection`` files into a local folder.

    Parameters
    ----------
    source : the bundle to copy from (defaults to the sibling folder).
    dest   : where to vendor it (defaults to ``flatmorphai/export_head_detection``).
    include_models : also copy the (large) ``models/`` weights, which makes the
        local copy a complete, self-contained bundle usable for export.
    """
    source = os.path.abspath(source or SIBLING_BUNDLE_DIR)
    dest = os.path.abspath(dest or LOCAL_BUNDLE_DIR)
    if not _has_code(source):
        raise FileNotFoundError(
            "Source Export_head_detection bundle not found or incomplete: "
            + source
        )
    if os.path.abspath(source) == os.path.abspath(dest):
        raise ValueError("Source and destination are the same folder.")
    os.makedirs(dest, exist_ok=True)

    for entry in CODE_ENTRIES:
        src = os.path.join(source, entry)
        if os.path.isdir(src):
            shutil.copytree(
                src, os.path.join(dest, entry),
                dirs_exist_ok=True, ignore=_ignore_pycache,
            )
            log("  copied %s/" % entry)
    for fn in TOPLEVEL_FILES:
        src = os.path.join(source, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, fn))
    if include_models:
        src = os.path.join(source, MODELS_ENTRY)
        if os.path.isdir(src):
            log("  copying models/ (this is large, please wait)...")
            shutil.copytree(
                src, os.path.join(dest, MODELS_ENTRY),
                dirs_exist_ok=True, ignore=_ignore_pycache,
            )
            log("  copied models/")
    log("Vendored Export_head_detection into: %s" % dest)
    if not include_models:
        log("  (code only — run with the models option for a complete, "
            "exportable bundle)")
    return dest
