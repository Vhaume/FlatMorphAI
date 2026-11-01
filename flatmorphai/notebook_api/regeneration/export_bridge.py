"""
Bridge to the ``Export_head_detection/`` folder.

The export is written as a set of flat scripts that import each other by
bare module name (``import anchor_cascade``, ``from parameter_reader
import read``), so the sub-folders have to be on ``sys.path`` before any
of them is imported.  Everything here is lazy: nothing heavy (torch,
transformers, segmentation_models_pytorch) is imported until a pipeline
actually runs, so the GUI starts even on a machine without a GPU.

The export root is resolved from the application root
(``<root>/Export_head_detection``) or from the setting
``regeneration_export_dir``.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SUBFOLDERS = (
    "", "common", "parameters", "eyes", "blastema", "darkness", "head",
    "segmentation", "stage", "metrics",
)

_STATE: Dict[str, Any] = {"root": None}


def default_export_dir() -> str:
    # Prefer the local in-package copy when complete, else the sibling folder.
    try:
        from ...export_bundle import resolve_export_root
        return resolve_export_root()
    except Exception:  # noqa: BLE001
        here = Path(__file__).resolve()
        return str(here.parents[3] / "Export_head_detection")


def export_available(export_dir: Optional[str] = None) -> bool:
    root = export_dir or default_export_dir()
    need = (
        "parameters/models.csv", "common/anchor_cascade.py",
        "head/head_detection.py", "segmentation/segment_mask.py",
        "stage/stage_prediction.py",
    )
    return all(os.path.isfile(os.path.join(root, n)) for n in need)


def activate(export_dir: Optional[str] = None) -> str:
    """Put the export sub-folders on ``sys.path`` (once) and return the root."""
    root = os.path.abspath(export_dir or default_export_dir())
    if _STATE["root"] == root:
        return root
    if not export_available(root):
        raise FileNotFoundError(
            f"Export_head_detection folder not found or incomplete: {root}"
        )
    for sub in SUBFOLDERS:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    _STATE["root"] = root
    return root


def module(name: str):
    """Import one export module by its bare name (after :func:`activate`)."""
    if _STATE["root"] is None:
        activate()
    return importlib.import_module(name)


def parameters(name: str) -> Dict[str, Any]:
    """Effective values of ``parameters/<name>.csv``."""
    return module("parameter_reader").read(name)


def model_path(role: str) -> str:
    return module("parameter_reader").model_path(role)
