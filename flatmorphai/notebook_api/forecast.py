"""
Compatibility shim that re-exports the classification API from
:mod:`flatmorphai.notebook_api.classification`.

Earlier revisions of this module attempted to call helpers that lived
inside ``evaluate_test_images_v2`` but were in fact only defined in
notebook 4. To bring the GUI workers in line with the reference Jupyter
run the heavy logic was ported into ``classification.py``; this file
keeps existing imports such as
``from ..notebook_api.forecast import process_videos``
working without forcing every call site to migrate at once.

References
----------
  - Hatamizadeh A. et al. (2023). FasterViT. arXiv:2306.06189
  - Wilson G. et al. (2017). Good Enough Practices in Scientific
    Computing. PLOS Computational Biology.
    DOI: 10.1371/journal.pcbi.1005510
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from .classification import (
    forecast_classes_in_image_dataloader,
    process_videos,
)

# The ConvNeXt-V2 re-exports (list_classification_checkpoints,
# load_model_and_prototype, resolve_most_recent_checkpoint) were dropped on
# 31 October 2026 together with that loading path. Nothing imported them: the
# segmentation checkpoint resolver of the same name lives in
# ``notebook_api.segmentation`` and is the one every page uses.
__all__ = [
    "forecast_classes_in_image_dataloader",
    "process_videos",
]


