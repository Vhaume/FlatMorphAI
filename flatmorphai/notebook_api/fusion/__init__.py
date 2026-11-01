"""FusionNetV2 image + 3-mask metric fusion classifier.

This subpackage hosts the three files of the Export_fusion_v2 bundle
(model_fusion.py, metrics_3mask.py) plus the integration glue used by
classification.py to detect a fusion checkpoint, build the FusionNetV2,
generate per-video metrics CSVs from Mask2Former outputs, and run the
two-tensor forward at inference time.

Lazy imports.  ``model_fusion`` imports torch; ``metrics_csv`` and
``metrics_3mask`` only need cv2 / numpy / pandas.  We expose
``compute_video_metrics_csv`` at package level for cheap importers and
defer ``FusionNetV2`` / ``load_fusion_model`` to module-getattr so
``from flatmorphai.notebook_api.fusion import compute_video_metrics_csv``
works in environments where torch is absent (the unit tests, the
notebook-only diagnostics, etc.).

References for the underlying SimCLR-v2 head: Chen, T. et al., "Big
Self-Supervised Models are Strong Semi-Supervised Learners", NeurIPS
2020, arXiv:2006.10029.  ConvNeXt-V2 backbone: Woo, S. et al.,
"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
Autoencoders", CVPR 2023, arXiv:2301.00808.
"""

# Light imports (no torch).
from .metrics_3mask import METRIC_COLS, compute_metric_vector  # noqa: F401
from .metrics_csv import compute_video_metrics_csv  # noqa: F401


def __getattr__(name):
    """Defer torch-dependent imports until the symbol is first used.

    This keeps ``from .fusion import compute_video_metrics_csv`` import-
    safe in environments without torch (the package was importable in
    the AST self-tests this way), while still surfacing
    ``FusionNetV2`` / ``load_fusion_model`` to the inference path.
    """
    if name in ("FusionNetV2", "load_fusion_model"):
        from . import model_fusion
        return getattr(model_fusion, name)
    raise AttributeError(name)
