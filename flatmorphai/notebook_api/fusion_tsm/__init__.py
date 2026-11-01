"""FusionNetV2 + Temporal Shift Module (TSM) classifier.

Verbatim files from the ``export_model_and_code`` bundle plus a light
integration layer used by the stereotype descriptor computation
and ``classification.process_videos`` so the FlatMorphAI Pipeline A /
Step 4 can route inference through the new model without touching the
legacy CE-head / Mahalanobis paths.

Architecture summary (full detail in ``model_fusion_tsm.py``):

  - ConvNeXt-V2 backbone, timm ``convnextv2_base.fcmae_ft_in22k_in1k_384``,
    1024-d global-avg pooled features.
  - Temporal Shift Module (Lin, J., Gan, C. and Han, S., "TSM:
    Temporal Shift Module for Efficient Video Understanding",
    ICCV 2019, arXiv:1811.08383) hooks installed on every backbone
    stage so a 2D network models motion at zero extra parameters
    across a clip of T=3 frames (T-1, T0, T+1) packed into the input
    channels.
  - Gated metric branch.  308 metric columns from 16 metric groups
    are standardised with the per-column statistics stored in the
    checkpoint, then a learnable per-metric gate (L1-sparsified at
    training time) plus a 2-layer MLP maps them to 128 dims; reference
    for the in-model selection: Guyon, I. and Elisseeff, A., "An
    Introduction to Variable and Feature Selection", JMLR 3, 1157-1182,
    2003.
  - Concat(1024 visual, 128 metric) -> linear classifier over five
    classes [C-shape, Screw-like, Wildtype, Snake-like, Walnut].

Lazy imports.  ``model_fusion_tsm`` pulls torch / timm; ``metrics_tsm``
and ``temporal_format`` only need cv2 / numpy / pandas.  We expose the
light helpers at package level so a torch-free environment (the AST
self-test, the per-video metrics CSV writer) can still import them.
"""

# Light imports (no torch).
from .nomenclature import (  # noqa: F401
    temporal_window,
    temporal_window_app,
    temporal_window_training,
    dyn_window,
    list_clip_frames,
    app_second,
    parse_frame,
)
from .temporal_format import build_temporal_image  # noqa: F401
from .metrics_tsm import (  # noqa: F401
    compute_required_metrics,
    required_metrics,
    required_groups,
)
# tsm_pipeline does the FlatMorphAI-side orchestration: per-video
# metrics CSV writer and per-video inference loop.  Both helpers are
# called from classification.process_videos when the fusion-TSM
# branch is active.
from .tsm_pipeline import (  # noqa: F401
    compute_video_metrics_tsm_csv,
    forecast_video_tsm,
)


def __getattr__(name):
    """Defer torch-dependent imports until the symbol is first used.

    Loading ``model_fusion_tsm`` and ``predictor_tsm`` requires torch +
    timm.  Surfacing them via module-getattr keeps
    ``from .fusion_tsm import build_temporal_image`` import-safe in
    environments without torch (which is how unit tests and the
    per-video metrics writer run).
    """
    if name in ("FusionNetV2", "load_fusion_tsm", "simclr_v2_head"):
        from . import model_fusion_tsm
        return getattr(model_fusion_tsm, name)
    if name in ("TSMPredictor",):
        from . import predictor_tsm
        return getattr(predictor_tsm, name)
    raise AttributeError(name)
