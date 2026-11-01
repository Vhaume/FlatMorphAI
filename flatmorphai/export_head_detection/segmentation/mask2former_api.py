"""
Notebook 2 port: Mask2Former semantic segmentation inference.

This module replicates verbatim the two entry points defined inline in
``2 - Generate image mask.ipynb`` so that the PySide6 workers call the
exact same code path as the reference Jupyter run. Prior revisions of
this file relied on a thin shim over ``evaluate_test_images_v2`` which
no longer exposes ``load_model_for_segmentation`` or
``forecast_mask_with_a_mask2former``. The resulting silent fallback
used an incorrect ``initialize_model_with_loss`` signature and was the
root cause of the segmentation fault observed when Step 2 was launched
from Pipeline A.

References
----------
  - Cheng B. et al. (2022). Masked-attention Mask Transformer for
    Universal Image Segmentation. CVPR 2022. arXiv:2112.01527
  - Liu Z. et al. (2021). Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows. ICCV 2021. arXiv:2103.14030
  - He K. et al. (2017). Mask R-CNN. ICCV 2017. DOI: 10.1109/ICCV.2017.322
  - Isensee F. et al. (2021). nnU-Net. Nature Methods 18, 203-211.
    DOI: 10.1038/s41592-020-01008-z
  - Shit S. et al. (2021). clDice loss. CVPR 2021.
    DOI: 10.1109/CVPR46437.2021.01629
  - Filatov A., Shiryaev E. (2010). Connected-component labelling.
    ISSPA 2010. DOI: 10.1109/ISSPA.2010.5605426
  - Wilson G. et al. (2017). Good Enough Practices in Scientific
    Computing. PLOS Computational Biology.
    DOI: 10.1371/journal.pcbi.1005510
"""

from __future__ import annotations

import ast
import os
import pickle
import re as _re
import time
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

CHECKPOINTS_SEGMENTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "mask2former_segmentation")   # standalone copy, checkpoint root of the export


# ---------------------------------------------------------------------------
# Checkpoint listing helpers
# ---------------------------------------------------------------------------
def list_segmentation_checkpoints(
    root: str = CHECKPOINTS_SEGMENTATION_DIR,
) -> List[str]:
    """Return checkpoint subdirectory names sorted by mtime, newest first."""
    if not os.path.isdir(root):
        return []
    entries = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    entries.sort(
        key=lambda n: os.path.getmtime(os.path.join(root, n)), reverse=True,
    )
    return entries


def resolve_most_recent_checkpoint(
    root: str = CHECKPOINTS_SEGMENTATION_DIR,
) -> Optional[str]:
    """Return the newest checkpoint name or None."""
    names = list_segmentation_checkpoints(root)
    return names[0] if names else None


# ---------------------------------------------------------------------------
# Internal helpers mirrored from the notebook
# ---------------------------------------------------------------------------
def _parse_param(value: Any) -> Any:
    """Parse a stringified list/tuple parameter from a checkpoint pickle.

    Accepts formats such as ``"[True, 4]"``, ``"True, 4"`` or
    ``"['gaussian', 0.1]"``.
    """
    if isinstance(value, (list, tuple)):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Expected str/list/tuple, got {type(value)}")
    value = value.strip()
    if not value.startswith("["):
        value = f"[{value}]"
    try:
        return ast.literal_eval(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse parameter: {value}") from exc


def _infer_backbone_depths(sd: dict) -> Optional[List[int]]:
    """Read Swin Transformer stage depths from an ImprovedMask2Former state dict.

    Keys of the form ``encoder.layers.N.blocks.M.*`` encode stage ``N``
    having at least ``M + 1`` transformer blocks.

    Reference: Liu Z. et al., Swin Transformer, ICCV 2021, arXiv:2103.14030.
    """
    sub = sd.get("base_model", sd)
    depths: dict = {}
    for key in sub.keys():
        match = _re.search(r"encoder\.layers\.(\d+)\.blocks\.(\d+)", key)
        if match:
            layer = int(match.group(1))
            block = int(match.group(2))
            depths[layer] = max(depths.get(layer, 0), block + 1)
    if not depths:
        return None
    n_stages = max(depths.keys()) + 1
    return [depths.get(i, 1) for i in range(n_stages)]


def _patch_albumentations_position_type() -> None:
    """Restore the ``PositionType`` enum removed in albumentations 2.0.

    Required to unpickle transforms stored by albumentations 1.4.x.
    Reference: albumentations changelog v2.0.0.
    """
    try:
        import albumentations as A  # type: ignore
    except ImportError:  # pragma: no cover
        return
    if not hasattr(A.PadIfNeeded, "PositionType"):
        class _PositionType(str, Enum):
            CENTER = "center"
            TOP_LEFT = "top_left"
            TOP_RIGHT = "top_right"
            BOTTOM_LEFT = "bottom_left"
            BOTTOM_RIGHT = "bottom_right"
            RANDOM = "random"
        A.PadIfNeeded.PositionType = _PositionType  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public loader (verbatim port of notebook 2 cell 3)
# ---------------------------------------------------------------------------
def load_model_for_segmentation(
    model_name: str,
    device: Optional["torch.device"] = None,
    checkpoint_root: str = CHECKPOINTS_SEGMENTATION_DIR,
) -> Tuple[Any, Any]:
    """Load a Mask2Former segmentation model from a checkpoint directory.

    The implementation mirrors the notebook "2 - Generate image mask.ipynb"
    exactly and rebuilds the full pipeline:
      - albumentations 1.x compatibility patch
      - dynamic loss selection via ``initialize_model_with_loss``
      - thin-worm structural patches when required
      - ``ImprovedMask2Former`` wrapper with all optional modules

    The HuggingFace pretrained backbone weights are never downloaded. The
    processor is loaded from the local cache via a four-level fallback,
    ending in a manual ``Mask2FormerImageProcessor`` with ImageNet
    defaults. The architecture is built from the cached ``config.json``
    only via ``Mask2FormerForUniversalSegmentation(config)``, which
    randomly initialises weights without any network access. When the
    cached config belongs to a smaller backbone variant than the
    fine-tuned checkpoint the backbone depths and head counts are
    patched in place before the model is instantiated. All parameters
    are then overwritten by the fine-tuned state dict via
    ``load_state_dict``.

    A single ``model.to(device)`` call is placed after ALL wrapping
    steps. Each wrapping step may create a new ``nn.Module`` object so
    any earlier call on the bare base model would not propagate to the
    final wrapper.

    Parameters
    ----------
    model_name : str
        Name of the checkpoint sub-directory under ``checkpoint_root``.
    device : torch.device, optional
        Target device. When None, CUDA is used if available, otherwise
        CPU. Passing the caller's device explicitly is recommended so
        the worker thread initialises its CUDA context first.
    checkpoint_root : str, optional
        Root directory containing the checkpoint sub-directories. Defaults
        to :data:`CHECKPOINTS_SEGMENTATION_DIR`.

    Returns
    -------
    Tuple[ImprovedMask2Former, BaseImageProcessor]
        The fully restored model in eval mode on the target device, and
        its associated HuggingFace processor.

    References
    ----------
      - Cheng B. et al. (2022). Mask2Former. CVPR 2022. arXiv:2112.01527
      - Liu Z. et al. (2021). Swin Transformer. ICCV 2021. arXiv:2103.14030
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required to load the segmentation model.")

    _patch_albumentations_position_type()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    # These imports are deferred so they live entirely in the worker
    # thread, matching the notebook execution pattern.
    from loss_initialization import initialize_model_with_loss  # type: ignore
    from improved_mask2former_wrapper import ImprovedMask2Former  # type: ignore
    from transformers import (  # type: ignore
        AutoConfig,
        AutoImageProcessor,
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )

    # ------------------------------------------------------------------ paths
    pth_model_file_path = os.path.join(
        checkpoint_root, model_name, model_name + "_loss_best_loss.pth",
    )
    pth_parameters_file_path = os.path.join(
        checkpoint_root, model_name, model_name + "_parameters.pkl",
    )

    # --------------------------------------------------------- load checkpoint
    # Weights are loaded on CPU first to avoid device mismatch issues.
    data_model = torch.load(
        pth_model_file_path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )

    # Backward-compatibility: inject fields absent from older checkpoints.
    if "list_train_f1" not in data_model:
        data_model["list_train_f1"] = []
    if "list_valid_f1" not in data_model:
        data_model["list_valid_f1"] = []
    if "seuil_es" not in data_model:
        data_model["seuil_es"] = 5

    # ------------------------------------------------------- load parameters
    try:
        with open(pth_parameters_file_path, "rb") as fh:
            data_parameters = pickle.load(fh)
    except Exception:
        # Fallback for checkpoints whose parameter file is missing or
        # corrupt. The defaults match the notebook.
        data_parameters = {
            "loss_name": "FocalBoundaryDiceCBDiceWTLoss",
            "Offset_Adjustment": "[True, None, 0.0]",
            "increase_point_supervision_density": "True",
            "train_tfms": "",
        }

    pretrained_model_name_or_path = data_model.get(
        "pretrained_model_name_or_path",
        "mask2former-swin-tiny-ade-semantic",
    )
    print(f"Pretrained model used: {pretrained_model_name_or_path}")

    # ------------------------------------------------------ device resolution
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(
                f"Device selected: {device} "
                f"(GPU: {torch.cuda.get_device_name(0)})"
            )
        else:
            device = torch.device("cpu")
            print(
                "WARNING: CUDA is not available. The model will run on CPU.\n"
                "  On a cluster, ensure you have requested a GPU node and "
                "that the correct CUDA-enabled PyTorch is active."
            )

    # ------------------------------------------------- processor candidates
    _processor_candidates = [
        pretrained_model_name_or_path,
        f"facebook/{pretrained_model_name_or_path}",
        os.path.join("facebook", "mask2former-swin-tiny-ade-semantic"),
        "mask2former-swin-tiny-ade-semantic",
    ]
    processor = None
    for _candidate in _processor_candidates:
        try:
            processor = AutoImageProcessor.from_pretrained(
                _candidate, local_files_only=True,
            )
            print(f"  Processor loaded from cache: {_candidate}")
            break
        except Exception:
            continue

    if processor is None:
        # Manual fallback with ImageNet statistics used by Mask2Former
        # pre-training (Cheng et al. 2022, arXiv:2112.01527).
        processor = Mask2FormerImageProcessor(
            do_resize=True,
            size={"shortest_edge": 800, "longest_edge": 1333},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_rescale=True,
            ignore_index=255,
            reduce_labels=False,
        )
        print("  Processor: constructed manually with ImageNet defaults.")

    # ---------------------------------------------------- config candidates
    _config_candidates = [
        pretrained_model_name_or_path,
        f"facebook/{pretrained_model_name_or_path}",
        "facebook/mask2former-swin-tiny-ade-semantic",
        "mask2former-swin-tiny-ade-semantic",
    ]
    _config = None
    _config_source = None
    for _candidate in _config_candidates:
        try:
            _config = AutoConfig.from_pretrained(
                _candidate, local_files_only=True,
            )
            _config_source = _candidate
            print(f"  Config loaded from cache: {_candidate}")
            break
        except Exception:
            continue

    if _config is None:
        raise RuntimeError(
            "Could not load any Mask2Former config from the local cache.\n"
            "Connect to the internet once and run:\n"
            "  from transformers import AutoConfig\n"
            f"  AutoConfig.from_pretrained('facebook/{pretrained_model_name_or_path}')\n"
            "to populate the cache, then re-run offline."
        )

    # ------------------------- backbone depth patching from state dict
    _inferred_depths = _infer_backbone_depths(data_model["model_state_dict"])
    if _inferred_depths is not None:
        _bc = getattr(_config, "backbone_config", None)
        if _bc is not None and hasattr(_bc, "depths"):
            _current_depths = list(_bc.depths)
            if _current_depths != _inferred_depths:
                print(
                    f"  Backbone depth mismatch detected.\n"
                    f"    Config '{_config_source}' has depths "
                    f"{_current_depths}.\n"
                    f"    State dict requires depths {_inferred_depths}.\n"
                    f"  Patching config to match state dict."
                )
                _bc.depths = _inferred_depths
                if (
                    hasattr(_bc, "num_heads")
                    and len(_bc.num_heads) != len(_inferred_depths)
                ):
                    _bc.num_heads = [
                        3 * (2 ** i) for i in range(len(_inferred_depths))
                    ]
                    print(f"  Patched num_heads to: {_bc.num_heads}")

    _config.num_labels = len(data_model["classes"])

    # The bare model is built on CPU. A single move after all wrapping
    # steps (below) guarantees the fully assembled model lands on the
    # correct device.
    model = Mask2FormerForUniversalSegmentation(_config)

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
        if hasattr(module, "checkpoint"):
            module.checkpoint = False

    print("  Model architecture built from config (no pretrained weights).")

    # ---------------------------------------------------------- loss wrapping
    loss_name = data_parameters.get("loss_name", "BCE and Dice loss")

    if loss_name == "FocalBoundaryDiceLoss":
        # Yin XX et al. (2023). J Cancer 14(5), 717-736. DOI: 10.7150/jca.82592
        model = initialize_model_with_loss(
            model, loss_name="FocalBoundaryDiceLoss", num_classes=2,
            lambda_dice=0.6, lambda_focal=0.1, lambda_boundary=0.3,
            focal_alpha=0.25, focal_gamma=2.0,
        )
    elif loss_name == "FocalBoundaryDiceCBDiceLoss":
        # Shi P et al. MICCAI 2024. DOI: 10.1007/978-3-031-72111-3_5
        # Shit S et al. clDice, CVPR 2021. arXiv:2003.07311
        # Yin XX et al. (2023). J Cancer 14(5), 717-736. DOI: 10.7150/jca.82592
        model = initialize_model_with_loss(
            model, loss_name="FocalBoundaryDiceCBDiceLoss", num_classes=2,
            lambda_dice=0.6, lambda_focal=0.2, lambda_boundary=0.2,
            focal_alpha=0.25, focal_gamma=2.0,
            cbdice_lambda_cbdice=0.3, cbdice_iter=5,
            cbdice_warmup_epochs=20, cbdice_dt_clip=40.0,
            cbdice_lambda_hd=0.0, cbdice_boundary_kernel_size=3,
            cbdice_logit_mode="sigmoid",
        )
    elif loss_name == "FocalBoundaryDiceCBDiceWTLoss":
        # Shi P et al. MICCAI 2024. arXiv:2407.01517
        # Shit S et al. CVPR 2021. arXiv:2003.07311
        # Salehi SSM et al. MICCAI Workshop 2017. arXiv:1706.05721
        # Yin XX et al. (2023). J Cancer 14(5), 717-736. DOI: 10.7150/jca.82592
        # Saha PK et al. Pattern Recogn. Lett. 2016, 76, 3-12.
        # DOI: 10.1016/j.patrec.2015.04.006
        model = initialize_model_with_loss(
            model, loss_name="FocalBoundaryDiceCBDiceWTLoss", num_classes=2,
            lambda_dice=0.65, lambda_focal=0.15, lambda_boundary=0.20,
            focal_alpha=0.25, focal_gamma=2.0,
            cbdice_lambda_cbdice=0.25, cbdice_iter=3,
            cbdice_logit_mode="sigmoid",
            cbdicewt_lambda_wt=0.25, cbdicewt_wt_alpha=0.30,
            cbdicewt_wt_beta=0.70, cbdicewt_wt_endpoint_weight=3.0,
        )
    elif loss_name == "TverskyLoss":
        # Salehi SSM et al. MICCAI Workshop 2017. arXiv:1706.05721
        model = initialize_model_with_loss(
            model, loss_name="TverskyLoss", num_classes=2,
            tversky_alpha=0.3, tversky_beta=0.7,
        )
    elif loss_name == "FocalTverskyLoss":
        # Abraham N, Khan NM. ISBI 2019. DOI: 10.1109/ISBI.2019.8759329
        model = initialize_model_with_loss(
            model, loss_name="FocalTverskyLoss", num_classes=2,
            tversky_alpha=0.5, tversky_beta=0.5, focal_gamma=1.5,
        )
    elif loss_name == "ComboLoss":
        # Taghanaki SA et al. arXiv:1805.02798
        model = initialize_model_with_loss(
            model, loss_name="ComboLoss", num_classes=2,
            combo_alpha=0.5, ce_weight=0.5,
        )
    # "BCE and Dice loss" requires no explicit wrapping.

    # --------------------------------------- thin-worm structural patches
    if data_parameters.get("increase_point_supervision_density") == "True":
        from thin_worm_patch import patch_mask2former_for_thin_worms  # type: ignore
        from refine_1x_patch import patch_refine_1x_for_320  # type: ignore
        from train_tfms_320 import setup_model_320  # type: ignore

        patch_mask2former_for_thin_worms(model, target_stride=1, verbose=True)
        patch_refine_1x_for_320(
            model, bottleneck_dim=64, use_checkpoint=True, verbose=True,
        )
        setup_model_320(model, target_stride=1, verbose=True)

    # ------------------------------------------- optional module parameters
    register_tokens = _parse_param(
        data_parameters.get("Register_Tokens", "[False, 0]")
    )
    offset_adjustment = _parse_param(
        data_parameters.get("Offset_Adjustment", "[False, None, 0.0]")
    )
    auxiliary_branch = _parse_param(
        data_parameters.get("Auxiliary_Branch", "[False, 0.0]")
    )
    noisy_guidance = _parse_param(
        data_parameters.get("Noisy_Guidance", "[False, None, 0.0]")
    )
    query_position_module = _parse_param(
        data_parameters.get("Query_Position_Module", "[False, 0.0]")
    )

    # ---------------------------------------- ImprovedMask2Former wrapper
    model = ImprovedMask2Former(
        model,
        use_register_tokens=register_tokens[0],
        num_registers=register_tokens[1],
        use_offset_adjustment=offset_adjustment[0],
        offset_strategy=offset_adjustment[1],
        offset_scale_factor=offset_adjustment[2],
        use_auxiliary_branch=auxiliary_branch[0],
        aux_loss_weight=auxiliary_branch[1],
        use_noisy_guidance=noisy_guidance[0],
        noise_type=noisy_guidance[1],
        noise_level=noisy_guidance[2],
        use_qpm=query_position_module[0],
        qpm_temperature=query_position_module[1],
        num_classes=2,
        compute_loss_per_sample=True,
    )

    # ------------------------------ restore trained heads and state dict
    model.class_predictor = data_model["class_predictor"].to(device)
    model.criterion = data_model["criterion"].to(device)
    print(f"Criterion: {data_model['criterion']}")
    print("Loading weights...")
    model.load_state_dict(data_model["model_state_dict"])

    # Single move after all wrapping steps.
    model = model.to(device)
    print(
        f"Model successfully moved to: {next(model.parameters()).device}"
    )
    model.eval()
    model.use_checkpointing = False  # type: ignore[attr-defined]

    return model, processor


# ---------------------------------------------------------------------------
# Public forecaster (verbatim port of notebook 2 cell 7)
# ---------------------------------------------------------------------------
def forecast_mask_with_a_mask2former(
    directory_all_images: str,
    model: Any,
    processor: Any,
    device: Any,
    model_name: str = "",
    folders_stereotypie: Optional[List[str]] = None,
    batch_size: int = 8,
    name_mask_dir: Optional[str] = None,
    image_augmentation: Any = None,
    two_pass: bool = True,
    two_pass_merge: str = "skeleton_guided",
    two_pass_padding_ratio: float = 0.50,
    two_pass_model_size: int = 320,
    two_pass_context_fill: bool = True,
    min_component_pixels: int = 10,
    min_pad_pixels: int = 32,
    min_roi_fraction: float = 0.0,
    two_pass_verbose: bool = False,
    keep_largest_component: bool = True,
    boundary_dilation_ratio: float = 0.02,
    two_pass_merge_dilation_radius: Any = None,
    two_pass_merge_dilation_fraction: float = 0.05,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    is_interrupted: Optional[Callable[[], bool]] = None,
) -> int:
    """Run two-pass Mask2Former inference on every frame under a tree.

    Exact port of the ``forecast_mask_with_a_mask2former`` defined in
    notebook 2 cell 7, extended with four UI-related keyword arguments
    (``model_name``, ``overwrite``, ``log``, ``progress_cb``,
    ``is_interrupted``) that are required by the PySide6 workers.

    Returns the total number of frames processed.

    References
    ----------
      - Cheng B. et al. (2022). Mask2Former. CVPR 2022. arXiv:2112.01527
      - He K. et al. (2017). Mask R-CNN. ICCV 2017.
        DOI: 10.1109/ICCV.2017.322
      - Isensee F. et al. (2021). nnU-Net. Nature Methods.
        DOI: 10.1038/s41592-020-01008-z
      - Shit S. et al. (2021). clDice. CVPR 2021.
        DOI: 10.1109/CVPR46437.2021.01629
      - Filatov A., Shiryaev E. (2010). ISSPA 2010.
        DOI: 10.1109/ISSPA.2010.5605426
    """
    # Lazy imports so the module is importable even if the repository is
    # analysed outside a GPU environment.
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    from evaluate_test_images_v2 import (  # type: ignore
        _apply_fusion,
        _ensure_uint8_rgb,
        _extract_p2_jobs,
        _keep_largest_component,
        _maybe_suppress_stdout,
        _paste_p2_result,
        _predict_batch_gpu,
    )

    _log = log if log is not None else (lambda _msg: None)

    def _interrupted() -> bool:
        return bool(is_interrupted() if is_interrupted is not None else False)

    if folders_stereotypie is None:
        folders_stereotypie = []

    if name_mask_dir is None:
        name_mask_dir = f"masks_{model_name}" if model_name else "masks"

    # Guard: ensure model resides on the intended device.
    # ``str(torch.device('cuda'))`` is ``"cuda"`` while
    # ``str(torch.device('cuda:0'))`` is ``"cuda:0"``, so a naive string
    # comparison kept moving the model on every call. Use the shared
    # helper that treats an unset index as the current CUDA device,
    # which is always 0 on the single-GPU workstations we target.
    class BaseWorker:
        @staticmethod
        def devices_equivalent(a, b): return (a.type == b.type) and ((a.index or 0) == (b.index or 0))

    model.eval()
    try:
        model_device = next(model.parameters()).device
        if not BaseWorker.devices_equivalent(model_device, device):
            _log(
                f"WARNING: model is on {model_device} but device={device}. "
                f"Moving model to {device}."
            )
            model = model.to(device)
    except StopIteration:
        pass

    _log(f"Model device: {next(model.parameters()).device}")
    _log(f"Target device: {device}")

    images_dir = directory_all_images
    if not folders_stereotypie:
        folders_stereotypie = [
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ]
    _log(f"Stereotypy folders: {folders_stereotypie}")

    use_two_pass = two_pass and two_pass_context_fill

    # Total frames for progress callback.
    total_frames = 0
    for folder in folders_stereotypie:
        images_subdir = os.path.join(images_dir, folder, "images")
        if os.path.isdir(images_subdir):
            total_frames += len(os.listdir(images_subdir))
    processed_frames = 0

    def _emit_progress(video_name: str = "") -> None:
        if progress_cb is not None:
            progress_cb(processed_frames, total_frames or 1, video_name)

    t_start = time.time()

    for folder_stereotypie in folders_stereotypie:
        if _interrupted():
            _log("Interrupted by user.")
            break

        path_mask_folder = os.path.join(
            images_dir, folder_stereotypie, name_mask_dir,
        )
        if not os.path.exists(path_mask_folder):
            os.makedirs(path_mask_folder)

        path_image_folder = os.path.join(
            images_dir, folder_stereotypie, "images",
        )
        if not os.path.isdir(path_image_folder):
            _log(
                f"Skipping {folder_stereotypie}: no images/ sub-folder."
            )
            continue

        img_list = os.listdir(path_image_folder)

        groups_images = [
            img_list[i:i + batch_size]
            for i in range(0, len(img_list), batch_size)
        ]

        total_groups = len(groups_images)
        _log(
            f"### Processing: {folder_stereotypie} | "
            f"{len(img_list)} images | "
            f"{total_groups} batches of {batch_size}"
        )

        for batch_idx, group_images in enumerate(groups_images, start=1):
            if _interrupted():
                _log("Interrupted by user.")
                break
            # Emit a plain "done/total" line so the progress area shows
            # the current batch index during model inference.
            _log(f"Batch {batch_idx}/{total_groups} - {folder_stereotypie}")

            batch_images_np = []
            for image_name in group_images:
                img_path = os.path.join(path_image_folder, image_name)
                img = cv2.imdecode(
                    np.fromfile(img_path, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                img = _ensure_uint8_rgb(img)
                if image_augmentation is not None:
                    img = image_augmentation(image=img)["image"]
                    img = _ensure_uint8_rgb(img)
                batch_images_np.append(img)

            n_b = len(batch_images_np)

            p1_masks = _predict_batch_gpu(
                model, processor, device,
                batch_images_np,
                model_size=two_pass_model_size,
            )

            if not use_two_pass:
                final_masks = p1_masks
            else:
                all_jobs: list = []
                jobs_per_image: list = [[] for _ in range(n_b)]
                for i in range(n_b):
                    jobs_i = _extract_p2_jobs(
                        image=batch_images_np[i],
                        p1_mask=p1_masks[i],
                        image_idx=i,
                        padding_ratio=two_pass_padding_ratio,
                        model_size=two_pass_model_size,
                        min_component_pixels=min_component_pixels,
                        min_pad_pixels=min_pad_pixels,
                        min_roi_fraction=min_roi_fraction,
                    )
                    all_jobs.extend(jobs_i)
                    jobs_per_image[i] = jobs_i

                p2_crop_results: list = []
                if all_jobs:
                    crops = [j["crop"] for j in all_jobs]
                    for sub_s in range(0, len(crops), batch_size):
                        sub_crops = crops[sub_s:sub_s + batch_size]
                        sub_masks = _predict_batch_gpu(
                            model, processor, device, sub_crops,
                            model_size=two_pass_model_size,
                        )
                        p2_crop_results.extend(sub_masks)

                p2_full_masks = [
                    np.zeros(batch_images_np[i].shape[:2], dtype=np.uint8)
                    for i in range(n_b)
                ]
                for job, crop_mask in zip(all_jobs, p2_crop_results):
                    _paste_p2_result(
                        crop_mask,
                        p2_full_masks[job["image_idx"]],
                        job,
                    )

                for i in range(n_b):
                    if (
                        p2_full_masks[i].sum() == 0
                        and p1_masks[i].sum() > 0
                    ):
                        p2_full_masks[i] = p1_masks[i].copy()

                final_masks = []
                for i in range(n_b):
                    crop_shape = None
                    if jobs_per_image[i]:
                        crop_shape = jobs_per_image[i][0]["crop"].shape
                    with _maybe_suppress_stdout(two_pass_verbose):
                        fused = _apply_fusion(
                            p1_mask=p1_masks[i],
                            p2_full=p2_full_masks[i],
                            two_pass_merge=two_pass_merge,
                            boundary_dilation_ratio=boundary_dilation_ratio,
                            crop_shape=crop_shape,
                            verbose=two_pass_verbose,
                            two_pass_merge_dilation_radius=(
                                two_pass_merge_dilation_radius
                            ),
                            two_pass_merge_dilation_fraction=(
                                two_pass_merge_dilation_fraction
                            ),
                        )
                    final_masks.append(fused)

            if keep_largest_component:
                for i in range(n_b):
                    with _maybe_suppress_stdout(two_pass_verbose):
                        final_masks[i], _ = _keep_largest_component(
                            final_masks[i], k=255,
                        )

            for i, image_name in enumerate(group_images):
                save_path = os.path.join(path_mask_folder, image_name)
                if (not overwrite) and os.path.isfile(save_path):
                    processed_frames += 1
                    _emit_progress(folder_stereotypie)
                    continue

                pred = final_masks[i].astype(float)
                if pred.max() == 0:
                    _log(
                        f"WARNING: empty prediction for image: {image_name}"
                    )
                else:
                    pred *= 255.0 / pred.max()

                (
                    Image.fromarray(pred)
                    .convert("L")
                    .save(save_path)
                )
                processed_frames += 1
                _emit_progress(folder_stereotypie)

    elapsed = time.time() - t_start
    _log(
        f"Done. Total time: {elapsed:.1f}s "
        f"({elapsed / max(1, processed_frames):.2f}s/image, "
        f"{processed_frames} images written)"
    )
    return processed_frames
