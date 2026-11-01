"""
Wrapper for the FlatMorphAI_deployment consolidated production bundle.

This module replaces every legacy classification loader (FasterViT +
Mahalanobis prototypes, ConvNeXt-V2 CE head + projection, older Fusion
model, TSM fusion model) with a single call into the deployment API
described in ``FlatMorphAI_deployment/README.md``.  The exposed
inference API is:

    from flatmorphai.notebook_api.deployment_bundle import (
        load_planaria_model,
        predict_video,
    )
    model, bundle_dir = load_planaria_model()
    names, idx = predict_video(model, image_paths, metrics_std,
                               use_bridge=True)

The deployment folder ships as a top-level sibling of the
``flatmorphai`` package (``<root>/FlatMorphAI_deployment``).
Its modules import each other by unqualified name
(``import proto_encoder as pe``), so we prepend the deployment folder
to ``sys.path`` on first use.  A user-specified ``bundle_dir`` from
Settings takes precedence over the default location.

The bundle exposes two prediction modes:

  * ``predict_base_class`` -- five base classes only
    (C-shape, Screw-like, Wildtype, Snake-like, Walnut), argmax of
    the production fusion gate.
  * ``predict_class``      -- five base classes plus Bridge-like
    (open-set surcharge above a locked threshold tau).

The ``use_bridge`` flag on :func:`predict_video` selects between
these two modes -- the same semantics the Settings ``Use Bridge-like
prototype`` toggle carried previously.

References
----------
  * FlatMorphAI_deployment/README.md (see the *Utilisation* section
    for the two prediction modes).
  * Woo, S. et al. "ConvNeXt V2", CVPR 2023, arXiv:2301.00808.
  * Lin, J., Gan, C. and Han, S. "TSM: Temporal Shift Module for
    Efficient Video Understanding", ICCV 2019, arXiv:1811.08383.
  * Scheirer, W. J. et al. "Toward Open Set Recognition", IEEE PAMI
    35(7), 2013, DOI: 10.1109/TPAMI.2012.256.
  * Chow, C. K. "On Optimum Recognition Error and Reject Tradeoff",
    IEEE Trans. Inf. Theory 16(1), 1970,
    DOI: 10.1109/TIT.1970.1054406.
  * Khosla, P. et al. "Supervised Contrastive Learning", NeurIPS
    2020, arXiv:2004.11362.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


# ---------------------------------------------------------------------
# Location of the deployment folder
# ---------------------------------------------------------------------
def _default_deployment_dir() -> str:
    """Return the default ``FlatMorphAI_deployment`` folder path.

    Resolved relative to this file so the app still works when
    launched from any working directory.  The layout is::

        <root>/
          flatmorphai/
            notebook_api/
              deployment_bundle.py  <-- this file
          FlatMorphAI_deployment/   <-- sibling of the package
    """
    here = Path(__file__).resolve()
    return str(here.parent.parent.parent / "FlatMorphAI_deployment")


def _default_bundle_dir() -> str:
    return os.path.join(_default_deployment_dir(), "bundle")


def _register_deployment_on_sys_path(deployment_dir: str) -> None:
    """Prepend the deployment folder to ``sys.path``.

    Idempotent.  Required because the deployment modules use
    top-level unqualified imports (``import proto_encoder``,
    ``import proto_production_features``, ...).
    """
    if not os.path.isdir(deployment_dir):
        raise FileNotFoundError(
            "FlatMorphAI deployment folder not found: "
            f"{deployment_dir}. "
            "Set 'step4_bundle_dir' in Settings to point at the "
            "'FlatMorphAI_deployment/bundle' directory."
        )
    if deployment_dir not in sys.path:
        sys.path.insert(0, deployment_dir)


def resolve_bundle_dir(
    bundle_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(deployment_dir, bundle_dir)`` validated on disk.

    A user-provided ``bundle_dir`` is honoured verbatim.  When empty,
    the default sibling layout is used.  Raises ``FileNotFoundError``
    if the resolved bundle does not carry a ``config.json``.
    """
    if bundle_dir:
        bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
        deployment_dir = os.path.dirname(bundle_dir)
    else:
        deployment_dir = _default_deployment_dir()
        bundle_dir = _default_bundle_dir()
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(
            f"Bundle directory not found: {bundle_dir}. "
            "Point 'step4_bundle_dir' at "
            "FlatMorphAI_deployment/bundle."
        )
    if not os.path.isfile(os.path.join(bundle_dir, "config.json")):
        raise FileNotFoundError(
            "Bundle 'config.json' not found under "
            f"{bundle_dir}. "
            "Point 'step4_bundle_dir' at the folder that contains "
            "config.json / production_noproj.pth / "
            "proto_encoder_head_6cls.pt / openset_config.pkl."
        )
    return deployment_dir, bundle_dir


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_planaria_model(
    bundle_dir: Optional[str] = None,
    device: Any = None,
) -> Tuple[Any, str]:
    """Load the consolidated ``PlanariaModel`` from ``bundle_dir``.

    Returns ``(planaria_model, absolute_bundle_dir)``.  The device
    (``cuda`` when available, else ``cpu``) is picked by the
    deployment module itself.

    The bundle's internal modules (``flatmorph_planaria_model``,
    ``proto_production_features``, ``proto_encoder``) print several
    diagnostic lines on load.  We keep those files pristine (they are
    the customer-facing deployment export) and instead swallow their
    stdout during the load call so the terminal stays quiet.  A
    single terse "Model ... is loaded" line is emitted by
    :func:`classification.load_deployment_bundle` afterwards.
    """
    import contextlib
    import io

    deployment_dir, bundle_dir = resolve_bundle_dir(bundle_dir)
    _register_deployment_on_sys_path(deployment_dir)
    import flatmorph_planaria_model as fm  # noqa: E402 -- sys.path setup
    with contextlib.redirect_stdout(io.StringIO()):
        model = fm.load(bundle_dir, device=device)
    return model, bundle_dir


def load_bundle_config(bundle_dir: str) -> dict:
    """Read ``config.json`` as a plain dict."""
    with open(
        os.path.join(bundle_dir, "config.json"), "r", encoding="utf-8",
    ) as fh:
        return json.load(fh)


def load_bundle_metric_cols(bundle_dir: str) -> List[str]:
    """Extract the ``metric_cols`` list from the production checkpoint.

    Used by ``process_videos`` to align the per-video metrics CSV
    columns (produced by ``compute_video_metrics_tsm_csv``) with what
    the bundle's metric branch expects.  Missing columns are
    tolerated: the bundle's :func:`standardize_metrics` fills them
    with the training median (README, section *Preparing the
    inputs*).
    """
    import torch  # local import so the app boots without torch
    cfg = load_bundle_config(bundle_dir)
    ckpt = torch.load(
        os.path.join(bundle_dir, cfg["production_weights"]),
        map_location="cpu",
    )
    return list(ckpt.get("metric_cols", []))


# ---------------------------------------------------------------------
# Metrics standardisation (best-effort)
# ---------------------------------------------------------------------
def standardize_metrics_matrix(
    metric_csv_path: str,
    image_basenames: Sequence[str],
    bundle_dir: str,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """Standardise metrics via the bundle helper, when possible.

    Returns an ``(N, metric_dim)`` float32 array when
    ``bundle/metric_stats.npz`` is present, otherwise ``None``.  The
    caller then substitutes a zero matrix (median-standardised value
    per README Option B): the image alone carries the decision, at a
    small accuracy cost.
    """
    _log = log if log is not None else (lambda _msg: None)
    stats = os.path.join(bundle_dir, "metric_stats.npz")
    if not os.path.isfile(stats):
        _log(
            f"  metric_stats.npz not found under {bundle_dir}; "
            "using standardised-median (zero) matrix per bundle "
            "README (image alone carries the decision)."
        )
        return None
    if not os.path.isfile(metric_csv_path):
        _log(
            f"  per-video metrics CSV missing "
            f"({metric_csv_path}); using standardised-median (zero) "
            "matrix (README Option B)."
        )
        return None
    deployment_dir = os.path.dirname(bundle_dir)
    _register_deployment_on_sys_path(deployment_dir)
    import flatmorph_planaria_model as fm  # noqa: E402
    return fm.standardize_metrics(
        metric_csv_path, list(image_basenames), bundle_dir=bundle_dir,
    )


def zero_metrics_matrix(planaria_model: Any, n: int) -> Any:
    """Return an ``(n, metric_dim)`` float32 zero matrix.

    Matches what the README calls Option B: metrics fall back to the
    standardised median so the model bases its decision on the image
    alone.
    """
    if not _HAS_NUMPY:
        raise ImportError("NumPy is required for zero_metrics_matrix")
    metric_dim = int(planaria_model.model.metric_dim)
    return np.zeros((n, metric_dim), dtype=np.float32)


# ---------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------
def predict_video(
    planaria_model: Any,
    image_paths: Sequence[str],
    metrics_std: Any,
    use_bridge: bool = False,
    batch_size: int = 8,
    io_workers: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[str], List[int]]:
    """Run inference over ``image_paths`` in ONE pass.

    Two optimisations over the deployment API's public
    :meth:`PlanariaModel.predict_class` / ``predict_base_class`` calls:

      1. **Parallel image loading.** The upstream API decodes each
         PNG sequentially inside a Python list comprehension.  We
         load every clip through a :class:`ThreadPoolExecutor` so
         cv2's imdecode + resize scale across cores while the GPU
         forward pass proceeds on the previous batch.
      2. **Single decision pass for Bridge-like.** In ``use_bridge``
         mode we do the dorsal pass ONCE per batch, cache the 1152-d
         fusion feature ``z``, and only at the end compute the
         encoder + open-set gate for every image in one vectorised
         numpy call.  This avoids re-running the encoder / gate per
         batch and keeps the "process all images then decide the
         class in one shot" ordering the operator requested.

    ``use_bridge`` selects the prediction mode:

      * ``False`` -> five base classes (argmax of the production
        fusion gate).
      * ``True``  -> five base classes plus Bridge-like via the
        locked open-set threshold ``tau``.

    Returns ``(pred_names, pred_indices)`` where the indices are
    into the active class list (``classes5`` when ``use_bridge`` is
    False, ``classes6`` when True).

    Reference for the "single global decision after all embeddings
    are computed" pattern: Wang, Y.-X. et al., "Simpleshot:
    Revisiting Nearest-Neighbor Classification for Few-Shot
    Learning", arXiv 1911.04623, 2019, sec. 3 (offline embedding +
    metric-space decision at the tail).
    """
    _log = log if log is not None else (lambda _msg: None)
    n = len(image_paths)
    if n == 0:
        return [], []

    import torch
    from concurrent.futures import ThreadPoolExecutor

    # Resolve the parallel-loader worker count.  ``_load_clip`` is
    # dominated by cv2's imdecode + INTER_AREA resize which release
    # the GIL, so threads scale nearly linearly.  We cap at 8 to
    # avoid saturating the SSD's queue depth on typical NVMe drives.
    try:
        _cpu = os.cpu_count() or 4
    except Exception:  # noqa: BLE001
        _cpu = 4
    if io_workers is None or io_workers <= 0:
        io_workers = max(2, min(8, _cpu - 1))

    # Re-register the deployment folder in case the caller loaded
    # PlanariaModel via a different code path.
    deployment_dir = _default_deployment_dir()
    _register_deployment_on_sys_path(deployment_dir)
    import proto_production_features as ppf  # noqa: E402

    # Parallel decode of every clip.  ``_load_clip`` returns a
    # (9, 384, 384) float32 tensor -- ~5.3 MB per clip.  For a
    # 1000-frame run that peaks at ~5 GB of host RAM; well within
    # budget for the machines this app runs on.
    ms = np.asarray(metrics_std, dtype=np.float32)
    if ms.shape[0] != n:
        raise ValueError(
            f"metrics_std has {ms.shape[0]} row(s) but "
            f"{n} image_paths were provided"
        )

    clips: List[Any] = [None] * n
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for i, tensor in enumerate(pool.map(ppf._load_clip, image_paths)):
            clips[i] = tensor
    _log(
        f"  Bundle inference: pre-loaded {n} clip(s) with "
        f"{io_workers} decode worker(s)"
    )

    device = planaria_model.device
    model = planaria_model.model
    Z_list: List[Any] = []
    LG_list: List[Any] = []
    total_batches = (n + batch_size - 1) // batch_size
    done_batches = 0
    for i in range(0, n, batch_size):
        batch_clips = clips[i:i + batch_size]
        batch_m = ms[i:i + batch_size]
        xs = torch.stack(batch_clips).to(
            device, non_blocking=True,
        ).to(memory_format=torch.channels_last)
        mm = torch.from_numpy(batch_m).to(device, non_blocking=True)
        B = xs.shape[0]
        H, W = xs.shape[-2], xs.shape[-1]
        xc = xs.reshape(B * model.T, 3, H, W)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(str(device) != "cpu"),
            ):
                feat = model.backbone(xc).view(
                    B, model.T, -1,
                ).mean(1)
                f = model.vis_norm(feat)
                g = model.metric_mlp(mm * model.metric_gate)
                z = torch.cat([f, g], dim=1)
                lf = model.classifier(z)
                lm = model.metric_aux(g)
                lv = model.vis_aux(f)
                w = torch.softmax(model.gate_net(z), dim=1)
                logits = (
                    w[:, 0:1] * lf + w[:, 1:2] * lm + w[:, 2:3] * lv
                )
        Z_list.append(z.float().cpu().numpy())
        LG_list.append(logits.float().cpu().numpy())
        done_batches += 1
        if progress_cb is not None:
            label = (
                os.path.basename(image_paths[i])
                if image_paths else ""
            )
            progress_cb(done_batches, total_batches, label)
        _log(f"  Bundle batch {done_batches}/{total_batches}")

    Z = np.concatenate(Z_list, axis=0)
    LG = np.concatenate(LG_list, axis=0)
    base_pred = LG.argmax(1)

    if use_bridge:
        # ONE encoder + gate pass over every image, as the operator
        # requested ("focus on embeddings for every image first,
        # then decide the classes at the end").
        E = planaria_model._encode(Z)
        E = planaria_model._prep(E)
        p_bridge = planaria_model.gate.predict_proba(E)[:, 1]
        idx = np.where(
            p_bridge >= planaria_model.tau,
            len(planaria_model.base_classes),
            base_pred,
        )
        pred_names = [
            planaria_model.classes6[k]
            if k == len(planaria_model.base_classes)
            else planaria_model.classes5[k]
            for k in idx
        ]
        classes = planaria_model.classes6
    else:
        pred_names = [planaria_model.classes5[k] for k in base_pred]
        classes = planaria_model.classes5

    class_to_idx = {c: i for i, c in enumerate(classes)}
    pred_idx = [class_to_idx.get(nm, -1) for nm in pred_names]
    return pred_names, pred_idx


# ---------------------------------------------------------------------
# Convenience introspection
# ---------------------------------------------------------------------
def bundle_available(bundle_dir: Optional[str] = None) -> bool:
    """Return True when :func:`resolve_bundle_dir` would succeed.

    Used by the Settings + Pipeline A pages to decide whether to
    route through the bundle by default or to fall back to the
    legacy checkpoint-based loader.
    """
    try:
        resolve_bundle_dir(bundle_dir)
    except FileNotFoundError:
        return False
    return True
