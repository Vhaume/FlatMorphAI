# -*- coding: utf-8 -*-
"""Drop the model-construction debug from the pipeline log.

The segmentation subprocess relays everything its child writes, and the Export
bundle prints about ninety lines of construction trace on every single run:
the loss banner, the ThinWorm patch report, the 320x320 setup, the
"IMPROVED MASK2FORMER INITIALIZATION" checklist and the first-forward-pass
summary. They are identical at every run, they say nothing about the data
being processed, and they bury the lines that do matter (which video is
segmented, how long a batch took, how much VRAM is left).

Two mechanisms remove them, because they are printed at two different moments:
  * the load-time blocks are silenced at the source, by wrapping the model
    construction in ``_quiet_env.suppressed_output()`` (file-descriptor level,
    the only thing that catches C-level writes);
  * the few lines printed DURING the run (the first forward pass, the library
    warnings) cannot be silenced that way without also swallowing the driver's
    own protocol lines, so they are dropped here, on the relay side.

Nothing is lost silently: the relay counts what it dropped and says so once,
and ``FLATMORPHAI_VERBOSE=1`` brings every line back.
"""
import os
import re

_PATTERNS = (
    r"^=+$",                                   # banner rules
    r"FIRST FORWARD PASS",
    r"MODULES THAT WILL BE USED",
    r"^\s*Input shape:\s*torch\.Size",
    r"^\s*Training mode:\s",
    r"^\s*Has mask labels:\s",
    r"^\s*[✓✔]\s+Offset Adjustment\s*$",
    # the model-construction banner, in case a load path escapes the
    # file-descriptor suppression: the loss report, the ThinWorm patch, the
    # 320x320 setup and the "improvements" checklist. Printed identically at
    # every run and about the model's own assembly, never about the data.
    r"^LOSS INITIALIZATION|^Loss name:|^Num classes:",
    r"FOCAL BOUNDARY DICE|cbDice Citation|FBD Citation|Skeleton Tversky",
    r"^Endpoint detection:|^Parameters:$|^\s*(lambda_|focal_|cbdice|cbdicewt)",
    r"FBDCBDiceWT|FocalBoundaryDiceCBDiceWTLoss|fbd_cbdice",
    r"ThinWormSolution|setup_model_320|patch_refine_1x_for_320",
    r"IMPROVED MASK2FORMER INITIALIZATION|INITIALIZING IMPROVEMENTS",
    r"TOTAL NEW PARAMETERS|^SUMMARY:$|^DEBUG MODE$",
    r"Register Tokens:|Auxiliary Branch:|Noisy Mask Guiding:|Query Position Module:",
    r"Offset Adjustment",
    r"^\s*(refine_2x|refine_1x|skip_proj|Total new params|Cascade:|Refine blocks:)",
    r"^\s*(local|remote) +branch:|1px worm representation",
    r"^\s*(Swin backbone|Pixel decoder enc|target_stride|checkpoint=True)",
    r"^\s*Backward workspace:|gradient checkpointing",
    r"^\+ FocalBoundaryDiceCBDiceWT configured",
    # internal VRAM bookkeeping of the blastema-refined driver: three lines
    # per video plus the park/unpark accounting, useful when chasing an
    # out-of-memory, noise the rest of the time.
    r"^\s*\[VRAM\]",
    r"model\(s\) parked on the host",
    r"nothing to park yet",
    # non-fatal library notices
    #
    # NOT hidden any more: "Error importing huggingface_hub" and
    # "RequestsDependencyWarning". Both come from one broken package in the
    # environment -- charset_normalizer, whose compiled part no longer matches
    # its Python part -- and I filed them here as cosmetic noise. They are
    # not: they announce a real defect, which stayed invisible until it broke
    # an unrelated import outright. A message that names a broken dependency
    # belongs in the log.
    r"^\s*warnings\.warn\(",
    r"UserWarning: None of the inputs have requires_grad",
    r"None of the inputs have requires_grad=True",
)
_RE = re.compile("|".join(_PATTERNS))

#: Second family: the measurement trace. These lines exist for TUNING -- block
#: durations, CPU seconds, cache hit rates per worker, the RAM planner's
#: arithmetic, the GPU self-test. They were the instrument used to take the
#: chain from 40 minutes to 10, and they are still how a slowdown gets
#: diagnosed, but they say nothing about the worms being measured, and they
#: are the bulk of what the operator reads during a normal run. Hidden by
#: default, restored whole by FLATMORPHAI_VERBOSE=1.
#: What is deliberately NOT here: the duration of each step, which videos were
#: processed, how many frames were written, the predicted classes, the 4-second
#: smoothing, where the results were copied. Those describe the work, not the
#: machine.
_DIAG_PATTERNS = (
    r"^\s*block \d+/\d+ done in ",
    r"^\s*worker \d+ \(pid \d+\) :",
    r"^\s*RAM: PSS per worker",
    r"^\s*RAM planner:",
    r"^\s*adaptive pool:",
    r"^\s*GPU self-test on fixed images",
    r"^\s*GPU server: (ON|OFF)",
    r"^\s*descriptor memo: (ON|OFF)",
    r"^\s*MCP numba: (ON|OFF)",
    r"^\s*CPU: .* logical available",
    r"^LightGBM: BALANCED frame pool:",
    r"^\s*stage \d+ timing per image:",
    r"^\s*\(timing: ",
    r"^\s*Segmentation batch \d+ \(best measured",
    r"^\s*Segmentation batch: calibrating",
    r"^Step \d+ resolver expectations:",
    r"^Note: DataLoader num_workers forced",
    r"^\s*Fed resolution",
    r"^\s*GPU memory: sub-batch capped",
    # regeneration / eyespots chain: the same instrumentation, under other
    # names. The per-stage profile, the U-Net cache hit rate, the number of
    # workers and the free VRAM are tuning figures; what the run PRODUCED
    # (the images analysed, their duration, the CSVs written, where they
    # were copied) goes through untouched.
    r"^\s*GPU memory before loading the model",
    r"^\s*concurrency: \d",
    r"^\s*U-Net inference: \d+ call\(s\)",
    r"^\s*--- per-stage profile",
    r"^\s+\S.*\d+\.\d s \| \d+\.\d{3} s/img \| ",
    r"^\s*GPU \(segmentation \+ U-Nets \+ state\) ~",
    r"^\s*\(per-stage profile unavailable",
    r"^\s*descriptor functions loaded:",
)
_DIAG_RE = re.compile("|".join(_DIAG_PATTERNS))


def diagnostic(line: str) -> bool:
    """True for a tuning / measurement line (durations, caches, planner)."""
    if verbose():
        return False
    try:
        return bool(_DIAG_RE.search(line))
    except Exception:  # noqa: BLE001
        return False


def verbose() -> bool:
    return os.environ.get("FLATMORPHAI_VERBOSE", "0") in ("1", "true", "True")


def drop(line: str) -> bool:
    """True when ``line`` is construction noise rather than run information."""
    if verbose():
        return False
    try:
        return bool(_RE.search(line))
    except Exception:  # noqa: BLE001
        return False


def summary(n: int) -> str:
    """One line for the operator, or "" when nothing was dropped."""
    if n <= 0:
        return ""
    return ("  (%d line(s) of model-construction debug hidden; "
            "FLATMORPHAI_VERBOSE=1 shows them)" % n)
