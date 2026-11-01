# -*- coding: utf-8 -*-
"""Installation self-check: is this machine able to reproduce the results?

The Dependencies card already answers "is every package importable". That is
not the same question as "will this installation compute the same numbers as
the reference machine", and it is the second one that decides whether a run is
trustworthy. The checks below cover what the package list cannot see:

* the interpreter and the five calculation-sensitive versions -- a different
  OpenCV or scikit-image silently moves descriptor VALUES, so a run can look
  perfectly healthy and still disagree with the golden tables;
* the torch build -- the default PyPI wheel has no CUDA, imports fine, and
  falls back to the processor, which turns a 9-minute run into an hour;
* the model weights -- present, non-empty, not truncated, and (in deep mode)
  actually loadable;
* the data folders and the write permissions, including the WSL case where the
  output folder sits on a Windows drive.

Every failing check carries the NATURE of the problem and the action that
fixes it, because a verdict without a remedy only moves the puzzle.

No Qt import here on purpose: this module is the single source of truth for
both the About page card and the command line, so a third party can self-check
an installation without opening the application::

    python -m flatmorphai.install_check
    python -m flatmorphai.install_check --deep     # also loads every checkpoint

Exit status is 0 when nothing failed, 1 when at least one check failed; a
warning alone does not fail the run.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional

#: A check never raises. ``status`` is one of:
#:   "ok"   -- verified good;
#:   "warn" -- works, but the result may differ from the reference, or the run
#:             will be much slower; the operator must know;
#:   "fail" -- a chain cannot run, or would produce wrong numbers;
#:   "skip" -- could not be tested here (dependency absent), which is itself
#:             reported rather than silently passed.
OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

#: Minimum VRAM the reference configuration was measured on, in GB.
VRAM_MIN_GB = 6.0
#: Python version the environment was validated with.
PYTHON_VALIDATED = (3, 10)


def _check(key: str, title: str, status: str, detail: str, fix: str = "") -> Dict[str, str]:
    return {"key": key, "title": title, "status": status, "detail": detail, "fix": fix}


def project_root() -> str:
    """Folder that holds ``flatmorphai/``, i.e. where the weights and the data
    folders are expected. Derived from this file, never from the cwd."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- interpreter

def _script(name: str) -> str:
    """Command line of a helper script: ``scripts/<name>`` in the repository
    layout, ``<name>`` at the root in an older one."""
    if os.path.isfile(os.path.join(project_root(), "scripts", name)):
        return "python scripts/" + name
    return "python " + name


def check_python() -> Dict[str, str]:
    v = sys.version_info
    got = "%d.%d.%d" % (v.major, v.minor, v.micro)
    if (v.major, v.minor) == PYTHON_VALIDATED:
        return _check("python", "Python interpreter", OK,
                      "Python %s, the validated version." % got)
    if (v.major, v.minor) < (3, 9):
        return _check("python", "Python interpreter", FAIL,
                      "Python %s. The application needs 3.9 or later and was "
                      "validated on 3.10." % got,
                      "Create an environment on Python 3.10: "
                      "conda create -n flatmorphai python=3.10")
    return _check("python", "Python interpreter", WARN,
                  "Python %s; the environment was validated on %d.%d. The "
                  "application runs, but the pinned wheels were resolved for "
                  "3.10." % (got, PYTHON_VALIDATED[0], PYTHON_VALIDATED[1]),
                  "To stay on the validated stack: "
                  "conda create -n flatmorphai python=3.10")


# ------------------------------------------------- calculation-sensitive stack
def check_calc_versions() -> Dict[str, str]:
    """The five packages whose version changes descriptor VALUES."""
    try:
        from .dependency_check import (CALC_SENSITIVE, VALIDATED_VERSIONS,
                                       calc_version_warnings)
    except Exception as exc:  # noqa: BLE001
        return _check("calc", "Calculation-sensitive versions", SKIP,
                      "Could not read the version table (%s)." % exc)
    warn = calc_version_warnings()
    names = ", ".join(sorted(VALIDATED_VERSIONS))
    # ``calc_version_warnings`` only compares what is INSTALLED, so a pinned
    # package that is absent altogether would leave this check green while the
    # chain cannot run at all. Report that here rather than let another check
    # carry it alone.
    absent = []
    try:
        from .dependency_check import _dist_version  # noqa: PLC0415,SLF001
        absent = [p for p in sorted(VALIDATED_VERSIONS) if not _dist_version(p)]
    except Exception:  # noqa: BLE001
        absent = []
    if absent:
        return _check(
            "calc", "Calculation-sensitive versions", FAIL,
            "%d of the %d pinned packages are not installed: %s. They decide "
            "the descriptor values, so nothing can be computed without them."
            % (len(absent), len(VALIDATED_VERSIONS), ", ".join(absent)),
            "Install them at the pinned versions: "
            "python -m pip install -r requirements.txt (and torch from the "
            "pytorch.org index matching your CUDA).")
    if not warn:
        return _check("calc", "Calculation-sensitive versions", OK,
                      "The %d pinned packages (%s) are on their validated "
                      "major.minor." % (len(VALIDATED_VERSIONS), names))
    lines = []
    for pip, inst, val, reason in warn:
        lines.append("%s: installed %s, validated %s. %s"
                     % (pip, inst or "?", val or "?",
                        CALC_SENSITIVE.get(pip, reason)))
    return _check(
        "calc", "Calculation-sensitive versions", FAIL,
        "%d package(s) differ from the validated version, so the descriptor "
        "values -- and therefore the forecasted classes -- may differ from the "
        "reference tables:\n  - %s" % (len(warn), "\n  - ".join(lines)),
        "Reinstall the validated versions, then prove the recomputation is "
        "unchanged: %s compare" % _script("test_csv_stereotypies.py"))


# ------------------------------------------------------------------ torch / GPU
def check_torch_build() -> Dict[str, str]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return _check("torch", "PyTorch build", FAIL,
                      "torch cannot be imported (%s)." % exc,
                      "Install the wheel matching your CUDA from "
                      "https://pytorch.org -- NOT the default PyPI wheel.")
    built = getattr(torch.version, "cuda", None)
    if not built:
        return _check(
            "torch", "PyTorch build", FAIL,
            "torch %s is a build WITHOUT CUDA support. It imports and runs, so "
            "nothing complains, but every model falls back to the processor: "
            "the measured 9 min 25 s of pipeline A becomes of the order of an "
            "hour." % torch.__version__,
            "Reinstall from the index matching your CUDA, for example:\n"
            "  python -m pip install torch==2.5.1 torchvision==0.20.1 "
            "--index-url https://download.pytorch.org/whl/cu121")
    if not torch.cuda.is_available():
        return _check(
            "torch", "PyTorch build", FAIL,
            "torch %s is built for CUDA %s but no GPU is visible. Under WSL "
            "this is almost always the NVIDIA driver: it is installed on the "
            "WINDOWS side, not inside the distribution."
            % (torch.__version__, built),
            "Check `nvidia-smi` inside the shell that launches the "
            "application. If it fails, install the NVIDIA driver on Windows "
            "(not in WSL) and reopen the shell.")
    return _check("torch", "PyTorch build", OK,
                  "torch %s, CUDA %s, %d GPU(s) visible."
                  % (torch.__version__, built, torch.cuda.device_count()))


def check_vram() -> Dict[str, str]:
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return _check("vram", "Graphics memory", SKIP,
                          "No GPU visible; see the PyTorch build check.")
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        free = total
        try:
            free = torch.cuda.mem_get_info()[0] / 1e9
        except Exception:  # noqa: BLE001 - older torch
            pass
    except Exception as exc:  # noqa: BLE001
        return _check("vram", "Graphics memory", SKIP,
                      "Could not read the GPU properties (%s)." % exc)
    if total >= VRAM_MIN_GB:
        return _check("vram", "Graphics memory", OK,
                      "%s, %.1f GB total, %.1f GB free."
                      % (props.name, total, free))
    return _check(
        "vram", "Graphics memory", WARN,
        "%s, %.1f GB total. The reference configuration has %.0f GB. The "
        "memory planner will shrink the batches and the number of concurrent "
        "workers, so the run is slower; the RESULTS do not change."
        % (props.name, total, VRAM_MIN_GB),
        "Nothing to fix for correctness. To go faster, close the other GPU "
        "applications before a run.")


def check_cores() -> Dict[str, str]:
    try:
        from .cpu_topology import logical_cores, physical_cores
        phys, log = physical_cores(), logical_cores()
    except Exception as exc:  # noqa: BLE001
        return _check("cores", "Processor cores", SKIP,
                      "Could not read the processor topology (%s)." % exc)
    if phys is None:
        return _check(
            "cores", "Processor cores", WARN,
            "The physical-core count could not be read (%d logical cores "
            "seen). The process pools are then sized on the LOGICAL count, "
            "which oversizes them: the descriptor workers compete for the same "
            "physical cores." % log,
            "On Windows this probe is not implemented yet; see the export "
            "notes. Results are unaffected, only the run time.")
    if phys >= 4:
        return _check("cores", "Processor cores", OK,
                      "%d physical cores, %d logical." % (phys, log))
    return _check("cores", "Processor cores", WARN,
                  "%d physical cores, %d logical; the reference configuration "
                  "has 4. Expect a longer run; the results do not change."
                  % (phys, log))


# ----------------------------------------------------------------- the weights
#: ``(relative folder, what it is, at least one file matching)``. Sizes are the
#: reference ones, used only to spot a truncated download.
WEIGHT_DIRS = [
    ("checkpoints_segmentation_model",
     "Mask2Former segmentation checkpoints", 700e6),
    ("mask2former-swin-tiny-ade-semantic",
     "Mask2Former base model (config + weights)", 150e6),
    (os.path.join("flatmorphai", "regen_stage_v20", "models"),
     "regeneration chain: U-Nets and state classifier", 1000e6),
    (os.path.join("flatmorphai", "export_head_detection", "models"),
     "blastema and eye U-Nets of the mask refinement (Image segmentation, "
     "Stereotype classification with blastema refinement)", 40e6),
]


def _tree_size(path: str) -> int:
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def check_weights(root: Optional[str] = None) -> List[Dict[str, str]]:
    root = root or project_root()
    out = []
    for rel, what, expected in WEIGHT_DIRS:
        path = os.path.join(root, rel)
        key = "weights:" + rel
        if not os.path.isdir(path):
            out.append(_check(
                key, "Weights -- %s" % what, FAIL,
                "%s is missing. The chains that need it refuse to start." % rel,
                "The weights are not in the repository, they are published as "
                "Release assets: " + _script("download_models.py")))
            continue
        size = _tree_size(path)
        if size == 0:
            out.append(_check(
                key, "Weights -- %s" % what, FAIL,
                "%s exists but is empty." % rel,
                "Delete the folder and download it again: " + _script("download_models.py")))
        elif size < 0.5 * expected:
            out.append(_check(
                key, "Weights -- %s" % what, FAIL,
                "%s holds only %.0f MB; the reference is about %.0f MB. The "
                "download is probably truncated."
                % (rel, size / 1e6, expected / 1e6),
                "Delete the folder and download it again, and let the script "
                "verify the SHA-256: " + _script("download_models.py")))
        else:
            out.append(_check(key, "Weights -- %s" % what, OK,
                              "%s, %.0f MB." % (rel, size / 1e6)))
    return out


def check_weights_loadable(root: Optional[str] = None) -> List[Dict[str, str]]:
    """Deep mode: really read every checkpoint with torch.

    Presence and size catch a missing or truncated download; they do not catch
    a corrupted archive, a file written by an incompatible torch, or a pickle
    the current interpreter refuses. Only loading does. It costs seconds and
    GPU memory, which is why it is not part of the quick pass.
    """
    root = root or project_root()
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return [_check("load", "Weights -- loadable", SKIP,
                       "torch unavailable (%s)." % exc)]
    out, tried, bad = [], 0, []
    for rel, _what, _exp in WEIGHT_DIRS:
        base = os.path.join(root, rel)
        if not os.path.isdir(base):
            continue
        for dp, _dn, fn in os.walk(base):
            for f in sorted(fn):
                if not f.endswith((".pt", ".pth", ".bin", ".ckpt")):
                    continue
                p = os.path.join(dp, f)
                tried += 1
                try:
                    torch.load(p, map_location="cpu", weights_only=True)
                except TypeError:
                    # torch < 2.0 has no weights_only
                    try:
                        torch.load(p, map_location="cpu")
                    except Exception as exc:  # noqa: BLE001
                        bad.append("%s (%s)" % (os.path.relpath(p, root),
                                                type(exc).__name__))
                except Exception as exc:  # noqa: BLE001
                    bad.append("%s (%s: %s)"
                               % (os.path.relpath(p, root),
                                  type(exc).__name__, str(exc)[:80]))
    if tried == 0:
        out.append(_check("load", "Weights -- loadable", SKIP,
                          "No checkpoint file found to read."))
    elif bad:
        out.append(_check(
            "load", "Weights -- loadable", FAIL,
            "%d of %d checkpoint(s) could not be read:\n  - %s"
            % (len(bad), tried, "\n  - ".join(bad)),
            "A readable file of the right size can still be corrupted. "
            "Download the Release assets again and let the script verify the "
            "SHA-256: " + _script("download_models.py")))
    else:
        out.append(_check("load", "Weights -- loadable", OK,
                          "%d checkpoint(s) read without error." % tried))
    return out


# -------------------------------------------------------------- data and rights
DATA_DIRS = [
    ("Processed_regeneration_images", "regeneration images", (".png", ".jpg")),
    ("Processed_videos_Frame_by_frame", "extracted frames", (".png",)),
    ("Videos_to_process", "source videos", (".avi", ".mkv", ".mp4")),
]


def check_data(root: Optional[str] = None) -> List[Dict[str, str]]:
    root = root or project_root()
    out = []
    for rel, what, exts in DATA_DIRS:
        path = os.path.join(root, rel)
        key = "data:" + rel
        if not os.path.isdir(path):
            out.append(_check(
                key, "Data -- %s" % what, WARN,
                "%s is missing. The application starts; the page that reads it "
                "has nothing to work on." % rel,
                "This folder ships with the repository. If it is absent, the "
                "clone is incomplete, or the folder was moved: point the page "
                "at its new location in Settings."))
            continue
        n = 0
        for dp, _dn, fn in os.walk(path):
            n += sum(1 for f in fn if f.lower().endswith(exts))
        if n == 0:
            out.append(_check(key, "Data -- %s" % what, WARN,
                              "%s contains no %s file." % (rel, "/".join(exts)),
                              "Check that the clone is complete."))
        else:
            out.append(_check(key, "Data -- %s" % what, OK,
                              "%s, %d file(s)." % (rel, n)))
    return out


def _writable(path: str) -> bool:
    probe = os.path.join(path, ".flatmorphai_write_probe")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("probe")
        os.remove(probe)
        return True
    except Exception:  # noqa: BLE001
        return False


def check_write_rights(root: Optional[str] = None) -> List[Dict[str, str]]:
    root = root or project_root()
    out = []
    cfg = os.path.join(os.path.expanduser("~"), ".flatmorphai")
    if _writable(cfg):
        out.append(_check("write:config", "Write access -- settings", OK,
                          "%s is writable." % cfg))
    else:
        out.append(_check(
            "write:config", "Write access -- settings", FAIL,
            "%s cannot be written. The application cannot save its settings, "
            "its caches or its comparison references." % cfg,
            "Give your account write access to your home folder, or set HOME "
            "to a folder you own before launching."))
    if _writable(root):
        out.append(_check("write:root", "Write access -- results", OK,
                          "The project folder is writable."))
    else:
        out.append(_check(
            "write:root", "Write access -- results", FAIL,
            "The project folder %s is read-only. Every chain writes its masks "
            "and its CSVs next to the images and would fail at the end of the "
            "run, after the computation." % root,
            "Move the clone to a folder you own, or fix its permissions."))
    return out


def check_wsl_workdir(root: Optional[str] = None) -> Dict[str, str]:
    """Under WSL, results written straight to /mnt/c are slow; the application
    mirrors them on the Linux disk. Worth stating, because it explains where
    the files appear during a run."""
    root = root or project_root()
    try:
        from .workers import linux_workdir
        on_win = linux_workdir.on_windows_drive(root)
        enabled = linux_workdir.enabled()
    except Exception as exc:  # noqa: BLE001
        return _check("wsl", "Working disk", SKIP,
                      "Could not be determined (%s)." % exc)
    if not on_win:
        return _check("wsl", "Working disk", OK,
                      "The project is on a native disk; no mirroring needed.")
    if enabled:
        return _check(
            "wsl", "Working disk", OK,
            "The project is on a Windows drive under WSL. Results are computed "
            "on the Linux disk and copied back during and at the end of the "
            "run -- which is why the output folder fills up in bursts.")
    return _check(
        "wsl", "Working disk", WARN,
        "The project is on a Windows drive under WSL and the Linux mirror is "
        "disabled (FLATMORPHAI_LINUX_WORKDIR=0). Writing a full-resolution PNG "
        "to /mnt/c costs about four times what it costs on the Linux disk.",
        "Unset FLATMORPHAI_LINUX_WORKDIR to let the application mirror again.")


# --------------------------------------------------------------------- runner
def run_checks(root: Optional[str] = None, deep: bool = False,
               progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    """Every check, in the order the operator should read them."""
    root = root or project_root()
    out: List[Dict[str, str]] = []

    def step(title, fn):
        if progress is not None:
            progress(title)
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001 - a check must never crash the page
            r = _check(title, title, SKIP,
                       "The check itself failed (%s: %s)." % (type(exc).__name__, exc))
        out.extend(r if isinstance(r, list) else [r])

    step("Python interpreter", check_python)
    step("Calculation-sensitive versions", check_calc_versions)
    step("PyTorch build", check_torch_build)
    step("Graphics memory", check_vram)
    step("Processor cores", check_cores)
    step("Model weights", lambda: check_weights(root))
    if deep:
        step("Reading the checkpoints", lambda: check_weights_loadable(root))
    step("Data folders", lambda: check_data(root))
    step("Write access", lambda: check_write_rights(root))
    step("Working disk", lambda: check_wsl_workdir(root))
    return out


def counts(results: List[Dict[str, str]]) -> Dict[str, int]:
    c = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for r in results:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return c


def format_text(results: List[Dict[str, str]]) -> str:
    """Plain-text report, for the command line and for a bug report."""
    mark = {OK: "  OK  ", WARN: " WARN ", FAIL: " FAIL ", SKIP: " SKIP "}
    lines = []
    for r in results:
        lines.append("[%s] %s" % (mark.get(r["status"], "  ?   "), r["title"]))
        for ln in r["detail"].splitlines():
            lines.append("         " + ln)
        if r["fix"] and r["status"] in (WARN, FAIL):
            for i, ln in enumerate(r["fix"].splitlines()):
                lines.append("         %s%s" % ("-> " if i == 0 else "   ", ln))
    c = counts(results)
    lines.append("")
    lines.append("%d ok, %d warning(s), %d failure(s), %d not tested"
                 % (c[OK], c[WARN], c[FAIL], c[SKIP]))
    if c[FAIL]:
        lines.append("A failure means a chain cannot run, or would not produce "
                     "the reference numbers. Fix those first.")
    elif c[WARN]:
        lines.append("No failure. The warnings do not change the results, "
                     "except where they say otherwise.")
    else:
        lines.append("This installation matches the reference configuration.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    deep = "--deep" in argv
    out = sys.__stdout__
    res = run_checks(deep=deep,
                     progress=lambda t: out.write("  ... %s\n" % t) or out.flush())
    out.write("\n" + format_text(res) + "\n")
    out.flush()
    return 1 if counts(res)[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
