"""
Pre-boot dependency check.

This module is imported at the very top of :mod:`flatmorphai.__main__`
(and :mod:`flatmorphai.main`) BEFORE any heavy third-party import.
It uses only the Python standard library so it never itself triggers
an ImportError.  When one or more required packages are missing, it
prints a clean actionable error naming each missing package plus the
exact ``pip install`` (and conda-forge) command to install them, then
exits with status 1 so the process ends before PySide6 gets a chance
to fail with a cryptic ``ImportError`` deep in the app code.

Reference for the "fail fast, fail loudly" pattern that motivates
this pre-boot gate: Wilson, G. et al., "Good Enough Practices in
Scientific Computing", PLOS Computational Biology 13(6):e1005510,
2017, DOI:10.1371/journal.pcbi.1005510, rule 4.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
from typing import List, Tuple


# Ordered list of ``(import_name, pip_package_name)`` tuples.  Some
# packages have a distinct import name and pip name (cv2 ->
# opencv-python, PIL -> Pillow, sklearn -> scikit-learn, skimage ->
# scikit-image) so the two are tracked separately.
#
# The list is deliberately RUNTIME-critical only: packages the app
# would crash on the moment the operator loads a page.  Optional
# extras (albumentations, jupytext, ...) are not included so a lean
# install does not fail this gate.
_REQUIRED: List[Tuple[str, str]] = [
    ("PySide6",       "PySide6"),
    ("torch",         "torch"),
    ("numpy",         "numpy"),
    ("cv2",           "opencv-python"),
    ("pandas",        "pandas"),
    ("PIL",           "Pillow"),
    ("scipy",         "scipy"),
    ("skimage",       "scikit-image"),
    ("sklearn",       "scikit-learn"),
    ("transformers",  "transformers"),
    # LightGBM is the new stereotype classifier (replaces the
    # ConvNeXt-V2 bundle).  ``lightgbm >= 4.7`` is the version the
    # trained models were saved with.
    ("lightgbm",      "lightgbm"),
]


# Optional packages: required only to RUN the Forecast regeneration
# states / Eyespots counter / Regeneration states editor pipelines (in
# their subprocess), NOT to start the application.  They are reported as
# a non-fatal warning when missing, never installed automatically, and
# never block launch.  This matters because several of them are
# torch-family (torchvision, and the encoder backends pulled by
# segmentation-models-pytorch): installing them with a bare
# ``pip install`` would fetch a CPU-only torch wheel from PyPI and
# silently clobber a CUDA-enabled torch, disabling the GPU.  The warning
# therefore tells the operator to install them matching their existing
# torch / CUDA build rather than printing a plain pip command.
_OPTIONAL: List[Tuple[str, str]] = [
    # Segmentation runtime: matplotlib and albumentations are imported when
    # the Mask2Former segmenter loads (evaluate_test_images_v2 imports
    # matplotlib at module top; mask2former_api imports albumentations to
    # unpickle stored transforms), so every segmentation run needs them.
    ("matplotlib",    "matplotlib"),
    ("albumentations", "albumentations"),
    # Shape metrics used by the stereotype feature extraction and the mask
    # metrics pipeline (fusion_tsm.metrics_lib.shape_metrics).
    ("mahotas",       "mahotas"),
    # Skeleton graph of the worm axis: the v20 regeneration chain imports
    # ``skan.Skeleton`` at module top in production/cadrage_squelette_skan2,
    # which production/enrich_metrics_v20 imports at module top too.  Without
    # it the Forecast regeneration states run dies on an ImportError before
    # the first image.  Pure Python/NumPy (pulls networkx), so it is safe to
    # install with a plain pip/conda command.
    ("skan",          "skan"),
    # Export head-detection models (eyespots, blastema, state) for the
    # Forecast regeneration states / Eyespots counter / editor pipelines.
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("timm",          "timm"),
    # ``datasets.py`` at the application root imports torchvision directly
    # (transforms of the classification loaders), on top of being required by
    # segmentation-models-pytorch and timm.
    ("torchvision",   "torchvision"),
]
# NOTE. ``pretrainedmodels`` and ``efficientnet-pytorch`` are deliberately NOT
# declared. They were install_requires of segmentation-models-pytorch < 0.4
# only -- pip pulls them in automatically on those versions, and 0.4+ (the
# reference environment uses 0.5) does not need them at all. Listing them here
# would flag a perfectly working install as incomplete, and the About page now
# turns red and pulls the operator to it on any missing package, so a false
# positive would fire at every launch.


# Packages each processing chain needs IN ITS SUBPROCESS, on top of the core
# requirements.  The workers probe their own entry before spawning the
# subprocess so a missing package is reported in the page's execution log as
# a readable message instead of killing the run with a raw ImportError.
#
# The names below are looked up in ``_OPTIONAL`` / ``_REQUIRED`` so a package
# is declared in exactly one place.
_SEGMENTATION = (
    "torch", "transformers", "cv2", "numpy", "skimage",
    "albumentations", "matplotlib",
)
_SMP = ("segmentation_models_pytorch", "timm", "torchvision")

PIPELINE_REQUIREMENTS: "dict[str, Tuple[str, ...]]" = {
    # Images -> Masks (page "Image segmentation").
    "segmentation": _SEGMENTATION + ("segmentation_models_pytorch", "timm",
                                     "torchvision"),
    # Videos/Images -> Stereotype (page "Stereotype classification").
    "stereotype": _SEGMENTATION + ("lightgbm", "mahotas", "sklearn",
                                   "pandas", "scipy"),
    # Images -> States (page "Forecast regeneration states", v20 chain).
    "regeneration": _SEGMENTATION + _SMP + ("skan", "pandas", "scipy",
                                            "sklearn"),
    # Images -> N Eyespots (page "Eyespots counter").
    "eyespots": _SEGMENTATION + _SMP + ("pandas",),
}

# Human-readable page name per chain, used in the error shown in the log.
PIPELINE_LABELS = {
    "segmentation": "Image segmentation",
    "stereotype": "Stereotype classification",
    "regeneration": "Forecast regeneration states",
    "eyespots": "Eyespots counter",
}


def _entry(import_name: str) -> Tuple[str, str]:
    """Return the ``(import_name, pip_name)`` pair declared for a module."""
    for entries in (_REQUIRED, _OPTIONAL):
        for name, pip_name in entries:
            if name == import_name:
                return (name, pip_name)
    return (import_name, import_name)


def missing_for(pipeline: str) -> List[Tuple[str, str]]:
    """Return the packages a processing chain needs but cannot import.

    ``pipeline`` is a key of :data:`PIPELINE_REQUIREMENTS`.  An unknown key
    yields an empty list so a new chain never blocks itself by accident.
    """
    names = PIPELINE_REQUIREMENTS.get(pipeline, ())
    seen, entries = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            entries.append(_entry(n))
    return find_missing(entries)


def _fusion_tsm_dir() -> str:
    """Absolute path of ``flatmorphai/notebook_api/fusion_tsm`` (this file is in
    ``flatmorphai/``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "notebook_api", "fusion_tsm")


def _metrics_tsm_imports() -> "Tuple[List[str], str]":
    """Module-level bare import names of ``metrics_tsm.py`` and the metrics_lib
    path. Parsed statically (``ast``) so nothing is imported here."""
    import ast
    d = _fusion_tsm_dir()
    with open(os.path.join(d, "metrics_tsm.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names: List[str] = []
    for node in tree.body:  # module level only (what fails at import time)
        if isinstance(node, ast.Import):
            names += [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module.split(".")[0])
    return names, os.path.join(d, "metrics_lib")


def missing_descriptor_modules(pipeline: str) -> List[str]:
    """Bundled descriptor SOURCE modules a chain needs but cannot resolve.

    The stereotype chain's ``metrics_tsm`` imports its descriptor modules by
    bare name from ``metrics_lib/`` (they are source files shipped with the
    app, NOT pip packages). A file missing from that folder makes the whole run
    die with a raw ``ModuleNotFoundError`` at import time -- the package gate
    above never sees it. This probes each module (with metrics_lib on the path,
    exactly as the run does) and returns the names that do not resolve.

    Only genuine bundled modules are reported: names that are a declared pip
    package (handled by :func:`missing_for`) or a standard-library module are
    excluded, so nothing is double-counted. Chains other than ``stereotype``
    have no such bundle and return an empty list.
    """
    if pipeline != "stereotype":
        return []
    try:
        names, lib = _metrics_tsm_imports()
    except Exception:  # noqa: BLE001 - never block a run on this probe
        return []
    declared = {imp for imp, _ in (_REQUIRED + _OPTIONAL)}
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    missing: List[str] = []
    sys.path.insert(0, lib)
    try:
        for name in dict.fromkeys(names):  # de-dup, keep order
            if name in declared or name in stdlib:
                continue
            try:
                spec = importlib.util.find_spec(name)
            except (ValueError, ModuleNotFoundError, ImportError):
                spec = None
            if spec is None:
                missing.append(name)
    finally:
        try:
            sys.path.remove(lib)
        except ValueError:
            pass
    return missing


def format_descriptor_module_error(pipeline: str, modules: List[str]) -> str:
    """One readable message naming the missing bundled descriptor modules."""
    label = PIPELINE_LABELS.get(pipeline, pipeline)
    return (
        "%s cannot run: %d descriptor module(s) are missing from the bundled "
        "feature-extraction folder (flatmorphai/notebook_api/fusion_tsm/"
        "metrics_lib/). These are source files shipped with the app, not pip "
        "packages.\nMissing: %s\nRestore them from the stereotype v4j export "
        "bundle (stereotypes_v4j/.../feature_extraction/), which is the "
        "reference copy the golden descriptors were computed with."
        % (label, len(modules), ", ".join(m + ".py" for m in modules))
    )


def format_pipeline_error(pipeline: str,
                          missing: List[Tuple[str, str]]) -> str:
    """One readable message naming the missing packages and how to install."""
    label = PIPELINE_LABELS.get(pipeline, pipeline)
    pip_names = sorted({p for _, p in missing})
    torch_family = TORCH_FAMILY
    lines = [
        "%s cannot run: %d required Python package(s) are missing from this "
        "environment." % (label, len(missing)),
        "Missing: " + ", ".join(
            n if n == p else "%s (pip name: %s)" % (n, p) for n, p in missing
        ),
    ]
    safe = [p for p in pip_names if p not in torch_family]
    risky = [p for p in pip_names if p in torch_family]
    if safe:
        lines.append("Install with:  pip install " + " ".join(safe)
                     + "   (or: conda install -c conda-forge "
                     + " ".join(safe) + ")")
    if risky:
        lines.append(
            "Also missing (torch-family): " + " ".join(risky)
            + " - install these matching your EXISTING torch / CUDA build, "
            "from the same channel you used for torch. A bare 'pip install' "
            "would pull a CPU-only torch wheel over a CUDA build and disable "
            "the GPU."
        )
    return "\n".join(lines)


def find_missing(entries: "List[Tuple[str, str]] | None" = None) -> List[Tuple[str, str]]:
    """Return the ``(import_name, pip_name)`` tuples that cannot be resolved.

    Defaults to the core (fatal) requirements; pass ``_OPTIONAL`` to probe the
    pipeline-only extras instead.
    """
    missing: List[Tuple[str, str]] = []
    for import_name, pip_name in (entries if entries is not None else _REQUIRED):
        try:
            spec = importlib.util.find_spec(import_name)
        except (ValueError, ModuleNotFoundError, ImportError):
            spec = None
        if spec is None:
            missing.append((import_name, pip_name))
    return missing


def _format_error(missing: List[Tuple[str, str]]) -> str:
    sep = "=" * 72
    lines: List[str] = [
        "",
        sep,
        "FlatMorphAI cannot start: required Python packages are missing.",
        sep,
        "",
        "Missing package(s):",
    ]
    for import_name, pip_name in missing:
        if import_name == pip_name:
            lines.append(f"  - {import_name}")
        else:
            lines.append(f"  - {import_name}  (pip name: {pip_name})")
    # De-duplicate the pip names in case future entries share a
    # single wheel (they do not today but the guard is cheap).
    pip_names = " ".join(sorted({p for _, p in missing}))
    lines += [
        "",
        "Install them with:",
        f"  pip install {pip_names}",
        "",
        "If you use conda, run instead:",
        f"  conda install -c conda-forge {pip_names}",
        "",
        sep,
        "",
    ]
    return "\n".join(lines)


def try_auto_install(missing: List[Tuple[str, str]]) -> bool:
    """Install the missing pip packages in the current interpreter.

    Only runs when the environment variable ``FLATMORPHAI_AUTO_INSTALL``
    is set to ``1`` (opt-in, so a plain launch never modifies the
    environment silently).  Returns True when every package resolves
    after the installation attempt.
    """
    if os.environ.get("FLATMORPHAI_AUTO_INSTALL", "0") != "1":
        return False
    pip_names = sorted({p for _, p in missing})
    sys.stderr.write(
        "[FlatMorphAI] FLATMORPHAI_AUTO_INSTALL=1: installing "
        + " ".join(pip_names) + "\n"
    )
    sys.stderr.flush()
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *pip_names]
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[FlatMorphAI] automatic install failed: {exc}\n")
        return False
    importlib.invalidate_caches()
    return not find_missing()


def _warn_optional(missing: List[Tuple[str, str]]) -> None:
    """Print a NON-fatal warning for missing pipeline-only extras.

    Deliberately does not print a bare ``pip install`` command: several of
    these are torch-family and a plain pip install would pull a CPU-only
    torch wheel over a CUDA build and disable the GPU.
    """
    torch_family = TORCH_FAMILY
    missing_pip = {p for _, p in missing}
    lines = [
        "[FlatMorphAI] WARNING - package(s) needed to RUN computations are "
        "missing (the app still starts, but the matching page will refuse to "
        "run until they are installed):",
    ]
    # Name the pages each missing package blocks, so the operator knows what
    # is affected without reading the source.
    for import_name, pip_name in sorted(missing, key=lambda e: e[1]):
        pages = [PIPELINE_LABELS[k] for k, names in PIPELINE_REQUIREMENTS.items()
                 if import_name in names]
        if pages:
            lines.append("    %s  ->  needed by: %s"
                         % (pip_name, ", ".join(sorted(pages))))
        else:
            lines.append("    %s" % pip_name)
    if missing_pip & torch_family:
        lines.append(
            "  Some are torch-family: install them matching your EXISTING "
            "torch / CUDA build (same channel you used for torch). Do NOT run "
            "a bare 'pip install torchvision' - it would pull a CPU torch "
            "wheel over a CUDA build and disable the GPU."
        )
    non_torch = missing_pip - torch_family
    if non_torch:
        lead = "  The rest are" if (missing_pip & torch_family) else "  These are"
        lines.append(
            lead + " safe to install directly: 'pip install "
            + " ".join(sorted(non_torch)) + "' (or 'conda install -c "
            "conda-forge " + " ".join(sorted(non_torch)) + "')."
        )
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


# Light, import-safe packages that are "present but can break" (a mixed
# pip/conda install leaves the Python files and the compiled extension on
# different versions). ``find_spec`` cannot see this - only a real import
# does - so these are probed by importing them. Keep the list to packages
# whose import has no heavy side effects (no torch / CUDA).
_IMPORT_PROBES = ("safetensors", "charset_normalizer")


def check_critical_imports() -> None:
    """Import a few light packages to catch 'installed but broken' cases.

    ``find_spec`` only confirms a package is locatable; it cannot detect a
    version-skewed install (e.g. safetensors' Python layer newer than its
    compiled ``_safetensors_rust`` library). This surfaces such breakage at
    startup - in the terminal and session log - instead of mid-computation.
    Non-fatal and best-effort.
    """
    for mod in _IMPORT_PROBES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                "[FlatMorphAI] WARNING - '%s' is installed but FAILS to import "
                "(so pipelines that need it will fail):\n    %s: %s\n"
                "  This is usually a mixed pip/conda install (Python files and "
                "the compiled library on different versions). Reinstall it "
                "cleanly: 'pip uninstall -y %s' then "
                "'conda install -c conda-forge %s'.\n"
                % (mod, type(exc).__name__, exc, mod, mod)
            )
            sys.stderr.flush()


def check_gpu() -> None:
    """Report, at startup, whether PyTorch imports and sees a CUDA GPU.

    The probe runs in a SHORT-LIVED CHILD PROCESS, not in the GUI process. That
    matters: calling ``torch.cuda.*`` in the GUI creates a CUDA context that
    reserves VRAM for the whole session -- memory the segmentation /
    regeneration SUBPROCESSES then cannot use, which caused out-of-memory
    errors a subprocess hit while the in-process pipelines (same model) did
    not. By probing in a child that exits immediately, the GUI keeps a ZERO
    GPU footprint and every pipeline subprocess gets the whole card.

    Non-fatal and best-effort. Still catches a broken/CPU-only torch or an
    import-breaking package conflict, reported in the terminal and session log.
    """
    code = (
        "import torch\n"
        "try:\n"
        "    ok = bool(torch.cuda.is_available())\n"
        "    name = torch.cuda.get_device_name(0) if ok else ''\n"
        "    cu = getattr(getattr(torch, 'version', None), 'cuda', None)\n"
        "    print('RESULT|%d|%s|%s|%s' % (int(ok), torch.__version__, cu, name))\n"
        "except Exception as e:\n"
        "    print('IMPORTERR|%s|%s' % (type(e).__name__, e))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[FlatMorphAI] GPU check: could not run probe (%s)\n" % exc)
        sys.stderr.flush()
        return

    line = ""
    for raw in (proc.stdout or "").splitlines():
        if raw.startswith(("RESULT|", "IMPORTERR|")):
            line = raw.strip()
    if line.startswith("RESULT|"):
        _, ok, ver, cu, name = (line.split("|", 4) + ["", "", "", ""])[:5]
        if ok == "1":
            sys.stderr.write(
                "[FlatMorphAI] GPU check: OK - CUDA available on %s "
                "(torch %s, CUDA build %s). GUI holds no GPU memory; the whole "
                "card is free for the pipelines.\n" % (name, ver, cu)
            )
        else:
            sys.stderr.write(
                "[FlatMorphAI] GPU check: WARNING - no CUDA GPU detected "
                "(torch %s, CUDA build %s). Computations will run on CPU "
                "(slow). If you expect a GPU, your torch build is likely "
                "CPU-only or mismatched with your driver's CUDA.\n" % (ver, cu)
            )
    elif line.startswith("IMPORTERR|"):
        _, etype, emsg = (line.split("|", 2) + ["", ""])[:3]
        sys.stderr.write(
            "[FlatMorphAI] GPU check: WARNING - PyTorch could not be imported, "
            "so the GPU is unavailable and computations will fail.\n"
            "    %s: %s\n"
            "  This is almost always a package conflict (protobuf/abseil ABI "
            "mismatch, or a broken torch install). Fix torch before running "
            "any computation.\n" % (etype, emsg)
        )
    else:
        sys.stderr.write(
            "[FlatMorphAI] GPU check: probe returned no result.\n%s\n"
            % ((proc.stderr or "").strip()[:500])
        )
    sys.stderr.flush()


def enforce(include_optional: bool = False) -> None:
    """Gate startup on the CORE packages; only WARN about the extras.

    Core packages (the app imports them on launch) are fatal: when missing
    they are installed if ``FLATMORPHAI_AUTO_INSTALL=1`` is set, otherwise the
    process exits with a clear message.  The pipeline-only extras in
    ``_OPTIONAL`` never block launch and are never auto-installed - they are
    reported as a non-fatal warning so a GPU-configured environment is never
    perturbed by this check.

    Called from :mod:`flatmorphai.__main__` and :mod:`flatmorphai.main`
    BEFORE any PySide6 / torch / cv2 import -- which is also BEFORE the
    session log exists.  That is why the extras warning is NOT emitted here
    by default: written this early it would reach the raw terminal only, and
    an operator launching from a desktop shortcut (no console attached) would
    never see it.  :func:`warn_missing_optional` emits it instead, once
    :mod:`flatmorphai.logging_setup` has teed stderr into the session log,
    and :func:`optional_report` feeds the startup dialog and the About page.
    """
    missing = find_missing()
    if missing and try_auto_install(missing):
        missing = []
    if missing:
        sys.stderr.write(_format_error(missing))
        sys.stderr.flush()
        sys.exit(1)
    if include_optional:
        warn_missing_optional()


def warn_missing_optional() -> List[Tuple[str, str]]:
    """Emit the non-fatal extras warning; return what is missing.

    Safe to call once the session log is in place so the warning is captured
    on disk as well as printed.
    """
    optional_missing = find_missing(_OPTIONAL)
    if optional_missing:
        _warn_optional(optional_missing)
    return optional_missing


def warn_calc_versions() -> "List[Tuple[str, str, str, str]]":
    """Print a NON-fatal alert when a calc-sensitive package version differs.

    Mirrors :func:`warn_missing_optional`: emitted once the session log is in
    place so it is captured on disk and shown in the terminal. Returns the same
    rows as :func:`calc_version_warnings` so the caller can reuse them.
    """
    warns = calc_version_warnings()
    if warns:
        lines = [
            "[FlatMorphAI] WARNING - calculation-sensitive package version(s) "
            "differ from the validated golden environment. Descriptor values "
            "and predictions MAY DIFFER from the reference:",
        ]
        for pip_name, installed, validated, reason in warns:
            lines.append("    %s : installed %s, validated %s"
                         % (pip_name, installed or "?", validated or "?"))
            lines.append("        %s" % reason)
        lines.append("  Reinstall the validated version to guarantee "
                     "reproducibility (see the About page, Dependencies card).")
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()
    return warns


def optional_report() -> "List[Tuple[str, str, List[str]]]":
    """Missing extras as ``(pip_name, import_name, [page labels])`` rows.

    Used by the GUI (startup dialog and About page) so the operator sees the
    same information as the terminal, in the application itself.
    """
    rows = []
    for import_name, pip_name in find_missing(_OPTIONAL):
        pages = sorted(
            PIPELINE_LABELS[k] for k, names in PIPELINE_REQUIREMENTS.items()
            if import_name in names
        )
        rows.append((pip_name, import_name, pages))
    return sorted(rows)


#: Package versions the stereotype / regeneration DESCRIPTORS were validated
#: against -- the environment in which the LightGBM v4j golden table was
#: reproduced to the 2nd decimal (RECALCULES ~1e-6, inherited mask groups
#: ~1e-14). These are keyed by pip (distribution) name.
#:
#: WHY THIS MATTERS: several of these libraries changed numerical behaviour
#: between releases in ways that DIRECTLY CHANGE descriptor values fed to the
#: classifier -- not just performance. The golden descriptors were produced
#: with the exact versions below; a different version of a ``calc_sensitive``
#: package can shift the inputs and therefore the stereotype / regeneration
#: prediction. The About page flags any such mismatch.
VALIDATED_VERSIONS: "dict[str, str]" = {
    "opencv-python": "5.0.0",
    "numpy": "2.2.6",
    "scikit-image": "0.25.2",
    "mahotas": "1.4.18",
    "torch": "2.5.1",
}

#: Packages where a version DIFFERENT from VALIDATED_VERSIONS can FALSIFY the
#: descriptor computation (change the numbers the model sees), with the reason.
#: Verified in practice: OpenCV's CLAHE / INTER_AREA resize / convexityDefects,
#: scikit-image's skeletonize / frangi-sato-meijering / regionprops, and
#: mahotas' Zernike moments all moved descriptor values across versions.
CALC_SENSITIVE: "dict[str, str]" = {
    "opencv-python": "CLAHE, INTER_AREA resize and convexityDefects changed "
                     "between OpenCV 4.x and 5.x; they feed the texture, shape "
                     "and terminal descriptors. OpenCV 5.0.0 reproduces the golden.",
    "scikit-image": "skeletonize (body axis), frangi/sato/meijering vesselness "
                    "and regionprops changed across releases; they feed the "
                    "axial, geodesic and shape descriptors.",
    "mahotas": "Zernike moments (sh_zern_*) depend on the mahotas version.",
    "numpy": "float reductions / FFT can shift the last digits of many "
             "descriptors; pinned for bit-stability.",
}


def _dist_version(pip_name: str) -> "str | None":
    """Installed version of a distribution by its pip name, WITHOUT importing it.

    Uses :mod:`importlib.metadata` so this pre-boot module never imports a heavy
    package (torch / cv2) just to read a version. OpenCV ships under several
    distribution names (opencv-python, -headless, -contrib); the first that
    resolves wins. Returns None when the distribution is absent.
    """
    try:
        from importlib import metadata as _md
    except Exception:  # noqa: BLE001 - very old interpreters
        return None
    candidates = [pip_name]
    if pip_name == "opencv-python":
        candidates += ["opencv-python-headless", "opencv-contrib-python",
                       "opencv-contrib-python-headless"]
    for name in candidates:
        try:
            return _md.version(name)
        except Exception:  # noqa: BLE001 - PackageNotFoundError and friends
            continue
    return None


def _version_tuple(v: "str | None") -> "Tuple[int, ...]":
    """Leading numeric components of a version string, e.g. '5.0.0.93' -> (5,0,0,93)."""
    if not v:
        return ()
    out = []
    for part in re.split(r"[.\-+]", v):
        if part.isdigit():
            out.append(int(part))
        else:
            break
    return tuple(out)


def version_report() -> "List[Tuple[str, str, str, str, bool, str]]":
    """Per declared package: ``(pip_name, installed, validated, sensitive, verdict)``.

    Actually returns ``(pip_name, import_name, installed, validated, sensitive,
    verdict)`` where ``verdict`` is one of:

    - ``"missing"``   : not installed;
    - ``"match"``     : installed == validated (or no validated version pinned);
    - ``"minor_diff"``: a ``calc_sensitive`` package whose MAJOR.MINOR differs
      from the validated version -- descriptor values may be falsified (alert);
    - ``"patch_diff"``: a ``calc_sensitive`` package differing only in the patch
      / build suffix -- usually safe, shown as a mild note;
    - ``"info_diff"`` : a non-sensitive package on a different version (FYI only).
    """
    rows: "List[Tuple[str, str, str, str, bool, str]]" = []
    for entries in (_REQUIRED, _OPTIONAL):
        for import_name, pip_name in entries:
            installed = _dist_version(pip_name)
            validated = VALIDATED_VERSIONS.get(pip_name, "")
            sensitive = pip_name in CALC_SENSITIVE
            if installed is None:
                verdict = "missing"
            elif not validated or installed == validated:
                verdict = "match"
            elif _version_tuple(installed)[:2] == _version_tuple(validated)[:2]:
                verdict = "patch_diff" if sensitive else "info_diff"
            else:
                verdict = "minor_diff" if sensitive else "info_diff"
            rows.append((pip_name, import_name, installed or "",
                         validated, sensitive, verdict))
    return rows


def calc_version_warnings() -> "List[Tuple[str, str, str, str]]":
    """Calc-sensitive packages whose version may change descriptor values.

    Returns ``(pip_name, installed, validated, reason)`` for every
    ``calc_sensitive`` package whose MAJOR.MINOR differs from the validated
    version (the ``minor_diff`` verdict). These are the ones that can falsify
    the stereotype / regeneration computation; the About page shows them as a
    red alert and the startup log prints them.
    """
    out = []
    for pip_name, _imp, installed, validated, sensitive, verdict in version_report():
        if sensitive and verdict == "minor_diff":
            out.append((pip_name, installed, validated,
                        CALC_SENSITIVE.get(pip_name, "")))
    return out


def installed_report() -> "List[Tuple[str, str, bool, bool]]":
    """Every declared package as ``(pip_name, import_name, found, core)``.

    Powers the "Dependencies" card of the About page so the operator can
    check the environment at any time, not only when a warning scrolled past.
    """
    rows = []
    for entries, core in ((_REQUIRED, True), (_OPTIONAL, False)):
        for import_name, pip_name in entries:
            try:
                found = importlib.util.find_spec(import_name) is not None
            except (ValueError, ModuleNotFoundError, ImportError):
                found = False
            rows.append((pip_name, import_name, found, core))
    return rows


#: Packages whose install must follow the operator's existing torch build.
#: They are deliberately excluded from the generated commands: a bare
#: ``pip install torchvision`` resolves a CPU-only torch wheel and silently
#: replaces a CUDA build, disabling the GPU for every pipeline.
TORCH_FAMILY = frozenset({
    "torch", "torchvision", "segmentation-models-pytorch", "timm",
})


def installable_names(missing: "List[Tuple[str, str]]") -> List[str]:
    """Sorted pip names of the missing packages that are safe to install."""
    return sorted({p for _, p in missing} - TORCH_FAMILY)


def pip_command(missing: "List[Tuple[str, str]]") -> str:
    """``pip install`` line for the non-torch-family packages missing.

    Several packages are separated by spaces, which is what pip expects for
    multiple requirements in one invocation.
    """
    names = installable_names(missing)
    return ("pip install " + " ".join(names)) if names else ""


def conda_command(missing: "List[Tuple[str, str]]") -> str:
    """``conda install`` equivalent, from conda-forge (same packages)."""
    names = installable_names(missing)
    return ("conda install -c conda-forge " + " ".join(names)) if names else ""


# Backwards-compatible alias: the pip form was the only command before the
# About page started offering conda as well.
install_command = pip_command


if __name__ == "__main__":  # pragma: no cover
    enforce(include_optional=True)
