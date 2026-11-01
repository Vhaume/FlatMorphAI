# -*- coding: utf-8 -*-
# MINIMAL ENVIRONMENT: what the application REALLY imports, and nothing else.
#
# `conda env export` describes the conda environment, not the application. On
# the reference machine it reports numpy 2.0 while 2.2.6 is installed (pip
# upgraded over it), it omits torch, opencv, PySide6 and transformers
# (installed by pip, hence invisible to conda), and it carries dozens of
# packages belonging to other work sharing the same environment.
#
# This script starts from the other end: it reads the CODE.
#
#   1. syntax analysis (ast) of every file of the application, of the LightGBM
#      model bundle and of the root modules the application loads -- not one
#      import is executed;
#   2. union with the lists declared in flatmorphai/dependency_check.py.
#      This step is not a precaution: lightgbm appears NOWHERE as an import
#      statement in the application -- it is imported by
#      lightgbm_stereotype_classifier/classifier.py, loaded dynamically. A
#      syntax analysis alone would lose the classifier itself;
#   3. resolution import name -> distribution name -> installed version, via
#      importlib.metadata, without importing the packages;
#   4. writing of requirements.txt and environment.yml.
#
# The five packages of VALIDATED_VERSIONS are pinned with ==, with the exact
# reason why they are not to be changed lightly as a comment. The others are
# pinned to the installed version, which can be relaxed knowingly.
#
# Usage (from the FlatMorphAI root):
#   python tools/make_environment.py          (writes both files)
#   python tools/make_environment.py --check  (prints only, writes nothing)
import ast
import os
import sys

# The repository root: this script lives one folder below it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = sys.__stdout__

#: Trees walked: the application, the model bundle loaded at run time, and the
#: root modules the application imports by their bare name.
#: ``model_function.py`` is deliberately NOT listed: it belonged to the
#: ConvNeXt-V2 path, removed on 31 October 2026, and nothing imports it any
#: more. Scanning it would put fastervit back into requirements.txt.
SCAN_DIRS = ["flatmorphai", "lightgbm_stereotype_classifier"]
SCAN_FILES = [os.path.join("mask2former_modules", "datasets.py"),
              os.path.join("mask2former_modules", "custom_dataset.py")]
#: Folder of the Mask2Former support modules imported by their bare names.
SUPPORT_DIR = "mask2former_modules"

#: Pinned even though no import names them (see main()).
ALSO_PIN = ["llvmlite", "opencv-python-headless"]

#: Import names whose distribution does not carry the same name.
ALIAS = {
    "cv2": "opencv-python", "PIL": "Pillow", "skimage": "scikit-image",
    "sklearn": "scikit-learn", "segmentation_models_pytorch":
    "segmentation-models-pytorch", "yaml": "PyYAML",
}

#: Bare names that are SOURCES shipped with the application (loaded from the
#: export folders placed on sys.path), not packages to install.
BUNDLED = {
    "cadrage_squelette", "cadrage_squelette_skan2", "ancrage_cascade",
    "classifier", "compute_expert_image_features", "compute_new_metrics",
    "feature_extraction", "flatmorph_planaria_model",
    "proto_production_features", "resolution_upgrade_784", "parameter_reader",
    "anchor_cascade", "preprocessing", "eye_detection", "blastema_detection",
    "head_detection", "segment_mask", "stage_prediction", "darkness_index",
    "evaluate_test_images_v2", "mask2former_api", "regen_pipeline",
    "eyespot_shared", "metrics_tsm",
}


def say(m=""):
    OUT.write(m + "\n"); OUT.flush()


def _local_names():
    """Project module names: every .py file and every walked folder."""
    names = set(BUNDLED)
    support = os.path.join(ROOT, SUPPORT_DIR)
    for f in (os.listdir(support) if os.path.isdir(support) else []):
        if f.endswith(".py"):
            names.add(f[:-3])
    for f in os.listdir(ROOT):
        if f.endswith(".py"):
            names.add(f[:-3])
        elif os.path.isdir(os.path.join(ROOT, f)):
            names.add(f)
    for d in SCAN_DIRS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            names.update(dn)
            names.update(f[:-3] for f in fn if f.endswith(".py"))
    return names


def scan():
    """Top-level names imported, excluding the standard library and the project."""
    std = set(getattr(sys, "stdlib_module_names", ()))
    local = _local_names()
    found = {}
    paths = [os.path.join(ROOT, f) for f in SCAN_FILES]
    for d in SCAN_DIRS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            paths += [os.path.join(dp, f) for f in fn
                      if f.endswith(".py") and ".bak" not in f]
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module.split(".")[0]]
            else:
                continue
            for nm in names:
                if nm in std or nm in local or nm.startswith("_"):
                    continue
                found.setdefault(nm, set()).add(os.path.relpath(p, ROOT))
    return found


def declared():
    """The dependency_check lists: the implicit imports are in there."""
    try:
        from flatmorphai import dependency_check as DC
        return (list(DC._REQUIRED), list(DC._OPTIONAL),  # noqa: SLF001
                dict(DC.VALIDATED_VERSIONS), dict(DC.CALC_SENSITIVE))
    except Exception as exc:  # noqa: BLE001
        say("dependency_check unreadable (%s): carrying on without it." % exc)
        return [], [], {}, {}


def _mm(v):
    """Major.minor, the comparison grain of dependency_check."""
    return ".".join((v or "").split(".")[:2])


def _note_pypi(v):
    """Flag a version pip will not be able to find on PyPI."""
    v = v or ""
    if ".dev" in v or "+" in v or v.startswith("0.0.0"):
        return ("   # local build / third-party channel: ABSENT from PyPI, "
                "replace with a published version before publishing")
    return ""


def coherence(presents):
    """Do the pinned versions contradict one another?

    Each package declares what it requires of the others (Requires-Dist). An
    environment built by successive additions can work perfectly well while
    being inconsistent on paper: pip never had to resolve the whole set at
    once. The day someone installs requirements.txt on a fresh machine, the
    full resolution applies and the inconsistency becomes a failure. It is
    detected here, before publication.
    """
    try:
        from importlib import metadata as md
        from packaging.requirements import Requirement
        from packaging.version import Version
    except Exception:  # noqa: BLE001
        say("\n(coherence check impossible: packaging unavailable)")
        return
    low = {k.lower().replace("_", "-"): v for k, v in presents.items()}
    clashes = []
    for d in sorted(presents):
        try:
            reqs = md.requires(d) or []
        except Exception:  # noqa: BLE001
            continue
        for raw in reqs:
            try:
                r = Requirement(raw)
            except Exception:  # noqa: BLE001
                continue
            if r.marker is not None and not r.marker.evaluate():
                continue
            name = r.name.lower().replace("_", "-")
            got = low.get(name)
            if not got or not r.specifier:
                continue
            try:
                if not r.specifier.contains(Version(got), prereleases=True):
                    clashes.append((d, r.name, str(r.specifier), got))
            except Exception:  # noqa: BLE001
                continue
    if clashes:
        say("\nINCONSISTENCY between pinned versions -- a fresh installation")
        say("from requirements.txt would fail or downgrade a package:")
        for a, b, spec, got in clashes:
            say("   %-28s requires %s%s, yet %s is pinned" % (a, b, spec, got))
        say("   The current environment works because it was built by")
        say("   successive additions, with no whole-set resolution.")
    else:
        say("\nCoherence of the pinned versions: no declared contradiction.")
    return clashes


def version_of(dist):
    try:
        from importlib import metadata as md
        return md.version(dist)
    except Exception:  # noqa: BLE001
        return None


def main():
    check_only = "--check" in sys.argv
    found = scan()
    req, opt, valid, why = declared()

    dists = {}          # distribution name -> (version, origin)
    src = {}            # distribution name -> files that import it
    for imp in sorted(found):
        d = ALIAS.get(imp, imp)
        dists[d] = [version_of(d), "code"]
        src[d] = sorted(found[imp])
    for imp, pip in req + opt:
        d = ALIAS.get(imp, pip)
        if d not in dists:
            dists[d] = [version_of(d), "declared"]
    for d in list(valid):
        if d not in dists:
            dists[d] = [version_of(d), "pinned"]
    # Distributions that no import statement names but that decide what code
    # runs: llvmlite is numba's compiler and must match it; the headless OpenCV
    # wheel is pulled in by albumentations and installs the same cv2/ folder as
    # opencv-python, so it must carry the same version or it may silently
    # replace the calculation-sensitive OpenCV.
    for d in ALSO_PIN:
        if d not in dists:
            dists[d] = [version_of(d), "pinned"]

    missing = sorted(d for d, (v, _o) in dists.items() if v is None)
    presents = {d: v for d, (v, _o) in dists.items() if v}

    say("%d third-party package(s) retained; %d found in the current interpreter"
        % (len(dists), len(presents)))
    say("    %s\n" % sys.executable)
    say("%-30s %-14s %s" % ("package", "version", "origin"))
    for d in sorted(dists):
        v, o = dists[d]
        mark = "  <- calculation" if d in valid else ""
        say("%-30s %-14s %s%s" % (d, v or "ABSENT", o, mark))
    rare = {d: f for d, f in src.items() if len(f) <= 2 and d in presents}
    if rare:
        say("\nPackage(s) pulled in by one or two files only -- worth checking,")
        say("this is where the dependencies of dead code hide:")
        for d in sorted(rare):
            say("   %-26s %s" % (d, ", ".join(rare[d])))
    if missing:
        say("\nNot installed here -- to check before publishing the file:")
        for d in missing:
            say("   %s" % d)
    # a gap with the validated versions is the most important thing to see
    def mm(v):
        parts = (v or "").split(".")
        return ".".join(parts[:2])
    # same criterion as dependency_check.calc_version_warnings: only a
    # MAJOR.MINOR gap counts; 5.0.0.93 and 5.0.0 are the same OpenCV.
    gaps = [(d, valid[d], presents.get(d)) for d in valid
            if presents.get(d) and mm(presents[d]) != mm(valid[d])]
    if gaps:
        say("\nWARNING -- installed version differs from the validated one:")
        for d, want, got in gaps:
            say("   %-20s validated %-10s installed %s" % (d, want, got))
        say("   The descriptors may differ. See CALC_SENSITIVE.")

    clashes = coherence(presents)

    if check_only:
        say("\n(--check: no file written)")
        return

    # ---- requirements.txt ------------------------------------------------
    lines = [
        "# FlatMorphAI 1.0.0 -- the application's dependencies, and those alone.",
        "# Produced by make_environment.py, which reads the code rather than",
        "# the conda environment: that one holds other work as well.",
        "#",
        "# The packages marked CALCULATION change the VALUES of the descriptors",
        "# from one version to the next, hence possibly the forecasted class.",
        "# Do not relax them without replaying test_csv_stereotypies.py.",
        "",
    ]
    if clashes:
        lines += [
            "# ---------------------------------------------------------------",
            "# THESE VERSIONS DO NOT INSTALL IN ONE COMMAND.",
            "# One package declares it requires a version of another that is",
            "# not the one pinned here. The reference environment works",
            "# because it was built by successive additions; pip resolves the",
            "# whole set at once and will refuse.",
            "#",
        ]
        for a, b, spec, got in clashes:
            lines.append("#   %s requires %s%s, yet %s==%s is pinned"
                         % (a, b, spec, b, got))
        lines += [
            "#",
            "# Install this way to obtain EXACTLY these versions:",
        ]
        for a, _b, _spec, _got in clashes:
            lines.append("#   pip install %s==%s --no-deps"
                         % (a, presents.get(a, "")))
        lines += [
            "#   pip install -r requirements.txt",
            "#",
            "# The declared cap is a precaution taken by the package author,",
            "# not an observed incompatibility: the full chain runs with these",
            "# versions. Lifting the cap properly (raising the package that",
            "# sets it) changes compiled code and requires replaying",
            "# test_csv_stereotypies.py and test_csv_regeneration.py.",
            "# ---------------------------------------------------------------",
            "",
        ]
    for d in sorted(presents):
        if d in valid:
            # The PINNED version is the installed one, not the one in
            # VALIDATED_VERSIONS: the installed one produced the golden table,
            # and VALIDATED_VERSIONS is truncated to major.minor for the
            # comparison (5.0.0 there denotes the whole 5.0.0.x series).
            # Writing "opencv-python==5.0.0" would name a version that does
            # not exist on PyPI and would make the installation fail.
            r = why.get(d, "").replace("\n", " ").strip()
            if r:
                lines.append("# CALCULATION: %s" % r)
            if _mm(presents[d]) != _mm(valid[d]):
                lines.append("# WARNING: validated version %s, installed %s"
                             % (valid[d], presents[d]))
        lines.append("%s==%s%s" % (d, presents[d],
                                   _note_pypi(presents[d])))
    lines.append("")
    lines.append("# torch / torchvision: install the wheel matching your CUDA")
    lines.append("# (https://pytorch.org), NOT the default PyPI wheel, which")
    lines.append("# has no GPU support and would silently replace an existing")
    lines.append("# CUDA installation.")
    open(os.path.join(ROOT, "requirements.txt"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")

    # ---- environment.yml -------------------------------------------------
    # Python and pip only. torch must come from the pytorch.org index matching
    # the CUDA version, before requirements.txt; every other version lives in
    # requirements.txt, so the two files cannot drift apart.
    pyver = "%d.%d" % sys.version_info[:2]
    y = ["# FlatMorphAI 1.0.0 -- conda environment.",
         "#",
         "#   conda env create -f environment.yml",
         "#   conda activate flatmorphai",
         "#   python -m pip install torch==%s torchvision==%s \\"
         % (presents.get("torch", "2.5.1"), presents.get("torchvision", "0.20.1")),
         "#       --index-url https://download.pytorch.org/whl/cu121",
         "#   python -m pip install -r requirements.txt",
         "name: flatmorphai",
         "channels:",
         "  - conda-forge",
         "dependencies:",
         "  - python=%s" % pyver,
         "  - pip"]
    open(os.path.join(ROOT, "environment.yml"), "w", encoding="utf-8").write(
        "\n".join(y) + "\n")

    say("\nwritten: requirements.txt (%d package(s)) and environment.yml"
        % len(presents))
    say("Read both before publishing them: this script describes what the code")
    say("imports, not what your hardware requires (CUDA, driver).")


if __name__ == "__main__":
    main()
