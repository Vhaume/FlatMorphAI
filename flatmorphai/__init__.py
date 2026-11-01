"""
FlatMorphAI Dashboard package.

PySide6 application for the planarian video-to-stereotype pipeline.
See README.md for details.
"""

__version__ = "1.0.0"
__app_name__ = "FlatMorphAI Dashboard"


def _put_support_modules_on_path() -> None:
    """Make the Mask2Former support modules importable by their bare names.

    The segmentation code and the segmentation checkpoints refer to modules
    such as ``improved_mask2former_wrapper`` or ``train_tfms_320`` by their
    bare name. In the repository they live in ``<root>/mask2former_modules/``;
    in an older layout they sit directly in ``<root>``, which is on sys.path
    already, and then nothing is done here.

    The folder goes FIRST on sys.path (``datasets.py`` must win over an
    installed Hugging Face ``datasets`` package, as it did at the root), and
    into PYTHONPATH so that every subprocess the application starts (the
    segmentation, descriptor and regeneration drivers) finds it as well.
    Processes started with the ``spawn`` method inherit sys.path anyway.
    """
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    support = os.path.join(root, "mask2former_modules")
    if not os.path.isdir(support):
        return
    if support not in sys.path:
        sys.path.insert(0, support)
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if support not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([support] + parts)


_put_support_modules_on_path()


def _utf8_for_subprocesses() -> None:
    """Native Windows: start every Python subprocess in UTF-8 mode, as the
    application's files are UTF-8 (Linux is UTF-8 already: nothing to do)."""
    import os
    if os.name == "nt":
        os.environ.setdefault("PYTHONUTF8", "1")


_utf8_for_subprocesses()
