"""
Application entry point.

Usage
-----
    python -m flatmorphai
    python flatmorphai/main.py

Performs a one-time Qt application setup, delegates theme and UI
scale handling to :class:`ThemeManager` and shows the
:class:`MainWindow`.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import warnings


# ---------------------------------------------------------------------
# OpenMP / MKL runtime safeguard (must run before torch, MKL or Qt load)
# ---------------------------------------------------------------------
# In conda environments that mix Intel MKL, PyTorch and Qt (e.g. a
# ``gputensorflow`` env that also carries TensorFlow), MKL's own OpenMP
# runtime (libiomp5) and the GNU OpenMP runtime (libgomp, used by Qt and
# NumPy) can both end up loaded in one process and fight on the first
# thread-pool spin-up.  Pinning MKL to the GNU layer makes everything use
# a single OpenMP runtime, which is the safe, recommended remedy.
# ``setdefault`` leaves any value the operator set themselves untouched.
# (The ``KMP_DUPLICATE_LIB_OK`` override is intentionally NOT set: Intel
# documents that it can crash or silently return wrong numbers.)
# Reference: Intel oneMKL "Threading Layer" documentation.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

# Silence Albumentations' online "a new version is available" check (it also
# issues a network request on import). Set before albumentations is imported
# anywhere -- including the subprocess pipelines, which inherit this env.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


# ---------------------------------------------------------------------
# Pre-boot dependency check
# ---------------------------------------------------------------------
# Runs BEFORE any PySide6 / torch / cv2 import below so a missing
# wheel surfaces a clean actionable error naming each missing package
# and the exact ``pip install`` command instead of the cryptic
# ImportError deep in the app code.  Import path uses a relative form
# only when we are being imported as part of the ``flatmorphai``
# package (i.e. via ``python -m flatmorphai``); when executed directly
# via ``python flatmorphai/main.py`` we synthesise the same guard
# via importlib so the check still fires.
try:
    from . import dependency_check as _depcheck
except ImportError:  # pragma: no cover -- ``python main.py`` case
    import importlib.util as _iu
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = _iu.spec_from_file_location(
        "_flatmorphai_depcheck",
        os.path.join(_here, "dependency_check.py"),
    )
    if _spec is not None and _spec.loader is not None:
        _depcheck = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(_depcheck)
    else:
        _depcheck = None

if _depcheck is not None:
    _depcheck.enforce()


# ---------------------------------------------------------------------
# Fatal-signal diagnostics
# ---------------------------------------------------------------------
# ``faulthandler`` intercepts SIGSEGV/SIGBUS/SIGFPE/SIGABRT and prints
# the Python stack of every thread to stderr (and to a log file) right
# before the process dies. Without this, a C-level crash in torch,
# cv2 or a CUDA driver shows up as a bare "Segmentation fault (core
# dumped)" line with no indication of which code path triggered it.
# Policy recommendation: Wilson G. et al., "Good Enough Practices in
# Scientific Computing", PLOS Comput. Biol. 2017,
# DOI: 10.1371/journal.pcbi.1005510, rule 4 (fail loudly, with enough
# context to debug).
def _install_faulthandler() -> None:
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:  # pragma: no cover
        pass
    # Also mirror the crash trace into ~/.flatmorphai/fatal.log so the
    # GUI user can recover it even when the terminal output is gone.
    try:
        log_dir = os.path.join(
            os.path.expanduser("~"), ".flatmorphai",
        )
        os.makedirs(log_dir, exist_ok=True)
        fatal_log_path = os.path.join(log_dir, "fatal.log")
        # The file handle must stay open for the lifetime of the
        # process; attach it to the module so the garbage collector
        # does not close it.
        _install_faulthandler._fatal_log = open(  # type: ignore[attr-defined]
            fatal_log_path, "a", buffering=1,
        )
        faulthandler.enable(
            file=_install_faulthandler._fatal_log,  # type: ignore[attr-defined]
            all_threads=True,
        )
    except Exception:  # pragma: no cover
        pass


_install_faulthandler()


# ---------------------------------------------------------------------
# Persistent session log (errors survive a crash AND echo to the console)
# ---------------------------------------------------------------------
# Opened before the heavy imports below so even an import-time error is
# captured. It tees stdout/stderr to ~/.flatmorphai/logs/session_*.log and
# hooks uncaught exceptions; the in-app log panel also writes every line it
# shows to this file. Combined with the faulthandler fatal trace above, an
# operator always has (a) the Python/pipeline error log and (b) the native
# crash backtrace on disk after any failure -- including a segfault, which
# wipes the terminal before it can be read.
try:
    from . import logging_setup as _logging_setup
except ImportError:  # pragma: no cover -- ``python main.py`` case
    import importlib.util as _iu2
    _spec2 = _iu2.spec_from_file_location(
        "_flatmorphai_logging",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_setup.py"),
    )
    if _spec2 is not None and _spec2.loader is not None:
        _logging_setup = _iu2.module_from_spec(_spec2)
        _spec2.loader.exec_module(_logging_setup)
    else:  # pragma: no cover
        _logging_setup = None

_SESSION_LOG_PATH = _logging_setup.init_session_log() if _logging_setup else None
if _SESSION_LOG_PATH:
    _fatal_log_path = os.path.join(
        os.path.expanduser("~"), ".flatmorphai", "fatal.log"
    )
    sys.stderr.write(
        "[FlatMorphAI] Session log: %s\n"
        "[FlatMorphAI] Fatal-crash trace (incl. segfault): %s\n"
        % (_SESSION_LOG_PATH, _fatal_log_path)
    )
    sys.stderr.flush()


# ---------------------------------------------------------------------
# Pipeline-only packages: warn AFTER the session log is open
# ---------------------------------------------------------------------
# ``enforce()`` above runs before any heavy import, which is also before the
# session log exists -- a warning written there would reach the raw terminal
# only, so an operator launching from a desktop shortcut (no console) would
# never see that, say, ``skan`` is missing and that Forecast regeneration
# states will refuse to run. Emitting it here tees it into the session log
# too, and ``main()`` additionally shows it in a dialog.
_MISSING_OPTIONAL = []
if _depcheck is not None:
    try:
        _MISSING_OPTIONAL = _depcheck.warn_missing_optional()
    except Exception:  # pragma: no cover - never block launch on this
        _MISSING_OPTIONAL = []
    try:
        # Non-fatal: alert when a calc-sensitive package (OpenCV, scikit-image,
        # mahotas, NumPy) differs from the validated golden versions, since that
        # can change descriptor values and predictions. Shown in the About page
        # Dependencies card too.
        _depcheck.warn_calc_versions()
    except Exception:  # pragma: no cover - never block launch on this
        pass


# ---------------------------------------------------------------------
# Qt plugin path sanitisation
# ---------------------------------------------------------------------
# ``opencv-python`` ships its own Qt 5 platform plugins under
# ``site-packages/cv2/qt/plugins`` and, when imported anywhere in the
# interpreter, exports ``QT_QPA_PLATFORM_PLUGIN_PATH`` pointing at that
# directory. PySide6 then tries to load those plugins and fails with
# ``Could not load the Qt platform plugin 'xcb'`` / ``'wayland'``
# because the ABI does not match. Stripping the variable before the
# first PySide6 import forces Qt to use its own bundled plugins. This
# must run before any ``from PySide6...`` statement.
for _var in (
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QT_PLUGIN_PATH",
    "QML2_IMPORT_PATH",
):
    _value = os.environ.get(_var, "")
    if "cv2" in _value or "opencv" in _value.lower():
        os.environ.pop(_var, None)


from pathlib import Path  # noqa: E402

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# GPU / torch startup check. Runs AFTER Qt is imported (so the Qt-before-torch
# import order is preserved) but BEFORE the app modules that import torch, so a
# broken torch or a disabled GPU is reported cleanly at launch - in the
# terminal and the session log - instead of surfacing mid-computation.
if _depcheck is not None:
    _depcheck.check_gpu()
    _depcheck.check_critical_imports()

from . import __app_name__, __version__  # noqa: E402
from .config import AppSettings  # noqa: E402
from .main_window import MainWindow  # noqa: E402
from .theme_manager import ThemeManager  # noqa: E402


# Bundled icon shipped under flatmorphai/resources. Replacing this
# file at install time is enough to rebrand the taskbar and the
# window title-bar icon, so we keep the lookup data-driven.
_APP_ICON_FILENAME = "planaria_icon.png"


def _resolve_app_icon() -> QIcon:
    """Locate the planaria icon on disk and return a :class:`QIcon`."""
    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root / "resources" / _APP_ICON_FILENAME,
        package_root.parent / "resources" / _APP_ICON_FILENAME,
        package_root.parent / _APP_ICON_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return QIcon(str(path))
    return QIcon()


# Suppress noisy warnings that do not carry actionable information for
# the end user. Libraries such as transformers and albumentations emit
# a lot of deprecation or future warnings at import time.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def configure_environment() -> None:
    """Silence a few backend chatter messages without breaking Qt features."""
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*.warning=false")
    # Mouse pointer size. Under WSLg / X11 the cursor theme often has no size
    # set, so the pointer renders minuscule inside the app. The size is driven
    # by XCURSOR_SIZE, which the theme reads when it draws EVERY shape, so the
    # arrow, the open and closed hands of the viewers, the text beam and the
    # busy pointer all keep their own drawing and their animation -- only
    # bigger. This used to be hard-coded to 48; it now follows the value saved
    # in the settings (Theme and scale > Pointer size). A size already set in
    # the environment wins, so a system-wide choice is never overridden.
    if not (os.environ.get("XCURSOR_SIZE", "") or "").strip():
        try:
            from .config import cursor_size_from_disk
            os.environ["XCURSOR_SIZE"] = str(cursor_size_from_disk())
        except Exception:  # noqa: BLE001 - never block the start-up
            os.environ["XCURSOR_SIZE"] = "48"
    # Some WSLg / X setups only honour the cursor size when a theme is named.
    os.environ.setdefault("XCURSOR_THEME", "Adwaita")
    # Second-line defence against the opencv-python Qt plugin clash: if
    # something reintroduces the variable between interpreter start and
    # ``QApplication`` creation (for example a late ``import cv2``) we
    # strip it again here. ``configure_environment`` is always called
    # before :class:`QApplication` is instantiated.
    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        value = os.environ.get(var, "")
        if "cv2" in value or "opencv" in value.lower():
            os.environ.pop(var, None)


def _show_missing_packages_dialog(parent) -> None:
    """Tell the operator, IN THE APP, which extras are missing and how to get them.

    The terminal warning is easy to miss (and absent entirely when the app is
    started from a shortcut), so the same information is shown once at launch
    as a dismissible dialog naming every missing package, the pages it blocks
    and the install command. Set ``FLATMORPHAI_NO_DEP_DIALOG=1`` to silence it
    on a machine where the operator knowingly runs a partial install.
    """
    if not _MISSING_OPTIONAL or _depcheck is None:
        return
    if os.environ.get("FLATMORPHAI_NO_DEP_DIALOG", "0") == "1":
        return
    from PySide6.QtWidgets import QMessageBox  # noqa: E402

    rows = _depcheck.optional_report()
    lines = [
        "%s is installed but %d package(s) its processing chains need are "
        "missing." % (__app_name__, len(rows)),
        "",
        "The application still starts, but the pages below will refuse to run "
        "until you install them:",
        "",
    ]
    for pip_name, _import_name, pages in rows:
        lines.append("  • %s" % pip_name
                     + ("   — needed by: " + ", ".join(pages) if pages else ""))
    cmd = _depcheck.install_command(_MISSING_OPTIONAL)
    torch_family = {"torch", "torchvision", "segmentation-models-pytorch",
                    "timm", "pretrainedmodels", "efficientnet-pytorch"}
    risky = sorted({p for _, p in _MISSING_OPTIONAL} & torch_family)
    if cmd:
        lines += ["", "Install them with:", "    " + cmd]
    if risky:
        lines += [
            "",
            "Also missing, from the torch family: " + " ".join(risky) + ".",
            "Install these from the SAME channel you used for torch: a bare "
            "'pip install' would pull a CPU-only torch wheel over your CUDA "
            "build and disable the GPU.",
        ]
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(__app_name__ + " - missing packages")
    box.setText("Some processing chains cannot run in this environment.")
    box.setInformativeText("\n".join(lines))
    box.setStandardButtons(QMessageBox.Ok)
    _centre_on(box, parent)
    box.exec()


def _centre_on(dialog, parent) -> None:
    """Centre a dialog on its parent window, or on the active screen.

    A QMessageBox sizes itself only once its layout is activated, so the
    geometry is computed first; without that the move would use a stale,
    too-small size hint and the box would sit off-centre.
    """
    try:
        dialog.adjustSize()
        dialog.layout().activate()
        size = dialog.sizeHint()
        if parent is not None and parent.isVisible():
            area = parent.frameGeometry()
        else:
            from PySide6.QtGui import QGuiApplication  # noqa: E402

            screen = QGuiApplication.screenAt(
                dialog.pos()) or QGuiApplication.primaryScreen()
            area = screen.availableGeometry()
        dialog.move(
            area.center().x() - size.width() // 2,
            area.center().y() - size.height() // 2,
        )
    except Exception:  # noqa: BLE001 - placement must never block the warning
        pass


def main() -> int:
    """Build the QApplication, show :class:`MainWindow` and run the event loop."""
    configure_environment()
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    # Apply the bundled planaria icon before any widget is shown so
    # the splash on launch, the task-bar entry and Alt+Tab all pick
    # up the branded icon instead of the default Qt penguin/logo.
    app_icon = _resolve_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # Diagnostic safe mode: ``FLATMORPHAI_SAFE_MODE=1`` shows a tiny
    # Qt-only window instead of the full application. It builds NO pages
    # and imports nothing heavy, so it isolates whether Qt itself can
    # paint in the current environment (useful when the app crashes with
    # a bare native "Segmentation fault" at the first event-loop tick):
    #   * if safe mode also crashes  -> the fault is in Qt / the display
    #     stack (WSL, OpenGL, platform plugin), not in the app code ;
    #   * if safe mode shows a window -> Qt is fine and the fault is in a
    #     page or a heavy library (torch, cv2) loaded by the full UI.
    if os.environ.get("FLATMORPHAI_SAFE_MODE", "0") == "1":
        from PySide6.QtWidgets import QLabel  # noqa: E402

        label = QLabel(
            "FlatMorphAI safe mode\n\nQt renders correctly in this "
            "environment.\nUnset FLATMORPHAI_SAFE_MODE to start the app."
        )
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(420, 180)
        if not app_icon.isNull():
            label.setWindowIcon(app_icon)
        label.show()
        print("[FlatMorphAI] SAFE MODE: Qt-only window, no pages loaded.",
              flush=True)
        return app.exec()

    settings = AppSettings()
    # Log the hardware calibration outcome so the operator can see
    # what CPU-count-derived defaults were applied on first launch.
    hw = settings.get("hardware_defaults_applied_v1")
    if isinstance(hw, dict):
        print(
            f"[FlatMorphAI] Hardware calibration: cpu_count="
            f"{hw.get('cpu_count')}, "
            f"recommended workers={hw.get('recommended')}, "
            f"applied={hw.get('applied')}",
            flush=True,
        )
    theme_manager = ThemeManager(app, settings)

    window = MainWindow(theme_manager=theme_manager)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    # Installing after the main window exists lets the shortcut parent
    # (the window) own the QShortcut objects, so they live exactly as
    # long as the UI does.
    theme_manager.install(shortcut_parent=window)
    # Pointer size, applied once the window exists so it propagates to every
    # child widget. Drawn by the application (flatmorphai/cursors.py), because
    # the distribution has no cursor theme for XCURSOR_SIZE to pick from.
    try:
        from . import cursors as _cursors
        _cursors.apply(int(settings.get("cursor_size", _cursors.DEFAULT_SIZE)),
                       str(settings.get("cursor_color", _cursors.DEFAULT_COLOR)),
                       on=str(settings.get("cursor_mode", "system")) == "drawn")
    except Exception:  # noqa: BLE001 - a pointer must never stop the launch
        pass
    window.show()
    # Bring the window to the front and give it the keyboard. Launched from a
    # terminal, a Qt window often opens behind it or unfocused -- under WSLg
    # the window manager is Windows', which does not necessarily raise a new
    # X window. The three calls do three different things and all are needed:
    # raise_() puts it above the others, activateWindow() gives it the focus,
    # and the deferred repeat catches the case where the manager finishes
    # mapping the window after this code has already run (the activation is
    # then ignored, which is the usual reason a window opens behind).
    def _focus() -> None:
        try:
            window.raise_()
            window.activateWindow()
            window.setFocus(Qt.ActiveWindowFocusReason)
        except Exception:  # noqa: BLE001 - focus must never block the launch
            pass
    _focus()
    QTimer.singleShot(0, _focus)
    QTimer.singleShot(250, _focus)
    # After show() so the dialog is parented to a visible window and never
    # appears behind it.
    _show_missing_packages_dialog(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
