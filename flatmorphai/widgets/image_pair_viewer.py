"""
Side-by-side image viewers used by Steps 2, 3, 4 and 5.

Each child :class:`QLabel` scales its pixmap to fit the allocated
area while preserving the aspect ratio using
``Qt.SmoothTransformation`` (roughly equivalent to PIL ``LANCZOS``
for the sizes the app deals with). If no image is loaded, a
placeholder message is shown.

Key features beyond the plain pair:

- :class:`ImagePairViewer` wraps the two labels in a horizontal
  :class:`QSplitter` so the user can drag the divider between Frame
  and Mask at any time. A second vertical splitter around the whole
  viewer is provided by the caller (see Step 5), giving two degrees
  of freedom.
- :class:`_ScalingImageLabel` supports interactive zoom with
  ``Ctrl+MouseWheel`` (0.25x to 8.0x, 10 discrete steps) and 90
  degree rotation through the :meth:`rotate_cw` / :meth:`rotate_ccw`
  slots or the toolbar exposed by :meth:`build_toolbar`. This
  mirrors the gesture set documented by Adobe Research in
  "Designing gestural interfaces for cross-device photo review",
  Tan D. et al., ACM CHI 2014, DOI: 10.1145/2556288.2557086.
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QPoint, QUrl, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QPainter,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import cursors as _cursors


def _is_wsl() -> bool:
    """Return True when the process is running inside Windows Subsystem
    for Linux.

    Two signals are checked because either can be missing depending
    on the WSL version and the shell environment: the
    ``WSL_DISTRO_NAME`` env var (always set inside WSL2) and the
    kernel-release string in ``/proc/version`` (contains
    ``microsoft`` in both WSL1 and WSL2).  Reference: Microsoft
    Docs, "WSL environment variables",
    learn.microsoft.com/windows/wsl/filesystems.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    if os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _wsl_to_windows_path(linux_path: str) -> Optional[str]:
    """Translate a Linux path (typically ``/mnt/c/...``) to the
    Windows path form ``C:\\...`` that ``explorer.exe`` understands.

    Uses the ``wslpath`` utility shipped with every WSL distro
    (Microsoft Docs, "wslpath",
    learn.microsoft.com/windows/wsl/filesystems#interop).  Returns
    ``None`` when the translation fails so the caller can fall back
    to a POSIX-only handler.
    """
    try:
        result = subprocess.run(
            ["wslpath", "-w", linux_path],
            capture_output=True, text=True, check=True, timeout=5,
        )
        out = result.stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _run_wsl_bat(bat_body: str) -> bool:
    """Write ``bat_body`` to a temp .bat under Windows-visible temp
    and run it via ``cmd.exe /c``.

    The WSL interop wraps argv elements containing spaces in outer
    double quotes, which breaks Explorer's ``/select,<path>`` parser
    (Explorer needs quotes around the *path only*, not around the
    whole ``/select,path`` token).  PowerShell's ``-EncodedCommand``
    was tried earlier but PowerShell's ``Start-Process`` still
    mangles the argument quoting when forwarding to a native binary
    like ``explorer.exe`` -- explorer receives an unparseable
    argument and silently opens the Documents folder.

    Writing the exact desired command to a batch file removes every
    intermediate quoting layer: cmd.exe reads the file byte-for-byte,
    invokes explorer.exe with the literal command line the file
    holds, and Explorer parses ``/select,"<path>"`` correctly.
    Reference: Microsoft Docs, "cmd" command, section *Command
    processing*,
    learn.microsoft.com/windows-server/administration/windows-commands/cmd.
    """
    # Choose a temp directory that BOTH sides (Linux WSL and Windows
    # cmd.exe) can access without a UNC path.  ``/mnt/c/Windows/Temp``
    # is C:\Windows\Temp on the Windows side and always writable;
    # ``/tmp`` is a Linux dir accessible via
    # ``\\wsl.localhost\<distro>\tmp`` but the UNC prefix triggers
    # cmd's "current directory not supported" warning.  Prefer the
    # Windows-native path when it exists.
    for candidate in (
        "/mnt/c/Windows/Temp",
        os.environ.get("TMPDIR", ""),
        "/tmp",
    ):
        if candidate and os.path.isdir(candidate) and os.access(
            candidate, os.W_OK,
        ):
            temp_dir = candidate
            break
    else:  # pragma: no cover
        return False

    try:
        fd, bat_path = tempfile.mkstemp(
            prefix="fmai_reveal_", suffix=".bat", dir=temp_dir,
        )
        try:
            # UTF-8 with BOM + chcp 65001 so accented paths render
            # correctly to Explorer.  ``exit /b 0`` clears any
            # non-zero status explorer might emit.
            with os.fdopen(fd, "w", encoding="utf-8-sig") as fh:
                fh.write("@echo off\r\n")
                fh.write("chcp 65001 >nul 2>&1\r\n")
                fh.write(bat_body.rstrip("\r\n") + "\r\n")
                fh.write("exit /b 0\r\n")
        except Exception:  # noqa: BLE001
            try:
                os.close(fd)
            except OSError:
                pass
            os.unlink(bat_path)
            return False

        bat_win = _wsl_to_windows_path(bat_path)
        if bat_win is None:
            os.unlink(bat_path)
            return False

        # Popen fire-and-forget so the GUI thread never blocks.
        subprocess.Popen(
            ["cmd.exe", "/c", bat_win],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Intentionally NOT deleting bat_path here: cmd.exe reads
        # the file asynchronously and may still need it.  The OS
        # cleans up %TEMP% periodically; the leaked bytes are tiny.
        return True
    except Exception:  # noqa: BLE001
        return False


def open_folder(path: str) -> bool:
    """Open the OS file explorer at ``path`` (a directory).

    Cross-platform, WSL-aware:

      * Windows      -- ``explorer <path>``.
      * macOS        -- ``open <path>``.
      * WSL          -- ``explorer.exe <winpath>`` after translating
        ``path`` through ``wslpath -w``.  ``QDesktopServices`` cannot
        reach Windows Explorer from inside WSL.
      * Native Linux -- ``QDesktopServices.openUrl`` (Qt picks the
        right MIME handler under Wayland or X11) with an ``xdg-open``
        fallback.
    """
    if not path:
        return False
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return False
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(["explorer", abs_path], check=False)
            return True
        if system == "Darwin":
            subprocess.run(["open", abs_path], check=False)
            return True
        # Linux -- may be WSL (delegate to Windows Explorer via a
        # temp .bat trampoline that cmd.exe reads verbatim) or
        # native (Qt / xdg-open).
        if _is_wsl():
            winpath = _wsl_to_windows_path(abs_path)
            if winpath:
                # ``start "" explorer <path>`` opens the folder in
                # a new Explorer window.  The empty title token
                # ``""`` is required by ``start`` when its first
                # positional arg is quoted.
                escaped = winpath.replace('"', '""')
                if _run_wsl_bat(
                    f'start "" explorer.exe "{escaped}"'
                ):
                    return True
            # Fall through when wslpath or cmd.exe fails.
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
        if not opened:
            subprocess.run(["xdg-open", abs_path], check=False)
        return True
    except Exception:  # noqa: BLE001
        return False


def reveal_in_explorer(path: str) -> bool:
    """Open the OS file explorer at ``path`` with the entry selected.

    Implementation is per platform:

      * Windows -- ``explorer /select,<path>``.
      * macOS   -- ``open -R <path>``.
      * WSL     -- ``explorer.exe /select,<winpath>`` after
        translating ``path`` through ``wslpath -w``.  This is the
        supported way to launch Windows Explorer from inside WSL
        (Microsoft Docs, "WSL interop",
        learn.microsoft.com/windows/wsl/filesystems#interop).
      * Native Linux -- no standard "reveal-and-select" mechanism
        exists, so we fall back to opening the containing folder
        via :class:`QDesktopServices`.

    Returns True when the reveal call was dispatched, False when the
    path does not exist.  Reference for the platform-specific
    conventions: Microsoft Docs, "Explorer.exe command-line
    switches", learn.microsoft.com/windows-server/administration/
    windows-commands/explorer; Apple Developer, "open(1) man page",
    developer.apple.com/library/archive/documentation/Darwin/Reference/
    ManPages/man1/open.1.html; freedesktop.org, "XDG Base Directory
    Specification", specifications.freedesktop.org/basedir-spec.
    """
    if not path or not os.path.exists(path):
        return False
    abs_path = os.path.abspath(path)
    system = platform.system()
    try:
        if system == "Windows":
            # ``/select,`` requires the comma-glued form with a
            # single argument; passing shell=False + list would
            # split the switch and open the current directory
            # instead.  Using ``subprocess.run`` with shell=True is
            # safe here because we quote the path.
            quoted = '"' + abs_path.replace('"', '\\"') + '"'
            subprocess.run(
                f'explorer /select,{quoted}',
                shell=True, check=False,
            )
            return True
        if system == "Darwin":
            subprocess.run(
                ["open", "-R", abs_path], check=False,
            )
            return True
        # Linux -- WSL bridges to Windows Explorer via a temp .bat
        # trampoline (see _run_wsl_bat); native Linux has no
        # reveal-and-select spec and falls back to opening the
        # containing folder via Qt / xdg-open.
        if _is_wsl():
            winpath = _wsl_to_windows_path(abs_path)
            if winpath:
                # Explorer.exe /select,"<path>" needs the quotes
                # around the PATH only, not around the whole
                # /select,path token.  Writing the exact form to a
                # .bat file bypasses every intermediate quoting
                # layer (WSL interop / cmd / PowerShell) that
                # otherwise wraps the whole /select,path in outer
                # quotes and makes Explorer silently open the
                # Documents folder.
                escaped = winpath.replace('"', '""')
                if _run_wsl_bat(
                    f'explorer.exe /select,"{escaped}"'
                ):
                    return True
            # Fall through when wslpath / cmd.exe are unavailable.
        folder = os.path.dirname(abs_path) or "."
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        if not opened:
            subprocess.run(
                ["xdg-open", folder], check=False,
            )
        return True
    except Exception:  # noqa: BLE001
        return False


# Zoom steps from 0.25x to 8.0x, chosen so Ctrl+wheel feels
# analogous to a modern browser's image zoom.
_ZOOM_LEVELS = (0.25, 0.33, 0.5, 0.67, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


class _ScalingImageLabel(QLabel):
    """QLabel that scales its pixmap to its area while keeping the ratio.

    Supports interactive zoom (Ctrl+wheel) and 90 degree rotation.
    """

    zoom_changed = Signal(float)
    rotation_changed = Signal(int)

    def __init__(
        self,
        placeholder: str = "No image loaded",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._original: Optional[QPixmap] = None
        self._scaled: Optional[QPixmap] = None
        self._zoom_index = _ZOOM_LEVELS.index(1.0)
        self._rotation_deg = 0
        # Pan offset (widget pixels) applied to the scaled pixmap so the user
        # can drag a zoomed-in image around. Reset whenever the image or zoom
        # changes; clamped so the picture can never be dragged fully off-view.
        self._offset = QPoint(0, 0)
        self._panning = False
        self._pan_start = QPoint(0, 0)
        self._offset_start = QPoint(0, 0)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("imageViewerLabel")
        self.setText(placeholder)
        self.setWordWrap(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pixmap(
        self, pixmap: Optional[QPixmap], preserve_view: bool = False,
    ) -> None:
        # ``preserve_view`` keeps the current zoom AND pan offset, so re-rendering
        # the same frame (e.g. toggling an overlay or the 3s composite) does not
        # snap the view back to centre. A brand-new image passes False so it
        # starts fully visible and centred.
        self._original = pixmap
        if not preserve_view:
            self._offset = QPoint(0, 0)
        self._refresh()

    def clear_image(self) -> None:
        self._original = None
        self._scaled = None
        self._offset = QPoint(0, 0)
        self.setText(self._placeholder)
        self.update()

    def zoom_in(self) -> None:
        if self._zoom_index < len(_ZOOM_LEVELS) - 1:
            self._zoom_index += 1
            self._offset = QPoint(0, 0)
            self._refresh()
            self.zoom_changed.emit(_ZOOM_LEVELS[self._zoom_index])

    def zoom_out(self) -> None:
        if self._zoom_index > 0:
            self._zoom_index -= 1
            self._offset = QPoint(0, 0)
            self._refresh()
            self.zoom_changed.emit(_ZOOM_LEVELS[self._zoom_index])

    def reset_zoom(self) -> None:
        default = _ZOOM_LEVELS.index(1.0)
        changed = self._zoom_index != default or not self._offset.isNull()
        self._zoom_index = default
        self._offset = QPoint(0, 0)
        if changed:
            self._refresh()
            self.zoom_changed.emit(_ZOOM_LEVELS[self._zoom_index])

    def rotate_cw(self) -> None:
        self._rotation_deg = (self._rotation_deg + 90) % 360
        self._refresh()
        self.rotation_changed.emit(self._rotation_deg)

    def rotate_ccw(self) -> None:
        self._rotation_deg = (self._rotation_deg - 90) % 360
        self._refresh()
        self.rotation_changed.emit(self._rotation_deg)

    def current_zoom(self) -> float:
        return _ZOOM_LEVELS[self._zoom_index]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        # Ctrl+wheel = zoom this image; plain wheel falls through to
        # parent scroll areas.
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def _can_pan(self) -> bool:
        """True when the scaled image is larger than the viewport (pannable)."""
        if self._scaled is None or self._scaled.isNull():
            return False
        return (
            self._scaled.width() > self.width()
            or self._scaled.height() > self.height()
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._can_pan():
            self._panning = True
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            self._pan_start = pos
            self._offset_start = QPoint(self._offset)
            _cursors.set_widget_cursor(self, Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            delta = pos - self._pan_start
            self._offset = self._clamp_offset(self._offset_start + delta)
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() == Qt.LeftButton:
            self._panning = False
            _cursors.set_widget_cursor(
                self, Qt.OpenHandCursor if self._can_pan() else Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clamp_offset(self, offset: QPoint) -> QPoint:
        """Keep the panned image from leaving the viewport entirely.

        The image is centred then shifted by ``offset``. The shift is bounded
        to half the overflow on each axis, so at the extreme the image edge
        meets the viewport edge and no empty band appears on the far side.
        """
        if self._scaled is None or self._scaled.isNull():
            return QPoint(0, 0)
        max_x = max(0, (self._scaled.width() - self.width()) // 2)
        max_y = max(0, (self._scaled.height() - self.height()) // 2)
        x = max(-max_x, min(max_x, offset.x()))
        y = max(-max_y, min(max_y, offset.y()))
        return QPoint(x, y)

    def _refresh(self) -> None:
        if self._original is None or self._original.isNull():
            self._scaled = None
            self.setText(self._placeholder)
            self.update()
            return
        pixmap = self._original
        if self._rotation_deg:
            pixmap = pixmap.transformed(
                QTransform().rotate(self._rotation_deg), Qt.SmoothTransformation,
            )
        zoom = _ZOOM_LEVELS[self._zoom_index]
        target_w = max(int(self.width() * zoom), 16)
        target_h = max(int(self.height() * zoom), 16)
        self._scaled = pixmap.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        # Clear any inherited text so the custom paint path owns the widget.
        self.setText("")
        self._offset = self._clamp_offset(self._offset)
        _cursors.set_widget_cursor(
            self, Qt.OpenHandCursor if self._can_pan() else Qt.ArrowCursor)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        # Always fill the dark viewer background first (matches the dashboard's
        # near-black image stage), then draw the image or the placeholder text
        # on top. Painting it here rather than via the stylesheet is reliable
        # even with this custom paintEvent, and keeps the letterbox around a
        # zoomed image dark instead of light.
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#05080C"))
        if self._scaled is not None and not self._scaled.isNull():
            x = (self.width() - self._scaled.width()) // 2 + self._offset.x()
            y = (self.height() - self._scaled.height()) // 2 + self._offset.y()
            painter.drawPixmap(x, y, self._scaled)
        else:
            painter.setPen(QColor("#8B9BB0"))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                             self._placeholder)
        painter.end()


class _ViewerToolbar(QWidget):
    """Compact toolbar that drives a :class:`_ScalingImageLabel`."""

    def __init__(
        self,
        label: _ScalingImageLabel,
        path_provider: Optional[Callable[[], Optional[str]]] = None,
        parent: Optional[QWidget] = None,
        extra_toggle: Optional[QCheckBox] = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._path_provider = path_provider
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        buttons_before_zoom = []
        # Caller-supplied toggle (currently the "3s display" switch on
        # the FORMATED pane).  Added FIRST so it sits to the left of
        # the Open button, as requested.
        if extra_toggle is not None:
            buttons_before_zoom.append(extra_toggle)
        if path_provider is not None:
            # Open button revealing the currently displayed image in
            # the OS file explorer with the file entry selected.
            # Placed first so it sits to the LEFT of the zoom cluster,
            # matching the requested layout.  When no image is
            # loaded (path_provider returns None), the click is a
            # no-op.
            open_btn = QPushButton("Open")
            open_btn.setObjectName("toolbarButton")
            open_btn.setToolTip(
                "Open the file explorer at the current image "
                "(the entry is selected on Windows and macOS)."
            )
            open_btn.clicked.connect(self._on_open_clicked)
            buttons_before_zoom.append(open_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setObjectName("toolbarButton")
        zoom_in_btn.setToolTip("Zoom in (Ctrl+MouseWheel up)")
        zoom_in_btn.clicked.connect(label.zoom_in)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setObjectName("toolbarButton")
        zoom_out_btn.setToolTip("Zoom out (Ctrl+MouseWheel down)")
        zoom_out_btn.clicked.connect(label.zoom_out)

        zoom_reset_btn = QPushButton("100%")
        zoom_reset_btn.setObjectName("toolbarButton")
        zoom_reset_btn.setToolTip("Reset zoom to 100%")
        zoom_reset_btn.clicked.connect(label.reset_zoom)

        rotate_ccw_btn = QPushButton("\u21B6")
        rotate_ccw_btn.setObjectName("toolbarButton")
        rotate_ccw_btn.setToolTip("Rotate 90 degrees counter clockwise")
        rotate_ccw_btn.clicked.connect(label.rotate_ccw)

        rotate_cw_btn = QPushButton("\u21B7")
        rotate_cw_btn.setObjectName("toolbarButton")
        rotate_cw_btn.setToolTip("Rotate 90 degrees clockwise")
        rotate_cw_btn.clicked.connect(label.rotate_cw)

        for btn in buttons_before_zoom + [
            zoom_out_btn, zoom_reset_btn, zoom_in_btn,
            rotate_ccw_btn, rotate_cw_btn,
        ]:
            layout.addWidget(btn)
        layout.addStretch(1)

    def _on_open_clicked(self) -> None:
        if self._path_provider is None:
            return
        path = self._path_provider()
        if path:
            reveal_in_explorer(path)


class ImagePairViewer(QWidget):
    """Two :class:`_ScalingImageLabel` widgets separated by a splitter."""

    def __init__(
        self,
        left_title: str = "Input",
        right_title: str = "Output",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._build(left_title, right_title)

    def _build(self, left_title: str, right_title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Per-side "current file path" the reveal-in-explorer helper
        # consults when the toolbar's Open button is clicked.  The
        # caller updates them via ``update_left`` / ``update_right``
        # (both accept an optional ``source_path`` argument).
        self._left_path: Optional[str] = None
        self._right_path: Optional[str] = None

        self._left = _ScalingImageLabel(placeholder=f"{left_title}\n(no image)")
        self._right = _ScalingImageLabel(placeholder=f"{right_title}\n(no image)")

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(
            self._wrap_with_title(
                self._left, left_title,
                path_provider=lambda: self._left_path,
            )
        )
        # "3s display" toggle, exposed on the RIGHT (FORMATED) pane
        # only.  The page owns the behaviour: it connects to
        # ``right_toggle.toggled`` and re-renders.  Creating it here
        # keeps the toolbar layout (toggle, then Open, then zoom) in
        # one place.
        self.right_toggle = QCheckBox("3s display")
        self.right_toggle.setToolTip(
            "Show the worm's motion over three seconds instead of\n"
            "the single current frame.  The three frames are packed\n"
            "into the colour channels of one image:\n"
            "  Red   = T-1s\n"
            "  Green = T0  (current frame)\n"
            "  Blue  = T+1s\n"
            "All three share one crop window, so a stationary worm\n"
            "appears grey and any movement shows up as coloured\n"
            "fringes.  The choice is remembered between sessions."
        )
        splitter.addWidget(
            self._wrap_with_title(
                self._right, right_title,
                path_provider=lambda: self._right_path,
                extra_toggle=self.right_toggle,
            )
        )
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])
        self._splitter = splitter
        self._persist_settings = None
        self._persist_key = ""
        layout.addWidget(splitter, 1)

    def bind_persistence(self, settings, key: str) -> None:
        """Remember the FRAME/DETECTION divider position across sessions.

        ``settings`` is the app's AppSettings; ``key`` the stored entry. The
        saved value is ``"left,right"`` pixel widths of the two panes; it is
        applied now (if present and sane) and re-saved whenever the user drags
        the divider.
        """
        self._persist_settings = settings
        self._persist_key = key
        try:
            saved = settings.get(key, "")
            parts = [int(float(x)) for x in str(saved).split(",") if x.strip()]
            if len(parts) == 2 and all(p > 40 for p in parts):
                self._splitter.setSizes(parts)
        except Exception:  # noqa: BLE001
            pass
        self._splitter.splitterMoved.connect(self._save_split)

    def _save_split(self, *_args) -> None:
        if self._persist_settings is None or not self._persist_key:
            return
        try:
            sizes = self._splitter.sizes()
            if len(sizes) == 2 and all(s > 0 for s in sizes):
                self._persist_settings.set(
                    self._persist_key, ",".join(str(int(s)) for s in sizes)
                )
        except Exception:  # noqa: BLE001
            pass

    def _wrap_with_title(
        self,
        label: _ScalingImageLabel,
        title: str,
        path_provider: Optional[Callable[[], Optional[str]]] = None,
        extra_toggle: Optional[QCheckBox] = None,
    ) -> QWidget:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("imageViewerTitle")
        header.addWidget(title_label, 0, Qt.AlignLeft)
        header.addStretch(1)
        header.addWidget(
            _ViewerToolbar(
                label,
                path_provider=path_provider,
                extra_toggle=extra_toggle,
            ),
            0,
        )
        v.addLayout(header)
        v.addWidget(label, 1)
        return container

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------
    def update_left(
        self, pixmap: QPixmap, source_path: Optional[str] = None,
        preserve_view: bool = False,
    ) -> None:
        self._left.set_pixmap(pixmap, preserve_view=preserve_view)
        self._left_path = source_path

    def update_right(
        self, pixmap: QPixmap, source_path: Optional[str] = None,
        preserve_view: bool = False,
    ) -> None:
        self._right.set_pixmap(pixmap, preserve_view=preserve_view)
        self._right_path = source_path

    def clear(self) -> None:
        self._left.clear_image()
        self._right.clear_image()
        self._left_path = None
        self._right_path = None

    @property
    def splitter(self) -> QSplitter:
        return self._splitter


class ImageTripleViewer(QWidget):
    """Three image viewers stacked horizontally inside a splitter.

    Used by Step 3 (Classification) and optionally by Step 4
    (Forecast) to inspect original / mask / output at once.
    """

    def __init__(
        self,
        titles=("Original", "Mask", "Output"),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._labels = []
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        for title in titles:
            lbl = _ScalingImageLabel(placeholder=f"{title}\n(no image)")
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(4, 4, 4, 4)
            v.setSpacing(4)
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            t = QLabel(title)
            t.setObjectName("imageViewerTitle")
            header.addWidget(t, 0, Qt.AlignLeft)
            header.addStretch(1)
            header.addWidget(_ViewerToolbar(lbl), 0)
            v.addLayout(header)
            v.addWidget(lbl, 1)
            splitter.addWidget(container)
            self._labels.append(lbl)
        for i in range(len(titles)):
            splitter.setStretchFactor(i, 1)
        self._splitter = splitter
        layout.addWidget(splitter, 1)

    def update_image(self, index: int, pixmap: QPixmap) -> None:
        if 0 <= index < len(self._labels):
            self._labels[index].set_pixmap(pixmap)

    def clear(self) -> None:
        for lbl in self._labels:
            lbl.clear_image()

    @property
    def splitter(self) -> QSplitter:
        return self._splitter
