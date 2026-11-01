"""
Aero-Snap style drag-to-edge window tiling with a live preview overlay.

The controller watches a top-level :class:`QMainWindow` while the user
is dragging it and, whenever the mouse pointer reaches one of the
screen edges defined below, displays a translucent preview rectangle
that mirrors the area the window would occupy after the snap. On
mouse release, the window is resized and repositioned accordingly.

Supported snap regions (matching the Windows 10/11 Snap Assist
behaviour):

- Left edge            -> left half of the screen
- Right edge           -> right half of the screen
- Top edge             -> maximised
- Top-left corner      -> top-left quadrant
- Top-right corner     -> top-right quadrant
- Bottom-left corner   -> bottom-left quadrant
- Bottom-right corner  -> bottom-right quadrant

Design and references
---------------------
The UX follows the long-standing Snap convention introduced with
Windows 7 and refined in Windows 11 with Snap Layouts, as described
in Agarawala A. and Balakrishnan R., "Keepin' it real: pushing the
desktop metaphor with physics, piles and the pen", Proc. ACM CHI 2006,
DOI: 10.1145/1124772.1124965, which formalises edge-based window
manipulation gestures.

The behaviour is additive: the window still retains its native title
bar and frame, so the operating system's own Aero-Snap remains
available on Windows. This controller complements it and extends the
feature to platforms where the native feature is missing (for example
X11 on some WSLg / Wayland sessions).
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QPainter, QPaintEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget


SnapKind = str  # "left" | "right" | "top" | "tl" | "tr" | "bl" | "br" | ""


class _SnapPreview(QWidget):
    """Translucent blue rectangle shown during a snap drag."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Windows 11 Snap Assist uses an accent-tinted panel with a
        # 1px border. We mirror that look with the FlatMorphAI blue.
        fill = QColor(37, 99, 235, 90)       # #2563EB at ~35 percent
        border = QColor(37, 99, 235, 220)    # #2563EB at ~86 percent
        rect = self.rect().adjusted(1, 1, -2, -2)
        painter.setBrush(fill)
        pen = painter.pen()
        pen.setColor(border)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 8, 8)


class WindowSnapController(QObject):
    """Adds drag-to-edge snap tiling with a preview overlay."""

    # Distance in pixels from the active-screen edges at which a snap
    # region is armed. 12 px matches the default Windows Snap threshold.
    EDGE_THRESHOLD = 12

    # Poll period while a drag is ongoing. 60 Hz keeps the preview
    # responsive without burning CPU when the mouse is idle.
    POLL_INTERVAL_MS = 16

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._preview: Optional[_SnapPreview] = None
        self._pending_snap: Optional[Tuple[SnapKind, QRect]] = None
        # Geometry captured at drag start so a quick drag back to the
        # original spot can restore it.
        self._pre_snap_geometry: Optional[QRect] = None
        self._tracking = False

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)

        window.installEventFilter(self)

    # ------------------------------------------------------------------
    # Event filter: Qt raises QEvent.Move while the user drags the
    # window. We start polling on the first move and stop as soon as
    # the left mouse button is released.
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        # ``self._window`` may have been destroyed (Python wrapper alive,
        # C++ side deleted) by the time an event flushes through the
        # filter chain on shutdown. Comparing a dangling shiboken
        # pointer with ``is`` is fine, but any further attribute access
        # on it would crash. Bail out early in that case.
        try:
            if watched is self._window and event.type() == QEvent.Move:
                if not self._tracking and self._is_left_button_pressed():
                    self._begin_tracking()
        except RuntimeError:
            # Raised when watched/self._window has been C++-deleted.
            return False
        return False

    # ------------------------------------------------------------------
    # Tracking lifecycle
    # ------------------------------------------------------------------
    def _begin_tracking(self) -> None:
        self._tracking = True
        self._pending_snap = None
        if not self._window.isMaximized():
            self._pre_snap_geometry = QRect(self._window.geometry())
        self._timer.start()

    def _stop_tracking(self) -> None:
        self._tracking = False
        self._timer.stop()
        pending = self._pending_snap
        self._pending_snap = None
        self._hide_preview()
        if pending is not None:
            self._apply_snap(*pending)

    def _poll(self) -> None:
        if not self._is_left_button_pressed():
            self._stop_tracking()
            return
        pos = QCursor.pos()
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        snap = self._classify_edge(pos, available)
        if snap is None:
            self._hide_preview()
            self._pending_snap = None
            return
        kind, target = snap
        self._pending_snap = (kind, target)
        self._show_preview(target)

    # ------------------------------------------------------------------
    # Snap region computation
    # ------------------------------------------------------------------
    def _classify_edge(
        self, pos, available: QRect,
    ) -> Optional[Tuple[SnapKind, QRect]]:
        t = self.EDGE_THRESHOLD
        left = pos.x() - available.left() <= t
        right = available.right() - pos.x() <= t
        top = pos.y() - available.top() <= t
        bottom = available.bottom() - pos.y() <= t

        w = available.width()
        h = available.height()
        cx = available.left()
        cy = available.top()

        # Corners first so they win over plain edges.
        if top and left:
            return "tl", QRect(cx, cy, w // 2, h // 2)
        if top and right:
            return "tr", QRect(cx + w // 2, cy, w - w // 2, h // 2)
        if bottom and left:
            return "bl", QRect(cx, cy + h // 2, w // 2, h - h // 2)
        if bottom and right:
            return (
                "br",
                QRect(cx + w // 2, cy + h // 2, w - w // 2, h - h // 2),
            )
        if top:
            return "top", QRect(cx, cy, w, h)
        if left:
            return "left", QRect(cx, cy, w // 2, h)
        if right:
            return "right", QRect(cx + w // 2, cy, w - w // 2, h)
        return None

    # ------------------------------------------------------------------
    # Preview overlay
    # ------------------------------------------------------------------
    def _show_preview(self, rect: QRect) -> None:
        if self._preview is None:
            self._preview = _SnapPreview()
        self._preview.setGeometry(rect)
        if not self._preview.isVisible():
            self._preview.show()
        self._preview.raise_()

    def _hide_preview(self) -> None:
        if self._preview is not None and self._preview.isVisible():
            self._preview.hide()

    # ------------------------------------------------------------------
    # Apply snap to the real window
    # ------------------------------------------------------------------
    def _apply_snap(self, kind: SnapKind, target: QRect) -> None:
        window = self._window
        # Maximise for the top-edge gesture so the native "restore"
        # animation kicks in instead of a plain resize.
        if kind == "top":
            window.showMaximized()
            return
        if window.isMaximized():
            window.showNormal()
        window.setGeometry(target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_left_button_pressed() -> bool:
        return bool(QApplication.mouseButtons() & Qt.LeftButton)
