"""
Main window for the FlatMorphAI dashboard.

Layout
------
- Left sidebar (240 px): a ``QScrollArea`` of :class:`SidebarCard`
  instances, one per step/pipeline.
- Central area: a :class:`QStackedWidget` with the six step pages and
  the two pipeline pages.
- Status bar: global messages and a :class:`GPUBadge` live at the
  bottom of the window.

The whole layout is wrapped in a :class:`QSplitter` so the user can
resize the sidebar.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from . import __app_name__, __version__
from .config import AppSettings
from .model_cache import ModelCache
from .pages import (
    AboutPage,
    PipelineAPage,
    PipelineBPage,
    PipelineCPage,
    PipelineDPage,
    PipelineEPage,
    RegenerationPage,
    EyespotsCounterPage,
    RegenerationEditorPage,
    SettingsPage,
    StereotypeClassificationPage,
    Step1Page,
    Step2Page,
    Step3Page,
    Step4Page,
    Step5Page,
    Step6Page,
)
from .theme_manager import ThemeManager
from .widgets import GPUBadge, SidebarCard, WindowSnapController
from .worker_registry import WorkerRegistry


# Sidebar entries arranged so end-to-end workflows come first, then
# individual steps, finally auxiliary pages. Section headers and the
# divider between Pipelines and Steps are inserted programmatically in
# :meth:`MainWindow._build_sidebar` so the user can easily distinguish
# "one-click pipelines" from the "step by step" mode.
# Each tuple: (object_key, number, name, icon_color).
SIDEBAR_ITEMS = [
    ("stereotype", "Stereotype classification", "Videos/Images -> Stereotype", "#2563EB"),
    ("pipeline_c", "Image segmentation", "Images -> Masks", "#2563EB"),
    ("regeneration", "Forecast regeneration states", "Images -> States", "#0D9488"),
    ("eyespots", "Eyespots counter", "Images -> N Eyespots", "#0D9488"),
    # Pipelines D (mask metrics) and E (scrunching detector) are
    # intentionally hidden from the sidebar for now while their
    # algorithms are being validated. The page classes and worker
    # implementations stay registered in :meth:`_build_stack` so
    # re-adding the two tuples here is the only step required to
    # surface them again.
    # ("pipeline_d", "Mask metrics", "Mask metrics", "#2563EB"),
    # ("pipeline_e", "Scrunching detector", "Scrunching detector", "#2563EB"),
    # Steps 1-4 and 6 are intentionally hidden from the sidebar --
    # the operator's canonical entry point is now Stereotype classification (or
    # Stereotype classification from images / C for narrower reruns).  The individual Step
    # pages remain registered in :meth:`_build_stack` for internal
    # debugging and unit-test coverage; re-adding the tuples here
    # is the only step required to surface them again.
    # ("step1", "Step 1", "Video to frames", "#22D3EE"),
    # ("step2", "Step 2", "Frames to masks", "#6366F1"),
    # ("step3", "Step 3", "Classification images", "#8B5CF6"),
    # ("step4", "Step 4", "Forecast", "#F59E0B"),
    # ("step6", "Step 6", "Period fusion", "#EC4899"),
    ("step5", "Stereotype editor", "Manual review", "#10B981"),
    ("regeneration_editor", "Regeneration states editor", "Manual review", "#10B981"),
    ("settings", "Settings", "Preferences and defaults", "#64748B"),
    ("about", "About", "Project, team and funding", "#0EA5E9"),
]

# Index markers for section headers and the visual divider.
# "Pipelines" header sits above the first Pipeline card; "Tools"
# groups the Stereotype editor with Settings and About.  The
# per-Step section is intentionally gone -- the individual Step
# pages are hidden from the sidebar (see SIDEBAR_ITEMS above).
_SIDEBAR_SECTION_HEADERS = {
    "stereotype": "Pipelines",
    "step5": "Tools",
}


class MainWindow(QMainWindow):
    """Top-level window wiring sidebar and pages together."""

    def __init__(
        self,
        theme_manager: Optional[ThemeManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 720)
        self._settings = AppSettings()
        self._theme_manager = theme_manager

        self._pages: Dict[str, QWidget] = {}
        self._sidebar_cards: List[SidebarCard] = []
        self._build_ui()
        self._connect_pages()
        # Aero-Snap / Snap Assist style drag-to-edge tiling. The
        # controller watches move events and displays the translucent
        # preview overlay; the native title bar and resize handles are
        # preserved so Windows can still perform its own snap when
        # available.
        self._snap_controller = WindowSnapController(self)
        # Re-fit the sidebar whenever the global UI scale changes
        # (Ctrl+wheel or Settings page). Keep the handle draggable so
        # the user can always override the automatic width.
        if self._theme_manager is not None:
            try:
                self._theme_manager.scale_changed.connect(
                    lambda _v: self._auto_fit_sidebar()
                )
            except Exception:  # noqa: BLE001
                pass
        # Restore the last window size/position (or maximize on first launch).
        self._restore_window_state()
        # Restore the folded/unfolded navigation bar.
        if bool(self._settings.get("sidebar_collapsed", False)):
            self._toggle_sidebar_collapsed()

    #: Settings keys of the window layout, written when the window closes.
    #: ``window_rect`` is the last NORMAL (un-maximized) window, "x,y,w,h":
    #: the position of the frame (title bar included, as move() expects) and
    #: the size of the client area (as resize() expects), so that the window
    #: does not drift by the height of its title bar between launches.
    #: ``window_maximized`` says whether it was closed maximized.
    _WINDOW_RECT_KEY = "window_rect"
    _WINDOW_MAX_KEY = "window_maximized"
    #: Room kept for the window frame (title bar, borders) before the window
    #: is shown, when its real size is not known yet.
    _FRAME_ALLOWANCE = (16, 48)

    def _restore_window_state(self) -> None:
        """Open the window as it was when it was closed.

        A window closed maximized reopens maximized; otherwise it reopens at
        its last size and place. The rectangle is applied BEFORE the window is
        shown. It is always fitted inside the available area of a screen,
        frame included: a window larger than the screen can stop receiving
        input under WSLg. First launch, or a rectangle no longer on any
        screen: centred at 85 % of the screen.
        """
        from PySide6.QtCore import QByteArray, QRect
        from PySide6.QtGui import QGuiApplication

        maximized = bool(self._settings.get(self._WINDOW_MAX_KEY, False))
        rect = QRect()
        saved = str(self._settings.get(self._WINDOW_RECT_KEY, "") or "")
        try:
            x, y, w, h = (int(float(v)) for v in saved.split(","))
            rect = QRect(x, y, w, h)
        except (TypeError, ValueError):
            rect = QRect()
        if not rect.isValid():
            # Configuration written before window_rect existed: Qt's own
            # saved geometry gives the normal rectangle and the maximized flag.
            geo = str(self._settings.get("window_geometry", "") or "")
            if geo:
                try:
                    ba = QByteArray.fromHex(geo.encode("ascii"))
                    if not ba.isEmpty() and self.restoreGeometry(ba):
                        maximized = bool(self.windowState() & Qt.WindowMaximized)
                        n = QRect(self.normalGeometry())
                        rect = QRect(n) if n.isValid() else QRect(self.geometry())
                except Exception:  # noqa: BLE001
                    rect = QRect()
        rect = self._fit_on_screen(rect)
        self.setWindowState(
            self.windowState() & ~(Qt.WindowMaximized | Qt.WindowFullScreen))
        self.resize(rect.size())
        self.move(rect.topLeft())
        self._last_normal_rect = QRect(rect)
        if maximized:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _fit_on_screen(self, rect):
        """``rect`` (frame position, client size) moved and shrunk so that the
        whole window, frame included, lies inside one screen's available area;
        a default rectangle when it is not on any screen."""
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QGuiApplication
        fw, fh = self._FRAME_ALLOWANCE
        screen = None
        if rect.isValid():
            best = 0
            for scr in QGuiApplication.screens():
                inter = scr.availableGeometry().intersected(rect)
                area = inter.width() * inter.height()
                if inter.width() >= 200 and inter.height() >= 120 and area > best:
                    best, screen = area, scr
        if screen is None:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                return rect
            avail = screen.availableGeometry()
            w, h = int(avail.width() * 0.85), int(avail.height() * 0.85)
            return QRect(avail.x() + (avail.width() - w) // 2,
                         avail.y() + (avail.height() - h) // 2, w, h)
        avail = screen.availableGeometry()
        w = max(200, min(rect.width(), avail.width() - fw))
        h = max(120, min(rect.height(), avail.height() - fh))
        x = min(max(rect.x(), avail.x()), avail.x() + avail.width() - (w + fw))
        y = min(max(rect.y(), avail.y()), avail.y() + avail.height() - (h + fh))
        return QRect(max(x, avail.x()), max(y, avail.y()), w, h)

    def _keep_frame_on_screen(self) -> None:
        """After the first show, when the real frame is known: pull a normal
        window whose frame overflows its screen back inside it."""
        try:
            if not self.isVisible() or self.isMaximized() or self.isFullScreen():
                return
            screen = self.screen()
            if screen is None:
                return
            avail = screen.availableGeometry()
            fr = self.frameGeometry()
            if avail.contains(fr):
                return
            extra_w = fr.width() - self.width()
            extra_h = fr.height() - self.height()
            w = min(self.width(), avail.width() - extra_w)
            h = min(self.height(), avail.height() - extra_h)
            self.resize(max(200, w), max(120, h))
            x = min(max(fr.x(), avail.x()), avail.right() + 1 - (w + extra_w))
            y = min(max(fr.y(), avail.y()), avail.bottom() + 1 - (h + extra_h))
            self.move(max(x, avail.x()), max(y, avail.y()))
        except Exception:  # noqa: BLE001 - layout memory must never block the UI
            pass

    def _track_normal_rect(self) -> None:
        """Remember the last normal rectangle (the one to reopen with)."""
        try:
            if self.isVisible() and not (self.isMaximized() or self.isFullScreen()
                                         or self.isMinimized()):
                from PySide6.QtCore import QRect
                self._last_normal_rect = QRect(self.pos(), self.size())
        except Exception:  # noqa: BLE001
            pass

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._track_normal_rect()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._track_normal_rect()

    def _save_window_state(self) -> None:
        """Remember the last normal rectangle and whether it is maximized."""
        try:
            self._settings.set(self._WINDOW_MAX_KEY, bool(self.isMaximized()))
            self._track_normal_rect()
            r = getattr(self, "_last_normal_rect", None)
            if r is not None and r.width() > 0 and r.height() > 0:
                self._settings.set(
                    self._WINDOW_RECT_KEY,
                    "%d,%d,%d,%d" % (r.x(), r.y(), r.width(), r.height()))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _logo_pixmap(size: int):
        """Load the app logo (same image as the taskbar icon) at ``size`` px."""
        from pathlib import Path
        from PySide6.QtGui import QPixmap
        root = Path(__file__).resolve().parent
        for p in (root / "resources" / "planaria_icon.png",
                  root.parent / "resources" / "planaria_icon.png",
                  root.parent / "planaria_icon.png"):
            if p.is_file():
                pm = QPixmap(str(p))
                if not pm.isNull():
                    return pm.scaled(size, size, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
        return None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        # A grabbable handle so the user can drag the sidebar wider/narrower.
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            "QSplitter::handle { background: transparent; }"
            "QSplitter::handle:hover { background: rgba(127,127,127,0.25); }"
        )

        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_stack())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 1200])

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")
        try:
            self._gpu_badge = GPUBadge()
            self._status.addPermanentWidget(self._gpu_badge)
        except Exception as exc:  # noqa: BLE001
            self._status.addPermanentWidget(QLabel(f"GPU probe failed: {exc}"))

    def _build_sidebar(self) -> QWidget:
        container = QWidget()
        container.setObjectName("sidebarContainer")
        self._sidebar_container = container
        self._sidebar_section_labels: List[QLabel] = []
        self._sidebar_collapsed = False
        # The sidebar width auto-fits its content. We compute the preferred
        # width from the current font metrics of every card and of the
        # header label. A small minimum (160 px) keeps the column usable
        # even when the user drags the splitter narrower. Rationale
        # follows Nielsen J., Usability Engineering, 1993,
        # ISBN: 0-12-518406-9, chapter 5, which recommends that
        # text-bearing panels should not be fixed at a zoom-independent
        # pixel width.
        container.setMinimumWidth(160)
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Branded header: the app logo (same image as the taskbar icon) next
        # to the product name, like the dashboard mockup.
        header_widget = QWidget()
        header_widget.setObjectName("sidebarHeader")
        header_widget.setAttribute(Qt.WA_StyledBackground, True)
        hh = QHBoxLayout(header_widget)
        hh.setContentsMargins(18, 16, 16, 14)
        hh.setSpacing(12)
        logo = QLabel()
        logo.setObjectName("sidebarLogo")
        pix = self._logo_pixmap(40)
        if pix is not None:
            logo.setPixmap(pix)
            logo.setFixedSize(40, 40)
            logo.setScaledContents(True)
            hh.addWidget(logo, 0, Qt.AlignVCenter)
        header = QLabel(f"{__app_name__}")
        self._sidebar_header = header
        header.setObjectName("sidebarHeaderText")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setWordWrap(True)
        hh.addWidget(header, 1, Qt.AlignVCenter)
        outer_layout.addWidget(header_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        cards_widget = QWidget()
        cards_widget.setObjectName("sidebarCards")
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setContentsMargins(8, 8, 8, 8)
        cards_layout.setSpacing(6)

        for index, (key, number, name, color) in enumerate(SIDEBAR_ITEMS):
            # Insert a section header + horizontal rule before the
            # first card of each group. Pipelines get a thicker divider
            # below them so the eye immediately distinguishes the
            # "pipelines" cluster from the step-by-step cluster.
            header_text = _SIDEBAR_SECTION_HEADERS.get(key)
            if header_text is not None:
                if index != 0:
                    cards_layout.addSpacing(6)
                    divider = QFrame()
                    divider.setObjectName("sidebarDivider")
                    divider.setFrameShape(QFrame.HLine)
                    divider.setFrameShadow(QFrame.Plain)
                    divider.setFixedHeight(2)
                    cards_layout.addWidget(divider)
                    cards_layout.addSpacing(4)
                header_label = QLabel(header_text)
                header_label.setObjectName("sidebarSectionHeader")
                cards_layout.addWidget(header_label)
                self._sidebar_section_labels.append(header_label)
            card = SidebarCard(
                step_index=index,
                step_number=number,
                step_name=name,
                icon_color=color,
            )
            card.step_selected.connect(self._on_card_selected)
            cards_layout.addWidget(card)
            self._sidebar_cards.append(card)
        cards_layout.addStretch(1)

        scroll.setWidget(cards_widget)
        self._sidebar_scroll = scroll
        self._sidebar_cards_widget = cards_widget
        outer_layout.addWidget(scroll, 1)

        # Foot of the sidebar: a theme toggle and the collapse/expand toggle
        # (same two buttons as the dashboard mockup).
        from PySide6.QtCore import QSize as _QSize

        self._theme_btn = QPushButton("  Toggle theme")
        self._theme_btn.setObjectName("sidebarThemeBtn")
        self._theme_btn.setIconSize(_QSize(18, 18))
        self._theme_btn.setIcon(self._theme_icon())
        self._theme_btn.setToolTip("Switch between the light and dark theme")
        self._theme_btn.clicked.connect(self._toggle_theme)

        self._collapse_btn = QPushButton("  Collapse")
        self._collapse_btn.setObjectName("sidebarCollapseBtn")
        self._collapse_btn.setIconSize(_QSize(18, 18))
        self._collapse_btn.setIcon(self._sidebar_toggle_icon(point_left=True))
        self._collapse_btn.setToolTip("Collapse the navigation bar")
        self._collapse_btn.clicked.connect(self._toggle_sidebar_collapsed)

        foot = QVBoxLayout()
        foot.setContentsMargins(8, 4, 8, 8)
        foot.setSpacing(6)
        foot.addWidget(self._theme_btn)
        foot.addWidget(self._collapse_btn)
        outer_layout.addLayout(foot)
        return container

    def _toggle_theme(self) -> None:
        """Switch between the light and dark theme."""
        if self._theme_manager is None:
            return
        current = self._theme_manager.theme
        self._theme_manager.set_theme("dark" if current == "light" else "light")
        # Repaint both foot-button icons in the new theme's contrast colour.
        self._theme_btn.setIcon(self._theme_icon())
        self._collapse_btn.setIcon(
            self._sidebar_toggle_icon(point_left=not self._sidebar_collapsed)
        )

    def _theme_icon(self):
        """A moon (switch to dark) when light, a sun (switch to light) when dark."""
        import math

        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import (
            QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap,
        )

        dark = (
            self._theme_manager is not None
            and self._theme_manager.theme == "dark"
        )
        color = QColor("#E2E8F0") if dark else QColor("#0F172A")
        size = 18
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        c = size / 2.0
        if dark:
            # Sun: a disc with eight rays.
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(c, c), 3.0, 3.0)
            pen = QPen(color, 1.5)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            for k in range(8):
                a = math.radians(k * 45)
                p.drawLine(
                    QPointF(c + math.cos(a) * 5.4, c + math.sin(a) * 5.4),
                    QPointF(c + math.cos(a) * 7.4, c + math.sin(a) * 7.4),
                )
        else:
            # Moon: a crescent (one disc minus an offset disc).
            disc = QPainterPath()
            disc.addEllipse(QRectF(3.0, 3.0, 12.0, 12.0))
            cut = QPainterPath()
            cut.addEllipse(QRectF(6.5, 1.5, 12.0, 12.0))
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawPath(disc.subtracted(cut))
        p.end()
        return QIcon(pm)

    def _sidebar_toggle_icon(self, point_left: bool):
        """A double chevron (« / »), like the dashboard mockup's collapse icon.

        ``point_left`` draws it pointing left (« -- the button collapses the
        bar); when the bar is already collapsed it points right (» -- expand).
        """
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import (
            QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF,
        )

        dark = (
            self._theme_manager is not None
            and self._theme_manager.theme == "dark"
        )
        color = QColor("#E2E8F0") if dark else QColor("#0F172A")
        size = 18
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(color, 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        top, mid, bot = 4.5, 9.0, 13.5
        for off in (0.0, 5.0):
            if point_left:  # «
                chev = QPolygonF([
                    QPointF(11.0 - off, top), QPointF(6.0 - off, mid),
                    QPointF(11.0 - off, bot),
                ])
            else:  # »
                chev = QPolygonF([
                    QPointF(7.0 + off, top), QPointF(12.0 + off, mid),
                    QPointF(7.0 + off, bot),
                ])
            p.drawPolyline(chev)
        p.end()
        return QIcon(pm)

    def _toggle_sidebar_collapsed(self) -> None:
        """Fold the sidebar to icon-only, or expand it back to full width."""
        self._sidebar_collapsed = not self._sidebar_collapsed
        collapsed = self._sidebar_collapsed
        self._settings.set("sidebar_collapsed", bool(collapsed))
        self._sidebar_header.setVisible(not collapsed)
        for lbl in self._sidebar_section_labels:
            lbl.setVisible(not collapsed)
        for card in self._sidebar_cards:
            card.set_collapsed(collapsed)
        splitter = self.centralWidget().layout().itemAt(0).widget()
        if collapsed:
            self._collapse_btn.setText("")
            self._collapse_btn.setIcon(self._sidebar_toggle_icon(point_left=False))
            self._collapse_btn.setToolTip("Expand the navigation bar")
            self._theme_btn.setText("")
            self._sidebar_container.setMinimumWidth(56)
            width = 72
        else:
            self._collapse_btn.setText("  Collapse")
            self._collapse_btn.setIcon(self._sidebar_toggle_icon(point_left=True))
            self._collapse_btn.setToolTip("Collapse the navigation bar")
            self._theme_btn.setText("  Toggle theme")
            # _auto_fit_sidebar recomputes and re-applies the full width.
            self._auto_fit_sidebar()
            return
        if isinstance(splitter, QSplitter):
            sizes = splitter.sizes()
            if sizes:
                splitter.setSizes([width, max(0, sum(sizes) - width)])

    # ------------------------------------------------------------------
    # Sidebar auto-fit
    # ------------------------------------------------------------------
    def _auto_fit_sidebar(self) -> None:
        """Compute and apply a preferred width that shows every card in full.

        The width is the maximum between the header's text width and the
        widest card's sizeHint. We add the scrollbar thickness and a
        small padding. Called at startup and whenever the UI scale
        (Ctrl+wheel) changes; also invoked when the window is resized
        for the first time. The QSplitter handle stays draggable so the
        user can always shrink or enlarge the column further.
        """
        if getattr(self, "_sidebar_collapsed", False):
            # Folded to icons: the full-width fit would unfold it visually.
            return
        try:
            fm_header = self._sidebar_header.fontMetrics()
            header_padding = 40  # left/right padding inside sidebarHeader
            header_w = (
                fm_header.horizontalAdvance(self._sidebar_header.text())
                + header_padding
            )
            card_w = 0
            for card in self._sidebar_cards:
                # ``natural_width`` measures the real text (word-wrapped labels
                # report a near-zero size hint, which used to collapse the
                # sidebar to a sliver at large fonts).
                nw = card.natural_width() if hasattr(card, "natural_width") \
                    else card.sizeHint().width()
                card_w = max(card_w, nw)
            # Add a bit of slack for scrollbar + margins (8 px on each
            # side from cards_layout plus 12 px scrollbar gutter).
            scrollbar = 14
            margins = 16
            target = max(header_w, card_w + scrollbar + margins)
            # Hard-cap so a very large font does not consume the whole
            # viewport: the user can still drag wider via the splitter.
            target = max(200, min(target, int(self.width() * 0.5)))
            # Enforce the content width as the sidebar minimum so the splitter
            # (or a restored layout) can never squeeze the text into a clip.
            self._sidebar_container.setMinimumWidth(target)
            splitter = self.centralWidget().layout().itemAt(0).widget()
            if isinstance(splitter, QSplitter):
                sizes = splitter.sizes()
                if sizes:
                    remaining = max(0, sum(sizes) - target)
                    splitter.setSizes([target, remaining])
        except Exception:  # noqa: BLE001
            # The computation is purely cosmetic; never crash the UI.
            pass

    def _build_stack(self) -> QWidget:
        self._stack = QStackedWidget()
        self._pages["step1"] = Step1Page()
        self._pages["step2"] = Step2Page()
        self._pages["step3"] = Step3Page()
        self._pages["step4"] = Step4Page()
        self._pages["step5"] = Step5Page()
        self._pages["step6"] = Step6Page()
        # Merged classification page (former Stereotype classification + Stereotype classification from images). Its two
        # embedded workflows are also kept in ``_pages`` under their old keys
        # so any code that still references them keeps working.
        stereotype_page = StereotypeClassificationPage()
        self._pages["stereotype"] = stereotype_page
        self._pages["pipeline_a"] = stereotype_page.page_a
        self._pages["pipeline_b"] = stereotype_page.page_b
        self._pages["pipeline_c"] = PipelineCPage()
        self._pages["regeneration"] = RegenerationPage()
        self._pages["eyespots"] = EyespotsCounterPage()
        self._pages["regeneration_editor"] = RegenerationEditorPage()
        self._pages["pipeline_d"] = PipelineDPage()
        self._pages["pipeline_e"] = PipelineEPage()
        self._pages["settings"] = SettingsPage(theme_manager=self._theme_manager)
        self._pages["about"] = AboutPage()
        # Pages that carry an execution-log panel are made vertically
        # scrollable so the log can be dragged taller: when it grows past the
        # viewport the Run/Cancel bar scrolls below it instead of being pushed
        # off-screen. Editor/settings/about pages keep their own fixed layout.
        # The merged "stereotype" page already embeds its own scroll area, so
        # it is excluded here.
        scrollable_keys = {
            "step1", "step2", "step3", "step4", "step6",
            "pipeline_c", "regeneration", "eyespots",
            "pipeline_d", "pipeline_e",
        }
        for key, _number, _name, _color in SIDEBAR_ITEMS:
            page = self._pages[key]
            self._stack.addWidget(
                self._scrollable(page) if key in scrollable_keys else page
            )
        return self._stack

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Wrap ``page`` in a vertically scrollable, frameless container."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setWidget(page)
        return area

    def _connect_pages(self) -> None:
        for idx, (key, _number, _name, _color) in enumerate(SIDEBAR_ITEMS):
            page = self._pages[key]
            if hasattr(page, "page_status_changed"):
                page.page_status_changed.connect(
                    lambda status, i=idx: self._on_page_status(i, status)
                )
            if hasattr(page, "run_state_changed"):
                page.run_state_changed.connect(
                    lambda name, running, i=idx: self._on_run_state(i, name, running)
                )
            # A page can ask to switch to another page on success (e.g. a
            # finished classification jumps to its editor).
            if hasattr(page, "navigate_requested"):
                page.navigate_requested.connect(self._navigate_to_key)
        # Restore the last visited page (falls back to the first card).
        # ``_forced_about`` keeps the saved "last_page" intact when we divert
        # to About because of a missing package: the operator installs it,
        # restarts, and lands back on the page they were actually using.
        self._forced_about = self._missing_packages()
        self._on_card_selected(self._initial_page_index())
        self._forced_about = False

    def _initial_page_index(self) -> int:
        """Index of the page to open at launch.

        Normally the last visited page. But when a package a processing chain
        needs is missing, the application opens on About with the Dependencies
        card unfolded instead: going straight to a pipeline page the operator
        cannot run would only end in a refused launch, whereas About states
        what is missing and offers the install command.
        """
        if self._missing_packages():
            for i, item in enumerate(SIDEBAR_ITEMS):
                if item[0] == "about":
                    return i
        saved = str(self._settings.get("last_page", "")).strip()
        if saved:
            for i, item in enumerate(SIDEBAR_ITEMS):
                if item[0] == saved:
                    return i
        return 0

    @staticmethod
    def _missing_packages() -> bool:
        """True when any declared package cannot be imported.

        Best-effort: a failure here must never prevent the window from
        opening, so any error falls back to the normal navigation.
        """
        try:
            from .dependency_check import _OPTIONAL, find_missing
            return bool(find_missing() or find_missing(_OPTIONAL))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Show / resize hooks
    # ------------------------------------------------------------------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Fit the sidebar to its content on the first show event.

        QTimer.singleShot defers the call until the event loop has
        processed the show, so the card widgets have computed their
        final size hints. Further changes in UI scale trigger
        ``_auto_fit_sidebar`` directly through ``scale_changed``.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._auto_fit_sidebar)
        QTimer.singleShot(300, self._keep_frame_on_screen)
        self._schedule_full_repaint()

    # Under WSLg (and some other X11 compositors) the first frames of a
    # newly mapped -- and especially a just-maximized -- window can reach the
    # screen before the compositor is ready to display them: the window stays
    # black until something repaints it, typically the pointer moving over a
    # widget. Repainting the whole window a few times during the first two
    # seconds after it is shown or changes state (maximized, restored) puts the
    # interface on screen without waiting for the pointer. A repaint of a
    # visible window is cheap and has no other effect.
    _REPAINT_DELAYS_MS = (30, 150, 400, 900, 1800)

    def _schedule_full_repaint(self) -> None:
        for ms in self._REPAINT_DELAYS_MS:
            QTimer.singleShot(ms, self._full_repaint)

    def _full_repaint(self) -> None:
        try:
            if self.isVisible():
                self.update()
                self.repaint()
        except RuntimeError:  # window already destroyed on shutdown
            pass

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._schedule_full_repaint()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _navigate_to_key(self, key: str) -> None:
        """Switch to the sidebar page with this key, if it is visible."""
        for i, item in enumerate(SIDEBAR_ITEMS):
            if item[0] == key:
                self._on_card_selected(i)
                return

    def _on_card_selected(self, index: int) -> None:
        if 0 <= index < len(self._sidebar_cards):
            for i, card in enumerate(self._sidebar_cards):
                card.set_active(i == index)
            self._stack.setCurrentIndex(index)
            key, number, name, _color = SIDEBAR_ITEMS[index]
            self._status.showMessage(f"{number}: {name}")
            # Remember the last visited page so the next launch opens it --
            # except when this selection is the forced diversion to About.
            if not getattr(self, "_forced_about", False):
                self._settings.set("last_page", key)

    def _on_page_status(self, index: int, status: str) -> None:
        if 0 <= index < len(self._sidebar_cards):
            self._sidebar_cards[index].set_status(status)

    def _on_run_state(self, index: int, name: str, running: bool) -> None:
        if running:
            self._sidebar_cards[index].set_status("running")
            self._status.showMessage(f"{name}: running...")
        else:
            self._status.showMessage(f"{name}: finished")
            # Same release as on cancel and on exit. A run that ends -- whether
            # it completed, was cancelled or died on an out-of-memory error --
            # has no further use for the cached models, and after a crash the
            # blocks are exactly what the NEXT attempt needs back.
            self._release_gpu_resources()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Flush settings, stop any worker, and release GPU memory."""
        registry = WorkerRegistry()
        if registry.active():
            from .widgets.file_dialogs import question as _msg_question
            reply = _msg_question(
                self, "Confirm exit",
                "A task is still running. Cancel and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            registry.stop_all(wait_ms=3000)
        self._save_window_state()
        try:
            self._settings.flush()
        except Exception:
            pass
        self._release_gpu_resources()
        super().closeEvent(event)

    def _release_gpu_resources(self) -> None:
        """Drop cached models and empty the CUDA allocator."""
        try:
            ModelCache().clear()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
