"""
Runtime theme and UI-scale manager for the FlatMorphAI dashboard.

Responsibilities
----------------
- Load the dark or light stylesheet from disk and apply it to the
  running :class:`QApplication`.
- Scale every font-size rule in the stylesheet (and the
  :class:`QApplication` base font) by a user controllable factor so
  the whole UI grows or shrinks uniformly.
- Install a global event filter on :class:`QApplication` so that
  ``Ctrl+MouseWheel`` over any widget bumps the scale up or down, and
  ``Ctrl+Plus`` / ``Ctrl+Minus`` / ``Ctrl+0`` behave as the usual
  browser zoom shortcuts.
- Persist every change through :class:`AppSettings` so the same theme
  and scale factor are restored on the next launch.

Design rationale
----------------
Qt Style Sheets that set ``font-size: 11pt`` on ``QWidget`` override
the base font assigned through :meth:`QApplication.setFont`, so
simply bumping the application font does not scale styled controls.
The manager therefore parses each ``\\d+(?:\\.\\d+)?pt`` literal in
the stylesheet and multiplies it by the current scale before
assigning the resulting text with :meth:`QApplication.setStyleSheet`.

The accessibility floor (200 percent zoom) is taken from the W3C
Recommendation WCAG 2.1, 11 December 2018, success criterion 1.4.4
"Resize text". The Ctrl+MouseWheel gesture mirrors the long-standing
browser-zoom convention documented in Nielsen Norman Group,
"Zoomed-Out Design: Why Modern Web Users Need Larger Text", 2021
(no DOI, identifier: NNGroup article, nngroup.com).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPalette, QShortcut, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from .config import AppSettings
from .constants import (
    AVAILABLE_THEMES,
    STYLESHEET_FILES,
    THEME_DARK,
    THEME_LIGHT,
    UI_SCALE_DEFAULT,
    UI_SCALE_MAX,
    UI_SCALE_MIN,
    UI_SCALE_STEP,
)


# Matches integer and decimal pt literals, e.g. "11pt", "10.5pt".
_PT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)pt\b")
# Matches px literals, e.g. "14px", "1px". These (padding, margins, radii,
# min-heights) are scaled too so the whole UI zooms proportionally with the
# font instead of leaving cramped padding around huge text.
_PX_PATTERN = re.compile(r"(\d+(?:\.\d+)?)px\b")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ThemeManager(QObject):
    """Applies theme + scale to ``QApplication`` and persists the state."""

    theme_changed = Signal(str)
    scale_changed = Signal(float)

    _BASE_FONT_POINT_SIZE = 11.0  # baseline that 1.0 scale corresponds to

    def __init__(
        self,
        app: QApplication,
        settings: AppSettings,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._app = app
        self._settings = settings
        self._theme = self._coerce_theme(settings.get("theme", THEME_LIGHT))
        self._scale = _clamp(
            float(settings.get("ui_scale", UI_SCALE_DEFAULT) or UI_SCALE_DEFAULT),
            UI_SCALE_MIN,
            UI_SCALE_MAX,
        )
        # Cached raw stylesheet text keyed by theme, to avoid re-reading
        # disk on every scale change.
        self._raw_qss_cache: dict[str, str] = {}
        # The shortcuts must be referenced from the Python side or the
        # shiboken garbage collector can drop the wrappers as soon as
        # ``_install_shortcuts`` returns. A subsequent activation would
        # then dispatch to a deleted Python receiver and crash the GUI
        # thread inside ``app.exec()``. Keep a list alive on the manager.
        self._shortcuts: list = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def install(self, shortcut_parent: Optional[QWidget] = None) -> None:
        """Apply the current theme/scale and hook up input handlers."""
        self._apply_all()
        self._app.installEventFilter(self)
        if shortcut_parent is not None:
            self._install_shortcuts(shortcut_parent)

    def _install_shortcuts(self, parent: QWidget) -> None:
        # Build the shortcuts with explicit signal/slot connections and
        # store strong references in :attr:`_shortcuts`. The keyword
        # argument ``activated=`` form has been observed to leak the
        # connection into a wrapper that shiboken collects, which then
        # leaves a dangling C++ slot (Qt for Python bug PYSIDE-2131,
        # bugreports.qt.io/browse/PYSIDE-2131).
        for sequence, handler in (
            ("Ctrl+=", self.increase_scale),
            ("Ctrl++", self.increase_scale),
            ("Ctrl+-", self.decrease_scale),
            ("Ctrl+0", self.reset_scale),
        ):
            shortcut = QShortcut(QKeySequence(sequence), parent)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def theme(self) -> str:
        return self._theme

    @property
    def scale(self) -> float:
        return self._scale

    def set_theme(self, theme: str) -> None:
        theme = self._coerce_theme(theme)
        if theme == self._theme:
            return
        self._theme = theme
        self._apply_stylesheet()
        self._settings.set("theme", theme)
        self.theme_changed.emit(theme)

    def set_scale(self, scale: float) -> None:
        scale = _clamp(round(scale, 2), UI_SCALE_MIN, UI_SCALE_MAX)
        if abs(scale - self._scale) < 1e-3:
            return
        self._scale = scale
        self._apply_all()
        self._settings.set("ui_scale", scale)
        self.scale_changed.emit(scale)

    def increase_scale(self) -> None:
        self.set_scale(self._scale + UI_SCALE_STEP)

    def decrease_scale(self) -> None:
        self.set_scale(self._scale - UI_SCALE_STEP)

    def reset_scale(self) -> None:
        self.set_scale(UI_SCALE_DEFAULT)

    # ------------------------------------------------------------------
    # Application hooks
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Intercept Ctrl+MouseWheel anywhere in the app to change scale."""
        if event.type() == QEvent.Wheel and isinstance(event, QWheelEvent):
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.increase_scale()
                elif delta < 0:
                    self.decrease_scale()
                event.accept()
                return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_theme(theme: str) -> str:
        if theme in AVAILABLE_THEMES:
            return theme
        return THEME_LIGHT

    # Palette overrides per theme. Only the roles that affect custom
    # painted widgets (pipeline timeline, log severity dots, etc.)
    # are set explicitly. The stylesheet still drives control colours.
    _PALETTE_BY_THEME = {
        THEME_DARK: {
            QPalette.Window: "#0B1017",       # gray canvas (cards are lighter)
            QPalette.WindowText: "#E6EDF6",
            QPalette.Text: "#E6EDF6",
            QPalette.Base: "#111823",
            QPalette.Button: "#172130",
            QPalette.ButtonText: "#E6EDF6",
            QPalette.Highlight: "#2DD4BF",    # brand teal
            QPalette.HighlightedText: "#04211E",
            QPalette.PlaceholderText: "#8B9BB0",
        },
        THEME_LIGHT: {
            QPalette.Window: "#F4F6F9",       # gray canvas so white cards stand out
            QPalette.WindowText: "#0F172A",
            QPalette.Text: "#0F172A",
            QPalette.Base: "#FFFFFF",
            QPalette.Button: "#FFFFFF",
            QPalette.ButtonText: "#0F172A",
            QPalette.Highlight: "#0D9488",    # brand teal
            QPalette.HighlightedText: "#FFFFFF",
            QPalette.PlaceholderText: "#64748B",
        },
    }

    def _apply_all(self) -> None:
        self._apply_palette()
        self._apply_font_scale()
        self._apply_stylesheet()

    def _apply_palette(self) -> None:
        roles = self._PALETTE_BY_THEME.get(self._theme, {})
        palette = self._app.palette()
        for role, hex_color in roles.items():
            palette.setColor(role, QColor(hex_color))
        self._app.setPalette(palette)

    def _apply_font_scale(self) -> None:
        font: QFont = self._app.font()
        font.setPointSizeF(self._BASE_FONT_POINT_SIZE * self._scale)
        self._app.setFont(font)

    def _apply_stylesheet(self) -> None:
        qss = self._read_stylesheet(self._theme)
        scaled = self._scale_literals(qss, self._scale)
        # Clear first so Qt does not merge the previous rules with the
        # new ones on cached painter paths.
        self._app.setStyleSheet("")
        self._app.setStyleSheet(scaled)

    @staticmethod
    def _scale_literals(qss: str, scale: float) -> str:
        """Multiply every ``Npt`` and ``Npx`` literal in the QSS by ``scale``.

        Scaling the px values (padding, margins, border-radius, min sizes)
        alongside the font keeps the whole interface proportional at any UI
        scale, so cards and controls breathe instead of wrapping huge text into
        tiny, fixed padding.
        """

        def _mk(unit: str):
            def _substitute(match: "re.Match[str]") -> str:
                try:
                    value = float(match.group(1))
                except ValueError:
                    return match.group(0)
                scaled_value = round(value * scale, 2)
                if abs(scaled_value - int(scaled_value)) < 1e-3:
                    return f"{int(scaled_value)}{unit}"
                return f"{scaled_value:g}{unit}"
            return _substitute

        qss = _PT_PATTERN.sub(_mk("pt"), qss)
        qss = _PX_PATTERN.sub(_mk("px"), qss)
        return qss

    def _read_stylesheet(self, theme: str) -> str:
        if theme in self._raw_qss_cache:
            return self._raw_qss_cache[theme]
        filename = STYLESHEET_FILES.get(theme, STYLESHEET_FILES[THEME_LIGHT])
        package_root = Path(__file__).resolve().parent
        candidates = [
            package_root / "resources" / "themes" / filename,  # repository layout
            package_root.parent / filename,   # <root>/<file> (older layout)
            package_root / filename,          # fallback next to the package
        ]
        for path in candidates:
            if path.is_file():
                try:
                    data = path.read_text(encoding="utf-8")
                    self._raw_qss_cache[theme] = data
                    return data
                except OSError:
                    continue
        return ""
