"""
Interactive evolution graphs for the Regeneration states editor.

:class:`EvolutionGraph` is a light, dependency-free line chart drawn with
``QPainter``.  It plots one or two series against the ``Day`` axis of a single
planaria:

* a primary series (the rounded / corrected regeneration state, or the eyespot
  count) drawn in the accent colour with a dot on every point ;
* an optional secondary series (the decimal state forecast by the model) drawn
  in grey.

Interactions:

* a dashed vertical line marks the Day of the image currently displayed ;
* hovering highlights the primary point of the pointed Day and shows its value ;
* clicking a column emits :attr:`EvolutionGraph.daySelected` so the page can
  open that Day's image ;
* the mouse wheel zooms the Day axis (centred on the cursor), dragging pans it,
  and a horizontal scrollbar slides the visible window when zoomed in - so a
  long time course is never cut off ;
* a toolbar gives Zoom -, Zoom +, taller/shorter, Reset, per-curve visibility
  checkboxes and an Export-to-image button.

The value axis takes fixed bounds (e.g. 1..6 for the regeneration state) and,
for a small integer range, labels every integer so the axis reads 1, 2, ... 6.

:class:`CollapsibleSection` wraps any widget under a clickable header that folds
it away and shows the current value in bold.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .file_dialogs import get_save_file_name

Point = Tuple[float, float]  # (day, value)

# Palette (kept close to the rest of the app's blue/teal accents).
_PRIMARY = QColor("#0b63c4")
_SECONDARY = QColor("#9aa5b1")
_CURRENT = QColor("#d64545")
_AXIS = QColor("#52606d")
_GRID = QColor("#e4e9f0")
_HOVER = QColor("#0b63c4")

# Day-axis zoom bounds (1.0 = show the whole time course).
_XZOOM_MIN = 1.0
_XZOOM_MAX = 20.0
_XZOOM_STEP = 1.5

# Canvas height bounds and step (pixels) driven by the taller/shorter buttons.
# The default is tall enough that the locked 1..6 state axis shows all six
# integer labels with clear separation even on a HiDPI display.
_H_MIN = 170
_H_MAX = 700
_H_STEP = 60
_H_DEFAULT = 300

# Scrollbar works in tenths of a Day so the slide feels smooth.
_SB_UNITS = 10


class _GraphCanvas(QWidget):
    """The painted chart surface with hover, click, wheel-zoom and drag-pan."""

    daySelected = Signal(int)
    viewChanged = Signal()  # emitted whenever the Day window changes

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._primary: List[Point] = []
        self._secondary: List[Point] = []
        self._current_day: Optional[int] = None
        self._y_min: Optional[float] = None
        self._y_max: Optional[float] = None
        self._y_integer: bool = False
        self._hover_day: Optional[int] = None
        self._show_primary = True
        self._show_secondary = True
        # Cached geometry recomputed on every paint so hover/click can map
        # pixels back to a Day without duplicating the layout maths.
        self._days: List[int] = []
        self._prim_by_day: Dict[int, float] = {}
        self._plot = (0, 0, 0, 0)  # left, top, width, height
        self._dmin = 0
        self._dmax = 0
        self._vmin = 0.0
        self._vmax = 1.0
        # Day-axis view: zoom factor and the day at the centre of the view.
        self._xzoom = 1.0
        self._xcenter: Optional[float] = None
        self._view_lo = 0.0
        self._view_hi = 1.0
        # Drag state (pan vs click disambiguation).
        self._press_x: Optional[int] = None
        self._press_center: Optional[float] = None
        self._panning = False
        self.setMinimumHeight(_H_MIN)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_data(
        self,
        primary: List[Point],
        secondary: Optional[List[Point]] = None,
        current_day: Optional[int] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        y_integer: bool = False,
    ) -> None:
        self._primary = sorted(primary or [])
        self._secondary = sorted(secondary or [])
        self._current_day = current_day
        self._y_min = y_min
        self._y_max = y_max
        self._y_integer = y_integer
        self.update()
        self.viewChanged.emit()

    def value_at(self, day: int) -> Optional[float]:
        for d, v in self._primary:
            if d == day:
                return v
        return None

    def set_show_primary(self, show: bool) -> None:
        self._show_primary = bool(show)
        self.update()

    def set_show_secondary(self, show: bool) -> None:
        self._show_secondary = bool(show)
        self.update()

    # --- view controls ------------------------------------------------
    def set_x_zoom(self, factor: float) -> None:
        factor = max(_XZOOM_MIN, min(_XZOOM_MAX, float(factor)))
        if factor != self._xzoom:
            self._xzoom = factor
            self.update()
            self.viewChanged.emit()

    def zoom_x_in(self) -> None:
        self.set_x_zoom(self._xzoom * _XZOOM_STEP)

    def zoom_x_out(self) -> None:
        self.set_x_zoom(self._xzoom / _XZOOM_STEP)

    def reset_view(self) -> None:
        self._xzoom = 1.0
        self._xcenter = None
        self.update()
        self.viewChanged.emit()

    def set_view_lo(self, day_lo: float) -> None:
        """Slide the window so it starts at ``day_lo`` (used by the scrollbar)."""
        span = self._view_hi - self._view_lo
        self._xcenter = day_lo + span / 2.0
        self.update()
        self.viewChanged.emit()

    def view_state(self) -> Tuple[int, int, float, float, float]:
        """Return (dmin, dmax, view_lo, view_hi, xzoom) for the scrollbar."""
        return (self._dmin, self._dmax, self._view_lo, self._view_hi, self._xzoom)

    # ------------------------------------------------------------------
    # View range helpers
    # ------------------------------------------------------------------
    def _compute_view(self) -> Tuple[float, float]:
        dmin, dmax = self._dmin, self._dmax
        span = max(1e-9, dmax - dmin)
        if self._xzoom <= 1.0:
            self._xcenter = None
            return dmin, dmax
        view_span = span / self._xzoom
        center = self._xcenter if self._xcenter is not None else (dmin + dmax) / 2.0
        lo = center - view_span / 2.0
        hi = center + view_span / 2.0
        if lo < dmin:
            lo, hi = dmin, dmin + view_span
        if hi > dmax:
            hi, lo = dmax, dmax - view_span
        self._xcenter = (lo + hi) / 2.0
        return lo, hi

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------
    def _nearest_day(self, mx: int) -> Optional[int]:
        if not self._days:
            return None
        left, _top, width, _h = self._plot
        if width <= 0 or self._view_hi == self._view_lo:
            return self._days[0]
        frac = (mx - left) / float(width)
        day_f = self._view_lo + frac * (self._view_hi - self._view_lo)
        return min(self._days, key=lambda d: abs(d - day_f))

    def _day_at_x(self, mx: int) -> float:
        left, _top, width, _h = self._plot
        if width <= 0:
            return (self._view_lo + self._view_hi) / 2.0
        frac = (mx - left) / float(width)
        return self._view_lo + frac * (self._view_hi - self._view_lo)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position() if hasattr(event, "position") else event.pos()
        mx, my = int(pos.x()), int(pos.y())
        if self._press_x is not None and not self._panning:
            if abs(mx - self._press_x) > 4:
                self._panning = True
        if self._panning and self._xzoom > 1.0:
            left, _top, width, _h = self._plot
            if width > 0:
                view_span = self._view_hi - self._view_lo
                dx_days = (self._press_x - mx) / float(width) * view_span
                base = self._press_center if self._press_center is not None else self._xcenter
                if base is not None:
                    self._xcenter = base + dx_days
            self.update()
            self.viewChanged.emit()
            return
        left, top, width, height = self._plot
        inside = (left <= mx <= left + width) and (top <= my <= top + height)
        new_hover = self._nearest_day(mx) if inside else None
        if new_hover != self._hover_day:
            self._hover_day = new_hover
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_day is not None:
            self._hover_day = None
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            pos = event.position() if hasattr(event, "position") else event.pos()
            self._press_x = int(pos.x())
            self._press_center = self._xcenter
            self._panning = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._press_x is not None:
            pos = event.position() if hasattr(event, "position") else event.pos()
            mx, my = int(pos.x()), int(pos.y())
            was_pan = self._panning
            self._press_x = None
            self._press_center = None
            self._panning = False
            if not was_pan:
                left, top, width, height = self._plot
                inside = (left <= mx <= left + width) and (top <= my <= top + height)
                day = self._nearest_day(mx)
                if day is not None and inside:
                    self.daySelected.emit(int(day))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        pos = event.position() if hasattr(event, "position") else event.pos()
        anchor_day = self._day_at_x(int(pos.x()))
        step = _XZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / _XZOOM_STEP
        new_zoom = max(_XZOOM_MIN, min(_XZOOM_MAX, self._xzoom * step))
        if new_zoom != self._xzoom:
            self._xzoom = new_zoom
            self._xcenter = anchor_day if new_zoom > 1.0 else None
            self.update()
            self.viewChanged.emit()
        event.accept()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def _theme_colors(self):
        """(primary, secondary, current, axis, grid, hover) for the theme.

        Matches the dashboard: corrected = indigo accent, forecast = muted
        grey, current-day = warn red. Adapts to the dark theme by reading the
        widget palette so the graph is never light-on-light or dark-on-dark.
        """
        dark = self.palette().color(QPalette.Window).lightness() < 128
        if dark:
            return (QColor("#818cf8"), QColor("#8b9bb0"), QColor("#fb7185"),
                    QColor("#8b9bb0"), QColor("#243146"), QColor("#818cf8"))
        return (QColor("#4f46e5"), QColor("#64748b"), QColor("#e11d48"),
                QColor("#64748b"), QColor("#e2e8f0"), QColor("#4f46e5"))

    def paintEvent(self, _event) -> None:  # noqa: N802
        # Theme-aware colours (assigning these names here makes every
        # reference below use the local, palette-driven values).
        _PRIMARY, _SECONDARY, _CURRENT, _AXIS, _GRID, _HOVER = self._theme_colors()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Axis labels use a slightly smaller font than the widget so all the
        # state integers (1..6) and the "D<day>" labels fit without crowding.
        label_font = QFont(self.font())
        _base_pt = self.font().pointSizeF()
        if _base_pt > 0:
            label_font.setPointSizeF(max(7.0, _base_pt - 2.0))
        fm = QFontMetrics(label_font)

        left = 12 + fm.horizontalAdvance("0.0")
        # Enough right room that the last "D<day>" label stays centred on its
        # point without being cut off by the widget edge.
        right = 28
        # Head room so the top value tick never collides with the toolbar just
        # above the canvas; foot room for the X tick marks + the "D<day>" row
        # (tick 4 px + 6 px gap + the label height + a little margin).
        top = 18
        bottom = 22 + fm.height()
        width = max(1, self.width() - left - right)
        height = max(1, self.height() - top - bottom)
        self._plot = (left, top, width, height)

        if not self._primary and not self._secondary:
            painter.setPen(QPen(_SECONDARY))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data")
            painter.end()
            return

        days = sorted(
            {int(d) for d, _ in self._primary} | {int(d) for d, _ in self._secondary}
        )
        self._days = days
        self._prim_by_day = {int(d): v for d, v in self._primary}
        self._dmin, self._dmax = days[0], days[-1]

        self._view_lo, self._view_hi = self._compute_view()
        vspan = max(1e-9, self._view_hi - self._view_lo)

        vals = [v for _, v in self._primary] + [v for _, v in self._secondary]
        vmin = self._y_min if self._y_min is not None else min(vals)
        vmax = self._y_max if self._y_max is not None else max(vals)
        if vmax <= vmin:
            vmax = vmin + 1.0
        if self._y_min is None:
            vmin -= 0.05 * (vmax - vmin)
        if self._y_max is None:
            vmax += 0.05 * (vmax - vmin)
        self._vmin, self._vmax = vmin, vmax

        def px(day: float) -> float:
            return left + (day - self._view_lo) / vspan * width

        def py(val: float) -> float:
            return top + (1.0 - (val - vmin) / (vmax - vmin)) * height

        # --- value grid + ticks ---
        # For an integer axis EVERY integer is labelled (the state axis always
        # reads 1,2,3,4,5,6; the eyespot axis 0,1,2,...), as requested. A grid
        # line is drawn at each one. Non-integer axes space their ticks by the
        # available height.
        if self._y_integer:
            lo, hi = int(round(vmin)), int(round(vmax))
            tick_vals: List[float] = [float(v) for v in range(lo, hi + 1)]
        else:
            n_ticks = max(2, min(10, height // 24))
            tick_vals = [vmin + i * (vmax - vmin) / n_ticks for i in range(n_ticks + 1)]

        painter.setFont(label_font)
        for tv in tick_vals:
            y = py(tv)
            painter.setPen(QPen(_GRID, 1))
            painter.drawLine(int(left), int(y), int(left + width), int(y))
            painter.setPen(QPen(_AXIS))
            label = ("%d" % round(tv)) if self._y_integer else ("%.1f" % tv)
            # Centre the label exactly on its grid line (the head/foot margins
            # leave room for the end labels), so every value sits level with
            # its line like in the reference chart.
            ly = int(round(y - fm.height() / 2.0))
            painter.drawText(0, ly, left - 6, fm.height(),
                             Qt.AlignRight | Qt.AlignVCenter, label)

        # No solid axis lines or tick marks (the HTML uses only horizontal grid
        # lines + labels). ``axis_y`` is kept for positioning the Day labels.
        axis_y = int(top + height)

        # --- x (Day) labels just below the plot, thinned when there is not
        #     enough width for them all ---
        visible_days = [d for d in days if self._view_lo - 1e-6 <= d <= self._view_hi + 1e-6]
        # Reserve ~50 px per "D<day>" label so each has room and never collides
        # with its neighbour; days in between are skipped.
        max_labels = max(2, width // 50)
        label_step = max(1, -(-len(visible_days) // max_labels))
        painter.setPen(QPen(_AXIS))
        for i, d in enumerate(visible_days):
            if i % label_step and d != visible_days[-1]:
                continue
            x = int(px(d))
            # Centre each label on its point; clamp only to the widget edges so
            # the first/last labels are never cut off (but stay under their
            # point, like the reference chart).
            bx = int(max(0, min(x - 24, self.width() - 48)))
            painter.drawText(bx, axis_y + 6, 48, fm.height(),
                             Qt.AlignHCenter | Qt.AlignTop, "D%d" % d)

        painter.save()
        painter.setClipRect(int(left), int(top), int(width) + 1, int(height) + 1)

        # --- dashed current-day line ---
        if (
            self._current_day is not None
            and self._view_lo <= self._current_day <= self._view_hi
        ):
            painter.setPen(QPen(_CURRENT, 1, Qt.DashLine))
            x = px(self._current_day)
            painter.drawLine(int(x), int(top), int(x), int(top + height))

        # --- secondary (forecast) polyline: dashed grey, like the HTML ---
        if self._secondary and self._show_secondary:
            pen = QPen(_SECONDARY, 2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            self._draw_polyline(painter, self._secondary, px, py)

        # --- primary (corrected) polyline: solid accent with dots ---
        if self._primary and self._show_primary:
            painter.setPen(QPen(_PRIMARY, 2.5))
            self._draw_polyline(painter, self._primary, px, py)
            painter.setBrush(_PRIMARY)
            for d, v in self._primary:
                painter.drawEllipse(int(px(d)) - 3, int(py(v)) - 3, 6, 6)

        painter.restore()

        # --- hover highlight on the primary point of the pointed column ---
        if (
            self._show_primary
            and self._hover_day is not None
            and self._hover_day in self._prim_by_day
            and self._view_lo <= self._hover_day <= self._view_hi
        ):
            v = self._prim_by_day[self._hover_day]
            x, y = px(self._hover_day), py(v)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(_HOVER, 2))
            painter.drawEllipse(int(x) - 6, int(y) - 6, 12, 12)
            txt = ("%d" % round(v)) if self._y_integer else ("%.2f" % v)
            txt = "D%d: %s" % (self._hover_day, txt)
            tw = fm.horizontalAdvance(txt) + 8
            tx = min(int(x) + 8, left + width - tw)
            tx = max(tx, left + 2)
            painter.setPen(QPen(_AXIS))
            painter.drawText(tx, int(top) + fm.height(), txt)

        painter.end()

    @staticmethod
    def _draw_polyline(painter, pts, px, py) -> None:
        for i in range(1, len(pts)):
            d0, v0 = pts[i - 1]
            d1, v1 = pts[i]
            painter.drawLine(int(px(d0)), int(py(v0)), int(px(d1)), int(py(v1)))


class EvolutionGraph(QWidget):
    """A navigable Day chart: toolbar + canvas + sliding scrollbar."""

    daySelected = Signal(int)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        primary_name: str = "Corrected",
        secondary_name: str = "Forecast",
        show_curve_toggles: bool = False,
        export_basename: str = "graph",
    ) -> None:
        super().__init__(parent)
        self._height = _H_DEFAULT
        self._export_basename = export_basename

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        bar = QHBoxLayout()
        bar.setContentsMargins(2, 2, 2, 4)
        bar.setSpacing(4)

        def _btn(text: str, tip: str, slot) -> QToolButton:
            b = QToolButton()
            b.setText(text)
            # A visible bordered button (not auto-raise) so the graph controls
            # read as real buttons, like the HTML's .btn-icon.
            b.setObjectName("chartBtn")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        self._canvas = _GraphCanvas()
        self._canvas.daySelected.connect(self.daySelected)
        self._canvas.viewChanged.connect(self._sync_scrollbar)
        self._canvas.setFixedHeight(self._height)

        bar.addWidget(QLabel("Zoom"))
        bar.addWidget(_btn("-", "Zoom the Day axis out (or scroll the wheel down)",
                           self._canvas.zoom_x_out))
        bar.addWidget(_btn("+", "Zoom the Day axis in (or scroll the wheel up)",
                           self._canvas.zoom_x_in))
        bar.addSpacing(10)
        bar.addWidget(QLabel("Height"))
        bar.addWidget(_btn("−", "Make the graph shorter", self._shorter))
        bar.addWidget(_btn("+", "Make the graph taller", self._taller))
        bar.addSpacing(10)
        bar.addWidget(_btn("Reset", "Reset zoom, pan and height", self._reset))

        self._chk_primary: Optional[QCheckBox] = None
        self._chk_secondary: Optional[QCheckBox] = None
        if show_curve_toggles:
            bar.addSpacing(10)
            self._chk_primary = QCheckBox(primary_name)
            self._chk_primary.setChecked(True)
            self._chk_primary.setToolTip("Show or hide the %s curve." % primary_name.lower())
            self._chk_primary.toggled.connect(self._canvas.set_show_primary)
            bar.addWidget(self._chk_primary)
            self._chk_secondary = QCheckBox(secondary_name)
            self._chk_secondary.setChecked(True)
            self._chk_secondary.setToolTip("Show or hide the %s curve." % secondary_name.lower())
            self._chk_secondary.toggled.connect(self._canvas.set_show_secondary)
            bar.addWidget(self._chk_secondary)

        bar.addStretch(1)
        bar.addWidget(_btn("Export", "Save this graph as a PNG image "
                                     "(honours the curve checkboxes)", self._export))
        layout.addLayout(bar)
        layout.addWidget(self._canvas)

        # Sliding window: shown only when the Day axis is zoomed in.
        self._scroll = QScrollBar(Qt.Horizontal)
        self._scroll.setToolTip("Slide the visible Day window.")
        self._scroll.valueChanged.connect(self._on_scroll)
        self._scroll.setVisible(False)
        layout.addWidget(self._scroll)
        self._sb_guard = False

    # --- API delegated to the canvas ---------------------------------
    def set_data(self, *args, **kwargs) -> None:
        self._canvas.set_data(*args, **kwargs)

    def value_at(self, day: int) -> Optional[float]:
        return self._canvas.value_at(day)

    # --- scrollbar sync ----------------------------------------------
    def _sync_scrollbar(self) -> None:
        dmin, dmax, view_lo, view_hi, xzoom = self._canvas.view_state()
        if xzoom <= 1.0 or dmax <= dmin:
            self._scroll.setVisible(False)
            return
        span = view_hi - view_lo
        self._sb_guard = True
        self._scroll.setMinimum(int(round(dmin * _SB_UNITS)))
        self._scroll.setMaximum(int(round((dmax - span) * _SB_UNITS)))
        self._scroll.setPageStep(max(1, int(round(span * _SB_UNITS))))
        self._scroll.setSingleStep(_SB_UNITS)
        self._scroll.setValue(int(round(view_lo * _SB_UNITS)))
        self._sb_guard = False
        self._scroll.setVisible(True)

    def _on_scroll(self, value: int) -> None:
        if self._sb_guard:
            return
        self._canvas.set_view_lo(value / float(_SB_UNITS))

    # --- toolbar slots -----------------------------------------------
    def _taller(self) -> None:
        self._height = min(_H_MAX, self._height + _H_STEP)
        self._canvas.setFixedHeight(self._height)

    def _shorter(self) -> None:
        self._height = max(_H_MIN, self._height - _H_STEP)
        self._canvas.setFixedHeight(self._height)

    def _reset(self) -> None:
        self._height = _H_DEFAULT
        self._canvas.setFixedHeight(self._height)
        self._canvas.reset_view()

    def _export(self) -> None:
        pixmap = self._canvas.grab()
        path, _flt = get_save_file_name(
            self, "Export graph as image", self._export_basename + ".png",
            "PNG image (*.png);;JPEG image (*.jpg)",
        )
        if not path:
            return
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            path += ".png"
        pixmap.save(path)


class CollapsibleSection(QWidget):
    """A header that folds its content, showing the current value in bold."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        expanded: bool = True,
        on_toggle: Optional[Callable[[bool], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._content = content
        self._on_toggle = on_toggle
        self._value_text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QWidget()
        header.setObjectName("collapsibleHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(6)
        from PySide6.QtCore import QSize as _QSize
        self._arrow = QToolButton()
        self._arrow.setAutoRaise(True)
        self._arrow.setArrowType(Qt.NoArrow)
        self._arrow.setIconSize(_QSize(18, 18))
        self._arrow.clicked.connect(self._toggle)
        hl.addWidget(self._arrow)
        self._label = QLabel()
        self._label.setTextFormat(Qt.RichText)
        font = QFont()
        font.setBold(True)
        self._label.setFont(font)
        hl.addWidget(self._label, 1)
        layout.addWidget(header)
        layout.addWidget(self._content)

        self._expanded = expanded
        self._content.setVisible(expanded)
        self._refresh_arrow()
        self._refresh_label()

    def _refresh_arrow(self) -> None:
        """Paint a large chevron (sized to the title text) in the theme colour."""
        from PySide6.QtGui import QIcon, QPalette

        from .collapsible_card import _chevron_pixmap
        color = self.palette().color(QPalette.WindowText)
        self._arrow.setIcon(QIcon(_chevron_pixmap(self._expanded, color, size=18)))

    def _toggle(self) -> None:
        self.set_expanded(not self._expanded, notify=True)

    def set_expanded(self, expanded: bool, notify: bool = False) -> None:
        self._expanded = bool(expanded)
        self._content.setVisible(self._expanded)
        self._refresh_arrow()
        if notify and self._on_toggle is not None:
            self._on_toggle(self._expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_value_text(self, text: str) -> None:
        """Set the current-value shown (in bold) in the header."""
        self._value_text = text
        self._refresh_label()

    def _refresh_label(self) -> None:
        if self._value_text:
            # The current value is highlighted in blue (indigo accent) so it
            # reads apart from the title at a glance.
            self._label.setText(
                "%s [currently <b style='color:#4F46E5;'>%s</b>]"
                % (self._title, self._value_text)
            )
        else:
            self._label.setText(self._title)
