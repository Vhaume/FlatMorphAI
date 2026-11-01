"""Mouse pointers drawn by the application, at the size the operator chose.

Why the application draws them
------------------------------
The pointer size of an X11 application is normally the cursor THEME's business:
the theme holds one bitmap per shape and per size, and ``XCURSOR_SIZE`` picks
which one. That is what this application tried first, and it changed nothing,
for a simple reason: the WSL distribution has no cursor theme installed at all
(``/usr/share/icons`` holds only ``hicolor`` and ``locolor``, neither with a
``cursors/`` folder). With no theme file to read, Qt falls back to its own
built-in cursor bitmaps, which are a fixed size and ignore ``XCURSOR_SIZE``
entirely. No value, and no restart, could ever have changed anything.

So the shapes are drawn here instead. A cursor built from a pixmap is handed
to Qt whole, so it needs no theme, no font and no system support -- and it can
be rebuilt at any moment, which is why the size now applies immediately rather
than at the next start-up.

What is preserved
-----------------
The SHAPES keep their meaning and keep changing as before: the viewers still
switch to an open hand when panning is possible and to a closed hand while
dragging, text fields still show a beam. Only their drawing is ours, and it
scales. ``shape_of`` maps a Qt shape to the drawing, and anything not drawn
here falls back to the Qt shape, unscaled, rather than to nothing.

Everything is pure Qt drawing (QPainterPath on a transparent QImage), so the
pointer stays crisp at any size instead of being a blown-up 16-pixel bitmap.
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import (QColor, QCursor, QImage, QPainter, QPainterPath,
                           QPen, QPixmap)

#: Size in pixels of the square the pointer is drawn in.
DEFAULT_SIZE = 48
MIN_SIZE = 16
MAX_SIZE = 128

_WHITE = QColor(255, 255, 255)
_BLACK = QColor(0, 0, 0)

#: Default body colour. White with a black outline is what a pointer has been
#: since pointers existed, and it is the only pair that stays readable on a
#: dark page as on a light one.
DEFAULT_COLOR = "#FFFFFF"

_cache: Dict[tuple, QCursor] = {}
#: ``enabled`` false means the application draws nothing and every widget keeps
#: the pointer Qt and the system give it. That is the default, because the
#: drawn pointers are consistent inside the window but differ from the one
#: Windows shows on the frame and everywhere else on the desktop -- two
#: pointers for one session reads as a bug, not as a feature.
_state = {"size": DEFAULT_SIZE, "color": DEFAULT_COLOR, "enabled": False}


def _body_color() -> QColor:
    c = QColor(str(_state.get("color") or DEFAULT_COLOR))
    return c if c.isValid() else _WHITE


def _outline_for(body: QColor) -> QColor:
    """Black outline on a light body, white on a dark one.

    Relative luminance of ITU-R BT.709 (the coefficients sRGB uses), so a
    saturated yellow counts as light and a saturated blue as dark -- which a
    plain average of the three channels gets wrong.
    """
    lum = (0.2126 * body.redF() + 0.7152 * body.greenF() + 0.0722 * body.blueF())
    return _BLACK if lum > 0.45 else _WHITE


# --------------------------------------------------------------------- shapes
def _arrow(p: QPainter, s: float) -> tuple:
    """The classic pointer. Hotspot at the tip, i.e. (0, 0)."""
    path = QPainterPath()
    pts = [(0.03, 0.03), (0.03, 0.74), (0.22, 0.56), (0.34, 0.86),
           (0.47, 0.80), (0.35, 0.51), (0.58, 0.50)]
    path.moveTo(pts[0][0] * s, pts[0][1] * s)
    for x, y in pts[1:]:
        path.lineTo(x * s, y * s)
    path.closeSubpath()
    p.drawPath(path)
    return (int(0.03 * s), int(0.03 * s))


def _beam(p: QPainter, s: float) -> tuple:
    """Text beam: a vertical bar with the two serifs. Hotspot at its centre."""
    w = max(1.0, s * 0.055)
    cx, top, bot = s * 0.5, s * 0.14, s * 0.86
    path = QPainterPath()
    path.addRect(cx - w / 2, top, w, bot - top)
    arm = s * 0.16
    for y in (top, bot - w):
        path.addRect(cx - arm, y, arm * 2, w)
    p.drawPath(path)
    return (int(cx), int(s * 0.5))


def _cross(p: QPainter, s: float) -> tuple:
    w = max(1.0, s * 0.055)
    c = s * 0.5
    path = QPainterPath()
    path.addRect(c - w / 2, s * 0.08, w, s * 0.84)
    path.addRect(s * 0.08, c - w / 2, s * 0.84, w)
    p.drawPath(path)
    return (int(c), int(c))


def _hand(p: QPainter, s: float, closed: bool) -> tuple:
    """Open or closed hand, as the viewers use while panning.

    Drawn as a palm plus four fingers and a thumb; closing the hand folds the
    fingers down into the palm instead of changing the silhouette, so the two
    states read as the same hand.
    """
    path = QPainterPath()
    palm_top = s * (0.46 if closed else 0.40)
    path.addRoundedRect(s * 0.22, palm_top, s * 0.56, s * (0.40 if closed else 0.44),
                        s * 0.14, s * 0.14)
    fw = s * 0.115                      # finger width
    tops = (s * 0.14, s * 0.09, s * 0.12, s * 0.20) if not closed else \
           (s * 0.34, s * 0.31, s * 0.33, s * 0.37)
    for i, top in enumerate(tops):
        x = s * 0.255 + i * s * 0.135
        path.addRoundedRect(x, top, fw, palm_top + s * 0.10 - top,
                            fw / 2.0, fw / 2.0)
    # thumb, on the left, angled
    path.addRoundedRect(s * 0.10, s * (0.52 if closed else 0.47), fw * 1.15,
                        s * 0.26, fw / 2.0, fw / 2.0)
    p.drawPath(path.simplified())
    return (int(s * 0.5), int(s * 0.5))


def _double_arrow(p: QPainter, s: float, angle: float) -> tuple:
    """Double-headed arrow used by every resize shape, drawn at ``angle``.

    One drawing covers the four of them -- horizontal, vertical and the two
    diagonals -- by rotating it, which keeps the four exactly consistent.
    """
    p.save()
    p.translate(s * 0.5, s * 0.5)
    p.rotate(angle)
    L = s * 0.36            # half length
    bar = s * 0.075         # shaft thickness
    head = s * 0.17         # arrow head half-height
    path = QPainterPath()
    path.addRect(-L + head, -bar / 2, 2 * (L - head), bar)
    for sign in (-1.0, 1.0):
        tip = sign * L
        path.moveTo(tip, 0.0)
        path.lineTo(tip - sign * head, -head)
        path.lineTo(tip - sign * head, head)
        path.closeSubpath()
    p.drawPath(path.simplified())
    p.restore()
    return (int(s * 0.5), int(s * 0.5))


def _size_all(p: QPainter, s: float) -> tuple:
    _double_arrow(p, s, 0.0)
    _double_arrow(p, s, 90.0)
    return (int(s * 0.5), int(s * 0.5))


def _split(p: QPainter, s: float, vertical: bool) -> tuple:
    """Splitter handle: the double arrow with the separating line."""
    _double_arrow(p, s, 90.0 if vertical else 0.0)
    path = QPainterPath()
    t = max(1.0, s * 0.055)
    if vertical:
        path.addRect(s * 0.12, s * 0.5 - t / 2, s * 0.76, t)
    else:
        path.addRect(s * 0.5 - t / 2, s * 0.12, t, s * 0.76)
    p.drawPath(path)
    return (int(s * 0.5), int(s * 0.5))


def _up_arrow(p: QPainter, s: float) -> tuple:
    path = QPainterPath()
    path.moveTo(s * 0.5, s * 0.08)
    path.lineTo(s * 0.74, s * 0.44)
    path.lineTo(s * 0.58, s * 0.44)
    path.lineTo(s * 0.58, s * 0.92)
    path.lineTo(s * 0.42, s * 0.92)
    path.lineTo(s * 0.42, s * 0.44)
    path.lineTo(s * 0.26, s * 0.44)
    path.closeSubpath()
    p.drawPath(path)
    return (int(s * 0.5), int(s * 0.08))


def _forbidden(p: QPainter, s: float) -> tuple:
    path = QPainterPath()
    path.addEllipse(QPointF(s * 0.5, s * 0.5), s * 0.36, s * 0.36)
    path.addEllipse(QPointF(s * 0.5, s * 0.5), s * 0.24, s * 0.24)
    p.drawPath(path)
    bar = QPainterPath()
    bar.addRect(s * 0.16, s * 0.44, s * 0.68, s * 0.12)
    p.drawPath(bar)
    return (int(s * 0.5), int(s * 0.5))


def _wait(p: QPainter, s: float) -> tuple:
    path = QPainterPath()
    path.addEllipse(QPointF(s * 0.5, s * 0.5), s * 0.34, s * 0.34)
    p.drawPath(path)
    inner = QPainterPath()
    inner.moveTo(s * 0.5, s * 0.5)
    inner.lineTo(s * 0.5, s * 0.22)
    inner.lineTo(s * 0.72, s * 0.5)
    inner.closeSubpath()
    p.drawPath(inner)
    return (int(s * 0.5), int(s * 0.5))


_DRAW = {
    Qt.ArrowCursor: lambda p, s: _arrow(p, s),
    Qt.IBeamCursor: lambda p, s: _beam(p, s),
    Qt.CrossCursor: lambda p, s: _cross(p, s),
    Qt.OpenHandCursor: lambda p, s: _hand(p, s, False),
    Qt.ClosedHandCursor: lambda p, s: _hand(p, s, True),
    Qt.PointingHandCursor: lambda p, s: _hand(p, s, False),
    Qt.WaitCursor: lambda p, s: _wait(p, s),
    Qt.BusyCursor: lambda p, s: _wait(p, s),
    # Resize and split shapes. These are the ones Qt sets ITSELF, on text
    # fields, splitter handles and the edges of sub-windows, which is why they
    # stayed small while the arrow and the hands followed the setting: they
    # never pass through the application's own code.
    Qt.SizeHorCursor: lambda p, s: _double_arrow(p, s, 0.0),
    Qt.SizeVerCursor: lambda p, s: _double_arrow(p, s, 90.0),
    Qt.SizeBDiagCursor: lambda p, s: _double_arrow(p, s, -45.0),
    Qt.SizeFDiagCursor: lambda p, s: _double_arrow(p, s, 45.0),
    Qt.SizeAllCursor: lambda p, s: _size_all(p, s),
    Qt.SplitHCursor: lambda p, s: _split(p, s, False),
    Qt.SplitVCursor: lambda p, s: _split(p, s, True),
    Qt.UpArrowCursor: lambda p, s: _up_arrow(p, s),
    Qt.ForbiddenCursor: lambda p, s: _forbidden(p, s),
}


def image(shape, size: int, color: Optional[str] = None) -> Optional[QImage]:
    """The pointer as a transparent image, or None for a shape we do not draw.

    Separated from :func:`cursor` so the drawing can be rendered and inspected
    without a running application (a QImage needs no QGuiApplication).
    """
    fn = _DRAW.get(shape)
    if fn is None:
        return None
    s = float(max(MIN_SIZE, min(MAX_SIZE, int(size))))
    img = QImage(int(s), int(s), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    try:
        p.setRenderHint(QPainter.Antialiasing, True)
        # A body colour and its contrasting outline: a single-colour pointer
        # disappears on a page of its own colour, whatever that colour is.
        body = QColor(color) if color else _body_color()
        if not body.isValid():
            body = _WHITE
        p.setPen(QPen(_outline_for(body), max(1.0, s * 0.035)))
        p.setBrush(body)
        hot = fn(p, s)
    finally:
        p.end()
    img.setOffset(img.offset())
    img.setText("hotx", str(hot[0]))
    img.setText("hoty", str(hot[1]))
    return img


def enabled() -> bool:
    return bool(_state.get("enabled"))


def last_error() -> str:
    """Why the last apply() did nothing visible, or "" when it worked."""
    return str(_state.get("error") or "")


def cursor(shape) -> QCursor:
    """A QCursor for ``shape`` at the configured size.

    Returns the plain Qt shape when the drawing is off, and also when the
    shape is not one we draw, so a caller never ends up with no cursor at all.
    """
    if not _state.get("enabled"):
        return QCursor(shape)
    size = int(_state["size"])
    key = (shape, size, str(_state.get("color")))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    img = image(shape, size)
    if img is None:
        cur = QCursor(shape)
    else:
        try:
            hx, hy = int(img.text("hotx") or 0), int(img.text("hoty") or 0)
            cur = QCursor(QPixmap.fromImage(img), hx, hy)
        except Exception:  # noqa: BLE001 - never leave a widget cursorless
            cur = QCursor(shape)
    _cache[key] = cur
    return cur


def size() -> int:
    return int(_state["size"])


def color() -> str:
    return str(_state.get("color") or DEFAULT_COLOR)


def restore_native() -> None:
    """Give every widget we touched its Qt shape back, and stop watching.

    Called when the operator switches back to the system pointer: the shape
    each widget asked for was remembered, so handing back ``QCursor(shape)``
    restores exactly what Qt would have set on its own.
    """
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return
        w_obj = _watcher.pop("obj", None)
        if w_obj is not None:
            try:
                app.removeEventFilter(w_obj)
            except Exception:  # noqa: BLE001
                pass
        _watcher["obj"] = None
        for w in app.allWidgets():
            try:
                shape = getattr(w, "_flatmorphai_cursor_shape", None)
                if shape is not None:
                    w._flatmorphai_cursor_shape = None      # noqa: SLF001
                    w.setCursor(QCursor(shape))
            except Exception:  # noqa: BLE001
                pass
        for w in app.topLevelWidgets():
            try:
                w.unsetCursor()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    _cache.clear()


def apply(new_size: Optional[int] = None, new_color: Optional[str] = None,
          on: Optional[bool] = None) -> None:
    """Set the pointer size and/or colour and refresh what is already shown.

    Either argument may be left out, which keeps the current value. Immediate:
    every widget carrying one of our cursors is handed the new drawing, and
    the top-level windows get the new arrow, which every child that does not
    set its own cursor inherits. No restart.
    """
    if new_size is not None:
        try:
            _state["size"] = max(MIN_SIZE, min(MAX_SIZE, int(new_size)))
        except (TypeError, ValueError):
            pass
    if new_color is not None:
        c = QColor(str(new_color))
        if c.isValid():
            _state["color"] = c.name()
    if on is not None:
        was = bool(_state.get("enabled"))
        _state["enabled"] = bool(on)
        if was and not on:
            restore_native()
            return
    if not _state.get("enabled"):
        return
    _cache.clear()
    _state["error"] = ""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            _state["error"] = "no QApplication"
            return
        arrow = cursor(Qt.ArrowCursor)
        if arrow.shape() != Qt.BitmapCursor:
            # The drawing did not survive the conversion to a cursor: the X
            # server may refuse a cursor of this size, or refuse ARGB cursors
            # altogether. Say so rather than appear to do nothing.
            _state["error"] = ("the display refused a %d px drawn cursor"
                               % int(_state["size"]))
        for w in app.topLevelWidgets():
            try:
                w.setCursor(arrow)
            except Exception:  # noqa: BLE001
                pass
        # Every widget already built: the ones we tagged keep their shape at
        # the new size, and the ones Qt gave a shape of its own (text fields,
        # splitter handles) are converted now -- the filter below only sees
        # the changes that come after it is installed.
        for w in app.allWidgets():
            try:
                want = getattr(w, "_flatmorphai_cursor_shape", None)
                if want is None:
                    sh = w.cursor().shape()
                    want = sh if sh in _DRAW else None
                if want is not None:
                    set_widget_cursor(w, want)
            except Exception:  # noqa: BLE001
                pass
        # Installed on the first call, size change or not: it is what covers
        # the pages built later, and the pointer entering a widget for the
        # first time.
        if _watcher["obj"] is None:
            try:
                _watcher["obj"] = _Watcher(app)
                app.installEventFilter(_watcher["obj"])
            except Exception as exc:  # noqa: BLE001
                _watcher["obj"] = None
                _state["error"] = "%s: %s" % (type(exc).__name__, exc)
        # Qt only re-reads the cursor under the pointer when something makes
        # it look again. Without this the new pointer appears only after the
        # mouse has moved out of the widget it is sitting on, which reads as
        # "nothing happened" when the change is made from a combo box.
        try:
            app.setOverrideCursor(arrow)
            app.restoreOverrideCursor()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        _state["error"] = "%s: %s" % (type(exc).__name__, exc)
    if _state.get("error"):
        import sys as _sys
        _sys.__stderr__.write("[cursors] %s\n" % _state["error"])


class _Watcher(QObject):
    """Replace, as it happens, any cursor Qt sets on a widget by ours.

    A text field gives itself a beam, a splitter handle a double arrow, a
    scroll area an arrow -- from inside Qt, never through this application.
    Those cursors therefore kept the built-in size whatever the setting said.
    Qt announces each of them with a CursorChange event, which is what this
    filter listens to: it reads the shape the widget just took, and if it is
    one we draw, hands back our drawing of it at the chosen size. The original
    shape is remembered on the widget so a later size change can redraw it.
    """

    #: Two moments are watched, because one of them is not enough.
    #: CursorChange catches a widget that gives itself a shape WHILE the
    #: filter is already installed. Enter catches everything else: a page
    #: built before the filter existed, or built lazily the first time it is
    #: opened -- the pointer entering the widget is the last possible moment,
    #: and it is also the first time the shape is actually seen. Enter fires
    #: often, so the test is kept to a dictionary lookup, and a widget already
    #: converted carries a bitmap cursor, whose shape is not in the table.
    _TYPES = (QEvent.CursorChange, QEvent.Enter)

    def eventFilter(self, obj, event):      # noqa: N802 - Qt naming
        try:
            if event.type() in self._TYPES and obj.isWidgetType():
                if not getattr(obj, "_flatmorphai_cursor_busy", False):
                    sh = obj.cursor().shape()
                    if sh in _DRAW:
                        obj._flatmorphai_cursor_busy = True      # noqa: SLF001
                        try:
                            set_widget_cursor(obj, sh)
                        finally:
                            obj._flatmorphai_cursor_busy = False  # noqa: SLF001
        except Exception:  # noqa: BLE001 - a filter must never raise
            pass
        return False


_watcher = {"obj": None}


def set_widget_cursor(widget, shape) -> None:
    """Give ``widget`` our drawing of ``shape`` and remember the choice.

    The shape is stored on the widget so :func:`apply` can redraw it at a new
    size without the widget having to listen for anything.
    """
    try:
        widget._flatmorphai_cursor_shape = shape      # noqa: SLF001
        widget.setCursor(cursor(shape))
    except Exception:  # noqa: BLE001
        pass
