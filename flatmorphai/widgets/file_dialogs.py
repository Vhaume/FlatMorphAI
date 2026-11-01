"""
Comfortably sized file/directory pickers.

On WSLg (and any platform without a native file chooser) Qt falls back to its
own dialog, which opens at a cramped default size: the "Look in" row, the side
bar and the name column are all squeezed and long names are clipped. These
helpers build the same dialog explicitly, force the non-native widget (so the
look is identical on every machine), and give it a generous initial and
minimum size so nothing is deformed.

They return exactly what the ``QFileDialog`` static helpers return, so call
sites can swap ``QFileDialog.getExistingDirectory(...)`` for
``get_existing_directory(...)`` with no other change.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QSize
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QListView,
    QMessageBox,
    QTreeView,
    QWidget,
)


class CenteredMessageBox(QMessageBox):
    """A QMessageBox that opens centred on its parent window.

    Qt places a message box at the top-left of the screen on some platforms
    (notably WSLg) instead of over the application. Re-centring on first show
    keeps confirmation prompts in the middle of the program where the user is
    looking, per Nielsen's "visibility of system status" heuristic.
    """

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # Centre now, then once more after the event loop has given the box its
        # final size (the first showEvent can fire before the layout settles,
        # and some window managers -- notably WSLg -- reposition the dialog
        # right after it is shown), so it always lands in the middle.
        self._recenter()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._recenter)

    def _recenter(self):
        try:
            screen = None
            parent = self.parentWidget()
            if parent is not None and parent.window().screen() is not None:
                screen = parent.window().screen()
            if screen is None:
                screen = QGuiApplication.screenAt(QCursor.pos())
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()
            frame = self.frameGeometry()
            x = avail.x() + (avail.width() - frame.width()) // 2
            y = avail.y() + (avail.height() - frame.height()) // 2
            self.move(x, y)
        except Exception:  # noqa: BLE001
            pass


def _message(parent, icon, title, text, buttons, default):
    box = CenteredMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(buttons)
    box.setDefaultButton(default)
    return box.exec()


def question(parent, title, text,
             buttons=QMessageBox.Yes | QMessageBox.No,
             default=QMessageBox.No):
    """Ask a Yes/No question, centred on the parent window."""
    return _message(parent, QMessageBox.Question, title, text, buttons, default)


def critical(parent, title, text,
             buttons=QMessageBox.Ok, default=QMessageBox.Ok):
    """Show an error, centred on the parent window."""
    return _message(parent, QMessageBox.Critical, title, text, buttons, default)


def warning(parent, title, text,
            buttons=QMessageBox.Ok, default=QMessageBox.Ok):
    """Show a warning, centred on the parent window."""
    return _message(parent, QMessageBox.Warning, title, text, buttons, default)


def information(parent, title, text,
                buttons=QMessageBox.Ok, default=QMessageBox.Ok):
    """Show an info message, centred on the parent window."""
    return _message(parent, QMessageBox.Information, title, text, buttons, default)


def _preferred_size() -> tuple:
    """Dialog size scaled to the screen so it is comfortable at any UI scale.

    At a large UI font scale a fixed 960x640 dialog becomes cramped (the path
    row, side bar and columns get squeezed). Sizing to ~72% of the available
    screen - clamped to a sane range - keeps it usable whatever the scale.
    """
    try:
        avail = QGuiApplication.primaryScreen().availableGeometry()
        w = int(max(840, min(avail.width() * 0.72, 1500)))
        h = int(max(560, min(avail.height() * 0.78, 980)))
        return w, h
    except Exception:  # noqa: BLE001
        return 1000, 680


def _prepare(dialog: QFileDialog, caption: str, directory: str) -> None:
    dialog.setWindowTitle(caption or dialog.windowTitle())
    # Force Qt's own widget: consistent sizing and behaviour on every host,
    # and the only reliable one under WSLg.
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    if directory:
        dialog.setDirectory(directory)
    w, h = _preferred_size()
    dialog.setMinimumSize(int(w * 0.8), int(h * 0.8))
    dialog.resize(w, h)
    _enlarge_icons(dialog)


def _enlarge_icons(dialog: QFileDialog) -> None:
    """Make the icons and rows inside the dialog bigger.

    Sizing the dialog up alone leaves the internal views (the side bar, the
    file list and the detail tree) with Qt's tiny default 16px icons, so the
    extra room reads as empty. Scale the icons to the dialog font and give the
    rows matching height so names and folders are comfortable to read and
    click at any UI scale.
    """
    try:
        px = dialog.fontMetrics().height()
        icon = QSize(max(24, int(px * 1.6)), max(24, int(px * 1.6)))
        for view in dialog.findChildren(QAbstractItemView):
            view.setIconSize(icon)
            if isinstance(view, QListView):
                view.setGridSize(QSize())  # let rows size to the icon
            if isinstance(view, QTreeView):
                try:
                    view.header().setMinimumSectionSize(max(80, int(px * 6)))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass


def get_existing_directory(
    parent: Optional[QWidget],
    caption: str = "Select directory",
    directory: str = "",
) -> str:
    """Pick a directory. Returns the path, or "" when cancelled."""
    dialog = QFileDialog(parent, caption, directory)
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    _prepare(dialog, caption, directory)
    if dialog.exec() == QFileDialog.Accepted:
        files = dialog.selectedFiles()
        return files[0] if files else ""
    return ""


def get_save_file_name(
    parent: Optional[QWidget],
    caption: str = "Save file",
    directory: str = "",
    filter: str = "",  # noqa: A002 - mirror QFileDialog's parameter name
) -> Tuple[str, str]:
    """Pick a save path. Returns ``(path, selected_filter)``; "" when cancelled."""
    dialog = QFileDialog(parent, caption, directory)
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    if filter:
        dialog.setNameFilter(filter)
    _prepare(dialog, caption, directory)
    if dialog.exec() == QFileDialog.Accepted:
        files = dialog.selectedFiles()
        return (files[0] if files else "", dialog.selectedNameFilter())
    return "", ""


def get_open_file_name(
    parent: Optional[QWidget],
    caption: str = "Open file",
    directory: str = "",
    filter: str = "",  # noqa: A002 - mirror QFileDialog's parameter name
) -> Tuple[str, str]:
    """Pick one existing file. Returns ``(path, selected_filter)``.

    Mirrors ``QFileDialog.getOpenFileName`` so the tuple unpacking at call
    sites keeps working; both items are "" when cancelled.
    """
    dialog = QFileDialog(parent, caption, directory)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    if filter:
        dialog.setNameFilter(filter)
    _prepare(dialog, caption, directory)
    if dialog.exec() == QFileDialog.Accepted:
        files = dialog.selectedFiles()
        return (files[0] if files else "", dialog.selectedNameFilter())
    return "", ""
