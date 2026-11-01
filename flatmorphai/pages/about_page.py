"""
About page: application description, authorship, funding and contact.

The page is a plain :class:`QWidget` with a vertical layout that
presents, in order:

- An application header (name, version, parent project PlanarIA).
- A short description of FlatMorphAI.
- A "Contact the authors" block with an email address and a clickable
  link to the BTSB team website.
- A "Rights and licence" block that states ownership and usage terms.
- A "Download and publication" block pointing at the release page
  and at the associated paper (when available).
- A "Funding" block crediting the Region Occitanie and INU
  Champollion.

Design rationale
----------------
A dedicated About page is a long-standing convention for scientific
desktop tools: it anchors provenance, citation metadata and funding
attribution in one place, next to the software the user is running.
The structure of the page follows recommendations from:

- The Research Software Citation Principles, Smith, A. M. et al.,
  PeerJ Computer Science, 2016, DOI: 10.7717/peerj-cs.86, which ask
  for clear author, version and licence information to be shipped
  with the software.
- The FORCE11 Software Citation Implementation Working Group report,
  Katz, D. S. et al., PeerJ Computer Science, 2019, DOI:
  10.7717/peerj-cs.205, which motivates a persistent "how to cite"
  section in the UI.

No external network calls are made: every link is rendered as clickable
rich text that hands off to the system browser through
:class:`QDesktopServices`.
"""

from __future__ import annotations

import re
import shutil
import sys
from typing import List, Optional

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __app_name__, __version__
from ..widgets.resize_grip import VerticalResizeGrip

# Status colours of the Dependencies card. Chosen to stay readable on both
# the light and the dark stylesheet.
_MISSING_COLOR = "#d93025"
_DEP_TABLE_HEIGHT_KEY = "about_dep_table_height"
_DEP_TABLE_MIN_HEIGHT = 120
_OK_COLOR = "#1a7f37"
#: Amber for a calc-sensitive package that differs only at the patch/build level
#: (usually safe), vs the red _MISSING_COLOR used for a major/minor mismatch.
_WARN_COLOR = "#b5820a"


def _dep_message(parent, title: str, text: str) -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(title)
    box.setText(text)
    box.exec()


# conda and pip redraw progress in place: an ANSI sequence to move the cursor,
# a backspace run, then one spinner frame. Piped into a QProcess there is no
# terminal to interpret any of it, so the raw output piles up as "[A[A[A" rows
# and lone "- \\ | /" characters. The helpers below replay the control codes and
# then drop whatever carries no information.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][B0]")
#: A run of spinner frames standing alone as a token: the ASCII wheel used by
#: conda and the braille dots used by some pip front-ends. The look-around
#: keeps hyphens and slashes that sit inside a word (a wheel filename, a path).
_SPINNER_RUN_RE = re.compile(r"(?<![^\s])[-\\|/\u2800-\u28ff]+(?![^\s])")


def _apply_backspaces(text: str) -> str:
    """Replay ``\\b`` as a terminal would: each one deletes the char before it."""
    out = []
    for ch in text:
        if ch == "\x08":
            if out:
                out.pop()
        else:
            out.append(ch)
    return "".join(out)


def _clean_console_line(raw: str) -> "Optional[str]":
    """Return a printable line, or None when it is only progress-redraw noise.

    A line rewritten in place keeps its last ``\\r`` segment, escape sequences
    and backspaces are replayed, then standalone spinner frames are removed.
    A line left empty, or left with only the label of a progress message
    ("Retrieving notices:"), is dropped; indentation and real content such as
    "Successfully installed skan-0.13.1" come through untouched.
    """
    text = raw.split("\r")[-1]
    text = _ANSI_RE.sub("", text)
    text = _apply_backspaces(text).replace("\x07", "")
    text = _SPINNER_RUN_RE.sub("", text).rstrip()
    if not text.strip() or text.strip().endswith(":"):
        return None
    return text


def _resolve_install_argv(command: str) -> "Optional[List[str]]":
    """Turn a displayed command into the argv actually executed.

    ``pip install X`` becomes ``<running interpreter> -m pip install X`` so the
    package lands in the environment the application runs from, not in whatever
    Python happens to come first on PATH. ``conda install ...`` keeps its own
    executable (resolved on PATH) and gains ``-y`` so it does not block waiting
    for a confirmation nobody can type. Returns None when conda is unavailable.
    """
    parts = command.split()
    if not parts:
        return None
    if parts[0] == "pip":
        return [sys.executable, "-m", "pip", *parts[1:]]
    if parts[0] == "conda":
        exe = shutil.which("conda")
        if not exe:
            return None
        rest = parts[1:]
        if "-y" not in rest and "--yes" not in rest:
            rest = [rest[0], "-y", *rest[1:]] if rest else rest
        return [exe, *rest]
    return None


class _InstallDialog(QDialog):
    """Run an install command with QProcess and stream its output live.

    A QProcess keeps the event loop running, so the window stays responsive
    and the operator sees pip/conda progress instead of a frozen application.
    """

    def __init__(self, argv: "List[str]", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Installing packages")
        self.setMinimumSize(760, 420)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(" ".join(argv)))
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(QFont("Consolas" if sys.platform.startswith("win")
                                else "Monospace"))
        v.addWidget(self._out, 1)
        self._status = QLabel("Running...")
        v.addWidget(self._status)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.Close).setEnabled(False)
        v.addWidget(self._buttons)

        self._buffer = ""
        self._last_line = ""
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(argv[0], argv[1:])

    def _on_output(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        self._buffer += data
        # Keep an unterminated tail in the buffer: a progress line is rewritten
        # with \r and would otherwise be appended half-drawn.
        *lines, self._buffer = self._buffer.replace("\r\n", "\n").split("\n")
        for raw in lines:
            text = _clean_console_line(raw)
            # conda repeats the same status line once per spinner frame.
            if text is not None and text != self._last_line:
                self._last_line = text
                self._out.appendPlainText(text)

    def _on_finished(self, code: int, _status) -> None:
        if code == 0:
            self._status.setText(
                "Done. Restart FlatMorphAI so the new packages are imported.")
        else:
            self._status.setText(
                "The command failed (exit code %d). Read the output above; "
                "you may need to run it yourself with the right permissions."
                % code)
        self._buttons.button(QDialogButtonBox.Close).setEnabled(True)

    def reject(self) -> None:  # noqa: D102
        if self._proc.state() != QProcess.NotRunning:
            return
        super().reject()


# ---------------------------------------------------------------------
# Static metadata. Update here when the team, funding or release URL
# changes; the page reads these at construction time.
# ---------------------------------------------------------------------
APP_TAGLINE = "Part of the PlanarIA project"
APP_DESCRIPTION = (
    "FlatMorphAI is a PySide6 dashboard that orchestrates the full "
    "planarian video-to-stereotype pipeline: frame extraction, "
    "semantic segmentation with Mask2Former, stereotype "
    "classification with a LightGBM classifier built on shape and "
    "motion descriptors, manual review in the Stereotype Editor and "
    "period fusion. It is designed to let biologists reproduce and "
    "audit the analysis end to end on a single workstation, with "
    "strict determinism (random_state=42) and GPU-aware model caching."
)

CONTACT_EMAIL = "contact@btsb-albi.fr"
TEAM_WEBSITE_URL = "https://btsb-albi.fr/home/"
TEAM_WEBSITE_LABEL = "BTSB team website (btsb-albi.fr)"

# Author and developer of FlatMorphAI: the application, the descriptor
# chains and the trained models.
AUTHOR_NAME = "Valentin PIERRAT"
AUTHOR_EMAIL = "valentin.p02@gmail.com"

# Project manager for the PlanarIA / FlatMorphAI work.
MANAGER_NAME = "Caroline VIGNET"
MANAGER_EMAIL = "caroline.vignet@univ-jfc.fr"

LICENCE_NAME = "CC BY-NC 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"

#: The licence covers the code, the trained weights and the reference data
#: alike. For the weights it is not only a choice: the segmentation model is
#: fine-tuned from the Mask2Former checkpoint
#: facebook/mask2former-swin-tiny-ade-semantic, whose weights Meta publishes
#: under CC BY-NC 4.0, so the non-commercial restriction is inherited and
#: follows every derivative. CC BY-NC 4.0 requires that a modification be
#: indicated, which is why the fine-tuning is stated here and not only in the
#: repository NOTICE.
RIGHTS_STATEMENT = (
    "Licence: Creative Commons Attribution-NonCommercial 4.0 International "
    "(CC BY-NC 4.0).\n\n"
    "Copyright 2026, PlanarIA project, BTSB (Biologie des Temps "
    "et des Systemes Biologiques), INU Champollion, Albi, France. "
    "Author and developer: Valentin PIERRAT.\n\n"
    "You may use, copy, modify and redistribute this software and its models, "
    "including for research and teaching, provided that you credit the "
    "authors, state the licence, indicate any change you made, and do NOT use "
    "the material for commercial purposes. The software is provided as is, "
    "without warranty of any kind, and the authors accept no liability for any "
    "damage arising from its use.\n\n"
    "This is deliberately a non-commercial licence, not an open-source licence "
    "in the sense of the Open Source Initiative.\n\n"
    "Third-party components keep their own terms. Two of the models derive "
    "from pre-trained weights that Meta AI publishes for non-commercial use "
    "only, and both were modified by fine-tuning: the segmentation model, "
    "from the Mask2Former checkpoint facebook/mask2former-swin-tiny-ade-semantic "
    "(Cheng et al., arXiv:2112.01527; weights under CC BY-NC 4.0), and the "
    "regeneration state classifier, from the ConvNeXt V2 backbone "
    "convnextv2_base (Woo et al., arXiv:2301.00808; ImageNet weights under "
    "CC BY-NC). A commercial use of either would therefore require retraining "
    "it from a permissively licensed backbone. The Mask2Former and ConvNeXt V2 "
    "source code, distinct from their weights, is MIT. The U-Net encoders come "
    "from timm (timm-efficientnet-b0), and two GPU routines reproduce "
    "scikit-image code under BSD-3-Clause. See the NOTICE file of the "
    "repository for the full list."
)

DOWNLOAD_URL = "https://btsb-albi.fr/home/"   # replace on release
DOWNLOAD_LABEL = "Download the latest release"
PAPER_URL = "https://btsb-albi.fr/home/"      # replace when the paper is online
PAPER_LABEL = "Associated paper (PlanarIA)"

FUNDING_STATEMENT = (
    "This work was funded by the Region Occitanie (France) and by "
    "INU Champollion (Institut National Universitaire Jean-Francois "
    "Champollion, Albi). The funders had no role in the software "
    "design or in the decision to publish."
)


class AboutPage(QWidget):
    """Read-only informational page shown from the sidebar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aboutPage")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(16)

        outer.addWidget(self._build_header())

        # A scroll area keeps the page usable at small window sizes and
        # at high UI scale factors (the Ctrl+wheel zoom easily pushes
        # the content past the viewport).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        # Each section is a collapsible card titled with its group title,
        # matching the rest of the app; the expanded state is persisted.
        from ..config import AppSettings
        from ..widgets import CollapsibleCard
        settings = AppSettings()
        self._settings = settings
        sections = [
            ("about_sec_description", self._build_description_group),
            ("about_sec_dependencies", self._build_dependencies_group),
            ("about_sec_install_check", self._build_install_check_group),
            ("about_sec_contact", self._build_contact_group),
            ("about_sec_rights", self._build_rights_group),
            ("about_sec_download", self._build_download_group),
            ("about_sec_funding", self._build_funding_group),
        ]
        for i, (key, builder) in enumerate(sections):
            gb = builder()
            card_title = gb.title()
            gb.setTitle("")
            gb.setObjectName("collapsibleBody")
            gb.setFlat(True)
            expanded = bool(settings.get(key + "_expanded", True))
            if key == "about_sec_dependencies" and self._dep_missing:
                # Something is missing: the operator is sent here on purpose,
                # so the card is unfolded whatever the saved state was.
                expanded = True
            card = CollapsibleCard(
                card_title, gb, expanded=expanded,
                on_toggle=lambda e, k=key: settings.set(k + "_expanded", bool(e)),
            )
            if key == "about_sec_dependencies":
                self._dep_card = card
                self._update_dep_title()
            content_layout.addWidget(card)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        box = QFrame()
        box.setObjectName("aboutHeader")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel(f"{__app_name__}")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setObjectName("aboutTitle")
        # Explicit left alignment: on Windows the native style centres
        # short single-line labels inside their layout cell. Setting the
        # flag pins the text to the left margin on every platform.
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        subtitle = QLabel(f"Version {__version__}   -   {APP_TAGLINE}")
        subtitle.setObjectName("aboutSubtitle")
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        return box

    def _build_description_group(self) -> QGroupBox:
        group = QGroupBox("Description")
        layout = QVBoxLayout(group)
        body = QLabel(APP_DESCRIPTION)
        body.setWordWrap(True)
        layout.addWidget(body)
        return group

    # ------------------------------------------------------------------
    # Dependencies card
    # ------------------------------------------------------------------
    def _save_dep_table_height(self, height: int) -> None:
        """Persist the package-list height chosen by dragging the grip."""
        try:
            self._settings.set(_DEP_TABLE_HEIGHT_KEY, int(height))
        except Exception:  # noqa: BLE001
            pass

    def _update_dep_title(self) -> None:
        """Put the missing-package count in the card header, in red when any."""
        card = getattr(self, "_dep_card", None)
        if card is None:
            return
        n = len(getattr(self, "_dep_missing", []))
        if n:
            title = ('Dependencies &nbsp;<span style="color:%s;">&mdash; %d '
                     'package%s missing</span>'
                     % (_MISSING_COLOR, n, "" if n == 1 else "s"))
        else:
            title = 'Dependencies &nbsp;<span style="color:%s;">&mdash; all ' \
                    'installed</span>' % _OK_COLOR
        try:
            label = card._header._title  # noqa: SLF001 - header is ours
            label.setTextFormat(Qt.RichText)
            label.setText(title)
        except Exception:  # noqa: BLE001
            pass

    def _build_dependencies_group(self) -> QGroupBox:
        """Live status of every declared package, so the environment is checkable.

        The startup warning scrolls past (and is invisible when the app is
        launched from a shortcut), so the same information is always available
        here: which packages are present, which are missing, which pages a
        missing one blocks, and the exact pip / conda command - copyable, or
        runnable from the page itself.
        """
        group = QGroupBox("Dependencies")
        self._dep_layout = QVBoxLayout(group)
        self._dep_layout.setSpacing(8)
        self._dep_missing = []
        self._populate_dependencies()
        return group

    def _populate_dependencies(self) -> None:
        """(Re)build the card contents from a fresh probe of the environment."""
        layout = self._dep_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        try:
            from ..dependency_check import (
                PIPELINE_LABELS, PIPELINE_REQUIREMENTS, calc_version_warnings,
                conda_command, installed_report, pip_command, version_report,
            )
            rows = installed_report()
            # (installed, validated, sensitive, verdict) keyed by pip name, so
            # the table can show the version and flag calc-sensitive mismatches.
            vmap = {pip: (inst, val, sens, verd)
                    for pip, _imp, inst, val, sens, verd in version_report()}
            calc_warn = calc_version_warnings()
        except Exception as exc:  # noqa: BLE001
            layout.addWidget(QLabel("Could not read the dependency list: %s" % exc))
            return

        missing = [(imp, pip) for pip, imp, found, _core in rows if not found]
        self._dep_missing = missing

        if not missing:
            head = QLabel("All %d declared packages are installed. Every "
                          "processing chain can run." % len(rows))
        else:
            head = QLabel(
                '<span style="color:%s;"><b>%d package(s) are missing.</b></span> '
                "The application runs, but the pages listed below refuse to "
                "start until they are installed."
                % (_MISSING_COLOR, len(missing))
            )
            head.setTextFormat(Qt.RichText)
        head.setWordWrap(True)
        layout.addWidget(head)

        # Alert: a calculation-sensitive package is on a version that can change
        # descriptor values (and therefore predictions) vs the validated golden
        # environment. This is the "results may be falsified" warning asked for.
        if calc_warn:
            items = "".join(
                "<li><b>%s</b>: installed <b>%s</b>, validated <b>%s</b>.<br/>%s</li>"
                % (p, inst or "?", val or "?", reason)
                for p, inst, val, reason in calc_warn
            )
            warn = QLabel(
                '<span style="color:%s;"><b>&#9888; %d calculation-sensitive '
                "package(s) differ from the validated versions.</b></span> "
                "Descriptor values &mdash; and therefore the stereotype / "
                "regeneration predictions &mdash; may differ from the golden "
                "reference. Reinstall the validated version to guarantee "
                "reproducibility:<ul>%s</ul>" % (_MISSING_COLOR, len(calc_warn), items)
            )
            warn.setTextFormat(Qt.RichText)
            warn.setWordWrap(True)
            layout.addWidget(warn)

        table = QTableWidget(len(rows), 5)
        table.setHorizontalHeaderLabels(
            ["Package", "Status", "Version", "Validated", "Used by"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setAlternatingRowColors(True)
        for r, (pip_name, import_name, found, core) in enumerate(rows):
            pages = sorted(
                PIPELINE_LABELS[k] for k, names in PIPELINE_REQUIREMENTS.items()
                if import_name in names
            )
            inst, val, sens, verd = vmap.get(pip_name, ("", "", False, "match"))
            cells = [
                pip_name,
                "installed" if found else "MISSING",
                inst or "-",
                (val or "-") + (" *" if sens else ""),
                ", ".join(pages) if pages else ("application" if core else "-"),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if not found:
                    font = item.font(); font.setBold(True); item.setFont(font)
                    item.setForeground(QColor(_MISSING_COLOR))
                elif c in (2, 3) and verd == "minor_diff":
                    # calc-sensitive version mismatch that can change results
                    font = item.font(); font.setBold(True); item.setFont(font)
                    item.setForeground(QColor(_MISSING_COLOR))
                elif c in (2, 3) and verd == "patch_diff":
                    item.setForeground(QColor(_WARN_COLOR))
                table.setItem(r, c, item)
        # Footnote on the asterisk so the operator knows what "Validated *" means.
        legend = QLabel(
            "* calculation-sensitive: a different version can change descriptor "
            "values (OpenCV CLAHE/resize, scikit-image skeleton/vesselness, "
            "mahotas Zernike, NumPy reductions). Validated = the environment in "
            "which the v4j golden was reproduced to the 2nd decimal."
        )
        legend.setWordWrap(True)
        legend.setObjectName("depLegend")
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        # The list is enlargeable from the bottom like the execution-log
        # panels, and the height the operator picks is saved so the card
        # comes back with the same layout at the next launch.
        natural = min(460, 40 + 28 * len(rows))
        saved = 0
        try:
            saved = int(self._settings.get(_DEP_TABLE_HEIGHT_KEY, 0) or 0)
        except Exception:  # noqa: BLE001
            saved = 0
        height = max(_DEP_TABLE_MIN_HEIGHT, saved or natural)
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)
        layout.addWidget(table)
        layout.addWidget(VerticalResizeGrip(
            table,
            min_height=_DEP_TABLE_MIN_HEIGHT,
            tooltip="Drag down to show more packages, up to shrink the list",
            on_resized=self._save_dep_table_height,
        ))
        layout.addWidget(legend)
        self._dep_table = table

        if not missing:
            self._update_dep_title()
            return

        # ---- install commands, copyable and runnable -------------------
        pip_cmd = pip_command(missing)
        conda_cmd = conda_command(missing)
        if pip_cmd:
            layout.addWidget(self._command_block(
                "Install the non-torch packages with pip:", pip_cmd))
        if conda_cmd:
            layout.addWidget(self._command_block(
                "Install the non-torch packages with conda:", conda_cmd))

        torch_family = {"torch", "torchvision", "segmentation-models-pytorch",
                        "timm", "pretrainedmodels", "efficientnet-pytorch"}
        risky = sorted({p for _, p in missing} & torch_family)
        if risky:
            warn = QLabel(
                '<span style="color:%s;"><b>Also missing, from the torch '
                "family: %s.</b></span> Install these yourself, from the SAME "
                "channel you used for torch - a bare pip install would pull a "
                "CPU-only torch wheel over a CUDA build and disable the GPU. "
                "They are deliberately left out of the commands above."
                % (_MISSING_COLOR, " ".join(risky))
            )
            warn.setTextFormat(Qt.RichText)
            warn.setWordWrap(True)
            layout.addWidget(warn)
        self._update_dep_title()

    def _command_block(self, caption: str, command: str) -> QWidget:
        """A caption, the command in a read-only field, and Copy / Run buttons."""
        box = QFrame()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(6)
        cap = QLabel(caption)
        cap.setWordWrap(True)
        v.addWidget(cap)

        row = QHBoxLayout()
        row.setSpacing(8)
        field = QLineEdit(command)
        field.setReadOnly(True)
        field.setFont(QFont("Consolas" if sys.platform.startswith("win")
                            else "Monospace"))
        field.setCursorPosition(0)
        row.addWidget(field, 1)

        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy this command to the clipboard")
        copy_btn.clicked.connect(lambda: self._copy_command(command, copy_btn))
        row.addWidget(copy_btn)

        run_btn = QPushButton("Run")
        run_btn.setToolTip("Run this command now, in the Python environment "
                           "FlatMorphAI is running from")
        run_btn.clicked.connect(lambda: self._run_command(command))
        row.addWidget(run_btn)
        v.addLayout(row)
        return box

    def _copy_command(self, command: str, button: QPushButton) -> None:
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.clipboard().setText(command)
        button.setText("Copied")
        QTimer.singleShot(1500, lambda: button.setText("Copy"))

    def _run_command(self, command: str) -> None:
        """Run an install command and stream its output into a dialog.

        pip is invoked as ``<this interpreter> -m pip`` rather than the bare
        ``pip`` on PATH: an operator often has several environments, and a
        terminal's active one is frequently NOT the one the application runs
        from. Installing into the running interpreter is the only way the new
        package is actually importable by the pipelines afterwards.
        """
        argv = _resolve_install_argv(command)
        if argv is None:
            _dep_message(self, "Install", "conda was not found on PATH. Open "
                         "an Anaconda prompt and paste the command instead.")
            return
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Question)
        confirm.setWindowTitle("Install packages")
        confirm.setText("Run this install now?")
        confirm.setInformativeText(
            " ".join(argv) + "\n\nTarget environment:\n" + sys.prefix
        )
        confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        confirm.setDefaultButton(QMessageBox.Yes)
        if confirm.exec() != QMessageBox.Yes:
            return
        dlg = _InstallDialog(argv, self)
        dlg.exec()
        # The environment may have changed: re-probe so the table and the card
        # title reflect what is installed now.
        try:
            import importlib
            importlib.invalidate_caches()
        except Exception:  # noqa: BLE001
            pass
        self._populate_dependencies()

    # ------------------------------------------------------------------
    # Installation check
    # ------------------------------------------------------------------
    def _build_install_check_group(self) -> QGroupBox:
        """Is this machine able to reproduce the reference results?

        The Dependencies card answers "is every package importable", which is a
        different question. A CUDA-less torch imports fine and silently moves
        every model to the processor; an OpenCV one minor version off imports
        fine and silently moves the descriptor VALUES. Neither shows up as a
        missing package, and both change what the operator gets. So this card
        probes the environment itself -- interpreter, the five
        calculation-sensitive versions, the torch build and the GPU, the
        weights, the data folders, the write permissions -- and for every
        problem states its nature and the action that fixes it.

        Nothing runs on page build: the probe touches the GPU and walks about
        2.2 GB of weights, so it starts when the operator asks.
        """
        group = QGroupBox("Installation check")
        self._ic_layout = QVBoxLayout(group)
        self._ic_layout.setSpacing(8)

        intro = QLabel(
            "Checks that this installation can reproduce the reference "
            "results, which is not the same thing as having every package "
            "installed: a torch built without CUDA and an OpenCV one minor "
            "version off both import perfectly and both change what you get "
            "&mdash; the first the run time, the second the descriptor values."
        )
        intro.setWordWrap(True)
        self._ic_layout.addWidget(intro)

        row = QHBoxLayout()
        self._ic_button = QPushButton("Run the check")
        self._ic_button.clicked.connect(lambda: self._run_install_check(False))
        row.addWidget(self._ic_button)
        self._ic_deep_button = QPushButton("Run the check + read every checkpoint")
        self._ic_deep_button.setToolTip(
            "Also loads each checkpoint with torch. Presence and size catch a "
            "missing or truncated download; only loading catches a corrupted "
            "one. Takes a few seconds and some GPU memory."
        )
        self._ic_deep_button.clicked.connect(lambda: self._run_install_check(True))
        row.addWidget(self._ic_deep_button)
        row.addStretch(1)
        self._ic_layout.addLayout(row)

        self._ic_summary = QLabel(
            "Not run yet. The same report is available without the interface, "
            "which is what to use when the application will not start:<br/>"
            "<tt>python -m flatmorphai.install_check</tt> &nbsp;&mdash;&nbsp; "
            "add <tt>--deep</tt> to read the checkpoints too."
        )
        self._ic_summary.setTextFormat(Qt.RichText)
        self._ic_summary.setWordWrap(True)
        self._ic_layout.addWidget(self._ic_summary)

        self._ic_report = QPlainTextEdit()
        self._ic_report.setReadOnly(True)
        self._ic_report.setMinimumHeight(140)
        self._ic_report.setPlaceholderText(
            "The detailed report appears here, selectable so it can be pasted "
            "into a bug report."
        )
        self._ic_layout.addWidget(self._ic_report)
        return group

    def _run_install_check(self, deep: bool) -> None:
        """Run the probe and show the report. Never raises: a diagnostic that
        crashes is worse than no diagnostic."""
        for b in (self._ic_button, self._ic_deep_button):
            b.setEnabled(False)
        self._ic_summary.setText("Running&hellip;")
        self._ic_report.setPlainText("")
        # Let the button repaint before the blocking probe starts.
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            from .. import install_check as ic
            steps: List[str] = []

            def progress(title: str) -> None:
                steps.append(title)
                self._ic_report.setPlainText("\n".join("  ... " + s for s in steps))
                QApplication.processEvents()

            results = ic.run_checks(deep=deep, progress=progress)
            counts = ic.counts(results)
            self._ic_report.setPlainText(ic.format_text(results))
            if counts[ic.FAIL]:
                colour, verdict = _MISSING_COLOR, (
                    "%d failure(s)" % counts[ic.FAIL])
                tail = ("A failure means a chain cannot run, or would not "
                        "produce the reference numbers. Each one is listed "
                        "below with what to do about it.")
            elif counts[ic.WARN]:
                colour, verdict = _WARN_COLOR, "%d warning(s)" % counts[ic.WARN]
                tail = ("No failure. A warning does not change the results "
                        "unless it says so; most of them are about speed.")
            else:
                colour, verdict = _OK_COLOR, "everything checks out"
                tail = "This installation matches the reference configuration."
            self._ic_summary.setText(
                '<span style="color:%s;"><b>%s</b></span> &mdash; %d ok, '
                "%d warning(s), %d failure(s), %d not tested.<br/>%s"
                % (colour, verdict, counts[ic.OK], counts[ic.WARN],
                   counts[ic.FAIL], counts[ic.SKIP], tail)
            )
        except Exception as exc:  # noqa: BLE001
            self._ic_summary.setText(
                '<span style="color:%s;">The check could not run: %s: %s</span>'
                % (_MISSING_COLOR, type(exc).__name__, exc)
            )
        finally:
            for b in (self._ic_button, self._ic_deep_button):
                b.setEnabled(True)

    def _build_contact_group(self) -> QGroupBox:
        group = QGroupBox("Contact the authors")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        author_label = QLabel(
            f'Author, developer: {AUTHOR_NAME} - '
            f'<a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a>'
        )
        author_label.setTextFormat(Qt.RichText)
        author_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        author_label.setOpenExternalLinks(False)
        author_label.linkActivated.connect(self._open_link)

        manager_label = QLabel(
            f'Project manager: {MANAGER_NAME} - '
            f'<a href="mailto:{MANAGER_EMAIL}">{MANAGER_EMAIL}</a>'
        )
        manager_label.setTextFormat(Qt.RichText)
        manager_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        manager_label.setOpenExternalLinks(False)
        manager_label.linkActivated.connect(self._open_link)

        email_label = QLabel(
            f'Team email: <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>'
        )
        email_label.setTextFormat(Qt.RichText)
        email_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        email_label.setOpenExternalLinks(False)
        email_label.linkActivated.connect(self._open_link)

        website_label = QLabel(
            f'Team website: <a href="{TEAM_WEBSITE_URL}">{TEAM_WEBSITE_LABEL}</a>'
        )
        website_label.setTextFormat(Qt.RichText)
        website_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        website_label.setOpenExternalLinks(False)
        website_label.linkActivated.connect(self._open_link)

        button_row = QHBoxLayout()
        email_author = QPushButton(f"Email {AUTHOR_NAME}")
        email_author.clicked.connect(lambda: self._open_link(f"mailto:{AUTHOR_EMAIL}"))
        email_manager = QPushButton(f"Email {MANAGER_NAME}")
        email_manager.clicked.connect(
            lambda: self._open_link(f"mailto:{MANAGER_EMAIL}")
        )
        open_site = QPushButton("Open team website")
        open_site.clicked.connect(lambda: self._open_link(TEAM_WEBSITE_URL))
        send_email = QPushButton("Send an email")
        send_email.clicked.connect(lambda: self._open_link(f"mailto:{CONTACT_EMAIL}"))
        button_row.addWidget(email_author)
        button_row.addWidget(email_manager)
        button_row.addWidget(open_site)
        button_row.addWidget(send_email)
        button_row.addStretch(1)

        layout.addWidget(author_label)
        layout.addWidget(manager_label)
        layout.addWidget(email_label)
        layout.addWidget(website_label)
        layout.addLayout(button_row)
        return group

    def _build_rights_group(self) -> QGroupBox:
        # The licence name is in the card header too, so it is readable
        # without unfolding the section.
        group = QGroupBox("Rights and licence — %s" % LICENCE_NAME)
        layout = QVBoxLayout(group)
        body = QLabel(RIGHTS_STATEMENT)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(body)
        # The full legal text is the only authority; the summary above is a
        # summary. Linked rather than pasted, so it cannot drift from the
        # official wording.
        link = QLabel('Full licence text: <a href="%s">%s</a>'
                      % (LICENCE_URL, LICENCE_URL))
        link.setTextFormat(Qt.RichText)
        link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        link.setOpenExternalLinks(False)
        link.linkActivated.connect(self._open_link)
        link.setWordWrap(True)
        layout.addWidget(link)
        return group

    def _build_download_group(self) -> QGroupBox:
        group = QGroupBox("Download and publication")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        download_label = QLabel(
            f'{DOWNLOAD_LABEL}: <a href="{DOWNLOAD_URL}">{DOWNLOAD_URL}</a>'
        )
        download_label.setTextFormat(Qt.RichText)
        download_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        download_label.setOpenExternalLinks(False)
        download_label.linkActivated.connect(self._open_link)

        paper_label = QLabel(
            f'{PAPER_LABEL}: <a href="{PAPER_URL}">{PAPER_URL}</a>'
        )
        paper_label.setTextFormat(Qt.RichText)
        paper_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        paper_label.setOpenExternalLinks(False)
        paper_label.linkActivated.connect(self._open_link)

        layout.addWidget(download_label)
        layout.addWidget(paper_label)
        return group

    def _build_funding_group(self) -> QGroupBox:
        group = QGroupBox("Funding")
        layout = QVBoxLayout(group)
        body = QLabel(FUNDING_STATEMENT)
        body.setWordWrap(True)
        layout.addWidget(body)
        return group

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _open_link(url: str) -> None:
        if not url:
            return
        QDesktopServices.openUrl(QUrl(url))
