"""
Stereotype classification page.

Merges the former "Pipeline A" (videos -> stereotypes) and "Pipeline B"
(already-extracted images -> stereotypes) into a single page with two
collapsible sections:

* ``Videos -> Stereotype``  -- wraps :class:`PipelineAPage`.
* ``Images -> Stereotype``  -- wraps :class:`PipelineBPage`.

Each section's expanded/collapsed state is persisted so the page reopens with
the same layout. The two embedded pages keep their own run/cancel/log controls
and worker lifecycles; this page only forwards their status signals so the
sidebar card reflects activity from either one.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..widgets import CollapsibleCard
from .base_page import BasePage
from .pipeline_a_page import PipelineAPage
from .pipeline_b_page import PipelineBPage


class StereotypeClassificationPage(BasePage):
    """Two collapsible classification workflows (from videos / from images)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Stereotype classification", parent)
        self._settings = AppSettings()
        self.page_a = PipelineAPage()
        self.page_b = PipelineBPage()

        # Each embedded page is designed to fill a window; give them a sensible
        # minimum so they stay usable inside a collapsible card, and let the
        # outer scroll area handle any overflow when both are open.
        self.page_a.setMinimumHeight(680)
        self.page_b.setMinimumHeight(560)

        for sub in (self.page_a, self.page_b):
            # The card header already names the workflow, so hide each embedded
            # page's own big title and tighten its margins so it sits cleanly
            # inside the card body.
            for lbl in sub.findChildren(QLabel):
                if lbl.objectName() == "pageTitle":
                    lbl.hide()
                    break
            lay = sub.layout()
            if lay is not None:
                lay.setContentsMargins(12, 6, 12, 12)
            # Transparent body so the card (not the canvas) shows behind it.
            sub.setObjectName("collapsibleBody")
            # Forward the sub-page status so the sidebar card lights up.
            sub.page_status_changed.connect(self.page_status_changed)
            sub.run_state_changed.connect(self.run_state_changed)
            # Forward the sub-page's "go to the editor" request on success.
            sub.navigate_requested.connect(self.navigate_requested)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Stereotype classification")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        self._sec_videos = CollapsibleCard(
            "Videos -> Stereotype", self.page_a,
            expanded=bool(self._settings.get("stereo_sec_videos_expanded", True)),
            on_toggle=lambda e: self._settings.set(
                "stereo_sec_videos_expanded", bool(e)
            ),
        )
        self._sec_images = CollapsibleCard(
            "Images -> Stereotype", self.page_b,
            expanded=bool(self._settings.get("stereo_sec_images_expanded", False)),
            on_toggle=lambda e: self._settings.set(
                "stereo_sec_images_expanded", bool(e)
            ),
        )

        # Bold divider separating the two workflow cards.
        separator = QFrame()
        separator.setObjectName("sectionDivider")
        separator.setFrameShape(QFrame.HLine)
        separator.setFixedHeight(3)
        separator.setStyleSheet(
            "#sectionDivider { background-color: #0D9488; border: none; }"
        )

        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)
        cl.addWidget(self._sec_videos)
        cl.addWidget(separator)
        cl.addWidget(self._sec_images)
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

    def cancel_current(self) -> None:
        """Cancel whichever embedded workflow is running."""
        self.page_a.cancel_current()
        self.page_b.cancel_current()
