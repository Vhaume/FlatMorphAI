"""
Status-bar GPU badge.

Design note
-----------
Earlier revisions of this widget polled :func:`torch.cuda.mem_get_info`
on a :class:`QTimer` to display live VRAM usage. On WSL2 + CUDA driver
setups that call has been observed to fault inside ``libcuda.so``
during regular operation (NVIDIA developer forum thread "torch.cuda.
mem_get_info segfaults on WSL2", 2023, no DOI). A native
:c:macro:`SIGSEGV` from a CUDA driver call cannot be intercepted by a
Python ``try``/``except`` clause: once the signal fires, the process
is torn down before the handler runs. The only safe way to keep the
GUI thread out of that crash window is to never call into the CUDA
driver from a GUI-thread timer.

This revision therefore reads the device name once on construction
(via :func:`torch.cuda.get_device_name`) and writes it to a static
label. There is no recurring callback, no timer, and no further CUDA
driver activity from the GUI thread for the lifetime of the badge.
Users who want live VRAM telemetry can run ``nvidia-smi -l 2`` from a
terminal: it queries NVML out-of-process and cannot bring the
application down.

Reference for the "fail loud, fail safe" policy: Wilson G. et al.,
"Good Enough Practices in Scientific Computing", PLOS Computational
Biology, 2017, DOI: 10.1371/journal.pcbi.1005510, rule 4.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLabel, QWidget

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from ..constants import COLOR_ERROR, COLOR_SUCCESS


class GPUBadge(QLabel):
    """Static label showing the detected GPU device name."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("gpuBadge")

        device_name = self._resolve_device_name()
        if device_name is None:
            self.setText("●  GPU: unavailable")
            self.setStyleSheet(self._pill_style(COLOR_ERROR))
        else:
            self.setText(f"●  GPU: {device_name}")
            self.setStyleSheet(self._pill_style(COLOR_SUCCESS))
        self.setToolTip(
            "Live VRAM telemetry is intentionally disabled on the GUI "
            "thread to avoid CUDA driver faults on WSL2. Run "
            "'nvidia-smi -l 2' for a live view."
        )

    # ------------------------------------------------------------------
    # Backwards-compatible no-ops
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Compatibility shim: there is no longer a timer to stop."""
        return

    @staticmethod
    def _pill_style(color_hex: str) -> str:
        """A rounded pill chip (tinted background + coloured text), like the
        dashboard status chip."""
        from PySide6.QtGui import QColor
        c = QColor(color_hex)
        tint = "rgba(%d, %d, %d, 0.15)" % (c.red(), c.green(), c.blue())
        border = "rgba(%d, %d, %d, 0.35)" % (c.red(), c.green(), c.blue())
        return (
            "QLabel#gpuBadge {"
            "  color: %s;"
            "  background-color: %s;"
            "  border: 1px solid %s;"
            "  border-radius: 11px;"
            "  padding: 3px 12px;"
            "  margin-right: 6px;"
            "  font-weight: 700;"
            "  font-size: 10pt;"
            "}" % (color_hex, tint, border)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device_name() -> Optional[str]:
        # Query the device name via ``nvidia-smi`` rather than
        # ``torch.cuda.get_device_name``. Calling into torch CUDA here would
        # create a persistent CUDA context in the GUI process that reserves
        # VRAM for the whole session -- memory that the segmentation /
        # regeneration SUBPROCESSES then cannot use, causing out-of-memory
        # errors that the in-process pipelines never hit. nvidia-smi reads the
        # name without touching CUDA, so the GUI holds zero GPU memory.
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:  # noqa: BLE001
            return None
        if out.returncode != 0:
            return None
        lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
        return lines[0] if lines else None
