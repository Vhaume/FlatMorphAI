# -*- coding: utf-8 -*-
"""Worker for Image segmentation - segmentation only.

It runs :mod:`flatmorphai.workers.segmentation_c_pipeline` in a dedicated
subprocess (see that module for the reason) and streams its stdout back to the
page as log lines and progress.  Cancellation touches a sentinel file the
driver polls, then terminates the process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from ..base_worker import BaseWorker


def _export_root() -> str:
    """Absolute path of the ``Export_head_detection`` bundle to use.

    Prefers the local in-package copy when complete (see
    :func:`flatmorphai.export_bundle.resolve_export_root`), else the sibling
    folder two levels up from ``flatmorphai/workers/``.
    """
    try:
        from ..export_bundle import resolve_export_root
        return resolve_export_root()
    except Exception:  # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "Export_head_detection")


@dataclass
class SegmentationCParams:
    """Parameters of a Image segmentation run.

    ``seg_checkpoint`` is the absolute path of the segmentation checkpoint
    folder resolved from the Settings page; the driver points the bundle at it
    so Image segmentation masks match Pipelines A and B.  ``seg_model`` is its basename
    and drives the ``masks_<model>/`` output-folder name.
    """

    processing_root: str
    seg_checkpoint: str
    seg_model: str
    blasteme_refined: bool = False
    blasteme_keep_biggest: bool = False
    overwrite: bool = False
    save_body_masks: bool = True
    save_blasteme_masks: bool = False
    save_blasteme_thumbnails: bool = False
    save_head_csv: bool = False
    save_body_thumbnails: bool = False


class SegmentationCWorker(BaseWorker):
    """Drive the Image segmentation segmentation subprocess and report progress."""

    def __init__(self, params: SegmentationCParams) -> None:
        super().__init__()
        self.params = params
        self._proc = None

    def run(self) -> None:
        p = self.params
        if not self.require_packages("segmentation"):
            return
        export_root = _export_root()
        if not os.path.isdir(export_root):
            self.error_signal.emit(
                "Export_head_detection folder not found next to the application: "
                + export_root
            )
            return
        if not os.path.isdir(p.processing_root):
            self.error_signal.emit(
                "Processing root does not exist: " + p.processing_root
            )
            return

        cancel_file = os.path.join(
            tempfile.gettempdir(), "flatmorphai_segc_cancel_%d" % os.getpid()
        )
        try:
            if os.path.exists(cancel_file):
                os.remove(cancel_file)
        except OSError:
            pass

        cfg = {
            "root": os.path.abspath(p.processing_root),
            "export_root": os.path.abspath(export_root),
            "cancel_file": cancel_file,
            "seg_checkpoint": os.path.abspath(p.seg_checkpoint) if p.seg_checkpoint else "",
            "seg_model": p.seg_model,
            "blasteme_refined": bool(p.blasteme_refined),
            "blasteme_keep_biggest": bool(p.blasteme_keep_biggest),
            "overwrite": bool(p.overwrite),
            "save_body_masks": bool(p.save_body_masks),
            "save_blasteme_masks": bool(p.save_blasteme_masks),
            "save_blasteme_thumbnails": bool(p.save_blasteme_thumbnails),
            "save_head_csv": bool(p.save_head_csv),
            "save_body_thumbnails": bool(p.save_body_thumbnails),
        }
        cfg_file = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(cfg, cfg_file)
        cfg_file.close()

        driver = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "segmentation_c_pipeline.py")
        self.log("Starting Image segmentation segmentation (separate process)")
        self.log("Processing root: " + cfg["root"])
        self.log("Blastema refined %s"
                 % ("on" if cfg["blasteme_refined"] else "off"))
        self.progress(0)

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            [sys.executable, driver, cfg_file.name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(driver), env=env,
            text=True, bufsize=1,
        )

        ok = False
        try:
            for raw in self._proc.stdout:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if self.is_interrupted():
                    self._cancel(cancel_file)
                if line.startswith("PROGRESS "):
                    try:
                        _, d, t = line.split()
                        pct = int(100.0 * float(d) / float(t)) if float(t) else 0
                        self.progress(max(0, min(100, pct)))
                    except ValueError:
                        pass
                elif line.startswith("LOG "):
                    self.log(line[4:])
                elif line.startswith("DONE"):
                    ok = True
                    self.log(line)
                elif line.startswith("ERROR "):
                    self.error_signal.emit(line[6:])
                else:
                    self.log(line)
        finally:
            self._proc.wait()
            try:
                os.remove(cfg_file.name)
            except OSError:
                pass
            try:
                if os.path.exists(cancel_file):
                    os.remove(cancel_file)
            except OSError:
                pass

        if self.is_interrupted():
            self.log("Cancelled by the user")
            return
        if self._proc.returncode != 0 and not ok:
            if self._proc.returncode is not None:
                self.error_signal.emit(
                    "Pipeline exited with code %d" % self._proc.returncode
                )
            return
        self.progress(100)
        self.result_signal.emit({"root": cfg["root"]})

    def force_terminate(self) -> None:
        """Kill the child process now so no subprocess survives app quit."""
        self.request_interruption()
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def _cancel(self, cancel_file: str) -> None:
        try:
            open(cancel_file, "w").close()
        except OSError:
            pass
        # give the driver a moment to stop cleanly, then terminate
        if self._proc and self._proc.poll() is None:
            time.sleep(0.2)
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
