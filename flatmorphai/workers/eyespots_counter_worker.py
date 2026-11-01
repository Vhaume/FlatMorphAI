# -*- coding: utf-8 -*-
"""Worker for the Eyespots counter page.

It runs :mod:`flatmorphai.workers.eyespots_counter_pipeline` in a dedicated
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
    :func:`flatmorphai.export_bundle.resolve_export_root`), else the sibling.
    """
    try:
        from ..export_bundle import resolve_export_root
        return resolve_export_root()
    except Exception:  # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "Export_head_detection")


@dataclass
class EyespotsCounterParams:
    """Parameters of a Eyespots counter run.

    ``seg_checkpoint`` is the absolute path of the segmentation checkpoint
    folder resolved from the Settings page; the driver points the bundle at it
    so generated masks match Pipelines A and B.  ``seg_model`` is its basename
    and drives the ``masks_<model>/`` folder name.
    """

    processing_root: str
    seg_checkpoint: str
    seg_model: str
    save_heatmaps: bool = False
    save_thumbnails: bool = False
    # Two counting thresholds, each independently switchable. At least one is
    # always on (the page refuses to turn the last one off): a disabled
    # threshold is simply not counted and its two columns are not written.
    threshold_balanced: float = 0.80
    threshold_positive: float = 0.60
    use_balanced: bool = True
    use_positive: bool = True
    #: A certain eyespot (strict threshold) opens the door to the others at
    #: the permissive one. With no certain eyespot, nothing is admitted.
    eyespots_pair_rule: bool = False
    # Mask refinement. When ``blasteme_refined`` is on, the full-body mask is
    # grown with the blastema prediction (keeping only its biggest component).
    blasteme_refined: bool = False


class EyespotsCounterWorker(BaseWorker):
    """Drive the Eyespots counter subprocess and report progress."""

    def __init__(self, params: EyespotsCounterParams) -> None:
        super().__init__()
        self.params = params
        self._proc = None

    def run(self) -> None:
        p = self.params
        if not self.require_packages("eyespots"):
            return
        # The eyespots counter now uses the SAME v20 bundle as the regeneration
        # page (its head cascade is what detects the eyes). Resolve it the same
        # way the regeneration worker does.
        try:
            from ..regen_stage_bundle import (
                available as _v20_available,
                resolve_regen_stage_root as _v20_root,
            )
            bundle_root = _v20_root() if _v20_available() else ""
        except Exception:  # noqa: BLE001
            bundle_root = ""
        if not bundle_root or not os.path.isdir(bundle_root):
            self.error_signal.emit(
                "Regeneration v20 bundle not found (needed for the eyespot "
                "detector): " + str(bundle_root)
            )
            return
        if not os.path.isdir(p.processing_root):
            self.error_signal.emit(
                "Processing root does not exist: " + p.processing_root
            )
            return

        cancel_file = os.path.join(
            tempfile.gettempdir(), "flatmorphai_eyespots_cancel_%d" % os.getpid()
        )
        try:
            if os.path.exists(cancel_file):
                os.remove(cancel_file)
        except OSError:
            pass

        cfg = {
            "root": os.path.abspath(p.processing_root),
            "bundle_root": os.path.abspath(bundle_root),
            "cancel_file": cancel_file,
            "seg_checkpoint": os.path.abspath(p.seg_checkpoint) if p.seg_checkpoint else "",
            "seg_model": p.seg_model,
            "save_heatmaps": bool(p.save_heatmaps),
            "save_thumbnails": bool(p.save_thumbnails),
            "threshold_balanced": float(p.threshold_balanced),
            "threshold_positive": float(p.threshold_positive),
            "use_balanced": bool(getattr(p, "use_balanced", True)),
            "use_positive": bool(getattr(p, "use_positive", True)),
            "eyespots_pair_rule": bool(getattr(p, "eyespots_pair_rule", False)),
            "overwrite": bool(getattr(p, "overwrite", False)),
        }
        cfg_file = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(cfg, cfg_file)
        cfg_file.close()

        driver = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "eyespots_counter_pipeline.py")
        self.log("Starting the eyespots counter (separate process)")
        self.log("Processing root: " + cfg["root"])
        _thr = []
        if cfg["use_balanced"]:
            _thr.append("strict %g" % cfg["threshold_balanced"])
        if cfg["use_positive"]:
            _thr.append("permissive %g" % cfg["threshold_positive"])
        if cfg.get("eyespots_pair_rule"):
            # The gate is the SELECTED threshold, the admission is always the
            # permissive one from the Settings -- say both values, because a
            # rule whose two ends are not stated cannot be checked against the
            # CSV it produced.
            _gate = (cfg["threshold_balanced"] if cfg["use_balanced"]
                     else cfg["threshold_positive"])
            _adm = cfg["threshold_positive"]
            if _adm < _gate:
                _thr.append("pair rule on (one eyespot at %g admits the "
                            "others at %g)" % (_gate, _adm))
            else:
                _thr.append("pair rule on but inactive (gate %g and admission "
                            "%g are the same threshold)" % (_gate, _adm))
        self.log("Save heatmaps %s | Save thumbnails %s | threshold(s): %s"
                 % ("on" if cfg["save_heatmaps"] else "off",
                    "on" if cfg["save_thumbnails"] else "off",
                    ", ".join(_thr) or "none"))
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
        if self._proc and self._proc.poll() is None:
            time.sleep(0.2)
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
