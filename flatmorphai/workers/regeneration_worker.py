# -*- coding: utf-8 -*-
"""Worker for the Forecast regeneration states page.

It runs :mod:`flatmorphai.workers.regeneration_pipeline` in a dedicated
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

from PySide6.QtCore import Signal

from ..base_worker import BaseWorker


def _export_root() -> str:
    """Absolute path of the bundled ``Export_head_detection`` folder.

    ``regeneration_worker.py`` lives at ``<app>/flatmorphai/workers/``, so the
    application root is two levels up and the export folder is its sibling.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "Export_head_detection")


@dataclass
class RegenerationParams:
    processing_root: str
    save_body_masks: bool = True
    save_eyespots_heatmaps: bool = False
    save_blasteme_masks: bool = False
    save_blasteme_csv: bool = False
    save_thumb_all_body: bool = False
    save_thumb_head_zoomed: bool = False
    save_thumb_regeneration: bool = False
    save_thumb_blastemes: bool = False
    save_thumb_eyespots: bool = False
    auto_state_increase_correction: bool = True
    auto_eyespots_increase_correction: bool = True
    #: A certain eyespot opens the door to the others at the permissive
    #: threshold.
    eyespots_pair_rule: bool = False
    eyespots_balanced_threshold: float = 0.80
    eyespots_positive_threshold: float = 0.60
    blasteme_refined: bool = False
    blasteme_keep_biggest: bool = False


class RegenerationWorker(BaseWorker):
    """Drive the regeneration pipeline subprocess and report progress."""

    # Emitted with the 0-based index of the step the driver just entered
    # (0 segmentation, 1 head detection, 2 state classification).
    step_signal = Signal(int)

    def __init__(self, params: RegenerationParams) -> None:
        super().__init__()
        self.params = params
        self._proc = None

    def run(self) -> None:
        p = self.params
        if not self.require_packages("regeneration"):
            return
        if not os.path.isdir(p.processing_root):
            self.error_signal.emit(
                "Processing root does not exist: " + p.processing_root
            )
            return

        # Prefer the upgraded v20 regeneration chain (skan2 axis, cascade v3c13,
        # refinement, two-branch state classifier) when its bundle is available.
        # Set FLATMORPHAI_REGEN_LEGACY=1 to force the previous pipeline.
        use_v20 = False
        bundle_root = ""
        if os.environ.get("FLATMORPHAI_REGEN_LEGACY") != "1":
            try:
                from ..regen_stage_bundle import (
                    available as _v20_available,
                    resolve_regen_stage_root as _v20_root,
                )
                if _v20_available():
                    bundle_root = _v20_root()
                    use_v20 = True
            except Exception:  # noqa: BLE001
                use_v20 = False

        if not use_v20:
            # Legacy chain needs the Export_head_detection bundle.
            export_root = _export_root()
            if not os.path.isdir(export_root):
                self.error_signal.emit(
                    "Export_head_detection folder not found next to the "
                    "application: " + export_root
                )
                return
        else:
            export_root = ""

        cancel_file = os.path.join(
            tempfile.gettempdir(), "flatmorphai_regen_cancel_%d" % os.getpid()
        )
        try:
            if os.path.exists(cancel_file):
                os.remove(cancel_file)
        except OSError:
            pass

        cfg = {
            "root": os.path.abspath(p.processing_root),
            "export_root": os.path.abspath(export_root) if export_root else "",
            "bundle_root": os.path.abspath(bundle_root) if bundle_root else "",
            "overwrite_masks": bool(getattr(p, "overwrite_masks", False)),
            "cancel_file": cancel_file,
            "save_body_masks": bool(p.save_body_masks),
            "save_eyespots_heatmaps": bool(p.save_eyespots_heatmaps),
            "save_blasteme_masks": bool(p.save_blasteme_masks),
            "save_blasteme_csv": bool(p.save_blasteme_csv),
            "save_thumb_all_body": bool(p.save_thumb_all_body),
            "save_thumb_head_zoomed": bool(p.save_thumb_head_zoomed),
            "save_thumb_regeneration": bool(p.save_thumb_regeneration),
            "save_thumb_blastemes": bool(p.save_thumb_blastemes),
            "save_thumb_eyespots": bool(p.save_thumb_eyespots),
            "auto_state_increase_correction": bool(p.auto_state_increase_correction),
            "auto_eyespots_increase_correction": bool(p.auto_eyespots_increase_correction),
            "eyespots_pair_rule": bool(getattr(p, "eyespots_pair_rule", False)),
            "eyespots_balanced_threshold": float(p.eyespots_balanced_threshold),
            "eyespots_positive_threshold": float(p.eyespots_positive_threshold),
            "blasteme_refined": bool(p.blasteme_refined),
            "blasteme_keep_biggest": bool(p.blasteme_keep_biggest),
        }
        cfg_file = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(cfg, cfg_file)
        cfg_file.close()

        driver = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "regen_stage_v20_pipeline.py" if use_v20 else "regeneration_pipeline.py",
        )
        self.log("Starting the regeneration pipeline (separate process)")
        if use_v20:
            self.log("Using the upgraded regeneration chain.")
        self.log("Processing root: " + cfg["root"])
        self.progress(0)

        env = dict(os.environ)
        # The allocator configuration is opt-in, see the note in
        # regen_stage_v20_pipeline.py: forcing expandable segments routes
        # allocations through the CUDA virtual-memory API, whose failures
        # are driver-level and unrecoverable under WSL.
        _ac = os.environ.get("FLATMORPHAI_CUDA_ALLOC_CONF", "").strip()
        if _ac:
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ac)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            [sys.executable, driver, cfg_file.name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(driver), env=env,
            text=True, bufsize=1,
        )

        n_folders = 1
        cur_folder = 0
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
                        frac = (float(d) / float(t)) if float(t) else 0.0
                        base = max(0, cur_folder - 1)
                        self.progress(int(100.0 * (base + frac) / max(1, n_folders)))
                    except ValueError:
                        pass
                elif line.startswith("STEP "):
                    try:
                        self.step_signal.emit(int(line[5:]))
                    except ValueError:
                        pass
                elif line.startswith("LOG "):
                    msg = line[4:]
                    if msg.startswith("folder "):
                        # "folder d/N: path"
                        try:
                            frac_part = msg.split()[1]
                            cur_folder, n_folders = (int(x) for x in frac_part.split("/"))
                        except (ValueError, IndexError):
                            pass
                    self.log(msg)
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
