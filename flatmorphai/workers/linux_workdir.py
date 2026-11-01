"""Run a pipeline on the LINUX disk when its output folder is on /mnt/<drive>.

Under WSL the Windows drives are reached through /mnt/c, where writing a
full-resolution PNG costs ~0.30 s instead of ~0.07 s on the Linux disk and
random reads in a video are ~3x slower (measured on the user's machine with
test_disque.py). Every intermediate file of Pipeline A (frames, masks,
formatted images, CSVs) is written and re-read several times, so the pipeline
works in ~/.flatmorphai/workdir/<id>/ and the results are copied back to the
user's folder at the end (the work copy is then removed).

Exactness: same files, same names, same content -- only their location during
the run changes. FLATMORPHAI_LINUX_WORKDIR=0 disables it.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from typing import Callable, Optional


def enabled() -> bool:
    return (sys.platform.startswith("linux")
            and os.environ.get("FLATMORPHAI_LINUX_WORKDIR", "1") in ("1", "true", "True"))


def on_windows_drive(path: str) -> bool:
    """True for paths under /mnt/<x>/ (WSL view of a Windows drive)."""
    try:
        p = os.path.realpath(os.path.abspath(path))
    except Exception:  # noqa: BLE001
        return False
    parts = p.split(os.sep)
    return len(parts) > 2 and parts[1] == "mnt" and len(parts[2]) == 1


def _root() -> str:
    return os.path.join(os.path.expanduser("~"), ".flatmorphai", "workdir")


def work_path(output_dir: str) -> Optional[str]:
    """Linux work folder for ``output_dir`` (path only, nothing created), or None."""
    if not enabled() or not on_windows_drive(output_dir):
        return None
    src = os.path.realpath(os.path.abspath(output_dir))
    return os.path.join(_root(), "%s_%s" % (os.path.basename(src.rstrip(os.sep)) or "out",
                                            hashlib.md5(src.encode("utf-8")).hexdigest()[:10]))


def prepare(work: str, output_dir: str, log: Optional[Callable[[str], None]] = None) -> bool:
    """Create ``work`` and pre-fill it with what already exists in ``output_dir``."""
    try:
        src = os.path.realpath(os.path.abspath(output_dir))
        if os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)
        t = time.time()
        if os.path.isdir(src) and any(os.scandir(src)):
            shutil.copytree(src, work, dirs_exist_ok=True)
        else:
            os.makedirs(work, exist_ok=True)
        if log is not None:
            log("Working on the Linux disk (%s); results are copied back to %s at the end%s"
                % (work, src, (" (existing content pre-copied in %.0f s)" % (time.time() - t))
                   if time.time() - t > 1 else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        if log is not None:
            log("Linux work folder could not be prepared (%s: %s)" % (type(exc).__name__, exc))
        return False


def plan(output_dir: str, log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Return the Linux work folder to use for ``output_dir`` (created and
    pre-filled with what already exists there), or None to work in place."""
    if not enabled() or not on_windows_drive(output_dir):
        return None
    try:
        src = os.path.realpath(os.path.abspath(output_dir))
        work = os.path.join(_root(), "%s_%s" % (os.path.basename(src.rstrip(os.sep)) or "out",
                                                hashlib.md5(src.encode("utf-8")).hexdigest()[:10]))
        if os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)       # stale copy of an interrupted run
        t = time.time()
        if os.path.isdir(src) and any(os.scandir(src)):
            shutil.copytree(src, work, dirs_exist_ok=True)
        else:
            os.makedirs(work, exist_ok=True)
        if log is not None:
            log("Working on the Linux disk (%s), results copied back to %s at the end%s"
                % (work, src, (" (existing content pre-copied in %.0f s)" % (time.time() - t))
                   if time.time() - t > 1 else ""))
        return work
    except Exception as exc:  # noqa: BLE001
        if log is not None:
            log("Linux work folder not used (%s: %s); working in place." % (type(exc).__name__, exc))
        return None


import threading as _threading

_BG = {}


def _copy_tree(work: str, output_dir: str, log=None, threads: int = 8) -> int:
    """Copy new / changed files of ``work`` to ``output_dir`` (8 at a time:
    /mnt/c is latency-bound, parallel copies overlap the round trips)."""
    from concurrent.futures import ThreadPoolExecutor
    src = os.path.realpath(os.path.abspath(output_dir))
    jobs = []
    for dp, _dn, fn in os.walk(work):
        rel = os.path.relpath(dp, work)
        dst = os.path.join(src, rel) if rel != "." else src
        os.makedirs(dst, exist_ok=True)
        for f in fn:
            a, b = os.path.join(dp, f), os.path.join(dst, f)
            try:
                if os.path.exists(b):
                    sa, sb = os.stat(a), os.stat(b)
                    if sa.st_size == sb.st_size and int(sa.st_mtime) <= int(sb.st_mtime):
                        continue
            except OSError:
                pass
            jobs.append((a, b))

    def _one(ab):
        try:
            shutil.copy2(ab[0], ab[1])
            return 1
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log("  WARNING: could not copy %s back (%s)" % (ab[1], exc))
            return 0
    if not jobs:
        return 0
    with ThreadPoolExecutor(max(1, threads)) as ex:
        return sum(ex.map(_one, jobs))


_PENDING = {}


def defer_background_sync(work: str, output_dir: str, log=None) -> None:
    """Arm the background copy; it starts on kick_pending() (called by the
    Step-4 pool once its workers are loaded, so the copy does not compete with
    their start-up reads on /mnt/c) or at the latest in sync_back()."""
    if work and os.path.isdir(work):
        _PENDING["job"] = (work, output_dir, log)


def kick_pending() -> None:
    job = _PENDING.pop("job", None)
    if job:
        start_background_sync(*job)


def start_background_sync(work: str, output_dir: str, log=None) -> None:
    """Start copying what already exists in ``work`` while the next steps run."""
    if not work or not os.path.isdir(work) or work in _BG:
        return
    box = {"n": 0, "t": time.time()}

    def _run():
        try:
            box["n"] = _copy_tree(work, output_dir, None, threads=4)
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc
        box["dt"] = time.time() - box["t"]
    th = _threading.Thread(target=_run, name="flatmorphai-copy-back", daemon=True)
    box["th"] = th
    _BG[work] = box
    th.start()
    if log is not None:
        log("Copying frames / masks / formatted images back to the Windows folder in the "
            "background while the next steps run")


def sync_back(work: str, output_dir: str, log: Optional[Callable[[str], None]] = None) -> None:
    """Finish the copy of ``work`` to ``output_dir`` (waits for the background
    copy, then copies what is new), then remove ``work``."""
    t = time.time()
    src = os.path.realpath(os.path.abspath(output_dir))
    _PENDING.pop("job", None)
    box = _BG.pop(work, None)
    n_bg = 0
    if box is not None:
        box["th"].join()
        n_bg = box.get("n", 0)
    n = _copy_tree(work, output_dir, log)
    shutil.rmtree(work, ignore_errors=True)
    if log is not None:
        log("Results copied back to %s (%d file(s) in the background%s, %d at the end in %.0f s)"
            % (src, n_bg, (", %.0f s" % box["dt"]) if box is not None and "dt" in box else "",
               n, time.time() - t))


def local_video_copy(path: str) -> Optional[str]:
    """Copy one input video from /mnt/<x> to the Linux disk (one sequential
    read, ~1 s per 100 MB) so the per-frame seeks are fast. None if not
    applicable; the caller deletes the copy."""
    if not enabled() or not on_windows_drive(path):
        return None
    try:
        d = os.path.join(os.path.expanduser("~"), ".flatmorphai", "video_cache")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "%s_%s" % (hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:10],
                                          os.path.basename(path)))
        shutil.copyfile(path, dst)
        return dst
    except Exception:  # noqa: BLE001
        return None
