"""
Persistent session logging.

A native crash (segfault in torch / Qt / a CUDA driver) wipes the terminal
and the in-app log panel before the operator can read or copy the error.  To
make every message recoverable, this module opens a line-buffered session log
file under ``~/.flatmorphai/logs/`` at startup and:

* tees ``stdout`` / ``stderr`` into it (so prints, forwarded subprocess output
  and Python tracebacks all land on disk as they happen) ;
* installs a ``sys.excepthook`` that records uncaught exceptions ;
* exposes :func:`write_line`, which the in-app log panel calls so every line it
  shows is also flushed to the file immediately.

Because the file is opened line-buffered and flushed on every write, the log is
intact on disk even when the process dies a line later.  This complements the
``faulthandler`` fatal-trace file set up in :mod:`flatmorphai.main` (which
captures the native C stack); together they give a Python-level error log AND a
native backtrace after any crash.

Everything here is best-effort and never raises: logging must not itself be a
source of failure.
"""
from __future__ import annotations

import atexit
import datetime
import os
import sys
import traceback
from typing import Optional

_SESSION_FILE = None          # the open file handle (kept alive for the process)
_LOG_PATH: Optional[str] = None


class _Tee:
    """Write to the original stream and the session file; never raise."""

    def __init__(self, stream, fh) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, data) -> int:
        try:
            if self._stream is not None:
                self._stream.write(data)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            return len(data)
        except Exception:  # noqa: BLE001
            return 0

    def flush(self) -> None:
        for target in (self._stream, self._fh):
            try:
                target.flush()
            except Exception:  # noqa: BLE001
                pass

    def __getattr__(self, name):
        # Delegate everything else (encoding, isatty, fileno, ...) to the
        # original stream so libraries that introspect it keep working.
        return getattr(self._stream, name)


def log_path() -> Optional[str]:
    """Absolute path of the current session log, or None when disabled."""
    return _LOG_PATH


def write_line(text: str) -> None:
    """Append one line to the session log immediately (best-effort)."""
    if _SESSION_FILE is None:
        return
    try:
        _SESSION_FILE.write(str(text).rstrip("\n") + "\n")
        _SESSION_FILE.flush()
    except Exception:  # noqa: BLE001
        pass


def init_session_log() -> Optional[str]:
    """Open the session log, tee the streams and hook exceptions.

    Idempotent: a second call returns the existing path without re-wrapping the
    streams.  Returns the log path, or None when it could not be created.
    """
    global _SESSION_FILE, _LOG_PATH
    if _SESSION_FILE is not None:
        return _LOG_PATH
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".flatmorphai", "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_PATH = os.path.join(log_dir, "session_%s.log" % stamp)
        _SESSION_FILE = open(
            _LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace"
        )
        _SESSION_FILE.write("=== FlatMorphAI session log %s ===\n" % stamp)
        _SESSION_FILE.flush()
    except Exception:  # noqa: BLE001
        _SESSION_FILE = None
        _LOG_PATH = None
        return None

    # Tee stdout / stderr so prints, forwarded subprocess output and
    # tracebacks are persisted. ``sys.__stdout__`` is the real console.
    try:
        sys.stdout = _Tee(sys.__stdout__, _SESSION_FILE)
        sys.stderr = _Tee(sys.__stderr__, _SESSION_FILE)
    except Exception:  # noqa: BLE001
        pass

    # Record uncaught exceptions, then chain to the previous hook.
    _previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb) -> None:
        try:
            write_line("UNCAUGHT EXCEPTION:")
            for chunk in traceback.format_exception(exc_type, exc_value, exc_tb):
                write_line(chunk.rstrip("\n"))
        except Exception:  # noqa: BLE001
            pass
        try:
            _previous_hook(exc_type, exc_value, exc_tb)
        except Exception:  # noqa: BLE001
            pass

    try:
        sys.excepthook = _hook
    except Exception:  # noqa: BLE001
        pass

    atexit.register(lambda: _SESSION_FILE and _SESSION_FILE.flush())
    return _LOG_PATH
