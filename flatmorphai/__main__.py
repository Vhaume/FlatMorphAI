"""Enable ``python -m flatmorphai`` invocation."""

# Native Windows: Python reads and writes text in the ANSI code page unless the
# UTF-8 mode is on, whereas the application's files (CSV, JSON, parameters,
# names with "µ") are UTF-8, as under Linux. Restart once in UTF-8 mode; the
# package already exported PYTHONUTF8=1, so every subprocess starts in it too.
import os as _os
import sys as _sys

if (__name__ == "__main__" and _os.name == "nt"
        and not _sys.flags.utf8_mode):
    import subprocess as _sp
    _sys.exit(_sp.call([_sys.executable, "-X", "utf8", "-m", "flatmorphai"]
                       + _sys.argv[1:]))

# The dependency check runs FIRST -- before ``from .main import main``
# -- because ``main`` transitively imports PySide6, torch and cv2.
# If any of those wheels are missing, the operator would otherwise
# see a cryptic ImportError deep in the app code.  The check below
# uses only the standard library so it never itself triggers an
# ImportError.
from .dependency_check import enforce as _enforce_dependencies

_enforce_dependencies()

from .main import main  # noqa: E402 -- must run AFTER the check

if __name__ == "__main__":
    import sys
    sys.exit(main())
