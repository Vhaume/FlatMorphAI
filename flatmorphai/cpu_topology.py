"""How many cores this machine really has, in ONE place.

Why this module exists
----------------------
The application used to ask the question in six different ways. Two places
(the LightGBM descriptor pool and the regeneration chain) counted PHYSICAL
cores by reading ``/proc/cpuinfo``, with two identical copies of the same
function. Everywhere else used ``os.cpu_count()``, which returns the LOGICAL
count -- 8 on the reference machine (4 physical cores with hyper-threading)
and, worse, the host's count rather than the allowed one when the process is
restricted to a subset of CPUs. So the same run could size one pool for 4
cores and the next for 8.

Two numbers, two uses
---------------------
``logical_cores()`` is what the scheduler will actually run in parallel, after
CPU affinity. It is the right base for pools whose workers spend part of their
time waiting (reading a video, encoding and writing a PNG): a waiting thread
costs nothing and a hyper-thread covers the wait.

``physical_cores()`` is the number of real execution units. It is the right
CAP for a stage that saturates its cores AND costs memory per worker, because
past that point the workers only take turns while each still holds its RAM.

Neither is a universal pool size, and this module deliberately does not
pretend otherwise: on the reference machine the MEASURED best count for the
heaviest stage (the descriptors) is 5 to 6 processes on 4 physical cores, i.e.
above the physical count -- hyper-threading does buy something when the work
mixes computation with memory waits. ``pool_size`` therefore keeps each
caller's own rule and only makes it affinity-aware and portable.

Detection
---------
Three independent readings, first one that answers wins:
  1. ``/proc/cpuinfo``: count the distinct (physical id, core id) pairs;
  2. ``/sys/devices/system/cpu/cpu*/topology/thread_siblings_list``: count the
     distinct sibling groups (one group per physical core);
  3. ``lscpu -p=Core,Socket``: count the distinct (core, socket) rows.
All three agree on the machines tested (4 physical / 8 logical under WSL2;
1 physical / 2 logical in a 2-vCPU container). The result is clamped to the
number of logical CPUs the process may use, so a container restricted to two
CPUs never gets told it owns eight cores.

``FLATMORPHAI_PHYSICAL_CORES=<n>`` overrides the detection (0 or an invalid
value means "undetected", which makes every caller fall back to its previous
logical-core rule). Nothing here changes any computed value: it only changes
how many workers run at once, and every pool in this application is built so
that the number of workers does not affect the result.
"""
from __future__ import annotations

import os
import re
from typing import Optional

_CACHE: dict = {}


def logical_cores() -> int:
    """CPUs this process may run on (affinity included), at least 1."""
    n = 0
    try:
        n = len(os.sched_getaffinity(0))          # Linux: the allowed set
    except (AttributeError, OSError):
        n = 0
    if not n:
        try:
            n = os.cpu_count() or 0
        except Exception:  # noqa: BLE001
            n = 0
    return max(1, int(n or 1))


def _from_cpuinfo() -> Optional[int]:
    seen, phys, core = set(), None, None
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("physical id"):
                    phys = ln.split(":", 1)[1].strip()
                elif ln.startswith("core id"):
                    core = ln.split(":", 1)[1].strip()
                elif not ln.strip():
                    if core is not None:
                        seen.add((phys, core))
                    phys = core = None
        if core is not None:
            seen.add((phys, core))
    except Exception:  # noqa: BLE001
        return None
    return len(seen) or None


def _from_sysfs() -> Optional[int]:
    groups = set()
    try:
        base = "/sys/devices/system/cpu"
        for name in os.listdir(base):
            if not re.fullmatch(r"cpu\d+", name):
                continue
            path = os.path.join(base, name, "topology", "thread_siblings_list")
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    groups.add(fh.read().strip())
    except Exception:  # noqa: BLE001
        return None
    return len(groups) or None


def _from_lscpu() -> Optional[int]:
    try:
        import subprocess
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True,
                             text=True, timeout=5).stdout
        rows = {ln.strip() for ln in out.splitlines()
                if ln.strip() and not ln.startswith("#")}
    except Exception:  # noqa: BLE001
        return None
    return len(rows) or None


def physical_cores() -> Optional[int]:
    """Physical cores (hyper-threads excluded), or None when unknown.

    ``None`` is a real answer and means "do not cap on the CPU": a wrong count
    must never silently shrink a pool to one worker.
    """
    if "phys" in _CACHE:
        return _CACHE["phys"]
    env = (os.environ.get("FLATMORPHAI_PHYSICAL_CORES") or "").strip()
    if env:
        try:
            v = int(env)
            _CACHE["phys"] = v if v > 0 else None
            return _CACHE["phys"]
        except ValueError:
            pass
    n = None
    probes = (_from_cpuinfo, _from_sysfs, _from_lscpu)
    if os.name == "nt":          # native Windows: no /proc, no sysfs, no lscpu
        from .platform_compat import physical_cores as _win_pc
        probes = (_win_pc,)
    for probe in probes:
        n = probe()
        if n:
            break
    if n:
        n = min(int(n), logical_cores())      # never promise more than allowed
        n = n if n >= 1 else None
    _CACHE["phys"] = n or None
    return _CACHE["phys"]


def pool_size(reserve: int = 0, cap: Optional[int] = None,
              items: Optional[int] = None, base: str = "logical",
              floor: int = 1) -> int:
    """How many workers to start.

    ``base`` picks the number to start from: "logical" for work that waits
    (disk, encoding), "physical" for work that saturates a core and costs
    memory per worker. With ``base="physical"`` and an undetected count, the
    logical number is used instead, so the caller keeps its previous value.
    ``reserve`` leaves cores for the interface and the parent process,
    ``cap`` is a hard ceiling, ``items`` the amount of work available (no
    point starting more workers than there are tasks), and ``floor`` the
    minimum returned.
    """
    n = logical_cores()
    if base == "physical":
        n = physical_cores() or n
    n -= max(0, int(reserve))
    if cap is not None:
        n = min(n, int(cap))
    if items is not None:
        n = min(n, int(items))
    return max(int(floor), int(n))


def summary() -> str:
    """One line for the operator's log."""
    p = physical_cores()
    return ("%s physical core(s), %d logical available"
            % (p if p else "unknown number of", logical_cores()))
