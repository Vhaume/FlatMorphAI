"""Native-Windows equivalents of the Linux interfaces the application reads.

Under Linux (and WSL) the application reads ``/proc`` for the memory and the
processor topology. Those files do not exist under native Windows, where the
same figures come from the Win32 API through ``ctypes`` (no extra package).
Every function here returns ``None`` (or an empty result) on any other system
or on any failure, so a caller keeps its own Linux path untouched and only
falls back to this module when ``os.name == "nt"``.

References: Microsoft Learn, "GlobalMemoryStatusEx", "GetProcessMemoryInfo",
"GetLogicalProcessorInformationEx" (Win32 API documentation).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

IS_WINDOWS = os.name == "nt"


def _kernel32():
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error=True)


def memory_status_mb() -> Dict[str, float]:
    """``{'MemTotal': MB, 'MemAvailable': MB, 'SwapTotal': MB, 'SwapFree': MB}``
    (the keys of ``/proc/meminfo`` the application uses), or ``{}``."""
    if not IS_WINDOWS:
        return {}
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        k32 = _kernel32()
        k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        if not k32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return {}
        mb = 1024.0 * 1024.0
        swap_total = max(0.0, (st.ullTotalPageFile - st.ullTotalPhys) / mb)
        swap_free = max(0.0, (st.ullAvailPageFile - st.ullAvailPhys) / mb)
        return {
            "MemTotal": st.ullTotalPhys / mb,
            "MemAvailable": st.ullAvailPhys / mb,
            "MemFree": st.ullAvailPhys / mb,
            "SwapTotal": swap_total,
            "SwapFree": swap_free,
        }
    except Exception:  # noqa: BLE001
        return {}


def _process_counters(pid: Optional[int] = None):
    """PROCESS_MEMORY_COUNTERS_EX of ``pid`` (this process when None)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    k32 = _kernel32()
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    # Explicit prototypes: a 64-bit HANDLE must not be squeezed into a C int.
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                           wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_VM_READ = 0x0010
    own = pid is None
    if own:
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        h = k32.GetCurrentProcess()
    else:
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                            False, int(pid))
        if not h:
            return None
    try:
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return None
        return c
    finally:
        if not own:
            k32.CloseHandle(h)


def process_private_mb(pid: Optional[int] = None) -> Optional[float]:
    """Private memory of a process in MB (the closest Windows figure to the
    Linux PSS the memory planner reads), or None."""
    if not IS_WINDOWS:
        return None
    try:
        c = _process_counters(pid)
        return None if c is None else c.PrivateUsage / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def peak_working_set_mb() -> Optional[float]:
    """Peak resident memory of this process in MB, or None."""
    if not IS_WINDOWS:
        return None
    try:
        c = _process_counters(None)
        return None if c is None else c.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def physical_cores() -> Optional[int]:
    """Physical processor cores (hyper-threads excluded), or None."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = _kernel32()
        k32.GetLogicalProcessorInformationEx.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        k32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
        RelationProcessorCore = 0
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(RelationProcessorCore, None,
                                             ctypes.byref(size))
        if size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(RelationProcessorCore, buf,
                                                    ctypes.byref(size)):
            return None
        # A sequence of variable-size records: Relationship (DWORD), Size (DWORD), ...
        n, off = 0, 0
        raw = buf.raw
        while off + 8 <= size.value:
            rel = int.from_bytes(raw[off:off + 4], "little")
            rec = int.from_bytes(raw[off + 4:off + 8], "little")
            if rec <= 0:
                break
            if rel == RelationProcessorCore:
                n += 1
            off += rec
        return n or None
    except Exception:  # noqa: BLE001
        return None
