# -*- coding: utf-8 -*-
"""One shared GPU process for the Step-4 descriptor workers.

Measured on the user's machine: importing torch + creating a CUDA context +
running the Hessian / Farneback kernels costs ~810 MB of private RAM in EVERY
worker process (90 MB without torch, 900 MB with). With ~1.6 GB per worker
that is half of it, and it is what limits the pool to ~3 workers in 8 GB.

Here ONE process owns torch/CUDA and runs the GPU twins (batched Hessian,
Farneback v2 float32, Gabor bank FFT float32); the workers never import torch:
they send their arrays over a local socket (multiprocessing.connection,
AF_UNIX, authkey) and get the result back. The server executes the very same
functions as the in-process path, so the results are the same; requests are
served one at a time (the GPU was already serialised by the lock).

  start(log)  -> parent, before the pool: spawns the server, waits for it,
                 exports FLATMORPHAI_GPU_SERVER / _KEY for the workers.
  call(op, *) -> worker side; raises on any error (callers fall back to CPU).
  stop()      -> parent, after the pool.
  active()    -> True in a worker when a server address is exported.

FLATMORPHAI_GPU_SERVER_ON=0 disables it (each worker then uses the GPU itself,
as before).
"""
from __future__ import annotations

import os
import sys
import threading

ENV_ADDR = "FLATMORPHAI_GPU_SERVER"
ENV_KEY = "FLATMORPHAI_GPU_SERVER_KEY"
STATS = {"calls": 0, "errors": 0}
_PROC = None
_CONN = None
_CLOCK = threading.Lock()


def enabled():
    return os.environ.get("FLATMORPHAI_GPU_SERVER_ON", "1") in ("1", "true", "True")


def active():
    return bool(os.environ.get(ENV_ADDR))


# ------------------------------------------------- shared-memory transport
# Big arrays (images in, flow / Hessian / Gabor maps out: up to ~50 MB per
# call) are not pickled through the socket: each worker owns two PERSISTENT
# shared-memory buffers (input, output), created once and grown when needed,
# and only their layout travels in the message. Measured: 48 ms -> 19 ms per
# call for 7.7 MB in / 46 MB out, and the server (which serves every worker
# one request at a time) no longer spends that time pickling. Same bytes, so
# the results are identical. Any problem -> the pickled path, as before.
# FLATMORPHAI_GPU_SHM=0 disables.
_SHM_TAG = "__shm_array__"


def shm_enabled():
    return os.environ.get("FLATMORPHAI_GPU_SHM", "1") in ("1", "true", "True")


def _untrack(shm):
    """The process that only ATTACHES must not let its resource tracker
    unlink (or warn about) a segment it does not own."""
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


def _pack(obj, buf, state):
    """Replace every numpy array of ``obj`` (nested list / tuple / dict) by a
    descriptor and copy its bytes into ``buf`` at 64-byte aligned offsets.
    Raises OverflowError when ``buf`` is too small (state['need'] = bytes)."""
    import numpy as np
    if isinstance(obj, np.ndarray) and not obj.dtype.hasobject:
        a = np.ascontiguousarray(obj)
        off = state["off"]
        end = off + a.nbytes
        state["off"] = (end + 63) & ~63
        state["need"] = state["off"]
        if buf is None or end > len(buf):
            state["over"] = True
            return (_SHM_TAG, off, a.shape, a.dtype.str)
        if a.nbytes:
            np.ndarray(a.shape, a.dtype, buffer=buf, offset=off)[...] = a
        return (_SHM_TAG, off, a.shape, a.dtype.str)
    if isinstance(obj, list):
        return [_pack(x, buf, state) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_pack(x, buf, state) for x in obj)
    if isinstance(obj, dict):
        return {k: _pack(v, buf, state) for k, v in obj.items()}
    return obj


def _unpack(obj, buf, copy):
    import numpy as np
    if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == _SHM_TAG:
        _t, off, shape, dt = obj
        a = np.ndarray(tuple(shape), np.dtype(dt), buffer=buf, offset=off)
        return a.copy() if copy else a
    if isinstance(obj, list):
        return [_unpack(x, buf, copy) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_unpack(x, buf, copy) for x in obj)
    if isinstance(obj, dict):
        return {k: _unpack(v, buf, copy) for k, v in obj.items()}
    return obj


# ----------------------------------------------------------------- server side
def _ops():
    from . import gpu_batch_ops as G
    from . import hessian_batch_gpu as HB

    def farneback(a, b, params):
        return G.farneback_v2([a], [b], *params[:6], store="float32")[0]

    def hessian(image, sigmas, dtype):
        return HB.hessian_batch_torch(image, sigmas, dtype=dtype)

    def gabor(image, specs):
        res = G.gabor_bank_torch(image, specs)
        # float32 values (already float32-exact): halve the transfer
        import numpy as np
        return [(r.astype(np.float32), i.astype(np.float32)) for r, i in res]

    return {"farneback": farneback, "hessian": hessian, "gabor": gabor,
            "ping": lambda: "pong", "selftest": lambda: _selftest(farneback, hessian, gabor)}


def _selftest(farneback, hessian, gabor):
    """GPU-vs-CPU check on FIXED synthetic images (seeded texture + sub-pixel
    shift), run once per run in the server. Every worker then takes the same
    decision, so no frame's path depends on which worker validated which
    frames (one worker used to fall back to the CPU Farneback alone after an
    unlucky first frame, EPE 1.1e-3 > 1e-3, making its frames differ).
    Thresholds unchanged: Farneback mean EPE <= 1e-3 px and <= 0.5 % of pixels
    > 0.01 px; Hessian and Gabor max relative error <= 1e-4."""
    import numpy as np
    import cv2
    rng = np.random.default_rng(20261028)
    H, W = 480, 640
    tex = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3.0)
    tex = (255 * (tex - tex.min()) / (np.ptp(tex) + 1e-9)).astype(np.uint8)
    blob = np.zeros((H, W), np.uint8)
    cv2.ellipse(blob, (320, 240), (180, 60), 25, 0, 360, 255, -1)
    a = cv2.addWeighted(tex, 0.6, blob, 0.4, 0)
    M = np.float32([[1, 0, 1.3], [0, 1, -0.7]])
    b = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = {}
    try:
        fbp = (0.5, 3, 15, 3, 5, 1.2, 0)
        ref = cv2.calcOpticalFlowFarneback(a, b, None, *fbp)
        g = farneback(a, b, fbp)
        e = np.sqrt(((g - ref) ** 2).sum(-1))
        epe, frac = float(e.mean()), float((e > 0.01).mean())
        out["farneback"] = (bool(epe <= 1e-3 and frac <= 5e-3), "EPE %.1e px, %.2f %% > 0.01 px" % (epe, 100 * frac))
    except Exception as exc:  # noqa: BLE001
        out["farneback"] = (False, "%s: %s" % (type(exc).__name__, exc))
    try:
        from skimage.feature import corner as _CO
        from . import hessian_batch_gpu as HB
        img = a.astype(np.float64)
        sig = [1.0, 2.0, 4.0]
        res = hessian(img, sig, os.environ.get("FLATMORPHAI_HESSIAN_GPU_DTYPE", "float32"))
        err = max(HB.max_rel_err(_CO._hessian_matrix_with_gaussian(img, sigma=s, mode="reflect", cval=0,
                                                                  order="rc"), res[s]) for s in sig)
        out["hessian"] = (bool(err <= 1e-4), "max rel err %.1e" % err)
    except Exception as exc:  # noqa: BLE001
        out["hessian"] = (False, "%s: %s" % (type(exc).__name__, exc))
    try:
        from .fusion_tsm.metrics_lib import gabor_gpu as GG
        crop = a[48:432, 128:512].astype(np.float64)
        specs = [(0.1, t) for t in np.linspace(0, np.pi, 4, endpoint=False)] + [(0.25, 0.0)]
        ref = GG._fft_gabor_many(crop, specs)
        g = gabor(crop, specs)
        e = 0.0
        for (r1, i1), (r2, i2) in zip(ref, g):
            m = max(np.abs(r1).max(), np.abs(i1).max()) + 1e-30
            e = max(e, float(max(np.abs(np.asarray(r2, np.float64) - r1).max(),
                                 np.abs(np.asarray(i2, np.float64) - i1).max()) / m))
        out["gabor"] = (bool(e <= 1e-4), "max rel err %.1e" % e)
    except Exception as exc:  # noqa: BLE001
        out["gabor"] = (False, "%s: %s" % (type(exc).__name__, exc))
    return out


ENV_DECISION = "FLATMORPHAI_GPU_DECISION"


def decision(op):
    """Run-wide GPU decision for ``op`` from the server self-test: True / False,
    or None when there is none (no server, or an older caller)."""
    v = os.environ.get(ENV_DECISION, "")
    for part in v.split(","):
        k, _, x = part.partition("=")
        if k == op and x in ("0", "1"):
            return x == "1"
    return None


#: Connection family: a Unix-domain socket under Linux (as validated), a named
#: pipe under native Windows, which has no Unix-domain sockets in
#: multiprocessing.connection.
_FAMILY = "AF_PIPE" if os.name == "nt" else "AF_UNIX"


def _serve(addr, key, ready):
    from multiprocessing.connection import Listener, wait
    import torch
    torch.zeros(1, device="cuda")                      # CUDA context up front
    ops = _ops()
    lst = Listener(addr, family=_FAMILY, authkey=key)
    conns = []
    maps = {}            # connection -> {"in": (name, shm), "out": (name, shm)}
    stop = threading.Event()

    def _accept():
        while not stop.is_set():
            try:
                c = lst.accept()
            except Exception:  # noqa: BLE001
                if stop.is_set():
                    return
                continue
            conns.append(c)
    th = threading.Thread(target=_accept, daemon=True)
    th.start()
    ready.set()
    n = 0
    while not stop.is_set():
        live = [c for c in conns if not c.closed]
        if not live:
            stop.wait(0.05)
            continue
        for c in wait(live, timeout=0.5):
            try:
                msg = c.recv()
            except (EOFError, OSError):
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
                continue
            op, args = msg[0], msg[1:]
            if op == "stop":
                stop.set()
                break
            try:
                if op == "shm":
                    c.send(_serve_shm(ops, maps, c, *args))
                else:
                    out = ops[op](*args)
                    c.send(("ok", out))
            except Exception as exc:  # noqa: BLE001
                c.send(("err", "%s: %s" % (type(exc).__name__, exc)))
            n += 1
            if n % 64 == 0:
                torch.cuda.empty_cache()
    try:
        lst.close()
    except Exception:  # noqa: BLE001
        pass


def _attach(maps, c, slot, name):
    from multiprocessing import shared_memory
    m = maps.setdefault(id(c), {})
    cur = m.get(slot)
    if cur is not None and cur[0] == name:
        return cur[1]
    if cur is not None:                      # the worker grew its buffer
        try:
            cur[1].close()
        except Exception:  # noqa: BLE001
            pass
    # NOT untracked: the workers and this server are spawned by the same
    # parent and share its resource tracker (a set of names): attaching adds
    # nothing, and the owning worker's unlink() removes the entry.
    shm = shared_memory.SharedMemory(name=name)
    m[slot] = (name, shm)
    return shm


def _serve_shm(ops, maps, c, op, in_name, out_name, args):
    shm_in = _attach(maps, c, "in", in_name)
    shm_out = _attach(maps, c, "out", out_name) if out_name else None
    real = _unpack(args, shm_in.buf, copy=False)     # read-only views
    out = ops[op](*real)
    del real
    st = {"off": 0, "need": 0, "over": False}
    desc = _pack(out, shm_out.buf if shm_out is not None else None, st)
    if st["over"]:          # does not fit: pickled this time, bigger buffer next time
        return ("ok_pickled", out, st["need"])
    return ("ok_shm", desc, st["need"])


def _main(addr, key_hex, ready):
    try:
        _serve(addr, bytes.fromhex(key_hex), ready)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[FlatMorphAI] GPU server failed: %s: %s\n" % (type(exc).__name__, exc))
        ready.set()


def start(log=None, timeout=120.0):
    """Start the shared GPU server (parent process). Returns True if up."""
    global _PROC
    _log = log or (lambda _m: None)
    if not enabled() or _PROC is not None:
        return _PROC is not None
    try:
        import multiprocessing as mp
        import secrets
        import tempfile
        import time
        try:
            import torch
            if not torch.cuda.is_available():
                _log("  GPU server: no CUDA GPU -> not started (CPU paths)")
                return False
        except Exception:  # noqa: BLE001
            _log("  GPU server: torch unavailable -> not started (CPU paths)")
            return False
        if _FAMILY == "AF_PIPE":
            addr = r"\\.\pipe\flatmorphai_gpu_%d" % os.getpid()
        else:
            addr = os.path.join(tempfile.gettempdir(), "flatmorphai_gpu_%d.sock" % os.getpid())
            try:
                os.unlink(addr)
            except OSError:
                pass
        key = secrets.token_hex(16)
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        p = ctx.Process(target=_main, args=(addr, key, ready), daemon=True,
                        name="flatmorphai-gpu-server")
        p.start()
        t0 = time.time()
        if not ready.wait(timeout) or not p.is_alive():
            p.terminate()
            _log("  GPU server: failed to start -> each worker uses the GPU itself")
            return False
        os.environ[ENV_ADDR] = addr
        os.environ[ENV_KEY] = key
        _PROC = p
        try:
            r = call("ping")
        except Exception as exc:  # noqa: BLE001
            r = repr(exc)
        if r != "pong":
            stop()
            _log("  GPU server: no answer (%s) -> each worker uses the GPU itself" % r)
            return False
        _log("  GPU server: ON (one process owns torch/CUDA for all workers, "
             "~0.8 GB RAM saved per worker; up in %.1f s)" % (time.time() - t0))
        try:
            t1 = time.time()
            res = call("selftest")
            os.environ[ENV_DECISION] = ",".join("%s=%d" % (k, 1 if v[0] else 0) for k, v in res.items())
            _log("  GPU self-test on fixed images (one decision for every worker, %.1f s): %s"
                 % (time.time() - t1, "; ".join("%s %s (%s)" % (k, "GPU" if v[0] else "CPU", v[1])
                                                 for k, v in res.items())))
        except Exception as exc:  # noqa: BLE001 - workers validate themselves, as before
            os.environ.pop(ENV_DECISION, None)
            _log("  GPU self-test unavailable (%s); each worker validates itself" % exc)
        return True
    except Exception as exc:  # noqa: BLE001
        _log("  GPU server: not started (%s: %s)" % (type(exc).__name__, exc))
        return False


def stop():
    global _PROC, _CONN
    addr = os.environ.pop(ENV_ADDR, None)
    os.environ.pop(ENV_KEY, None)
    os.environ.pop(ENV_DECISION, None)
    try:
        if _CONN is not None:
            _CONN.send(("stop",))
    except Exception:  # noqa: BLE001
        pass
    _CONN = None
    if _PROC is not None:
        _PROC.join(5.0)
        if _PROC.is_alive():
            _PROC.terminate()
    _PROC = None
    if addr and _FAMILY != "AF_PIPE":       # a named pipe leaves no file behind
        try:
            os.unlink(addr)
        except OSError:
            pass


# ----------------------------------------------------------------- client side
def call(op, *args):
    """Run ``op`` on the shared GPU process; raises on any problem."""
    global _CONN
    from multiprocessing.connection import Client
    addr = os.environ.get(ENV_ADDR)
    if not addr:
        raise RuntimeError("no GPU server")
    with _CLOCK:
        if _CONN is None:
            _CONN = Client(addr, family=_FAMILY,
                           authkey=bytes.fromhex(os.environ.get(ENV_KEY, "")))
        try:
            if op != "ping" and shm_enabled() and _SHM["ok"]:
                try:
                    return _call_shm(op, args)
                except _ShmUnavailable:
                    _SHM["ok"] = False          # this process: pickled path from now on
            _CONN.send((op,) + tuple(args))
            st, out = _CONN.recv()
        except Exception:
            STATS["errors"] += 1
            try:
                _CONN.close()
            except Exception:  # noqa: BLE001
                pass
            _CONN = None
            raise
    if st != "ok":
        STATS["errors"] += 1
        raise RuntimeError(out)
    STATS["calls"] += 1
    return out


class _ShmUnavailable(Exception):
    pass


_SHM = {"ok": True, "in": None, "out": None, "calls": 0, "pickled": 0}


def _cap_bytes():
    """Per-buffer ceiling: the buffers stay allocated for the whole run (they
    are RAM), so a rare huge result goes through the pickled path instead."""
    try:
        return int(float(os.environ.get("FLATMORPHAI_GPU_SHM_MAX_MB", "64")) * (1 << 20))
    except ValueError:
        return 64 << 20


class _TooBig(Exception):
    pass


def _grow(slot, need):
    """(Re)create this worker's persistent buffer ``slot`` with >= need bytes."""
    from multiprocessing import shared_memory
    cur = _SHM[slot]
    if cur is not None and cur.size >= need:
        return cur
    if need > _cap_bytes():
        raise _TooBig()
    size = min(max(int(need * 1.25) + 4096, 8 << 20), _cap_bytes())
    new = shared_memory.SharedMemory(create=True, size=size)
    if cur is not None:
        try:
            cur.close()
            cur.unlink()
        except Exception:  # noqa: BLE001
            pass
    _SHM[slot] = new
    return new


def _cleanup_shm():
    for slot in ("in", "out"):
        cur = _SHM.get(slot)
        if cur is not None:
            try:
                cur.close()
                cur.unlink()
            except Exception:  # noqa: BLE001
                pass
            _SHM[slot] = None


def _call_pickled(op, args):
    _SHM["pickled"] += 1
    _CONN.send((op,) + tuple(args))
    st, out = _CONN.recv()
    if st != "ok":
        raise RuntimeError(out)
    STATS["calls"] += 1
    return out


def shm_summary():
    if not _SHM["calls"] and not _SHM["pickled"]:
        return ""
    mb = sum(_SHM[k].size for k in ("in", "out") if _SHM[k] is not None) / 1048576.0
    return "shared-memory transfer %d calls (%d pickled), buffers %.0f MB" % (
        _SHM["calls"], _SHM["pickled"], mb)


def _call_shm(op, args):
    """One request through the persistent shared buffers (lock held)."""
    try:
        st = {"off": 0, "need": 0, "over": False}
        _pack(args, None, st)                         # size of the inputs
        try:
            shm_in = _grow("in", max(st["need"], 1))
        except _TooBig:
            return _call_pickled(op, args)
        st = {"off": 0, "need": 0, "over": False}
        desc = _pack(args, shm_in.buf, st)
        if _SHM["out"] is None:
            _grow("out", 8 << 20)
            import atexit
            atexit.register(_cleanup_shm)
    except Exception as exc:  # noqa: BLE001
        raise _ShmUnavailable(str(exc))
    _CONN.send(("shm", op, shm_in.name, _SHM["out"].name, desc))
    res = _CONN.recv()
    kind = res[0]
    if kind == "err":
        raise RuntimeError(res[1])
    _SHM["calls"] += 1
    if kind == "ok_pickled":
        _SHM["pickled"] += 1
        try:
            _grow("out", res[2])
        except Exception:  # noqa: BLE001 - too big: stays pickled for such results
            pass
        out = res[1]
    else:
        out = _unpack(res[1], _SHM["out"].buf, copy=True)   # buffer reused next call
    STATS["calls"] += 1
    return out
