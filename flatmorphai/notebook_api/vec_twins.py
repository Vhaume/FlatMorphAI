# -*- coding: utf-8 -*-
"""Exact (bit-for-bit) twins of the descriptor loops that make tens of
thousands of tiny numpy calls per frame (profiled on the user's machine):

  * twist_border_evolution._border_evolution_impl : the cross-section loop
    (be_cross_std / be_cross_lrasym / be_cross_lr_flips), ~0.47 s/frame with
    its siblings. The normals are still computed per point by the original
    Python (np.linalg.norm kept as is); the sampling is vectorised (same
    float64 broadcasting as the original scalar expression) and the float32
    mean / std of the masked samples are done in numba with NumPy's own
    pairwise summation (verified bit-identical on 20 000 random arrays).
  * twist_snake_temporal._profile : per-column first/last/mean row of a mask,
    vectorised (integer sums are exact in float64).
  * texture_features.fractal_features : box counting and box means,
    vectorised (same boxes, same order, same float64 arithmetic).

Each function is re-compiled from its ORIGINAL source with only the loop
replaced (pattern not found -> function untouched). FLATMORPHAI_VEC_TWINS=0
disables everything.
"""
from __future__ import annotations

import os
import textwrap

import numpy as np

STATS = {"patched": []}
_K = {}
# NumPy >= 2 (NEP 50): np.float32 op python-float stays float32 ; NumPy 1.x -> float64
_NEP50 = (np.float32(1.0) / 1.5).dtype == np.float32


def enabled():
    return os.environ.get("FLATMORPHAI_VEC_TWINS", "1") in ("1", "true", "True")


def _kernels():
    if _K:
        return _K
    from numba import njit

    @njit(nogil=True, cache=False)
    def _pw(a, lo, n):
        if n < 8:
            res = np.float32(0.0)
            for i in range(n):
                res = np.float32(res + a[lo + i])
            return res
        elif n <= 128:
            r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
            r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
            i = 8
            m = n - (n % 8)
            while i < m:
                r0 = np.float32(r0 + a[lo + i]); r1 = np.float32(r1 + a[lo + i + 1])
                r2 = np.float32(r2 + a[lo + i + 2]); r3 = np.float32(r3 + a[lo + i + 3])
                r4 = np.float32(r4 + a[lo + i + 4]); r5 = np.float32(r5 + a[lo + i + 5])
                r6 = np.float32(r6 + a[lo + i + 6]); r7 = np.float32(r7 + a[lo + i + 7])
                i += 8
            res = np.float32(np.float32(np.float32(r0 + r1) + np.float32(r2 + r3))
                             + np.float32(np.float32(r4 + r5) + np.float32(r6 + r7)))
            while i < n:
                res = np.float32(res + a[lo + i])
                i += 1
            return res
        else:
            n2 = n // 2
            n2 -= n2 % 8
            return np.float32(_pw(a, lo, n2) + _pw(a, lo + n2, n - n2))

    @njit(nogil=True, cache=False)
    def mean32(a):
        return np.float32(_pw(a, 0, a.shape[0]) / np.float32(a.shape[0]))

    @njit(nogil=True, cache=False)
    def std32(a):
        n = a.shape[0]
        m = np.float32(_pw(a, 0, n) / np.float32(n))
        x = np.empty(n, np.float32)
        for i in range(n):
            d = np.float32(a[i] - m)
            x[i] = np.float32(d * d)
        return np.float32(np.sqrt(np.float32(_pw(x, 0, n) / np.float32(n))))

    @njit(nogil=True, cache=False)
    def cross(seg, inb, rows, gm, cs, cla, nep50):
        nr, nc = seg.shape
        half = nc // 2
        buf = np.empty(nc, np.float32)
        lb = np.empty(nc, np.float32)
        rb = np.empty(nc, np.float32)
        for r in range(nr):
            n = 0; nl = 0; nrr = 0
            for j in range(nc):
                if inb[r, j]:
                    buf[n] = seg[r, j]; n += 1
                    if j < half:
                        lb[nl] = seg[r, j]; nl += 1
                    elif j > half:
                        rb[nrr] = seg[r, j]; nrr += 1
            if n > 4:
                sv = std32(buf[:n])
                i = rows[r]
                if nep50:
                    cs[i] = np.float32(sv / np.float32(gm))
                else:
                    cs[i] = np.float64(sv) / gm
                if nl > 1 and nrr > 1:
                    dm = np.float32(mean32(lb[:nl]) - mean32(rb[:nrr]))
                    mm = mean32(buf[:n])
                    if nep50:
                        cla[i] = np.float32(dm / np.float32(mm + np.float32(1e-6)))
                    else:
                        cla[i] = np.float64(dm) / (np.float64(mm) + 1e-6)

    @njit(nogil=True, cache=False)
    def _n32(x0, x1):          # np.linalg.norm of a float32 2-vector (sdot)
        return np.float32(np.sqrt(np.float32(np.float32(x0 * x0) + np.float32(x1 * x1))))

    @njit(nogil=True, cache=False)
    def _turns(Ps, k, i0, i1, wrap, strict, eps):
        """For i in [i0, i1): a = P[i]-P[i-k], b = P[i+k]-P[i] (wrap: modulo),
        keep i when |a||b| passes the test; returns (idx, cos=(a.b)/(|a||b|),
        cross1 = a1*b0 - a0*b1, cross2 = a0*b1 - a1*b0), all float32."""
        n = Ps.shape[0]
        m = i1 - i0
        idx = np.empty(m, np.int64); cv = np.empty(m, np.float32)
        c1 = np.empty(m, np.float32); c2 = np.empty(m, np.float32)
        q = 0
        for i in range(i0, i1):
            im = i - k
            ip = i + k
            if wrap:
                if im < 0:
                    im += n
                ip = ip % n
            a0 = np.float32(Ps[i, 0] - Ps[im, 0]); a1 = np.float32(Ps[i, 1] - Ps[im, 1])
            b0 = np.float32(Ps[ip, 0] - Ps[i, 0]); b1 = np.float32(Ps[ip, 1] - Ps[i, 1])
            na = np.float32(_n32(a0, a1) * _n32(b0, b1))
            ok = (na > eps) if strict else (not (na < eps))
            if not ok:
                continue
            dot = np.float32(np.float32(a0 * b0) + np.float32(a1 * b1))
            idx[q] = i; cv[q] = np.float32(dot / na)
            c1[q] = np.float32(np.float32(a1 * b0) - np.float32(a0 * b1))
            c2[q] = np.float32(np.float32(a0 * b1) - np.float32(a1 * b0))
            q += 1
        return idx[:q], cv[:q], c1[:q], c2[:q]

    @njit(nogil=True, cache=False)
    def _fs_rays(Ps, k, L, worm, eps):
        """_frame_state_impl normals + left/right ray walks (same arithmetic as
        the original: float32 normal, float64 r * step, round half-even)."""
        H, W = worm.shape
        lrL = np.zeros(L, np.float32); lrR = np.zeros(L, np.float32)
        for i in range(k, L - k):
            t0 = np.float32(Ps[i + k, 0] - Ps[i - k, 0]); t1 = np.float32(Ps[i + k, 1] - Ps[i - k, 1])
            v0 = np.float32(-t1); v1 = t0
            nn2 = _n32(v0, v1)
            if nn2 < eps:
                continue
            v0 = np.float32(v0 / nn2); v1 = np.float32(v1 / nn2)
            for side in range(2):
                sg = np.float32(1.0) if side == 0 else np.float32(-1.0)
                s0 = np.float64(np.float32(sg * v0)); s1 = np.float64(np.float32(sg * v1))
                d = 0
                for r in range(1, 60):
                    yy = int(np.rint(np.float64(Ps[i, 0]) + s0 * np.float64(r)))
                    xx = int(np.rint(np.float64(Ps[i, 1]) + s1 * np.float64(r)))
                    if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0:
                        break
                    d = r
                if side == 0:
                    lrL[i] = d
                else:
                    lrR[i] = d
        return lrL, lrR

    _K.update(pw=_pw, mean32=mean32, std32=std32, cross=cross, turns=_turns, fs_rays=_fs_rays)
    return _K


def _eps32():
    # the original compares a float32 with the python float 1e-6: NumPy >= 2
    # casts the python float to float32, NumPy 1.x promotes the float32.
    return np.float32(1e-6) if _NEP50 else 1e-6


def turn_angles(Ps, k, i0, i1, wrap, strict):
    """(idx, theta_deg float32, sign1*theta, sign2*theta) as the original
    scalar loops: theta = degrees(arccos(clip(cos, -1, 1))) by numpy itself."""
    idx, cv, c1, c2 = _kernels()["turns"](np.ascontiguousarray(Ps, np.float32), int(k), int(i0), int(i1),
                                          bool(wrap), bool(strict), _eps32())
    th = np.degrees(np.arccos(np.clip(cv, -1, 1)))
    return idx, th, np.sign(c1) * th, np.sign(c2) * th


def frame_state_loop(Ps, k, L, worm, cur, lrL, lrR):
    idx, th, _s1, _s2 = turn_angles(Ps, k, k, L - k, False, True)
    cur[idx] = th
    a, b = _kernels()["fs_rays"](np.ascontiguousarray(Ps, np.float32), int(k), int(L),
                                 np.ascontiguousarray(worm), _eps32())
    lrL[:] = a
    lrR[:] = b


# ---------------------------------------------------------------- runtime helpers
def cross_section(gray, worm, Ps, idx, NV, offs, H, W, gm, cs, cla):
    """Vectorised body of the be_cross_* loop for the points ``idx`` whose
    normal ``NV`` (float32, already normalised by the original code) is valid."""
    I = np.asarray(idx)
    yy = np.clip((Ps[I, 0][:, None] + offs[None, :] * NV[:, 0][:, None]).round().astype(int), 0, H - 1)
    xx = np.clip((Ps[I, 1][:, None] + offs[None, :] * NV[:, 1][:, None]).round().astype(int), 0, W - 1)
    seg = np.ascontiguousarray(gray[yy, xx].astype(np.float32))
    inb = np.ascontiguousarray(worm[yy, xx] > 0)
    _kernels()["cross"](seg, inb, I.astype(np.int64), float(gm), cs, cla, _NEP50)


def column_profile(mk, x0, x1):
    """width / mid per column of a boolean mask, as the original loop."""
    sub = mk[:, x0:x1 + 1]
    cnt = sub.sum(0)
    has = cnt > 0
    H = sub.shape[0]
    first = np.argmax(sub, 0)
    last = H - 1 - np.argmax(sub[::-1], 0)
    rsum = (sub * np.arange(H, dtype=np.int64)[:, None]).sum(0)
    width = np.full(sub.shape[1], np.nan)
    mid = np.full(sub.shape[1], np.nan)
    width[has] = (last[has] - first[has] + 1)
    mid[has] = rsum[has].astype(np.float64) / cnt[has]
    return width, mid


def box_count(sub, s):
    h, w = sub.shape
    nh, nw = -(-h // s), -(-w // s)
    pad = np.zeros((nh * s, nw * s), bool)
    pad[:h, :w] = sub
    return int(pad.reshape(nh, s, nw, s).any(axis=(1, 3)).sum())


def box_means(g, mm, box):
    """Means of g over the mask inside each box of the original double loop
    (rows 0..H-box-1 step box, same for columns), in the same order."""
    H, W = mm.shape
    ys = list(range(0, H - box, box))
    xs = list(range(0, W - box, box))
    if not ys or not xs:
        return []
    ny, nx = len(ys), len(xs)
    m = mm[:ny * box, :nx * box].reshape(ny, box, nx, box)
    gv = g[:ny * box, :nx * box].astype(np.float64).reshape(ny, box, nx, box)
    cnt = m.sum(axis=(1, 3))
    out = []
    for a in range(ny):
        for b in range(nx):
            if cnt[a, b]:
                out.append(g[ys[a]:ys[a] + box, xs[b]:xs[b] + box][mm[ys[a]:ys[a] + box, xs[b]:xs[b] + box]].mean())
    return out


# ---------------------------------------------------------------- source patches
_P_CROSS = (
    """    for i in range(k, L - k):
        t = Ps[i + k] - Ps[i - k]; nv = np.array([-t[1], t[0]], np.float32); nn2 = np.linalg.norm(nv)
        if nn2 < 1e-6: continue
        nv /= nn2; yy = np.clip((Ps[i, 0] + offs * nv[0]).round().astype(int), 0, H - 1); xx = np.clip((Ps[i, 1] + offs * nv[1]).round().astype(int), 0, W - 1)
        seg = gray[yy, xx].astype(np.float32); inb = worm[yy, xx] > 0
        if inb.sum() > 4:
            cs[i] = seg[inb].std() / gm
            lh = seg[:40][inb[:40]]; rh = seg[41:][inb[41:]]
            if len(lh) > 1 and len(rh) > 1: cla[i] = (lh.mean() - rh.mean()) / (seg[inb].mean() + 1e-6)
""",
    """    _vI = []; _vN = []
    for i in range(k, L - k):
        t = Ps[i + k] - Ps[i - k]; nv = np.array([-t[1], t[0]], np.float32); nn2 = np.linalg.norm(nv)
        if nn2 < 1e-6: continue
        nv /= nn2; _vI.append(i); _vN.append(nv)
    if _vI:
        _VT_.cross_section(gray, worm, Ps, _vI, np.stack(_vN), offs, H, W, gm, cs, cla)
""")

_P_PROFILE = (
    """    for j, x in enumerate(range(x0, x1 + 1)):
        r = np.where(mk[:, x])[0]
        if len(r): width[j] = r.max() - r.min() + 1; mid[j] = r.mean()
""",
    """    width, mid = _VT_.column_profile(mk, x0, x1)
""")

_P_FRACTAL_BOX = (
    """    for s in sizes:
        h,w=sub.shape; c=0
        for yy in range(0,h,s):
            for xx in range(0,w,s):
                if sub[yy:yy+s,xx:xx+s].any(): c+=1
        cnts.append(max(c,1))
""",
    """    for s in sizes:
        cnts.append(max(_VT_.box_count(sub, s),1))
""")


def _same(a, b):
    if a is b:
        return True
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype.kind in "fc":
            return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
        return bool(np.array_equal(a, b))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    try:
        return type(a) is type(b) and bool(a == b)
    except Exception:  # noqa: BLE001
        return False


CHECKS = int(os.environ.get("FLATMORPHAI_VEC_CHECKS", "3") or 3)
STATE = {}          # name -> {"ok": None/True/False, "n": checks done}


def _guard(orig, fast, name):
    """First CHECKS calls: run BOTH and compare bit for bit (the original's
    result is returned); any difference -> the original for good."""
    import functools
    st = STATE.setdefault(name, {"ok": None, "n": 0, "calls": 0})

    @functools.wraps(orig)
    def wrapper(*a, **kw):
        if st["ok"] is False:
            return orig(*a, **kw)
        if st["ok"] is None:
            ref = orig(*a, **kw)
            try:
                got = fast(*a, **kw)
                same = _same(ref, got)
            except Exception:  # noqa: BLE001
                same = False
            st["n"] += 1
            if not same:
                st["ok"] = False
            elif st["n"] >= CHECKS:
                st["ok"] = True
            return ref
        st["calls"] += 1
        return fast(*a, **kw)
    wrapper.__wrapped_original__ = orig
    return wrapper


def status():
    parts = []
    for k, v in sorted(STATE.items()):
        parts.append("%s %s" % (k.split(".")[-1], "ON" if v["ok"] else ("OFF (mismatch)" if v["ok"] is False else "?")))
    return ", ".join(parts)


def _patched(fn, patches, name):
    import inspect
    fn = getattr(fn, "__wrapped_original__", fn)      # always from the ORIGINAL source
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new in patches:
        if src.count(old) != 1:
            return None
        src = src.replace(old, new)
    g = fn.__globals__
    g["_VT_"] = __import__(__name__, fromlist=["cross_section"])
    loc = {}
    exec(compile(src, getattr(fn.__code__, "co_filename", "<vec_twins>"), "exec"), g, loc)  # noqa: S102
    nf = loc.get(fn.__name__)
    if nf is None:
        return None
    STATS["patched"].append(name)
    w = _guard(fn, nf, name)
    w.__vec_twin__ = True
    return w


_P_CURV = (
    """    for i in range(k, L - k):
        a = Ps[i] - Ps[i - k]; b = Ps[i + k] - Ps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
        if na < 1e-6: continue
        th = np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1))); curv[i] = th
        sgnc[i] = np.sign(a[1] * b[0] - a[0] * b[1]) * th
""",
    """    _ci, _cth, _cs1, _cs2 = _VT_.turn_angles(Ps, k, k, L - k, False, False)
    curv[_ci] = _cth; sgnc[_ci] = _cs1
""")

_P_BWAV = (
    """                kk = 3; sg = []
                for i in range(len(cps)):
                    a = cps[i] - cps[i - kk]; b = cps[(i + kk) % len(cps)] - cps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
                    if na > 1e-6: sg.append(np.sign(a[0] * b[1] - a[1] * b[0]) * np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1))))
                sg = np.array(sg); strong = sg[np.abs(sg) > 12]
""",
    """                kk = 3
                _wi, _wth, _ws1, sg = _VT_.turn_angles(cps, kk, 0, len(cps), True, True)
                strong = sg[np.abs(sg) > 12]
""")

_P_FS = (
    """    for i in range(k, L - k):
        a = Ps[i] - Ps[i - k]; b = Ps[i + k] - Ps[i]; na = np.linalg.norm(a) * np.linalg.norm(b)
        if na > 1e-6: cur[i] = np.degrees(np.arccos(np.clip((a @ b) / na, -1, 1)))
        t = Ps[i + k] - Ps[i - k]; nv = np.array([-t[1], t[0]], np.float32); nn2 = np.linalg.norm(nv)
        if nn2 < 1e-6: continue
        nv /= nn2
        for sgn, store in ((1.0, 'L'), (-1.0, 'R')):
            d = 0
            for r in rr:
                yy = int(round(Ps[i, 0] + sgn * nv[0] * r)); xx = int(round(Ps[i, 1] + sgn * nv[1] * r))
                if yy < 0 or xx < 0 or yy >= H or xx >= W or worm[yy, xx] == 0: break
                d = r
            if store == 'L': lrL[i] = d
            else: lrR[i] = d
""",
    """    _VT_.frame_state_loop(Ps, k, L, worm, cur, lrL, lrR)
""")

_TARGETS = (("_frame_state_impl", (_P_FS,)),
            ("_border_evolution_impl", (_P_CROSS, _P_CURV, _P_BWAV)),
            ("_profile", (_P_PROFILE,)),
            ("fractal_features", (_P_FRACTAL_BOX,)))


def patch_modules(log=None, namespaces=()):
    """Patch the target functions in every loaded module (and in the given
    exec namespaces, e.g. the v4j chain). Idempotent."""
    if not enabled():
        return []
    import sys
    done = []
    targets = [(m, getattr(m, "__dict__", None)) for m in list(sys.modules.values()) if m is not None]
    targets += [(None, ns) for ns in namespaces]
    for mod, d in targets:
        if not isinstance(d, dict):
            continue
        for attr, patches in _TARGETS:
            fn = d.get(attr)
            if fn is None or not callable(fn) or getattr(fn, "__vec_twin__", False):
                continue
            base = getattr(fn, "__wrapped_original__", fn)
            if mod is not None and getattr(base, "__module__", None) != getattr(mod, "__name__", None):
                continue
            try:
                nf = _patched(fn, patches, "%s.%s" % (getattr(mod, "__name__", "ns"), attr))
            except Exception:  # noqa: BLE001
                nf = None
            if nf is not None:
                d[attr] = nf
                done.append("%s.%s" % (getattr(mod, "__name__", "ns"), attr))
    if done and log is not None:
        log("  vec twins: %s (exact vectorised/numba loops)" % ", ".join(done))
    return done
