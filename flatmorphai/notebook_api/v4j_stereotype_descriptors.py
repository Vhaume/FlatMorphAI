# -*- coding: utf-8 -*-
"""v4j stereotype descriptors the app did not compute for the older model.

The v4j LightGBM model reads 217 descriptors; 54 of them are NEW families
(ag_, cm_, cr_, fw_, l_, lp_, ond_, pl_, rs_, ru_, ta_, tn_, tt_, wn_, wz_, zl_)
produced by the v4j descriptor chain (lightgbm_stereotype_classifier/v4j_chain/).
This module reuses the EXACT per-frame functions of those programs -- never a
retyped formula -- so a value matches the chain by construction, as long as it is
fed the same inputs. The inputs the chain uses are available in the app:

  * the 384 px formatted image (mask applied, colour) -- the chain's format_384 /
    couleur_384 with the SAME parameters (image_size 384, margin 0, apply_mask
    True, b&w False); the app writes it to each video's
    ``Formated_image-size(384)_margin(0.0)_b&w(False)_apply-mask(True)/`` folder,
  * the T-1 / T0 / T+1 neighbours (ordered by second), and
  * the registered motion image (prev/cur/next aligned).

The functions are loaded by slicing their source (``_load_funcs``) rather than
importing the program, because importing a chain program scans the training
dataset at module load.
"""
from __future__ import annotations

import ast
import os
from typing import Dict, List

# Geodesic memo hit/miss counters (diagnostic; printed when
# FLATMORPHAI_V4J_GEOD_STATS=1). Shows how many of the per-frame geodesic calls
# the content cache collapses -- the geodesic is ~42% of the v4j chain.
_GEOD_STATS = {"hit": 0, "miss": 0}

# Batch-fill histogram of the geodesic GPU server (diagnostic): batch size -> count
# of GPU calls made at that size. A healthy server fills batches (>=5); all-1
# means the collect window is too short under the GIL and the GPU loses.
_GEOD_SRV_BATCHES: "dict" = {}


def _chain_dir() -> str:
    """The vendored v4j chain folder inside the bundle."""
    here = os.path.dirname(os.path.abspath(__file__))
    # flatmorphai/notebook_api/ -> <app root>/lightgbm_stereotype_classifier/v4j_chain
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, "lightgbm_stereotype_classifier", "v4j_chain")


def _load_funcs(filename: str, want, bundle_dir: str = "") -> dict:
    """Return named callables/constants from a chain program WITHOUT running its
    module body (which execs the previous program and scans the dataset).

    Parses the file, keeps only imports / assignments / function / class
    definitions, drops every other top-level statement (the module-level
    ``exec`` of the previous program, the dataset-index building, ``main`` call),
    then execs the reduced tree. Returns only the requested names.
    """
    path = os.path.join(bundle_dir or _chain_dir(), filename)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    _BAD = ("open(", "exec(", "compile(", "eval(", ".read(", "getcwd",
            "ROOT", "os.environ", "pd.read", "glob",
            "_L[", "_G[", "_C[", "_D[", "_mm[", "_src")

    def _keep(n):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                          ast.AsyncFunctionDef, ast.ClassDef)):
            return True
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            # keep only pure constant assignments; drop ones that do I/O or
            # read the dataset (``_src = open(...)``, module-level exec setup).
            try:
                src = ast.unparse(n.value) if n.value is not None else ""
            except Exception:  # noqa: BLE001
                return False
            return not any(b in src for b in _BAD)
        return False

    tree.body = [n for n in tree.body if _keep(n)]
    ast.fix_missing_locations(tree)
    ns: dict = {"__name__": "_v4j_%s" % filename}
    exec(compile(tree, path, "exec"), ns)  # noqa: S102
    return {w: ns[w] for w in want if w in ns}


# --------------------------------------------------------------------------
# wn_ : Walnut descriptors (single frame T0, 384 px colour, mask applied).
# walnut(bgr) is fully self-contained (cv2, numpy, scipy only), so it is reused
# as-is on the app's formatted image.
# --------------------------------------------------------------------------
_WN = None
# The 9 wn_ descriptors the v4j model reads (of walnut()'s 14 outputs).
WN_USED = ["wn_courbure_cv", "wn_dark03", "wn_dark10", "wn_droit", "wn_red_med",
           "wn_red_p90", "wn_red_q80_spread", "wn_rouge_gamma_frac",
           "wn_rouge_gamma_spread"]


def _walnut_fn():
    global _WN
    if _WN is None:
        d = _load_funcs("calcul_descripteurs_walnut_stereotypies_v3j.py",
                        ("walnut",))
        _WN = d.get("walnut")
    return _WN


def compute_wn(formatted_bgr) -> Dict[str, float]:
    """The wn_ descriptors from the 384 px formatted colour image, or {}."""
    fn = _walnut_fn()
    if fn is None or formatted_bgr is None:
        return {}
    try:
        out = fn(formatted_bgr)
        return {k: float(out[k]) for k in WN_USED if k in out}
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------
# tt_ : temporal twist/snake metrics from the registered motion image
# (B = aligned prev, G = current, R = aligned next). twist_snake_temporal() is
# self-contained; the app already builds that motion image (see _build_motion_image
# in lightgbm_bundle), so it is reused directly.
# --------------------------------------------------------------------------
_TT = None
TT_USED = ["tt_area_osc", "tt_midline_lat_max", "tt_midline_lat_osc",
           "tt_width_zone_max", "tt_width_zone_osc"]


def _tt_fn():
    global _TT
    if _TT is None:
        d = _load_funcs("twist_snake_temporal.py", ("twist_snake_temporal",))
        _TT = d.get("twist_snake_temporal")
        try:   # exact vectorised _profile (FLATMORPHAI_VEC_TWINS=0 disables)
            from . import vec_twins as _VT
            if _TT is not None:
                _VT.patch_modules(namespaces=[_TT.__globals__])
        except Exception:  # noqa: BLE001
            pass
    return _TT


def compute_tt(motion_bgr) -> Dict[str, float]:
    fn = _tt_fn()
    if fn is None or motion_bgr is None:
        return {}
    try:
        out = fn(motion_bgr)
        return {k: float(out[k]) for k in TT_USED if k in out}
    except Exception:  # noqa: BLE001
        return {}


# Families still to be wired (need the shared chain helpers format_384 / axe /
# _objet / oriente / NS_MOT motion alignment, extracted the same way):
#   ta_ (torsion_onde v3z), tt_ (twist_snake_temporal), cm_ (twist_snake_metrics),
#   rs_/lp_/tn_ (commentaires v3g), ag_ (angles_persistance v4a),
#   cr_/fw_/zl_ (croissant_voisines v3s), l_ (axe_pointes v4c),
#   ru_ (rupture_coupes v4d), pl_ (ligne_pli v4e), ond_ (ondulation v3o),
#   wz_ (mouvement_largeur v3n).
#   DONE: wn_ (single frame), tt_ (motion image).
_PENDING_FAMILIES: List[str] = [
    "ta_", "cm_", "rs_", "lp_", "tn_", "ag_", "cr_", "fw_", "zl_",
    "l_", "ru_", "pl_", "ond_", "wz_",
]


# --------------------------------------------------------------------------
# Axis / temporal families: call each chain program's own ``calcule`` with the
# app's I/O injected (format_384 = grey of the Formated image, voisins = the
# app's T-1/T+1 neighbours, NS_MOT = make_motion's pose/align/encode, and the
# chain helpers _objet/axe/_geod/oriente), so the output matches the chain.
# --------------------------------------------------------------------------
def _ns_mot() -> dict:
    """pose / align_to / encode_motion, copied verbatim from the bundle's
    make_motion_dataset.py (the module itself can't be imported -- its top
    import of format_dataset_size, used only for dataset paths, is absent)."""
    import cv2
    import numpy as np

    def pose(gray):
        m = cv2.moments((gray > 8).astype(np.uint8), binaryImage=True)
        if m['m00'] < 50:
            return None
        cx = m['m10'] / m['m00']; cy = m['m01'] / m['m00']
        mu20 = m['mu20'] / m['m00']; mu02 = m['mu02'] / m['m00']; mu11 = m['mu11'] / m['m00']
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        ecc = ((mu20 - mu02) ** 2 + 4 * mu11 ** 2) ** 0.5 / (mu20 + mu02 + 1e-9)
        return cx, cy, float(theta), float(ecc)

    def align_to(src_gray, src_pose, dst_pose, dst_gray):
        H, W = src_gray.shape
        scx, scy, sth, sec = src_pose; dcx, dcy, dth, dec = dst_pose
        angles = [0.0] if min(sec, dec) < 0.15 else [dth - sth, dth - sth + np.pi]
        cand = []
        for a in angles:
            M = cv2.getRotationMatrix2D((scx, scy), np.degrees(a), 1.0)
            M[0, 2] += (dcx - scx); M[1, 2] += (dcy - scy)
            cand.append(cv2.warpAffine(src_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0))
        if len(cand) == 1:
            return cand[0]
        dm = dst_gray > 8
        sads = [float(np.abs((c - dst_gray)[dm]).mean()) if dm.any() else np.abs(c).mean() for c in cand]
        return cand[int(np.argmin(sads))]

    def encode_motion(R, gt, B, worm):
        out = np.zeros((gt.shape[0], gt.shape[1], 3), np.uint8)
        out[..., 0] = (np.clip(R, 0, 255) * worm).astype(np.uint8)
        out[..., 1] = (np.clip(gt, 0, 255) * worm).astype(np.uint8)
        out[..., 2] = (np.clip(B, 0, 255) * worm).astype(np.uint8)
        return out

    return {"pose": pose, "align_to": align_to, "encode_motion": encode_motion}


class _GeodServer:
    """Single GPU-owner micro-batching server for the per-call geodesic.

    The per-call geodesic (``_geod_crop``) is the single biggest cost of the v4j
    chain (~31% of its time, MCP_Geometric on the CPU), and it is launched
    dynamically ~14x/frame by the 9 programs. The 7 pool threads calling the GPU
    each on their own frame CONTEND (measured: 7-thread GPU geodesic is x10
    SLOWER than one owner batching). This server resolves that exactly as asked:
    the threads SUBMIT ``(mask, source)`` and block; ONE owner thread collects
    the requests that arrive within a short window and runs them as a SINGLE
    batched GPU geodesic (``geodesic_gpu._gpu_geod_multi``), then hands each
    thread its full-shape field. GPU work is funnelled to one owner (no
    contention) AND batched (the ~10x win); the threads keep doing the rest
    (axe_pointes, nonzero, ...) in parallel.

    Opt-in via FLATMORPHAI_V4J_GEOD_SERVER=1. On any error, no CUDA, or an
    unservable request it serves on the CPU (``cpu_fn``) and never leaves a
    caller blocked. NOTE: the GPU geodesic is the same float32 chamfer field the
    axis GPU path already uses (that gave identical class distributions), but it
    now feeds every geodesic-derived descriptor, so a run MUST be checked against
    the benchmark class distributions before trusting it."""

    def __init__(self, cpu_fn, window=0.005, maxbatch=24):
        import threading
        import queue
        self._cpu = cpu_fn
        # window (s) and max batch are tunable: under the GIL a longer window
        # lets more pool threads reach their geodesic and fill the batch.
        self._win = float(os.environ.get("FLATMORPHAI_V4J_GEOD_WINDOW", window))
        self._maxb = int(os.environ.get("FLATMORPHAI_V4J_GEOD_MAXBATCH", maxbatch))
        self._q = queue.Queue()
        self._gpu_ok = None            # None=untried, True=ok, False=CPU only
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def geod(self, m, depart):
        if self._gpu_ok is False:
            return self._cpu(m, depart)
        import threading
        ev = threading.Event()
        box = {}
        self._q.put((m, depart, box, ev))
        ev.wait()
        return box.get("r")

    def _loop(self):
        import time
        import queue
        import numpy as np
        from . import geodesic_gpu as _gg
        while True:
            first = self._q.get()
            batch = [first]
            deadline = time.time() + self._win
            while len(batch) < self._maxb:
                to = deadline - time.time()
                if to <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=to))
                except queue.Empty:
                    break
            self._serve(batch, _gg, np)

    def _serve(self, batch, _gg, np):
        if self._gpu_ok is None:
            try:
                self._gpu_ok = bool(_gg._try_init())
            except Exception:  # noqa: BLE001
                self._gpu_ok = False
        if not self._gpu_ok:
            for (m, depart, box, ev) in batch:
                try:
                    box["r"] = self._cpu(m, depart)
                except Exception:  # noqa: BLE001
                    box["r"] = None
                ev.set()
            return
        crops = []; srcs = []; meta = []
        for (m, depart, box, ev) in batch:
            try:
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    box["r"] = self._cpu(m, depart); ev.set(); continue
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                crops.append(m[y0:y1, x0:x1])
                srcs.append((int(depart[0]) - y0, int(depart[1]) - x0))
                meta.append((m, box, ev, y0, x0, depart))
            except Exception:  # noqa: BLE001
                try:
                    box["r"] = self._cpu(m, depart)
                except Exception:  # noqa: BLE001
                    box["r"] = None
                ev.set()
        if not crops:
            return
        _GEOD_SRV_BATCHES[len(crops)] = _GEOD_SRV_BATCHES.get(len(crops), 0) + 1
        try:
            fields = _gg._gpu_geod_multi(crops, srcs)
        except Exception:  # noqa: BLE001 -- GPU problem: permanent CPU fallback
            self._gpu_ok = False
            for (m, box, ev, y0, x0, depart) in meta:
                try:
                    box["r"] = self._cpu(m, depart)
                except Exception:  # noqa: BLE001
                    box["r"] = None
                ev.set()
            return
        for (m, box, ev, y0, x0, depart), fc in zip(meta, fields):
            try:
                out = np.full(m.shape, np.inf)
                out[y0:y0 + fc.shape[0], x0:x0 + fc.shape[1]] = fc
                out[~m] = np.inf
                box["r"] = out
            except Exception:  # noqa: BLE001
                try:
                    box["r"] = self._cpu(m, depart)
                except Exception:  # noqa: BLE001
                    box["r"] = None
            ev.set()


class V4jChain:
    # program file -> the v4j-model columns it provides (of all it computes).
    _PROGRAMS = {
        "calcul_descripteurs_commentaires_stereotypies_v3g.py":
            ["cm_axial_tex_step", "cm_border_wav", "cm_border_wav_amp",
             "cm_darkwide", "cm_max_turn", "cm_n_sharp", "cm_neck_depth",
             "cm_neck_len", "rs_area_max", "rs_disp", "rs_disp0", "rs_len_max",
             "rs_thin", "rs_wmax", "rs_wmean", "lp_dark_rel", "lp_dark_rel_rng",
             "lp_lum_rng", "tn_neck_chg"],
        "calcul_descripteurs_torsion_onde_stereotypies_v3z.py": ["ta_saut_w"],
        "calcul_descripteurs_angles_persistance_stereotypies_v4a.py":
            ["ag_double_min", "ag_double_t0", "ag_opp_min"],
        "calcul_descripteurs_axe_pointes_stereotypies_v4c.py":
            ["l_bras_min", "l_deux_cotes_t0", "l_symetrie_t0"],
        "calcul_descripteurs_rupture_coupes_stereotypies_v4d.py":
            ["ru_bord_deux_t0"],
        "calcul_descripteurs_ligne_pli_stereotypies_v4e.py":
            ["pl_contraste_t0", "pl_long_lignes", "pl_n_t0"],
        "calcul_descripteurs_ondulation_persistance_stereotypies_v3o.py":
            ["ond_courb_dist", "ond_inflex_moy"],
        "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py":
            ["wz_col_chg"],
        "calcul_descripteurs_croissant_voisines_stereotypies_v3s.py":
            ["cr_courb_min", "cr_meme_sens", "cr_resid_max", "fw_hu_med",
             "fw_vit_med", "zl_coin_frac", "zl_coin_max"],
    }

    # cross-program functions a program pulls from ANOTHER chain program's
    # module body (the pull is an exec of that program, dropped by the AST
    # filter). We extract them in isolation and inject them per program, so a
    # program's own definitions are never clobbered (unlike the shared helpers).
    #   v4a pulls axe_profil / corr / NB from v3z;
    #   v4d and v4e pull axe_pointes from v4c.
    _PROG_EXTRA = {
        "calcul_descripteurs_angles_persistance_stereotypies_v4a.py":
            ("calcul_descripteurs_torsion_onde_stereotypies_v3z.py",
             ("axe_profil", "corr", "NB")),
        "calcul_descripteurs_rupture_coupes_stereotypies_v4d.py":
            ("calcul_descripteurs_axe_pointes_stereotypies_v4c.py",
             ("axe_pointes",)),
        "calcul_descripteurs_ligne_pli_stereotypies_v4e.py":
            ("calcul_descripteurs_axe_pointes_stereotypies_v4c.py",
             ("axe_pointes",)),
    }

    def __init__(self):
        self._helpers = None          # chain helpers _objet/axe/_geod/oriente + NS_MOT
        self._ns = {}                 # program file -> execed namespace (functions)
        self._ok = True
        self._ctx = None              # (images_dir, masks_dir, formatted_dir, frames, mask_folder)
        import threading
        self._tl = threading.local()  # per-worker per-frame memo caches
        self._gpu_axe = None          # None=untried/validating, True=use, False=off
        self._gpu_lock = threading.Lock()
        # Video-level SHARED axis cache (mask-bytes -> axe dict), filled once by a
        # single owner thread in a batched GPU pass (``_prepass_axes``) before the
        # frame pool runs. The per-frame workers then READ this cache instead of
        # each of the 7 pool threads calling the GPU on its own frame -- measured
        # ~10x faster on the axis (7-thread GPU contention vs one batched owner).
        self._axe_shared: dict = {}
        self._axe_shared_lock = threading.Lock()
        self._prepass_done = False

    def _caches(self):
        """Per-thread memo caches (one worker processes one frame at a time, so
        these collapse the duplicate format_384 / _objet / axe calls the 9
        programs make for the SAME T-1/T0/T+1 masks within a single frame).
        They are cleared at the start of every ``compute`` call, which also keeps
        ``id()`` keys from aliasing across frames (a freed array's id could be
        reused)."""
        tl = self._tl
        if not getattr(tl, "ready", False):
            tl.f384 = {}
            tl.f384f = {}
            tl.mbrut = {}
            tl.objet = {}
            tl.axe = {}
            tl.axe_pre = {}      # mask-bytes -> axe dict, from the GPU batch
            tl.geod = {}         # (mask-bytes, depart) -> geodesic field (per-frame)
            tl.ready = True
        return tl

    def _ensure_loaded(self):
        if self._helpers is not None or not self._ok:
            return
        try:
            cd = _chain_dir()
            import sys as _sys
            if cd not in _sys.path:
                _sys.path.insert(0, cd)
            helpers = {}
            helpers.update(_load_funcs(
                "calcul_descripteurs_commentaires_stereotypies_v3g.py",
                ("_objet", "axe", "_geod")))
            helpers.update(_load_funcs(
                "calcul_descripteurs_mouvement_largeur_stereotypies_v3n.py",
                ("oriente",)))
            helpers["NS_MOT"] = _ns_mot()
            # import the app's image formatter (CIM) used by format_384_fond
            from . import classification_images as _cim
            helpers["CIM"] = _cim
            # Memoize the two most expensive pure per-frame functions by the
            # identity of their (cached) input array, so the axis/geodesic of a
            # given mask is computed once per frame instead of ~14x across the 9
            # programs. Results are treated read-only by the chain (verified: the
            # descriptor values are unchanged), so sharing them is safe.
            _raw_objet = helpers["_objet"]
            _orig_axe = helpers["axe"]
            _orig_geod = helpers["_geod"]

            def _geod_crop_raw(m, depart):
                # Bit-identical speedup of v3g._geod: the geodesic only ever runs
                # through in-mask (finite-cost) cells, all inside the mask's
                # bounding box, so running MCP_Geometric on that box instead of
                # the full 384x384 image gives the SAME in-mask distances (and
                # the same inf elsewhere), on far fewer cells. Validated bit-for-
                # bit (max abs diff 0.0) on every local frame.
                import numpy as np
                from skimage.graph import MCP_Geometric
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    return _orig_geod(m, depart)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                mc = m[y0:y1, x0:x1]
                dc, _ = MCP_Geometric(np.where(mc, 1.0, np.inf)).find_costs(
                    [(int(depart[0]) - y0, int(depart[1]) - x0)])
                out = np.full(m.shape, np.inf)
                out[y0:y1, x0:x1] = np.asarray(dc, float)
                out[~m] = np.inf
                return out

            def _geod_crop(m, depart):
                # Per-frame memoization: the 9 programs call the geodesic ~14x on
                # the SAME (mask, source), and it was NOT cached before (only axe
                # and _objet were). Keying by mask CONTENT + source collapses the
                # repeats. BIT-IDENTICAL (pure function of its inputs), pure CPU,
                # no GPU/GIL involved -- the geodesic is ~42% of the v4j chain.
                try:
                    import numpy as np
                    tl = self._caches()
                    key = (m.tobytes(), int(depart[0]), int(depart[1]))
                    hit = tl.geod.get(key)
                    if hit is not None:
                        _GEOD_STATS["hit"] += 1
                        return hit
                    _GEOD_STATS["miss"] += 1
                    v = _geod_crop_raw(m, depart)
                    tl.geod[key] = v
                    return v
                except Exception:  # noqa: BLE001 - never let the cache break a run
                    return _geod_crop_raw(m, depart)

            # Per-call geodesic: CPU by default (bbox-cropped MCP_Geometric), bit
            # exact. A single geodesic is launch-bound on the GPU and loses when
            # each pool thread calls it on its own frame (measured x10 SLOWER by
            # contention). FLATMORPHAI_V4J_GEOD_SERVER=1 funnels every geodesic
            # through ONE GPU owner that batches the concurrent requests (see
            # _GeodServer) -- the threads submit and the owner serves them in one
            # batched GPU call. Opt-in; the GPU field is the float32 chamfer the
            # axis path already uses, so a run must be checked against the
            # benchmark classes before trusting it.
            import os as _os
            if _os.environ.get("FLATMORPHAI_V4J_GEOD_SERVER", "0") in ("1", "true", "True"):
                self._geod_server = _GeodServer(_geod_crop)

                def _geod(m, depart):
                    return self._geod_server.geod(m, depart)
            else:
                _geod = _geod_crop
            helpers["_geod"] = _geod

            def _axe_from_field(m, d, a, NB=40):
                # The CPU-exact tail of axe: given the geodesic field ``d`` from
                # endpoint ``a`` (either the CPU MCP field or a GPU-batched one),
                # bin the mask along the axis and measure the local widths. The
                # per-bin arg-max is vectorised (stable argsort preserves the
                # row-major order, matching v3g.axe's np.nonzero(q==k)+argmax).
                import numpy as np
                import cv2
                b = np.unravel_index(
                    int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape)
                L = float(d[b])
                assert L > 10, "axis too short"
                dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                q = np.full(m.shape, -1)
                q[m] = np.minimum((d[m] / L * NB).astype(int), NB - 1)
                P = np.zeros((NB, 2)); w = np.zeros(NB)
                ys2, xs2 = np.nonzero(m)           # row-major, same as the scan
                bins = q[ys2, xs2]
                dv = dt[ys2, xs2]
                order = np.argsort(bins, kind="stable")
                bs = bins[order]; dvs = dv[order]
                yso = ys2[order]; xso = xs2[order]
                lo = np.searchsorted(bs, np.arange(NB), side="left")
                hi = np.searchsorted(bs, np.arange(NB), side="right")
                for k in range(NB):
                    s, en = int(lo[k]), int(hi[k])
                    if en > s:
                        j = int(np.argmax(dvs[s:en]))
                        P[k] = (yso[s + j], xso[s + j])
                        w[k] = 2.0 * dvs[s + j]
                    else:
                        P[k] = P[k - 1] if k else a
                        w[k] = w[k - 1] if k else 1.0
                e = max(1, int(0.15 * NB))
                wmed = float(np.median(w[e:NB - e])) + 1e-6
                return dict(a=np.array(a, float), b=np.array(b, float), L=L,
                            P=P, w=w, wmed=wmed, dt=dt, e=e,
                            bend=1.0 - float(np.hypot(
                                *(np.array(a, float) - np.array(b, float)))) / L)

            def axe_fast(m, NB=40):
                # Two geodesic passes (arbitrary start -> endpoint a; a -> field),
                # then the CPU-exact binning. _geod is GPU or CPU per-call.
                import numpy as np
                ys, xs = np.nonzero(m)
                d = _geod(m, (ys[0], xs[0]))
                a = np.unravel_index(
                    int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape)
                d = _geod(m, a)
                return _axe_from_field(m, d, a, NB)

            self._axe_from_field = _axe_from_field
            self._axe_fast = axe_fast

            def _axe_cpu(m, NB=40):
                # exact CPU axis (bbox-cropped MCP geodesic), the trusted
                # reference the GPU-batched axes are validated against.
                import numpy as np
                ys, xs = np.nonzero(m)
                d = _geod_crop(m, (ys[0], xs[0]))
                a = np.unravel_index(
                    int(np.argmax(np.where(np.isfinite(d), d, -1))), d.shape)
                d = _geod_crop(m, a)
                return _axe_from_field(m, d, a, NB)

            self._axe_cpu = _axe_cpu
            _raw_axe = axe_fast

            def _objet_memo(arr, _raw=_raw_objet):
                c = self._caches().objet
                k = id(arr)
                hit = c.get(k)
                if hit is not None:
                    return hit[1]
                v = _raw(arr)
                c[k] = (arr, v)  # keep arr alive so its id stays valid/unique
                return v

            def _axe_memo(m, *a, **k):
                # all chain call sites use the default NB=40; only memoize that
                # case so a non-default call can never get a wrong cached axis.
                if a or (k and k != {"NB": 40}):
                    return _raw_axe(m, *a, **k)
                tl = self._caches()
                key = id(m)
                hit = tl.axe.get(key)
                if hit is not None:
                    return hit[1]
                # per-frame GPU-batched precompute: look up by mask CONTENT
                # (the arrays are freshly built each call, so id won't match).
                pre = getattr(tl, "axe_pre", None)
                if pre:
                    v = pre.get(m.tobytes())
                    if v is not None:
                        tl.axe[key] = (m, v)
                        return v
                v = _raw_axe(m)
                tl.axe[key] = (m, v)
                return v

            helpers["_objet"] = _objet_memo
            helpers["axe"] = _axe_memo
            self._helpers = helpers
            # cache the cross-program extras (axe_pointes from v4c, axe_profil/
            # corr/NB from v3z), extracted once in isolation.
            self._xtra_cache: dict = {}
            for prog in self._PROGRAMS:
                ns = self._load_program_ns(prog)
                spec = self._PROG_EXTRA.get(prog)
                if spec is not None:
                    src_file, names = spec
                    if src_file not in self._xtra_cache:
                        self._xtra_cache[src_file] = _load_funcs(src_file, names)
                    # inject AFTER the helper re-assert so it is never clobbered
                    ns.update({k: v for k, v in self._xtra_cache[src_file].items()
                               if k in names})
                self._ns[prog] = ns
            # axe_pointes (8 weighted Dijkstra per mask) is called by v4c, v4d
            # AND v4e on the same T-1/T0/T+1 masks (~9 calls/frame), and frame
            # t's T+1 mask is frame t+1's T0. It is pure -> share one bounded,
            # copy-on-hit cache keyed by mask content + code signature: exact.
            if os.environ.get("FLATMORPHAI_DESC_MEMO", "1") in ("1", "true", "True"):
                try:
                    from . import memo_cache as _MC
                    _fast_ok = (os.environ.get("FLATMORPHAI_MCP_NUMBA", "1") in ("1", "true", "True")
                                and os.environ.get("FLATMORPHAI_FAST_AXE", "1") in ("1", "true", "True"))
                    for _nsx in self._ns.values():
                        for _fname in ("axe_pointes", "axe_profil"):
                            _f = _nsx.get(_fname)
                            if callable(_f):
                                if _fname == "axe_pointes" and _fast_ok:
                                    # exact twin: no Dijkstra from the last tip,
                                    # early stop once the target tips are settled
                                    try:
                                        from .mcp_numba import make_fast_axe_pointes
                                        _f = make_fast_axe_pointes(_f)
                                    except Exception:  # noqa: BLE001
                                        pass
                                _nsx[_fname] = _MC.memoize(_f, _fname)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            self._ok = False
            self._helpers = None

    def _load_program_ns(self, filename: str) -> dict:
        """Exec a program's definitions (minus its dataset body) into a namespace
        pre-seeded with the chain helpers; the app I/O helpers are injected per
        video in set_video()."""
        import ast as _ast
        path = os.path.join(_chain_dir(), filename)
        with open(path, encoding="utf-8") as fh:
            tree = _ast.parse(fh.read())
        _BAD = ("open(", "exec(", "compile(", "eval(", ".read(", "getcwd",
                "ROOT", "os.environ", "pd.read", "glob",
                "_L[", "_G[", "_C[", "_D[", "_Z[", "_mm[", "_src")

        def _keep(n):
            if isinstance(n, (_ast.Import, _ast.ImportFrom, _ast.FunctionDef,
                              _ast.AsyncFunctionDef, _ast.ClassDef)):
                return True
            if isinstance(n, (_ast.Assign, _ast.AnnAssign)):
                try:
                    src = _ast.unparse(n.value) if n.value is not None else ""
                except Exception:  # noqa: BLE001
                    return False
                return not any(b in src for b in _BAD)
            return False

        tree.body = [n for n in tree.body if _keep(n)]
        _ast.fix_missing_locations(tree)
        ns: dict = {"__name__": "_v4jchain_%s" % filename}
        ns.update(self._helpers)      # _objet/axe/_geod/oriente/NS_MOT/CIM
        exec(compile(tree, path, "exec"), ns)  # noqa: S102
        ns.update(self._helpers)      # re-assert (a kept Assign may shadow)
        return ns

    def set_video(self, images_dir, masks_dir, formatted_dir, frames):
        """``frames`` is the list of frame file names ordered by acquisition
        second (same order the app uses). Builds the app I/O helpers and injects
        them into every program namespace."""
        self._ensure_loaded()
        if not self._ok:
            return
        import cv2
        import numpy as np
        order = list(frames)
        idx = {n: i for i, n in enumerate(order)}

        def _mask_path(nm):
            p = os.path.join(masks_dir, nm)
            return p

        try:   # file bytes through the per-process read cache (path + mtime)
            from . import lightgbm_bundle as _LBM
        except Exception:  # noqa: BLE001
            _LBM = None

        def _lire(path, flag=cv2.IMREAD_GRAYSCALE):
            try:
                buf = _LBM._readbytes(path) if _LBM is not None else np.fromfile(path, np.uint8)
                return cv2.imdecode(buf, flag)
            except Exception:  # noqa: BLE001
                return None

        def fixe(nm):
            return os.path.join(images_dir, nm), _mask_path(nm)

        def format_384(nm):
            c = self._caches().f384
            hit = c.get(nm)
            if hit is not None:
                return hit
            im = _lire(os.path.join(formatted_dir, nm), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError("no formatted image: " + nm)
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            c[nm] = g
            return g

        def format_384_fond(nm):
            c = self._caches().f384f
            hit = c.get(nm)
            if hit is not None:
                return hit
            raw = _lire(os.path.join(images_dir, nm), cv2.IMREAD_COLOR)
            m = _lire(_mask_path(nm))
            obj = self._helpers["CIM"].biggest_objects_in_the_binary_mask(
                m, num_objects_to_keep=1, num_objects_by_row=1)
            out = self._helpers["CIM"].compute_mask_image_planarian_reshaped(
                raw, obj[0], image_size=384, margin_percent=0.0,
                apply_mask=False, black_and_white=False)
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            c[nm] = g
            return g

        def voisins(nm):
            i = idx.get(nm, 0)
            pv = order[i - 1] if i > 0 else order[min(i + 1, len(order) - 1)]
            nx = order[i + 1] if i + 1 < len(order) else order[max(i - 1, 0)]
            return pv, nx

        def masque_brut(nm):
            c = self._caches().mbrut
            if nm in c:
                return c[nm]
            m = _lire(_mask_path(nm))
            v = (m > 127) if m is not None else None
            c[nm] = v
            return v

        def parse_frame(nm):
            # (video id, frame index, total) -- v3s' choix_frames uses only the
            # frame index (element [1]) to order the T-1/T+1 neighbours, which is
            # exactly the app's acquisition order.
            return 0, idx.get(nm, 0), len(order)

        inj = dict(format_384=format_384, format_384_fond=format_384_fond,
                   voisins=voisins, fixe=fixe, _lire=_lire,
                   masque_brut=masque_brut, parse_frame=parse_frame)
        for ns in self._ns.values():
            ns.update(inj)
            ns.update(self._helpers)
        self._ctx = (images_dir, masks_dir, formatted_dir, order)
        # Owner-thread batched GPU axis pre-pass (no-op unless the GPU axis is
        # opted in). Runs here, in the single preparing thread, BEFORE the frame
        # pool, so the GPU is driven by one owner in big batches instead of being
        # hammered by every pool worker.
        self._prepass_axes()

    def _prepass_axes(self, log=None):
        """Owner-thread, up-front batched GPU axis pass over EVERY frame's masks.

        Fills ``self._axe_shared`` (mask-bytes -> axe dict) in big GPU batches
        from this single preparing thread, so the per-frame pool reads the axis
        instead of each of the pool threads calling the GPU on its own frame.
        Measured ~10x faster on the axis (7-thread GPU contention vs one batched
        owner). Gated by FLATMORPH_V4J_GPU_GEOD=1 -- same opt-in and same
        batched_geod_fields + validation as the per-frame GPU path (which gave
        identical class distributions), so fidelity is unchanged; it only moves
        the work to one owner. Best-effort: any problem leaves the shared cache
        empty and the per-frame path (GPU or exact CPU) is used as before."""
        import os as _os
        if _os.environ.get("FLATMORPH_V4J_GPU_GEOD", "0") not in ("1", "true", "True"):
            return
        if self._prepass_done or self._gpu_axe is False or not self._ns or self._ctx is None:
            return
        import numpy as np
        try:
            from . import geodesic_gpu as _gg
            ns0 = next(iter(self._ns.values()))
            fmt = ns0["format_384"]; obj = ns0["_objet"]; mb = ns0["masque_brut"]
            order = self._ctx[3]
            items = []; seen = set()
            for x in order:
                cands = []
                try:
                    cands.append(obj(fmt(x)))
                except Exception:  # noqa: BLE001
                    pass
                m = mb(x)
                if m is not None:
                    cands.append(m)
                for m in cands:
                    if m is None:
                        continue
                    k = m.tobytes()
                    if k in seen:
                        continue
                    seen.add(k); items.append((k, m))
            if not items:
                return
            built = []           # (key, mask, axe) across all frames
            CH = 8               # masks per GPU batch
            for i in range(0, len(items), CH):
                chunk = items[i:i + CH]
                cropped = []; offs = []; starts = []
                for _k, m in chunk:
                    ys, xs = np.nonzero(m)
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    cropped.append(m[y0:y1, x0:x1]); offs.append((y0, x0))
                    starts.append((int(ys[0]) - y0, int(xs[0]) - x0))
                f1 = _gg.batched_geod_fields(cropped, starts)
                if f1 is None:
                    self._gpu_axe = False
                    return
                a_crop = [np.unravel_index(
                    int(np.argmax(np.where(np.isfinite(f), f, -1))), f.shape)
                    for f in f1]
                f2 = _gg.batched_geod_fields(cropped, a_crop)
                if f2 is None:
                    self._gpu_axe = False
                    return
                for (k, m), (y0, x0), fc, ac in zip(chunk, offs, f2, a_crop):
                    dfull = np.full(m.shape, np.inf, dtype=np.float64)
                    dfull[y0:y0 + fc.shape[0], x0:x0 + fc.shape[1]] = fc
                    dfull[~m] = np.inf
                    afull = (ac[0] + y0, ac[1] + x0)
                    try:
                        ax = self._axe_from_field(m, dfull, afull)
                    except Exception:  # noqa: BLE001
                        continue
                    built.append((k, m, ax))
            if not built:
                return
            # one-time validation against the exact CPU axis (same guard as the
            # per-frame path); on mismatch, disable GPU axis entirely.
            if self._gpu_axe is None:
                with self._gpu_lock:
                    if self._gpu_axe is None:
                        ok, why = self._validate_axes(built)
                        self._gpu_axe = ok
                        _gg.note(ok, why)
                if self._gpu_axe is False:
                    return
            with self._axe_shared_lock:
                for k, _m, ax in built:
                    self._axe_shared[k] = ax
            self._prepass_done = True
        except Exception:  # noqa: BLE001
            self._gpu_axe = False

    def _precompute_axes(self, tl, name):
        """Compute the axes of this frame's canonical masks (T-1/T0/T+1, the
        formatted-object and raw masks) in ONE batched GPU pass and cache them by
        mask content, so the per-call CPU geodesic is skipped for the masks that
        dominate. Pure best-effort: any problem -> leave the cache empty and the
        chain uses the exact CPU path. The first frame validates the GPU-batched
        axes against the exact CPU axis and disables the path on any mismatch."""
        if self._gpu_axe is False:
            return
        # Fast path: the owner pre-pass already computed every axis on the GPU in
        # big batches; here we only seed this frame's keys from the shared cache
        # (dict lookups, NO GPU call from the pool thread -> no contention).
        if self._prepass_done and self._axe_shared:
            try:
                ns0 = next(iter(self._ns.values()))
                fmt = ns0["format_384"]; obj = ns0["_objet"]
                mb = ns0["masque_brut"]; vois = ns0["voisins"]
                pv, nx = vois(name); names = []
                for x in (name, pv, nx):
                    if x not in names:
                        names.append(x)
                for x in names:
                    cands = []
                    try:
                        cands.append(obj(fmt(x)))
                    except Exception:  # noqa: BLE001
                        pass
                    m = mb(x)
                    if m is not None:
                        cands.append(m)
                    for m in cands:
                        if m is None:
                            continue
                        ax = self._axe_shared.get(m.tobytes())
                        if ax is not None:
                            tl.axe_pre[m.tobytes()] = ax
                return
            except Exception:  # noqa: BLE001
                pass
        import numpy as np
        try:
            from . import geodesic_gpu as _gg
            ns0 = next(iter(self._ns.values()))
            fmt = ns0["format_384"]; obj = ns0["_objet"]
            mb = ns0["masque_brut"]; vois = ns0["voisins"]
            pv, nx = vois(name)
            names = []
            for x in (name, pv, nx):
                if x not in names:
                    names.append(x)
            items = []           # (key_bytes, full_mask)
            seen = set()
            for x in names:
                cands = []
                try:
                    cands.append(obj(fmt(x)))
                except Exception:  # noqa: BLE001
                    pass
                mm = mb(x)
                if mm is not None:
                    cands.append(mm)
                for m in cands:
                    if m is None:
                        continue
                    k = m.tobytes()
                    if k in seen:
                        continue
                    seen.add(k); items.append((k, m))
            if not items:
                return
            cropped = []; offs = []; starts = []
            for _k, m in items:
                ys, xs = np.nonzero(m)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                cropped.append(m[y0:y1, x0:x1])
                offs.append((y0, x0))
                starts.append((int(ys[0]) - y0, int(xs[0]) - x0))
            f1 = _gg.batched_geod_fields(cropped, starts)
            if f1 is None:
                self._gpu_axe = False
                return
            a_crop = [np.unravel_index(
                int(np.argmax(np.where(np.isfinite(f), f, -1))), f.shape)
                for f in f1]
            f2 = _gg.batched_geod_fields(cropped, a_crop)
            if f2 is None:
                self._gpu_axe = False
                return
            built = []           # (key, full_mask, axe_dict)
            for (k, m), (y0, x0), fc, ac in zip(items, offs, f2, a_crop):
                dfull = np.full(m.shape, np.inf, dtype=np.float64)
                dfull[y0:y0 + fc.shape[0], x0:x0 + fc.shape[1]] = fc
                dfull[~m] = np.inf
                afull = (ac[0] + y0, ac[1] + x0)
                try:
                    ax = self._axe_from_field(m, dfull, afull)
                except Exception:  # noqa: BLE001
                    continue
                built.append((k, m, ax))
            if not built:
                return
            # one-time validation against the exact CPU axis
            if self._gpu_axe is None:
                with self._gpu_lock:
                    if self._gpu_axe is None:
                        ok, why = self._validate_axes(built)
                        self._gpu_axe = ok
                        _gg.note(ok, why)
                if self._gpu_axe is False:
                    return
            for k, _m, ax in built:
                tl.axe_pre[k] = ax
        except Exception:  # noqa: BLE001
            self._gpu_axe = False

    def _validate_axes(self, built):
        """Compare GPU-batched axes to the exact CPU axis on a few masks.
        Returns (ok, reason)."""
        import numpy as np
        tol = 0.05
        checked = 0
        for k, m, ax in built:
            try:
                ref = self._axe_cpu(m)
            except Exception:  # noqa: BLE001
                continue
            dL = abs(ax["L"] - ref["L"]) / max(ref["L"], 1.0)
            dw = float(np.max(np.abs(ax["w"] - ref["w"]))) / (ref["wmed"] + 1e-6)
            dP = float(np.max(np.abs(ax["P"] - ref["P"])))
            if dL > tol or dw > tol or dP > 2.0:
                return False, ("GPU-batched axis mismatch (dL=%.3f dw=%.3f "
                               "dP=%.2f) -- CPU fallback" % (dL, dw, dP))
            checked += 1
            if checked >= 3:
                break
        if checked == 0:
            return False, "could not validate GPU axes -- CPU fallback"
        return True, ("GPU batched axes active (validated on %d masks)" % checked)

    def compute(self, name) -> Dict[str, float]:
        self._ensure_loaded()
        if not self._ok or self._ctx is None:
            return {}
        # fresh per-frame memo caches (also prevents id() aliasing across frames)
        tl = self._caches()
        tl.f384.clear(); tl.f384f.clear(); tl.mbrut.clear()
        tl.objet.clear(); tl.axe.clear(); tl.axe_pre.clear()
        if getattr(tl, "geod", None) is not None:
            tl.geod.clear()
        self._precompute_axes(tl, name)
        out: Dict[str, float] = {}
        for prog, cols in self._PROGRAMS.items():
            ns = self._ns.get(prog)
            if ns is None or "calcule" not in ns:
                continue
            try:
                # calcule returns (sp, bn, {col: val}[, extra]); v3s adds a 4th
                # element (the chosen neighbour names), so index rather than
                # unpack a fixed arity.
                res = ns["calcule"]((None, name))
                d = res[2]
                for c in cols:
                    if c in d:
                        out[c] = float(d[c])
            except Exception:  # noqa: BLE001 - a family that fails is left NaN
                pass
        return out


_CHAIN = None


def get_chain():
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = V4jChain()
    return _CHAIN


def compute_v4j_extra(formatted_bgr=None, motion_bgr=None, frame_name=None,
                      chain=None, **_ignored) -> Dict[str, float]:
    """All wired v4j-only descriptors for one frame.

    wn_ (single frame) and tt_ (motion image) are computed directly. The axis /
    temporal families (cm_, rs_, lp_, tn_, ta_, ag_, l_, ru_, pl_, ond_, wz_,
    cr_, fw_, zl_) are computed by ``chain`` (a V4jChain whose ``set_video`` has
    been called for this video); pass the frame file name as ``frame_name``.
    """
    out: Dict[str, float] = {}
    out.update(compute_wn(formatted_bgr))
    out.update(compute_tt(motion_bgr))
    if chain is not None and frame_name is not None:
        try:
            out.update(chain.compute(frame_name))
        except Exception:  # noqa: BLE001
            pass
    return out
