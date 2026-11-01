# -*- coding: utf-8 -*-
"""Shared environment of the export: paths, seed 42, device check, axis substitution, source slices.

Every module of regen_pipeline imports this file first. It
  * puts production/, segmentation/ and parameters/ on sys.path,
  * fixes the seeds (numpy, torch, CUDA/ROCm) to 42,
  * installs cadrage_squelette_skan2 under the name cadrage_squelette (sys.modules), exactly as the production
    programs do (raffiner_..._skan2, regenerate_composite_final_v20, enrich_metrics_v20), so that the head cascade
    (ancrage_cascade_v3c13_rayon_blasteme_v5s1neg_s040, which imports cadrage_squelette) uses the skan + Dijkstra axis,
  * selects the device and stops if no GPU is visible (CUDA or ROCm), unless REGEN_ALLOW_CPU=1 is set
    (slow, used only for the cloud check of the export),
  * gives tranche/remplace/exec_slice, which read a function block from a production program and execute it
    unchanged, the same mechanism as enrich_metrics_v20 (the formulas are never retyped).
"""
import os, sys, random
import numpy as np, torch

EXPORT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = os.path.join(EXPORT_ROOT, "production")
SEGM = os.path.join(EXPORT_ROOT, "segmentation")
PARAM = os.path.join(EXPORT_ROOT, "parameters")
for _p in (PARAM, SEGM, PROD):
    if _p not in sys.path: sys.path.insert(0, _p)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

import cadrage_squelette_skan2 as _CSK          # noqa: E402  (production/cadrage_squelette_skan2.py)
sys.modules["cadrage_squelette"] = _CSK          # skan + Dijkstra axis for every module imported afterwards
import ancrage_cascade as AC                     # noqa: E402  (loading of the U-Nets, 4 channels, peaks, darkness)
import ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 as A3b   # noqa: E402  (head cascade in service, v3c14 = v3c13 with the tip-correction start at the largest border radius in contact with the blastema)
import cadrage_squelette as CS                   # noqa: E402
assert CS is _CSK and A3b.CS is _CSK, "the axis must come from cadrage_squelette_skan2"

from parameter_reader import read, read_table, write_correction, model_path   # noqa: E402,F401

ALLOW_CPU = os.environ.get("REGEN_ALLOW_CPU", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda" and not ALLOW_CPU:
    raise SystemExit("No GPU visible to PyTorch (CUDA or ROCm). Install the GPU build of PyTorch, or set "
                     "REGEN_ALLOW_CPU=1 for a slow CPU check.")
AC.DEVICE = DEVICE                               # ancrage_cascade._charge and _pred use this attribute
_SHOWN = []


def show_device():
    if not _SHOWN:
        name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU (REGEN_ALLOW_CPU=1)"
        print("regen_pipeline | torch %s | device %s | %s" % (torch.__version__, DEVICE, name))
        _SHOWN.append(1)


def check_on_device(module, what):
    """Stops if a model is not on the selected device (frequent cause of silent CPU runs)."""
    dev = next(module.parameters()).device
    assert dev.type == DEVICE.type, "%s is on %s instead of %s" % (what, dev, DEVICE)


def source(name):
    return open(os.path.join(PROD, name), encoding="utf-8").read()


def tranche(s, debut, fin):
    """Text of s from the first occurrence of debut up to the next occurrence of fin (excluded)."""
    i = s.index(debut); j = s.index(fin, i); return s[i:j]


def remplace(s, a, b, n=1):
    assert s.count(a) == n, ("pattern expected %d time(s)" % n, a)
    return s.replace(a, b)


def exec_slice(name, debut, fin, ns, replacements=()):
    """Executes, in namespace ns, the block of production/<name> between debut and fin, after the listed
    text replacements (each checked to occur exactly once). Returns ns."""
    s = tranche(source(name), debut, fin)
    for a, b in replacements: s = remplace(s, a, b)
    exec(compile(s, os.path.join(PROD, name), "exec"), ns)
    return ns


def apply_parameters(table, targets):
    """Writes the effective values of parameters/<table>.csv on module attributes.
    targets maps a prefix to a module or dict: a parameter "AC.NB_COUPES" goes to targets["AC"], a parameter
    without prefix to targets[""]. Returns the dictionary of effective values."""
    vals = read(table)
    for k, v in vals.items():
        pre, _, att = k.rpartition(".")
        tgt = targets.get(pre)
        if tgt is None: continue
        if isinstance(tgt, dict): tgt[att] = v
        else: setattr(tgt, att, v)
    return vals


def imread_color(path, segmentation=False):
    """BGR image read as in production. Steps 2 to 6 used cv2.imread (IMREAD_COLOR, EXIF orientation applied),
    the segmentation used cv2.imdecode(np.fromfile(...), IMREAD_COLOR). imdecode is also the fallback for
    paths that cv2.imread cannot open (non ASCII characters under Windows)."""
    import cv2
    im = None if segmentation else cv2.imread(path)
    return im if im is not None else cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imread_gray(path):
    import cv2
    im = cv2.imread(path, 0)
    return im if im is not None else cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def jpeg_enabled():
    """Is the in-memory JPEG round trip of the masks kept?

    The production chain SAVED its masks as .jpg and read them back, so every
    step downstream -- the refinement, the cascade, the descriptors and the two
    models -- was calibrated on masks carrying that compression. The chain
    reproduces the round trip in memory for that reason only: nothing is written
    as JPEG, the files on disk are PNG. Measured cost: ~17 ms per mask after the
    segmentation (PIL, quality 75) and ~4.5 ms after the refinement (OpenCV,
    quality 100), about 1.2 % of a run.

    DROPPED BY DEFAULT since it was measured to change nothing: every step after
    the mask reads it through masque_ver, which thresholds at 128, and the
    compression never moved a single pixel across that threshold. Checked on the
    reference machine: a full run of 52 worms without the round trip gave the
    four result tables IDENTICAL to the run with it (state, decimal state to four
    places, head positions, eyespot counts, blastema areas), and the 12-image
    descriptor reference stayed identical bit for bit (1 560 values). Saves about
    1.1 s per 52 worms (17 ms per mask after the segmentation, 4.5 ms after the
    refinement).

    FLATMORPHAI_REGEN_JPEG=1 brings it back, for a strict byte-for-byte
    reproduction of how the production chain stored its masks."""
    return os.environ.get("FLATMORPHAI_REGEN_JPEG", "0") in ("1", "true", "True")


def jpeg_roundtrip(mask_u8, quality=None, pil=False):
    """Mask encoded then decoded as in production, which stored the masks as .jpg files read back by the next step:
    Mask2Former masks by PIL (default quality 75, after scaling to 255), refined masks by cv2.imwrite quality 100.
    Returns the decoded grey mask (uint8), what the next production step actually read."""
    import cv2
    if pil:
        import io
        from PIL import Image
        pr = mask_u8.astype(float)
        if pr.max() > 0: pr *= 255.0 / pr.max()
        buf = io.BytesIO(); Image.fromarray(pr).convert("L").save(buf, format="JPEG")
        return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    ok, enc = cv2.imencode(".jpg", mask_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality or 100)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
