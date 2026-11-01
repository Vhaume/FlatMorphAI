# -*- coding: utf-8 -*-
"""Step 5. Three-channel composite of the state classifier (processed_composite_384_uni_final_v20).

The functions worm_mask, orient, _sq, whole_channel, composite_v11 and tete_en_haut are read from
production/regenerate_composite_final_v20.py and executed unchanged. Path of the production loop:
  channel 0, whole worm: grey image rotated so that the principal axis of the mask (SVD of the pixel coordinates) is
     vertical, turned by 180 degrees when the retained tip falls in the lower half, background 0, bounding box, padded
     to a square and resized to 384,
  channel 1, zoom: thumbnail of half-side FRAC_Z1 = 0.30 L cut by cadrage_squelette.affine on the tip and centre
     retained by the cascade (centre 0.12 L from the tip, tip up), grey, background outside the worm 0 (not read by
     the classifier, kept for the structure of the file),
  channel 2, super-zoom: same framing, half-side FRAC_SZ = 0.15 L (side 0.30 L), CLAHE clip 2 tile 8, background 0.
The composite is written as a PNG (lossless), channels in the order 0, 1, 2 of cv2.merge (B, G, R of the file).
Reference (verified). Zuiderveld K. (1994). CLAHE. Graphics Gems IV, DOI 10.1016/B978-0-12-336156-1.50061-6.
"""
import threading

import numpy as np, cv2
from . import _env
from ._env import CS, read, exec_slice

PROD_FILE = "regenerate_composite_final_v22.py"


class _ThreadCLAHE:
    """cv2.CLAHE keeps internal buffers, so one shared instance is not safe when
    several worms are analysed at once. This proxy hands each thread its own
    instance, built with the same clipLimit and tileGridSize: the histogram
    equalisation of a given image is therefore unchanged, only the buffer it
    uses is private. Production code reaches the object through ``.apply``."""

    def __init__(self, clip, tile):
        self._clip = float(clip); self._tile = int(tile); self._tl = threading.local()

    def _inst(self):
        c = getattr(self._tl, "c", None)
        if c is None:
            c = cv2.createCLAHE(clipLimit=self._clip, tileGridSize=(self._tile, self._tile))
            self._tl.c = c
        return c

    def apply(self, img):
        return self._inst().apply(img)


class CompositeBuilder:
    def __init__(self):
        self.P = read("composite")
        ns = dict(np=np, cv2=cv2, CS=CS, SIZE=int(self.P["SIZE"]), MASK_THR=int(self.P["MASK_THR"]),
                  FRAC_SZ=float(self.P["FRAC_SZ"]), FRAC_Z1=float(self.P["FRAC_Z1"]),
                  _clahe=_ThreadCLAHE(self.P["CLAHE_CLIP"], self.P["CLAHE_TILE"]))
        exec_slice(PROD_FILE, "def worm_mask(m):", "NB = int(", ns)
        self.ns = ns

    def _superzoom_channel(self, bgr, wm, r3=None):
        """Channel 2 alone, by the exact operations of composite_v11's second pass.

        Same affine, same CS.vignette, same CLAHE instance and same background
        rule, so the pixels are bit-identical to the ones composite_v11 puts in
        channel 2 -- which is what makes the fast path below safe.
        """
        ns = self.ns
        # r3 comes in as an ARGUMENT (it used to be stashed on self between the
        # two calls): with several worms analysed at once, one thread's r3 could
        # be overwritten by another's between build() and this call, which mixed
        # two worms in one super-zoom. A private CLAHE per thread for the same
        # reason -- cv2.CLAHE keeps internal buffers and is not thread-safe.
        if r3 is None:                      # kept for an external caller
            r3 = getattr(self, "_r3", None)
        M = CS.affine(r3["tip"], r3["centre"], r3["L"], frac=ns["FRAC_SZ"])
        v = CS.vignette(bgr, wm.astype(np.uint8), M)
        foot = cv2.warpAffine(wm.astype(np.uint8) * 255, M, (ns["SIZE"], ns["SIZE"]),
                              flags=cv2.INTER_NEAREST) > 127
        g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        g = ns["_clahe"].apply(g)
        g[~foot] = 0
        return g

    def build(self, bgr, mask, r3, full=True):
        """bgr original image, mask refined mask (grey 0..255), r3 result of the cascade. Returns (composite 384 x 384
        x 3 uint8, info) or (black composite, info) when the mask is unusable, as in production.

        ``full=False`` skips channel 1 (the 0.30 L zoom). The state classifier
        reads the composite through ``ch_map`` = (0, 2, 2): it takes the whole
        worm once and the super-zoom twice, and NEVER reads channel 1 -- the
        production docstring already notes it is "kept for the structure of the
        file". Building it costs two extra warpAffine passes over the full-size
        image per worm for pixels nothing consumes, so the fast path fills
        channel 1 with the super-zoom instead. The tensor the classifier builds
        is then bit-identical, while the two warpAffine calls disappear.
        Pass ``full=True`` (the default) whenever the PNG is written to disk, so
        a saved composite keeps exactly the three production channels."""
        ns = self.ns; S = ns["SIZE"]
        m = ns["worm_mask"](mask)
        if m.sum() < 300 or r3 is None:
            return np.zeros((S, S, 3), np.uint8), dict(composite="vide")
        rbgr, rm, M = ns["orient"](bgr, m)
        rg = cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY)
        rg, rm, tourne = ns["tete_en_haut"](rg, rm, M, r3)
        c0 = ns["whole_channel"](rg, rm)
        wm = CS.masque_ver(mask)
        if full:
            comp = ns["composite_v11"](bgr, wm, r3, c0)
        else:
            c2 = self._superzoom_channel(bgr, wm, r3)
            comp = cv2.merge([c0, c2, c2])
        Msz = CS.affine(r3["tip"], r3["centre"], r3["L"], frac=ns["FRAC_SZ"])
        hx, hy = (Msz @ np.array([r3["x"], r3["y"], 1.0])).tolist()
        return comp, dict(composite="ok", tourne_180=int(tourne),
                          canal1=("zoom" if full else "copy of the super-zoom"),
                          point_tete_dans_superzoom=int(0 <= hx < S and 0 <= hy < S))
