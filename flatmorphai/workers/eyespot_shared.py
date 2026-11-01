# -*- coding: utf-8 -*-
"""Shared eyespot-counting helpers for the subprocess drivers.

Both :mod:`flatmorphai.workers.eyespots_counter_pipeline` and
:mod:`flatmorphai.workers.regeneration_pipeline` run as standalone scripts with
the ``Export_head_detection`` sub-packages on ``sys.path``.  They both count the
eyespots of the two worm extremities and write a
``eyespots_detection.csv`` with, per threshold, a count and the list of (x, y)
positions.  This module is the single definition of that logic and of the CSV
column names so the two drivers always produce the exact same schema - which is
what the "Regeneration states editor" page relies on when it overlays the
detected eyespots and plots their number.

The module is pure Python and imports nothing heavy at load time; the Export
modules (``eye_detection``, ``preprocessing``) are passed in by the caller.
"""
from __future__ import annotations


def eyespot_columns(t_bal, t_pos):
    """Return the four CSV column names (count + positions, per threshold).

    The threshold value is embedded in the name, matching the convention of
    the rest of the application.  ``%g`` drops trailing zeros (0.80 -> ``0.8``).
    """
    return (
        "n_eyespots_strong_balanced(T_Min=%g)" % float(t_bal),
        "eyespots_positions_balanced(T_Min=%g)" % float(t_bal),
        "n_eyespots_strong_positive(T_Min=%g)" % float(t_pos),
        "eyespots_positions_positive(T_Min=%g)" % float(t_pos),
    )


def positions_str(points):
    """Format a list of (x, y) points as ``[(x, y), (x, y), ...]`` (``%g``)."""
    return "[" + ", ".join("(%g, %g)" % (p[0], p[1]) for p in points) + "]"


def count_peaks(r3, A3b, t_bal, t_pos, pair_rule=False):
    """(positions at the strict threshold, positions at the permissive one).

    Every local maximum of the eyespot heat map of the extremity the cascade
    kept, above each threshold, mapped back into the original image. Same
    separation neighbourhood as the cascade (AC.MINDIST), but WITHOUT its cap
    at the two strongest: that cap exists to choose the two peaks whose
    probability goes to the state model, and it has no business in a count.

    Shared by both chains, which read the same map of the same extremity:
    only the threshold differs -- the strict one for the regeneration chain,
    the one the operator selected for the counter.

    ``pair_rule``: if AT LEAST ONE eyespot passes the strict threshold, the
    other peaks are accepted at the permissive one. Eyespots come in pairs; a
    certain one makes the second plausible, and that prior is what the rule
    uses instead of judging each blob in isolation. The guard is the entry
    condition: with no certain eyespot nothing is admitted, so a noisy image
    without eyespots stays protected by the strict threshold.
    Measured on the 52 reference images: one single image changes (a second
    eyespot at 0.612 that missed 0.80), none goes above two eyespots, and the
    19 images without a certain eyespot peak at 0.314, far below the
    permissive threshold.
    """
    cands = (r3 or {}).get("candidates") or []
    i_head = int((r3 or {}).get("i_head", 0))
    if not (0 <= i_head < len(cands)):
        return [], []
    c = cands[i_head]
    p = c.get("eye_heatmap")
    M = c.get("eye_M")
    if p is None or M is None:
        return [], []
    bal, pos = [], []
    for px, py, h in A3b._pics_tous(p, min(float(t_bal), float(t_pos))):
        xy = A3b._vers_image(M, px, py)
        pt = (round(float(xy[0]), 1), round(float(xy[1]), 1))
        if h >= float(t_pos):
            pos.append(pt)
        if h >= float(t_bal):
            bal.append(pt)
    if pair_rule and bal and float(t_pos) < float(t_bal):
        # A certain eyespot opens the door to the others at the permissive
        # threshold. The permissive list already holds the strict peaks
        # (t_pos < t_bal).
        bal = list(pos)
    return bal, pos


def count_from_ends(ends, pr, M, t_bal, t_pos):
    """Count the eyespots of BOTH extremities from precomputed heat maps.

    ``ends`` is the two-element list returned by ``eye_detection.eye_maps`` (one
    dict per extremity, each carrying ``peaks_straight`` as (x, y, height) in the
    straightened frame).  ``pr`` is the Export ``preprocessing`` module and ``M``
    the straightening matrix, used to map each peak back to the ORIGINAL image
    frame.  The heat map must have been predicted at a threshold no higher than
    ``min(t_bal, t_pos)`` so both counts can be obtained by height filtering.

    Returns ``(positions_balanced, positions_positive)`` as two lists of (x, y)
    tuples rounded to one decimal; the counts are their lengths.
    """
    bal, pos = [], []
    for end in ends:
        for px, py, h in end.get("peaks_straight", []):
            xo, yo = pr.to_original(px, py, M)
            point = (round(float(xo), 1), round(float(yo), 1))
            if h >= float(t_pos):
                pos.append(point)
            if h >= float(t_bal):
                bal.append(point)
    return bal, pos


def count_eyespots_both_ends(eye_mod, pr, rbgr, rm, span, M, t_bal, t_pos):
    """Convenience wrapper: predict the heat maps then count both extremities.

    Predicts the eyespot heat map once at ``min(t_bal, t_pos)`` and delegates to
    :func:`count_from_ends`.  Returns ``(ends, positions_balanced,
    positions_positive)`` so the caller can also reuse ``ends`` (thumbnails,
    heat maps) without a second model pass.
    """
    t_low = min(float(t_bal), float(t_pos))
    ends = eye_mod.eye_maps(rbgr, rm, span, threshold=t_low)
    bal, pos = count_from_ends(ends, pr, M, t_bal, t_pos)
    return ends, bal, pos
