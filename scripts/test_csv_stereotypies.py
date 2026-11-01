# -*- coding: utf-8 -*-
# Compare the tables the STEREOTYPE chain produces (Stereotype classification) against a
# reference copy, to check that a speed-up moves NOT ONE value: descriptors,
# forecasted classes and the merged table.
#
# Usage (from the FlatMorphAI root):
#   python scripts/test_csv_stereotypies.py keep      -> save the current CSVs as the reference
#   python scripts/test_csv_stereotypies.py redo      -> reference + DELETE the CSVs, so that
#                                                the next run recomputes them
#   python scripts/test_csv_stereotypies.py compare   -> compare the current CSVs with that copy
#   ... [folder]                              -> a folder other than
#                                                Processed_videos_Frame_by_frame
#
# The protocol that validates a change -- the only one that proves anything:
#   1) "redo": the reference is saved and the tables are deleted. The MASKS are
#      kept, so the next run reuses them (overwrite=False) and segmentation
#      stays out of the comparison -- only the descriptors and the classes are
#      recomputed, which is exactly what a speed-up touches.
#   2) run Stereotype classification again.
#   3) "compare": everything must be IDENTICAL.
# Without step 1 the next run finds the CSVs already there, recomputes nothing,
# and the comparison only compares the files with themselves.
import os
import sys
import csv
import shutil
import hashlib
import re

# The repository root: this script lives one folder below it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.__stdout__
# The reference is kept INSIDE the project, not under ~/.flatmorphai: the home
# folder differs between the environments that may launch the script, and a
# reference one side cannot see is of no use. Deletable by hand.
_REF_ROOT = os.path.join(ROOT, "_references_csv")
#: tables at the root of the processed folder
ROOT_FILES = ("all_videos.csv", "grouped_periods.csv")
#: the forecasted class, the one value that must never move
CLASS_COLS = ("forecasted_Stereotype_index", "forecasted_Stereotype_name")


def say(m=""):
    OUT.write(m + "\n"); OUT.flush()


def _collect(base):
    """{relative path -> absolute path} of the stereotype chain's CSVs."""
    found = {}
    for f in ROOT_FILES:
        p = os.path.join(base, f)
        if os.path.isfile(p):
            found[f] = p
    for d in sorted(os.listdir(base)):
        vd = os.path.join(base, d)
        if not os.path.isdir(vd):
            continue
        for e in sorted(os.listdir(vd)):
            if not e.lower().endswith(".csv"):
                continue
            if e.endswith("_features.csv") or e.startswith("results_Stereotype_"):
                found[os.path.join(d, e)] = os.path.join(vd, e)
    return found


def _ref_dir(base):
    """One reference folder per processed folder, named after its path
    RELATIVE to the project: the same name on both sides, wherever the project
    happens to be mounted."""
    try:
        rel = os.path.relpath(os.path.abspath(base), ROOT)
    except ValueError:
        rel = ""
    if rel.startswith("..") or not rel or rel == ".":
        rel = "abs_" + hashlib.md5(os.path.abspath(base).encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel).strip("_")[:60]
    return os.path.join(_REF_ROOT, slug)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter=";"))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # NaN and empty treated alike


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    base = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Processed_videos_Frame_by_frame")
    if not os.path.isabs(base):
        base = os.path.join(ROOT, base)
    if not os.path.isdir(base):
        say("folder not found: %s" % base); return
    REF = _ref_dir(base)
    cur = _collect(base)
    if not cur:
        say("no stereotype CSV in %s: run the pipeline first" % base); return
    say("folder: %s" % base)
    say("%d table(s) found\n" % len(cur))

    _keep = mode.startswith("keep")
    _redo = mode.startswith("redo")
    if _keep or _redo:
        # Guard: a reference must never SHRINK silently.
        # On 31 October 2026 a 14-table reference fell to 4 because "redo" was
        # run while ten of them had already been deleted by an earlier pass.
        # The next comparison then reported "IDENTICAL" on 4 tables and "new"
        # on ten, that is, a validation ten times weaker than it looked.
        # Better to refuse than to half-validate without saying so.
        old = set()
        if os.path.isdir(REF):
            for dp, _dn, fn in os.walk(REF):
                for f in fn:
                    if f.lower().endswith(".csv"):
                        old.add(os.path.relpath(os.path.join(dp, f), REF))
        if old and len(cur) < len(old) and "--force" not in sys.argv:
            missing = sorted(old - set(cur))
            say("REFUSED: the existing reference holds %d table(s), the folder"
                % len(old))
            say("   holds only %d. Overwriting it now would weaken the next" % len(cur))
            say("   comparison with nothing to signal it.")
            say("")
            say("   Missing from the processed folder:")
            for m in missing[:8]:
                say("      %s" % m)
            if len(missing) > 8:
                say("      ... and %d more" % (len(missing) - 8))
            say("")
            say("   Run the pipeline again to regenerate them, then start over.")
            say("   If shrinking the set is intended: add --force.")
            return
        shutil.rmtree(REF, ignore_errors=True)
        for rel, p in cur.items():
            dst = os.path.join(REF, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
        say("reference saved: %d table(s) in %s" % (len(cur), REF))
        if _redo:
            n = 0
            for rel, p in sorted(cur.items()):
                try:
                    os.remove(p); n += 1
                except OSError as exc:
                    say("  could not be deleted: %s (%s)" % (rel, exc))
            say("%d table(s) deleted from the processed folder: the next run will" % n)
            say("   recompute them. The masks and the formatted images are intact,")
            say("   so segmentation will be skipped and the comparison will cover")
            say("   the descriptors and the classes.")
        else:
            say("WARNING: the tables are still there. With overwrite=False the next")
            say("   run will find them already done and recompute NOTHING; the")
            say('   comparison would be meaningless. Use "redo" to delete them')
            say("   after taking the reference.")
        say("Now run the pipeline again, then:")
        say("   python scripts/test_csv_stereotypies.py compare%s"
            % ("" if base.endswith("Processed_videos_Frame_by_frame") else ' "%s"' % base))
        return

    if not os.path.isdir(REF):
        # No local reference: fall back on the one shipped with the repository,
        # so that a fresh clone can check it reproduces the AUTHOR's numbers.
        # Comparing a run with a reference you took from that same run only
        # proves the chain is deterministic, which is a much weaker statement.
        shipped = os.path.join(ROOT, "reference_csv", "stereotypes")
        if os.path.isdir(shipped):
            REF = shipped
            say("no local reference; using the one shipped with the repository:")
            say("   %s" % os.path.relpath(REF, ROOT))
            say("   (a difference here means your installation does not "
                "reproduce the published results)")
            say("")
        else:
            say("no reference for this folder: run")
            say("   python scripts/test_csv_stereotypies.py keep"); return
    else:
        say("reference: %s" % os.path.relpath(REF, ROOT))
        say("")

    ref = {}
    for dp, _dn, fn in os.walk(REF):
        for f in fn:
            if f.lower().endswith(".csv"):
                rel = os.path.relpath(os.path.join(dp, f), REF)
                ref[rel] = os.path.join(dp, f)

    missing = sorted(set(ref) - set(cur))
    added = sorted(set(cur) - set(ref))
    total_diff = 0
    cls_diff = 0
    worst = []
    for rel in sorted(set(ref) & set(cur)):
        ra, rb = _rows(ref[rel]), _rows(cur[rel])
        label = rel if len(rel) <= 54 else "..." + rel[-51:]
        if len(ra) != len(rb):
            say("  %-56s %d rows against %d" % (label, len(ra), len(rb)))
            total_diff += 1; continue
        if not ra:
            continue
        head = ra[0]
        cls_idx = [i for i, h in enumerate(head) if h in CLASS_COLS]
        nd = 0
        first = []
        for i, (x, y) in enumerate(zip(ra, rb)):
            for j, (u, v) in enumerate(zip(x, y)):
                if u == v:
                    continue
                a, b = _num(u), _num(v)
                if a is None and b is None and (u or "") == (v or ""):
                    continue
                if a is not None and b is not None and a == b:
                    continue        # "1.0" against "1": the same number
                nd += 1
                if j in cls_idx:
                    cls_diff += 1
                if a is not None and b is not None:
                    worst.append((abs(a - b) / (abs(a) + 1e-12), head[j] if j < len(head) else "col%d" % j, rel))
                if len(first) < 3:
                    col = head[j] if j < len(head) else "col%d" % j
                    first.append("row %d, %s: %r -> %r" % (i, col, u, v))
        total_diff += nd
        say("  %-56s %s" % (label, "IDENTICAL (%d rows)" % (len(ra) - 1) if nd == 0
                            else "%d differing value(s)" % nd))
        for e in first:
            say("        " + e)

    for rel in missing:
        say("  %-56s missing now" % rel[-56:])
    for rel in added:
        say("  %-56s new (not in the reference)" % rel[-56:])

    say("")
    if total_diff == 0 and not missing:
        say("=> the %d table(s) are IDENTICAL: the speed-up changes no descriptor"
            % len(set(ref) & set(cur)))
        say("   and no forecasted class.")
    else:
        say("=> TO BE EXAMINED.")
        say("   differing forecasted classes: %d" % cls_diff)
        for e, k, rel in sorted(worst, reverse=True)[:5]:
            say("   relative difference %.2e on %s (%s)" % (e, k, os.path.basename(rel)))
    if added and total_diff == 0:
        say("   (new tables are not a difference: the reference was taken")
        say("    before they existed.)")


if __name__ == "__main__":
    main()
