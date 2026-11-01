# -*- coding: utf-8 -*-
# Compare the tables a page produces (Regeneration, Eyespots counter,
# Segmentation only) against a reference copy, to check that a change moved
# NOT ONE row.
#
# Usage (from the FlatMorphAI root):
#   python scripts/test_csv_regeneration.py keep      -> save the current CSVs as the reference
#   python scripts/test_csv_regeneration.py compare   -> compare the current CSVs with that copy
#   ... [folder]                              -> a folder other than
#                                                Processed_regeneration_images
import os
import sys
import csv
import shutil

# The repository root: this script lives one folder below it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.__stdout__
_REF_ROOT = os.path.join(os.path.expanduser("~"), ".flatmorphai")
FILES = ("regeneration_states.csv", "head_position.csv",
         "eyespots_detection.csv", "blasteme_detection.csv")


def say(m=""):
    OUT.write(m + "\n"); OUT.flush()


def _folder(base):
    """The folder holding at least one of the tables, under ``base``."""
    if not os.path.isdir(base):
        return None
    if any(os.path.isfile(os.path.join(base, f)) for f in FILES):
        return base
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p) and any(os.path.isfile(os.path.join(p, f)) for f in FILES):
            return p
    return None


def _ref_dir(base):
    """One reference per processed folder, so the pages never mix."""
    import hashlib
    return os.path.join(_REF_ROOT, "csv_reference_%s_%s"
                        % (os.path.basename(base.rstrip(os.sep))[:28],
                           hashlib.md5(os.path.abspath(base).encode("utf-8")).hexdigest()[:8]))


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter=";"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    base = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Processed_regeneration_images")
    if not os.path.isabs(base):
        base = os.path.join(ROOT, base)
    REF = _ref_dir(base)
    folder = _folder(base)
    if folder is None:
        say("no table found in %s: run the page first" % base); return
    say("folder: %s" % folder)
    if mode.startswith("keep"):
        os.makedirs(REF, exist_ok=True)
        n = 0
        for f in FILES:
            p = os.path.join(folder, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(REF, f)); n += 1
        say("reference saved: %d table(s) in %s" % (n, REF))
        say("Now run the page again with your change, then:")
        say("   python scripts/test_csv_regeneration.py compare%s"
            % ("" if base.endswith("Processed_regeneration_images") else " " + base))
        return

    if not os.path.isdir(REF):
        # No local reference: fall back on the one shipped with the repository,
        # so that a fresh clone can check it reproduces the AUTHOR's numbers.
        # Comparing a run with a reference you took from that same run only
        # proves the chain is deterministic, which is a much weaker statement.
        shipped = os.path.join(ROOT, "reference_csv", "regeneration")
        if os.path.isdir(shipped):
            REF = shipped
            say("no local reference; using the one shipped with the repository:")
            say("   %s" % os.path.relpath(REF, ROOT))
            say("   (a difference here means your installation does not "
                "reproduce the published results)")
        else:
            say("no reference for this folder: run "
                "python scripts/test_csv_regeneration.py keep [folder] first"); return
    else:
        say("reference: %s" % REF)
    total_diff = 0
    seen = 0
    for f in FILES:
        a, b = os.path.join(REF, f), os.path.join(folder, f)
        if not (os.path.isfile(a) or os.path.isfile(b)):
            continue
        seen += 1
        if not (os.path.isfile(a) and os.path.isfile(b)):
            say("  %-28s missing on one side" % f); total_diff += 1; continue
        ra, rb = _rows(a), _rows(b)
        if len(ra) != len(rb):
            say("  %-28s %d rows against %d" % (f, len(ra), len(rb))); total_diff += 1; continue
        head = ra[0]
        nd = 0
        first = []
        for i, (x, y) in enumerate(zip(ra, rb)):
            for j, (u, v) in enumerate(zip(x, y)):
                if u != v:
                    nd += 1
                    if len(first) < 3:
                        col = head[j] if j < len(head) else "col%d" % j
                        first.append("row %d, %s: %r -> %r" % (i, col, u, v))
        total_diff += nd
        say("  %-28s %s" % (f, "IDENTICAL (%d rows)" % (len(ra) - 1) if nd == 0
                            else "%d differing value(s)" % nd))
        for e in first:
            say("        " + e)
    if total_diff == 0:
        say("\n=> the %d table(s) are IDENTICAL: the change moves no result."
            % seen)
    else:
        say("\n=> TO BE EXAMINED: see the differences above.")


if __name__ == "__main__":
    main()
