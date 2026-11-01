# -*- coding: utf-8 -*-
"""Build the four weight archives for the GitHub Release, and print their
SHA-256 so they can be pasted into download_models.py.

Why this script exists: the weights are 2.2 GB and GitHub refuses any file over
100 MB inside a repository, so they travel as Release assets (2 GB each is
allowed). That creates an ordering problem -- download_models.py needs the asset
URLs, which do not exist until the Release does. The order is therefore:

    1. python tools/make_release_assets.py            -> four .zip + their SHA-256
    2. create the Release on GitHub, upload the four files
    3. paste the tag and the digests into download_models.py
    4. python scripts/download_models.py                -> verify it works from scratch

The archives are plain zip without compression: these files are already
compressed tensors, so deflating them costs minutes and saves almost nothing.

Usage (from the FlatMorphAI root):
    python tools/make_release_assets.py            (list what would be built)
    python tools/make_release_assets.py build      (write release_assets/)
"""
import hashlib
import os
import sys
import zipfile

# The repository root: this script lives one folder below it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.__stdout__
DEST = os.path.join(ROOT, "release_assets")

#: (archive name, folder to archive relative to the project root)
ASSETS = [
    ("checkpoints_segmentation_model.zip", "checkpoints_segmentation_model"),
    ("mask2former-swin-tiny-ade-semantic.zip", "mask2former-swin-tiny-ade-semantic"),
    ("regen_stage_v20_models.zip", os.path.join("flatmorphai", "regen_stage_v20", "models")),
    ("export_head_detection_models.zip",
     os.path.join("flatmorphai", "export_head_detection", "models")),
]
#: Files inside the weight folders that must NOT travel.
SKIP_SUFFIX = (".bak", ".pyc", ".log")
SKIP_DIR = {"__pycache__"}


def say(m=""):
    OUT.write(m + "\n"); OUT.flush()


def _files(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in SKIP_DIR]
        for f in sorted(fn):
            if any(s in f for s in SKIP_SUFFIX):
                continue
            yield os.path.join(dp, f)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    build = len(sys.argv) > 1 and sys.argv[1].startswith("build")
    rows = []
    for name, rel in ASSETS:
        src = os.path.join(ROOT, rel)
        if not os.path.isdir(src):
            say("MISSING  %-44s (%s)" % (rel, "folder not found")); continue
        files = list(_files(src))
        size = sum(os.path.getsize(p) for p in files)
        rows.append((name, rel, src, files, size))
        say("%-44s %5d file(s)  %7.0f MB -> %s" % (rel, len(files), size / 1e6, name))

    if not rows:
        say("\nNothing to archive."); return
    total = sum(r[4] for r in rows)
    say("\ntotal %.1f GB" % (total / 1e9))
    over = [r for r in rows if r[4] > 2e9]
    if over:
        say("WARNING: an asset over 2 GB exceeds the GitHub per-asset limit:")
        for r in over:
            say("   %s (%.1f GB) -- split it" % (r[0], r[4] / 1e9))

    if not build:
        say("")
        say("Nothing was written. To build the archives:")
        say("    python tools/make_release_assets.py build")
        return

    os.makedirs(DEST, exist_ok=True)
    say("")
    digests = []
    for name, rel, src, files, _size in rows:
        dst = os.path.join(DEST, name)
        # ZIP_STORED: the payload is already compressed, deflate would cost
        # minutes for a percent.
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in files:
                # Paths inside the archive are relative to the project root, so
                # extracting at the root puts every file back in place.
                z.write(p, os.path.relpath(p, ROOT))
        d = _sha256(dst)
        digests.append((name, d, os.path.getsize(dst)))
        say("  wrote %-44s %7.0f MB" % (name, os.path.getsize(dst) / 1e6))

    say("")
    say("Paste this into download_models.py, then create the Release and")
    say("upload the four files from %s :" % os.path.basename(DEST))
    say("")
    say("SHA256 = {")
    for name, d, _s in digests:
        say('    "%s":\n        "%s",' % (name, d))
    say("}")
    say("")
    say("Keep these digests with the Release notes too: they are the only way")
    say("a user can tell a truncated download from a complete one.")


if __name__ == "__main__":
    main()
