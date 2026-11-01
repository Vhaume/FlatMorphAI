# -*- coding: utf-8 -*-
"""Download the model weights from the GitHub Release and verify them.

The weights are not in the repository: GitHub refuses any file over 100 MB, and
these are 2.2 GB. They are published as Release assets and fetched here.

Each archive is checked against its SHA-256 before extraction. That is not
ceremony: a truncated download leaves a file of plausible size that loads and
then produces wrong numbers, which is the worst possible failure for this
application. A digest turns that into a clear error.

Already-installed folders are left alone, so running this twice is free.

Usage (from the FlatMorphAI root):
    python scripts/download_models.py                 (download what is missing)
    python scripts/download_models.py --check         (report only, download nothing)
    python scripts/download_models.py --force         (re-download everything)

See make_release_assets.py for how the archives and the digests are produced.
"""
import hashlib
import os
import shutil
import sys
import zipfile

# The repository root: this script lives one folder below it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.__stdout__
TMP = os.path.join(ROOT, ".model_downloads")

# ---------------------------------------------------------------------------
# TO FILL IN ONCE THE RELEASE EXISTS.
# The tag of the GitHub Release that carries the four assets, and the digests
# printed by `python tools/make_release_assets.py build`. Until both are filled in,
# this script refuses to run rather than download something it cannot verify.
# ---------------------------------------------------------------------------
REPO = "Vhaume/FlatMorphAI"       # <- GitHub owner/repository
TAG = "weights-v1.0.0"            # <- the Release tag

SHA256 = {
    # "checkpoints_segmentation_model.zip": "<64 hex chars>",
    # "mask2former-swin-tiny-ade-semantic.zip": "<64 hex chars>",
    # "regen_stage_v20_models.zip": "<64 hex chars>",
    # "export_head_detection_models.zip": "<64 hex chars>",
}

#: (archive name, folder it restores, roughly expected size in bytes)
ASSETS = [
    ("checkpoints_segmentation_model.zip", "checkpoints_segmentation_model", 807e6),
    ("mask2former-swin-tiny-ade-semantic.zip", "mask2former-swin-tiny-ade-semantic", 182e6),
    ("regen_stage_v20_models.zip",
     os.path.join("flatmorphai", "regen_stage_v20", "models"), 1200e6),
    ("export_head_detection_models.zip",
     os.path.join("flatmorphai", "export_head_detection", "models"), 45e6),
]


def say(m=""):
    OUT.write(m + "\n"); OUT.flush()


def url_for(name):
    return "https://github.com/%s/releases/download/%s/%s" % (REPO, TAG, name)


def _tree_size(path):
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def installed(rel, expected):
    """True when the folder is there and big enough to be complete."""
    path = os.path.join(ROOT, rel)
    return os.path.isdir(path) and _tree_size(path) >= 0.5 * expected


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(name, dest):
    """Download with a progress line. Standard library only, so the script
    works before any dependency is installed."""
    import urllib.error
    import urllib.request
    url = url_for(name)
    say("  downloading %s" % url)
    try:
        with urllib.request.urlopen(url) as r, open(dest, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                blk = r.read(1 << 20)
                if not blk:
                    break
                fh.write(blk); got += len(blk)
                if total:
                    OUT.write("\r    %5.1f%%  %6.0f / %6.0f MB"
                              % (100.0 * got / total, got / 1e6, total / 1e6))
                else:
                    OUT.write("\r    %6.0f MB" % (got / 1e6))
                OUT.flush()
        OUT.write("\n"); OUT.flush()
        return True
    except urllib.error.HTTPError as exc:
        say("    FAILED: HTTP %s. Check REPO and TAG at the top of this script, "
            "and that the asset is attached to the Release." % exc.code)
    except Exception as exc:  # noqa: BLE001
        say("    FAILED: %s: %s" % (type(exc).__name__, exc))
    return False


def main():
    check = "--check" in sys.argv
    force = "--force" in sys.argv

    missing_digest = [n for n, _r, _s in ASSETS if not SHA256.get(n)]
    if missing_digest and not check:
        say("This script is not configured yet.")
        say("")
        say("Missing SHA-256 for: %s" % ", ".join(missing_digest))
        say("Fill in REPO, TAG and SHA256 at the top of this file. The digests")
        say("come from:   python tools/make_release_assets.py build")
        say("")
        say("Refusing to download an archive that cannot be verified: a")
        say("truncated file would load and then produce wrong numbers.")
        return 1

    say("repository %s, release %s\n" % (REPO, TAG))
    todo = []
    for name, rel, expected in ASSETS:
        if installed(rel, expected) and not force:
            say("present   %-44s %7.0f MB"
                % (rel, _tree_size(os.path.join(ROOT, rel)) / 1e6))
        else:
            say("TO FETCH  %-44s ~%6.0f MB" % (rel, expected / 1e6))
            todo.append((name, rel, expected))

    if check:
        say("")
        say("--check: nothing downloaded. %d folder(s) to fetch." % len(todo))
        if missing_digest:
            say("Note: SHA256 is not filled in yet for %s."
                % ", ".join(missing_digest))
        return 0
    if not todo:
        say("\nEverything is in place.")
        return 0

    os.makedirs(TMP, exist_ok=True)
    say("")
    failed = []
    for name, rel, _expected in todo:
        say("%s" % name)
        archive = os.path.join(TMP, name)
        if not (os.path.isfile(archive) and _sha256(archive) == SHA256[name]):
            if not fetch(name, archive):
                failed.append(rel); continue
        else:
            say("  already downloaded and verified, reusing it")
        got = _sha256(archive)
        if got != SHA256[name]:
            say("  SHA-256 MISMATCH")
            say("    expected %s" % SHA256[name])
            say("    got      %s" % got)
            say("  The download is incomplete or the asset was replaced. The")
            say("  archive is kept in %s; delete it and run again."
                % os.path.basename(TMP))
            failed.append(rel); continue
        say("  SHA-256 verified")
        target = os.path.join(ROOT, rel)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        try:
            with zipfile.ZipFile(archive) as z:
                # Paths inside the archive are relative to the project root.
                z.extractall(ROOT)
            say("  extracted to %s" % rel)
            os.remove(archive)
        except Exception as exc:  # noqa: BLE001
            say("  extraction FAILED: %s: %s" % (type(exc).__name__, exc))
            failed.append(rel)

    shutil.rmtree(TMP, ignore_errors=True)
    say("")
    if failed:
        say("%d folder(s) could not be installed: %s" % (len(failed), ", ".join(failed)))
        return 1
    say("Weights installed. Check the whole installation with:")
    say("    python -m flatmorphai.install_check --deep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
