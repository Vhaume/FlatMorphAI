"""
Image naming convention of the regeneration pipeline.

``Species_Compound_Concentration_Plate_Position_Date[_OptionalComment].<ext>``

* six mandatory fields separated by the underscore, in this order: species,
  compound, concentration, plate, position and acquisition date; the optional
  comment is everything after the sixth underscore (it may itself contain
  underscores);
* any image extension that OpenCV can decode is accepted (the list below is
  the one used to enumerate the folders);
* the date is kept verbatim in the CSV outputs; :func:`parse_date` tries the
  usual numeric layouts so that images of one planaria can be sorted in time
  for the monotone correction of the predicted state.

Reference for the "derive, don't type" rule that motivates reading the
metadata from the file name: Sandve, G. K. et al., "Ten Simple Rules for
Reproducible Computational Research", PLOS Comput. Biol., 2013,
DOI: 10.1371/journal.pcbi.1003285, rule 6.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".jp2",
)

# Folders created or consumed by the pipeline next to ``images/``.  They
# are skipped when scanning for ``images/`` folders so that a second run
# never descends into its own outputs.
PIPELINE_SIDE_FOLDERS: Tuple[str, ...] = (
    "masks", "thumbnails_all_body", "thumbnails_head_zoomed",
    "thumbnails_blastemes", "thumbnails_eyespots", "Eyespots heatmaps",
    "blasteme_masks", "temp_formated",
)

_TOKEN = re.compile(r"^[A-Za-z0-9()\-\.]+$")


@dataclass(frozen=True)
class ImageName:
    file_name: str
    species: str
    compound: str
    concentration: str
    plate: str
    position: str
    date: str
    comment: str
    extension: str

    @property
    def planaria_key(self) -> Tuple[str, str]:
        """Fields that identify one animal across dates (plate + position)."""
        return (self.plate, self.position)


def parse_image_name(file_name: str) -> Optional[ImageName]:
    """Return the parsed fields, or ``None`` when the name is not compliant.

    Convention: ``Species_Compound_Concentration_Plate_Position_Date``
    plus an optional trailing comment.
    """
    base = os.path.basename(file_name)
    stem, ext = os.path.splitext(base)
    if ext.lower() not in IMAGE_EXTENSIONS:
        return None
    parts = stem.split("_")
    if len(parts) < 6:
        return None
    species, compound, concentration, plate, position, date = parts[:6]
    if not (species and compound and concentration and plate and position and date):
        return None
    if not (_TOKEN.match(plate) and _TOKEN.match(position)):
        return None
    comment = "_".join(parts[6:]) if len(parts) > 6 else ""
    return ImageName(
        base, species, compound, concentration,
        plate, position, date, comment, ext.lower(),
    )


_DATE_LAYOUTS = (
    "%Y%m%d", "%Y-%m-%d", "%d%m%Y", "%d-%m-%Y", "%y%m%d", "%d%m%y",
    "%Y_%m_%d", "%Y.%m.%d", "%d.%m.%Y",
)


def parse_date(token: str) -> Optional[_dt.date]:
    """Best-effort conversion of the DATE field to a :class:`datetime.date`.

    Returns ``None`` when no layout matches; callers then fall back to a
    lexicographic sort on the raw token.
    """
    t = str(token).strip()
    digits = re.sub(r"\D", "", t)
    for layout in _DATE_LAYOUTS:
        for cand in (t, digits):
            try:
                return _dt.datetime.strptime(cand, layout).date()
            except ValueError:
                continue
    return None


def sort_key(name: ImageName):
    d = parse_date(name.date)
    return (0, d.toordinal(), name.file_name) if d is not None else (1, 0, name.date, name.file_name)


def day_index(names: List[ImageName]) -> Dict[str, int]:
    """Day number of every image relative to the first acquisition of the
    same planaria (0 for the first date).  When a date cannot be parsed the
    rank of the raw token among the planaria's dates is used instead."""
    out: Dict[str, int] = {}
    groups: Dict[Tuple[str, str], List[ImageName]] = {}
    for n in names:
        groups.setdefault(n.planaria_key, []).append(n)
    for members in groups.values():
        dated = [(parse_date(m.date), m) for m in members]
        if all(d is not None for d, _ in dated):
            d0 = min(d for d, _ in dated)
            for d, m in dated:
                out[m.file_name] = (d - d0).days
        else:
            toks = sorted({m.date for m in members})
            for m in members:
                out[m.file_name] = toks.index(m.date)
    return out


def list_images(images_dir: str) -> List[str]:
    try:
        return sorted(
            f for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
    except OSError:
        return []


def scan_images_folders(root: str) -> List[str]:
    """Every folder named ``images`` under ``root`` (any depth), as absolute
    paths, skipping the pipeline's own side folders."""
    found: List[str] = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, _files in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in PIPELINE_SIDE_FOLDERS and not d.startswith("masks")
        )
        if "images" in dirnames:
            found.append(os.path.join(dirpath, "images"))
            dirnames.remove("images")
    return found


def audit_folder(images_dir: str) -> Tuple[List[ImageName], List[str]]:
    """Split the files of one ``images/`` folder into compliant parsed names
    and the list of non-compliant file names."""
    ok: List[ImageName] = []
    bad: List[str] = []
    for f in list_images(images_dir):
        p = parse_image_name(f)
        (ok if p is not None else bad).append(p if p is not None else f)
    return ok, bad
