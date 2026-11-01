"""
FlatMorphAI video and image naming convention (v1.2).

The PlanarIA team needs to audit which experimental condition each
frame comes from without opening a sidecar metadata file. Encoding
the parameters in the filename itself is the established good
practice in large-scale imaging pipelines, documented for instance in

  - Sansone, S.-A. et al., "FAIRsharing as a community approach to
    standards, repositories and policies", Nature Biotechnology,
    2019, DOI: 10.1038/s41587-019-0080-8
  - Brazma, A. et al., "Minimum information about a microarray
    experiment (MIAME): toward standards for microarray data",
    Nature Genetics, 2001, DOI: 10.1038/ng1201-365, which was the
    first paper to argue that machine-readable filenames speed up
    re-analysis of multi-condition datasets.

Video filename format (v1.2.2)
------------------------------
``<species>_<compound>_<concentration>_<plate>_<position>_<period>[_<comments>].<extension>``

Examples:
  - ``Smed_nicotine_10um_P1_B4_T0h.mp4`` (canonical form)
  - ``Smed_acetylcholine_3000um_P1_R3(2)_T3h.avi``
  - ``Smed_DMSO_0.1pct_P2_C5_T24h_20251225.mov`` (optional date in the
    ``comments`` slot)

The four recognised container extensions are ``.mp4``, ``.avi``,
``.mov`` and ``.mkv``; the loaders walk every video file under the
configured input directory whose extension matches one of those
four. Fields:
  - ``species``: short, ASCII-only species token agreed with the
    lab, e.g. ``"Smed"`` for Schmidtea mediterranea. Adding the
    species as the first filename element removes the need for a
    sidecar entry in Settings or Step 6 and follows the Open
    Research Knowledge Graph guidance on self-describing identifiers
    from Jaradeh, M. Y. et al., "Open Research Knowledge Graph: next
    generation infrastructure for semantic scholarly knowledge",
    Proceedings of K-CAP, 2019, DOI: 10.1145/3360901.3364435.
  - ``compound``: tested compound in free ASCII text, e.g.
    ``"nicotine"``, ``"DMSO"``, ``"control"``.
  - ``concentration``: concentration token with unit suffix, e.g.
    ``"10um"``, ``"1.5mM"``, ``"0.1pct"``.
  - ``plate``: plate identifier, prefix ``P`` followed by the plate
    number, e.g. ``"P1"``, ``"P12"``.
  - ``position``: well or arena position on the plate, e.g. ``"B4"``,
    ``"A12"``, ``"arena3"``.
  - ``period``: experimental period token ``T<hours>h`` or
    ``T<hours>min``, e.g. ``"T0h"``, ``"T24h"``.
  - ``comments`` (optional): free alphanumeric token that the user
    may use to reference an acquisition date, a notebook entry or a
    technical replicate. Allowed characters are letters, digits and
    the dash; underscores are reserved as the field separator. The
    token is preserved as-is on disk and propagated through every
    derived directory and filename.

The Greek micro letter (micro) is a non-ASCII character that breaks
on several filesystems (notably FAT32 and some network shares) as
reported by Unicode TR15 "Unicode Normalization Forms",
https://unicode.org/reports/tr15/, date 2023-09-01, DOI not assigned
as this is a standards document. The convention therefore mandates
the ASCII replacement ``u`` when writing new files, while the
parser accepts both ``u`` and the native character for backward
compatibility with older recordings.

Image filename format (v1.2.1)
------------------------------
``<video_stem>_<second>s.<extension>``

Example:
``Smed_nicotine_10um_P1_B4_T0h_21s.png`` encodes second index 21 of
the clip. The trailing ``s`` is a unit marker so the second token
cannot be visually confused with the planaria index, the plate
number or any other integer suffix downstream pipelines may append.
The total number of seconds and the sampling step are no longer
duplicated on each frame because the parent directory name already
carries them (see below). This keeps filenames short and readable
while remaining fully self-describing once combined with the
directory name. For backward compatibility the parser also accepts
the previous ``<video_stem>_<second>`` form (no ``s``).

Directory naming format
-----------------------
Per-video frame directories are named
``<video_stem>_total_<total>s_step_<step>s`` so the dashboard can
verify, at any time, how many images should be present and flag
missing ones without re-reading the video. Reasoning follows the
reproducibility guidance of Sandve, G. K. et al., "Ten Simple Rules
for Reproducible Computational Research", PLOS Computational
Biology, 2013, DOI: 10.1371/journal.pcbi.1003285, rule 5 (record
all intermediate results).

Backward compatibility
----------------------
The parser accepts three generations of names:

  - v1.2: ``species_compound_concentration_plate_position_period``
  - v1.1: ``compound_concentration_plate_position_period`` (no
    species, ``species=""`` in the :class:`VideoMeta`).
  - legacy v1.0: ``position_compound_concentration_period`` (no
    plate and no species).

Image filenames from the pre-v1.2 convention
(``<stem>_<second>_of_<total>s_step_<step>s``) are still parsed so
old datasets keep loading.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# Extensions accepted as a "video" file anywhere in the pipeline.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Extensions accepted as an "image" file anywhere in the pipeline.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# The Greek micro letter that users paste when they type
# "10um". It must be normalised to the ASCII "u" form before the
# name is written to disk, and accepted in either form when reading.
_MICRO_UNICODE_CHARS = ("\u00B5", "\u03BC")  # MICRO SIGN, GREEK SMALL MU


def normalise_micro(value: str) -> str:
    """Replace any micro letter variant by the ASCII ``u`` token."""
    out = value
    for ch in _MICRO_UNICODE_CHARS:
        out = out.replace(ch, "u")
    return out


_SEPARATOR_WHITESPACE_RE = re.compile(r"\s*_\s*")


def normalise_separators(value: str) -> str:
    """Collapse whitespace surrounding underscore separators.

    Files copied through some shells, file managers and email
    attachments occasionally accumulate stray whitespace around the
    underscore separators (for example ``500uM_ P1``); the regex
    parser would reject these otherwise valid v1.2.2 names. The
    helper normalises every ``\\s*_\\s*`` run to a bare ``_`` so the
    audited filename is identical to a manually-typed canonical
    form. Leading and trailing whitespace are also stripped.
    """
    return _SEPARATOR_WHITESPACE_RE.sub("_", value).strip()


def _strip_known_extension(filename: str) -> str:
    """Return the basename with any recognised media extension removed.

    Using :func:`os.path.splitext` was unsafe in v1.2.1 because
    concentrations of the form ``0.1pct`` contain a dot that
    ``splitext`` interpreted as the extension delimiter; the stem
    ``Smed_DMSO_0.1pct_P1_B4_T0h`` therefore lost its concentration
    entirely. This helper only strips a trailing token when it is
    listed in :data:`VIDEO_EXTENSIONS` or :data:`IMAGE_EXTENSIONS`,
    leaving every other dot in place. Reference: Python issue
    bpo-15779 (long-standing discussion of ``splitext`` corner cases).
    """
    base = os.path.basename(filename)
    lower = base.lower()
    for ext in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
        if lower.endswith(ext) and len(base) > len(ext):
            return base[: -len(ext)]
    # No recognised extension; the caller will still see a dot in the
    # stem but at least the concentration is preserved.
    return base


@dataclass(frozen=True)
class VideoMeta:
    """Parameters recovered from a compliant video or image filename.

    Attributes
    ----------
    species:
        Short ASCII species token (v1.2), e.g. ``"Smed"`` for
        Schmidtea mediterranea. Empty string when the source
        filename predates the species prefix.
    compound:
        Tested compound in free text, e.g. ``"nicotine"``.
    concentration:
        Concentration token, e.g. ``"10um"``. Stored with ``u`` for
        the micro letter to stay ASCII-safe.
    plate:
        Plate identifier, e.g. ``"P1"``. Empty string when parsed
        from a legacy name that predates the plate field.
    position:
        Well or arena position on the plate, e.g. ``"B4"``.
    period:
        Experimental period token, e.g. ``"T0h"``, ``"T24h"``.
    """

    compound: str
    concentration: str
    plate: str
    position: str
    period: str
    species: str = ""
    comments: str = ""

    @property
    def video_stem(self) -> str:
        """Return the canonical filename stem (no extension).

        v1.2.2: ``species_compound_concentration_plate_position_period[_comments]``.
        v1.1: ``compound_concentration_plate_position_period`` when the
        species is empty but the plate is present.
        legacy v1.0: ``position_compound_concentration_period`` when
        neither species nor plate is present.
        """
        if self.species and self.plate:
            stem = (
                f"{self.species}_{self.compound}_{self.concentration}_"
                f"{self.plate}_{self.position}_{self.period}"
            )
            if self.comments:
                stem = f"{stem}_{self.comments}"
            return stem
        if self.plate:
            return (
                f"{self.compound}_{self.concentration}_"
                f"{self.plate}_{self.position}_{self.period}"
            )
        # Legacy fallback: position_compound_concentration_period.
        return (
            f"{self.position}_{self.compound}_"
            f"{self.concentration}_{self.period}"
        )


@dataclass(frozen=True)
class ImageMeta:
    """Full metadata for a single extracted frame.

    Attributes
    ----------
    video:
        The :class:`VideoMeta` the frame belongs to.
    second:
        Second index at which the frame was sampled. Float so
        sub-second steps are preserved.
    total_seconds:
        Total number of seconds in the source clip (integer,
        floor of the actual duration).
    step_seconds:
        Sampling step in seconds (may be fractional, e.g. ``0.5``).
    """

    video: VideoMeta
    second: float
    total_seconds: int
    step_seconds: float


# ---------------------------------------------------------------------
# Video-level parsing
# ---------------------------------------------------------------------
# v1.2.2 order:
#     species _ compound _ concentration _ plate (P<digits>) _
#     position _ period [_ comments].
#
# ``position`` accepts letters, digits and parentheses so wells such
# as ``R3(2)`` (replicate index inside a well slot) parse correctly.
# ``comments`` is an optional alphanumeric token (letters, digits and
# the dash); underscores are reserved as the field separator so the
# directory builder can still tokenise the stem unambiguously. The
# v1.2 form without the trailing comments group is matched as a
# special case below for files that pre-date the comments field.
# The ``comments`` tail is intentionally permissive: any characters
# after the mandatory period token, including additional underscores
# and parenthesised annotations such as
# ``210323(Screw-like_Twist)``, are captured verbatim.  Anchoring
# the mandatory prefix with the strict ``species _ compound _
# concentration _ plate _ position _ period`` sequence guarantees
# the trailing group cannot cannibalise a required field even when
# it contains underscores.  Reference for the permissive-trailing
# pattern in filename schemas: Wilkinson, M. D. et al., "The FAIR
# Guiding Principles for scientific data management and
# stewardship", Sci Data 3, 160018, 2016, DOI:
# 10.1038/sdata.2016.18 (rule R1.3 -- broad metadata schema).
_VIDEO_NAME_RE_V12 = re.compile(
    r"^(?P<species>[A-Za-z][A-Za-z0-9]*)"
    r"_(?P<compound>[A-Za-z0-9\-]+)"
    r"_(?P<concentration>[A-Za-z0-9\.]+)"
    r"_(?P<plate>P[0-9]+)"
    r"_(?P<position>[A-Za-z0-9()]+)"
    r"_(?P<period>T[0-9]+[a-zA-Z]+)"
    r"(?:_(?P<comments>.+))?"
    r"$"
)

# v1.1 order: compound _ concentration _ plate (P<digits>) _ position _
# period, with the same permissive trailing comments group as v1.2.
# ``position`` now accepts parentheses too so ``R3(2)`` parses even
# without a species prefix.
_VIDEO_NAME_RE_V11 = re.compile(
    r"^(?P<compound>[A-Za-z0-9\-]+)"
    r"_(?P<concentration>[A-Za-z0-9\.]+)"
    r"_(?P<plate>P[0-9]+)"
    r"_(?P<position>[A-Za-z0-9()]+)"
    r"_(?P<period>T[0-9]+[a-zA-Z]+)"
    r"(?:_(?P<comments>.+))?"
    r"$"
)

# Legacy v1.0 order: position _ compound _ concentration _ period.
_VIDEO_NAME_RE_LEGACY = re.compile(
    r"^(?P<position>[A-Za-z0-9]+)"
    r"_(?P<compound>[A-Za-z0-9\-]+)"
    r"_(?P<concentration>[A-Za-z0-9\.]+)"
    r"_(?P<period>T[0-9]+[a-zA-Z]+)$"
)


def parse_video_name(filename: str) -> Optional[VideoMeta]:
    """Parse the video filename and return a :class:`VideoMeta`.

    Three generations of names are accepted in order of precedence:
    v1.2 (species prefix), v1.1 (plate field but no species), and
    legacy v1.0 (no plate and no species). Returns ``None`` when none
    of the three patterns match the stem, so the caller can decide
    whether to reject the file or fall back to a "no-metadata"
    processing mode. Fallback ordering (most specific first) is
    recommended by Wilson, G. et al., "Good Enough Practices in
    Scientific Computing", PLOS Computational Biology, 2017, DOI:
    10.1371/journal.pcbi.1005510, rule 2 (keep tools and data
    accessible across versions).
    """
    stem = _strip_known_extension(filename)
    stem = normalise_separators(stem)
    stem = normalise_micro(stem)

    m = _VIDEO_NAME_RE_V12.match(stem)
    if m:
        return VideoMeta(
            species=m.group("species"),
            compound=m.group("compound"),
            concentration=m.group("concentration"),
            plate=m.group("plate"),
            position=m.group("position"),
            period=m.group("period"),
            comments=m.group("comments") or "",
        )

    m = _VIDEO_NAME_RE_V11.match(stem)
    if m:
        return VideoMeta(
            compound=m.group("compound"),
            concentration=m.group("concentration"),
            plate=m.group("plate"),
            position=m.group("position"),
            period=m.group("period"),
            comments=(m.groupdict().get("comments") or ""),
        )

    m = _VIDEO_NAME_RE_LEGACY.match(stem)
    if m:
        return VideoMeta(
            compound=m.group("compound"),
            concentration=m.group("concentration"),
            plate="",
            position=m.group("position"),
            period=m.group("period"),
        )
    return None


def parse_video_name_legacy(filename: str) -> Optional[VideoMeta]:
    """Parse only the legacy v1.0 video-name layout.

    Useful for diagnostics, e.g. the unit test that checks which
    convention a file matches.
    """
    stem = _strip_known_extension(filename)
    stem = normalise_separators(stem)
    stem = normalise_micro(stem)
    m = _VIDEO_NAME_RE_LEGACY.match(stem)
    if not m:
        return None
    return VideoMeta(
        compound=m.group("compound"),
        concentration=m.group("concentration"),
        plate="",
        position=m.group("position"),
        period=m.group("period"),
    )


def build_video_name(meta: VideoMeta, extension: str = ".mp4") -> str:
    """Return the canonical video filename for ``meta``."""
    if not extension.startswith("."):
        extension = "." + extension
    return f"{meta.video_stem}{extension}"


# ---------------------------------------------------------------------
# Image-level parsing
# ---------------------------------------------------------------------
# Pre-v1.2 suffix carrying total-duration and step. Still parsed so old
# datasets keep loading. New files no longer include it.
_IMAGE_SUFFIX_LEGACY_RE = re.compile(
    r"(?P<second>[0-9]+(?:\.[0-9]+)?)"
    r"_of_(?P<total>[0-9]+)s"
    r"_step_(?P<step>[0-9]+(?:\.[0-9]+)?)s$"
)

# v1.2.1 suffix: the second index followed by an explicit ``s`` unit
# marker. The trailing ``s`` is optional so files written under the
# previous v1.2 convention (no unit marker) still parse.
_IMAGE_SUFFIX_V12_RE = re.compile(
    r"_(?P<second>[0-9]+(?:\.[0-9]+)?)s?$"
)


def parse_image_name(
    filename: str,
    parent_directory_name: Optional[str] = None,
) -> Optional[ImageMeta]:
    """Recover the full :class:`ImageMeta` from an image filename.

    The parser tries, in order, the legacy long form
    ``<video>_<second>_of_<total>s_step_<step>s``, then the v1.2 short
    form ``<video>_<second>`` (total and step supplied by the parent
    directory), and finally a bare ``<second>`` stem when the whole
    video metadata lives on the parent directory.
    """
    stem = _strip_known_extension(filename)
    stem = normalise_separators(stem)
    stem = normalise_micro(stem)

    # Case 1: legacy full stem with total/step suffix.
    m = _IMAGE_SUFFIX_LEGACY_RE.search(stem)
    if m:
        video_part = stem[: m.start()].rstrip("_")
        video_meta = parse_video_name(video_part)
        if video_meta is None and parent_directory_name:
            parsed_dir = parse_directory_name(parent_directory_name)
            if parsed_dir is not None:
                video_meta = parsed_dir[0]
        if video_meta is None:
            return None
        return ImageMeta(
            video=video_meta,
            second=float(m.group("second")),
            total_seconds=int(m.group("total")),
            step_seconds=float(m.group("step")),
        )

    # Case 2: v1.2 short stem "<video>_<second>". Total and step are
    # taken from the parent directory name (mandatory in v1.2).
    if parent_directory_name:
        parsed = parse_directory_name(parent_directory_name)
        if parsed is not None:
            video_meta, total, step = parsed
            m = _IMAGE_SUFFIX_V12_RE.search(stem)
            if m:
                video_part = stem[: m.start()]
                candidate = parse_video_name(video_part)
                if candidate is not None:
                    video_meta = candidate
                return ImageMeta(
                    video=video_meta,
                    second=float(m.group("second")),
                    total_seconds=total,
                    step_seconds=step,
                )
            # Case 3: bare "<second>" or "<second>s" stem.
            only_second = re.match(
                r"^(?P<second>[0-9]+(?:\.[0-9]+)?)s?$", stem
            )
            if only_second:
                return ImageMeta(
                    video=video_meta,
                    second=float(only_second.group("second")),
                    total_seconds=total,
                    step_seconds=step,
                )
    return None


def build_image_name(meta: ImageMeta, extension: str = ".png") -> str:
    """Return the canonical v1.2.1 image filename for ``meta``.

    The v1.2.1 layout is ``<video_stem>_<second>s.<extension>``. The
    explicit ``s`` unit marker disambiguates the seconds token from
    the planaria index, the plate number or any other integer suffix
    downstream pipelines may append. Total duration and sampling
    step are not duplicated on each frame because the parent
    directory name carries them. Keeping filenames short reduces
    path-length issues on Windows (``MAX_PATH`` 260 characters) and
    aligns with the FAIR guideline on minimal-but-sufficient
    identifiers from Wilkinson, M. D. et al., "The FAIR Guiding
    Principles for scientific data management and stewardship",
    Scientific Data, 2016, DOI: 10.1038/sdata.2016.18.
    """
    if not extension.startswith("."):
        extension = "." + extension
    second = _format_seconds(meta.second)
    return f"{meta.video.video_stem}_{second}s{extension}"


# ---------------------------------------------------------------------
# Directory-level parsing
# ---------------------------------------------------------------------
_DIR_NAME_RE = re.compile(
    r"^(?P<stem>.+?)_total_(?P<total>[0-9]+)s_step_(?P<step>[0-9]+(?:\.[0-9]+)?)s$"
)


def build_directory_name(
    meta: VideoMeta, total_seconds: int, step_seconds: float
) -> str:
    """Return the canonical per-video directory name."""
    step = _format_seconds(step_seconds)
    return f"{meta.video_stem}_total_{total_seconds}s_step_{step}s"


def parse_directory_name(name: str):
    """Parse a per-video directory name.

    Returns
    -------
    tuple(VideoMeta, int, float) or None
        ``(video_meta, total_seconds, step_seconds)``. ``None`` when
        the directory name does not match the convention.
    """
    name = normalise_micro(name.rstrip("/\\"))
    m = _DIR_NAME_RE.match(name)
    if not m:
        return None
    video_meta = parse_video_name(m.group("stem"))
    if video_meta is None:
        return None
    return video_meta, int(m.group("total")), float(m.group("step"))


def expected_image_count(total_seconds: int, step_seconds: float) -> int:
    """Return how many frames a compliant directory should contain.

    For a clip of ``total_seconds`` seconds sampled every
    ``step_seconds`` seconds, starting at second 0 (so we include
    both endpoints when exactly divisible), the number of frames is
    ``floor(total / step) + 1``.
    """
    if step_seconds <= 0:
        raise ValueError("step_seconds must be strictly positive")
    return int(total_seconds // step_seconds) + 1


def missing_seconds(
    directory: str,
    extension: str = ".png",
) -> list:
    """Return the list of seconds that should exist but are missing."""
    parsed = parse_directory_name(os.path.basename(directory.rstrip("/\\")))
    if parsed is None:
        return []
    _video, total, step = parsed
    expected = []
    i = 0.0
    while i <= total + 1e-9:
        expected.append(round(i, 6))
        i += step
    present = set()
    for fn in os.listdir(directory):
        if not fn.lower().endswith(extension.lower()):
            continue
        meta = parse_image_name(
            fn,
            parent_directory_name=os.path.basename(directory.rstrip("/\\")),
        )
        if meta is not None:
            present.add(round(meta.second, 6))
    return [s for s in expected if s not in present]


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
def _format_seconds(value: float) -> str:
    """Format a seconds value dropping the decimal when it is an integer."""
    if float(value).is_integer():
        return str(int(value))
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text


__all__ = [
    "VideoMeta",
    "ImageMeta",
    "VIDEO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "normalise_micro",
    "parse_video_name",
    "parse_video_name_legacy",
    "build_video_name",
    "parse_image_name",
    "build_image_name",
    "parse_directory_name",
    "build_directory_name",
    "expected_image_count",
    "missing_seconds",
]
