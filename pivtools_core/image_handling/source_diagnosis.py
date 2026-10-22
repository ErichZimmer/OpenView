"""Source diagnosis: a bounded filesystem scan plus the shared diagnosis types.

The GUI validates an image source (Setup page and the calibration tabs) and, when the
configured source does not load, needs to say what is actually on disk and offer
one-click fixes. This module holds the types those suggestions are built from and
``scan_source``, the single filesystem look every rule shares. Rules only build
suggestions; nothing here writes configuration.

Import direction is one-way at module level: this module imports ``path_utils``;
``path_utils`` imports this module only lazily.
"""

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

from .path_utils import normalise_source_path

# One-level scan caps. A source folder with tens of thousands of frames must not
# stall the validate request, and a rule only needs a representative sample.
SCAN_MAX_ENTRIES = 2000
SCAN_MAX_SUBDIRS = 64
# Total non-hidden entries examined per directory before giving up, as a multiple of
# SCAN_MAX_ENTRIES. Bounds the walk of a huge folder whose other kind of entry
# (files vs. subdirectories) has not yet reached its own cap.
SCAN_HARD_STOP_FACTOR = 10

FAMILY_EXTS: Dict[str, Tuple[str, ...]] = {
    "set": (".set",),
    "ims": (".ims",),
    "im7": (".im7",),
    "cine": (".cine",),
    "raster": (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"),
}
TYPE_FAMILY: Dict[str, str] = {
    "lavision_set": "set",
    "lavision_im7": "im7",
    "cine": "cine",
    "standard": "raster",
}
IMAGE_TYPE_LABELS: Dict[str, str] = {
    "standard": "Standard (TIFF/PNG/JPG)",
    "cine": "Phantom CINE",
    "lavision_set": "LaVision recording (.set + .ims folder)",
    "lavision_im7": "LaVision images (.im7)",
}
# The closed vocabulary a suggestion's ``apply`` may write to the GUI form.
APPLY_FIELDS: FrozenSet[str] = frozenset(
    {
        "source_path",
        "image_type",
        "image_format",
        "start_index",
        "use_camera_subfolders",
        "camera_subfolders",
        "camera_count",
        "camera_numbers",
        "pairing_preset",
    }
)
CAMERA_DIR_RE = re.compile(r"(cam|camera|view|c)?[\s_\-]*0*(\d+)", re.IGNORECASE)

_EXT_TO_FAMILY: Dict[str, str] = {
    ext: family for family, exts in FAMILY_EXTS.items() for ext in exts
}
_STREAMSET_XML = "streamset.xml"
# A DaVis data folder holds one ``<prefix>-0.ims`` per stream (the first frame).
_DAVIS_FIRST_FRAME_RE = re.compile(r".*-0\.ims", re.IGNORECASE)
_DIGITS_RE = re.compile(r"(\d+)")


class Suggestion(TypedDict):
    """One ranked, one-click fix offered to the user."""

    id: str
    message: str
    apply: Dict[str, Any]
    label: str


class StreamInfo(TypedDict):
    """One image stream inside a LaVision ``.set`` recording."""

    index: int
    prefix: str
    width: int
    height: int
    decoder: str
    camera: Optional[int]


@dataclass(frozen=True)
class SubdirInfo:
    """What one immediate subfolder of the listed directory holds."""

    name: str
    camera_number: Optional[int]
    ext_counts: Mapping[str, int]
    is_davis_companion: bool
    truncated: bool


@dataclass(frozen=True)
class SourceScan:
    """Result of ``scan_source``; see the field comments for the exact semantics."""

    raw: str
    path: Path  # normalised
    exists: bool
    is_dir: bool
    ancestor: Optional[Path]  # nearest existing ancestor when not exists
    missing_tail: Tuple[str, ...]  # path components below ``ancestor``
    listed: Optional[Path]  # directory actually listed
    root_names: Tuple[str, ...]  # natural-sorted image-family file names in listed
    root_ext_counts: Mapping[str, int]
    subdirs: Tuple[SubdirInfo, ...]  # natural-sorted
    listed_is_companion: bool  # listed is a DaVis ``.set`` data folder
    companion_of: Optional[Path]  # the ``.set`` whose data folder ``listed`` is
    truncated: bool
    errors: Tuple[str, ...]


@dataclass(frozen=True)
class DiagnosisContext:
    """The configured source plus the page facts the diagnosis rules need."""

    source_path: Path
    image_type: str
    image_format: Tuple[str, ...]
    camera_count: int
    camera_numbers: Tuple[int, ...]
    use_camera_subfolders: bool
    camera_subfolders: Tuple[str, ...]
    pairing_preset: Optional[str]  # None on calibration
    pair_reader: bool  # Setup: not cfg.time_resolved; calibration: False
    start_index: int
    calibration: bool


@dataclass(frozen=True)
class _Listing:
    """Single-directory listing (internal)."""

    image_names: Tuple[str, ...]
    ext_counts: Mapping[str, int]
    subdir_names: Tuple[str, ...]
    is_companion: bool
    truncated: bool
    error: Optional[str]


def natural_key(name: str) -> Tuple:
    """Case-insensitive natural sort key (``B2`` before ``B10``)."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in _DIGITS_RE.split(name)
        if part != ""
    )


def camera_number_of(name: str) -> Optional[int]:
    """Camera number encoded in a folder name (``cam_01``, ``Camera 1``, ``12``), else None."""
    match = CAMERA_DIR_RE.fullmatch(name.strip())
    return int(match.group(2)) if match else None


def family_of(name: str) -> Optional[str]:
    """Image family (a ``FAMILY_EXTS`` key) of a file name by extension, else None."""
    return _EXT_TO_FAMILY.get(os.path.splitext(name)[1].lower())


def make_suggestion(
    rule_id: str, message: str, label: str, apply: Dict[str, Any]
) -> Suggestion:
    """Build a ``Suggestion`` after checking ``apply`` against the closed vocabulary.

    Raises
    ------
    ValueError
        On a key outside ``APPLY_FIELDS``, a value that is not JSON-safe
        (``str``/``int``/``bool`` or a list of those), ``camera_count`` without
        ``camera_numbers``, or an ``image_format`` that is not a list.
    """
    unknown = set(apply) - APPLY_FIELDS
    if unknown:
        raise ValueError(
            f"suggestion {rule_id!r}: unknown apply keys {sorted(unknown)}"
        )
    for key, value in apply.items():
        items = value if isinstance(value, list) else [value]
        if not all(isinstance(v, (str, int, bool)) for v in items):
            raise ValueError(
                f"suggestion {rule_id!r}: apply[{key!r}] is not JSON-safe: {value!r}"
            )
    if "camera_count" in apply and "camera_numbers" not in apply:
        raise ValueError(f"suggestion {rule_id!r}: camera_count without camera_numbers")
    if "image_format" in apply and not isinstance(apply["image_format"], list):
        raise ValueError(f"suggestion {rule_id!r}: image_format must be a list")
    json.dumps(apply)  # final guard: must serialise
    return Suggestion(id=rule_id, message=message, apply=apply, label=label)


def _list_dir(directory: Path) -> _Listing:
    """List one directory (no recursion), skipping dot entries, honouring the caps."""
    names: List[str] = []
    ext_counts: Dict[str, int] = {}
    dirs: List[str] = []
    companion = False
    truncated = False
    seen = 0
    hard_stop = SCAN_HARD_STOP_FACTOR * SCAN_MAX_ENTRIES
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                seen += 1
                if seen > hard_stop:
                    truncated = True
                    break
                lower = entry.name.lower()
                if lower == _STREAMSET_XML or _DAVIS_FIRST_FRAME_RE.fullmatch(lower):
                    companion = True
                if entry.is_dir():
                    if len(dirs) < SCAN_MAX_SUBDIRS:
                        dirs.append(entry.name)
                    else:
                        truncated = True
                    continue
                if family_of(entry.name) is None:
                    continue
                if len(names) < SCAN_MAX_ENTRIES:
                    names.append(entry.name)
                    ext = os.path.splitext(lower)[1]
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                else:
                    truncated = True
                if (
                    len(names) >= SCAN_MAX_ENTRIES
                    and len(dirs) >= SCAN_MAX_SUBDIRS
                    and truncated
                ):
                    break
    except OSError as exc:
        return _Listing(
            (),
            {},
            (),
            False,
            False,
            f"could not list {directory.as_posix()}: {exc.strerror}",
        )
    return _Listing(
        tuple(sorted(names, key=natural_key)),
        ext_counts,
        tuple(sorted(dirs, key=natural_key)),
        companion,
        truncated,
        None,
    )


def _probe(path: Path, errors: List[str]) -> Optional[os.stat_result]:
    """``os.stat`` that never raises. None when the path is absent or unreadable.

    A missing path (ENOENT, ENOTDIR) is an ordinary answer. Any other failure, such
    as a parent directory without search permission, is appended to ``errors``.
    """
    try:
        return os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        errors.append(f"could not stat {path.as_posix()}: {exc.strerror}")
        return None


def scan_source(path: Union[str, Path]) -> SourceScan:
    """Scan a configured source path once, for every diagnosis rule to share.

    The path is quote-normalised. A missing path is walked up to its nearest
    existing ancestor, which is what gets listed. A file lists its parent. Only the
    directory itself and its immediate subfolders are inspected, with the
    ``SCAN_MAX_*`` caps, ignoring dot-files and dot-directories. Listing failures
    are recorded in ``errors`` and never raised.
    """
    raw = str(path)
    norm = Path(normalise_source_path(raw))
    errors: List[str] = []
    norm_stat = _probe(norm, errors)
    exists = norm_stat is not None
    is_dir = exists and stat.S_ISDIR(norm_stat.st_mode)

    ancestor: Optional[Path] = None
    tail: List[str] = []
    if not exists:
        ancestor = norm
        ancestor_stat = None
        while ancestor_stat is None and ancestor.parent != ancestor:
            tail.append(ancestor.name)
            ancestor = ancestor.parent
            ancestor_stat = _probe(ancestor, errors)
        tail.reverse()
        if ancestor_stat is None:
            ancestor = None
    if exists:
        listed: Optional[Path] = norm if is_dir else norm.parent
    elif ancestor is not None:
        listed = ancestor if stat.S_ISDIR(ancestor_stat.st_mode) else ancestor.parent
    else:
        listed = None

    truncated = False
    root = _Listing((), {}, (), False, False, None)
    subdirs: List[SubdirInfo] = []
    companion_of: Optional[Path] = None
    if listed is not None:
        root = _list_dir(listed)
        truncated = root.truncated
        if root.error:
            errors.append(root.error)
        for name in root.subdir_names:
            sub = _list_dir(listed / name)
            if sub.error:
                errors.append(sub.error)
            truncated = truncated or sub.truncated
            subdirs.append(
                SubdirInfo(
                    name=name,
                    camera_number=camera_number_of(name),
                    ext_counts=sub.ext_counts,
                    is_davis_companion=sub.is_companion,
                    truncated=sub.truncated,
                )
            )
        if listed.name:
            candidate = listed.with_name(listed.name + ".set")
            candidate_stat = _probe(candidate, errors)
            if candidate_stat is not None and stat.S_ISREG(candidate_stat.st_mode):
                companion_of = candidate

    return SourceScan(
        raw=raw,
        path=norm,
        exists=exists,
        is_dir=is_dir,
        ancestor=ancestor,
        missing_tail=tuple(tail),
        listed=listed,
        root_names=root.image_names,
        root_ext_counts=root.ext_counts,
        subdirs=tuple(subdirs),
        listed_is_companion=root.is_companion,
        companion_of=companion_of,
        truncated=truncated,
        errors=tuple(errors),
    )
