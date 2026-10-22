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
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from .path_utils import normalise_source_path
from .readers.set_reader import read_set_info, set_camera_streams

# One-level scan caps. A source folder with tens of thousands of frames must not
# stall the validate request, and a rule only needs a representative sample.
SCAN_MAX_ENTRIES = 2000
SCAN_MAX_SUBDIRS = 64
# Total non-hidden entries examined per directory before giving up, as a multiple of
# SCAN_MAX_ENTRIES. Bounds the walk of a huge folder whose other kind of entry
# (files vs. subdirectories) has not yet reached its own cap.
SCAN_HARD_STOP_FACTOR = 10

# Folder names listed in a ``found`` sentence before the rest collapse to "and N more".
FOUND_MAX_NAMES = 5

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
_SET_SUFFIX = ".set"
# A stream's data file is ``{prefix}-1.ims`` (see IMSFrameInfo.data_path).
_STREAM_DATA_SUFFIX = "-1.ims"
_STREAM_INDEX_SUFFIX = "-0.ims"


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


def _plural(count: int, noun: str) -> str:
    """``1 folder`` / ``3 folders``."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _count_text(count: int, truncated: bool) -> str:
    """A count, suffixed ``+`` when the scan stopped counting early."""
    return f"{count}+" if truncated else str(count)


def _ext_counts_text(counts: Mapping[str, int], truncated: bool, files: bool) -> str:
    """``500 .tif files`` (several extensions joined by ``, ``), ``files`` word optional."""
    parts = []
    for ext in sorted(counts):
        count = counts[ext]
        text = f"{_count_text(count, truncated)} {ext}"
        if files:
            text += " file" if count == 1 and not truncated else " files"
        parts.append(text)
    return ", ".join(parts)


def _names_text(names: Sequence[str]) -> str:
    """Up to ``FOUND_MAX_NAMES`` names, then ``and N more``."""
    shown = ", ".join(names[:FOUND_MAX_NAMES])
    extra = len(names) - FOUND_MAX_NAMES
    return f"{shown} and {extra} more" if extra > 0 else shown


def _folders_clause(subdirs: Sequence[SubdirInfo]) -> str:
    """The subfolder part of a scan sentence."""
    head = (
        f"{_plural(len(subdirs), 'folder')} ({_names_text([d.name for d in subdirs])})"
    )
    if all(not d.ext_counts for d in subdirs):
        return f"{head}, none with images"
    signatures = {tuple(sorted(d.ext_counts.items())) + (d.truncated,) for d in subdirs}
    first = subdirs[0]
    if len(signatures) == 1 and len(first.ext_counts) == 1:
        files = _ext_counts_text(first.ext_counts, first.truncated, files=True)
        return (
            f"{head} with {files}"
            if len(subdirs) == 1
            else f"{head}, each with {files}"
        )
    details = []
    for d in subdirs[:FOUND_MAX_NAMES]:
        text = (
            _ext_counts_text(d.ext_counts, d.truncated, files=False)
            if d.ext_counts
            else "no images"
        )
        details.append(f"{d.name}: {text}")
    return f"{head}: {', '.join(details)}"


def scan_sentence(scan: SourceScan) -> str:
    """Describe what is on disk at the source, in one or two plain sentences.

    Used as ``found`` when the source does not load. Counts that hit the scan caps
    carry a ``+``. Listing failures recorded in ``scan.errors`` are appended.
    """
    parts: List[str] = []
    if scan.listed_is_companion and scan.companion_of is not None and scan.listed:
        prefixes = {
            name[: -len(_STREAM_INDEX_SUFFIX)]
            for name in scan.root_names
            if name.lower().endswith(_STREAM_INDEX_SUFFIX)
        }
        parts.append(
            f"{scan.listed.as_posix()} is the data folder of "
            f"{scan.companion_of.as_posix()} ({_plural(len(prefixes), 'stream')})."
        )
    else:
        if not scan.exists:
            parts.append(f"{scan.path.as_posix()} does not exist.")
        if scan.listed is None:
            parts.append("None of its parent folders exist.")
        else:
            total = sum(scan.root_ext_counts.values())
            root_truncated = scan.truncated and total >= SCAN_MAX_ENTRIES
            clauses = []
            if scan.root_ext_counts:
                clauses.append(
                    _ext_counts_text(scan.root_ext_counts, root_truncated, files=True)
                )
            elif scan.subdirs:
                clauses.append("no images here")
            if scan.subdirs:
                clauses.append(_folders_clause(scan.subdirs))
            body = "; ".join(clauses) if clauses else "no images here and no folders"
            parts.append(f"In {scan.listed.as_posix()}: {body}.")
    for error in scan.errors:
        parts.append(f"{error[:1].upper()}{error[1:]}.")
    return " ".join(parts)


def pass_sentence(
    camera_count: int, found_count: int, unit: str, image_size: Tuple[int, int]
) -> str:
    """``3 cameras × 500 .im7, 1280×1024.`` for a source that loads.

    ``image_size`` is ``(W, H)`` exactly as the validator returns it.
    """
    return f"{_pass_head(camera_count, found_count, unit)}, {image_size[0]}\u00d7{image_size[1]}."


def _pass_head(camera_count: int, found_count: Any, unit: str) -> str:
    """``3 cameras × 500 .im7`` (the camera factor only for several cameras)."""
    count = f"{found_count} {unit}"
    return f"{camera_count} cameras \u00d7 {count}" if camera_count > 1 else count


def streams_sentence(set_name: str, streams: Sequence[StreamInfo]) -> str:
    """List a ``.set`` recording's streams, merging consecutive equal ones into ranges."""
    groups: List[List[StreamInfo]] = []
    for stream in streams:
        key = (stream["width"], stream["height"], stream["decoder"])
        if groups and key == (
            groups[-1][0]["width"],
            groups[-1][0]["height"],
            groups[-1][0]["decoder"],
        ):
            groups[-1].append(stream)
        else:
            groups.append([stream])
    items = []
    for group in groups:
        first, last = group[0], group[-1]
        label = (
            str(first["index"])
            if len(group) == 1
            else f"{first['index']}\u2013{last['index']}"
        )
        items.append(
            f"{label}: {first['width']}\u00d7{first['height']} {first['decoder']}"
        )
    return f"{set_name}: {_plural(len(streams), 'stream')} \u2014 {', '.join(items)}."


def set_stream_infos(set_path: Path, camera_count: int) -> List[StreamInfo]:
    """Per-stream table of a ``.set`` recording, with each stream's camera.

    ``camera`` is the 1-based position in the unique stream split for
    ``camera_count`` cameras, or None for every stream when no unique split exists.

    Raises
    ------
    Exception
        Whatever ``read_set_info`` raises for an unreadable recording.
    """
    info = read_set_info(set_path)
    try:
        split = set_camera_streams(set_path, camera_count, info)
    except ValueError:
        split = []
    camera_of = {idx: cam for cam, group in enumerate(split, start=1) for idx in group}
    return [
        StreamInfo(
            index=frame.frame_idx,
            prefix=frame.data_path.name[: -len(_STREAM_DATA_SUFFIX)],
            width=frame.width,
            height=frame.height,
            decoder=frame.decoder,
            camera=camera_of.get(frame.frame_idx),
        )
        for frame in info.frames
    ]


def found_for(
    scan: SourceScan,
    ctx: DiagnosisContext,
    result: Mapping[str, Any],
    streams: Optional[List[StreamInfo]],
    streams_error: Optional[str],
) -> str:
    """The ``found`` sentence for a validation result.

    A ``.set`` file target reports its stream table (or why there is none). Any
    other valid source reports what loaded. Everything else reports what is on disk.
    """
    if ctx.image_type == "lavision_set" and scan.path.suffix.lower() == _SET_SUFFIX:
        companion = scan.path.with_suffix("")
        if streams:
            return streams_sentence(scan.path.name, streams)
        if not companion.is_dir():
            return (
                f"{scan.path.as_posix()} has no data folder "
                f"{companion.as_posix()} beside it."
            )
        if streams_error is not None:
            return f"{scan.path.name}: streams could not be read ({streams_error})."
    fmt = result.get("format_detected")
    if result.get("valid") and fmt:
        unit = {"set": "entries", "cine": "frames"}.get(fmt, f".{fmt}")
        head = _pass_head(ctx.camera_count, result["found_count"], unit)
        size = result.get("image_size")
        if size is None:
            return f"{head}, image size could not be read."
        return pass_sentence(ctx.camera_count, result["found_count"], unit, size)
    return scan_sentence(scan)
