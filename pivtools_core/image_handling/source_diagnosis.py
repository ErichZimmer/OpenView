"""Source diagnosis: a bounded filesystem scan plus the shared diagnosis types.

The GUI validates an image source (Setup page and the calibration tabs) and, when the
configured source does not load, needs to say what is actually on disk and offer
one-click fixes. This module holds the types those suggestions are built from and
``scan_source``, the single filesystem look every rule shares. Rules only build
suggestions; nothing here writes configuration.

Import direction is one-way at module level: this module imports ``path_utils``;
``path_utils`` imports this module only lazily.
"""

import fnmatch
import json
import logging
import os
import re
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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

from ..calibration_settings import NON_CALIBRATION_IMAGE_TYPES
from .path_utils import (
    RASTER_EXTENSIONS,
    _detect_ab_pair_pattern,
    _suggest_pattern,
    _suggest_pattern_for_role,
    format_to_glob,
    is_calibration_output_dir,
    normalise_source_path,
    start_index_explains_failure,
    start_index_hint_message,
)
from .readers.davis_attr import LabelledCamera
from .readers.set_reader import (
    SetInfo,
    read_set_info,
    set_camera_streams,
    valid_stream_splits,
)

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# The validate routes' answer for a blank source path, which is never scanned
# (``Path("")`` is the server's working directory).
BLANK_SOURCE_MESSAGE = "No source path set."

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
    "pivraw": (".pivraw",),
    "raster": RASTER_EXTENSIONS,  # exactly what the validator globs
}
TYPE_FAMILY: Dict[str, str] = {
    "lavision_set": "set",
    "lavision_im7": "im7",
    "cine": "cine",
    "pivraw": "pivraw",
    "standard": "raster",
}
IMAGE_TYPE_LABELS: Dict[str, str] = {
    "standard": "Standard (TIFF/PNG/JPG)",
    "cine": "Phantom CINE",
    "pivraw": "PIVTOOLS Acquisition (.pivraw)",
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

# Most ``.set`` suggestions one rule offers (several .set files, or several camera counts).
SET_SUGGESTIONS_MAX = 5
# A fix that edits camera selection in calibration also edits the Setup (and PIV) state.
CROSS_STORE_SUFFIX: Dict[str, str] = {"set_streams": " (also changes Setup and PIV)"}
DEFAULT_CROSS_STORE_SUFFIX = " (also changes Setup)"
_CAMERA_APPLY_KEYS = ("camera_count", "camera_numbers")
# A camera owns one stream (single frame) or two (an A/B pair).
_PAIR_STREAMS = 2
# What read_set_info raises for a recording that is present but unreadable:
# a missing or short index/data file, a declared-but-unsupported layout, or
# malformed decoder / manifest XML. Not a catch-all for reader bugs.
_UNREADABLE_RECORDING = (OSError, ValueError, ET.ParseError)


class Suggestion(TypedDict):
    """One ranked, one-click fix offered to the user."""

    id: str
    message: str
    apply: Dict[str, Any]
    label: str


class StreamInfo(TypedDict):
    """One image stream inside a LaVision ``.set`` recording.

    ``width``/``height`` are the output size, None when a transformer id
    PIVTOOLs does not know leaves it unknown. ``label``, ``camera_id``,
    ``model`` and ``role`` come from the recording's camera labels and are None
    when it has none; ``role`` is "A"/"B" for a double-frame camera's frames and
    "" for a single-frame camera.
    """

    index: int
    prefix: str
    width: Optional[int]
    height: Optional[int]
    decoder: str
    camera: Optional[int]
    label: Optional[str]
    camera_id: Optional[str]
    model: Optional[str]
    role: Optional[str]


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
    subdirs_truncated: bool  # listed held more than SCAN_MAX_SUBDIRS subfolders
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
    pair_reader: bool  # PIV pairs streams (not cfg.time_resolved), on both pages
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
    subdirs_truncated: bool
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
    subdirs_truncated = False
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
                    if is_calibration_output_dir(Path(entry.path)):
                        continue
                    if len(dirs) < SCAN_MAX_SUBDIRS:
                        dirs.append(entry.name)
                    else:
                        truncated = subdirs_truncated = True
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
            False,
            f"could not list {directory.as_posix()}: {exc.strerror}",
        )
    return _Listing(
        tuple(sorted(names, key=natural_key)),
        ext_counts,
        tuple(sorted(dirs, key=natural_key)),
        companion,
        truncated,
        subdirs_truncated,
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
    root = _Listing((), {}, (), False, False, False, None)
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
        subdirs_truncated=root.subdirs_truncated,
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


def _folders_clause(subdirs: Sequence[SubdirInfo], capped: bool) -> str:
    """The subfolder part of a scan sentence; ``capped`` marks the count with ``+``."""
    noun = "folder" if len(subdirs) == 1 and not capped else "folders"
    head = (
        f"{_count_text(len(subdirs), capped)} {noun} "
        f"({_names_text([d.name for d in subdirs])})"
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
        # Files, not frame streams: one DaVis 11.4 stream file holds A and B.
        parts.append(
            f"{scan.listed.as_posix()} is the data folder of "
            f"{scan.companion_of.as_posix()} ({_plural(len(prefixes), 'stream file')})."
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
                clauses.append(_folders_clause(scan.subdirs, scan.subdirs_truncated))
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
        size = (
            "size unknown"
            if first["width"] is None
            else f"{first['width']}\u00d7{first['height']}"
        )
        items.append(f"{label}: {size} {first['decoder']}")
    return f"{set_name}: {_plural(len(streams), 'stream')} \u2014 {', '.join(items)}."


def _streams_text(streams: Sequence[int]) -> str:
    """``stream 4`` / ``streams 0\u20131``."""
    if len(streams) == 1:
        return f"stream {streams[0]}"
    return f"streams {streams[0]}\u2013{streams[-1]}"


def _labelled_cameras_text(cameras: Sequence[LabelledCamera]) -> str:
    """``4 cameras from the recording's own labels: Camera 1 (...) on streams 0\u20131; ...``."""
    listing = "; ".join(f"{c.describe()} on {_streams_text(c.frames)}" for c in cameras)
    return f"{_cameras(len(cameras))} from the recording's own labels: {listing}."


def recording_notes(info: SetInfo) -> str:
    """Where a ``.set`` recording's cameras come from, as facts for ``found``.

    Partial labels (DaVis 11.4 writes the camera keys on a leading run of
    images only) are counted. DaVis's own warning is not a fact about the
    cameras: :func:`recording_warnings` reports it.
    """
    notes: List[str] = []
    if info.cameras is not None:
        notes.append(_labelled_cameras_text(info.cameras))
        counts = {c.labelled_images for c in info.cameras}
        if counts != {info.n_entries}:
            if len(counts) == 1:
                where = f"{counts.pop()} of {info.n_entries} images"
            else:
                where = ", ".join(
                    f"{c.labelled_images} of {info.n_entries} images for {c.label}"
                    for c in info.cameras
                )
            notes.append(f"Camera labels on {where}; every labelled image agrees.")
    elif info.attribute_files:
        notes.append(
            "No camera labels: its .attr files carry no camera keys, so cameras "
            "are matched to streams by shape."
        )
    else:
        notes.append(
            "No camera labels: the recording has no .attr files, so cameras are "
            "matched to streams by shape."
        )
    return " ".join(notes)


def recording_warnings(info: SetInfo) -> List[str]:
    """What DaVis itself warns about a ``.set`` recording, one sentence each.

    ``SetIsSuspicious`` set to anything but 0 is a warning, not a refusal
    (decided 17 Nov): DaVis does not say why, so the user checks the images.
    """
    if info.suspicious_flag is None or info.suspicious_flag == "0":
        return []
    return [
        f"DaVis marked this recording as suspicious (SetIsSuspicious = "
        f"{info.suspicious_flag} in {info.set_dir.name}.set) without saying "
        f"why; check its images before processing."
    ]


def set_stream_infos(set_path: Path, camera_count: int) -> List[StreamInfo]:
    """Per-stream table of a ``.set`` recording, with each stream's camera.

    With camera labels, ``camera`` is the labelled camera's number whatever
    ``camera_count`` says. Without, it is the 1-based position in the unique
    stream split for ``camera_count`` cameras, or None for every stream when no
    unique split exists.

    Raises
    ------
    Exception
        Whatever ``read_set_info`` raises for an unreadable recording.
    """
    info = read_set_info(set_path)
    if info.cameras is not None:
        split = [c.frames for c in info.cameras]
    else:
        try:
            split = set_camera_streams(set_path, camera_count, info)
        except ValueError:
            split = []
    camera_of = {idx: cam for cam, group in enumerate(split, start=1) for idx in group}
    model_of = {idx: c.model for c in info.cameras or () for idx in c.frames}
    out: List[StreamInfo] = []
    for frame in info.frames:
        shape = None if frame.has_unknown_transformer else frame.shape
        out.append(
            StreamInfo(
                index=frame.frame_idx,
                prefix=frame.data_path.name[: -len(_STREAM_DATA_SUFFIX)],
                width=None if shape is None else shape[1],
                height=None if shape is None else shape[0],
                decoder=frame.decoder,
                camera=camera_of.get(frame.frame_idx),
                label=frame.camera_label,
                camera_id=frame.camera_id,
                model=model_of.get(frame.frame_idx),
                role=frame.frame_role,
            )
        )
    return out


def found_for(
    scan: SourceScan,
    ctx: DiagnosisContext,
    result: Mapping[str, Any],
    streams: Optional[List[StreamInfo]],
    streams_error: Optional[str],
) -> str:
    """The ``found`` sentence for a validation result.

    A ``.set`` file target reports its stream table and where its cameras come
    from (or why there is none). Any other valid source reports what loaded.
    Everything else reports what is on disk.
    """
    if ctx.image_type == "lavision_set" and scan.path.suffix.lower() == _SET_SUFFIX:
        companion = scan.path.with_suffix("")
        if streams:
            # The stream table was just read from the same cached SetInfo.
            notes = recording_notes(read_set_info(scan.path))
            return f"{streams_sentence(scan.path.name, streams)} {notes}"
        # Only a .set that exists can lack its data folder. A missing one falls
        # through to the disk description, which says it does not exist.
        if scan.exists and not companion.is_dir():
            return (
                f"{scan.path.as_posix()} has no data folder "
                f"{companion.as_posix()} beside it."
            )
        if streams_error is not None:
            return f"{scan.path.name}: streams could not be read ({streams_error})."
    loaded = loaded_sentence(ctx, result)
    return loaded if loaded is not None else scan_sentence(scan)


def loaded_sentence(ctx: DiagnosisContext, result: Mapping[str, Any]) -> Optional[str]:
    """What loaded, for a valid result that says how much; else None.

    A result without ``found_count`` cannot say how much loaded, so the caller
    describes the disk instead.
    """
    fmt = result.get("format_detected")
    if not (result.get("valid") and fmt and "found_count" in result):
        return None
    unit = {"set": "entries", "cine": "frames", "pivraw": "frames"}.get(fmt, f".{fmt}")
    head = _pass_head(ctx.camera_count, result["found_count"], unit)
    size = result.get("image_size")
    if size is None:
        return f"{head}, image size could not be read."
    return pass_sentence(ctx.camera_count, result["found_count"], unit, size)


@dataclass(frozen=True)
class RuleInput:
    """Everything a diagnosis rule may look at."""

    scan: SourceScan
    ctx: DiagnosisContext
    camera: int
    camera_path: Path
    expected_count: Optional[int]
    # The core validator's first frame index over the FULL matching listing
    # (``result["detected_start_index"]``); the scan's names are only a sample.
    detected_start_index: Optional[int]
    # One request's scans, so a rule that lists another directory (the parent,
    # in ``inside_camera_folder``) shares it across the request's cameras.
    scan_cache: Optional[Dict[str, SourceScan]] = None


Rule = Callable[[RuleInput], List[Suggestion]]


def cross_store_label(
    ctx: DiagnosisContext, rule_id: str, label: str, apply: Mapping[str, Any]
) -> str:
    """Append the cross-store note to ``label`` on calibration, for camera-selection fixes."""
    if ctx.calibration and any(key in apply for key in _CAMERA_APPLY_KEYS):
        return label + CROSS_STORE_SUFFIX.get(rule_id, DEFAULT_CROSS_STORE_SUFFIX)
    return label


def _is_set_file(path: Path) -> bool:
    """True for a path whose name ends in ``.set``."""
    return path.suffix.lower() == _SET_SUFFIX


def _ordered_set_names(scan: SourceScan) -> List[str]:
    """``.set`` names at the listed folder, recordings with a data folder first."""
    names = [n for n in scan.root_names if _is_set_file(Path(n))]
    # The sort is stable, so natural order holds within each group.
    names.sort(key=lambda n: not (scan.listed / n).with_suffix("").is_dir())
    return names


def rule_set_in_folder(inp: RuleInput) -> List[Suggestion]:
    """The source is a folder holding ``.set`` files (or a missing ``.set`` beside some)."""
    scan = inp.scan
    if inp.ctx.image_type != "lavision_set" or scan.listed is None:
        return []
    folder_target = scan.exists and scan.is_dir
    missing_set = (
        not scan.exists
        and _is_set_file(scan.path)
        and scan.ancestor is not None
        and scan.ancestor == scan.path.parent
    )
    if not (folder_target or missing_set):
        return []
    out: List[Suggestion] = []
    for name in _ordered_set_names(scan)[:SET_SUGGESTIONS_MAX]:
        file = scan.listed / name
        out.append(
            make_suggestion(
                "set_in_folder",
                f"{scan.listed.as_posix()} holds the recording {name}. "
                "Use the .set file itself as the source.",
                f'Use "{name}"',
                {"source_path": str(file)},
            )
        )
    return out


def rule_set_companion(inp: RuleInput) -> List[Suggestion]:
    """The source is a ``.set`` recording's data folder (or a file in it)."""
    scan = inp.scan
    set_file = scan.companion_of
    if set_file is None or not scan.exists or _is_set_file(scan.path):
        return []
    apply: Dict[str, Any] = {"source_path": str(set_file)}
    if inp.ctx.image_type != "lavision_set":
        apply["image_type"] = "lavision_set"
    return [
        make_suggestion(
            "set_companion",
            f"{scan.listed.as_posix()} is the data folder of {set_file.name}. "
            "Use the .set file as the source.",
            f'Use "{set_file.name}"',
            apply,
        )
    ]


def _and_join(items: Sequence[Any]) -> str:
    """``2`` / ``2 and 3`` / ``1, 2 and 3``."""
    texts = [str(i) for i in items]
    return texts[0] if len(texts) == 1 else f"{', '.join(texts[:-1])} and {texts[-1]}"


def _runs(numbers: Sequence[int]) -> List[str]:
    """Collapse consecutive numbers into ``a\u2013b`` runs (``[2, 3, 5]`` -> ``["2\u20133", "5"]``)."""
    out: List[str] = []
    start = prev = None
    for n in numbers:
        if start is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            out.append(str(start) if start == prev else f"{start}\u2013{prev}")
        start = prev = n
    if start is not None:
        out.append(str(start) if start == prev else f"{start}\u2013{prev}")
    return out


def _cameras(count: int) -> str:
    """``1 camera`` / ``3 cameras``."""
    return _plural(count, "camera")


def set_streams_message(
    streams: Sequence[StreamInfo],
    split: Sequence[Tuple[int, ...]],
    process: Sequence[int],
    pair_reader: bool,
) -> str:
    """Explain a stream split and what to set, e.g. for a PLIF camera beside A/B pairs.

    ``split`` holds the 0-based stream indices of each camera, ``process`` the 1-based
    cameras to process. "single frame, likely PLIF" is said only when the pair reader
    is in use and the split also holds A/B pairs.
    """
    by_index = {s["index"]: s for s in streams}
    has_pair = any(len(g) == _PAIR_STREAMS for g in split)
    clauses: List[str] = []
    pair_cams: List[int] = []
    pair_streams: List[str] = []
    for cam, group in enumerate(split, start=1):
        if len(group) == _PAIR_STREAMS:
            pair_cams.append(cam)
            pair_streams.append(f"{group[0]}\u2013{group[-1]}")
            continue
        info = by_index[group[0]]
        size = f"{info['width']}\u00d7{info['height']}"
        note = (
            f"{size}, single frame, likely PLIF" if pair_reader and has_pair else size
        )
        clauses.append(f"Camera {cam} = stream {group[0]} ({note})")
    if pair_cams:
        noun = "A/B pair" if len(pair_cams) == 1 else "A/B pairs"
        clauses.append(
            f"{'camera' if len(pair_cams) == 1 else 'cameras'} "
            f"{_and_join(_runs(pair_cams))} = {noun} ({', '.join(pair_streams)})"
        )
    body = "; ".join(clauses)
    body = body[0].upper() + body[1:]
    k = len(split)
    if list(process) == list(range(1, k + 1)):
        tail = f"Set {_cameras(k)}."
    else:
        noun = "camera" if len(process) == 1 else "cameras"
        tail = f"Set {_cameras(k)} and process {noun} {_and_join(process)}."
    return f"{_plural(len(streams), 'stream')} = {_cameras(k)}. {body}. {tail}"


def _stream_candidates(
    shapes: Sequence[Tuple[int, int]], pair_reader: bool
) -> List[Tuple[int, Tuple[Tuple[int, ...], ...]]]:
    """Camera counts with a unique valid split, best first.

    With the pair reader, a split with no A/B camera is dropped (it would select no
    camera). Ranked by most A/B cameras, then fewest cameras.
    """
    found = []
    for k in range(1, len(shapes) + 1):
        splits = valid_stream_splits(shapes, k)
        if len(splits) != 1:
            continue
        n_pairs = sum(len(g) == _PAIR_STREAMS for g in splits[0])
        if pair_reader and n_pairs == 0:
            continue
        found.append((n_pairs, k, splits[0]))
    found.sort(key=lambda item: (-item[0], item[1]))
    return [(k, split) for _, k, split in found]


def rule_set_streams(inp: RuleInput) -> List[Suggestion]:
    """Offer a camera count (and selection) that fits a ``.set`` recording's streams."""
    scan, ctx = inp.scan, inp.ctx
    if not (scan.exists and not scan.is_dir and _is_set_file(scan.path)):
        return []
    try:
        info = read_set_info(scan.path)
    except _UNREADABLE_RECORDING:
        # ``found`` already reports why (apply_diagnosis read the same container
        # for the stream table). Anything else is a reader bug and propagates, so
        # it is logged and named in ``found`` rather than swallowed here.
        return []
    if info.cameras is not None:
        return _labelled_suggestions(inp, info.cameras)
    try:
        split = set_camera_streams(scan.path, ctx.camera_count, info)
    except ValueError:
        return _count_suggestions(inp, info)
    if not ctx.pair_reader:
        return []
    pairs = [
        c
        for c in sorted(set(ctx.camera_numbers))
        if 1 <= c <= len(split) and len(split[c - 1]) == _PAIR_STREAMS
    ]
    has_single_selected = any(
        1 <= c <= len(split) and len(split[c - 1]) != _PAIR_STREAMS
        for c in ctx.camera_numbers
    )
    if not has_single_selected:
        return []
    # The pairing check fails for a selected camera. A count whose split holds more
    # A/B cameras than this one (e.g. 3 for PLIF at count 5, where every camera is a
    # single stream) is offered first. Otherwise the count is right and only the
    # single-stream cameras are deselected.
    n_pairs = sum(len(g) == _PAIR_STREAMS for g in split)
    better = _count_suggestions(inp, info, more_pairs_than=n_pairs)
    if better:
        return better
    if not pairs:
        return []
    return [_streams_suggestion(inp, ctx.camera_count, split, pairs, False)]


def _labelled_suggestions(
    inp: RuleInput, cameras: Sequence[LabelledCamera]
) -> List[Suggestion]:
    """The one suggestion for a recording whose cameras carry labels.

    The count is the labels'. With the pair reader the double-frame cameras are
    selected (a single-frame camera has no A/B pair); time-resolved, every
    camera is. Offered when the configured count differs, or when the pair
    reader would pair a selected single-frame camera. A rig with no
    double-frame camera gets no pair selection: ``SetPairingError`` names why.
    """
    ctx = inp.ctx
    k = len(cameras)
    double = [c.number for c in cameras if c.double_frame]
    process = double if ctx.pair_reader else list(range(1, k + 1))
    if not process:
        return []
    if ctx.camera_count != k:
        change_count = True
    elif ctx.pair_reader and set(ctx.camera_numbers) - set(double):
        change_count = False
    else:
        return []
    noun = "camera" if len(process) == 1 else "cameras"
    message = (
        f"{_cameras(k)} by the recording's own camera labels: "
        + "; ".join(
            f"{c.number} = {c.describe()}, {_streams_text(c.frames)}" for c in cameras
        )
        + f". Set {_cameras(k)} and process {noun} {_and_join(process)}."
    )
    return [_set_streams_suggestion(inp, k, process, change_count, message)]


def _count_suggestions(
    inp: RuleInput, info: Any, more_pairs_than: int = -1
) -> List[Suggestion]:
    """Suggestions for a camera count whose split resolves (or pairs) better.

    ``more_pairs_than`` keeps only candidates with more A/B cameras than that; the
    default keeps every candidate (the configured count does not resolve).
    """
    ctx = inp.ctx
    shapes = [f.shape for f in info.frames]
    candidates = [
        (k, split)
        for k, split in _stream_candidates(shapes, ctx.pair_reader)
        if sum(len(g) == _PAIR_STREAMS for g in split) > more_pairs_than
    ]
    out: List[Suggestion] = []
    for k, split in candidates[:SET_SUGGESTIONS_MAX]:
        if ctx.pair_reader:
            process = [c for c, g in enumerate(split, 1) if len(g) == _PAIR_STREAMS]
        else:
            process = list(range(1, k + 1))
        out.append(_streams_suggestion(inp, k, split, process, True))
    return out


def _streams_suggestion(
    inp: RuleInput,
    camera_count: int,
    split: Sequence[Tuple[int, ...]],
    process: List[int],
    change_count: bool,
) -> Suggestion:
    """Build one ``set_streams`` suggestion for a split by stream shape."""
    streams = set_stream_infos(inp.scan.path, camera_count)
    message = set_streams_message(streams, split, process, inp.ctx.pair_reader)
    return _set_streams_suggestion(inp, camera_count, process, change_count, message)


def _set_streams_suggestion(
    inp: RuleInput,
    camera_count: int,
    process: List[int],
    change_count: bool,
    message: str,
) -> Suggestion:
    """One ``set_streams`` suggestion: its label and ``apply`` around ``message``."""
    ctx = inp.ctx
    apply: Dict[str, Any] = {"camera_numbers": process}
    if change_count:
        apply["camera_count"] = camera_count
        label = f"Use {_cameras(camera_count)}"
        if process != list(range(1, camera_count + 1)):
            label += f", process {_and_join(process)}"
    else:
        noun = "camera" if len(process) == 1 else "cameras"
        label = f"Process {noun} {_and_join(process)}"
    return make_suggestion(
        "set_streams",
        message,
        cross_store_label(ctx, "set_streams", label, apply),
        apply,
    )


# Image family -> image type it selects (inverse of TYPE_FAMILY).
_FAMILY_TYPE: Dict[str, str] = {family: t for t, family in TYPE_FAMILY.items()}
# Families a camera subfolder can hold. Container formats (.set, .cine, .pivraw)
# never use one.
_FOLDER_FAMILIES = ("raster", "im7")
# Entries read from the nearest existing ancestor when looking for a path_case match.
_CASE_SCAN_MAX = SCAN_MAX_ENTRIES * SCAN_HARD_STOP_FACTOR


def _per_camera_container_pattern(
    names: Sequence[str],
    folder: Path,
    camera_count: int,
    camera_numbers: Sequence[int],
    suffix: str,
) -> Optional[str]:
    """Filename pattern for one-file-per-camera containers ending in ``suffix``.

    ``resolve_file_path`` formats a ``%`` pattern with the camera number and reads a
    pattern without ``%`` as one literal file for every camera. So several names that
    differ by a digit run give a ``%d`` pattern, accepted only when the file it names
    exists for every selected camera. One file is suggested only for a single
    camera. Anything else gives None (never a literal for ``camera_count > 1``).
    """
    files = sorted(n for n in names if n.lower().endswith(suffix))
    if len(files) == 1 and camera_count == 1:
        return files[0]
    if len(files) < 2:
        return None
    pattern = _suggest_pattern(files).pattern
    if pattern is None:
        return None
    try:
        paths = [folder / (pattern % camera) for camera in camera_numbers]
    except (TypeError, ValueError):  # a literal ``%`` in the file name
        return None
    return pattern if all(path.exists() for path in paths) else None


def cine_pattern(
    names: Sequence[str],
    folder: Path,
    camera_count: int,
    camera_numbers: Sequence[int],
) -> Optional[str]:
    """Filename pattern for ``.cine`` files, or None when none is safe to suggest."""
    return _per_camera_container_pattern(
        names, folder, camera_count, camera_numbers, ".cine"
    )


def pivraw_pattern(
    names: Sequence[str],
    folder: Path,
    camera_count: int,
    camera_numbers: Sequence[int],
) -> Optional[str]:
    """Filename pattern for ``.pivraw`` files, or None when none is safe to suggest.

    Only names ending in ``.pivraw`` count, so the ``.pivraw.json`` / ``.pivraw.idx``
    sidecars beside each container never shape the pattern.
    """
    return _per_camera_container_pattern(
        names, folder, camera_count, camera_numbers, ".pivraw"
    )


def format_apply(
    names: Sequence[str], image_type: str, folder: Path, ctx: DiagnosisContext
) -> Optional[Dict[str, Any]]:
    """The ``image_format`` (and A/B ``pairing_preset``) fragment for real file names.

    Standard on Setup tries an A/B pair first. Standard on calibration, IM7, CINE and
    PIVRAW use a single pattern. None when no pattern can be derived from ``names``.
    """
    if not names:
        return None
    if image_type == "standard":
        if not ctx.calibration:
            pair = _detect_ab_pair_pattern(list(names))
            if pair is not None:
                fragment: Dict[str, Any] = {
                    "image_format": [pair["pattern_a"], pair["pattern_b"]]
                }
                if ctx.pairing_preset != pair["mode"]:
                    fragment["pairing_preset"] = pair["mode"]
                return fragment
        pattern = _suggest_pattern(list(names)).pattern
    elif image_type == "lavision_im7":
        pattern = _suggest_pattern(list(names), "im7").pattern
    elif image_type == "cine":
        pattern = cine_pattern(names, folder, ctx.camera_count, ctx.camera_numbers)
    elif image_type == "pivraw":
        pattern = pivraw_pattern(names, folder, ctx.camera_count, ctx.camera_numbers)
    else:
        return None
    return None if pattern is None else {"image_format": [pattern]}


def camera_numbers_for(ctx: DiagnosisContext, k: int) -> List[int]:
    """Camera selection for a new count ``k``.

    A full selection (``1..camera_count``) grows or shrinks to ``1..k``. A partial
    selection (e.g. PLIF excluded as ``[2, 3]``) is clipped to ``1..k``, and falls back
    to ``1..k`` when nothing survives the clip.
    """
    selected = sorted(set(ctx.camera_numbers))
    clipped = [c for c in selected if 1 <= c <= k]
    if selected == list(range(1, ctx.camera_count + 1)) or not clipped:
        return list(range(1, k + 1))
    return clipped


def _family_count(counts: Mapping[str, int], family: str) -> int:
    """Files of ``family`` in an extension-count mapping."""
    return sum(counts.get(ext, 0) for ext in FAMILY_EXTS[family])


def _qualifying_folders(scan: SourceScan, family: str) -> List[SubdirInfo]:
    """Subfolders with a camera number that hold ``family``, ordered by camera number."""
    found = [
        d
        for d in scan.subdirs
        if d.camera_number is not None and _family_count(d.ext_counts, family) > 0
    ]
    return sorted(found, key=lambda d: (d.camera_number, natural_key(d.name)))


def _configured_patterns(ctx: DiagnosisContext) -> List[str]:
    """The non-blank configured filename patterns."""
    return [p for p in ctx.image_format if p and p.strip()]


def _root_matches(scan: SourceScan, ctx: DiagnosisContext) -> List[str]:
    """Names at the listed folder that match a configured pattern (a sample if truncated)."""
    globs = [format_to_glob(p) for p in _configured_patterns(ctx)]
    return [n for n in scan.root_names if any(fnmatch.fnmatchcase(n, g) for g in globs)]


def _effective_subfolders(ctx: DiagnosisContext, k: int) -> List[str]:
    """Folder names cameras ``1..k`` resolve to: an explicit entry, else ``Cam{N}``."""
    return [
        (
            ctx.camera_subfolders[i]
            if i < len(ctx.camera_subfolders) and ctx.camera_subfolders[i]
            else f"Cam{i + 1}"
        )
        for i in range(k)
    ]


def _folder_variants(
    scan: SourceScan,
    ctx: DiagnosisContext,
    image_type: str,
    folders: Sequence[SubdirInfo],
) -> List[Tuple[Dict[str, Any], str]]:
    """The ``(apply, label)`` ways to use camera folders, main option first.

    The main ``apply`` holds only fields that differ from ``ctx`` and may be empty. With
    one folder the source moves into it (a count-1 setup ignores subfolders, so
    ``camera_subfolders`` would be pruned). With several, a configured count of 1 also
    gets a ``Use {first}/ only`` alternative.

    ``camera_subfolders`` is indexed by camera number, so the folders' camera numbers
    must be exactly ``1..k`` with no duplicates. Otherwise no mapping is invented and
    the list is empty.
    """
    names = [d.name for d in folders]
    k = len(names)
    if [d.camera_number for d in folders] != list(range(1, k + 1)):
        return []
    if k == 1:
        apply: Dict[str, Any] = {"source_path": str(scan.path / names[0])}
        if ctx.camera_count > 1:
            apply.update(camera_count=1, camera_numbers=[1])
        return [(apply, f"Use {names[0]}/ as the source")]
    main: Dict[str, Any] = {}
    if image_type == "lavision_im7" and not ctx.use_camera_subfolders:
        main["use_camera_subfolders"] = True
    if names != _effective_subfolders(ctx, k):
        main["camera_subfolders"] = names
    if k != ctx.camera_count:
        main["camera_count"] = k
        main["camera_numbers"] = camera_numbers_for(ctx, k)
        label = f"Set camera count to {k}"
    else:
        label = f"Use folders {', '.join(names)}"
    variants = [(main, label)]
    if ctx.camera_count == 1:
        variants.append(
            ({"source_path": str(scan.path / names[0])}, f"Use {names[0]}/ only")
        )
    return variants


def _squash(name: str) -> str:
    """Name with all whitespace removed and case folded, for path_case matching."""
    return re.sub(r"\s+", "", name).casefold()


def rule_path_case(inp: RuleInput) -> List[Suggestion]:
    """A missing path whose first missing component differs from a real one only in case or spacing."""
    scan = inp.scan
    if scan.exists or scan.ancestor is None or not scan.missing_tail:
        return []
    wanted = _squash(scan.missing_tail[0])
    try:
        with os.scandir(scan.ancestor) as it:
            children = [
                e.name
                for _, e in zip(range(_CASE_SCAN_MAX), it)
                if not e.name.startswith(".")
            ]
    except OSError:  # unreadable ancestor: ``found`` already reports it
        return []
    matches = sorted(
        (c for c in children if c != scan.missing_tail[0] and _squash(c) == wanted),
        key=natural_key,
    )
    return [
        make_suggestion(
            "path_case",
            f"{scan.path.as_posix()} does not exist, but "
            f"{(scan.ancestor / child).as_posix()} does and differs only in "
            "letter case or spacing.",
            f'Use "{child}"',
            {"source_path": str(scan.ancestor.joinpath(child, *scan.missing_tail[1:]))},
        )
        for child in matches[:SET_SUGGESTIONS_MAX]
    ]


def rule_picked_file(inp: RuleInput) -> List[Suggestion]:
    """The source is an image file. Offer its folder with a pattern from its sibling files."""
    scan, ctx = inp.scan, inp.ctx
    if not (scan.exists and not scan.is_dir):
        return []
    family = family_of(scan.path.name)
    if family == "set":
        if ctx.image_type == "lavision_set":
            return []
        return [
            make_suggestion(
                "picked_file",
                f"{scan.path.name} is a LaVision recording. Switch the image type "
                "to match.",
                f"Switch to {IMAGE_TYPE_LABELS['lavision_set']}",
                {"image_type": "lavision_set"},
            )
        ]
    if family not in _FOLDER_FAMILIES and family not in ("cine", "pivraw"):
        return []  # ``.ims`` belongs to set_companion; anything else is not an image
    folder = scan.path.parent
    image_type = _FAMILY_TYPE[family]
    names = [n for n in scan.root_names if family_of(n) == family]
    fragment = format_apply(names, image_type, folder, ctx)
    if fragment is None:
        return []
    apply: Dict[str, Any] = {"source_path": str(folder)}
    if image_type != ctx.image_type:
        apply["image_type"] = image_type
    apply.update(fragment)
    folder_name = folder.name or folder.as_posix()
    return [
        make_suggestion(
            "picked_file",
            f"{scan.path.name} is a single file. The source is the folder "
            f"{folder.as_posix()} with a filename pattern.",
            f'Use folder "{folder_name}"',
            apply,
        )
    ]


def _dominant_family(scan: SourceScan) -> Optional[Tuple[str, bool]]:
    """The largest image family as ``(family, only_in_camera_folders)``, else None.

    Root files decide when there are any (``.ims`` ignored). Otherwise camera folders
    are summed per family. A tie for the largest gives None.
    """
    root = {
        f: _family_count(scan.root_ext_counts, f) for f in FAMILY_EXTS if f != "ims"
    }
    folders = {
        f: sum(_family_count(d.ext_counts, f) for d in _qualifying_folders(scan, f))
        for f in _FOLDER_FAMILIES
    }
    for totals, in_folders in ((root, False), (folders, True)):
        present = {f: n for f, n in totals.items() if n > 0}
        if not present:
            continue
        top = max(present.values())
        leaders = [f for f, n in present.items() if n == top]
        return (leaders[0], in_folders) if len(leaders) == 1 else None
    return None


def rule_wrong_type(inp: RuleInput) -> List[Suggestion]:
    """The folder mostly holds a different image family than the configured type."""
    scan, ctx = inp.scan, inp.ctx
    if not (scan.exists and scan.is_dir):
        return []
    dominant = _dominant_family(scan)
    if dominant is None or dominant[0] == TYPE_FAMILY[ctx.image_type]:
        return []
    family, in_folders = dominant
    image_type = _FAMILY_TYPE[family]
    label = f"Switch to {IMAGE_TYPE_LABELS[image_type]}"
    found_in = " in its camera folders" if in_folders else ""
    message = (
        f"{scan.path.as_posix()} mostly holds {_family_exts_text(family)} "
        f"files{found_in}, but the image type is {IMAGE_TYPE_LABELS[ctx.image_type]}."
    )
    if family == "set":
        ordered = _ordered_set_names(scan)
        return [
            make_suggestion(
                "wrong_type",
                message,
                label,
                {
                    "image_type": image_type,
                    "source_path": str(scan.path / ordered[0]),
                },
            )
        ]
    if in_folders:
        folders = _qualifying_folders(scan, family)
        names = [
            n
            for n in _list_dir(scan.path / folders[0].name).image_names
            if family_of(n) == family
        ]
        folder = scan.path / folders[0].name
    else:
        folders = []
        names = [n for n in scan.root_names if family_of(n) == family]
        folder = scan.path
    fragment = format_apply(names, image_type, folder, ctx)
    if fragment is None:
        return []
    base: Dict[str, Any] = {"image_type": image_type, **fragment}
    if not in_folders:
        return [make_suggestion("wrong_type", message, label, base)]
    out: List[Suggestion] = []
    for index, (extra, variant_label) in enumerate(
        _folder_variants(scan, ctx, image_type, folders)
    ):
        apply = {**base, **extra}
        text = (
            label
            if index == 0
            else f"{label}, {variant_label[0].lower()}{variant_label[1:]}"
        )
        out.append(
            make_suggestion(
                "wrong_type",
                message,
                cross_store_label(ctx, "wrong_type", text, apply),
                apply,
            )
        )
    return out


def _folders_format(
    root: Path, folders: Sequence[SubdirInfo], ctx: DiagnosisContext
) -> Dict[str, Any]:
    """The pattern fragment to apply together with camera ``folders`` of ``root``.

    Empty when a configured pattern matches a file of the configured family in any
    of the folders (listing samples, as capped by the scan), or when ``format_apply``
    derives no pattern from the first folder's files. Otherwise the fragment, so one
    click fixes the folders and the pattern together.
    """
    family = TYPE_FAMILY[ctx.image_type]
    globs = [format_to_glob(p) for p in _configured_patterns(ctx)]
    first_names: List[str] = []
    for index, folder in enumerate(folders):
        names = [
            n
            for n in _list_dir(root / folder.name).image_names
            if family_of(n) == family
        ]
        if any(fnmatch.fnmatchcase(n, g) for n in names for g in globs):
            return {}
        if index == 0:
            first_names = names
    fragment = format_apply(first_names, ctx.image_type, root / folders[0].name, ctx)
    return fragment or {}


def _with_format_note(message: str, fragment: Mapping[str, Any]) -> str:
    """``message`` plus the pattern a folder fix also applies, when it applies one."""
    if not fragment:
        return message
    patterns = " / ".join(f'"{p}"' for p in fragment["image_format"])
    return f"{message} Their files match {patterns}."


def rule_camera_folders(inp: RuleInput) -> List[Suggestion]:
    """No file matches at the root, but camera-numbered subfolders hold the configured family."""
    scan, ctx = inp.scan, inp.ctx
    family = TYPE_FAMILY[ctx.image_type]
    if not (scan.exists and scan.is_dir) or family not in _FOLDER_FAMILIES:
        return []
    if _root_matches(scan, ctx):
        return []
    folders = _qualifying_folders(scan, family)
    if not folders:
        return []
    names = [d.name for d in folders]
    message = (
        f"No file matching {', '.join(_configured_patterns(ctx)) or 'the pattern'} "
        f"sits directly in {scan.path.as_posix()}, but "
        f"{_plural(len(names), 'camera folder')} ({_names_text(names)}) hold "
        f"{_family_exts_text(family)} files."
    )
    variants = _folder_variants(scan, ctx, ctx.image_type, folders)
    # Folder listings only when a folder fix will be offered.
    offered = any(apply for apply, _ in variants)
    fragment = _folders_format(scan.path, folders, ctx) if offered else {}
    message = _with_format_note(message, fragment)
    out: List[Suggestion] = []
    for apply, label in variants:
        if apply:  # a fix only for the pattern belongs to the pattern rule
            apply = {**apply, **fragment}
            out.append(
                make_suggestion(
                    "camera_folders",
                    message,
                    cross_store_label(ctx, "camera_folders", label, apply),
                    apply,
                )
            )
    return out


def _family_exts_text(family: str) -> str:
    """``.tif/.tiff/.png`` for a family, as used in messages."""
    return "/".join(FAMILY_EXTS[family])


def rule_inside_camera_folder(inp: RuleInput) -> List[Suggestion]:
    """The source is one camera folder of several. Offer the parent with its camera folders."""
    scan, ctx = inp.scan, inp.ctx
    family = TYPE_FAMILY[ctx.image_type]
    if not (scan.exists and scan.is_dir) or family not in _FOLDER_FAMILIES:
        return []
    if camera_number_of(scan.path.name) is None:
        return []
    parent = scan.path.parent
    parent_scan = _scan_once(parent, inp.scan_cache)
    folders = _qualifying_folders(parent_scan, family)
    if len(folders) < 2 or scan.path.name not in {d.name for d in folders}:
        return []
    variants = _folder_variants(parent_scan, ctx, ctx.image_type, folders)
    if not variants:
        return []
    fragment = _folders_format(parent, folders, ctx)
    apply: Dict[str, Any] = {"source_path": str(parent), **variants[0][0], **fragment}
    parent_name = parent.name or parent.as_posix()
    return [
        make_suggestion(
            "inside_camera_folder",
            _with_format_note(
                f"{scan.path.as_posix()} is one of {len(folders)} camera folders in "
                f"{parent.as_posix()} ({_names_text([d.name for d in folders])}).",
                fragment,
            ),
            cross_store_label(
                ctx,
                "inside_camera_folder",
                f"Use {parent_name}/ with its camera folders",
                apply,
            ),
            apply,
        )
    ]


def rule_flat_with_subfolders(inp: RuleInput) -> List[Suggestion]:
    """Camera subfolders are expected but the matching files sit at the root."""
    scan, ctx = inp.scan, inp.ctx
    if not (scan.exists and scan.is_dir) or inp.camera_path.exists():
        return []
    expects_folders = (
        ctx.image_type == "lavision_im7" and ctx.use_camera_subfolders
    ) or (ctx.image_type == "standard" and ctx.camera_count > 1)
    if not expects_folders:
        return []
    matches = _root_matches(scan, ctx)
    if not matches or _qualifying_folders(scan, TYPE_FAMILY[ctx.image_type]):
        return []
    shown = ", ".join(matches[:FOUND_MAX_NAMES])
    if ctx.image_type == "lavision_im7":
        apply: Dict[str, Any] = {"use_camera_subfolders": False}
        label = "Turn off camera subfolders"
    else:
        apply = {"camera_count": 1, "camera_numbers": [1]}
        label = "Use 1 camera"
    return [
        make_suggestion(
            "flat_with_subfolders",
            f"{inp.camera_path.as_posix()} does not exist, but files matching the "
            f"pattern sit directly in {scan.path.as_posix()} ({shown}). The images "
            "are in one folder, not in camera subfolders.",
            cross_store_label(ctx, "flat_with_subfolders", label, apply),
            apply,
        )
    ]


def _family_names(inp: RuleInput) -> Optional[List[str]]:
    """Files of the configured image family in ``camera_path``, or None if it is not a folder."""
    if not inp.camera_path.is_dir():
        return None
    if inp.camera_path == inp.scan.path:
        names: Sequence[str] = inp.scan.root_names
    else:
        names = _list_dir(inp.camera_path).image_names
    family = TYPE_FAMILY[inp.ctx.image_type]
    return [n for n in names if family_of(n) == family]


def pattern_reason(
    names: Sequence[str], image_type: str, calibration: bool
) -> Optional[str]:
    """Why no filename pattern can be made from ``names``, or None when one can.

    Mirrors ``format_apply``: standard on Setup tries an A/B pair first, so a pair
    means a pattern exists. Container types never take a derived pattern.
    """
    if not names or image_type not in ("standard", "lavision_im7"):
        return None
    if image_type == "standard":
        if not calibration and _detect_ab_pair_pattern(list(names)) is not None:
            return None
        return _suggest_pattern(list(names)).reason
    return _suggest_pattern(list(names), "im7").reason


def _pattern_label(fragment: Mapping[str, Any]) -> str:
    """``Use "{p}"`` or ``Use "{a}" / "{b}"`` for an ``image_format`` fragment."""
    return "Use " + " / ".join(f'"{p}"' for p in fragment["image_format"])


def rule_pattern(inp: RuleInput) -> List[Suggestion]:
    """No file in the camera folder matches the first configured pattern.

    Fires when the folder holds files of the configured family and none matches
    ``format_to_glob(primary)``, or the primary pattern is empty. The suggestion is
    the pattern (or A/B pair, or CINE / PIVRAW ``%d`` pattern) derived from the real
    names.
    """
    ctx = inp.ctx
    if ctx.image_type == "lavision_set":
        return []
    names = _family_names(inp)
    if not names:
        return []
    configured = _configured_patterns(ctx)
    primary = ctx.image_format[0] if ctx.image_format else ""
    if primary and primary.strip():
        glob = format_to_glob(primary)
        if any(fnmatch.fnmatchcase(n, glob) for n in names):
            return []
    fragment = format_apply(names, ctx.image_type, inp.camera_path, ctx)
    if fragment is None or fragment["image_format"] == configured:
        return []
    shown = ", ".join(names[:FOUND_MAX_NAMES])
    wanted = f'"{primary}"' if primary.strip() else "an empty pattern"
    return [
        make_suggestion(
            "pattern",
            f"No file in {inp.camera_path.as_posix()} matches {wanted}. "
            f"The files look like {shown}.",
            _pattern_label(fragment),
            fragment,
        )
    ]


def role_pattern_suggestion(
    names: Sequence[str],
    role: Optional[str],
    patterns: Sequence[str],
    index: int,
) -> Optional[Suggestion]:
    """A ``pattern`` suggestion replacing ``patterns[index]`` for A/B ``role``.

    None when no pattern can be made, or when it equals the configured one.
    """
    pattern = _suggest_pattern_for_role(list(names), role).pattern
    if pattern is None or pattern == patterns[index]:
        return None
    image_format = list(patterns)
    image_format[index] = pattern
    return make_suggestion(
        "pattern",
        f'The files look like "{pattern}".',
        f'Use "{pattern}"',
        {"image_format": image_format},
    )


def start_index_suggestion(found: int, start: int) -> Suggestion:
    """Switch the start index to the files' first index ``found``."""
    return make_suggestion(
        "start_index",
        start_index_hint_message(found, start),
        f"Switch to {found}-based",
        {"start_index": found},
    )


def rule_start_index(inp: RuleInput) -> List[Suggestion]:
    """Files start at 0 or 1 but the start index is the other, leaving a file missing.

    The first index is the core validator's, read from the full matching listing:
    the scan's capped sample may miss the lowest-numbered file.
    """
    ctx = inp.ctx
    primary = ctx.image_format[0] if ctx.image_format else ""
    found = inp.detected_start_index
    if found is None or inp.expected_count is None or not primary.strip():
        return []
    if not start_index_explains_failure(
        inp.camera_path, primary, found, ctx.start_index, inp.expected_count
    ):
        return []
    return [start_index_suggestion(found, ctx.start_index)]


# Spec rank order (section 4.4). Earlier rules explain the failure more directly.
RULES: Tuple[Tuple[str, Rule], ...] = (
    ("path_case", rule_path_case),
    ("picked_file", rule_picked_file),
    ("set_in_folder", rule_set_in_folder),
    ("set_companion", rule_set_companion),
    ("wrong_type", rule_wrong_type),
    ("camera_folders", rule_camera_folders),
    ("inside_camera_folder", rule_inside_camera_folder),
    ("flat_with_subfolders", rule_flat_with_subfolders),
    ("set_streams", rule_set_streams),
    ("pattern", rule_pattern),
    ("start_index", rule_start_index),
)


def diagnose(
    scan: SourceScan,
    ctx: DiagnosisContext,
    camera: int,
    camera_path: Path,
    expected_count: Optional[int],
    valid: bool,
    *,
    detected_start_index: Optional[int],
    scan_cache: Optional[Dict[str, SourceScan]] = None,
) -> List[Suggestion]:
    """Ranked, de-duplicated suggestions for one camera's source.

    A passing validation runs only ``set_streams`` (it also annotates passes). A failing
    one runs every rule in ``RULES`` order and drops later suggestions whose ``apply``
    equals an earlier one. Merging across cameras is the caller's job.
    ``detected_start_index`` is the core result's ``detected_start_index``.
    On calibration, a fix that switches to a type calibration refuses
    (``NON_CALIBRATION_IMAGE_TYPES``) is never offered.
    """
    inp = RuleInput(
        scan, ctx, camera, camera_path, expected_count, detected_start_index, scan_cache
    )
    out: List[Suggestion] = []
    seen: List[Dict[str, Any]] = []
    for rule_id, rule in RULES:
        if valid and rule_id != "set_streams":
            continue
        for suggestion in rule(inp):
            if suggestion["apply"] in seen:
                continue
            if (
                ctx.calibration
                and suggestion["apply"].get("image_type") in NON_CALIBRATION_IMAGE_TYPES
            ):
                continue
            seen.append(suggestion["apply"])
            out.append(suggestion)
    return out


def _exception_text(exc: BaseException) -> str:
    """An exception's reason for a ``found`` sentence (``strerror`` for OS errors)."""
    return getattr(exc, "strerror", None) or str(exc) or type(exc).__name__


def _unfinished(exc: BaseException) -> str:
    """The ``found`` sentence for a diagnosis stopped by ``exc``."""
    return f"Could not finish the diagnosis: {_exception_text(exc)}."


def _targets_set_file(ctx: DiagnosisContext) -> bool:
    """True when the context's source is a ``.set`` recording (its streams are shown)."""
    source = Path(normalise_source_path(str(ctx.source_path)))
    return ctx.image_type == "lavision_set" and _is_set_file(source)


def _scan_once(path: Path, scan_cache: Optional[Dict[str, SourceScan]]) -> SourceScan:
    """``scan_source(path)``, reused from ``scan_cache`` (one request's scans) if given."""
    if scan_cache is None:
        return scan_source(path)
    key = str(path)
    if key not in scan_cache:
        scan_cache[key] = scan_source(path)
    return scan_cache[key]


def apply_diagnosis(
    result: Dict[str, Any],
    ctx: DiagnosisContext,
    camera: int,
    camera_path: Path,
    expected_count: Optional[int],
    *,
    scan_cache: Optional[Dict[str, SourceScan]] = None,
) -> None:
    """Add ``found``, ``suggestions``, ``streams`` and ``warnings`` to a validation result.

    ``warnings`` (always a list) holds what should look like a warning rather
    than a fact: today DaVis's ``SetIsSuspicious`` flag of a ``.set`` whose
    stream table was read (:func:`recording_warnings`); empty otherwise.

    The core ``error`` is never touched. A passing source that is not a ``.set``
    recording is not scanned: ``found`` says what loaded and no rule runs on a pass
    except ``set_streams``, which needs the ``.set``. A ``.set`` whose stream table
    cannot be read keeps ``streams = None`` and ``found`` says why. Any failure while
    scanning, describing or running the rules is logged, leaves ``suggestions``
    empty and is reported in ``found`` as ``Could not finish the diagnosis: ...``.
    ``scan_cache`` shares one scan per source across the cameras of one request.
    """
    if result["valid"] and not _targets_set_file(ctx):
        loaded = loaded_sentence(ctx, result)
        if loaded is not None:
            result.update(found=loaded, suggestions=[], streams=None, warnings=[])
            return
    streams: Optional[List[StreamInfo]] = None
    warnings: List[str] = []
    try:
        scan = _scan_once(ctx.source_path, scan_cache)
        streams_error: Optional[str] = None
        if (
            ctx.image_type == "lavision_set"
            and scan.exists
            and not scan.is_dir
            and _is_set_file(scan.path)
        ):
            try:
                streams = set_stream_infos(scan.path, ctx.camera_count)
            except Exception as exc:  # any unreadable recording: ``found`` says why
                logger.exception("Could not read the streams of %s", scan.path)
                streams_error = str(exc)
        if streams:
            # The stream table was just read from the same cached SetInfo.
            warnings = recording_warnings(read_set_info(scan.path))
        found = found_for(scan, ctx, result, streams, streams_error)
    except Exception as exc:
        logger.exception("Source diagnosis of %s failed", ctx.source_path)
        result.update(
            found=_unfinished(exc), suggestions=[], streams=streams, warnings=[]
        )
        return
    try:
        suggestions = diagnose(
            scan,
            ctx,
            camera,
            camera_path,
            expected_count,
            bool(result["valid"]),
            detected_start_index=result["detected_start_index"],
            scan_cache=scan_cache,
        )
    except Exception as exc:
        logger.exception("Source diagnosis rules failed for %s", ctx.source_path)
        suggestions = []
        found += f" {_unfinished(exc)}"
    result["found"] = found
    result["suggestions"] = suggestions
    result["streams"] = streams
    result["warnings"] = warnings


def piv_diagnosis_context(cfg: "Config", source_path_idx: int) -> DiagnosisContext:
    """The Setup-page diagnosis context for one configured PIV source path."""
    return DiagnosisContext(
        source_path=cfg.source_paths[source_path_idx],
        image_type=cfg.image_type,
        image_format=tuple(cfg.image_format),
        camera_count=cfg.camera_count,
        camera_numbers=tuple(cfg.camera_numbers),
        use_camera_subfolders=cfg.images_use_camera_subfolders,
        camera_subfolders=tuple(cfg.camera_subfolders or ()),
        pairing_preset=cfg.pairing_preset,
        pair_reader=not cfg.time_resolved,
        start_index=cfg.start_index,
        calibration=False,
    )
