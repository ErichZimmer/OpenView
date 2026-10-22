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
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import (
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

from .path_utils import (
    _detect_ab_pair_pattern,
    _suggest_pattern,
    format_to_glob,
    normalise_source_path,
)
from .readers.set_reader import (
    read_set_info,
    set_camera_streams,
    valid_stream_splits,
)

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

# Most ``.set`` suggestions one rule offers (several .set files, or several camera counts).
SET_SUGGESTIONS_MAX = 5
# A fix that edits camera selection in calibration also edits the Setup (and PIV) state.
CROSS_STORE_SUFFIX: Dict[str, str] = {"set_streams": " (also changes Setup and PIV)"}
DEFAULT_CROSS_STORE_SUFFIX = " (also changes Setup)"
_CAMERA_APPLY_KEYS = ("camera_count", "camera_numbers")
# A camera owns one stream (single frame) or two (an A/B pair).
_PAIR_STREAMS = 2


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


@dataclass(frozen=True)
class RuleInput:
    """Everything a diagnosis rule may look at."""

    scan: SourceScan
    ctx: DiagnosisContext
    camera: int
    camera_path: Path
    expected_count: Optional[int]


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
    except Exception:  # any unreadable recording: ``found`` reports why, not this rule
        return []
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
    if not has_single_selected or not pairs:
        return []
    return [_streams_suggestion(inp, ctx.camera_count, split, pairs, False)]


def _count_suggestions(inp: RuleInput, info: Any) -> List[Suggestion]:
    """Suggestions for a camera count whose stream split does not resolve."""
    ctx = inp.ctx
    shapes = [(f.height, f.width) for f in info.frames]
    out: List[Suggestion] = []
    for k, split in _stream_candidates(shapes, ctx.pair_reader)[:SET_SUGGESTIONS_MAX]:
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
    """Build one ``set_streams`` suggestion (label, message and ``apply``)."""
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
    streams = set_stream_infos(inp.scan.path, camera_count)
    return make_suggestion(
        "set_streams",
        set_streams_message(streams, split, process, ctx.pair_reader),
        cross_store_label(ctx, "set_streams", label, apply),
        apply,
    )


# Image family -> image type it selects (inverse of TYPE_FAMILY).
_FAMILY_TYPE: Dict[str, str] = {family: t for t, family in TYPE_FAMILY.items()}
# Families a camera subfolder can hold. Container formats (.set, .cine) never use one.
_FOLDER_FAMILIES = ("raster", "im7")
# Entries read from the nearest existing ancestor when looking for a path_case match.
_CASE_SCAN_MAX = SCAN_MAX_ENTRIES * SCAN_HARD_STOP_FACTOR


def cine_pattern(
    names: Sequence[str],
    folder: Path,
    camera_count: int,
    camera_numbers: Sequence[int],
) -> Optional[str]:
    """Filename pattern for ``.cine`` files, or None when none is safe to suggest.

    ``resolve_file_path`` formats a ``%`` pattern with the camera number and reads a
    pattern without ``%`` as one literal file for every camera. So several names that
    differ by a digit run give a ``%d`` pattern, accepted only when the file it names
    exists for every selected camera. One ``.cine`` is suggested only for a single
    camera. Anything else gives None (never a literal for ``camera_count > 1``).
    """
    cines = sorted(n for n in names if n.lower().endswith(".cine"))
    if len(cines) == 1 and camera_count == 1:
        return cines[0]
    if len(cines) < 2:
        return None
    pattern = _suggest_pattern(cines).pattern
    if pattern is None:
        return None
    try:
        files = [folder / (pattern % camera) for camera in camera_numbers]
    except (TypeError, ValueError):  # a literal ``%`` in the file name
        return None
    return pattern if all(file.exists() for file in files) else None


def format_apply(
    names: Sequence[str], image_type: str, folder: Path, ctx: DiagnosisContext
) -> Optional[Dict[str, Any]]:
    """The ``image_format`` (and A/B ``pairing_preset``) fragment for real file names.

    Standard on Setup tries an A/B pair first. Standard on calibration, IM7 and CINE
    use a single pattern. None when no pattern can be derived from ``names``.
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
    """
    names = [d.name for d in folders]
    k = len(names)
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
    if family not in _FOLDER_FAMILIES and family != "cine":
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
    found_in = "its camera folders" if in_folders else "it"
    message = (
        f"{scan.path.as_posix()} mostly holds {IMAGE_TYPE_LABELS[image_type]} "
        f"files in {found_in}, not {IMAGE_TYPE_LABELS[ctx.image_type]}."
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
    out: List[Suggestion] = []
    for apply, label in _folder_variants(scan, ctx, ctx.image_type, folders):
        if apply:
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
    parent_scan = scan_source(parent)
    folders = _qualifying_folders(parent_scan, family)
    if len(folders) < 2 or scan.path.name not in {d.name for d in folders}:
        return []
    apply: Dict[str, Any] = {
        "source_path": str(parent),
        **_folder_variants(parent_scan, ctx, ctx.image_type, folders)[0][0],
    }
    parent_name = parent.name or parent.as_posix()
    return [
        make_suggestion(
            "inside_camera_folder",
            f"{scan.path.as_posix()} is one of {len(folders)} camera folders in "
            f"{parent.as_posix()} ({_names_text([d.name for d in folders])}).",
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
