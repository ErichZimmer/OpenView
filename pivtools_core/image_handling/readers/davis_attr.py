"""Strict reader for DaVis ``MetaData{N}.attr`` attribute streams, and the
camera identity they carry.

A DaVis 10.2+ stream recording stores per-image attributes beside its image
streams. Which cameras the recording holds comes from three of those keys,
``Acq.Camera.Label``, ``Acq.Camera.ID`` and ``CameraName``, not from the stream
shapes. Every byte the parser accepts is decoded into a typed value or checked
against a fixed expectation; anything else raises :class:`AttrFormatError`
naming the file, the byte offset and what was found.

File layout (decoded 2026-11-16 from DaVis 10.2.1 and 11.4.0.91 recordings and
checked against lvpyio 1.3.1, attr-decode REPORT §2 A)::

    offset 0      header, HEADER_SIZE bytes: u32 version (== 1), then zeros
    offset 1024   entry 0: records ..., END_ENTRY record
                  entry 1: records ..., END_ENTRY record
                  ...
                  trailing block: records ..., END_STREAM record
                  offset table: count x u64 (start of each entry, then of the
                                trailing block)
    EOF - 4       u32 count == number of entries + 1

    record        i32 type | u32 key_len | u32 value_len | key | value

Record types:

    -1  END_ENTRY   key_len == value_len == 0; closes one image entry
     0  END_STREAM  key_len == value_len == 0; closes the trailing block
     1  STRING      value is UTF-8 text, no terminator
     2  UINT16      value is little-endian uint16[value_len / 2]
     3  INT32       value is little-endian int32[value_len / 4]
     4  FLOAT32     value is little-endian float32[value_len / 4]
     5  FLOAT64     value is little-endian float64[value_len / 8]
     6  EMPTY       key_len == value_len == 0; carries nothing (lvpyio drops it)

Types 1, 3, 4, 5, 6 occur in real recordings. Type 2 was seen through lvpyio on
a synthetic file only (it returns uint16 (1, N)) and is accepted on that
evidence (Morgan, 17 Nov). lvpyio raises "Attribute of type N is unknown" for
N >= 7. lvpyio also skips a marker record that carries a value; no real file has
one, so it is refused.

Entry ``i`` belongs to image ``i`` of the recording. The trailing block belongs
to no image: lvpyio reports ``count - 1`` images.

Which frame a file describes is not in the file and not in ``N``: StreamSet.xml
binds each AttributeReader to one frame or to the whole image buffer
(:class:`AttrBinding`, built by ``set_reader``). Several files may bind to one
frame (DaVis 11 splits camera, timing and trigger keys into separate files); a
frame's attributes are their union, and for a key in two of them the file
declared later in StreamSet.xml wins (lvpyio's rule, attr-decode REPORT A5).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------

# Header block before the first record. Word 0 is a version; the other 1020
# bytes were zero in every file surveyed (47 files, DaVis 10.2.1 and 11.4.0.91).
# They are required to be zero so a future layout that uses them fails loudly.
HEADER_SIZE = 1024
HEADER_VERSION = 1

_RECORD_HEADER = struct.Struct("<iII")  # type, key_len, value_len
_TABLE_COUNT = struct.Struct("<I")  # trailing u32 entry count
_OFFSET_WIDTH = 8  # each offset-table slot is a u64

TYPE_END_ENTRY = -1
TYPE_END_STREAM = 0
TYPE_STRING = 1
TYPE_UINT16 = 2
TYPE_INT32 = 3
TYPE_FLOAT32 = 4
TYPE_FLOAT64 = 5
TYPE_EMPTY = 6

# Array record type -> little-endian numpy dtype. lvpyio returns these as
# (1, N) arrays of the same dtype.
_ARRAY_DTYPES: Dict[int, np.dtype] = {
    TYPE_UINT16: np.dtype("<u2"),
    TYPE_INT32: np.dtype("<i4"),
    TYPE_FLOAT32: np.dtype("<f4"),
    TYPE_FLOAT64: np.dtype("<f8"),
}
_MARKER_TYPES = (TYPE_END_ENTRY, TYPE_END_STREAM, TYPE_EMPTY)
_MARKER_NAMES = {TYPE_END_ENTRY: "END_ENTRY", TYPE_END_STREAM: "END_STREAM"}

# Keys are printable ASCII in every file surveyed. lvpyio turns a non-ASCII key
# byte into '?' without a message; this reader refuses it rather than decode it
# under a guessed code page.
_KEY_BYTES = re.compile(rb"[\x20-\x7e]+")

AttrValue = Union[str, np.ndarray]


class AttrFormatError(ValueError):
    """A ``.attr`` file holds bytes, or a binding, this reader does not understand."""


class CameraIdentityError(ValueError):
    """The recording's camera keys do not give one consistent camera layout."""


# ---------------------------------------------------------------------------
# Parsed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrRecord:
    """One record. ``value`` is None only for ``TYPE_EMPTY``."""

    type_code: int
    key: str
    value: Optional[AttrValue]


@dataclass(frozen=True)
class AttrEntry:
    """The records of one image entry, in file order (EMPTY records included)."""

    records: Tuple[AttrRecord, ...]

    def as_dict(self) -> Dict[str, AttrValue]:
        """Key -> value, EMPTY records dropped (as lvpyio does)."""
        return {r.key: r.value for r in self.records if r.type_code != TYPE_EMPTY}


@dataclass(frozen=True)
class AttrStream:
    """A parsed ``.attr`` file: one entry per image, and the trailing block."""

    path: str
    entries: Tuple[AttrEntry, ...]
    trailing: AttrEntry  # belongs to no image

    @property
    def n_entries(self) -> int:
        """Number of images the file describes."""
        return len(self.entries)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _fail(source: str, offset: int, what: str) -> AttrFormatError:
    return AttrFormatError(f"{source}: byte {offset}: {what}")


def _decode_value(
    source: str, offset: int, type_code: int, key: str, raw: bytes
) -> AttrValue:
    """The typed value of one record, or the refusal naming why not."""
    if type_code == TYPE_STRING:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fail(
                source, offset, f"string value of '{key}' is not UTF-8 ({exc.reason})"
            ) from exc
    dtype = _ARRAY_DTYPES.get(type_code)
    if dtype is not None:
        if len(raw) == 0 or len(raw) % dtype.itemsize:
            raise _fail(
                source,
                offset,
                f"'{key}' is type {type_code} ({dtype.name}) but its value is "
                f"{len(raw)} bytes, not a positive multiple of {dtype.itemsize}",
            )
        return np.frombuffer(raw, dtype=dtype).reshape(1, -1).copy()
    raise _fail(
        source,
        offset,
        f"record type {type_code} for key '{key}' is not one of the known types "
        f"(string 1, uint16 2, int32 3, float32 4, float64 5, empty 6). The value's "
        f"layout is unknown, so it is refused rather than guessed",
    )


def _parse_block(
    data: bytes, pos: int, limit: int, source: str, closer: int
) -> Tuple[AttrEntry, int]:
    """Records from ``pos`` up to and including the ``closer`` marker."""
    records: List[AttrRecord] = []
    seen: Dict[str, int] = {}
    while True:
        if pos + _RECORD_HEADER.size > limit:
            raise _fail(
                source,
                pos,
                f"block runs into the offset table at byte {limit} without its "
                f"{_MARKER_NAMES[closer]} marker",
            )
        type_code, key_len, value_len = _RECORD_HEADER.unpack_from(data, pos)
        if type_code in _MARKER_TYPES:
            if key_len or value_len:
                raise _fail(
                    source,
                    pos,
                    f"marker record type {type_code} has key_len {key_len} and "
                    f"value_len {value_len}; every marker seen has both 0",
                )
            if type_code == closer:
                return AttrEntry(tuple(records)), pos + _RECORD_HEADER.size
            if type_code != TYPE_EMPTY:
                raise _fail(
                    source,
                    pos,
                    f"marker {_MARKER_NAMES[type_code]} where a record or the "
                    f"{_MARKER_NAMES[closer]} marker was expected",
                )
            records.append(AttrRecord(TYPE_EMPTY, "", None))
            pos += _RECORD_HEADER.size
            continue
        key_start = pos + _RECORD_HEADER.size
        end = key_start + key_len + value_len
        if key_len == 0 or end > limit:
            raise _fail(
                source,
                pos,
                f"record type {type_code} with key_len {key_len}, value_len "
                f"{value_len} does not fit before byte {limit}",
            )
        key_raw = data[key_start : key_start + key_len]
        if not _KEY_BYTES.fullmatch(key_raw):
            raise _fail(source, pos, f"key {key_raw!r} is not printable ASCII")
        key = key_raw.decode("ascii")
        if key in seen:
            raise _fail(
                source,
                pos,
                f"key '{key}' appears twice in one entry (first at byte "
                f"{seen[key]}); which one DaVis means is unknown",
            )
        seen[key] = pos
        value = _decode_value(
            source, pos, type_code, key, data[key_start + key_len : end]
        )
        records.append(AttrRecord(type_code, key, value))
        pos = end


def parse_attr_bytes(data: bytes, source: str = "<bytes>") -> AttrStream:
    """Parse the bytes of one ``.attr`` file.

    Args:
        data: whole file contents.
        source: name used in error messages.

    Returns:
        The file's entries (one per image) and its trailing block.

    Raises:
        AttrFormatError: on any byte that does not fit the layout in the module
            docstring (header, record types, lengths, markers, offset table),
            naming the byte offset.
    """
    min_size = HEADER_SIZE + _RECORD_HEADER.size + _OFFSET_WIDTH + _TABLE_COUNT.size
    if len(data) < min_size:
        raise _fail(source, 0, f"file is {len(data)} bytes, shorter than {min_size}")
    (version,) = struct.unpack_from("<I", data, 0)
    if version != HEADER_VERSION:
        raise _fail(source, 0, f"header version {version}, expected {HEADER_VERSION}")
    nonzero = next((i for i in range(4, HEADER_SIZE) if data[i]), None)
    if nonzero is not None:
        raise _fail(
            source, nonzero, "header bytes 4..1023 are not all zero; layout unknown"
        )

    (count,) = _TABLE_COUNT.unpack_from(data, len(data) - _TABLE_COUNT.size)
    table_pos = len(data) - _TABLE_COUNT.size - _OFFSET_WIDTH * count
    if count < 1 or table_pos < HEADER_SIZE + _RECORD_HEADER.size:
        raise _fail(
            source,
            len(data) - _TABLE_COUNT.size,
            f"offset-table count {count} does not fit a {len(data)}-byte file",
        )
    offsets = struct.unpack_from(f"<{count}Q", data, table_pos)

    entries: List[AttrEntry] = []
    pos = HEADER_SIZE
    for i in range(count - 1):
        if offsets[i] != pos:
            raise _fail(
                source,
                pos,
                f"offset table says entry {i} starts at {offsets[i]}, records "
                f"say {pos}",
            )
        entry, pos = _parse_block(data, pos, table_pos, source, TYPE_END_ENTRY)
        entries.append(entry)
    if offsets[count - 1] != pos:
        raise _fail(
            source,
            pos,
            f"offset table says the trailing block starts at {offsets[count - 1]}, "
            f"records say {pos}",
        )
    trailing, pos = _parse_block(data, pos, table_pos, source, TYPE_END_STREAM)
    if pos != table_pos:
        raise _fail(
            source,
            pos,
            f"{table_pos - pos} unexplained bytes between the trailing block and "
            f"the offset table",
        )
    return AttrStream(path=source, entries=tuple(entries), trailing=trailing)


# ---------------------------------------------------------------------------
# Encoding (the inverse; the tests use it to prove every byte is accounted for)
# ---------------------------------------------------------------------------


def _encode_block(entry: AttrEntry, closer: int) -> bytes:
    out = bytearray()
    for r in entry.records:
        if r.type_code == TYPE_EMPTY:
            out += _RECORD_HEADER.pack(TYPE_EMPTY, 0, 0)
            continue
        key = r.key.encode("ascii")
        if r.type_code == TYPE_STRING:
            if not isinstance(r.value, str):
                raise TypeError(f"string record '{r.key}' holds {type(r.value)}")
            val = r.value.encode("utf-8")
        else:
            if not isinstance(r.value, np.ndarray):
                raise TypeError(f"array record '{r.key}' holds {type(r.value)}")
            val = r.value.astype(_ARRAY_DTYPES[r.type_code], copy=False).tobytes()
        out += _RECORD_HEADER.pack(r.type_code, len(key), len(val)) + key + val
    out += _RECORD_HEADER.pack(closer, 0, 0)
    return bytes(out)


def encode_attr_stream(stream: AttrStream) -> bytes:
    """Bytes that :func:`parse_attr_bytes` turns back into ``stream``."""
    out = bytearray(struct.pack("<I", HEADER_VERSION) + bytes(HEADER_SIZE - 4))
    offsets = []
    for entry in stream.entries:
        offsets.append(len(out))
        out += _encode_block(entry, TYPE_END_ENTRY)
    offsets.append(len(out))
    out += _encode_block(stream.trailing, TYPE_END_STREAM)
    out += struct.pack(f"<{len(offsets)}Q", *offsets)
    out += _TABLE_COUNT.pack(len(offsets))
    return bytes(out)


# ---------------------------------------------------------------------------
# A recording's attribute files, bound as StreamSet.xml says
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrBinding:
    """One AttributeReader of StreamSet.xml: its file and what it describes.

    ``frame`` is the frame-stream index the file is bound to, or None for a file
    bound to the whole image buffer. Bindings are kept in declaration order,
    which decides which file wins a key both carry.
    """

    prefix: str
    frame: Optional[int]


@dataclass(frozen=True)
class RecordingAttributes:
    """Every ``.attr`` file of one recording, bound as the manifest says.

    ``frame_files[k]`` lists the files bound to frame stream ``k`` (lvpyio's
    ``buffer.frames[k]``) in declaration order; ``buffer_files`` those bound to
    the whole buffer.
    """

    set_dir: Path
    n_frames: int
    frame_files: Dict[int, Tuple[AttrStream, ...]]
    buffer_files: Tuple[AttrStream, ...]
    n_entries: int

    def frame_attributes(self, entry: int, frame: int) -> Dict[str, AttrValue]:
        """Union of every file bound to ``frame`` at image ``entry`` (0-based)."""
        return _merge(self.frame_files.get(frame, ()), entry)

    def buffer_attributes(self, entry: int) -> Dict[str, AttrValue]:
        """Union of every file bound to the whole buffer at image ``entry``."""
        return _merge(self.buffer_files, entry)


def _merge(streams: Sequence[AttrStream], entry: int) -> Dict[str, AttrValue]:
    """Union of one entry across files in declaration order; a later file's
    value replaces an earlier one's (lvpyio's rule, measured on synthetic
    recordings in both declaration orders)."""
    merged: Dict[str, AttrValue] = {}
    for s in streams:
        merged.update(s.entries[entry].as_dict())
    return merged


def read_recording_attributes(
    set_dir: Path, bindings: Sequence[AttrBinding], n_frames: int, n_images: int
) -> Optional[RecordingAttributes]:
    """Parse every ``.attr`` file the manifest declares, bound to its frame.

    Args:
        set_dir: the recording's companion folder.
        bindings: the AttributeReaders of StreamSet.xml, in declaration order.
        n_frames: number of frame streams of the recording.
        n_images: images per frame stream, from the ``.ims`` indexes.

    Returns:
        None when ``bindings`` is empty (the recording has no attribute files).

    Raises:
        FileNotFoundError: if a declared ``.attr`` file is missing, naming it.
            lvpyio returns empty attributes then, without a message.
        AttrFormatError: on a malformed file, a binding to a frame stream the
            recording does not have, or a file whose image count differs from
            the image streams'. lvpyio is silent on both.
    """
    if not bindings:
        return None
    frame_files: Dict[int, List[AttrStream]] = {}
    buffer_files: List[AttrStream] = []
    for binding in bindings:
        path = set_dir / f"{binding.prefix}.attr"
        where = f"StreamSet.xml in {set_dir} declares the attribute file {path.name}"
        if not path.is_file():
            raise FileNotFoundError(
                f"{where}, which is missing. Copy the complete companion folder: "
                f"the camera labels of the recording are in it."
            )
        stream = parse_attr_bytes(path.read_bytes(), str(path))
        if stream.n_entries != n_images:
            raise AttrFormatError(
                f"{where}, which describes {stream.n_entries} images, but the "
                f"image streams hold {n_images}. The file does not belong to "
                f"these images, or one of them is incomplete."
            )
        if binding.frame is None:
            buffer_files.append(stream)
            continue
        if not 0 <= binding.frame < n_frames:
            raise AttrFormatError(
                f"{where} for frame stream {binding.frame}, but the recording has "
                f"frame streams 0..{n_frames - 1}."
            )
        frame_files.setdefault(binding.frame, []).append(stream)
    return RecordingAttributes(
        set_dir=set_dir,
        n_frames=n_frames,
        frame_files={k: tuple(v) for k, v in sorted(frame_files.items())},
        buffer_files=tuple(buffer_files),
        n_entries=n_images,
    )


# ---------------------------------------------------------------------------
# Camera identity
# ---------------------------------------------------------------------------

KEY_LABEL = "Acq.Camera.Label"
KEY_ID = "Acq.Camera.ID"
KEY_NAME = "CameraName"
CAMERA_KEYS = (KEY_LABEL, KEY_ID, KEY_NAME)

# CameraName is "<camera number><role>: <model>", role "A"/"B" for the two
# frames of a double-frame camera and empty for a single-frame camera. Seen:
# "1A: Imager MX 25M", "3: Imager sCMOS CLHS", "1: Imager CX-25".
_CAMERA_NAME = re.compile(r"(?P<number>[1-9][0-9]*)(?P<role>[AB]?): (?P<model>.+)")
_DOUBLE_FRAME_ROLES = ("A", "B")


@dataclass(frozen=True)
class FrameIdentity:
    """What the camera keys say about one frame stream."""

    frame: int
    label: str  # Acq.Camera.Label, e.g. "Camera 1"
    camera_id: str  # Acq.Camera.ID, e.g. "Camera.ImagerMX25M: 290187"
    number: int  # from CameraName
    role: str  # "A", "B" or "" (single frame)
    model: str  # from CameraName, e.g. "Imager MX 25M"
    labelled_images: int  # how many images carry the three keys


@dataclass(frozen=True)
class LabelledCamera:
    """One physical camera of a recording, as its own camera keys name it.

    ``frames`` are its frame-stream indices, A then B for a double-frame camera.
    ``labelled_images`` counts the images that carry its camera keys: DaVis
    11.4 writes them on a leading run of images only (264 of 500 on one real
    recording), so it can be below the recording's image count.
    """

    number: int
    label: str
    camera_id: str
    model: str
    frames: Tuple[int, ...]
    labelled_images: int

    @property
    def double_frame(self) -> bool:
        """True for an A/B camera (two frames per image)."""
        return len(self.frames) == len(_DOUBLE_FRAME_ROLES)

    def describe(self) -> str:
        """``Camera 3 (Imager sCMOS CLHS, Camera.ImagerSCMOS5MCLHS: 61008639,
        single frame)``: what the user sees in DaVis, word for word."""
        kind = "double frame (A/B)" if self.double_frame else "single frame"
        return f"{self.label} ({self.model}, {self.camera_id}, {kind})"


def frame_identities(rec: RecordingAttributes) -> List[Optional[FrameIdentity]]:
    """The camera keys of each frame stream, checked across every image.

    An image that carries none of the three keys is skipped. An image that
    carries some but not all raises, and so does one whose keys differ from an
    earlier image's, naming both images.

    Returns:
        One entry per frame stream; None for a stream no image labels.

    Raises:
        CameraIdentityError: as above, or a CameraName that is not
            ``<number>[A|B]: <model>``.
    """
    name = rec.set_dir.name
    out: List[Optional[FrameIdentity]] = []
    for frame in range(rec.n_frames):
        triple: Optional[Tuple[str, str, str]] = None
        first = -1
        carried = 0
        for e in range(rec.n_entries):
            attrs = rec.frame_attributes(e, frame)
            present = [k for k in CAMERA_KEYS if k in attrs]
            if not present:
                continue
            if len(present) != len(CAMERA_KEYS):
                missing = [k for k in CAMERA_KEYS if k not in attrs]
                raise CameraIdentityError(
                    f"{name}: image {e + 1}, frame stream {frame} carries "
                    f"{', '.join(present)} but not {', '.join(missing)}. A camera "
                    f"is named by all three keys, so this image's camera is unknown."
                )
            values = tuple(attrs[k] for k in CAMERA_KEYS)
            if not all(isinstance(v, str) for v in values):
                raise CameraIdentityError(
                    f"{name}: image {e + 1}, frame stream {frame}: the camera keys "
                    f"are not all text: {values}"
                )
            carried += 1
            if triple is None:
                triple, first = values, e
            elif values != triple:
                raise CameraIdentityError(
                    f"{name}: frame stream {frame} is labelled {triple} on image "
                    f"{first + 1} but {values} on image {e + 1}. The camera of one "
                    f"frame stream cannot change within a recording, so which "
                    f"camera these images come from is unknown."
                )
        if triple is None:
            out.append(None)
            continue
        label, camera_id, camera_name = triple
        m = _CAMERA_NAME.fullmatch(camera_name)
        if m is None:
            raise CameraIdentityError(
                f"{name}: frame stream {frame} has CameraName {camera_name!r}, not "
                f"'<number>[A|B]: <model>' as DaVis writes it."
            )
        out.append(
            FrameIdentity(
                frame=frame,
                label=label,
                camera_id=camera_id,
                number=int(m["number"]),
                role=m["role"],
                model=m["model"],
                labelled_images=carried,
            )
        )
    return out


def group_cameras(rec: RecordingAttributes) -> Optional[Tuple[LabelledCamera, ...]]:
    """The recording's physical cameras, from its camera keys.

    Rules, each failure raising :class:`CameraIdentityError` in plain words:

    - when any frame stream is labelled, every frame stream is;
    - frame streams sharing (Acq.Camera.Label, Acq.Camera.ID) are one camera;
    - a one-stream camera has role ""; a two-stream camera has roles "A" then
      "B" on adjacent streams;
    - CameraName's number and model agree within a camera;
    - numbered by first frame stream, the cameras' CameraName numbers are
      1..n in that order (true on every real recording, attr-decode REPORT C1).

    Returns:
        Cameras in order of their first frame stream. None when no image of the
        recording carries camera keys (decided 17 Nov: treated as a recording
        without ``.attr``).
    """
    name = rec.set_dir.name
    ids = frame_identities(rec)
    if all(fi is None for fi in ids):
        return None
    unlabelled = [k for k, fi in enumerate(ids) if fi is None]
    if unlabelled:
        labelled = [k for k, fi in enumerate(ids) if fi is not None]
        raise CameraIdentityError(
            f"{name}: frame streams {labelled} carry camera labels, but frame "
            f"streams {unlabelled} carry none on any image. Which camera those "
            f"streams belong to is unknown."
        )
    groups: Dict[Tuple[str, str], List[FrameIdentity]] = {}
    for fi in ids:
        groups.setdefault((fi.label, fi.camera_id), []).append(fi)

    cameras: List[LabelledCamera] = []
    for (label, camera_id), members in groups.items():
        roles = tuple(m.role for m in members)
        frames = tuple(m.frame for m in members)
        what = f"{name}: camera '{label}' ({camera_id}) on frame streams {frames}"
        if roles not in (("",), _DOUBLE_FRAME_ROLES):
            raise CameraIdentityError(
                f"{what} has frame roles {list(roles)}. A camera is single frame "
                f"(no role) or double frame (A then B)."
            )
        if len(frames) == 2 and frames[1] != frames[0] + 1:
            raise CameraIdentityError(
                f"{what}: its A and B frames are not on adjacent frame streams."
            )
        if (
            len({m.number for m in members}) != 1
            or len({m.model for m in members}) != 1
        ):
            raise CameraIdentityError(
                f"{what}: its CameraName number or model differs between frames."
            )
        cameras.append(
            LabelledCamera(
                number=members[0].number,
                label=label,
                camera_id=camera_id,
                model=members[0].model,
                frames=frames,
                labelled_images=min(m.labelled_images for m in members),
            )
        )
    cameras.sort(key=lambda c: c.frames[0])
    numbers = [c.number for c in cameras]
    if numbers != list(range(1, len(cameras) + 1)):
        raise CameraIdentityError(
            f"{name}: in frame-stream order the cameras carry CameraName numbers "
            f"{numbers}, not 1..{len(cameras)}. Camera numbers would then differ "
            f"between DaVis and PIVTOOLs. Cameras: "
            f"{'; '.join(c.describe() for c in cameras)}."
        )
    return tuple(cameras)
