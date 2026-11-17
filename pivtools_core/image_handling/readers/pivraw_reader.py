"""Reader for PIVTOOLS Acquisition run containers (``.pivraw``), format version 2.

One run folder per recording, laid out like Phantom ``.cine`` (one container per
camera, the ``%d`` in the pattern names the camera)::

    <run_dir>/Camera1.pivraw         frames back to back, frame k at k * frame_stride
    <run_dir>/Camera1.pivraw.json    header sidecar
    <run_dir>/Camera1.pivraw.idx     one "<qQq" record (host_ns, frame_id, camera_ns) per frame
    <run_dir>/Camera2.pivraw ...

The format is defined by the acquisition app (``pivtools-acquisition``
``docs/CONTRACTS.md`` section 4). PIVTOOLs does not import that package; the
container is read here from its specification alone, and the tests read the
recorder's own output (``unit-tests/fixtures/pivraw_contract``).

Layout facts the reader relies on:

* ``frame_stride = ceil(bytes_per_frame / 4096) * 4096``; the bytes between the
  payload and the stride are zero padding. A finalised file therefore holds
  exactly ``n_frames * frame_stride`` bytes, and its ``.idx`` exactly
  ``n_frames * 24``.
* Run index = ``frame_id - first_frame_id``. In a double-frame run, run index
  2t is frame A of the camera's run trigger t and 2t + 1 its B. The file's run
  indices and ``missing_indices`` are disjoint and together fill
  ``[0, span)``, the span rounded up to a whole pair for double-frame.
* Mono12p is the PFNC LSB-first packing (two pixels in three bytes), the layout
  ``set_reader._decode_mono12p`` already decodes, so that decoder is reused.
* BayerRG8 becomes a 2x2 superpixel sum R+G+G+B at half resolution, with no
  demosaicing (decision 2026-10-26). The sum of four 8-bit photosites is at most
  1020, exact in float32.
* A double-frame run holds whole pairs A, B, A, B, ... so a PIV pair is frames
  ``(2k-1, 2k)`` (1-based): the ``skip_frames`` pairing preset (frame_stride 1,
  pair_stride 2). A time-resolved run pairs any two frames.

The reader is strict and forward-only: a missing or unknown sidecar field, a
field of the wrong type, a ``format_version`` other than the integer 2, a
``slot`` that differs from the file name, an unfinalised run, or a sidecar and
``.idx`` that disagree with each other or with the files on disk raise
:class:`PivrawError` (a ``ValueError``) naming the file and the field. There is
no version-1 path. Frame indices are 1-based, like every other container in
PIVTOOLs. Returned pixels are raw counts as float32: (H, W) for a frame,
(2, H, W) for a pair.
"""

import json
import logging
import math
import numbers
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from .out_buffer import check_out
from .set_reader import _decode_mono12p

PathLike = Union[str, Path]

FORMAT_NAME = "pivraw"
FORMAT_VERSION = 2
SIDECAR_SUFFIX = ".json"  # Camera1.pivraw -> Camera1.pivraw.json
INDEX_SUFFIX = ".idx"  # Camera1.pivraw -> Camera1.pivraw.idx
# One record per frame, in file order, little-endian and packed ("<qQq", 24 B):
# host_ns = host perf_counter_ns at delivery (one clock for every camera of a run),
# frame_id = the camera's transport image id (not a trigger count),
# camera_ns = the camera's own timestamp (its own clock, arbitrary origin).
INDEX_DTYPE = np.dtype([("host_ns", "<i8"), ("frame_id", "<u8"), ("camera_ns", "<i8")])
INDEX_RECORD_BYTES = INDEX_DTYPE.itemsize
# Frames start on 4096-byte boundaries so the recorder can write with direct I/O.
STRIDE_ALIGNMENT = 4096
# The recorder names camera k's files Camera<k>.pivraw; the sidecar's slot must be k.
_DATA_NAME = re.compile(r"^Camera(\d+)\.pivraw$")
# How many offending run indices an error message lists.
_LIST_IN_MESSAGE = 8

MONO12P = "Mono12p"
BAYER_RG8 = "BayerRG8"
PIXEL_FORMATS = (MONO12P, BAYER_RG8)

DOUBLE_FRAME = "double_frame"
TIME_RESOLVED = "time_resolved"
MODES = (DOUBLE_FRAME, TIME_RESOLVED)

# Every sidecar field, in the writer's order (container.Header); no other is accepted.
REQUIRED_FIELDS = (
    "camera_model",
    "camera_serial",
    "slot",
    "width",
    "height",
    "pixel_format",
    "bytes_per_frame",
    "frame_stride",
    "mode",
    "exposure_us",
    "exposure_effective_us",
    "max_rate_hz",
    "rate_hz",
    "dt_s",
    "camera_settings",
    "created_at",
    "n_frames",
    "finalised",
    "dropped_pairs",
    "missing_indices",
    "first_frame_id",
    "format",
    "format_version",
)


class PivrawError(ValueError):
    """A ``.pivraw`` container that cannot be read as specified."""


@dataclass(frozen=True)
class PivrawHeader:
    """The validated sidecar and ``.idx`` of one camera's ``.pivraw`` container.

    ``index`` is the read-only ``.idx`` as an ``INDEX_DTYPE`` array, one record
    per file frame in file order.
    """

    path: Path
    camera_model: str
    camera_serial: str
    slot: int
    width: int
    height: int
    pixel_format: str
    bytes_per_frame: int
    frame_stride: int
    mode: str
    exposure_us: float
    exposure_effective_us: float
    max_rate_hz: float
    rate_hz: float
    dt_s: float
    camera_settings: Dict[str, Any]
    created_at: str
    n_frames: int
    dropped_pairs: int
    missing_indices: Tuple[int, ...]
    first_frame_id: Optional[int]
    index: np.ndarray = field(compare=False, repr=False)

    @property
    def image_shape(self) -> Tuple[int, int]:
        """(H, W) of a decoded frame: full sensor for Mono12p, halved for BayerRG8."""
        if self.pixel_format == BAYER_RG8:
            return (self.height // 2, self.width // 2)
        return (self.height, self.width)


# {str(path): (stamp, header)}; the stamp covers all three files of the camera.
_header_cache: Dict[str, Tuple[Tuple[int, ...], PivrawHeader]] = {}


def sidecar_path(file_path: PathLike) -> Path:
    """``Camera1.pivraw`` -> ``Camera1.pivraw.json``."""
    p = Path(file_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def index_path(file_path: PathLike) -> Path:
    """``Camera1.pivraw`` -> ``Camera1.pivraw.idx``."""
    p = Path(file_path)
    return p.with_name(p.name + INDEX_SUFFIX)


def expected_bytes_per_frame(width: int, height: int, pixel_format: str) -> int:
    """Payload bytes of one ``width`` x ``height`` frame in ``pixel_format``.

    Raises:
        PivrawError: For an unknown pixel format, or a geometry the format cannot
            hold (Mono12p packs pixels in twos; the Bayer superpixel needs even
            width and height).
    """
    if pixel_format == MONO12P:
        if (width * height) % 2:
            raise PivrawError(
                f"Mono12p packs two pixels into three bytes; {width}x{height} "
                "is an odd pixel count."
            )
        return width * height * 3 // 2
    if pixel_format == BAYER_RG8:
        if width % 2 or height % 2:
            raise PivrawError(
                "BayerRG8 is read as 2x2 superpixels, which needs even width and "
                f"height; got {width}x{height}."
            )
        return width * height
    raise PivrawError(
        f"unknown pixel_format {pixel_format!r}; expected one of {PIXEL_FORMATS}."
    )


def _stride_for(bytes_per_frame: int) -> int:
    """``ceil(bytes_per_frame / 4096) * 4096``."""
    return -(-bytes_per_frame // STRIDE_ALIGNMENT) * STRIDE_ALIGNMENT


def _is_int(value: Any) -> bool:
    """A JSON integer: ``int`` and not ``bool`` (``2.0`` and ``True`` are not)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _int_field(sidecar: Mapping[str, Any], key: str, where: str, minimum: int) -> int:
    """A required integer field >= ``minimum`` (bool is not an integer here)."""
    value = sidecar[key]
    if not _is_int(value) or value < minimum:
        raise PivrawError(
            f"{where}: {key} must be an integer >= {minimum}, got {value!r}."
        )
    return value


def _number_field(sidecar: Mapping[str, Any], key: str, where: str) -> float:
    """A required finite real number."""
    value = sidecar[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise PivrawError(f"{where}: {key} must be a finite number, got {value!r}.")
    return float(value)


def _str_field(sidecar: Mapping[str, Any], key: str, where: str) -> str:
    """A required string field."""
    value = sidecar[key]
    if not isinstance(value, str):
        raise PivrawError(f"{where}: {key} must be a string, got {value!r}.")
    return value


def _listed(values: np.ndarray) -> str:
    """The first few of ``values`` for an error message."""
    shown = ", ".join(str(int(v)) for v in values[:_LIST_IN_MESSAGE])
    more = len(values) - _LIST_IN_MESSAGE
    return f"[{shown}{f', ... {more} more' if more > 0 else ''}]"


def _check_index(
    index: np.ndarray,
    first_frame_id: Optional[int],
    missing_indices: Tuple[int, ...],
    mode: str,
    idx_file: Path,
    where: str,
) -> None:
    """The ``.idx`` agrees with itself and with the sidecar, as the writer defines them.

    Run index = ``frame_id - first_frame_id``. The run indices increase strictly
    from >= 0; host times never go backwards; double-frame file positions
    (2k, 2k+1) carry run indices (2t, 2t+1); and the run indices and
    ``missing_indices`` together fill ``[0, span)`` exactly once each. The span
    ends at the last index the writer received, which can be a frame A it
    discarded because its B never came, so it is checked as a partition rather
    than recomputed from the ``.idx``.

    Raises:
        PivrawError: Naming the field or file that disagrees: a stale, edited or
            foreign ``.idx`` or sidecar.
    """
    if index.size and first_frame_id is None:
        raise PivrawError(
            f"{where}: first_frame_id is null, but the run holds {index.size} "
            "frames: their run indices cannot be known."
        )
    run_idx = (
        index["frame_id"].astype(np.int64) - first_frame_id
        if index.size
        else np.zeros(0, dtype=np.int64)
    )
    if run_idx.size and (run_idx[0] < 0 or np.any(np.diff(run_idx) <= 0)):
        raise PivrawError(
            f"{idx_file}: frame ids are not strictly increasing from first_frame_id "
            f"{first_frame_id} (an id wrap, a camera reset, or a foreign .idx)."
        )
    backwards = np.flatnonzero(np.diff(index["host_ns"]) < 0)
    if backwards.size:
        raise PivrawError(
            f"{idx_file}: host_ns goes backwards at record(s) {_listed(backwards + 1)} "
            "(0-based): not one recording's index."
        )
    if mode == DOUBLE_FRAME:
        a, b = run_idx[0::2], run_idx[1::2]
        if np.any(a % 2) or np.any(b != a + 1):
            raise PivrawError(
                f"{idx_file}: file positions (2k, 2k+1) are not frames A, B of one "
                "trigger."
            )
    missing = np.asarray(missing_indices, dtype=np.int64)
    if not (run_idx.size or missing.size):
        return
    top = int(max(run_idx.max(initial=-1), missing.max(initial=-1)))
    span = (top // 2 + 1) * 2 if mode == DOUBLE_FRAME else top + 1
    covered = np.bincount(np.concatenate([run_idx, missing]), minlength=span)
    if np.any(covered != 1):
        raise PivrawError(
            f"{where}: missing_indices and the .idx do not partition run indices "
            f"[0, {span}): neither holds {_listed(np.flatnonzero(covered == 0))}, "
            f"both hold {_listed(np.flatnonzero(covered > 1))}. The sidecar or "
            ".idx is stale, edited or from another recording."
        )


def _parse_sidecar(raw_path: Path, sidecar: Any) -> PivrawHeader:
    """Validate a decoded sidecar against the format, the ``.idx`` and the files.

    Raises:
        PivrawError: On any missing or unknown field, a field of the wrong type,
            a wrong format/version, a slot that is not the file name's, an
            unfinalised run, inconsistent geometry, a data/index file whose size
            disagrees with ``n_frames``, or an ``.idx`` that disagrees with the
            sidecar (see :func:`_check_index`).
    """
    side = sidecar_path(raw_path)
    where = str(side)
    if not isinstance(sidecar, dict):
        raise PivrawError(
            f"{where}: expected a JSON object, got {type(sidecar).__name__}."
        )
    missing = [k for k in REQUIRED_FIELDS if k not in sidecar]
    if missing:
        raise PivrawError(
            f"{where}: missing required field(s) {', '.join(missing)} (a stale "
            "file: re-record, or re-export with the current pivtools-acquisition)."
        )
    unknown = sorted(set(sidecar) - set(REQUIRED_FIELDS))
    if unknown:
        raise PivrawError(f"{where}: unknown field(s) {', '.join(unknown)}.")
    if sidecar["format"] != FORMAT_NAME:
        raise PivrawError(
            f"{where}: format is {sidecar['format']!r}, expected {FORMAT_NAME!r}."
        )
    version = sidecar["format_version"]
    if not _is_int(version) or version != FORMAT_VERSION:
        raise PivrawError(
            f"{where}: format_version {version!r} is not supported; this PIVTOOLs "
            f"reads format_version {FORMAT_VERSION} (the integer) only."
        )

    finalised = sidecar["finalised"]
    if not isinstance(finalised, bool):
        raise PivrawError(
            f"{where}: finalised must be true or false, got {finalised!r}."
        )
    if not finalised or sidecar["n_frames"] is None:
        raise PivrawError(
            f"{where}: run not finalised (recorder stopped early?): run "
            f"`pivtools-acq recover {raw_path.parent}`"
        )

    slot = _int_field(sidecar, "slot", where, 1)
    name = _DATA_NAME.match(raw_path.name)
    if name is None or int(name.group(1)) != slot:
        raise PivrawError(
            f"{where}: sidecar slot {slot} does not match the file name "
            f"{raw_path.name}; the recorder names camera {slot}'s file "
            f"Camera{slot}.pivraw."
        )
    width = _int_field(sidecar, "width", where, 1)
    height = _int_field(sidecar, "height", where, 1)
    pixel_format = sidecar["pixel_format"]
    if pixel_format not in PIXEL_FORMATS:
        raise PivrawError(
            f"{where}: pixel_format {pixel_format!r} is not one of {PIXEL_FORMATS}."
        )
    try:
        bpf_expected = expected_bytes_per_frame(width, height, pixel_format)
    except PivrawError as exc:
        raise PivrawError(f"{where}: {exc}") from exc
    bytes_per_frame = _int_field(sidecar, "bytes_per_frame", where, 1)
    if bytes_per_frame != bpf_expected:
        raise PivrawError(
            f"{where}: bytes_per_frame {bytes_per_frame} does not match "
            f"{width}x{height} {pixel_format} ({bpf_expected} bytes)."
        )
    frame_stride = _int_field(sidecar, "frame_stride", where, 1)
    if frame_stride != _stride_for(bytes_per_frame):
        raise PivrawError(
            f"{where}: frame_stride {frame_stride} is not bytes_per_frame rounded up "
            f"to {STRIDE_ALIGNMENT} ({_stride_for(bytes_per_frame)})."
        )
    mode = sidecar["mode"]
    if mode not in MODES:
        raise PivrawError(f"{where}: mode {mode!r} is not one of {MODES}.")
    n_frames = _int_field(sidecar, "n_frames", where, 0)
    if mode == DOUBLE_FRAME and n_frames % 2:
        raise PivrawError(
            f"{where}: a double-frame run holds whole A/B pairs, but n_frames is "
            f"{n_frames} (odd)."
        )

    dropped_pairs = _int_field(sidecar, "dropped_pairs", where, 0)
    missing_indices = sidecar["missing_indices"]
    if not isinstance(missing_indices, list) or any(
        not _is_int(m) or m < 0 for m in missing_indices
    ):
        raise PivrawError(
            f"{where}: missing_indices must be a list of non-negative integers, "
            f"got {missing_indices!r}."
        )
    first_frame_id = sidecar["first_frame_id"]
    if first_frame_id is not None and (
        not _is_int(first_frame_id) or first_frame_id < 0
    ):
        raise PivrawError(
            f"{where}: first_frame_id must be a non-negative integer or null, got "
            f"{first_frame_id!r}."
        )
    camera_settings = sidecar["camera_settings"]
    if not isinstance(camera_settings, dict):
        raise PivrawError(
            f"{where}: camera_settings must be an object, got {camera_settings!r}."
        )

    raw_size = raw_path.stat().st_size
    if raw_size != n_frames * frame_stride:
        raise PivrawError(
            f"{raw_path}: holds {raw_size} bytes, but the sidecar declares "
            f"{n_frames} frames x {frame_stride} bytes = {n_frames * frame_stride}. "
            "The data file and its sidecar disagree."
        )
    idx_file = index_path(raw_path)
    if not idx_file.is_file():
        raise PivrawError(f"{raw_path}: per-frame index {idx_file} is missing.")
    idx_size = idx_file.stat().st_size
    if idx_size != n_frames * INDEX_RECORD_BYTES:
        raise PivrawError(
            f"{idx_file}: holds {idx_size} bytes, expected {n_frames} records x "
            f"{INDEX_RECORD_BYTES} bytes = {n_frames * INDEX_RECORD_BYTES} "
            "(format_version 2 records are host_ns, frame_id, camera_ns)."
        )
    index = np.fromfile(idx_file, dtype=INDEX_DTYPE)
    index.setflags(write=False)
    _check_index(index, first_frame_id, tuple(missing_indices), mode, idx_file, where)

    header = PivrawHeader(
        path=raw_path,
        camera_model=_str_field(sidecar, "camera_model", where),
        camera_serial=_str_field(sidecar, "camera_serial", where),
        slot=slot,
        width=width,
        height=height,
        pixel_format=pixel_format,
        bytes_per_frame=bytes_per_frame,
        frame_stride=frame_stride,
        mode=mode,
        exposure_us=_number_field(sidecar, "exposure_us", where),
        exposure_effective_us=_number_field(sidecar, "exposure_effective_us", where),
        max_rate_hz=_number_field(sidecar, "max_rate_hz", where),
        rate_hz=_number_field(sidecar, "rate_hz", where),
        dt_s=_number_field(sidecar, "dt_s", where),
        camera_settings=dict(camera_settings),
        created_at=_str_field(sidecar, "created_at", where),
        n_frames=n_frames,
        dropped_pairs=dropped_pairs,
        missing_indices=tuple(missing_indices),
        first_frame_id=first_frame_id,
        index=index,
    )
    if header.dropped_pairs or header.missing_indices:
        logging.warning(
            "%s: the recorder reports %d dropped pair(s) and %d missing frame "
            "index(es) %s. Frames are read by their position in the file.",
            raw_path,
            header.dropped_pairs,
            len(header.missing_indices),
            list(header.missing_indices[:10]),
        )
    return header


def _stamp(raw_path: Path) -> Tuple[int, ...]:
    """mtime and size of the data, sidecar and index files (missing index -> -1).

    The sidecar is rewritten at stop and by ``pivtools-acq recover``, which also
    truncates the data and index files, so all three take part.
    """
    raw = raw_path.stat()
    side = sidecar_path(raw_path).stat()
    try:
        idx = index_path(raw_path).stat()
        idx_stamp: Tuple[int, int] = (idx.st_mtime_ns, idx.st_size)
    except FileNotFoundError:
        idx_stamp = (-1, -1)
    return (raw.st_mtime_ns, raw.st_size, side.st_mtime_ns, side.st_size) + idx_stamp


def read_pivraw_header(file_path: PathLike) -> PivrawHeader:
    """The validated header of one camera's ``.pivraw``, cached by file stamps.

    Raises:
        FileNotFoundError: The ``.pivraw`` or its ``.json`` sidecar is missing.
        PivrawError: The sidecar or files do not satisfy the format.
    """
    raw_path = Path(file_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"PIVRAW file not found: {raw_path}")
    side = sidecar_path(raw_path)
    if not side.is_file():
        raise FileNotFoundError(f"PIVRAW sidecar not found: {side} (for {raw_path})")

    stamp = _stamp(raw_path)
    key = str(raw_path)
    cached = _header_cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        sidecar = json.loads(side.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PivrawError(f"{side}: not valid JSON ({exc}).") from exc
    header = _parse_sidecar(raw_path, sidecar)
    _header_cache[key] = (stamp, header)
    return header


def clear_header_cache() -> None:
    """Drop every cached header. Mirrors ``cine_reader.clear_metadata_cache``."""
    _header_cache.clear()


def get_pivraw_frame_count(file_path: PathLike) -> int:
    """Frames in the container (a double-frame run counts A and B separately)."""
    return read_pivraw_header(file_path).n_frames


def get_pivraw_image_shape(file_path: PathLike) -> Tuple[int, int]:
    """(H, W) of a decoded frame, from the sidecar alone (no pixel decode)."""
    return read_pivraw_header(file_path).image_shape


def _check_frame_idx(header: PivrawHeader, idx: int, role: str) -> None:
    """Raise unless ``idx`` is a 1-based frame number inside the container."""
    if isinstance(idx, bool) or not isinstance(idx, numbers.Integral):
        raise PivrawError(
            f"{header.path}: {role} index must be an integer, got {idx!r}."
        )
    if not 1 <= idx <= header.n_frames:
        raise PivrawError(
            f"{header.path}: {role} {idx} is out of range; the container holds "
            f"frames 1 to {header.n_frames} (1-based)."
        )


def _decode_into(header: PivrawHeader, idx: int, dest: np.ndarray) -> None:
    """Decode 1-based frame ``idx`` of ``header`` straight into float32 ``dest``."""
    offset = (idx - 1) * header.frame_stride
    raw = np.fromfile(
        header.path, dtype=np.uint8, count=header.bytes_per_frame, offset=offset
    )
    if raw.size != header.bytes_per_frame:
        raise PivrawError(
            f"{header.path}: frame {idx} is truncated ({raw.size} of "
            f"{header.bytes_per_frame} bytes at offset {offset})."
        )
    if header.pixel_format == MONO12P:
        np.copyto(dest, _decode_mono12p(raw, header.width, header.height))
    else:
        mosaic = raw.reshape(header.height // 2, 2, header.width // 2, 2)
        mosaic.sum(axis=(1, 3), dtype=np.float32, out=dest)


def read_pivraw_single(
    file_path: PathLike, idx: int, out: Optional[np.ndarray] = None
) -> np.ndarray:
    """Read 1-based frame ``idx`` as float32 raw counts, shape (H, W).

    Args:
        file_path: The camera's ``.pivraw`` file.
        idx: Frame number, 1-based.
        out: Optional (H, W) float32 C-contiguous destination, written in place
            and returned. Undefined after an exception.
    """
    header = read_pivraw_header(file_path)
    _check_frame_idx(header, idx, "frame")
    if out is None:
        out = np.empty(header.image_shape, dtype=np.float32)
    else:
        check_out(
            out,
            header.image_shape,
            f"read_pivraw_single frame {idx} of {header.path.name}",
        )
    _decode_into(header, idx, out)
    return out


def read_pivraw_pair(
    file_path: PathLike,
    idx: int = 1,
    out: Optional[np.ndarray] = None,
    frame_b_idx: Optional[int] = None,
) -> np.ndarray:
    """Read a frame pair as float32 raw counts, shape (2, H, W).

    Args:
        file_path: The camera's ``.pivraw`` file.
        idx: Frame A number, 1-based (from ``Config.get_frame_pair_indices``).
        out: Optional (2, H, W) float32 C-contiguous destination, written in
            place and returned. Undefined after an exception.
        frame_b_idx: Frame B number, 1-based. None means ``idx + 1``.

    Raises:
        PivrawError: A frame is out of range, or a double-frame run is asked for
            anything but one of its recorded A/B pairs ``(2k-1, 2k)``. Pairing a B
            frame with the next A frame (time-resolved preset), or a frame with
            itself (A/B-format preset), would correlate the wrong exposures
            without any other error, so it is refused by name.
    """
    header = read_pivraw_header(file_path)
    frame_b = idx + 1 if frame_b_idx is None else frame_b_idx
    _check_frame_idx(header, idx, "frame A")
    _check_frame_idx(header, frame_b, "frame B")
    if header.mode == DOUBLE_FRAME and (idx % 2 != 1 or frame_b != idx + 1):
        raise PivrawError(
            f"{header.path}: a double-frame run holds A/B pairs as frames "
            f"(2k-1, 2k), but frames ({idx}, {frame_b}) were requested. Use the "
            "skip_frames pairing preset (frame_stride 1, pair_stride 2) with "
            "start_index 1."
        )
    shape = (2,) + header.image_shape
    if out is None:
        out = np.empty(shape, dtype=np.float32)
    else:
        check_out(out, shape, f"read_pivraw_pair frame {idx} of {header.path.name}")
    _decode_into(header, idx, out[0])
    _decode_into(header, frame_b, out[1])
    return out
