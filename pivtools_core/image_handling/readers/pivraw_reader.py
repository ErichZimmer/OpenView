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
* A double-frame run holds whole pairs A, B, A, B, ... so a recorded pair is
  frames ``(2k-1, 2k)`` (1-based): the ``skip_frames`` pairing preset
  (frame_stride 1, pair_stride 2). A time-resolved run pairs any two frames.
* PIV pairs are matched across cameras by trigger, not by file position
  (:func:`align_pivraw_run`, decision 02-1): with unequal drops each camera
  holds pair n at its own file frames, incomplete pairs are skipped and
  counted, and a run whose cameras cannot be shown to count from the same
  trigger is refused by name.

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
import yaml

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


def _frames_per_trigger(mode: str) -> int:
    """Frames one trigger fires: A and B for double-frame, one for time-resolved."""
    return 2 if mode == DOUBLE_FRAME else 1


def _run_span(run_idx: np.ndarray, missing: np.ndarray, mode: str) -> int:
    """Run indices the writer accounts for, ``[0, span)``: file frames plus ``missing_indices``.

    The span ends at the largest index of either, rounded up to a whole pair for
    double-frame (``container.py`` of the writer). 0 when the camera accounts
    for no frame.
    """
    top = int(max(run_idx.max(initial=-1), missing.max(initial=-1)))
    if top < 0:
        return 0
    return (top // 2 + 1) * 2 if mode == DOUBLE_FRAME else top + 1


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
    span = _run_span(run_idx, missing, mode)
    if span == 0:
        return
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
    # The writer derives dropped_pairs from missing_indices: whole missing pairs
    # for double-frame, always 0 for time-resolved.
    dropped_expected = len(missing_indices) // 2 if mode == DOUBLE_FRAME else 0
    if dropped_pairs != dropped_expected:
        raise PivrawError(
            f"{where}: dropped_pairs {dropped_pairs} disagrees with "
            f"missing_indices ({len(missing_indices)} run indices), which give "
            f"{dropped_expected} for a {mode} run. The sidecar is stale or edited."
        )

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


# ---------------------------------------------------------------------------
# Trigger alignment (decision 02-1)
# ---------------------------------------------------------------------------
#
# Frames are paired across cameras by trigger, never by position in the file:
# with unequal drops, file pair k of one camera is another trigger than file
# pair k of the next. The rules and constants are the decided alignment,
# prototyped in the pivraw-contract research
# (``02-pivraw-reader/followup/pivraw-contract/scripts/pivraw_align.py``) and
# mirrored by the acquisition app's TIFF export (``recorder/export.py``
# ``align_run``). Both are held to the outcome of every case in
# ``pivraw_contract/expected.json``, which keeps the two aligners in agreement.

# The recorder's run description beside the camera files (CONTRACTS section 4).
RUN_FILE = "run.yaml"
# run.yaml ``trigger``: what fired the cameras.
SOFTWARE_TRIGGER = "software"
RUN_TRIGGERS = (SOFTWARE_TRIGGER, "daq")
# Fraction of a trigger period two host times may differ by and still be the
# same trigger. The rig measured its cameras within 0.74 ms of each other at a
# 200 ms period (pivraw-contract REPORT C8); a quarter period keeps a wide
# margin and stays below the half period at which a one-trigger offset becomes
# ambiguous. Prototype ``pivraw_align.TOL_FRAC``, acquisition
# ``export.TRIGGER_TOL_FRAC``.
TRIGGER_TOL_FRAC = 0.25
# Double-frame A/B phase on the camera clock: B->next A no longer than
# (1 - margin) of A->B is a flip, A->B no longer than (1 - margin) of B->next A
# is proven, and in between the phase is not provable (the CX-25 near its
# 9.6 pairs/s maximum reads B out 52 ms after A and the next A 59 ms after B,
# REPORT C9). Prototype ``pivraw_align.PHASE_MARGIN``, acquisition
# ``export.AB_PHASE_MARGIN``.
AB_PHASE_MARGIN = 0.2
# Why a run cannot be aligned. The same names as the acquisition app's
# ``export.REASONS`` and the fixtures' ``must_raise:<reason>``.
ALIGNMENT_REASONS = (
    "inconsistent_files",
    "cameras_disagree",
    "no_frames",
    "no_complete_pair",
    "camera_offset",
    "tr_interval",
    "ab_phase",
)
# The one pairing of a double-frame run, in recorded frames: A and B of each
# trigger (the skip_frames preset).
DOUBLE_FRAME_STRIDES = (1, 2)
_RERECORD = "Re-record the run."


class PivrawAlignmentError(PivrawError):
    """A run whose cameras cannot be paired by trigger.

    ``reason`` is one of :data:`ALIGNMENT_REASONS`; the message names the run,
    the camera and, where there is one, the trigger.
    """

    def __init__(self, reason: str, message: str) -> None:
        if reason not in ALIGNMENT_REASONS:
            raise ValueError(f"unknown alignment reason {reason!r}")
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class AlignedPair:
    """One PIV pair complete on every camera.

    ``run_triggers`` are the run triggers of its two frames (equal for
    double-frame); ``file_frames`` maps each camera slot to its 1-based file
    frames (A, B).
    """

    run_triggers: Tuple[int, int]
    file_frames: Mapping[int, Tuple[int, int]]


@dataclass(frozen=True)
class PivrawAlignment:
    """Every camera of one run, paired by trigger.

    Attributes:
        run_dir: The run folder.
        mode: ``double_frame`` or ``time_resolved``.
        period_s: Trigger period, ``1 / rate_hz``.
        cameras: Camera slots of the run, ascending.
        pairs: The complete pairs in trigger order; pair number n is
            ``pairs[n - 1]``.
        skipped_incomplete: Pairs that should exist but lack a frame on at
            least one camera.
        skipped_by_camera: Per slot, the skipped pairs that camera lacks a
            frame of (a pair can count against several cameras).
        triggers_requested: run.yaml ``config.record.n``.
        triggers_unaccounted: Triggers requested that no camera accounts for:
            lost before the first or after the last delivered frame, which the
            files cannot tell apart.
        trigger_source: run.yaml ``trigger``.
        notes: Plain sentences for validation: phases that could not be proven
            and unaccounted triggers.
    """

    run_dir: Path
    mode: str
    period_s: float
    cameras: Tuple[int, ...]
    pairs: Tuple[AlignedPair, ...]
    skipped_incomplete: int
    skipped_by_camera: Mapping[int, int]
    triggers_requested: int
    triggers_unaccounted: int
    trigger_source: str
    notes: Tuple[str, ...]

    def file_frames(self, pair_number: int, camera: int) -> Tuple[int, int]:
        """1-based file frames (A, B) of 1-based ``pair_number`` on ``camera``.

        Raises:
            PivrawError: The pair number is outside ``1..len(pairs)`` or the
                camera is not in the run.
        """
        if not 1 <= pair_number <= len(self.pairs):
            raise PivrawError(
                f"{self.run_dir}: pair {pair_number} is out of range; the run "
                f"aligns {len(self.pairs)} pair(s) on every camera (1-based)."
            )
        if camera not in self.cameras:
            raise PivrawError(
                f"{self.run_dir}: camera {camera} is not in the run; its cameras "
                f"are {list(self.cameras)}."
            )
        return self.pairs[pair_number - 1].file_frames[camera]

    @property
    def summary(self) -> str:
        """One sentence: pairs aligned, and pairs skipped with the cameras lacking them."""
        text = (
            f"{len(self.pairs)} pair(s) aligned by trigger on every camera; "
            f"{self.skipped_incomplete} skipped as incomplete"
        )
        lacking = [
            f"Camera{s} lacks a frame of {n}"
            for s, n in self.skipped_by_camera.items()
            if n
        ]
        return text + (f" ({', '.join(lacking)})." if lacking else ".")

    def report(self) -> Dict[str, Any]:
        """JSON-safe summary for validation results (camera keys as strings)."""
        return {
            "pairs": len(self.pairs),
            "skipped_incomplete": self.skipped_incomplete,
            "skipped_by_camera": {str(s): n for s, n in self.skipped_by_camera.items()},
            "triggers_requested": self.triggers_requested,
            "triggers_unaccounted": self.triggers_unaccounted,
            "summary": self.summary,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _CameraTriggers:
    """One camera's frames by run trigger (internal to the alignment)."""

    slot: int
    span: int  # run indices the writer accounts for: [0, span)
    position: Dict[int, int]  # run trigger -> 0-based file position of its first frame
    host_ns: np.ndarray  # per file frame
    camera_ns: np.ndarray  # per file frame


# {(run_dir, frame_stride, pair_stride): (stamp, alignment)}
_alignment_cache: Dict[
    Tuple[str, int, int], Tuple[Tuple[Any, ...], PivrawAlignment]
] = {}


def clear_alignment_cache() -> None:
    """Drop every cached alignment."""
    _alignment_cache.clear()


def _run_cameras(run_dir: Path) -> Tuple[int, ...]:
    """Slots of the ``Camera<k>.pivraw`` files in ``run_dir``, ascending."""
    try:
        names = [p.name for p in run_dir.iterdir()]
    except OSError as exc:
        raise PivrawError(f"{run_dir}: cannot list the run folder ({exc}).") from exc
    slots = sorted(int(m.group(1)) for n in names if (m := _DATA_NAME.match(n)))
    if not slots:
        raise PivrawError(f"{run_dir}: no Camera<k>.pivraw file; not a run folder.")
    return tuple(slots)


def _read_run_file(run_dir: Path) -> Tuple[int, str]:
    """run.yaml's ``config.record.n`` (triggers requested) and ``trigger``.

    Raises:
        PivrawError: run.yaml is missing or unreadable, or either key is absent
            or of the wrong kind.
    """
    path = run_dir / RUN_FILE
    try:
        run = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PivrawError(
            f"{path}: cannot read the run description ({exc}). Every recorded "
            "run holds one; the folder is incomplete."
        ) from exc
    try:
        requested = run["config"]["record"]["n"]
        trigger = run["trigger"]
    except (KeyError, TypeError) as exc:
        raise PivrawError(
            f"{path}: config.record.n or trigger is missing ({exc!r}); not a "
            "run.yaml the recorder wrote."
        ) from exc
    if not _is_int(requested) or requested < 1:
        raise PivrawError(
            f"{path}: config.record.n must be a positive integer, got {requested!r}."
        )
    if trigger not in RUN_TRIGGERS:
        raise PivrawError(
            f"{path}: trigger must be one of {RUN_TRIGGERS}, got {trigger!r}."
        )
    return requested, trigger


def _camera_triggers(header: PivrawHeader) -> _CameraTriggers:
    """Run trigger -> file position of one camera, from its validated header."""
    per = _frames_per_trigger(header.mode)
    missing = np.asarray(header.missing_indices, dtype=np.int64)
    if header.n_frames == 0:
        empty = np.zeros(0, dtype=np.int64)
        return _CameraTriggers(
            header.slot, _run_span(empty, missing, header.mode), {}, empty, empty
        )
    run_idx = header.index["frame_id"].astype(np.int64) - header.first_frame_id
    position = {int(run_idx[k]) // per: k for k in range(0, header.n_frames, per)}
    return _CameraTriggers(
        header.slot,
        _run_span(run_idx, missing, header.mode),
        position,
        header.index["host_ns"].astype(np.int64),
        header.index["camera_ns"].astype(np.int64),
    )


def _ab_phase_note(run_dir: Path, cam: _CameraTriggers) -> Optional[str]:
    """None when the camera's A/B phase is proven, a note when it cannot be.

    Judged on the camera's own clock (``camera_ns``): its origin is arbitrary
    and differs between cameras by hours, but within one camera it times
    exposures without host delivery jitter. A ratio of two intervals is used,
    so the clock's unit does not matter.

    Raises:
        PivrawAlignmentError: ``ab_phase``, the phase is flipped: every file
            pair is B of one trigger and A of the next.
    """
    a_to_b = [cam.camera_ns[k + 1] - cam.camera_ns[k] for k in cam.position.values()]
    b_to_next_a = [
        cam.camera_ns[cam.position[t + 1]] - cam.camera_ns[k + 1]
        for t, k in cam.position.items()
        if t + 1 in cam.position
    ]
    name = f"Camera{cam.slot}"
    if not b_to_next_a:
        return (
            f"{name}: A/B phase not provable (fewer than two consecutive "
            "pairs); pairs are read as recorded."
        )
    median_ab, median_ba = float(np.median(a_to_b)), float(np.median(b_to_next_a))
    if median_ba <= (1 - AB_PHASE_MARGIN) * median_ab:
        raise PivrawAlignmentError(
            "ab_phase",
            f"{run_dir}: {name}: A->B {median_ab / 1e6:.2f} ms is longer than "
            f"B->next A {median_ba / 1e6:.2f} ms on the camera clock: each file "
            "pair is B of one trigger and A of the next (the run's first frame "
            f"was lost). {_RERECORD}",
        )
    if median_ab <= (1 - AB_PHASE_MARGIN) * median_ba:
        return None
    return (
        f"{name}: A/B phase not provable (A->B {median_ab / 1e6:.2f} ms and "
        f"B->next A {median_ba / 1e6:.2f} ms on the camera clock are within "
        f"{AB_PHASE_MARGIN:.0%} of each other); pairs are read as recorded."
    )


def _check_tr_spacing(
    run_dir: Path,
    cam: _CameraTriggers,
    pairs: Tuple[AlignedPair, ...],
    period_ns: float,
    trigger_source: str,
) -> None:
    """Time-resolved: the two frames of each pair are their triggers apart in host time.

    Host time, as in the prototype and the acquisition export: it is one
    nanosecond clock for every camera, while the unit of a camera's own clock
    is not yet confirmed on the rig (lab visit L1).

    Raises:
        PivrawAlignmentError: ``tr_interval``, naming the camera, the triggers
            and, for a software-triggered run, the DAQ.
    """
    name = f"Camera{cam.slot}"
    for pair in pairs:
        first, second = (f - 1 for f in pair.file_frames[cam.slot])
        dt = cam.host_ns[second] - cam.host_ns[first]
        expected = (pair.run_triggers[1] - pair.run_triggers[0]) * period_ns
        if abs(dt - expected) <= TRIGGER_TOL_FRAC * period_ns:
            continue
        if trigger_source == SOFTWARE_TRIGGER:
            why = (
                "The run was software-triggered (run.yaml trigger: software), "
                "and a late software tick spaces frames unevenly. Software "
                "triggering is for setup, focus and calibration: record "
                "measurements with the DAQ trigger."
            )
        else:
            why = f"The camera ignored a trigger (its ids stay contiguous). {_RERECORD}"
        raise PivrawAlignmentError(
            "tr_interval",
            f"{run_dir}: {name} run triggers {pair.run_triggers}: frames "
            f"{dt / 1e6:.1f} ms apart, expected {expected / 1e6:.1f} ms. {why}",
        )


def _check_pairing(
    run_dir: Path, mode: str, frame_stride: int, pair_stride: int
) -> None:
    """The pairing strides (in recorded frames) suit the run's mode.

    Raises:
        PivrawError: A double-frame run paired other than A with B of each
            trigger, or a time-resolved stride below 1.
    """
    if mode == DOUBLE_FRAME:
        if (frame_stride, pair_stride) != DOUBLE_FRAME_STRIDES:
            raise PivrawError(
                f"{run_dir}: a double-frame run pairs A with B of each trigger: "
                "use the skip_frames pairing preset (frame_stride 1, "
                f"pair_stride 2), not frame_stride {frame_stride}, pair_stride "
                f"{pair_stride}."
            )
        return
    if frame_stride < 1 or pair_stride < 1:
        raise PivrawError(
            f"{run_dir}: a time-resolved run pairs two different frames: "
            f"frame_stride and pair_stride must be >= 1, got {frame_stride} and "
            f"{pair_stride}."
        )


def _run_stamp(run_dir: Path, slots: Tuple[int, ...]) -> Tuple[Any, ...]:
    """Stamps of the files the alignment reads: every camera's three and run.yaml."""
    try:
        run = (run_dir / RUN_FILE).stat()
        run_stamp: Tuple[int, int] = (run.st_mtime_ns, run.st_size)
    except FileNotFoundError:
        run_stamp = (-1, -1)
    return (slots, run_stamp) + tuple(
        _stamp(run_dir / f"Camera{s}.pivraw") for s in slots
    )


def align_pivraw_run(
    run_dir: PathLike, frame_stride: int, pair_stride: int
) -> PivrawAlignment:
    """Pair every camera of a run by trigger, cached by the files' stamps.

    Steps (the decided alignment, section comment above):

    1. Each camera's header is read strictly (:func:`read_pivraw_header`): run
       index = ``frame_id - first_frame_id``, and run trigger = run index
       (time-resolved) or run index // 2 (double-frame).
    2. Every trigger up to the last one any camera accounts for is a pair that
       should exist: double-frame A_t with B_t, time-resolved t with
       t + ``frame_stride``, stepped by ``pair_stride``. A pair is complete when
       every camera holds all its frames; the others are skipped and counted.
    3. The ids alone cannot see a camera that lost its first frame(s) or
       ignored a trigger, so: (a) each complete pair's first frame arrives at
       the same host time on every camera within :data:`TRIGGER_TOL_FRAC` of
       the period (host time is the one clock all cameras share); (b)
       time-resolved, the two frames of a pair are their triggers apart in host
       time; (c) double-frame, each camera's A/B phase on its own clock.
    4. Triggers run.yaml requested that no camera accounts for become a note.

    Args:
        run_dir: The run folder (``Camera<k>.pivraw`` files and ``run.yaml``).
        frame_stride: Recorded frames from A to B of a pair. Double-frame
            accepts only the skip_frames preset (1, with ``pair_stride`` 2).
        pair_stride: Recorded frames from one pair's A to the next pair's A.

    Raises:
        PivrawError: A camera's files are refused (strict header), run.yaml is
            missing or malformed, or the strides do not suit the mode.
        PivrawAlignmentError: The run cannot be aligned; ``reason`` says why.
    """
    run_dir = Path(run_dir)
    slots = _run_cameras(run_dir)
    key = (str(run_dir), int(frame_stride), int(pair_stride))
    stamp = _run_stamp(run_dir, slots)
    cached = _alignment_cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    headers = {s: read_pivraw_header(run_dir / f"Camera{s}.pivraw") for s in slots}
    requested, trigger_source = _read_run_file(run_dir)
    modes = {h.mode for h in headers.values()}
    rates = {h.rate_hz for h in headers.values()}
    if len(modes) != 1 or len(rates) != 1:
        raise PivrawAlignmentError(
            "cameras_disagree",
            f"{run_dir}: cameras disagree on the mode {sorted(modes)} or the "
            f"trigger rate {sorted(rates)} Hz: not one recording. {_RERECORD}",
        )
    mode, rate = modes.pop(), rates.pop()
    _check_pairing(run_dir, mode, frame_stride, pair_stride)
    period_ns = 1e9 / rate
    cams = {s: _camera_triggers(h) for s, h in headers.items()}
    empty = [f"Camera{s}" for s, c in cams.items() if not c.position]
    if empty:
        raise PivrawAlignmentError(
            "no_frames",
            f"{run_dir}: {', '.join(empty)} holds no frames, so no pair is "
            "complete on every camera. Check that camera, then re-record the run.",
        )

    per = _frames_per_trigger(mode)
    last = max(c.span // per for c in cams.values()) - 1
    if mode == DOUBLE_FRAME:
        wanted = [(t, t) for t in range(last + 1)]
    else:
        wanted = [
            (t, t + frame_stride)
            for t in range(0, last - frame_stride + 1, pair_stride)
        ]
    pairs = []
    skipped = 0
    skipped_by = {s: 0 for s in slots}
    for ta, tb in wanted:
        lacking = [
            s for s, c in cams.items() if ta not in c.position or tb not in c.position
        ]
        if lacking:
            skipped += 1
            for s in lacking:
                skipped_by[s] += 1
            continue
        if mode == DOUBLE_FRAME:
            files = {
                s: (c.position[ta] + 1, c.position[ta] + 2) for s, c in cams.items()
            }
        else:
            files = {
                s: (c.position[ta] + 1, c.position[tb] + 1) for s, c in cams.items()
            }
        pairs.append(AlignedPair((ta, tb), files))
    if not pairs:
        raise PivrawAlignmentError(
            "no_complete_pair",
            f"{run_dir}: no PIV pair is complete on every camera ({skipped} "
            f"incomplete). {_RERECORD}",
        )

    ref = cams[slots[0]]
    for cam in (cams[s] for s in slots[1:]):
        offsets = np.array(
            [
                cam.host_ns[cam.position[p.run_triggers[0]]]
                - ref.host_ns[ref.position[p.run_triggers[0]]]
                for p in pairs
            ],
            dtype=np.float64,
        )
        k = int(np.argmax(np.abs(offsets)))
        if abs(offsets[k]) > TRIGGER_TOL_FRAC * period_ns:
            raise PivrawAlignmentError(
                "camera_offset",
                f"{run_dir}: Camera{cam.slot} run trigger "
                f"{pairs[k].run_triggers[0]} arrives {offsets[k] / 1e6:+.1f} ms "
                f"({offsets[k] / period_ns:+.2f} periods) from Camera{ref.slot} "
                "in host time: its frame ids count from a different trigger "
                "(first frame(s) lost, or a trigger ignored), and the files "
                f"cannot show which. {_RERECORD}",
            )

    pairs_t = tuple(pairs)
    notes = []
    for cam in cams.values():
        if mode == DOUBLE_FRAME:
            note = _ab_phase_note(run_dir, cam)
            if note is not None:
                notes.append(note)
        else:
            _check_tr_spacing(run_dir, cam, pairs_t, period_ns, trigger_source)
    unaccounted = requested - (last + 1)
    if unaccounted > 0:
        notes.append(
            f"run.yaml requested {requested} trigger(s) and the cameras account "
            f"for {last + 1}: {unaccounted} trigger(s) were lost before the "
            "first or after the last delivered frame (the files cannot say "
            "which end)."
        )
    elif unaccounted < 0:
        notes.append(
            f"run.yaml requested {requested} trigger(s) but the cameras account "
            f"for {last + 1}."
        )

    alignment = PivrawAlignment(
        run_dir=run_dir,
        mode=mode,
        period_s=period_ns / 1e9,
        cameras=slots,
        pairs=pairs_t,
        skipped_incomplete=skipped,
        skipped_by_camera=dict(skipped_by),
        triggers_requested=requested,
        triggers_unaccounted=unaccounted,
        trigger_source=trigger_source,
        notes=tuple(notes),
    )
    if skipped or notes:
        logging.warning(
            "%s: %s", run_dir, " ".join((alignment.summary,) + alignment.notes)
        )
    _alignment_cache[key] = (stamp, alignment)
    return alignment
