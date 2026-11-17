"""Reader for PIVTOOLS Acquisition run containers (``.pivraw``).

One run folder per recording, laid out like Phantom ``.cine`` (one container per
camera, the ``%d`` in the pattern names the camera)::

    <run_dir>/Camera1.pivraw         frames back to back, frame k at k * frame_stride
    <run_dir>/Camera1.pivraw.json    header sidecar
    <run_dir>/Camera1.pivraw.idx     one "<qQ" record (host ns, camera frame id) per frame
    <run_dir>/Camera2.pivraw ...

The format is defined by the acquisition app (``pivtools-acquisition``
``docs/CONTRACTS.md`` section 4). PIVTOOLs does not import that package; the
container is read here from its specification alone.

Layout facts the reader relies on:

* ``frame_stride = ceil(bytes_per_frame / 4096) * 4096``; the bytes between the
  payload and the stride are zero padding. A finalised file therefore holds
  exactly ``n_frames * frame_stride`` bytes, and its ``.idx`` exactly
  ``n_frames * 16``.
* Mono12p is the PFNC LSB-first packing (two pixels in three bytes), the layout
  ``set_reader._decode_mono12p`` already decodes, so that decoder is reused.
* BayerRG8 becomes a 2x2 superpixel sum R+G+G+B at half resolution, with no
  demosaicing (decision 2026-10-26). The sum of four 8-bit photosites is at most
  1020, exact in float32.
* A double-frame run holds whole pairs A, B, A, B, ... so a PIV pair is frames
  ``(2k-1, 2k)`` (1-based): the ``skip_frames`` pairing preset (frame_stride 1,
  pair_stride 2). A time-resolved run pairs any two frames.

The reader is strict: a missing sidecar field, a wrong ``format_version``, an
unfinalised run, or a sidecar that disagrees with the files on disk raises
:class:`PivrawError` (a ``ValueError``) naming the file. Frame indices are
1-based, like every other container in PIVTOOLs. Returned pixels are raw
counts as float32: (H, W) for a frame, (2, H, W) for a pair.
"""

import json
import logging
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from .out_buffer import check_out
from .set_reader import _decode_mono12p

PathLike = Union[str, Path]

FORMAT_NAME = "pivraw"
FORMAT_VERSION = 1
SIDECAR_SUFFIX = ".json"  # Camera1.pivraw -> Camera1.pivraw.json
INDEX_SUFFIX = ".idx"  # Camera1.pivraw -> Camera1.pivraw.idx
# One "<qQ" record per frame: host timestamp (ns, int64) + camera frame id (uint64).
INDEX_RECORD_BYTES = 16
# Frames start on 4096-byte boundaries so the recorder can write with direct I/O.
STRIDE_ALIGNMENT = 4096

MONO12P = "Mono12p"
BAYER_RG8 = "BayerRG8"
PIXEL_FORMATS = (MONO12P, BAYER_RG8)

DOUBLE_FRAME = "double_frame"
TIME_RESOLVED = "time_resolved"
MODES = (DOUBLE_FRAME, TIME_RESOLVED)

REQUIRED_FIELDS = (
    "format",
    "format_version",
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
)


class PivrawError(ValueError):
    """A ``.pivraw`` container that cannot be read as specified."""


@dataclass(frozen=True)
class PivrawHeader:
    """The validated sidecar of one camera's ``.pivraw`` container."""

    path: Path
    camera_model: str
    camera_serial: str
    slot: Any
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


def _int_field(sidecar: Mapping[str, Any], key: str, where: str, minimum: int) -> int:
    """A required integer field >= ``minimum`` (bool is not an integer here)."""
    value = sidecar[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
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


def _parse_sidecar(raw_path: Path, sidecar: Any) -> PivrawHeader:
    """Validate a decoded sidecar against the format and the files on disk.

    Raises:
        PivrawError: On any missing field, wrong format/version, unfinalised run,
            inconsistent geometry, or a data/index file whose size disagrees with
            ``n_frames``.
    """
    side = sidecar_path(raw_path)
    where = str(side)
    if not isinstance(sidecar, dict):
        raise PivrawError(
            f"{where}: expected a JSON object, got {type(sidecar).__name__}."
        )
    missing = [k for k in REQUIRED_FIELDS if k not in sidecar]
    if missing:
        raise PivrawError(f"{where}: missing required field(s) {', '.join(missing)}.")
    if sidecar["format"] != FORMAT_NAME:
        raise PivrawError(
            f"{where}: format is {sidecar['format']!r}, expected {FORMAT_NAME!r}."
        )
    version = sidecar["format_version"]
    if isinstance(version, bool) or version != FORMAT_VERSION:
        raise PivrawError(
            f"{where}: format_version {version!r} is not supported; this PIVTOOLs "
            f"reads format_version {FORMAT_VERSION}."
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
        isinstance(m, bool) or not isinstance(m, int) or m < 0 for m in missing_indices
    ):
        raise PivrawError(
            f"{where}: missing_indices must be a list of non-negative integers, "
            f"got {missing_indices!r}."
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
            f"{INDEX_RECORD_BYTES} bytes = {n_frames * INDEX_RECORD_BYTES}."
        )

    header = PivrawHeader(
        path=raw_path,
        camera_model=str(sidecar["camera_model"]),
        camera_serial=str(sidecar["camera_serial"]),
        slot=sidecar["slot"],
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
        created_at=str(sidecar["created_at"]),
        n_frames=n_frames,
        dropped_pairs=dropped_pairs,
        missing_indices=tuple(missing_indices),
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
