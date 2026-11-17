"""Pure-Python reader for LaVision .im7 image files.

Reads the Image_Header_7 binary format (256-byte header + pixel data + attributes).
No dependency on lvpyio -- works on macOS, Linux, and Windows.

Supports:
- Uncompressed float32 and uint16 pixel data (pack_type=0)
- DaVis token stream, uint16 (pack_type=1), decoded by :mod:`.davis_tokens`
- Zlib-compressed data (pack_type=2), one block per row or one per frame
- LZ4 (pack_type=20): header version 0 holds one block over the whole
  buffer, version 1 one block per frame plus a frame-offset table
- Intensity scale attributes (slope/offset)
- Multi-frame files (sizeF > 1)

Fixed 12-bit packed data (pack_type=3) is refused by name, see
:func:`_unsupported_pack_error`. So is a header version other than 0 or 1, and
version 1 on any pack type but 20, see :func:`_parse_header`.

Reference: LaVision ReadIM7.h / ReadIM7.cpp (liorshig/readim on GitHub)
"""

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .davis_tokens import TokenStreamError, decode_pack1
from .out_buffer import check_out

# buffer_format values
BUFFER_FORMAT_FLOAT = -3
BUFFER_FORMAT_WORD = -4

# pack_type values
PACK_UNCOMPRESSED = 0
# DaVis token stream (grammar in davis_tokens): a 3-byte preamble before frame 0
# only, then int8 deltas, 0x80 / 0x7f absolutes and 0x81 nibble runs, with the
# running value carried across frames. Equal to lvpyio 1.3.1 on every frame of
# all 28 real pack-1 files (64 frames, numba-decoder report, 2026-11-17).
PACK_RLE = 1
PACK_ZLIB = 2
# Refused by name. Its layout for widths not divisible by 4 is measured
# (container-unknowns D4) but no real pack-3 file exists among 34,509 surveyed
# .im7, so there is nothing to check a decoder against.
PACK_FIXED_12BIT = 3
# DaVis 10/11 lossless compression. Header version 0: one LZ4 *block* (not
# framed) over the whole pixel buffer, prefixed by an int64 little-endian
# compressed-size (version 1, one block per frame, is described at
# IM7_VERSION_PER_FRAME). The version-0 layout was
# reverse-engineered and verified bit-exact against LaVision's own lvpyio on
# merle + andre-x25 calibration frames (2026-06-12, see the calibration PRD).
PACK_LZ4 = 20

# Image_Header_7 version. Of 34,509 real .im7 headers on four drives (drive
# survey, 2026-11-16) every one is version 0 or 1, and all 10 version-1 files
# are pack 20 with 8 frames (DaVis "Subtract sliding minimum" output).
IM7_VERSION_BUFFER = 0
# Version 1, pack 20, measured byte for byte on the real files (D3, 2026-11-17):
# per frame an int64 compressed size and one LZ4 block; a table of size_f int64
# frame offsets; the attribute records; then, not read here, per-frame LZ4 mask
# planes (H x W uint8, all 1 in every real file), a mask-offset table and a
# 256-byte footer of offsets. lvpyio 1.3.1 agrees bit for bit on all 10 files.
IM7_VERSION_PER_FRAME = 1

# Extended header record types (int32)
IEH_END = 0
IEH_SCALE_X = 1
IEH_SCALE_Y = 2
IEH_SCALE_Z = 3
IEH_SCALE_I = 4
IEH_COMMENT = 5
IEH_ATTRIBUTE = 6
# Highest record type seen in real files: a 2026-10-23 survey of 398 DaVis .im7
# files (pack types 0, 2, 20) found types 0-6, 8, 9 and 11-13, with the first
# record always IEH_SCALE_X. _read_attributes rejects anything outside 1..13 so
# a decoder that stopped at the wrong byte fails instead of parsing pixel bytes.
IEH_TYPE_MAX = 13

HEADER_SIZE = 256
HEADER_FORMAT = "<hhhh iiii hhh"  # 30 bytes of fields + 226 reserved


@dataclass
class IM7Header:
    """Parsed Image_Header_7 from an .im7 file."""

    version: int
    pack_type: int
    buffer_format: int
    is_sparse: int
    size_x: int
    size_y: int
    size_z: int
    size_f: int
    scalar_n: int
    vector_grid: int
    extra_flags: int


@dataclass
class IM7Scales:
    """Intensity scale information extracted from .im7 attributes.

    The physical pixel value is: raw_pixel * slope + offset.
    """

    slope: float = 1.0
    offset: float = 0.0
    description: str = ""
    unit: str = ""


def _parse_header(raw: bytes, source: str) -> IM7Header:
    """Unpack the 256-byte Image_Header_7.

    Parameters
    ----------
    raw : bytes
        Exactly 256 bytes from the start of the .im7 file.
    source : str
        File name for error messages.

    Returns
    -------
    IM7Header
        Parsed header fields.

    Raises
    ------
    ValueError
        If raw is not 256 bytes, or any of size_x/size_y/size_z/size_f is below
        1. Every byte count the pixel decoders compute is a product of these, so
        a zero or negative one would surface as a division by zero, a negative
        read length or a huge allocation instead of as a damaged header.
        Also if the version is not 0 or 1, or is 1 on a pack type other than 20:
        neither has been seen in a real file, and the version decides the pixel
        layout, so reading on would decode a guess.
    """
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"Header must be {HEADER_SIZE} bytes, got {len(raw)}")

    values = struct.unpack_from(HEADER_FORMAT, raw, 0)
    dims = dict(zip(("size_x", "size_y", "size_z", "size_f"), values[4:8]))
    bad = {k: v for k, v in dims.items() if v < 1}
    if bad:
        raise ValueError(
            f"Damaged .im7 header: every dimension must be at least 1, got "
            f"{', '.join(f'{k}={v}' for k, v in bad.items())}"
        )
    version, pack_type = values[0], values[1]
    if version not in (IM7_VERSION_BUFFER, IM7_VERSION_PER_FRAME):
        raise ValueError(
            f"{source}: .im7 header version {version} is not read. Known "
            f"versions are {IM7_VERSION_BUFFER} (every pack type) and "
            f"{IM7_VERSION_PER_FRAME} (pack_type {PACK_LZ4}, one LZ4 block per "
            f"frame). Export the images from DaVis again, or report the file."
        )
    if version == IM7_VERSION_PER_FRAME and pack_type != PACK_LZ4:
        raise ValueError(
            f"{source}: .im7 header version {version} with pack_type "
            f"{pack_type} is not read. Version {version} has only been seen "
            f"with pack_type {PACK_LZ4}, so its layout for other pack types is "
            f"unknown. Export the images from DaVis again, or report the file."
        )
    return IM7Header(
        version=values[0],
        pack_type=values[1],
        buffer_format=values[2],
        is_sparse=values[3],
        size_x=values[4],
        size_y=values[5],
        size_z=values[6],
        size_f=values[7],
        scalar_n=values[8],
        vector_grid=values[9],
        extra_flags=values[10],
    )


def _pixel_dtype(header: IM7Header) -> np.dtype:
    """Return the numpy dtype for the pixel data based on buffer_format."""
    if header.buffer_format == BUFFER_FORMAT_FLOAT:
        return np.dtype("<f4")  # float32 little-endian
    elif header.buffer_format == BUFFER_FORMAT_WORD:
        return np.dtype("<u2")  # uint16 little-endian
    else:
        raise ValueError(
            f"Unsupported buffer_format: {header.buffer_format}. "
            f"Expected {BUFFER_FORMAT_FLOAT} (FLOAT) or {BUFFER_FORMAT_WORD} (WORD/uint16)."
        )


def _frame_byte_size(header: IM7Header) -> int:
    """Byte count for one frame (one sizeZ * sizeY plane) of uncompressed data."""
    return header.size_x * header.size_y * header.size_z * _pixel_dtype(header).itemsize


def _read_pixels_uncompressed(f, header: IM7Header) -> np.ndarray:
    """Read every frame of uncompressed pixel data (pack_type=0).

    Whole buffer only: the attribute records follow the last frame, so a reader
    that stopped early would hand ``_read_attributes`` pixel bytes. Per-camera
    partial reads live in :func:`read_im7_camera`, which seeks to the end of
    the buffer before reading attributes.

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data (after 256-byte header).
    header : IM7Header
        Parsed header.

    Returns
    -------
    np.ndarray
        Pixel array with shape (size_f, sizeZ, sizeY, sizeX).
    """
    dt = _pixel_dtype(header)
    frame_pixels = header.size_x * header.size_y * header.size_z
    frame_bytes = frame_pixels * dt.itemsize

    n_frames = header.size_f
    n_bytes = n_frames * frame_bytes
    raw = f.read(n_bytes)
    if len(raw) != n_bytes:
        raise IOError(
            f"Expected {n_bytes} bytes of pixel data, got {len(raw)}. "
            f"File may be truncated."
        )
    pixels = np.frombuffer(raw, dtype=dt)
    return pixels.reshape(n_frames, header.size_z, header.size_y, header.size_x)


try:  # optional C accelerator; the pure-python path below is correct without it
    import lz4.block as _lz4_block_mod
except ImportError:
    _lz4_block_mod = None


def _lz4_block_decompress(src: bytes, expected_size: int) -> bytes:
    """Decompress one raw LZ4 *block* (no frame header, no checksums).

    Uses the ``lz4`` package when installed; otherwise a pure-python decoder
    (~30 MB/s — fine for frame-at-a-time calibration reads). Raises if the
    output size does not match ``expected_size`` exactly: a truncated or
    misread buffer must fail visibly, never produce a short image.
    """
    if _lz4_block_mod is not None:
        out = _lz4_block_mod.decompress(src, uncompressed_size=expected_size)
        if len(out) != expected_size:
            raise IOError(
                f"LZ4 pixel buffer decompressed to {len(out)} bytes, "
                f"expected {expected_size}"
            )
        return out

    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i : i + lit]
        i += lit
        if i >= n:
            break  # final sequence carries literals only
        off = src[i] | (src[i + 1] << 8)
        i += 2
        mlen = (token & 0xF) + 4
        if (token & 0xF) == 15:
            while True:
                b = src[i]
                i += 1
                mlen += b
                if b != 255:
                    break
        start = len(out) - off
        if off == 0 or start < 0:
            raise IOError(f"corrupt LZ4 stream: match offset {off} at {len(out)}")
        if off >= mlen:
            out += out[start : start + mlen]
        else:
            # Overlapping match: LZ4 semantics require byte-serial copy.
            for _ in range(mlen):
                out.append(out[start])
                start += 1
        if len(out) > expected_size:
            raise IOError(
                f"corrupt LZ4 stream: output exceeds expected {expected_size} bytes"
            )
    if len(out) != expected_size:
        raise IOError(
            f"LZ4 pixel buffer decompressed to {len(out)} bytes, "
            f"expected {expected_size}"
        )
    return bytes(out)


def _read_pixels_lz4(
    f, header: IM7Header, source: str, frame_range: Optional[tuple] = None
) -> np.ndarray:
    """Read LZ4-packed pixel data (pack_type=20) in the layout its version names.

    Version 0 holds one LZ4 block over the whole buffer, version 1 one block
    per frame (see :data:`IM7_VERSION_PER_FRAME`). Either way ``f`` is left at
    the attribute records.

    Returns
    -------
    np.ndarray
        Shape (end - start, size_z, size_y, size_x) for ``frame_range``
        (start, end), default every frame.
    """
    if header.version == IM7_VERSION_PER_FRAME:
        return _read_lz4_per_frame(f, header, source, frame_range)
    return _read_lz4_whole_buffer(f, header, frame_range)


def _read_lz4_whole_buffer(
    f, header: IM7Header, frame_range: Optional[tuple] = None
) -> np.ndarray:
    """Version-0 pack 20: one int64 size, then one LZ4 block over every frame.

    The full buffer is decompressed and the requested frames sliced out.
    Leaves ``f`` positioned at the attribute records (the byte after the
    compressed block).
    """
    dt = _pixel_dtype(header)
    frame_pixels = header.size_x * header.size_y * header.size_z
    total_bytes = frame_pixels * dt.itemsize * header.size_f

    comp_size = struct.unpack("<q", f.read(8))[0]
    if comp_size <= 0:
        raise IOError(f"invalid LZ4 compressed size: {comp_size}")
    comp = f.read(comp_size)
    if len(comp) != comp_size:
        raise IOError(
            f"Expected {comp_size} bytes of LZ4 data, got {len(comp)}. "
            f"File may be truncated."
        )
    out = _lz4_block_decompress(comp, total_bytes)
    pixels = np.frombuffer(out, dtype=dt).reshape(
        header.size_f, header.size_z, header.size_y, header.size_x
    )

    if frame_range is None:
        return pixels
    start, end = frame_range
    return pixels[start:end]


def _read_lz4_frame_size(f, k: int, n_frames: int, file_size: int, source: str) -> int:
    """Read and check the int64 compressed size in front of version-1 frame ``k``.

    The size must be positive and the block it announces must fit in the file,
    so a truncated file fails here by name for a frame that is skipped as well
    as for one that is decoded.
    """
    pos = f.tell()
    raw = f.read(8)
    if len(raw) != 8:
        raise IOError(
            f"{source}: version-1 LZ4 frame {k} of {n_frames} is cut off: its "
            f"8-byte size at byte {pos} has {len(raw)} bytes. The file is truncated."
        )
    comp_size = struct.unpack("<q", raw)[0]
    if comp_size <= 0:
        raise IOError(
            f"{source}: version-1 LZ4 frame {k} of {n_frames} declares "
            f"{comp_size} compressed bytes at byte {pos}. The file is damaged."
        )
    remaining = file_size - pos - 8
    if comp_size > remaining:
        raise IOError(
            f"{source}: version-1 LZ4 frame {k} of {n_frames} declares "
            f"{comp_size} compressed bytes at byte {pos}, but only {remaining} "
            f"remain. The file is truncated."
        )
    return comp_size


def _read_lz4_per_frame(
    f, header: IM7Header, source: str, frame_range: Optional[tuple] = None
) -> np.ndarray:
    """Version-1 pack 20: one int64 size and one LZ4 block per frame.

    After the last block comes a table of size_f int64 values, the file offset
    of each frame's size prefix, and then the attribute records. Frames outside
    ``frame_range`` are skipped by seeking past their blocks, not decoded. The
    table must equal the offsets the walk found, which checks the layout on
    every read. Leaves ``f`` positioned at the attribute records.

    The per-frame mask planes and the offset footer DaVis writes after the
    attribute records (see :data:`IM7_VERSION_PER_FRAME`) are not read.
    """
    dt = _pixel_dtype(header)
    frame_bytes = header.size_x * header.size_y * header.size_z * dt.itemsize
    n_frames = header.size_f
    start, end = frame_range if frame_range is not None else (0, n_frames)
    file_size = os.fstat(f.fileno()).st_size

    pixels = np.empty((end - start, header.size_z, header.size_y, header.size_x), dt)
    starts = []
    for k in range(n_frames):
        starts.append(f.tell())
        comp_size = _read_lz4_frame_size(f, k, n_frames, file_size, source)
        if not start <= k < end:
            f.seek(comp_size, 1)
            continue
        try:
            raw = _lz4_block_decompress(f.read(comp_size), frame_bytes)
        except IOError as exc:
            raise IOError(
                f"{source}: version-1 LZ4 frame {k} of {n_frames}: {exc}"
            ) from exc
        pixels[k - start] = np.frombuffer(raw, dtype=dt).reshape(pixels.shape[1:])

    pos = f.tell()
    table_raw = f.read(8 * n_frames)
    if len(table_raw) != 8 * n_frames:
        raise IOError(
            f"{source}: the version-1 frame-offset table at byte {pos} has "
            f"{len(table_raw)} of its {8 * n_frames} bytes. The file is truncated."
        )
    table = list(struct.unpack(f"<{n_frames}q", table_raw))
    if table != starts:
        raise IOError(
            f"{source}: the version-1 frame-offset table at byte {pos} lists "
            f"{table}, but the frames start at {starts}. The file is damaged or "
            f"is a pack-20 layout this reader does not know."
        )
    return pixels


def _zlib_block_bytes(f, header: IM7Header) -> int:
    """Uncompressed size of each zlib block in a pack_type-2 file: a row or a frame.

    DaVis writes pack_type 2 in two layouts, both a sequence of int32-size-
    prefixed zlib blocks: one block per row (the ReadIM7 reference layout), or
    one block per frame (DaVis dark images, ``Transformer*-dark.im7`` at
    1248x1680, found 2026-10-23). The first block tells them apart: inflating at
    most ``row_bytes + 1`` bytes of it yields exactly ``row_bytes`` for a per-row
    file and more for a per-frame one. Per-frame blocks are verified on
    single-frame files only; every block's inflated size is checked on read, so
    a multi-frame file laid out differently raises instead of decoding wrongly.

    Leaves ``f`` where it found it (at the first block's size prefix).
    """
    row_bytes = header.size_x * _pixel_dtype(header).itemsize
    frame_bytes = row_bytes * header.size_y * header.size_z
    pos = f.tell()
    size_raw = f.read(4)
    if len(size_raw) < 4:
        raise IOError("Unexpected EOF reading compressed size")
    comp_size = struct.unpack("<i", size_raw)[0]
    if comp_size <= 0:
        raise IOError(f"invalid zlib block size: {comp_size}")
    head = zlib.decompressobj().decompress(f.read(comp_size), row_bytes + 1)
    f.seek(pos)
    if len(head) > row_bytes:
        return frame_bytes
    if len(head) == row_bytes:
        return row_bytes
    raise IOError(
        f"First zlib block inflated to {len(head)} bytes; expected one row "
        f"({row_bytes}) or one frame ({frame_bytes})"
    )


def _read_zlib_block(f, block_bytes: int) -> bytes:
    """Read one int32-size-prefixed zlib block that must inflate to block_bytes.

    The stream must end exactly at the declared size with nothing left over, so
    a misread size prefix or layout fails here rather than shifting the image.
    """
    size_raw = f.read(4)
    if len(size_raw) < 4:
        raise IOError("Unexpected EOF reading compressed size")
    comp_size = struct.unpack("<i", size_raw)[0]
    if comp_size <= 0:
        raise IOError(f"invalid zlib block size: {comp_size}")
    comp = f.read(comp_size)
    if len(comp) != comp_size:
        raise IOError("Unexpected EOF reading compressed data")
    inflater = zlib.decompressobj()
    out = inflater.decompress(comp)
    if not inflater.eof or inflater.unused_data:
        raise IOError(
            f"zlib block of {comp_size} bytes does not end at its declared size"
        )
    if len(out) != block_bytes:
        raise IOError(
            f"zlib block inflated to {len(out)} bytes, expected {block_bytes}"
        )
    return out


def _skip_zlib_blocks(f, n_blocks: int) -> None:
    """Skip n_blocks zlib blocks by reading their size prefixes and seeking."""
    for _ in range(n_blocks):
        comp_size_raw = f.read(4)
        if len(comp_size_raw) < 4:
            raise IOError("Unexpected EOF skipping compressed blocks")
        comp_size = struct.unpack("<i", comp_size_raw)[0]
        f.seek(comp_size, 1)


def _read_pixels_zlib(f, header: IM7Header) -> np.ndarray:
    """Read every frame of zlib-compressed pixel data (pack_type=2).

    Format: a sequence of blocks, each a 4-byte little-endian int32 giving the
    compressed size followed by that many bytes of zlib data, one block per row
    or one per frame (see :func:`_zlib_block_bytes`). Whole buffer only, for the
    reason given on :func:`_read_pixels_uncompressed`.

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data.
    header : IM7Header
        Parsed header.

    Returns
    -------
    np.ndarray
        Pixel array with shape (size_f, sizeZ, sizeY, sizeX).
    """
    dt = _pixel_dtype(header)
    frame_bytes = header.size_x * header.size_y * header.size_z * dt.itemsize
    block_bytes = _zlib_block_bytes(f, header)

    n_frames = header.size_f
    n_blocks = n_frames * (frame_bytes // block_bytes)
    buf = b"".join(_read_zlib_block(f, block_bytes) for _ in range(n_blocks))
    pixels = np.frombuffer(buf, dtype=dt)
    return pixels.reshape(n_frames, header.size_z, header.size_y, header.size_x)


def _unsupported_pack_error(header: IM7Header, source: str) -> ValueError:
    """The error both dispatchers raise for a pack type this reader does not read."""
    if header.pack_type == PACK_FIXED_12BIT:
        return ValueError(
            f"{source}: pack_type 3 (fixed 12-bit packed) is not read. No real "
            f"DaVis file of this type has been found to check a decoder against. "
            f"Export the recording from DaVis uncompressed or with lossless "
            f"compression (pack types 0, 1, 2 or 20)."
        )
    return ValueError(
        f"{source}: unsupported pack_type {header.pack_type}. Expected 0 "
        f"(uncompressed), 1 (token stream), 2 (zlib) or 20 (LZ4)."
    )


def _read_pixels_packtype1(
    f, header: IM7Header, source: str, frame_range: Optional[tuple] = None
) -> np.ndarray:
    """Read pack_type=1 (DaVis token stream) pixel data.

    The frames form one token stream that cannot be seeked into, so every
    frame is decoded in order, including the ones after the requested range:
    the attribute records sit after the last frame, and decoding up to it is the
    only way to find that byte. Only the requested range is returned. Leaves
    ``f`` positioned at the attribute records.

    The remaining bytes are read into a pre-allocated writable uint8 array,
    which is the one array type :func:`.davis_tokens.decode_pack1` accepts.

    Parameters
    ----------
    f : file object
        Positioned at the start of pixel data (after the 256-byte header).
    header : IM7Header
        Parsed header.
    source : str
        File name for error messages.
    frame_range : (start, end), optional
        Frames to return. Default all.

    Returns
    -------
    np.ndarray
        uint16, shape (end - start, size_z, size_y, size_x).

    Raises
    ------
    ValueError
        If the buffer format is not WORD. The tokens code 16-bit values, and
        all 28 real pack-1 files are WORD.
    TokenStreamError
        A named subclass, with the file name, if the stream does not decode.
    """
    if header.buffer_format != BUFFER_FORMAT_WORD:
        raise ValueError(
            f"{source}: pack_type 1 with buffer_format {header.buffer_format} is "
            f"not read. The token stream codes 16-bit values, and every pack-1 "
            f"file seen so far is WORD ({BUFFER_FORMAT_WORD})."
        )
    npix = header.size_x * header.size_y * header.size_z
    nf = header.size_f
    start, end = frame_range if frame_range is not None else (0, nf)

    base = f.tell()
    remaining = os.fstat(f.fileno()).st_size - base
    data = np.empty(remaining, dtype=np.uint8)
    got = f.readinto(data)
    if got != remaining:
        raise IOError(f"{source}: read {got} of the {remaining} bytes after the header")
    try:
        frames, off = decode_pack1(data, npix, nf, range(start, end))
    except TokenStreamError as exc:
        raise type(exc)(f"{source}: {exc}") from exc
    f.seek(base + off)
    return frames.reshape(end - start, header.size_z, header.size_y, header.size_x)


def _read_attributes(f, source: str, pack_type: int) -> IM7Scales:
    """Read extended header records (attributes) after the pixel data.

    The extended header is a sequence of records, each with:
        int32  type   (IEH_END=0 terminates)
        int32  size   (byte count of data that follows)
        bytes  data   (size bytes)

    Scale records (types 1-4) contain: 'slope offset\\0unit\\0'
    Attribute records (type 6) contain: 'name=value' (single string)

    Only IEH_SCALE_I (type 4) is extracted for the intensity scale.

    This walk is also the check that the pixel decoder stopped at the right
    byte, for every pack type: a decoder that misread the stream leaves ``f``
    inside pixel data, and parsing on would yield a garbled image with a
    default scale and no error. So EOF straight after the pixels is accepted
    (no records), but otherwise the first record must be a real one (type
    1..IEH_TYPE_MAX, not a bare IEH_END), every record must have a type in that
    range and fit in the file, and the chain ends at IEH_END or EOF. Whatever
    follows IEH_END (real files carry further data there) is not read.

    Parameters
    ----------
    f : file object
        Positioned at the start of the attribute section.
    source : str
        File name for error messages.
    pack_type : int
        The pixel pack type, for error messages.

    Returns
    -------
    IM7Scales
        Intensity scale (slope, offset, unit). Defaults to slope=1, offset=0
        if no SCALE_I record is found.

    Raises
    ------
    IOError
        If the bytes after the pixel data are not a valid record chain.
    """

    def fail(pos: int, reason: str) -> IOError:
        return IOError(
            f"{source}: pixel data (pack_type {pack_type}) does not end where the "
            f"attribute records begin ({reason} at byte {pos}). Either the file "
            f"is damaged or it uses a variant of this pack type the reader does "
            f"not decode correctly."
        )

    scales = IM7Scales()
    first = True

    while True:
        pos = f.tell()
        type_raw = f.read(4)
        if len(type_raw) == 0:
            break  # EOF: no (more) records
        if len(type_raw) < 4:
            raise fail(pos, f"{len(type_raw)} stray trailing bytes")

        ieh_type = struct.unpack("<i", type_raw)[0]
        if ieh_type == IEH_END:
            if first:
                raise fail(pos, "an empty record list")
            break
        if not 1 <= ieh_type <= IEH_TYPE_MAX:
            raise fail(pos, f"record type {ieh_type}")

        size_raw = f.read(4)
        if len(size_raw) < 4:
            raise fail(pos, f"record type {ieh_type} with no size field")
        ieh_size = struct.unpack("<i", size_raw)[0]
        if ieh_size < 0:
            raise fail(pos, f"record type {ieh_type} of size {ieh_size}")

        data = f.read(ieh_size)
        if len(data) != ieh_size:
            raise fail(
                pos, f"record type {ieh_type} of size {ieh_size} past end of file"
            )

        if ieh_type == IEH_SCALE_I:
            _parse_scale_record(data, scales)
        first = False

    return scales


def _parse_scale_record(data: bytes, scales: IM7Scales) -> None:
    """Parse a scale record into slope/offset/unit.

    The data format is: 'slope offset\\0unit\\0[padding]'
    where slope and offset are space-separated ASCII floats.
    """
    parts = data.split(b"\x00")
    if not parts:
        return

    # First part: "slope offset"
    values_str = parts[0].decode("utf-8", errors="replace").strip()
    tokens = values_str.split()
    if len(tokens) >= 2:
        try:
            scales.slope = float(tokens[0])
            scales.offset = float(tokens[1])
        except ValueError:
            pass
    elif len(tokens) == 1:
        try:
            scales.slope = float(tokens[0])
        except ValueError:
            pass

    # Second part: unit string
    if len(parts) > 1:
        scales.unit = parts[1].decode("utf-8", errors="replace").strip()

    # Store the raw description
    scales.description = values_str


def _read_im7_internal(filepath: Path) -> tuple:
    """Core whole-file reader: every frame, then the attribute records.

    There is deliberately no frame range here. The attributes follow the last
    frame of the buffer, and for pack types 0 and 2 a partial read would
    leave the pointer inside pixel data; per-camera partial reads are
    :func:`read_im7_camera`'s job, which repositions before reading them.

    Parameters
    ----------
    filepath : Path
        Path to the .im7 file (must exist).

    Returns
    -------
    header, pixels, scales
    """
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
        if len(raw_header) != HEADER_SIZE:
            raise IOError(
                f"File too small for header ({len(raw_header)} < {HEADER_SIZE})"
            )
        header = _parse_header(raw_header, filepath.name)

        if header.pack_type == PACK_UNCOMPRESSED:
            pixels = _read_pixels_uncompressed(f, header)
        elif header.pack_type == PACK_RLE:
            pixels = _read_pixels_packtype1(f, header, filepath.name)
        elif header.pack_type == PACK_ZLIB:
            pixels = _read_pixels_zlib(f, header)
        elif header.pack_type == PACK_LZ4:
            pixels = _read_pixels_lz4(f, header, filepath.name)
        else:
            raise _unsupported_pack_error(header, filepath.name)

        scales = _read_attributes(f, filepath.name, header.pack_type)

    return header, pixels, scales


def read_im7(filepath: Union[str, Path]) -> tuple:
    """Read a LaVision .im7 image file.

    Returns the parsed header, pixel data array, and intensity scales.
    Single-frame (sizeF=1), single-plane (sizeZ=1) data is squeezed
    to 2D (H, W). Multi-frame data retains all dimensions.

    Parameters
    ----------
    filepath : str or Path
        Path to the .im7 file.

    Returns
    -------
    header : IM7Header
        Parsed binary header.
    pixels : np.ndarray
        Pixel data. Shape is (H, W) for single-frame single-plane,
        or (F, H, W) for multi-frame with single plane,
        or (F, Z, H, W) for multi-frame/multi-plane.
    scales : IM7Scales
        Intensity scale (slope, offset, unit).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    header, pixels, scales = _read_im7_internal(filepath)

    # Squeeze singleton dimensions
    if pixels.shape[0] == 1 and header.size_z == 1:
        pixels = pixels[0, 0]  # (H, W)
    elif pixels.shape[0] == 1:
        pixels = pixels[0]  # (Z, H, W)
    elif header.size_z == 1:
        pixels = pixels[:, 0]  # (F, H, W)

    return header, pixels, scales


def get_im7_frame_count(filepath: Union[str, Path]) -> int:
    """Return the number of frames (size_f) in an .im7 file.

    Reads only the 256-byte header — no pixel decode. Used to detect how many
    frames a multi-camera buffer holds so ``frames_per_camera`` can be derived
    from the configured camera count rather than hard-coded.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
    return _parse_header(raw_header, filepath.name).size_f


def get_im7_frame_shape(filepath: Union[str, Path]) -> tuple:
    """Return the (height, width) every frame of an .im7 buffer has.

    Reads only the 256-byte header -- no pixel decode. All frames in one buffer
    share the header's size_y x size_x, so this is the shape of any camera's
    frame regardless of which camera or frame is asked for.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
    header = _parse_header(raw_header, filepath.name)
    return (header.size_y, header.size_x)


def read_im7_camera(
    filepath: Union[str, Path],
    camera_no: int = 1,
    frames_per_camera: int = 2,
    frames: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read a specific camera's frames from a multi-camera .im7 file.

    Only reads the frames belonging to the requested camera — skips the
    rest on disk (seek for uncompressed, skip for zlib).

    Parameters
    ----------
    filepath : str or Path
        Path to the .im7 file.
    camera_no : int
        Camera number (1-based).
    frames_per_camera : int
        Frames stored per camera (2 for standard PIV A/B pairs, 1 for
        time-resolved or calibration snapshots).
    frames : int, optional
        How many of this camera's frames to read, from its first. Default
        (None) reads the whole camera slice. ``frames=1`` on a double-frame
        buffer reads frame A only: for pack types 0 and 2 that halves the
        bytes decoded (measured 61.3 -> 30.9 ms, 63 -> 32 MB on a 4-camera
        4872x3248 pack-0 calibration file), and version-1 pack 20 (one LZ4
        block per frame) decodes only the frames returned. Version-0 pack 20
        (one LZ4 block for the whole buffer) and pack 1 (one token stream,
        sequential) decode every frame of the buffer by format design, so
        there it narrows only the result.
    out : np.ndarray, optional
        Destination (n, H, W) float32 C-contiguous buffer, where n is the number
        of frames this call will actually produce AFTER clamping to the file. A
        single-frame file asked for two frames with a (2, H, W) buffer raises by
        name rather than leaving out[1] as uninitialised memory. Written in
        place and returned; undefined after an exception.

    Returns
    -------
    np.ndarray
        Pixel data with shape (n, H, W), dtype float32, with intensity scale
        applied, where n = min(frames or frames_per_camera, frames available).
        ``out`` itself when given.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Parse header first to validate frame range
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
    header = _parse_header(raw_header, filepath.name)

    if frames is not None and frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")

    start = (camera_no - 1) * frames_per_camera
    wanted = frames_per_camera if frames is None else min(frames, frames_per_camera)
    end = start + wanted

    if start >= header.size_f:
        raise ValueError(
            f"Camera {camera_no} with {frames_per_camera} frames/camera "
            f"requires frame index {start}, but file only has "
            f"{header.size_f} frames."
        )

    # Clamp to available frames (single-frame calibration .im7 files have
    # sizeF=1 but callers may request frames_per_camera=2 by default)
    end = min(end, header.size_f)
    n_frames = end - start

    # Pre-allocate float32 output (or take the caller's), read one frame at a
    # time so only one raw-dtype frame is in memory at any point.
    if out is None:
        result = np.empty((n_frames, header.size_y, header.size_x), dtype=np.float32)
    else:
        if out.ndim == 3 and out.shape[0] != n_frames:
            raise ValueError(
                f"{filepath.name} holds {n_frames} frame(s) for camera {camera_no} "
                f"(asked for {wanted}), but a {tuple(out.shape)} buffer was "
                f"supplied. A single-frame file cannot fill a pair slot; nothing "
                f"is written into the missing frames."
            )
        result = check_out(
            out,
            (n_frames, header.size_y, header.size_x),
            f"read_im7_camera camera {camera_no} of {filepath.name}",
        )

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)  # skip header

        if header.pack_type == PACK_UNCOMPRESSED:
            dt = _pixel_dtype(header)
            frame_bytes = header.size_x * header.size_y * header.size_z * dt.itemsize
            # Seek past earlier cameras' frames
            if start > 0:
                f.seek(start * frame_bytes, 1)
            for i in range(n_frames):
                raw = f.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise IOError("Unexpected EOF reading frame data")
                frame = np.frombuffer(raw, dtype=dt).reshape(
                    header.size_z, header.size_y, header.size_x
                )
                result[i] = (
                    frame[0]
                    if header.size_z == 1
                    else frame.reshape(header.size_y, header.size_x)
                )
                del frame, raw  # free before next iteration's read
            # The attribute records follow the LAST frame of the buffer, not the
            # last frame read. Without this seek, every camera but the last had
            # its intensity scale parsed out of the next camera's pixel bytes.
            f.seek(HEADER_SIZE + header.size_f * frame_bytes)

        elif header.pack_type == PACK_ZLIB:
            dt = _pixel_dtype(header)
            frame_bytes = header.size_x * header.size_y * header.size_z * dt.itemsize
            block_bytes = _zlib_block_bytes(f, header)
            blocks_per_frame = frame_bytes // block_bytes
            if start > 0:
                _skip_zlib_blocks(f, start * blocks_per_frame)
            for i in range(n_frames):
                buf = b"".join(
                    _read_zlib_block(f, block_bytes) for _ in range(blocks_per_frame)
                )
                frame = np.frombuffer(buf, dtype=dt).reshape(
                    header.size_z, header.size_y, header.size_x
                )
                result[i] = (
                    frame[0]
                    if header.size_z == 1
                    else frame.reshape(header.size_y, header.size_x)
                )
                del frame, buf
            # Walk the remaining frames' block-size prefixes (no decompression) so
            # the attribute records are read from where they actually are.
            _skip_zlib_blocks(f, (header.size_f - end) * blocks_per_frame)

        elif header.pack_type == PACK_LZ4:
            # Version 0 decompresses the whole buffer and slices; version 1
            # decodes only this camera's frames and seeks past the others.
            frames = _read_pixels_lz4(f, header, filepath.name, (start, end))
            for i in range(n_frames):
                frame = frames[i]
                result[i] = (
                    frame[0]
                    if header.size_z == 1
                    else frame.reshape(header.size_y, header.size_x)
                )
            del frames

        elif header.pack_type == PACK_RLE:
            # One token stream over every frame; decode in order, keep the range.
            frames = _read_pixels_packtype1(f, header, filepath.name, (start, end))
            for i in range(n_frames):
                frame = frames[i]
                result[i] = (
                    frame[0]
                    if header.size_z == 1
                    else frame.reshape(header.size_y, header.size_x)
                )
            del frames
        else:
            raise _unsupported_pack_error(header, filepath.name)

        # Read attributes for intensity scale
        scales = _read_attributes(f, filepath.name, header.pack_type)

    # Apply intensity scale in-place
    if scales.slope != 1.0:
        result *= scales.slope
    if scales.offset != 0.0:
        result += scales.offset
    return result
