"""DaVis token decoder shared by ``.im7`` pack 1 and IMX-compressed ``.set`` entries.

DaVis codes a run of 16-bit values as a stream of tokens that update a running
value. The grammar was measured against LaVision's lvpyio 1.3.1, which is the
oracle (container-unknowns C5-C7, numba-decoder report, 2026-11-16/17)::

    int8 byte (not 0x7f/0x80/0x81)  signed delta added to the running value
    0x80 lo hi                      one absolute 16-bit little-endian value
    0x7f n (lo hi)*n                n absolute 16-bit little-endian values
    0x81 nibble bytes...            signed 4-bit deltas, high nibble first.
                                    Only a LOW nibble 0x8 ends the run; a high
                                    nibble 0x8 is a delta of -8.

``.im7`` pack 1 puts a 3-byte preamble (``01 01 00`` in all 28 real files) before
frame 0 only, and the running value carries from the last pixel of frame k into
frame k+1. An IMX ``.set`` entry has no preamble, its running value starts at 0,
and the tokens code the stored words of the packed payload, not pixels.

The kernel is compiled lazily by numba on first use and cached in the in-tree
``__pycache__`` (numba falls back to a user-wide directory where that cannot be
written). numba does not bounds-check array reads by default, so the kernel
guards every read itself and reports a failure as a status code, which the
wrappers raise as a named :class:`TokenStreamError` subclass. Set
``NUMBA_DISABLE_JIT=1`` to run the same source as plain Python.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

# Kernel status codes. The wrappers raise on anything but _STATUS_OK.
_STATUS_OK = 0
_STATUS_TRUNCATED = 1  # the input ended inside a token
_STATUS_OVERRUN = 2  # a 0x7f run would write past the requested value count
_STATUS_RANGE = 3  # the running value left 0..65535

# "01 01 00" before frame 0 of an .im7 pack-1 buffer only (drive-survey D2,
# container-unknowns C7: 28 of 28 real files).
PACK1_PREAMBLE_BYTES = 3

# Output is uint16. lvpyio's behaviour outside this range is unmeasured, so the
# kernel stops there rather than wrapping (Morgan, 2026-11-17).
_UINT16_MAX = 65535


@njit(cache=True, nogil=True)
def _decode_tokens(data, off, out, prev):
    """Decode ``out.shape[0]`` values from ``data[off:]`` into ``out``.

    Parameters
    ----------
    data : uint8[::1]
        The whole token stream.
    off : int
        Byte offset of the first token.
    out : uint16[::1]
        Destination. Its length is the number of values to decode.
    prev : int
        Running value before the first token.

    Returns
    -------
    (off, prev, n_written, status)
        Byte offset after the last token read, running value after the last
        value written, number of values written, and a ``_STATUS_*`` code.
    """
    n = data.shape[0]
    npix = out.shape[0]
    i = 0
    while i < npix:
        if off >= n:
            return off, prev, i, _STATUS_TRUNCATED
        b = np.int64(data[off])
        off += 1
        if b == 0x81:
            while i < npix:
                if off >= n:
                    return off, prev, i, _STATUS_TRUNCATED
                byte = np.int64(data[off])
                off += 1
                hi = byte >> 4
                prev += hi - 16 if hi >= 8 else hi
                if prev < 0 or prev > _UINT16_MAX:
                    return off, prev, i, _STATUS_RANGE
                out[i] = prev
                i += 1
                if i >= npix:
                    break
                lo = byte & 0xF
                if lo == 8:
                    break
                prev += lo - 16 if lo >= 8 else lo
                if prev < 0 or prev > _UINT16_MAX:
                    return off, prev, i, _STATUS_RANGE
                out[i] = prev
                i += 1
        elif b == 0x80:
            if off + 2 > n:
                return off, prev, i, _STATUS_TRUNCATED
            prev = np.int64(data[off]) | (np.int64(data[off + 1]) << 8)
            off += 2
            out[i] = prev
            i += 1
        elif b == 0x7F:
            if off >= n:
                return off, prev, i, _STATUS_TRUNCATED
            cnt = np.int64(data[off])
            off += 1
            if i + cnt > npix:
                return off, prev, i, _STATUS_OVERRUN
            if off + 2 * cnt > n:
                return off, prev, i, _STATUS_TRUNCATED
            for _ in range(cnt):
                prev = np.int64(data[off]) | (np.int64(data[off + 1]) << 8)
                off += 2
                out[i] = prev
                i += 1
        else:
            prev += b - 256 if b >= 128 else b
            if prev < 0 or prev > _UINT16_MAX:
                return off, prev, i, _STATUS_RANGE
            out[i] = prev
            i += 1
    return off, prev, i, _STATUS_OK


class TokenStreamError(IOError):
    """A DaVis token stream that does not decode under the measured grammar."""


class TruncatedTokenStreamError(TokenStreamError):
    """The stream ended inside a token, before every value was decoded."""


class TokenRunOverrunError(TokenStreamError):
    """A 0x7f run of absolutes runs past the number of values requested."""


class TokenValueRangeError(TokenStreamError):
    """The running value left the uint16 range 0..65535."""


class TrailingTokenBytesError(TokenStreamError):
    """A self-contained token entry holds bytes after its last value."""


_STATUS_ERRORS = {
    _STATUS_TRUNCATED: (
        TruncatedTokenStreamError,
        "the token stream ended inside a token (truncated file or unknown token)",
    ),
    _STATUS_OVERRUN: (
        TokenRunOverrunError,
        "a 0x7f run of absolutes runs past the values requested",
    ),
    _STATUS_RANGE: (
        TokenValueRangeError,
        "the running value left the uint16 range 0..65535",
    ),
}


def _raise_on_status(
    status: int, where: str, off: int, written: int, wanted: int
) -> None:
    """Raise the named error for a kernel status other than OK.

    An unknown status is a kernel bug and raises too, never passes.
    """
    if status == _STATUS_OK:
        return
    if status not in _STATUS_ERRORS:
        raise RuntimeError(f"{where}: token kernel returned unknown status {status}")
    cls, text = _STATUS_ERRORS[status]
    raise cls(f"{where}: {text} (byte {off}, value {written} of {wanted})")


def _check_input(data: np.ndarray, where: str) -> None:
    """Refuse any token array but a writable, C-contiguous, 1-D uint8 one.

    numba compiles one specialisation per array type. A read-only array (from
    ``np.frombuffer(bytes)``) or a strided one would silently compile and cache
    a second copy of the kernel (numba-decoder claim 11), and another dtype
    would decode the wrong values. Readers fill a pre-allocated array instead.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"{where}: token data must be a numpy array, got {type(data).__name__}"
        )
    problems = []
    if data.dtype != np.uint8:
        problems.append(f"dtype {data.dtype}, not uint8")
    if data.ndim != 1:
        problems.append(f"{data.ndim} dimensions, not 1-D")
    if not data.flags.c_contiguous:
        problems.append("not C-contiguous")
    if not data.flags.writeable:
        problems.append("not writeable")
    if problems:
        raise TypeError(f"{where}: token data is {'; '.join(problems)}")


def decode_pack1(
    data: np.ndarray, npix: int, n_frames: int, keep: Optional[range] = None
) -> tuple[np.ndarray, int]:
    """Decode the frames of an ``.im7`` pack-1 buffer.

    The stream cannot be seeked into, so every frame is decoded in order, and
    frames outside ``keep`` go to one scratch buffer.

    Parameters
    ----------
    data : np.ndarray
        The bytes after the 256-byte header, as a writable C-contiguous uint8
        array. It may run on past the last frame (the attribute records).
    npix : int
        Values per frame (size_x * size_y * size_z).
    n_frames : int
        Frames in the buffer (size_f).
    keep : range, optional
        Contiguous frames to return. Default all.

    Returns
    -------
    frames : np.ndarray
        uint16, shape (len(keep), npix).
    end : int
        Byte offset in ``data`` just after the last frame's tokens, where the
        attribute records begin.
    """
    _check_input(data, "pack-1 buffer")
    keep = range(n_frames) if keep is None else keep
    if keep.step != 1 or len(keep) == 0 or keep.start < 0 or keep.stop > n_frames:
        raise ValueError(
            f"pack-1 buffer: keep {keep} must be a non-empty contiguous range "
            f"inside the buffer's {n_frames} frames"
        )
    frames = np.empty((len(keep), npix), dtype=np.uint16)
    scratch = None
    off, prev = PACK1_PREAMBLE_BYTES, 0
    for fi in range(n_frames):
        if fi in keep:
            dst = frames[fi - keep.start]
        else:
            if scratch is None:
                scratch = np.empty(npix, dtype=np.uint16)
            dst = scratch
        off, prev, written, status = _decode_tokens(data, off, dst, prev)
        _raise_on_status(status, f"pack-1 frame {fi} of {n_frames}", off, written, npix)
    return frames, off


def decode_imx_words(data: np.ndarray, n_words: int) -> np.ndarray:
    """Decode one IMX-compressed ``.set`` entry to its stored uint16 words.

    The entry has no preamble and its running value starts at 0. The words are
    the packed payload the pixel decoder (e.g. mono-12p) reads next, not pixels.
    Every real entry's tokens end exactly at its last byte (container-unknowns
    C5, 6 of 6 streams), so trailing bytes are refused.

    Parameters
    ----------
    data : np.ndarray
        The entry's stored bytes, as a writable C-contiguous uint8 array.
    n_words : int
        uint16 words the entry decodes to (packed payload bytes / 2). The caller
        refuses a payload with an odd byte count before calling.

    Returns
    -------
    np.ndarray
        uint16, shape (n_words,).
    """
    _check_input(data, "IMX entry")
    words = np.empty(n_words, dtype=np.uint16)
    off, _prev, written, status = _decode_tokens(data, 0, words, 0)
    _raise_on_status(status, "IMX entry", off, written, n_words)
    if off != data.shape[0]:
        raise TrailingTokenBytesError(
            f"IMX entry: tokens end at byte {off}, entry holds {data.shape[0]}"
        )
    return words
