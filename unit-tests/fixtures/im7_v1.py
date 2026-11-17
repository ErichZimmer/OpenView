"""Synthetic DaVis .im7 version-1 pack-20 files, and the committed lvpyio fixture.

Layout, measured byte for byte on ``G:\\MyProjects4\\gF_Ramp_experiment\\60deg_3000\\
SubOverTimeMin_sl=all\\B00003.im7`` (D3, 2026-11-17), with the frame-offset table
checked on all 10 real version-1 files:

  header        256 B Image_Header_7, version 1, pack 20. Bytes 30..43 are the
                constant every real pack-20 file carries (version 0 and 1);
                lvpyio refuses a pack-20 header without them.
  pixels        per frame: int64 compressed size, then one raw LZ4 block (no
                frame header) holding that frame's uint16 pixels.
  frame table   size_f int64, each the file offset of that frame's size prefix.
  attributes    the buffer's extended-header records, ended by IEH_END (int32
                type 0, int32 size 0).
  masks         per frame: int32 20 (LZ4), int64 compressed size, one LZ4 block
                of an H x W uint8 mask, 1 = valid (lvpyio ``frame.masks[0]``).
  mask table    size_f int64, the file offset of each mask block.
  footer        256 B: int64 [0] start of the attribute records, [1] start of
                the frame table, [7] start of the mask table, every other byte 0.

:func:`build_im7_v1_pack20` writes that layout for the tests. Run as a script,
this module writes ``im7_v1_pack20/three_frames.im7`` and lvpyio's decode of it,
``three_frames.lvpyio.npy``, after checking that lvpyio returns the designed
pixels, the designed intensity scale (2 x + 1) on every frame and all-ones
masks. The script needs lvpyio (Windows workstation):

  python unit-tests/fixtures/im7_v1.py [--out <folder>]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import lz4.block
import numpy as np

HEADER_SIZE = 256
FOOTER_SIZE = 256
VERSION_PER_FRAME_LZ4 = 1
PACK_LZ4 = 20
BUFFER_FORMAT_WORD = -4
# extra_flags of every real version-1 file (drive-survey distinct_values.md).
EXTRA_FLAGS_REAL = 5
# Header bytes 30..43 of every real pack-20 file checked (version 0 and 1).
HEADER_PACK20_BYTES_30_44 = bytes.fromhex("1400000000000000" "0a000200" "0000")
IEH_END = 0
# Real three-string form: "slope offset", unit, description.
SCALE_RECORDS = (
    (1, b"1 0\x00pixel\x00x\x00"),
    (2, b"1 0\x00pixel\x00y\x00"),
    (3, b"1 0\x00\x00\x00"),
    (4, b"2 1\x00counts\x00\x00"),
)
SLOPE_I, OFFSET_I = 2.0, 1.0


def _header(size_x: int, size_y: int, size_f: int) -> bytes:
    """Image_Header_7: version 1, pack 20, WORD, size_z 1."""
    h = struct.pack(
        "<hhhh iiii hhh",
        VERSION_PER_FRAME_LZ4,
        PACK_LZ4,
        BUFFER_FORMAT_WORD,
        0,
        size_x,
        size_y,
        1,
        size_f,
        0,
        1,
        EXTRA_FLAGS_REAL,
    )
    h += HEADER_PACK20_BYTES_30_44
    return h + b"\x00" * (HEADER_SIZE - len(h))


def _lz4_block(raw: bytes) -> bytes:
    """One raw LZ4 block, no size header."""
    return lz4.block.compress(raw, store_size=False)


def build_im7_v1_pack20(
    frames: np.ndarray, masks: np.ndarray | None = None
) -> tuple[bytes, dict]:
    """A version-1 pack-20 .im7 in the measured layout.

    Parameters
    ----------
    frames : np.ndarray
        uint16, shape (n_frames, H, W).
    masks : np.ndarray, optional
        uint8, same shape, 1 = valid. Default all ones, as in every real file.

    Returns
    -------
    data : bytes
        The whole file.
    layout : dict
        Byte offsets: ``frame_starts``, ``table_at``, ``attr_at``,
        ``mask_starts``, ``mask_table_at``, ``footer_at``.
    """
    n_frames, height, width = frames.shape
    if masks is None:
        masks = np.ones(frames.shape, np.uint8)
    out = bytearray(_header(width, height, n_frames))
    frame_starts = []
    for frame in frames:
        frame_starts.append(len(out))
        block = _lz4_block(frame.astype("<u2").tobytes())
        out += struct.pack("<q", len(block)) + block
    table_at = len(out)
    out += struct.pack(f"<{n_frames}q", *frame_starts)
    attr_at = len(out)
    for ieh_type, text in SCALE_RECORDS:
        out += struct.pack("<ii", ieh_type, len(text)) + text
    out += struct.pack("<ii", IEH_END, 0)
    mask_starts = []
    for mask in masks:
        mask_starts.append(len(out))
        block = _lz4_block(mask.astype(np.uint8).tobytes())
        out += struct.pack("<iq", PACK_LZ4, len(block)) + block
    mask_table_at = len(out)
    out += struct.pack(f"<{n_frames}q", *mask_starts)
    footer_at = len(out)
    footer = [0] * 8
    footer[0], footer[1], footer[7] = attr_at, table_at, mask_table_at
    out += struct.pack("<8q", *footer) + b"\x00" * (FOOTER_SIZE - 64)
    layout = {
        "frame_starts": frame_starts,
        "table_at": table_at,
        "attr_at": attr_at,
        "mask_starts": mask_starts,
        "mask_table_at": mask_table_at,
        "footer_at": footer_at,
    }
    return bytes(out), layout


def three_frames() -> np.ndarray:
    """Three distinct 8x4 frames: a ramp with a repeat run (so LZ4 emits a match)."""
    base = np.arange(32, dtype=np.uint16).reshape(4, 8)
    base[2:] = base[0]  # rows 2-3 repeat row 0: an LZ4 back-reference
    return np.stack([base + 1000 * (k + 1) for k in range(3)]).astype(np.uint16)


def main() -> None:
    import lvpyio as lv

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "im7_v1_pack20")
    )
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    designed = three_frames()
    data, _ = build_im7_v1_pack20(designed)
    path = out / "three_frames.im7"
    path.write_bytes(data)
    buf = lv.read_buffer(str(path))
    ref = np.stack([np.asarray(f.components["PIXEL"].planes[0]) for f in buf.frames])
    if ref.dtype != np.uint16 or not np.array_equal(ref, designed):
        raise SystemExit(
            f"lvpyio does not return the designed pixels (dtype {ref.dtype}, "
            f"shape {ref.shape}, first {ref.ravel()[:8].tolist()})"
        )
    for k, frame in enumerate(buf.frames):
        scale = frame.scales.i
        if (scale.slope, scale.offset) != (SLOPE_I, OFFSET_I):
            raise SystemExit(
                f"frame {k}: lvpyio reads intensity scale {scale}, designed "
                f"slope {SLOPE_I} offset {OFFSET_I}"
            )
        masks = [np.asarray(m) for m in frame.masks]
        if len(masks) != 1 or not np.all(masks[0] == 1):
            raise SystemExit(f"frame {k}: lvpyio masks are not one all-ones plane")
    np.save(out / "three_frames.lvpyio.npy", ref)
    print(
        f"three_frames: {path.stat().st_size} B, {ref.shape}, lvpyio equals the "
        f"design, intensity scale {SLOPE_I:g} x + {OFFSET_I:g} on every frame"
    )


if __name__ == "__main__":
    main()
