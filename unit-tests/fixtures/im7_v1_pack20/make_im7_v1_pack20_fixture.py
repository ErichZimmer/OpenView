"""Write the synthetic .im7 version-1 pack-20 fixture and record lvpyio's decode of it.

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
  masks         per frame: int32 20, int64 compressed size, one LZ4 block of an
                H x W uint8 mask (lvpyio ``frame.masks[0]``; all 1 in the real
                files).
  mask table    size_f int64, the file offset of each mask block.
  footer        256 B: int64 [0] start of the attribute records, [1] start of
                the frame table, [7] start of the mask table, the rest zero.

A version-0 pack-20 file holds one size and one block for the whole buffer, no
frame table, and no masks.

The fixture ``three_frames`` is 8x4 pixels, three frames, WORD. Each frame has
known values, so lvpyio is cross-checked against the design. The scale records
are SCALE_X..SCALE_I in the real three-string form, with a non-identity
intensity scale (slope 2, offset 1): a camera read that does not end exactly at
the records would either fail the attribute walk or apply no scale. This script
checks that lvpyio returns the designed pixels, the designed scale and all-ones
masks, then saves lvpyio's frames as ``three_frames.lvpyio.npy`` (uint16,
(3, 4, 8)).

Needs lvpyio (Windows workstation) and lz4. Usage:
  python make_im7_v1_pack20_fixture.py [--out <folder>]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import lvpyio as lv
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
SCALE_RECORDS = (
    (1, b"1 0\x00pixel\x00x\x00"),
    (2, b"1 0\x00pixel\x00y\x00"),
    (3, b"1 0\x00\x00\x00"),
    (4, b"2 1\x00counts\x00\x00"),
)
SLOPE_I, OFFSET_I = 2.0, 1.0
WIDTH, HEIGHT, N_FRAMES = 8, 4, 3


def header(size_x: int, size_y: int, size_f: int) -> bytes:
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


def lz4_block(raw: bytes) -> bytes:
    """One raw LZ4 block, no size header."""
    return lz4.block.compress(raw, store_size=False)


def designed_frames() -> np.ndarray:
    """Three distinct frames: a ramp with a repeat run (so LZ4 emits a match)."""
    base = np.arange(WIDTH * HEIGHT, dtype=np.uint16).reshape(HEIGHT, WIDTH)
    base[2:] = base[0]  # rows 2-3 repeat row 0: an LZ4 back-reference
    return np.stack([base + 1000 * (k + 1) for k in range(N_FRAMES)]).astype(np.uint16)


def three_frames() -> tuple[bytes, np.ndarray]:
    """The fixture bytes and the designed pixels."""
    frames = designed_frames()
    out = bytearray(header(WIDTH, HEIGHT, N_FRAMES))
    starts = []
    for frame in frames:
        starts.append(len(out))
        block = lz4_block(frame.astype("<u2").tobytes())
        out += struct.pack("<q", len(block)) + block
    table_at = len(out)
    out += struct.pack(f"<{N_FRAMES}q", *starts)
    attr_at = len(out)
    for ieh_type, text in SCALE_RECORDS:
        out += struct.pack("<ii", ieh_type, len(text)) + text
    out += struct.pack("<ii", IEH_END, 0)
    mask_starts = []
    for _ in range(N_FRAMES):
        mask_starts.append(len(out))
        block = lz4_block(np.ones(WIDTH * HEIGHT, np.uint8).tobytes())
        out += struct.pack("<iq", PACK_LZ4, len(block)) + block
    mask_table_at = len(out)
    out += struct.pack(f"<{N_FRAMES}q", *mask_starts)
    footer = [0] * 8
    footer[0], footer[1], footer[7] = attr_at, table_at, mask_table_at
    out += struct.pack("<8q", *footer) + b"\x00" * (FOOTER_SIZE - 64)
    return bytes(out), frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    data, designed = three_frames()
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
