"""Write the synthetic .im7 pack-1 fixtures and record lvpyio's decode of each.

Each fixture is a bare 256-byte Image_Header_7 (pack 1, WORD) followed by a
hand-built token stream and EOF, which lvpyio 1.3.1 and the production reader
both accept. The tokens are designed to give known values. This script checks
that lvpyio returns exactly those values, then saves lvpyio's frames as
``<name>.lvpyio.npy`` (uint16, (n_frames, H, W)). The tests compare the
production reader with those arrays, so the oracle is lvpyio, cross-checked by
the design.

Fixtures:
  high_nibble_8   16x4, one frame. The nibble run 0x12 0x88 reads +1, +2, then a
                  HIGH nibble 8 as -8, and the LOW nibble 8 ends the run
                  (container-unknowns C6).
  carry_3frames   8x2, three frames. The preamble precedes frame 0 only, and
                  frames 1 and 2 open with a delta and a nibble run, so their
                  values depend on the running value carried from the frame
                  before. Every token kind appears.

Needs lvpyio (Windows workstation). Usage:
  python make_im7_pack1_fixtures.py [--out <folder>]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import lvpyio as lv
import numpy as np

PREAMBLE = b"\x01\x01\x00"
HEADER_SIZE = 256
PACK_RLE = 1
BUFFER_FORMAT_WORD = -4


def header(size_x: int, size_y: int, size_f: int) -> bytes:
    """Bare Image_Header_7: version 0, pack 1, WORD, size_z 1."""
    h = struct.pack(
        "<hhhh iiii hhh",
        0,
        PACK_RLE,
        BUFFER_FORMAT_WORD,
        0,
        size_x,
        size_y,
        1,
        size_f,
        0,
        1,
        0,
    )
    return h + b"\x00" * (HEADER_SIZE - len(h))


def high_nibble_8() -> tuple[bytes, np.ndarray]:
    """0x80 5000 | 0x81 0x12 0x88 (+1 +2 -8 end) | +10 for the other 59 pixels."""
    w, h = 16, 4
    tokens = bytearray(b"\x80" + struct.pack("<H", 5000) + b"\x81\x12\x88")
    values = [5000, 5001, 5003, 4995]
    while len(values) < w * h:
        tokens.append(10)
        values.append(values[-1] + 10)
    return header(w, h, 1) + PREAMBLE + bytes(tokens), np.array(
        values, np.uint16
    ).reshape(1, h, w)


def carry_3frames() -> tuple[bytes, np.ndarray]:
    """Three 8x2 frames; see the module docstring."""
    w, h = 8, 2
    frame0 = (
        b"\x80\xe8\x03"  # 1000
        + b"\x81\x12\x3f\x28"  # +1 +2 +3 -1 +2, low nibble 8 ends
        + b"\x7f\x02\x10\x27\x11\x27"  # 10000 10001
        + bytes(
            [0x05, 0xFB, 0x0A, 0xF6, 0x7E, 0x82, 0x01, 0xFF]
        )  # +5 -5 +10 -10 +126 -126 +1 -1
    )
    values0 = [
        1000,
        1001,
        1003,
        1006,
        1005,
        1007,
        10000,
        10001,
        10006,
        10001,
        10011,
        10001,
        10127,
        10001,
        10002,
        10001,
    ]
    frame1 = (
        b"\x03"  # +3 on the carried 10001
        + b"\x80\x34\x12"  # 4660
        + b"\x81\x77\x78"  # +7 +7 +7, low nibble 8 ends
        + b"\x7f\x03\x00\x01\x00\x02\x00\x03"  # 256 512 768
        + bytes(
            [0x01, 0x02, 0x03, 0x04, 0xFC, 0xFD, 0xFE, 0xFF]
        )  # +1 +2 +3 +4 -4 -3 -2 -1
    )
    values1 = [
        10004,
        4660,
        4667,
        4674,
        4681,
        256,
        512,
        768,
        769,
        771,
        774,
        778,
        774,
        771,
        769,
        768,
    ]
    abs2 = list(range(2000, 2011))
    frame2 = (
        b"\x81\xf1\x28"  # -1 +1 +2, low nibble 8 ends, on the carried 768
        + b"\x64\x9c"  # +100 -100
        + b"\x7f\x0b"
        + b"".join(struct.pack("<H", v) for v in abs2)
    )
    values2 = [767, 768, 770, 870, 770] + abs2
    data = header(w, h, 3) + PREAMBLE + frame0 + frame1 + frame2
    return data, np.array([values0, values1, values2], np.uint16).reshape(3, h, w)


def lvpyio_frames(path: Path) -> np.ndarray:
    """Every frame's PIXEL plane as lvpyio reads it, (n_frames, H, W) uint16."""
    buf = lv.read_buffer(str(path))
    return np.stack([np.asarray(f.components["PIXEL"].planes[0]) for f in buf.frames])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("high_nibble_8", high_nibble_8),
        ("carry_3frames", carry_3frames),
    ):
        data, designed = build()
        path = out / f"{name}.im7"
        path.write_bytes(data)
        ref = lvpyio_frames(path)
        if ref.dtype != np.uint16 or not np.array_equal(ref, designed):
            raise SystemExit(
                f"{name}: lvpyio does not return the designed values "
                f"(dtype {ref.dtype}, first {ref.ravel()[:8].tolist()}, "
                f"designed {designed.ravel()[:8].tolist()})"
            )
        np.save(out / f"{name}.lvpyio.npy", ref)
        print(f"{name}: {path.stat().st_size} B, {ref.shape}, lvpyio equals the design")


if __name__ == "__main__":
    main()
