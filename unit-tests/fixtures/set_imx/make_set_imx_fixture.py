"""Cut one real IMX-compressed ``.set`` entry down to a committable fixture.

The source is a DaVis 11.4 Live Image: one mono-12p stream, 5312x4600, one entry
stored with the "IMX" codec (36.8 MB), no stream transformer. An IMX entry is a
token stream over the little-endian uint16 words of the packed payload
(container-unknowns REPORT C5). It cannot be seeked into, but a cut that starts
on a token holding absolute values and ends on a token boundary is itself a
complete IMX entry: the running value starts at 0 and the first token overwrites
it.

This script walks the entry's tokens, takes the stored bytes of whole rows
``--rows`` (the default band, rows 1239..1345, is one of the 7 places in the
entry where a token boundary falls on a row boundary, and holds every token kind
the entry uses: deltas, single absolutes and 0x7f runs), and writes a minimal
``.set`` around them:

  fixture.set                 the source's, first comment line made generic
  fixture/StreamSet.xml       the source's FrameReader for Frame0, nothing else
  fixture/Frame0-0.ims        the source's 1024-byte header, height set to the
                              band, one entry covering the cut bytes
  fixture/Frame0-1.ims        the cut bytes, verbatim
  fixture/Frame0-decoder.xml  the source's, verbatim

lvpyio 1.3.1 then reads the cut ``.set``. Its pixels must equal the same rows of
its own read of the full source recording, otherwise the script stops. They are
saved as ``fixture.lvpyio.npy`` (uint16, (rows, width)), the
oracle the tests compare with.

Needs lvpyio and the source recording (Windows workstation). Usage:
  python make_set_imx_fixture.py [--source <.set>] [--rows 1239 1346] [--out <folder>]
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import lvpyio as lv
import numpy as np

HEADER_BYTES = 1024
ENTRY = struct.Struct("<Iqq")
WORD_HEIGHT = 4
TOKEN_ABSOLUTE = 0x80
TOKEN_RUN = 0x7F
TOKEN_NIBBLES = 0x81
MONO12P_BYTES_PER_2PX = 3
FRAME_READER_MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<StreamCfg>\n    <Version>2</Version>\n'
    "    <MinDavisVersion>10.2.0</MinDavisVersion>\n"
    '    <ReaderInfo Name="Images_16bit_8Bit" Type="Core.Set.Recording.FrameReader" '
    'FilePrefix="Frame0" FormatDescription="LaVision image stream format">\n'
    '        <ContentPurpose IsAssociatedToFrames="true" '
    'IsAssociatedToEntireImage="false" IsAssociatedToAllImages="false" '
    'StartFrame="0" EndFrame="0"/>\n    </ReaderInfo>\n</StreamCfg>\n'
)


def token_starts(data: bytes) -> dict[int, tuple[int, int]]:
    """Map word index -> (byte offset, token byte) for every token in ``data``."""
    starts = {}
    i = words = 0
    while i < len(data):
        b = data[i]
        starts[words] = (i, b)
        if b == TOKEN_NIBBLES:
            raise SystemExit(
                "nibble run in the source entry: the cut walk does not handle it"
            )
        if b == TOKEN_ABSOLUTE:
            i, words = i + 3, words + 1
        elif b == TOKEN_RUN:
            n = data[i + 1]
            i, words = i + 2 + 2 * n, words + n
        else:
            i, words = i + 1, words + 1
    if i != len(data):
        raise SystemExit(f"tokens end at byte {i}, entry holds {len(data)}")
    starts[words] = (i, -1)
    return starts


def lvpyio_pixels(set_path: Path, height: int, width: int) -> np.ndarray:
    """Entry 0, stream 0 as lvpyio reads it, cropped to (height, width).

    lvpyio pads the height up to a multiple of 16 with zero rows.
    """
    s = lv.read_set(str(set_path))
    plane = np.asarray(s[0].frames[0].components["PIXEL"].planes[0])
    return plane[:height, :width]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source",
        default=r"D:\planar calibration\Live images"
        r"\Live Image_Date=261014_Time=145509.set",
    )
    ap.add_argument("--rows", nargs=2, type=int, default=[1239, 1346])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    source = Path(args.source)
    src_dir = source.with_suffix("")
    index = (src_dir / "Frame0-0.ims").read_bytes()
    if len(index) != HEADER_BYTES + ENTRY.size or index[24:27] != b"IMX":
        raise SystemExit(f"{src_dir}: expected one IMX entry")
    width, height = struct.unpack_from("<ii", index, 12)
    _, offset, size = ENTRY.unpack_from(index, HEADER_BYTES)
    with open(src_dir / "Frame0-1.ims", "rb") as f:
        f.seek(offset)
        entry = f.read(size)

    row_words = width * MONO12P_BYTES_PER_2PX // 4
    r0, r1 = args.rows
    starts = token_starts(entry)
    w0, w1 = r0 * row_words, r1 * row_words
    if w0 not in starts or w1 not in starts:
        raise SystemExit(f"rows {r0}..{r1} do not start and end on a token boundary")
    (b0, first), (b1, _) = starts[w0], starts[w1]
    if first not in (TOKEN_ABSOLUTE, TOKEN_RUN):
        raise SystemExit(f"row {r0} opens with a delta, so the cut needs prior state")
    cut = entry[b0:b1]
    rows = r1 - r0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_file = out / "fixture.set"
    fix_dir = out / "fixture"
    if fix_dir.exists():
        shutil.rmtree(fix_dir)
    fix_dir.mkdir()
    # lvpyio opens no buffer for a stub .set, so the source's is kept, with its
    # first comment line (which names the folder DaVis wrote it to) made generic.
    set_lines = source.read_bytes().split(b"\n")
    if not set_lines[0].startswith(b"// set file <"):
        raise SystemExit(f"{source.name}: unexpected first line {set_lines[0][:40]!r}")
    set_lines[0] = b"// set file <fixture.set> created by DaVis"
    set_file.write_bytes(b"\n".join(set_lines))
    header = bytearray(index[:HEADER_BYTES])
    struct.pack_into("<i", header, 4 * WORD_HEIGHT, rows)
    (fix_dir / "Frame0-0.ims").write_bytes(bytes(header) + ENTRY.pack(1, 0, len(cut)))
    (fix_dir / "Frame0-1.ims").write_bytes(cut)
    shutil.copyfile(src_dir / "Frame0-decoder.xml", fix_dir / "Frame0-decoder.xml")
    (fix_dir / "StreamSet.xml").write_text(FRAME_READER_MANIFEST, encoding="utf-8")

    full = lvpyio_pixels(source, height, width)[r0:r1]
    ref = lvpyio_pixels(set_file, rows, width)
    if ref.dtype != np.uint16 or not np.array_equal(ref, full):
        raise SystemExit(
            "lvpyio's read of the cut differs from its read of the source rows"
        )
    np.save(out / "fixture.lvpyio.npy", ref)
    kinds = {"delta": 0, "absolute": 0, "run": 0}
    for w, (_, b) in starts.items():
        if w0 <= w < w1:
            kinds[{TOKEN_ABSOLUTE: "absolute", TOKEN_RUN: "run"}.get(b, "delta")] += 1
    print(
        f"rows {r0}..{r1 - 1} of {source.name}: {len(cut)} B stored, "
        f"{ref.shape} pixels, tokens {kinds}; lvpyio cut == lvpyio source rows"
    )


if __name__ == "__main__":
    main()
