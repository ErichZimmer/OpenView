"""Regenerate the committed DaVis ``.attr`` fixtures from two real recordings.

Each fixture is a real recording cut down to a few images:

- every ``MetaData{N}.attr`` keeps the header, the chosen image entries and the
  trailing block as the recording's own bytes, sliced with the file's offset
  table only (no parser involved), plus a new offset table and count;
- StreamSet.xml is the recording's own, with its ``<Transformer>`` nodes removed
  (their correction images are tens of MB and are not what these fixtures test);
- the ``.set`` file is the recording's own, with the first comment line (which
  names the acquisition PC's folder) made generic;
- the ``.scales`` files are the recording's own;
- the pixel streams are tiny ``raw-16-bit`` stand-ins. Streams of equal real
  shape get equal tiny shapes, so the shape rule sees the same equalities as on
  the real recording. Entry ``e`` of frame stream ``k`` is filled with
  ``100 * k + e``.

lvpyio (LaVision's reader) then reads every cut, and its buffer and frame
attributes are saved to ``lvpyio_attributes.json`` beside the fixtures, as the
ground truth the tests compare the production parser against.

Sources (defaults): the NTNU PIV plus PLIF sample (DaVis 10.2.1, 4 cameras, all
150 images labelled) cut to image 1, and ``Recording_Date=261030_Time=145749``
(DaVis 11.4.0.91, A and B in one stream, camera keys on images 1..501 of 900,
``SetIsSuspicious = 1``) cut to images 501 and 502, the last labelled image and
the first unlabelled one.

Usage (workstation venv, lvpyio 1.3.1)::

    python unit-tests/fixtures/davis_attr/make_davis_attr_fixture.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # unit-tests/, for fixtures.lavision

from fixtures.lavision import ims_index  # noqa: E402

# .attr framing (davis_attr module docstring): 1024-byte header, entries, trailing
# block, then count x u64 offsets and a u32 count.
_ATTR_HEADER = 1024
_U64 = 8
_U32 = 4

# .ims index: int32 header words 2 (frames per image), 3 (width), 4 (height).
_IMS_WORD = struct.Struct("<i")
_WORD_SPAN, _WORD_WIDTH, _WORD_HEIGHT = 2, 3, 4

_TRANSFORMER_NODE = re.compile(rb"[ \t]*<Transformer\b.*?</Transformer>\r?\n?", re.S)
_FRAME_READER = re.compile(
    r'<ReaderInfo[^>]*Type="Core\.Set\.Recording\.FrameReader"[^>]*'
    r'FilePrefix="(?P<prefix>[^"]+)"[^>]*>\s*<ContentPurpose[^>]*'
    r'StartFrame="(?P<start>\d+)"\s+EndFrame="(?P<end>\d+)"'
)
_SET_COMMENT = re.compile(rb"\A// set file <[^>\r\n]*>")
GENERIC_SET_COMMENT = b"// set file <recording.set>"

NTNU = "REF_background_check_1006"
REC_145749 = "Recording_Date=261030_Time=145749"
DEFAULT_SOURCES = {
    NTNU: (
        Path(
            "D:/PIVTOOLS_data/calibration_check/srikar_ntnu_2026-11/PIV_sample/"
            "REF_background_check_1006.set"
        ),
        [0],
    ),
    REC_145749: (
        Path("D:/planar calibration/Recording_Date=261030_Time=145749.set"),
        [500, 501],
    ),
}


def cut_attr(data: bytes, images: Sequence[int]) -> bytes:
    """The ``.attr`` bytes holding only ``images`` (0-based), sliced raw."""
    (count,) = struct.unpack_from("<I", data, len(data) - _U32)
    table = len(data) - _U32 - _U64 * count
    offsets = list(struct.unpack_from(f"<{count}Q", data, table)) + [table]
    n_images = count - 1
    out = bytearray(data[:_ATTR_HEADER])
    new_offsets = []
    for i in list(images) + [n_images]:  # the trailing block is slot n_images
        if not 0 <= i <= n_images:
            raise ValueError(f"image {i} is outside 0..{n_images - 1}")
        new_offsets.append(len(out))
        out += data[offsets[i] : offsets[i + 1]]
    out += struct.pack(f"<{len(new_offsets)}Q", *new_offsets)
    out += struct.pack("<I", len(new_offsets))
    return bytes(out)


def _own_files(folder: Path, pattern: str) -> List[Path]:
    """Files matching ``pattern`` in ``folder``, macOS ``._`` copies left out."""
    return [p for p in folder.glob(pattern) if not p.name.startswith(".")]


def _stored_shape(index_path: Path) -> Tuple[int, int, int]:
    """(span, width, height) from a real ``-0.ims`` header."""
    head = index_path.read_bytes()[:20]
    return tuple(
        _IMS_WORD.unpack_from(head, 4 * w)[0]
        for w in (_WORD_SPAN, _WORD_WIDTH, _WORD_HEIGHT)
    )


def make_fixture(name: str, source: Path, images: Sequence[int], out: Path) -> Path:
    """Write the cut of ``source`` as ``out/<name>.set`` plus its folder."""
    src_dir = source.with_suffix("")
    dst_set = out / f"{name}.set"
    dst_dir = out / name
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    raw_set = source.read_bytes()
    if not _SET_COMMENT.match(raw_set):
        raise ValueError(f"{source} does not start with a '// set file <...>' line")
    dst_set.write_bytes(_SET_COMMENT.sub(GENERIC_SET_COMMENT, raw_set, count=1))

    manifest = (src_dir / "StreamSet.xml").read_bytes()
    (dst_dir / "StreamSet.xml").write_bytes(_TRANSFORMER_NODE.sub(b"", manifest))

    # Dot-files are macOS AppleDouble copies (``._MetaData0.attr``), not DaVis's.
    for attr in sorted(_own_files(src_dir, "*.attr")):
        (dst_dir / attr.name).write_bytes(cut_attr(attr.read_bytes(), images))
    for scales in sorted(_own_files(src_dir, "*.scales")):
        shutil.copyfile(scales, dst_dir / scales.name)

    tiny: Dict[Tuple[int, int], Tuple[int, int]] = {}  # real (w, h) -> tiny (w, h)
    for m in _FRAME_READER.finditer(manifest.decode("utf-8")):
        prefix, start = m["prefix"], int(m["start"])
        span, width, height = _stored_shape(src_dir / f"{prefix}-0.ims")
        if span != int(m["end"]) - start + 1:
            raise ValueError(f"{prefix}: header span {span} disagrees with manifest")
        w, h = tiny.setdefault((width, height), (5 + len(tiny), 3 + len(tiny)))
        payloads: List[bytes] = []
        for e in range(len(images)):
            for f in range(span):
                value = 100 * (start + f) + e
                payloads.append(np.full((h, w), value, dtype="<u2").tobytes())
        index = ims_index(w, h, [len(p) for p in payloads], words={_WORD_SPAN: span})
        (dst_dir / f"{prefix}-0.ims").write_bytes(index)
        (dst_dir / f"{prefix}-1.ims").write_bytes(b"".join(payloads))
        (dst_dir / f"{prefix}-decoder.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<FrameDecoder>\n<id>raw-16-bit</id>\n</FrameDecoder>\n"
        )
    return dst_set


def _jsonable(value):
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "hex": arr.tobytes().hex(),
    }


def lvpyio_attributes(set_path: Path) -> List[dict]:
    """lvpyio's buffer and frame attributes for every image of ``set_path``."""
    import lvpyio as lv

    s = lv.read_set(str(set_path))
    try:
        rows = []
        for i in range(len(s)):
            buf = s[i]
            rows.append(
                {
                    "buffer": {k: _jsonable(v) for k, v in buf.attributes.items()},
                    "frames": [
                        {k: _jsonable(v) for k, v in fr.attributes.items()}
                        for fr in buf.frames
                    ],
                }
            )
        return rows
    finally:
        s.close()


def main(argv: Sequence[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--ntnu", type=Path, default=DEFAULT_SOURCES[NTNU][0])
    p.add_argument("--rec145749", type=Path, default=DEFAULT_SOURCES[REC_145749][0])
    a = p.parse_args(argv)
    sources = {NTNU: a.ntnu, REC_145749: a.rec145749}

    import lvpyio as lv

    oracle = {"lvpyio_version": lv.__version__, "recordings": {}}
    for name, (_default, images) in DEFAULT_SOURCES.items():
        source = sources[name]
        set_path = make_fixture(name, source, images, a.out)
        oracle["recordings"][name] = {
            "source_images": [i + 1 for i in images],
            "images": lvpyio_attributes(set_path),
        }
        print(f"wrote {set_path} (images {[i + 1 for i in images]} of {source.name})")
    target = a.out / "lvpyio_attributes.json"
    target.write_text(json.dumps(oracle, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
