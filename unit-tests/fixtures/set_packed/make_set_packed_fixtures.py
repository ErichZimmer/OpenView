"""lvpyio's decode of a synthetic payload per packed pixel encoding (ticket D8).

For each of ``mono-10pmsb``, ``mono-12pmsb`` and ``mono-12packed`` this writes a
one-stream, one-entry recording with ``fixtures/lavision.py::build_set`` (the
container the tests build), replaces its stub ``.set`` with the DaVis set text
lvpyio needs to open a recording (01-lavision-readers REPORT A2), and reads it with
lvpyio 1.3.1, LaVision's own reader. The payload and lvpyio's pixels are saved:

  {id}.payload.npy   uint8, the entry's stored bytes
  {id}.lvpyio.npy    uint16 (height, width), lvpyio's PIXEL plane, top-left block

so the test suite carries LaVision's answer on machines without lvpyio.

Payload, 64 x 48 pixels: the zero group, the all-ones group, every single bit set
in a zero group and every single bit cleared in an all-ones group, then seeded
random bytes. The decoders only shift, mask and OR fixed bits, so each output bit
is 0 or a copy of one input bit; the zero and single-bit groups fix that map
(container-unknowns REPORT E3). The random bytes cover every value besides.

Checks before anything is saved, each a stop on failure:
  - lvpyio's frame is uint16 and its pad outside the top-left block is zero;
  - lvpyio's three 12-bit and two 10-bit readings of the same bytes all differ,
    so the payload discriminates the layouts.

Needs lvpyio (Windows workstation). Usage, from the python-PIVTOOLs root:
  env/Scripts/python.exe unit-tests/fixtures/set_packed/make_set_packed_fixtures.py
      [--out <folder>] [--work <folder>] [--seed N]
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import tempfile
from pathlib import Path

import lvpyio as lv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # unit-tests, for fixtures.lavision
sys.path.insert(0, str(HERE.parents[2]))  # repo root, for pivtools_core

from fixtures.lavision import build_set  # noqa: E402

WIDTH, HEIGHT = 64, 48
# Bytes per group and bits per pixel of each id.
GROUPS = {"mono-10pmsb": (5, 10), "mono-12pmsb": (3, 12), "mono-12packed": (3, 12)}
# The extra ids lvpyio decodes the same bytes as, to show the payload discriminates.
CONTROLS = {
    10: ("mono-10p", "mono-10pmsb"),
    12: ("mono-12p", "mono-12pmsb", "mono-12packed"),
}
# A DaVis recording's .set text with no path or date of a real recording. lvpyio
# opens no buffer unless SetIdentifier is "StreamSet" (REPORT A2).
DAVIS_SET_TEXT = (
    "// set file <synthetic.set> created by DaVis\r\n\r\n#GROUP Sets\r\n"
    'SetType = 4352;\r\nSetGroups = "";\r\nSetTime = "";\r\nSetComments = "";\r\n'
    'SetStart = 1;\r\nSetInc = 1;\r\nSetSourceSet = "";\r\nSetViewCallback = "";\r\n'
    'SetLoadCallback = "";\r\nSetIdentifier = "StreamSet";\r\n'
)
DEFAULT_SEED = 20261117


def payload(group_bytes: int, bits: int, rng: np.random.Generator) -> bytes:
    """Edge groups then random bytes, exactly one WIDTH x HEIGHT frame."""
    n_bytes = WIDTH * HEIGHT * bits // 8
    groups = [bytes(group_bytes), b"\xff" * group_bytes]
    for pos, bit in itertools.product(range(group_bytes), range(8)):
        g = bytearray(group_bytes)
        g[pos] = 1 << bit
        groups.append(bytes(g))
        g = bytearray(b"\xff" * group_bytes)
        g[pos] ^= 1 << bit
        groups.append(bytes(g))
    edges = b"".join(groups)
    rest = rng.integers(0, 256, n_bytes - len(edges), dtype=np.uint8).tobytes()
    return edges + rest


def lvpyio_read(work: Path, decoder: str, raw: bytes) -> np.ndarray:
    """lvpyio's top-left block of a build_set container labelled ``decoder``."""
    name = f"packed_{decoder}"
    for stale in (work / name, work / f"{name}.set"):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    set_file = build_set(work, decoder, [[raw]], width=WIDTH, height=HEIGHT, name=name)
    set_file.write_text(DAVIS_SET_TEXT, newline="")
    recording = lv.read_set(str(set_file))
    try:
        if len(recording) != 1:
            raise SystemExit(
                f"{decoder}: lvpyio opened {len(recording)} buffers, not 1"
            )
        plane = np.asarray(recording[0].frames[0].components["PIXEL"].planes[0])
    finally:
        recording.close()
    block = plane[:HEIGHT, :WIDTH]
    if plane.dtype != np.uint16 or block.shape != (HEIGHT, WIDTH):
        raise SystemExit(f"{decoder}: lvpyio gave {plane.shape} {plane.dtype}")
    if np.count_nonzero(plane) != np.count_nonzero(block):
        raise SystemExit(f"{decoder}: lvpyio's pad outside the frame is not zero")
    return block.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE), help="folder for the .npy files")
    ap.add_argument("--work", default=None, help="scratch folder (default: a temp dir)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.work) if args.work else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        results = {}
        for decoder, (group_bytes, bits) in GROUPS.items():
            raw = payload(group_bytes, bits, rng)
            controls = {c: lvpyio_read(work, c, raw) for c in CONTROLS[bits]}
            for a, b in itertools.combinations(controls, 2):
                if np.array_equal(controls[a], controls[b]):
                    raise SystemExit(f"payload for {decoder}: lvpyio reads {a} == {b}")
            results[decoder] = (raw, controls[decoder])

    for decoder, (raw, ref) in results.items():
        np.save(out / f"{decoder}.payload.npy", np.frombuffer(raw, np.uint8))
        np.save(out / f"{decoder}.lvpyio.npy", ref)
        print(
            f"{decoder}: {len(raw)} B -> lvpyio {ref.shape} {ref.dtype}, "
            f"min {ref.min()} max {ref.max()}; differs from the other "
            f"{len(CONTROLS[GROUPS[decoder][1]]) - 1} ids of its depth"
        )


if __name__ == "__main__":
    main()
