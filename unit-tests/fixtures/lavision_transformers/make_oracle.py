"""Regenerate ``lvpyio_oracle.npz``: LaVision's answer for every stream transformer.

Workstation only: needs lvpyio (1.3.1 was used; V0 found 1.5.0 identical on these
cases) and the two research writers this oracle was proven with,

- ``rotation-shape/scripts/synth_set.py`` (``write_recording``), from the
  rotation-shape follow-up, for one transformer per stream;
- ``transformer-order/transformer_order.py`` (``build``), from the
  transformer-order follow-up, for several transformers on one stream.

Each case is written as a small DaVis recording, read with lvpyio, and stored
twice in the archive: the exact bytes lvpyio read (every file of the recording)
and lvpyio's frames. The unit tests unpack the bytes and compare the production
reader against lvpyio bit for bit, shape included, so the Mac checks LaVision's
answer without lvpyio.

``.gitignore`` ignores ``unit-tests/**/*.npz`` (test outputs), so a regenerated
archive is committed with ``git add -f``.

Archive keys:``{case}__file__{relative path}`` (uint8 bytes, ``/`` written as
``~``), ``{case}__lvpyio`` ((entries, H, W) uint16) and ``{case}__meta`` (JSON).

Usage::

    python make_oracle.py
    python make_oracle.py --research-dir <followup folder> --out <npz>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_RESEARCH = (
    Path.home()
    / "OneDrive - University of Southampton/Documents/pivtools_fullstack/wiki/plans"
    / "2026-11-14-overnight-research/01-lavision-readers/followup"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "lvpyio_oracle.npz"

# Non-square and small: the eight symmetries of a 12 x 20 frame are pairwise
# distinct, and the archive stays a few hundred kB.
H, W = 12, 20
SEED = 20261117

GEOMETRIC = (
    None,
    "rotate-180",
    "vertical-mirror",
    "horinzontal-mirror",
    "rotate-cw-90",
    "rotate-ccw-90",
    "rotate-cw-90-vertical-mirror",
    "rotate-ccw-90-vertical-mirror",
)
FILE_BACKED = (
    "dark-image-subtraction",
    "background-subtraction",
    "intensity-correction-1",
)


def order_cases() -> list[tuple[str, list[str], str]]:
    """(name, ids in manifest order, prefix scheme): transformer-order REPORT §6 (c) 1."""
    cases = []
    for geo, short in (("rotate-cw-90", "cw90"), ("rotate-ccw-90", "ccw90")):
        for first in ("dark", "geo"):
            ids = (
                ["dark-image-subtraction", geo]
                if first == "dark"
                else [geo, "dark-image-subtraction"]
            )
            for scheme in ("prefix=manifest", "prefix=reversed"):
                tag = "pm" if scheme == "prefix=manifest" else "pr"
                cases.append((f"o_dark_{short}_{first}first_{tag}", ids, scheme))
    pm = "prefix=manifest"
    cases += [
        ("o_ic1_dark", ["intensity-correction-1", "dark-image-subtraction"], pm),
        ("o_dark_ic1", ["dark-image-subtraction", "intensity-correction-1"], pm),
        ("o_ic1_bgnd", ["intensity-correction-1", "background-subtraction"], pm),
        ("o_dark_bgnd", ["dark-image-subtraction", "background-subtraction"], pm),
        ("o_ic1_rot180", ["intensity-correction-1", "rotate-180"], pm),
        ("o_bgnd_vflip", ["background-subtraction", "vertical-mirror"], pm),
        (
            "o_ic1_dark_cw90",
            ["intensity-correction-1", "dark-image-subtraction", "rotate-cw-90"],
            pm,
        ),
        (
            "o_cw90_dark_ic1_pr",
            ["rotate-cw-90", "dark-image-subtraction", "intensity-correction-1"],
            "prefix=reversed",
        ),
    ]
    return cases


def archive(store: dict, case: str, set_file: Path, frames, meta: dict) -> None:
    """Add one recording's files, lvpyio's frames and the case description."""
    root = set_file.parent
    paths = [set_file] + sorted(
        p for p in set_file.with_suffix("").rglob("*") if p.is_file()
    )
    for p in paths:
        rel = p.relative_to(root).as_posix().replace("/", "~")
        store[f"{case}__file__{rel}"] = np.frombuffer(p.read_bytes(), dtype=np.uint8)
    stack = np.stack(frames)
    if stack.dtype != np.uint16:
        raise ValueError(f"{case}: lvpyio returned {stack.dtype}, expected uint16")
    store[f"{case}__lvpyio"] = stack
    store[f"{case}__meta"] = np.array(json.dumps(meta))
    print(f"{case}: lvpyio {stack.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    sys.path.insert(0, str(a.research_dir / "rotation-shape" / "scripts"))
    sys.path.insert(0, str(a.research_dir / "transformer-order"))
    import synth_set as ss  # noqa: E402  (research writer, path set above)
    import transformer_order as to  # noqa: E402

    rng = np.random.default_rng(SEED)
    store: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def read_single_stream(sf: Path):
            entries = ss.read_lvpyio(sf)
            if not entries:
                raise RuntimeError(f"lvpyio opened 0 buffers for {sf}")
            if any(len(e) != 1 for e in entries):
                raise RuntimeError(f"{sf}: expected one frame per buffer")
            return [e[0] for e in entries]

        # One geometric transformer (or none), two entries, two encodings.
        for enc in ("raw-16-bit", "mono-12p"):
            for tid in GEOMETRIC:
                case = f"g_{enc}_{tid or 'none'}".replace("-", "_")
                frames = [ss.asymmetric_frame(H, W, enc, entry=e) for e in range(2)]
                sf = ss.write_recording(
                    work / case, "rec", [ss.Stream(enc, frames, transformer=tid)]
                )
                archive(
                    store,
                    case,
                    sf,
                    read_single_stream(sf),
                    {"ids": [tid] if tid else []},
                )

        # One file-backed correction. Raw and correction hit every clip boundary:
        # x - C < 0, x - C + 300 > 65535 (raw near saturation, C < 300).
        for tid in FILE_BACKED:
            case = f"f_{tid}".replace("-", "_")
            raw, corr = to.make_images(rng, H, W)
            c = corr((H, W))
            raw[0, :6] = [0, 1, 299, 300, 65535, 65400]
            c[0, :6] = [0, 300, 300, 300, 0, 299]
            sf = ss.write_recording(
                work / case,
                "rec",
                [ss.Stream("raw-16-bit", [raw], transformer=tid, correction=c)],
            )
            archive(store, case, sf, read_single_stream(sf), {"ids": [tid]})

        # Several transformers on one stream, written by transformer_order.build.
        template = work / "template.im7"
        ss.write_im7_word(template, np.zeros((H, W), np.uint16))
        for case, ids, scheme in order_cases():
            raw, corr = to.make_images(rng, H, W)
            pre = to.prefixes(len(ids), scheme)
            ops = [
                {
                    "id": t,
                    "prefix": pre[i],
                    "corr": corr((H, W)) if t in to.INTENSITY else None,
                }
                for i, t in enumerate(ids)
            ]
            (work / case).mkdir()
            sf = to.build(work / case, template, "rec", raw, ops)
            archive(
                store,
                case,
                sf,
                read_single_stream(sf),
                {"ids": ids, "prefixes": pre},
            )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **store)
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes, {len(store)} keys)")


if __name__ == "__main__":
    main()
