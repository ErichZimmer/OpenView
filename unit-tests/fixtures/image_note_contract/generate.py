"""Write the PIVTOOLs image-note contract fixture with the acquisition app's own writer.

Run with the acquisition venv and PYTHONPATH=src from the acquisition repo root. Takes the
output folder as argv[1] and the writer commit as argv[2]. Refuses to overwrite a sample.
"""

import json
import sys
from pathlib import Path

from pivtools_acq.common.image_note import (
    NOTE_FORMAT_VERSION,
    NOTE_KEYS,
    PIXEL_RULES,
    ImageNote,
    write_note,
)
from pivtools_acq.common.pixels import VIEW_ROTATION_K

out = Path(sys.argv[1])
commit = sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
samples = {
    "calibration_cx25.tif.json": ImageNote(
        view_rotation="cw90",
        dark_id="dark_20261117_101500",
        pixel_rule="mono12p",
        camera_serial="118581",
    ),
    "export_jai.image_note.json": ImageNote(
        view_rotation="none",
        dark_id=None,
        pixel_rule="bayer_rggb_superpixel_sum",
        camera_serial="U512345",
    ),
}
for name, note in samples.items():
    if (out / name).exists():
        raise SystemExit(f"{out / name} exists; delete the old samples first")
    write_note(out / name, note)
manifest = {
    "writer_commit": commit,
    "format_version": NOTE_FORMAT_VERSION,
    "note_keys": list(NOTE_KEYS),
    "pixel_rules": PIXEL_RULES,
    "view_rotations": list(VIEW_ROTATION_K),
    "samples": sorted(samples),
}
(out / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
)
print("wrote", sorted(p.name for p in out.iterdir()))
