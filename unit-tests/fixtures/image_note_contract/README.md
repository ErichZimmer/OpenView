# Image-note contract fixtures

Written by the acquisition app's own writer (`pivtools_acq.common.image_note.write_note`) at `pivtools-acquisition` commit `0081187`, the A3 merge that introduced the note (`docs/CONTRACTS.md` §3.2). `MANIFEST.json` holds that writer's constants: `NOTE_KEYS`, `NOTE_FORMAT_VERSION`, `PIXEL_RULES` and the view rotations of `pixels.VIEW_ROTATION_K`.

- `calibration_cx25.tif.json`: a calibration image's note (`<n>.tif.json`), CX-25, rotated `cw90`, dark subtracted.
- `export_jai.image_note.json`: an exported camera folder's note (`image_note.json`), JAI, no rotation, no dark.

Do not edit them here. To regenerate, delete every file here but this README and `generate.py`, then run `generate.py <this folder> <acquisition commit>` with the acquisition venv and `PYTHONPATH=src` from that repo's root, and update the commit above. `generate.py` imports `pivtools_acq`, so it runs only in that environment. `unit-tests/test_image_note.py` checks the reader's constants against `MANIFEST.json` and reads both samples.
