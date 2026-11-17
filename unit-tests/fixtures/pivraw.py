"""PIVTOOLS Acquisition run folders (``.pivraw``) for reader tests.

Every run comes from the committed contract fixtures in ``pivraw_contract/``:
runs the real recorder wrote (pivtools-acquisition ``7210202``, see that
folder's README), with every frame decoded by the acquisition package in
``expected_arrays.npz`` and the per-case truth in ``expected.json``. Tests copy
a case into a temporary folder and change one thing at a time, so no test
builds a container from its own reading of the format.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

CONTRACT_DIR = Path(__file__).resolve().parent / "pivraw_contract"
# Sentinel for ``edit_sidecar``: remove the field from the sidecar.
DROP = object()


def contract_cases() -> Dict[str, Any]:
    """``expected.json`` ``cases``: per case its mode, cameras and decided outcome."""
    return json.loads((CONTRACT_DIR / "expected.json").read_text(encoding="utf-8"))[
        "cases"
    ]


def contract_manifest() -> Dict[str, Any]:
    """``MANIFEST.json``: writer commit, format version, ``.idx`` layout, sizes."""
    return json.loads((CONTRACT_DIR / "MANIFEST.json").read_text(encoding="utf-8"))


def expected_frames(case: str, camera: int) -> np.ndarray:
    """uint16 ``(n_frames, H, W)``: every frame of one camera as the acquisition decoded it."""
    with np.load(CONTRACT_DIR / "expected_arrays.npz") as arrays:
        return arrays[f"{case}__Camera{camera}"]


def copy_case(tmp_path: Path, case: str, name: str = "") -> Path:
    """Copy one contract run folder to ``tmp_path / (name or case)`` and return it."""
    run = tmp_path / (name or case)
    shutil.copytree(CONTRACT_DIR / case, run)
    return run


def edit_sidecar(run: Path, camera: int, **changes: Any) -> Path:
    """Change fields of ``Camera{camera}.pivraw.json`` in ``run``; :data:`DROP` deletes one.

    Returns the camera's ``.pivraw`` data path.
    """
    side = run / f"Camera{camera}.pivraw.json"
    sidecar = json.loads(side.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            sidecar.pop(key)
        else:
            sidecar[key] = value
    side.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return run / f"Camera{camera}.pivraw"


def write_piv_config(
    tmp_path: Path,
    run_dir: Path,
    camera_count: int,
    num_images: int,
    preset: str = "skip_frames",
) -> Path:
    """A ``config.yaml`` pointing PIVTOOLs at ``run_dir`` the way a user would."""
    strides = {"skip_frames": (1, 2), "time_resolved": (1, 1), "ab_format": (0, 1)}
    frame_stride, pair_stride = strides[preset]
    cfg = {
        "paths": {
            "source_paths": [str(run_dir)],
            "camera_count": camera_count,
            "camera_numbers": list(range(1, camera_count + 1)),
        },
        "images": {
            "image_format": ["Camera%d.pivraw"],
            "image_type": "pivraw",
            "num_images": num_images,
            "start_index": 1,
            "pairing_preset": preset,
            "frame_stride": frame_stride,
            "pair_stride": pair_stride,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path
