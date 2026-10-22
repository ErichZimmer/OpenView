"""Shared helpers for the source-diagnosis tests (contexts, file trees, configs)."""

from pathlib import Path
from typing import Any, Iterable

import yaml

from pivtools_core.config import Config
from pivtools_core.image_handling.source_diagnosis import DiagnosisContext


def make_ctx(source_path: Any, **overrides: Any) -> DiagnosisContext:
    """Build a ``DiagnosisContext`` for a standard single-camera Setup page."""
    fields = dict(
        source_path=Path(source_path),
        image_type="standard",
        image_format=("B%05d.tif",),
        camera_count=1,
        camera_numbers=(1,),
        use_camera_subfolders=False,
        camera_subfolders=(),
        pairing_preset="time_resolved",
        pair_reader=False,
        start_index=1,
        calibration=False,
    )
    fields.update(overrides)
    return DiagnosisContext(**fields)


def touch(folder: Path, names: Iterable[str]) -> None:
    """Create ``folder`` (and parents) and an empty file for each name in it."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


def write_config(tmp_path: Path, **blocks: dict) -> Config:
    """Write a minimal ``config.yaml`` and load it.

    ``blocks`` (``paths`` / ``images`` / ``calibration``) are merged key by key
    over a valid single-camera standard default.
    """
    cfg = {
        "paths": {
            "source_paths": [str(tmp_path / "src")],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {"image_format": ["B%05d.tif"], "start_index": 1},
        "calibration": {},
    }
    for name, block in blocks.items():
        cfg.setdefault(name, {}).update(block)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return Config(path=str(config_path))
