"""Shared helpers for the source-diagnosis tests (contexts, file trees, configs)."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from flask import Flask
from flask.testing import FlaskClient

from pivtools_core.config import Config
from pivtools_core.image_handling.path_utils import (
    camera_subfolder_name,
    detect_start_index,
    format_to_glob,
    glob_images,
)
from pivtools_core.image_handling.source_diagnosis import (
    DiagnosisContext,
    RuleInput,
    scan_source,
)


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


def _inp(
    source: Any, camera: int = 1, expected_count: int = 1, **ctx_overrides: Any
) -> RuleInput:
    """Build a ``RuleInput`` for ``source`` the way the validators do."""
    ctx = make_ctx(source, **ctx_overrides)
    camera_path = Path(source) / camera_subfolder_name(
        ctx.image_type,
        camera,
        ctx.camera_count,
        use_camera_subfolders=ctx.use_camera_subfolders,
        camera_subfolders=ctx.camera_subfolders,
    )
    return RuleInput(
        scan_source(source),
        ctx,
        camera,
        camera_path,
        expected_count,
        detected_start_index_of(camera_path, ctx.image_format[0]),
    )


def detected_start_index_of(camera_path: Path, pattern: str) -> Optional[int]:
    """The core validator's ``detected_start_index``: over the FULL matching listing."""
    if not camera_path.is_dir() or not pattern.strip():
        return None
    names = [p.name for p in glob_images(camera_path, format_to_glob(pattern))]
    return detect_start_index(names, pattern)


def frame_names(fmt: str, start: int, stop: int) -> List[str]:
    """File names ``fmt % i`` for ``i`` in ``start..stop`` inclusive."""
    return [fmt % i for i in range(start, stop + 1)]


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


def piv_route_response(
    tmp_path: Path,
    monkeypatch: Any,
    paths: Optional[Dict[str, Any]] = None,
    images: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
) -> Any:
    """``POST /backend/validate_files`` against a config built by ``write_config``.

    ``get_config`` is bound to that config for the request; the Flask response is
    returned unparsed so status codes can be asserted.
    """
    import pivtools_gui.app as gui_app

    blocks: Dict[str, Dict[str, Any]] = {}
    if paths is not None:
        blocks["paths"] = paths
    if images is not None:
        blocks["images"] = images
    config = write_config(tmp_path, **blocks)
    monkeypatch.setattr(gui_app, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    return app.test_client().post("/backend/validate_files", json=json or {})


class FakeCalibrationConfig:
    """The ``Config`` members the calibration loader and blueprint read."""

    def __init__(
        self, source: Path, camera_count: int = 1, time_resolved: bool = False
    ):
        self._source = source
        self.camera_count = camera_count
        self.camera_numbers = list(range(1, camera_count + 1))
        self.time_resolved = time_resolved
        # The blueprint's helpers also read the calibration pointer block.
        self.calibration = {"active": "dotboard"}

    def get_calibration_source(self, idx: int) -> Path:
        """The one configured calibration source, for any index."""
        return self._source


def calibration_client(
    source: Path, monkeypatch: Any, camera_count: int = 1, time_resolved: bool = False
) -> FlaskClient:
    """A test client for the calibration blueprint with ``source`` as its source."""
    import pivtools_gui.calibration.app.views as views

    config = FakeCalibrationConfig(source, camera_count, time_resolved)
    monkeypatch.setattr(views, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(views.calibration_bp)
    return app.test_client()
