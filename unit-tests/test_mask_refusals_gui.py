"""The GUI's mask routes refuse a mask that does not fit its camera (R2c).

Same decision as ``test_mask_refusals.py``: a wrong-size mask is refused by
name, never stretched, and with masking on a missing mask is refused. Here the
refusal must reach the browser as an error response carrying the message,
not a log line on the server. The routes are driven for real over a
two-camera TIFF rig of transposed shapes.

Usage:
    pytest unit-tests/test_mask_refusals_gui.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mask_refusals import CAM_SHAPES, FILE_MASKING, _make_config, _save_mask


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: Any) -> SimpleNamespace:
    """A masked two-camera rig, cwd at its config, with a test client on it."""
    import pivtools_gui.app as gui_app
    from pivtools_gui.errors import register_json_error_handlers
    from pivtools_gui.masking.app import views as masking_views

    cfg = _make_config(tmp_path, FILE_MASKING)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(masking_views, "_cfg", lambda: cfg)
    app = Flask(__name__)
    register_json_error_handlers(app)
    app.register_blueprint(gui_app.api_bp)
    app.register_blueprint(masking_views.masking_bp, url_prefix="/backend")
    return SimpleNamespace(cfg=cfg, client=app.test_client(), gui_app=gui_app)


def test_single_frame_preview_refuses_a_wrong_size_mask_with_400(rig):
    _save_mask(rig.cfg, 2, CAM_SHAPES[1])
    response = rig.client.post(
        "/backend/filter_single_frame",
        json={"camera": 2, "frame_idx": 1, "filters": [], "source_path_idx": 0},
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Camera 2" in error and "rotated" in error


def test_batch_preview_refuses_a_missing_mask_before_its_worker_starts(
    rig, monkeypatch: Any
):
    """The batch preview's worker thread used to log the problem and carry on."""

    class NoThread:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise AssertionError("worker thread started")

    monkeypatch.setattr(rig.gui_app, "threading", SimpleNamespace(Thread=NoThread))
    response = rig.client.post(
        "/backend/filter", json={"camera": 1, "start_idx": 1, "source_path_idx": 0}
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Camera 1 has no mask file" in error


def _save_payload(camera: int, height: int, width: int) -> dict:
    return {
        "meta": {"basePathIdx": 0, "camera": f"Cam{camera}"},
        "width": width,
        "height": height,
        "polygons": [{"index": 0, "name": "wall", "points": [[0, 0], [9, 0], [9, 3]]}],
    }


def test_saving_a_mask_of_another_size_than_the_camera_is_refused(rig):
    response = rig.client.post(
        "/backend/save_mask_array", json=_save_payload(2, *CAM_SHAPES[1])
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Camera 2" in error and "64 x 96" in error and "96 x 64" in error
    assert not rig.cfg.get_mask_path(2).exists()


def test_saving_a_mask_of_the_cameras_size_writes_it(rig):
    """Control: the saved mask then loads for that camera."""
    from pivtools_core.image_handling.load_images import load_mask_for_camera

    response = rig.client.post(
        "/backend/save_mask_array", json=_save_payload(2, *CAM_SHAPES[2])
    )
    assert response.status_code == 200
    assert load_mask_for_camera(2, rig.cfg).shape == CAM_SHAPES[2]


def test_loading_a_stale_mask_into_the_editor_is_refused_with_409(rig):
    _save_mask(rig.cfg, 2, CAM_SHAPES[1])
    response = rig.client.get(
        "/backend/load_mask?basepath_idx=0&camera=Cam2&polygons_only=true"
    )
    assert response.status_code == 409
    error = response.get_json()["error"]
    assert "Camera 2" in error and "rotated" in error


def test_loading_a_missing_mask_into_the_editor_is_still_404(rig):
    """Control: no mask yet is the editor's normal start, not a refusal."""
    response = rig.client.get(
        "/backend/load_mask?basepath_idx=0&camera=Cam1&polygons_only=true"
    )
    assert response.status_code == 404
