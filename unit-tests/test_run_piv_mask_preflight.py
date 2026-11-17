"""Run PIV refuses a mask that cannot be used before any job starts (R2d).

Decision (Morgan, 17 Nov 2026): the GUI's Run PIV checks every camera's mask
on every dataset the job will process, before launching it. A refused mask is
a 400 with the MaskError message, as the .pivraw pairing preflight refuses.
With masking off there is nothing to check.

The datasets and cameras checked are the ones the job processes: the same
active-path resolution the job runs (``Config.resolve_active_paths`` fed the
value ``start_piv_job`` passes to the job), and ``config.camera_numbers``.

GET /backend/mask_status reports the same check for one camera and dataset, so
the Masking tab can show a refusal before Run.

Usage:
    pytest unit-tests/test_run_piv_mask_preflight.py -v
"""

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mask_refusals import CAM_SHAPES, FILE_MASKING, _make_config, _save_mask


class RecordingRunner:
    """Stands in for the PIV runner: records each job start, launches nothing."""

    def __init__(self) -> None:
        self.started: List[dict] = []

    def start_piv_job(self, **kwargs: Any) -> dict:
        self.started.append(kwargs)
        return {"status": "started", "job_id": "piv_test"}


def _two_dataset_rig(tmp_path: Path, masking: dict):
    """The two-camera rig twice over: datasets 0 and 1, each with its own masks."""
    cfg = _make_config(tmp_path, masking)
    second = tmp_path / "source2"
    shutil.copytree(cfg.source_paths[0], second)
    (tmp_path / "output2").mkdir()
    cfg.data["paths"]["source_paths"].append(str(second))
    cfg.data["paths"]["base_paths"].append(str(tmp_path / "output2"))
    return cfg


@pytest.fixture
def make_client(tmp_path: Path, monkeypatch: Any):
    """Build a client on a rig, with the production error handlers and a recording runner."""
    import pivtools_gui.app as gui_app
    from pivtools_gui.errors import register_json_error_handlers
    from pivtools_gui.masking.app import views as masking_views

    monkeypatch.delenv("PIV_ACTIVE_PATHS", raising=False)

    def build(cfg) -> SimpleNamespace:
        runner = RecordingRunner()
        monkeypatch.setattr(gui_app, "get_config", lambda *a, **k: cfg)
        monkeypatch.setattr(masking_views, "_cfg", lambda: cfg)
        monkeypatch.setattr(gui_app, "get_runner", lambda *a, **k: runner)
        app = Flask(__name__)
        register_json_error_handlers(app)
        app.register_blueprint(gui_app.api_bp)
        app.register_blueprint(masking_views.masking_bp, url_prefix="/backend")
        return SimpleNamespace(client=app.test_client(), runner=runner, cfg=cfg)

    return build


@pytest.mark.parametrize("mode", ["instantaneous", "ensemble"])
def test_run_piv_refuses_a_wrong_size_mask_before_launching(
    tmp_path: Path, make_client: Any, mode: str
) -> None:
    cfg = _make_config(tmp_path, FILE_MASKING)
    _save_mask(cfg, 1, CAM_SHAPES[1])
    _save_mask(cfg, 2, CAM_SHAPES[1])  # camera 1's shape: camera 2 is rotated
    rig = make_client(cfg)
    response = rig.client.post("/backend/run_piv", json={"mode": mode})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Camera 2" in error and "rotated" in error
    assert rig.runner.started == []


def test_run_piv_refuses_a_missing_mask_with_masking_on(
    tmp_path: Path, make_client: Any
) -> None:
    cfg = _make_config(tmp_path, FILE_MASKING)
    _save_mask(cfg, 1, CAM_SHAPES[1])
    rig = make_client(cfg)
    response = rig.client.post("/backend/run_piv", json={"mode": "instantaneous"})
    assert response.status_code == 400
    assert "Camera 2 has no mask file" in response.get_json()["error"]
    assert rig.runner.started == []


def test_run_piv_starts_when_every_mask_fits(tmp_path: Path, make_client: Any) -> None:
    """Control: good masks for both cameras, the job starts."""
    cfg = _make_config(tmp_path, FILE_MASKING)
    for cam, shape in CAM_SHAPES.items():
        _save_mask(cfg, cam, shape)
    rig = make_client(cfg)
    response = rig.client.post("/backend/run_piv", json={"mode": "instantaneous"})
    assert response.status_code == 200
    assert len(rig.runner.started) == 1


def test_run_piv_with_masking_off_checks_nothing(
    tmp_path: Path, make_client: Any
) -> None:
    """Control: no mask files anywhere, masking off, the job starts."""
    rig = make_client(_make_config(tmp_path, {"enabled": False, "mode": "file"}))
    response = rig.client.post("/backend/run_piv", json={"mode": "instantaneous"})
    assert response.status_code == 200
    assert len(rig.runner.started) == 1


def test_run_piv_checks_only_the_datasets_the_job_will_process(
    tmp_path: Path, make_client: Any
) -> None:
    """Dataset 1 has no masks. Running dataset 0 alone starts; dataset 1 is refused,
    and with no active_paths the job takes every dataset, so it is refused too."""
    cfg = _two_dataset_rig(tmp_path, FILE_MASKING)
    for cam, shape in CAM_SHAPES.items():
        _save_mask(cfg, cam, shape)  # dataset 0 only
    rig = make_client(cfg)

    only_first = rig.client.post("/backend/run_piv", json={"active_paths": [0]})
    assert only_first.status_code == 200

    only_second = rig.client.post("/backend/run_piv", json={"active_paths": [1]})
    assert only_second.status_code == 400
    assert (
        str(Path(cfg.source_paths[1]) / "mask_Cam1.mat")
        in only_second.get_json()["error"]
    )

    every_dataset = rig.client.post("/backend/run_piv", json={})
    assert every_dataset.status_code == 400
    assert len(rig.runner.started) == 1


def _status(rig, camera: Any, basepath_idx: Any = 0):
    return rig.client.get(
        f"/backend/mask_status?basepath_idx={basepath_idx}&camera={camera}"
    )


def test_mask_status_reports_a_refused_mask(tmp_path: Path, make_client: Any) -> None:
    cfg = _make_config(tmp_path, FILE_MASKING)
    _save_mask(cfg, 2, CAM_SHAPES[1])
    rig = make_client(cfg)
    response = _status(rig, "Cam2")
    assert response.status_code == 200
    body = response.get_json()
    assert body["masking_enabled"] is True and body["mode"] == "file"
    assert body["ok"] is False
    assert "Camera 2" in body["message"] and "rotated" in body["message"]


def test_mask_status_reports_a_missing_mask_and_an_unknown_mode(
    tmp_path: Path, make_client: Any
) -> None:
    rig = make_client(_make_config(tmp_path, FILE_MASKING))
    body = _status(rig, 1).get_json()
    assert body["ok"] is False and "Camera 1 has no mask file" in body["message"]

    rig.cfg.data["masking"]["mode"] = "polygon"
    body = _status(rig, 1).get_json()
    assert body["mode"] == "polygon" and body["ok"] is False
    assert "masking.mode 'polygon'" in body["message"]


def test_mask_status_is_ok_for_a_fitting_mask_and_with_masking_off(
    tmp_path: Path, make_client: Any
) -> None:
    cfg = _make_config(tmp_path, FILE_MASKING)
    _save_mask(cfg, 1, CAM_SHAPES[1])
    rig = make_client(cfg)
    assert _status(rig, 1).get_json() == {
        "masking_enabled": True,
        "mode": "file",
        "ok": True,
        "message": None,
    }
    rig.cfg.data["masking"]["enabled"] = False
    assert _status(rig, 2).get_json() == {
        "masking_enabled": False,
        "mode": "file",
        "ok": True,
        "message": None,
    }


@pytest.mark.parametrize(
    "query", ["basepath_idx=5&camera=1", "basepath_idx=0&camera=x", "basepath_idx=0"]
)
def test_mask_status_rejects_a_bad_request(
    tmp_path: Path, make_client: Any, query: str
) -> None:
    rig = make_client(_make_config(tmp_path, FILE_MASKING))
    response = rig.client.get(f"/backend/mask_status?{query}")
    assert response.status_code == 400
    assert response.get_json()["error"]
