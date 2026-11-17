"""GUI backend routes on a ``.pivraw`` run the reader refuses.

``Config.num_frame_pairs`` and ``read_pair`` run the trigger alignment, which
raises ``PivrawError`` naming the camera and the reason. A route that lets it
become a 500 shows the user "internal error" for a recording they must re-take.
Routes that read pairs answer 400 with the reader's text. The two config routes
must still return the config (the GUI loads it on mount and cannot change the
source otherwise), so they carry the reader's text in ``images.pairing_error``.
"""

import pytest
import yaml
from fixtures.pivraw import copy_case, write_piv_config
from flask import Flask

from pivtools_core.config import Config
from pivtools_core.image_handling.readers.pivraw_reader import clear_alignment_cache
from pivtools_gui import app as A
from pivtools_gui.errors import register_json_error_handlers

# df_2cam_missed_trigger_cam2: Camera2's ids count from a different trigger.
REFUSAL = "Camera2 run trigger 1"


@pytest.fixture(autouse=True)
def _fresh_alignment():
    clear_alignment_cache()
    yield
    clear_alignment_cache()


def _client(tmp_path, monkeypatch, case="df_2cam_missed_trigger_cam2"):
    run = copy_case(tmp_path, case)
    work = tmp_path / "work"
    work.mkdir()
    config_path = write_piv_config(work, run, 2, 4)
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["base_paths"] = [str(tmp_path / "out")]
    data["calibration"] = {"active": "charuco", "calibration_sources": []}
    config_path.write_text(yaml.dump(data))
    # Some routes take a fresh Config() from the working directory.
    monkeypatch.chdir(work)
    monkeypatch.setattr(A, "get_config", lambda: Config(path=str(config_path)))
    monkeypatch.setattr(A, "reload_config", lambda: None)
    app = Flask(__name__)
    app.register_blueprint(A.api_bp)
    register_json_error_handlers(app)
    return app.test_client()


def test_pair_preview_returns_the_refusal_as_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/backend/preview_frame_pairs?count=5")
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]


def test_frame_pair_returns_the_refusal_as_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/backend/get_frame_pair?camera=1&idx=1&source_path_idx=0")
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]


def test_filter_preview_returns_the_refusal_as_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/backend/filter", json={"camera": 1, "start_idx": 1})
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]


def test_single_frame_filter_returns_the_refusal_as_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/backend/filter_single_frame", json={"camera": 1, "frame_idx": 1})
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]


def test_config_loads_and_carries_the_refusal(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/backend/config")
    assert r.status_code == 200
    images = r.get_json()["images"]
    assert images["num_frame_pairs"] is None
    assert images["per_loop_frame_pairs"] is None
    assert REFUSAL in images["pairing_error"]


def test_config_loads_and_carries_a_selection_that_names_no_dataset(
    tmp_path, monkeypatch
):
    """A .pivraw pair count aligns the selected runs, so a selection naming no
    real run has no count: the setup page must still load and show why."""
    monkeypatch.delenv("PIV_ACTIVE_PATHS", raising=False)
    client = _client(tmp_path, monkeypatch, case="df_2cam_clean")
    config_path = tmp_path / "work" / "config.yaml"
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["active_paths"] = [5]
    config_path.write_text(yaml.dump(data))

    r = client.get("/backend/config")
    assert r.status_code == 200
    images = r.get_json()["images"]
    assert images["num_frame_pairs"] is None
    assert "Dataset 5 in paths.active_paths is out of range" in images["pairing_error"]


def test_update_config_saves_and_carries_the_refusal(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/backend/update_config", json={"images": {"num_images": 6}})
    assert r.status_code == 200
    images = r.get_json()["updated"]["images"]
    assert images["num_frame_pairs"] is None
    assert REFUSAL in images["pairing_error"]
    assert yaml.safe_load(open("config.yaml"))["images"]["num_images"] == 6


def test_a_source_change_carries_the_new_pair_status(tmp_path, monkeypatch):
    """Changing only the source re-pairs the run: the GUI must not keep the old status.

    Moving from the refused run to a clean one answers with the clean count and no
    pairing_error, and moving back answers with the refusal, though no images key
    was posted either time.
    """
    client = _client(tmp_path, monkeypatch)
    clean = copy_case(tmp_path, "df_2cam_clean")
    refused = tmp_path / "df_2cam_missed_trigger_cam2"

    r = client.post(
        "/backend/update_config", json={"paths": {"source_paths": [str(clean)]}}
    )
    assert r.status_code == 200
    images = r.get_json()["updated"]["images"]
    assert images["num_frame_pairs"] == 2
    assert "pairing_error" not in images

    r = client.post(
        "/backend/update_config", json={"paths": {"source_paths": [str(refused)]}}
    )
    images = r.get_json()["updated"]["images"]
    assert images["num_frame_pairs"] is None
    assert REFUSAL in images["pairing_error"]
    assert set(images) == {"num_frame_pairs", "per_loop_frame_pairs", "pairing_error"}


def test_an_update_of_another_section_carries_no_images(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/backend/update_config", json={"filters": []})
    assert r.status_code == 200
    assert "images" not in r.get_json()["updated"]


@pytest.mark.parametrize("mode", ["instantaneous", "ensemble"])
def test_run_piv_refuses_the_run_and_starts_no_job(tmp_path, monkeypatch, mode):
    """A null pair count in the config response must never launch a job."""
    started = []

    class _Runner:
        def start_piv_job(self, **kwargs):
            started.append(kwargs)
            return {"status": "started", "job_id": "x"}

    monkeypatch.setattr(A, "get_runner", lambda: _Runner())
    client = _client(tmp_path, monkeypatch)
    r = client.post("/backend/run_piv", json={"mode": mode})
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]
    assert started == []


def test_run_piv_pairs_the_runs_the_request_selects(tmp_path, monkeypatch):
    """Run PIV's datasets come from the request (the Run tab's checkboxes), not
    from paths.active_paths in the config file, and the pairing check follows
    them: run 0 pairs, run 1 cannot, and the file says [0]."""
    started = []

    class _Runner:
        def start_piv_job(self, **kwargs):
            started.append(kwargs)
            return {"status": "started", "job_id": "x"}

    monkeypatch.delenv("PIV_ACTIVE_PATHS", raising=False)
    monkeypatch.setattr(A, "get_runner", lambda: _Runner())
    client = _client(tmp_path, monkeypatch, case="df_2cam_clean")
    unpairable = copy_case(tmp_path, "df_2cam_missed_trigger_cam2")
    config_path = tmp_path / "work" / "config.yaml"
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["source_paths"].append(str(unpairable))
    data["paths"]["base_paths"].append(str(tmp_path / "out2"))
    data["paths"]["active_paths"] = [0]
    config_path.write_text(yaml.dump(data))

    r = client.post("/backend/run_piv", json={"active_paths": [1]})
    assert r.status_code == 400
    assert str(unpairable) in r.get_json()["error"]
    assert REFUSAL in r.get_json()["error"]
    assert started == []

    r = client.post("/backend/run_piv", json={"active_paths": [0]})
    assert r.status_code == 200
    assert len(started) == 1

    # No active_paths in the request: the job takes the file's [0], and so does
    # the check (the CLI's behaviour, unchanged).
    r = client.post("/backend/run_piv", json={})
    assert r.status_code == 200
    assert len(started) == 2


def test_run_piv_starts_a_clean_run(tmp_path, monkeypatch):
    started = []

    class _Runner:
        def start_piv_job(self, **kwargs):
            started.append(kwargs)
            return {"status": "started", "job_id": "x"}

    monkeypatch.setattr(A, "get_runner", lambda: _Runner())
    client = _client(tmp_path, monkeypatch, case="df_2cam_clean")
    r = client.post("/backend/run_piv", json={"mode": "instantaneous"})
    assert r.status_code == 200
    assert len(started) == 1


@pytest.mark.parametrize(
    "url, body",
    [
        ("/backend/merge_vectors/merge_all", {"base_path_idx": 0}),
        ("/backend/statistics/calculate", {"cameras": [1]}),
        ("/backend/video/start_batch", {}),
        (
            "/backend/plot/transform_all_frames",
            {"camera": 1, "frame": 1},
        ),
    ],
)
def test_vector_job_starts_refuse_the_run_as_400(tmp_path, monkeypatch, url, body):
    """Job routes on other blueprints read the shared config singleton, so the
    real app is used with that singleton pointed at the refused run."""
    import pivtools_core.config as config_module

    _client(tmp_path, monkeypatch)  # builds the run and chdirs to its config
    monkeypatch.setattr(config_module, "_CONFIG", None)
    results = tmp_path / "out"
    results.mkdir()
    body = dict(body)
    if url.endswith("calculate"):
        body["base_paths"] = [str(results)]
    if url.endswith("transform_all_frames"):
        body["base_path"] = str(results)
    r = A.app.test_client().post(url, json=body)
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]


def test_config_of_a_clean_run_has_its_pairs_and_no_error(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, case="df_2cam_clean")
    images = client.get("/backend/config").get_json()["images"]
    assert images["num_frame_pairs"] == 2
    assert "pairing_error" not in images
