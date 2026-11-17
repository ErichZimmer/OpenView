"""``POST /backend/import_run``: the GUI's "Import acquisition run" action.

The route writes the GUI's own ``config.yaml`` (the one ``get_config`` loaded)
from a ``.pivraw`` run through :func:`pivtools_cli.run_import.import_run`, then
reloads the config. The GUI always has a config, so replacing it needs
``force``: the first attempt answers 409 with every conflict, and the user
confirms. A run the reader refuses, or a request without a run folder, answers
400 with the reason. Ground truth: the contract fixtures (``df_2cam_clean``
records ``dt_s`` 0.0002 s and aligns 2 pairs; ``df_2cam_missed_trigger_cam2``
is refused naming Camera2's run trigger 1).
"""

import pytest
import yaml
from fixtures.pivraw import copy_case
from flask import Flask

from pivtools_cli.cli import create_default_config
from pivtools_core import calibration_settings as cs
from pivtools_core.config import Config
from pivtools_core.image_handling.readers.pivraw_reader import clear_alignment_cache
from pivtools_gui import app as A
from pivtools_gui.errors import register_json_error_handlers

REFUSAL = "Camera2 run trigger 1"


@pytest.fixture(autouse=True)
def _fresh_alignment():
    clear_alignment_cache()
    yield
    clear_alignment_cache()


@pytest.fixture
def gui(tmp_path, monkeypatch):
    """A GUI backend whose working config is ``work/config.yaml`` (the CLI default).

    Returns (client, config_path, project, reloads) where ``reloads`` counts
    ``reload_config`` calls.
    """
    work = tmp_path / "work"
    work.mkdir()
    config_path = work / "config.yaml"
    create_default_config(config_path)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(A, "get_config", lambda: Config(path=str(config_path)))
    reloads = []
    monkeypatch.setattr(A, "reload_config", lambda: reloads.append(1))
    app = Flask(__name__)
    app.register_blueprint(A.api_bp)
    register_json_error_handlers(app)
    return app.test_client(), config_path, project, reloads


def _post(client, **body):
    return client.post("/backend/import_run", json=body)


def test_an_existing_config_answers_409_and_is_kept(gui):
    client, config_path, project, reloads = gui
    run = copy_case(project, "df_2cam_clean")
    before = config_path.read_text()
    r = _post(client, run_dir=str(run))
    assert r.status_code == 409
    body = r.get_json()
    assert str(config_path) in body["error"] and "already exists" in body["error"]
    assert "--force" not in body["error"]
    assert config_path.read_text() == before
    assert reloads == []


def test_force_imports_the_run_and_reloads_the_config(gui):
    client, config_path, project, reloads = gui
    run = copy_case(project, "df_2cam_clean")
    r = _post(client, run_dir=str(run), force=True)
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert f"for run {run}" in body["summary"]
    assert "2 pair(s) aligned by trigger" in body["summary"]
    cfg = Config(path=str(config_path))
    assert cfg.image_type == "pivraw"
    assert cfg.source_paths == [run]
    assert cfg.num_frame_pairs == 2
    assert reloads == [1]


def test_base_path_sets_where_results_go(gui, tmp_path):
    client, config_path, project, _ = gui
    run = copy_case(project, "df_2cam_clean")
    out = tmp_path / "elsewhere"
    r = _post(client, run_dir=str(run), base_path=str(out), force=True)
    assert r.status_code == 200
    assert yaml.safe_load(config_path.read_text())["paths"]["base_paths"] == [str(out)]


def test_a_refused_run_answers_400_and_writes_nothing(gui):
    client, config_path, project, reloads = gui
    run = copy_case(project, "df_2cam_missed_trigger_cam2")
    before = config_path.read_text()
    r = _post(client, run_dir=str(run), force=True)
    assert r.status_code == 400
    assert REFUSAL in r.get_json()["error"]
    assert config_path.read_text() == before
    assert reloads == []


def test_a_folder_that_is_not_a_run_answers_400(gui, tmp_path):
    client, _, _, _ = gui
    r = _post(client, run_dir=str(tmp_path / "nowhere"), force=True)
    assert r.status_code == 400
    assert "nowhere" in r.get_json()["error"]


@pytest.mark.parametrize(
    "body, named",
    [
        ({}, "run_dir"),
        ({"run_dir": "  "}, "run_dir"),
        ({"run_dir": 3}, "run_dir"),
        ({"run_dir": "x", "force": "yes"}, "force"),
        ({"run_dir": "x", "base_path": 5}, "base_path"),
    ],
)
def test_a_malformed_request_answers_400_naming_the_field(gui, body, named):
    client, _, _, _ = gui
    r = client.post("/backend/import_run", json=body)
    assert r.status_code == 400
    assert named in r.get_json()["error"]


def test_a_different_rig_dt_answers_409_with_both_values(gui):
    """Velocity scales with dt: the user sees the stored and recorded values first."""
    client, config_path, project, _ = gui
    run = copy_case(project, "df_2cam_clean")
    source = project / "calibration"
    cs.save_settings(source, {"rig": {"dt": 5e-4}})
    r = _post(client, run_dir=str(run))
    assert r.status_code == 409
    error = r.get_json()["error"]
    assert "already exists" in error
    assert "0.0005" in error and "0.0002" in error


def test_a_settings_write_failure_answers_500_naming_the_file(gui, monkeypatch):
    client, config_path, project, reloads = gui
    run = copy_case(project, "df_2cam_clean")
    path = cs.settings_path(project / "calibration")

    def failing_save(source, partial):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(cs, "save_settings", failing_save)
    before = config_path.read_text()
    r = _post(client, run_dir=str(run), force=True)
    assert r.status_code == 500
    error = r.get_json()["error"]
    assert str(path) in error and "Permission denied" in error
    assert config_path.read_text() == before
    assert reloads == []


def test_malformed_calibration_settings_answer_400_naming_the_file(gui):
    client, config_path, project, reloads = gui
    run = copy_case(project, "df_2cam_clean")
    path = cs.settings_path(project / "calibration")
    path.parent.mkdir(parents=True)
    path.write_text("- a\n- b\n", encoding="utf-8")
    before = config_path.read_text()
    r = _post(client, run_dir=str(run), force=True)
    assert r.status_code == 400
    assert str(path) in r.get_json()["error"]
    assert config_path.read_text() == before
    assert reloads == []


def test_force_replaces_a_different_rig_dt(gui):
    client, _, project, _ = gui
    run = copy_case(project, "df_2cam_clean")
    source = project / "calibration"
    cs.save_settings(source, {"rig": {"dt": 5e-4}})
    r = _post(client, run_dir=str(run), force=True)
    assert r.status_code == 200
    stored = yaml.safe_load(cs.settings_path(source).read_text(encoding="utf-8"))
    assert stored["rig"]["dt"] == 2e-4
