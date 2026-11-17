"""``pivtools-cli init --from-run``: a PIVTOOLs config for a recorded ``.pivraw`` run.

The acquisition app records a run to ``<project>/<run>/`` and calibration images
to ``<project>/calibration/Cam{k}/``. The import writes a config from the CLI
template plus the run's facts, read through the strict reader: the cameras, the
pairing for the run's mode, the image count, and ``rig.dt`` into the
calibration source's settings. Ground truth is the contract fixtures'
``expected.json`` and their sidecars (``df_2cam_clean`` records ``dt_s``
0.0002, ``tr_bayer_clean`` 0.04 at 25 Hz).
"""

import sys

import pytest
import yaml
from fixtures.pivraw import contract_cases, copy_case, edit_index, edit_sidecar

from pivtools_cli import cli
from pivtools_cli.run_import import RunImportConflict, import_run
from pivtools_core import calibration_settings as cs
from pivtools_core.config import Config
from pivtools_core.image_handling.readers.pivraw_reader import (
    PivrawAlignmentError,
    PivrawError,
    clear_alignment_cache,
)
from pivtools_core.validation import validate_config

CASES = contract_cases()


@pytest.fixture(autouse=True)
def _fresh_alignment():
    clear_alignment_cache()
    yield
    clear_alignment_cache()


def _project_run(tmp_path, case):
    """A project folder holding one recorded run, as the app lays it out."""
    project = tmp_path / "project"
    project.mkdir()
    return project, copy_case(project, case)


def _rig_dt(source):
    return yaml.safe_load(cs.settings_path(source).read_text(encoding="utf-8"))["rig"][
        "dt"
    ]


def test_double_frame_run_imports_to_a_config_that_validates(tmp_path):
    """PC T8: two pairs, the source is the run folder, rig.dt is 2e-4 s exactly."""
    project, run = _project_run(tmp_path, "df_2cam_clean")
    config_path = tmp_path / "work" / "config.yaml"
    config_path.parent.mkdir()
    import_run(run, config_path)

    cfg = Config(path=str(config_path))
    ok, error, _warnings = validate_config(cfg)
    assert ok, error
    assert cfg.num_frame_pairs == len(CASES["df_2cam_clean"]["pairs"]) == 2
    # The run folder, not the recording machine's temp dir the snippet names.
    assert cfg.source_paths == [run]
    assert cfg.base_paths == [project / "results" / "df_2cam_clean"]
    assert cfg.image_type == "pivraw"
    assert cfg.image_format == ("Camera%d.pivraw",)
    assert cfg.camera_numbers == [1, 2]
    assert cfg.camera_count == 2
    assert cfg.pairing_preset == "skip_frames"
    assert (cfg.start_index, cfg.frame_stride, cfg.pair_stride) == (1, 1, 2)
    assert cfg.num_loops == 1
    assert cfg.num_images == 4  # max n_frames over cameras
    calibration = project / "calibration"
    assert cfg.data["calibration"]["calibration_sources"] == [str(calibration)]
    assert _rig_dt(calibration) == 2e-4


def test_one_camera_time_resolved_run_points_at_cam1(tmp_path):
    """The app always writes calibration/Cam1/; PIVTOOLs reads a one-camera
    source at its root, so the source is calibration/Cam1 (Morgan, 17 Nov).
    Time-resolved: rig.dt = frame_stride x dt_s = 1 x 0.04 s."""
    project, run = _project_run(tmp_path, "tr_bayer_clean")
    config_path = tmp_path / "config.yaml"
    import_run(run, config_path)

    cfg = Config(path=str(config_path))
    ok, error, _warnings = validate_config(cfg)
    assert ok, error
    assert cfg.num_frame_pairs == len(CASES["tr_bayer_clean"]["pairs"]) == 2
    assert cfg.pairing_preset == "time_resolved"
    assert (cfg.frame_stride, cfg.pair_stride) == (1, 1)
    assert cfg.camera_numbers == [1]
    assert cfg.num_images == 3
    source = project / "calibration" / "Cam1"
    assert cfg.data["calibration"]["calibration_sources"] == [str(source)]
    assert _rig_dt(source) == 0.04


def test_a_run_with_a_drop_is_refused_and_nothing_is_written(tmp_path):
    """Camera 2 dropped run trigger 1 (test_pivraw_alignment
    ``_cam2_dropped_pair_2``). Pairs are never dropped, so the import fails
    with the reader's ``dropped`` refusal and writes no config, calibration
    settings or results folder."""
    project = tmp_path / "project"
    project.mkdir()
    run = copy_case(project, "df_2cam_missed_trigger_cam2", "cam2_dropped_pair_2")
    edit_index(run, 2, frame_id=[5000, 5001, 5004, 5005])
    edit_sidecar(run, 2, missing_indices=[2, 3], dropped_pairs=1)
    config_path = tmp_path / "config.yaml"
    with pytest.raises(PivrawAlignmentError) as info:
        import_run(run, config_path)
    assert info.value.reason == "dropped"
    assert "Camera2 lacks run trigger(s) [1]" in str(info.value)
    assert str(info.value).endswith("Re-record the run.")
    assert not config_path.exists()
    assert not (project / "calibration").exists()
    assert not (project / "results").exists()


def test_a_run_that_cannot_align_is_refused_and_nothing_is_written(tmp_path):
    project, run = _project_run(tmp_path, "df_2cam_missed_trigger_cam2")
    config_path = tmp_path / "config.yaml"
    with pytest.raises(PivrawAlignmentError, match="Camera2 run trigger 1"):
        import_run(run, config_path)
    assert not config_path.exists()
    assert not (project / "calibration").exists()
    assert not (project / "results").exists()


def test_new_calibration_settings_load_with_the_app_layout(tmp_path):
    """Created settings name the app's calibration files, so they load at once."""
    project, run = _project_run(tmp_path, "df_2cam_clean")
    import_run(run, tmp_path / "config.yaml")
    settings = cs.load_settings(project / "calibration")
    assert settings["image"]["image_format"] == "%05d.tif"
    assert settings["image"]["image_type"] == "standard"


def test_the_summary_says_the_image_block_came_from_the_app_layout(tmp_path):
    _project, run = _project_run(tmp_path, "df_2cam_clean")
    summary = import_run(run, tmp_path / "config.yaml").summary()
    assert "created" in summary and "acquisition app's calibration layout" in summary
    assert "%05d.tif" in summary and "standard" in summary


def test_existing_calibration_settings_keep_their_image_block(tmp_path):
    project, run = _project_run(tmp_path, "df_2cam_clean")
    source = project / "calibration"
    cs.save_settings(
        source, {"image": {"image_format": "cal_%03d.tif", "image_type": "standard"}}
    )
    result = import_run(run, tmp_path / "config.yaml")
    assert "calibration layout" not in result.summary()
    settings = cs.load_settings(source)
    assert settings["image"]["image_format"] == "cal_%03d.tif"
    assert settings["rig"]["dt"] == 2e-4


def test_a_different_rig_dt_in_existing_settings_is_refused(tmp_path):
    """Velocity scales with dt: another run's calibration is never re-scaled silently."""
    project, run = _project_run(tmp_path, "df_2cam_clean")
    source = project / "calibration"
    cs.save_settings(source, {"rig": {"dt": 5e-4}})
    config_path = tmp_path / "config.yaml"
    with pytest.raises(RunImportConflict, match=r"rig\.dt") as excinfo:
        import_run(run, config_path)
    assert "0.0005" in str(excinfo.value) and "0.0002" in str(excinfo.value)
    assert len(excinfo.value.conflicts) == 1
    assert not config_path.exists()
    assert _rig_dt(source) == 5e-4

    import_run(run, config_path, force=True)
    assert _rig_dt(source) == 2e-4


def test_the_same_rig_dt_in_existing_settings_is_accepted(tmp_path):
    project, run = _project_run(tmp_path, "df_2cam_clean")
    cs.save_settings(project / "calibration", {"rig": {"dt": 2e-4}})
    import_run(run, tmp_path / "config.yaml")
    assert _rig_dt(project / "calibration") == 2e-4


def test_an_existing_config_needs_force(tmp_path):
    _project, run = _project_run(tmp_path, "df_2cam_clean")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\n")
    with pytest.raises(FileExistsError, match="--force"):
        import_run(run, config_path)
    assert config_path.read_text() == "paths: {}\n"
    import_run(run, config_path, force=True)
    assert Config(path=str(config_path)).image_type == "pivraw"


def test_an_existing_config_and_a_different_rig_dt_are_reported_together(tmp_path):
    """force replaces both, so the user must see both before choosing it."""
    project, run = _project_run(tmp_path, "df_2cam_clean")
    cs.save_settings(project / "calibration", {"rig": {"dt": 5e-4}})
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\n")
    with pytest.raises(RunImportConflict) as excinfo:
        import_run(run, config_path)
    conflicts = excinfo.value.conflicts
    assert len(conflicts) == 2
    assert "already exists" in conflicts[0]
    assert "0.0005" in conflicts[1] and "0.0002" in conflicts[1]
    assert str(excinfo.value).endswith("Use --force to replace them.")
    assert config_path.read_text() == "paths: {}\n"
    assert _rig_dt(project / "calibration") == 5e-4

    import_run(run, config_path, force=True)
    assert Config(path=str(config_path)).image_type == "pivraw"
    assert _rig_dt(project / "calibration") == 2e-4


def _failing_save(source, partial):
    raise PermissionError(13, "Permission denied", str(cs.settings_path(source)))


def test_a_settings_write_failure_leaves_the_config_unchanged(tmp_path, monkeypatch):
    """The settings are written before the config is moved into place, so a
    settings write that fails leaves the old config as it was."""
    project, run = _project_run(tmp_path, "df_2cam_clean")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\n")
    monkeypatch.setattr(cs, "save_settings", _failing_save)
    with pytest.raises(PermissionError):
        import_run(run, config_path, force=True)
    assert config_path.read_text() == "paths: {}\n"
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml", "project"]
    # Validating the staged config made the results folder; a refused import
    # leaves none behind.
    assert sorted(p.name for p in project.iterdir()) == ["df_2cam_clean"]


@pytest.mark.parametrize("text", ["- a\n- b\n", "rig: {dt: [\n"])
def test_malformed_calibration_settings_are_refused_by_name(tmp_path, text):
    project, run = _project_run(tmp_path, "df_2cam_clean")
    path = cs.settings_path(project / "calibration")
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    with pytest.raises(cs.CalibrationSettingsError) as excinfo:
        import_run(run, config_path)
    assert str(path) in str(excinfo.value)
    assert not config_path.exists()
    assert path.read_text(encoding="utf-8") == text


def test_cameras_that_disagree_on_dt_are_refused(tmp_path):
    _project, run = _project_run(tmp_path, "df_2cam_clean")
    edit_sidecar(run, 2, dt_s=3e-4)
    with pytest.raises(PivrawError, match=r"dt_s") as excinfo:
        import_run(run, tmp_path / "config.yaml")
    assert "Camera1" in str(excinfo.value) and "Camera2" in str(excinfo.value)


def test_base_path_can_be_chosen(tmp_path):
    _project, run = _project_run(tmp_path, "df_2cam_clean")
    out = tmp_path / "elsewhere"
    import_run(run, tmp_path / "config.yaml", base_path=out)
    assert Config(path=str(tmp_path / "config.yaml")).base_paths == [out]


def _cli(monkeypatch, cwd, *args):
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "argv", ["pivtools-cli", *args])
    cli.main()


def test_cli_writes_the_config_in_the_working_directory(tmp_path, monkeypatch):
    _project, run = _project_run(tmp_path, "df_2cam_clean")
    work = tmp_path / "work"
    work.mkdir()
    _cli(monkeypatch, work, "init", "--from-run", str(run))
    assert Config(path=str(work / "config.yaml")).source_paths == [run]


def test_cli_base_path_without_from_run_is_refused(tmp_path, monkeypatch, capsys):
    """--base-path only means something for an imported run; argparse refuses
    it alone (exit 2) before any config is written."""
    with pytest.raises(SystemExit) as excinfo:
        _cli(monkeypatch, tmp_path, "init", "--base-path", str(tmp_path / "out"))
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--base-path" in err and "--from-run" in err
    assert not (tmp_path / "config.yaml").exists()


def test_cli_refusal_exits_non_zero_with_the_reason(tmp_path, monkeypatch, capsys):
    _project, run = _project_run(tmp_path, "df_2cam_missed_trigger_cam2")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        _cli(monkeypatch, work, "init", "--from-run", str(run))
    assert excinfo.value.code == 1
    assert "Camera2 run trigger 1" in capsys.readouterr().err
    assert not (work / "config.yaml").exists()
