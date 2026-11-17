"""A dataset selection that names no real dataset is refused by name (R2d Q2).

Decision (Morgan, 17 Nov 2026): ``Config.resolve_active_paths`` used to drop
out-of-range indices and ignore a malformed ``PIV_ACTIVE_PATHS`` (falling back
to the config file), so Run PIV on dataset 5 of a one-dataset config started a
job that processed nothing. Now an index outside the datasets, a non-integer
entry, an empty selection or a malformed override raises ``ActivePathsError``,
naming the bad value and how many datasets the config has. Every caller gets
it: the PIV job (the CLI exits non-zero), Run PIV and the output routes (400).

Usage:
    pytest unit-tests/test_active_paths_refusals.py -v
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mask_refusals import _make_config
from test_run_piv_mask_preflight import _two_dataset_rig

from pivtools_core.config import ACTIVE_PATHS_ENV, ActivePathsError

NO_MASKING = {"enabled": False, "mode": "file"}


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: Any) -> None:
    monkeypatch.delenv(ACTIVE_PATHS_ENV, raising=False)


@pytest.fixture
def two(tmp_path: Path):
    """A config with two datasets (0 and 1)."""
    return _two_dataset_rig(tmp_path, NO_MASKING)


def test_without_a_selection_every_dataset_is_active(two) -> None:
    """Control: paths.active_paths unset means all datasets."""
    assert two.active_paths == [0, 1]


def test_a_valid_selection_is_kept(two, monkeypatch: Any) -> None:
    """Control: a file selection and an override within range are returned as given."""
    two.data["paths"]["active_paths"] = [1]
    assert two.active_paths == [1]
    monkeypatch.setenv(ACTIVE_PATHS_ENV, "0,1")
    assert two.active_paths == [0, 1]


@pytest.mark.parametrize("index", [2, 5, -1])
def test_an_out_of_range_dataset_in_the_file_is_refused(two, index: int) -> None:
    two.data["paths"]["active_paths"] = [0, index]
    with pytest.raises(ActivePathsError) as err:
        _ = two.active_paths
    text = str(err.value)
    assert f"Dataset {index}" in text and "paths.active_paths" in text
    assert "2 datasets" in text and "0 to 1" in text


@pytest.mark.parametrize("entry", ["a", 1.5, True, None])
def test_a_non_integer_dataset_in_the_file_is_refused(two, entry: Any) -> None:
    two.data["paths"]["active_paths"] = [entry]
    with pytest.raises(ActivePathsError, match=r"is not a dataset number.*2 datasets"):
        _ = two.active_paths


def test_an_empty_selection_in_the_file_is_refused(two) -> None:
    two.data["paths"]["active_paths"] = []
    with pytest.raises(ActivePathsError, match=r"selects no dataset.*2 datasets"):
        _ = two.active_paths


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0,x", r"Dataset 'x' in PIV_ACTIVE_PATHS '0,x' is not a dataset number"),
        ("1;2", r"Dataset '1;2' in PIV_ACTIVE_PATHS '1;2' is not a dataset number"),
        ("7", r"Dataset 7 in PIV_ACTIVE_PATHS '7' is out of range"),
        ("", r"PIV_ACTIVE_PATHS '' selects no dataset"),
    ],
)
def test_a_malformed_or_out_of_range_override_is_refused(
    two, monkeypatch: Any, value: str, expected: str
) -> None:
    """It used to be ignored (malformed) or dropped (out of range)."""
    monkeypatch.setenv(ACTIVE_PATHS_ENV, value)
    with pytest.raises(ActivePathsError, match=expected):
        _ = two.active_paths


@pytest.fixture
def routes(two, monkeypatch: Any):
    """A client on the two-dataset config, with a runner that records job starts."""
    from flask import Flask
    from test_run_piv_mask_preflight import RecordingRunner

    import pivtools_gui.app as gui_app
    from pivtools_gui.errors import register_json_error_handlers

    runner = RecordingRunner()
    monkeypatch.setattr(gui_app, "get_config", lambda *a, **k: two)
    monkeypatch.setattr(gui_app, "get_runner", lambda *a, **k: runner)
    app = Flask(__name__)
    register_json_error_handlers(app)
    app.register_blueprint(gui_app.api_bp)
    return app.test_client(), runner


@pytest.mark.parametrize("selection", [[5], [], [0, "x"]])
def test_run_piv_refuses_a_selection_that_names_no_dataset(routes, selection) -> None:
    client, runner = routes
    response = client.post("/backend/run_piv", json={"active_paths": selection})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "PIV_ACTIVE_PATHS" in error and "2 datasets" in error
    assert runner.started == []


def test_clear_output_refuses_a_negative_dataset_and_deletes_nothing(
    routes, two
) -> None:
    """[-1] used to clear the LAST dataset's output (Python's negative index)."""
    from pivtools_core.paths import get_data_paths

    client, _ = routes
    data_dir = get_data_paths(
        two.base_paths[1],
        two.num_frame_pairs,
        1,
        "instantaneous",
        use_uncalibrated=True,
    )["data_dir"]
    data_dir.mkdir(parents=True)
    (data_dir / "B00001.mat").write_bytes(b"vectors")
    response = client.post(
        "/backend/clear_output",
        json={"active_paths": [-1], "camera_numbers": [1], "mode": "instantaneous"},
    )
    assert response.status_code == 400
    assert "Dataset -1" in response.get_json()["error"]
    assert (data_dir / "B00001.mat").exists()


def test_check_output_exists_refuses_an_out_of_range_dataset(routes) -> None:
    client, _ = routes
    response = client.get("/backend/check_output_exists?active_paths=0,9")
    assert response.status_code == 400
    assert "Dataset 9" in response.get_json()["error"]
    # Control: no selection asks about nothing, as before.
    assert client.get("/backend/check_output_exists").get_json() == {
        "exists": False,
        "details": {},
    }


def test_validate_config_reports_the_refusal(two) -> None:
    from pivtools_core.validation import validate_config

    two.data["paths"]["active_paths"] = [5]
    ok, message, _warnings = validate_config(two)
    assert ok is False
    assert "Dataset 5 in paths.active_paths is out of range" in message


def test_the_cli_override_is_refused_the_same_way(two) -> None:
    """The CLI's own --active-paths parse used int() alone, with no range check."""
    from types import SimpleNamespace

    from pivtools_cli.cli import get_active_paths_from_args

    with pytest.raises(ActivePathsError, match=r"Dataset 9 in PIV_ACTIVE_PATHS '0,9'"):
        get_active_paths_from_args(SimpleNamespace(active_paths="0,9"), two)
    assert get_active_paths_from_args(SimpleNamespace(active_paths="1"), two) == [1]
    assert get_active_paths_from_args(SimpleNamespace(active_paths=None), two) == [0, 1]


def test_a_config_with_no_datasets_is_refused(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, NO_MASKING)
    cfg.data["paths"]["source_paths"] = []
    with pytest.raises(ActivePathsError, match=r"no datasets"):
        _ = cfg.active_paths
