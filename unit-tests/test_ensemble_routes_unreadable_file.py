"""Viewer routes that read ``ensemble_result.mat`` answer 400 naming a bad file.

``plot_ensemble``, ``get_vector_at_position`` (ensemble source) and ``check_runs``
(ensemble view) used to call ``scipy.io.loadmat`` inline, so a corrupt file raised
whatever scipy raised and the route answered 500 "Internal server error" with the
file nowhere in the message. They now read through
``vector_loading.load_result_mat``, whose ``ResultFileError`` names the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from flask import Flask

import pivtools_gui.plotting.app.plotting_views as PV
import pivtools_gui.plotting.app.shared_utils as SU
from pivtools_core.paths import get_data_paths

NUM_FRAME_PAIRS = 4
VECTOR_FORMAT = "%05d.mat"


class _FakeConfig:
    def __init__(self, base):
        self.base_paths = [str(base)]
        self.num_frame_pairs = NUM_FRAME_PAIRS
        self.vector_format = VECTOR_FORMAT
        self.camera_numbers = [1]
        self.is_stereo_setup = False
        self.stereo_pairs = []


def _ensemble_file(base: Path) -> Path:
    folder = Path(
        get_data_paths(
            base_dir=base,
            num_frame_pairs=NUM_FRAME_PAIRS,
            cam=1,
            type_name="ensemble",
            use_uncalibrated=True,
        )["data_dir"]
    )
    folder.mkdir(parents=True)
    return folder / "ensemble_result.mat"


def _write_corrupt(path: Path) -> None:
    path.write_bytes(b"not a MATLAB file, a half-written one")


def _write_without_struct(path: Path) -> None:
    scipy.io.savemat(str(path), {"something_else": np.ones((2, 2))})


ROUTES = {
    "plot_ensemble": ("/plot/plot_ensemble", {"var": "ens:ux"}),
    "get_vector_at_position": (
        "/plot/get_vector_at_position",
        {"var": "ens:ux", "var_source": "ens", "x_percent": "0.5", "y_percent": "0.5"},
    ),
    "check_runs": ("/plot/check_runs", {"var": "ux"}),
}


@pytest.mark.parametrize(
    ("writer", "expected"),
    [(_write_corrupt, "cannot be read"), (_write_without_struct, "ensemble_result")],
    ids=["corrupt", "no-ensemble_result-struct"],
)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_bad_ensemble_file_answers_400_naming_it(
    tmp_path, monkeypatch, route, writer, expected
):
    path = _ensemble_file(tmp_path)
    writer(path)
    monkeypatch.setattr(PV, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(SU, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(PV.vector_plot_bp, url_prefix="/plot")
    url, args = ROUTES[route]

    res = app.test_client().get(
        url,
        query_string={
            "base_path": str(tmp_path),
            "camera": "1",
            "merged": "0",
            "is_uncalibrated": "1",
            "type_name": "ensemble",
            **args,
        },
    )

    assert res.status_code == 400, res.get_data(as_text=True)
    error = res.get_json()["error"]
    assert str(path) in error
    assert expected in error
