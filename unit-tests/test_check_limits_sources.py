"""``/plot/check_limits`` (the viewer's Auto-Calculate) reads the files the view draws.

The dropdown sends a prefixed name (``inst:ux``, ``mean:ux``, ``inst_stat:uu_inst``,
``ens:ux``). The route used to look the prefixed name up as a field and only ever
sample instantaneous frame files, and the viewer sent neither ``is_uncalibrated``
nor ``type_name``, so it failed for every source in every mode. These tests build a
small run with the production writers, calibrated and uncalibrated, and send what
the viewer's ``fetchLimits`` now sends.

Each mode and source has its own value range, so a limit drawn from the wrong
file shows up as the wrong number:

* frames ``f = 1..4``: ``ux = f`` uncalibrated, ``10 f`` calibrated
* mean statistics: the mean of those, 2.5 and 25
* per-frame ``uu_inst = (ux - mean)^2``: 0.25..2.25 and 25..225
* ensemble: ``ux`` 50 and 500
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from flask import Flask

import pivtools_gui.plotting.app.plotting_views as PV
import pivtools_gui.plotting.app.shared_utils as SU
from pivtools_cli.piv.piv_result import (
    PIVEnsemblePassResult,
    PIVEnsembleResult,
    PIVPassResult,
    PIVResult,
)
from pivtools_cli.piv.save_results import (
    save_ensemble_result_distributed,
    save_piv_result_distributed,
)
from pivtools_core.paths import get_data_paths
from pivtools_gui.calibration.runio import _vectors_struct
from pivtools_gui.vector_statistics.instantaneous_statistics import (
    VectorStatisticsProcessor,
)

NUM_FRAME_PAIRS = 4
VECTOR_FORMAT = "%05d.mat"
GRID = (3, 4)
# ux per frame is FRAME_SCALE[mode] * frame number; the ensemble ux is ENSEMBLE_UX.
FRAME_SCALE = {"uncalibrated": 1.0, "calibrated": 10.0}
ENSEMBLE_UX = {"uncalibrated": 50.0, "calibrated": 500.0}


class _FakeConfig:
    def __init__(self, base):
        self.base_paths = [str(base)]
        self.num_frame_pairs = NUM_FRAME_PAIRS
        self.vector_format = VECTOR_FORMAT
        self.camera_numbers = [1]
        self.is_stereo_setup = False
        self.stereo_pairs = []


def _dir(base: Path, type_name: str, uncalibrated: bool) -> Path:
    return Path(
        get_data_paths(
            base_dir=base,
            num_frame_pairs=NUM_FRAME_PAIRS,
            cam=1,
            type_name=type_name,
            use_uncalibrated=uncalibrated,
        )["data_dir"]
    )


def _write_coordinates(folder: Path) -> None:
    coords = np.empty((1,), dtype=[("x", object), ("y", object)])
    coords["x"][0] = np.tile(np.arange(GRID[1], dtype=float), (GRID[0], 1))
    coords["y"][0] = np.tile(np.arange(GRID[0], dtype=float)[:, None], (1, GRID[1]))
    scipy.io.savemat(str(folder / "coordinates.mat"), {"coordinates": coords})


def _write_mode(base: Path, mode: str) -> None:
    """Frames, statistics and an ensemble result for one calibration mode."""
    uncalibrated = mode == "uncalibrated"
    frames = _dir(base, "instantaneous", uncalibrated)
    frames.mkdir(parents=True)
    for f in range(1, NUM_FRAME_PAIRS + 1):
        ux = np.full(GRID, FRAME_SCALE[mode] * f)
        b_mask = np.zeros(GRID, dtype=bool)
        if uncalibrated:
            result = PIVResult()
            result.add_pass(PIVPassResult(ux_mat=ux, uy_mat=ux, b_mask=b_mask))
            save_piv_result_distributed(
                result, frames, f, vector_fmt=VECTOR_FORMAT, save_mode="minimal"
            )
        else:
            # Calibration apply's writer (calibration/runio.py).
            scipy.io.savemat(
                str(frames / (VECTOR_FORMAT % f)),
                {"piv_result": _vectors_struct({0: (ux, ux, b_mask)}, 1)},
                oned_as="row",
            )
    _write_coordinates(frames)

    stats = VectorStatisticsProcessor(
        data_dir=frames,
        base_dir=base,
        num_frame_pairs=NUM_FRAME_PAIRS,
        vector_format=VECTOR_FORMAT,
        use_uncalibrated=uncalibrated,
    ).process(
        requested_statistics=["mean_velocity", "inst_stresses"], save_figures=False
    )
    assert stats["success"], stats

    ensemble = _dir(base, "ensemble", uncalibrated)
    ensemble.mkdir(parents=True)
    result = PIVEnsembleResult()
    ux = np.full(GRID, ENSEMBLE_UX[mode])
    result.add_pass(
        PIVEnsemblePassResult(ux_mat=ux, uy_mat=ux, b_mask=np.zeros(GRID, dtype=bool))
    )
    save_ensemble_result_distributed(result, ensemble)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("limits_run")
    for mode in FRAME_SCALE:
        _write_mode(base, mode)
    return base


@pytest.fixture
def client(run_dir, monkeypatch):
    monkeypatch.setattr(PV, "get_config", lambda: _FakeConfig(run_dir))
    monkeypatch.setattr(SU, "get_config", lambda: _FakeConfig(run_dir))
    app = Flask(__name__)
    app.register_blueprint(PV.vector_plot_bp, url_prefix="/plot")
    return app.test_client()


def _view_limits(client, base, var, mode, type_name):
    """The request the viewer's fetchLimits sends for the current view."""
    return client.get(
        "/plot/check_limits",
        query_string={
            "base_path": str(base),
            "camera": "1",
            "merged": "0",
            "is_uncalibrated": "1" if mode == "uncalibrated" else "0",
            "type_name": type_name,
            "var": var,
        },
    )


EXPECTED = {
    # (dropdown name, type_name of the view): (min, max) per mode
    ("inst:ux", "instantaneous"): {"uncalibrated": (1.0, 4.0), "calibrated": (10, 40)},
    ("mean:ux", "instantaneous"): {"uncalibrated": (2.5, 2.5), "calibrated": (25, 25)},
    ("inst_stat:uu_inst", "instantaneous"): {
        "uncalibrated": (0.25, 2.25),
        "calibrated": (25.0, 225.0),
    },
    ("ens:ux", "ensemble"): {"uncalibrated": (50, 50), "calibrated": (500, 500)},
}


@pytest.mark.parametrize("mode", sorted(FRAME_SCALE))
@pytest.mark.parametrize(("var", "type_name"), sorted(EXPECTED))
def test_limits_come_from_the_files_the_view_draws(
    client, run_dir, var, type_name, mode
):
    res = _view_limits(client, run_dir, var, mode, type_name)

    assert res.status_code == 200, res.get_data(as_text=True)
    payload = res.get_json()
    low, high = EXPECTED[(var, type_name)][mode]
    assert payload["min"] == pytest.approx(low)
    assert payload["max"] == pytest.approx(high)


def test_mean_limits_without_mean_stats_404_naming_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(PV, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(SU, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(PV.vector_plot_bp, url_prefix="/plot")

    res = _view_limits(
        app.test_client(), tmp_path, "mean:ux", "uncalibrated", "instantaneous"
    )

    assert res.status_code == 404
    assert "mean_stats.mat" in res.get_json()["error"]


def test_correlation_quality_figures_have_no_colour_range(client, run_dir):
    res = _view_limits(
        client, run_dir, "corrq:timeseries", "uncalibrated", "instantaneous"
    )

    assert res.status_code == 400
    assert "corrq" in res.get_json()["error"]


def test_an_unknown_source_prefix_is_refused(client, run_dir):
    res = _view_limits(client, run_dir, "bogus:ux", "uncalibrated", "instantaneous")

    assert res.status_code == 400
    assert "bogus" in res.get_json()["error"]
