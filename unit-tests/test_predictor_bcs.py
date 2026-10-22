"""Predictor boundary conditions (BCs): config validation, padded-grid layout, override.

Covers
  * ``pivtools_core.config._validate_predictor_boundary_conditions`` via both the
    ensemble and instantaneous properties (same schema, same failures).
  * ``CrossCorrelator._cache_window_padding_unified``: a BC inside the measured
    grid must NOT insert a row (the predictor is only edge-padded at its ends, so
    an interior row would shift every later measured row by one); a BC in the
    padding zone is still inserted; ``y_position > H - 1`` raises.
  * ``CrossCorrelator._apply_predictor_boundary_conditions``: identical result on
    ensemble ``(ny, nx, 2)`` and instantaneous ``(N, ny, nx, 2)`` fields, correct
    rows for top/bottom, alignment guard, empty-list no-op.
"""

import logging

import numpy as np
import pytest

from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU
from pivtools_core.config import Config

# Synthetic geometry. Window centres 16..112 step 8 give padded y rows
# [0, 8, 16, ..., 112, 119, 127] (n_pre = n_post = 2) on a 128 px image.
IMG_H = 128
IMG_W = 128
SPACING = 8
FIRST_CTR = 16
LAST_CTR = 112
CTRS = np.arange(FIRST_CTR, LAST_CTR + 1, SPACING, dtype=np.float64)
N_CTRS = CTRS.size
WINDOW = (16, 16)

# The pre-fix code inserted an interior row (not within 0.5 px of a grid row)
# and counted it on the nearer side: n_post + 1 below the grid midpoint, else
# n_pre + 1. Rows the override does not touch were then shifted by one in two
# cases, which these positions exercise.
GRID_MID = 0.5 * (FIRST_CTR + LAST_CTR)
# Top BC in the lower half (y_img = 84, between rows 80 and 88).
TOP_INSIDE_Y = 84
# Bottom BC in the upper half (y_img = 127 - 83 = 44, between rows 40 and 48).
BOTTOM_INSIDE_Y = 83
# 4 px from either edge lies in the padding zone (outside [16, 112]).
PADDING_ZONE_Y = 4

SECTIONS = (
    ("ensemble_piv", "ensemble_predictor_boundary_conditions"),
    ("instantaneous_piv", "instantaneous_predictor_boundary_conditions"),
)


def _config(data: dict) -> Config:
    """Build a Config without a YAML file and with a fixed image shape."""
    cfg = Config.__new__(Config)
    cfg.data = data
    cfg._detected_image_shape = (IMG_H, IMG_W)
    return cfg


def _window_fn(pass_idx: int, config: Config) -> tuple:
    """Fixed window-centre layout: (spacing_x, spacing_y, ctrs_x, ctrs_y, padding)."""
    return SPACING, SPACING, CTRS.copy(), CTRS.copy(), (0, 0, 0, 0)


def _correlator(bcs: list) -> EnsembleCorrelatorCPU:
    """Run the single-pass padding cache on a bare correlator with ``bcs``."""
    corr = EnsembleCorrelatorCPU.__new__(EnsembleCorrelatorCPU)
    corr.H, corr.W = IMG_H, IMG_W
    corr._cache_window_padding_unified(
        _config({}),
        [WINDOW],
        "square",
        _window_fn,
        predictor_boundary_conditions=bcs,
    )
    return corr


def _bc(y: int, edge: str, ux: float = 0.0, uy: float = 0.0) -> dict:
    return {"y_position": y, "ux": ux, "uy": uy, "edge": edge}


# ---------------------------------------------------------------- config validator


BAD_BC_VALUES = [
    pytest.param({"a": 1}, id="non-list-dict"),
    pytest.param(None, id="none"),
    pytest.param("[]", id="string"),
    pytest.param([42], id="non-dict-entry"),
    pytest.param([{"ux": 0.0}], id="missing-y"),
    pytest.param([{"y_position": -1}], id="negative-y"),
    pytest.param([{"y_position": 1.5}], id="non-integer-float-y"),
    pytest.param([{"y_position": True}], id="bool-y"),
    pytest.param([{"y_position": 3, "uz": 0.0}], id="unknown-key"),
    pytest.param([{"y_position": 3, "edge": "left"}], id="bad-edge"),
    pytest.param([{"y_position": 3, "ux": "fast"}], id="non-numeric-ux"),
    pytest.param([{"y_position": 3, "ux": float("nan")}], id="nan-ux"),
    pytest.param([{"y_position": 3, "uy": float("inf")}], id="inf-uy"),
    pytest.param([{"y_position": float("nan")}], id="nan-y"),
    pytest.param([{"y_position": float("inf")}], id="inf-y"),
]


@pytest.mark.parametrize("section, prop", SECTIONS)
@pytest.mark.parametrize("raw", BAD_BC_VALUES)
def test_validator_rejects_bad_entries(section, prop, raw):
    cfg = _config({section: {"predictor_boundary_conditions": raw}})
    with pytest.raises(ValueError, match=f"{section}.predictor_boundary_conditions"):
        getattr(cfg, prop)


@pytest.mark.parametrize("section, prop", SECTIONS)
def test_validator_accepts_and_normalises(section, prop):
    raw = [
        {"y_position": 41.0},
        {"y_position": 7, "ux": 2, "uy": -0.5, "edge": "top"},
    ]
    cfg = _config({section: {"predictor_boundary_conditions": raw}})
    got = getattr(cfg, prop)
    assert got == [
        {"y_position": 41, "ux": 0.0, "uy": 0.0, "edge": "bottom"},
        {"y_position": 7, "ux": 2.0, "uy": -0.5, "edge": "top"},
    ]
    assert isinstance(got[0]["y_position"], int)


@pytest.mark.parametrize("section, prop", SECTIONS)
def test_validator_default_empty(section, prop):
    assert getattr(_config({section: {}}), prop) == []


# ---------------------------------------------------------------- padded grid layout


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(_bc(TOP_INSIDE_Y, "top"), id="top-inside"),
        pytest.param(_bc(BOTTOM_INSIDE_Y, "bottom"), id="bottom-inside"),
    ],
)
def test_inside_grid_bc_preserves_alignment(bc, caplog):
    with caplog.at_level(logging.WARNING):
        corr = _correlator([bc])
    ctrs_all = corr.win_ctrs_y_all[0]
    n_pre = corr.n_pre_all[0][0]
    n_post = corr.n_post_all[0][0]
    assert len(ctrs_all) == n_pre + N_CTRS + n_post
    np.testing.assert_array_equal(ctrs_all[n_pre : n_pre + N_CTRS], CTRS)
    assert (n_pre, n_post) == (2, 2)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "within the measured grid" in warnings[0].getMessage()
    assert corr.predictor_bcs == [bc]


@pytest.mark.parametrize(
    "edge, y_img, pre_post",
    [
        ("bottom", IMG_H - 1 - PADDING_ZONE_Y, (2, 3)),
        ("top", PADDING_ZONE_Y, (3, 2)),
    ],
)
def test_padding_zone_bc_inserted(edge, y_img, pre_post):
    corr = _correlator([_bc(PADDING_ZONE_Y, edge)])
    ctrs_all = corr.win_ctrs_y_all[0]
    n_pre = corr.n_pre_all[0][0]
    n_post = corr.n_post_all[0][0]
    assert (n_pre, n_post) == pre_post
    assert np.any(np.isclose(ctrs_all, y_img))
    assert np.all(np.diff(ctrs_all) >= 0)
    assert len(ctrs_all) == n_pre + N_CTRS + n_post
    np.testing.assert_array_equal(ctrs_all[n_pre : n_pre + N_CTRS], CTRS)


@pytest.mark.parametrize("edge", ["bottom", "top"])
def test_y_position_beyond_image_raises(edge):
    with pytest.raises(ValueError, match="outside the image"):
        _correlator([_bc(IMG_H, edge)])


# y_position 120 from either edge lands 7 px from the opposite edge of the image,
# beyond the far side of the measured grid [16, 112]: the override would replace
# every predictor row, so it must raise instead of silently doing so.
FAR_SIDE_Y = 120


@pytest.mark.parametrize("edge", ["bottom", "top"])
def test_far_side_bc_raises(edge):
    with pytest.raises(ValueError, match="far side of the measured grid"):
        _correlator([_bc(FAR_SIDE_Y, edge)])


# ---------------------------------------------------------------- runtime override


def _bare(bcs: list) -> EnsembleCorrelatorCPU:
    corr = EnsembleCorrelatorCPU.__new__(EnsembleCorrelatorCPU)
    corr.H = IMG_H
    corr.predictor_bcs = bcs
    return corr


PADDED_Y = np.concatenate([[0.0, 8.0], CTRS, [119.0, 127.0]]).astype(np.float32)
NX_PAD = 5


def _field(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((PADDED_Y.size, NX_PAD, 2)).astype(np.float32)


@pytest.mark.parametrize(
    "bc, expected_rows",
    [
        pytest.param(
            _bc(TOP_INSIDE_Y, "top", ux=1.5, uy=-0.25),
            PADDED_Y <= TOP_INSIDE_Y,
            id="top",
        ),
        pytest.param(
            _bc(BOTTOM_INSIDE_Y, "bottom", ux=-2.0, uy=0.75),
            PADDED_Y >= IMG_H - 1 - BOTTOM_INSIDE_Y,
            id="bottom",
        ),
    ],
)
def test_apply_sets_correct_rows_2d_and_stacked(bc, expected_rows):
    rng = np.random.default_rng(0)
    base = _field(rng)
    corr = _bare([bc])

    single = base.copy()
    corr._apply_predictor_boundary_conditions(single, PADDED_Y, pass_idx=1)

    n_frames = 3
    stacked = np.stack([base] * n_frames)
    corr._apply_predictor_boundary_conditions(stacked, PADDED_Y, pass_idx=1)

    assert expected_rows.any() and not expected_rows.all()
    np.testing.assert_array_equal(single[expected_rows, :, 0], bc["uy"])
    np.testing.assert_array_equal(single[expected_rows, :, 1], bc["ux"])
    np.testing.assert_array_equal(single[~expected_rows], base[~expected_rows])
    for k in range(n_frames):
        np.testing.assert_array_equal(stacked[k], single)


def test_apply_alignment_guard():
    corr = _bare([_bc(TOP_INSIDE_Y, "top")])
    field = _field(np.random.default_rng(1))
    with pytest.raises(ValueError, match="rows"):
        corr._apply_predictor_boundary_conditions(field, PADDED_Y[:-1], pass_idx=2)


def test_apply_empty_is_noop():
    corr = _bare([])
    field = _field(np.random.default_rng(2))
    before = field.copy()
    corr._apply_predictor_boundary_conditions(field, PADDED_Y[:-1], pass_idx=0)
    np.testing.assert_array_equal(field, before)


# ---------------------------------------------------------------- figure


def _ux_profile(y: np.ndarray) -> np.ndarray:
    """Top-wall boundary-layer-like ux(y) [px/frame] used as the measured field."""
    u_inf, delta = 5.0, 80.0
    return u_inf * (1.0 - np.exp(-y / delta))


def test_top_bc_alignment_figure(make_figures, output_dir):
    """Old (interior row insert + end-only edge pad) vs fixed padded predictor."""
    bc = _bc(TOP_INSIDE_Y, "top")
    measured = np.zeros((N_CTRS, 1, 2), dtype=np.float32)
    measured[:, 0, 1] = _ux_profile(CTRS)

    # Fixed path: production padding cache + production override.
    corr = _correlator([bc])
    ctrs_new = corr.win_ctrs_y_all[0]
    n_pre, n_post = corr.n_pre_all[0][0], corr.n_post_all[0][0]
    field_new = np.pad(measured, ((n_pre, n_post), (0, 0), (0, 0)), mode="edge")
    corr._apply_predictor_boundary_conditions(field_new, ctrs_new, pass_idx=1)

    # Old path, simulated: interior row inserted into the coordinates and counted
    # on the nearer side (n_post + 1 here, since the BC is below the grid
    # midpoint), field padded only at its ends.
    assert TOP_INSIDE_Y > GRID_MID
    ctrs_old = np.insert(
        ctrs_new, np.searchsorted(ctrs_new, TOP_INSIDE_Y), float(TOP_INSIDE_Y)
    )
    field_old = np.pad(measured, ((n_pre, n_post + 1), (0, 0), (0, 0)), mode="edge")
    corr._apply_predictor_boundary_conditions(field_old, ctrs_old, pass_idx=1)

    # The fixed predictor equals the measured profile on every measured row
    # beyond the BC; the old one is shifted by one row there.
    beyond = CTRS > TOP_INSIDE_Y
    np.testing.assert_allclose(
        field_new[n_pre : n_pre + N_CTRS, 0, 1][beyond], measured[beyond, 0, 1]
    )
    assert np.all(field_new[ctrs_new <= TOP_INSIDE_Y, 0, 1] == 0.0)
    old_on_grid = np.interp(CTRS[beyond], ctrs_old, field_old[:, 0, 1])
    assert not np.allclose(old_on_grid, measured[beyond, 0, 1])

    if not make_figures:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_fine = np.linspace(0, IMG_H - 1, 400)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(_ux_profile(y_fine), y_fine, color="0.6", lw=1, label="true ux(y)")
    ax.plot(measured[:, 0, 1], CTRS, "ks", ms=5, mfc="none", label="measured rows")
    ax.plot(
        field_old[:, 0, 1],
        ctrs_old,
        "o--",
        color="tab:red",
        label="old: interior insert + end edge-pad (misaligned)",
    )
    ax.plot(
        field_new[:, 0, 1],
        ctrs_new,
        "^-",
        color="tab:blue",
        label="fix: no insert, override existing rows",
    )
    ax.axhline(
        TOP_INSIDE_Y, color="k", ls=":", lw=1, label=f"top BC y = {TOP_INSIDE_Y} px"
    )
    ax.invert_yaxis()
    ax.set_xlabel("padded predictor ux [px/frame]")
    ax.set_ylabel("row y (image, down) [px]")
    ax.set_title("Top predictor BC inside the measured grid: row alignment")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "test_predictor_bc_top_alignment.png", dpi=150)
    plt.close(fig)
