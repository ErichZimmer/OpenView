"""Tests for pivtools_cli.piv.piv_backend.infilling (physical-spacing gap fill).

The least-squares core is checked against a frozen copy of the verified
prototype (``_reference_fill_ls``), and the nearest fill against a frozen copy of
the old index-space implementation (``_reference_old_nearest``). Both references
are deliberately duplicated here, not imported, so a change to production maths
cannot silently move the reference with it.
"""

import cv2
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import distance_transform_edt, gaussian_filter

from pivtools_cli.piv.piv_backend.infilling import (
    BIHARMONIC_ORDER,
    HARMONIC_ORDER,
    INFILL_METHODS,
    _fill_least_squares,
    _fill_nearest_stack,
    apply_infilling,
    fill_masked_predictor,
    infill_biharmonic,
    infill_harmonic,
    infill_knn,
    infill_nearest,
    infill_rbf_local,
)

# 16x96 windows at 50 % overlap: the retro_tbl geometry that motivated the change.
RECT_PITCH = (8.0, 48.0)
UNIT_PITCH = (1.0, 1.0)
PROTOTYPE_TOL = 1e-12  # production must reproduce the prototype to round-off
EXACT_TOL = 1e-9  # bilinear fields are reproduced exactly up to solver round-off


# --------------------------------------------------------------------- references
def _reference_fill_ls(u, fill, spacing, order):
    """Frozen copy of the verified prototype fill_ls (scratchpad fill_methods.py)."""
    u = np.asarray(u, float)
    fill = np.asarray(fill, bool) | ~np.isfinite(u)
    known = ~fill
    ny, nx = u.shape
    py, px = spacing
    idx = np.arange(ny * nx).reshape(ny, nx)
    stencil = {1: (-1.0, 1.0), 2: (1.0, -2.0, 1.0)}[order]
    rows, cols, vals, r = [], [], [], 0
    for axis, h in ((0, py), (1, px)):
        if u.shape[axis] <= order:
            continue
        sl = [slice(None)] * 2
        cs = []
        for j in range(order + 1):
            sl[axis] = slice(j, u.shape[axis] - order + j)
            cs.append(idx[tuple(sl)].ravel())
        m = cs[0].size
        rr = r + np.arange(m)
        for c, w in zip(cs, stencil):
            rows.append(rr)
            cols.append(c)
            vals.append(np.full(m, w / h**order))
        r += m
    L = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r, ny * nx),
    )
    f, k = fill.ravel(), known.ravel()
    Lu, Lk = L[:, f], L[:, k]
    touch = np.diff(Lu.indptr) > 0
    Lu, Lk = Lu[touch], Lk[touch]
    x = spla.spsolve((Lu.T @ Lu).tocsc(), Lu.T @ (-(Lk @ u.ravel()[k])))
    out = u.copy().ravel()
    out[f] = x
    return out.reshape(ny, nx)


def _reference_old_nearest(ux, uy, mask):
    """Frozen copy of the pre-change index-space infill_nearest."""
    mask = np.asarray(mask, dtype=bool)
    ux_out = np.asarray(ux, dtype=np.float32).copy()
    uy_out = np.asarray(uy, dtype=np.float32).copy()
    invalid = mask | ~np.isfinite(ux_out) | ~np.isfinite(uy_out)
    idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    fx, fy = ux_out[tuple(idx)][mask], uy_out[tuple(idx)][mask]
    ux_out[mask], uy_out[mask] = fx, fy
    return ux_out, uy_out


def _random_case(rng, frac=0.15, edge_rows=True):
    ny, nx = int(rng.integers(20, 70)), int(rng.integers(10, 35))
    u = rng.normal(size=(ny, nx))
    fill = rng.random((ny, nx)) < frac
    if edge_rows:
        fill[: int(rng.integers(0, 4))] = True
    return u, fill


def _bilinear(shape, spacing):
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]] * np.array(spacing)[:, None, None]
    return 3.0 + 0.02 * X - 0.05 * Y + 1e-4 * X * Y


# ------------------------------------------------------- least-squares core
@pytest.mark.parametrize("spacing", [RECT_PITCH, UNIT_PITCH])
@pytest.mark.parametrize("order", [HARMONIC_ORDER, BIHARMONIC_ORDER])
def test_least_squares_matches_prototype(spacing, order):
    rng = np.random.default_rng(1234 + order)
    worst = 0.0
    for _ in range(15):
        u, fill = _random_case(rng)
        ref = _reference_fill_ls(u, fill, spacing, order)
        got = _fill_least_squares(u[None], fill, spacing, order)[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    assert worst <= PROTOTYPE_TOL


def test_harmonic_and_biharmonic_reproduce_bilinear():
    shape = (60, 25)
    u = _bilinear(shape, RECT_PITCH)
    holes = np.zeros(shape, bool)
    holes[20:35, 8:15] = True
    holes[45:48, 3:5] = True
    edge = holes.copy()
    edge[:4] = True  # strip open to the frame edge
    h, _ = infill_harmonic(u, u, holes, spacing=RECT_PITCH)
    b, _ = infill_biharmonic(u, u, edge, spacing=RECT_PITCH)
    assert np.abs(h - u).max() < EXACT_TOL
    assert np.abs(b - u).max() < EXACT_TOL


def test_least_squares_known_nodes_unchanged_and_stack_equals_per_field():
    rng = np.random.default_rng(7)
    u, fill = _random_case(rng)
    stack = np.stack([u, 2 * u + 1, -u])
    out = _fill_least_squares(stack, fill, RECT_PITCH, HARMONIC_ORDER)
    for k in range(3):
        single = _fill_least_squares(stack[k][None], fill, RECT_PITCH, HARMONIC_ORDER)
        assert np.abs(out[k] - single[0]).max() < PROTOTYPE_TOL
    assert np.array_equal(out[:, ~fill], stack[:, ~fill])


def test_uniform_scaling_equals_unit_spacing():
    rng = np.random.default_rng(3)
    u, fill = _random_case(rng)
    for order in (HARMONIC_ORDER, BIHARMONIC_ORDER):
        a = _fill_least_squares(u[None], fill, (5.0, 5.0), order)
        b = _fill_least_squares(u[None], fill, UNIT_PITCH, order)
        assert np.abs(a - b).max() < 1e-10


def test_exclude_nodes_are_not_donors_and_unchanged():
    rng = np.random.default_rng(11)
    u, fill = _random_case(rng, edge_rows=False)
    excl = np.zeros_like(fill)
    excl[:, :2] = True
    excl &= ~fill
    a = u.copy()
    b = u.copy()
    b[excl] = 1e6
    for order in (HARMONIC_ORDER, BIHARMONIC_ORDER):
        ra = _fill_least_squares(a[None], fill, RECT_PITCH, order, excl)[0]
        rb = _fill_least_squares(b[None], fill, RECT_PITCH, order, excl)[0]
        assert np.abs(ra[fill] - rb[fill]).max() < 1e-9
        assert np.array_equal(rb[excl], b[excl])
    na = _fill_nearest_stack(a[None], fill, RECT_PITCH, excl)[0]
    nb = _fill_nearest_stack(b[None], fill, RECT_PITCH, excl)[0]
    assert np.array_equal(na[fill], nb[fill])


def test_least_squares_raise_paths():
    u = np.ones((10, 10))
    allfill = np.ones((10, 10), bool)
    with pytest.raises(ValueError, match="valid donor"):
        _fill_least_squares(u[None], allfill, RECT_PITCH, HARMONIC_ORDER)
    # an isolated fill region walled off by excluded nodes has no anchor
    fill = np.zeros((10, 10), bool)
    fill[4:6, 4:6] = True
    excl = np.zeros((10, 10), bool)
    excl[3:7, 3:7] = True
    excl &= ~fill
    with pytest.raises(ValueError, match="not linked to any valid"):
        _fill_least_squares(u[None], fill, RECT_PITCH, HARMONIC_ORDER, excl)
    bad = u.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _fill_least_squares(bad[None], fill, RECT_PITCH, HARMONIC_ORDER)
    with pytest.raises(ValueError, match="too small"):
        _fill_least_squares(
            np.ones((1, 1, 2)),
            np.array([[True, False]]),
            RECT_PITCH,
            BIHARMONIC_ORDER,
        )
    with pytest.raises(ValueError, match="spacing"):
        _fill_least_squares(u[None], fill, (0.0, 1.0), HARMONIC_ORDER)


# --------------------------------------------------------------- nearest
def test_nearest_unit_spacing_bit_identical_to_old():
    rng = np.random.default_rng(5)
    ux = rng.normal(size=(60, 30)).astype(np.float32)
    uy = rng.normal(size=(60, 30)).astype(np.float32)
    mask = rng.random((60, 30)) < 0.2
    ux[5, 5] = np.nan  # stray NaN outside the mask: never a donor, left in place
    mask[5, 5] = False
    o1, o2 = _reference_old_nearest(ux, uy, mask)
    n1, n2 = infill_nearest(ux, uy, mask, spacing=UNIT_PITCH)
    assert n1.dtype == np.float32 and n2.dtype == np.float32
    assert np.array_equal(o1, n1, equal_nan=True)
    assert np.array_equal(o2, n2, equal_nan=True)


def test_nearest_uses_physical_distance():
    # Target at (5, 5). Donors: 3 rows below (24 px) and 1 column right (48 px).
    ux = np.zeros((12, 12), np.float32)
    mask = np.ones((12, 12), bool)
    mask[8, 5] = False
    mask[5, 6] = False
    ux[8, 5] = 1.0  # physically nearer (3 x 8 px = 24 px)
    ux[5, 6] = 2.0  # index-nearer (1 node) but 48 px away
    out_phys, _ = infill_nearest(ux, ux, mask, spacing=RECT_PITCH)
    out_idx, _ = infill_nearest(ux, ux, mask, spacing=UNIT_PITCH)
    assert out_phys[5, 5] == 1.0
    assert out_idx[5, 5] == 2.0


def test_nearest_components_share_donor():
    rng = np.random.default_rng(9)
    ux = rng.normal(size=(30, 20)).astype(np.float32)
    uy = ux * 3.0 + 1.0
    mask = rng.random((30, 20)) < 0.3
    fx, fy = infill_nearest(ux, uy, mask, spacing=RECT_PITCH)
    np.testing.assert_allclose(fy[mask], fx[mask] * 3.0 + 1.0, rtol=1e-6)


# ------------------------------------------------- public dispatcher contract
@pytest.mark.parametrize("method", INFILL_METHODS)
def test_every_method_fills_and_returns_float32(method):
    rng = np.random.default_rng(21)
    u = _bilinear((40, 25), RECT_PITCH) + 0.01 * rng.normal(size=(40, 25))
    mask = rng.random((40, 25)) < 0.1
    excl = np.zeros_like(mask)
    excl[:, 0] = True
    excl &= ~mask
    u_in = u.copy()
    u_in[excl] = 0.0
    ox, oy = apply_infilling(
        u_in, u_in, mask, {"method": method}, spacing=RECT_PITCH, exclude=excl
    )
    assert ox.dtype == np.float32 and oy.dtype == np.float32
    assert np.all(np.isfinite(ox[mask])) and np.all(np.isfinite(oy[mask]))
    assert np.array_equal(ox[excl], u_in[excl].astype(np.float32))
    assert np.array_equal(ox[~mask & ~excl], u_in[~mask & ~excl].astype(np.float32))


def test_spacing_is_required():
    u = np.zeros((5, 5))
    m = np.zeros((5, 5), bool)
    with pytest.raises(TypeError):
        apply_infilling(u, u, m, {"method": "nearest"})
    with pytest.raises(TypeError):
        infill_nearest(u, u, m)


def test_unknown_method_raises():
    u = np.zeros((5, 5))
    with pytest.raises(ValueError, match="Unknown infilling method"):
        apply_infilling(
            u, u, np.zeros((5, 5), bool), {"method": "bogus"}, spacing=(1, 1)
        )


def test_knn_and_rbf_use_physical_coordinates():
    # Same donor layout as test_nearest_uses_physical_distance: a 1-neighbour KNN
    # must pick the physically nearer donor.
    ux = np.zeros((12, 12))
    mask = np.zeros((12, 12), bool)
    mask[5, 5] = True
    excl = np.ones((12, 12), bool)
    excl[8, 5] = excl[5, 6] = False
    excl[5, 5] = False
    ux[8, 5], ux[5, 6] = 1.0, 2.0
    kx, _ = infill_knn(ux, ux, mask, n_neighbors=1, spacing=RECT_PITCH, exclude=excl)
    assert kx[5, 5] == pytest.approx(1.0)
    # RBF on a linear field in physical coordinates is exact (TPS reproduces linear)
    u = _bilinear((30, 20), RECT_PITCH) - 1e-4 * np.prod(
        np.mgrid[0:30, 0:20] * np.array(RECT_PITCH)[:, None, None], axis=0
    )
    m = np.zeros((30, 20), bool)
    m[12:15, 8:10] = True
    rx, _ = infill_rbf_local(u, u, m, spacing=RECT_PITCH)
    assert np.abs(rx[m] - u[m]).max() < 1e-6


def test_griddata_uniform_scaling_matches_index_space():
    rng = np.random.default_rng(2)
    u = _bilinear((30, 20), UNIT_PITCH)
    mask = rng.random((30, 20)) < 0.15
    a, _ = apply_infilling(
        u, u, mask, {"method": "griddata_linear"}, spacing=(4.0, 4.0)
    )
    b, _ = apply_infilling(
        u, u, mask, {"method": "griddata_linear"}, spacing=UNIT_PITCH
    )
    assert np.abs(a[mask] - b[mask]).max() < 1e-4


# ------------------------------------------------------- predictor fill
def test_fill_masked_predictor_shapes_dtype_and_none():
    rng = np.random.default_rng(4)
    ny, nx, N = 40, 20, 3
    mask = np.zeros((ny, nx), bool)
    mask[:, 0] = True
    mask[-4:] = True
    f3 = rng.normal(size=(ny, nx, 2)).astype(np.float32)
    f4 = rng.normal(size=(N, ny, nx, 2)).astype(np.float32)
    f3[mask] = 0
    f4[:, mask] = 0
    for method in ("harmonic", "nearest"):
        o3 = fill_masked_predictor(f3, mask, RECT_PITCH, method)
        o4 = fill_masked_predictor(f4, mask, RECT_PITCH, method)
        assert o3.shape == f3.shape and o3.dtype == np.float32
        assert o4.shape == f4.shape and o4.dtype == np.float32
        assert np.array_equal(o3[~mask], f3[~mask])
        per = np.stack(
            [fill_masked_predictor(f4[i], mask, RECT_PITCH, method) for i in range(N)]
        )
        assert np.abs(o4 - per).max() < 1e-6
        assert np.any(o3[mask] != 0)
    assert fill_masked_predictor(f3, mask, RECT_PITCH, "none") is f3
    assert fill_masked_predictor(f3, None, RECT_PITCH, "harmonic") is f3
    assert fill_masked_predictor(f3, np.zeros_like(mask), RECT_PITCH, "harmonic") is f3
    with pytest.raises(ValueError, match="Unknown predictor fill"):
        fill_masked_predictor(f3, mask, RECT_PITCH, "biharmonic")
    with pytest.raises(ValueError, match="mask shape"):
        fill_masked_predictor(f3, mask[:-1], RECT_PITCH, "harmonic")
    bad = f3.copy()
    bad[5, 5, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fill_masked_predictor(bad, mask, RECT_PITCH, "harmonic")


def _predictor_through_warp_chain(field, upsample=4):
    """Edge-pad, Gaussian-smooth and bicubic-remap one channel, as the predictor is."""
    pad = 2
    p = np.pad(field, pad, mode="edge")
    p = gaussian_filter(p, sigma=1.0, mode="nearest")
    ny, nx = field.shape
    yq = np.linspace(0, ny - 1, (ny - 1) * upsample + 1) + pad
    xq = np.linspace(0, nx - 1, (nx - 1) * upsample + 1) + pad
    mx, my = np.meshgrid(xq.astype(np.float32), yq.astype(np.float32))
    return cv2.remap(
        p.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def test_predictor_fill_removes_mask_edge_bias(make_figures, output_dir):
    # Uniform flow ux = 10 px with the first two columns masked (stored as 0).
    ny, nx = 40, 20
    truth = np.full((ny, nx), 10.0, np.float32)
    mask = np.zeros((ny, nx), bool)
    mask[:, :2] = True
    field = np.stack([np.zeros_like(truth), truth], axis=-1)
    field[mask] = 0.0
    up = 4
    bias = {}
    for method in ("none", "nearest", "harmonic"):
        filled = fill_masked_predictor(field, mask, RECT_PITCH, method)
        warped = _predictor_through_warp_chain(filled[..., 1], up)
        bias[method] = warped.mean(axis=0) - 10.0
    first_valid = 2 * up  # interpolated column index of the first unmasked window
    assert abs(bias["none"][first_valid]) > 1.0  # the zero step leaks in
    assert abs(bias["harmonic"][first_valid]) < 1e-4
    assert abs(bias["nearest"][first_valid]) < 1e-4
    if make_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(bias["none"].size) / up
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, b in bias.items():
            ax.plot(x, b, label=f"predictor_fill = {method}")
        ax.axvspan(-0.5, 1.5, color="0.85", label="masked windows")
        ax.set_xlabel("window column")
        ax.set_ylabel("predictor ux bias after pad + smooth + bicubic remap [px]")
        ax.set_title("Masked-window zeros leak into the predictor; filling removes it")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "infilling_predictor_mask_edge_bias.png", dpi=110)
        plt.close(fig)
