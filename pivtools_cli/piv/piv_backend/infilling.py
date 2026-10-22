"""
Infilling methods for PIV velocity fields.

This module is the single home for filling invalid vectors on a PIV grid:

- Nearest-valid-neighbour infilling (Euclidean distance in physical units)
- Harmonic least-squares inpainting (min sum |grad u|^2)
- Biharmonic least-squares inpainting (min sum of squared second differences)
- K-nearest neighbors (KNN) regression
- Griddata linear interpolation
- Radial basis function (RBF) interpolation
- :func:`fill_masked_predictor`: filling masked windows of a multi-pass
  predictor field before it is padded, smoothed and interpolated

Every method takes the grid pitch ``spacing = (pitch_y, pitch_x)`` in pixels,
so distances and derivatives are measured in physical units. Rectangular
interrogation windows give anisotropic pitch (e.g. 8 x 48 px for 16x96 windows
at 50 % overlap); ignoring it treats nodes 48 px apart as if they were as close
as nodes 8 px apart. For square windows the pitch is uniform and the results
equal the grid-index results.

Every method also takes an optional ``exclude`` mask: excluded nodes are neither
filled nor used as donors, and are returned unchanged (used for masked windows,
whose stored zeros are not data). Non-finite values outside ``mask`` are treated
as excluded too, so a stray NaN is never a donor and is left in place.

Contract: every method (and :func:`apply_infilling`) either returns fields that
are finite at every masked cell, or raises ``ValueError`` when the input is too
degenerate to fill (insufficient or disconnected donor points, singular
system). There are no silent partial fills. :func:`apply_infilling`
additionally normalises outputs to float32.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator, griddata
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, splu

from pivtools_core.config import PREDICTOR_FILL_METHODS

try:
    from sklearn.neighbors import KNeighborsRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

Spacing = tuple[float, float]

# Least-squares inpainting orders: the penalised finite-difference order.
HARMONIC_ORDER = 1  # min sum (du/dx)^2 + (du/dy)^2 ; natural BC du/dn = 0
BIHARMONIC_ORDER = 2  # min sum (d2u/dx2)^2 + (d2u/dy2)^2 ; natural BC = linear extrap.
_DIFF_STENCILS: dict[int, tuple[float, ...]] = {
    HARMONIC_ORDER: (-1.0, 1.0),
    BIHARMONIC_ORDER: (1.0, -2.0, 1.0),
}
# Relative residual of the normal equations above which a least-squares solve is
# treated as numerically singular (well-posed systems solve to ~1e-12).
_LS_RESIDUAL_TOL = 1e-6
# Cap on (row, col) nodes listed in an unanchored-group error message.
_MAX_REPORTED_NODES = 20

INFILL_METHODS = (
    "nearest",
    "harmonic",
    "biharmonic",
    "knn",
    "griddata_linear",
    "rbf_local",
)


def _degenerate_msg(
    method: str, n_valid: int, n_required: int, mask: np.ndarray, shape
) -> str:
    return (
        f"Infilling method '{method}' cannot run: {n_valid} valid donor point(s) "
        f"available (requires >= {n_required}) for field of shape {tuple(shape)} "
        f"with {int(np.asarray(mask, dtype=bool).sum())} cell(s) to fill. "
        f"The field is degenerate -- check upstream correlation/outlier settings."
    )


def _check_spacing(spacing: Spacing) -> tuple[float, float]:
    """Validate ``spacing = (pitch_y, pitch_x)`` and return it as floats."""
    try:
        py, px = (float(s) for s in spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"spacing must be (pitch_y, pitch_x), got {spacing!r}"
        ) from exc
    if not (np.isfinite(py) and np.isfinite(px) and py > 0 and px > 0):
        raise ValueError(f"spacing must be positive and finite, got {spacing!r}")
    return py, px


def _effective_exclude(
    ux: np.ndarray, uy: np.ndarray, mask: np.ndarray, exclude: np.ndarray | None
) -> np.ndarray:
    """Excluded nodes plus any non-finite node outside ``mask`` (never donors)."""
    stray = ~mask & ~(np.isfinite(ux) & np.isfinite(uy))
    if exclude is None:
        return stray
    exclude = np.asarray(exclude, dtype=bool)
    if exclude.shape != mask.shape:
        raise ValueError(f"exclude shape {exclude.shape} != mask shape {mask.shape}")
    return exclude | stray


def _fill_nearest_stack(
    fields: np.ndarray,
    fill: np.ndarray,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Nearest-donor fill of a stack ``(K, ny, nx)`` sharing one fill pattern.

    The donor is the nearest node, in physical distance, that is neither filled
    nor excluded and is finite in every field. All K fields take the SAME donor,
    so vector components stay coherent. Returns float64; non-fill nodes are
    returned unchanged.
    """
    py, px = _check_spacing(spacing)
    fields = np.asarray(fields, dtype=np.float64)
    fill = np.asarray(fill, dtype=bool)
    exclude = np.zeros_like(fill) if exclude is None else np.asarray(exclude, bool)
    out = fields.copy()
    if not fill.any():
        return out
    donor = ~fill & ~exclude & np.all(np.isfinite(fields), axis=0)
    if not donor.any():
        raise ValueError(_degenerate_msg("nearest", 0, 1, fill, fill.shape))
    iy, ix = ndi.distance_transform_edt(
        ~donor, sampling=(py, px), return_distances=False, return_indices=True
    )
    out[:, fill] = fields[:, iy, ix][:, fill]
    return out


def _difference_rows(
    shape: tuple[int, int], spacing: tuple[float, float], order: int
) -> sp.csr_matrix:
    """Finite-difference operator of ``order`` along y and x, in physical units.

    One row per stencil position that fits inside the grid; an axis whose length
    is <= ``order`` contributes no rows. Each difference is divided by
    pitch**order so the penalised energy is in physical units.
    """
    stencil = _DIFF_STENCILS[order]
    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    rows, cols, vals = [], [], []
    r = 0
    for axis, h in ((0, spacing[0]), (1, spacing[1])):
        n = shape[axis]
        if n <= order:
            continue
        w = 1.0 / h**order
        sl = [slice(None), slice(None)]
        cs = []
        for j in range(order + 1):
            sl[axis] = slice(j, n - order + j)
            cs.append(idx[tuple(sl)].ravel())
        m = cs[0].size
        rr = r + np.arange(m)
        for c, coef in zip(cs, stencil):
            rows.append(rr)
            cols.append(c)
            vals.append(np.full(m, w * coef))
        r += m
    if r == 0:
        return sp.csr_matrix((0, shape[0] * shape[1]))
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r, shape[0] * shape[1]),
    )


def _fill_least_squares(
    fields: np.ndarray,
    fill: np.ndarray,
    spacing: Spacing,
    order: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares variational fill of a stack ``(K, ny, nx)``, known nodes fixed.

    Minimises the sum of squared finite differences of ``order`` (1 = harmonic,
    2 = biharmonic) in physical units over the fill nodes, with every other
    non-excluded node held at its value. Only difference equations touching at
    least one fill node are kept; equations touching an excluded node are
    dropped, which gives the same natural boundary condition at an excluded
    region as at the frame edge. All K right-hand sides share one sparse LU
    factorisation of the normal equations.

    Raises
    ------
    ValueError
        No known node, a non-finite known node, no difference equations, a group
        of fill nodes with no equation linking it to a known node, or a
        numerically singular system (``np.linalg.LinAlgError``).
    """
    if order not in _DIFF_STENCILS:
        raise ValueError(f"order must be one of {sorted(_DIFF_STENCILS)}, got {order}")
    py, px = _check_spacing(spacing)
    fields = np.asarray(fields, dtype=np.float64)
    fill = np.asarray(fill, dtype=bool)
    exclude = np.zeros_like(fill) if exclude is None else np.asarray(exclude, bool)
    out = fields.copy()
    if not fill.any():
        return out
    method = "harmonic" if order == HARMONIC_ORDER else "biharmonic"
    known = ~fill & ~exclude
    if not known.any():
        raise ValueError(_degenerate_msg(method, 0, 1, fill, fill.shape))
    if not np.all(np.isfinite(fields[:, known])):
        raise ValueError(
            f"Infilling method '{method}': non-finite values at known nodes; "
            f"the caller must put invalid nodes in the fill or exclude mask."
        )

    K, ny, nx = fields.shape
    L = _difference_rows((ny, nx), (py, px), order)
    if L.shape[0] == 0:
        raise ValueError(
            f"Infilling method '{method}': grid {fill.shape} is too small for "
            f"order-{order} differences along either axis."
        )
    f, k, e = fill.ravel(), known.ravel(), exclude.ravel()
    Lu = L[:, f].tocsr()
    touches_fill = np.diff(Lu.indptr) > 0
    touches_excl = np.diff(L[:, e].tocsr().indptr) > 0
    keep = touches_fill & ~touches_excl
    Lu = Lu[keep]
    Lk = L[:, k].tocsr()[keep]

    # Every group of coupled fill nodes must be tied to at least one known node,
    # otherwise its values are undetermined (the system is singular).
    A = (Lu.T @ Lu).tocsc()
    n_unknown = A.shape[0]
    n_groups, group = connected_components(A, directed=False)
    anchored_rows = np.diff(Lk.indptr) > 0
    anchored = np.zeros(n_groups, dtype=bool)
    anchored[np.unique(group[Lu[anchored_rows].indices])] = True
    if not anchored.all():
        sizes = np.bincount(group, minlength=n_groups)[~anchored]
        fill_rows, fill_cols = np.nonzero(fill)
        stuck = ~anchored[group]
        nodes = list(zip(fill_rows[stuck].tolist(), fill_cols[stuck].tolist()))
        raise ValueError(
            f"Infilling method '{method}': {int((~anchored).sum())} group(s) of "
            f"fill nodes (sizes {sizes.tolist()}) are not linked to any valid "
            f"node: every order-{order} difference stencil through them along x "
            f"or y touches an excluded (masked) or off-grid node. "
            f"(row, col) nodes: {nodes[:_MAX_REPORTED_NODES]}"
            f"{' ...' if len(nodes) > _MAX_REPORTED_NODES else ''}. Check the mask "
            f"for enclosed pockets, or use infill method 'nearest'."
        )

    U_k = fields[:, known].T  # (n_known, K)
    B = Lu.T @ (-(Lk @ U_k))  # (n_unknown, K)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X = splu(A).solve(np.asarray(B))
    except (RuntimeError, MatrixRankWarning) as exc:
        raise np.linalg.LinAlgError(
            f"Infilling method '{method}': singular least-squares system ({exc})"
        ) from exc
    X = np.asarray(X).reshape(n_unknown, K)
    resid = np.linalg.norm(A @ X - B) / max(np.linalg.norm(B), np.finfo(float).tiny)
    if not np.all(np.isfinite(X)) or resid > _LS_RESIDUAL_TOL:
        raise np.linalg.LinAlgError(
            f"Infilling method '{method}': numerically singular least-squares "
            f"system (relative residual {resid:.2e})."
        )
    out[:, fill] = X.T
    return out


def _as_stack(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(ux, np.float64), np.asarray(uy, np.float64)])


def infill_nearest(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill masked cells with the value of the nearest valid vector.

    Distance is Euclidean in physical units (``spacing``). A donor is valid
    only when it is not masked, not excluded, and both components are finite.
    BOTH components are taken from the SAME donor cell, so filled (ux, uy)
    values remain coherent vector pairs.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates cells to fill.
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled (returned
        unchanged), e.g. masked windows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays, float32.

    Raises
    ------
    ValueError
        If there are no valid donor cells to fill from.
    """
    mask = np.asarray(mask, dtype=bool)
    ux32 = np.asarray(ux, dtype=np.float32)
    uy32 = np.asarray(uy, dtype=np.float32)
    excl = _effective_exclude(ux32, uy32, mask, exclude)
    out = _fill_nearest_stack(_as_stack(ux32, uy32), mask, spacing, excl)
    return out[0].astype(np.float32), out[1].astype(np.float32)


def infill_harmonic(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill by harmonic least-squares inpainting in physical units.

    Minimises sum (du/dx)^2 + (du/dy)^2 over the masked cells with every other
    valid cell held fixed. Satisfies a maximum principle (filled values stay
    within the range of their donors) and extends flat (zero normal gradient)
    into regions open to the frame edge, so it is robust to noisy donors.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates cells to fill.
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays (float64).

    Raises
    ------
    ValueError
        If the fill is degenerate (see :func:`_fill_least_squares`).
    """
    mask = np.asarray(mask, dtype=bool)
    excl = _effective_exclude(ux, uy, mask, exclude)
    out = _fill_least_squares(_as_stack(ux, uy), mask, spacing, HARMONIC_ORDER, excl)
    return out[0], out[1]


def infill_biharmonic(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill by biharmonic least-squares inpainting in physical units.

    Minimises the sum of squared second differences (d2u/dx2)^2 + (d2u/dy2)^2
    over the masked cells with every other valid cell held fixed. Matches value
    and slope at hole edges (exact for bilinear fields, including holes open to
    the frame edge, where it extrapolates linearly). Amplifies donor noise more
    than :func:`infill_harmonic`.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates cells to fill (outliers only).
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays (float64).

    Raises
    ------
    ValueError
        If the fill is degenerate (see :func:`_fill_least_squares`).
    """
    mask = np.asarray(mask, dtype=bool)
    excl = _effective_exclude(ux, uy, mask, exclude)
    out = _fill_least_squares(_as_stack(ux, uy), mask, spacing, BIHARMONIC_ORDER, excl)
    return out[0], out[1]


def _training_set(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    spacing: Spacing,
    exclude: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Physical node coordinates (y, x) of valid donors and of mask cells."""
    py, px = _check_spacing(spacing)
    H, W = ux.shape
    yy, xx = np.mgrid[0:H, 0:W]
    yy = yy * py
    xx = xx * px
    valid = ~mask & ~_effective_exclude(ux, uy, mask, exclude)
    return yy, xx, valid, mask


def infill_griddata_linear(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    method: str = "linear",
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill using scipy griddata interpolation in physical coordinates.

    Uses Delaunay triangulation for linear/cubic interpolation with
    nearest-neighbor fallback for edge regions.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    method : str, optional
        Interpolation method: 'linear', 'cubic', or 'nearest', defaults to 'linear'.
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.

    Raises
    ------
    ValueError
        If there are no valid donor cells to interpolate from.
    """
    mask = np.asarray(mask, dtype=bool)
    yy, xx, valid, _ = _training_set(ux, uy, mask, spacing, exclude)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(_degenerate_msg("griddata_linear", n_valid, 1, mask, ux.shape))

    pts = np.c_[yy[valid], xx[valid]]

    ux_f = griddata(pts, ux[valid], (yy, xx), method=method)
    uy_f = griddata(pts, uy[valid], (yy, xx), method=method)

    # Fallback to nearest for anything left unfilled (edges)
    if np.isnan(ux_f).any() or np.isnan(uy_f).any():
        ux_nn = griddata(pts, ux[valid], (yy, xx), method="nearest")
        uy_nn = griddata(pts, uy[valid], (yy, xx), method="nearest")
        ux_f = np.where(np.isnan(ux_f), ux_nn, ux_f)
        uy_f = np.where(np.isnan(uy_f), uy_nn, uy_f)

    # Only replace the masked outliers, keep everything else
    ux_out = ux.copy()
    uy_out = uy.copy()
    ux_out[mask] = ux_f[mask]
    uy_out[mask] = uy_f[mask]

    return ux_out, uy_out


def infill_rbf_local(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    neighbors: int = 64,
    kernel: str = "thin_plate_spline",
    epsilon: float = None,
    smoothing: float = 0.0,
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill masked PIV vectors with local RBFs (memory-safe), physical coordinates.

    Uses radial basis function interpolation with local neighborhood
    support to keep memory usage reasonable for large fields.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    neighbors : int, optional
        Number of nearest neighbors used per query, defaults to 64.
        Typical range: 32-128.
    kernel : str, optional
        RBF kernel type, defaults to "thin_plate_spline".
        Options: "thin_plate_spline", "multiquadric", "inverse_multiquadric",
        "cubic", "quintic", "linear", "gaussian".
    epsilon : float, optional
        Shape parameter in pixels (physical units). Auto-determined if None.
        TPS ignores epsilon.
    smoothing : float, optional
        Regularization parameter, defaults to 0.0.
        Try 1e-3 to 1e-1 if data is noisy.
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.

    Raises
    ------
    ValueError
        If there are fewer than 4 valid donor cells.
    """
    mask = np.asarray(mask, dtype=bool)
    yy, xx, valid, _ = _training_set(ux, uy, mask, spacing, exclude)
    n_valid = int(valid.sum())
    if n_valid < 4:
        raise ValueError(_degenerate_msg("rbf_local", n_valid, 4, mask, ux.shape))

    X_train = np.c_[xx[valid].ravel(), yy[valid].ravel()].astype(np.float64)
    # Only query points that need filling (the mask)
    X_query = np.c_[xx[mask].ravel(), yy[mask].ravel()].astype(np.float64)

    u_train = ux[valid].astype(np.float64)
    v_train = uy[valid].astype(np.float64)

    # Build local RBF models (KD-tree inside; memory ~ O(N))
    rbf_u = RBFInterpolator(
        X_train,
        u_train,
        kernel=kernel,
        epsilon=epsilon,
        neighbors=neighbors,
        smoothing=smoothing,
    )
    rbf_v = RBFInterpolator(
        X_train,
        v_train,
        kernel=kernel,
        epsilon=epsilon,
        neighbors=neighbors,
        smoothing=smoothing,
    )

    u_pred = rbf_u(X_query)
    v_pred = rbf_v(X_query)

    # Keep originals; fill only masked cells
    ux_filled = ux.copy()
    uy_filled = uy.copy()
    ux_filled[mask] = u_pred
    uy_filled[mask] = v_pred

    return ux_filled, uy_filled


def infill_knn(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 32,
    weights: str = "distance",
    algorithm: str = "kd_tree",
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill masked PIV vectors using K-nearest-neighbor regression.

    Uses scikit-learn's KNN regressor in physical coordinates for fast, local
    interpolation. Requires scikit-learn to be installed.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    n_neighbors : int, optional
        Number of neighbors for interpolation, defaults to 32.
        Typical range: 16-64.
    weights : str, optional
        Weighting scheme: "uniform" or "distance", defaults to "distance".
        "distance" usually gives smoother results.
    algorithm : str, optional
        Nearest neighbor search algorithm, defaults to "kd_tree".
        Options: "auto", "ball_tree", "kd_tree", "brute".
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If there are no valid donor cells to interpolate from.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for KNN infilling. "
            "Install with: pip install scikit-learn"
        )

    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    # Early exit if nothing to fill
    if not np.any(mask):
        return ux, uy

    yy, xx, valid, _ = _training_set(ux, uy, mask, spacing, exclude)

    X_train = np.c_[xx[valid], yy[valid]]
    X_query = np.c_[xx[mask], yy[mask]]

    # Clamp n_neighbors to avoid errors
    n_valid = X_train.shape[0]
    if n_valid < 1:
        raise ValueError(_degenerate_msg("knn", n_valid, 1, mask, ux.shape))
    if n_valid < n_neighbors:
        n_neighbors = n_valid

    # Fit single KNN model for both components (vectorized)
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        n_jobs=1,  # Always 1: runs inside Dask daemon workers which can't fork
    )

    # Stack u and v as multi-output targets
    y_train = np.column_stack([ux[valid], uy[valid]])
    knn.fit(X_train, y_train)

    # Single prediction call for both components
    predictions = knn.predict(X_query)

    ux_filled = ux.copy()
    uy_filled = uy.copy()
    ux_filled[mask] = predictions[:, 0]
    uy_filled[mask] = predictions[:, 1]

    return ux_filled, uy_filled


def apply_infilling(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    method_cfg: dict,
    *,
    spacing: Spacing,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the configured infilling method.

    This function dispatches to the appropriate infilling method based on
    the configuration dictionary.

    Contract: returns (ux, uy) that are finite at every masked cell and
    float32, or raises ValueError when the field is too degenerate to fill
    (insufficient valid donor points). No silent partial fills. Excluded cells
    (and non-finite cells outside ``mask``) are never donors and are returned
    unchanged.

    Parameters
    ----------
    ux, uy : np.ndarray
        Velocity components (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    method_cfg : dict
        Configuration dictionary with 'method' and 'parameters' keys.
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) in pixels. Required: every caller must
        state the vector-grid geometry.
    exclude : np.ndarray, optional
        Boolean mask of cells that are neither donors nor filled (e.g. masked
        windows, whose stored zeros are not data).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays, float32.

    Raises
    ------
    ValueError
        If an unknown infilling method is specified, or the input field is
        degenerate (insufficient valid donor points).
    """
    method = method_cfg.get("method", "nearest").lower()
    params = method_cfg.get("parameters", {})
    mask = np.asarray(mask, dtype=bool)
    geo = {"spacing": spacing, "exclude": exclude}

    if method == "nearest":
        ux_out, uy_out = infill_nearest(ux, uy, mask, **geo)

    elif method == "harmonic":
        ux_out, uy_out = infill_harmonic(ux, uy, mask, **geo)

    elif method == "knn":
        n_neighbors = params.get("n_neighbors", 32)
        weights = params.get("weights", "distance")
        algorithm = params.get("algorithm", "kd_tree")
        ux_out, uy_out = infill_knn(
            ux,
            uy,
            mask,
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            **geo,
        )

    elif method == "biharmonic":
        ux_out, uy_out = infill_biharmonic(ux, uy, mask, **geo)

    elif method == "griddata_linear":
        interp_method = params.get("method", "linear")
        ux_out, uy_out = infill_griddata_linear(
            ux, uy, mask, method=interp_method, **geo
        )

    elif method == "rbf_local":
        neighbors = params.get("neighbors", 64)
        kernel = params.get("kernel", "thin_plate_spline")
        epsilon = params.get("epsilon", None)
        smoothing = params.get("smoothing", 0.0)
        ux_out, uy_out = infill_rbf_local(
            ux,
            uy,
            mask,
            neighbors=neighbors,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
            **geo,
        )

    else:
        raise ValueError(
            f"Unknown infilling method: '{method}'. Valid methods: "
            f"{', '.join(repr(m) for m in INFILL_METHODS)}. "
            f"Note: 'local_median' has been removed; use 'nearest' instead."
        )

    return (
        np.asarray(ux_out, dtype=np.float32),
        np.asarray(uy_out, dtype=np.float32),
    )


def fill_masked_predictor(
    field: np.ndarray,
    mask: np.ndarray | None,
    spacing: Spacing,
    method: str,
) -> np.ndarray:
    """Fill masked windows of a multi-pass predictor field before it is used.

    Masked windows are stored as 0, which is not data: the flow under a mask is
    moving but unseen. Fed to the predictor, the zeros act as a fake step that
    padding, smoothing and the bicubic warp spread into the valid windows next
    to the mask. This replaces them with a fill from the valid windows.

    Parameters
    ----------
    field : np.ndarray
        Predictor field, channels last ``[uy, ux]``: shape ``(ny, nx, 2)``
        (ensemble) or ``(N, ny, nx, 2)`` (instantaneous, one field per pair).
    mask : np.ndarray or None
        Boolean ``(ny, nx)`` mask of masked windows, shared by all N fields.
    spacing : tuple[float, float]
        Grid pitch (pitch_y, pitch_x) of the predictor grid, in pixels.
    method : str
        ``'harmonic'`` (least-squares, robust to noise), ``'nearest'``
        (physical-distance donor) or ``'none'`` (leave zeros, old behaviour).

    Returns
    -------
    np.ndarray
        Field of the same shape and dtype with masked windows filled. The input
        is returned unchanged for ``'none'``, a ``None`` mask or an empty mask.

    Raises
    ------
    ValueError
        Unknown method, shape mismatch, non-finite values outside the mask, or a
        degenerate fill.
    """
    if method not in PREDICTOR_FILL_METHODS:
        raise ValueError(
            f"Unknown predictor fill method '{method}'. "
            f"Valid: {', '.join(repr(m) for m in PREDICTOR_FILL_METHODS)}."
        )
    if method == "none" or mask is None:
        return field
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return field
    field = np.asarray(field)
    if field.ndim not in (3, 4) or field.shape[-1] != 2:
        raise ValueError(
            f"predictor field must be (ny, nx, 2) or (N, ny, nx, 2), got {field.shape}"
        )
    ny, nx = field.shape[-3:-1]
    if mask.shape != (ny, nx):
        raise ValueError(f"mask shape {mask.shape} != predictor grid {(ny, nx)}")

    # (..., ny, nx, 2) -> (K, ny, nx) with K = 2 * N
    stack = np.moveaxis(field, -1, -3).reshape(-1, ny, nx)
    if not np.all(np.isfinite(stack[:, ~mask])):
        raise ValueError("predictor field has non-finite values outside the mask")
    if method == "harmonic":
        filled = _fill_least_squares(stack, mask, spacing, HARMONIC_ORDER)
    else:
        filled = _fill_nearest_stack(stack, mask, spacing)
    out = np.moveaxis(filled.reshape(field.shape[:-3] + (2, ny, nx)), -3, -1)
    return np.ascontiguousarray(out, dtype=field.dtype)
