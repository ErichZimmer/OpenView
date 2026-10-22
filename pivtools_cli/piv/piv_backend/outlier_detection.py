"""
Outlier detection methods for PIV velocity fields.

This module provides various methods for detecting outliers in PIV data:
- Peak magnitude thresholding
- Median-based 2D outlier detection
- Sigma-based outlier detection
- Divergence/vorticity-based outlier detection

Masked vector windows (stored as 0 by both pipelines) are excluded from every
test via the ``masked`` argument of ``apply_outlier_detection``.
"""

import bottleneck as bn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi
from scipy.signal import convolve2d

# Border/hole guard of median_outlier_detection. The guard window is the
# immediate 3x3 around a node (the node itself included) and is deliberately
# independent of the median window ``size``. A node is rejected when fewer
# than GUARD_MIN_PRESENT of those 9 cells hold a present vector.
GUARD_WINDOW = 3
GUARD_MIN_PRESENT = 6


def _as_mask(masked: np.ndarray, shape: tuple) -> np.ndarray:
    """Validate a vector-window mask against the field shape and cast to bool.

    Parameters
    ----------
    masked : np.ndarray
        Mask of masked vector windows (True = masked).
    shape : tuple
        Expected shape (that of the velocity field).

    Returns
    -------
    np.ndarray
        Boolean mask of ``shape``.

    Raises
    ------
    ValueError
        If ``masked`` does not have ``shape``.
    """
    masked = np.asarray(masked)
    if masked.shape != tuple(shape):
        raise ValueError(
            f"masked shape {masked.shape} does not match field shape {tuple(shape)}"
        )
    return masked.astype(bool, copy=False)


def peak_magnitude_detection(
    peak_mag: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """
    Detect outliers based on peak magnitude threshold.

    Vectors with peak magnitudes below the threshold are marked as outliers.

    Parameters
    ----------
    peak_mag : np.ndarray
        Peak magnitude array from PIV correlation.
    threshold : float, optional
        Minimum acceptable peak magnitude, defaults to 0.2 (matching the
        shipped config template). This signature default is not what a
        pipeline run uses: the only production call site passes ``threshold``
        explicitly from the config method dict, and that read falls back to
        0.5 when the ``peak_mag`` entry omits a ``threshold`` key
        (``detect_outliers`` below). A config that names the method without a
        threshold therefore gets 0.5, not the 0.2 documented here.

    Returns
    -------
    np.ndarray
        Boolean array indicating outliers (True = outlier).
    """
    b_filter = (peak_mag < threshold) | np.isnan(peak_mag)
    return b_filter


def median_outlier_detection(
    ux: np.ndarray,
    uy: np.ndarray,
    epsilon: float = 0.1,
    threshold: float = 2.0,
    size: int = 5,
    combine: str = "norm",
    masked: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalized-median (universal) outlier detection for 2D PIV velocity
    fields, replicating Westerweel & Scarano's published PIVware
    ``pwValidate`` (nan-aware via bottleneck).

    Each vector is compared to the component-wise median of its ``size``x
    ``size`` neighbourhood (the vector itself excluded). The residual is the
    **vector norm** of that difference,
    ``r0 = sqrt((ux - med_u)^2 + (uy - med_v)^2)`` (PIVware ``pwVectorNorm``),
    a single scalar per node. It is normalized by the neighbourhood median of
    the neighbour residual norms plus ``epsilon`` and compared against
    ``threshold`` once. Because the vector is judged as a whole -- rather than
    each component being tested and OR'd -- a large but legitimate fluctuation
    in the quiet (wall-normal) component is not flagged on its own. This
    matters most in high-shear regions such as the near-wall boundary layer,
    where a per-component test over-rejects valid vectors.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    epsilon : float, optional
        Regularization term (px) for the normalization, defaults to 0.1
        (PIVware ``EPS``).
    threshold : float, optional
        Normalized-residual detection level, defaults to 2.0
        (PIVware ``LEVEL``).
    size : int, optional
        Side length of the square neighbourhood; must be an odd integer >= 3.
        Defaults to 5 (PIVware ``SIZE``). A smaller window (3) keeps shear from
        spreading across the neighbourhood; a larger one gives smoother
        statistics.
    combine : {"norm", "per_component"}, optional
        How the two input fields are combined into a verdict. Defaults to
        ``"norm"`` — the PIVware vector-norm form described above, correct when
        the inputs are the components of a genuine vector.

        ``"per_component"`` instead thresholds each field independently against
        its own neighbourhood scale and ORs the two masks. Use it when the
        inputs are NOT vector components — notably the ensemble stress pair
        ``(UU, VV)``, which are independent variances from one LM fit. There the
        norm is dominated by the larger field (measured: ``|dUU|`` 0.069 vs
        ``|dVV|`` 0.031 px^2, so VV carries only 17 % of ``r_0**2``) and hides
        40 % of the VV failures and 26 % of the UU failures a per-component test
        finds. Velocity does not have this problem — its components share a
        single failure mode (one bad correlation peak corrupts both), so the
        norm misses nothing there and ``"norm"`` remains correct for ``(ux, uy)``.
    masked : np.ndarray or None, optional
        Boolean mask of masked vector windows (True = masked), same shape as
        ``ux``. Masked nodes are set to NaN internally, so they drop out of
        every neighbourhood median and residual median, and they are never
        flagged. In the 3x3 border/hole guard a masked neighbour counts as
        PRESENT, otherwise every node along a curved or stair-step mask edge
        would fall below ``GUARD_MIN_PRESENT`` and be auto-flagged. A node
        whose entire neighbourhood is masked or missing has a NaN median and
        is still flagged. ``None`` (default) leaves the behaviour unchanged.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier). Never True at masked nodes.

    Raises
    ------
    ValueError
        If ``combine`` or ``size`` is invalid, or ``ux``, ``uy`` and
        ``masked`` shapes disagree.
    """
    if combine not in ("norm", "per_component"):
        raise ValueError(f"combine must be 'norm' or 'per_component', got {combine!r}")
    if ux.shape != uy.shape:
        raise ValueError("ux and uy must have identical shapes")
    if not isinstance(size, (int, np.integer)) or size < 3 or size % 2 == 0:
        raise ValueError(f"size must be an odd integer >= 3, got {size!r}")

    ux = ux.astype(np.float32, copy=False)
    uy = uy.astype(np.float32, copy=False)
    if masked is not None:
        masked = _as_mask(masked, ux.shape)
        ux = np.where(masked, np.float32(np.nan), ux)
        uy = np.where(masked, np.float32(np.nan), uy)

    def _neighbours(U):
        """Stack the size*size-1 neighbours of every node (self excluded,
        NaN outside the grid)."""
        pad = size // 2
        U_pad = np.pad(U, pad, mode="constant", constant_values=np.nan)
        win = sliding_window_view(U_pad, (size, size))  # (H, W, size, size)
        flat = win.reshape(U.shape[0], U.shape[1], size * size)
        centre = (size * size) // 2  # flat index of the node itself
        return np.concatenate(
            [flat[..., :centre], flat[..., centre + 1 :]], axis=-1
        )  # (H, W, size*size - 1)

    ux_nn = _neighbours(ux)
    uy_nn = _neighbours(uy)

    # Component-wise neighbour median (PIVware pwMedianUV)
    med_u = bn.nanmedian(ux_nn, axis=-1)
    med_v = bn.nanmedian(uy_nn, axis=-1)

    if combine == "norm":
        # Residual as a VECTOR NORM (PIVware pwVectorNorm): one scalar per node
        r_0 = np.sqrt((ux - med_u) ** 2 + (uy - med_v) ** 2)

        # Normalize by the median of the neighbour residual norms (relative to the
        # same neighbour median) plus epsilon (PIVware pwMedian of the residual).
        r_i = np.sqrt((ux_nn - med_u[..., None]) ** 2 + (uy_nn - med_v[..., None]) ** 2)
        r_m = bn.nanmedian(r_i, axis=-1)

        norm_resid = r_0 / (r_m + epsilon)
        exceeded = (norm_resid > threshold) | ~np.isfinite(norm_resid)
    else:
        # Independent verdicts: each field is scaled by its OWN neighbourhood
        # residual median, so a small-magnitude field cannot be hidden behind a
        # large one. Same normalized-median statistic, applied twice and OR-ed.
        exceeded = np.zeros(ux.shape, dtype=bool)
        for U, U_nn, med in ((ux, ux_nn, med_u), (uy, uy_nn, med_v)):
            r_0 = np.abs(U - med)
            r_m = bn.nanmedian(np.abs(U_nn - med[..., None]), axis=-1)
            norm_resid = r_0 / (r_m + epsilon)
            exceeded |= (norm_resid > threshold) | ~np.isfinite(norm_resid)

    # Border/hole guard on the immediate 3×3 (deliberately independent of the
    # median window `size`): reject a node only when its closest neighbours are
    # missing, so a wall-row node whose larger window pokes past the grid edge
    # is not thrown out just for being near the wall. A cell is present when
    # both components are finite, or when it is masked (a mask edge is not a
    # hole in the measurement).
    present = np.isfinite(ux) & np.isfinite(uy)
    if masked is not None:
        present |= masked
    guard_kernel = np.ones((GUARD_WINDOW, GUARD_WINDOW), dtype=np.float32)
    n_present = convolve2d(
        present.astype(np.float32),
        guard_kernel,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
    )

    # Boolean mask: true = outlier
    b_filter = exceeded | (n_present < GUARD_MIN_PRESENT)
    if masked is not None:
        b_filter &= ~masked
    return b_filter


def sigma_outlier_detection(
    ux: np.ndarray,
    uy: np.ndarray,
    sigma_threshold: float = 2.0,
    combine: str = "norm",
) -> np.ndarray:
    """
    Detect outliers based on local standard deviation (sigma-based).

    Vectors that deviate from the 8-neighbor mean by more than sigma_threshold
    times the local standard deviation are marked as outliers.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    sigma_threshold : float, optional
        Number of standard deviations for outlier threshold, defaults to 2.0.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier).
    """
    if combine not in ("norm", "per_component"):
        raise ValueError(f"combine must be 'norm' or 'per_component', got {combine!r}")
    if combine == "norm":
        return _sigma_single(np.sqrt(ux**2 + uy**2), sigma_threshold)
    # Independent verdicts — see median_outlier_detection's `combine` docs for
    # why the magnitude is wrong when the inputs are not vector components.
    return _sigma_single(ux, sigma_threshold) | _sigma_single(uy, sigma_threshold)


def _sigma_single(field: np.ndarray, sigma_threshold: float) -> np.ndarray:
    """Local 8-neighbour sigma test on ONE scalar field."""
    v = np.asarray(field).astype(np.float32, copy=False)
    finite = np.isfinite(v).astype(np.float32)
    v0 = np.where(np.isfinite(v), v, 0.0)

    # 3x3 window means for value, value^2, and for the finite-mask
    m9_v = ndi.uniform_filter(v0, size=3, mode="constant", cval=0.0)
    m9_v2 = ndi.uniform_filter(v0 * v0, size=3, mode="constant", cval=0.0)
    m9_cnt = ndi.uniform_filter(finite, size=3, mode="constant", cval=0.0)

    # uniform_filter returns MEAN; convert to SUM by * 9
    sum9 = m9_v * 9.0
    sumsq9 = m9_v2 * 9.0
    cnt9 = m9_cnt * 9.0

    # Exclude the center pixel to get 8-neighbour stats
    center_val = np.where(np.isfinite(v), v, 0.0)
    center_cnt = finite
    sum8 = sum9 - center_val
    sumsq8 = sumsq9 - center_val * center_val
    cnt8 = cnt9 - center_cnt

    # Mean and std over the 8 neighbours (ignore divisions by <=0 count)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean8 = sum8 / cnt8
        var8 = (sumsq8 / cnt8) - mean8 * mean8
    var8 = np.maximum(var8, 0.0)
    std8 = np.sqrt(var8)

    # Outlier mask
    b_filter = (
        (np.abs(v - mean8) > sigma_threshold * std8)
        | ~np.isfinite(v)
        | ~np.isfinite(mean8)
        | (std8 == 0)
        | (cnt8 < 1)
    )
    return b_filter


def div_vort_outliers(
    ux: np.ndarray,
    uy: np.ndarray,
    div_thresh: float = None,
    vort_thresh: float = None,
) -> np.ndarray:
    """
    Detect outliers based on divergence and vorticity thresholds.

    Computes divergence and vorticity using central differences, then
    identifies outliers as vectors with extreme values. Thresholds are
    automatically computed from the field statistics if not provided.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    div_thresh : float, optional
        Divergence threshold. If None, computed as 6*MAD.
    vort_thresh : float, optional
        Vorticity threshold. If None, computed as 6*MAD.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier).
    """
    # Central differences
    dudx = 0.5 * (np.roll(ux, -1, 1) - np.roll(ux, 1, 1))
    dudy = 0.5 * (np.roll(ux, -1, 0) - np.roll(ux, 1, 0))
    dvdx = 0.5 * (np.roll(uy, -1, 1) - np.roll(uy, 1, 1))
    dvdy = 0.5 * (np.roll(uy, -1, 0) - np.roll(uy, 1, 0))

    div = dudx + dvdy
    vort = dvdx - dudy

    # Robust thresholds from field using MAD (Median Absolute Deviation)
    if div_thresh is None:
        div_finite = div[np.isfinite(div)]
        if div_finite.size > 0:
            div_median = np.median(div_finite)
            mad = 1.4826 * np.median(np.abs(div_finite - div_median))
            div_thresh = 6 * mad
        else:
            div_thresh = np.inf

    if vort_thresh is None:
        vort_finite = vort[np.isfinite(vort)]
        if vort_finite.size > 0:
            vort_median = np.median(vort_finite)
            mad = 1.4826 * np.median(np.abs(vort_finite - vort_median))
            vort_thresh = 6 * mad
        else:
            vort_thresh = np.inf

    return (np.abs(div) > div_thresh) | (np.abs(vort) > vort_thresh)


def apply_outlier_detection(
    ux: np.ndarray,
    uy: np.ndarray,
    methods: list,
    peak_mag: np.ndarray = None,
    combine: str = "norm",
    masked: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply multiple outlier detection methods and combine results.

    This function applies a stack of outlier detection methods configured
    in the YAML file and combines their results with logical OR.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    methods : list
        List of method dictionaries from config, each with 'type' and parameters.
    peak_mag : np.ndarray, optional
        Peak magnitude array (required for 'peak_mag' method).
    combine : {"norm", "per_component"}, optional
        Forwarded to the component-combining methods (``median_2d``, ``sigma``);
        ignored by ``peak_mag`` and ``div_vort``, which take a single field or a
        genuine vector. Defaults to ``"norm"``. The ensemble stress call site
        passes ``"per_component"`` because ``(UU, VV)`` are not vector
        components — see ``median_outlier_detection``.
    masked : np.ndarray or None, optional
        Boolean mask of masked vector windows (True = masked), same shape as
        ``ux``. Masked windows are stored as 0 by the pipelines, and a 0 would
        otherwise count as a real neighbour. When given, every method runs on
        float copies of ``ux``/``uy`` with masked nodes set to NaN, and masked
        nodes are removed from the combined result. Per method:

        - ``peak_mag`` is unaffected except that masked nodes cannot be flagged.
        - ``median_2d`` excludes masked nodes from the neighbourhood and
          residual medians, and counts them as present in its 3x3 border/hole
          guard (``masked`` is forwarded).
        - ``sigma`` excludes masked nodes from the 8-neighbour mean, standard
          deviation and neighbour count.
        - ``div_vort`` yields NaN derivatives wherever a central-difference
          stencil touches a masked node, so those nodes are not flagged and are
          left out of the automatic MAD thresholds.

        ``None`` (default) leaves the behaviour of every method unchanged.

    Returns
    -------
    np.ndarray
        Combined boolean mask of outliers (True = outlier). Never True at
        masked nodes.

    Raises
    ------
    ValueError
        If a method is unknown, ``peak_mag`` is required but missing, or
        ``masked`` does not match the shape of ``ux``.
    """
    if masked is not None:
        masked = _as_mask(masked, ux.shape)
        ux = np.where(masked, np.nan, ux)
        uy = np.where(masked, np.nan, uy)

    combined_mask = np.zeros(ux.shape, dtype=bool)

    for method_cfg in methods:
        method_type = method_cfg.get("type", "").lower()

        if method_type == "peak_mag":
            if peak_mag is None:
                raise ValueError(
                    "peak_mag array required for 'peak_mag' outlier detection"
                )
            threshold = method_cfg.get("threshold", 0.5)
            mask = peak_magnitude_detection(peak_mag, threshold=threshold)
            combined_mask |= mask

        elif method_type == "median_2d":
            epsilon = method_cfg.get("epsilon", 0.1)
            threshold = method_cfg.get("threshold", 2.0)
            size = method_cfg.get("size", 5)
            mask = median_outlier_detection(
                ux,
                uy,
                epsilon=epsilon,
                threshold=threshold,
                size=size,
                combine=combine,
                masked=masked,
            )
            combined_mask |= mask

        elif method_type == "sigma":
            sigma_threshold = method_cfg.get("sigma_threshold", 2.0)
            mask = sigma_outlier_detection(
                ux, uy, sigma_threshold=sigma_threshold, combine=combine
            )
            combined_mask |= mask

        elif method_type == "div_vort":
            div_thresh = method_cfg.get("div_thresh", None)
            vort_thresh = method_cfg.get("vort_thresh", None)
            mask = div_vort_outliers(
                ux, uy, div_thresh=div_thresh, vort_thresh=vort_thresh
            )
            combined_mask |= mask

        else:
            raise ValueError(f"Unknown outlier detection method: {method_type}")

    if masked is not None:
        combined_mask &= ~masked
    return combined_mask
