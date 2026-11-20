"""
Shared utilities for plotting views.

Contains common functions used across plotting and interactive endpoints.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.io import loadmat

from pivtools_core.config import get_config
from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import find_non_empty_run

from ...utils import camera_number
from ..plot_maker import make_scalar_settings

# Unit mapping for different PIV variables
# Used to automatically select correct units when plotting
VARIABLE_UNITS = {
    # Velocities, in m/s
    "ux": "m/s",
    "uy": "m/s",
    "uz": "m/s",
    "mean_ux": "m/s",
    "mean_uy": "m/s",
    "mean_uz": "m/s",
    # Fluctuations, in m/s
    "u_prime": "m/s",
    "v_prime": "m/s",
    "w_prime": "m/s",
    # Mean stresses (m^2/s^2)
    "uu": "m^2/s^2",
    "vv": "m^2/s^2",
    "ww": "m^2/s^2",
    "uv": "m^2/s^2",
    "uw": "m^2/s^2",
    "vw": "m^2/s^2",
    # Instantaneous stresses (m^2/s^2)
    "uu_inst": "m^2/s^2",
    "vv_inst": "m^2/s^2",
    "ww_inst": "m^2/s^2",
    "uv_inst": "m^2/s^2",
    "uw_inst": "m^2/s^2",
    "vw_inst": "m^2/s^2",
    # Turbulent kinetic energy (m^2/s^2)
    "tke": "m^2/s^2",
    # Vorticity and divergence, in 1/s
    "vorticity": "1/s",
    "divergence": "1/s",
    # Gamma vortex criteria (dimensionless)
    "gamma1": "-",
    "gamma2": "-",
    # Peak height (dimensionless)
    "peak_mag": "-",
    # Peak detectability peak1/peak2 (dimensionless, PIVware SNR type 2)
    "peak_ratio": "-",
    "peakheight": "-",
    # Correlation-quality maps (correlation_quality statistic)
    "nan_pct": "%",
    "peak_ratio_median": "-",
    "peak_mag_mean": "-",
    # Ensemble diagnostics
    "nan_reason": "-",
    "b_mask": "-",
    "c_a": "-",
    "c_b": "-",
    "c_ab": "-",
    "window_size": "px",
    "sig_ab_x": "px",
    "sig_ab_y": "px",
    "sig_ab_xy": "px",
    "sig_a_x": "px",
    "sig_a_y": "px",
    "sig_a_xy": "px",
    "win_ctrs_x": "px",
    "win_ctrs_y": "px",
    "pred_x": "px",
    "pred_y": "px",
}

# Ensemble/statistics variable names carry these suffixes on top of the base
# stress/velocity names (e.g. UU_stress_uncorrected -> uu). Stripped repeatedly
# until the name is stable, so units are defined once per base variable.
_VAR_SUFFIXES = (
    "_uncorrected",
    "_window_correction",
    "_particle_correction",
    "_correction",
    "_stress",
    "_inst",
)


# Source prefixes the Results-tab variable dropdown puts on a name ("mean:uu"), as
# parsed by parseVarType in gui/src/hooks/useVectorViewer.ts. An unprefixed name is
# an instantaneous frame-file variable there too.
VAR_SOURCES = ("inst", "inst_stat", "mean", "ens", "corrq")


def split_var_source(var: str) -> Tuple[str, str]:
    """
    Split a dropdown variable name into (source, bare name).

    ``"mean:uu"`` gives ``("mean", "uu")``; ``"ux"`` gives ``("inst", "ux")``.

    Raises:
        ValueError: The prefix is not one of :data:`VAR_SOURCES`.
    """
    if ":" not in var:
        return "inst", var
    source, name = var.split(":", 1)
    if source not in VAR_SOURCES:
        raise ValueError(
            f"Unknown variable source '{source}' in '{var}'; "
            f"expected one of {', '.join(VAR_SOURCES)}"
        )
    return source, name


def normalize_var_name(var: str) -> str:
    """'ens:UU_stress_uncorrected' -> 'uu': strip source prefix, lowercase,
    strip ensemble suffixes until stable."""
    v = split_var_source(var)[1].strip().lower()
    changed = True
    while changed:
        changed = False
        for suf in _VAR_SUFFIXES:
            if v.endswith(suf) and len(v) > len(suf):
                v = v[: -len(suf)]
                changed = True
    return v


# Uncalibrated data is pixel displacement per frame pair, so units derive
# mechanically from the calibrated ones.
_UNCALIBRATED_UNITS = {"m/s": "px", "m^2/s^2": "px^2", "1/s": "1/frame"}


def units_for_var(var: str, uncalibrated: bool = False) -> str:
    """Canonical colorbar units for any plottable variable. All plot routes
    must use this rather than hardcoding units."""
    cal = VARIABLE_UNITS.get(normalize_var_name(var), "-")
    return _UNCALIBRATED_UNITS.get(cal, cal) if uncalibrated else cal


def length_units_for(uncalibrated: bool) -> str:
    """Axis (coordinate) units: px for uncalibrated data, mm for calibrated."""
    return "px" if uncalibrated else "mm"


# --- Wall-units (viscous scaling) view -------------------------------------
# View-only normalisation by friction velocity: u+ = u/u_tau, stresses/u_tau^2,
# y+ = (y_mm/1000)*u_tau/nu. Only velocity and stress variables scale; anything
# else (vorticity, peak heights, diagnostics) is left untouched.
_WALL_VELOCITY = {
    "ux",
    "uy",
    "uz",
    "mean_ux",
    "mean_uy",
    "mean_uz",
    "u_prime",
    "v_prime",
    "w_prime",
}
_WALL_STRESS = {"uu", "vv", "ww", "uv", "uw", "vw", "tke"}


def wall_scale_divisor(var: str, u_tau: float) -> Optional[float]:
    """Divisor to convert a variable to wall units, or None if the variable
    does not scale (left in physical units)."""
    base = normalize_var_name(var)
    if base in _WALL_VELOCITY:
        return u_tau
    if base in _WALL_STRESS:
        return u_tau**2
    return None


def apply_wall_units(var_arr, cy, var: str, plot_kwargs: Dict[str, Any]):
    """Apply view-only wall-unit scaling ahead of rendering.

    Returns (var_arr, cy, extra) where extra holds make_scalar_settings
    overrides: variable, variable_units, ylabel, y_units, lower_limit,
    upper_limit. No-op (empty extra) unless plot_kwargs['wall_units'].
    Colour limits arrive in raw units from the client and are divided here so
    the frontend never needs to know the scaling.
    """
    if not plot_kwargs.get("wall_units"):
        return var_arr, cy, {}
    u_tau = plot_kwargs["u_tau"]
    nu = plot_kwargs["nu"]
    extra: Dict[str, Any] = {}
    div = wall_scale_divisor(var, u_tau)
    if div:
        var_arr = np.asarray(var_arr, dtype=float) / div
        extra["variable"] = f"{var}+"
        extra["variable_units"] = ""
        if plot_kwargs.get("lower_limit") is not None:
            extra["lower_limit"] = plot_kwargs["lower_limit"] / div
        if plot_kwargs.get("upper_limit") is not None:
            extra["upper_limit"] = plot_kwargs["upper_limit"] / div
    if cy is not None:
        cy = np.asarray(cy, dtype=float) / 1000.0 * u_tau / nu  # mm -> m -> y+
        extra["ylabel"] = "y+"
        extra["y_units"] = ""
        # x stays mm while y is y+: equal aspect would collapse the plot
        extra["aspect"] = "auto"
    return var_arr, cy, extra


def load_piv_result(mat_path: Path) -> Any:
    """
    Load a .mat file and return its piv_result.

    Args:
        mat_path: Path to the .mat file

    Returns:
        piv_result array/struct from the mat file

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If piv_result not found in mat file
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"PIV result file not found: {mat_path}")
    try:
        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        # Truncated/corrupted .mat files (e.g. still being written) raise
        # various errors from scipy.io.  Surface them as FileNotFoundError
        # so callers that already handle 404 will skip gracefully.
        logger.warning(f"Failed to read {mat_path} (possibly truncated): {e}")
        raise FileNotFoundError(f"PIV result file unreadable: {mat_path}") from e
    if "piv_result" not in mat:
        raise ValueError(f"Variable 'piv_result' not found in mat: {mat_path}")
    return mat["piv_result"]


# dtype kinds a stored b_mask may have: bool, or 0/1 as int, uint or float
# (merging writes uint8, the PIV writer bool). NumPy casts strings and objects to
# bool without complaint, so the kind is checked rather than relying on astype.
B_MASK_DTYPE_KINDS = "biuf"


def b_mask_for(pr: Any, var_arr: np.ndarray, mat_path: Path) -> np.ndarray:
    """
    The boolean ``b_mask`` of one result pass element, checked against its data.

    Every writer of a plotted result emits ``b_mask`` on every pass element (PIV
    ``save_results`` instantaneous and ensemble, calibration apply and stereo
    reconstruction in ``calibration/runio.py``, ``vector_merger``, mean and
    per-frame statistics). One that is missing, not numeric/boolean, or not the
    shape of the plotted field is a stale or broken file.

    Args:
        pr: A ``piv_result`` or ``ensemble_result`` element (``loadmat`` with
            ``struct_as_record=False``)
        var_arr: The field being plotted from the same element
        mat_path: The file the element came from, for the error message

    Returns:
        ``b_mask`` as a bool array of ``var_arr``'s shape

    Raises:
        ValueError: ``b_mask`` is missing, of a non-mask dtype, or of another
            shape. The message names the file.
    """
    if not hasattr(pr, "b_mask"):
        raise ValueError(f"{mat_path}: result element has no b_mask field")
    stored = np.asarray(pr.b_mask)
    if stored.dtype.kind not in B_MASK_DTYPE_KINDS:
        raise ValueError(
            f"{mat_path}: b_mask has dtype {stored.dtype}, not a boolean mask"
        )
    if stored.shape != var_arr.shape:
        raise ValueError(
            f"{mat_path}: b_mask shape {stored.shape} does not match the "
            f"field's shape {var_arr.shape}"
        )
    return stored.astype(bool)


def extract_var_and_mask(
    pr: Any, var: str, mat_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract variable and mask arrays from a piv_result element.

    Args:
        pr: piv_result struct with attributes
        var: Variable name to extract
        mat_path: The file ``pr`` came from, for error messages

    Returns:
        Tuple of (variable_array, mask_array)

    Raises:
        ValueError: If the variable is not found, or ``b_mask`` is unusable
            (see :func:`b_mask_for`)
    """
    if not hasattr(pr, var):
        raise ValueError(f"{mat_path}: '{var}' not found in piv_result element")
    var_arr = np.asarray(getattr(pr, var))
    return var_arr, b_mask_for(pr, var_arr, mat_path)


def create_and_return_plot(
    var_arr: np.ndarray,
    mask_arr: np.ndarray,
    settings,
    raw: bool = False,
) -> Tuple[str, int, int, Dict]:
    """
    Create a plot and return base64 encoded PNG with metadata.

    Args:
        var_arr: Variable array to plot (H, W)
        mask_arr: Boolean mask array (H, W)
        settings: Plot settings object from make_scalar_settings
        raw: If True, create marginless image (pixel grid == data grid)

    Returns:
        Tuple of (base64_image, width, height, metadata_dict)
    """
    from ..plot_maker import plot_scalar_field

    H, W = int(var_arr.shape[0]), int(var_arr.shape[1])

    if raw:
        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        arr = np.asarray(var_arr).squeeze()

        vmin = (
            getattr(settings, "lower_limit", None)
            if hasattr(settings, "lower_limit")
            else None
        )
        vmax = (
            getattr(settings, "upper_limit", None)
            if hasattr(settings, "upper_limit")
            else None
        )

        # Handle NaN/Inf in automatic limit calculation
        if vmin is None or vmax is None:
            valid_data = arr[~(np.isnan(arr) | np.isinf(arr))]
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = float(np.min(valid_data))
                if vmax is None:
                    vmax = float(np.max(valid_data))
            else:
                if vmin is None:
                    vmin = 0.0
                if vmax is None:
                    vmax = 1.0

        cmap_name = getattr(settings, "cmap", None) or "bwr"
        cmap = plt.cm.get_cmap(cmap_name)
        cmap.set_bad(color="gray")
        masked = np.ma.masked_invalid(arr)
        ax.imshow(
            masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper"
        )
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # For raw mode, axes_bbox covers the entire image
        # Include data limits for cursor position mapping (raw mode uses pixel indices)
        # Normalize ylim to [0, H] for consistent cursor mapping
        axes_bbox = {
            "left": 0,
            "top": 0,
            "width": W,
            "height": H,
            "png_width": W,
            "png_height": H,
            "xlim": [0, W],
            "ylim": [0, H],
        }
        return (
            b64_img,
            W,
            H,
            {"axes_bbox": axes_bbox, "raw": True, "grid_dims": {"nx": W, "ny": H}},
        )

    # Full plot with decorations
    fig, ax, cf = plot_scalar_field(var_arr, mask_arr, settings)

    # Apply axis limits if set
    if hasattr(settings, "xlim") and settings.xlim is not None:
        ax.set_xlim(settings.xlim)
    if hasattr(settings, "ylim") and settings.ylim is not None:
        ax.set_ylim(settings.ylim)

    # Capture visible axis limits for cursor position mapping
    # Use matplotlib's actual axis limits which reflect user-set xlim/ylim or auto-scaling
    actual_xlim = ax.get_xlim()
    actual_ylim = ax.get_ylim()

    # Compute axes bounding box in PNG pixel coordinates
    # Draw canvas first to finalize layout before getting accurate pixel bounds
    fig.canvas.draw()

    dpi = fig.dpi
    fig_width, fig_height = fig.get_size_inches()
    png_width = int(fig_width * dpi)
    png_height = int(fig_height * dpi)

    # Use get_window_extent for accurate pixel coordinates after layout
    renderer = fig.canvas.get_renderer()
    ax_bbox_display = ax.get_window_extent(renderer=renderer)

    # Convert from display coords (origin bottom-left) to PNG coords (origin top-left)
    axes_left = int(round(ax_bbox_display.x0))
    axes_top = int(round(png_height - ax_bbox_display.y1))
    axes_width = int(round(ax_bbox_display.width))
    axes_height = int(round(ax_bbox_display.height))

    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    axes_left = clamp(axes_left, 0, png_width)
    axes_top = clamp(axes_top, 0, png_height)
    axes_width = clamp(axes_width, 0, png_width - axes_left)
    axes_height = clamp(axes_height, 0, png_height - axes_top)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Normalize limits for consistent cursor mapping (always min, max order)
    xlim_normalized = (min(actual_xlim), max(actual_xlim))
    ylim_normalized = (min(actual_ylim), max(actual_ylim))

    axes_bbox = {
        "left": axes_left,
        "top": axes_top,
        "width": axes_width,
        "height": axes_height,
        "png_width": png_width,
        "png_height": png_height,
        "xlim": [float(xlim_normalized[0]), float(xlim_normalized[1])],
        "ylim": [float(ylim_normalized[0]), float(ylim_normalized[1])],
    }
    return b64_img, W, H, {"axes_bbox": axes_bbox}


def parse_plot_params(req) -> Dict:
    """
    Parse plot parameters from request.

    Args:
        req: Flask request object

    Returns:
        Dict with normalized parameter fields

    Raises:
        ValueError: If required parameters are invalid
    """
    base_path = req.args.get("base_path", default=None, type=str)
    base_idx = req.args.get("base_path_idx", default=0, type=int)
    cfg = get_config()

    if not base_path:
        try:
            base_path = cfg.base_paths[base_idx]
        except Exception as err:
            raise ValueError(
                "Invalid base_path and base_path_idx fallback failed"
            ) from err

    camera = camera_number(req.args.get("camera", default=1))
    merged_raw = req.args.get("merged", default="0", type=str)
    use_merged = merged_raw in ("1", "true", "True", "TRUE")
    is_uncal_raw = req.args.get("is_uncalibrated", default="0", type=str)
    use_uncalibrated = is_uncal_raw in ("1", "true", "True", "TRUE")
    is_stereo_raw = req.args.get("is_stereo", default="0", type=str)
    use_stereo = is_stereo_raw in ("1", "true", "True", "TRUE")
    # Camera pair for stereo (comma-separated, e.g., "1,2")
    camera_pair_raw = req.args.get("camera_pair", default=None, type=str)
    stereo_camera_pair = None
    if camera_pair_raw and use_stereo:
        try:
            parts = camera_pair_raw.split(",")
            stereo_camera_pair = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            # Fall back to config stereo pairs
            if cfg.is_stereo_setup and cfg.stereo_pairs:
                stereo_camera_pair = cfg.stereo_pairs[0]
    type_name = req.args.get("type_name", default="instantaneous", type=str)
    frame = req.args.get("frame", default=1, type=int)
    run = req.args.get("run", default=1, type=int)
    endpoint = req.args.get("endpoint", default="", type=str)
    var = req.args.get("var", default="ux", type=str)
    lower_limit = req.args.get("lower_limit", type=float)
    upper_limit = req.args.get("upper_limit", type=float)
    cmap = req.args.get("cmap", default=None, type=str)

    if cmap and (cmap.strip() == "" or cmap.lower() == "default"):
        cmap = None

    raw_mode = req.args.get("raw", default="0", type=str) in (
        "1",
        "true",
        "True",
        "TRUE",
    )

    # Parse axis limits (optional)
    xlim_min = req.args.get("xlim_min", type=float)
    xlim_max = req.args.get("xlim_max", type=float)
    ylim_min = req.args.get("ylim_min", type=float)
    ylim_max = req.args.get("ylim_max", type=float)

    xlim = (
        (xlim_min, xlim_max) if xlim_min is not None and xlim_max is not None else None
    )
    ylim = (
        (ylim_min, ylim_max) if ylim_min is not None and ylim_max is not None else None
    )

    # Parse custom title
    custom_title = req.args.get("title", default=None, type=str)
    if custom_title and custom_title.strip() == "":
        custom_title = None

    # Wall-units (viscous scaling) view: active only when a positive u_tau is
    # supplied and the data is calibrated (uncalibrated px data cannot scale).
    u_tau = req.args.get("u_tau", type=float)
    nu = req.args.get("nu", default=1.5e-5, type=float)
    wall_units = bool(
        u_tau is not None and u_tau > 0 and nu and nu > 0 and not use_uncalibrated
    )

    # Validate frame number for calibrated data
    if not use_uncalibrated and not use_merged and not use_stereo:
        if frame < 1 or frame > cfg.num_frame_pairs:
            raise ValueError(
                f"Frame {frame} out of range. Valid: 1-{cfg.num_frame_pairs}"
            )

    return {
        "base_path": base_path,
        "camera": camera,
        "frame": frame,
        "run": run,
        "endpoint": endpoint,
        "var": var,
        "use_merged": use_merged,
        "use_uncalibrated": use_uncalibrated,
        "use_stereo": use_stereo,
        "stereo_camera_pair": stereo_camera_pair,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "cmap": cmap,
        "type_name": type_name,
        "raw": raw_mode,
        "xlim": xlim,
        "ylim": ylim,
        "custom_title": custom_title,
        "u_tau": u_tau,
        "nu": nu,
        "wall_units": wall_units,
    }


def validate_and_get_paths(params: Dict) -> Dict[str, Path]:
    """
    Validate parameters and resolve data paths.

    Args:
        params: Dict from parse_plot_params

    Returns:
        Dict of resolved paths

    Raises:
        ValueError: If path resolution fails
    """
    try:
        return get_data_paths(
            base_dir=params["base_path"],
            num_frame_pairs=get_config().num_frame_pairs,
            cam=params["camera"],
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
            use_uncalibrated=params["use_uncalibrated"],
            use_stereo=params.get("use_stereo", False),
            stereo_camera_pair=params.get("stereo_camera_pair"),
        )
    except Exception as e:
        logger.error(f"Path resolution failed: {e}")
        raise ValueError(f"Failed to resolve paths: {e}") from e


def load_and_plot_data(
    mat_path: Path,
    coords_path: Optional[Path],
    var: str,
    run: int,
    save_basepath: Path,
    **plot_kwargs,
) -> Tuple[str, int, int, Dict, int]:
    """
    Load data from mat_path, find non-empty run, extract var/mask, and create plot.

    Args:
        mat_path: Path to piv_result .mat file
        coords_path: Optional path to coordinates.mat
        var: Variable name to plot
        run: 1-based run number to start search from
        save_basepath: Base path for save operations
        **plot_kwargs: Additional plot settings (lower_limit, upper_limit, cmap, etc.)

    Returns:
        Tuple of (base64_image, width, height, extra_metadata, effective_run)
    """
    piv_result = load_piv_result(mat_path)
    pr, effective_run = find_non_empty_run(piv_result, var, run)
    if pr is None:
        raise ValueError(f"No non-empty run found for variable {var}")

    # Special handling for peak_mag (requires indexing via peak_choice)
    if var == "peak_mag":
        try:
            peak_mag = np.asarray(pr.peak_mag)
            peak_choice = np.asarray(pr.peak_choice)
        except Exception as err:
            raise ValueError(
                "peak_mag or peak_choice not found in piv_result element"
            ) from err

        if peak_mag.ndim == 3:
            if peak_mag.shape[0] == 1:
                var_arr = np.squeeze(peak_mag, axis=0)
            else:
                h, w = peak_choice.shape
                idx = peak_choice
                var_arr = peak_mag[idx, np.arange(h)[:, None], np.arange(w)[None, :]]
        else:
            var_arr = peak_mag
        mask_arr = np.zeros_like(var_arr, dtype=bool)
    else:
        var_arr, mask_arr = extract_var_and_mask(pr, var, mat_path)

    cx = cy = None
    if coords_path and coords_path.exists():
        mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in mat:
            raise ValueError("Variable 'coordinates' not found in coords mat")
        coords = mat["coordinates"]
        cx, cy = extract_coordinates(coords, effective_run)

    var_arr, cy, wall_extra = apply_wall_units(var_arr, cy, var, plot_kwargs)

    is_uncalibrated = plot_kwargs.get("is_uncalibrated", False)

    # Plot the data exactly as stored — no coordinate reflection, no sign negation.
    # The renderer orients the y-axis to the array's row order (row 0 at top), so the
    # stored coordinates alone decide whether 0 is at the top (y-down) or bottom (y-up).
    settings = make_scalar_settings(
        get_config(),
        variable=wall_extra.get("variable", var),
        run_label=effective_run,
        save_basepath=save_basepath,
        variable_units=wall_extra.get(
            "variable_units",
            plot_kwargs.get("variable_units", units_for_var(var, is_uncalibrated)),
        ),
        length_units=plot_kwargs.get("length_units", length_units_for(is_uncalibrated)),
        ylabel=wall_extra.get("ylabel"),
        y_units=wall_extra.get("y_units"),
        aspect=wall_extra.get("aspect", "equal"),
        coords_x=cx,
        coords_y=cy,
        lower_limit=wall_extra.get("lower_limit", plot_kwargs.get("lower_limit")),
        upper_limit=wall_extra.get("upper_limit", plot_kwargs.get("upper_limit")),
        cmap=plot_kwargs.get("cmap"),
        xlim=plot_kwargs.get("xlim"),
        ylim=plot_kwargs.get("ylim"),
        custom_title=plot_kwargs.get("custom_title"),
    )

    b64_img, W, H, extra = create_and_return_plot(
        var_arr, mask_arr, settings, raw=plot_kwargs.get("raw", False)
    )
    if plot_kwargs.get("wall_units"):
        extra = dict(extra or {})
        extra.update(
            {
                "wall_units": True,
                "u_tau": plot_kwargs["u_tau"],
                "nu": plot_kwargs["nu"],
            }
        )
    return b64_img, W, H, extra, effective_run


def build_response_meta(
    effective_run: int,
    var: str,
    width: int,
    height: int,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Build standardized metadata for plot responses.

    Args:
        effective_run: The run number used
        var: Variable name
        width: Image width
        height: Image height
        extra: Additional metadata to include

    Returns:
        Dict of metadata
    """
    meta = {"run": effective_run, "var": var, "width": width, "height": height}
    if extra:
        meta.update(extra)
    return meta
