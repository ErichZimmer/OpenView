"""
Configuration Validation Module

Validates PIV configuration before processing starts.
Used by both instantaneous.py and ensemble.py entry points.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

from pivtools_core.config import ActivePathsError, Config
from pivtools_core.fft_sizes import (
    BUILT_FFT_SIZES,
    SINGLE_MODE_A_SIZES,
    SINGLE_MODE_WINDOW_SIZES,
)
from pivtools_core.image_handling.load_images import (
    create_piv_frame_reader,
    read_pair,
)
from pivtools_core.image_handling.path_utils import (
    camera_path_for,
    format_to_glob,
    validate_images_generic,
)
from pivtools_core.image_handling.source_diagnosis import piv_diagnosis_context

logger = logging.getLogger(__name__)


def later_loop_findings(config: Config, base_source: Path, camera: int) -> List[str]:
    """Loops 1..N-1 of a looped recording, each read through its own source.

    Loop 0 is ``base_source`` itself, which the callers (``validate_config``,
    the GUI's ``/validate_files``) check in full. Every later loop must exist,
    and its first and last pair must read from its own camera path, as the
    pipeline (``load_images``) will read them.

    Args:
        config: Configuration object.
        base_source: The configured ``paths.source_paths`` entry (loop 0).
        camera: Camera number (1-based).

    Returns:
        One sentence per problem, each naming the loop and its source; empty
        for a single-loop recording.
    """
    findings: List[str] = []
    if config.num_loops <= 1:
        # Nothing to read, and the pair count is not asked for: a .pivraw run
        # that cannot be paired raises there, and validate_config reports that
        # refusal itself (_pivraw_pairing_findings).
        return findings
    pairs_per_loop = config.per_loop_frame_pairs
    for loop_idx in range(1, config.num_loops):
        loop_source = config.get_loop_source_path(base_source, loop_idx)
        if not loop_source.exists():
            findings.append(f"Loop {loop_idx} source not found: {loop_source.name}")
            continue
        camera_path = camera_path_for(config, loop_source, camera)
        for which, pair in (("first", 1), ("last", pairs_per_loop)):
            try:
                read_pair(pair, camera_path, camera, config)
            except Exception as e:
                findings.append(
                    f"Loop {loop_idx} ({loop_source.name}): {which} pair "
                    f"({pair}) could not be read: {e}"
                )
    return findings


def _pivraw_pairing_findings(config: Config, warnings: List[str]) -> List[str]:
    """The trigger pairing of a ``.pivraw`` config, reported for every active run.

    Appends to ``warnings``, per active source path: the alignment notes
    (unprovable A/B phase), each led by the run folder, and a camera in the
    run that is not selected but would refuse it if it were; then a
    ``num_images`` cap that bites. Returns the errors: a run that cannot be
    paired or lost a frame (named by the reader), or active runs that align
    different pair counts.
    """
    from pivtools_core.image_handling.readers.pivraw_reader import PivrawError

    try:
        n_pairs = config.num_frame_pairs
        alignments = [config.pivraw_alignment(s) for s in config.active_source_paths]
    except PivrawError as e:
        return [str(e)]
    for alignment in alignments:
        warnings.extend(f"{alignment.run_dir}: {note}" for note in alignment.notes)
        warnings.extend(_unselected_camera_notes(config, alignment))
    # num_frame_pairs refuses runs that align different pair counts, so one
    # count stands for every run.
    aligned = len(alignments[0].pairs)
    if n_pairs < aligned:
        warnings.append(
            f"Processing {n_pairs} of {aligned} aligned pairs: "
            f"images.num_images ({config.num_images}) caps them."
        )
    return []


def _unselected_camera_notes(config: Config, alignment) -> List[str]:
    """A note per camera in the run but not selected that would refuse it.

    The alignment covers the selected cameras only. Each other camera of the
    run is aligned with them in turn, and one the reader refuses is named with
    the reader's reason, so the user knows the run is not whole.
    """
    from pivtools_core.image_handling.readers.pivraw_reader import (
        PivrawError,
        align_pivraw_run,
        run_cameras,
    )

    notes = []
    for slot in run_cameras(alignment.run_dir):
        if slot in alignment.cameras:
            continue
        try:
            align_pivraw_run(
                alignment.run_dir,
                config.frame_stride,
                config.pair_stride,
                cameras=alignment.cameras + (slot,),
            )
        except PivrawError as e:
            notes.append(
                f"{alignment.run_dir}: Camera{slot} is in the run but not selected "
                f"for processing; selected, it would refuse the run: {e}"
            )
    return notes


def validate_config(config: Config) -> Tuple[bool, str, List[str]]:
    """
    Validate configuration before starting PIV processing.

    Checks:
    - Source and base paths alignment
    - Active paths configuration
    - Source paths exist
    - Base paths exist (creates if missing)
    - Image files for each camera

    Args:
        config: Configuration object

    Returns:
        Tuple of (is_valid, error_message, warnings)
        - is_valid: True if configuration is valid
        - error_message: Combined error messages if invalid
        - warnings: List of warning messages (processing continues)
    """
    errors = []
    warnings = []

    # Check source_paths and base_paths have the same length
    if len(config.source_paths) != len(config.base_paths):
        errors.append(
            f"source_paths ({len(config.source_paths)}) and base_paths ({len(config.base_paths)}) "
            "must have the same number of entries for paired processing"
        )

    # The dataset selection must name real datasets (an empty one is refused too)
    try:
        config.active_paths
    except ActivePathsError as e:
        errors.append(str(e))

    # Check source paths exist
    for i, source_path in enumerate(config.source_paths):
        if not source_path.exists():
            errors.append(f"Source path {i+1} does not exist: {source_path}")

    # Check base paths exist (create if missing)
    for i, base_path in enumerate(config.base_paths):
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created base path {i+1}: {base_path}")
            except Exception as e:
                errors.append(f"Failed to create base path {i+1}: {base_path} - {e}")

    if errors:
        return False, "\n".join(errors), warnings

    # Check image files for each camera
    camera_numbers = config.camera_numbers
    source_path = config.source_paths[0]
    image_type = config.image_type  # Already case-normalized via _detect_image_type()
    diagnosis = piv_diagnosis_context(config, 0)

    for camera_num in camera_numbers:
        format_str = config.image_format[0]
        camera_path = camera_path_for(config, source_path, camera_num)

        # Use the canonical validator (handles .set, .cine, .pivraw, .im7, standard)
        read_frame = create_piv_frame_reader(camera_path, camera_num, config)

        result = validate_images_generic(
            camera_path,
            camera_num,
            format_str,
            image_type,
            config.num_images,
            config.start_index == 0,
            read_frame_fn=read_frame,
            diagnosis=diagnosis,
            pair_strides=(config.frame_stride, config.pair_stride),
        )

        warnings.extend(f"Camera {camera_num}: {note}" for note in result["notes"])
        # The diagnosis warnings are about the recording (DaVis's SetIsSuspicious
        # flag), the same for every camera of it: reported once.
        warnings.extend(w for w in result["warnings"] if w not in warnings)
        if result["error"]:
            errors.append(f"Camera {camera_num}: {result['error']}")
        elif result["valid"] and image_type != "pivraw":
            # Check for processing subset (found > expected). A .pivraw run is
            # paired by trigger, so its subset is counted in pairs, below.
            found = result["found_count"]
            if isinstance(found, int) and found > config.num_images:
                warnings.append(
                    f"Camera {camera_num}: Processing subset - using {config.num_images} of {found} available files"
                )

        errors.extend(
            f"Camera {camera_num}: {finding}"
            for finding in later_loop_findings(config, source_path, camera_num)
        )

        # Supplementary: indexing mismatch check (standard types only)
        if image_type == "standard" and result["valid"] and camera_path.exists():
            pattern = format_to_glob(format_str)
            matching_files = list(camera_path.glob(pattern))
            if matching_files:
                indices = []
                for f in matching_files:
                    try:
                        m = re.search(r"(\d+)", f.name)
                        if m:
                            indices.append(int(m.group(1)))
                    except Exception:
                        pass
                if indices:
                    min_idx = min(indices)
                    expected_min = config.start_index
                    if min_idx != expected_min:
                        warnings.append(
                            f"Camera {camera_num}: File indexing mismatch - found files starting at {min_idx}, "
                            f"but start_index is {expected_min}"
                        )

    if image_type == "pivraw" and not errors:
        errors.extend(_pivraw_pairing_findings(config, warnings))

    # Window sizes must be sizes the codelet FFT engine was built for.
    # Fail loud here, at config load, rather than deep in the C hot path.
    errors.extend(validate_window_sizes(config))

    if errors:
        return False, "\n".join(errors), warnings

    # Memory estimation check
    memory_warning = validate_memory_for_images(config)
    if memory_warning:
        warnings.append(memory_warning)

    # Native-thread budget. Every Dask worker is its own process and each opens
    # native threadpools sized to omp_threads, so the machine spends
    # workers * omp_threads threads at once. The C kernels (correlation, warp,
    # and the k-space LM fitter) honour omp_threads directly, so an
    # over-committed product shows up as contention, not as an error.
    n_cores = os.cpu_count()
    if n_cores:
        # omp_threads is a str property with no coercion, so a YAML value like
        # "auto" would make int() raise here — and validate_config is contracted
        # to RETURN (False, msg, warnings), never to raise.
        try:
            budget = int(config.dask_workers_per_node) * int(config.omp_threads)
        except (TypeError, ValueError):
            errors.append(
                f"processing.omp_threads / dask_workers_per_node must be "
                f"integers, got omp_threads={config.omp_threads!r} "
                f"dask_workers_per_node={config.dask_workers_per_node!r}"
            )
            return False, "\n".join(errors), warnings
        if budget > n_cores:
            warnings.append(
                f"Thread budget {budget} exceeds {n_cores} cores "
                f"(dask_workers_per_node={config.dask_workers_per_node} x "
                f"omp_threads={config.omp_threads}). Native threadpools will "
                f"contend; reduce either knob in processing:."
            )

    # Ensemble-specific validation (when ensemble processing is enabled)
    if config.data.get("processing", {}).get("ensemble", False):
        ens_valid, ens_errors, ens_warnings = validate_ensemble_config(config)
        errors.extend(ens_errors)
        warnings.extend(ens_warnings)
        if not ens_valid:
            return False, "\n".join(errors), warnings

    return True, "", warnings


def _check_pass_size(
    wh,
    label: str,
    pass_number: int,
    allowed: Tuple[int, ...] = BUILT_FFT_SIZES,
    is_fft: bool = True,
) -> List[str]:
    """Return errors for one ``[h, w]`` pass whose axes are not permitted.

    Split out of :func:`_check_built_sizes` because the ensemble schedule needs
    a per-pass ``allowed`` set (single-mode passes admit the Frame-A sizes)
    while still reporting the pass's true position in the schedule.

    Parameters
    ----------
    wh
        One ``[height, width]`` pair.
    label
        Config key being checked, used verbatim in the error message.
    pass_number
        1-based pass index as it appears in the config.
    allowed
        Permitted axis lengths. Defaults to the built codelet sizes.
    is_fft
        True when ``allowed`` is a genuine FFT constraint (the axis length
        reaches the codelet engine). False for an ensemble ``single`` pass,
        where ``window_size`` is only the Frame-A mask support. Controls the
        wording so the message never claims a non-FFT length is an FFT one.
    """
    try:
        h, w = int(wh[0]), int(wh[1])
    except (TypeError, ValueError, IndexError):
        return [f"{label} pass {pass_number}: malformed window size {wh!r}"]

    bad = [v for v in (h, w) if v not in allowed]
    if not bad:
        return []

    if is_fft:
        reason = f"The FFT engine only supports {list(allowed)}."
        # Most likely mistake: a small Frame-A size used on a pass that is not
        # single mode. Say so rather than just listing the FFT sizes.
        if any(v in SINGLE_MODE_A_SIZES for v in bad):
            reason += (
                f" Sizes {list(SINGLE_MODE_A_SIZES)} are legal only on an "
                f"ensemble single-mode pass (ensemble_piv.type: single)."
            )
    else:
        reason = (
            f"A single-mode pass supports {list(allowed)} "
            f"(window_size is the Frame-A mask support, not an FFT size)."
        )
    return [
        f"{label} pass {pass_number}: window size [{h}, {w}] uses unsupported "
        f"size(s) {bad}. {reason}"
    ]


def _check_built_sizes(
    window_sizes,
    label: str,
    allowed: Tuple[int, ...] = BUILT_FFT_SIZES,
    is_fft: bool = True,
) -> List[str]:
    """Return an error per [h, w] pass whose axes are not permitted sizes."""
    errors = []
    for pass_idx, wh in enumerate(window_sizes or []):
        errors.extend(
            _check_pass_size(wh, label, pass_idx + 1, allowed=allowed, is_fft=is_fft)
        )
    return errors


def validate_window_sizes(config: Config) -> List[str]:
    """Window axis lengths must be sizes the codelet FFT engine was built for.

    Checks the instantaneous schedule always (it also drives the predictor
    passes) and the ensemble schedule when ensemble processing is enabled.

    In ``single`` ensemble mode the correlation is computed on the full
    ``sum_window`` (see ``cpu_ensemble.py`` -> ``window_sizes_for_computation``),
    so ``sum_window`` is a genuine FFT size and is checked too. Only
    ``sum_fitting_window`` is a correlation-plane crop (not an FFT), so it is
    intentionally not checked here.

    The ensemble schedule is checked per pass, because a ``single`` pass's
    ``window_size`` is not an FFT length either — it is the support of the
    Frame-A ``singlepix`` mask inside the sum window, so it additionally admits
    ``SINGLE_MODE_A_SIZES``. A ``std`` pass stays restricted to built sizes.
    """
    errors = _check_built_sizes(config.window_sizes, "instantaneous_piv.window_size")
    if not config.data.get("processing", {}).get("ensemble", False):
        return errors

    # Use the property so the all-'std' default, the 'standard' alias and the
    # length-vs-passes check live in config.py only. A malformed type list is
    # collected as an error here rather than raising mid-validation.
    try:
        ensemble_types = config.ensemble_type
    except ValueError as e:
        errors.append(f"ensemble_piv.type: {e}")
        return errors

    for pass_idx, (wh, pass_type) in enumerate(
        zip(config.ensemble_window_sizes, ensemble_types)
    ):
        if pass_type == "single":
            errors.extend(
                _check_pass_size(
                    wh,
                    "ensemble_piv.window_size",
                    pass_idx + 1,
                    allowed=SINGLE_MODE_WINDOW_SIZES,
                    is_fft=False,
                )
            )
        else:
            errors.extend(
                _check_pass_size(wh, "ensemble_piv.window_size", pass_idx + 1)
            )

    if "single" in ensemble_types:
        # Read raw so a malformed sum_window is still a collected error here.
        sum_window = config.data.get("ensemble_piv", {}).get("sum_window", [16, 16])
        errors.extend(_check_built_sizes([sum_window], "ensemble_piv.sum_window"))
    return errors


def validate_ensemble_config(config: Config) -> Tuple[bool, List[str], List[str]]:
    """
    Validate ensemble PIV configuration before processing starts.

    Catches ValueError from config properties (which validate internally)
    and adds additional cross-field checks not covered by individual properties.

    Args:
        config: Configuration object

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # 1. Validate ensemble_type list (length, valid values)
    try:
        ensemble_types = config.ensemble_type
    except ValueError as e:
        errors.append(f"Ensemble type: {e}")
        return False, errors, warnings

    # 2. Validate window sizes and overlaps
    try:
        window_sizes = config.ensemble_window_sizes
    except ValueError as e:
        errors.append(f"Ensemble window sizes: {e}")
        return False, errors, warnings

    try:
        overlaps = config.ensemble_overlaps
    except ValueError as e:
        errors.append(f"Ensemble overlaps: {e}")
        return False, errors, warnings

    # 3. Validate overlap values are in range
    for i, ovlp in enumerate(overlaps):
        if ovlp < 0 or ovlp > 95:
            errors.append(f"Pass {i+1}: overlap {ovlp}% out of range (0-95%)")

    # 4. Window sizes should decrease or stay the same across passes
    for i in range(1, len(window_sizes)):
        prev = window_sizes[i - 1]
        curr = window_sizes[i]
        if curr[0] > prev[0] or curr[1] > prev[1]:
            warnings.append(
                f"Pass {i+1}: window size {curr} is larger than pass {i} ({prev}). "
                "Typically window sizes decrease across passes."
            )

    # 5. Validate sum_window when single mode is used
    if "single" in ensemble_types:
        try:
            config.ensemble_sum_window
        except ValueError as e:
            errors.append(f"Ensemble sum window: {e}")

    # 6. Validate sum_fitting_window when enabled
    if config.ensemble_sum_fitting_window_enabled:
        try:
            config.ensemble_sum_fitting_window
        except ValueError as e:
            errors.append(f"Ensemble sum fitting window: {e}")

    # 7. Validate fit_method
    fit_method = None
    try:
        fit_method = config.ensemble_fit_method
    except ValueError as e:
        errors.append(f"Ensemble fit method: {e}")

    # 7b. Validate kspace_shape (quartic kurtosis terms are LM-fitter-only)
    try:
        kspace_shape = config.ensemble_kspace_shape
        if (
            kspace_shape != "gaussian"
            and fit_method is not None
            and fit_method != "kspace"
        ):
            errors.append(
                f"ensemble_piv.kspace_shape '{kspace_shape}' requires "
                f"fit_method 'kspace' (the '{fit_method}' fitter has no quartic "
                "shape terms and would silently ignore it), got "
                f"fit_method '{fit_method}'."
            )
    except ValueError as e:
        errors.append(f"Ensemble kspace shape: {e}")

    # 7c. Validate kspace_floor (coloured analytic floor is LM-fitter-only)
    try:
        kspace_floor = config.ensemble_kspace_floor
        if kspace_floor != "flat" and fit_method is not None and fit_method != "kspace":
            errors.append(
                f"ensemble_piv.kspace_floor '{kspace_floor}' requires "
                f"fit_method 'kspace' (only the LM fitter carries the "
                "coloured-floor model and would otherwise silently ignore "
                f"it), got fit_method '{fit_method}'."
            )
    except ValueError as e:
        errors.append(f"Ensemble kspace floor: {e}")

    # 8. Validate background subtraction / per-pair normalization consistency
    try:
        bg_method = config.ensemble_background_subtraction_method
    except ValueError as e:
        errors.append(f"Ensemble background subtraction: {e}")
        bg_method = None
    if config.ensemble_per_pair_normalization and bg_method != "window_mean":
        errors.append(
            "ensemble_piv.per_pair_normalization requires "
            "background_subtraction_method exactly 'window_mean' (the per-pair "
            "energies must be fluctuation energies, and any ensemble-level "
            "background term — 'correlation'/'image', plain or '+window_mean' — "
            "is built from raw mean images, inconsistent with normalized sums), "
            f"got '{bg_method}'."
        )

    # 9. Validate resume_from_pass
    resume = config.ensemble_resume_from_pass
    num_passes = config.ensemble_num_passes
    if resume != 0:
        if resume < 1 or resume > num_passes:
            errors.append(
                f"resume_from_pass={resume} is out of range. "
                f"Must be 0 (disabled) or 1-{num_passes}."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def estimate_memory_requirement(
    image_height: int, image_width: int, batch_size: int
) -> float:
    """
    Estimate minimum memory per worker in GB for PIV processing.

    Uses a 3x multiplier on raw image batch size to account for
    raw images + filtered copies + warped copies in memory simultaneously.

    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        batch_size: Number of image pairs per batch

    Returns:
        Estimated minimum memory in GB
    """
    # 3 × batch_size × 2 images_per_pair × H × W × 4 bytes (float32)
    bytes_needed = 3 * batch_size * 2 * image_height * image_width * 4
    return bytes_needed / (1024**3)


def parse_memory_limit_gb(memory_limit: str) -> float:
    """
    Parse a Dask memory limit string (e.g. '12GB', '512MB') to GB.

    Args:
        memory_limit: Memory string with unit suffix

    Returns:
        Memory in GB
    """
    match = re.match(r"^([\d.]+)\s*(GB|MB)$", memory_limit.strip(), re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "MB":
        return value / 1024
    return value


def validate_memory_for_images(config: Config) -> str:
    """
    Check if configured memory per worker is sufficient for the image
    resolution and batch size.

    Cameras are processed one after another on the same workers, so the
    largest selected camera sets the requirement.

    Returns:
        Warning message string, or empty string if memory is sufficient or
        nothing is configured yet (no source path, no camera, or a source path
        that does not exist). A camera whose image size cannot be read gives a
        warning naming the camera and the reader's error.
    """
    sources = config.source_paths
    cameras = config.camera_numbers
    if not sources or not cameras or not sources[0].exists():
        return ""
    shapes = []
    for cam in cameras:
        try:
            # (H, W) -- reads an actual image or container metadata
            shapes.append(config.image_shape_for(cam))
        except (ValueError, OSError) as e:
            return (
                f"Memory per worker not checked: the image size of camera {cam} "
                f"cannot be read ({e})."
            )

    h, w = max(shapes, key=lambda hw: hw[0] * hw[1])
    batch_size = config.batch_size
    memory_limit = config.dask_memory_limit
    megapixels = (h * w) / 1e6

    required_gb = estimate_memory_requirement(h, w, batch_size)
    configured_gb = parse_memory_limit_gb(memory_limit)

    if configured_gb <= 0:
        return ""

    if required_gb > configured_gb:
        return (
            f"Memory per worker ({memory_limit}) may be insufficient for "
            f"{megapixels:.1f} MP images with batch size {batch_size}. "
            f"Estimated minimum: {required_gb:.1f} GB. "
            f"Reduce batch size or increase memory per worker in Performance Settings "
            f"to avoid out-of-memory crashes."
        )

    return ""


def validate_batch_size_for_pod(config: Config, batch_size: int) -> Tuple[bool, str]:
    """
    Validate batch size is sufficient for POD filtering.

    POD requires a minimum number of images for meaningful decomposition.
    A batch size < 10 will likely not produce useful results.

    Args:
        config: Configuration object
        batch_size: Configured batch size

    Returns:
        Tuple of (is_valid, warning_message)
    """
    # Check if POD is configured
    filters = config.filters or []
    has_pod = any(f.get("type") == "pod" for f in filters)

    if not has_pod:
        return True, ""

    MIN_POD_BATCH_SIZE = 10

    if batch_size < MIN_POD_BATCH_SIZE:
        return False, (
            f"POD filter requires batch_size >= {MIN_POD_BATCH_SIZE} for meaningful decomposition. "
            f"Current batch_size: {batch_size}. Either increase batch_size or remove POD filter."
        )

    if batch_size < 20:
        return True, (
            f"POD filter works best with batch_size >= 20. "
            f"Current batch_size: {batch_size} may produce suboptimal results."
        )

    return True, ""


def log_validation_result(
    is_valid: bool,
    error_msg: str,
    warnings: List[str],
    config: Config,
) -> None:
    """
    Log validation results in a formatted way.

    Args:
        is_valid: Whether validation passed
        error_msg: Error message if failed
        warnings: List of warnings
        config: Configuration object
    """
    logger.info("=" * 80)
    logger.info("VALIDATING CONFIGURATION")
    logger.info("=" * 80)

    if not is_valid:
        logger.error("Configuration validation failed!")
        logger.error("=" * 80)
        logger.error("ERRORS:")
        logger.error(error_msg)
        logger.error("=" * 80)
        logger.error(
            "\nPlease fix the configuration errors in config.yaml and try again."
        )
        return

    logger.info("Configuration validated successfully")
    logger.info(f"  Source paths: {config.source_paths}")
    logger.info(f"  Cameras: {config.camera_numbers}")
    logger.info(f"  Image files: {config.num_images}")
    logger.info(f"  Frame pairs: {config.num_frame_pairs}")
    logger.info(f"  Image format: {config.image_format}")

    if warnings:
        logger.info("")
        logger.info("NOTES:")
        for warning in warnings:
            logger.info(f"  - {warning}")

    logger.info("=" * 80)
    logger.info("")
