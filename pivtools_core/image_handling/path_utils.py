"""Shared path utilities and validation for image loading.

This module provides centralized path-building and validation functions used by both
PIV image loading (load_images.py) and calibration image loading
(calibration_loader.py), eliminating code duplication.
"""

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ..calibration_settings import CALIBRATION_DIRNAME, SETTINGS_FILENAME

if TYPE_CHECKING:
    from ..config import Config
    from .source_diagnosis import DiagnosisContext, SourceScan


def format_to_glob(fmt: str) -> str:
    """Convert printf-style format string to glob pattern.

    Replaces any %Nd format specifier (%d, %02d, %03d, %05d, etc.) with *.
    """
    return re.sub(r"%\d*d", "*", fmt)


# Quote characters a pasted path may be wrapped in (one matching pair is removed).
_SOURCE_QUOTES = "\"'"


def normalise_source_path(raw: str) -> str:
    """Strip whitespace, then one matching pair of surrounding quotes, then whitespace.

    A path pasted from a shell or file manager often carries ``"..."`` / ``'...'``
    quotes and stray spaces, which the filesystem would take literally. Only a pair
    whose opening and closing quote are the same character is removed, and only one
    pair, so a name that genuinely ends in an apostrophe (``runs'``) is kept.
    """
    path = raw.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in _SOURCE_QUOTES:
        path = path[1:-1]
    return path.strip()


def infer_image_type(image_format: str) -> str:
    """Infer the image type from a format string's extension.

    Standalone twin of ``Config._detect_image_type`` for callers that hold a format
    string but not a ``Config`` (e.g. the calibration CLI, which takes ``--source`` /
    ``--image-format`` directly). Returns one of ``standard``, ``cine``,
    ``pivraw``, ``lavision_set``, ``lavision_im7``.
    """
    fmt = (image_format or "").lower()
    if ".cine" in fmt:
        return "cine"
    if ".pivraw" in fmt:
        return "pivraw"
    if ".set" in fmt or ".ims" in fmt:
        return "lavision_set"
    if ".im7" in fmt:
        return "lavision_im7"
    return "standard"


# Extensions of the "standard" raster image family. The single source of truth for
# both the validator's fallback globs and the diagnosis scan's raster family.
RASTER_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def glob_images(directory: Path, pattern: str) -> List[Path]:
    """glob() a directory, excluding dotfiles.

    macOS writes AppleDouble sidecar files ("._foo.tif") next to real images on
    FAT/exFAT/network volumes. These sort before the real files (".") and would
    otherwise poison pattern suggestion (yielding "._foo%05d.tif") and file counts.
    Any leading-dot name (hidden file or AppleDouble sidecar) is excluded.
    """
    return [f for f in directory.glob(pattern) if not f.name.startswith(".")]


def _raster_images(directory: Path) -> List[Path]:
    """Every ``RASTER_EXTENSIONS`` file in ``directory`` (dot-names excluded)."""
    found: List[Path] = []
    for ext in RASTER_EXTENSIONS:
        found.extend(glob_images(directory, f"*{ext}"))
    return found


def camera_subfolder_name(
    image_type: str,
    camera: int,
    camera_count: int,
    *,
    use_camera_subfolders: bool,
    camera_subfolders: Sequence[str],
) -> str:
    """Subfolder holding one camera's images: THE rule for PIV and calibration alike.

    - Container formats (``lavision_set``, ``cine``, ``pivraw``) never use a
      subfolder: all cameras live in one ``.set``, and ``.cine`` / ``.pivraw``
      files sit side by side in the source directory (``%d`` in the pattern
      names the camera).
    - IM7 uses a subfolder only when ``use_camera_subfolders`` is True (one
      single-camera ``.im7`` per folder); otherwise one multi-camera ``.im7``
      per time instance sits at the source root. This is the only format the
      flag applies to.
    - A single-camera setup (``camera_count == 1``) always resolves at the
      source root. A stored subfolder entry there is stale state from a
      previous multi-camera session, never a valid override.
    - Otherwise (standard formats at ``camera_count > 1``, or IM7 with the flag
      on) an explicit ``camera_subfolders`` entry wins, else ``Cam{N}``.

    ``Config.get_camera_folder`` (PIV images, ``paths.camera_subfolders``) and
    :func:`calibration_camera_folder` (calibration images, the sidecar's
    ``image`` block) both call this, so the two can never disagree.
    """
    if image_type in ("lavision_set", "cine", "pivraw"):
        return ""
    if image_type == "lavision_im7" and not use_camera_subfolders:
        return ""
    if camera_count == 1:
        return ""
    idx = camera - 1  # camera is 1-based
    subfolders = list(camera_subfolders or [])
    if idx < len(subfolders) and subfolders[idx]:
        return str(subfolders[idx])
    return f"Cam{camera}"


def calibration_camera_folder(
    image_settings: dict,
    image_type: str,
    camera: int,
    num_cameras: int,
) -> str:
    """Subfolder name for one camera's calibration images (settings-sidecar based).

    The same rule as PIV images (:func:`camera_subfolder_name`), fed from the
    sidecar ``image`` block: ``use_camera_subfolders`` gates IM7 only, and
    ``num_cameras`` is the configured rig camera count.
    """
    return camera_subfolder_name(
        image_type,
        camera,
        num_cameras,
        use_camera_subfolders=bool(image_settings.get("use_camera_subfolders", False)),
        camera_subfolders=image_settings.get("camera_subfolders") or [],
    )


def build_calibration_camera_path(
    source: Path,
    image_settings: dict,
    camera: int = 1,
    num_cameras: int = 1,
) -> Path:
    """Path to one camera's calibration images for an explicit source.

    ``image_settings`` is the ``image`` block of the source's settings sidecar
    (``pivtools_core.calibration_settings``) — the persisted YAML config plays
    no part in calibration image resolution.

    The camera folder follows :func:`camera_subfolder_name` (the PIV rule):
    containers resolve to ``source`` itself (the ``.set`` file, or the
    directory holding the ``.cine`` files), IM7 takes a subfolder only with
    ``use_camera_subfolders``, standard formats take ``Cam{N}`` (or the explicit
    entry) whenever ``num_cameras > 1``.
    """
    source = Path(source)
    fmt = image_settings.get("image_format") or ""
    image_type = image_settings.get("image_type") or infer_image_type(fmt)
    folder = calibration_camera_folder(image_settings, image_type, camera, num_cameras)
    if folder:
        return source / folder

    return source


def camera_path_for(config: "Config", base_source: Path, camera: int) -> Path:
    """Where one camera's PIV images are, given a source (one loop's, if looped).

    THE camera-path rule for PIV images: every reader (the pipeline, the GUI
    routes, shape detection, validation) calls this, so none can look for a
    camera somewhere the others do not. The folder comes from
    ``Config.get_camera_folder`` (:func:`camera_subfolder_name`):

    - ``.set``: ``base_source`` is the container file itself.
    - ``.cine`` / ``.pivraw``: the directory holding one file per camera.
    - ``.im7``: ``base_source / Cam{N}`` with ``images.use_camera_subfolders``
      (one single-camera file per folder), else ``base_source`` (one
      multi-camera file per instant).
    - standard rasters: ``base_source / Cam{N}`` (or the configured
      subfolder) when there is more than one camera, else ``base_source``.

    Args:
        config: Configuration object.
        base_source: A ``paths.source_paths`` entry, or the loop source
            ``Config.get_loop_source_path`` derived from one. The camera folder
            sits inside the loop's source, so a loop is resolved first.
        camera: Camera number (1-based).
    """
    folder = config.get_camera_folder(camera)
    return Path(base_source) / folder if folder else Path(base_source)


def build_piv_camera_path(
    config: "Config",
    source_path_idx: int = 0,
    camera: int = 1,
) -> Path:
    """Path to one camera's PIV images in ``source_paths[source_path_idx]``.

    :func:`camera_path_for` on the configured source (loop 0 of a looped
    recording).

    Args:
        config: Configuration object
        source_path_idx: Index into source_paths list (default: 0)
        camera: Camera number (1-based, default: 1)

    Returns:
        Path: Path to PIV image file (.set) or directory for other formats
    """
    return camera_path_for(config, config.source_paths[source_path_idx], camera)


def resolve_file_path(
    camera_path: Path,
    camera: int,
    frame_idx: int,
    format_pattern: str,
    image_type: str,
    zero_based_indexing: bool = False,
) -> Path:
    """Resolve the file path for a specific frame.

    Handles the different file path patterns for each image type:
    - lavision_set: camera_path IS the .set file (return directly)
    - lavision_im7: camera_path / (format_pattern % frame_idx)
    - cine, pivraw: camera_path / (format_pattern % camera)
    - standard: camera_path / (format_pattern % frame_idx)

    Args:
        camera_path: Base path - for .set this IS the file, otherwise directory
        camera: Camera number (1-based)
        frame_idx: Frame index (1-based)
        format_pattern: File format pattern (e.g., "%05d.tif", "Camera%d.cine")
        image_type: One of "lavision_set", "lavision_im7", "cine", "pivraw",
            "standard"
        zero_based_indexing: Whether file indices are 0-based

    Returns:
        Path: Full path to the image file
    """
    # Apply zero-based indexing if needed
    file_idx = frame_idx - 1 if zero_based_indexing else frame_idx

    if image_type == "lavision_set":
        # For .set files, camera_path IS the .set file - return directly
        # (format_pattern is ignored as source_path contains the full file path)
        return camera_path

    elif image_type in ("cine", "pivraw"):
        # .cine / .pivraw files: pattern uses camera number
        if "%" in format_pattern:
            return camera_path / (format_pattern % camera)
        return camera_path / format_pattern

    elif image_type == "lavision_im7":
        # .im7 files: pattern uses frame index
        if "%" in format_pattern:
            return camera_path / (format_pattern % file_idx)
        return camera_path / format_pattern

    else:
        # Standard formats: pattern uses frame index
        if "%" in format_pattern:
            return camera_path / (format_pattern % file_idx)
        return camera_path / format_pattern


def _image_to_base64(img: np.ndarray, max_size: int = 512) -> str:
    """Convert image array to base64 PNG string.

    Parameters
    ----------
    img : np.ndarray
        Image array (H, W) or (H, W, C)
    max_size : int
        Maximum dimension for resizing

    Returns
    -------
    str
        Base64-encoded PNG image
    """
    from PIL import Image

    # Normalize to 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype == bool:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Create PIL image
    pil_img = Image.fromarray(img)

    # Resize if too large
    if max(pil_img.size) > max_size:
        ratio = max_size / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


AMBIGUOUS_FRAME_INDEX_MSG = "could not tell which number is the frame index in '{name}'"
"""Reason given when no single digit run can be identified as the frame index."""

NO_FRAME_NUMBER_MSG = "no frame number found in '{name}'"
"""Reason given when a filename contains no digits at all."""

_DIGIT_RUN = re.compile(r"(\d+)")


class PatternSuggestion(NamedTuple):
    """A suggested filename pattern, or the reason none could be made.

    Exactly one of the two fields is set: ``pattern`` is the printf-style format
    (e.g. ``"Cam1_%05d.tif"``), ``reason`` explains why no pattern was suggested
    (e.g. the frame number is ambiguous). Callers surface ``reason`` to the user
    rather than dropping it.
    """

    pattern: Optional[str]
    reason: Optional[str]


def _split_digit_runs(stem: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split a filename stem into its non-digit skeleton and its digit runs.

    ``"run3_img0001_cam1"`` -> ``(("run", "_img", "_cam", ""), ("3", "0001", "1"))``.
    The skeleton always has one more element than the runs.
    """
    parts = _DIGIT_RUN.split(stem)
    return tuple(parts[0::2]), tuple(parts[1::2])


def _run_placeholder(values: List[str]) -> str:
    """printf placeholder for a digit run given its observed string values.

    A leading zero in any value proves zero padding, so the width is that run's
    true length (``%0Nd`` for any N). Otherwise the run is unpadded (``%d``).
    """
    padded = [v for v in values if len(v) > 1 and v.startswith("0")]
    if padded:
        return f"%0{max(len(v) for v in padded)}d"
    return "%d"


def _suggest_pattern(
    sample_files: List[str], forced_ext: Optional[str] = None
) -> PatternSuggestion:
    """Suggest a filename pattern from a folder's sample filenames.

    The frame index is the digit run whose value varies between files. The first
    sorted file fixes the naming skeleton (non-digit text, run count and
    extension); only files sharing that skeleton are compared, so other file
    families in the folder (e.g. the B frames of an A/B pair) do not interfere.

    - Exactly one run position varies: it is the frame index.
    - Several vary: the one strictly increasing in sorted order is chosen; if
      none or more than one is, the frame index is ambiguous.
    - None vary (single file or identical names): a name with exactly one digit
      run uses it; otherwise the frame index is ambiguous.

    Parameters
    ----------
    sample_files : List[str]
        Filenames (not paths) found in the folder.
    forced_ext : str, optional
        Extension to write into the pattern instead of the file's own.

    Returns
    -------
    PatternSuggestion
        The pattern, or the reason no pattern could be suggested.
    """
    if not sample_files:
        raise ValueError("_suggest_pattern needs at least one filename")
    names = sorted(sample_files)
    first = Path(names[0])
    skeleton, first_runs = _split_digit_runs(first.stem)
    ext = forced_ext or first.suffix.lstrip(".")

    if not first_runs:
        return PatternSuggestion(None, NO_FRAME_NUMBER_MSG.format(name=first.name))

    group: List[Tuple[str, ...]] = []
    for name in names:
        path = Path(name)
        name_skeleton, runs = _split_digit_runs(path.stem)
        if name_skeleton == skeleton and path.suffix == first.suffix:
            group.append(runs)

    varying = [
        i for i in range(len(first_runs)) if len({runs[i] for runs in group}) > 1
    ]
    if len(varying) > 1:
        varying = [
            i
            for i in varying
            if all(int(a[i]) < int(b[i]) for a, b in zip(group[:-1], group[1:]))
        ]
    elif not varying and len(first_runs) == 1:
        varying = [0]

    if len(varying) != 1:
        return PatternSuggestion(
            None, AMBIGUOUS_FRAME_INDEX_MSG.format(name=first.name)
        )

    frame_run = varying[0]
    placeholder = _run_placeholder([runs[frame_run] for runs in group])
    pieces = [skeleton[0]]
    for i, run in enumerate(first_runs):
        pieces.append(placeholder if i == frame_run else run)
        pieces.append(skeleton[i + 1])
    suffix = f".{ext}" if ext else ""
    return PatternSuggestion(f"{''.join(pieces)}{suffix}", None)


def _detect_ab_pair_pattern(
    sample_files: List[str], forced_ext: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Detect if files follow A/B naming convention and suggest both patterns.

    Looks for paired files with _A/_B or _a/_b suffixes in the filename.
    Returns both pattern_a and pattern_b if A/B pairs are detected.

    Parameters
    ----------
    sample_files : List[str]
        List of filenames to analyze (just names, not full paths)
    forced_ext : str, optional
        Force a specific extension

    Returns
    -------
    Optional[Dict[str, Any]]
        If A/B pairs detected:
        {
            "pattern_a": str,  # Pattern for A files (e.g., "B%05d_A.tif")
            "pattern_b": str,  # Pattern for B files (e.g., "B%05d_B.tif")
            "mode": "ab_format"
        }
        If not A/B format, returns None
    """
    if not sample_files:
        return None

    # Look for A/B pattern in filenames
    # Common patterns: _A.tif/_B.tif, _a.png/_b.png, -A.jpg/-B.jpg
    a_pattern = re.compile(r"[_-][Aa]\.[a-zA-Z]+$")
    b_pattern = re.compile(r"[_-][Bb]\.[a-zA-Z]+$")

    a_files = [f for f in sample_files if a_pattern.search(f)]
    b_files = [f for f in sample_files if b_pattern.search(f)]

    # Need both A and B files present, with similar counts
    if not a_files or not b_files:
        return None

    # Check that counts are roughly similar (within factor of 2)
    if max(len(a_files), len(b_files)) > 2 * min(len(a_files), len(b_files)):
        return None

    # Suggest each role's pattern from that role's files; the _A/_B suffix is
    # non-digit skeleton text, so it is preserved.
    pattern_a = _suggest_pattern(a_files, forced_ext).pattern
    pattern_b = _suggest_pattern(b_files, forced_ext).pattern
    if pattern_a is None or pattern_b is None:
        # No frame index identifiable; the caller's single-pattern suggestion
        # reports why.
        return None

    # Verify the patterns differ only in A/B
    pattern_a_normalized = re.sub(r"[_-][Aa]\.", "_X.", pattern_a)
    pattern_b_normalized = re.sub(r"[_-][Bb]\.", "_X.", pattern_b)

    if pattern_a_normalized != pattern_b_normalized:
        # Patterns don't match in structure - not a valid A/B pair
        return None

    return {
        "pattern_a": pattern_a,
        "pattern_b": pattern_b,
        "mode": "ab_format",
    }


_PRINTF_INT = re.compile(r"%(0?)(\d*)d")


def _pattern_to_regex(pattern: str) -> Optional["re.Pattern[str]"]:
    """Compile a single-placeholder printf pattern into a filename regex.

    ``"B%05d_A.tif"`` -> ``^B(\d{5,})_A\.tif$``; ``%d`` -> ``(\d+)``. A number
    wider than the pad width still formats wider, hence the open upper bound.
    Returns None unless the pattern has exactly one ``%d``-style placeholder.
    """
    placeholders = list(_PRINTF_INT.finditer(pattern))
    if len(placeholders) != 1:
        return None
    m = placeholders[0]
    width = int(m.group(2)) if m.group(2) else 0
    digits = rf"(\d{{{width},}})" if width > 1 else r"(\d+)"
    return re.compile(
        "^"
        + re.escape(pattern[: m.start()])
        + digits
        + re.escape(pattern[m.end() :])
        + "$"
    )


def detect_start_index(sample_files: List[str], pattern: str) -> Optional[int]:
    """Smallest frame index among the filenames that match ``pattern``.

    The index is parsed with the pattern itself (its ``%0Nd`` placeholder), so a
    camera number elsewhere in the name (``Cam1_00000.tif``) is never mistaken for
    the frame index. Pass the FULL matching listing, not a truncated sample: the
    minimum is only meaningful over every file.

    Returns None when the pattern has no single placeholder or nothing matches.
    """
    regex = _pattern_to_regex(pattern)
    if regex is None:
        return None
    indices = [int(m.group(1)) for m in map(regex.match, sample_files) if m]
    return min(indices) if indices else None


def start_index_hint_message(found: int, start: int) -> str:
    """User-facing text for files whose first index differs from the start index."""
    message = f"Files start at {found} but the start index is {start}"
    if found in (0, 1):
        message += f" — switch to {found}-based indexing"
    return message


def _start_index_mismatch(
    camera_path: Path,
    pattern: str,
    matching_names: List[str],
    start_idx: int,
    expected_count: int,
) -> Optional[int]:
    """The files' first index when a wrong 0/1 start index explains a failure.

    Returns the detected first index only when the first or last expected file
    is missing AND the files start at the other of 0/1. Otherwise None (no
    mismatch, or a missing file the start index does not explain).
    """
    found = detect_start_index(matching_names, pattern)
    if found is None or not start_index_explains_failure(
        camera_path, pattern, found, start_idx, expected_count
    ):
        return None
    return found


def start_index_explains_failure(
    camera_path: Path,
    pattern: str,
    found: int,
    start_idx: int,
    expected_count: int,
) -> bool:
    """True when files starting at ``found`` explain a failure at ``start_idx``.

    ``found`` must be the other of 0/1, and the first or last expected file
    (``pattern % start_idx`` / ``pattern % (start_idx + expected_count - 1)``)
    must be missing from ``camera_path``.
    """
    if found == start_idx or found not in (0, 1):
        return False
    try:
        first = pattern % start_idx
        last = pattern % (start_idx + max(expected_count, 1) - 1)
    except (TypeError, ValueError):
        return False
    return not ((camera_path / first).exists() and (camera_path / last).exists())


def _validate_file_listing(
    result: Dict[str, Any],
    camera_path: Path,
    image_format: str,
    matching_files: List[Path],
    start_idx: int,
    expected_count: int,
) -> Optional[int]:
    """Set ``valid``/``error`` for a non-empty listing of per-frame files.

    Records ``detected_start_index`` over the full listing. A wrong 0/1 start
    index that leaves the first or last expected file missing fails with the
    start-index hint; otherwise the count decides.

    Returns the files' first index when the start index explains the failure,
    else None.
    """
    names = [f.name for f in matching_files]
    result["detected_start_index"] = detect_start_index(names, image_format)
    found = _start_index_mismatch(
        camera_path, image_format, names, start_idx, expected_count
    )
    if found is not None:
        result["valid"] = False
        result["error"] = start_index_hint_message(found, start_idx)
    elif len(matching_files) < expected_count:
        result["valid"] = False
        result["error"] = (
            f"Found {len(matching_files)} files, expected {expected_count}"
        )
    else:
        result["valid"] = True
    return found


def _append_error(result: Dict[str, Any], text: str) -> None:
    """Append ``text`` to the result's ``error`` with ``"; "``, or set it."""
    result["error"] = f"{result['error']}; {text}" if result["error"] else text


def _fold_pattern_reason(
    result: Dict[str, Any], names: List[str], image_type: str, calibration: bool
) -> None:
    """Append why no filename pattern can be made from ``names`` to ``error``.

    The ``pattern`` diagnosis rule offers the pattern when one exists; when none
    can be made (e.g. an ambiguous frame index) the reason must still reach the user.
    """
    from .source_diagnosis import pattern_reason

    reason = pattern_reason(names, image_type, calibration)
    if reason is not None:
        _append_error(result, reason)


def cine_extension_mismatch(cine_file: Path) -> Optional[str]:
    """The error for a CINE image type whose pattern names a non-``.cine`` file, else None.

    Same guard as the ``.set`` branch of :func:`validate_images_generic`: the CINE
    header reader on another format's bytes fails with an unrelated decode error
    (a PNG gives "'utf-8' codec can't decode byte 0x89"), so name the mismatch.
    """
    if cine_file.suffix.lower() == ".cine":
        return None
    kind = f"a {cine_file.suffix}" if cine_file.suffix else "an extension-less"
    return f"Image type is CINE but the pattern names {kind} file: {cine_file}"


def pivraw_extension_mismatch(pivraw_file: Path) -> Optional[str]:
    """The error for a PIVRAW image type whose pattern names a non-``.pivraw`` file.

    Twin of :func:`cine_extension_mismatch`: without it the reader would report a
    missing ``<name>.json`` sidecar for, say, a ``.png``, which hides the real
    mistake. None when the suffix is ``.pivraw``.
    """
    if pivraw_file.suffix.lower() == ".pivraw":
        return None
    kind = f"a {pivraw_file.suffix}" if pivraw_file.suffix else "an extension-less"
    return f"Image type is PIVRAW but the pattern names {kind} file: {pivraw_file}"


def _container_count_failure(
    result: Dict[str, Any], container: Path, exc: Exception
) -> Dict[str, Any]:
    """Fail a container validation whose image count could not be read.

    The message names the file and the reader's reason; it also goes into
    ``read_error`` so callers that rebuild the message (the PIV route) keep it.
    """
    message = f"Could not read the image count of {container}: {exc}"
    logging.warning(message)
    result["valid"] = False
    result["error"] = message
    result["read_error"] = message
    return result


def _check_container_count(
    result: Dict[str, Any], expected_count: Optional[int]
) -> None:
    """Set ``valid`` for a container whose entry count was read into ``found_count``.

    Fewer entries than the caller expects fails with "found N entries, expected
    M" (the same shortfall rule the per-file listing applies); more is a
    processing subset and passes. ``expected_count=None`` means the caller holds
    no count, so any readable count passes.
    """
    found = result["found_count"]
    if expected_count is not None and found < expected_count:
        result["valid"] = False
        result["error"] = f"found {found} entries, expected {expected_count}"
        return
    result["valid"] = True


def validate_images_generic(
    camera_path: Path,
    camera: int,
    image_format: str,
    image_type: str,
    expected_count: Optional[int],
    zero_based_indexing: bool,
    read_frame_fn: Callable[[int], np.ndarray],
    *,
    diagnosis: "DiagnosisContext",
    scan_cache: Optional[Dict[str, "SourceScan"]] = None,
    pair_strides: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Generic image validation for both PIV and calibration images.

    This is the core validation logic used by both validate_files() in app.py
    and validate_calibration_images() in calibration_loader.py.

    Parameters
    ----------
    camera_path : Path
        Path to the camera directory or source directory
    camera : int
        Camera number (1-based)
    image_format : str
        Image format pattern (e.g., "%05d.tif", "data.set")
    image_type : str
        One of "lavision_set", "lavision_im7", "cine", "pivraw",
        "standard"
    expected_count : int or None
        Expected number of images/frames. A container (.set / .cine) holding
        fewer entries fails validation; more is a processing subset and passes.
        ``None`` is allowed for containers only, when the caller has no count to
        hold the container to (calibration with a blank ``n_views``).
    zero_based_indexing : bool
        Whether file indices are 0-based
    read_frame_fn : Callable[[int], np.ndarray]
        Function to read a single frame by index, for preview generation.
        Should accept frame index (1-based) and return image array.
    diagnosis : DiagnosisContext
        The configured source and page facts; its ``image_type`` must equal
        ``image_type``. Drives ``found`` / ``suggestions`` / ``streams``.
    scan_cache : dict, optional
        One request's source scans keyed by source path, shared by the cameras a
        route validates against the same source so the source is scanned once.
        None scans afresh. Never keep it beyond one request: the disk changes.
    pair_strides : (int, int), optional
        ``(frame_stride, pair_stride)`` of the PIV pairing. Required for a
        ``.pivraw`` source on the PIV page, whose run is paired by trigger
        here; ignored otherwise.

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - Overall validation result
        - found_count: int - Number of files (or container entries/frames) found
        - expected_count: int - Expected number of files
        - camera_path: str - Path to camera directory
        - first_image_preview: str - Base64 PNG of first image (if valid)
        - image_size: tuple - (width, height) of images
        - sample_files: list - Sample of matching filenames
        - format_detected: str - Detected file format
        - error: str or None - Error message if validation failed
        - read_error: str or None - Why a container file that EXISTS would not
          decode. Callers that build their own message must check this: a status
          flag alone cannot tell "file absent" from "file present but
          undecodable", and reporting the former for the latter sends the user
          hunting for a missing file that is right there.
          Kept as a string, not the exception object: this dict is returned
          straight to `jsonify` by the calibration validate route
          (`pivtools_gui/calibration/app/views.py`), and an Exception in it
          raises TypeError there — turning the very failure this field exists to
          report into an HTTP 500 with no message.
        - detected_start_index: int or None - First frame index among the files
          matching the pattern (full listing, standard/im7 only)
        - notes: list - Plain-words facts about how the images are read that do
          not fail validation (``.pivraw``: the run has no dark, so the images
          are raw counts)
        - found: str - What is on disk (or what loaded), one plain sentence
        - suggestions: list - Ranked one-click fixes (``Suggestion`` dicts)
        - streams: list or None - Per-stream table of a ``.set`` recording
        - warnings: list - What should look like a warning, not a fact (a
          ``.set`` that DaVis marked ``SetIsSuspicious``)
        - pivraw_alignment: dict or None - PIV ``.pivraw`` only: the run's
          trigger alignment (``PivrawAlignment.report``: pairs, triggers
          requested, a summary sentence, notes)

        When no pattern can be derived from the folder's files, the reason is
        appended to ``error``.

    Raises
    ------
    ValueError
        When ``diagnosis.image_type`` differs from ``image_type``, or a PIV
        ``.pivraw`` source is validated without ``pair_strides``.
    """
    from .source_diagnosis import apply_diagnosis

    if diagnosis.image_type != image_type:
        raise ValueError(
            f"diagnosis image_type {diagnosis.image_type!r} does not match "
            f"image_type {image_type!r}"
        )
    result = _validate_core(
        camera_path,
        camera,
        image_format,
        image_type,
        expected_count,
        zero_based_indexing,
        read_frame_fn,
        calibration=diagnosis.calibration,
        pair_strides=pair_strides,
        selected_cameras=diagnosis.camera_numbers,
    )
    apply_diagnosis(
        result, diagnosis, camera, camera_path, expected_count, scan_cache=scan_cache
    )
    return result


def _validate_core(
    camera_path: Path,
    camera: int,
    image_format: str,
    image_type: str,
    expected_count: Optional[int],
    zero_based_indexing: bool,
    read_frame_fn: Callable[[int], np.ndarray],
    *,
    calibration: bool,
    pair_strides: Optional[Tuple[int, int]] = None,
    selected_cameras: Sequence[int] = (),
) -> Dict[str, Any]:
    """The validation body of :func:`validate_images_generic`, before diagnosis.

    ``calibration`` selects how a missing-pattern reason is derived (calibration
    never pairs A/B patterns) and whether a ``.pivraw`` run is aligned by
    trigger (PIV only, with ``pair_strides``, over ``selected_cameras``, the
    cameras selected for processing). Returns the result dict without
    ``found`` / ``suggestions`` / ``streams`` / ``warnings``.
    """
    result: Dict[str, Any] = {
        "valid": False,
        "found_count": 0,
        "expected_count": expected_count,
        "camera_path": str(camera_path),
        "first_image_preview": None,
        "read_error": None,
        "image_size": None,
        "sample_files": [],
        "format_detected": None,
        "error": None,
        "detected_start_index": None,  # first frame index among matching files
        "notes": [],  # plain-words facts about how the images are read
        "pivraw_alignment": None,
    }

    if image_type == "pivraw" and not calibration and pair_strides is None:
        raise ValueError(
            "pair_strides is required to validate a .pivraw source for PIV: the "
            "run is paired by trigger with the config's frame and pair strides"
        )
    if expected_count is None and image_type not in ("lavision_set", "cine", "pivraw"):
        raise ValueError(
            f"expected_count is required for {image_type} images (only containers "
            "can be validated without one)"
        )

    start_idx = 0 if zero_based_indexing else 1

    # Check for empty pattern first (glob('') throws ValueError)
    if not image_format or not image_format.strip():
        result["error"] = "Pattern is empty"
        # Still provide suggestions from files in directory
        if camera_path.exists() and camera_path.is_dir():
            all_images = []
            if image_type == "lavision_im7":
                all_images = glob_images(camera_path, "*.im7")
            else:
                all_images.extend(_raster_images(camera_path))
            if all_images:
                all_image_names = [f.name for f in sorted(all_images)]
                result["sample_files"] = all_image_names[:5]
                _fold_pattern_reason(result, all_image_names, image_type, calibration)
        return result

    # Handle container formats
    if image_type == "lavision_set":
        # For .set files, camera_path IS the .set file itself
        set_file = camera_path
        if not set_file.exists():
            result["error"] = f"Set file not found: {set_file}"
            # List sibling .set files (the set_in_folder rule suggests them).
            parent_dir = set_file.parent
            if parent_dir.exists():
                set_files = glob_images(parent_dir, "*.set")
                if set_files:
                    result["sample_files"] = [f.name for f in set_files[:5]]
            return result

        if not set_file.suffix.lower() == ".set":
            result["error"] = f"Expected .set file but got: {set_file}"
            return result

        result["format_detected"] = "set"
        result["sample_files"] = [set_file.name]

        # Get entry count from .set file (opens file briefly, no pixel decode).
        # Import from the concrete submodule (mirrors the .cine branch below) —
        # `get_set_entry_count` is NOT re-exported from the .readers package.
        # A count that cannot be read fails validation visibly: the file exists
        # but its index is unreadable, and an unknown count must not pass.
        try:
            from .readers.set_reader import get_set_entry_count

            result["found_count"] = get_set_entry_count(str(set_file))
        except Exception as e:
            return _container_count_failure(result, set_file, e)
        _check_container_count(result, expected_count)

        # Read the first frame. A read failure is a genuine error (corrupt/locked
        # container, wrong camera), so surface it instead of passing silently.
        # A stream-split or pairing failure is a configuration problem (camera
        # count or selection), not a read failure, so its own text is the error.
        from .readers.set_reader import SetConfigError

        try:
            img = read_frame_fn(1)
        except SetConfigError as e:
            result["valid"] = False
            result["error"] = str(e)
            result["read_error"] = str(e)
            return result
        except Exception as e:
            # Log as well as return: the caller may replace `error` with its own
            # text, and without this the real reason reached neither the screen
            # nor pypiv.log. An unsupported .set pixel encoding was invisible in
            # both for exactly this reason.
            logging.warning(f"Could not read first frame of {set_file}: {e}")
            result["valid"] = False
            result["error"] = f"Set file found but could not read the first frame: {e}"
            result["read_error"] = str(e)
            return result
        try:
            result["image_size"] = (img.shape[1], img.shape[0])  # (W, H)
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not render preview from .set file: {e}")

        return result

    # For non-.set types, camera_path must be a directory
    if not camera_path.exists():
        result["error"] = f"Camera path does not exist: {camera_path}"
        return result

    if image_type == "cine":
        if "%" in image_format:
            cine_filename = image_format % camera
        else:
            cine_filename = image_format
        cine_file = camera_path / cine_filename

        if not cine_file.exists():
            result["error"] = f"CINE file not found: {cine_file}"
            # List the .cine files present (the pattern rule suggests a pattern).
            cine_files = glob_images(camera_path, "*.cine")
            if cine_files:
                result["sample_files"] = [f.name for f in cine_files[:5]]
            return result

        mismatch = cine_extension_mismatch(cine_file)
        if mismatch is not None:
            # read_error too: the PIV route rebuilds its message from read_error
            # for a container whose first frame did not read, and would otherwise
            # replace this with "First frame not found" for a file that exists.
            result["error"] = mismatch
            result["read_error"] = mismatch
            return result

        result["format_detected"] = "cine"
        result["sample_files"] = [cine_filename]

        # A header whose frame count cannot be read fails validation visibly.
        try:
            from .readers.cine_reader import get_cine_frame_count

            result["found_count"] = get_cine_frame_count(str(cine_file))
        except Exception as e:
            return _container_count_failure(result, cine_file, e)
        _check_container_count(result, expected_count)

        # Surface a genuine first-frame read failure.
        try:
            img = read_frame_fn(1)
        except Exception as e:
            logging.warning(f"Could not read first frame of {cine_file}: {e}")
            result["valid"] = False
            result["error"] = f"CINE file found but could not read the first frame: {e}"
            result["read_error"] = str(e)
            return result
        try:
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not render preview from .cine file: {e}")

        return result

    elif image_type == "pivraw":
        if "%" in image_format:
            pivraw_filename = image_format % camera
        else:
            pivraw_filename = image_format
        pivraw_file = camera_path / pivraw_filename

        if not pivraw_file.exists():
            result["error"] = f"PIVRAW file not found: {pivraw_file}"
            # List the .pivraw files present (the pattern rule suggests a pattern).
            pivraw_files = glob_images(camera_path, "*.pivraw")
            if pivraw_files:
                result["sample_files"] = [f.name for f in pivraw_files[:5]]
            return result

        mismatch = pivraw_extension_mismatch(pivraw_file)
        if mismatch is not None:
            # read_error too, for the same reason as the CINE branch above.
            result["error"] = mismatch
            result["read_error"] = mismatch
            return result

        result["format_detected"] = "pivraw"
        result["sample_files"] = [pivraw_filename]

        # A sidecar whose frame count cannot be read (missing field, wrong
        # version, unfinalised run, size mismatch) fails validation visibly.
        try:
            from .readers.pivraw_reader import get_pivraw_frame_count

            result["found_count"] = get_pivraw_frame_count(pivraw_file)
        except Exception as e:
            return _container_count_failure(result, pivraw_file, e)
        # No per-camera count check against num_images: the trigger alignment
        # below refuses a run in which a selected camera lost any frame, and
        # the pair count is min(aligned pairs, the num_images cap)
        # (Config.per_loop_frame_pairs), so no num_images can read past a
        # camera's frames. found_count stays for display.
        result["valid"] = True

        # PIV pairs the run's cameras by trigger. A run that cannot be aligned
        # (a camera offset by a trigger, a flipped A/B phase, a camera with no
        # frames, a dropped frame, ...) fails here with the camera, trigger and reason; read_error
        # too, so the PIV route does not replace it with "First frame not found".
        if not calibration:
            from .readers.pivraw_reader import PivrawError, align_pivraw_run

            try:
                alignment = align_pivraw_run(
                    camera_path, *pair_strides, cameras=selected_cameras
                )
            except PivrawError as e:
                result["valid"] = False
                result["error"] = str(e)
                result["read_error"] = str(e)
                return result
            result["pivraw_alignment"] = alignment.report()

        try:
            img = read_frame_fn(1)
        except Exception as e:
            logging.warning(f"Could not read first frame of {pivraw_file}: {e}")
            result["valid"] = False
            result["error"] = (
                f"PIVRAW file found but could not read the first frame: {e}"
            )
            result["read_error"] = str(e)
            return result
        # The frame read, so the run file and its dark are valid: say when the
        # images are raw because the run was recorded without a dark.
        from .readers.pivraw_reader import pivraw_dark_note

        note = pivraw_dark_note(pivraw_file)
        if note is not None:
            result["notes"].append(note)
        # The frame read, so a preview that will not render is cosmetic: the
        # user reads why there is none, and no size is reported for it.
        try:
            preview = _image_to_base64(img)
        except Exception as e:
            result["notes"].append(
                f"No preview of {pivraw_file.name} frame 1: it read, but could "
                f"not be rendered ({type(e).__name__}: {e})."
            )
        else:
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = preview

        return result

    elif image_type == "lavision_im7":
        # Count .im7 files in directory
        pattern = format_to_glob(image_format)
        matching_files = sorted(glob_images(camera_path, pattern))

        if not matching_files:
            try:
                first_expected = image_format % start_idx
            except (TypeError, ValueError):
                first_expected = image_format

            error_msg = f"First frame not found. Looking for: {first_expected}"
            error_msg = _append_folder_contents(error_msg, camera_path)
            result["error"] = error_msg

            # List the .im7 files present (the pattern rule suggests a pattern).
            im7_files = sorted(glob_images(camera_path, "*.im7"))
            if im7_files:
                im7_names = [f.name for f in im7_files]
                _fold_pattern_reason(result, im7_names, image_type, calibration)
                result["sample_files"] = im7_names[:5]
            return result

        result["found_count"] = len(matching_files)
        result["sample_files"] = [f.name for f in matching_files[:5]]
        result["format_detected"] = "im7"

        start_hint = (
            _validate_file_listing(
                result,
                camera_path,
                image_format,
                matching_files,
                start_idx,
                expected_count,
            )
            is not None
        )

        # Read the first frame. A failure here is a genuine read error (e.g. a
        # multi-camera .im7 frame-count mismatch), not a cosmetic preview issue —
        # surface it and fail validation instead of passing with a missing preview.
        try:
            img = read_frame_fn(1)
        except Exception as e:
            result["valid"] = False
            # The reason already names the file (e.g. the frame-count mismatch).
            # A start-index hint already explains a missing first file; keep it.
            if not start_hint:
                result["error"] = str(e) or type(e).__name__
            return result
        try:
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not render preview from .im7 file: {e}")

        return result

    else:
        # Standard formats
        pattern = format_to_glob(image_format)
        matching_files = sorted(glob_images(camera_path, pattern))

        if not matching_files:
            try:
                first_expected = image_format % start_idx
            except (TypeError, ValueError):
                first_expected = image_format

            error_msg = f"First frame not found. Looking for: {first_expected}"
            error_msg = _append_folder_contents(error_msg, camera_path)
            result["error"] = error_msg

            # Try to find image files and suggest pattern
            all_images = []
            all_images.extend(_raster_images(camera_path))

            if all_images:
                all_image_names = [f.name for f in sorted(all_images)]
                result["sample_files"] = all_image_names[:5]
                _fold_pattern_reason(result, all_image_names, image_type, calibration)
            return result

        result["found_count"] = len(matching_files)
        result["sample_files"] = [f.name for f in matching_files[:5]]
        result["format_detected"] = matching_files[0].suffix.lstrip(".")

        _validate_file_listing(
            result, camera_path, image_format, matching_files, start_idx, expected_count
        )

        # Try to read first for preview (non-fatal if it fails)
        try:
            img = read_frame_fn(1)
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not read preview for camera: {e}")

        return result


# Subfolder names listed by _append_folder_contents before "and N more".
_FOLDER_NAMES_SHOWN = 10
# File names listed by _append_folder_contents before "and N more...".
_FILE_NAMES_SHOWN = 5


def is_calibration_output_dir(path: Path) -> bool:
    """True for PIVTOOLs' own calibration-output folder inside a directory source.

    ``calibration_settings.root_for_source`` puts ``calibration/settings.yaml``
    inside every directory source, so it is never image data. A ``calibration``
    folder without that sidecar is the user's own and is not matched.
    """
    return path.name == CALIBRATION_DIRNAME and os.path.isfile(path / SETTINGS_FILENAME)


def _append_folder_contents(error_msg: str, folder_path: Path) -> str:
    """Append what a folder holds to an error message, ignoring dot-names.

    Files are listed when there are any. A folder with only subfolders says so
    rather than claiming to be empty, and ``Folder is empty`` is kept for a truly
    empty directory. PIVTOOLs' own calibration-output folder is not listed.
    """
    if folder_path.exists() and folder_path.is_dir():
        entries = [
            e
            for e in folder_path.iterdir()
            if not e.name.startswith(".") and not is_calibration_output_dir(e)
        ]
        all_files = sorted(e.name for e in entries if e.is_file())
        folders = sorted(e.name for e in entries if e.is_dir())
        if all_files:
            shown = ", ".join(all_files[:_FILE_NAMES_SHOWN])
            error_msg += f". Found {len(all_files)} files: {shown}"
            if len(all_files) > _FILE_NAMES_SHOWN:
                error_msg += f" and {len(all_files) - _FILE_NAMES_SHOWN} more..."
        elif folders:
            shown = ", ".join(folders[:_FOLDER_NAMES_SHOWN])
            extra = len(folders) - _FOLDER_NAMES_SHOWN
            if extra > 0:
                shown += f" and {extra} more"
            error_msg += f". No files here, only folders: {shown}"
        else:
            error_msg += f". Folder is empty: {folder_path}"
    return error_msg


def _suggest_pattern_for_role(
    sample_files: List[str],
    role: Optional[str] = None,
    forced_ext: Optional[str] = None,
) -> PatternSuggestion:
    """Suggest a pattern from sample files, optionally filtering by role (A/B).

    Parameters
    ----------
    sample_files : List[str]
        List of filenames to analyze
    role : str, optional
        If "A", filter to files with _A suffix. If "B", filter to _B suffix.
    forced_ext : str, optional
        Force a specific extension

    Returns
    -------
    PatternSuggestion
        The suggested pattern, or the reason none could be made.
    """
    if not sample_files:
        raise ValueError("_suggest_pattern_for_role needs at least one filename")

    candidates = sample_files

    # Patterns to match A/B suffixes:
    # - [_-][Aa] matches _A, -A, _a, -a (with separator)
    # - \d[Aa] matches 1A, 2A, etc. (directly after digit, like IMG00001A.tif)
    pattern_a = r"(?:[_-]|(?<=\d))[Aa]\.[a-zA-Z0-9]+$"
    pattern_b = r"(?:[_-]|(?<=\d))[Bb]\.[a-zA-Z0-9]+$"

    # Filter by role if specified
    if role == "A":
        candidates = [f for f in sample_files if re.search(pattern_a, f)]
    elif role == "B":
        candidates = [f for f in sample_files if re.search(pattern_b, f)]

    if not candidates and role:
        # No files match this role - try to derive from opposite role
        opposite_pattern = pattern_b if role == "A" else pattern_a
        opposite_files = [f for f in sample_files if re.search(opposite_pattern, f)]

        if opposite_files:
            # Generate pattern from opposite role, then transform A<->B
            opposite = _suggest_pattern(opposite_files, forced_ext)
            if opposite.pattern is None:
                return opposite
            if role == "A":
                # Transform B to A (preserving case) - handles both _B and digit+B
                return PatternSuggestion(
                    re.sub(
                        r"([Bb])(\.[a-zA-Z0-9]+$)",
                        lambda m: ("A" if m.group(1) == "B" else "a") + m.group(2),
                        opposite.pattern,
                    ),
                    None,
                )
            # Transform A to B (preserving case) - handles both _A and digit+A
            return PatternSuggestion(
                re.sub(
                    r"([Aa])(\.[a-zA-Z0-9]+$)",
                    lambda m: ("B" if m.group(1) == "A" else "b") + m.group(2),
                    opposite.pattern,
                ),
                None,
            )

        # No A/B files at all - fall back to first file (non-A/B naming)
        candidates = sample_files

    return _suggest_pattern(candidates, forced_ext)


def _single_pattern_reason_or_suggestion(
    result: Dict[str, Any],
    names: List[str],
    role: Optional[str],
    patterns: Sequence[str],
    index: int,
) -> None:
    """Offer the role-aware pattern for ``patterns[index]``, or fold why there is none."""
    from .source_diagnosis import role_pattern_suggestion

    reason = _suggest_pattern_for_role(names, role).reason
    if reason is not None:
        _append_error(result, reason)
        return
    suggestion = role_pattern_suggestion(names, role, patterns, index)
    if suggestion is not None:
        result["suggestions"].append(suggestion)


def validate_single_pattern(
    camera_path: Path,
    pattern: str,
    image_type: str,
    expected_count: int,
    zero_based_indexing: bool,
    role: Optional[str] = None,
    *,
    patterns: Sequence[str],
    index: int,
) -> Dict[str, Any]:
    """Validate a single image pattern.

    This function validates one pattern (e.g., pattern A or pattern B) independently,
    returning pattern-specific validation results including suggestions filtered by role.

    Parameters
    ----------
    camera_path : Path
        Path to the camera directory
    pattern : str
        Image format pattern to validate (e.g., "B%05d_A.tif")
    image_type : str
        One of "lavision_set", "lavision_im7", "cine", "pivraw",
        "standard"
    expected_count : int
        Expected number of files matching this pattern
    zero_based_indexing : bool
        Whether file indices are 0-based
    role : str, optional
        "A", "B", or None. Used to filter suggestions to matching role.
    patterns : Sequence[str]
        Every configured pattern; a ``pattern`` suggestion replaces entry ``index``.
    index : int
        Position of ``pattern`` in ``patterns``.

    Returns
    -------
    Dict[str, Any]
        Validation result with keys:
        - valid: bool - Whether pattern validation passed
        - found_count: int - Number of files matching pattern
        - error: str or None - Error message if validation failed; when no
          pattern can be derived from the folder, the reason is appended
        - suggestions: list - ``pattern`` (role-aware) and ``start_index``
          suggestions. A missing camera folder carries none here: the
          camera-level diagnosis owns it.
        - sample_files: List[str] - Sample of matching filenames
    """
    from .source_diagnosis import start_index_suggestion

    result: Dict[str, Any] = {
        "valid": False,
        "found_count": 0,
        "error": None,
        "suggestions": [],
        "sample_files": [],
    }

    # Check for empty pattern first
    if not pattern or not pattern.strip():
        result["error"] = "Pattern is empty"
        # Still provide suggestions from files in directory
        if camera_path.exists() and camera_path.is_dir():
            all_images = []
            if image_type == "lavision_im7":
                all_images = glob_images(camera_path, "*.im7")
            else:
                all_images.extend(_raster_images(camera_path))
            if all_images:
                all_image_names = [f.name for f in sorted(all_images)]
                result["sample_files"] = all_image_names[:5]
                _single_pattern_reason_or_suggestion(
                    result, all_image_names, role, patterns, index
                )
        return result

    # Container formats don't use per-pattern validation
    if image_type in ("lavision_set", "cine", "pivraw"):
        result["valid"] = True
        result["found_count"] = "container"
        return result

    if not camera_path.exists():
        result["error"] = f"Camera path does not exist: {camera_path}"
        return result

    if not camera_path.is_dir():
        result["error"] = f"Camera path is not a directory: {camera_path}"
        return result

    # Convert pattern to glob pattern
    glob_pattern = format_to_glob(pattern)
    matching_files = sorted(glob_images(camera_path, glob_pattern))

    # Calculate the first expected filename
    start_idx = 0 if zero_based_indexing else 1
    try:
        first_expected = pattern % start_idx
    except (TypeError, ValueError):
        first_expected = pattern

    # Check if the first expected file actually exists
    first_file_path = camera_path / first_expected
    first_file_exists = first_file_path.exists()

    # Find all image files (for suggestions)
    all_images = []
    if image_type == "lavision_im7":
        all_images = glob_images(camera_path, "*.im7")
    else:
        all_images.extend(_raster_images(camera_path))
    all_image_names = [f.name for f in sorted(all_images)] if all_images else []

    if not matching_files:
        result["error"] = f"No files matching pattern '{pattern}'"
        if all_image_names:
            result["sample_files"] = all_image_names[:5]
            _single_pattern_reason_or_suggestion(
                result, all_image_names, role, patterns, index
            )
        return result

    # The pattern matches but the first or last expected file is missing because
    # the files start at the other of 0/1: name the start index as the cause.
    # Checked before "Pattern incomplete", whose same-pattern suggestion would
    # otherwise be dropped and leave the user with no reason at all.
    matching_names = [f.name for f in matching_files]
    found_start = _start_index_mismatch(
        camera_path, pattern, matching_names, start_idx, expected_count
    )
    if found_start is not None:
        result["found_count"] = len(matching_files)
        result["sample_files"] = matching_names[:5]
        result["suggestions"].append(start_index_suggestion(found_start, start_idx))
        result["error"] = start_index_hint_message(found_start, start_idx)
        return result

    # Files match the glob, but check if first expected file actually exists
    # This catches cases like "B%05d" matching "B00001_A.tif" via glob "B*"
    # but the actual file "B00001" doesn't exist
    if not first_file_exists:
        result["error"] = f"Pattern incomplete: '{first_expected}' not found"
        result["sample_files"] = (
            all_image_names[:5] if all_image_names else matching_names[:5]
        )
        # Suggest a corrected pattern based on role (an identical one says nothing)
        if all_image_names:
            _single_pattern_reason_or_suggestion(
                result, all_image_names, role, patterns, index
            )
        return result

    result["found_count"] = len(matching_files)
    result["sample_files"] = [f.name for f in matching_files[:5]]

    # Validate count
    if len(matching_files) < expected_count:
        result["error"] = (
            f"Found {len(matching_files)} files, expected {expected_count}"
        )
        result["valid"] = False
    else:
        result["valid"] = True

    return result
