"""
CINE file reader for Phantom high-speed cameras.

.cine files are single-camera containers with custom frame indexing.
Key difference from .im7/.set: one .cine file per camera, not all cameras in one file.

Directory structure:
    source_dir/Camera1.cine
    source_dir/Camera2.cine

Frame indexing:
    .cine files have a FirstImageNo offset (can be negative for pre-trigger frames)
    User sees: Pair 1, Pair 2, Pair 3...
    Internal: actual_frame = user_frame + FirstImageNo - 1 (handled transparently)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cinereader as cr
import numpy as np

from .out_buffer import check_out

# Metadata cache: {file_path: (metadata, last_modified_time)}
_metadata_cache = {}


def _get_cached_metadata(file_path: str):
    """Get cached cine metadata, reload if file changed.

    Caching avoids re-reading the header for each frame pair,
    which significantly improves performance for large datasets.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CINE file not found: {file_path}")

    mtime = path.stat().st_mtime

    if file_path in _metadata_cache:
        cached_meta, cached_mtime = _metadata_cache[file_path]
        if cached_mtime == mtime:
            return cached_meta

    logging.debug(f"Reading CINE metadata: {file_path}")
    metadata = cr.read_metadata(file_path)
    _metadata_cache[file_path] = (metadata, mtime)
    return metadata


def _internal_frame(metadata, frame: int, which: str, file_path: str) -> int:
    """The camera's frame number for 1-based ``frame``, refused if outside the file.

    Frames are numbered from 1 for the user; the camera numbers them from
    ``FirstImageNo``, which is negative for pre-trigger frames.
    """
    first_frame = metadata.FirstImageNo
    last_frame = first_frame + metadata.ImageCount - 1
    internal = frame + first_frame - 1
    if not first_frame <= internal <= last_frame:
        raise ValueError(
            f"{os.path.basename(file_path)}: {which} {frame} (camera frame "
            f"{internal}) is outside the file, which holds frames 1 to "
            f"{metadata.ImageCount} (camera frames {first_frame} to {last_frame})."
        )
    return internal


def read_cine_pair(
    file_path: str,
    idx: int,
    *,
    frame_b_idx: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read frames A and B of one pair from a .cine file.

    ``load_images.read_pair`` passes both frame numbers from
    ``Config.get_frame_pair_indices``. Frame B is required: this reader once
    read ``idx + 1`` as frame B whatever ``frame_stride`` asked for, so a
    wide-gap pairing correlated the wrong frames with no error.

    Always reads exactly two frames — use :func:`read_cine_single` for one.

    Args:
        file_path: Path to .cine file
        idx: Frame A number (1-based, from pairing logic)
        frame_b_idx: Frame B number (1-based, from pairing logic)
        out: Optional (2, H, W) float32 C-contiguous destination; the two
            decoded frames are widened straight into it (one copy instead of
            the stack-then-astype two), and it is returned. Undefined after an
            exception.

    Returns:
        np.ndarray: Shape (2, H, W) with frame A and B as float32

    Raises:
        FileNotFoundError: If the .cine file doesn't exist
        ValueError: If either frame is outside the file
    """
    metadata = _get_cached_metadata(file_path)
    internal_a = _internal_frame(metadata, idx, "frame A", file_path)
    internal_b = _internal_frame(metadata, frame_b_idx, "frame B", file_path)

    frame_a = cr.read_image(metadata, file_path, internal_a)
    frame_b = cr.read_image(metadata, file_path, internal_b)

    if out is None:
        return np.stack([frame_a, frame_b], axis=0).astype(np.float32)

    # Explicit shape checks before np.copyto: it broadcasts silently, and a
    # (1, H, W) or (H, 1) frame would fill the slot with plausible wrong pixels.
    what = (
        f"read_cine_pair frames ({idx}, {frame_b_idx}) of {os.path.basename(file_path)}"
    )
    if frame_b.shape != frame_a.shape:
        raise ValueError(
            f"{what}: frames A and B decoded to different shapes "
            f"{frame_a.shape} and {frame_b.shape}."
        )
    check_out(out, (2,) + tuple(frame_a.shape), what)
    np.copyto(out[0], frame_a)
    np.copyto(out[1], frame_b)
    return out


def read_cine_single(file_path: str, idx: int = 1) -> np.ndarray:
    """Read a single frame from .cine file.

    Args:
        file_path: Path to .cine file
        idx: Frame index (1-based, user-facing)

    Returns:
        np.ndarray: Shape (H, W) single frame as float32
    """
    metadata = _get_cached_metadata(file_path)
    internal_idx = _internal_frame(metadata, idx, "frame", file_path)
    frame = cr.read_image(metadata, file_path, internal_idx)
    return frame.astype(np.float32)


def get_cine_frame_count(file_path: str) -> int:
    """Get total number of frames in .cine file.

    Args:
        file_path: Path to .cine file

    Returns:
        int: Total frame count
    """
    metadata = _get_cached_metadata(file_path)
    return metadata.ImageCount


def get_cine_first_frame_no(file_path: str) -> int:
    """Get FirstImageNo for a .cine file.

    This is mainly useful for debugging - user-facing code should
    use 1-based indexing which is translated automatically.

    Args:
        file_path: Path to .cine file

    Returns:
        int: First frame number in camera's numbering scheme
    """
    metadata = _get_cached_metadata(file_path)
    return metadata.FirstImageNo


def get_cine_image_shape(file_path: str) -> Tuple[int, int]:
    """Get image dimensions from .cine file metadata.

    Args:
        file_path: Path to .cine file

    Returns:
        Tuple[int, int]: (height, width) of images
    """
    metadata = _get_cached_metadata(file_path)
    # Read first frame to get actual shape (metadata has biWidth/biHeight but
    # reading a frame ensures we get the exact array dimensions)
    first_frame = cr.read_image(metadata, file_path, metadata.FirstImageNo)
    return first_frame.shape


def get_cine_metadata_info(file_path: str) -> dict:
    """Get comprehensive metadata info for a .cine file.

    Useful for debugging and validation.

    Args:
        file_path: Path to .cine file

    Returns:
        dict: Metadata information including frame count, dimensions, etc.
    """
    metadata = _get_cached_metadata(file_path)

    return {
        "first_image_no": metadata.FirstImageNo,
        "image_count": metadata.ImageCount,
        "last_image_no": metadata.FirstImageNo + metadata.ImageCount - 1,
        "width": metadata.biWidth,
        "height": metadata.biHeight,
        "bit_depth": metadata.biBitCount,
        "real_bpp": getattr(metadata, "RealBPP", metadata.biBitCount),
    }


def clear_metadata_cache():
    """Clear the metadata cache.

    Useful for testing or when files have been modified.
    """
    global _metadata_cache
    _metadata_cache = {}
