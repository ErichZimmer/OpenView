import logging
import os
from pathlib import Path

import cv2
import numpy as np

# BT.601 luma (0.299 R + 0.587 G + 0.114 B), OpenCV's RGB[A]-to-gray rule, by
# channel order and count. One rule for every colour image PIVTOOLs reads.
_LUMA_CODES = {
    ("RGB", 3): cv2.COLOR_RGB2GRAY,
    ("RGB", 4): cv2.COLOR_RGBA2GRAY,
    ("BGR", 3): cv2.COLOR_BGR2GRAY,
    ("BGR", 4): cv2.COLOR_BGRA2GRAY,
}

# PNG specification, IHDR: the first chunk follows the 8-byte signature, and
# after its 4-byte length and b"IHDR" come width (4), height (4), bit depth (1)
# and colour type (1), so the colour type is byte 25.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_BYTES = 26
_PNG_COLOUR_TYPE_BYTE = 25
_PNG_GRAY, _PNG_GRAY_ALPHA = 0, 4
_PNG_COLOUR_TYPES = {
    _PNG_GRAY: "gray",
    2: "RGB",
    3: "palette",
    _PNG_GRAY_ALPHA: "gray+alpha",
    6: "RGBA",
}


def _luma(img: np.ndarray, order: str, file_path: str) -> np.ndarray:
    """BT.601 luma of a channels-last colour image at its own depth.

    Args:
        img: (H, W, 3) or (H, W, 4) colour samples, alpha last and ignored.
        order: "RGB" (tifffile) or "BGR" (OpenCV's imread).
        file_path: Named in the debug line and in a refusal.

    Raises:
        ValueError: The image does not have 3 or 4 channels.
    """
    code = _LUMA_CODES.get((order, img.shape[-1]))
    if code is None:
        raise ValueError(
            f"{Path(file_path).name} has {img.shape[-1]} colour channels; "
            "PIVTOOLs converts 3 (colour) or 4 (colour + alpha) to gray."
        )
    logging.debug(f"Converting colour image to grayscale: {Path(file_path).name}")
    return cv2.cvtColor(img, code)


def _png_colour_type(file_path: str) -> int:
    """The PNG colour type from the file's IHDR chunk.

    Raises:
        ValueError: The file does not start with a PNG signature and IHDR, or
            names a colour type the PNG specification does not define.
    """
    with open(file_path, "rb") as f:
        head = f.read(_PNG_IHDR_BYTES)
    if (
        len(head) < _PNG_IHDR_BYTES
        or head[:8] != _PNG_SIGNATURE
        or head[12:16] != b"IHDR"
    ):
        raise ValueError(
            f"{Path(file_path).name} does not start with a PNG signature and "
            "IHDR chunk, so it is not a PNG file."
        )
    colour_type = head[_PNG_COLOUR_TYPE_BYTE]
    if colour_type not in _PNG_COLOUR_TYPES:
        raise ValueError(
            f"{Path(file_path).name} has PNG colour type {colour_type}, which "
            "the PNG specification does not define."
        )
    return colour_type


def read_tiff(file_path: str) -> np.ndarray:
    """Read a single-page TIFF image using tifffile.

    Colour is decided by the page's sample axis ``S``, which tifffile derives
    from the SamplesPerPixel tag. A page without ``S`` is grayscale whatever
    its width. A page with ``S`` has it moved last, so contiguous and planar
    (``PlanarConfiguration = 2``) colour pages convert alike.

    Returns:
        np.ndarray: Image as float32 for consistent processing.

    Raises:
        ValueError: The file holds more than one page. PIVTOOLs reads one
            image per TIFF file and does not treat pages as frames. Or the
            page has an axis other than Y, X and S (a volumetric page).
    """
    import tifffile

    with tifffile.TiffFile(file_path) as tif:
        n_pages = len(tif.pages)
        if n_pages != 1:
            raise ValueError(
                f"{Path(file_path).name} is a multi-page TIFF with {n_pages} "
                "pages. PIVTOOLs reads one image per TIFF file and does not "
                "treat pages as frames. Save each image as its own TIFF."
            )
        page = tif.pages[0]
        img = page.asarray()
        axes = page.axes
    if set(axes) - {"Y", "X", "S"}:
        raise ValueError(
            f"{Path(file_path).name} has TIFF page axes {axes!r}. PIVTOOLs "
            "reads a 2-D page, optionally with colour samples (axes YX, YXS "
            "or SYX)."
        )
    if "S" in axes:
        img = np.moveaxis(img, axes.index("S"), -1)
        if img.shape[-1] in (3, 4):
            img = _luma(img, "RGB", file_path)
        else:
            img = np.mean(img, axis=-1)
    return img.astype(np.float32)


def read_png_jpeg(file_path: str) -> np.ndarray:
    """Read a PNG or JPEG at its native bit depth, colour converted to gray.

    One decoder, OpenCV's ``imread`` with ``IMREAD_UNCHANGED``: it keeps a
    16-bit PNG's depth (Pillow cut 16-bit colour to its top byte), and it does
    not apply a JPEG's EXIF orientation (``IMREAD_COLOR`` would turn the
    pixels). Colour is converted with the TIFF path's BT.601 rule
    (:func:`_luma`).

    Whether a PNG is colour is decided by its IHDR colour type, not by the
    decoded channel count, because OpenCV expands gray+alpha to four channels
    that look like RGBA. A JPEG holds no alpha, so its decoded shape decides.

    Returns:
        np.ndarray: Image as float32 for consistent processing.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: A gray+alpha PNG, which PIVTOOLs does not read because what
            the alpha means for a particle image is unknown; a file that is not
            the PNG its extension says; a decode OpenCV refuses; or decoded
            channels that disagree with the PNG's colour type.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    name = Path(file_path).name

    colour_type = None
    if Path(file_path).suffix.lower() == ".png":
        colour_type = _png_colour_type(file_path)
        if colour_type == _PNG_GRAY_ALPHA:
            raise ValueError(
                f"{name} is a gray+alpha PNG (colour type 4). PIVTOOLs reads "
                "gray or colour images and does not guess what the alpha "
                "channel means for a particle image. Save it as gray, without "
                "alpha."
            )

    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV could not decode {name}; the file is damaged.")

    if colour_type is not None:
        expect_colour = colour_type != _PNG_GRAY
        if expect_colour != (img.ndim == 3):
            raise ValueError(
                f"{name} is a {_PNG_COLOUR_TYPES[colour_type]} PNG, but OpenCV "
                f"decoded it to shape {img.shape}. Report it, with the file."
            )
    if img.ndim == 3:
        img = _luma(img, "BGR", file_path)
    return img.astype(np.float32)
