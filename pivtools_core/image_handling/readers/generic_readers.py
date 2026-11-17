import logging
import os
from pathlib import Path

import cv2
import numpy as np


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
        logging.debug(
            f"Converting color TIFF image to grayscale: {Path(file_path).name}"
        )
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = np.mean(img, axis=-1)
    return img.astype(np.float32)


def read_png_jpeg(file_path: str) -> np.ndarray:
    """Read PNG/JPEG images using PIL or opencv.

    Returns:
        np.ndarray: Image as float32 for consistent processing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        from PIL import Image

        img = Image.open(file_path)
        if img.mode in ("RGB", "RGBA", "P", "L") and img.mode != "L":
            logging.debug(
                f"Converting color PNG/JPEG image to grayscale: {Path(file_path).name}"
            )
            img = img.convert("L")
        img_array = np.array(img)
        return img_array.astype(np.float32)
    except ImportError:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image file could not be read: {file_path}")
        if img.ndim > 2 and img.shape[-1] > 1:
            logging.debug(
                f"Converting color PNG/JPEG image to grayscale: {Path(file_path).name}"
            )
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = np.mean(img, axis=-1)
        # Handle (H, W, 1) grayscale images - squeeze singleton channel
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]
        return img.astype(np.float32)
