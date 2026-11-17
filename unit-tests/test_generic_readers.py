"""Standard-image readers, tested through ``load_images.read_image``.

Run:
    pytest unit-tests/test_generic_readers.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from pivtools_core.image_handling.load_images import read_image

H, W = 16, 24


def _write(path: Path, data: np.ndarray, **kwargs) -> str:
    tifffile.imwrite(path, data, **kwargs)
    return str(path)


def _uniform_rgb() -> np.ndarray:
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 100, 50, 200
    return rgb


# ITU-R BT.601 luma of (100, 50, 200): 0.299*100 + 0.587*50 + 0.114*200 = 82.05,
# which OpenCV's 8-bit RGB2GRAY rounds to 82.
RGB_LUMA = 82.0


@pytest.mark.parametrize(("pages", "width"), [(2, W), (5, W), (2, 3), (2, 4)], ids=str)
def test_multipage_tiff_is_refused_with_its_page_count(tmp_path, pages, width):
    data = np.arange(pages * H * width, dtype=np.uint16).reshape(pages, H, width)
    path = _write(tmp_path / "stack.tif", data, photometric="minisblack")

    with pytest.raises(ValueError, match=rf"stack\.tif.*{pages} pages"):
        read_image(path)


def test_volumetric_tiff_page_is_refused_with_its_axes(tmp_path):
    depth = 4
    data = np.zeros((depth, H, H), dtype=np.uint16)
    path = _write(
        tmp_path / "volume.tif",
        data,
        volumetric=True,
        tile=(depth, H, H),
        photometric="minisblack",
    )

    with pytest.raises(ValueError, match=r"volume\.tif.*axes 'ZYX'"):
        read_image(path)


@pytest.mark.parametrize("width", [3, 4, W])
def test_grayscale_tiff_keeps_every_column_and_value(tmp_path, width):
    data = np.arange(H * width, dtype=np.uint16).reshape(H, width) * 97
    path = _write(tmp_path / "gray.tif", data, photometric="minisblack")

    out = read_image(path)

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, data.astype(np.float32))


@pytest.mark.parametrize("planarconfig", ["contig", "separate"])
def test_rgb_tiff_reads_as_luma_whatever_its_sample_layout(tmp_path, planarconfig):
    rgb = _uniform_rgb()
    stored = rgb if planarconfig == "contig" else np.moveaxis(rgb, -1, 0).copy()
    path = _write(
        tmp_path / "rgb.tif", stored, photometric="rgb", planarconfig=planarconfig
    )

    out = read_image(path)

    assert out.shape == (H, W)
    np.testing.assert_array_equal(out, np.full((H, W), RGB_LUMA, np.float32))


@pytest.mark.parametrize(
    "name", ["B00001.raw", "B00001.cr2", "B00001.NEF", "B00001.arw"]
)
def test_camera_raw_is_refused_as_removed(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\x00" * 64)

    with pytest.raises(ValueError, match=rf"{Path(name).suffix}.*removed"):
        read_image(str(path))
