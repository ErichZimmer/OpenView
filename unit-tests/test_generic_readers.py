"""Standard-image readers, tested through ``load_images.read_image``.

Run:
    pytest unit-tests/test_generic_readers.py -v
"""

import struct
import zlib
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


# PNG colour types (PNG specification, IHDR): the number of samples each holds.
PNG_GRAY, PNG_RGB, PNG_PALETTE, PNG_GRAY_ALPHA, PNG_RGBA = 0, 2, 3, 4, 6
_PNG_SAMPLES = {PNG_GRAY: 1, PNG_RGB: 3, PNG_PALETTE: 1, PNG_GRAY_ALPHA: 2, PNG_RGBA: 4}


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body))
    )


def _write_png(path: Path, samples: np.ndarray, colour_type: int, palette=None) -> str:
    """A PNG written from the specification, so its colour type and sample order
    are the test's and not any decoder's convention.

    ``samples`` is (H, W) or (H, W, samples per pixel), uint8 or uint16, in the
    file's own order (R, G, B[, A] or gray[, alpha]). Rows use filter type 0.
    """
    data = samples.reshape(samples.shape[0], samples.shape[1], -1)
    assert data.shape[2] == _PNG_SAMPLES[colour_type]
    depth = 16 if data.dtype == np.uint16 else 8
    big_endian = data.astype(">u2" if depth == 16 else "u1")
    rows = b"".join(b"\x00" + row.tobytes() for row in big_endian)
    height, width = data.shape[:2]
    ihdr = struct.pack(">IIBBBBB", width, height, depth, colour_type, 0, 0, 0)
    chunks = _png_chunk(b"IHDR", ihdr)
    if palette is not None:
        chunks += _png_chunk(b"PLTE", np.asarray(palette, np.uint8).tobytes())
    chunks += _png_chunk(b"IDAT", zlib.compress(rows)) + _png_chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunks)
    return str(path)


def _ramp16() -> np.ndarray:
    return np.linspace(0, 65535, H * W).reshape(H, W).astype(np.uint16)


def test_gray16_png_keeps_every_value(tmp_path):
    ramp = _ramp16()

    out = read_image(_write_png(tmp_path / "gray16.png", ramp, PNG_GRAY))

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, ramp.astype(np.float32))


@pytest.mark.parametrize("colour_type", [PNG_RGB, PNG_RGBA], ids=["rgb16", "rgba16"])
def test_colour16_png_with_equal_channels_reads_as_the_gray16_file(
    tmp_path, colour_type
):
    ramp = _ramp16()
    channels = [ramp, ramp, ramp]
    if colour_type == PNG_RGBA:
        channels.append(np.full_like(ramp, 65535))

    out = read_image(
        _write_png(tmp_path / "colour16.png", np.dstack(channels), colour_type)
    )

    assert out.max() == 65535
    np.testing.assert_array_equal(
        out, read_image(_write_png(tmp_path / "gray16.png", ramp, PNG_GRAY))
    )


# ITU-R BT.601 luma, the TIFF path's rule (cv2 RGB2GRAY): 0.299 R + 0.587 G +
# 0.114 B. For (40000, 20000, 60000): 11960 + 11740 + 6840 = 30540. Red and blue
# swapped would give 34240, so the value also pins the channel order.
RGB16 = (40000, 20000, 60000)
RGB16_LUMA = 30540.0
# OpenCV 4.12's 16-bit RGB2GRAY uses fixed-point weights: measured within 1.52
# counts of exact BT.601 over 262144 random 16-bit pixels (2e-5 of full scale).
LUMA16_FIXED_POINT = 1.6


def test_rgb16_png_reads_as_full_depth_luma(tmp_path):
    rgb = np.empty((H, W, 3), dtype=np.uint16)
    rgb[...] = RGB16

    out = read_image(_write_png(tmp_path / "rgb16.png", rgb, PNG_RGB))

    np.testing.assert_allclose(out, RGB16_LUMA, atol=LUMA16_FIXED_POINT)


def test_rgb8_png_reads_as_the_tiff_luma(tmp_path):
    out = read_image(_write_png(tmp_path / "rgb8.png", _uniform_rgb(), PNG_RGB))

    np.testing.assert_array_equal(out, np.full((H, W), RGB_LUMA, np.float32))


def test_palette_png_reads_as_the_luma_of_its_colours(tmp_path):
    index = np.zeros((H, W), dtype=np.uint8)

    out = read_image(
        _write_png(
            tmp_path / "palette.png", index, PNG_PALETTE, palette=[(100, 50, 200)]
        )
    )

    np.testing.assert_array_equal(out, np.full((H, W), RGB_LUMA, np.float32))


def _random_rgb8(seed: int = 0, side: int = 512) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (side, side, 3), np.uint8)


def test_rgb8_luma_is_bt601_and_differs_from_pillow_by_rounding_only(tmp_path):
    """Colour PNG and JPEG moved from Pillow's convert("L") to the TIFF path's
    cv2 RGB2GRAY (decision of 17 Nov 2026). Both apply the BT.601 weights with
    fixed-point rounding: each is within half a count of the exact luma, so they
    differ by at most 1 count. Measured with Pillow 11.3.0 and OpenCV 4.12 on
    this image: 301 of 262144 pixels differ, all by exactly 1.
    """
    from PIL import Image

    rgb = _random_rgb8()
    path = _write_png(tmp_path / "noise.png", rgb, PNG_RGB)
    exact = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    pillow = np.array(Image.open(path).convert("L")).astype(np.float32)

    out = read_image(path)

    assert np.abs(out - exact).max() <= 0.51
    differ = out != pillow
    assert np.abs(out - pillow).max() == 1
    assert differ.mean() < 0.002


@pytest.mark.parametrize("colour", [True, False], ids=["colour", "gray"])
def test_jpeg_reads_as_its_decoded_samples(tmp_path, colour):
    """A JPEG's samples are lossy, so the truth is Pillow's decode of the same
    file (identical to OpenCV's at these versions), through BT.601 for colour."""
    from PIL import Image

    data = _random_rgb8(side=64) if colour else _random_rgb8(side=64)[..., 1]
    path = str(tmp_path / "frame.jpg")
    Image.fromarray(data).save(path, quality=90)
    decoded = np.array(Image.open(path)).astype(np.float64)
    if colour:
        decoded = decoded @ np.array([0.299, 0.587, 0.114])

    out = read_image(path)

    assert out.shape == (64, 64)
    assert np.abs(out - decoded).max() <= (0.51 if colour else 0)


def test_jpeg_exif_orientation_is_not_applied(tmp_path):
    """Pixels come back as stored: a rotation would move every vector. OpenCV's
    IMREAD_COLOR applies EXIF orientation 6 (a quarter turn); IMREAD_UNCHANGED,
    like Pillow's array, does not."""
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
    path = str(tmp_path / "rotated.jpg")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(path, exif=exif.tobytes())

    assert read_image(path).shape == (H, W)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16], ids=["la8", "la16"])
def test_gray_alpha_png_is_refused_by_name(tmp_path, dtype):
    gray_alpha = np.zeros((H, W, 2), dtype=dtype)

    path = _write_png(tmp_path / "gray_alpha.png", gray_alpha, PNG_GRAY_ALPHA)

    with pytest.raises(ValueError, match=r"gray_alpha\.png.*gray\+alpha"):
        read_image(path)
