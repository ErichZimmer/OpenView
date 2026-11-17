"""The .im7 reader on pack 1 (DaVis token stream) against lvpyio 1.3.1, and pack 3 refused.

The fixtures in ``fixtures/im7_pack1`` are synthetic pack-1 files, and the
``.lvpyio.npy`` beside each is lvpyio's decode of it, written by
``fixtures/im7_pack1/make_im7_pack1_fixtures.py``. That script also checks lvpyio
returns the values the tokens were designed to give.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from pivtools_core.image_handling.readers.im7_reader import (
    HEADER_SIZE,
    read_im7,
    read_im7_camera,
)

FIXTURES = Path(__file__).parent / "fixtures" / "im7_pack1"


def _fixture(name: str) -> tuple[Path, np.ndarray]:
    return FIXTURES / f"{name}.im7", np.load(FIXTURES / f"{name}.lvpyio.npy")


def test_high_nibble_8_reads_as_minus_8_equal_to_lvpyio():
    path, ref = _fixture("high_nibble_8")
    _, pixels, _ = read_im7(path)
    assert pixels.dtype == np.uint16
    np.testing.assert_array_equal(pixels, ref[0])
    assert pixels.ravel()[:5].tolist() == [5000, 5001, 5003, 4995, 5005]


def test_multi_frame_buffer_equals_lvpyio_on_every_frame():
    path, ref = _fixture("carry_3frames")
    _, pixels, _ = read_im7(path)
    assert pixels.dtype == np.uint16
    np.testing.assert_array_equal(pixels, ref)


@pytest.mark.parametrize(
    "camera_no,frames_per_camera,frames",
    [(1, 1, [0]), (2, 1, [1]), (3, 1, [2]), (1, 2, [0, 1])],
)
def test_camera_read_returns_the_lvpyio_frames_of_that_camera(
    camera_no, frames_per_camera, frames
):
    path, ref = _fixture("carry_3frames")
    got = read_im7_camera(
        path, camera_no=camera_no, frames_per_camera=frames_per_camera
    )
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, ref[frames].astype(np.float32))


def _header(
    pack_type: int, buffer_format: int, size_x: int = 4, size_y: int = 2
) -> bytes:
    h = struct.pack(
        "<hhhh iiii hhh", 0, pack_type, buffer_format, 0, size_x, size_y, 1, 1, 0, 1, 0
    )
    return h + b"\x00" * (HEADER_SIZE - len(h))


@pytest.mark.parametrize("reader", ["read_im7", "read_im7_camera"])
def test_pack3_is_refused_by_name(tmp_path, reader):
    p = tmp_path / "B00001.im7"
    p.write_bytes(_header(3, -4) + b"\x00" * 12)
    with pytest.raises(ValueError, match="pack_type 3"):
        read_im7(p) if reader == "read_im7" else read_im7_camera(p, 1, 1)


@pytest.mark.parametrize("reader", ["read_im7", "read_im7_camera"])
def test_pack1_with_a_float_buffer_is_refused_by_name(tmp_path, reader):
    p = tmp_path / "B00001.im7"
    p.write_bytes(_header(1, -3) + b"\x01\x01\x00" + b"\x05" * 8)
    with pytest.raises(ValueError, match="pack_type 1 with buffer_format -3"):
        read_im7(p) if reader == "read_im7" else read_im7_camera(p, 1, 1)
