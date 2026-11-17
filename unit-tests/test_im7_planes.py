"""``.im7`` camera reads take one plane per frame (``size_z`` = 1).

The ``.im7`` counterpart of ``.ims`` header word 14 (planes per frame, which
must be 0, Morgan 2026-11-17) is ``size_z``, whose single-plane value is 1.
Every camera and PIV ``.im7`` on the surveyed drives has ``size_z`` 1; the only
``size_z`` 2 files are DaVis's float dewarped-image output (drive survey D).
"""

import struct

import numpy as np
import pytest
from fixtures.lavision import _IM7_TRAILER

from pivtools_core.image_handling.readers.im7_reader import read_im7_camera

# Image_Header_7 "<hhhh iiii": version 0, pack_type 0 (uncompressed),
# buffer_format -4 (WORD), is_sparse 0, then size_x, size_y, size_z, size_f.
_WORD = -4


def _write_im7(path, size_z, size_f=2, width=4, height=3):
    header = bytearray(256)
    struct.pack_into("<hhhh", header, 0, 0, 0, _WORD, 0)
    struct.pack_into("<iiii", header, 8, width, height, size_z, size_f)
    pixels = np.arange(width * height * size_z * size_f, dtype="<u2")
    path.write_bytes(bytes(header) + pixels.tobytes() + _IM7_TRAILER)
    return pixels.reshape(size_f, size_z, height, width)


def test_a_single_plane_camera_reads(tmp_path):
    path = tmp_path / "B00001.im7"
    pixels = _write_im7(path, size_z=1)
    frames = read_im7_camera(path, 1, 2)
    np.testing.assert_array_equal(frames, pixels[:, 0].astype(np.float32))


@pytest.mark.parametrize("size_z", [2, 3])
def test_more_than_one_plane_is_refused_by_name(tmp_path, size_z):
    path = tmp_path / "B00001.im7"
    _write_im7(path, size_z=size_z)
    with pytest.raises(ValueError) as info:
        read_im7_camera(path, 1, 2)
    message = str(info.value)
    assert "B00001.im7" in message
    assert f"size_z (planes per frame) is {size_z}" in message
