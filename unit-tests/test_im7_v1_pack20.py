"""The .im7 reader on version-1 pack 20 (one LZ4 block per frame) against lvpyio 1.3.1.

``fixtures/im7_v1_pack20/three_frames.im7`` is a synthetic version-1 file built
to the layout measured on the real files, and ``three_frames.lvpyio.npy`` is
lvpyio's decode of it. Both are written by
``fixtures/im7_v1_pack20/make_im7_v1_pack20_fixture.py``, which also checks that
lvpyio returns the designed pixels and the intensity scale 2 x + 1.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from pivtools_core.image_handling.readers.im7_reader import (
    get_im7_frame_count,
    get_im7_frame_shape,
    read_im7,
    read_im7_camera,
)

FIXTURES = Path(__file__).parent / "fixtures" / "im7_v1_pack20"
FIXTURE = FIXTURES / "three_frames.im7"
LVPYIO = np.load(FIXTURES / "three_frames.lvpyio.npy")


def test_whole_buffer_equals_lvpyio_on_every_frame():
    header, pixels, scales = read_im7(FIXTURE)
    assert header.version == 1
    assert pixels.dtype == np.uint16
    np.testing.assert_array_equal(pixels, LVPYIO)
    assert (scales.slope, scales.offset) == (2.0, 1.0)


@pytest.mark.parametrize(
    "camera_no,frames_per_camera,frames,expected",
    [
        (1, 1, None, [0]),
        (2, 1, None, [1]),
        (3, 1, None, [2]),
        (1, 2, None, [0, 1]),
        (2, 2, None, [2]),  # clamped: the buffer ends after frame 2
        (1, 2, 1, [0]),
    ],
)
def test_camera_read_returns_that_cameras_lvpyio_frames_scaled(
    camera_no, frames_per_camera, frames, expected
):
    # The fixture's intensity scale is 2 x + 1, so a camera read that skipped
    # frames and then missed the attribute records would not apply it.
    got = read_im7_camera(
        FIXTURE, camera_no=camera_no, frames_per_camera=frames_per_camera, frames=frames
    )
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, LVPYIO[expected].astype(np.float32) * 2 + 1)


ENTRY_POINTS = {
    "read_im7": read_im7,
    "read_im7_camera": lambda p: read_im7_camera(p, 1, 1),
    "get_im7_frame_count": get_im7_frame_count,
    "get_im7_frame_shape": get_im7_frame_shape,
}


def _patched(tmp_path, version: int, pack_type: int) -> Path:
    """The fixture with its header version and pack type overwritten."""
    data = bytearray(FIXTURE.read_bytes())
    struct.pack_into("<hh", data, 0, version, pack_type)
    p = tmp_path / "B00007.im7"
    p.write_bytes(bytes(data))
    return p


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_unknown_header_version_is_refused_by_name(tmp_path, entry):
    p = _patched(tmp_path, version=2, pack_type=20)
    with pytest.raises(ValueError, match=r"B00007\.im7: \.im7 header version 2"):
        ENTRY_POINTS[entry](p)


def _frame_starts() -> list[int]:
    """Byte offset of each fixture frame's int64 size, walked from the header."""
    data = FIXTURE.read_bytes()
    starts, pos = [], 256
    for _ in range(LVPYIO.shape[0]):
        starts.append(pos)
        pos += 8 + struct.unpack_from("<q", data, pos)[0]
    return starts


READERS = {
    "read_im7": read_im7,
    # Camera 1 holds frame 0 only: frames 1 and 2 are skipped, not decoded.
    "read_im7_camera": lambda p: read_im7_camera(p, 1, 1),
}


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize(
    "cut,message",
    [
        (
            lambda s: s[1] + 8 + 5,
            r"B00007\.im7: version-1 LZ4 frame 1 of 3 declares \d+ compressed "
            r"bytes at byte \d+, but only 5 remain\. The file is truncated\.",
        ),
        (
            lambda s: s[2] + 3,
            r"B00007\.im7: version-1 LZ4 frame 2 of 3 is cut off: its 8-byte "
            r"size at byte \d+ has 3 bytes\. The file is truncated\.",
        ),
    ],
    ids=["inside-frame-1-block", "inside-frame-2-size"],
)
def test_truncated_file_is_refused_by_name(tmp_path, reader, cut, message):
    p = tmp_path / "B00007.im7"
    p.write_bytes(FIXTURE.read_bytes()[: cut(_frame_starts())])
    with pytest.raises(IOError, match=message):
        READERS[reader](p)


@pytest.mark.parametrize("reader", READERS)
def test_frame_offset_table_that_disagrees_with_the_blocks_is_refused(tmp_path, reader):
    starts = _frame_starts()
    data = bytearray(FIXTURE.read_bytes())
    table_at = starts[-1] + 8 + struct.unpack_from("<q", data, starts[-1])[0]
    struct.pack_into("<q", data, table_at + 8, starts[1] + 1)
    p = tmp_path / "B00007.im7"
    p.write_bytes(bytes(data))
    with pytest.raises(IOError, match=r"B00007\.im7: the version-1 frame-offset table"):
        READERS[reader](p)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_version_1_on_a_pack_type_other_than_20_is_refused_by_name(tmp_path, entry):
    p = _patched(tmp_path, version=1, pack_type=0)
    with pytest.raises(
        ValueError, match=r"B00007\.im7: \.im7 header version 1 with pack_type 0"
    ):
        ENTRY_POINTS[entry](p)
