"""Shape-based stream -> camera resolution for LaVision .set containers.

A recording may hold a single-frame camera (e.g. PLIF) beside A/B PIV cameras, so
the positional ``(cam-1)*frames_per_camera`` mapping is wrong. The resolver
partitions the streams into consecutive groups of 1 or 2 (a pair must match in
shape) and refuses anything but exactly one valid partition.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import build_raw16_set  # noqa: E402

from pivtools_core.image_handling.readers import set_reader  # noqa: E402
from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    camera_streams,
    clear_set_info_cache,
    set_camera_streams,
    valid_stream_splits,
)

# Stream 0 differs, streams 1-4 are equal (2160x2560 + 4 x 2048x2048, scaled down).
PLIF = [(6, 8)] + [(4, 6)] * 4


@pytest.fixture()
def _clean_set_cache():
    clear_set_info_cache()
    yield
    clear_set_info_cache()


def test_valid_splits_plif_three_cameras():
    assert valid_stream_splits(PLIF, 3) == [((0,), (1, 2), (3, 4))]


def test_valid_splits_counts_partitions_when_shapes_equal():
    assert len(valid_stream_splits([(4, 6)] * 4, 3)) == 3  # 1+1+2, 1+2+1, 2+1+1
    assert valid_stream_splits([(4, 6)] * 6, 3) == [((0, 1), (2, 3), (4, 5))]
    assert valid_stream_splits([(4, 6)] * 3, 3) == [((0,), (1,), (2,))]
    assert valid_stream_splits([(4, 6)] * 5, 2) == []  # max 4 streams for 2 cameras
    assert valid_stream_splits([(4, 6)] * 2, 0) == []


def test_uniform_layouts_match_old_stride():
    # Every n = k or n = 2k case gives the old (cam-1)*fpc mapping, shapes irrelevant
    # (alk235: 5 differing shapes).
    alk = [(3472 + 16 * i, 4) for i in range(5)]
    assert valid_stream_splits(alk, 5) == [tuple((i,) for i in range(5))]
    for k in (1, 2, 3, 4):
        assert valid_stream_splits([(4, 6)] * (2 * k), k) == [
            tuple((2 * c, 2 * c + 1) for c in range(k))
        ]
        assert valid_stream_splits([(4, 6)] * k, k) == [tuple((c,) for c in range(k))]


def test_set_camera_streams_plif(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    assert set_camera_streams(s, 3) == [(0,), (1, 2), (3, 4)]


def test_set_camera_streams_cannot_split_names_counts_that_fit(
    tmp_path, _clean_set_cache
):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"cannot split into 2 cameras") as e:
        set_camera_streams(s, 2)
    assert "Camera counts that fit: 3, 5" in str(e.value)
    assert "0: 6x8 raw-16-bit" in str(e.value)  # stream table


def test_set_camera_streams_ambiguous_raises(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)] * 4)
    with pytest.raises(ValueError, match=r"3 different splits fit 3 cameras"):
        set_camera_streams(s, 3)


def test_camera_streams_pair_and_single(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    assert camera_streams(s, 2, 3, pair=True) == (1, 2)
    assert camera_streams(s, 1, 3, pair=False) == (0,)


def test_pairing_check_names_single_stream_camera(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(
        ValueError, match=r"Camera 1 has one stream \(0: 6x8\).*Deselect camera 1"
    ):
        camera_streams(s, 1, 3, pair=True)


def test_camera_streams_rejects_out_of_range_camera(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"Camera 4 is outside 1\.\.3"):
        camera_streams(s, 4, 3, pair=False)


def test_split_is_computed_once_per_container(tmp_path, _clean_set_cache, monkeypatch):
    s = build_raw16_set(tmp_path, PLIF)
    calls = []
    real = set_reader.valid_stream_splits
    monkeypatch.setattr(
        set_reader, "valid_stream_splits", lambda *a: calls.append(a) or real(*a)
    )
    for _ in range(3):
        set_camera_streams(s, 3)
    assert len(calls) == 1


def test_split_cache_follows_set_info_invalidation(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)] * 4, name="shrink")
    assert set_camera_streams(s, 2) == [(0, 1), (2, 3)]

    # Shrink the container to 2 streams and bump the mtimes that
    # _set_info_cache_key stats (the .set file and its companion folder). The
    # camera count stays 2, so a stale stream_splits entry would return the old
    # split instead of [(0,), (1,)].
    folder = tmp_path / "shrink"
    for stream in (2, 3):
        for f in folder.glob(f"Frame{stream}-*"):
            f.unlink()
    for path in (s, folder):
        st = path.stat()
        os.utime(path, (st.st_atime, st.st_mtime + 10))

    assert set_camera_streams(s, 2) == [(0,), (1,)]
