"""Shape-based stream -> camera resolution for LaVision .set containers.

A recording may hold a single-frame camera (e.g. PLIF) beside A/B PIV cameras, so
the positional ``(cam-1)*frames_per_camera`` mapping is wrong. The resolver
partitions the streams into consecutive groups of 1 or 2 (a pair must match in
shape) and refuses anything but exactly one valid partition.
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import (  # noqa: E402,F401
    PLIF,
    build_raw16_set,
    clean_set_cache,
    make_set_config,
    write_manifest,
)

from pivtools_core.image_handling.calibration_loader import (  # noqa: E402
    _read_native_frame,
)
from pivtools_core.image_handling.load_images import (  # noqa: E402
    read_pair,
    read_single_frame,
)
from pivtools_core.image_handling.readers import set_reader  # noqa: E402
from pivtools_core.image_handling.readers.lavision_reader import (  # noqa: E402
    read_lavision_ims,
)
from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    camera_streams,
    read_set_pair,
    set_camera_streams,
    valid_stream_splits,
)


def test_valid_splits_plif_three_cameras():
    assert valid_stream_splits(PLIF, 3) == [((0,), (1, 2), (3, 4))]


def test_valid_splits_counts_partitions_when_shapes_equal():
    assert len(valid_stream_splits([(4, 6)] * 4, 3)) == 3  # 1+1+2, 1+2+1, 2+1+1
    assert valid_stream_splits([(4, 6)] * 6, 3) == [((0, 1), (2, 3), (4, 5))]
    assert valid_stream_splits([(4, 6)] * 3, 3) == [((0,), (1,), (2,))]
    assert valid_stream_splits([(4, 6)] * 5, 2) == []  # max 4 streams for 2 cameras
    assert valid_stream_splits([(4, 6)] * 2, 0) == []


def test_valid_splits_prune_keeps_long_dead_end_lists_fast():
    # 24 equal-shape streams into 12 cameras has exactly one split (all pairs).
    # Unpruned, the walk explores every 1/2 composition of the prefix; the prune
    # cuts branches whose remaining streams cannot fill the remaining cameras.
    assert valid_stream_splits([(4, 6)] * 24, 12) == [
        tuple((2 * c, 2 * c + 1) for c in range(12))
    ]


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


def test_set_camera_streams_plif(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    assert set_camera_streams(s, 3) == [(0,), (1, 2), (3, 4)]


def test_set_camera_streams_cannot_split_names_counts_that_fit(
    tmp_path, clean_set_cache
):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"cannot split into 2 cameras") as e:
        set_camera_streams(s, 2)
    assert "Camera counts that fit: 3, 5" in str(e.value)
    assert "0: 8\u00d76 raw-16-bit" in str(e.value)  # stream table, width first


def test_set_camera_streams_ambiguous_raises(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)] * 4)
    with pytest.raises(ValueError, match=r"3 different splits fit 3 cameras"):
        set_camera_streams(s, 3)


def test_camera_streams_pair_and_single(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    assert camera_streams(s, 2, 3, pair=True) == (1, 2)
    assert camera_streams(s, 1, 3, pair=False) == (0,)


def test_pairing_check_names_single_stream_camera(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(
        ValueError, match=r"Camera 1 has one stream \(0: 8\u00d76\).*Deselect camera 1"
    ):
        camera_streams(s, 1, 3, pair=True)


def test_camera_streams_rejects_out_of_range_camera(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"Camera 4 is outside 1\.\.3"):
        camera_streams(s, 4, 3, pair=False)


def test_split_is_computed_once_per_container(tmp_path, clean_set_cache, monkeypatch):
    s = build_raw16_set(tmp_path, PLIF)
    calls = []
    real = set_reader.valid_stream_splits
    monkeypatch.setattr(
        set_reader, "valid_stream_splits", lambda *a: calls.append(a) or real(*a)
    )
    for _ in range(3):
        set_camera_streams(s, 3)
    assert len(calls) == 1


def test_split_cache_follows_set_info_invalidation(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)] * 4, name="shrink")
    assert set_camera_streams(s, 2) == [(0, 1), (2, 3)]

    # Shrink the container to 2 streams (files and manifest) and bump the mtimes
    # that _set_info_cache_key stats (the .set file and its companion folder). The
    # camera count stays 2, so a stale stream_splits entry would return the old
    # split instead of [(0,), (1,)].
    folder = tmp_path / "shrink"
    for stream in (2, 3):
        for f in folder.glob(f"Frame{stream}-*"):
            f.unlink()
    write_manifest(folder, frame_readers=[("Frame0", 0), ("Frame1", 1)])
    for path in (s, folder):
        st = path.stat()
        os.utime(path, (st.st_atime, st.st_mtime + 10))

    assert set_camera_streams(s, 2) == [(0,), (1,)]


def test_plif_camera2_prepaired_reads_streams_1_2(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF, n_entries=2, fill_with_index=True)
    cfg = make_set_config(
        tmp_path, s, camera_count=3, camera_numbers=[2, 3], frame_stride=0
    )
    pair = read_pair(1, s, 2, cfg)
    assert pair[0, 0, 0] == 100 and pair[1, 0, 0] == 200  # streams 1, 2 -- not 2, 3
    pair3 = read_pair(1, s, 3, cfg)
    assert pair3[0, 0, 0] == 300 and pair3[1, 0, 0] == 400


def test_plif_time_resolved_camera2_reads_stream_1_across_entries(
    tmp_path, clean_set_cache
):
    s = build_raw16_set(tmp_path, PLIF, n_entries=2, fill_with_index=True)
    cfg = make_set_config(
        tmp_path, s, camera_count=3, camera_numbers=[2, 3], frame_stride=1
    )
    pair = read_pair(1, s, 2, cfg)
    assert (
        pair[0, 0, 0] == 100 and pair[1, 0, 0] == 101
    )  # first stream of camera 2, entries 1 and 2


@pytest.mark.parametrize("num_cameras", [0, -1])
def test_invalid_camera_count_raises_clearly(tmp_path, clean_set_cache, num_cameras):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=rf"Invalid camera count {num_cameras} for"):
        set_camera_streams(s, num_cameras)
    with pytest.raises(ValueError, match=rf"Invalid camera count {num_cameras} for"):
        camera_streams(s, 1, num_cameras, pair=False)


def test_read_set_pair_rejects_wrong_tuple_length(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"exactly two frame streams"):
        read_set_pair(s, (1, 2, 3), 1)


def test_read_set_pair_rejects_out_of_range_stream(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"Frame stream 5 is outside 0\.\.4"):
        read_set_pair(s, (1, 5), 1)


def test_read_set_pair_rejects_shape_mismatch(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"differ in shape"):
        read_set_pair(s, (0, 1), 1)


def test_prepaired_read_of_single_stream_camera_raises(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    cfg = make_set_config(
        tmp_path, s, camera_count=3, camera_numbers=[1, 2, 3], frame_stride=0
    )
    with pytest.raises(ValueError, match=r"Camera 1 has one stream"):
        read_pair(1, s, 1, cfg)


def test_plif_shape_probe_answers_per_camera(tmp_path, clean_set_cache):
    """The PLIF camera (6 x 8) and the PIV cameras (4 x 6) each report their
    own shape from one config, whichever camera is asked first."""
    s = build_raw16_set(tmp_path, PLIF)
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    pre_paired = make_set_config(dir_a, s, 3, [2, 3], frame_stride=0)
    assert pre_paired.image_shape_for(3) == (4, 6)
    assert pre_paired.image_shape_for(2) == (4, 6)
    time_resolved = make_set_config(dir_b, s, 3, [1, 2, 3], frame_stride=1)
    assert time_resolved.image_shape_for(2) == (4, 6)
    assert time_resolved.image_shape_for(1) == (6, 8)
    assert time_resolved.image_shape_for(3) == (4, 6)


def test_plif_calibration_camera2_reads_stream_1(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF, n_entries=2, fill_with_index=True)
    img = _read_native_frame(
        file_path=s,
        camera=2,
        frame_idx=1,
        image_type="lavision_set",
        num_cameras=3,
        use_camera_subfolders=False,
    )
    assert img.shape == (4, 6) and img[0, 0] == 100


def test_read_single_frame_set_requires_num_cameras(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(ValueError, match=r"num_cameras"):
        read_single_frame(s, camera=2, frame_idx=1, image_type="lavision_set")


def test_registry_set_reader_resolves_streams(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF, fill_with_index=True)
    pair = read_lavision_ims(str(s), camera_no=2, im_no=1, num_cameras=3)
    assert pair[0, 0, 0] == 100 and pair[1, 0, 0] == 200


def test_resolver_config_errors_are_set_config_errors(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    with pytest.raises(set_reader.SetConfigError, match=r"cannot split into 2"):
        set_camera_streams(s, 2)
    with pytest.raises(set_reader.SetConfigError, match=r"ambiguous"):
        set_camera_streams(s, 4)
    with pytest.raises(set_reader.SetConfigError, match=r"outside 1\.\.3"):
        camera_streams(s, 4, 3, pair=False)
    with pytest.raises(set_reader.SetPairingError):
        camera_streams(s, 1, 3, pair=True)
    assert issubclass(set_reader.SetPairingError, set_reader.SetConfigError)
    assert issubclass(set_reader.SetConfigError, ValueError)
