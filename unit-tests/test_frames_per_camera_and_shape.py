"""Frames-per-camera guards and .set shape detection.

Pins three Batch-1 fixes from the image-reading audit (2026-08-24):

* ``_detect_im7_frames_per_camera`` accepts a derived frames-per-camera of
  exactly 1 or 2, never any other divisor, and ``set_camera_streams`` rejects
  a 10-stream .set with ``camera_count: 2`` (it used to yield stride 5 and
  silently read another camera's pixels).
* ``Config.image_shape_for`` has no config override: the old ``images["shape"]``
  branch read a key nothing wrote, and honoring a stale stored shape would
  silently break the window grids.
* ``Config._detect_image_shape`` takes a .set camera's shape from container
  index metadata (its OWN stream), not from a pixel decode through the
  pre-paired pair reader — which crashed on time-resolved containers and
  returned a wrong stream's shape on mixed-resolution rigs.

Usage:
    pytest unit-tests/test_frames_per_camera_and_shape.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import build_raw16_set, make_set_config

from pivtools_core.config import Config
from pivtools_core.image_handling.load_images import (
    _detect_im7_frames_per_camera,
)
from pivtools_core.image_handling.readers import im7_reader
from pivtools_core.image_handling.readers.set_reader import (
    clear_set_info_cache,
    set_camera_streams,
)


@pytest.fixture()
def _clean_set_cache():
    clear_set_info_cache()
    yield
    clear_set_info_cache()


# ---------------------------------------------------------------------------
# .set stream split: only unique consecutive groups of 1 or 2 are real
# ---------------------------------------------------------------------------


def test_set_split_rejects_ten_streams_two_cameras(tmp_path, _clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)] * 10)
    with pytest.raises(
        ValueError, match=r"cannot split into 2 cameras.*Camera counts that fit: 5, 10"
    ):
        set_camera_streams(s, 2)


def test_set_split_accepts_the_two_uniform_layouts(tmp_path, _clean_set_cache):
    four = build_raw16_set(tmp_path, [(4, 6)] * 4, name="four")
    five = build_raw16_set(tmp_path, [(4, 6)] * 5, name="five")
    assert set_camera_streams(four, 2) == [(0, 1), (2, 3)]
    assert set_camera_streams(four, 4) == [(0,), (1,), (2,), (3,)]
    assert set_camera_streams(five, 5) == [(i,) for i in range(5)]
    with pytest.raises(ValueError, match=r"cannot split into 1 camera of 1 or 2"):
        set_camera_streams(five, 1)


# ---------------------------------------------------------------------------
# .im7 frames-per-camera: same guard, same reasoning
# ---------------------------------------------------------------------------


def _fake_frame_count(monkeypatch, size_f):
    monkeypatch.setattr(im7_reader, "get_im7_frame_count", lambda _p: size_f)


def test_im7_fpc_rejects_divisor_that_is_not_a_layout(tmp_path, monkeypatch):
    """size_f=8 with camera_count 2 divides to 4 — camera 2's slice would start
    at frame 4, which is camera 3's data in the real 4-camera file."""
    _fake_frame_count(monkeypatch, 8)
    with pytest.raises(ValueError, match="not 1 or 2"):
        _detect_im7_frames_per_camera(tmp_path / "b.im7", 2)


def test_im7_fpc_accepts_the_two_real_layouts(tmp_path, monkeypatch):
    _fake_frame_count(monkeypatch, 8)
    assert _detect_im7_frames_per_camera(tmp_path / "b.im7", 4) == 2
    assert _detect_im7_frames_per_camera(tmp_path / "b.im7", 8) == 1
    _fake_frame_count(monkeypatch, 3)
    assert _detect_im7_frames_per_camera(tmp_path / "b.im7", 3) == 1
    with pytest.raises(ValueError, match=r"3 cameras \(single-frame\)"):
        _detect_im7_frames_per_camera(tmp_path / "b.im7", 1)


# ---------------------------------------------------------------------------
# Config.image_shape_for: metadata-based .set detection, no config override
# ---------------------------------------------------------------------------


def test_set_shape_detection_uses_the_cameras_own_stream(tmp_path, _clean_set_cache):
    """Time-resolved, mixed-resolution container: camera 2 must report camera
    2's shape. The old pair-reader route read streams 0+1 regardless."""
    set_file = build_raw16_set(tmp_path, [(4, 6), (8, 10), (12, 14)])
    cfg = make_set_config(tmp_path, set_file, camera_count=3, camera_numbers=[2])
    assert cfg.image_shape_for(2) == (8, 10)


def test_set_shape_detection_survives_single_stream_time_resolved(
    tmp_path, _clean_set_cache
):
    """One-stream TR container: the old route demanded streams [0, 1] and
    crashed. Metadata has the answer without touching pixels."""
    set_file = build_raw16_set(tmp_path, [(6, 9)], name="single")
    cfg = make_set_config(tmp_path, set_file, camera_count=1, camera_numbers=[1])
    assert cfg.image_shape_for(1) == (6, 9)


def test_set_shape_detection_pre_paired_uses_stride_two(tmp_path, _clean_set_cache):
    set_file = build_raw16_set(
        tmp_path, [(4, 6), (4, 6), (8, 10), (8, 10)], name="paired"
    )
    cfg = make_set_config(
        tmp_path, set_file, camera_count=2, camera_numbers=[2], frame_stride=0
    )
    assert cfg.image_shape_for(2) == (8, 10)


def _make_im7_config(tmp_path, camera_count, camera_numbers, subfolders=False):
    cfg = {
        "paths": {
            "source_paths": [str(tmp_path)],
            "camera_count": camera_count,
            "camera_numbers": list(camera_numbers),
            "camera_subfolders": (
                [f"Cam{n}" for n in range(1, camera_count + 1)] if subfolders else []
            ),
        },
        "images": {
            "image_format": ["B%05d.im7"],
            "image_type": "lavision_im7",
            "frame_stride": 1,
            "start_index": 1,
            "use_camera_subfolders": subfolders,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return Config(path=str(config_path))


def test_im7_shape_detection_time_resolved_high_camera(tmp_path):
    """5 cameras x 1 frame, first processed camera is 4. The pair-reader route
    assumed 2 frames per camera and asked for frame 6 of 5 -- a crash on a
    correctly configured recording. The header has the shape; no decode."""
    from test_calibration_cli_loader import _write_pack0_im7

    _write_pack0_im7(tmp_path / "B00001.im7", np.zeros((5, 7, 9), np.uint16))
    cfg = _make_im7_config(tmp_path, camera_count=5, camera_numbers=[4, 5])
    assert cfg.image_shape_for(4) == (7, 9)


def test_im7_shape_detection_wrong_camera_count_names_it(tmp_path):
    from test_calibration_cli_loader import _write_pack0_im7

    _write_pack0_im7(tmp_path / "B00001.im7", np.zeros((8, 7, 9), np.uint16))
    cfg = _make_im7_config(tmp_path, camera_count=2, camera_numbers=[1])
    with pytest.raises(ValueError, match="not 1 or 2 frames per camera"):
        cfg.image_shape_for(1)


def test_im7_shape_detection_single_camera_subfolder(tmp_path):
    from test_calibration_cli_loader import _write_pack0_im7

    # camera_count 2: at 1 the source root is always used (single-camera rule).
    (tmp_path / "Cam2").mkdir()
    _write_pack0_im7(tmp_path / "Cam2" / "B00001.im7", np.zeros((2, 7, 9), np.uint16))
    cfg = _make_im7_config(
        tmp_path, camera_count=2, camera_numbers=[2], subfolders=True
    )
    assert cfg.image_shape_for(2) == (7, 9)


def test_images_shape_key_is_ignored(tmp_path, _clean_set_cache):
    """The dead override is gone: a stored shape never bypasses detection."""
    set_file = build_raw16_set(tmp_path, [(6, 9)], name="ignored")
    cfg = make_set_config(tmp_path, set_file, camera_count=1, camera_numbers=[1])
    cfg.data["images"]["shape"] = [999, 999]
    cfg.data["images"]["image_shape"] = [999, 999]
    assert cfg.image_shape_for(1) == (6, 9)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
