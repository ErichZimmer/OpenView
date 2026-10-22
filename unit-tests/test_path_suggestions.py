"""Tests for rule_flat_with_subfolders, the stale multi-camera layout rule.

When camera subfolder paths (source/Cam1, source/Cam2) don't exist but files
matching the image pattern sit directly in the source folder, the GUI offers to
switch to a single-camera setup (or, for IM7, to turn camera subfolders off).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fixtures.diagnosis import _inp, touch  # noqa: E402

from pivtools_core.image_handling.source_diagnosis import (  # noqa: E402
    rule_flat_with_subfolders,
)

TWO_CAMERAS = {"camera_count": 2, "camera_numbers": (1, 2)}


def _flat(source, patterns, **overrides):
    return rule_flat_with_subfolders(
        _inp(source, image_format=tuple(patterns), **{**TWO_CAMERAS, **overrides})
    )


def test_flat_layout_detected(tmp_path):
    touch(tmp_path, [f"{i:05d}.tif" for i in range(1, 4)])
    out = _flat(tmp_path, ["%05d.tif"])
    assert [(s["apply"], s["label"]) for s in out] == [
        ({"camera_count": 1, "camera_numbers": [1]}, "Use 1 camera")
    ]
    assert "00001.tif, 00002.tif, 00003.tif" in out[0]["message"]


def test_files_only_in_camera_subfolder_not_detected(tmp_path):
    touch(tmp_path / "Cam1", [f"{i:05d}.tif" for i in range(1, 4)])
    assert _flat(tmp_path, ["%05d.tif"]) == []


def test_ab_patterns_detected_when_only_b_matches(tmp_path):
    touch(tmp_path, [f"B{i:05d}_B.tif" for i in range(1, 3)])
    out = _flat(tmp_path, ["B%05d_A.tif", "B%05d_B.tif"])
    assert len(out) == 1
    assert "B00001_B.tif, B00002_B.tif" in out[0]["message"]


def test_capped_at_five_files(tmp_path):
    touch(tmp_path, [f"{i:05d}.tif" for i in range(1, 20)])
    [s] = _flat(tmp_path, ["%05d.tif"])
    assert "00005.tif" in s["message"]
    assert "00006.tif" not in s["message"]


def test_empty_and_specifierless_patterns_do_not_crash(tmp_path):
    touch(tmp_path, ["literal.tif"])
    assert _flat(tmp_path, ["", "   "]) == []
    # A pattern without %d globs as a literal filename
    assert len(_flat(tmp_path, ["literal.tif"])) == 1


def test_missing_source_dir_returns_empty(tmp_path):
    assert _flat(tmp_path / "nope", ["%05d.tif"]) == []


def test_dotfiles_excluded(tmp_path):
    touch(tmp_path, ["._00001.tif"])
    assert _flat(tmp_path, ["%05d.tif"]) == []
    touch(tmp_path, ["00001.tif"])
    [s] = _flat(tmp_path, ["%05d.tif"])
    assert "._00001.tif" not in s["message"]


def test_im7_flag_on_files_at_root(tmp_path):
    touch(tmp_path, ["B00001.im7", "B00002.im7"])
    out = _flat(
        tmp_path,
        ["B%05d.im7"],
        image_type="lavision_im7",
        use_camera_subfolders=True,
    )
    assert [(s["apply"], s["label"]) for s in out] == [
        ({"use_camera_subfolders": False}, "Turn off camera subfolders")
    ]


def test_qualifying_camera_folder_present_is_silent(tmp_path):
    touch(tmp_path, ["B00001.im7"])
    touch(tmp_path / "camera1", ["B00001.im7"])
    out = _flat(
        tmp_path,
        ["B%05d.im7"],
        image_type="lavision_im7",
        use_camera_subfolders=True,
    )
    assert out == []


def test_calibration_label_carries_cross_store_note(tmp_path):
    touch(tmp_path, ["B00001.tif"])
    [s] = _flat(tmp_path, ["B%05d.tif"], calibration=True)
    assert s["label"] == "Use 1 camera (also changes Setup)"
