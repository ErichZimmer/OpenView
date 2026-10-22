#!/usr/bin/env python3
"""
test_pattern_suggestion.py

Filename-pattern suggestion picks the digit run that VARIES across the folder.

Background: the old suggester read only the first sorted file, took its first
digit run and capped padding at %05d. ``Cam1_00001.tif`` became ``Cam%d_00001.tif``
(the camera number, not the frame index) and six-digit names became ``%05d``.
The suggester now takes the whole sample list, picks the run position whose value
changes between files, and keeps the run's true width. When it cannot tell which
number is the frame index it says so instead of guessing.

Usage:
    pytest unit-tests/test_pattern_suggestion.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures.diagnosis import _inp, frame_names, touch  # noqa: E402

from pivtools_core.image_handling.path_utils import (  # noqa: E402
    _detect_ab_pair_pattern,
    _suggest_pattern,
    _suggest_pattern_for_role,
    validate_images_generic,
    validate_single_pattern,
)
from pivtools_core.image_handling.source_diagnosis import (  # noqa: E402
    pattern_reason,
    role_pattern_suggestion,
    rule_pattern,
)


@pytest.mark.parametrize(
    "names, expected",
    [
        (frame_names("Cam1_%05d.tif", 1, 10), "Cam1_%05d.tif"),
        (frame_names("run3_img%04d_cam1.tif", 1, 10), "run3_img%04d_cam1.tif"),
        (frame_names("img%d.tif", 1, 12), "img%d.tif"),
        (frame_names("frame%06d.png", 0, 7), "frame%06d.png"),
        (frame_names("IMG%05dA.tif", 1, 5), "IMG%05dA.tif"),
    ],
)
def test_varying_run_is_the_frame_index(names, expected):
    """The run that changes between files is chosen, at its true width."""
    suggestion = _suggest_pattern(sorted(names))
    assert suggestion.pattern == expected
    assert suggestion.reason is None


def test_ab_pairs_give_both_patterns():
    names = sorted(frame_names("B%05d_A.tif", 1, 6) + frame_names("B%05d_B.tif", 1, 6))
    ab = _detect_ab_pair_pattern(names)
    assert ab is not None
    assert ab["pattern_a"] == "B%05d_A.tif"
    assert ab["pattern_b"] == "B%05d_B.tif"


def test_role_filter_digit_letter_suffix():
    """IMG00001A.tif style (letter directly after the digits) keeps working per role."""
    names = sorted(
        frame_names("IMG%05dA.tif", 1, 5) + frame_names("IMG%05dB.tif", 1, 5)
    )
    assert _suggest_pattern_for_role(names, "A").pattern == "IMG%05dA.tif"
    assert _suggest_pattern_for_role(names, "B").pattern == "IMG%05dB.tif"


def test_single_file_one_run_uses_it():
    assert _suggest_pattern(["Cam_00007.tif"]).pattern == "Cam_%05d.tif"
    assert _suggest_pattern(["img7.tif"]).pattern == "img%d.tif"


def test_single_file_two_runs_is_ambiguous():
    suggestion = _suggest_pattern(["Cam1_00001.tif"])
    assert suggestion.pattern is None
    assert suggestion.reason == (
        "could not tell which number is the frame index in 'Cam1_00001.tif'"
    )


def test_two_runs_varying_together_resolved_by_sort_order():
    """Only the strictly increasing run is the frame index."""
    names = ["a2_0001.tif", "a1_0002.tif", "a3_0003.tif"]
    # Sorted, the first run reads 1,2,3 (strictly increasing) while the 4-digit
    # run reads 2,1,3, so the first run is the frame index.
    assert _suggest_pattern(sorted(names)).pattern == "a%d_0002.tif"


def test_two_runs_both_monotone_is_ambiguous():
    names = ["t1_img0001.tif", "t2_img0002.tif", "t3_img0003.tif"]
    suggestion = _suggest_pattern(names)
    assert suggestion.pattern is None
    assert "t1_img0001.tif" in suggestion.reason


def _fake_frame(_idx):
    return np.zeros((8, 8), dtype=np.uint8)


def test_set_suggestion_is_a_filename(tmp_path):
    """A missing .set suggests the sibling's filename, not its full path."""
    (tmp_path / "recording.set").write_bytes(b"stub")
    result = validate_images_generic(
        camera_path=tmp_path / "missing.set",
        camera=1,
        image_format="missing.set",
        image_type="lavision_set",
        expected_count=1,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )
    assert result["suggested_pattern"] == "recording.set"


def test_ambiguous_message_reaches_validation_error(tmp_path):
    (tmp_path / "Cam1_00001.tif").write_bytes(b"")
    generic = validate_images_generic(
        camera_path=tmp_path,
        camera=1,
        image_format="wrong%05d.tif",
        image_type="standard",
        expected_count=1,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )
    assert generic["suggested_pattern"] is None
    assert generic["suggestion_error"] == (
        "could not tell which number is the frame index in 'Cam1_00001.tif'"
    )
    assert generic["suggestion_error"] in generic["error"]

    single = validate_single_pattern(
        camera_path=tmp_path,
        pattern="wrong%05d.tif",
        image_type="standard",
        expected_count=1,
        zero_based_indexing=False,
    )
    assert single["suggested_pattern"] is None
    assert generic["suggestion_error"] in single["error"]


def test_validation_suggests_from_folder(tmp_path):
    for name in frame_names("Cam1_%05d.tif", 1, 10):
        (tmp_path / name).write_bytes(b"")
    result = validate_images_generic(
        camera_path=tmp_path,
        camera=1,
        image_format="nope%05d.tif",
        image_type="standard",
        expected_count=10,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )
    assert result["suggested_pattern"] == "Cam1_%05d.tif"


def test_pattern_rule_single(tmp_path):
    touch(tmp_path, frame_names("Cam1_%05d.tif", 1, 10))
    [s] = rule_pattern(_inp(tmp_path, image_format=("nope%05d.tif",)))
    assert (s["id"], s["apply"], s["label"]) == (
        "pattern",
        {"image_format": ["Cam1_%05d.tif"]},
        'Use "Cam1_%05d.tif"',
    )


def test_pattern_rule_ab_sets_preset_on_setup_only(tmp_path):
    touch(
        tmp_path,
        frame_names("B%05d_A.tif", 1, 6) + frame_names("B%05d_B.tif", 1, 6),
    )
    [s] = rule_pattern(
        _inp(tmp_path, image_format=("x%05d.tif",), pairing_preset="time_resolved")
    )
    assert s["apply"] == {
        "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
        "pairing_preset": "ab_format",
    }
    assert s["label"] == 'Use "B%05d_A.tif" / "B%05d_B.tif"'
    [t] = rule_pattern(
        _inp(tmp_path, image_format=("x%05d.tif",), pairing_preset="ab_format")
    )
    assert t["apply"] == {"image_format": ["B%05d_A.tif", "B%05d_B.tif"]}
    [c] = rule_pattern(
        _inp(
            tmp_path,
            image_format=("x%05d.tif",),
            calibration=True,
            pairing_preset=None,
        )
    )
    assert c["apply"] == {"image_format": ["B%05d_A.tif"]}


def test_pattern_rule_im7(tmp_path):
    touch(tmp_path, frame_names("B%05d.im7", 1, 3))
    [s] = rule_pattern(
        _inp(tmp_path, image_type="lavision_im7", image_format=("x%05d.im7",))
    )
    assert s["apply"] == {"image_format": ["B%05d.im7"]}


def test_pattern_rule_ambiguous_gives_reason_not_suggestion(tmp_path):
    touch(tmp_path, ["Cam1_00001.tif"])
    assert rule_pattern(_inp(tmp_path, image_format=("wrong%05d.tif",))) == []
    assert pattern_reason(["Cam1_00001.tif"], "standard", False) == (
        "could not tell which number is the frame index in 'Cam1_00001.tif'"
    )


def test_pattern_rule_silent_when_pattern_matches_or_no_family_files(tmp_path):
    touch(tmp_path, frame_names("Cam1_%05d.tif", 1, 3))
    assert rule_pattern(_inp(tmp_path, image_format=("Cam1_%05d.tif",))) == []
    other = tmp_path / "other"
    touch(other, ["notes.txt"])
    assert rule_pattern(_inp(other, image_format=("nope%05d.tif",))) == []
    assert rule_pattern(_inp(tmp_path / "missing", image_format=("a%05d.tif",))) == []


def test_pattern_rule_empty_pattern_and_container_types(tmp_path):
    touch(tmp_path, frame_names("Cam1_%05d.tif", 1, 3))
    [s] = rule_pattern(_inp(tmp_path, image_format=("",)))
    assert s["apply"] == {"image_format": ["Cam1_%05d.tif"]}
    touch(tmp_path / "s", ["run.set"])
    assert rule_pattern(_inp(tmp_path / "s", image_type="lavision_set")) == []


def test_role_pattern_suggestion_digit_letter_suffix():
    names = sorted(
        frame_names("IMG%05dA.tif", 1, 5) + frame_names("IMG%05dB.tif", 1, 5)
    )
    s = role_pattern_suggestion(names, "B", ["IMG%05dA.tif", "wrong%05d.tif"], 1)
    assert s["apply"] == {"image_format": ["IMG%05dA.tif", "IMG%05dB.tif"]}


def test_role_pattern_suggestion_none_when_unchanged_or_ambiguous():
    names = frame_names("IMG%05dA.tif", 1, 5)
    assert role_pattern_suggestion(names, "A", ["IMG%05dA.tif"], 0) is None
    assert role_pattern_suggestion(["Cam1_00001.tif"], None, ["x"], 0) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
