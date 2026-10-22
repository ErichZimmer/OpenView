"""Tests for the source-diagnosis scan, types and path normalisation."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fixtures.diagnosis import touch, write_config  # noqa: E402

import pivtools_core.image_handling.source_diagnosis as SD  # noqa: E402
from pivtools_core.image_handling.path_utils import (  # noqa: E402
    infer_image_type,
    normalise_source_path,
)
from pivtools_core.image_handling.source_diagnosis import (  # noqa: E402
    camera_number_of,
    make_suggestion,
    scan_source,
)

# --- scan ---


@pytest.mark.parametrize(
    "raw",
    ['  "/data/run.set"  ', "'/data/run.set'", ' " /data/run.set " ', "/data/run.set"],
)
def test_normalise_source_path(raw):
    assert normalise_source_path(raw) == "/data/run.set"


@pytest.mark.parametrize(
    "name, number",
    [
        ("cam_01", 1),
        ("Camera 1", 1),
        ("camera1", 1),
        ("C1", 1),
        ("view-3", 3),
        ("12", 12),
        ("run2", None),
        ("camera", None),
    ],
)
def test_camera_number_of(name, number):
    assert camera_number_of(name) == number


def test_scan_lists_families_one_level_and_skips_hidden(tmp_path):
    touch(
        tmp_path,
        [
            "B00002.TIF",
            "B00001.tif",
            "run.set",
            "._B00003.tif",
            ".hidden.tif",
            "notes.txt",
        ],
    )
    touch(tmp_path / "camera1", ["B00001.im7", "B00002.im7"])
    touch(tmp_path / "camera1" / "nested", ["deep.tif"])
    touch(tmp_path / ".git", ["x.tif"])
    scan = scan_source(f"  '{tmp_path}/'  ")
    assert scan.exists and scan.is_dir and scan.listed == tmp_path
    assert scan.root_names == ("B00001.tif", "B00002.TIF", "run.set")
    assert dict(scan.root_ext_counts) == {".tif": 2, ".set": 1}
    assert [s.name for s in scan.subdirs] == ["camera1"]
    assert scan.subdirs[0].camera_number == 1 and dict(scan.subdirs[0].ext_counts) == {
        ".im7": 2
    }


def test_scan_marks_davis_companion(tmp_path):
    touch(tmp_path / "run", ["Frame0-0.ims", "Frame0-1.ims"])
    touch(tmp_path / "other", ["StreamSet.xml"])
    (tmp_path / "run.set").write_bytes(b"stub")
    assert all(s.is_davis_companion for s in scan_source(tmp_path).subdirs)
    inner = scan_source(tmp_path / "run")
    assert inner.listed_is_companion and inner.companion_of == tmp_path / "run.set"
    picked = scan_source(tmp_path / "run" / "Frame0-0.ims")
    assert (
        not picked.is_dir
        and picked.listed == tmp_path / "run"
        and picked.companion_of == tmp_path / "run.set"
    )


def test_scan_missing_records_ancestor_and_tail(tmp_path):
    scan = scan_source(tmp_path / "a" / "b")
    assert (
        not scan.exists
        and scan.ancestor == tmp_path
        and scan.missing_tail == ("a", "b")
        and scan.listed == tmp_path
    )


def test_scan_caps_entries_and_subdirs(tmp_path, monkeypatch):  # Review Focus 4
    monkeypatch.setattr(SD, "SCAN_MAX_ENTRIES", 10)
    monkeypatch.setattr(SD, "SCAN_MAX_SUBDIRS", 3)
    touch(tmp_path, [f"B{i:05d}.tif" for i in range(25)])
    for i in range(5):
        (tmp_path / f"cam{i}").mkdir()
    scan = scan_source(tmp_path)
    assert scan.truncated and len(scan.root_names) == 10 and len(scan.subdirs) == 3


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_scan_unlistable_dir_records_error(tmp_path):  # Review Focus 3
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        scan = scan_source(locked)
        assert scan.errors and "could not list" in scan.errors[0]
    finally:
        locked.chmod(0o755)


def test_ims_infers_lavision_set(tmp_path):
    assert infer_image_type("frame.ims") == "lavision_set"
    # Config._detect_image_type: config.yaml with images.image_format ["run.ims"] and no image_type
    assert (
        write_config(tmp_path, images={"image_format": ["run.ims"]}).image_type
        == "lavision_set"
    )


def test_config_readers_normalise_quoted_paths(tmp_path):
    cfg = write_config(
        tmp_path,
        paths={"source_paths": ['  "/x/run.set" ']},
        calibration={"calibration_sources": [" '/x/cal' "]},
    )
    assert cfg.source_paths == [Path("/x/run.set")] and cfg.calibration_sources == [
        Path("/x/cal")
    ]


def test_make_suggestion_guards_the_vocabulary(tmp_path):  # Review Focus 5
    ok = make_suggestion("set_in_folder", "m", "l", {"source_path": "/x/run.set"})
    assert ok == {
        "id": "set_in_folder",
        "message": "m",
        "label": "l",
        "apply": {"source_path": "/x/run.set"},
    }
    for bad in (
        {"foo": 1},
        {"camera_count": 3},
        {"source_path": tmp_path},
        {"image_format": "B%05d.tif"},
    ):
        with pytest.raises(ValueError):
            make_suggestion("x", "m", "l", bad)
