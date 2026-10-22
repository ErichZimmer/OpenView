"""Tests for the source-diagnosis scan, types and path normalisation."""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fixtures.diagnosis import make_ctx, touch, write_config  # noqa: E402
from fixtures.lavision import (  # noqa: E402,F401
    PLIF,
    build_raw16_set,
    clean_set_cache,
)

import pivtools_core.image_handling.source_diagnosis as SD  # noqa: E402
from pivtools_core.image_handling.path_utils import (  # noqa: E402
    _append_folder_contents,
    infer_image_type,
    normalise_source_path,
)
from pivtools_core.image_handling.source_diagnosis import (  # noqa: E402
    StreamInfo,
    SubdirInfo,
    camera_number_of,
    found_for,
    make_suggestion,
    pass_sentence,
    scan_sentence,
    scan_source,
    set_stream_infos,
    streams_sentence,
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


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_scan_path_beneath_unsearchable_dir_records_error(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        scan = scan_source(locked / "sub" / "x")
        assert not scan.exists and scan.ancestor == locked
        assert any(
            e.startswith("could not stat ") and "locked" in e for e in scan.errors
        )
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


# --- found ---


def _scan(**fields):
    """A hand-built ``SourceScan`` defaulting to an existing, empty directory."""
    base = dict(
        raw="",
        path=Path("D:/x"),
        exists=True,
        is_dir=True,
        ancestor=None,
        missing_tail=(),
        listed=Path("D:/x"),
        root_names=(),
        root_ext_counts={},
        subdirs=(),
        listed_is_companion=False,
        companion_of=None,
        truncated=False,
        errors=(),
    )
    base.update(fields)
    return SD.SourceScan(**base)


def test_scan_sentence_camera_folders_verbatim():
    subs = tuple(
        SubdirInfo(f"camera{i}", i, {".im7": 12}, False, False) for i in (1, 2, 3)
    )
    assert scan_sentence(
        _scan(path=Path("D:/calib"), listed=Path("D:/calib"), subdirs=subs)
    ) == (
        "In D:/calib: no images here; 3 folders (camera1, camera2, camera3), each with 12 .im7 files."
    )


def test_scan_sentence_companion_verbatim():
    names = tuple(f"Frame{i}-{j}.ims" for i in range(5) for j in (0, 1))
    scan = _scan(
        path=Path("D:/run"),
        listed=Path("D:/run"),
        root_names=names,
        root_ext_counts={".ims": 10},
        listed_is_companion=True,
        companion_of=Path("D:/run.set"),
    )
    assert scan_sentence(scan) == "D:/run is the data folder of D:/run.set (5 streams)."


def test_streams_sentence_plif_verbatim():
    s = [
        StreamInfo(
            index=0,
            prefix="Frame0",
            width=2160,
            height=2560,
            decoder="mono-12p",
            camera=1,
        )
    ] + [
        StreamInfo(
            index=i,
            prefix=f"Frame{i}",
            width=2048,
            height=2048,
            decoder="mono-12p",
            camera=None,
        )
        for i in range(1, 5)
    ]
    assert (
        streams_sentence("run.set", s)
        == "run.set: 5 streams \u2014 0: 2160\u00d72560 mono-12p, 1\u20134: 2048\u00d72048 mono-12p."
    )


def test_pass_sentence_verbatim():
    assert (
        pass_sentence(3, 500, ".im7", (1280, 1024))
        == "3 cameras \u00d7 500 .im7, 1280\u00d71024."
    )
    assert pass_sentence(1, 500, ".tif", (1280, 1024)) == "500 .tif, 1280\u00d71024."


def test_scan_sentence_other_forms():
    assert (
        scan_sentence(
            _scan(path=Path("D:/x"), listed=Path("D:/x"), root_ext_counts={".tif": 500})
        )
        == "In D:/x: 500 .tif files."
    )
    assert scan_sentence(
        _scan(
            path=Path("D:/x"),
            listed=Path("D:/x"),
            root_ext_counts={".set": 1},
            subdirs=(SubdirInfo("run", None, {".ims": 1}, True, False),),
        )
    ) == (
        "In D:/x: 1 .set file; 1 folder (run) with 1 .ims file."
    )  # singular for 1
    assert (
        scan_sentence(_scan(path=Path("D:/x"), listed=Path("D:/x")))
        == "In D:/x: no images here and no folders."
    )
    assert scan_sentence(
        _scan(
            path=Path("D:/x/run"),
            exists=False,
            is_dir=False,
            ancestor=Path("D:/x"),
            missing_tail=("run",),
            listed=Path("D:/x"),
            root_ext_counts={".set": 2},
        )
    ) == ("D:/x/run does not exist. In D:/x: 2 .set files.")


def test_scan_sentence_marks_truncated_counts():  # Review Focus 4
    assert (
        scan_sentence(
            _scan(
                path=Path("D:/x"),
                listed=Path("D:/x"),
                root_ext_counts={".tif": 2000},
                truncated=True,
            )
        )
        == "In D:/x: 2000+ .tif files."
    )


def test_scan_sentence_reports_listing_errors():  # Review Focus 3
    text = scan_sentence(
        _scan(
            path=Path("D:/x"),
            listed=Path("D:/x"),
            errors=("could not list D:/x: Permission denied",),
        )
    )
    assert text.endswith(" Could not list D:/x: Permission denied.")


def test_scan_sentence_folder_clause_variants():
    def subs(*specs):
        return tuple(SubdirInfo(n, None, c, False, False) for n, c in specs)

    mixed = scan_sentence(
        _scan(subdirs=subs(("a", {".im7": 3}), ("b", {".tif": 2}), ("c", {})))
    )
    assert (
        mixed
        == "In D:/x: no images here; 3 folders (a, b, c): a: 3 .im7, b: 2 .tif, c: no images."
    )
    none = scan_sentence(_scan(subdirs=subs(("a", {}), ("b", {}))))
    assert none == "In D:/x: no images here; 2 folders (a, b), none with images."
    many = scan_sentence(
        _scan(subdirs=subs(*[(f"c{i}", {".im7": 1}) for i in range(1, 8)]))
    )
    assert (
        many
        == "In D:/x: no images here; 7 folders (c1, c2, c3, c4, c5 and 2 more), each with 1 .im7 file."
    )


def test_found_never_says_empty_when_folders_exist(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"camera{i}").mkdir()
    # tmp_path's own name derives from the test name and contains "empty"; strip it before checking.
    assert (
        "empty"
        not in scan_sentence(scan_source(tmp_path))
        .replace(tmp_path.as_posix(), "")
        .lower()
    )
    assert _append_folder_contents(
        "First frame not found. Looking for: calib00001.tif", tmp_path
    ) == (
        "First frame not found. Looking for: calib00001.tif. No files here, only folders: camera1, camera2, camera3"
    )
    empty = tmp_path / "camera1"
    assert _append_folder_contents("x", empty) == f"x. Folder is empty: {empty}"


def test_append_folder_contents_ignores_dot_names(tmp_path):
    (tmp_path / ".DS_Store").touch()
    (tmp_path / ".hidden").mkdir()
    assert _append_folder_contents("x", tmp_path) == f"x. Folder is empty: {tmp_path}"


def test_set_stream_infos_plif(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)  # PLIF = [(6, 8)] + [(4, 6)] * 4, (H, W)
    infos = set_stream_infos(s, 3)
    assert [i["camera"] for i in infos] == [1, 2, 2, 3, 3]
    assert (infos[0]["width"], infos[0]["height"], infos[0]["prefix"]) == (
        8,
        6,
        "Frame0",
    )
    assert [i["camera"] for i in set_stream_infos(s, 2)] == [None] * 5


def test_found_for_set_without_companion(tmp_path):
    stub = tmp_path / "run.set"
    stub.write_bytes(b"stub")
    ctx = make_ctx(stub, image_type="lavision_set")
    text = found_for(scan_source(stub), ctx, {"valid": False}, None, None)
    assert (
        text
        == f"{stub.as_posix()} has no data folder {(tmp_path / 'run').as_posix()} beside it."
    )


def test_found_for_set_streams_and_unreadable(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    ctx = make_ctx(s, image_type="lavision_set", camera_count=3)
    infos = set_stream_infos(s, 3)
    assert found_for(scan_source(s), ctx, {"valid": True}, infos, None).startswith(
        "shaped.set: 5 streams"
    )
    assert found_for(
        scan_source(s), ctx, {"valid": False}, None, "index unreadable"
    ) == ("shaped.set: streams could not be read (index unreadable).")


def test_found_for_valid_and_invalid_results(tmp_path):
    ctx = make_ctx(tmp_path, image_type="lavision_im7", camera_count=3)
    ok = {
        "valid": True,
        "found_count": 500,
        "format_detected": "im7",
        "image_size": (1280, 1024),
    }
    assert (
        found_for(scan_source(tmp_path), ctx, ok, None, None)
        == "3 cameras \u00d7 500 .im7, 1280\u00d71024."
    )
    cine = dict(ok, format_detected="cine")
    assert (
        found_for(scan_source(tmp_path), make_ctx(tmp_path), cine, None, None)
        == "500 frames, 1280\u00d71024."
    )
    bad = found_for(scan_source(tmp_path), ctx, {"valid": False}, None, None)
    assert bad == scan_sentence(scan_source(tmp_path))
