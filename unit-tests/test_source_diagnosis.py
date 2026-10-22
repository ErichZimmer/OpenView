"""Tests for the source-diagnosis scan, types and path normalisation."""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fixtures.diagnosis import _inp, make_ctx, touch, write_config  # noqa: E402
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
    camera_numbers_for,
    cine_pattern,
    found_for,
    make_suggestion,
    pass_sentence,
    rule_camera_folders,
    rule_inside_camera_folder,
    rule_path_case,
    rule_picked_file,
    rule_set_companion,
    rule_set_in_folder,
    rule_set_streams,
    rule_wrong_type,
    scan_sentence,
    scan_source,
    set_stream_infos,
    set_streams_message,
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


# --- set rules ---


def test_set_in_folder_one_set(tmp_path):
    touch(tmp_path / "run", ["Frame0-0.ims"])
    (tmp_path / "run.set").write_bytes(b"stub")
    [s] = rule_set_in_folder(
        _inp(tmp_path, image_type="lavision_set", image_format=("data.set",))
    )
    assert (s["id"], s["apply"], s["label"]) == (
        "set_in_folder",
        {"source_path": str(tmp_path / "run.set")},
        'Use "run.set"',
    )


def test_set_in_folder_several_ranked_by_companion_and_capped(tmp_path):
    for n in ("a", "b", "c", "d", "e", "f"):
        (tmp_path / f"{n}.set").write_bytes(b"stub")
    touch(tmp_path / "e", ["Frame0-0.ims"])
    out = rule_set_in_folder(_inp(tmp_path, image_type="lavision_set"))
    assert [s["label"] for s in out] == [
        'Use "e.set"',
        'Use "a.set"',
        'Use "b.set"',
        'Use "c.set"',
        'Use "d.set"',
    ]


def test_set_in_folder_missing_set_names_sibling(tmp_path):
    (tmp_path / "recording.set").write_bytes(b"stub")
    [s] = rule_set_in_folder(_inp(tmp_path / "missing.set", image_type="lavision_set"))
    assert s["apply"] == {"source_path": str(tmp_path / "recording.set")}


def test_set_in_folder_silent_for_other_types(tmp_path):
    (tmp_path / "run.set").write_bytes(b"stub")
    assert rule_set_in_folder(_inp(tmp_path, image_type="standard")) == []


def test_set_companion_dir_and_picked_ims(tmp_path):
    touch(tmp_path / "run", ["Frame0-0.ims", "Frame0-1.ims"])
    (tmp_path / "run.set").write_bytes(b"stub")
    [d] = rule_set_companion(_inp(tmp_path / "run", image_type="lavision_set"))
    assert d["apply"] == {"source_path": str(tmp_path / "run.set")}
    [f] = rule_set_companion(
        _inp(tmp_path / "run" / "Frame0-0.ims", image_type="lavision_im7")
    )
    assert f["apply"] == {
        "source_path": str(tmp_path / "run.set"),
        "image_type": "lavision_set",
    }


def test_set_companion_without_sibling_set_is_silent(tmp_path):
    touch(tmp_path / "run", ["Frame0-0.ims"])
    assert rule_set_companion(_inp(tmp_path / "run", image_type="lavision_set")) == []


@pytest.mark.parametrize("count", [1, 2])
def test_set_streams_plif_wrong_count(tmp_path, clean_set_cache, count):
    s = build_raw16_set(tmp_path, PLIF)
    out = rule_set_streams(
        _inp(
            s,
            image_type="lavision_set",
            camera_count=count,
            camera_numbers=tuple(range(1, count + 1)),
            pair_reader=True,
        )
    )
    assert [(x["apply"], x["label"]) for x in out] == [
        (
            {"camera_count": 3, "camera_numbers": [2, 3]},
            "Use 3 cameras, process 2 and 3",
        )
    ]


def test_set_streams_plif_right_count_deselects_plif(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    [x] = rule_set_streams(
        _inp(
            s,
            image_type="lavision_set",
            camera_count=3,
            camera_numbers=(1, 2, 3),
            pair_reader=True,
        )
    )
    assert (x["apply"], x["label"]) == (
        {"camera_numbers": [2, 3]},
        "Process cameras 2 and 3",
    )
    assert (
        rule_set_streams(
            _inp(
                s,
                image_type="lavision_set",
                camera_count=3,
                camera_numbers=(2, 3),
                pair_reader=True,
            )
        )
        == []
    )


def test_set_streams_six_equal_count_one(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)] * 6)
    [x] = rule_set_streams(
        _inp(s, image_type="lavision_set", camera_count=1, pair_reader=True)
    )
    assert (x["apply"], x["label"]) == (
        {"camera_count": 3, "camera_numbers": [1, 2, 3]},
        "Use 3 cameras",
    )


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_set_streams_five_equal_pair_reader_never_empty_selection(
    tmp_path, clean_set_cache, count
):
    s = build_raw16_set(tmp_path, [(4, 6)] * 5)
    assert (
        rule_set_streams(
            _inp(
                s,
                image_type="lavision_set",
                camera_count=count,
                camera_numbers=tuple(range(1, count + 1)),
                pair_reader=True,
            )
        )
        == []
    )


def test_set_streams_equal_resolution_blind_spot_offers_nothing(
    tmp_path, clean_set_cache
):
    # 2 cameras: 1+2 or 2+1, ambiguous; 3 singles excluded
    s = build_raw16_set(tmp_path, [(4, 6)] * 3)
    assert (
        rule_set_streams(
            _inp(
                s,
                image_type="lavision_set",
                camera_count=2,
                camera_numbers=(1, 2),
                pair_reader=True,
            )
        )
        == []
    )


def test_set_streams_time_resolved_ranks_pairs_first(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    out = rule_set_streams(
        _inp(
            s,
            image_type="lavision_set",
            camera_count=2,
            camera_numbers=(1, 2),
            pair_reader=False,
        )
    )
    assert [x["apply"] for x in out] == [
        {"camera_count": 3, "camera_numbers": [1, 2, 3]},
        {"camera_count": 5, "camera_numbers": [1, 2, 3, 4, 5]},
    ]


def test_set_streams_calibration_label(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, PLIF)
    out = rule_set_streams(
        _inp(s, image_type="lavision_set", camera_count=1, calibration=True)
    )
    assert out[0]["label"] == "Use 3 cameras (also changes Setup and PIV)"


def test_set_streams_unreadable_set_is_silent(tmp_path, clean_set_cache):
    (tmp_path / "bad.set").write_bytes(b"stub")
    assert (
        rule_set_streams(
            _inp(tmp_path / "bad.set", image_type="lavision_set", pair_reader=True)
        )
        == []
    )


def test_set_streams_message_plif_verbatim():
    streams = [
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
            camera=2 + (i - 1) // 2,
        )
        for i in range(1, 5)
    ]
    assert set_streams_message(streams, [(0,), (1, 2), (3, 4)], [2, 3], True) == (
        "5 streams = 3 cameras. Camera 1 = stream 0 (2160\u00d72560, single frame, "
        "likely PLIF); cameras 2\u20133 = A/B pairs (1\u20132, 3\u20134). "
        "Set 3 cameras and process cameras 2 and 3."
    )
    # Time-resolved: no PLIF guess, and nothing to deselect.
    assert set_streams_message(streams, [(0,), (1, 2), (3, 4)], [1, 2, 3], False) == (
        "5 streams = 3 cameras. Camera 1 = stream 0 (2160\u00d72560); "
        "cameras 2\u20133 = A/B pairs (1\u20132, 3\u20134). Set 3 cameras."
    )


# --- folder rules ---


def test_camera_folders_im7_flag_off_count_one(tmp_path):
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.im7", "B00002.im7"])
    out = rule_camera_folders(
        _inp(tmp_path, image_type="lavision_im7", image_format=("B%05d.im7",))
    )
    assert [(s["apply"], s["label"]) for s in out] == [
        (
            {
                "use_camera_subfolders": True,
                "camera_subfolders": ["camera1", "camera2", "camera3"],
                "camera_count": 3,
                "camera_numbers": [1, 2, 3],
            },
            "Set camera count to 3",
        ),
        ({"source_path": str(tmp_path / "camera1")}, "Use camera1/ only"),
    ]
    cal = rule_camera_folders(
        _inp(
            tmp_path,
            image_type="lavision_im7",
            image_format=("B%05d.im7",),
            calibration=True,
        )
    )
    assert cal[0]["label"] == "Set camera count to 3 (also changes Setup)"
    assert cal[1]["label"] == "Use camera1/ only"


def test_camera_folders_im7_flag_off_count_three(tmp_path):
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.im7"])
    [s] = rule_camera_folders(
        _inp(
            tmp_path,
            image_type="lavision_im7",
            image_format=("B%05d.im7",),
            camera_count=3,
            camera_numbers=(1, 2, 3),
        )
    )
    assert s["apply"] == {
        "use_camera_subfolders": True,
        "camera_subfolders": ["camera1", "camera2", "camera3"],
    }
    assert s["label"] == "Use folders camera1, camera2, camera3"


def test_camera_folders_cam_01_naming(tmp_path):
    for i in (1, 2):
        touch(tmp_path / f"cam_0{i}", ["B00001.im7"])
    [s] = rule_camera_folders(
        _inp(
            tmp_path,
            image_type="lavision_im7",
            image_format=("B%05d.im7",),
            use_camera_subfolders=True,
            camera_count=2,
            camera_numbers=(1, 2),
        )
    )
    assert s["apply"] == {"camera_subfolders": ["cam_01", "cam_02"]}


def test_camera_folders_two_folders_count_three(tmp_path):
    for i in (1, 2):
        touch(tmp_path / f"camera{i}", ["B00001.im7"])
    [s] = rule_camera_folders(
        _inp(
            tmp_path,
            image_type="lavision_im7",
            image_format=("B%05d.im7",),
            use_camera_subfolders=True,
            camera_count=3,
            camera_numbers=(1, 2, 3),
        )
    )
    assert s["apply"] == {
        "camera_subfolders": ["camera1", "camera2"],
        "camera_count": 2,
        "camera_numbers": [1, 2],
    }


def test_camera_folders_standard_spaced_names(tmp_path):
    for i in (1, 2):
        touch(tmp_path / f"Camera {i}", ["B00001.tif"])
    [s] = rule_camera_folders(_inp(tmp_path, camera_count=2, camera_numbers=(1, 2)))
    assert s["apply"] == {"camera_subfolders": ["Camera 1", "Camera 2"]}


def test_camera_folders_single_folder_becomes_source_path(tmp_path):  # B9
    touch(tmp_path / "camera1", ["B00001.tif"])
    [s] = rule_camera_folders(_inp(tmp_path, camera_count=2, camera_numbers=(1, 2)))
    assert s["apply"] == {
        "source_path": str(tmp_path / "camera1"),
        "camera_count": 1,
        "camera_numbers": [1],
    }
    assert s["label"] == "Use camera1/ as the source"


def test_camera_folders_silent_when_root_matches_or_already_configured(tmp_path):
    for i in (1, 2):
        touch(tmp_path / f"Cam{i}", ["B00001.tif"])
    # Subfolders already resolve to Cam1, Cam2 at the configured count: nothing to change.
    assert (
        rule_camera_folders(_inp(tmp_path, camera_count=2, camera_numbers=(1, 2))) == []
    )
    touch(tmp_path, ["B00001.tif"])
    assert rule_camera_folders(_inp(tmp_path)) == []


def test_no_rule_writes_subfolders_into_a_single_camera_setup(tmp_path):  # B9 invariant
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.tif"])
    single = tmp_path / "single"
    touch(single / "camera1", ["B00001.tif"])  # reaches k == 1
    for inp in (
        _inp(tmp_path),
        _inp(tmp_path / "camera1", camera_count=3, camera_numbers=(1, 2, 3)),
        _inp(single, camera_count=2, camera_numbers=(1, 2)),
    ):
        for rule in (rule_camera_folders, rule_wrong_type, rule_inside_camera_folder):
            for s in rule(inp):
                count_after = s["apply"].get("camera_count", inp.ctx.camera_count)
                assert not ("camera_subfolders" in s["apply"] and count_after == 1)


@pytest.mark.parametrize(
    "current, count, k, expected",
    [
        ((1,), 1, 3, [1, 2, 3]),
        ((2, 3), 3, 2, [2]),
        ((2, 3), 3, 4, [2, 3]),
        ((3,), 3, 1, [1]),
    ],
)
def test_camera_numbers_for(tmp_path, current, count, k, expected):
    ctx = make_ctx(tmp_path, camera_count=count, camera_numbers=current)
    assert camera_numbers_for(ctx, k) == expected


def test_inside_camera_folder(tmp_path):
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.tif"])
    [s] = rule_inside_camera_folder(
        _inp(tmp_path / "camera1", camera_count=3, camera_numbers=(1, 2, 3))
    )
    assert s["apply"] == {
        "source_path": str(tmp_path),
        "camera_subfolders": ["camera1", "camera2", "camera3"],
    }
    assert s["label"] == f"Use {tmp_path.name}/ with its camera folders"


def test_inside_camera_folder_needs_siblings(tmp_path):
    touch(tmp_path / "camera1", ["B00001.tif"])
    assert rule_inside_camera_folder(_inp(tmp_path / "camera1")) == []


def test_wrong_type_tif_present_im7_chosen(tmp_path):
    touch(tmp_path, ["B00001.tif", "B00002.tif", "B00003.tif"])
    [s] = rule_wrong_type(
        _inp(tmp_path, image_type="lavision_im7", image_format=("B%05d.im7",))
    )
    assert (s["apply"], s["label"]) == (
        {"image_type": "standard", "image_format": ["B%05d.tif"]},
        "Switch to Standard (TIFF/PNG/JPG)",
    )


def test_wrong_type_set_present_standard_chosen(tmp_path):
    (tmp_path / "run.set").write_bytes(b"stub")
    touch(tmp_path / "run", ["Frame0-0.ims"])
    [s] = rule_wrong_type(_inp(tmp_path))
    assert s["apply"] == {
        "image_type": "lavision_set",
        "source_path": str(tmp_path / "run.set"),
    }
    assert s["label"] == "Switch to LaVision recording (.set + .ims folder)"


def test_wrong_type_tie_and_same_family_are_silent(tmp_path):
    touch(tmp_path, ["B00001.tif", "x00001.im7"])
    assert rule_wrong_type(_inp(tmp_path, image_type="cine")) == []
    other = tmp_path / "other"
    touch(other, ["B00001.tif"])
    assert rule_wrong_type(_inp(other)) == []


def test_wrong_type_family_only_in_camera_folders(tmp_path):
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.im7", "B00002.im7"])
    out = rule_wrong_type(_inp(tmp_path))
    assert [(s["apply"], s["label"]) for s in out] == [
        (
            {
                "image_type": "lavision_im7",
                "image_format": ["B%05d.im7"],
                "use_camera_subfolders": True,
                "camera_subfolders": ["camera1", "camera2", "camera3"],
                "camera_count": 3,
                "camera_numbers": [1, 2, 3],
            },
            "Switch to LaVision images (.im7)",
        ),
        (
            {
                "image_type": "lavision_im7",
                "image_format": ["B%05d.im7"],
                "source_path": str(tmp_path / "camera1"),
            },
            "Switch to LaVision images (.im7), use camera1/ only",
        ),
    ]


def test_picked_file(tmp_path):
    touch(tmp_path, ["B00001.tif", "B00002.tif"])
    [s] = rule_picked_file(_inp(tmp_path / "B00001.tif"))
    assert s["apply"] == {"source_path": str(tmp_path), "image_format": ["B%05d.tif"]}
    touch(tmp_path, ["x00001.im7", "x00002.im7"])
    [t] = rule_picked_file(_inp(tmp_path / "x00001.im7"))
    assert t["apply"] == {
        "source_path": str(tmp_path),
        "image_type": "lavision_im7",
        "image_format": ["x%05d.im7"],
    }


def test_picked_file_set_and_ims(tmp_path):
    (tmp_path / "run.set").write_bytes(b"stub")
    [s] = rule_picked_file(_inp(tmp_path / "run.set"))
    assert s["apply"] == {"image_type": "lavision_set"}
    assert rule_picked_file(_inp(tmp_path / "run.set", image_type="lavision_set")) == []
    touch(tmp_path / "run", ["Frame0-0.ims"])
    assert rule_picked_file(_inp(tmp_path / "run" / "Frame0-0.ims")) == []


def test_path_case_whitespace(tmp_path):
    touch(tmp_path / "Run 1", ["B00001.tif"])
    [s] = rule_path_case(_inp(tmp_path / "run1"))
    assert (s["apply"], s["label"]) == (
        {"source_path": str(tmp_path / "Run 1")},
        'Use "Run 1"',
    )


def test_path_case_letter_case(tmp_path):
    touch(tmp_path / "Run1", ["B00001.tif"])
    if (tmp_path / "RUN1").exists():
        pytest.skip("case-insensitive filesystem")
    [s] = rule_path_case(_inp(tmp_path / "run1" / "sub"))
    assert s["apply"] == {"source_path": str(tmp_path / "Run1" / "sub")}


def test_path_case_silent_when_present_or_unmatched(tmp_path):
    touch(tmp_path / "Run1", ["B00001.tif"])
    assert rule_path_case(_inp(tmp_path / "Run1")) == []
    assert rule_path_case(_inp(tmp_path / "nothing")) == []


@pytest.mark.parametrize(
    "names, count, numbers, expected",
    [
        (["Camera1.cine", "Camera2.cine"], 2, [1, 2], "Camera%d.cine"),
        (["run.cine"], 1, [1], "run.cine"),
        (["run.cine"], 2, [1, 2], None),
        (["a.cine", "b.cine"], 1, [1], None),
    ],
)
def test_cine_pattern_b8(tmp_path, names, count, numbers, expected):
    touch(tmp_path, names)
    assert cine_pattern(names, tmp_path, count, numbers) == expected


def test_cine_pattern_requires_every_selected_camera_file(tmp_path):
    touch(tmp_path, ["Camera1.cine", "Camera2.cine"])
    names = ["Camera1.cine", "Camera2.cine"]
    assert cine_pattern(names, tmp_path, 3, [1, 2, 3]) is None
