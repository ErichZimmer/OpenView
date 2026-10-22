"""Tests for the source-diagnosis scan, types and path normalisation."""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fixtures.diagnosis import (  # noqa: E402
    _inp,
    calibration_client,
    detected_start_index_of,
    frame_names,
    make_ctx,
    piv_route_response,
    touch,
    write_config,
)
from fixtures.lavision import (  # noqa: E402,F401
    PLIF,
    build_raw16_set,
    clean_set_cache,
)

import pivtools_core.image_handling.source_diagnosis as SD  # noqa: E402
from pivtools_core.calibration_settings import save_settings  # noqa: E402
from pivtools_core.image_handling.path_utils import (  # noqa: E402
    _append_folder_contents,
    infer_image_type,
    normalise_source_path,
    validate_images_generic,
    validate_single_pattern,
)
from pivtools_core.image_handling.source_diagnosis import (  # noqa: E402
    APPLY_FIELDS,
    RULES,
    StreamInfo,
    SubdirInfo,
    apply_diagnosis,
    camera_number_of,
    camera_numbers_for,
    cine_pattern,
    diagnose,
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
    rule_start_index,
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
        subdirs_truncated=False,
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


@pytest.mark.parametrize(
    "folders",
    [
        ["camera1", "camera3"],
        ["camera2", "camera3"],
        ["camera1", "cam_1"],
        ["camera2"],
    ],
)
def test_folder_rules_silent_unless_camera_numbers_are_exactly_1_to_k(
    tmp_path, folders
):
    for name in folders:
        touch(tmp_path / name, ["B00001.tif"])
    for count, numbers in ((1, (1,)), (2, (1, 2))):
        inp = _inp(tmp_path, camera_count=count, camera_numbers=numbers)
        assert rule_camera_folders(inp) == []
    touch(tmp_path / "x", ["B00001.im7"])
    for name in folders:
        touch(tmp_path / "w" / name, ["B00001.im7"])
    assert rule_wrong_type(_inp(tmp_path / "w")) == []
    assert (
        rule_inside_camera_folder(
            _inp(tmp_path / folders[0], camera_count=2, camera_numbers=(1, 2))
        )
        == []
    )


# --- diagnose ---


def test_start_index_rule(tmp_path):
    touch(tmp_path, [f"B{i:05d}.tif" for i in range(0, 10)])
    [s] = rule_start_index(_inp(tmp_path, start_index=1, expected_count=10))
    assert s == {
        "id": "start_index",
        "apply": {"start_index": 0},
        "label": "Switch to 0-based",
        "message": "Files start at 0 but the start index is 1 \u2014 switch to 0-based indexing",
    }


def test_start_index_rule_silent_when_files_present(tmp_path):
    touch(tmp_path, frame_names("B%05d.tif", 1, 10))
    assert rule_start_index(_inp(tmp_path, start_index=1, expected_count=10)) == []


def test_diagnose_ranks_by_rule(tmp_path):
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.im7"])
    touch(tmp_path, ["stray00001.im7"])
    ctx = make_ctx(tmp_path, image_type="lavision_im7", image_format=("B%05d.im7",))
    ids = [
        s["id"]
        for s in diagnose(
            scan_source(tmp_path),
            ctx,
            1,
            tmp_path,
            1,
            valid=False,
            detected_start_index=None,
        )
    ]
    assert ids == ["camera_folders", "camera_folders", "pattern"]


def test_diagnose_dedupes_identical_apply(tmp_path, monkeypatch):
    def same(rule_id):
        return lambda inp: [make_suggestion(rule_id, "m", "l", {"start_index": 0})]

    monkeypatch.setattr(
        SD,
        "RULES",
        (("pattern", same("pattern")), ("start_index", same("start_index"))),
    )
    out = diagnose(
        scan_source(tmp_path),
        make_ctx(tmp_path),
        1,
        tmp_path,
        1,
        False,
        detected_start_index=None,
    )
    assert [s["id"] for s in out] == ["pattern"]


def test_diagnose_on_pass_runs_only_set_streams(tmp_path):
    touch(tmp_path, frame_names("Cam1_%05d.tif", 1, 3))
    ctx = make_ctx(tmp_path, image_format=("nope%05d.tif",))
    out = diagnose(
        scan_source(tmp_path), ctx, 1, tmp_path, 3, True, detected_start_index=None
    )
    assert out == []


def test_diagnose_set_target_with_wrong_type_puts_type_fix_first(
    tmp_path, clean_set_cache
):
    s = build_raw16_set(tmp_path, PLIF)
    ctx = make_ctx(s, image_type="standard", camera_count=1, camera_numbers=(1,))
    out = diagnose(scan_source(s), ctx, 1, s, 1, valid=False, detected_start_index=None)
    assert out[0]["id"] == "picked_file"
    assert out[0]["apply"] == {"image_type": "lavision_set"}
    ids = [x["id"] for x in out]
    assert ids.index("picked_file") < ids.index("set_streams")


def test_rule_pattern_cine_uses_percent_d_when_every_camera_file_exists(tmp_path):
    touch(tmp_path, ["rec_1.cine", "rec_2.cine"])
    [s] = SD.rule_pattern(
        _inp(
            tmp_path,
            image_type="cine",
            image_format=("x.cine",),
            camera_count=2,
            camera_numbers=(1, 2),
        )
    )
    assert s["apply"] == {"image_format": ["rec_%d.cine"]}


def tree_path_case(root):
    touch(root / "Run 1", ["B00001.tif"])
    return root / "run1", make_ctx(root / "run1"), root / "run1", 1


def tree_picked_file(root):
    touch(root, ["B00001.tif", "B00002.tif"])
    return root / "B00001.tif", make_ctx(root / "B00001.tif"), root / "B00001.tif", 1


def tree_set_in_folder(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.set").write_bytes(b"stub")
    ctx = make_ctx(root, image_type="lavision_set", image_format=("data.set",))
    return root, ctx, root, 1


def tree_set_companion(root):
    touch(root / "run", ["Frame0-0.ims"])
    (root / "run.set").write_bytes(b"stub")
    ctx = make_ctx(root / "run", image_type="lavision_set")
    return root / "run", ctx, root / "run", 1


def tree_wrong_type(root):
    touch(root, ["B00001.tif", "B00002.tif", "B00003.tif"])
    ctx = make_ctx(root, image_type="lavision_im7", image_format=("B%05d.im7",))
    return root, ctx, root, 1


def tree_camera_folders(root):
    for i in (1, 2, 3):
        touch(root / f"camera{i}", ["B00001.im7"])
    ctx = make_ctx(root, image_type="lavision_im7", image_format=("B%05d.im7",))
    return root, ctx, root, 1


def tree_inside_camera_folder(root):
    for i in (1, 2, 3):
        touch(root / f"camera{i}", ["B00001.tif"])
    ctx = make_ctx(root / "camera1", camera_count=3, camera_numbers=(1, 2, 3))
    return root / "camera1", ctx, root / "camera1", 1


def tree_flat_with_subfolders(root):
    touch(root, ["B00001.tif"])
    ctx = make_ctx(root, camera_count=2, camera_numbers=(1, 2))
    return root, ctx, root / "Cam1", 1


def tree_set_streams(root):
    root.mkdir(parents=True, exist_ok=True)
    s = build_raw16_set(root, PLIF)
    ctx = make_ctx(
        s,
        image_type="lavision_set",
        camera_count=1,
        camera_numbers=(1,),
        pair_reader=True,
    )
    return s, ctx, s, 1


def tree_pattern(root):
    touch(root, frame_names("Cam1_%05d.tif", 1, 10))
    return root, make_ctx(root, image_format=("nope%05d.tif",)), root, 10


def tree_start_index(root):
    touch(root, [f"B{i:05d}.tif" for i in range(0, 10)])
    return root, make_ctx(root, start_index=1), root, 10


ALL_FIXTURE_TREES = [
    tree_path_case,
    tree_picked_file,
    tree_set_in_folder,
    tree_set_companion,
    tree_wrong_type,
    tree_camera_folders,
    tree_inside_camera_folder,
    tree_flat_with_subfolders,
    tree_set_streams,
    tree_pattern,
    tree_start_index,
]


@pytest.mark.usefixtures("clean_set_cache")
@pytest.mark.parametrize("build", ALL_FIXTURE_TREES)  # Review Focus 5
def test_every_rule_output_is_json_safe(tmp_path, build):
    source, ctx, camera_path, expected_count = build(tmp_path)
    out = diagnose(
        scan_source(source),
        ctx,
        1,
        camera_path,
        expected_count,
        False,
        detected_start_index=detected_start_index_of(camera_path, ctx.image_format[0]),
    )
    assert out, build.__name__  # every tree must fire its rule
    for s in out:
        json.loads(json.dumps(s))
        assert set(s["apply"]) <= APPLY_FIELDS


@pytest.mark.usefixtures("clean_set_cache")
def test_fixture_trees_cover_every_rule(tmp_path):
    emitted = set()
    for i, build in enumerate(ALL_FIXTURE_TREES):
        source, ctx, camera_path, expected_count = build(tmp_path / str(i))
        emitted |= {
            s["id"]
            for s in diagnose(
                scan_source(source),
                ctx,
                1,
                camera_path,
                expected_count,
                False,
                detected_start_index=detected_start_index_of(
                    camera_path, ctx.image_format[0]
                ),
            )
        }
    assert emitted == {rule_id for rule_id, _ in RULES}


# --- validators ---


def _fake_frame(_idx):
    import numpy as np

    return np.zeros((8, 8), dtype=np.uint8)


def test_found_for_without_found_count_describes_the_disk(tmp_path):
    touch(tmp_path, ["B00001.tif"])
    scan = scan_source(tmp_path)
    result = {"valid": True, "format_detected": "tif"}
    assert found_for(scan, make_ctx(tmp_path), result, None, None) == scan_sentence(
        scan
    )


def test_validate_images_generic_rejects_mismatched_diagnosis(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        validate_images_generic(
            tmp_path,
            1,
            "B%05d.im7",
            "lavision_im7",
            1,
            False,
            _fake_frame,
            diagnosis=make_ctx(tmp_path),
        )


def test_validate_images_generic_adds_diagnosis_on_every_branch(tmp_path):
    touch(tmp_path, ["Cam1_00001.tif"])
    for fmt in ("", "nope%05d.tif"):
        result = validate_images_generic(
            tmp_path,
            1,
            fmt,
            "standard",
            1,
            False,
            _fake_frame,
            diagnosis=make_ctx(tmp_path, image_format=(fmt,)),
        )
        assert {"found", "suggestions", "streams"} <= set(result)
        assert not any(k.startswith("suggest") and k != "suggestions" for k in result)


def test_apply_diagnosis_reports_an_unfinished_diagnosis(tmp_path, monkeypatch):
    touch(tmp_path, ["Cam1_00001.tif"])

    def boom(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(SD, "diagnose", boom)
    result = {
        "valid": False,
        "error": "core text",
        "found_count": 0,
        "detected_start_index": None,
    }
    apply_diagnosis(result, make_ctx(tmp_path), 1, tmp_path, 1)
    assert result["error"] == "core text"
    assert result["suggestions"] == []
    assert result["found"].endswith(
        "Could not finish the diagnosis: Permission denied."
    )


def test_single_pattern_role_suggestion_replaces_only_its_index(tmp_path):
    touch(tmp_path, frame_names("B%05d_A.tif", 1, 3) + frame_names("B%05d_B.tif", 1, 3))
    result = validate_single_pattern(
        tmp_path,
        "wrong%05d_B.tif",
        "standard",
        3,
        False,
        role="B",
        patterns=["B%05d_A.tif", "wrong%05d_B.tif"],
        index=1,
    )
    assert set(result) == {
        "valid",
        "found_count",
        "error",
        "suggestions",
        "sample_files",
    }
    assert [s["apply"] for s in result["suggestions"]] == [
        {"image_format": ["B%05d_A.tif", "B%05d_B.tif"]}
    ]


def test_single_pattern_missing_camera_path_has_no_suggestions(tmp_path):
    result = validate_single_pattern(
        tmp_path / "Cam1",
        "B%05d.tif",
        "standard",
        1,
        False,
        patterns=["B%05d.tif"],
        index=0,
    )
    assert result["error"] == f"Camera path does not exist: {tmp_path / 'Cam1'}"
    assert result["suggestions"] == []


# --- routes ---


def test_validate_files_passes_core_set_message_through(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    touch(folder / "run", ["Frame0-0.ims"])
    (folder / "run.set").write_bytes(b"stub")
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(folder)]},
        images={
            "image_type": "lavision_set",
            "image_format": ["data.set"],
            "pairing_preset": "pre_paired",
            "frame_stride": 0,
        },
    ).get_json()
    cam = body["details"]["camera_1"]
    assert cam["error"] == f"Expected .set file but got: {folder}"
    assert "data.set" not in cam["error"]
    assert (cam["suggestions"][0]["apply"], cam["suggestions"][0]["label"]) == (
        {"source_path": str(folder / "run.set")},
        'Use "run.set"',
    )
    assert cam["found"] == (
        f"In {folder.as_posix()}: 1 .set file; 1 folder (run) with 1 .ims file."
    )
    assert cam["streams"] is None
    assert not any(k.startswith("suggested_") for k in body)
    assert not any(k.startswith("suggested_") for k in cam)


def test_validate_files_flat_files_suggest_one_camera(tmp_path, monkeypatch):
    src = tmp_path / "src"
    touch(src, ["B00001.tif", "B00002.tif", "B00003.tif"])
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(src)], "camera_count": 2, "camera_numbers": [1, 2]},
        images={"image_format": ["B%05d.tif"], "num_images": 3},
    ).get_json()
    applies = [s["apply"] for s in body["details"]["camera_1"]["suggestions"]]
    assert {"camera_count": 1, "camera_numbers": [1]} in applies


@pytest.mark.parametrize("idx", [3, -1, "0"])
def test_validate_files_rejects_out_of_range_source_idx(tmp_path, monkeypatch, idx):
    r = piv_route_response(tmp_path, monkeypatch, json={"source_path_idx": idx})
    assert r.status_code == 400
    assert r.get_json()["error"] == (
        f"source_path_idx {idx} is out of range: 1 source paths configured"
    )


def test_calibration_route_set_in_folder(tmp_path, monkeypatch):
    folder = tmp_path / "cal"
    touch(folder / "run", ["Frame0-0.ims"])
    (folder / "run.set").write_bytes(b"stub")
    save_settings(
        folder, {"image": {"image_type": "lavision_set", "image_format": "data.set"}}
    )
    out = (
        calibration_client(folder, monkeypatch)
        .post("/calibration/validate", json={"camera": 1, "source_path_idx": 0})
        .get_json()
    )
    assert out["suggestions"][0]["apply"]["source_path"] == str(folder / "run.set")


def test_calibration_im7_camera_folders_at_count_one(tmp_path, monkeypatch):
    src = tmp_path / "calib"
    for i in (1, 2, 3):
        touch(src / f"camera{i}", ["B00001.im7", "B00002.im7"])
    save_settings(
        src,
        {
            "image": {
                "image_type": "lavision_im7",
                "image_format": "calib%05d.tif",
                "n_views": 2,
            }
        },
    )
    out = (
        calibration_client(src, monkeypatch)
        .post("/calibration/validate", json={"camera": 1, "source_path_idx": 0})
        .get_json()
    )
    assert out["error"] == (
        "First frame not found. Looking for: calib00001.tif. "
        "No files here, only folders: camera1, camera2, camera3"
    )
    assert out["found"] == (
        f"In {src.as_posix()}: no images here; 3 folders (camera1, camera2, camera3), "
        "each with 2 .im7 files."
    )
    assert [s["label"] for s in out["suggestions"][:2]] == [
        "Set camera count to 3 (also changes Setup)",
        "Use camera1/ only",
    ]


def test_calibration_route_missing_sidecar_carries_empty_diagnosis(
    tmp_path, monkeypatch
):
    out = (
        calibration_client(tmp_path, monkeypatch)
        .post("/calibration/validate", json={"camera": 1, "source_path_idx": 0})
        .get_json()
    )
    assert out["valid"] is False
    assert (out["found"], out["suggestions"], out["streams"]) == (None, [], None)


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_validate_files_survives_unlistable_subfolder(tmp_path, monkeypatch):
    src = tmp_path / "src"
    touch(src, ["Cam1_00001.tif"])
    locked = src / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        r = piv_route_response(
            tmp_path,
            monkeypatch,
            paths={"source_paths": [str(src)]},
            images={"image_format": ["nope%05d.tif"], "num_images": 1},
        )
        assert r.status_code == 200
        cam = r.get_json()["details"]["camera_1"]
        assert cam["error"].startswith(
            "First frame not found. Looking for: nope00001.tif"
        )
        assert cam["found"].endswith(
            f"Could not list {locked.as_posix()}: Permission denied."
        )
    finally:
        locked.chmod(0o755)


def test_validate_files_unreadable_set_index_reports_reason(
    tmp_path, monkeypatch, clean_set_cache
):
    folder = tmp_path / "data"
    touch(folder / "run", [])
    (folder / "run" / "Frame0-0.ims").write_bytes(b"xx")  # index too small
    (folder / "run" / "Frame0-1.ims").write_bytes(b"yy")
    set_file = folder / "run.set"
    set_file.write_bytes(b"stub")
    cam = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(set_file)]},
        images={
            "image_type": "lavision_set",
            "image_format": ["run.set"],
            "pairing_preset": "pre_paired",
            "frame_stride": 0,
        },
    ).get_json()["details"]["camera_1"]
    reason = f"Index file too small: {folder / 'run' / 'Frame0-0.ims'}"
    assert cam["status"] == "error"
    assert cam["error"] == f"Could not read the image count of {set_file}: {reason}"
    assert cam["found"] == f"run.set: streams could not be read ({reason})."
    assert cam["suggestions"] == []
    assert cam["streams"] is None


def test_validate_files_multi_camera_cine_never_suggests_a_literal(
    tmp_path, monkeypatch
):
    src = tmp_path / "src"
    touch(src, ["rec_1.cine", "rec_2.cine"])
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(src)], "camera_count": 2, "camera_numbers": [1, 2]},
        images={"image_type": "cine", "image_format": ["run.cine"], "num_images": 1},
    ).get_json()
    for key in ("camera_1", "camera_2"):
        formats = [
            s["apply"]["image_format"]
            for s in body["details"][key]["suggestions"]
            if "image_format" in s["apply"]
        ]
        assert formats == [["rec_%d.cine"]]


def test_validate_files_single_cine_at_two_cameras_suggests_nothing(
    tmp_path, monkeypatch
):
    src = tmp_path / "src"
    touch(src, ["rec_1.cine"])
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(src)], "camera_count": 2, "camera_numbers": [1, 2]},
        images={"image_type": "cine", "image_format": ["run.cine"], "num_images": 1},
    ).get_json()
    for key in ("camera_1", "camera_2"):
        assert not any(
            "image_format" in s["apply"] for s in body["details"][key]["suggestions"]
        )


def test_calibration_output_folder_hidden_only_with_its_sidecar(tmp_path):
    touch(tmp_path / "camera1", ["B00001.im7"])
    touch(tmp_path / "calibration", ["B00001.tif"])  # the user's own folder
    assert "calibration" in [d.name for d in scan_source(tmp_path).subdirs]
    assert "calibration" in _append_folder_contents("x", tmp_path)
    save_settings(
        tmp_path,
        {"image": {"image_type": "lavision_im7", "image_format": "B%05d.im7"}},
    )
    assert [d.name for d in scan_source(tmp_path).subdirs] == ["camera1"]
    assert _append_folder_contents("x", tmp_path) == (
        "x. No files here, only folders: camera1"
    )


# --- final review fixes ---


def _malformed_decoder_set(root: Path) -> Path:
    """A ``.set`` whose stream-0 decoder XML is truncated (an ``ET.ParseError``)."""
    from fixtures.lavision import build_set

    set_file = build_set(
        root, "raw-16-bit", [[bytes(48)], [bytes(48)]], 6, 4, name="run"
    )
    (root / "run" / "Frame0-decoder.xml").write_text(
        "<FrameDecoder><id>raw-16-bit</id>"
    )
    return set_file


def test_validate_files_malformed_decoder_keeps_core_error(
    tmp_path, monkeypatch, clean_set_cache
):
    set_file = _malformed_decoder_set(tmp_path)
    cam = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(set_file)]},
        images={
            "image_type": "lavision_set",
            "image_format": ["run.set"],
            "pairing_preset": "pre_paired",
            "frame_stride": 0,
            "num_images": 1,
        },
    ).get_json()["details"]["camera_1"]
    reason = "no element found: line 1, column 33"
    assert cam["error"] == f"Could not read the image count of {set_file}: {reason}"
    assert cam["found"] == f"run.set: streams could not be read ({reason})."
    assert cam["suggestions"] == []


def test_calibration_route_malformed_decoder_keeps_core_error(
    tmp_path, monkeypatch, clean_set_cache
):
    set_file = _malformed_decoder_set(tmp_path)
    save_settings(
        set_file, {"image": {"image_type": "lavision_set", "image_format": "run.set"}}
    )
    out = (
        calibration_client(set_file, monkeypatch)
        .post("/calibration/validate", json={"camera": 1, "source_path_idx": 0})
        .get_json()
    )
    reason = "no element found: line 1, column 33"
    assert out["error"] == f"Could not read the image count of {set_file}: {reason}"
    assert out["found"] == f"run.set: streams could not be read ({reason})."
    assert out["suggestions"] == []


@pytest.mark.parametrize("exc", [KeyError("weird_type"), SyntaxError("bad xml")])
def test_apply_diagnosis_reports_any_rule_failure(tmp_path, monkeypatch, exc):
    touch(tmp_path, ["Cam1_00001.tif"])

    def boom(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(SD, "diagnose", boom)
    result = {
        "valid": False,
        "error": "core text",
        "found_count": 0,
        "detected_start_index": None,
    }
    apply_diagnosis(result, make_ctx(tmp_path), 1, tmp_path, 1)
    assert result["error"] == "core text"
    assert result["suggestions"] == []
    assert result["found"].endswith(f"Could not finish the diagnosis: {exc}.")


def test_apply_diagnosis_reports_a_found_builder_failure(tmp_path, monkeypatch):
    touch(tmp_path, ["Cam1_00001.tif"])

    def boom(*_args, **_kwargs):
        raise ValueError("found exploded")

    monkeypatch.setattr(SD, "found_for", boom)
    result = {
        "valid": False,
        "error": "core text",
        "found_count": 0,
        "detected_start_index": None,
    }
    apply_diagnosis(result, make_ctx(tmp_path), 1, tmp_path, 1)
    assert result["error"] == "core text"
    assert result["found"] == "Could not finish the diagnosis: found exploded."


def test_start_index_rule_ignores_a_truncated_sample(tmp_path):
    import dataclasses

    touch(tmp_path, frame_names("B%05d.tif", 0, 49))
    inp = _inp(tmp_path, start_index=0, expected_count=60)
    # A capped listing whose arbitrary sample happens to miss frame 0.
    sample = dataclasses.replace(
        inp.scan,
        root_names=tuple(n for n in inp.scan.root_names if n != "B00000.tif"),
        truncated=True,
    )
    assert inp.detected_start_index == 0
    assert rule_start_index(dataclasses.replace(inp, scan=sample)) == []


def _count_scans(monkeypatch) -> list:
    """Record every ``scan_source`` path the diagnosis makes."""
    calls = []
    real = SD.scan_source

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(SD, "scan_source", counting)
    return calls


def test_validate_files_passing_multi_camera_does_not_scan(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for c in (1, 2, 3):
        touch(src / f"Cam{c}", frame_names("B%05d.tif", 1, 3))
    calls = _count_scans(monkeypatch)
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={
            "source_paths": [str(src)],
            "camera_count": 3,
            "camera_numbers": [1, 2, 3],
        },
        images={"image_format": ["B%05d.tif"], "num_images": 3},
    ).get_json()
    assert calls == []
    for c in (1, 2, 3):
        cam = body["details"][f"camera_{c}"]
        assert cam["found"] == "3 cameras × 3 .tif, image size could not be read."
        assert cam["suggestions"] == []


def test_validate_files_failing_multi_camera_scans_once(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for c in (1, 2, 3):
        touch(src / f"Cam{c}", frame_names("B%05d.tif", 1, 3))
    calls = _count_scans(monkeypatch)
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={
            "source_paths": [str(src)],
            "camera_count": 3,
            "camera_numbers": [1, 2, 3],
        },
        images={"image_format": ["nope%05d.tif"], "num_images": 3},
    ).get_json()
    assert len(calls) == 1
    assert all(body["details"][f"camera_{c}"]["found"] for c in (1, 2, 3))


def test_calibration_im7_camera_folders_also_fix_the_pattern(tmp_path, monkeypatch):
    src = tmp_path / "calib"
    for i in (1, 2, 3):
        touch(src / f"camera{i}", ["B00001.im7", "B00002.im7"])
    save_settings(
        src,
        {
            "image": {
                "image_type": "lavision_im7",
                "image_format": "calib%05d.tif",
                "n_views": 2,
            }
        },
    )
    out = (
        calibration_client(src, monkeypatch, camera_count=3)
        .post("/calibration/validate", json={"camera": 1, "source_path_idx": 0})
        .get_json()
    )
    [s] = [s for s in out["suggestions"] if s["id"] == "camera_folders"]
    assert s["apply"] == {
        "use_camera_subfolders": True,
        "camera_subfolders": ["camera1", "camera2", "camera3"],
        "image_format": ["B%05d.im7"],
    }


def test_camera_folders_keep_a_matching_pattern(tmp_path):
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
    assert "image_format" not in s["apply"]


def test_inside_camera_folder_also_fixes_the_pattern(tmp_path):
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.tif", "B00002.tif"])
    [s] = rule_inside_camera_folder(
        _inp(
            tmp_path / "camera1",
            camera_count=3,
            camera_numbers=(1, 2, 3),
            image_format=("nope%05d.tif",),
        )
    )
    assert s["apply"] == {
        "source_path": str(tmp_path),
        "camera_subfolders": ["camera1", "camera2", "camera3"],
        "image_format": ["B%05d.tif"],
    }


def test_found_marks_the_subfolder_cap(tmp_path):
    for i in range(SD.SCAN_MAX_SUBDIRS + 1):
        (tmp_path / f"run{i:03d}").mkdir()
    scan = scan_source(tmp_path)
    assert scan.subdirs_truncated
    assert f"; {SD.SCAN_MAX_SUBDIRS}+ folders (" in scan_sentence(scan)
    (tmp_path / f"run{SD.SCAN_MAX_SUBDIRS:03d}").rmdir()
    assert f"; {SD.SCAN_MAX_SUBDIRS} folders (" in scan_sentence(scan_source(tmp_path))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("/data/runs'", "/data/runs'"),  # a trailing apostrophe is part of the name
        ("'/data/Bob's run'", "/data/Bob's run"),
        ("\"'/data/run'\"", "'/data/run'"),  # exactly one pair
        ("\"/data/run'", "\"/data/run'"),  # not a matching pair
    ],
)
def test_normalise_source_path_strips_one_matching_pair(raw, expected):
    assert normalise_source_path(raw) == expected


@pytest.mark.parametrize("blank", ["", "   ", '""'])
def test_validate_files_blank_source_is_reported_not_scanned(
    tmp_path, monkeypatch, blank
):
    calls = _count_scans(monkeypatch)
    cam = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [blank]},
        images={"image_format": ["B%05d.tif"], "num_images": 1},
    ).get_json()["details"]["camera_1"]
    assert calls == []
    assert cam["status"] == "error"
    assert cam["error"] == "No source path set."
    assert cam["found"] == "No source path set."
    assert (cam["suggestions"], cam["streams"]) == ([], None)


def test_append_folder_contents_reports_the_true_count(tmp_path):
    touch(tmp_path, [f"f{i:02d}.txt" for i in range(12)])
    assert _append_folder_contents("x", tmp_path) == (
        "x. Found 12 files: f00.txt, f01.txt, f02.txt, f03.txt, f04.txt and 7 more..."
    )


def test_raster_family_is_exactly_what_the_core_globs(tmp_path):
    from pivtools_core.image_handling.path_utils import RASTER_EXTENSIONS

    assert SD.FAMILY_EXTS["raster"] == RASTER_EXTENSIONS
    touch(tmp_path, ["B00001.bmp", "B00002.bmp"])
    scan = scan_source(tmp_path)
    assert scan.root_names == ()
    assert SD.family_of("B00001.bmp") is None


def test_inside_camera_folder_reuses_the_request_scan_cache(tmp_path, monkeypatch):
    # Three cameras validated in one request, each inside camera1/: the parent
    # listing must be done once, not once per camera.
    for i in (1, 2, 3):
        touch(tmp_path / f"camera{i}", ["B00001.tif"])
    calls = []
    real = SD.scan_source
    monkeypatch.setattr(SD, "scan_source", lambda p: calls.append(str(p)) or real(p))
    cache = {}
    for camera in (1, 2, 3):
        inp = _inp(
            tmp_path / "camera1",
            camera=camera,
            camera_count=3,
            camera_numbers=(1, 2, 3),
        )
        inp = replace(inp, scan_cache=cache)
        assert rule_inside_camera_folder(inp)
    assert calls.count(str(tmp_path)) == 1


def test_set_streams_reader_bug_is_named_in_found(
    tmp_path, monkeypatch, clean_set_cache
):
    # An OSError/ValueError from the reader is an unreadable recording and the rule
    # stays silent (found reports it). Anything else is a bug: it must not be
    # swallowed, and apply_diagnosis names it in ``found``.
    s = build_raw16_set(tmp_path, [(4, 6)] * 2)
    monkeypatch.setattr(
        SD, "read_set_info", lambda *a, **k: (_ for _ in ()).throw(KeyError("Frame0"))
    )
    with pytest.raises(KeyError):
        rule_set_streams(_inp(s, image_type="lavision_set", pair_reader=True))
    ctx = make_ctx(s, image_type="lavision_set", pair_reader=True, camera_count=1)
    result = {"valid": False, "error": "x", "detected_start_index": None}
    apply_diagnosis(result, ctx, 1, s, 1)
    assert result["suggestions"] == []
    assert "Could not finish the diagnosis" in result["found"]
