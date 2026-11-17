"""The image note (pivtools-acquisition CONTRACTS §3.2): the strict reader.

The note says what the acquisition app applied to the images it saved for
PIVTOOLs (decode rule, dark, view rotation, camera). The fixtures in
``fixtures/image_note_contract/`` were written by the app's own writer, so the
key set and the vocabularies are pinned to the writer, not to this reader.
"""

import json
from pathlib import Path

import pytest

from pivtools_core.image_handling.readers import image_note
from pivtools_core.image_handling.readers.image_note import (
    ImageNoteError,
    read_image_note,
)

CONTRACT_DIR = Path(__file__).resolve().parent / "fixtures" / "image_note_contract"


def _manifest() -> dict:
    return json.loads((CONTRACT_DIR / "MANIFEST.json").read_text(encoding="utf-8"))


def _sample(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


def _write(tmp_path: Path, content) -> Path:
    path = tmp_path / "00001.tif.json"
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


def test_reader_constants_match_the_writer_manifest():
    """A regenerated fixture that adds a key, a pixel rule or a rotation fails
    here until the reader follows."""
    manifest = _manifest()
    assert manifest["format_version"] == image_note.NOTE_FORMAT_VERSION
    assert manifest["note_keys"] == list(image_note.NOTE_KEYS)
    assert manifest["pixel_rules"] == image_note.PIXEL_RULES
    assert manifest["view_rotations"] == list(image_note.VIEW_ROTATIONS)


def test_writer_samples_read_as_written():
    cal = read_image_note(CONTRACT_DIR / "calibration_cx25.tif.json")
    assert (
        cal.view_rotation,
        cal.dark_id,
        cal.dark_subtracted,
        cal.pixel_rule,
        cal.camera_serial,
    ) == ("cw90", "dark_20261117_101500", True, "mono12p", "118581")
    export = read_image_note(CONTRACT_DIR / "export_jai.image_note.json")
    assert (
        export.view_rotation,
        export.dark_id,
        export.dark_subtracted,
        export.pixel_rule,
        export.camera_serial,
    ) == ("none", None, False, "bayer_rggb_superpixel_sum", "U512345")


@pytest.mark.parametrize("key", ["view_rotation", "camera_serial", "dark_id"])
def test_a_missing_key_is_refused_by_name(tmp_path, key):
    data = _sample("calibration_cx25.tif.json")
    del data[key]
    path = _write(tmp_path, data)
    with pytest.raises(ImageNoteError, match=rf"missing {key}") as exc:
        read_image_note(path)
    assert str(path) in str(exc.value)


def test_an_unknown_key_is_refused_by_name(tmp_path):
    data = _sample("calibration_cx25.tif.json")
    data["exposure_us"] = 100
    with pytest.raises(ImageNoteError, match=r"unknown exposure_us"):
        read_image_note(_write(tmp_path, data))


@pytest.mark.parametrize("version", [2, 1.0, True, "1"])
def test_another_format_version_is_refused(tmp_path, version):
    data = _sample("calibration_cx25.tif.json")
    data["format_version"] = version
    with pytest.raises(ImageNoteError, match=r"format_version must be 1"):
        read_image_note(_write(tmp_path, data))


@pytest.mark.parametrize(
    "dark_id, dark_subtracted",
    [(None, True), ("dark_20261117_101500", False), ("dark_20261117_101500", 1)],
)
def test_dark_subtracted_disagreeing_with_dark_id_is_refused(
    tmp_path, dark_id, dark_subtracted
):
    data = _sample("calibration_cx25.tif.json")
    data["dark_id"], data["dark_subtracted"] = dark_id, dark_subtracted
    with pytest.raises(ImageNoteError, match=r"dark_subtracted"):
        read_image_note(_write(tmp_path, data))


@pytest.mark.parametrize(
    "key, value, words",
    [
        ("view_rotation", "cw270", "view rotation"),
        ("pixel_rule", "Mono12p", "pixel rule"),
        ("camera_serial", "", "camera_serial"),
        ("camera_serial", 118581, "camera_serial"),
    ],
)
def test_a_value_outside_the_contract_is_refused(tmp_path, key, value, words):
    data = _sample("calibration_cx25.tif.json")
    data[key] = value
    with pytest.raises(ImageNoteError, match=words):
        read_image_note(_write(tmp_path, data))


def test_an_empty_dark_id_is_refused(tmp_path):
    data = _sample("calibration_cx25.tif.json")
    data["dark_id"] = ""
    with pytest.raises(ImageNoteError, match=r"dark_id"):
        read_image_note(_write(tmp_path, data))


@pytest.mark.parametrize("text", ["{not json", "[1, 2]"])
def test_a_file_that_is_not_a_note_object_is_refused(tmp_path, text):
    path = _write(tmp_path, text)
    with pytest.raises(ImageNoteError) as exc:
        read_image_note(path)
    assert str(path) in str(exc.value)


def test_a_missing_file_is_refused_by_name(tmp_path):
    path = tmp_path / "00001.tif.json"
    with pytest.raises(ImageNoteError, match=r"cannot read the image note"):
        read_image_note(path)


def test_note_paths_follow_the_contract(tmp_path):
    assert image_note.calibration_note_path(tmp_path / "00001.tif") == (
        tmp_path / "00001.tif.json"
    )
    assert image_note.folder_note_path(tmp_path / "Camera1") == (
        tmp_path / "Camera1" / "image_note.json"
    )


@pytest.mark.parametrize(
    "name, is_note",
    [
        ("00001.tif.json", True),
        ("image_note.json", True),
        ("00001.tif", False),
        ("settings.yaml", False),
    ],
)
def test_is_note_file_names_both_note_kinds(name, is_note):
    assert image_note.is_note_file(name) is is_note
