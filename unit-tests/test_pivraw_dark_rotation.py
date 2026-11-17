#!/usr/bin/env python3
"""
test_pivraw_dark_rotation.py

The ``.pivraw`` reader applies the run's in-run dark and its view rotation, so
a ``.pivraw`` image reaches PIVTOOLs the way a DaVis image does (ticket P3,
Morgan's rules of 2026-11-17).

Locks in:
  * Every frame of the contract dark cases equals the acquisition package's
    own ``pixels.decode_minus_dark`` (``<case>__Camera<k>__dark_subtracted`` in
    ``expected_arrays.npz``): the dark of the frame's own A/B role, subtracted
    per photosite and clipped at 0, before the Bayer 2x2 sum.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.diagnosis import make_ctx, piv_route_response  # noqa: E402
from fixtures.pivraw import (  # noqa: E402
    CONTRACT_DIR,
    DROP,
    contract_cases,
    copy_case,
    expected_frames,
    write_piv_config,
)

from pivtools_core.config import Config  # noqa: E402
from pivtools_core.image_handling.path_utils import (  # noqa: E402
    validate_images_generic,
)
from pivtools_core.image_handling.readers.pivraw_reader import (  # noqa: E402
    PivrawError,
    clear_header_cache,
    get_pivraw_image_shape,
    pivraw_dark_note,
    read_pivraw_single,
)
from pivtools_core.validation import validate_config  # noqa: E402

CASES = contract_cases()
DARK_CASES = [case for case, spec in CASES.items() if "dark" in spec]


@pytest.fixture(autouse=True)
def _clean_header_cache():
    clear_header_cache()
    yield
    clear_header_cache()


def _dark_subtracted(case: str, camera: int) -> np.ndarray:
    """uint16 ``(n_frames, H, W)``: the acquisition's ``decode_minus_dark`` of every frame."""
    with np.load(CONTRACT_DIR / "expected_arrays.npz") as arrays:
        return arrays[f"{case}__Camera{camera}__dark_subtracted"]


def test_the_contract_has_both_dark_cases():
    assert sorted(DARK_CASES) == ["df_2cam_dark", "tr_bayer_dark"]


@pytest.mark.parametrize(
    "case, camera",
    [(case, int(cam)) for case in DARK_CASES for cam in CASES[case]["cameras"]],
)
def test_dark_case_frames_equal_the_apps_decode_minus_dark(case, camera):
    expected = _dark_subtracted(case, camera)
    raw = CONTRACT_DIR / case / f"Camera{camera}.pivraw"
    assert expected.shape[0] == CASES[case]["cameras"][str(camera)]["n_frames"]
    for k in range(expected.shape[0]):
        np.testing.assert_array_equal(
            read_pivraw_single(raw, k + 1), expected[k].astype(np.float32)
        )


def _mosaic(path: Path, k: int, side: int = 8) -> np.ndarray:
    """Frame ``k`` (0-based) of an 8 x 8 BayerRG8 fixture as int64 photosites."""
    blob = path.read_bytes()
    return np.frombuffer(blob, np.uint8, side * side, k * 4096).reshape(side, side)


def _superpixel_sum(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.int64)
    return m[0::2, 0::2] + m[0::2, 1::2] + m[1::2, 0::2] + m[1::2, 1::2]


def test_bayer_dark_is_subtracted_per_photosite_before_the_sum():
    """Hand computation from the raw bytes: clip each photosite at 0, then sum.

    Summing first and subtracting the dark's sum gives a different image on
    this fixture, so the test would catch the wrong order.
    """
    run = CONTRACT_DIR / "tr_bayer_dark"
    dark = _mosaic(run / "dark" / "Camera1.pivraw", 0).astype(np.int64)
    differs = False
    for k in range(3):
        frame = _mosaic(run / "Camera1.pivraw", k).astype(np.int64)
        per_photosite = _superpixel_sum(np.clip(frame - dark, 0, None))
        sum_first = np.clip(_superpixel_sum(frame) - _superpixel_sum(dark), 0, None)
        got = read_pivraw_single(run / "Camera1.pivraw", k + 1)
        np.testing.assert_array_equal(got, per_photosite.astype(np.float32))
        differs |= bool(np.any(per_photosite != sum_first))
    assert differs, "the fixture cannot tell subtract-then-sum from sum-then-subtract"


@pytest.mark.parametrize("case", ["df_2cam_clean", "tr_bayer_clean"])
def test_run_without_a_dark_reads_raw_with_a_note(case):
    raw = CONTRACT_DIR / case / "Camera1.pivraw"
    expected = expected_frames(case, 1)
    for k in range(expected.shape[0]):
        np.testing.assert_array_equal(
            read_pivraw_single(raw, k + 1), expected[k].astype(np.float32)
        )
    note = pivraw_dark_note(raw)
    assert note is not None and "No dark was subtracted" in note
    assert "dark: null" in note


def test_run_with_a_dark_has_no_note():
    assert pivraw_dark_note(CONTRACT_DIR / "df_2cam_dark" / "Camera1.pivraw") is None


def _edit_run_yaml(run: Path, **changes) -> None:
    path = run / "run.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            data.pop(key)
        else:
            data[key] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _swap_dark_camera(run: Path, camera: int, source: Path) -> None:
    """Replace the in-run dark's three files for ``camera`` with ``source``'s."""
    for suffix in ("", ".json", ".idx"):
        shutil.copyfile(
            Path(str(source) + suffix),
            run / "dark" / f"Camera{camera}.pivraw{suffix}",
        )


def _foreign_geometry(run: Path) -> None:
    """A BayerRG8 8 x 8 time-resolved dark in a Mono12p 16 x 8 double-frame run."""
    _swap_dark_camera(run, 1, CONTRACT_DIR / "tr_bayer_dark" / "dark" / "Camera1.pivraw")


def _run_frames_as_dark(run: Path) -> None:
    """The run's own 4 frames in place of a one-trigger dark (same geometry)."""
    _swap_dark_camera(run, 1, CONTRACT_DIR / "df_2cam_dark" / "Camera1.pivraw")


def _dark_camera_missing(run: Path) -> None:
    for suffix in ("", ".json", ".idx"):
        (run / "dark" / f"Camera1.pivraw{suffix}").unlink()


def _dark_folder_missing(run: Path) -> None:
    shutil.rmtree(run / "dark")


_DARK_REFUSALS = {
    "foreign geometry": (
        _foreign_geometry,
        r"does not match camera 1 .*width 8 \(camera 16\).*pixel_format "
        r"'BayerRG8' \(camera 'Mono12p'\).*mode 'time_resolved' "
        r"\(camera 'double_frame'\)",
    ),
    "not one trigger": (_run_frames_as_dark, r"holds 4 frames; a double_frame dark"),
    "camera missing from the dark": (
        _dark_camera_missing,
        r"names the in-run dark dark_20261117_120000, but .*Camera1\.pivraw is missing",
    ),
    "dark folder missing": (_dark_folder_missing, r"Camera1\.pivraw is missing"),
    "dark key absent": (
        lambda run: _edit_run_yaml(run, dark=DROP),
        r"no 'dark' key \(a stale run.yaml",
    ),
    "dark outside the run": (
        lambda run: _edit_run_yaml(
            run, dark={"id": "dark_20261117_120000", "path": "../dark"}
        ),
        r"dark must be null or \{id, path: 'dark'\}",
    ),
    "dark without an id": (
        lambda run: _edit_run_yaml(run, dark={"path": "dark"}),
        r"dark must be null or",
    ),
}


@pytest.mark.parametrize("what", list(_DARK_REFUSALS))
def test_a_dark_that_does_not_fit_is_refused_by_name(tmp_path, what):
    edit, message = _DARK_REFUSALS[what]
    run = copy_case(tmp_path, "df_2cam_dark")
    edit(run)
    with pytest.raises(PivrawError, match=message):
        read_pivraw_single(run / "Camera1.pivraw", 1)
    with pytest.raises(PivrawError, match=message):
        get_pivraw_image_shape(run / "Camera1.pivraw")


def _validate(run: Path) -> dict:
    return validate_images_generic(
        camera_path=run,
        camera=1,
        image_format="Camera%d.pivraw",
        image_type="pivraw",
        expected_count=4,
        zero_based_indexing=False,
        read_frame_fn=lambda idx: read_pivraw_single(run / "Camera1.pivraw", idx),
        diagnosis=make_ctx(run, image_type="pivraw", image_format=("Camera%d.pivraw",)),
    )


def test_validation_notes_a_run_without_a_dark(tmp_path):
    raw_run = _validate(copy_case(tmp_path, "df_2cam_clean"))
    assert raw_run["valid"] is True
    assert len(raw_run["notes"]) == 1 and "No dark was subtracted" in raw_run["notes"][0]
    dark_run = _validate(copy_case(tmp_path, "df_2cam_dark"))
    assert dark_run["valid"] is True and dark_run["notes"] == []


def test_config_validation_warns_that_no_dark_was_subtracted(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    config_path = write_piv_config(tmp_path, run, 2, 4)
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["base_paths"] = [str(tmp_path / "out")]
    config_path.write_text(yaml.dump(data))
    ok, error, warnings = validate_config(Config(path=str(config_path)))
    assert ok, error
    for camera in (1, 2):
        assert any(
            w.startswith(f"Camera {camera}: No dark was subtracted") for w in warnings
        ), warnings


def test_gui_validate_route_carries_the_note(tmp_path, monkeypatch):
    run = copy_case(tmp_path, "df_2cam_clean")
    resp = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(run)], "camera_count": 2, "camera_numbers": [1, 2]},
        images={
            "image_type": "pivraw",
            "image_format": ["Camera%d.pivraw"],
            "num_images": 4,
            "pairing_preset": "skip_frames",
            "frame_stride": 1,
            "pair_stride": 2,
        },
        json={"camera": 1},
    )
    assert resp.status_code == 200, resp.get_json()
    camera = resp.get_json()["details"]["camera_1"]
    assert any("No dark was subtracted" in n for n in camera["notes"]), camera


def test_missing_run_yaml_is_refused(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    (run / "run.yaml").unlink()
    with pytest.raises(PivrawError, match=r"run.yaml: cannot read the run description"):
        read_pivraw_single(run / "Camera1.pivraw", 1)
