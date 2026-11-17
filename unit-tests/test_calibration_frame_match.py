"""K1: a calibration is applied only to PIV images in its own pixel frame.

Locks in:
  * T9: ``.pivraw`` is not a calibration image type. A settings sidecar naming
    it, a live override naming it and a direct frame read of it are all refused
    by name. Calibration images from PIVTOOLS Acquisition are TIFFs with notes.
  * Applying a calibration to a ``.pivraw`` run compares every calibration
    image's note with the run's camera on view_rotation, dark_subtracted,
    pixel_rule and camera_serial, and refuses a mismatch, a missing note or a
    malformed note by name. dark_id is never compared.
  * DaVis: a ``.set`` calibration against a ``.set`` PIV source compares each
    camera's geometry transformer.
  * The note files are not images: they are neither counted nor offered as a
    source by the calibration folder checks and the source diagnosis.

Usage:
    pytest unit-tests/test_calibration_frame_match.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import (  # noqa: E402
    build_labelled_set,
    build_raw16_set,
    make_set_config,
    write_manifest,
)
from fixtures.pivraw import copy_case, edit_run_yaml, edit_sidecar  # noqa: E402

from pivtools_core.calibration_settings import (  # noqa: E402
    load_settings,
    save_settings,
)
from pivtools_core.config import Config  # noqa: E402
from pivtools_core.image_handling.calibration_loader import (  # noqa: E402
    get_calibration_frame_count,
    read_calibration_frame_at,
    validate_calibration_images,
)
from pivtools_core.image_handling.readers.pivraw_reader import (  # noqa: E402
    clear_header_cache,
)
from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    clear_set_info_cache,
)
from pivtools_gui.calibration import record as REC  # noqa: E402
from pivtools_gui.calibration import runio  # noqa: E402
from pivtools_gui.calibration.frame_match import (  # noqa: E402
    CalibrationFrameMismatch,
)
from pivtools_gui.calibration.pipeline import build_scale_factor_record  # noqa: E402

DF_CLEAN = "df_2cam_clean"
# Two Mono12p cameras SIM1 and SIM2, view_rotation none, in-run dark below.
DF_DARK = "df_2cam_dark"
RUN_DARK_ID = "dark_20261117_120000"
# A calibration is taken at another exposure with its own dark (A3 Q3).
CAL_DARK_ID = "dark_20261117_090000"
BOARD = "scale_factor"


@pytest.fixture(autouse=True)
def _fresh_reader_caches():
    """The .pivraw header and .set info caches are keyed on path and stamps, and
    tests rewrite runs and manifests in place."""
    clear_header_cache()
    clear_set_info_cache()
    yield
    clear_header_cache()
    clear_set_info_cache()


def _note(
    view_rotation="none",
    dark_id=CAL_DARK_ID,
    pixel_rule="mono12p",
    camera_serial="SIM1",
) -> dict:
    """A calibration image note in the contract's form (CONTRACTS §3.2)."""
    return {
        "format_version": 1,
        "view_rotation": view_rotation,
        "dark_id": dark_id,
        "dark_subtracted": dark_id is not None,
        "pixel_rule": pixel_rule,
        "camera_serial": camera_serial,
    }


def _calibration(tmp_path: Path, notes=None, n_images: int = 2) -> Path:
    """An acquisition-app calibration folder: Cam<k>/%05d.tif, each with its note.

    ``notes`` maps (camera, image number) to a note dict, or to None for an image
    without a note. Unlisted images get the note that matches the DF_DARK run.
    A scale-factor model is saved per camera so the apply planner has records.
    """
    notes = notes or {}
    source = tmp_path / "project" / "calibration"
    for cam in (1, 2):
        folder = source / f"Cam{cam}"
        folder.mkdir(parents=True)
        for n in range(1, n_images + 1):
            (folder / f"{n:05d}.tif").write_bytes(b"tif")
            note = notes.get((cam, n), _note(camera_serial=f"SIM{cam}"))
            if note is not None:
                text = note if isinstance(note, str) else json.dumps(note)
                (folder / f"{n:05d}.tif.json").write_text(text, encoding="utf-8")
        REC.save_mono(
            build_scale_factor_record(
                camera=cam, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
            ),
            REC.mono_model_dir_for_source(source, cam, BOARD),
        )
    save_settings(
        source, {"image": {"image_format": "%05d.tif", "image_type": "standard"}}
    )
    return source


def _piv_config(tmp_path: Path, run: Path) -> Config:
    """A config whose one PIV source is the .pivraw ``run``."""
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "results")],
            "source_paths": [str(run)],
            "camera_count": 2,
            "camera_numbers": [1, 2],
        },
        "images": {
            "image_format": ["Camera%d.pivraw"],
            "image_type": "pivraw",
            "num_images": 2,
            "start_index": 1,
            "pairing_preset": "skip_frames",
            "frame_stride": 1,
            "pair_stride": 2,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    return Config(path=str(path))


def _plan(cfg: Config, source: Path, **kwargs):
    return runio.plan_apply_units(cfg, source, BOARD, False, "instantaneous", **kwargs)


class _CalConfig:
    """The two Config members the calibration loader reads."""

    def __init__(self, source: Path, camera_count: int = 2):
        self._source = source
        self.camera_count = camera_count
        self.camera_numbers = list(range(1, camera_count + 1))
        self.time_resolved = False

    def get_calibration_source(self, idx: int) -> Path:
        return self._source


# ---------------------------------------------------------------------------
# T9: .pivraw is not a calibration image type
# ---------------------------------------------------------------------------

_PIVRAW_REFUSAL = r"'pivraw' is not a calibration image type"


def test_settings_naming_pivraw_are_refused(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    save_settings(
        run, {"image": {"image_format": "Camera%d.pivraw", "image_type": "pivraw"}}
    )
    with pytest.raises(ValueError, match=_PIVRAW_REFUSAL) as exc:
        load_settings(run)
    assert "settings.yaml" in str(exc.value)
    with pytest.raises(ValueError, match=_PIVRAW_REFUSAL):
        get_calibration_frame_count(2, _CalConfig(run), 0)


def test_a_live_pivraw_override_is_refused(tmp_path):
    """The GUI validates live form values before they are saved: a pivraw format
    typed over a valid TIFF sidecar is refused too."""
    run = copy_case(tmp_path, DF_CLEAN)
    save_settings(
        run, {"image": {"image_format": "%05d.tif", "image_type": "standard"}}
    )
    with pytest.raises(ValueError, match=_PIVRAW_REFUSAL):
        validate_calibration_images(
            1, _CalConfig(run), 0, image_format="Camera%d.pivraw"
        )


def test_a_direct_pivraw_frame_read_is_refused(tmp_path):
    """The calibration CLI reads frames straight through read_calibration_frame_at."""
    run = copy_case(tmp_path, DF_CLEAN)
    with pytest.raises(ValueError, match=_PIVRAW_REFUSAL):
        read_calibration_frame_at(
            camera_path=run,
            camera=1,
            frame_idx=1,
            image_format="Camera%d.pivraw",
            image_type="pivraw",
            num_cameras=2,
        )


# ---------------------------------------------------------------------------
# .pivraw PIV source: every calibration note against the run's camera
# ---------------------------------------------------------------------------


def test_matching_notes_apply(tmp_path):
    run = copy_case(tmp_path, DF_DARK)
    units = _plan(_piv_config(tmp_path, run), _calibration(tmp_path))
    assert sorted(u["label"] for u in units) == ["results/Cam1", "results/Cam2"]


def test_a_different_dark_id_alone_still_applies(tmp_path):
    """dark_id is provenance only: the calibration's dark is its own (A3 Q3)."""
    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(
        tmp_path, {(1, 1): _note(dark_id="dark_19990101_000000", camera_serial="SIM1")}
    )
    assert len(_plan(_piv_config(tmp_path, run), source)) == 2


@pytest.mark.parametrize(
    "field, note, words",
    [
        (
            "view_rotation",
            _note(view_rotation="cw90"),
            "view_rotation cw90 in its note, none in the run",
        ),
        (
            "dark_subtracted",
            _note(dark_id=None),
            "dark_subtracted false in its note, true in the run",
        ),
        (
            "pixel_rule",
            _note(pixel_rule="bayer_rggb_superpixel_sum"),
            "pixel_rule bayer_rggb_superpixel_sum in its note, mono12p in the run",
        ),
        (
            "camera_serial",
            _note(camera_serial="SIM9"),
            "camera_serial SIM9 in its note, SIM1 in the run",
        ),
    ],
)
def test_each_mismatched_field_is_refused_by_name(tmp_path, field, note, words):
    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(1, 2): note})
    with pytest.raises(CalibrationFrameMismatch) as exc:
        _plan(_piv_config(tmp_path, run), source)
    message = str(exc.value)
    assert f"camera 1, 00002.tif: {words}" in message
    # Only the one image differs; camera 2 and image 1 are not named.
    assert "00001.tif" not in message and "camera 2" not in message
    assert str(source) in message and str(run) in message


def test_the_run_side_comes_from_run_yaml_and_the_sidecar(tmp_path):
    """Rotation from run.yaml's slot, dark from run.yaml's dark key, serial and
    pixel format from the camera sidecar: change the run, not the notes."""
    run = copy_case(tmp_path, DF_DARK)
    edit_sidecar(run, 2, camera_serial="SIM7")
    with pytest.raises(CalibrationFrameMismatch) as exc:
        _plan(_piv_config(tmp_path, run), _calibration(tmp_path))
    assert "camera 2, 00001.tif: camera_serial SIM2 in its note, SIM7 in the run" in (
        str(exc.value)
    )


def test_a_run_without_a_dark_refuses_dark_subtracted_notes(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    with pytest.raises(
        CalibrationFrameMismatch,
        match="dark_subtracted true in its " "note, false in the run",
    ):
        _plan(_piv_config(tmp_path, run), _calibration(tmp_path))


def test_a_rotated_run_refuses_unrotated_notes(tmp_path):
    run = copy_case(tmp_path, DF_DARK)
    data = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    data["config"]["cameras"]["slots"][0]["view_rotation"] = "ccw90"
    edit_run_yaml(run, config=data["config"])
    with pytest.raises(
        CalibrationFrameMismatch,
        match="camera 1, 00001.tif: "
        "view_rotation none in its note, ccw90 in the run",
    ):
        _plan(_piv_config(tmp_path, run), _calibration(tmp_path))


def test_a_missing_note_is_refused_by_name(tmp_path):
    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(2, 1): None})
    with pytest.raises(
        CalibrationFrameMismatch, match="camera 2, 00001.tif: no " "image note"
    ):
        _plan(_piv_config(tmp_path, run), source)


@pytest.mark.parametrize(
    "bad, words",
    [
        ("{not json", "not JSON"),
        ({**_note(), "exposure_us": 5}, "unknown exposure_us"),
        ({**_note(), "dark_subtracted": False}, "dark_subtracted is false"),
    ],
)
def test_a_malformed_note_is_refused_by_name(tmp_path, bad, words):
    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(1, 1): bad})
    with pytest.raises(CalibrationFrameMismatch) as exc:
        _plan(_piv_config(tmp_path, run), source)
    assert "camera 1, 00001.tif:" in str(exc.value)
    assert words in str(exc.value)


def test_plain_tiff_calibration_against_a_run_is_refused(tmp_path):
    """No notes at all: the frames cannot be proven to match the run."""
    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(c, n): None for c in (1, 2) for n in (1, 2)})
    with pytest.raises(CalibrationFrameMismatch, match="cannot be proven"):
        _plan(_piv_config(tmp_path, run), source)


def test_the_gui_apply_route_returns_the_refusal(tmp_path, monkeypatch):
    """POST /calibration/apply plans through the same planner, answers 400 and
    starts no job."""
    from flask import Flask

    import pivtools_gui.calibration.app.views as V

    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(2, 2): _note(camera_serial="SIM1")})
    monkeypatch.setattr(V, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(V, "_source_path", lambda idx: source)
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    resp = app.test_client().post(
        "/calibration/apply", json={"board": BOARD, "stereo": False}
    )
    body = resp.get_json()
    assert resp.status_code == 400
    assert body["success"] is False
    assert "job_id" not in body
    assert "camera 2, 00002.tif: camera_serial SIM1 in its note, SIM2 in the run" in (
        body["error"]
    )


def test_the_cli_apply_refuses_before_writing(tmp_path, monkeypatch):
    """apply-calibration --all-paths plans through the same planner; cli.main
    prints a ValueError as 'error: ...' and exits 1."""
    from types import SimpleNamespace

    import pivtools_cli.calibration_cli as cli

    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(1, 1): _note(view_rotation="cw90")})
    monkeypatch.setattr(cli, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(
        cli.runio, "calibrate_mono_run", lambda *a, **k: pytest.fail("applied")
    )
    args = SimpleNamespace(
        board=BOARD,
        source=str(source),
        camera=None,
        dt=None,
        z_world=None,
        tilt_x=None,
        tilt_y=None,
        uncalibrated_dir=None,
        calibrated_dir=None,
        all_paths=True,
        type_name=None,
        model_type=None,
    )
    with pytest.raises(CalibrationFrameMismatch, match="view_rotation cw90"):
        cli.apply_calibration_command(args)


def test_an_explicit_apply_checks_every_configured_source(tmp_path):
    """Explicit uncalibrated/calibrated dirs name no base path, so the vectors'
    PIV source is unknown: every configured source is checked."""
    run = copy_case(tmp_path, DF_DARK)
    source = _calibration(tmp_path, {(1, 1): _note(view_rotation="180")})
    explicit = {"uncal": tmp_path / "u", "out": tmp_path / "o"}
    with pytest.raises(CalibrationFrameMismatch, match="view_rotation 180"):
        _plan(_piv_config(tmp_path, run), source, camera=1, explicit=explicit)


def test_a_pivraw_run_and_an_image_less_calibration_are_refused(tmp_path):
    """A scale factor typed in without images records no frame at all."""
    run = copy_case(tmp_path, DF_DARK)
    source = tmp_path / "typed"
    for cam in (1, 2):
        REC.save_mono(
            build_scale_factor_record(
                camera=cam, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
            ),
            REC.mono_model_dir_for_source(source, cam, BOARD),
        )
    with pytest.raises(CalibrationFrameMismatch, match="has no settings.yaml"):
        _plan(_piv_config(tmp_path, run), source)


# ---------------------------------------------------------------------------
# Exported TIFF run (Camera<k>/image_note.json) as the PIV source
# ---------------------------------------------------------------------------


def _exported_config(root: Path, notes) -> Config:
    """An exported TIFF run under ``root``: Camera<k>/ with the folder note ``notes[k]``."""
    export = root / "run" / "tiff"
    for cam, note in notes.items():
        folder = export / f"Camera{cam}"
        folder.mkdir(parents=True)
        (folder / "image_note.json").write_text(json.dumps(note), encoding="utf-8")
    cfg = {
        "paths": {
            "base_paths": [str(root / "results")],
            "source_paths": [str(export)],
            "camera_count": 2,
            "camera_numbers": [1, 2],
            "camera_subfolders": ["Camera1", "Camera2"],
        },
        "images": {
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "num_images": 2,
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    return Config(path=str(path))


def test_an_exported_run_is_compared_note_to_note(tmp_path):
    source = _calibration(tmp_path)
    run_notes = {
        cam: _note(dark_id=RUN_DARK_ID, camera_serial=f"SIM{cam}") for cam in (1, 2)
    }
    assert len(_plan(_exported_config(tmp_path / "a", run_notes), source)) == 2

    run_notes[2] = _note(
        dark_id=RUN_DARK_ID,
        camera_serial="SIM2",
        pixel_rule="bayer_rggb_superpixel_sum",
    )
    with pytest.raises(
        CalibrationFrameMismatch,
        match="camera 2, 00001.tif: pixel_rule mono12p in its note, "
        "bayer_rggb_superpixel_sum in the run",
    ):
        _plan(_exported_config(tmp_path / "b", run_notes), source)


def test_an_exported_run_and_a_plain_calibration_are_refused(tmp_path):
    source = _calibration(tmp_path, {(c, n): None for c in (1, 2) for n in (1, 2)})
    run_notes = {cam: _note(camera_serial=f"SIM{cam}") for cam in (1, 2)}
    with pytest.raises(CalibrationFrameMismatch, match="TIFFs exported by PIVTOOLS"):
        _plan(_exported_config(tmp_path / "a", run_notes), source)


# ---------------------------------------------------------------------------
# DaVis .set against .set: each camera's geometry transformer
# ---------------------------------------------------------------------------


def _set_with_geometry(root: Path, name: str, n_streams: int, geometry):
    """A one-camera .set of ``n_streams`` (4, 6) streams, each turned by ``geometry``."""
    set_file = build_raw16_set(root, [(4, 6)] * n_streams, name=name)
    write_manifest(
        set_file.with_suffix(""),
        frame_readers=[(f"Frame{k}", k) for k in range(n_streams)],
        transformers=(
            [(geometry, f"Frame{k}", k) for k in range(n_streams)] if geometry else ()
        ),
    )
    return set_file


def _set_piv_config(root: Path, piv_set: Path, cameras: int = 1) -> Config:
    cfg = make_set_config(root, piv_set, cameras, list(range(1, cameras + 1)))
    cfg.data["paths"]["base_paths"] = [str(root / "results")]
    return cfg


def _davis_plan(tmp_path: Path, cal_geometry, piv_geometry):
    piv_set = _set_with_geometry(tmp_path, "piv", 2, piv_geometry)
    cal_set = _set_with_geometry(tmp_path, "cal", 1, cal_geometry)
    save_settings(
        cal_set,
        {"image": {"image_format": cal_set.name, "image_type": "lavision_set"}},
    )
    REC.save_mono(
        build_scale_factor_record(
            camera=1, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
        ),
        REC.mono_model_dir_for_source(cal_set, 1, BOARD),
    )
    return _plan(_set_piv_config(tmp_path, piv_set), cal_set)


def test_davis_matching_geometry_applies(tmp_path):
    (unit,) = _davis_plan(tmp_path, "rotate-cw-90", "rotate-cw-90")
    assert unit["frame_warnings"] == (_IDENTITY_NOT_COMPARED,)


# Neither recording carries camera labels, so camera k is taken as camera k.
_IDENTITY_NOT_COMPARED = (
    "The calibration recording or the PIV recording carries no camera labels, so "
    "camera k of the calibration is taken to be camera k of the PIV recording "
    "without checking that they are the same physical camera."
)

# A PIV + PLIF rig scaled down: two double-frame cameras (serials 11, 12) and
# two single-frame cameras (31, 32), as test_set_camera_identity builds it.
_PIV_RIG_KEYS = [
    ("Camera 1", "Camera.X: 11", "1A: X"),
    ("Camera 1", "Camera.X: 11", "1B: X"),
    ("Camera 2", "Camera.X: 12", "2A: X"),
    ("Camera 2", "Camera.X: 12", "2B: X"),
    ("Camera 3", "Camera.S: 31", "3: S"),
    ("Camera 4", "Camera.S: 32", "4: S"),
]


def _labelled_calibration(tmp_path: Path, keys) -> Path:
    """A labelled DaVis calibration recording, one single-frame stream per
    camera, with a scale-factor model saved for each."""
    cal_set = build_labelled_set(tmp_path, keys, name="cal")
    save_settings(
        cal_set,
        {"image": {"image_format": cal_set.name, "image_type": "lavision_set"}},
    )
    for cam in range(1, len(keys) + 1):
        REC.save_mono(
            build_scale_factor_record(
                camera=cam, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
            ),
            REC.mono_model_dir_for_source(cal_set, cam, BOARD),
        )
    return cal_set


def test_a_labelled_calibration_is_split_by_its_own_cameras(tmp_path):
    """The calibration holds 2 cameras, the PIV recording 4: the calibration is
    split by its own labels, and cameras 1 and 2 are the same serials."""
    piv_set = build_labelled_set(tmp_path, _PIV_RIG_KEYS, name="piv")
    cal_set = _labelled_calibration(
        tmp_path,
        [("Camera 1", "Camera.X: 11", "1: X"), ("Camera 2", "Camera.X: 12", "2: X")],
    )
    cfg = make_set_config(tmp_path, piv_set, 4, [1, 2])
    cfg.data["paths"]["base_paths"] = [str(tmp_path / "results")]
    units = _plan(cfg, cal_set)
    assert len(units) == 2
    assert all(u["frame_warnings"] == () for u in units)


def test_swapped_camera_serials_are_refused_by_name(tmp_path):
    piv_set = build_labelled_set(tmp_path, _PIV_RIG_KEYS[:4], name="piv")
    cal_set = _labelled_calibration(
        tmp_path,
        [("Camera 1", "Camera.X: 12", "1: X"), ("Camera 2", "Camera.X: 11", "2: X")],
    )
    cfg = make_set_config(tmp_path, piv_set, 2, [1, 2])
    cfg.data["paths"]["base_paths"] = [str(tmp_path / "results")]
    with pytest.raises(CalibrationFrameMismatch) as exc:
        _plan(cfg, cal_set)
    message = str(exc.value)
    for words in [
        "camera 1: the calibration recording's camera 1 is Camera.X: 12, the PIV "
        "recording's camera 1 is Camera.X: 11",
        "camera 2: the calibration recording's camera 2 is Camera.X: 11, the PIV "
        "recording's camera 2 is Camera.X: 12",
    ]:
        assert words in message, message


@pytest.mark.parametrize(
    "cal, piv, words",
    [
        ("rotate-cw-90", None, "rotate-cw-90 in the calibration recording, none in"),
        (None, "rotate-180", "none in the calibration recording, rotate-180 in"),
        ("rotate-ccw-90", "rotate-cw-90", "rotate-ccw-90 in the calibration"),
    ],
)
def test_davis_geometry_mismatch_is_refused_by_name(tmp_path, cal, piv, words):
    with pytest.raises(CalibrationFrameMismatch) as exc:
        _davis_plan(tmp_path, cal, piv)
    assert f"camera 1: geometry transformer {words}" in str(exc.value)


# ---------------------------------------------------------------------------
# Kinds that differ
# ---------------------------------------------------------------------------


def test_an_app_calibration_against_a_davis_recording_is_refused(tmp_path):
    # Two A/B cameras: 4 equal streams split uniquely as (0, 1), (2, 3).
    piv_set = _set_with_geometry(tmp_path, "piv", 4, None)
    source = _calibration(tmp_path)
    with pytest.raises(CalibrationFrameMismatch) as exc:
        _plan(_set_piv_config(tmp_path, piv_set, cameras=2), source)
    message = str(exc.value)
    assert "images saved by PIVTOOLS Acquisition" in message
    assert "a DaVis .set recording" in message and "cannot be proven" in message


# ---------------------------------------------------------------------------
# Note files are not images
# ---------------------------------------------------------------------------


def _real_tiffs(folder: Path, n: int, folder_note: bool = False) -> None:
    """``n`` readable 16-bit TIFFs, each with its note, plus a folder note if asked."""
    import cv2
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        cv2.imwrite(str(folder / f"{i:05d}.tif"), np.full((8, 12), i, np.uint16))
        (folder / f"{i:05d}.tif.json").write_text(json.dumps(_note()), encoding="utf-8")
    if folder_note:
        (folder / "image_note.json").write_text(json.dumps(_note()), encoding="utf-8")


def test_calibration_counts_and_validation_ignore_note_files(tmp_path):
    source = tmp_path / "calibration"
    for cam in (1, 2):
        _real_tiffs(source / f"Cam{cam}", 3)
    save_settings(
        source, {"image": {"image_format": "%05d.tif", "image_type": "standard"}}
    )
    cfg = _CalConfig(source)
    assert tuple(get_calibration_frame_count(1, cfg, 0)) == (3, "files")
    result = validate_calibration_images(1, cfg, 0)
    assert result["valid"] is True, result
    assert result["found_count"] == 3
    assert ".json" not in json.dumps(result.get("sample_files", []))
    assert "json" not in (result.get("found") or "")
    assert not result.get("suggestions")


@pytest.mark.parametrize("picked", ["", "Camera1.pivraw"])
def test_calibration_diagnosis_never_suggests_pivraw(tmp_path, picked):
    """A run folder (or one of its files) picked as a calibration source gets no
    'Switch to .pivraw' fix: that type is refused for calibration (T9). The same
    folder on the Setup page still gets it."""
    from fixtures.diagnosis import make_ctx

    from pivtools_core.image_handling.source_diagnosis import diagnose, scan_source

    run = copy_case(tmp_path, DF_CLEAN)
    path = run / picked if picked else run

    def suggestions(calibration: bool):
        ctx = make_ctx(
            path,
            image_type="standard",
            image_format=("%05d.tif",),
            calibration=calibration,
            pairing_preset=None if calibration else "skip_frames",
        )
        found = diagnose(
            scan_source(path), ctx, 1, run, None, False, detected_start_index=None
        )
        return [s["apply"].get("image_type") for s in found]

    assert "pivraw" not in suggestions(calibration=True)
    assert "pivraw" in suggestions(calibration=False)


def test_source_diagnosis_does_not_list_note_files(tmp_path):
    from pivtools_core.image_handling.source_diagnosis import scan_source

    _real_tiffs(tmp_path / "calibration" / "Cam1", 2)
    _real_tiffs(tmp_path / "export" / "Camera1", 2, folder_note=True)

    scan = scan_source(tmp_path / "calibration" / "Cam1")
    assert scan.root_names == ("00001.tif", "00002.tif")
    assert dict(scan.root_ext_counts) == {".tif": 2}

    (camera1,) = scan_source(tmp_path / "export").subdirs
    assert dict(camera1.ext_counts) == {".tif": 2}


# ---------------------------------------------------------------------------
# Applied without a comparison: the warning reaches the CLI and the GUI result
# ---------------------------------------------------------------------------

_NOT_COMPARED = (
    "The calibration (images with no frame record) and the PIV images (DaVis .set "
    "recording) carry no shared record of rotation or dark, so their pixel frames "
    "could not be compared."
)


def _plain_one_camera_calibration(tmp_path: Path) -> Path:
    """One camera's TIFF calibration without notes (a DaVis user's export, say)."""
    source = tmp_path / "plain_cal"
    source.mkdir()
    (source / "00001.tif").write_bytes(b"tif")
    save_settings(
        source, {"image": {"image_format": "%05d.tif", "image_type": "standard"}}
    )
    REC.save_mono(
        build_scale_factor_record(
            camera=1, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
        ),
        REC.mono_model_dir_for_source(source, 1, BOARD),
    )
    return source


def test_davis_against_plain_applies_with_a_warning_both_ways(tmp_path):
    piv_set = _set_with_geometry(tmp_path, "piv", 2, "rotate-cw-90")
    (unit,) = _plan(
        _set_piv_config(tmp_path, piv_set), _plain_one_camera_calibration(tmp_path)
    )
    assert unit["frame_warnings"] == (_NOT_COMPARED,)

    # The mirror: a .set calibration against PIV TIFFs without notes.
    cal_set = _set_with_geometry(tmp_path, "cal", 1, None)
    save_settings(
        cal_set,
        {"image": {"image_format": cal_set.name, "image_type": "lavision_set"}},
    )
    REC.save_mono(
        build_scale_factor_record(
            camera=1, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
        ),
        REC.mono_model_dir_for_source(cal_set, 1, BOARD),
    )
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "results")],
            "source_paths": [str(tmp_path / "tiffs")],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {"image_format": ["B%05d_A.tif", "B%05d_B.tif"], "num_images": 2},
    }
    path = tmp_path / "tiff_config.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    (unit,) = _plan(Config(path=str(path)), cal_set)
    assert unit["frame_warnings"] == (
        "The calibration (DaVis .set recording) and the PIV images (images with no "
        "frame record) carry no shared record of rotation or dark, so their pixel "
        "frames could not be compared.",
    )


def test_matched_frames_carry_no_warning(tmp_path):
    run = copy_case(tmp_path, DF_DARK)
    units = _plan(_piv_config(tmp_path, run), _calibration(tmp_path))
    assert all(u["frame_warnings"] == () for u in units)


def test_the_warning_reaches_the_cli_output(tmp_path, monkeypatch, capsys):
    import pivtools_cli.calibration_cli as cli

    piv_set = _set_with_geometry(tmp_path, "piv", 2, None)
    source = _plain_one_camera_calibration(tmp_path)
    monkeypatch.setattr(cli, "get_config", lambda: _set_piv_config(tmp_path, piv_set))
    monkeypatch.setattr(cli.runio, "calibrate_mono_run", lambda *a, **k: ["B1.mat"])
    monkeypatch.setattr(cli.runio, "stamp_unit", lambda unit, pointer: None)
    assert cli.apply_calibration_command(_cli_apply_args(source)) == 1
    assert f"[calibration] WARNING: {_NOT_COMPARED}" in capsys.readouterr().out


def test_the_warning_reaches_the_gui_apply_response(tmp_path, monkeypatch):
    import pivtools_gui.calibration.app.views as V

    piv_set = _set_with_geometry(tmp_path, "piv", 2, None)
    source = _plain_one_camera_calibration(tmp_path)
    monkeypatch.setattr(V, "get_config", lambda: _set_piv_config(tmp_path, piv_set))
    monkeypatch.setattr(V, "_source_path", lambda idx: source)
    monkeypatch.setattr(V.c2runio, "calibrate_mono_run", lambda *a, **k: ["B1.mat"])
    monkeypatch.setattr(V.c2runio, "stamp_unit", lambda unit, pointer: None)
    client = _client(V)
    resp = client.post("/calibration/apply", json={"board": BOARD, "stereo": False})
    body = resp.get_json()
    assert resp.status_code == 200 and body["success"] is True, body
    assert body["warnings"] == [_NOT_COMPARED]
    status = _wait_for_job(client, f"/calibration/apply/status/{body['job_id']}")
    assert status["status"] == "completed", status
    assert status["warnings"] == [_NOT_COMPARED]


# ---------------------------------------------------------------------------
# A .pivraw run the reader refuses: 400 (GUI) or exit 1 (CLI), never a job
# ---------------------------------------------------------------------------

# Camera 2 missed its first trigger: the alignment refuses the run (P2).
REFUSED_RUN = "df_2cam_missed_trigger_cam2"


def _client(V):
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client()


def _wait_for_job(client, url: str, timeout_s: float = 20.0) -> dict:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(url).get_json()
        if status.get("status") in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError(f"job at {url} did not finish in {timeout_s} s")


def _cli_apply_args(source: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        board=BOARD,
        source=str(source),
        camera=None,
        dt=None,
        z_world=None,
        tilt_x=None,
        tilt_y=None,
        uncalibrated_dir=None,
        calibrated_dir=None,
        all_paths=True,
        type_name=None,
        model_type=None,
    )


def _reader_refusal(cfg: Config) -> str:
    """The .pivraw reader's own text for the refused run."""
    from pivtools_core.image_handling.readers.pivraw_reader import PivrawError

    with pytest.raises(PivrawError) as exc:
        cfg.num_frame_pairs
    return str(exc.value)


def _jobs(job_type: str) -> int:
    from pivtools_gui.services.job_manager import job_manager

    return len(job_manager.list_jobs(job_type=job_type))


def test_gui_apply_answers_a_refused_run_with_400(tmp_path, monkeypatch):
    import pivtools_gui.calibration.app.views as V

    run = copy_case(tmp_path, REFUSED_RUN)
    refusal = _reader_refusal(_piv_config(tmp_path, run))
    source = _calibration(tmp_path)
    monkeypatch.setattr(V, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(V, "_source_path", lambda idx: source)
    before = _jobs("calibration_apply")
    resp = _client(V).post("/calibration/apply", json={"board": BOARD, "stereo": False})
    assert resp.status_code == 400
    assert resp.get_json() == {"success": False, "error": refusal}
    assert _jobs("calibration_apply") == before


def test_cli_apply_exits_1_with_the_reader_text(tmp_path, monkeypatch, capsys):
    import pivtools_cli.calibration_cli as cal_cli
    from pivtools_cli import cli

    run = copy_case(tmp_path, REFUSED_RUN)
    refusal = _reader_refusal(_piv_config(tmp_path, run))
    source = _calibration(tmp_path)
    monkeypatch.setattr(cal_cli, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(
        cal_cli.runio, "calibrate_mono_run", lambda *a, **k: pytest.fail("applied")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pivtools-cli",
            "apply-calibration",
            "--board",
            BOARD,
            "--source",
            str(source),
            "--all-paths",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert capsys.readouterr().err.strip() == f"error: {refusal}"


def _stereo_calibration(tmp_path: Path, notes=None) -> Path:
    """The app calibration of :func:`_calibration`, plus a pinhole stereo model."""
    from test_calibration_self_cal import _stereo_record

    source = _calibration(tmp_path, notes)
    REC.save_stereo(
        _stereo_record(), REC.stereo_model_dir_for_source(source, 1, 2, "dotboard")
    )
    return source


def _self_cal_body() -> dict:
    return {"board": "dotboard", "camera_pair": [1, 2], "base_path_idx": 0}


def test_gui_self_cal_answers_a_refused_run_with_400(tmp_path, monkeypatch):
    import pivtools_gui.calibration.app.views as V

    run = copy_case(tmp_path, REFUSED_RUN)
    refusal = _reader_refusal(_piv_config(tmp_path, run))
    source = _stereo_calibration(tmp_path)
    monkeypatch.setattr(V, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(V, "_source_path", lambda idx: source)
    before = _jobs("calibration_self_cal")
    resp = _client(V).post("/calibration/self_cal/run", json=_self_cal_body())
    assert resp.status_code == 400
    assert resp.get_json() == {"success": False, "error": refusal}
    assert _jobs("calibration_self_cal") == before


def _cli_self_cal_args(source: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        board="dotboard",
        source=str(source),
        camera_pair="1,2",
        base_path_idx=0,
        n_images=None,
        window_size=None,
        overlap=None,
        no_filters=True,
        no_figures=True,
        model_type=None,
    )


def test_cli_self_calibrate_refuses_a_refused_run(tmp_path, monkeypatch):
    import pivtools_cli.calibration_cli as cal_cli
    from pivtools_core.image_handling.readers.pivraw_reader import PivrawError

    run = copy_case(tmp_path, REFUSED_RUN)
    refusal = _reader_refusal(_piv_config(tmp_path, run))
    source = _stereo_calibration(tmp_path)
    monkeypatch.setattr(cal_cli, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(
        cal_cli.c2sc, "load_particle_pairs", lambda *a, **k: pytest.fail("loaded")
    )
    with pytest.raises(PivrawError) as exc:
        cal_cli.self_calibrate_command(_cli_self_cal_args(source))
    assert str(exc.value) == refusal


# ---------------------------------------------------------------------------
# Stereo self-calibration runs the same frame check
# ---------------------------------------------------------------------------


def test_gui_self_cal_refuses_a_frame_mismatch_before_any_job(tmp_path, monkeypatch):
    import pivtools_gui.calibration.app.views as V

    run = copy_case(tmp_path, DF_DARK)
    source = _stereo_calibration(tmp_path, {(2, 1): _note(view_rotation="cw90")})
    monkeypatch.setattr(V, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(V, "_source_path", lambda idx: source)
    before = _jobs("calibration_self_cal")
    resp = _client(V).post("/calibration/self_cal/run", json=_self_cal_body())
    assert resp.status_code == 400
    assert "camera 2, 00001.tif: view_rotation cw90 in its note, none in the run" in (
        resp.get_json()["error"]
    )
    assert _jobs("calibration_self_cal") == before


def test_cli_self_calibrate_refuses_a_frame_mismatch(tmp_path, monkeypatch):
    import pivtools_cli.calibration_cli as cal_cli

    run = copy_case(tmp_path, DF_DARK)
    source = _stereo_calibration(tmp_path, {(1, 2): _note(camera_serial="SIM2")})
    monkeypatch.setattr(cal_cli, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(
        cal_cli.c2sc, "load_particle_pairs", lambda *a, **k: pytest.fail("loaded")
    )
    with pytest.raises(CalibrationFrameMismatch, match="camera 1, 00002.tif: "):
        cal_cli.self_calibrate_command(_cli_self_cal_args(source))


def test_self_cal_with_matching_frames_reaches_the_image_load(tmp_path, monkeypatch):
    """The check passes matching frames on to the particle-image load."""
    import pivtools_cli.calibration_cli as cal_cli

    class _Loaded(Exception):
        pass

    def _load(*a, **k):
        raise _Loaded

    run = copy_case(tmp_path, DF_DARK)
    source = _stereo_calibration(tmp_path)
    monkeypatch.setattr(cal_cli, "get_config", lambda: _piv_config(tmp_path, run))
    monkeypatch.setattr(cal_cli.c2sc, "load_particle_pairs", _load)
    with pytest.raises(_Loaded):
        cal_cli.self_calibrate_command(_cli_self_cal_args(source))
