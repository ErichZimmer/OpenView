#!/usr/bin/env python3
"""
test_pivraw_reader.py

The ``pivraw`` image type: PIVTOOLS Acquisition run containers, one
``Camera{k}.pivraw`` (+ ``.json`` sidecar, ``.idx`` records) per camera, read
from the committed contract fixtures (``unit-tests/fixtures/pivraw_contract``),
which the real recorder wrote.

Locks in:
  * The reader's format constants equal the writer's MANIFEST, and the fixture
    copy is byte-exact.
  * T1: every fixture frame reads bit-identical to the acquisition package's
    own decode; Mono12p also equals an independent byte-plane decode past the
    zero padding, BayerRG8 the 2x2 superpixel sum R+G+G+B.
  * The ``.idx`` is read as version 2 (host_ns, frame_id, camera_ns).
  * T3: every strictness refusal raises PivrawError naming the field (missing
    or unknown field, field type, format_version other than the integer 2,
    slot against the file name, a foreign ``.idx``, blanked missing_indices).
  * Pairs honour ``frame_b_idx`` and ``out=``; a double-frame run refuses any
    pair but its recorded ``(2k-1, 2k)``.
  * read_pair / _read_batch through a Config, count validation and source
    diagnosis know the type. Calibration refuses it
    (``test_calibration_frame_match.py``, T9).

The cases with drops, offsets or a lost first frame are read frame by frame
only here: pairing them by trigger is ticket P2.

Usage:
    pytest unit-tests/test_pivraw_reader.py -v
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.diagnosis import _inp, make_ctx, piv_route_response  # noqa: E402
from fixtures.pivraw import (  # noqa: E402
    CONTRACT_DIR,
    DROP,
    contract_cases,
    contract_manifest,
    copy_case,
    edit_sidecar,
    expected_frames,
    write_piv_config,
)

from pivtools_core.config import Config  # noqa: E402
from pivtools_core.image_handling.load_images import (  # noqa: E402
    _read_batch,
    read_pair,
    read_single_frame,
)
from pivtools_core.image_handling.path_utils import (  # noqa: E402
    camera_subfolder_name,
    infer_image_type,
    validate_images_generic,
)
from pivtools_core.image_handling.readers import (
    get_reader,  # noqa: E402
    pivraw_reader,  # noqa: E402
)
from pivtools_core.image_handling.readers.pivraw_reader import (  # noqa: E402
    PivrawError,
    clear_header_cache,
    get_pivraw_frame_count,
    get_pivraw_image_shape,
    read_pivraw_header,
    read_pivraw_pair,
    read_pivraw_single,
)
from pivtools_core.image_handling.source_diagnosis import (  # noqa: E402
    pivraw_pattern,
    rule_picked_file,
    rule_wrong_type,
    scan_source,
)
from pivtools_core.validation import validate_config  # noqa: E402

# Double-frame Mono12p, 2 cameras x 4 frames of 16 x 8 (2 pairs), no drops.
DF_CLEAN = "df_2cam_clean"
# Time-resolved BayerRG8, 1 camera x 3 frames of 8 x 8 sensor (4 x 4 decoded).
TR_BAYER = "tr_bayer_clean"
DF_SHAPE = (8, 16)

CASES = contract_cases()


@pytest.fixture(autouse=True)
def _clean_header_cache():
    clear_header_cache()
    yield
    clear_header_cache()


# ---------------------------------------------------------------------------
# T1: the contract fixtures read bit-identical
# ---------------------------------------------------------------------------


def test_reader_constants_match_the_writer_manifest():
    """The writer's MANIFEST is the contract: a regenerated copy that adds a
    header field or changes the ``.idx`` record fails here until the reader
    follows."""
    manifest = contract_manifest()
    assert manifest["format_version"] == pivraw_reader.FORMAT_VERSION
    assert manifest["header_fields"] == list(pivraw_reader.REQUIRED_FIELDS)
    assert manifest["idx_record"] == "<qQq"
    assert manifest["idx_fields"] == list(pivraw_reader.INDEX_DTYPE.names)
    assert pivraw_reader.INDEX_DTYPE == np.dtype(
        [("host_ns", "<i8"), ("frame_id", "<u8"), ("camera_ns", "<i8")]
    )
    assert manifest["stride_align_bytes"] == pivraw_reader.STRIDE_ALIGNMENT


def test_fixture_copy_is_byte_exact():
    """Every file has the size the writer recorded: catches a line-ending
    conversion or an edit of the committed copy (see .gitattributes)."""
    sizes = contract_manifest()["files"]
    assert sizes, "MANIFEST.json lists no files"
    for rel, size in sizes.items():
        assert (CONTRACT_DIR / rel).stat().st_size == size, rel


_CAMERA_FRAMES = [
    (case, int(cam))
    for case, spec in CASES.items()
    for cam, facts in spec["cameras"].items()
    if facts["n_frames"] > 0
]
# A run with an in-run dark reads dark-subtracted (test_pivraw_dark_rotation.py).
_RAW_CAMERA_FRAMES = [(c, k) for c, k in _CAMERA_FRAMES if "dark" not in CASES[c]]


@pytest.mark.parametrize("case, camera", _RAW_CAMERA_FRAMES)
def test_every_contract_frame_reads_bit_identical(case, camera):
    """Every frame of a run without a dark equals the acquisition package's own
    decode (``expected_arrays.npz``).

    Includes the cases whose pairing P2 decides (drops, offsets, a lost first
    frame): here they are read frame by frame only.
    """
    path = CONTRACT_DIR / case / f"Camera{camera}.pivraw"
    expected = expected_frames(case, camera)
    facts = CASES[case]["cameras"][str(camera)]
    assert expected.shape == (facts["n_frames"], *facts["decoded_shape"])
    for k in range(expected.shape[0]):
        got = read_pivraw_single(path, k + 1)
        assert got.dtype == np.float32 and got.shape == expected.shape[1:]
        np.testing.assert_array_equal(got, expected[k].astype(np.float32))


def _decode_mono12p_byte_planes(raw: bytes, width: int, height: int) -> np.ndarray:
    """Reference Mono12p decode from the three byte planes (LSB-first)."""
    b = np.frombuffer(raw, dtype=np.uint8, count=width * height * 3 // 2)
    b0 = b[0::3].astype(np.uint16)
    b1 = b[1::3].astype(np.uint16)
    b2 = b[2::3].astype(np.uint16)
    out = np.empty(width * height, dtype=np.uint16)
    out[0::2] = b0 | ((b1 & 0x0F) << 8)
    out[1::2] = (b1 >> 4) | (b2 << 4)
    return out.reshape(height, width)


def test_mono12p_matches_byte_plane_reference_past_the_padding():
    """16 x 8 Mono12p is 192 payload bytes in a 4096-byte stride."""
    path = CONTRACT_DIR / DF_CLEAN / "Camera1.pivraw"
    payload, stride = 16 * 8 * 3 // 2, 4096
    blob = path.read_bytes()
    assert len(blob) == 4 * stride
    for k in range(4):
        raw = blob[k * stride : k * stride + payload]
        np.testing.assert_array_equal(
            read_pivraw_single(path, k + 1), _decode_mono12p_byte_planes(raw, 16, 8)
        )
        assert not any(blob[k * stride + payload : (k + 1) * stride])


def test_bayer_superpixel_matches_numpy_reference():
    path = CONTRACT_DIR / TR_BAYER / "Camera1.pivraw"
    blob = path.read_bytes()
    for k in range(3):
        mosaic = np.frombuffer(blob, np.uint8, 64, k * 4096).reshape(8, 8)
        m = mosaic.astype(np.float32)
        reference = m[0::2, 0::2] + m[0::2, 1::2] + m[1::2, 0::2] + m[1::2, 1::2]
        np.testing.assert_array_equal(read_pivraw_single(path, k + 1), reference)
    assert get_pivraw_image_shape(path) == (4, 4)


@pytest.mark.parametrize("case, camera", _CAMERA_FRAMES)
def test_index_is_read_as_version_2(case, camera):
    """The header carries the ``.idx``: frame ids as the writer recorded them,
    and a camera clock that advances with the host clock."""
    header = read_pivraw_header(CONTRACT_DIR / case / f"Camera{camera}.pivraw")
    facts = CASES[case]["cameras"][str(camera)]
    assert header.index.dtype.names == ("host_ns", "frame_id", "camera_ns")
    assert header.index["frame_id"].tolist() == facts["idx_frame_ids"]
    assert header.first_frame_id == facts["first_frame_id"]
    assert header.slot == camera
    assert np.all(np.diff(header.index["camera_ns"]) > 0)
    with pytest.raises(ValueError, match="read-only"):
        header.index["frame_id"][0] = 0


# ---------------------------------------------------------------------------
# T3: strict sidecar and index, each refusal named
# ---------------------------------------------------------------------------


def _idx_records(path: Path) -> np.ndarray:
    """The ``.idx`` of ``path`` decoded independently of the reader (``<qQq``)."""
    dtype = np.dtype([("host_ns", "<i8"), ("frame_id", "<u8"), ("camera_ns", "<i8")])
    return np.fromfile(Path(str(path) + ".idx"), dtype=dtype).copy()


def _write_idx(path: Path, records: np.ndarray) -> None:
    Path(str(path) + ".idx").write_bytes(records.tobytes())


def _v1_index(run: Path) -> Path:
    """A version-1 index: 16-byte ``<qQ`` records (host_ns, frame_id), same count."""
    path = run / "Camera1.pivraw"
    rec = _idx_records(path)
    v1 = np.empty(rec.size, dtype=[("host_ns", "<i8"), ("frame_id", "<u8")])
    v1["host_ns"], v1["frame_id"] = rec["host_ns"], rec["frame_id"]
    Path(str(path) + ".idx").write_bytes(v1.tobytes())
    return path


def _extra_record(run: Path) -> Path:
    path = run / "Camera1.pivraw"
    rec = _idx_records(path)
    _write_idx(path, np.concatenate([rec, rec[-1:]]))
    return path


def _other_cameras_index(run: Path) -> Path:
    """Camera 2's index beside camera 1's data: right size, foreign ids."""
    path = run / "Camera1.pivraw"
    Path(str(path) + ".idx").write_bytes((run / "Camera2.pivraw.idx").read_bytes())
    return path


def _host_time_backwards(run: Path) -> Path:
    path = run / "Camera1.pivraw"
    rec = _idx_records(path)
    rec["host_ns"][2] = rec["host_ns"][1] - 1
    _write_idx(path, rec)
    return path


def _renamed_camera(run: Path) -> Path:
    """Camera 1's three files renamed to Camera3: the sidecar still says slot 1."""
    for suffix in ("", ".json", ".idx"):
        (run / f"Camera1.pivraw{suffix}").rename(run / f"Camera3.pivraw{suffix}")
    return run / "Camera3.pivraw"


def _sidecar(case="df_2cam_clean", **changes):
    return case, lambda run: edit_sidecar(run, 1, **changes)


# (case, mutate(run) -> camera path, message the PivrawError must contain)
_REFUSALS = {
    "missing first_frame_id": (
        *_sidecar(first_frame_id=DROP),
        r"missing required field\(s\) first_frame_id",
    ),
    "unknown field": (
        *_sidecar(anchor_frame_id=999),
        r"unknown field\(s\) anchor_frame_id",
    ),
    "format_version 2.0": (*_sidecar(format_version=2.0), "format_version 2.0"),
    "format_version 1": (*_sidecar(format_version=1), "format_version 1"),
    "format_version true": (*_sidecar(format_version=True), "format_version True"),
    "format_version '2'": (*_sidecar(format_version="2"), "format_version '2'"),
    "slot differs from the file name": (*_sidecar(slot=2), "slot 2"),
    "slot not an int": (*_sidecar(slot="1"), "slot must be an integer"),
    "file name not Camera<k>": ("df_2cam_clean", _renamed_camera, "slot 1"),
    "camera_model not a string": (*_sidecar(camera_model=7), "camera_model"),
    "created_at not a string": (*_sidecar(created_at=None), "created_at"),
    "first_frame_id null with frames": (
        *_sidecar(first_frame_id=None),
        "first_frame_id is null",
    ),
    "first_frame_id not an int": (*_sidecar(first_frame_id=1000.0), "first_frame_id"),
    "version-1 .idx": ("df_2cam_clean", _v1_index, r"\.idx: holds 64 bytes"),
    ".idx count mismatch": ("df_2cam_clean", _extra_record, r"\.idx: holds 120 bytes"),
    "another camera's .idx": (
        "df_2cam_clean",
        _other_cameras_index,
        "missing_indices and the .idx do not partition",
    ),
    "ids before first_frame_id": (
        *_sidecar(first_frame_id=1001),
        "frame ids are not strictly increasing from first_frame_id",
    ),
    "B in an A position": (
        *_sidecar(first_frame_id=999),
        r"positions \(2k, 2k\+1\) are not frames A, B",
    ),
    "host time backwards": (
        "df_2cam_clean",
        _host_time_backwards,
        "host_ns goes backwards",
    ),
    "blanked missing_indices": (
        *_sidecar("df_2cam_unequal_drops", missing_indices=[]),
        "missing_indices and the .idx do not partition",
    ),
    "missing_indices overlapping the file": (
        *_sidecar("df_2cam_unequal_drops", missing_indices=[1, 2, 3]),
        "missing_indices and the .idx do not partition",
    ),
}


@pytest.mark.parametrize("what", list(_REFUSALS))
def test_strict_refusal_names_the_field(tmp_path, what):
    case, mutate, message = _REFUSALS[what]
    path = mutate(copy_case(tmp_path, case))
    with pytest.raises(PivrawError, match=message):
        read_pivraw_single(path, 1)


def test_empty_camera_is_a_valid_file(tmp_path):
    """A finalised camera that received nothing: n_frames 0, first_frame_id null.

    The file is as the writer defines it; refusing the run is P2's alignment
    ("camera holds no frames"), not the per-file reader's.
    """
    path = CONTRACT_DIR / "df_2cam_cam2_empty" / "Camera2.pivraw"
    assert get_pivraw_frame_count(path) == 0
    with pytest.raises(PivrawError, match="out of range"):
        read_pivraw_single(path, 1)


@pytest.mark.parametrize(
    "changes",
    [{"finalised": False}, {"n_frames": None}, {"finalised": False, "n_frames": None}],
)
def test_unfinalised_run_names_the_recover_command(tmp_path, changes):
    run = copy_case(tmp_path, DF_CLEAN)
    path = edit_sidecar(run, 1, **changes)
    with pytest.raises(PivrawError) as exc:
        get_pivraw_frame_count(path)
    assert "run not finalised (recorder stopped early?): run " in str(exc.value)
    assert f"`pivtools-acq recover {run}`" in str(exc.value)


@pytest.mark.parametrize(
    "field", ["dt_s", "pixel_format", "missing_indices", "finalised"]
)
def test_missing_field_raises_by_name(tmp_path, field):
    path = edit_sidecar(copy_case(tmp_path, DF_CLEAN), 1, **{field: DROP})
    with pytest.raises(PivrawError, match=f"missing required field\\(s\\) {field}"):
        read_pivraw_single(path, 1)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"format": "cine"}, "format is 'cine'"),
        ({"bytes_per_frame": 37}, "bytes_per_frame 37 does not match"),
        ({"frame_stride": 8192}, "frame_stride 8192"),
        ({"mode": "burst"}, "mode 'burst'"),
        ({"pixel_format": "Mono16"}, "pixel_format 'Mono16'"),
        ({"n_frames": 3}, "odd"),
        ({"n_frames": 6}, "data file and its sidecar disagree"),
        ({"dt_s": float("nan")}, "dt_s must be a finite number"),
    ],
)
def test_inconsistent_sidecar_raises(tmp_path, changes, message):
    path = edit_sidecar(copy_case(tmp_path, DF_CLEAN), 1, **changes)
    with pytest.raises(PivrawError, match=message):
        get_pivraw_frame_count(path)


def test_truncated_data_and_missing_index_raise(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    path = run / "Camera1.pivraw"
    Path(str(path) + ".idx").unlink()
    with pytest.raises(PivrawError, match="per-frame index .* is missing"):
        get_pivraw_frame_count(path)

    path = run / "Camera2.pivraw"
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(PivrawError, match="disagree"):
        get_pivraw_frame_count(path)


def test_missing_sidecar_is_a_pivraw_error(tmp_path):
    """An incomplete run folder, refused by name like every other run fault."""
    path = copy_case(tmp_path, DF_CLEAN) / "Camera1.pivraw"
    Path(str(path) + ".json").unlink()
    with pytest.raises(PivrawError, match="PIVRAW sidecar not found"):
        get_pivraw_frame_count(path)


def test_header_cache_revalidates_after_rewrite(tmp_path):
    """A recover truncates the data and rewrites the sidecar; the cache must notice.

    Stand-in: camera 1 of a 2-frame run copied over camera 1 of a 4-frame run.
    """
    run = copy_case(tmp_path, DF_CLEAN)
    path = run / "Camera1.pivraw"
    assert get_pivraw_frame_count(path) == 4
    for suffix in ("", ".json", ".idx"):
        name = f"Camera1.pivraw{suffix}"
        shutil.copyfile(
            CONTRACT_DIR / "df_2cam_lost_first_pair_cam1" / name, run / name
        )
    assert get_pivraw_frame_count(path) == 2


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def test_double_frame_pair_reads_recorded_pairs():
    pair = read_pivraw_pair(CONTRACT_DIR / DF_CLEAN / "Camera1.pivraw", idx=3)
    assert pair.shape == (2, *DF_SHAPE) and pair.dtype == np.float32
    np.testing.assert_array_equal(pair, expected_frames(DF_CLEAN, 1)[2:4])


@pytest.mark.parametrize("a, b", [(2, 3), (1, 1), (1, 3), (3, 2)])
def test_double_frame_refuses_other_pairings(a, b):
    with pytest.raises(PivrawError, match="skip_frames pairing preset"):
        read_pivraw_pair(
            CONTRACT_DIR / DF_CLEAN / "Camera1.pivraw", idx=a, frame_b_idx=b
        )


def test_time_resolved_pair_honours_frame_b_idx():
    path = CONTRACT_DIR / TR_BAYER / "Camera1.pivraw"
    frames = expected_frames(TR_BAYER, 1)
    pair = read_pivraw_pair(path, idx=1, frame_b_idx=3)
    np.testing.assert_array_equal(pair, frames[[0, 2]])
    # None means A + 1.
    np.testing.assert_array_equal(read_pivraw_pair(path, idx=2)[1], frames[2])


def test_pair_out_is_written_in_place():
    path = CONTRACT_DIR / DF_CLEAN / "Camera1.pivraw"
    out = np.full((2, *DF_SHAPE), -1.0, dtype=np.float32)
    assert read_pivraw_pair(path, idx=1, out=out) is out
    np.testing.assert_array_equal(out, expected_frames(DF_CLEAN, 1)[0:2])

    single = np.empty(DF_SHAPE, dtype=np.float32)
    assert read_pivraw_single(path, 2, out=single) is single


@pytest.mark.parametrize(
    "bad", [np.empty((2, 8, 15), np.float32), np.empty((2, 8, 16), np.float64)]
)
def test_pair_out_contract_is_enforced(bad):
    with pytest.raises(ValueError, match="out has"):
        read_pivraw_pair(CONTRACT_DIR / DF_CLEAN / "Camera1.pivraw", idx=1, out=bad)


@pytest.mark.parametrize("idx", [0, 5])
def test_out_of_range_frame_raises(idx):
    with pytest.raises(PivrawError, match="out of range"):
        read_pivraw_single(CONTRACT_DIR / DF_CLEAN / "Camera1.pivraw", idx)


# ---------------------------------------------------------------------------
# Config, read_pair, read_batch
# ---------------------------------------------------------------------------


def test_type_detection_agrees_between_config_and_path_utils(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4)))
    cfg.data["images"].pop("image_type")
    assert cfg.image_type == "pivraw" == infer_image_type("Camera%d.pivraw")
    assert cfg.is_container_format and cfg.is_single_camera_container
    assert not cfg.is_multi_camera_container
    assert cfg.get_camera_folder(2) == ""
    assert (
        camera_subfolder_name(
            "pivraw", 2, 2, use_camera_subfolders=True, camera_subfolders=["Cam2"]
        )
        == ""
    )
    assert get_reader("Camera1.pivraw") is read_pivraw_pair


def test_shape_comes_from_sidecar_without_decoding(tmp_path, monkeypatch):
    run = copy_case(tmp_path, TR_BAYER)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 1, 3, "time_resolved")))

    def _no_decode(*_a, **_k):
        raise AssertionError("shape detection decoded pixels")

    monkeypatch.setattr(pivraw_reader, "_decode_into", _no_decode)
    assert cfg.image_shape_for(1) == (4, 4)


def test_read_pair_and_read_batch_through_config(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4)))
    assert cfg.num_frame_pairs == 2

    for camera in (1, 2):
        frames = expected_frames(DF_CLEAN, camera)
        for p in (1, 2):
            np.testing.assert_array_equal(
                read_pair(p, run, camera, cfg), frames[2 * p - 2 : 2 * p]
            )
        batch = _read_batch(1, 2, run, camera, cfg)
        assert batch.shape == (2, 2, *DF_SHAPE) and batch.dtype == np.float32
        np.testing.assert_array_equal(batch, frames.reshape(2, 2, *DF_SHAPE))


def test_read_pair_with_wrong_preset_fails_loudly(tmp_path):
    """time_resolved pairing on a double-frame run would correlate B with the next A.

    The trigger alignment refuses the preset before any frame is read.
    """
    run = copy_case(tmp_path, DF_CLEAN)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4, "time_resolved")))
    for pair in (1, 2):
        with pytest.raises(PivrawError, match="skip_frames"):
            read_pair(pair, run, 1, cfg)


def test_read_pair_passes_frame_b_through(tmp_path):
    """frame_stride 2 on a time-resolved run pairs (1, 3): B is not A + 1."""
    run = copy_case(tmp_path, TR_BAYER)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 1, 3, "time_resolved")))
    cfg.data["images"]["pairing_preset"] = "custom"
    cfg.data["images"]["frame_stride"] = 2
    assert cfg.get_frame_pair_indices(1, camera=1) == (1, 3)
    np.testing.assert_array_equal(
        read_pair(1, run, 1, cfg), expected_frames(TR_BAYER, 1)[[0, 2]]
    )


def test_ab_format_preset_on_double_frame_run_is_refused(tmp_path):
    """ab_format gives (A, B) = (1, 1); the reader must not quietly read (1, 2)."""
    run = copy_case(tmp_path, DF_CLEAN)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4, "ab_format")))
    with pytest.raises(PivrawError, match="skip_frames pairing preset"):
        cfg.get_frame_pair_indices(1, camera=1)
    with pytest.raises(PivrawError, match="skip_frames pairing preset"):
        read_pair(1, run, 1, cfg)


def test_time_resolved_run_through_config(tmp_path):
    run = copy_case(tmp_path, TR_BAYER)
    frames = expected_frames(TR_BAYER, 1)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 1, 3, "time_resolved")))
    assert cfg.num_frame_pairs == 2
    np.testing.assert_array_equal(read_pair(2, run, 1, cfg), frames[1:3])
    np.testing.assert_array_equal(
        read_single_frame(run / "Camera1.pivraw", 1, 3, "pivraw"), frames[2]
    )


def test_end_to_end_validate_then_read(tmp_path):
    """A user's config for a two-camera double-frame run validates, then reads."""
    run = copy_case(tmp_path, DF_CLEAN)
    config_path = write_piv_config(tmp_path, run, 2, 4)
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["base_paths"] = [str(tmp_path / "out")]
    config_path.write_text(yaml.dump(data))
    cfg = Config(path=str(config_path))

    ok, error, _warnings = validate_config(cfg)
    assert ok, error
    assert cfg.image_shape_for(2) == DF_SHAPE
    np.testing.assert_array_equal(
        read_pair(2, run, 2, cfg), expected_frames(DF_CLEAN, 2)[2:4]
    )


def test_gui_validate_route_accepts_a_run(tmp_path, monkeypatch):
    run = copy_case(tmp_path, DF_CLEAN)
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
    body = resp.get_json()
    assert body["valid"] is True, body


# ---------------------------------------------------------------------------
# Count validation and calibration counting
# ---------------------------------------------------------------------------


def _validate(run, expected_count, image_format="Camera%d.pivraw"):
    path = run / (image_format % 1 if "%" in image_format else image_format)
    return validate_images_generic(
        camera_path=run,
        camera=1,
        image_format=image_format,
        image_type="pivraw",
        expected_count=expected_count,
        zero_based_indexing=False,
        read_frame_fn=lambda idx: read_pivraw_single(path, idx),
        diagnosis=make_ctx(run, image_type="pivraw", image_format=(image_format,)),
        pair_strides=(1, 2),  # skip_frames: every case here is double-frame
    )


def test_count_validation(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    ok = _validate(run, 4)
    assert ok["valid"] is True and ok["found_count"] == 4
    assert ok["format_detected"] == "pivraw" and ok["image_size"] == (16, 8)
    assert "4 frames" in ok["found"]

    # num_images above a camera's frames is not a shortfall for a .pivraw run:
    # its pairs come from the trigger alignment, which caps the pair count.
    more = _validate(run, 6)
    assert more["valid"] is True, more["error"]
    assert more["found_count"] == 4


def test_count_validation_surfaces_unfinalised_run(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    edit_sidecar(run, 1, finalised=False)
    result = _validate(run, 4)
    assert result["valid"] is False
    assert "run not finalised" in result["error"]


def test_count_validation_surfaces_a_refused_index(tmp_path):
    run = copy_case(tmp_path, "df_2cam_unequal_drops")
    edit_sidecar(run, 1, missing_indices=[])
    result = _validate(run, 4)
    assert result["valid"] is False
    assert "missing_indices and the .idx do not partition" in result["error"]


def test_count_validation_names_extension_mismatch(tmp_path):
    (tmp_path / "Camera1.png").write_bytes(b"\x89PNG")
    result = _validate(tmp_path, 2, image_format="Camera%d.png")
    assert result["error"] == (
        f"Image type is PIVRAW but the pattern names a .png file: {tmp_path / 'Camera1.png'}"
    )


# ---------------------------------------------------------------------------
# Source diagnosis
# ---------------------------------------------------------------------------


def test_scan_counts_containers_not_sidecars(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    scan = scan_source(run)
    assert scan.root_ext_counts == {".pivraw": 2}
    assert sorted(scan.root_names) == ["Camera1.pivraw", "Camera2.pivraw"]
    assert pivraw_pattern(scan.root_names, run, 2, [1, 2]) == "Camera%d.pivraw"


def test_wrong_type_suggests_pivraw_for_a_run_folder(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    [s] = rule_wrong_type(_inp(run))
    assert s["apply"] == {"image_type": "pivraw", "image_format": ["Camera%d.pivraw"]}
    assert s["label"] == "Switch to PIVTOOLS Acquisition (.pivraw)"


def test_picked_container_file_suggests_its_run_folder(tmp_path):
    run = copy_case(tmp_path, DF_CLEAN)
    [s] = rule_picked_file(_inp(run / "Camera1.pivraw"))
    assert s["apply"] == {
        "source_path": str(run),
        "image_type": "pivraw",
        "image_format": ["Camera%d.pivraw"],
    }
    # A sidecar is not an image: picking it offers nothing.
    assert rule_picked_file(_inp(run / "Camera1.pivraw.json")) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
