#!/usr/bin/env python3
"""
test_pivraw_reader.py

The ``pivraw`` image type: PIVTOOLS Acquisition run containers, one
``Camera{k}.pivraw`` (+ ``.json`` sidecar, ``.idx`` records) per camera.

Locks in:
  * Mono12p decodes to the same counts as an independent byte-plane decode, and
    round-trips the fixture's byte-by-byte packer; frames are found at
    ``k * frame_stride`` past the zero padding.
  * BayerRG8 becomes the 2x2 superpixel sum R+G+G+B at half resolution.
  * Pairs honour ``frame_b_idx`` and ``out=``; a double-frame run refuses any
    pair but its recorded ``(2k-1, 2k)``.
  * read_pair / _read_batch through a Config equal the source frames.
  * Image shape comes from the sidecar without decoding pixels.
  * Unfinalised runs, missing fields, wrong versions and size mismatches raise.
  * Count validation, calibration counting and source diagnosis know the type,
    and the ``.json`` / ``.idx`` sidecars are never taken for images.

Usage:
    pytest unit-tests/test_pivraw_reader.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.diagnosis import _inp, make_ctx, piv_route_response  # noqa: E402
from fixtures.pivraw import (  # noqa: E402
    DROP,
    bayer_frames,
    mono_frames,
    pack_mono12p,
    stride_for,
    write_camera,
    write_piv_config,
    write_run,
)

from pivtools_core.calibration_settings import save_settings  # noqa: E402
from pivtools_core.config import Config  # noqa: E402
from pivtools_core.image_handling.calibration_loader import (  # noqa: E402
    FrameCount,
    get_calibration_frame_count,
)
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
from pivtools_core.image_handling.readers import get_reader  # noqa: E402
from pivtools_core.image_handling.readers import pivraw_reader  # noqa: E402
from pivtools_core.image_handling.readers.pivraw_reader import (  # noqa: E402
    PivrawError,
    clear_header_cache,
    get_pivraw_frame_count,
    get_pivraw_image_shape,
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


@pytest.fixture(autouse=True)
def _clean_header_cache():
    clear_header_cache()
    yield
    clear_header_cache()


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


def _bayer_reference(mosaic: np.ndarray) -> np.ndarray:
    """R + G + G + B of every 2x2 RGGB block, as float32."""
    m = mosaic.astype(np.float32)
    return m[0::2, 0::2] + m[0::2, 1::2] + m[1::2, 0::2] + m[1::2, 1::2]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


# 64 x 48 Mono12p is 4608 bytes -> stride 8192: frames sit past 3584 padding bytes.
MONO_H, MONO_W = 48, 64


def test_mono12p_matches_byte_plane_reference_and_packer(tmp_path):
    frames = mono_frames(4, MONO_H, MONO_W)
    path = write_camera(tmp_path, 1, frames)
    stride = stride_for(MONO_H * MONO_W * 3 // 2)
    assert stride == 8192
    blob = path.read_bytes()

    for k, frame in enumerate(frames):
        got = read_pivraw_single(path, k + 1)
        raw = blob[k * stride : k * stride + MONO_H * MONO_W * 3 // 2]
        assert got.dtype == np.float32 and got.shape == (MONO_H, MONO_W)
        np.testing.assert_array_equal(
            got, _decode_mono12p_byte_planes(raw, MONO_W, MONO_H)
        )
        np.testing.assert_array_equal(got, frame.astype(np.float32))
        # The padding the reader skips really is zeros.
        assert not any(blob[k * stride + len(raw) : (k + 1) * stride])


@pytest.mark.parametrize("value", [0, 1, 0x0F0, 0x00F, 0xF00, 0xFFF])
def test_mono12p_bit_patterns(tmp_path, value):
    frame = np.full((2, 4), value, dtype=np.uint16)
    frame[0, 1] = 0xFFF - value  # odd pixel differs from even
    path = write_camera(tmp_path, 1, [frame, frame])
    np.testing.assert_array_equal(read_pivraw_single(path, 2), frame.astype(np.float32))


def test_bayer_superpixel_matches_numpy_reference(tmp_path):
    frames = bayer_frames(2, 8, 12)
    frames[1][:] = 255  # max: 4 * 255 = 1020, exact in float32
    path = write_camera(tmp_path, 1, frames, pixel_format="BayerRG8")

    a = read_pivraw_single(path, 1)
    assert a.shape == (4, 6) and a.dtype == np.float32
    np.testing.assert_array_equal(a, _bayer_reference(frames[0]))
    np.testing.assert_array_equal(read_pivraw_single(path, 2), np.full((4, 6), 1020.0))
    assert get_pivraw_image_shape(path) == (4, 6)


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def test_double_frame_pair_reads_recorded_pairs(tmp_path):
    frames = mono_frames(6, 4, 6)
    path = write_camera(tmp_path, 1, frames)
    pair = read_pivraw_pair(path, idx=3)
    assert pair.shape == (2, 4, 6) and pair.dtype == np.float32
    np.testing.assert_array_equal(pair, np.stack(frames[2:4]).astype(np.float32))


@pytest.mark.parametrize("a, b", [(2, 3), (1, 1), (1, 3), (3, 2)])
def test_double_frame_refuses_other_pairings(tmp_path, a, b):
    path = write_camera(tmp_path, 1, mono_frames(6, 4, 6))
    with pytest.raises(PivrawError, match="skip_frames pairing preset"):
        read_pivraw_pair(path, idx=a, frame_b_idx=b)


def test_time_resolved_pair_honours_frame_b_idx(tmp_path):
    frames = mono_frames(6, 4, 6)
    path = write_camera(tmp_path, 1, frames, mode="time_resolved")
    pair = read_pivraw_pair(path, idx=2, frame_b_idx=5)
    np.testing.assert_array_equal(pair[0], frames[1].astype(np.float32))
    np.testing.assert_array_equal(pair[1], frames[4].astype(np.float32))
    # None means A + 1.
    np.testing.assert_array_equal(
        read_pivraw_pair(path, idx=2)[1], frames[2].astype(np.float32)
    )


def test_pair_out_is_written_in_place(tmp_path):
    frames = mono_frames(2, 4, 6)
    path = write_camera(tmp_path, 1, frames)
    out = np.full((2, 4, 6), -1.0, dtype=np.float32)
    assert read_pivraw_pair(path, idx=1, out=out) is out
    np.testing.assert_array_equal(out, np.stack(frames).astype(np.float32))

    single = np.empty((4, 6), dtype=np.float32)
    assert read_pivraw_single(path, 2, out=single) is single


@pytest.mark.parametrize(
    "bad", [np.empty((2, 4, 5), np.float32), np.empty((2, 4, 6), np.float64)]
)
def test_pair_out_contract_is_enforced(tmp_path, bad):
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6))
    with pytest.raises(ValueError, match="out has"):
        read_pivraw_pair(path, idx=1, out=bad)


@pytest.mark.parametrize("idx", [0, 3])
def test_out_of_range_frame_raises(tmp_path, idx):
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6))
    with pytest.raises(PivrawError, match="out of range"):
        read_pivraw_single(path, idx)


# ---------------------------------------------------------------------------
# Strict sidecar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{"finalised": False}, {"n_frames": None}, {"finalised": False, "n_frames": None}],
)
def test_unfinalised_run_names_the_recover_command(tmp_path, overrides):
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6), sidecar_overrides=overrides)
    with pytest.raises(PivrawError) as exc:
        get_pivraw_frame_count(path)
    assert "run not finalised (recorder stopped early?): run " in str(exc.value)
    assert f"`pivtools-acq recover {tmp_path}`" in str(exc.value)


@pytest.mark.parametrize(
    "field", ["dt_s", "pixel_format", "missing_indices", "finalised"]
)
def test_missing_field_raises_by_name(tmp_path, field):
    path = write_camera(
        tmp_path, 1, mono_frames(2, 4, 6), sidecar_overrides={field: DROP}
    )
    with pytest.raises(PivrawError, match=f"missing required field\\(s\\) {field}"):
        read_pivraw_single(path, 1)


@pytest.mark.parametrize("version", [2, 0, "1", True])
def test_wrong_format_version_raises(tmp_path, version):
    path = write_camera(
        tmp_path, 1, mono_frames(2, 4, 6), sidecar_overrides={"format_version": version}
    )
    with pytest.raises(PivrawError, match="format_version"):
        get_pivraw_frame_count(path)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"format": "cine"}, "format is 'cine'"),
        ({"bytes_per_frame": 37}, "bytes_per_frame 37 does not match"),
        ({"frame_stride": 8192}, "frame_stride 8192"),
        ({"mode": "burst"}, "mode 'burst'"),
        ({"pixel_format": "Mono16"}, "pixel_format 'Mono16'"),
        ({"n_frames": 3}, "odd"),
        ({"n_frames": 4}, "data file and its sidecar disagree"),
    ],
)
def test_inconsistent_sidecar_raises(tmp_path, overrides, message):
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6), sidecar_overrides=overrides)
    with pytest.raises(PivrawError, match=message):
        get_pivraw_frame_count(path)


def test_truncated_data_and_missing_index_raise(tmp_path):
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6))
    Path(str(path) + ".idx").unlink()
    with pytest.raises(PivrawError, match="per-frame index .* is missing"):
        get_pivraw_frame_count(path)

    path = write_camera(tmp_path, 2, mono_frames(2, 4, 6))
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(PivrawError, match="disagree"):
        get_pivraw_frame_count(path)


def test_missing_sidecar_is_file_not_found(tmp_path):
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6))
    Path(str(path) + ".json").unlink()
    with pytest.raises(FileNotFoundError, match="PIVRAW sidecar not found"):
        get_pivraw_frame_count(path)


def test_header_cache_revalidates_after_recover_rewrite(tmp_path):
    """A recover truncates the data and rewrites the sidecar; the cache must notice."""
    path = write_camera(tmp_path, 1, mono_frames(4, 4, 6))
    assert get_pivraw_frame_count(path) == 4
    write_camera(tmp_path, 1, mono_frames(2, 4, 6))
    assert get_pivraw_frame_count(path) == 2


# ---------------------------------------------------------------------------
# Config, read_pair, read_batch
# ---------------------------------------------------------------------------


def _two_camera_run(tmp_path, n=6):
    cams = [mono_frames(n, 4, 6, seed=1), mono_frames(n, 4, 6, seed=2)]
    run = write_run(tmp_path / "run_0001", cams)
    return run, cams


def test_type_detection_agrees_between_config_and_path_utils(tmp_path):
    run, _ = _two_camera_run(tmp_path)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 6)))
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
    run = write_run(tmp_path / "run", [bayer_frames(2, 8, 12)], pixel_format="BayerRG8")
    cfg = Config(path=str(write_piv_config(tmp_path, run, 1, 2)))

    def _no_decode(*_a, **_k):
        raise AssertionError("shape detection decoded pixels")

    monkeypatch.setattr(pivraw_reader, "_decode_into", _no_decode)
    assert cfg.image_shape_for(1) == (4, 6)


def test_read_pair_and_read_batch_through_config(tmp_path):
    run, cams = _two_camera_run(tmp_path)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 6)))
    assert cfg.num_frame_pairs == 3

    for camera in (1, 2):
        for p in (1, 2, 3):
            expected = np.stack(cams[camera - 1][2 * p - 2 : 2 * p]).astype(np.float32)
            np.testing.assert_array_equal(read_pair(p, run, camera, cfg), expected)
        batch = _read_batch(1, 3, run, camera, cfg)
        assert batch.shape == (3, 2, 4, 6) and batch.dtype == np.float32
        np.testing.assert_array_equal(
            batch, np.stack([read_pair(p, run, camera, cfg) for p in (1, 2, 3)])
        )


def test_read_pair_with_wrong_preset_fails_loudly(tmp_path):
    """time_resolved pairing on a double-frame run would correlate B with the next A."""
    run, _ = _two_camera_run(tmp_path)
    cfg = Config(
        path=str(write_piv_config(tmp_path, run, 2, 6, preset="time_resolved"))
    )
    read_pair(1, run, 1, cfg)  # (1, 2) is a recorded pair
    with pytest.raises(PivrawError, match="skip_frames"):
        read_pair(2, run, 1, cfg)  # (2, 3) is not


def test_read_pair_passes_frame_b_through(tmp_path):
    """frame_stride 2 on a time-resolved run pairs (1, 3): B is not A + 1."""
    frames = mono_frames(5, 4, 6)
    run = write_run(tmp_path / "tr", [frames], mode="time_resolved")
    cfg = Config(
        path=str(write_piv_config(tmp_path, run, 1, 5, preset="time_resolved"))
    )
    cfg.data["images"]["pairing_preset"] = "custom"
    cfg.data["images"]["frame_stride"] = 2
    assert cfg.get_frame_pair_indices(1) == (1, 3)
    np.testing.assert_array_equal(
        read_pair(1, run, 1, cfg), np.stack([frames[0], frames[2]]).astype(np.float32)
    )


def test_ab_format_preset_on_double_frame_run_is_refused(tmp_path):
    """ab_format gives (A, B) = (1, 1); the reader must not quietly read (1, 2)."""
    run, _ = _two_camera_run(tmp_path)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 6, preset="ab_format")))
    assert cfg.get_frame_pair_indices(1) == (1, 1)
    with pytest.raises(PivrawError, match=r"frames \(1, 1\) were requested"):
        read_pair(1, run, 1, cfg)


def test_time_resolved_run_through_config(tmp_path):
    frames = mono_frames(5, 4, 6)
    run = write_run(tmp_path / "tr", [frames], mode="time_resolved")
    cfg = Config(
        path=str(write_piv_config(tmp_path, run, 1, 5, preset="time_resolved"))
    )
    assert cfg.num_frame_pairs == 4
    np.testing.assert_array_equal(
        read_pair(4, run, 1, cfg), np.stack(frames[3:5]).astype(np.float32)
    )
    np.testing.assert_array_equal(
        read_single_frame(run / "Camera1.pivraw", 1, 3, "pivraw"),
        frames[2].astype(np.float32),
    )


def test_end_to_end_validate_then_read(tmp_path):
    """A user's config for a two-camera double-frame run validates, then reads."""
    run, cams = _two_camera_run(tmp_path)
    config_path = write_piv_config(tmp_path, run, 2, 6)
    import yaml

    data = yaml.safe_load(config_path.read_text())
    data["paths"]["base_paths"] = [str(tmp_path / "out")]
    config_path.write_text(yaml.dump(data))
    cfg = Config(path=str(config_path))

    ok, error, _warnings = validate_config(cfg)
    assert ok, error
    assert cfg.image_shape_for(2) == (4, 6)
    np.testing.assert_array_equal(
        read_pair(2, run, 2, cfg), np.stack(cams[1][2:4]).astype(np.float32)
    )


def test_gui_validate_route_accepts_a_run(tmp_path, monkeypatch):
    run, _ = _two_camera_run(tmp_path)
    resp = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(run)], "camera_count": 2, "camera_numbers": [1, 2]},
        images={
            "image_type": "pivraw",
            "image_format": ["Camera%d.pivraw"],
            "num_images": 6,
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
    )


def test_count_validation(tmp_path):
    run, _ = _two_camera_run(tmp_path)
    ok = _validate(run, 6)
    assert ok["valid"] is True and ok["found_count"] == 6
    assert ok["format_detected"] == "pivraw" and ok["image_size"] == (6, 4)
    assert "6 frames" in ok["found"]

    short = _validate(run, 8)
    assert short["valid"] is False
    assert short["error"].startswith("found 6 entries, expected 8")


def test_count_validation_surfaces_unfinalised_run(tmp_path):
    write_camera(
        tmp_path, 1, mono_frames(2, 4, 6), sidecar_overrides={"finalised": False}
    )
    result = _validate(tmp_path, 2)
    assert result["valid"] is False
    assert "run not finalised" in result["error"]


def test_count_validation_names_extension_mismatch(tmp_path):
    (tmp_path / "Camera1.png").write_bytes(b"\x89PNG")
    result = _validate(tmp_path, 2, image_format="Camera%d.png")
    assert result["error"] == (
        f"Image type is PIVRAW but the pattern names a .png file: {tmp_path / 'Camera1.png'}"
    )


def test_calibration_frame_count_reads_the_header(tmp_path):
    write_run(tmp_path, [mono_frames(4, 4, 6), mono_frames(4, 4, 6, seed=3)])
    save_settings(
        tmp_path, {"image": {"image_format": "Camera%d.pivraw", "image_type": "pivraw"}}
    )

    class _Cfg:
        camera_count = 2

        def get_calibration_source(self, idx):
            return tmp_path

    assert get_calibration_frame_count(2, _Cfg(), 0) == FrameCount(4, ".pivraw header")


# ---------------------------------------------------------------------------
# Source diagnosis
# ---------------------------------------------------------------------------


def test_scan_counts_containers_not_sidecars(tmp_path):
    run, _ = _two_camera_run(tmp_path)
    scan = scan_source(run)
    assert scan.root_ext_counts == {".pivraw": 2}
    assert sorted(scan.root_names) == ["Camera1.pivraw", "Camera2.pivraw"]
    assert pivraw_pattern(scan.root_names, run, 2, [1, 2]) == "Camera%d.pivraw"


def test_wrong_type_suggests_pivraw_for_a_run_folder(tmp_path):
    run, _ = _two_camera_run(tmp_path)
    [s] = rule_wrong_type(_inp(run))
    assert s["apply"] == {"image_type": "pivraw", "image_format": ["Camera%d.pivraw"]}
    assert s["label"] == "Switch to PIVTOOLS Acquisition (.pivraw)"


def test_picked_container_file_suggests_its_run_folder(tmp_path):
    run, _ = _two_camera_run(tmp_path)
    [s] = rule_picked_file(_inp(run / "Camera1.pivraw"))
    assert s["apply"] == {
        "source_path": str(run),
        "image_type": "pivraw",
        "image_format": ["Camera%d.pivraw"],
    }
    # A sidecar is not an image: picking it offers nothing.
    assert rule_picked_file(_inp(run / "Camera1.pivraw.json")) == []


def test_sidecar_is_valid_json_written_by_fixture(tmp_path):
    """Guards the fixture itself: every field the reader requires is written."""
    path = write_camera(tmp_path, 1, mono_frames(2, 4, 6))
    sidecar = json.loads(Path(str(path) + ".json").read_text())
    assert set(pivraw_reader.REQUIRED_FIELDS) <= set(sidecar)
    assert pack_mono12p(np.array([[0xABC, 0x123]], np.uint16)) == bytes(
        [0xBC, 0x3A, 0x12]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
