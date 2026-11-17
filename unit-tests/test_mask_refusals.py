"""A pixel mask that does not fit its camera's images is refused by name.

Decision (Morgan, 17 Nov 2026, readers ledger R2c): a mask whose size does not
match the camera's image is refused, never stretched. With masking on, a
camera whose mask file is missing or unreadable is refused too, never run
unmasked. With masking off there is no mask, and that is fine.

The rig has two cameras of transposed shape, (64, 96) and (96, 64): camera 2
presents the shape of camera 1 turned a quarter turn, so camera 1's mask on
camera 2 is the "drawn before the camera was rotated" case.

Usage:
    pytest unit-tests/test_mask_refusals.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config
from pivtools_core.image_handling.load_images import MaskError, load_mask_for_camera
from pivtools_core.vector_loading import save_mask_to_mat

CAM_SHAPES = {1: (64, 96), 2: (96, 64)}


def _make_config(tmp_path: Path, masking: dict) -> Config:
    """Two-camera standard TIFF rig, one pair per camera, with ``masking``."""
    source = tmp_path / "source"
    base = tmp_path / "output"
    base.mkdir()
    for cam, (h, w) in CAM_SHAPES.items():
        folder = source / f"Cam{cam}"
        folder.mkdir(parents=True)
        for frame in ("A", "B"):
            tifffile.imwrite(
                folder / f"B00001_{frame}.tif", np.zeros((h, w), np.uint16)
            )
    cfg = {
        "paths": {
            "source_paths": [str(source)],
            "base_paths": [str(base)],
            "camera_count": 2,
            "camera_numbers": [1, 2],
        },
        "images": {
            "num_images": 1,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
            "vector_format": ["B%05d.mat"],
        },
        "instantaneous_piv": {"window_size": [[32, 32]], "overlap": [50]},
        "masking": masking,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return Config(path=str(config_path))


def _save_mask(cfg: Config, camera: int, shape: tuple) -> Path:
    """Write a mask of ``shape`` where ``camera``'s mask file belongs."""
    path = cfg.get_mask_path(camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=bool)
    mask[:4, :] = True
    save_mask_to_mat(str(path), mask, np.asarray([]))
    return path


FILE_MASKING = {"enabled": True, "mode": "file"}


def test_a_mask_of_the_wrong_size_is_refused_naming_camera_file_and_shapes(tmp_path):
    cfg = _make_config(tmp_path, FILE_MASKING)
    path = _save_mask(cfg, 1, (32, 48))
    with pytest.raises(MaskError) as err:
        load_mask_for_camera(1, cfg)
    text = str(err.value)
    assert "Camera 1" in text
    assert str(path) in text
    assert "32 x 48" in text and "64 x 96" in text
    assert "redraw the mask on this camera's images" in text.lower()
    assert "rotated" not in text


def test_a_mask_with_height_and_width_swapped_says_the_camera_is_rotated(tmp_path):
    """Camera 1's (64, 96) mask on camera 2, whose images are (96, 64)."""
    cfg = _make_config(tmp_path, FILE_MASKING)
    _save_mask(cfg, 2, CAM_SHAPES[1])
    with pytest.raises(MaskError) as err:
        load_mask_for_camera(2, cfg)
    text = str(err.value)
    assert "Camera 2" in text
    assert "64 x 96" in text and "96 x 64" in text
    assert "rotated" in text


def test_a_mask_that_fits_is_returned_unchanged(tmp_path):
    """Control: each camera's own-size mask loads, pixel for pixel."""
    cfg = _make_config(tmp_path, FILE_MASKING)
    for cam, shape in CAM_SHAPES.items():
        _save_mask(cfg, cam, shape)
        mask = load_mask_for_camera(cam, cfg)
        expected = np.zeros(shape, dtype=bool)
        expected[:4, :] = True
        np.testing.assert_array_equal(mask, expected)


def test_with_masking_off_there_is_no_mask(tmp_path):
    """Control: masking disabled and no mask file anywhere gives None."""
    cfg = _make_config(tmp_path, {"enabled": False, "mode": "file"})
    assert load_mask_for_camera(1, cfg) is None


def test_with_masking_on_a_missing_mask_file_is_refused(tmp_path):
    cfg = _make_config(tmp_path, FILE_MASKING)
    with pytest.raises(MaskError) as err:
        load_mask_for_camera(2, cfg)
    text = str(err.value)
    assert "Camera 2" in text
    assert str(cfg.get_mask_path(2)) in text
    assert "masking off" in text


def test_with_masking_on_an_unreadable_mask_file_is_refused(tmp_path):
    cfg = _make_config(tmp_path, FILE_MASKING)
    path = cfg.get_mask_path(1)
    path.write_bytes(b"this is not a mat file")
    with pytest.raises(MaskError) as err:
        load_mask_for_camera(1, cfg)
    text = str(err.value)
    assert "Camera 1" in text and str(path) in text
    assert "could not be read" in text
    assert err.value.__cause__ is not None


def test_compute_vector_mask_refuses_a_mask_of_another_shape(tmp_path):
    """It used to resize the mask with ndimage.zoom and log a warning."""
    from pivtools_core.image_handling.load_images import compute_vector_mask

    cfg = _make_config(tmp_path, FILE_MASKING)
    with pytest.raises(MaskError, match=r"Camera 2.*64 x 96.*96 x 64.*rotated"):
        compute_vector_mask(np.zeros(CAM_SHAPES[1], bool), cfg, 2, ensemble=False)


def test_the_filter_step_refuses_a_mask_of_another_shape():
    """It used to log a warning and filter the block unmasked."""
    from pivtools_cli.processing.dask_pipeline import apply_all_filters_slim

    block = np.ones((1, 2, *CAM_SHAPES[2]), dtype=np.float32)
    with pytest.raises(MaskError, match=r"64 x 96.*96 x 64"):
        apply_all_filters_slim(block, [], np.zeros(CAM_SHAPES[1], bool))


def test_the_filter_step_zeroes_a_mask_that_fits():
    """Control: the masked rows are zeroed, the rest untouched."""
    from pivtools_cli.processing.dask_pipeline import apply_all_filters_slim

    block = np.ones((1, 2, *CAM_SHAPES[1]), dtype=np.float32)
    mask = np.zeros(CAM_SHAPES[1], bool)
    mask[:4, :] = True
    out = apply_all_filters_slim(block, [], mask)
    assert np.all(out[:, :, :4, :] == 0)
    assert np.all(out[:, :, 4:, :] == 1)


def test_mask_log_lines_render(tmp_path, caplog):
    """Mask logging used str.format braces under stdlib logging, which fails to render."""
    import logging

    cfg = _make_config(
        tmp_path,
        {
            "enabled": True,
            "mode": "rectangular",
            "rectangular": {"top": 4, "bottom": 0, "left": 0, "right": 0},
        },
    )
    _save_mask(cfg, 1, CAM_SHAPES[1])
    with caplog.at_level(logging.DEBUG):
        load_mask_for_camera(1, cfg)
        cfg.data["masking"]["mode"] = "file"
        load_mask_for_camera(1, cfg)
    assert caplog.records
    for record in caplog.records:
        text = record.getMessage()  # raises TypeError if the args do not fit
        assert "{}" not in text and "{:" not in text, text


def test_an_unknown_masking_mode_is_refused(tmp_path):
    cfg = _make_config(tmp_path, {"enabled": True, "mode": "polygon"})
    with pytest.raises(
        MaskError, match=r"masking\.mode 'polygon'.*'file', 'rectangular'"
    ):
        load_mask_for_camera(1, cfg)
