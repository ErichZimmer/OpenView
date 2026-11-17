"""Each camera keeps its own image shape through the pipeline.

Decision (Morgan, 17 Nov 2026, RS report decision 1(a)): a rig whose cameras
differ in sensor size, or that has one camera rotated by 90 degrees, is
processed with every camera selected in one run. ``Config.image_shape_for``
replaces the single shape detected from the first camera, and every reader of
the shape passes the camera.

Merging cameras of different sizes into one field stays out of scope, so the
paths that combine cameras refuse mismatched shapes by name.

Usage:
    pytest unit-tests/test_per_camera_image_shape.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config
from pivtools_core.image_handling.load_images import load_images

# Two deliberately different, non-square sensors. Camera 2 is the transpose of
# camera 1's aspect, the shape a 90 degree rotated camera presents.
CAM_SHAPES = {1: (64, 96), 2: (96, 64)}
NUM_PAIRS = 2


def _write_tiff_rig(source: Path, shapes: dict, num_pairs: int) -> None:
    """Write A/B TIFF pairs of random particle-like noise, one folder per camera."""
    rng = np.random.default_rng(7)
    for cam, (h, w) in shapes.items():
        folder = source / f"Cam{cam}"
        folder.mkdir(parents=True)
        for idx in range(1, num_pairs + 1):
            for frame in ("A", "B"):
                img = rng.integers(0, 4096, size=(h, w), dtype=np.uint16)
                tifffile.imwrite(folder / f"B{idx:05d}_{frame}.tif", img)


def _make_config(tmp_path: Path, shapes: dict = CAM_SHAPES) -> Config:
    """Two-camera standard TIFF config over a synthetic rig in ``tmp_path``."""
    source = tmp_path / "source"
    base = tmp_path / "output"
    base.mkdir()
    _write_tiff_rig(source, shapes, NUM_PAIRS)
    cfg = {
        "calibration": {"active": "scale_factor"},
        "paths": {
            "source_paths": [str(source)],
            "base_paths": [str(base)],
            "camera_count": len(shapes),
            "camera_numbers": sorted(shapes),
        },
        "images": {
            "num_images": NUM_PAIRS,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
            "vector_format": ["B%05d.mat"],
        },
        "processing": {
            "backend": "cpu",
            "omp_threads": 1,
            "dask_workers_per_node": 1,
            "dask_memory_limit": "2GB",
            "dask_max_in_flight_per_worker": 2,
        },
        "batches": {"size": 1},
        "instantaneous_piv": {
            "window_size": [[32, 32]],
            "overlap": [50],
            "runs": [1],
            "peak_finder": "gauss3",
        },
        "outlier_detection": {"enabled": False},
        "filters": [],
        "masking": {"enabled": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return Config(path=str(config_path))


def test_each_camera_reports_its_own_shape(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.image_shape_for(1) == (64, 96)
    assert cfg.image_shape_for(2) == (96, 64)
    # Order of asking must not matter: no shape leaks from the first camera.
    assert cfg.image_shape_for(1) == (64, 96)


def test_load_images_gives_each_camera_its_own_shape(tmp_path):
    """Camera 2 used to fail at its first batch: the buffer was sized from
    camera 1's shape. Each camera's lazy array and its computed pixels now
    carry that camera's shape, and the pixels are the files' own."""
    cfg = _make_config(tmp_path)
    source = cfg.source_paths[0]
    for cam, (h, w) in CAM_SHAPES.items():
        images = load_images(cam, cfg, source=source, batch_size=1)
        assert images.shape == (NUM_PAIRS, 2, h, w)
        computed = images.compute()
        assert computed.shape == (NUM_PAIRS, 2, h, w)
        expected_b2 = tifffile.imread(source / f"Cam{cam}" / "B00002_B.tif")
        np.testing.assert_array_equal(computed[1, 1], expected_b2.astype(np.float32))


def test_a_pair_that_differs_from_its_cameras_first_image_fails_by_name(tmp_path):
    """Within one camera every pair must match the camera's detected shape;
    a stray resized file fails at its batch, naming camera and pair."""
    cfg = _make_config(tmp_path)
    stray = cfg.source_paths[0] / "Cam2" / "B00002_A.tif"
    tifffile.imwrite(stray, np.zeros((80, 64), np.uint16))
    tifffile.imwrite(stray.with_name("B00002_B.tif"), np.zeros((80, 64), np.uint16))
    images = load_images(2, cfg, batch_size=1)
    with pytest.raises(ValueError, match=r"Pair 2 of camera 2 does not fit"):
        images.compute()


# Window grid by hand for a 32 px window at 50 % overlap (16 px spacing, first
# centre at 16 px): 64 px holds centres 16, 32, 48 -> 3 windows; 96 px holds
# 16 ... 80 -> 5 windows.
EXPECTED_GRID = {1: (3, 5), 2: (5, 3)}


def test_instantaneous_runs_both_cameras_in_one_process(tmp_path):
    """The body of ``instantaneous.main``'s camera loop, for both cameras on
    one Config and one client: mask, vector mask, then the run. Each camera's
    vectors and coordinates land on its own window grid."""
    from dask.distributed import Client

    from pivtools_cli.piv.save_results import get_output_path
    from pivtools_cli.piv_cluster.cluster import LOOPBACK_ANY_PORT
    from pivtools_core.image_handling.load_images import (
        compute_vector_mask,
        load_mask_for_camera,
    )
    from pivtools_core.instantaneous import run_instantaneous_piv
    from pivtools_core.vector_loading import read_mat_contents

    cfg = _make_config(tmp_path)
    cfg.data["masking"] = {
        "enabled": True,
        "mode": "rectangular",
        "rectangular": {"top": 4, "bottom": 0, "left": 0, "right": 0},
    }
    source = cfg.source_paths[0]

    client = Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=LOOPBACK_ANY_PORT,
        worker_dashboard_address=LOOPBACK_ANY_PORT,
    )
    try:
        for cam in cfg.camera_numbers:
            mask = load_mask_for_camera(cam, cfg)
            assert mask.shape == CAM_SHAPES[cam]
            vector_masks = compute_vector_mask(mask, cfg, cam, ensemble=False)
            assert vector_masks[0].shape == EXPECTED_GRID[cam]
            output_path = get_output_path(
                cfg, cam, use_uncalibrated=True, piv_type="instantaneous"
            )
            run_instantaneous_piv(
                config=cfg,
                client=client,
                camera_num=cam,
                source_path=source,
                output_path=output_path,
                base_path=cfg.base_paths[0],
                vector_masks=vector_masks,
                pixel_mask=mask,
            )
            arr = read_mat_contents(str(output_path / "B00002.mat"), run_index=0)
            assert arr[0, 0].shape == EXPECTED_GRID[cam], f"camera {cam}"
    finally:
        client.close()


def test_ensemble_runs_both_cameras_in_one_process(tmp_path):
    """The ensemble driver, accumulator and worker accumulation for both
    cameras on one Config and one client. Each camera's ensemble result and
    coordinates land on its own window grid."""
    from dask.distributed import Client

    from pivtools_cli.piv.save_results import (
        get_ensemble_output_path,
        load_ensemble_result,
    )
    from pivtools_cli.piv_cluster.cluster import LOOPBACK_ANY_PORT
    from pivtools_core.ensemble import run_ensemble_piv
    from pivtools_core.vector_loading import load_coords_from_directory

    cfg = _make_config(tmp_path)
    cfg.data["ensemble_piv"] = {
        "window_size": [[32, 32]],
        "overlap": [50],
        "type": ["std"],
        "runs": [1],
        "fit_method": "kspace",
    }
    source = cfg.source_paths[0]

    client = Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=LOOPBACK_ANY_PORT,
        worker_dashboard_address=LOOPBACK_ANY_PORT,
    )
    try:
        for cam in cfg.camera_numbers:
            output_path = get_ensemble_output_path(cfg, cam, use_uncalibrated=True)
            run_ensemble_piv(
                config=cfg,
                client=client,
                camera_num=cam,
                source_path=source,
                output_path=output_path,
                base_path=cfg.base_paths[0],
            )
            x_list, _y_list = load_coords_from_directory(output_path, runs=[1])
            assert x_list[0].shape == EXPECTED_GRID[cam], f"camera {cam}"
            result, _n = load_ensemble_result(output_path / "ensemble_result.mat")
            assert result.passes[0].ux_mat.shape == EXPECTED_GRID[cam], f"camera {cam}"
    finally:
        client.close()


def test_correlator_cache_refuses_another_cameras_frames(tmp_path):
    """A correlator cache belongs to one camera. Handed another camera's frames
    (the one place two cameras' shapes meet in a run) it refuses by name,
    rather than laying camera 1's window grid over camera 2's pixels."""
    from pivtools_cli.piv.piv_backend.cpu_instantaneous import (
        InstantaneousCorrelatorCPU,
    )

    cfg = _make_config(tmp_path)
    cache_cam1 = InstantaneousCorrelatorCPU(cfg, camera=1).get_cache_data()
    on_worker = InstantaneousCorrelatorCPU(cfg, precomputed_cache=cache_cam1)
    cam2_batch = load_images(2, cfg, batch_size=1)[:1].compute()
    with pytest.raises(ValueError, match=r"built for 64 x 96 frames.*96 x 64"):
        on_worker.correlate_batch(cam2_batch, config=cfg)


def test_ensemble_correlator_refuses_another_cameras_frames(tmp_path):
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    cfg = _make_config(tmp_path)
    cfg.data["ensemble_piv"] = {"window_size": [[32, 32]], "overlap": [50]}
    correlator = EnsembleCorrelatorCPU(cfg, camera=2)
    cam1_batch = load_images(1, cfg, batch_size=1)[:1].compute()
    with pytest.raises(ValueError, match=r"built for 96 x 64 frames.*64 x 96"):
        correlator.correlate_batch_for_accumulation(cam1_batch, cfg, pass_idx=0)


@pytest.mark.parametrize("kwargs", [{}, {"camera": 1, "precomputed_cache": {}}])
def test_correlator_needs_exactly_one_shape_source(tmp_path, kwargs):
    from pivtools_cli.piv.piv_backend.cpu_instantaneous import (
        InstantaneousCorrelatorCPU,
    )

    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match=r"needs exactly one of camera"):
        InstantaneousCorrelatorCPU(cfg, **kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
