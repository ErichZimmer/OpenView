"""``POST /validate_files`` checks every loop of a looped recording through its own source.

14 Nov review, fault 2. With ``images.num_loops: 2`` and 3 pairs per loop the
pipeline reads 6 pairs, 3 from each loop's source. The validator read the
global last pair (6) from loop 0's source, so a correct recording failed
("Last pair (6) could not be read" for TIFFs, "Entry 5 out of range" for a
``.set``), and loops 1..N were only checked to exist.

The pipeline (``load_images``) is the reference for what a correct looped
recording is.
"""

import numpy as np
import pytest
import tifffile
from fixtures.diagnosis import piv_route_response, write_config
from fixtures.lavision import build_raw16_set

from pivtools_core.image_handling.load_images import load_images

PAIRS_PER_LOOP = 3
FRAME_SHAPE = (8, 12)


def _tif_loops(tmp_path):
    """``run_0`` and ``run_1``, each 3 A/B pairs; pixel value 100*loop + 10*pair (+5 for B)."""
    for loop in (0, 1):
        folder = tmp_path / f"run_{loop}"
        folder.mkdir()
        for pair in range(1, PAIRS_PER_LOOP + 1):
            for role, add in (("A", 0), ("B", 5)):
                tifffile.imwrite(
                    folder / f"B{pair:05d}_{role}.tif",
                    np.full(FRAME_SHAPE, 100 * loop + 10 * pair + add, np.uint16),
                )
    paths = {"source_paths": [str(tmp_path / "run_0")]}
    images = {
        "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
        "frame_stride": 0,
        "pair_stride": 1,
        "start_index": 1,
        "num_images": PAIRS_PER_LOOP,
        "num_loops": 2,
    }
    return paths, images


def _set_loops(tmp_path, loop_1_entries=PAIRS_PER_LOOP + 1):
    """``loop=0.set`` and ``loop=1.set``, time-resolved: 4 entries give 3 pairs."""
    set_0 = build_raw16_set(
        tmp_path, [FRAME_SHAPE], name="loop=0", n_entries=PAIRS_PER_LOOP + 1
    )
    build_raw16_set(tmp_path, [FRAME_SHAPE], name="loop=1", n_entries=loop_1_entries)
    paths = {"source_paths": [str(set_0)]}
    images = {
        "image_format": [set_0.name],
        "image_type": "lavision_set",
        "frame_stride": 1,
        "pair_stride": 1,
        "start_index": 1,
        "num_images": PAIRS_PER_LOOP + 1,
        "num_loops": 2,
    }
    return paths, images


def _camera_1(tmp_path, monkeypatch, paths, images):
    response = piv_route_response(tmp_path, monkeypatch, paths=paths, images=images)
    assert response.status_code == 200
    body = response.get_json()
    return body["valid"], body["details"]["camera_1"]


def test_pipeline_reads_both_tif_loops(tmp_path):
    """The reference: 6 pairs, the last one from run_1's pair 3."""
    paths, images = _tif_loops(tmp_path)
    cfg = write_config(tmp_path, paths=paths, images=images)
    pairs = load_images(1, cfg).compute()
    assert pairs.shape == (2 * PAIRS_PER_LOOP, 2, *FRAME_SHAPE)
    assert [int(v) for v in pairs[:, 0, 0, 0]] == [10, 20, 30, 110, 120, 130]


def test_pipeline_reads_both_set_loops(tmp_path):
    paths, images = _set_loops(tmp_path)
    cfg = write_config(tmp_path, paths=paths, images=images)
    assert load_images(1, cfg).shape == (2 * PAIRS_PER_LOOP, 2, *FRAME_SHAPE)


@pytest.mark.parametrize("dataset", [_tif_loops, _set_loops])
def test_a_correct_looped_recording_validates(tmp_path, monkeypatch, dataset):
    valid, camera = _camera_1(tmp_path, monkeypatch, *dataset(tmp_path))
    assert camera["error"] is None
    assert camera["status"] == "ok"
    assert valid


def test_loop_1_missing_its_last_tif_is_an_error_naming_loop_1(tmp_path, monkeypatch):
    paths, images = _tif_loops(tmp_path)
    (tmp_path / "run_1" / f"B{PAIRS_PER_LOOP:05d}_B.tif").unlink()

    valid, camera = _camera_1(tmp_path, monkeypatch, paths, images)

    assert not valid
    assert camera["status"] == "error"
    assert f"Loop 1 (run_1): last pair ({PAIRS_PER_LOOP}) could not be read" in (
        camera["error"]
    )
    assert "Loop 0" not in camera["error"]


def test_loop_1_short_of_set_entries_is_an_error_naming_loop_1(tmp_path, monkeypatch):
    """loop=1.set holds 3 entries (2 pairs) where the config asks for 4 (3 pairs)."""
    paths, images = _set_loops(tmp_path, loop_1_entries=PAIRS_PER_LOOP)

    valid, camera = _camera_1(tmp_path, monkeypatch, paths, images)

    assert not valid
    assert f"Loop 1 (loop=1.set): last pair ({PAIRS_PER_LOOP}) could not be read" in (
        camera["error"]
    )
    assert "Loop 0" not in camera["error"]


def test_a_missing_loop_source_is_named(tmp_path, monkeypatch):
    paths, images = _tif_loops(tmp_path)
    for f in (tmp_path / "run_1").iterdir():
        f.unlink()
    (tmp_path / "run_1").rmdir()

    valid, camera = _camera_1(tmp_path, monkeypatch, paths, images)

    assert not valid
    assert "Loop 1 source not found: run_1" in camera["error"]
