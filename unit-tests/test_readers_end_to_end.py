"""Readers and the per-camera pipeline together, end to end (ticket E1).

Two DaVis ``.set`` rigs, each run through configuration, image loading and an
instantaneous PIV job over both PIV cameras in one process:

1. Two cameras of equal sensor size, camera 2 carrying DaVis's
   ``rotate-cw-90`` transformer, pre-paired (A/B) and time-resolved. Camera 2's
   frames come out (W, H) and its vectors land on its own, transposed grid
   (rotation-shape REPORT §6 items 6, 7 and 10; C1, D6).
2. An NTNU-shaped rig: two double-frame PIV cameras and two single-frame PLIF
   cameras, identified by the camera labels in the recording's ``.attr`` files.
   PIV runs on cameras 1 and 2; a PLIF camera is refused for A/B pairing by
   name and allowed time-resolved (D7, decisions of 17 Nov 2026).

The truth is built in the stored (sensor) frame: the particle centres of
frame B are those of frame A moved by a stored shift (rows, columns), and the
rendered stored frames are what the ``.set`` holds. DaVis's transformer is
applied by the reader. The expected vector in each camera's OUTPUT frame is
derived in the test, not by hand: an index map of the sensor is pushed through
the camera's transform and the shifted pixel is located in it
(:func:`_output_shift`). Both rotated-rig cameras get the same stored shift
(2, 3), with both components non-zero, so the derived output shifts differ:
(2, 3) for camera 1 and, through ``rotate-cw-90``, (3, -2) for camera 2. That
checks both terms and the sign of the rotated displacement; a plain transpose
would give (3, 2).

Usage:
    pytest unit-tests/test_readers_end_to_end.py -v
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fixtures.lavision import build_set, clean_set_cache, write_manifest  # noqa: F401

from pivtools_core.config import Config

# Both cameras' sensors are 64 rows x 96 columns. Camera 2's quarter turn
# presents 96 x 64, the transpose: a mixed-shape rig made by rotation alone.
SENSOR_SHAPE = (64, 96)
OUTPUT_SHAPES = {1: (64, 96), 2: (96, 64)}

# Stored (sensor) frame displacement (rows, columns) from A to B, per camera.
# Both components non-zero, so a rotation changes both and flips one sign.
SENSOR_SHIFT = {1: (2, 3), 2: (2, 3)}

# rotate-cw-90 as lvpyio measures it (rotation-shape REPORT A1): the sensor's
# top-left pixel lands top-right. Camera 1 has no transformer.
# test_load_images_reads_each_camera_at_its_own_shape pins these to the pixels
# production returns, so the displacement derived from them is production's.
TO_OUTPUT = {1: lambda x: x, 2: lambda x: np.rot90(x, -1)}


def _output_shift(transform, sensor_shape, stored_shift):
    """The (rows, columns) displacement in the output frame of a stored-frame
    displacement, found on an index map: label every sensor pixel, apply the
    camera's transform, and locate an interior pixel and its shifted partner.
    """
    h, w = sensor_shape
    index = np.arange(h * w).reshape(h, w)
    out = transform(index)
    r, c = h // 2, w // 2
    before = np.argwhere(out == index[r, c])[0]
    after = np.argwhere(out == index[r + stored_shift[0], c + stored_shift[1]])[0]
    return tuple(int(v) for v in after - before)


def _expected_vector(output_shift):
    """(ux, uy) of an output-frame (rows, columns) shift. Uncalibrated vectors
    are in image pixels, ux along columns and uy along rows (down). Measured:
    a stored shift of +3 rows on an unrotated camera gave uy = +3.00. Camera 1
    of the rotated rig re-checks it on every run."""
    rows, cols = output_shift
    return float(cols), float(rows)


NUM_PAIRS = 2

# Particle images: ~0.03 particles per pixel, Gaussian of sigma 0.8 px
# (diameter ~3 px at e^-2, inside the 1.5-5 px range of Morgan's data), peak
# 4000 counts, seeded so a failure reproduces.
PARTICLE_DENSITY = 0.03
PARTICLE_SIGMA_PX = 0.8
PARTICLE_PEAK = 4000.0
# Particles are seeded this far outside the frame so the shifted frame is full.
SEED_MARGIN_PX = 8
SEED = 11


def _render(shape, rows, cols):
    """uint16 image of Gaussian particles centred at (rows, cols)."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = np.zeros(shape, np.float64)
    for r, c in zip(rows, cols):
        img += np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * PARTICLE_SIGMA_PX**2))
    return np.clip(PARTICLE_PEAK * img, 0, 65535).astype("<u2")


def _particle_frames(shape, shift, n_frames, rng):
    """``n_frames`` sensor frames of one particle field, frame k moved k*shift."""
    n = int(PARTICLE_DENSITY * shape[0] * shape[1])
    rows = rng.uniform(-SEED_MARGIN_PX, shape[0] + SEED_MARGIN_PX, n)
    cols = rng.uniform(-SEED_MARGIN_PX, shape[1] + SEED_MARGIN_PX, n)
    return [
        _render(shape, rows + k * shift[0], cols + k * shift[1])
        for k in range(n_frames)
    ]


def _rotated_rig(tmp_path, pre_paired):
    """A raw-16-bit ``.set`` of two cameras, camera 2 under ``rotate-cw-90``.

    Pre-paired: streams (0, 1) are camera 1's A and B, (2, 3) camera 2's, one
    entry per pair. Time-resolved: stream 0 is camera 1, stream 1 camera 2, one
    entry per frame, so NUM_PAIRS + 1 entries give NUM_PAIRS overlapping pairs.

    Returns:
        (set_file, sensor): sensor[camera] is the list of sensor-coordinate
        frames in read order (A, B, A, B, ... pre-paired; 1, 2, 3 time-resolved).
    """
    rng = np.random.default_rng(SEED)
    sensor = {}
    streams = []
    for cam in (1, 2):
        if pre_paired:
            pairs = [
                _particle_frames(SENSOR_SHAPE, SENSOR_SHIFT[cam], 2, rng)
                for _ in range(NUM_PAIRS)
            ]
            sensor[cam] = [frame for pair in pairs for frame in pair]
            streams += [
                [p[0].tobytes() for p in pairs],
                [p[1].tobytes() for p in pairs],
            ]
        else:
            frames = _particle_frames(
                SENSOR_SHAPE, SENSOR_SHIFT[cam], NUM_PAIRS + 1, rng
            )
            sensor[cam] = frames
            streams.append([f.tobytes() for f in frames])

    h, w = SENSOR_SHAPE
    set_file = build_set(tmp_path, "raw-16-bit", streams, w, h, name="rotated")
    cam2_streams = (2, 3) if pre_paired else (1,)
    write_manifest(
        tmp_path / "rotated",
        [(f"Frame{k}", k) for k in range(len(streams))],
        transformers=[
            ("rotate-cw-90", f"Transformer{n}", k) for n, k in enumerate(cam2_streams)
        ],
    )
    return set_file, sensor


def _set_config(
    tmp_path, set_file, camera_count, camera_numbers, pre_paired, num_images
):
    """A ``.set`` config ready for an instantaneous run, loaded as ``Config``."""
    base = tmp_path / "output"
    base.mkdir()
    cfg = {
        "calibration": {"active": "scale_factor"},
        "paths": {
            "source_paths": [str(set_file)],
            "base_paths": [str(base)],
            "camera_count": camera_count,
            "camera_numbers": list(camera_numbers),
        },
        "images": {
            "num_images": num_images,
            "image_format": [set_file.name],
            "image_type": "lavision_set",
            "start_index": 1,
            "frame_stride": 0 if pre_paired else 1,
            "pair_stride": 1,
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


def _rotated_config(tmp_path, pre_paired):
    set_file, sensor = _rotated_rig(tmp_path, pre_paired)
    num_images = NUM_PAIRS if pre_paired else NUM_PAIRS + 1
    cfg = _set_config(tmp_path, set_file, 2, [1, 2], pre_paired, num_images)
    return cfg, set_file, sensor


MODES = pytest.mark.parametrize(
    "pre_paired", [True, False], ids=["pre-paired", "time-resolved"]
)


# ---------------------------------------------------------------------------
# One camera rotated: shape, pixels, and an instantaneous run over both
# ---------------------------------------------------------------------------


@MODES
def test_rotated_camera_reports_the_transposed_shape(
    tmp_path, clean_set_cache, pre_paired
):
    cfg, _set_file, _sensor = _rotated_config(tmp_path, pre_paired)
    assert cfg.image_shape_for(1) == (64, 96)
    assert cfg.image_shape_for(2) == (96, 64)


def test_output_shift_of_a_quarter_turn_is_the_rotated_stored_shift():
    """The index-map derivation against the written-out rule for
    ``np.rot90(x, -1)``: a stored (dy, dx) appears as (dx, -dy). Two
    independent routes to the expected vector must agree, and the rig's own
    stored shift must come out different from itself and from its transpose."""
    for dy, dx in [(2, 3), (-1, 4), (3, 0), (0, 3)]:
        assert _output_shift(TO_OUTPUT[2], SENSOR_SHAPE, (dy, dx)) == (dx, -dy)
        assert _output_shift(TO_OUTPUT[1], SENSOR_SHAPE, (dy, dx)) == (dy, dx)
    rotated = _output_shift(TO_OUTPUT[2], SENSOR_SHAPE, SENSOR_SHIFT[2])
    assert rotated not in (SENSOR_SHIFT[2], SENSOR_SHIFT[2][::-1])


@MODES
def test_load_images_reads_each_camera_at_its_own_shape(
    tmp_path, clean_set_cache, pre_paired
):
    """Each camera's lazy batch array has its output shape, equals the GUI's
    ``read_pair`` pair by pair, and equals the sensor frames under the
    camera's transformer."""
    from pivtools_core.image_handling.load_images import load_images, read_pair

    cfg, set_file, sensor = _rotated_config(tmp_path, pre_paired)
    for cam, (h, w) in OUTPUT_SHAPES.items():
        images = load_images(cam, cfg, source=set_file, batch_size=1)
        assert images.shape == (NUM_PAIRS, 2, h, w), f"camera {cam}"
        computed = images.compute()
        for p in range(NUM_PAIRS):
            np.testing.assert_array_equal(
                computed[p], read_pair(p + 1, set_file, cam, cfg)
            )
            a, b = (2 * p, 2 * p + 1) if pre_paired else (p, p + 1)
            for frame, k in ((0, a), (1, b)):
                np.testing.assert_array_equal(
                    computed[p, frame],
                    TO_OUTPUT[cam](sensor[cam][k]).astype(np.float32),
                )


# Window grid by hand for a 32 px window at 50 % overlap (16 px spacing, first
# centre at 16 px): 64 px holds centres 16, 32, 48 -> 3 windows; 96 px holds
# 16 ... 80 -> 5 windows. Camera 2's grid is camera 1's transposed.
EXPECTED_GRID = {1: (3, 5), 2: (5, 3)}
# One 32 px pass with a 3-point Gaussian peak fit on clean synthetic particles;
# the median over the grid is well inside this.
SHIFT_TOLERANCE_PX = 0.25


def _run_instantaneous(cfg, cameras):
    """The body of ``instantaneous.main``'s camera loop, for ``cameras`` on one
    Config and one in-process Dask client. Returns {camera: output folder}."""
    from dask.distributed import Client

    from pivtools_cli.piv.save_results import get_output_path
    from pivtools_cli.piv_cluster.cluster import LOOPBACK_ANY_PORT
    from pivtools_core.image_handling.load_images import load_mask_for_camera
    from pivtools_core.instantaneous import run_instantaneous_piv

    client = Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=LOOPBACK_ANY_PORT,
        worker_dashboard_address=LOOPBACK_ANY_PORT,
    )
    outputs = {}
    try:
        for cam in cameras:
            output_path = get_output_path(
                cfg, cam, use_uncalibrated=True, piv_type="instantaneous"
            )
            run_instantaneous_piv(
                config=cfg,
                client=client,
                camera_num=cam,
                source_path=cfg.source_paths[0],
                output_path=output_path,
                base_path=cfg.base_paths[0],
                vector_masks=None,
                pixel_mask=load_mask_for_camera(cam, cfg),
            )
            outputs[cam] = output_path
    finally:
        client.close()
    return outputs


def _median_displacement(output_path, pair):
    """Median (ux, uy) and grid shape of pass 1 of ``B{pair:05d}.mat``."""
    from pivtools_core.vector_loading import read_mat_contents

    arr = read_mat_contents(str(output_path / f"B{pair:05d}.mat"), run_index=0)
    ux, uy = arr[0, 0], arr[0, 1]
    return float(np.nanmedian(ux)), float(np.nanmedian(uy)), ux.shape


def _assert_vectors(output_path, pair, grid, expected, label):
    """Pass 1 of ``pair`` is on ``grid`` and its median (ux, uy) is ``expected``."""
    ux, uy, got_grid = _median_displacement(output_path, pair)
    assert got_grid == grid, f"{label} pair {pair}: grid {got_grid}"
    assert (
        abs(ux - expected[0]) < SHIFT_TOLERANCE_PX
    ), f"{label} pair {pair}: ux {ux}, expected {expected[0]}"
    assert (
        abs(uy - expected[1]) < SHIFT_TOLERANCE_PX
    ), f"{label} pair {pair}: uy {uy}, expected {expected[1]}"


def _rotated_rig_vector(cam):
    """Camera ``cam``'s expected (ux, uy): its stored shift through its transform."""
    return _expected_vector(
        _output_shift(TO_OUTPUT[cam], SENSOR_SHAPE, SENSOR_SHIFT[cam])
    )


@MODES
def test_instantaneous_runs_both_cameras_each_on_its_own_grid(
    tmp_path, clean_set_cache, pre_paired
):
    """Both cameras in one job. Each camera's vectors land on its own grid and
    equal its stored-frame shift carried through its own transformer: camera 1
    (no transformer) reports the stored shift, camera 2 the cw-90-rotated one.
    Without the rotation camera 2 would report camera 1's vector."""
    cfg, _set_file, _sensor = _rotated_config(tmp_path, pre_paired)
    outputs = _run_instantaneous(cfg, [1, 2])
    for pair in range(1, NUM_PAIRS + 1):
        for cam in (1, 2):
            _assert_vectors(
                outputs[cam],
                pair,
                EXPECTED_GRID[cam],
                _rotated_rig_vector(cam),
                f"camera {cam}",
            )


# A cold interpreter, a LocalCluster and two cameras took 11 s on the
# workstation; a hang is a failure, not a wait.
CLI_TIMEOUT_S = 300


@pytest.mark.slow
def test_cli_instantaneous_processes_both_cameras(tmp_path, clean_set_cache):
    """``pivtools-cli instantaneous`` in a subprocess, from the config's folder,
    as a user runs it: it exits 0 (a failed camera exits non-zero, P4) and
    each camera's files sit on its own grid. Camera and path overrides are
    cleared, so a stale one cannot narrow the run to one camera."""
    cfg, _set_file, _sensor = _rotated_config(tmp_path, pre_paired=True)
    worktree = Path(__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(worktree))
    env.pop("PIV_CAMERA", None)
    env.pop("PIV_ACTIVE_PATHS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pivtools_cli.cli", "instantaneous"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    for cam in (1, 2):
        output_path = (
            cfg.base_paths[0] / "uncalibrated_piv" / str(NUM_PAIRS) / f"Cam{cam}"
        ) / "instantaneous"
        for pair in range(1, NUM_PAIRS + 1):
            assert (output_path / f"B{pair:05d}.mat").is_file(), (
                f"camera {cam} pair {pair} not written:\n" + result.stderr[-3000:]
            )
            _assert_vectors(
                output_path,
                pair,
                EXPECTED_GRID[cam],
                _rotated_rig_vector(cam),
                f"CLI camera {cam}",
            )


# ---------------------------------------------------------------------------
# NTNU-shaped rig: 2 PIV double-frame cameras, 2 PLIF single-frame cameras
# ---------------------------------------------------------------------------

# The NTNU sample's camera keys (fixtures/davis_attr, attr-decode REPORT §2 C2),
# one (Acq.Camera.Label, Acq.Camera.ID, CameraName) per frame stream.
NTNU_KEYS = [
    ("Camera 1", "Camera.ImagerMX25M: 290187", "1A: Imager MX 25M"),
    ("Camera 1", "Camera.ImagerMX25M: 290187", "1B: Imager MX 25M"),
    ("Camera 2", "Camera.ImagerMX25M: 290186", "2A: Imager MX 25M"),
    ("Camera 2", "Camera.ImagerMX25M: 290186", "2B: Imager MX 25M"),
    ("Camera 3", "Camera.ImagerSCMOS5MCLHS: 61008639", "3: Imager sCMOS CLHS"),
    ("Camera 4", "Camera.ImagerSCMOS5MCLHS: 61009193", "4: Imager sCMOS CLHS"),
]
# Scaled down: the PIV sensors share one size, the PLIF sensors another.
PIV_SHAPE = (64, 96)
PLIF_SHAPE = (48, 64)
# Every stream holds this many images: 3 A/B pairs for the PIV cameras, and 3
# frames (2 overlapping pairs) for a PLIF camera processed time-resolved.
NTNU_IMAGES = 3
# Each camera's stored-frame motion (rows, columns), different per camera so a
# camera reading another's streams shows. No transformers on this rig.
NTNU_SHIFT = {1: (3, 0), 2: (0, 3), 3: (2, -3), 4: (0, 3)}
NTNU_SHAPES = {1: PIV_SHAPE, 2: PIV_SHAPE, 3: PLIF_SHAPE, 4: PLIF_SHAPE}


def _ntnu_vector(cam):
    """Camera ``cam``'s expected (ux, uy): its stored shift, no transformer."""
    return _expected_vector(
        _output_shift(lambda x: x, NTNU_SHAPES[cam], NTNU_SHIFT[cam])
    )


# 32 px windows at 50 % overlap: 48 px -> 2 windows, 64 px -> 3, 96 px -> 5.
NTNU_GRID = {1: (3, 5), 2: (3, 5), 3: (2, 3)}


def _ntnu_rig(tmp_path):
    """A labelled raw-16-bit ``.set`` laid out like the NTNU PIV plus PLIF rig.

    Streams 0-3 are the PIV cameras' A and B frames (one entry per pair),
    streams 4 and 5 the PLIF cameras (one entry per frame), each with a
    ``MetaData{k}.attr`` carrying its camera keys on every image.

    Returns:
        (set_file, sensor): sensor[camera] lists its frames in read order.
    """
    from fixtures.lavision import camera_keys, write_attr

    rng = np.random.default_rng(SEED)
    sensor = {}
    streams = []
    shapes = []
    for cam in (1, 2):
        pairs = [
            _particle_frames(PIV_SHAPE, NTNU_SHIFT[cam], 2, rng)
            for _ in range(NTNU_IMAGES)
        ]
        sensor[cam] = [frame for pair in pairs for frame in pair]
        streams += [[p[0].tobytes() for p in pairs], [p[1].tobytes() for p in pairs]]
        shapes += [PIV_SHAPE, PIV_SHAPE]
    for cam in (3, 4):
        sensor[cam] = _particle_frames(PLIF_SHAPE, NTNU_SHIFT[cam], NTNU_IMAGES, rng)
        streams.append([f.tobytes() for f in sensor[cam]])
        shapes.append(PLIF_SHAPE)

    set_file = build_set(
        tmp_path, "raw-16-bit", streams, 0, 0, name="ntnu_like", shapes=shapes
    )
    folder = tmp_path / "ntnu_like"
    for k, keys in enumerate(NTNU_KEYS):
        write_attr(folder / f"MetaData{k}.attr", [camera_keys(*keys)] * NTNU_IMAGES)
    write_manifest(
        folder,
        [(f"Frame{k}", k) for k in range(len(streams))],
        attribute_readers=[(f"MetaData{k}", k) for k in range(len(NTNU_KEYS))],
    )
    return set_file, sensor


def test_ntnu_validation_finds_four_labelled_cameras(
    tmp_path, monkeypatch, clean_set_cache
):
    """The Setup page's ``/validate_files`` with cameras 1 and 2 selected
    pre-paired: both pass, and the stream table and the found line name the
    recording's four cameras from its labels."""
    from fixtures.diagnosis import piv_route_response

    set_file, _sensor = _ntnu_rig(tmp_path)
    response = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={
            "source_paths": [str(set_file)],
            "camera_count": 4,
            "camera_numbers": [1, 2],
        },
        images={
            "image_format": [set_file.name],
            "image_type": "lavision_set",
            "num_images": NTNU_IMAGES,
            "frame_stride": 0,
            "pair_stride": 1,
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["valid"] is True
    for cam in (1, 2):
        detail = body["details"][f"camera_{cam}"]
        assert detail["status"] == "ok", detail["error"]
        assert [s["camera"] for s in detail["streams"]] == [1, 1, 2, 2, 3, 4]
        assert [s["role"] for s in detail["streams"]] == ["A", "B", "A", "B", "", ""]
        assert "4 cameras from the recording's own labels" in detail["found"]


def test_ntnu_piv_runs_on_the_two_piv_cameras(tmp_path, clean_set_cache):
    set_file, _sensor = _ntnu_rig(tmp_path)
    cfg = _set_config(tmp_path, set_file, 4, [1, 2], True, NTNU_IMAGES)
    outputs = _run_instantaneous(cfg, [1, 2])
    for cam in (1, 2):
        for pair in range(1, NTNU_IMAGES + 1):
            _assert_vectors(
                outputs[cam], pair, NTNU_GRID[cam], _ntnu_vector(cam), f"camera {cam}"
            )


def test_ntnu_plif_camera_is_refused_for_ab_pairing_by_name(tmp_path, clean_set_cache):
    """Camera 3 selected for pre-paired PIV: the batch read the run would make
    refuses it, naming the camera, its identity and why."""
    from pivtools_core.image_handling.load_images import load_images, read_pair
    from pivtools_core.image_handling.readers.set_reader import SetPairingError

    set_file, _sensor = _ntnu_rig(tmp_path)
    cfg = _set_config(tmp_path, set_file, 4, [3], True, NTNU_IMAGES)
    for read in (
        lambda: read_pair(1, set_file, 3, cfg),
        lambda: load_images(3, cfg, source=set_file, batch_size=1).compute(),
    ):
        with pytest.raises(SetPairingError) as exc:
            read()
        message = str(exc.value)
        assert message.startswith("Camera 3 of ntnu_like.set")
        assert "Camera.ImagerSCMOS5MCLHS: 61008639" in message
        assert "is a single-frame camera, so it has no A/B pairs" in message


def test_ntnu_plif_camera_runs_time_resolved(tmp_path, clean_set_cache):
    """The same camera processed time-resolved: frames 1+2 and 2+3 read at
    its own shape and run through instantaneous PIV."""
    from pivtools_core.image_handling.load_images import load_images

    set_file, sensor = _ntnu_rig(tmp_path)
    cfg = _set_config(tmp_path, set_file, 4, [3], False, NTNU_IMAGES)
    assert cfg.image_shape_for(3) == PLIF_SHAPE
    computed = load_images(3, cfg, source=set_file, batch_size=1).compute()
    assert computed.shape == (NTNU_IMAGES - 1, 2, *PLIF_SHAPE)
    np.testing.assert_array_equal(computed[1, 1], sensor[3][2].astype(np.float32))

    outputs = _run_instantaneous(cfg, [3])
    for pair in range(1, NTNU_IMAGES):
        _assert_vectors(outputs[3], pair, NTNU_GRID[3], _ntnu_vector(3), "camera 3")
