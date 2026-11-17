"""DaVis stream transformers: the table, LaVision's fixed order, and output shape.

StreamSet.xml binds post-decode transformers to a frame stream. lvpyio 1.3.1
(LaVision's reader) applies them in a fixed order whatever the manifest lists:
intensity-correction-1, dark subtraction, background subtraction, each clipped
per step, in sensor coordinates; then at most one geometry. A 90 degree geometry
makes the stream read as (W, H) of its index header, so every caller sizes
buffers and splits cameras on the OUTPUT shape.

Evidence: rotation-shape and transformer-order follow-up REPORTs (2026-11-16).
The committed oracle ``fixtures/lavision_transformers/lvpyio_oracle.npz`` holds
the exact bytes lvpyio read and its frames (``make_oracle.py`` regenerates it).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fixtures.lavision import (
    build_set,
    make_set_config,
    write_dark_im7,
    write_manifest,
)

from pivtools_core.image_handling.load_images import read_pair
from pivtools_core.image_handling.readers.set_reader import (
    SetConfigError,
    camera_streams,
    clear_set_info_cache,
    read_set_frame,
    read_set_info,
    read_set_pair,
    set_camera_streams,
)
from pivtools_core.image_handling.source_diagnosis import set_stream_infos


@pytest.fixture(autouse=True)
def _clean_set_cache():
    """Empty the process-wide SetInfo cache around every test: the cache is keyed
    on path and mtimes, and tests rewrite manifests in place."""
    clear_set_info_cache()
    yield
    clear_set_info_cache()


# The 3 x 5 sensor frame of the hand-written cases: unique values, row 0 at top.
X35 = np.array(
    [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ],
    dtype=np.uint16,
)

# What each geometric id makes of X35, written out by hand from the rotation-shape
# REPORT table A1 (where the sensor's top-left pixel lands), not from numpy.
GEOMETRY_EXPECTED = {
    "rotate-180": [[14, 13, 12, 11, 10], [9, 8, 7, 6, 5], [4, 3, 2, 1, 0]],
    "vertical-mirror": [[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
    "horinzontal-mirror": [[4, 3, 2, 1, 0], [9, 8, 7, 6, 5], [14, 13, 12, 11, 10]],
    # Clockwise on screen: top-left lands top-right.
    "rotate-cw-90": [[10, 5, 0], [11, 6, 1], [12, 7, 2], [13, 8, 3], [14, 9, 4]],
    # Counter-clockwise: top-left lands bottom-left.
    "rotate-ccw-90": [[4, 9, 14], [3, 8, 13], [2, 7, 12], [1, 6, 11], [0, 5, 10]],
    # Top-left lands bottom-right.
    "rotate-cw-90-vertical-mirror": [
        [14, 9, 4],
        [13, 8, 3],
        [12, 7, 2],
        [11, 6, 1],
        [10, 5, 0],
    ],
    # The transpose: top-left stays top-left.
    "rotate-ccw-90-vertical-mirror": [
        [0, 5, 10],
        [1, 6, 11],
        [2, 7, 12],
        [3, 8, 13],
        [4, 9, 14],
    ],
}
QUARTER_TURNS = {
    "rotate-cw-90",
    "rotate-ccw-90",
    "rotate-cw-90-vertical-mirror",
    "rotate-ccw-90-vertical-mirror",
}
CORRECTION_FILES = {
    "dark-image-subtraction": "dark",
    "background-subtraction": "bgnd",
    "intensity-correction-1": "scmos-1",
}
REFUSED_IDS = ("white-image-correction", "add-offset", "image-intensity-correction-2")


def _one_stream_set(tmp_path, frame, transformers=(), corrections=None, name="t"):
    """A one-stream raw-16-bit recording of ``frame`` with the given transformers.

    Args:
        transformers: (id, FilePrefix) in manifest order, all bound to stream 0.
        corrections: {FilePrefix-suffix.im7 file name: uint16 image} to write.
    """
    h, w = frame.shape
    set_file = build_set(
        tmp_path, "raw-16-bit", [[frame.astype("<u2").tobytes()]], w, h, name=name
    )
    set_dir = tmp_path / name
    for file_name, image in (corrections or {}).items():
        write_dark_im7(set_dir / file_name, image)
    write_manifest(
        set_dir,
        [("Frame0", 0)],
        transformers=[(tid, prefix, 0) for tid, prefix in transformers],
    )
    return set_file


ORACLE = (
    Path(__file__).parent / "fixtures" / "lavision_transformers" / "lvpyio_oracle.npz"
)


def _oracle_cases():
    with np.load(ORACLE) as z:
        return sorted({k.split("__")[0] for k in z.files})


def unpack_oracle_case(case, dest):
    """Write one archived recording into ``dest``; return (set file, lvpyio frames, meta)."""
    with np.load(ORACLE) as z:
        for key in z.files:
            if key.startswith(f"{case}__file__"):
                rel = key.split("__file__", 1)[1].replace("~", "/")
                path = dest / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z[key].tobytes())
        frames = z[f"{case}__lvpyio"]
        meta = json.loads(str(z[f"{case}__meta"]))
    return dest / "rec.set", frames, meta


# ---------------------------------------------------------------------------
# lvpyio oracle: every archived case, bit for bit, shape included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _oracle_cases())
def test_read_equals_lvpyio(case, tmp_path):
    set_file, lv_frames, meta = unpack_oracle_case(case, tmp_path)

    info = read_set_info(set_file)
    assert info.frames[0].shape == lv_frames.shape[1:], meta
    for entry, want in enumerate(lv_frames, start=1):
        got = read_set_frame(set_file, entry, 0)
        assert got.shape == want.shape
        np.testing.assert_array_equal(got, want.astype(np.float32), err_msg=case)


def test_manifest_order_and_prefixes_do_not_change_the_read(tmp_path):
    """Dark + a quarter turn in both manifest orders and both prefix schemes reads
    one array (transformer-order REPORT §6 (c) 2)."""
    dark = (X35 % 4).astype(np.uint16)
    dark_id = "dark-image-subtraction"
    for geo in ("rotate-cw-90", "rotate-ccw-90"):
        reads = []
        for k, (first, second) in enumerate(
            [
                ((dark_id, "Transformer0"), (geo, "Transformer1")),
                ((dark_id, "Transformer1"), (geo, "Transformer0")),
                ((geo, "Transformer1"), (dark_id, "Transformer0")),
                ((geo, "Transformer0"), (dark_id, "Transformer1")),
            ]
        ):
            dark_prefix = first[1] if first[0] == dark_id else second[1]
            set_file = _one_stream_set(
                tmp_path,
                X35 + 3,
                [first, second],
                {f"{dark_prefix}-dark.im7": dark},
                name=f"{geo}_{k}",
            )
            reads.append(read_set_frame(set_file, 1, 0))
        want = np.array(GEOMETRY_EXPECTED[geo], np.float32) + 3
        want -= np.array(
            {"rotate-cw-90": np.rot90(dark, -1), "rotate-ccw-90": np.rot90(dark, 1)}[
                geo
            ],
            np.float32,
        )
        for got in reads:
            np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Each id by hand: geometry and the three corrections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tid", sorted(GEOMETRY_EXPECTED))
def test_geometry_by_hand(tid, tmp_path):
    set_file = _one_stream_set(tmp_path, X35, [(tid, "Transformer0")])
    want = np.array(GEOMETRY_EXPECTED[tid], dtype=np.float32)

    info = read_set_info(set_file)
    assert info.frames[0].stored_shape == (3, 5)
    assert info.frames[0].shape == want.shape
    np.testing.assert_array_equal(read_set_frame(set_file, 1, 0), want)

    # Into a caller's buffer of the output shape (the time-resolved PIV path).
    out = np.empty(want.shape, np.float32)
    read_set_frame(set_file, 1, 0, out=out)
    np.testing.assert_array_equal(out, want)


def test_every_geometry_output_shape_agrees_with_frame_info(tmp_path):
    """On a 4 x 7 frame, the read and IMSFrameInfo.shape are (7, 4) exactly for the
    four quarter turns (rotation-shape REPORT §6 item 3)."""
    frame = np.arange(28, dtype=np.uint16).reshape(4, 7)
    for tid in GEOMETRY_EXPECTED:
        set_file = _one_stream_set(tmp_path, frame, [(tid, "T0")], name=tid)
        shape = read_set_info(set_file).frames[0].shape
        assert read_set_frame(set_file, 1, 0).shape == shape
        assert shape == ((7, 4) if tid in QUARTER_TURNS else (4, 7)), tid


# Two rows: the correction loader squeezes a one-row image to 1-D.
RAW_CORR = np.array(
    [[100, 50, 10, 65535, 65400], [0, 1, 299, 300, 301]], dtype=np.uint16
)
CORR = np.array([[20, 20, 20, 0, 299], [0, 300, 300, 300, 0]], dtype=np.uint16)


@pytest.mark.parametrize(
    "tid, want",
    [
        # max(raw - C, 0): 10 - 20 and 1 - 300 clip to 0, never wrap.
        ("dark-image-subtraction", [[80, 30, 0, 65535, 65101], [0, 0, 0, 0, 301]]),
        ("background-subtraction", [[80, 30, 0, 65535, 65101], [0, 0, 0, 0, 301]]),
        # clip(raw - C + 300, 0, 65535): 65535 + 300 saturates; 65400 - 299 + 300
        # is 65401; 1 - 300 + 300 is 1.
        (
            "intensity-correction-1",
            [[380, 330, 290, 65535, 65401], [300, 1, 299, 300, 601]],
        ),
    ],
)
def test_correction_by_hand(tid, want, tmp_path):
    set_file = _one_stream_set(
        tmp_path,
        RAW_CORR,
        [(tid, "Transformer0")],
        {f"Transformer0-{CORRECTION_FILES[tid]}.im7": CORR},
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, 1, 0), np.array(want, dtype=np.float32)
    )


def test_corrections_clip_per_step_in_lavision_order(tmp_path):
    """ic1 runs first and clips, then dark (transformer-order REPORT §6 (c) 3).

    Raw 65500, T 100, D 50: per step clip(65500 - 100 + 300) = 65535, minus 50 is
    65485. Clipping once at the end would give 65535. Dark is listed first in the
    manifest, and manifest order would give clip(65450 - 100 + 300) = 65535 too.
    """
    raw = np.array([[65500, 1000, 0], [65500, 1000, 0]], dtype=np.uint16)
    ic1 = np.array([[100, 100, 400], [100, 100, 400]], dtype=np.uint16)
    dark = np.array([[50, 50, 0], [50, 50, 0]], dtype=np.uint16)
    set_file = _one_stream_set(
        tmp_path,
        raw,
        [
            ("dark-image-subtraction", "Transformer0"),
            ("intensity-correction-1", "Transformer1"),
        ],
        {"Transformer0-dark.im7": dark, "Transformer1-scmos-1.im7": ic1},
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, 1, 0), np.array([[65485, 1150, 0]] * 2, np.float32)
    )


def test_correction_runs_in_sensor_shape_before_a_quarter_turn(
    tmp_path,
):
    """Dark is subtracted from the stored frame, then the frame turns."""
    dark = np.zeros((3, 5), np.uint16)
    dark[0, 0] = 7  # sensor top-left only
    set_file = _one_stream_set(
        tmp_path,
        X35 + 10,
        [
            ("rotate-cw-90", "Transformer1"),
            ("dark-image-subtraction", "Transformer0"),
        ],
        {"Transformer0-dark.im7": dark},
    )
    want = np.array(GEOMETRY_EXPECTED["rotate-cw-90"], np.float32) + 10
    want[0, 2] -= 7  # where the sensor top-left lands after cw90
    np.testing.assert_array_equal(read_set_frame(set_file, 1, 0), want)


# ---------------------------------------------------------------------------
# Refusals by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 3), (4, 5)])
def test_correction_not_in_sensor_shape_is_refused(shape, tmp_path):
    """A transposed (output-shaped) dark on a quarter-turn stream and a wrong
    size are both refused, though lvpyio reads the transposed one in the sensor
    raster (transformer-order REPORT Q1b, §6 (c) 5)."""
    set_file = _one_stream_set(
        tmp_path,
        X35,
        [
            ("dark-image-subtraction", "Transformer0"),
            ("rotate-cw-90", "Transformer1"),
        ],
        {"Transformer0-dark.im7": np.zeros(shape, np.uint16)},
    )
    with pytest.raises(ValueError, match="does not belong to this recording") as exc:
        read_set_frame(set_file, 1, 0)
    assert "Transformer0-dark.im7" in str(exc.value)
    assert "(3, 5)" in str(exc.value)


def test_two_geometries_on_one_stream_are_refused(tmp_path):
    set_file = _one_stream_set(
        tmp_path,
        X35,
        [("rotate-cw-90", "Transformer0"), ("vertical-mirror", "Transformer1")],
    )
    with pytest.raises(ValueError, match="two geometric transformers") as exc:
        read_set_info(set_file)
    assert "'rotate-cw-90'" in str(exc.value)
    assert "'vertical-mirror'" in str(exc.value)


@pytest.mark.parametrize("tid", ["dark-image-subtraction", "rotate-180"])
def test_repeated_id_on_one_stream_is_refused(tid, tmp_path):
    corrections = None
    if tid == "dark-image-subtraction":
        corrections = {
            "Transformer0-dark.im7": np.zeros((3, 5), np.uint16),
            "Transformer1-dark.im7": np.zeros((3, 5), np.uint16),
        }
    set_file = _one_stream_set(
        tmp_path, X35, [(tid, "Transformer0"), (tid, "Transformer1")], corrections
    )
    with pytest.raises(ValueError, match=f"declares '{tid}' twice for frame stream 0"):
        read_set_info(set_file)


@pytest.mark.parametrize("tid", ["background-subtraction", "intensity-correction-1"])
def test_missing_correction_file_is_refused_at_parse(tid, tmp_path):
    """rotation-shape REPORT §6 item 8."""
    set_file = _one_stream_set(tmp_path, X35, [(tid, "Transformer3")])
    with pytest.raises(
        FileNotFoundError, match=f"Transformer3-{CORRECTION_FILES[tid]}.im7"
    ):
        read_set_info(set_file)


@pytest.mark.parametrize(
    "tid", ["rotate-45", "horizontal-mirror", "flat-field-correction"]
)
def test_unknown_id_refuses_the_whole_recording(tid, tmp_path):
    """Without camera labels, an unknown id makes every camera's stream
    mapping uncertain, so nothing is read (with labels it blocks only its
    camera: test_set_camera_identity.py). The correctly spelled
    'horizontal-mirror' is unknown: lvpyio opens 0 buffers for it."""
    frame = np.zeros((3, 5), np.uint16)
    set_file = build_set(
        tmp_path, "raw-16-bit", [[frame.tobytes()], [frame.tobytes()]], 5, 3
    )
    write_manifest(
        tmp_path / "synthetic",
        [("Frame0", 0), ("Frame1", 1)],
        transformers=[(tid, "Transformer1", 1)],
    )
    with pytest.raises(ValueError, match=f"'{tid}'") as exc:
        read_set_info(set_file)
    assert "frame stream 1" in str(exc.value)
    assert "no camera of this recording can be read" in str(exc.value)


@pytest.mark.parametrize("tid", REFUSED_IDS)
def test_refused_id_blocks_only_its_stream(tid, tmp_path):
    """rotation-shape REPORT §6 item 9: the other stream reads, its own refuses."""
    cam1 = np.arange(15, dtype=np.uint16).reshape(3, 5)
    set_file = build_set(
        tmp_path, "raw-16-bit", [[cam1.tobytes()], [cam1.tobytes()]], 5, 3
    )
    write_manifest(
        tmp_path / "synthetic",
        [("Frame0", 0), ("Frame1", 1)],
        transformers=[(tid, "Transformer1", 1)],
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, 1, 0), cam1.astype(np.float32)
    )
    with pytest.raises(ValueError) as exc:
        read_set_frame(set_file, 1, 1)
    message = str(exc.value)
    assert f"'{tid}'" in message and "Frame stream 1" in message
    assert "Other streams of this recording are unaffected" in message


# ---------------------------------------------------------------------------
# Output shape reaches every caller
# ---------------------------------------------------------------------------


def _two_streams_one_rotated(tmp_path, geometry="rotate-cw-90"):
    """Two raw-16-bit streams of stored 4 x 6, stream 1 with a quarter turn, two
    entries each, filled with 100*stream + entry."""
    streams = [
        [np.full((4, 6), 100 * k + e, "<u2").tobytes() for e in range(2)]
        for k in range(2)
    ]
    set_file = build_set(tmp_path, "raw-16-bit", streams, 6, 4)
    write_manifest(
        tmp_path / "synthetic",
        [("Frame0", 0), ("Frame1", 1)],
        transformers=[(geometry, "Transformer0", 1)],
    )
    return set_file


def test_frame_info_keeps_stored_and_output_shape(tmp_path):
    info = read_set_info(_two_streams_one_rotated(tmp_path))
    assert [f.stored_shape for f in info.frames] == [(4, 6), (4, 6)]
    assert [f.shape for f in info.frames] == [(4, 6), (6, 4)]


def test_split_uses_output_shapes(tmp_path):
    """Equal stored shapes, one stream rotated: not one camera's A/B pair
    (rotation-shape REPORT §6 item 5, C3)."""
    set_file = _two_streams_one_rotated(tmp_path)
    with pytest.raises(SetConfigError, match="Camera counts that fit: 2") as exc:
        set_camera_streams(set_file, 1)
    # The stream table names output size (width first) and the transformer.
    assert "1: 4×6 raw-16-bit rotate-cw-90" in str(exc.value)
    assert set_camera_streams(set_file, 2) == [(0,), (1,)]


def test_a_b_pair_of_rotated_streams_reads_at_output_shape(tmp_path):
    streams = [[np.full((4, 6), 100 * k, "<u2").tobytes()] for k in range(2)]
    set_file = build_set(tmp_path, "raw-16-bit", streams, 6, 4)
    write_manifest(
        tmp_path / "synthetic",
        [("Frame0", 0), ("Frame1", 1)],
        transformers=[
            ("rotate-ccw-90", "Transformer0", 0),
            ("rotate-ccw-90", "Transformer1", 1),
        ],
    )
    assert camera_streams(set_file, 1, 1, pair=True) == (0, 1)
    out = np.empty((2, 6, 4), np.float32)
    read_set_pair(set_file, (0, 1), 1, out=out)
    np.testing.assert_array_equal(out[1], np.full((6, 4), 100, np.float32))


def test_config_and_time_resolved_read_use_output_shape(tmp_path):
    set_file = _two_streams_one_rotated(tmp_path)
    cfg = make_set_config(tmp_path, set_file, camera_count=2, camera_numbers=[1, 2])
    assert cfg.image_shape_for(1) == (4, 6)
    assert cfg.image_shape_for(2) == (6, 4)

    pair = read_pair(1, set_file, 2, cfg)
    assert pair.shape == (2, 6, 4)
    np.testing.assert_array_equal(pair[0], np.full((6, 4), 100, np.float32))
    np.testing.assert_array_equal(pair[1], np.full((6, 4), 101, np.float32))

    out = np.empty((2, 6, 4), np.float32)
    assert read_pair(1, set_file, 2, cfg, out=out) is out


def test_config_shape_of_a_quarter_turn_needs_no_decode(tmp_path):
    """transformer-order REPORT §6 (c) 5: the data file is empty, so any decode
    would fail; the shape still comes out (W, H)."""
    set_file = build_set(tmp_path, "raw-16-bit", [[b""]], 6, 4, declared_sizes=[[48]])
    write_manifest(
        tmp_path / "synthetic",
        [("Frame0", 0)],
        transformers=[("rotate-cw-90", "T0", 0)],
    )
    cfg = make_set_config(tmp_path, set_file, camera_count=1, camera_numbers=[1])
    assert cfg.image_shape_for(1) == (6, 4)


def test_gui_stream_table_shows_output_shape(tmp_path):
    streams = set_stream_infos(_two_streams_one_rotated(tmp_path), 2)
    assert [(s["width"], s["height"], s["camera"]) for s in streams] == [
        (6, 4, 1),
        (4, 6, 2),
    ]
