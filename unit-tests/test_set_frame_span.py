"""A .set stream file that holds several frames of each image (frame span).

DaVis 11.4 writes a double-frame camera's A and B exposures into one stream file:
its StreamSet.xml FrameReader spans StartFrame..EndFrame, and index header word 2
holds the span length. Frame f of buffer b is entry ``b * word2 + (f - StartFrame)``
(container-unknowns REPORT §2 H1-H3; drive-survey REPORT §4 row 1).

The expected entries below are lvpyio 1.3.1's buffers for the five synthetic
layouts of container-unknowns REPORT §2 H1 (scripts/probe_frame_span.py), not a
re-derivation of the reader's formula. Every entry carries its own value, so each
frame read is traced to the one entry it came from.
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import (  # noqa: E402,F401
    clean_set_cache,
    ims_index,
    make_set_config,
    write_manifest,
    write_scales,
)

from pivtools_core.image_handling.load_images import read_pair  # noqa: E402
from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    camera_streams,
    get_set_entry_count,
    read_set_frame,
    read_set_info,
)

_H, _W = 2, 3


# Entry k of the i-th stream file holds 1000*i + 10*k plus the pixel's row-major
# position (0..5), so the top-left pixel names the entry and a rotation shows.
_RAMP = np.arange(_H * _W, dtype="<u2").reshape(_H, _W)


def _entry_base(prefix_no: int, entry: int) -> int:
    """Top-left pixel value of entry ``entry`` of stream file number ``prefix_no``."""
    return 1000 * prefix_no + 10 * entry


def _write_streams(tmp_path, readers, name="span", **manifest):
    """A raw-16-bit .set with one stream file per reader.

    Args:
        readers: (prefix, start_frame, end_frame, n_entries, word2) per stream
            file, in declaration order. Entry k of the i-th file is filled with
            ``_entry_base(i, k) + _RAMP``.
        manifest: scale_readers / transformers passed to ``write_manifest``.

    Returns:
        Path: the .set file.
    """
    set_file = tmp_path / f"{name}.set"
    set_file.write_bytes(b"stub")
    set_dir = tmp_path / name
    set_dir.mkdir()
    for i, (prefix, _start, _end, n_entries, word2) in enumerate(readers):
        payloads = [
            (_entry_base(i, k) + _RAMP).astype("<u2").tobytes()
            for k in range(n_entries)
        ]
        (set_dir / f"{prefix}-0.ims").write_bytes(
            ims_index(_W, _H, [len(p) for p in payloads], words={2: word2})
        )
        (set_dir / f"{prefix}-1.ims").write_bytes(b"".join(payloads))
        (set_dir / f"{prefix}-decoder.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<FrameDecoder>\n<id>raw-16-bit</id>\n</FrameDecoder>\n"
        )
    write_manifest(
        set_dir,
        frame_readers=[(p, s, e) for p, s, e, _n, _w in readers],
        **manifest,
    )
    return set_file


def _buffers(set_file):
    """Every frame of every buffer as (stream file number, entry) pairs."""
    info = read_set_info(set_file)
    out = []
    for b in range(info.n_entries):
        frames = []
        for f in range(len(info.frames)):
            value = int(read_set_frame(set_file, entry_no=b + 1, frame_idx=f)[0, 0])
            frames.append((value // 1000, value % 1000 // 10))
        out.append(frames)
    return out


# ---------------------------------------------------------------------------
# CU §6 test 5: the five H1 layouts map exactly as lvpyio maps them
# ---------------------------------------------------------------------------


def test_one_reader_spanning_two_frames(tmp_path, clean_set_cache):
    """H1 row 1: frames 0..1, word2 = 2, 6 entries -> [0,1], [2,3], [4,5]."""
    set_file = _write_streams(tmp_path, [("Camera1", 0, 1, 6, 2)])
    assert _buffers(set_file) == [
        [(0, 0), (0, 1)],
        [(0, 2), (0, 3)],
        [(0, 4), (0, 5)],
    ]


def test_spanning_reader_then_single_frame_reader(tmp_path, clean_set_cache):
    """H1 row 2: A frames 0..1 + B frame 2 -> [A0,A1,B0], [A2,A3,B1]."""
    set_file = _write_streams(tmp_path, [("A", 0, 1, 4, 2), ("B", 2, 2, 2, 1)])
    assert _buffers(set_file) == [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 2), (0, 3), (1, 1)],
    ]


def test_frame_position_not_declaration_order(tmp_path, clean_set_cache):
    """H1 row 3: B frame 0 declared after A frames 1..2 -> [B0,A0,A1], [B1,A2,A3]."""
    set_file = _write_streams(tmp_path, [("A", 1, 2, 4, 2), ("B", 0, 0, 2, 1)])
    assert _buffers(set_file) == [
        [(1, 0), (0, 0), (0, 1)],
        [(1, 1), (0, 2), (0, 3)],
    ]


def test_span_disagreeing_with_word2_is_refused(tmp_path, clean_set_cache):
    """H1 row 4: frames 0..1 with word2 = 1. lvpyio refuses every buffer
    ("FrameReader-FrameCount=1 / ContentPurpose-FrameCount=2")."""
    set_file = _write_streams(tmp_path, [("Camera1", 0, 1, 4, 1)])
    with pytest.raises(ValueError) as err:
        read_set_info(set_file)
    message = str(err.value)
    assert "word 2" in message and "Camera1" in message and "spans 2" in message


def test_one_reader_spanning_three_frames(tmp_path, clean_set_cache):
    """H1 row 5: frames 0..2, word2 = 3, 6 entries -> [0,1,2], [3,4,5]."""
    set_file = _write_streams(tmp_path, [("Camera1", 0, 2, 6, 3)])
    assert _buffers(set_file) == [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 3), (0, 4), (0, 5)],
    ]


# ---------------------------------------------------------------------------
# Layouts the span makes possible, refused by name
# ---------------------------------------------------------------------------


def test_partial_last_buffer_is_refused(tmp_path, clean_set_cache):
    """5 entries at word2 = 2 leave buffer 2 with frame 0 only. lvpyio counts
    3 buffers and fails reading the last ("Invalid argument: '!index'",
    container-unknowns REPORT §2 A1); the recording is refused at parse."""
    set_file = _write_streams(tmp_path, [("Camera1", 0, 1, 5, 2)])
    with pytest.raises(ValueError) as err:
        read_set_info(set_file)
    message = str(err.value)
    assert "Camera1-0.ims" in message and "5 entries" in message
    assert "2 frames per image" in message


def test_end_frame_before_start_frame_is_refused(tmp_path, clean_set_cache):
    set_file = _write_streams(tmp_path, [("Camera1", 1, 0, 2, 1)])
    with pytest.raises(ValueError, match=r"FrameReader 'Camera1'.*spans frames 1\.\.0"):
        read_set_info(set_file)


def test_overlapping_spans_are_refused(tmp_path, clean_set_cache):
    """A frames 0..1 and B frame 1 both claim frame 1."""
    set_file = _write_streams(tmp_path, [("A", 0, 1, 4, 2), ("B", 1, 1, 2, 1)])
    with pytest.raises(
        ValueError, match=r"two image streams to frame stream 1: 'A' and 'B'"
    ):
        read_set_info(set_file)


# ---------------------------------------------------------------------------
# Scale files and corrections bound per frame of a spanning stream
# ---------------------------------------------------------------------------


def test_each_frame_of_a_span_takes_its_own_scale_file(tmp_path, clean_set_cache):
    """145749's shape: one stream file spans A and B, CameraScale1 is bound to
    frame 0 and CameraScale2 to frame 1 (container-unknowns REPORT §2 H3)."""
    set_file = _write_streams(
        tmp_path,
        [("Camera1", 0, 1, 2, 2)],
        scale_readers=[("CameraScale1", 0), ("CameraScale2", 1)],
    )
    set_dir = set_file.with_suffix("")
    write_scales(set_dir / "CameraScale1.scales", factor=2.0)
    write_scales(set_dir / "CameraScale2.scales", factor=3.0)
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=0),
        (2.0 * (_entry_base(0, 0) + _RAMP)).astype(np.float32),
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=1),
        (3.0 * (_entry_base(0, 1) + _RAMP)).astype(np.float32),
    )


def test_correction_bound_to_one_frame_of_a_span(tmp_path, clean_set_cache):
    """A rotate-180 bound to frame 1 turns B only, though A shares its file."""
    set_file = _write_streams(
        tmp_path,
        [("Camera1", 0, 1, 2, 2)],
        transformers=[("rotate-180", "Transformer0", 1)],
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=0),
        (_entry_base(0, 0) + _RAMP).astype(np.float32),
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=1),
        (_entry_base(0, 1) + _RAMP[::-1, ::-1]).astype(np.float32),
    )


def test_scale_file_spanning_frames_is_refused(tmp_path, clean_set_cache):
    """Only FrameReaders span; a scale file bound to 0..1 is unseen."""
    set_file = _write_streams(
        tmp_path,
        [("Camera1", 0, 1, 2, 2)],
        scale_readers=[("CameraScale1", 0, 1)],
    )
    with pytest.raises(
        ValueError, match=r"ScaleReader 'CameraScale1'.*spans frames 0\.\.1"
    ):
        read_set_info(set_file)


# ---------------------------------------------------------------------------
# Pairing and counting: A and B of each camera in one stream file
# ---------------------------------------------------------------------------

# Two double-frame cameras, each writing A and B into one file (frames 0..1 and
# 2..3), three images: six entries per file.
_TWO_AB_CAMERAS = [("Camera1", 0, 1, 6, 2), ("Camera2", 2, 3, 6, 2)]
_N_IMAGES = 3


def _frame(prefix_no, entry):
    return (_entry_base(prefix_no, entry) + _RAMP).astype(np.float32)


def test_ab_stream_files_split_into_cameras(tmp_path, clean_set_cache):
    set_file = _write_streams(tmp_path, _TWO_AB_CAMERAS)
    assert get_set_entry_count(set_file) == _N_IMAGES
    assert camera_streams(set_file, 1, 2, pair=True) == (0, 1)
    assert camera_streams(set_file, 2, 2, pair=True) == (2, 3)


def test_pre_paired_read_pair_takes_a_and_b_from_one_entry_pair(
    tmp_path, clean_set_cache
):
    """Pair p of camera 2 is entries 2(p-1) and 2(p-1)+1 of Camera2-1.ims."""
    set_file = _write_streams(tmp_path, _TWO_AB_CAMERAS)
    cfg = make_set_config(tmp_path, set_file, 2, [1, 2], frame_stride=0)
    cfg.data["images"]["num_images"] = get_set_entry_count(set_file)
    assert cfg.num_frame_pairs == _N_IMAGES
    for p in range(1, _N_IMAGES + 1):
        pair = read_pair(p, set_file, 2, cfg)
        np.testing.assert_array_equal(pair[0], _frame(1, 2 * (p - 1)))
        np.testing.assert_array_equal(pair[1], _frame(1, 2 * (p - 1) + 1))


def test_time_resolved_read_pair_takes_frame_a_of_consecutive_images(
    tmp_path, clean_set_cache
):
    """Time-resolved pairs a camera's first frame across images: pair p of
    camera 1 is entries 2(p-1) and 2p of Camera1-1.ims (A of image p and p+1)."""
    set_file = _write_streams(tmp_path, _TWO_AB_CAMERAS)
    cfg = make_set_config(tmp_path, set_file, 2, [1, 2], frame_stride=1)
    cfg.data["images"]["num_images"] = get_set_entry_count(set_file)
    assert cfg.num_frame_pairs == _N_IMAGES - 1
    for p in range(1, _N_IMAGES):
        pair = read_pair(p, set_file, 1, cfg)
        np.testing.assert_array_equal(pair[0], _frame(0, 2 * (p - 1)))
        np.testing.assert_array_equal(pair[1], _frame(0, 2 * p))


def test_config_image_shape_of_an_ab_stream_camera(tmp_path, clean_set_cache):
    set_file = _write_streams(tmp_path, _TWO_AB_CAMERAS)
    cfg = make_set_config(tmp_path, set_file, 2, [2], frame_stride=0)
    assert cfg.image_shape_for(2) == (_H, _W)
