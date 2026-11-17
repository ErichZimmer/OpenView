"""Strict .ims index header and StreamSet.xml manifest for the LaVision .set reader.

Every field the reader does not decode must hold the value measured on real DaVis
recordings, or the recording is refused with the field and value named. The
measured values and lvpyio's behaviour for each other value are in
wiki/plans/2026-11-14-overnight-research/01-lavision-readers/followup/
container-unknowns/REPORT.md (§1 header table, §2 A, B, F, G). Each test below is
one of that report's §6 acceptance tests 1, 2, 4 and 6, with Morgan's 17 Nov calls
applied (word 1 must be 0, file number must be 1, foreign FrameReader names
refused, the Frame{N} name-guessing fallback removed).
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import (  # noqa: E402,F401
    build_set,
    clean_set_cache,
    ims_index,
    write_manifest,
)

from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    read_set_frame,
    read_set_info,
)

# 2x3 raw-16-bit frame, four entries, so a header can disagree with the entry count.
_H, _W = 2, 3
_FRAME_BYTES = _H * _W * 2
_N_ENTRIES = 4


def _one_stream_set(tmp_path, name="strict"):
    """A valid one-stream raw-16-bit .set of four distinct entries."""
    entries = [
        np.full((_H, _W), 10 * (k + 1), "<u2").tobytes() for k in range(_N_ENTRIES)
    ]
    set_file = build_set(tmp_path, "raw-16-bit", [entries], _W, _H, name=name)
    return set_file, tmp_path / name


def _rewrite_index(set_dir, **kwargs):
    """Replace Frame0-0.ims with a variant built by fixtures.lavision.ims_index."""
    (set_dir / "Frame0-0.ims").write_bytes(
        ims_index(_W, _H, [_FRAME_BYTES] * _N_ENTRIES, **kwargs)
    )


# ---------------------------------------------------------------------------
# Control: the measured header reads
# ---------------------------------------------------------------------------


def test_measured_header_reads(tmp_path, clean_set_cache):
    set_file, _ = _one_stream_set(tmp_path)
    info = read_set_info(set_file)
    assert info.n_entries == _N_ENTRIES
    assert [e.file_no for e in info.frames[0].entries] == [1] * _N_ENTRIES
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=3, frame_idx=0),
        np.full((_H, _W), 30, np.float32),
    )


# ---------------------------------------------------------------------------
# CU §6 test 1: word 1 (image count)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word1", [2, 5, -1])
def test_nonzero_image_count_is_refused_by_name(tmp_path, clean_set_cache, word1):
    """lvpyio uses a non-zero word 1 as the image count even when it disagrees with
    the entries (2 hides entries, 5 invents them). No real file has one."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, words={1: word1})
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert f"word 1 (image count) is {word1}" in message
    assert "Frame0-0.ims" in message


# ---------------------------------------------------------------------------
# CU §6 test 4: header strictness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,value,field",
    [
        (0, 2, "version"),
        (0, 0, "version"),
        (5, 8, "bits per decoded pixel"),
        (5, 12, "bits per decoded pixel"),
        (14, 2, "planes per frame"),
        (15, 1, "sample type"),
        (15, 2, "sample type"),
    ],
)
def test_unmeasured_header_word_is_refused_by_name(
    tmp_path, clean_set_cache, word, value, field
):
    """Each of these reads differently in lvpyio or not at all (CU §2 C4, G)."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, words={word: value})
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert f"word {word} ({field}) is {value}" in message
    assert "Frame0-0.ims" in message


def test_one_plane_per_frame_reads(tmp_path, clean_set_cache):
    """Word 14 = 1 opens in lvpyio as 0 does (CU §2 C4, §4 item 1), so both read."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, words={14: 1})
    assert read_set_info(set_file).n_entries == _N_ENTRIES


@pytest.mark.parametrize("word2", [0, 2, 3])
def test_frames_per_image_must_match_the_frame_reader_span(
    tmp_path, clean_set_cache, word2
):
    """lvpyio refuses a word 2 that disagrees with the FrameReader's
    EndFrame - StartFrame + 1 as a "frame count inconsistency" (CU §2 G, H1)."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, words={2: word2})
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert f"word 2 (frames per image) is {word2}" in message
    assert "FrameReader 'Frame0'" in message and "spans 1 frame" in message


def test_header_shorter_than_1024_bytes_is_refused(tmp_path, clean_set_cache):
    """The old reader guessed where the entry table started in a short header."""
    set_file, set_dir = _one_stream_set(tmp_path)
    (set_dir / "Frame0-0.ims").write_bytes(ims_index(_W, _H, [])[:256])
    with pytest.raises(ValueError, match="is 256 bytes, shorter than the 1024-byte"):
        read_set_info(set_file)


def test_stray_bytes_after_the_last_entry_are_refused(tmp_path, clean_set_cache):
    set_file, set_dir = _one_stream_set(tmp_path)
    index = ims_index(_W, _H, [_FRAME_BYTES] * _N_ENTRIES)
    (set_dir / "Frame0-0.ims").write_bytes(index + b"\x00" * 12)
    with pytest.raises(ValueError, match="12 bytes after the last whole 20-byte"):
        read_set_info(set_file)


# ---------------------------------------------------------------------------
# Codec name, bytes 24..55
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", [b"zlib", b"ZLIB", b"lz4", b"None", b"\xff\xfe"])
def test_unknown_codec_name_is_refused(tmp_path, clean_set_cache, codec):
    """Names are case-sensitive: lvpyio opens no stream for "zlib" (CU §2 C1)."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, codec=codec)
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert "compression name" in message and "bytes 24..55" in message
    assert repr(codec.decode("latin-1")) in message


@pytest.mark.parametrize("codec", ["", "Uncompressed"])
def test_uncompressed_codec_names_read(tmp_path, clean_set_cache, codec):
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, codec=codec.encode())
    assert read_set_info(set_file).frames[0].codec == codec
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=2, frame_idx=0),
        np.full((_H, _W), 20, np.float32),
    )


@pytest.mark.parametrize("codec", ["Zlib", "LZ4", "IMX"])
def test_compressed_codec_is_parsed_and_refused_at_read(
    tmp_path, clean_set_cache, codec
):
    """The name is carried on the stream. Reading compressed bytes as raw pixels
    would return wrong pixels whenever the stored size happens to fit (CU §2 C8)."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, codec=codec.encode())
    assert read_set_info(set_file).frames[0].codec == codec
    with pytest.raises(ValueError, match=f"'{codec}' compression"):
        read_set_frame(set_file, entry_no=1, frame_idx=0)


# ---------------------------------------------------------------------------
# CU §6 test 2: file number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_no,named", [(2, "Frame0-2.ims"), (10, "Frame0-10.ims")])
def test_entry_in_another_data_file_is_refused(
    tmp_path, clean_set_cache, file_no, named
):
    """Production read every entry from -1.ims, so a split stream returned the
    wrong frames with no error in 6 of 11 layouts (CU §2 B2)."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, file_numbers=[1, 1, file_no, 1])
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert named in message and "entry 2" in message


def test_file_number_zero_is_refused(tmp_path, clean_set_cache):
    """File number 0 names the index itself. lvpyio raises on it; production
    used to read the entry from -1.ims."""
    set_file, set_dir = _one_stream_set(tmp_path)
    _rewrite_index(set_dir, file_numbers=[1, 1, 1, 0])
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert "file number 0" in message and "entry 3" in message


# ---------------------------------------------------------------------------
# CU §6 test 6: FrameReader Name and IsAssociatedToFrames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,vendor",
    [
        ("MRAW", "Photron"),
        ("GRAW", "Photron"),
        ("CINE", "Vision Research"),
        ("cine", "Vision Research"),
    ],
)
def test_foreign_camera_reader_is_refused_by_vendor(
    tmp_path, clean_set_cache, name, vendor
):
    set_file, set_dir = _one_stream_set(tmp_path)
    write_manifest(set_dir, [("Frame0", 0)], reader_name=name)
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert f"Name '{name}'" in message and vendor in message


@pytest.mark.parametrize(
    "name", ["images_16bit_8bit", "Images_16bit", "ScaleReaderV1", "", "NotAReader"]
)
def test_frame_reader_name_must_be_images_16bit_8bit(tmp_path, clean_set_cache, name):
    """lvpyio opens a stream only for this exact, case-sensitive Name (CU §2 F1)."""
    set_file, set_dir = _one_stream_set(tmp_path)
    write_manifest(set_dir, [("Frame0", 0)], reader_name=name)
    with pytest.raises(ValueError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert f"Name '{name}'" in message and "'Images_16bit_8Bit'" in message


def test_frame_reader_not_associated_to_frames_is_refused(tmp_path, clean_set_cache):
    set_file, set_dir = _one_stream_set(tmp_path)
    write_manifest(set_dir, [("Frame0", 0)], associated_to_frames="false")
    with pytest.raises(ValueError, match="IsAssociatedToFrames is 'false'"):
        read_set_info(set_file)


# ---------------------------------------------------------------------------
# A .set whose images are .im7 files (drive-survey §4 row 6)
# ---------------------------------------------------------------------------


def test_manifest_over_im7_images_points_at_lavision_im7(tmp_path, clean_set_cache):
    """Real case E:/retro_tbl_35mm_far: StreamSet.xml declares Frame0/Frame1 .ims
    streams, with decoder XMLs and a dark image, but the images are B*.im7 files.
    This used to be reported as "Frame0-0.ims is missing"."""
    set_file, set_dir = _one_stream_set(tmp_path)
    for f in set_dir.glob("*.ims"):
        f.unlink()
    (set_dir / "Transformer0-dark.im7").write_bytes(b"stub")
    for k in (1, 2, 3):
        (set_dir / f"B{k:05d}.im7").write_bytes(b"stub")
    with pytest.raises(FileNotFoundError) as exc:
        read_set_info(set_file)
    message = str(exc.value)
    assert "lavision_im7" in message
    assert "B00001.im7 and 2 more" in message
    assert "StreamSet.xml declares .ims streams" in message
    assert "missing" not in message
