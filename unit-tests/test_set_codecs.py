"""Compressed ``.set`` entries: the Zlib, LZ4 and IMX codecs (ticket D4).

The codec name sits at index header bytes 24..55. lvpyio 1.3.1, the oracle,
decompresses each entry before the pixel decoder (container-unknowns REPORT §2
C1, C2, C5):

  Zlib  one bare RFC 1950 stream per entry
  LZ4   one bare LZ4 block, no size prefix
  IMX   the DaVis token grammar over the little-endian uint16 words of the packed
        payload, no preamble, running value from 0, ending exactly at the entry
        size

The expected pixels are hand-computed from each encoding's bit layout, or come
from lvpyio (the real IMX fixture in ``fixtures/set_imx``).
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import sys
import zlib
from pathlib import Path

import lz4.block
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import build_set, clean_set_cache  # noqa: E402,F401

from pivtools_core.image_handling.readers.davis_tokens import (  # noqa: E402
    TrailingTokenBytesError,
    TruncatedTokenStreamError,
)
from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    read_set_frame,
)

# (decoder, packed payload bytes, expected pixels), each hand-computed from the
# encoding's bit layout. Shapes are (2, 4).
_MONO12P_BYTES = bytes(
    [0x34, 0x12, 0xAB, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0xF0, 0x00]
)
# 0x34 0x12 0xAB: p0 = 0x34 | 0x2 << 8 = 564, p1 = 0x1 | 0xAB << 4 = 2737.
# 0x01 0xF0 0x00: p0 = 0x01 | 0x0 << 8 = 1,  p1 = 0xF | 0x00 << 4 = 15.
_MONO12P_PIXELS = [[564, 2737, 4095, 4095], [0, 0, 1, 15]]
# One little-endian bit stream, 10 bits per pixel: 1023 | 0 << 10 | 512 << 20 |
# 1 << 30 = 0x600003FF, then four pixels of all ones.
_MONO10P_BYTES = bytes([0xFF, 0x03, 0x00, 0x60, 0x00] + [0xFF] * 5)
_MONO10P_PIXELS = [[1023, 0, 512, 1], [1023, 1023, 1023, 1023]]
_RAW16_PIXELS = [[0, 3, 1021, 65535], [7, 256, 4095, 12]]
_RAW8_PIXELS = [[0, 1, 200, 255], [17, 42, 128, 254]]

PAYLOADS = {
    "raw-16-bit": (np.array(_RAW16_PIXELS, "<u2").tobytes(), _RAW16_PIXELS),
    "raw-8-bit": (np.array(_RAW8_PIXELS, np.uint8).tobytes(), _RAW8_PIXELS),
    "mono-12p": (_MONO12P_BYTES, _MONO12P_PIXELS),
    "mono-10p": (_MONO10P_BYTES, _MONO10P_PIXELS),
}
_H, _W = 2, 4


def _codec_set(tmp_path, decoder, codec, stored_entries):
    """A one-stream .set of ``stored_entries`` under ``codec``."""
    return build_set(tmp_path, decoder, [stored_entries], _W, _H, codec=codec.encode())


@pytest.mark.parametrize("decoder", list(PAYLOADS))
def test_zlib_entry_decodes_exactly(tmp_path, clean_set_cache, decoder):
    """Entry 2 sits after a compressed entry 1 of another size, so its offset
    comes from the index, not from the raw frame size."""
    payload, pixels = PAYLOADS[decoder]
    zeros = bytes(len(payload))
    set_file = _codec_set(
        tmp_path, decoder, "Zlib", [zlib.compress(zeros), zlib.compress(payload)]
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=2, frame_idx=0),
        np.array(pixels, np.float32),
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=0),
        np.zeros((_H, _W), np.float32),
    )


@pytest.mark.parametrize("decoder", list(PAYLOADS))
def test_lz4_entry_decodes_exactly(tmp_path, clean_set_cache, decoder):
    """A bare LZ4 block, no size prefix, as lvpyio opens it (CU §2 C1)."""
    payload, pixels = PAYLOADS[decoder]
    zeros = bytes(len(payload))
    set_file = _codec_set(
        tmp_path,
        decoder,
        "LZ4",
        [
            lz4.block.compress(zeros, store_size=False),
            lz4.block.compress(payload, store_size=False),
        ],
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=2, frame_idx=0),
        np.array(pixels, np.float32),
    )
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=0),
        np.zeros((_H, _W), np.float32),
    )


# Every IMX token kind, 9 words (CU §2 C5, the grammar of davis_tokens.py):
#   0x80 34 12             absolute 0x1234                      4660
#   0x81 12 88             nibbles +1 +2, high 8 = -8, low 8 ends  4661 4663 4655
#   0x7f 02 1027 1127      two absolutes                        10000 10001
#   05 FB 0A               int8 deltas +5 -5 +10                10006 10001 10011
_IMX_TOKENS = bytes.fromhex("8034128112887f021027112705fb0a")
_IMX_WORDS = [4660, 4661, 4663, 4655, 10000, 10001, 10006, 10001, 10011]


def test_imx_all_token_entry_decodes_raw16(tmp_path, clean_set_cache):
    """raw-16-bit: the decoded words are the pixels, so they are hand-known."""
    set_file = build_set(tmp_path, "raw-16-bit", [[_IMX_TOKENS]], 3, 3, codec=b"IMX")
    np.testing.assert_array_equal(
        read_set_frame(set_file, entry_no=1, frame_idx=0),
        np.array(_IMX_WORDS, np.float32).reshape(3, 3),
    )


def test_imx_all_token_entry_decodes_mono12p_as_its_uncompressed_twin(
    tmp_path, clean_set_cache
):
    """mono-12p: the words are the packed payload, decoded by mono-12p after.
    The twin stores those 18 bytes uncompressed (CU §2 C2 control)."""
    packed = np.array(_IMX_WORDS, "<u2").tobytes()
    imx = build_set(
        tmp_path, "mono-12p", [[_IMX_TOKENS]], 4, 3, name="imx", codec=b"IMX"
    )
    twin = build_set(tmp_path, "mono-12p", [[packed]], 4, 3, name="twin")
    expected = read_set_frame(twin, entry_no=1, frame_idx=0)
    assert expected.shape == (3, 4)
    np.testing.assert_array_equal(
        read_set_frame(imx, entry_no=1, frame_idx=0), expected
    )


_REAL_IMX = Path(__file__).resolve().parent / "fixtures" / "set_imx"


def test_real_imx_live_image_entry_equals_lvpyio(clean_set_cache):
    """Rows 1239..1345 of a real DaVis 11.4 Live Image (mono-12p, 5312 wide),
    the stored IMX bytes verbatim: 1109 deltas, 18 single absolutes and 2308
    0x7f runs. lvpyio's decode of the cut equals its decode of those rows of the
    full 36.8 MB entry (fixtures/set_imx/make_set_imx_fixture.py)."""
    expected = np.load(_REAL_IMX / "fixture.lvpyio.npy")
    assert expected.shape == (107, 5312) and expected.dtype == np.uint16
    img = read_set_frame(_REAL_IMX / "fixture.set", entry_no=1, frame_idx=0)
    np.testing.assert_array_equal(img, expected.astype(np.float32))


# ---------------------------------------------------------------------------
# Refusals, each naming the stream, the entry and the codec
# ---------------------------------------------------------------------------


def _assert_names_the_entry(message, codec):
    assert "Frame stream 0 of synthetic" in message
    assert "entry_no 1" in message and f"'{codec}' compression" in message


def test_imx_odd_packed_byte_count_is_refused(tmp_path, clean_set_cache):
    """mono-10p 2x2 packs into 5 bytes. lvpyio failed on all 3 layouts tried
    (CU §2 C5), so the layout is unknown."""
    set_file = build_set(tmp_path, "mono-10p", [[b"\x00" * 4]], 2, 2, codec=b"IMX")
    with pytest.raises(ValueError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)
    _assert_names_the_entry(str(exc.value), "IMX")
    assert "5 bytes, an odd count" in str(exc.value)


def test_imx_tokens_after_the_last_word_are_refused(tmp_path, clean_set_cache):
    set_file = build_set(
        tmp_path, "raw-16-bit", [[_IMX_TOKENS + b"\x05"]], 3, 3, codec=b"IMX"
    )
    with pytest.raises(TrailingTokenBytesError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)
    _assert_names_the_entry(str(exc.value), "IMX")
    assert "tokens end at byte 15, entry holds 16" in str(exc.value)


def test_truncated_imx_entry_is_refused(tmp_path, clean_set_cache):
    set_file = build_set(
        tmp_path, "raw-16-bit", [[_IMX_TOKENS[:-1]]], 3, 3, codec=b"IMX"
    )
    with pytest.raises(TruncatedTokenStreamError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)
    _assert_names_the_entry(str(exc.value), "IMX")


_PAYLOAD = PAYLOADS["raw-16-bit"][0]  # 16 bytes for a 2x4 raw-16-bit frame


@pytest.mark.parametrize(
    "stored,reason",
    [
        (zlib.compress(_PAYLOAD)[:-4], "without its end marker"),
        (zlib.compress(_PAYLOAD[:-2]), "decompresses to 14 bytes"),
        (zlib.compress(_PAYLOAD + b"\x00\x00"), "more than the 16 bytes"),
        (zlib.compress(_PAYLOAD) + b"\x00", "1 bytes follow the end"),
        (_PAYLOAD, "does not decompress"),
    ],
    ids=["truncated", "short", "long", "trailing", "not-zlib"],
)
def test_bad_zlib_entry_is_refused(tmp_path, clean_set_cache, stored, reason):
    set_file = _codec_set(tmp_path, "raw-16-bit", "Zlib", [stored])
    with pytest.raises(OSError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)
    _assert_names_the_entry(str(exc.value), "Zlib")
    assert reason in str(exc.value)


@pytest.mark.parametrize(
    "stored",
    [
        lz4.block.compress(_PAYLOAD, store_size=False)[:-3],
        lz4.block.compress(_PAYLOAD[:-2], store_size=False),
        lz4.block.compress(_PAYLOAD + b"\x00\x00", store_size=False),
        lz4.block.compress(_PAYLOAD, store_size=True),
    ],
    ids=["truncated", "short", "long", "size-prefixed"],
)
def test_bad_lz4_entry_is_refused(tmp_path, clean_set_cache, stored):
    set_file = _codec_set(tmp_path, "raw-16-bit", "LZ4", [stored])
    with pytest.raises(OSError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)
    _assert_names_the_entry(str(exc.value), "LZ4")
    assert "does not decompress to the 16 bytes" in str(exc.value)
