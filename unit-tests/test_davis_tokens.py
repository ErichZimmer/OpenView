"""DaVis token decoder shared by .im7 pack 1 and IMX-compressed .set entries.

Expected values are worked by hand from the token grammar measured against
lvpyio 1.3.1 (container-unknowns C5-C7, numba-decoder REPORT): an int8 byte is a
signed delta, 0x80 lo hi one absolute, 0x7f n (lo hi)*n n absolutes, 0x81 a run
of signed nibbles, high first, where only a LOW nibble 8 ends the run.
"""

import numpy as np
import pytest

from pivtools_core.image_handling.readers.davis_tokens import (
    TokenRunOverrunError,
    TokenStreamError,
    TokenValueRangeError,
    TrailingTokenBytesError,
    TruncatedTokenStreamError,
    decode_imx_words,
    decode_pack1,
)

PREAMBLE = bytes([0x01, 0x01, 0x00])

# C6 example: 0x80 5000 | 0x81 run 0x12 (+1, +2), 0x88 (high 8 = -8, low 8 ends)
# | 0x0a (+10). lvpyio gives 5000, 5001, 5003, 4995, 5005.
C6_TOKENS = bytes([0x80, 0x88, 0x13, 0x81, 0x12, 0x88, 0x0A])
C6_VALUES = [5000, 5001, 5003, 4995, 5005]


def _u8(data: bytes) -> np.ndarray:
    """Writable C-contiguous uint8 copy, the form the readers hand the decoder."""
    return np.frombuffer(data, dtype=np.uint8).copy()


def test_pack1_c6_example_high_nibble_8_is_minus_8():
    frames, end = decode_pack1(_u8(PREAMBLE + C6_TOKENS), npix=5, n_frames=1)
    assert frames.dtype == np.uint16
    assert frames.shape == (1, 5)
    assert frames[0].tolist() == C6_VALUES
    assert end == len(PREAMBLE) + len(C6_TOKENS)


# Three 3-pixel frames. The preamble comes before frame 0 only, and frames 1 and
# 2 open with int8 deltas, so their values exist only if the running value
# carries over from the previous frame's last pixel.
CARRY_TOKENS = (
    bytes([0x80, 0xE8, 0x03, 0x05, 0xFE])  # 1000, +5, -2
    + bytes([0x02, 0x81, 0x38, 0x01])  # +2, run (+3, end), +1
    + bytes([0xFF, 0x7F, 0x02, 0x10, 0x27, 0x11, 0x27])  # -1, two absolutes
)
CARRY_FRAMES = [[1000, 1005, 1003], [1005, 1008, 1009], [1008, 10000, 10001]]
CARRY_END = len(PREAMBLE) + len(CARRY_TOKENS)


def test_pack1_running_value_carries_across_frames():
    frames, end = decode_pack1(_u8(PREAMBLE + CARRY_TOKENS), npix=3, n_frames=3)
    assert frames.tolist() == CARRY_FRAMES
    assert end == CARRY_END


def test_pack1_keep_returns_only_the_wanted_frames_and_still_finds_the_end():
    data = _u8(PREAMBLE + CARRY_TOKENS + bytes([0x01, 0x00, 0x00, 0x00]))
    frames, end = decode_pack1(data, npix=3, n_frames=3, keep=range(1, 2))
    assert frames.tolist() == [CARRY_FRAMES[1]]
    assert end == CARRY_END


@pytest.mark.parametrize(
    "tokens",
    [
        bytes([0x64, 0x01]),  # two deltas for a four-pixel frame
        bytes([0x64, 0x80, 0x10]),  # 0x80 absolute missing its high byte
        bytes([0x64, 0x7F, 0x02, 0x10, 0x00, 0x11]),  # 0x7f run one byte short
        bytes([0x64, 0x81, 0x12]),  # nibble run cut off before its end
    ],
    ids=["deltas", "abs16", "abs-run", "nibble-run"],
)
def test_pack1_truncated_stream_raises_by_name(tokens):
    with pytest.raises(TruncatedTokenStreamError, match="pack-1 frame 0 of 1"):
        decode_pack1(_u8(PREAMBLE + tokens), npix=4, n_frames=1)


def test_pack1_0x7f_run_past_the_frame_raises_by_name():
    tokens = bytes([0x7F, 0x05] + [0x10, 0x00] * 5)  # five absolutes, frame of 3
    with pytest.raises(TokenRunOverrunError, match="0x7f"):
        decode_pack1(_u8(PREAMBLE + tokens), npix=3, n_frames=1)


@pytest.mark.parametrize(
    "tokens",
    [
        bytes([0x05, 0xF0]),  # 5 then -16: below 0
        bytes([0x80, 0xFF, 0xFF, 0x01]),  # 65535 then +1: above uint16
        bytes([0x80, 0x02, 0x00, 0x81, 0x8F]),  # 2 then high-nibble -8: below 0
    ],
    ids=["delta-below-0", "delta-above-65535", "nibble-below-0"],
)
def test_pack1_running_value_outside_uint16_raises_by_name(tokens):
    with pytest.raises(TokenValueRangeError, match="0..65535"):
        decode_pack1(_u8(PREAMBLE + tokens), npix=4, n_frames=1)


def _wrong_arrays():
    good = _u8(PREAMBLE + C6_TOKENS)
    read_only = np.frombuffer(PREAMBLE + C6_TOKENS, dtype=np.uint8)
    strided = np.repeat(good, 2)[::2]
    return {
        "int8": (good.view(np.int8), "uint8"),
        "uint16": (good[:10].view(np.uint16), "uint8"),
        "read-only": (read_only, "writeable"),
        "strided": (strided, "C-contiguous"),
        "2-D": (good.reshape(2, 5), "1-D"),
    }


@pytest.mark.parametrize("case", list(_wrong_arrays()))
def test_pack1_refuses_an_input_array_of_the_wrong_kind(case):
    arr, reason = _wrong_arrays()[case]
    with pytest.raises(TypeError, match=reason):
        decode_pack1(arr, npix=5, n_frames=1)


@pytest.mark.parametrize(
    "keep", [range(2, 4), range(-1, 1), range(0, 3, 2), range(1, 1)]
)
def test_pack1_refuses_a_keep_range_outside_the_buffer(keep):
    with pytest.raises(ValueError, match="keep"):
        decode_pack1(_u8(PREAMBLE + CARRY_TOKENS), npix=3, n_frames=3, keep=keep)


def test_imx_words_have_no_preamble_and_start_from_zero():
    words = decode_imx_words(_u8(C6_TOKENS), n_words=5)
    assert words.dtype == np.uint16
    assert words.tolist() == C6_VALUES
    # A leading int8 delta counts from 0, not from any earlier value.
    assert decode_imx_words(_u8(bytes([0x07, 0x02])), n_words=2).tolist() == [7, 9]


def test_imx_tokens_must_end_exactly_at_the_entry_size():
    with pytest.raises(
        TrailingTokenBytesError, match="tokens end at byte 7, entry holds 8"
    ):
        decode_imx_words(_u8(C6_TOKENS + bytes([0x01])), n_words=5)


def test_imx_short_entry_raises_by_name():
    with pytest.raises(TruncatedTokenStreamError, match="IMX entry"):
        decode_imx_words(_u8(C6_TOKENS), n_words=6)


def test_imx_refuses_a_read_only_array():
    with pytest.raises(TypeError, match="writeable"):
        decode_imx_words(np.frombuffer(C6_TOKENS, dtype=np.uint8), n_words=5)


def test_every_named_error_is_a_token_stream_error_and_an_ioerror():
    for cls in (
        TruncatedTokenStreamError,
        TokenRunOverrunError,
        TokenValueRangeError,
        TrailingTokenBytesError,
    ):
        assert issubclass(cls, TokenStreamError)
        assert issubclass(cls, IOError)
