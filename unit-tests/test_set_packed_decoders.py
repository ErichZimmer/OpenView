"""The three msb-first and grouped DaVis pixel encodings (ticket D8).

``mono-10pmsb`` (the NTNU Imager MX PIV cameras), ``mono-12pmsb`` and
``mono-12packed``. Their layouts were settled against lvpyio 1.3.1, LaVision's own
reader, not taken from the names (01-lavision-readers REPORT A1-A3):

    mono-10pmsb    PFNC Mono10pmsb, 4 px in 5 bytes, one big-endian bit stream
                       p0 =  b0         << 2 | b1 >> 6
                       p1 = (b1 & 0x3F) << 4 | b2 >> 4
                       p2 = (b2 & 0x0F) << 6 | b3 >> 2
                       p3 = (b3 & 0x03) << 8 | b4
    mono-12pmsb    PFNC msb packed, 2 px in 3 bytes
                       p0 =  b0         << 4 | b1 >> 4
                       p1 = (b1 & 0x0F) << 8 | b2
    mono-12packed  nibble byte in the middle, 2 px in 3 bytes
                       p0 = b0 << 4 | (b1 & 0x0F)
                       p1 = b2 << 4 | (b1 >> 4)

Each id is checked four ways: hand-computed bytes, the fast decoder against a
plain reference written from the layout on random bytes, lvpyio's own output for
a committed synthetic payload (``fixtures/set_packed``), and a frame size the
packing cannot hold refused by name.

Usage:
    pytest unit-tests/test_set_packed_decoders.py -v
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.lavision import build_set, clean_set_cache  # noqa: E402,F401

from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    _IMPLEMENTED_DECODERS,
    read_set_frame,
)


def _read(tmp_path, decoder, payload, width, height, name="synthetic"):
    """One entry of a one-stream ``decoder`` container through read_set_frame."""
    set_file = build_set(
        tmp_path, decoder, [[payload]], width=width, height=height, name=name
    )
    return read_set_frame(set_file, entry_no=1, frame_idx=0)


# ---------------------------------------------------------------------------
# Hand-computed bytes
# ---------------------------------------------------------------------------


def test_mono10pmsb_hand_computed_group(tmp_path, clean_set_cache):
    """One 5-byte group, worked as a 40-bit big-endian stream cut into 10-bit pixels.

    A5 3C 96 0F E1 = 10100101 00111100 10010110 00001111 11100001
    p0 = 1010010100 = 0x294 = 660
    p1 = 1111001001 = 0x3C9 = 969
    p2 = 0110000011 = 0x183 = 387
    p3 = 1111100001 = 0x3E1 = 993
    """
    img = _read(tmp_path, "mono-10pmsb", bytes([0xA5, 0x3C, 0x96, 0x0F, 0xE1]), 4, 1)

    np.testing.assert_array_equal(img, np.array([[660, 969, 387, 993]], np.float32))


# Two groups on two rows, so group order and row order are pinned as well as the
# bits inside a group. b = 34 12 AB | FF 0F 80.
_TWO_GROUPS_12 = bytes([0x34, 0x12, 0xAB, 0xFF, 0x0F, 0x80])


def test_mono12pmsb_hand_computed_groups(tmp_path, clean_set_cache):
    """p0 = b0 << 4 | b1 >> 4, p1 = (b1 & 0x0F) << 8 | b2.

    group 1: p0 = 0x340 | 0x1 = 0x341 = 833,  p1 = 0x200 | 0xAB = 0x2AB = 683
    group 2: p0 = 0xFF0 | 0x0 = 0xFF0 = 4080, p1 = 0xF00 | 0x80 = 0xF80 = 3968
    """
    img = _read(tmp_path, "mono-12pmsb", _TWO_GROUPS_12, 2, 2)

    np.testing.assert_array_equal(img, np.array([[833, 683], [4080, 3968]], np.float32))


def test_mono12packed_hand_computed_groups(tmp_path, clean_set_cache):
    """p0 = b0 << 4 | (b1 & 0x0F), p1 = b2 << 4 | b1 >> 4.

    group 1: p0 = 0x340 | 0x2 = 0x342 = 834,  p1 = 0xAB0 | 0x1 = 0xAB1 = 2737
    group 2: p0 = 0xFF0 | 0xF = 0xFFF = 4095, p1 = 0x800 | 0x0 = 0x800 = 2048
    """
    img = _read(tmp_path, "mono-12packed", _TWO_GROUPS_12, 2, 2)

    np.testing.assert_array_equal(
        img, np.array([[834, 2737], [4095, 2048]], np.float32)
    )


def test_the_three_12bit_ids_read_the_same_bytes_differently(tmp_path, clean_set_cache):
    """Control: the bytes above discriminate. mono-12p reads them as 564, 2737
    (``test_set_decoders.test_mono12p_still_decodes``), so a dispatch that sent
    either new id to the mono-12p decoder fails the two tests above."""
    reads = {
        dec: _read(tmp_path, dec, _TWO_GROUPS_12, 2, 2, name=dec)[0, 0]
        for dec in ("mono-12p", "mono-12pmsb", "mono-12packed")
    }
    assert reads == {"mono-12p": 564, "mono-12pmsb": 833, "mono-12packed": 834}


# ---------------------------------------------------------------------------
# Fast decoder against a plain reference written from the layout
# ---------------------------------------------------------------------------


def _msb_bitstream_reference(raw, bits, width, height):
    """A big-endian bit stream cut into ``bits``-wide pixels, one bit at a time.

    The PFNC msb-packed construction rule: the first pixel's most significant bit
    is the first byte's most significant bit. ``np.unpackbits`` with its default
    ``bitorder="big"`` lays out each byte msb first.
    """
    stream = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="big")
    per_px = stream[: width * height * bits].reshape(-1, bits).astype(np.uint32)
    weights = 1 << np.arange(bits - 1, -1, -1, dtype=np.uint32)
    return (per_px * weights).sum(axis=1).astype(np.uint16).reshape(height, width)


def _mono12packed_reference(raw, width, height):
    """The mono-12packed layout as byte gathers: nibble byte in the middle."""
    g = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.uint16)
    px = np.empty((g.shape[0], 2), np.uint16)
    px[:, 0] = (g[:, 0] << 4) | (g[:, 1] & 0x0F)
    px[:, 1] = (g[:, 2] << 4) | (g[:, 1] >> 4)
    return px.reshape(height, width)


_REFERENCES = {
    "mono-10pmsb": (10, lambda r, w, h: _msb_bitstream_reference(r, 10, w, h)),
    "mono-12pmsb": (12, lambda r, w, h: _msb_bitstream_reference(r, 12, w, h)),
    "mono-12packed": (12, _mono12packed_reference),
}


@pytest.mark.parametrize("decoder", sorted(_REFERENCES))
def test_fast_decoder_equals_reference_on_random_bytes(decoder):
    """Random bytes cover every value of every pixel position, unlike any image."""
    bits, reference = _REFERENCES[decoder]
    width, height = 64, 48
    rng = np.random.default_rng(20261117)
    raw = rng.integers(0, 256, width * height * bits // 8, dtype=np.uint8).tobytes()

    got = _IMPLEMENTED_DECODERS[decoder](raw, width, height)

    np.testing.assert_array_equal(got, reference(raw, width, height))
    assert got.dtype == np.uint16 and got.flags.c_contiguous
    assert got.shape == (height, width)
    # The top bit is reached (not masked off by both sides alike), and no value
    # exceeds the encoding's range.
    assert np.any(got >> (bits - 1)) and got.max() < (1 << bits)


def test_msb_reference_reproduces_the_hand_computed_group():
    """Verify the verifier: the bit-stream reference gives the worked example."""
    np.testing.assert_array_equal(
        _msb_bitstream_reference(bytes([0xA5, 0x3C, 0x96, 0x0F, 0xE1]), 10, 4, 1),
        [[660, 969, 387, 993]],
    )


# ---------------------------------------------------------------------------
# lvpyio's own answer, committed (fixtures/set_packed/make_set_packed_fixtures.py)
# ---------------------------------------------------------------------------

_LVPYIO_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "set_packed"
_SAME_DEPTH = {
    "mono-10pmsb": ("mono-10p",),
    "mono-12pmsb": ("mono-12p", "mono-12packed"),
    "mono-12packed": ("mono-12p", "mono-12pmsb"),
}


@pytest.mark.parametrize("decoder", sorted(_SAME_DEPTH))
def test_read_equals_lvpyio_on_committed_payload(tmp_path, clean_set_cache, decoder):
    """lvpyio 1.3.1 read this payload, labelled with this id, from the container
    build_set writes. Production must give the same pixels."""
    raw = np.load(_LVPYIO_FIXTURES / f"{decoder}.payload.npy").tobytes()
    expected = np.load(_LVPYIO_FIXTURES / f"{decoder}.lvpyio.npy")
    height, width = expected.shape

    img = _read(tmp_path, decoder, raw, width, height)

    assert expected.dtype == np.uint16
    np.testing.assert_array_equal(img, expected.astype(np.float32))


@pytest.mark.parametrize("decoder", sorted(_SAME_DEPTH))
def test_committed_payload_discriminates_layouts(decoder):
    """Control: every other decoder of the same depth disagrees with lvpyio on
    this payload, so the test above can fail."""
    raw = np.load(_LVPYIO_FIXTURES / f"{decoder}.payload.npy").tobytes()
    expected = np.load(_LVPYIO_FIXTURES / f"{decoder}.lvpyio.npy")
    height, width = expected.shape

    for other in _SAME_DEPTH[decoder]:
        got = _IMPLEMENTED_DECODERS[other](raw, width, height)
        assert not np.array_equal(got, expected), other


# ---------------------------------------------------------------------------
# A frame the packing cannot hold is refused by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decoder,width,payload_bytes,reason",
    [
        # 6 px * 10 / 8 = 7 bytes passes the size check; 6 is not a multiple of 4.
        ("mono-10pmsb", 6, 7, "multiple of four pixels"),
        # 3 px * 12 / 8 = 4 bytes passes the size check; 3 is odd.
        ("mono-12pmsb", 3, 4, "even number of pixels"),
        ("mono-12packed", 3, 4, "even number of pixels"),
    ],
)
def test_frame_the_packing_cannot_hold_is_refused_by_name(
    tmp_path, clean_set_cache, decoder, width, payload_bytes, reason
):
    with pytest.raises(ValueError, match=reason) as exc:
        _read(tmp_path, decoder, b"\x00" * payload_bytes, width, 1)

    assert f"{decoder} packs" in str(exc.value)
    assert f"{width}x1" in str(exc.value)
