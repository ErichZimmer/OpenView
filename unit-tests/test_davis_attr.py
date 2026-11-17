"""The DaVis ``.attr`` byte format: parse, encode, refuse.

Evidence: attr-decode follow-up REPORT (2026-11-17), §2 A and §6 items 1-3. The
committed fixtures under ``fixtures/davis_attr/`` are real DaVis 10.2.1 and
11.4.0.91 files cut to one or two images at the byte level
(``make_davis_attr_fixture.py``), so the round trip below runs on DaVis's own
bytes, not on bytes this module's encoder wrote.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from pivtools_core.image_handling.readers.davis_attr import (
    HEADER_SIZE,
    TYPE_EMPTY,
    TYPE_FLOAT32,
    TYPE_FLOAT64,
    TYPE_INT32,
    TYPE_STRING,
    AttrEntry,
    AttrFormatError,
    AttrRecord,
    AttrStream,
    encode_attr_stream,
    parse_attr_bytes,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "davis_attr"
REAL_ATTR_FILES = sorted(FIXTURES.glob("*/*.attr"))


def _record(type_code, key, value):
    return struct.pack("<iII", type_code, len(key), len(value)) + key + value


def _file(entries, trailing=b""):
    """A .attr file assembled by hand: each entry is raw record bytes."""
    end_entry = struct.pack("<iII", -1, 0, 0)
    end_stream = struct.pack("<iII", 0, 0, 0)
    out = bytearray(struct.pack("<I", 1) + bytes(HEADER_SIZE - 4))
    offsets = []
    for records in entries:
        offsets.append(len(out))
        out += records + end_entry
    offsets.append(len(out))
    out += trailing + end_stream
    out += struct.pack(f"<{len(offsets)}Q", *offsets) + struct.pack("<I", len(offsets))
    return bytes(out)


def test_real_fixture_files_are_present():
    names = {(p.parent.name, p.name) for p in REAL_ATTR_FILES}
    assert len(names) == 15  # NTNU MetaData0..6, 145749 MetaData1..8


@pytest.mark.parametrize(
    "path", REAL_ATTR_FILES, ids=lambda p: f"{p.parent.name[:12]}/{p.name}"
)
def test_real_bytes_round_trip(path):
    """§6 item 1: every byte of DaVis's own file is accounted for."""
    data = path.read_bytes()
    assert encode_attr_stream(parse_attr_bytes(data, path.name)) == data


def test_real_fixtures_hold_every_real_record_type_and_a_trailing_block():
    types = set()
    trailing_keys = set()
    for path in REAL_ATTR_FILES:
        stream = parse_attr_bytes(path.read_bytes(), path.name)
        types |= {r.type_code for e in stream.entries for r in e.records}
        trailing_keys |= {r.key for r in stream.trailing.records}
    assert types == {TYPE_STRING, TYPE_INT32, TYPE_FLOAT32, TYPE_FLOAT64, TYPE_EMPTY}
    assert trailing_keys == {"_DaVisVersion"}


def test_hand_assembled_bytes_parse_to_the_expected_values():
    """§6 item 2: bytes written here, not by the encoder, including the uint16
    type seen only through lvpyio (accepted on that evidence, 17 Nov)."""
    records = (
        _record(1, b"CCDExposureTime", "13761 µs".encode("utf-8"))
        + _record(2, b"U16", struct.pack("<2H", 1, 65535))
        + _record(3, b"RealFrameSize", struct.pack("<2i", 5120, -4096))
        + _record(4, b"DevDataTrace0", struct.pack("<f", 1.5))
        + _record(5, b"Acq.Time", struct.pack("<d", 12783.125))
        + struct.pack("<iII", 6, 0, 0)
        + _record(1, b"Empty", b"")
    )
    data = _file([records, b""], trailing=_record(1, b"_DaVisVersion", b"11.4.0.91"))
    stream = parse_attr_bytes(data)
    assert stream.n_entries == 2
    d = stream.entries[0].as_dict()
    assert d.keys() == {
        "CCDExposureTime",
        "U16",
        "RealFrameSize",
        "DevDataTrace0",
        "Acq.Time",
        "Empty",
    }
    assert d["CCDExposureTime"] == "13761 µs" and d["Empty"] == ""
    expected = {
        "U16": ("<u2", [[1, 65535]]),
        "RealFrameSize": ("<i4", [[5120, -4096]]),
        "DevDataTrace0": ("<f4", [[1.5]]),
        "Acq.Time": ("<f8", [[12783.125]]),
    }
    for key, (dtype, values) in expected.items():
        assert d[key].dtype == np.dtype(dtype) and d[key].tolist() == values, key
    assert [r.type_code for r in stream.entries[0].records][5] == TYPE_EMPTY
    assert stream.entries[1].records == ()
    assert stream.trailing.as_dict() == {"_DaVisVersion": "11.4.0.91"}


def _base():
    """One entry holding one string record "a" = "b", as hand-assembled bytes."""
    return _file([_record(1, b"a", b"b")])


def _patched(data, pos, fmt, value):
    b = bytearray(data)
    struct.pack_into(fmt, b, pos, value)
    return bytes(b)


# Byte positions in _base(): the record header starts at HEADER_SIZE (type,
# key_len, value_len), its key at +12, its value at +13; END_ENTRY at +14.
_REC = HEADER_SIZE
_END_ENTRY = HEADER_SIZE + 14

MALFORMED = {
    "version": (lambda: _patched(_base(), 0, "<I", 2), 0),
    "header not zero": (lambda: _patched(_base(), 100, "<B", 7), 100),
    "unknown type -2": (lambda: _patched(_base(), _REC, "<i", -2), _REC),
    "unknown type 7": (lambda: _patched(_base(), _REC, "<i", 7), _REC),
    "count": (lambda: _patched(_base(), len(_base()) - 4, "<I", 99), len(_base()) - 4),
    "offset": (lambda: _patched(_base(), len(_base()) - 20, "<Q", 1025), _REC),
    "truncated": (lambda: _base()[:-6], None),
    "array length": (lambda: _patched(_base(), _REC, "<i", 3), _REC),
    "utf8": (lambda: _patched(_base(), _REC + 13, "<B", 0xB5), _REC),
    "duplicate key": (
        lambda: _file([_record(1, b"a", b"b") + _record(1, b"a", b"c")]),
        _REC + 14,
    ),
    "marker length": (lambda: _patched(_base(), _END_ENTRY + 4, "<I", 3), _END_ENTRY),
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_input_is_refused_naming_the_byte(case):
    """§6 item 3: the 11 malformed inputs of attr_proto.self_test."""
    build, offset = MALFORMED[case]
    with pytest.raises(AttrFormatError, match=r"<bad>: byte \d+: ") as exc:
        parse_attr_bytes(build(), "<bad>")
    if offset is not None:
        assert f"byte {offset}:" in str(exc.value)


def test_non_ascii_key_is_refused():
    """lvpyio turns a non-ASCII key byte into '?' without a message."""
    with pytest.raises(AttrFormatError, match="not printable ASCII"):
        parse_attr_bytes(_file([_record(1, b"k\xb5", b"v")]))


def test_encoder_round_trips_every_type():
    stream = AttrStream(
        "<s>",
        (
            AttrEntry(
                (
                    AttrRecord(TYPE_STRING, "s", "µ"),
                    AttrRecord(2, "u", np.array([[1, 65535]], "<u2")),
                    AttrRecord(TYPE_INT32, "i", np.array([[-1]], "<i4")),
                    AttrRecord(TYPE_FLOAT32, "f", np.array([[0.5]], "<f4")),
                    AttrRecord(TYPE_FLOAT64, "d", np.array([[2.25]], "<f8")),
                    AttrRecord(TYPE_EMPTY, "", None),
                )
            ),
        ),
        AttrEntry((AttrRecord(TYPE_STRING, "_DaVisVersion", "11.4.0.91"),)),
    )
    data = encode_attr_stream(stream)
    assert encode_attr_stream(parse_attr_bytes(data)) == data
