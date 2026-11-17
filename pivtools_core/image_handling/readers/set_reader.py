"""Pure-Python reader for LaVision .set image containers.

Reads the .set companion directory structure: per frame stream an index file
({prefix}-0.ims), a data file ({prefix}-1.ims), a decoder XML and a scale XML.
No dependency on lvpyio -- works on macOS, Linux, and Windows.

Stream discovery: StreamSet.xml is the companion folder's manifest. Its
ReaderInfo nodes name each stream's file prefix and bind it to a frame-stream
index; the prefix is DaVis's choice per recording ("Frame0".. on one, "Camera1"..
on another), not a format constant. A recording without a manifest, or whose
manifest declares no FrameReader, is refused: streams are never guessed from
file names.

Frame span: a FrameReader may cover several frame positions of each image
(``StartFrame..EndFrame``). DaVis 11.4 writes a double-frame camera's A and B
into one stream file this way, with index header word 2 equal to the span. Each
covered position becomes its own frame stream over the shared files: frame f of
image b is entry ``b * span + (f - StartFrame)`` (container-unknowns REPORT §2
H1-H3).

Strictness: every .ims index header field and FrameReader attribute the reader
does not decode must hold the value measured on real DaVis recordings, otherwise
the recording is refused with the field and the value named (see
``_IMS_REQUIRED_WORDS`` and ``_FRAME_READER_NAME``).

Pixel encodings: DaVis names the encoding in {prefix}-decoder.xml. The full set of
ids DaVis 10/11 can write is in ``_DECODER_BITS_PER_PX`` below, and the ones
decoded here, each with its layout proven against lvpyio, are in
``_IMPLEMENTED_DECODERS``. ``rgb-24`` is recognised and refused by name -- its
channel order cannot be derived from the id, and a wrong guess yields a
plausible-looking wrong image rather than a failure.

Compression: the index header names a codec per stream (``_IMS_CODECS``). A
"Zlib", "LZ4" or "IMX" entry is decompressed to the packed payload before the
pixel decoder (``_decompress_entry``), and must come out at exactly the size the
encoding and geometry need, or the read is refused naming the codec.

Stream transformers: StreamSet.xml binds post-decode transformers to a stream,
which DaVis applies on read (see ``_CORRECTIONS`` and ``_GEOMETRIES``). Three
subtract a per-pixel image stored beside the streams, seven are file-less
geometry. They run in LaVision's fixed order whatever order the manifest lists:
the corrections in sensor coordinates, each clipped, then at most one geometry.
Four geometries turn the frame by 90 degrees, so a stream's OUTPUT shape
(``IMSFrameInfo.shape``) is (W, H) of what its index header stores
(``IMSFrameInfo.stored_shape``); every caller sizes buffers and splits cameras on
the output shape. Three known DaVis ids with no measured effect are refused when
their stream is read, not skipped, because skipping one returns different pixels
from DaVis with nothing to indicate it happened. An id outside the 13 DaVis ids
leaves its stream's output shape unknown. With camera labels (below) it blocks
only the camera that stream belongs to; without them the whole recording is
refused at parse, because cameras are then mapped to streams by shape.

Cameras: a DaVis 10.2+ recording names its cameras in its ``MetaData{N}.attr``
files (``davis_attr``), which StreamSet.xml binds to frame streams by
AttributeReader, never by ``N``. When they carry camera keys, the cameras are
those labels (``SetInfo.cameras``): numbered by first frame stream, double frame
(A/B) or single frame. Without labels, the stream shapes decide
(:func:`valid_stream_splits`), and only a unique split is accepted.

Also supports:
- Pre-paired pairs: the camera's two streams (camera_streams) via read_set_pair
- Single frames: read_set_frame; time-resolved pairing is assembled by callers
  from two single-frame reads at different entries (load_images.read_pair)
- Per-camera frame extraction with seek-based skipping

Reference: LaVision DaVis 10.x .set recording format
"""

import re
import struct
import threading
import xml.etree.ElementTree as ET
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from lz4.block import LZ4BlockError

from .davis_attr import (
    AttrBinding,
    AttrFormatError,
    LabelledCamera,
    RecordingAttributes,
    group_cameras,
    read_recording_attributes,
)
from .davis_tokens import TokenStreamError, decode_imx_words
from .im7_reader import _lz4_block_decompress
from .out_buffer import check_out

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IMSIndexEntry:
    """One 20-byte entry of a {prefix}-0.ims index: where one stored frame lives.

    ``file_no`` names the data file holding the bytes, ``{prefix}-{file_no}.ims``;
    ``offset`` is absolute inside that file and ``size`` is the stored size.
    """

    file_no: int
    offset: int
    size: int


@dataclass(frozen=True)
class StreamTransformer:
    """One post-decode transformer StreamSet.xml binds to a frame stream.

    ``id`` is the DaVis transformer id (``dark-image-subtraction``,
    ``rotate-180``, ...); ``label`` is what the DaVis UI shows for it, kept so a
    refusal names what the user can see. ``correction_path`` is the per-pixel
    image a correction subtracts (``{FilePrefix}-dark.im7``, ``-bgnd.im7``,
    ``-scmos-1.im7``), None for geometry and for the refused ids.
    """

    id: str
    label: str
    correction_path: Optional[Path] = None


@dataclass
class IMSFrameInfo:
    """Metadata for one frame stream: one frame position of every image.

    Usually one {prefix}-0/1.ims pair of its own. When the FrameReader spans
    several frames, the streams of that span share the pair, and ``entries``
    holds only this frame's entries, one per image.

    ``stored_shape`` is the index header's (H, W), in sensor coordinates: what
    the payload decodes to, before the transformers. Only the decoder and the
    correction-image check use it. Everything that sizes a buffer, compares
    cameras or reports a shape uses :attr:`shape`.
    """

    frame_idx: int  # 0-based frame-stream index (the frame position in an image)
    data_path: Path  # {prefix}-1.ims
    index_path: Path  # {prefix}-0.ims
    decoder: str  # DaVis encoding id, e.g. "mono-12p", "raw-16-bit"
    codec: str  # compression name from the index header, one of _IMS_CODECS
    stored_shape: Tuple[int, int]
    n_entries: int
    scale_slope: float
    scale_offset: float
    scale_unit: str
    entries: List[IMSIndexEntry]
    # In application order (LaVision's, not the manifest's); empty when none.
    transformers: Tuple[StreamTransformer, ...] = ()
    # From the recording's camera keys; all None when it carries none. Role is
    # "A" or "B" for a double-frame camera's frames, "" for a single frame.
    camera_label: Optional[str] = None
    camera_id: Optional[str] = None
    frame_role: Optional[str] = None
    # Why this stream cannot be read, when its camera carries a transformer id
    # PIVTOOLs does not know; None when it can.
    blocked: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """(H, W) of the frame this stream reads as, after its transformers.

        (W, H) of :attr:`stored_shape` when its geometry turns the frame by 90
        degrees. A refused id is taken to keep the shape: all three are intensity
        corrections by their DaVis label (HYPOTHESIS, rotation-shape REPORT §5).

        Raises:
            ValueError: If the stream carries a transformer id PIVTOOLs does not
                know: its output shape is unknown (the message is
                :attr:`blocked`).
        """
        if self.has_unknown_transformer:
            raise ValueError(self.blocked)
        h, w = self.stored_shape
        for tf in self.transformers:
            geometry = _GEOMETRIES.get(tf.id)
            if geometry is not None and geometry.swaps_axes:
                return (w, h)
        return (h, w)

    @property
    def has_unknown_transformer(self) -> bool:
        """True when a transformer id is outside the 13 DaVis ids PIVTOOLs knows."""
        return any(tf.id not in _APPLICATION_RANK for tf in self.transformers)


@dataclass(frozen=True)
class _StreamSource:
    """Where one frame stream's entries live, as StreamSet.xml declares them.

    One per frame position of an image buffer. A FrameReader spanning
    ``StartFrame..EndFrame`` stores that many frames of each buffer in one
    stream file (DaVis 11.4 writes A and B this way), so it yields one source
    per frame, all naming the same files. ``frames_per_image`` is the span
    ``EndFrame - StartFrame + 1``, which the index header's word 2 must equal;
    frame ``index`` of buffer ``b`` is entry ``b * frames_per_image +
    span_offset`` of the file (container-unknowns REPORT §2 H1).
    """

    index: int
    prefix: str
    scale_prefix: Optional[str]
    frames_per_image: int
    span_offset: int


@dataclass
class SetInfo:
    """Parsed .set container metadata.

    ``cameras`` are the recording's own camera labels, in frame-stream order;
    None when it has no ``.attr`` files or none of them carries camera keys
    (``attribute_files`` tells the two apart: the declared file names, empty
    when there are none). ``suspicious_flag`` is the text of DaVis's
    ``SetIsSuspicious`` key in the ``.set`` file, None when absent; DaVis does
    not record why it sets it.
    """

    set_dir: Path
    frames: List[IMSFrameInfo]
    n_entries: int  # number of images (same across all frames)
    cameras: Optional[Tuple[LabelledCamera, ...]]
    attribute_files: Tuple[str, ...]
    suspicious_flag: Optional[str]
    # Resolved stream->camera split per camera count, filled lazily by
    # set_camera_streams. Lives on SetInfo so it dies with the cache entry.
    stream_splits: Dict[int, Tuple[Tuple[int, ...], ...]] = field(
        default_factory=dict, repr=False, compare=False
    )


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------


# The {prefix}-0.ims index, as measured on 20 real stream files over four drives
# and against lvpyio 1.3.1 (container-unknowns REPORT, §1 and §2 A, B, C, G). A
# 1024-byte header of int32 words, then one 20-byte entry per stored frame.
_IMS_HEADER_BYTES = 1024
# File number, offset inside that file, stored size. The file number is unsigned:
# lvpyio names the data file with it as an unsigned decimal (-1 opens
# "-4294967295.ims"), so a refusal names the same file lvpyio would open.
_IMS_ENTRY = struct.Struct("<Iqq")
_IMS_WORD = struct.Struct("<i")
_IMS_WORD_FRAMES_PER_IMAGE = 2
_IMS_WORD_WIDTH = 3
_IMS_WORD_HEIGHT = 4

# Header words that hold one measured value on every real stream file. lvpyio
# reads other values differently or not at all, and no real file shows what
# DaVis means by them, so any other value is refused with the field named.
# word: (field name, accepted values, what another value does).
_IMS_REQUIRED_WORDS: Dict[int, Tuple[str, Tuple[int, ...], str]] = {
    0: (
        "version",
        (1,),
        "Every real stream file is version 1. lvpyio also opens 2 and 3, but "
        "what they change is unknown",
    ),
    1: (
        "image count",
        (0,),
        "0 means the count is the number of index entries. lvpyio takes a "
        "non-zero value as the count even where it hides entries or invents "
        "ones that do not exist",
    ),
    5: (
        "bits per decoded pixel",
        (16,),
        "At 8 lvpyio keeps only the low byte of each decoded pixel, and it opens "
        "no other value",
    ),
    # 1 is accepted beside the measured 0 because lvpyio reads both alike; 2 or
    # more fails or crashes lvpyio (HYPOTHESIS: planes per frame).
    14: (
        "planes per frame",
        (0, 1),
        "lvpyio fails or crashes on 2 or more",
    ),
    15: (
        "sample type",
        (0,),
        "0 is uint16 counts; 1 and 2 are float32 and float64 streams",
    ),
}

# Bytes 24..55 name the compression, NUL-terminated and case-sensitive. These
# are the names lvpyio opens a stream for; it opens none for "zlib", "ZLIB",
# "lz4" or "None" (container-unknowns REPORT §2 C1). Real files carry "" or
# "IMX".
_IMS_CODEC_FIELD = slice(24, 56)
_IMS_CODECS = ("", "Uncompressed", "Zlib", "LZ4", "IMX")
_IMS_UNCOMPRESSED_CODECS = ("", "Uncompressed")

# The data file every entry must name. A stream split across {prefix}-1.ims,
# {prefix}-2.ims, ... is read by lvpyio entry by entry, but no real recording on
# any surveyed drive has one, and when DaVis writes one is unknown. File number 0
# names the index itself and is invalid in lvpyio too.
_IMS_DATA_FILE_NO = 1


def _ims_header_word(data: bytes, word: int) -> int:
    """int32 header word ``word`` of an index file's bytes."""
    return _IMS_WORD.unpack_from(data, 4 * word)[0]


def _parse_ims_index(
    index_path: Path, prefix: str, frames_per_image: int
) -> Tuple[int, int, str, List[IMSIndexEntry]]:
    """Parse a {prefix}-0.ims index file, refusing every unmeasured layout.

    Args:
        index_path: the index file.
        prefix: the stream's file prefix, used to name data files in refusals.
        frames_per_image: the span of the stream's FrameReader
            (``EndFrame - StartFrame + 1``), which header word 2 must equal.

    Returns:
        (width, height, codec, entries).

    Raises:
        ValueError: If the file is shorter than the header, holds a partial
            entry, holds an entry count that is not a whole number of images at
            ``frames_per_image`` frames each, or any checked header word or entry
            field differs from the measured value. The message names the field
            and the value found.
    """
    data = index_path.read_bytes()
    where = f"{index_path.name} in {index_path.parent}"
    if len(data) < _IMS_HEADER_BYTES:
        raise ValueError(
            f"{where} is {len(data)} bytes, shorter than the {_IMS_HEADER_BYTES}-"
            f"byte .ims index header. The index is truncated."
        )
    stray = (len(data) - _IMS_HEADER_BYTES) % _IMS_ENTRY.size
    if stray:
        raise ValueError(
            f"{where} holds {stray} bytes after the last whole {_IMS_ENTRY.size}-"
            f"byte entry. The index is truncated or not an .ims index."
        )

    for word, (name, accepted, why) in _IMS_REQUIRED_WORDS.items():
        value = _ims_header_word(data, word)
        if value not in accepted:
            shown = " or ".join(str(a) for a in accepted)
            raise ValueError(
                f"{where}: index header word {word} ({name}) is {value}; PIVTOOLs "
                f"reads only {shown}. {why}. No real recording seen so far "
                f"carries this value, so it is refused rather than read on a guess."
            )

    word2 = _ims_header_word(data, _IMS_WORD_FRAMES_PER_IMAGE)
    if word2 != frames_per_image:
        raise ValueError(
            f"{where}: index header word {_IMS_WORD_FRAMES_PER_IMAGE} (frames per "
            f"image) is {word2}, but StreamSet.xml's FrameReader '{prefix}' spans "
            f"{frames_per_image} frame(s) (EndFrame - StartFrame + 1). The two must "
            f"agree; lvpyio refuses this as a frame count inconsistency."
        )

    codec_bytes = data[_IMS_CODEC_FIELD].split(b"\0", 1)[0]
    codec = codec_bytes.decode("latin-1")
    if codec not in _IMS_CODECS:
        raise ValueError(
            f"{where}: compression name at index header bytes 24..55 is "
            f"{codec!r}; PIVTOOLs knows {', '.join(repr(c) for c in _IMS_CODECS)}. "
            f"Names are case-sensitive, and lvpyio opens no stream for any other."
        )

    width = _ims_header_word(data, _IMS_WORD_WIDTH)
    height = _ims_header_word(data, _IMS_WORD_HEIGHT)

    entries = [
        IMSIndexEntry(*_IMS_ENTRY.unpack_from(data, pos))
        for pos in range(_IMS_HEADER_BYTES, len(data), _IMS_ENTRY.size)
    ]
    partial = len(entries) % frames_per_image
    if partial:
        # lvpyio counts ceil(entries / word2) images and fails reading the last
        # one (container-unknowns REPORT §2 A1); no real file has a partial image.
        raise ValueError(
            f"{where} holds {len(entries)} entries at {frames_per_image} frames per "
            f"image (header word {_IMS_WORD_FRAMES_PER_IMAGE}), so its last image "
            f"has only {partial} of its {frames_per_image} frames. The recording "
            f"is truncated; lvpyio fails reading that image too."
        )
    foreign = [k for k, e in enumerate(entries) if e.file_no != _IMS_DATA_FILE_NO]
    if foreign:
        k = foreign[0]
        file_no = entries[k].file_no
        what = (
            "file number 0, which names the index itself and is invalid"
            if file_no == 0
            else f"data file {prefix}-{file_no}.ims"
        )
        raise ValueError(
            f"{where}: entry {k} ({len(foreign)} of {len(entries)} entries in all) "
            f"names {what}, not {prefix}-{_IMS_DATA_FILE_NO}.ims. PIVTOOLs reads "
            f"only streams held in one data file: no real recording seen so far is "
            f"split across files, so this layout is refused rather than read on a "
            f"guess."
        )
    return width, height, codec, entries


# ---------------------------------------------------------------------------
# Decoder detection
# ---------------------------------------------------------------------------


def _read_decoder(decoder_path: Path) -> str:
    """Read the pixel encoding id from Frame{N}-decoder.xml.

    The encoding is never guessed. This used to default to ``"mono-16"`` (raw
    uint16) when the XML was missing or held no ``<id>``, which silently decoded
    packed data as unpacked and produced a scrambled image instead of an error.
    ``mono-16`` is not a DaVis id at all -- it was our own placeholder.

    Raises:
        FileNotFoundError: If the decoder XML is absent.
        ValueError: If the XML holds no usable ``<id>``.
    """
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Pixel encoding unknown: {decoder_path.name} is missing from "
            f"{decoder_path.parent}. The encoding will not be guessed -- copy the "
            f"complete companion folder, or re-export the recording from DaVis."
        )
    root = ET.parse(decoder_path).getroot()
    id_elem = root.find("id")
    if id_elem is None or not id_elem.text or not id_elem.text.strip():
        raise ValueError(
            f"Pixel encoding unknown: {decoder_path} has no <id> element. "
            f"The encoding will not be guessed."
        )
    return id_elem.text.strip()


# ---------------------------------------------------------------------------
# Stream transformers (post-decode corrections declared by DaVis)
# ---------------------------------------------------------------------------

# StreamSet.xml binds post-decode transformers to a stream; DaVis (and lvpyio)
# apply them on read, so skipping one silently returns different pixels from
# what DaVis shows for the same recording. The 13 ids are LaVision's list, read
# from lvpyio 1.3.1's Core.DataObjects.dll string table (01-lavision-readers
# REPORT B3). The effect of the 10 below was measured against lvpyio on
# synthetic non-square recordings (y down, row 0 at top; rotation-shape REPORT
# A1, A3); dark-image-subtraction, rotate-180 and intensity-correction-1 also on
# real recordings. Every transformer declares a FilePrefix, but only the three
# corrections own a file.
#
# Order (transformer-order REPORT R1, 324 of 324 synthetic cases and two real
# recordings): lvpyio applies the corrections in the order of _CORRECTIONS, in
# sensor coordinates, each clipped to its own range, then one geometry, whatever
# order the manifest lists them in and whatever their FilePrefixes. Dark and
# background commute (both orders are max(x - D - B, 0)).

# lvpyio adds this pedestal inside intensity-correction-1. Measured constant:
# lvpyio - raw + T == 300 on 44 M real NTNU pixels, and still 300 with T constant
# 1000, so it is not derived from T (01-lavision-readers REPORT B1).
_SCMOS1_PEDESTAL = 300
_UINT16_MAX = np.iinfo(np.uint16).max


def _subtract_clamped(img: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """``max(img - corr, 0)`` in uint16: one full-frame allocation plus the mask.

    The uint16 subtraction wraps where corr > img, and the wrapped elements are
    then zeroed in place, so they never reach the output. Not done in place on
    ``img``: _decode_raw16 returns a read-only np.frombuffer view. Measured on a
    5312x3528 frame: 23.0 ms here, against 42.9 ms for
    ``np.where(img > corr, img - corr, 0).astype(np.uint16)``, which allocates four
    full-frame temporaries.
    """
    out = img - corr
    np.copyto(out, 0, where=img <= corr)
    return out


def _scmos_correction_1(img: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """``clip(img - corr + 300, 0, 65535)``, DaVis 'Intensity correction 1'.

    int32 holds every intermediate: -65535 + 300 at the low end, 65535 + 300 at
    the high end. Both clips are observable: a raw pixel near saturation with
    corr below 300 exceeds 65535 (transformer-order REPORT Q3).
    """
    out = img.astype(np.int32)
    out -= corr
    out += _SCMOS1_PEDESTAL
    np.clip(out, 0, _UINT16_MAX, out=out)
    return out.astype(np.uint16)


@dataclass(frozen=True)
class _Correction:
    """A transformer that subtracts the per-pixel image ``{FilePrefix}-{suffix}.im7``.

    ``apply(decoded, image)`` takes both in sensor shape, uint16, and returns a
    new uint16 frame clipped to that step's range.
    """

    label: str
    suffix: str
    apply: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _Geometry:
    """A file-less transformer that rearranges pixels.

    ``swaps_axes`` is True when the output is (W, H) of the stored frame. ``apply``
    returns a negative-stride or transposed VIEW, deliberately not materialised:
    every caller performs exactly one widening copy into float32 (``result[i] =
    img``, ``np.copyto``, ``astype``), which absorbs any source strides, and none
    of them mutates the returned array. On a 90 degree view that copy costs about
    68 ms more per 5120 x 5120 frame (rotation-shape REPORT C6), accepted for now.
    """

    label: str
    swaps_axes: bool
    apply: Callable[[np.ndarray], np.ndarray]


# In LaVision's application order.
_CORRECTIONS: Dict[str, _Correction] = {
    "intensity-correction-1": _Correction(
        "Intensity correction 1", "scmos-1", _scmos_correction_1
    ),
    "dark-image-subtraction": _Correction(
        "Dark image subtraction", "dark", _subtract_clamped
    ),
    "background-subtraction": _Correction(
        "Background subtraction", "bgnd", _subtract_clamped
    ),
}

_GEOMETRIES: Dict[str, _Geometry] = {
    "rotate-180": _Geometry("Rotation by 180", False, lambda x: x[::-1, ::-1]),
    "vertical-mirror": _Geometry("Vertical mirroring", False, lambda x: x[::-1]),
    # LaVision's spelling. The correctly spelled id opens 0 buffers in lvpyio,
    # so it is not matched (it is refused as unknown).
    "horinzontal-mirror": _Geometry(
        "Horizontal mirroring", False, lambda x: x[:, ::-1]
    ),
    # Clockwise as displayed with row 0 at the top: the sensor's top-left pixel
    # lands top-right. np.rot90 turns from axis 0 towards axis 1, so k=-1.
    "rotate-cw-90": _Geometry(
        "Clockwise rotation by 90", True, lambda x: np.rot90(x, -1)
    ),
    "rotate-ccw-90": _Geometry(
        "Counterclockwise rotation by 90", True, lambda x: np.rot90(x, 1)
    ),
    # Rotate, then mirror rows: np.rot90(x, -1)[::-1] is x[::-1, ::-1].T.
    "rotate-cw-90-vertical-mirror": _Geometry(
        "Clockwise rotation by 90 plus vertical mirror",
        True,
        lambda x: x[::-1, ::-1].T,
    ),
    # np.rot90(x, 1)[::-1] is x.T.
    "rotate-ccw-90-vertical-mirror": _Geometry(
        "Counterclockwise rotation by 90 plus vertical mirror", True, lambda x: x.T
    ),
}

# Known DaVis ids that are refused when their stream is read: lvpyio opens no
# synthetic recording carrying any of them, so there is no oracle for their
# effect (01-lavision-readers REPORT B3). id: DaVis label.
_REFUSED_TRANSFORMERS: Dict[str, str] = {
    "white-image-correction": "White image subtraction",
    "add-offset": "Add fixed pixel offset",
    "image-intensity-correction-2": "Intensity correction 2",
}

# Position of each id in a stream's transformer tuple. A refused id never runs
# (its stream is refused first), so where it sits among the others is moot.
_APPLICATION_RANK: Dict[str, int] = {
    tid: rank
    for rank, tid in enumerate([*_CORRECTIONS, *_REFUSED_TRANSFORMERS, *_GEOMETRIES])
}

# StreamSet.xml is the companion folder's manifest. ReaderInfo nodes declare each
# file group by FilePrefix and bind it to a frame-stream index through
# ContentPurpose StartFrame. The prefix is DaVis's choice per recording, not a
# format constant: "Frame0".. on one recording, "Camera1".. on another, with
# scale files "FrameScales0" / "CameraScale1" respectively. Both seen in the wild.
_FRAME_READER_TYPE = "Core.Set.Recording.FrameReader"
_SCALE_READER_TYPE = "Core.Set.Recording.ScaleReader"
_ATTRIBUTE_READER_TYPE = "Core.Set.Recording.AttributeReader"
# The Name of every AttributeReader on the surveyed drives (DaVis 10.2.1, 11.4).
_ATTRIBUTE_READER_NAME = "AttributeReaderV1"

# The FrameReader Name of a LaVision image stream. lvpyio opens a stream only
# for this exact, case-sensitive Name, and one foreign Name makes it open no
# image of the whole recording (container-unknowns REPORT §2 F1, F2). It is the
# only Name on the 24 FrameReaders of the surveyed drives.
_FRAME_READER_NAME = "Images_16bit_8Bit"

# Reader Names LaVision's own DLL describes for other vendors' camera formats
# ("Reads Photron MRAW format.", "Reads Vision Research cine X format."), matched
# case-insensitively so the refusal names the vendor. Upper-cased Name: advice.
_FOREIGN_READER_NAMES = {
    "MRAW": "a Photron MRAW stream. PIVTOOLs has no Photron reader; export the "
    "images from DaVis as .im7 or TIFF",
    "GRAW": "a Photron GRAW stream. PIVTOOLs has no Photron reader; export the "
    "images from DaVis as .im7 or TIFF",
    "CINE": "a Vision Research cine stream. PIVTOOLs reads the original .cine "
    "file directly with image_type 'cine'",
}


def _load_stream_manifest(set_dir: Path) -> Optional[ET.Element]:
    """Parse StreamSet.xml once, or return None when the recording has none.

    Raises:
        ValueError: If the file exists but is not valid XML. The manifest drives
            both stream discovery and the corrections, so a broken one cannot be
            worked around.
    """
    stream_xml = set_dir / "StreamSet.xml"
    if not stream_xml.exists():
        return None
    try:
        return ET.parse(stream_xml).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{stream_xml} is not valid XML: {exc}") from exc


def _bound_frame_span(node: ET.Element, what: str, stream_xml: Path) -> Tuple[int, int]:
    """The frame positions ``(StartFrame, EndFrame)`` a manifest node is bound to.

    ``ContentPurpose StartFrame``/``EndFrame`` are frame positions within one
    image buffer, inclusive (container-unknowns REPORT §2 H1). Every failure
    names the node and what is wrong: a bare ``int()`` traceback or a KeyError
    would be the only unnamed failure in this module.

    Raises:
        ValueError: If ContentPurpose or either bound is missing, a bound is not
            an integer, StartFrame is negative, or EndFrame precedes StartFrame.
    """
    purpose = node.find("ContentPurpose")
    if purpose is None:
        raise ValueError(
            f"{what} in {stream_xml} has no ContentPurpose, so the frame stream "
            f"it applies to is unknown."
        )
    bounds = []
    for attr in ("StartFrame", "EndFrame"):
        text = purpose.get(attr)
        try:
            bounds.append(int(text))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{what} in {stream_xml} gives {attr} '{text}', which is not a "
                f"frame-stream number. The stream it belongs to cannot be "
                f"determined."
            ) from exc
    start, end = bounds
    if start < 0:
        raise ValueError(
            f"{what} in {stream_xml} gives StartFrame {start}; frame positions "
            f"start at 0."
        )
    if end < start:
        raise ValueError(
            f"{what} in {stream_xml} spans frames {start}..{end}: EndFrame comes "
            f"before StartFrame, so it is bound to no frame."
        )
    return start, end


def _bound_stream_index(node: ET.Element, what: str, stream_xml: Path) -> int:
    """The one frame position a ScaleReader or Transformer node is bound to.

    Only a FrameReader may span several frames (:func:`_streams_from_manifest`).
    Every scale file and correction on the surveyed drives is bound to exactly
    one frame, including on the two recordings whose FrameReader spans A and B
    (each frame gets its own ``CameraScale{N}``), so a span here is refused
    rather than applied to every frame on a guess.

    Raises:
        ValueError: Via :func:`_bound_frame_span`, or if the node spans frames.
    """
    start, end = _bound_frame_span(node, what, stream_xml)
    if start != end:
        raise ValueError(
            f"{what} in {stream_xml} spans frames {start}..{end}. Scale files and "
            f"corrections are read only when bound to exactly one frame; no real "
            f"recording seen so far binds one to several."
        )
    return start


def _check_frame_reader(node: ET.Element, prefix: str, stream_xml: Path) -> None:
    """Refuse a FrameReader that is not a LaVision image stream bound to frames.

    Raises:
        ValueError: If its Name is a foreign camera format, is any Name other
            than ``Images_16bit_8Bit``, or its ContentPurpose is not
            ``IsAssociatedToFrames="true"``. The message names the attribute and
            its value.
    """
    name = node.get("Name") or ""
    what = f"FrameReader '{prefix}' in {stream_xml}"
    if name.upper() in _FOREIGN_READER_NAMES:
        raise ValueError(
            f"{what} has Name '{name}': {_FOREIGN_READER_NAMES[name.upper()]}."
        )
    if name != _FRAME_READER_NAME:
        raise ValueError(
            f"{what} has Name '{name}'. The only LaVision image-stream reader is "
            f"'{_FRAME_READER_NAME}' (case-sensitive); lvpyio opens no image of "
            f"this recording either."
        )
    purpose = node.find("ContentPurpose")
    associated = None if purpose is None else purpose.get("IsAssociatedToFrames")
    if purpose is not None and associated != "true":
        raise ValueError(
            f"{what}: ContentPurpose IsAssociatedToFrames is '{associated}'; "
            f"an image stream is associated to frames ('true'). lvpyio refuses "
            f"this recording too."
        )


def _streams_from_manifest(
    set_dir: Path, root: ET.Element, frame_readers: List[ET.Element]
) -> List[_StreamSource]:
    """Frame streams as StreamSet.xml declares them, one per frame position.

    A FrameReader spanning ``StartFrame..EndFrame`` is expanded into one stream
    per frame it covers. The order is the frame position, not the declaration
    order: lvpyio returns a reader declared second but bound to frame 0 as the
    buffer's first frame (container-unknowns REPORT §2 H1).

    Callers locate a camera's stream positionally (``(camera-1) * fpc``), so the
    covered positions must be exactly 0..N-1. DaVis can write a sparse manifest
    when a camera is disabled mid-session; a gap is reported as such rather than
    silently shifting every camera after it.
    """
    stream_xml = set_dir / "StreamSet.xml"

    # frame position -> (prefix, StartFrame, EndFrame) of the reader covering it
    owners: Dict[int, Tuple[str, int, int]] = {}
    for node in frame_readers:
        prefix = (node.get("FilePrefix") or "").strip()
        if not prefix:
            raise ValueError(
                f"A FrameReader in {stream_xml} has no FilePrefix, so its stream "
                f"files cannot be located."
            )
        _check_frame_reader(node, prefix, stream_xml)
        start, end = _bound_frame_span(node, f"FrameReader '{prefix}'", stream_xml)
        for frame in range(start, end + 1):
            if frame in owners:
                raise ValueError(
                    f"{stream_xml.name} binds two image streams to frame stream "
                    f"{frame}: '{owners[frame][0]}' and '{prefix}'. Camera mapping "
                    f"is positional, so this cannot be resolved by choosing one."
                )
            owners[frame] = (prefix, start, end)

    found = sorted(owners)
    if found != list(range(len(found))):
        raise ValueError(
            f"{stream_xml.name} declares StartFrames {found}; expected contiguous "
            f"0..{len(found) - 1}. Cameras are located positionally, so a gap "
            f"would silently shift every camera after it. DaVis writes a sparse "
            f"manifest when a camera is disabled mid-session -- check the "
            f"recording, this is its manifest, not a reader fault."
        )

    scale_prefixes: dict = {}
    for node in root.iter("ReaderInfo"):
        if node.get("Type") != _SCALE_READER_TYPE:
            continue
        prefix = (node.get("FilePrefix") or "").strip()
        index = _bound_stream_index(node, f"ScaleReader '{prefix}'", stream_xml)
        if index in scale_prefixes:
            raise ValueError(
                f"{stream_xml.name} binds two scale files to frame stream {index}: "
                f"'{scale_prefixes[index]}' and '{prefix}'."
            )
        scale_prefixes[index] = prefix

    return [
        _StreamSource(
            index=i,
            prefix=owners[i][0],
            scale_prefix=scale_prefixes.get(i),
            frames_per_image=owners[i][2] - owners[i][1] + 1,
            span_offset=i - owners[i][1],
        )
        for i in found
    ]


def _discover_streams(set_dir: Path, root: Optional[ET.Element]) -> List[_StreamSource]:
    """Every frame stream StreamSet.xml declares, in stream order.

    The manifest is the only source. Guessing streams from ``Frame{N}`` file
    names read sets lvpyio opens no image for (a foreign FrameReader type, or no
    FrameReader at all), and named nothing when the guess was wrong.

    Raises:
        FileNotFoundError, ValueError: Via :func:`_raise_no_streams` when the
            recording has no StreamSet.xml or it declares no FrameReader.
    """
    frame_readers = (
        []
        if root is None
        else [n for n in root.iter("ReaderInfo") if n.get("Type") == _FRAME_READER_TYPE]
    )
    if not frame_readers:
        _raise_no_streams(set_dir, manifest_present=root is not None)
    return _streams_from_manifest(set_dir, root, frame_readers)


def _attribute_bindings(set_dir: Path, root: Optional[ET.Element]) -> List[AttrBinding]:
    """The AttributeReaders StreamSet.xml declares, in declaration order.

    Each binds ``{FilePrefix}.attr`` to one frame stream (``IsAssociatedToFrames
    ="true"``, ``StartFrame == EndFrame``) or to the whole image buffer
    (``IsAssociatedToFrames="false" IsAssociatedToEntireImage="true"``), the
    only two forms on the surveyed drives (attr-decode REPORT A4).

    Raises:
        AttrFormatError: If a reader has no FilePrefix, a Name other than
            ``AttributeReaderV1`` (the only one seen; another may be another
            layout), or any other binding.
        ValueError: Via :func:`_bound_stream_index` for a frame binding with a
            bad or spanning frame range.
    """
    if root is None:
        return []
    stream_xml = set_dir / "StreamSet.xml"
    bindings: List[AttrBinding] = []
    for node in root.iter("ReaderInfo"):
        if node.get("Type") != _ATTRIBUTE_READER_TYPE:
            continue
        prefix = (node.get("FilePrefix") or "").strip()
        what = f"AttributeReader '{prefix}'"
        if not prefix:
            raise AttrFormatError(
                f"An AttributeReader in {stream_xml} has no FilePrefix, so its "
                f".attr file cannot be located."
            )
        name = node.get("Name") or ""
        if name != _ATTRIBUTE_READER_NAME:
            raise AttrFormatError(
                f"{what} in {stream_xml} has Name '{name}'. The only attribute "
                f"reader seen is '{_ATTRIBUTE_READER_NAME}'; another may be another "
                f"file layout, so it is refused rather than parsed on a guess."
            )
        purpose = node.find("ContentPurpose")
        frames = None if purpose is None else purpose.get("IsAssociatedToFrames")
        entire = None if purpose is None else purpose.get("IsAssociatedToEntireImage")
        if (frames, entire) == ("false", "true"):
            bindings.append(AttrBinding(prefix, None))
        elif (frames, entire) == ("true", "false"):
            bindings.append(
                AttrBinding(prefix, _bound_stream_index(node, what, stream_xml))
            )
        else:
            raise AttrFormatError(
                f"{what} in {stream_xml} has IsAssociatedToFrames={frames!r} and "
                f"IsAssociatedToEntireImage={entire!r}. Only a file bound to one "
                f"frame or to the whole image is understood."
            )
    return bindings


# ``SetIsSuspicious = 1;`` in the .set file's "#GROUP Sets" block. DaVis 11.4
# wrote it on both recordings whose camera keys stop partway through, one of
# which holds blank images 501..899 (attr-decode REPORT C3, D5 ledger note).
_SUSPICIOUS_LINE = re.compile(rb"^SetIsSuspicious\s*=\s*([^;\r\n]*);", re.MULTILINE)


def _read_suspicious_flag(set_path: Path) -> Optional[str]:
    """The text of the ``.set`` file's ``SetIsSuspicious`` key, None when absent.

    A missing ``.set`` file has no flag to read; the companion folder alone is
    what the streams come from.
    """
    if not set_path.is_file():
        return None
    match = _SUSPICIOUS_LINE.search(set_path.read_bytes())
    return None if match is None else match.group(1).decode("latin-1").strip()


def _parse_stream_transformers(
    set_dir: Path, root: Optional[ET.Element]
) -> Dict[int, Tuple[StreamTransformer, ...]]:
    """Map frame-stream index to its declared transformers, from StreamSet.xml.

    DaVis declares each transformer as, for example::

        <Transformer ID="dark-image-subtraction" Label="Dark image subtraction"
                     FilePrefix="Transformer0" MinDaVisVersion="10.2.0">
            <ContentPurpose IsAssociatedToFrames="true" StartFrame="0" EndFrame="0"/>
        </Transformer>

    A stream may carry each correction once and at most one geometry. They are
    returned in LaVision's application order, not the manifest's. Refused ids
    are recorded here and refused when THAT stream is read. An id outside the
    13 DaVis ids is recorded too: :func:`_block_unknown_transformers` decides,
    once the camera labels are known, whether it blocks one camera or the whole
    recording. Everything else that is wrong is refused now.

    Args:
        set_dir: The .set companion directory.
        root: Parsed StreamSet.xml, or None when the recording has none.

    Returns:
        {stream index: transformers in application order}. Empty when nothing is
        declared.

    Raises:
        ValueError: If a transformer's binding is missing or spans several
            streams; a stream declares one id twice or two geometries (lvpyio
            silently keeps only the last of each, and this reader drops nothing
            silently).
        FileNotFoundError: If a declared correction image is not on disk.
    """
    if root is None:
        return {}
    stream_xml = set_dir / "StreamSet.xml"

    declared: Dict[int, List[Tuple[StreamTransformer, str]]] = {}
    for node in root.iter("Transformer"):
        tid = (node.get("ID") or "").strip()
        label = (node.get("Label") or "").strip()
        prefix = (node.get("FilePrefix") or "").strip()
        stream_index = _bound_stream_index(node, f"Transformer '{prefix}'", stream_xml)

        on_stream = declared.setdefault(stream_index, [])
        for other, other_prefix in on_stream:
            if other.id == tid:
                raise ValueError(
                    f"{stream_xml.name} declares '{tid}' twice for frame stream "
                    f"{stream_index} (FilePrefix '{other_prefix}' and '{prefix}'). "
                    f"lvpyio applies only the later declaration and drops the other "
                    f"without a message; PIVTOOLs refuses rather than drop one."
                )
            if tid in _GEOMETRIES and other.id in _GEOMETRIES:
                raise ValueError(
                    f"{stream_xml.name} declares two geometric transformers for "
                    f"frame stream {stream_index}: '{other.id}' (FilePrefix "
                    f"'{other_prefix}') and '{tid}' (FilePrefix '{prefix}'). lvpyio "
                    f"applies only the one listed last and drops the other without "
                    f"a message; PIVTOOLs refuses rather than drop one."
                )

        correction_path = None
        correction = _CORRECTIONS.get(tid)
        if correction is not None:
            correction_path = set_dir / f"{prefix}-{correction.suffix}.im7"
            if not correction_path.exists():
                raise FileNotFoundError(
                    f"{stream_xml.name} declares '{tid}' ({correction.label}) for "
                    f"frame stream {stream_index}, but {correction_path.name} is "
                    f"missing from {set_dir}. Copy the complete companion folder -- "
                    f"the correction is part of the recording, not an optional extra."
                )

        on_stream.append(
            (
                StreamTransformer(id=tid, label=label, correction_path=correction_path),
                prefix,
            )
        )

    # An unknown id never runs (its camera is blocked first), so where it sorts
    # is moot; last keeps the known ones in LaVision's order.
    unknown_rank = len(_APPLICATION_RANK)
    return {
        index: tuple(
            sorted(
                (tf for tf, _prefix in found),
                key=lambda tf: _APPLICATION_RANK.get(tf.id, unknown_rank),
            )
        )
        for index, found in declared.items()
    }


def _unknown_transformer_text(frame: IMSFrameInfo, set_name: str) -> str:
    """What is unknown about a stream's transformer id, in plain words."""
    tf = next(t for t in frame.transformers if t.id not in _APPLICATION_RANK)
    return (
        f"StreamSet.xml of {set_name} declares the transformer '{tf.id}' (DaVis "
        f"label '{tf.label}') for frame stream {frame.frame_idx}. It is not one of "
        f"the 13 DaVis transformer ids PIVTOOLs knows, so its effect on the frame, "
        f"and on the frame's shape, is unknown. PIVTOOLs applies: "
        f"{', '.join([*_CORRECTIONS, *_GEOMETRIES])}."
    )


def _block_unknown_transformers(
    set_name: str,
    frames: List[IMSFrameInfo],
    cameras: Optional[Tuple[LabelledCamera, ...]],
) -> None:
    """Refuse, or block one camera for, each transformer id PIVTOOLs does not know.

    With camera labels, the camera owning such a stream is blocked: every one of
    its streams carries the reason in ``blocked`` and refuses to read, while the
    other cameras read (decided 17 Nov). Without labels, cameras are mapped to
    streams by shape and this stream's is unknown, so the recording is refused.

    Raises:
        ValueError: Without camera labels, for the first such stream.
    """
    unknown = [f for f in frames if f.has_unknown_transformer]
    if not unknown:
        return
    if cameras is None:
        raise ValueError(
            f"{_unknown_transformer_text(unknown[0], set_name)} The recording "
            f"carries no camera labels, so cameras are mapped to streams by shape, "
            f"and no camera of this recording can be read."
        )
    for frame in unknown:
        camera = next(c for c in cameras if frame.frame_idx in c.frames)
        reason = (
            f"Camera {camera.number}, {camera.describe()}, of {set_name} cannot be "
            f"read. {_unknown_transformer_text(frame, set_name)} The recording's "
            f"other cameras are unaffected."
        )
        for stream in camera.frames:
            if frames[stream].blocked is None:
                frames[stream].blocked = reason


def _load_correction(path_str: str) -> np.ndarray:
    """Load one correction image as uint16, cached and revalidated by mtime.

    The dark (``-dark.im7``), background (``-bgnd.im7``) and sCMOS
    (``-scmos-1.im7``) maps are each the full sensor -- 37 MB as uint16 for a
    5312x3528 camera -- so the cache trades memory for not re-reading one on
    every pair, which would otherwise dominate read time. The cache is per
    process, which is what Dask workers need.

    The mtime is part of the cache key, not decoration. Keying on the path alone
    means re-exporting a recording leaves the previous image cached in a
    long-lived process -- the Flask GUI outlives many recordings -- and every
    subsequent read then subtracts the wrong pixels with nothing to indicate it.
    Silently wrong data is the one outcome this reader refuses everywhere else, so
    it pays one stat per frame to make it impossible. A superseded entry ages out
    of the LRU normally; it is never served again.

    Args:
        path_str: Path to the correction .im7, as a string so it is hashable.

    Returns:
        np.ndarray: (H, W) uint16 correction image, in sensor coordinates.

    Raises:
        ValueError: If the image is not 2D, or is not integer counts in
            [0, 65535] -- the range the uint16 corrections assume.
        OSError: If the image is missing. _parse_stream_transformers already
            checks existence at parse time, so reaching this means it was deleted
            mid-session.
    """
    return _load_correction_cached(path_str, Path(path_str).stat().st_mtime)


@lru_cache(maxsize=4)
def _load_correction_cached(path_str: str, mtime: float) -> np.ndarray:
    """Cached body of :func:`_load_correction`. Call that, not this.

    ``mtime`` is a cache key only and is deliberately unused in the body: a changed
    image produces a different key and therefore a fresh read.

    Size is a compromise, not a fit. Four entries hold two cameras' worth of
    pre-paired streams (~150 MB). A rig with three or more corrected cameras
    whose worker interleaves them will evict and re-read; the alk235 recordings
    have ten such streams, so sizing to hold them all would cost ~370 MB per
    worker. Raise this only with a measurement showing the re-reads matter more
    than the resident memory.
    """
    from .im7_reader import read_im7

    _header, pixels, _scales = read_im7(path_str)
    image = np.squeeze(np.asarray(pixels))
    if image.ndim != 2:
        raise ValueError(
            f"Correction image {Path(path_str).name} is {image.ndim}-dimensional "
            f"(shape {image.shape}); expected a single 2D frame."
        )

    # read_im7 hands back float32 for a WORD buffer, so the integrality check is
    # needed -- but only for a float dtype. Running `image == np.floor(image)` on
    # an integer array promotes a 19 Mpx frame to float64 (~150 MB) to learn
    # nothing.
    lo, hi = image.min(), image.max()
    integral = np.issubdtype(image.dtype, np.integer) or bool(
        np.all(image == np.floor(image))
    )
    if not integral or lo < 0 or hi > _UINT16_MAX:
        raise ValueError(
            f"Correction image {Path(path_str).name} is not integer counts in "
            f"[0, 65535] (min {lo}, max {hi}, dtype {image.dtype}). The uint16 "
            f"correction would wrap and silently return wrong pixels."
        )
    return image.astype(np.uint16)


# ---------------------------------------------------------------------------
# Scale parsing
# ---------------------------------------------------------------------------


def _read_scales(scales_path: Optional[Path]) -> Tuple[float, float, str]:
    """Read intensity scale from a {scaleprefix}.scales XML.

    ``None`` (the manifest declares no scale file for this stream) means
    identity scale -- that is the declared state, not a fallback. A declared
    file that is missing is refused before this is called.
    """
    if scales_path is None:
        return 1.0, 0.0, ""
    tree = ET.parse(scales_path)
    root = tree.getroot()
    scale_i = root.find(".//ScaleI")
    if scale_i is None:
        return 1.0, 0.0, ""
    slope = float(scale_i.get("Factor", "1"))
    offset = float(scale_i.get("Offset", "0"))
    unit = scale_i.get("Unit", "")
    return slope, offset, unit


# ---------------------------------------------------------------------------
# Pixel decoding
# ---------------------------------------------------------------------------


# Every pixel encoding DaVis 10/11 can write, with its bit depth per pixel.
# Recovered from LaVision's own reader: lvpyio 1.3.1, io/Core.DataObjects.dll,
# UTF-16LE string table at 0x2ec5a8-0x2ec8f8, stored as id/label pairs. Bits
# rather than bytes keeps the 10-bit (1.25 B/px) and 12-bit (1.5 B/px) sizes
# exact under integer arithmetic.
_DECODER_BITS_PER_PX = {
    "mono-10p": 10,
    "mono-10pmsb": 10,
    "mono-12p": 12,
    "mono-12packed": 12,
    "mono-12pmsb": 12,
    "raw-16-bit": 16,
    "raw-8-bit": 8,
    "rgb-24": 24,
}

# The label DaVis shows for each id, so an error message names what the user sees
# in DaVis rather than only the internal id.
_DECODER_LABELS = {
    "mono-10p": "Mono10p to 16 bit",
    "mono-10pmsb": "Mono10pmsb to 16 bit",
    "mono-12p": "Mono12p to 16 bit",
    "mono-12packed": "Mono12packed to 16 bit",
    "mono-12pmsb": "Mono12pmsb to 16 bit",
    "raw-16-bit": "Raw 16 bit",
    "raw-8-bit": "8 to 16 bit",
    "rgb-24": "RGB 24",
}


def _decode_mono10p(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode Mono10p (USB3 Vision packed 10-bit) bytes to uint16 array.

    Packing: 4 pixels in 5 bytes, least-significant bits first, as one continuous
    little-endian bit stream::

        pixel0 =  b0       | (b1 & 0x03) << 8
        pixel1 = (b1 >> 2) | (b2 & 0x0F) << 6
        pixel2 = (b2 >> 4) | (b3 & 0x3F) << 4
        pixel3 = (b3 >> 6) |  b4         << 2

    Each is a little-endian uint16 window over two adjacent bytes, shifted down
    by its bit offset and masked to 10 bits -- the same memory-access trick as
    :func:`_decode_mono12p`, four overlapping windows at stride 5 instead of two
    at stride 3. Verified bit-exact against lvpyio 1.3.1 on the 1984x1264
    MiniShaker streams of a real five-camera recording (2026-08-24).

    Raises:
        ValueError: If the pixel count is not a multiple of 4. Guarded here
            because the generic byte-count check in :func:`_decode_pixels`
            floor-divides ``w*h*10/8`` and would accept a truncated payload for a
            malformed geometry.
    """
    n_pixels = width * height
    if n_pixels % 4:
        raise ValueError(
            f"mono-10p packs four pixels into five bytes, so a frame must hold a "
            f"multiple of four pixels; {width}x{height} is {n_pixels}. The declared "
            f"frame geometry does not match a mono-10p payload."
        )
    n_quads = n_pixels // 4

    # The final window starts at byte 5*n_quads - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read.
    def window(offset: int) -> np.ndarray:
        return np.ndarray(
            (n_quads,), dtype="<u2", buffer=raw, offset=offset, strides=(5,)
        )

    pixels = np.empty((n_quads, 4), dtype=np.uint16)
    np.bitwise_and(window(0), 0x03FF, out=pixels[:, 0])
    np.right_shift(window(1), 2, out=pixels[:, 1])
    np.bitwise_and(pixels[:, 1], 0x03FF, out=pixels[:, 1])
    np.right_shift(window(2), 4, out=pixels[:, 2])
    np.bitwise_and(pixels[:, 2], 0x03FF, out=pixels[:, 2])
    # (b3 | b4<<8) >> 6 is at most 10 bits wide, so no mask is needed.
    np.right_shift(window(3), 6, out=pixels[:, 3])
    return pixels.reshape(height, width)


def _decode_mono10pmsb(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode PFNC Mono10pmsb (msb packed 10-bit) bytes to a uint16 array.

    Packing: 4 pixels in 5 bytes, most-significant bits first, as one continuous
    big-endian bit stream (PFNC 2.4, section 6.3.2, figure 6-11)::

        pixel0 =  b0         << 2 | b1 >> 6
        pixel1 = (b1 & 0x3F) << 4 | b2 >> 4
        pixel2 = (b2 & 0x0F) << 6 | b3 >> 2
        pixel3 = (b3 & 0x03) << 8 | b4

    Each is a BIG-endian uint16 window over two adjacent bytes (``b_k << 8 |
    b_k+1``), shifted down by (6, 4, 2, 0) and masked to 10 bits: the mirror of
    :func:`_decode_mono10p`, same four windows at stride 5. ``>u2`` pins the byte
    order whatever the host. This is the encoding of the NTNU Imager MX PIV
    cameras. Verified bit-exact against lvpyio 1.3.1 on 16 real 5120x5120 NTNU
    frames and on every single-bit and every pixel-pair input (2026-11-14 and
    2026-11-16 research).

    Raises:
        ValueError: If the pixel count is not a multiple of 4, for the reason
            given in :func:`_decode_mono10p`.
    """
    n_pixels = width * height
    if n_pixels % 4:
        raise ValueError(
            f"mono-10pmsb packs four pixels into five bytes, so a frame must hold a "
            f"multiple of four pixels; {width}x{height} is {n_pixels}. The declared "
            f"frame geometry does not match a mono-10pmsb payload."
        )
    n_quads = n_pixels // 4

    # The final window starts at byte 5*n_quads - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read.
    def window(offset: int) -> np.ndarray:
        return np.ndarray(
            (n_quads,), dtype=">u2", buffer=raw, offset=offset, strides=(5,)
        )

    pixels = np.empty((n_quads, 4), dtype=np.uint16)
    # (b0<<8 | b1) >> 6 is at most 10 bits wide, so no mask is needed.
    np.right_shift(window(0), 6, out=pixels[:, 0])
    np.right_shift(window(1), 4, out=pixels[:, 1])
    np.bitwise_and(pixels[:, 1], 0x03FF, out=pixels[:, 1])
    np.right_shift(window(2), 2, out=pixels[:, 2])
    np.bitwise_and(pixels[:, 2], 0x03FF, out=pixels[:, 2])
    np.bitwise_and(window(3), 0x03FF, out=pixels[:, 3])
    return pixels.reshape(height, width)


def _decode_mono12p(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode PFNC Mono12p (lsb packed) bytes to a uint16 array.

    Packing: 2 pixels in 3 bytes.
        pixel0 = byte0 | (byte1 & 0x0F) << 8
        pixel1 = (byte1 >> 4) | (byte2 << 4)

    Those two expressions are the same bytes read as two OVERLAPPING little-endian
    uint16 windows, because little-endian stores the low byte first::

        uint16 at byte 3k+0  =  b0 | b1<<8   ->  & 0x0FFF  ==  pixel0
        uint16 at byte 3k+1  =  b1 | b2<<8   ->  >> 4      ==  pixel1

    Reading the windows directly replaces three stride-3 byte-plane gathers and six
    full-frame temporaries with two gathers and two ufuncs writing straight into the
    output columns. This is a memory-access change, not an arithmetic one: measured
    on a 5312x3528 frame, 69.6 -> 17.5 ms on x86-64 and 26.4 -> 11.8 ms on an Apple
    M4, bit-identical to the byte-plane form on six edge-case bit patterns, on random
    full-frame data, and on real recordings.

    The windows are deliberately unaligned (numpy reports ``ALIGNED=False``). Free on
    x86-64, mildly penalised on ARM -- which is why the ARM gain is the smaller one --
    but this remains the fastest of the tried formulations on both. ``<u2`` pins the
    byte order, so a big-endian host still decodes correctly, just via numpy's
    byte-swapping path.

    Raises:
        ValueError: If the frame holds an odd number of pixels. mono-12p packs two
            pixels per three bytes, so an odd count means the container's declared
            geometry is wrong. Guarded explicitly because the pair-wise form below
            would otherwise drop the trailing pixel silently.
    """
    n_pixels = width * height
    if n_pixels % 2:
        raise ValueError(
            f"mono-12p packs two pixels into three bytes, so a frame must hold an "
            f"even number of pixels; {width}x{height} is {n_pixels}. The declared "
            f"frame geometry does not match a mono-12p payload."
        )
    n_pairs = n_pixels // 2

    # The final window starts at byte 3*n_pairs - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read. A short buffer raises from
    # np.ndarray itself rather than decoding garbage.
    lo = np.ndarray((n_pairs,), dtype="<u2", buffer=raw, offset=0, strides=(3,))
    hi = np.ndarray((n_pairs,), dtype="<u2", buffer=raw, offset=1, strides=(3,))

    pixels = np.empty((n_pairs, 2), dtype=np.uint16)
    np.bitwise_and(lo, 0x0FFF, out=pixels[:, 0])
    np.right_shift(hi, 4, out=pixels[:, 1])
    return pixels.reshape(height, width)


def _require_even_pixels(decoder: str, width: int, height: int) -> int:
    """Pixel pairs in a 2-px-in-3-bytes frame, refusing an odd pixel count.

    Raises:
        ValueError: If the frame holds an odd number of pixels, for the reason
            given in :func:`_decode_mono12p`.
    """
    n_pixels = width * height
    if n_pixels % 2:
        raise ValueError(
            f"{decoder} packs two pixels into three bytes, so a frame must hold an "
            f"even number of pixels; {width}x{height} is {n_pixels}. The declared "
            f"frame geometry does not match a {decoder} payload."
        )
    return n_pixels // 2


def _decode_mono12pmsb(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode PFNC msb packed 12-bit bytes (DaVis 'mono-12pmsb') to a uint16 array.

    Packing: 2 pixels in 3 bytes, most-significant bits first, as one continuous
    big-endian bit stream::

        pixel0 =  b0         << 4 | b1 >> 4
        pixel1 = (b1 & 0x0F) << 8 | b2

    The mirror of :func:`_decode_mono12p` with BIG-endian windows::

        uint16 at byte 3k+0  =  b0<<8 | b1  ->  >> 4      ==  pixel0
        uint16 at byte 3k+1  =  b1<<8 | b2  ->  & 0x0FFF  ==  pixel1

    Layout settled against lvpyio 1.3.1, which decodes a synthetic recording
    carrying this id equal to this rule on all 2^24 three-byte groups, and to no
    other candidate layout (01-lavision-readers REPORT A3). No real recording in
    this encoding has been seen.

    Raises:
        ValueError: If the frame holds an odd number of pixels.
    """
    n_pairs = _require_even_pixels("mono-12pmsb", width, height)

    # The final window starts at byte 3*n_pairs - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read.
    first = np.ndarray((n_pairs,), dtype=">u2", buffer=raw, offset=0, strides=(3,))
    second = np.ndarray((n_pairs,), dtype=">u2", buffer=raw, offset=1, strides=(3,))

    pixels = np.empty((n_pairs, 2), dtype=np.uint16)
    np.right_shift(first, 4, out=pixels[:, 0])
    np.bitwise_and(second, 0x0FFF, out=pixels[:, 1])
    return pixels.reshape(height, width)


def _decode_mono12packed(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode DaVis 'mono-12packed' bytes to a uint16 array.

    Packing: 2 pixels in 3 bytes, the two low nibbles in the MIDDLE byte::

        pixel0 = b0 << 4 | (b1 & 0x0F)
        pixel1 = b2 << 4 | (b1 >> 4)

    This is not a PFNC packed style, and not PFNC Mono12g, which puts the nibble
    byte last. Layout settled against lvpyio 1.3.1, which decodes a synthetic
    recording carrying this id equal to this rule on all 2^24 three-byte groups,
    and to no other candidate layout (01-lavision-readers REPORT A3). No real
    recording in this encoding has been seen.

    Two windows give it without byte gathers:

        big-endian uint16 at 3k+0     =  b0<<8 | b1  ->  (w >> 4) & 0x0FF0 | w & 0x000F
        little-endian uint16 at 3k+1  =  b1 | b2<<8  ->   w >> 4

    Raises:
        ValueError: If the frame holds an odd number of pixels.
    """
    n_pairs = _require_even_pixels("mono-12packed", width, height)

    # The final window starts at byte 3*n_pairs - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read.
    first = np.ndarray((n_pairs,), dtype=">u2", buffer=raw, offset=0, strides=(3,))
    second = np.ndarray((n_pairs,), dtype="<u2", buffer=raw, offset=1, strides=(3,))

    pixels = np.empty((n_pairs, 2), dtype=np.uint16)
    np.right_shift(first, 4, out=pixels[:, 0])
    np.bitwise_and(pixels[:, 0], 0x0FF0, out=pixels[:, 0])
    np.bitwise_or(pixels[:, 0], np.bitwise_and(first, 0x000F), out=pixels[:, 0])
    np.right_shift(second, 4, out=pixels[:, 1])
    return pixels.reshape(height, width)


def _decode_raw16(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode raw uint16 little-endian bytes (DaVis 'raw-16-bit')."""
    return np.frombuffer(raw, dtype=np.dtype("<u2")).reshape(height, width)


def _decode_raw8(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode raw uint8 bytes to uint16 (DaVis 'raw-8-bit').

    Values are widened, NOT rescaled: an 8-bit count of 200 stays 200, it does not
    become 51200. This matches :func:`_decode_mono12p`, which returns 0-4095 even
    though its DaVis label also reads "to 16 bit". Every encoding in this module
    therefore returns raw sensor counts, so intensity thresholds (masking, peak
    magnitude) mean the same thing whichever encoding a recording used.
    """
    return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).reshape(height, width)


# What this reader decodes, and the function that decodes each id. One table, so
# "implemented" and "dispatched" cannot disagree. The 10-bit ids share a byte
# count, as do the 12-bit ones, and differ only in bit order, so size alone never
# picks a layout: each layout here was settled against lvpyio, not from its name.
# mono-10pmsb on real NTNU frames, the two 12-bit ids on all 2^24 three-byte groups
# of a synthetic recording lvpyio decodes (01-lavision-readers REPORT A1-A3).
# rgb-24 alone is refused: lvpyio 1.3.1 opens no synthetic rgb-24 recording, so
# there is no oracle for its channel order (REPORT A5).
#
# The cross-check against lvpyio IS a plain equality assertion, with two
# adjustments: lvpyio pads the frame height up to a multiple of 16 with zero rows
# (3528 -> 3536), and in a multi-camera buffer it pads every frame to the LARGEST
# frame's shape, unmasked (a 1984x1264 stream comes back as 2160x2560 with zeros
# outside the top-left block). Compare against its top-left `height` x `width`
# block. Everything else agrees bit for bit, declared corrections included. See
# [[piv-data-formats]].
_IMPLEMENTED_DECODERS: dict[str, Callable[[bytes, int, int], np.ndarray]] = {
    "mono-10p": _decode_mono10p,
    "mono-10pmsb": _decode_mono10pmsb,
    "mono-12p": _decode_mono12p,
    "mono-12packed": _decode_mono12packed,
    "mono-12pmsb": _decode_mono12pmsb,
    "raw-16-bit": _decode_raw16,
    "raw-8-bit": _decode_raw8,
}


def _payload_bytes(decoder: str, width: int, height: int) -> int:
    """Bytes of one packed frame in encoding ``decoder``, refusing unproven ids.

    Args:
        decoder: Encoding id from Frame{N}-decoder.xml.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Raises:
        ValueError: If the id is unknown to DaVis 10/11, or if it is a known
            DaVis encoding this reader does not implement.
    """
    implemented = ", ".join(_IMPLEMENTED_DECODERS)

    if decoder not in _DECODER_BITS_PER_PX:
        raise ValueError(
            f"Unknown .set pixel encoding '{decoder}'. It is not one of the eight "
            f"encodings DaVis 10/11 writes ({', '.join(_DECODER_BITS_PER_PX)}). "
            f"Check the <id> in the Frame{{N}}-decoder.xml of this recording. "
            f"PIVTOOLs decodes: {implemented}."
        )

    if decoder not in _IMPLEMENTED_DECODERS:
        raise ValueError(
            f"PIVTOOLs cannot decode this .set: pixel encoding '{decoder}' "
            f"(DaVis '{_DECODER_LABELS[decoder]}') is a recognised DaVis encoding "
            f"but is not implemented. PIVTOOLs decodes: {implemented}. Adding "
            f"'{decoder}' needs a sample recording to verify against -- send one to "
            f"the PIVTOOLs maintainers. Re-exporting from DaVis as Raw 16 bit is a "
            f"workaround in the meantime."
        )
    return (width * height * _DECODER_BITS_PER_PX[decoder]) // 8


def _decode_pixels(raw: bytes, decoder: str, width: int, height: int) -> np.ndarray:
    """Dispatch to the correct pixel decoder, refusing anything unproven.

    Args:
        raw: Packed pixel bytes for one entry: the stored bytes of an
            uncompressed entry, or the output of :func:`_decompress_entry`.
        decoder: Encoding id from Frame{N}-decoder.xml.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        np.ndarray: (height, width) uint16 of raw sensor counts.

    Raises:
        ValueError: If the id is unknown to DaVis 10/11, if it is a known DaVis
            encoding this reader does not implement, or if the byte count does not
            match the id's bit depth.
    """
    expected = _payload_bytes(decoder, width, height)
    if len(raw) != expected:
        bits = _DECODER_BITS_PER_PX[decoder]
        raise ValueError(
            f"Pixel data size mismatch for encoding '{decoder}': entry holds "
            f"{len(raw)} bytes but a {width}x{height} frame at {bits} bits/pixel "
            f"needs {expected}. The container may be truncated, or the encoding id "
            f"may not match the stored data."
        )

    return _IMPLEMENTED_DECODERS[decoder](raw, width, height)


# ---------------------------------------------------------------------------
# Entry decompression (index header bytes 24..55)
# ---------------------------------------------------------------------------


def _decompress_entry(
    stored: bytes, frame_info: IMSFrameInfo, entry_idx: int, expected: int
) -> Union[bytes, np.ndarray]:
    """Turn one entry's stored bytes into the packed payload the pixel decoder reads.

    The layouts are the ones lvpyio 1.3.1 opens (container-unknowns REPORT §2
    C1, C2, C5): Zlib is one bare RFC 1950 stream, LZ4 one bare block with no
    size prefix, and IMX the DaVis token grammar over the payload's
    little-endian uint16 words. An uncompressed entry is returned as stored, and
    :func:`_decode_pixels` checks its size.

    Args:
        stored: The entry's bytes as stored in {prefix}-1.ims.
        frame_info: The stream, for its codec and for naming it in a refusal.
        entry_idx: 0-based entry, for naming it in a refusal.
        expected: Packed payload bytes the decoder needs (:func:`_payload_bytes`).

    Returns:
        The packed payload, ``expected`` bytes long.

    Raises:
        IOError: If the entry does not decompress, decompresses to a size other
            than ``expected``, or holds bytes after its compressed data. The
            message names the stream, the entry and the codec.
    """
    codec = frame_info.codec
    if codec in _IMS_UNCOMPRESSED_CODECS:
        return stored

    where = (
        f"Frame stream {frame_info.frame_idx} of {frame_info.data_path.parent.name}, "
        f"entry_no {entry_idx + 1} ({frame_info.data_path.name}), stored with "
        f"'{codec}' compression"
    )
    if codec == "Zlib":
        return _decompress_zlib(stored, expected, where)
    if codec == "LZ4":
        try:
            return _lz4_block_decompress(stored, expected)
        except (OSError, LZ4BlockError) as exc:
            raise IOError(
                f"{where}: the LZ4 block does not decompress to the {expected} "
                f"bytes the frame's geometry and encoding need ({exc})"
            ) from exc
    if codec == "IMX":
        return _decompress_imx(stored, expected, where)
    raise RuntimeError(f"{where}: no decompressor for codec {codec!r}")


def _decompress_zlib(stored: bytes, expected: int, where: str) -> bytes:
    """Decompress one bare zlib stream to exactly ``expected`` bytes.

    ``max_length`` caps the output one byte past ``expected``, so an entry that
    inflates to more is caught without inflating all of it.
    """
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(stored, expected + 1)
    except zlib.error as exc:
        raise IOError(f"{where}: the zlib stream does not decompress ({exc})") from exc
    if len(out) > expected:
        raise IOError(
            f"{where}: the zlib stream decompresses to more than the {expected} "
            f"bytes the frame's geometry and encoding need"
        )
    if not inflater.eof:
        raise IOError(
            f"{where}: the zlib stream ends after {len(out)} of {expected} bytes, "
            f"without its end marker. The entry is truncated."
        )
    if len(out) != expected:
        raise IOError(
            f"{where}: the zlib stream decompresses to {len(out)} bytes, but the "
            f"frame's geometry and encoding need {expected}"
        )
    if inflater.unused_data:
        raise IOError(
            f"{where}: {len(inflater.unused_data)} bytes follow the end of the zlib "
            f"stream inside the entry. lvpyio's reading of such an entry is "
            f"unmeasured, so it is refused rather than read on a guess."
        )
    return out


def _decompress_imx(stored: bytes, expected: int, where: str) -> np.ndarray:
    """Decode one IMX entry to its ``expected`` packed payload bytes.

    The tokens code the payload's little-endian uint16 words, so a payload with
    an odd byte count has no measured layout (CU §2 C5: three layouts tried on
    mono-10p, all failed in lvpyio) and is refused.

    The token kernel takes only a writable uint8 array (one compiled
    specialisation), so the stored bytes are copied into one.

    Returns:
        A little-endian byte view of the decoded words.
    """
    if expected % 2:
        raise ValueError(
            f"{where}: IMX codes the packed payload as 16-bit words, but this "
            f"frame's geometry and encoding pack it into {expected} bytes, an odd "
            f"count. How DaVis stores the last byte is unmeasured, so the entry is "
            f"refused rather than read on a guess."
        )
    try:
        words = decode_imx_words(np.frombuffer(stored, np.uint8).copy(), expected // 2)
    except TokenStreamError as exc:
        raise type(exc)(f"{where}: {exc}") from exc
    return words.astype("<u2", copy=False).view(np.uint8)


# ---------------------------------------------------------------------------
# Set container parsing
# ---------------------------------------------------------------------------


def _im7_image_files(set_dir: Path) -> List[Path]:
    """The .im7 files in a companion folder that hold images, sorted.

    Transformer data files (Transformer1-scmos-1.im7, Transformer0-dark.im7) are
    .im7 too, and are NOT evidence of .im7 layout: advising lavision_im7 on their
    account would read a correction map as image data.
    """
    return sorted(
        p for p in set_dir.glob("*.im7") if not p.name.startswith("Transformer")
    )


def _parse_set(set_path: Union[str, Path]) -> SetInfo:
    """Parse a .set container, returning metadata for all frame streams.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file (the companion directory is derived from it).
    """
    set_path = Path(set_path)
    set_dir = set_path.with_suffix("")  # companion directory

    if not set_dir.is_dir():
        raise FileNotFoundError(
            f"{set_path.name} has no companion data folder (expected {set_dir}). "
            f"A DaVis recording is a .set file plus a folder of the same name holding "
            f"its .ims image streams; both must be copied together. A .set with no "
            f"folder is usually a DaVis result set (processed vectors), which holds no "
            f"images -- point at the recording .set instead."
        )

    root = _load_stream_manifest(set_dir)

    # DaVis can leave a StreamSet.xml declaring .ims streams beside images it
    # stored as .im7 files (E:\retro_tbl_35mm_far, drive-survey C7). Why is
    # unknown; the fix is the same either way, so it is named before the
    # manifest's streams are looked for and reported missing.
    im7_files = _im7_image_files(set_dir)
    if im7_files and not any(set_dir.glob("*.ims")):
        _raise_im7_layout(set_dir, im7_files, manifest_present=root is not None)

    sources = _discover_streams(set_dir, root)

    # Post-decode corrections DaVis declares for this recording (usually none).
    transformers = _parse_stream_transformers(set_dir, root)

    frames = []
    n_entries = None
    # A spanning FrameReader yields one source per frame over the same files, so
    # each index is parsed once: prefix -> (width, height, codec, entries).
    parsed_indexes: Dict[str, Tuple[int, int, str, List[IMSIndexEntry]]] = {}

    for src in sources:
        index_path = set_dir / f"{src.prefix}-0.ims"
        data_path = set_dir / f"{src.prefix}-1.ims"
        decoder_path = set_dir / f"{src.prefix}-decoder.xml"

        for required in (index_path, data_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"StreamSet.xml declares frame stream {src.index} with "
                    f"prefix '{src.prefix}', but {required.name} is missing "
                    f"from {set_dir}. Copy the complete companion folder."
                )

        scales_path = None
        if src.scale_prefix is not None:
            scales_path = set_dir / f"{src.scale_prefix}.scales"
            if not scales_path.exists():
                raise FileNotFoundError(
                    f"StreamSet.xml declares scale file '{src.scale_prefix}' for "
                    f"frame stream {src.index}, but {scales_path.name} is missing "
                    f"from {set_dir}. Copy the complete companion folder."
                )

        if src.prefix not in parsed_indexes:
            parsed_indexes[src.prefix] = _parse_ims_index(
                index_path, src.prefix, src.frames_per_image
            )
        stored_width, stored_height, codec, file_entries = parsed_indexes[src.prefix]
        # Frame `index` of buffer b is entry b * span + span_offset of the file.
        entries = file_entries[src.span_offset :: src.frames_per_image]
        n_ent = len(entries)
        decoder = _read_decoder(decoder_path)
        slope, offset, unit = _read_scales(scales_path)

        if n_entries is None:
            n_entries = n_ent
        elif n_ent != n_entries:
            raise ValueError(
                f"Frame stream {src.index} ('{src.prefix}') holds {n_ent} images "
                f"but frame stream {sources[0].index} ('{sources[0].prefix}') holds "
                f"{n_entries}. Every frame stream of a recording holds one frame of "
                f"each image."
            )

        frames.append(
            IMSFrameInfo(
                frame_idx=src.index,
                data_path=data_path,
                index_path=index_path,
                decoder=decoder,
                codec=codec,
                stored_shape=(stored_height, stored_width),
                n_entries=n_ent,
                scale_slope=slope,
                scale_offset=offset,
                scale_unit=unit,
                entries=entries,
                transformers=transformers.get(src.index, ()),
            )
        )

    bindings = _attribute_bindings(set_dir, root)
    rec = read_recording_attributes(set_dir, bindings, len(frames), n_entries)
    cameras = None if rec is None else group_cameras(rec)
    for camera in cameras or ():
        for stream, role in zip(
            camera.frames, ("A", "B") if camera.double_frame else ("",)
        ):
            frames[stream].camera_label = camera.label
            frames[stream].camera_id = camera.camera_id
            frames[stream].frame_role = role
    _block_unknown_transformers(set_path.name, frames, cameras)

    return SetInfo(
        set_dir=set_dir,
        frames=frames,
        n_entries=n_entries,
        cameras=cameras,
        attribute_files=tuple(f"{b.prefix}.attr" for b in bindings),
        suspicious_flag=_read_suspicious_flag(set_path),
    )


def read_set_attributes(set_path: Union[str, Path]) -> Optional[RecordingAttributes]:
    """Every attribute of a recording, per image, frame stream and whole image.

    For inspection and checks against lvpyio; the reader itself keeps only the
    camera identity (:attr:`SetInfo.cameras`), not the attributes, so a cached
    SetInfo stays small.

    Returns:
        None when StreamSet.xml declares no AttributeReader.

    Raises:
        Whatever :func:`read_set_info` raises for the recording.
    """
    set_path = Path(set_path)
    info = read_set_info(set_path)
    bindings = _attribute_bindings(info.set_dir, _load_stream_manifest(info.set_dir))
    return read_recording_attributes(
        info.set_dir, bindings, len(info.frames), info.n_entries
    )


def _raise_im7_layout(
    set_dir: Path, im7_files: List[Path], manifest_present: bool
) -> None:
    """Refuse a companion folder whose images are .im7 files, pointing at the
    .im7 image type.

    Args:
        set_dir: the companion folder.
        im7_files: its image .im7 files, from :func:`_im7_image_files`.
        manifest_present: whether a StreamSet.xml sits beside them, which DaVis
            leaves declaring .ims streams on some recordings.
    """
    shown = im7_files[0].name
    if len(im7_files) > 1:
        shown += f" and {len(im7_files) - 1} more"
    leftover = (
        " Its StreamSet.xml declares .ims streams, but none are stored here."
        if manifest_present
        else ""
    )
    raise FileNotFoundError(
        f"{set_dir} holds .im7 files ({shown}), not .ims frame streams.{leftover} "
        f"This recording is in DaVis .im7 layout: set image_type to 'lavision_im7' "
        f"and point the source path at the folder {set_dir} rather than at the "
        f".set file."
    )


def _raise_no_streams(set_dir: Path, manifest_present: bool) -> None:
    """Name why a companion folder yields no declared frame stream.

    DaVis reuses the .set extension for project and calibration nodes, and the
    streams of a recording are found only through the FrameReader entries of its
    StreamSet.xml -- each case has a different fix. (A folder of .im7 images is
    named earlier, by :func:`_raise_im7_layout`.)

    Raises:
        FileNotFoundError: Stream files without a StreamSet.xml, or no image data.
        ValueError: Stream files whose StreamSet.xml declares no FrameReader.
    """
    ims_prefixes = sorted({p.name[: -len("-1.ims")] for p in set_dir.glob("*-1.ims")})
    if ims_prefixes:
        shown = ", ".join(ims_prefixes[:5])
        common = (
            f"{set_dir} holds image streams ({shown}), but the stream-to-camera "
            f"mapping comes only from the FrameReader entries of StreamSet.xml and "
            f"is not guessed from file names. Re-export the recording from DaVis, "
            f"or copy the complete companion folder."
        )
        if manifest_present:
            raise ValueError(
                f"StreamSet.xml in {set_dir} declares no FrameReader "
                f"(Type '{_FRAME_READER_TYPE}'). {common}"
            )
        raise FileNotFoundError(f"StreamSet.xml is missing from {set_dir}. {common}")

    found = sorted(p.name for p in set_dir.iterdir())
    raise FileNotFoundError(
        f"{set_dir} holds no image data -- no *-1.ims streams and no .im7 files. "
        f"This is a DaVis project or calibration node, not an image recording. "
        f"Found: {', '.join(found[:5])}"
        f"{f' (+{len(found) - 5} more)' if len(found) > 5 else ''}. Point at a "
        f"recording .set whose companion folder holds the image streams."
    )


# ---------------------------------------------------------------------------
# Single-image reader (one entry from one frame stream)
# ---------------------------------------------------------------------------


def _read_single_image(
    frame_info: IMSFrameInfo,
    entry_idx: int,
) -> np.ndarray:
    """Read one image from a frame stream, returning uint16.

    Parameters
    ----------
    frame_info : IMSFrameInfo
        Frame stream metadata.
    entry_idx : int
        0-based entry index.

    Returns
    -------
    np.ndarray
        Image array (H, W) uint16.
    """
    if entry_idx < 0 or entry_idx >= frame_info.n_entries:
        raise IndexError(f"Entry {entry_idx} out of range [0, {frame_info.n_entries})")
    if frame_info.blocked is not None:
        raise ValueError(frame_info.blocked)

    # Refuse an unimplemented encoding before the read; its size is also what a
    # compressed entry must decompress to.
    stored_h, stored_w = frame_info.stored_shape
    expected = _payload_bytes(frame_info.decoder, stored_w, stored_h)

    # Refuse before the 37 MB read, not after it. Per stream: a container whose
    # other streams carry only implemented transformers stays readable.
    for tf in frame_info.transformers:
        if tf.id in _REFUSED_TRANSFORMERS:
            raise ValueError(
                f"Frame stream {frame_info.frame_idx} of "
                f"{frame_info.data_path.parent.name} declares the stream transformer "
                f"'{tf.id}' (DaVis label '{tf.label}'). It is a DaVis transformer "
                f"whose effect PIVTOOLs has no measurement of (lvpyio opens no test "
                f"recording carrying it), so it is not applied, and skipping it "
                f"would return different pixels from DaVis for the same recording. "
                f"PIVTOOLs applies: {', '.join([*_CORRECTIONS, *_GEOMETRIES])}. "
                f"Other streams of this recording are unaffected."
            )

    entry = frame_info.entries[entry_idx]
    with open(frame_info.data_path, "rb") as f:
        f.seek(entry.offset)
        raw = f.read(entry.size)

    if len(raw) != entry.size:
        raise IOError(
            f"Expected {entry.size} bytes at offset {entry.offset}, "
            f"got {len(raw)}. File may be truncated."
        )

    payload = _decompress_entry(raw, frame_info, entry_idx, expected)
    # Decoded in sensor coordinates: the index header's stored shape.
    img = _decode_pixels(payload, frame_info.decoder, stored_w, stored_h)
    return _apply_transformers(img, frame_info)


def _apply_transformers(img: np.ndarray, frame_info: IMSFrameInfo) -> np.ndarray:
    """Apply a stream's transformers to its decoded sensor frame.

    ``frame_info.transformers`` is already in LaVision's order (parse sorts it):
    corrections in sensor coordinates, each clipped, then at most one geometry,
    which may return a view (see :class:`_Geometry`).

    Raises:
        ValueError: If a correction image is not the stream's stored (sensor)
            shape. lvpyio checks only the pixel count and reads a transposed
            image in the sensor raster; DaVis stores corrections in sensor shape
            (transformer-order REPORT Q1b), so any other shape is refused.
    """
    for tf in frame_info.transformers:
        correction = _CORRECTIONS.get(tf.id)
        if correction is None:
            img = _GEOMETRIES[tf.id].apply(img)
            continue
        image = _load_correction(str(tf.correction_path))
        if image.shape != frame_info.stored_shape:
            raise ValueError(
                f"Correction image {tf.correction_path.name} ('{tf.id}') is "
                f"{image.shape} but frame stream {frame_info.frame_idx} stores "
                f"{frame_info.stored_shape} (sensor rows, columns). DaVis stores "
                f"corrections in sensor shape, so this image does not belong to "
                f"this recording."
            )
        img = correction.apply(img, image)
    return img


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------


def _apply_scale_inplace(arr: np.ndarray, fi: IMSFrameInfo) -> None:
    """Apply intensity scale (slope/offset) in-place on a float32 array."""
    if fi.scale_slope != 1.0:
        arr *= fi.scale_slope
    if fi.scale_offset != 0.0:
        arr += fi.scale_offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parsed-container cache
# ---------------------------------------------------------------------------

# Parsing a .set opens every stream's index file and its two XML sidecars: 21 opens
# and ~43 stat calls on a 10-stream recording. read_pair needs that same metadata
# three times for one time-resolved pair (once to derive frames-per-camera, once per
# frame), so an uncached read spends 65 opens and 128 stats to deliver two frames --
# against 3 opens for the equivalent .im7 pair. Caching reduces it to two stats.
#
# Same shape as the .cine reader's _metadata_cache (see readers/cine_reader.py):
# per-process, keyed by path, invalidated on mtime.
#
# Bounded, unlike the .cine one. A 20,000-entry 10-stream SetInfo holds 200,000
# IMSIndexEntry objects, and each Dask worker gets its own copy of this dict, so an
# unbounded cache would let a long time-resolved run pin hundreds of MB per worker.
# Four covers a multi-loop acquisition's working set. Same reasoning, and the same
# "raise it only with a measurement" caveat, as _load_correction's lru_cache above.
_SET_INFO_CACHE_SIZE = 4
_set_info_cache: "OrderedDict[str, Tuple[Tuple[float, float], SetInfo]]" = OrderedDict()

# The Flask server is threaded, so this dict needs a lock. Each OrderedDict operation
# is individually atomic under the GIL, but the sequences here are not: a hit's
# move_to_end can race another thread's eviction and raise KeyError out of what
# should be a read. The Dask path cannot hit it (piv_cluster pins
# threads_per_worker=1) and functools.lru_cache is internally locked; only this dict
# is exposed. _parse_set deliberately runs OUTSIDE the lock -- it is slow file I/O,
# and two threads racing to parse the same container is wasteful, never wrong.
_set_info_lock = threading.Lock()


def _set_info_cache_key(set_path: Path) -> Tuple[float, float]:
    """Modification times that must both still hold for a cached SetInfo to be valid.

    The .set file alone is not enough. It is a few hundred bytes of project XML and
    does not change when the streams in its companion folder do. The folder's mtime
    moves whenever a stream file is added or removed.

    KNOWN LIMITATION, accepted rather than hidden: rewriting a {prefix}-0.ims or
    StreamSet.xml in place changes neither mtime, so a cached SetInfo would
    survive it. Acquisition data is written once and read many times, so paying
    two stats per read is the right trade. Call :func:`clear_set_info_cache` if a
    container is ever edited in place during a session.

    Raises:
        OSError: If the .set or its companion directory is missing. Callers should
            fall through to _parse_set, which diagnoses that case by name.
    """
    return (set_path.stat().st_mtime, set_path.with_suffix("").stat().st_mtime)


def clear_set_info_cache() -> None:
    """Drop every cached SetInfo. Mirrors cine_reader.clear_metadata_cache()."""
    with _set_info_lock:
        _set_info_cache.clear()


def read_set_info(set_path: Union[str, Path]) -> SetInfo:
    """Parse .set metadata without reading any pixel data.

    Useful for getting entry count, dimensions, frame count. Results are cached per
    process and revalidated by mtime, so repeated calls for the same container cost
    two stat calls rather than a full re-parse. See :func:`_set_info_cache_key` for
    what invalidates an entry and the one case it cannot see.
    """
    set_path = Path(set_path)

    try:
        key = _set_info_cache_key(set_path)
    except OSError:
        # Missing .set or companion folder. Let _parse_set raise: it names which of
        # the non-recording .set shapes this is, and each has a different fix. A
        # bare stat error from here would lose that.
        return _parse_set(set_path)

    cache_id = str(set_path.absolute())
    with _set_info_lock:
        cached = _set_info_cache.get(cache_id)
        if cached is not None and cached[0] == key:
            _set_info_cache.move_to_end(cache_id)
            return cached[1]

    info = _parse_set(set_path)  # outside the lock: slow I/O, idempotent

    with _set_info_lock:
        _set_info_cache[cache_id] = (key, info)
        _set_info_cache.move_to_end(cache_id)
        while len(_set_info_cache) > _SET_INFO_CACHE_SIZE:
            _set_info_cache.popitem(last=False)
    return info


def get_set_entry_count(set_path: Union[str, Path]) -> int:
    """Get the number of entries (image pairs) in a .set file."""
    return read_set_info(set_path).n_entries


# A camera owns one stream (single frame) or two (a pre-paired A/B).
_MAX_STREAMS_PER_CAMERA = 2


def valid_stream_splits(
    shapes: Sequence[Tuple[int, int]], num_cameras: int
) -> List[Tuple[Tuple[int, ...], ...]]:
    """Every way to partition the streams into ``num_cameras`` cameras.

    A partition cuts ``range(len(shapes))``, in order, into exactly
    ``num_cameras`` consecutive groups of size 1 or 2. A size-2 group is an A/B
    pair and needs equal shapes.

    Args:
        shapes: (height, width) of each frame stream, in stream order.
        num_cameras: number of cameras to split into.

    Returns:
        Each partition as a tuple of stream-index tuples. Empty when
        ``num_cameras < 1`` or no partition exists.
    """
    n_streams = len(shapes)
    if num_cameras < 1:
        return []

    def extend(start: int, cameras_left: int) -> List[Tuple[Tuple[int, ...], ...]]:
        # Prune: the remaining streams must be able to fill the remaining cameras
        # (1..MAX each). Without this the walk is exponential on long, dead-end
        # stream lists. It never drops a valid split, so results are unchanged.
        remaining = n_streams - start
        if not cameras_left <= remaining <= _MAX_STREAMS_PER_CAMERA * cameras_left:
            return []
        if cameras_left == 0:
            return [()]
        found: List[Tuple[Tuple[int, ...], ...]] = []
        for size in range(1, _MAX_STREAMS_PER_CAMERA + 1):
            end = start + size
            if end > n_streams:
                break
            if size == 2 and shapes[start] != shapes[start + 1]:
                continue
            group = tuple(range(start, end))
            found.extend((group,) + rest for rest in extend(end, cameras_left - 1))
        return found

    return extend(0, num_cameras)


class SetConfigError(ValueError):
    """The camera count or selection does not fit the recording's streams.

    No unique stream split, or a camera outside the count. A configuration
    problem, not a read failure, so callers report its text as it stands.
    """


class SetPairingError(SetConfigError):
    """A selected camera resolves to one stream where an A/B pair is needed."""


def _require_positive_cameras(num_cameras: int, set_path: Union[str, Path]) -> None:
    """Raise a clear ValueError for a camera count below 1."""
    if num_cameras < 1:
        raise ValueError(
            f"Invalid camera count {num_cameras} for {Path(set_path).name}."
        )


def _size(width: int, height: int) -> str:
    """``216\u00d7256``: width first, as the diagnosis found line and stream table show it."""
    return f"{width}\u00d7{height}"


def _stream_table(info: SetInfo) -> str:
    """One-line stream listing for error messages: output size, decoder, transformers."""
    return "; ".join(
        f"{f.frame_idx}: "
        + (
            "size unknown"
            if f.has_unknown_transformer
            else _size(f.shape[1], f.shape[0])
        )
        + f" {f.decoder}"
        + "".join(f" {tf.id}" for tf in f.transformers)
        for f in info.frames
    )


def _streams_text(streams: Tuple[int, ...]) -> str:
    """``stream 4`` / ``streams 0–1``."""
    if len(streams) == 1:
        return f"stream {streams[0]}"
    return f"streams {streams[0]}–{streams[-1]}"


def _camera_list(cameras: Sequence[LabelledCamera]) -> str:
    """``Camera 1 (Imager MX 25M, ..., double frame (A/B)) on streams 0–1; ...``."""
    return "; ".join(f"{c.describe()} on {_streams_text(c.frames)}" for c in cameras)


def set_camera_streams(
    set_path: Union[str, Path],
    num_cameras: int,
    set_info: Optional[SetInfo] = None,
) -> List[Tuple[int, ...]]:
    """The stream->camera split of a .set container.

    With camera labels (:attr:`SetInfo.cameras`) the split is the labelled
    cameras' streams, and the camera count must be theirs. Without labels it is
    the unique split by stream shape. Cached on the SetInfo, so it is computed
    once per container and follows the SetInfo cache's invalidation. Errors are
    not cached.

    Args:
        set_path: path to the .set file.
        num_cameras: number of cameras the recording is split into.
        set_info: pre-read container metadata, to skip the cache lookup.

    Returns:
        One tuple of 0-based stream indices per camera, in camera order.

    Raises:
        SetConfigError: With labels, if ``num_cameras`` is not their count (the
            message lists the cameras by label, model and ID). Without, if no
            shape split fits, or more than one does (ambiguous).
    """
    _require_positive_cameras(num_cameras, set_path)
    info = set_info if set_info is not None else read_set_info(set_path)
    cached = info.stream_splits.get(num_cameras)
    if cached is not None:
        return list(cached)

    if info.cameras is not None:
        n = len(info.cameras)
        if num_cameras != n:
            raise SetConfigError(
                f"{Path(set_path).name} holds {n} {'camera' if n == 1 else 'cameras'} "
                f"by its own camera labels, not {num_cameras}: "
                f"{_camera_list(info.cameras)}. Set the camera count to {n}."
            )
        split = tuple(c.frames for c in info.cameras)
        info.stream_splits[num_cameras] = split
        return list(split)

    # Output shapes: an A/B pair must match as read, after the transformers, and
    # a rotated stream must not pair with an unrotated one of equal stored shape.
    shapes = [f.shape for f in info.frames]
    splits = valid_stream_splits(shapes, num_cameras)
    if len(splits) == 1:
        info.stream_splits[num_cameras] = splits[0]
        return list(splits[0])

    name = Path(set_path).name
    fitting = [
        k for k in range(1, len(shapes) + 1) if len(valid_stream_splits(shapes, k)) == 1
    ]
    fits = ", ".join(str(k) for k in fitting) if fitting else "none"
    table = _stream_table(info)
    cameras = f"{num_cameras} camera" if num_cameras == 1 else f"{num_cameras} cameras"
    if not splits:
        raise SetConfigError(
            f"{name} holds {len(shapes)} frame streams, which cannot split into "
            f"{cameras} of 1 or 2 consecutive streams (an A/B pair must "
            f"match in shape). Camera counts that fit: {fits}. Streams: {table}"
        )
    raise SetConfigError(
        f"{name}: {len(splits)} different splits fit {cameras}, so the "
        f"stream->camera mapping is ambiguous. Camera counts that fit: {fits}. "
        f"Streams: {table}"
    )


def camera_streams(
    set_path: Union[str, Path],
    camera: int,
    num_cameras: int,
    pair: bool,
    set_info: Optional[SetInfo] = None,
) -> Tuple[int, ...]:
    """The 0-based stream indices of one camera.

    Args:
        set_path: path to the .set file.
        camera: 1-based camera number.
        num_cameras: number of cameras the recording is split into.
        pair: True when the caller needs an A/B pair (two streams).
        set_info: pre-read container metadata.

    Raises:
        SetConfigError: If ``camera`` is outside 1..num_cameras or the split is
            not unique.
        SetPairingError: If ``pair`` is set and the camera has a single stream.
    """
    _require_positive_cameras(num_cameras, set_path)
    name = Path(set_path).name
    if not 1 <= camera <= num_cameras:
        raise SetConfigError(f"Camera {camera} is outside 1..{num_cameras} for {name}.")
    info = set_info if set_info is not None else read_set_info(set_path)
    streams = set_camera_streams(set_path, num_cameras, info)[camera - 1]
    if pair and len(streams) == 1 and info.cameras is not None:
        raise SetPairingError(
            f"Camera {camera} of {name}, {info.cameras[camera - 1].describe()} on "
            f"{_streams_text(streams)}, is a single-frame camera, so it has no A/B "
            f"pairs. Deselect camera {camera} in the PIV camera selection, or "
            f"process it time-resolved."
        )
    if pair and len(streams) == 1:
        frame = info.frames[streams[0]]
        raise SetPairingError(
            f"Camera {camera} has one stream ({frame.frame_idx}: "
            f"{_size(frame.shape[1], frame.shape[0])}) in {name}, but A/B pairing "
            f"needs two. "
            f"Deselect camera {camera} in the PIV camera selection."
        )
    return streams


def read_set_pair(
    set_path: Union[str, Path],
    streams: Tuple[int, int],
    im_no: int,
    set_info: Optional[SetInfo] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read a pre-paired A/B frame pair from a .set container.

    Reads the two given frame streams from entry ``im_no``. Which streams belong
    to a camera is decided by :func:`camera_streams`, never here. Time-resolved
    pairing does NOT live here either: it is two single-frame reads at different
    entries, built from read_set_frame (see load_images.read_pair).

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file.
    streams : tuple of int
        0-based (A, B) frame-stream indices, as returned by
        :func:`camera_streams` with ``pair=True``.
    im_no : int
        Image/entry number (1-based).
    set_info : SetInfo, optional
        Pre-parsed container metadata. If provided, skips re-parsing the
        index/XML files. Use read_set_info() once, then pass it here for
        every pair in a batch.
    out : np.ndarray, optional
        Destination (2, H, W) float32 C-contiguous buffer, typically one slot of
        a batch. Validated by :func:`readers.out_buffer.check_out`, written in
        place and returned; undefined after an exception.

    Returns
    -------
    np.ndarray
        Array of shape (2, H, W), dtype float32, with intensity scale applied.
        ``out`` itself when given.

    Raises
    ------
    ValueError
        If ``streams`` does not hold exactly two indices, an index is outside the
        container, or the two streams differ in shape.
    """
    # read_set_info, not _parse_set: callers that do not thread set_info through
    # (the pre-paired PIV path and the calibration loader) would otherwise re-parse
    # the whole container on every single frame.
    info = set_info if set_info is not None else read_set_info(set_path)

    if len(streams) != 2:
        raise ValueError(
            f"read_set_pair needs exactly two frame streams, got {tuple(streams)}"
        )
    frame_idx_a, frame_idx_b = streams
    n_streams = len(info.frames)
    for idx in streams:
        if not 0 <= idx < n_streams:
            raise ValueError(
                f"Frame stream {idx} is outside 0..{n_streams - 1} in "
                f"{Path(set_path).name}"
            )

    fi_a = info.frames[frame_idx_a]
    fi_b = info.frames[frame_idx_b]
    entry_idx = im_no - 1

    # Streams legitimately differ in shape across cameras (alk235's five span
    # 3472-3536 rows). The resolver only pairs equal shapes, so this guards a
    # caller handing in a hand-built tuple rather than a numpy broadcast error.
    if fi_a.shape != fi_b.shape:
        raise ValueError(
            f"Frame streams {frame_idx_a} ({fi_a.shape[0]}x{fi_a.shape[1]}) and "
            f"{frame_idx_b} ({fi_b.shape[0]}x{fi_b.shape[1]}) differ in shape, so "
            f"they are not one camera's A/B pair."
        )

    if out is None:
        result = np.empty((2, *fi_a.shape), dtype=np.float32)
    else:
        result = check_out(
            out,
            (2, *fi_a.shape),
            f"read_set_pair streams {frame_idx_a}+{frame_idx_b} of {Path(set_path).name}",
        )

    img_a = _read_single_image(fi_a, entry_idx)
    result[0] = img_a
    del img_a

    img_b = _read_single_image(fi_b, entry_idx)
    result[1] = img_b
    del img_b

    # Apply per-frame-stream intensity scale (A and B may differ)
    _apply_scale_inplace(result[0], fi_a)
    _apply_scale_inplace(result[1], fi_b)

    return result


def read_set_frame(
    set_path: Union[str, Path],
    entry_no: int,
    frame_idx: int,
    set_info: Optional[SetInfo] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read a single frame from a .set container.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file.
    entry_no : int
        Entry number (1-based).
    frame_idx : int
        Frame index within the entry (0-based, maps to Frame{N} files).
    set_info : SetInfo, optional
        Pre-parsed metadata (avoids re-parsing).
    out : np.ndarray, optional
        Destination (H, W) float32 array to decode into, typically one slice of a
        caller's ``(2, H, W)`` pair buffer. Without it this allocates a fresh frame
        via ``astype``, and a caller stacking two frames then copies both again --
        two full-frame passes where one suffices. Writing into ``out`` performs the
        uint16-to-float32 widening straight into its final location, which is the
        same thing :func:`read_set_pair` does with ``result[0] = img_a``.

    Returns
    -------
    np.ndarray
        Image (H, W) float32 with intensity scale applied. ``out`` itself when given.

    Raises
    ------
    ValueError
        If ``frame_idx`` is out of range, or ``out`` does not match the frame
        stream's shape or is not float32.
    """
    # read_set_info, not _parse_set: callers that do not thread set_info through
    # (the pre-paired PIV path and the calibration loader) would otherwise re-parse
    # the whole container on every single frame.
    info = set_info if set_info is not None else read_set_info(set_path)

    if frame_idx < 0 or frame_idx >= len(info.frames):
        raise ValueError(
            f"Frame index {frame_idx} out of range [0, {len(info.frames)})"
        )

    fi = info.frames[frame_idx]

    if out is None:
        result = None
    else:
        if out.shape != fi.shape:
            raise ValueError(
                f"out has shape {out.shape}, but frame stream {fi.frame_idx} of "
                f"{Path(set_path).name} reads as {fi.shape}."
            )
        if out.dtype != np.float32:
            raise ValueError(
                f"out has dtype {out.dtype}; read_set_frame produces float32."
            )
        result = out

    img = _read_single_image(fi, entry_no - 1)

    if result is None:
        result = img.astype(np.float32)
    else:
        # Widens uint16 -> float32 directly into the caller's buffer.
        np.copyto(result, img)
    del img

    _apply_scale_inplace(result, fi)
    return result
