"""Synthetic LaVision DaVis ``.set`` / ``.im7`` writers shared by the reader tests.

Moved out of ``test_set_decoders.py`` and ``test_frames_per_camera_and_shape.py``
so the resolver, reader and config tests build containers one way.
"""

import struct

import numpy as np
import pytest
import yaml

from pivtools_core.config import Config
from pivtools_core.image_handling.readers.set_reader import clear_set_info_cache

# Index layout as DaVis writes it (container-unknowns REPORT §1, measured on 20
# real stream files): a 1024-byte header, then 20-byte entries.
TABLE_OFFSET = 1024
ENTRY_STRUCT = "<Iqq"  # file number uint32, offset int64, size int64

# The header words every real stream file carries, by int32 word index: version 1,
# image count 0 (derive), 1 frame per image, 16 bits per decoded pixel, planes 0,
# sample type 0 (uint16). Words 3 and 4 (width, height) are per stream.
MEASURED_HEADER_WORDS = {0: 1, 1: 0, 2: 1, 5: 16, 14: 0, 15: 0}
CODEC_OFFSET = 24  # bytes 24..55, NUL-terminated codec name; empty = uncompressed
CODEC_FIELD_BYTES = 32


def ims_index(width, height, sizes, words=None, codec=b"", file_numbers=None):
    """Bytes of a {prefix}-0.ims index in DaVis's measured layout.

    Args:
        width: stored width, header word 3.
        height: stored height, header word 4.
        sizes: per-entry stored size; entries are laid end to end from offset 0.
        words: optional {word index: int32} overriding MEASURED_HEADER_WORDS, to
            build a header DaVis was not seen to write.
        codec: raw bytes written at CODEC_OFFSET (at most CODEC_FIELD_BYTES).
        file_numbers: optional per-entry data-file number; default 1 for all.

    Returns:
        bytes: the whole index file.
    """
    header = bytearray(TABLE_OFFSET)
    for word, value in {**MEASURED_HEADER_WORDS, 3: width, 4: height}.items():
        struct.pack_into("<i", header, 4 * word, value)
    for word, value in (words or {}).items():
        struct.pack_into("<i", header, 4 * word, value)
    assert len(codec) <= CODEC_FIELD_BYTES
    header[CODEC_OFFSET : CODEC_OFFSET + len(codec)] = codec

    offset = 0
    for k, size in enumerate(sizes):
        file_no = 1 if file_numbers is None else file_numbers[k]
        header += struct.pack(ENTRY_STRUCT, file_no, offset, size)
        offset += size
    return bytes(header)


# Stream 0 differs, streams 1-4 are equal (2160x2560 + 4 x 2048x2048, scaled down).
# (height, width) per stream.
PLIF = [(6, 8)] + [(4, 6)] * 4


@pytest.fixture()
def clean_set_cache():
    """Empty the process-wide SetInfo cache before and after the test."""
    clear_set_info_cache()
    yield
    clear_set_info_cache()


def build_set(
    tmp_path,
    decoder,
    streams,
    width,
    height,
    name="synthetic",
    write_decoder_xml=True,
    declared_sizes=None,
    prefixes=None,
    shapes=None,
    manifest=True,
):
    """Write a minimal .set container and return the path to the .set file.

    Args:
        tmp_path: pytest tmp_path.
        decoder: encoding id written into {prefix}-decoder.xml (a str, or one
            per stream).
        streams: list (one per frame stream) of lists of entry payload bytes.
        width: frame width written into the index header.
        height: frame height written into the index header.
        name: container stem.
        write_decoder_xml: when False, omit the decoder XML entirely.
        declared_sizes: optional per-stream list of per-entry sizes to declare in
            the index, overriding the true payload length. Used to build a
            container whose declared geometry disagrees with its payload.
        prefixes: optional per-stream file prefix. Default ``Frame{N}``.
        shapes: optional per-stream (height, width), overriding width/height.
            Real recordings carry a different shape per camera.
        manifest: when True, write a StreamSet.xml declaring stream k under its
            prefix at StartFrame k. The reader finds streams only through it.

    Returns:
        Path: the .set file (its companion directory sits alongside).
    """
    set_file = tmp_path / f"{name}.set"
    set_file.write_bytes(b"stub")
    set_dir = tmp_path / name
    set_dir.mkdir()

    all_prefixes = [
        f"Frame{k}" if prefixes is None else prefixes[k] for k in range(len(streams))
    ]
    for stream_idx, entries in enumerate(streams):
        prefix = all_prefixes[stream_idx]
        h, w = (height, width) if shapes is None else shapes[stream_idx]
        dec = decoder if isinstance(decoder, str) else decoder[stream_idx]

        sizes = (
            [len(p) for p in entries]
            if declared_sizes is None
            else declared_sizes[stream_idx]
        )
        (set_dir / f"{prefix}-0.ims").write_bytes(ims_index(w, h, sizes))
        (set_dir / f"{prefix}-1.ims").write_bytes(b"".join(entries))

        if write_decoder_xml:
            (set_dir / f"{prefix}-decoder.xml").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f"<FrameDecoder>\n<id>{dec}</id>\n</FrameDecoder>\n"
            )

    if manifest:
        write_manifest(
            set_dir, frame_readers=[(p, k) for k, p in enumerate(all_prefixes)]
        )
    return set_file


def write_manifest(
    set_dir,
    frame_readers,
    scale_readers=(),
    transformers=(),
    reader_name="Images_16bit_8Bit",
    associated_to_frames="true",
):
    """Write a StreamSet.xml in DaVis's real shape (``<StreamCfg>`` root).

    Args:
        set_dir: companion directory.
        frame_readers: (prefix, stream_index) per image stream, or
            (prefix, start_frame, end_frame) for a stream file holding several
            frames of each image (DaVis 11.4 writes A and B into one file).
        scale_readers: (prefix, stream_index) per .scales file, or
            (prefix, start_frame, end_frame) to build a span DaVis was not seen
            to write.
        transformers: (id, prefix, stream_index) per correction.
        reader_name: the FrameReader ``Name``. Every real recording carries
            ``Images_16bit_8Bit``; anything else builds a foreign reader.
        associated_to_frames: the FrameReader's ``IsAssociatedToFrames``.
    """

    def purpose(frame, associated="true", end=None):
        return (
            f'<ContentPurpose IsAssociatedToFrames="{associated}" '
            f'IsAssociatedToEntireImage="false" IsAssociatedToAllImages="false" '
            f'StartFrame="{frame}" EndFrame="{frame if end is None else end}"/>'
        )

    body = "".join(
        f'<ReaderInfo Name="{reader_name}" Type="Core.Set.Recording.FrameReader" '
        f'FilePrefix="{prefix}" FormatDescription="LaVision image stream format">'
        f"{purpose(frames[0], associated_to_frames, end=frames[-1])}</ReaderInfo>"
        for prefix, *frames in frame_readers
    )
    body += "".join(
        f'<ReaderInfo Name="ScaleReaderV1" Type="Core.Set.Recording.ScaleReader" '
        f'FilePrefix="{prefix}" FormatDescription="Frame scale files">'
        f"{purpose(frames[0], end=frames[-1])}</ReaderInfo>"
        for prefix, *frames in scale_readers
    )
    body += "".join(
        f'<Transformer ID="{tid}" Label="L" FilePrefix="{prefix}" '
        f'MinDaVisVersion="10.2.0">{purpose(frame)}</Transformer>'
        for tid, prefix, frame in transformers
    )
    (set_dir / "StreamSet.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><StreamCfg><Version>2</Version>'
        f"{body}</StreamCfg>"
    )


def write_scales(path, factor, offset=0.0):
    """Write a {prefix}.scales file with the given intensity scale."""
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><ScalesList><Scales>'
        '<ScaleX Factor="1" Offset="0" Unit="pixel" Description=""/>'
        f'<ScaleI Factor="{factor}" Offset="{offset}" Unit="counts" Description=""/>'
        "</Scales></ScalesList>"
    )


# Extended-header records after the pixels, as DaVis writes them: an IEH_SCALE_X
# record first (type 1, "slope offset\0unit\0", copied from a real DaVis dark
# image), then int32 0 (IEH_END). The reader rejects a bare IEH_END, which real
# files never have, because it is indistinguishable from zero pixel bytes.
_IM7_SCALE_X_PAYLOAD = b"1 0\x00pixel\x00"
_IM7_TRAILER = (
    struct.pack("<ii", 1, len(_IM7_SCALE_X_PAYLOAD))
    + _IM7_SCALE_X_PAYLOAD
    + b"\x00\x00\x00\x00"
)


def write_dark_im7(path, arr):
    """Write a minimal uncompressed uint16 .im7 the in-tree reader can read."""
    import struct as _struct

    h, w = arr.shape
    header = bytearray(256)
    # Image_Header_7, "<hhhh iiii hhh": version, pack_type, buffer_format,
    # is_sparse, size_x, size_y, size_z, size_f, scalar_n, vector_grid, extra_flags.
    _struct.pack_into("<hhhh", header, 0, 0, 0, -4, 0)  # buffer_format -4 = WORD
    _struct.pack_into("<iiii", header, 8, w, h, 1, 1)
    path.write_bytes(bytes(header) + arr.astype("<u2").tobytes() + _IM7_TRAILER)


def write_dark_im7_float(path, arr):
    """Write a float32 .im7, for values that do not fit uint16."""
    import struct as _struct

    h, w = arr.shape
    header = bytearray(256)
    _struct.pack_into("<hhhh", header, 0, 0, 0, -3, 0)  # buffer_format -3 = FLOAT
    _struct.pack_into("<iiii", header, 8, w, h, 1, 1)
    path.write_bytes(bytes(header) + arr.astype("<f4").tobytes() + _IM7_TRAILER)


def build_raw16_set(
    tmp_path, shapes, name="shaped", n_entries=1, fill_with_index=False
):
    """A .set whose stream k holds ``n_entries`` entries of shapes[k] = (H, W).

    Mirrors :func:`build_set` but takes a per-stream shape, which is the point of
    the shape tests: real recordings carry different shapes per stream (alk235's
    five cameras span 3472-3536 rows).

    Args:
        tmp_path: pytest tmp_path.
        shapes: per-stream (height, width).
        name: container stem.
        n_entries: entries written per stream.
        fill_with_index: when True, entry ``e`` (0-based) of stream ``k`` is filled
            with ``100*k + e`` (uint16) so a read identifies its source; otherwise
            zeros.

    Returns:
        Path: the .set file.
    """
    set_file = tmp_path / f"{name}.set"
    set_file.write_bytes(b"stub")
    set_dir = tmp_path / name
    set_dir.mkdir()

    for idx, (h, w) in enumerate(shapes):
        payloads = [
            np.full(
                (h, w), 100 * idx + e if fill_with_index else 0, dtype="<u2"
            ).tobytes()
            for e in range(n_entries)
        ]
        index = ims_index(w, h, [len(p) for p in payloads])
        (set_dir / f"Frame{idx}-0.ims").write_bytes(index)
        (set_dir / f"Frame{idx}-1.ims").write_bytes(b"".join(payloads))
        (set_dir / f"Frame{idx}-decoder.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<FrameDecoder>\n<id>raw-16-bit</id>\n</FrameDecoder>\n"
        )
    write_manifest(
        set_dir, frame_readers=[(f"Frame{k}", k) for k in range(len(shapes))]
    )
    return set_file


def make_set_config(tmp_path, set_file, camera_count, camera_numbers, frame_stride=1):
    """Write a minimal lavision_set config.yaml and load it as a Config."""
    cfg = {
        "paths": {
            "source_paths": [str(set_file)],
            "camera_count": camera_count,
            "camera_numbers": list(camera_numbers),
        },
        "images": {
            "image_format": [set_file.name],
            "image_type": "lavision_set",
            "frame_stride": frame_stride,
            "start_index": 1,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return Config(path=str(config_path))
