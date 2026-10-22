"""Synthetic LaVision DaVis ``.set`` / ``.im7`` writers shared by the reader tests.

Moved out of ``test_set_decoders.py`` and ``test_frames_per_camera_and_shape.py``
so the resolver, reader and config tests build containers one way.
"""

import struct

import numpy as np
import yaml

from pivtools_core.config import Config

# Index layout, mirroring _parse_ims_index: 256-byte header (width at offset 12,
# height at 16) then 768 bytes of padding, then 20-byte entries.
TABLE_OFFSET = 256 + 768
ENTRY_STRUCT = "<iqq"  # flag int32, offset int64, size int64


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
        prefixes: optional per-stream file prefix. Default ``Frame{N}``, which
            the reader finds by glob; anything else needs a StreamSet.xml
            manifest (see ``write_manifest``) to be found at all.
        shapes: optional per-stream (height, width), overriding width/height.
            Real recordings carry a different shape per camera.

    Returns:
        Path: the .set file (its companion directory sits alongside).
    """
    set_file = tmp_path / f"{name}.set"
    set_file.write_bytes(b"stub")
    set_dir = tmp_path / name
    set_dir.mkdir()

    for stream_idx, entries in enumerate(streams):
        prefix = f"Frame{stream_idx}" if prefixes is None else prefixes[stream_idx]
        h, w = (height, width) if shapes is None else shapes[stream_idx]
        dec = decoder if isinstance(decoder, str) else decoder[stream_idx]

        index = bytearray(TABLE_OFFSET)
        struct.pack_into("<i", index, 12, w)
        struct.pack_into("<i", index, 16, h)

        offset = 0
        for entry_idx, payload in enumerate(entries):
            size = len(payload)
            if declared_sizes is not None:
                size = declared_sizes[stream_idx][entry_idx]
            index += struct.pack(ENTRY_STRUCT, 0, offset, size)
            offset += len(payload)

        (set_dir / f"{prefix}-0.ims").write_bytes(bytes(index))
        (set_dir / f"{prefix}-1.ims").write_bytes(b"".join(entries))

        if write_decoder_xml:
            (set_dir / f"{prefix}-decoder.xml").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f"<FrameDecoder>\n<id>{dec}</id>\n</FrameDecoder>\n"
            )

    return set_file


def write_manifest(set_dir, frame_readers, scale_readers=(), transformers=()):
    """Write a StreamSet.xml in DaVis's real shape (``<StreamCfg>`` root).

    Args:
        set_dir: companion directory.
        frame_readers: (prefix, stream_index) per image stream.
        scale_readers: (prefix, stream_index) per .scales file.
        transformers: (id, prefix, stream_index) per correction.
    """

    def purpose(frame):
        return (
            f'<ContentPurpose IsAssociatedToFrames="true" '
            f'IsAssociatedToEntireImage="false" IsAssociatedToAllImages="false" '
            f'StartFrame="{frame}" EndFrame="{frame}"/>'
        )

    body = "".join(
        f'<ReaderInfo Name="Images_16bit_8Bit" Type="Core.Set.Recording.FrameReader" '
        f'FilePrefix="{prefix}" FormatDescription="LaVision image stream format">'
        f"{purpose(frame)}</ReaderInfo>"
        for prefix, frame in frame_readers
    )
    body += "".join(
        f'<ReaderInfo Name="ScaleReaderV1" Type="Core.Set.Recording.ScaleReader" '
        f'FilePrefix="{prefix}" FormatDescription="Frame scale files">'
        f"{purpose(frame)}</ReaderInfo>"
        for prefix, frame in scale_readers
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


def write_stream_set(set_dir, entries):
    """Write a StreamSet.xml declaring one Transformer per (id, prefix, frame)."""
    body = "".join(
        f'<Transformer ID="{tid}" Label="L" FilePrefix="{prefix}" '
        f'MinDaVisVersion="10.2.0">'
        f'<ContentPurpose IsAssociatedToFrames="true" StartFrame="{frame}" '
        f'EndFrame="{frame}"/></Transformer>'
        for tid, prefix, frame in entries
    )
    (set_dir / "StreamSet.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><StreamSet>{body}</StreamSet>'
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
    # Trailing int32 0 terminates the extended-header record list (IEH_END).
    path.write_bytes(bytes(header) + arr.astype("<u2").tobytes() + b"\x00\x00\x00\x00")


def write_dark_im7_float(path, arr):
    """Write a float32 .im7, for values that do not fit uint16."""
    import struct as _struct

    h, w = arr.shape
    header = bytearray(256)
    _struct.pack_into("<hhhh", header, 0, 0, 0, -3, 0)  # buffer_format -3 = FLOAT
    _struct.pack_into("<iiii", header, 8, w, h, 1, 1)
    path.write_bytes(bytes(header) + arr.astype("<f4").tobytes() + b"\x00\x00\x00\x00")


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
        index = bytearray(TABLE_OFFSET)
        struct.pack_into("<i", index, 12, w)
        struct.pack_into("<i", index, 16, h)
        offset = 0
        for payload in payloads:
            index += struct.pack(ENTRY_STRUCT, 0, offset, len(payload))
            offset += len(payload)
        (set_dir / f"Frame{idx}-0.ims").write_bytes(bytes(index))
        (set_dir / f"Frame{idx}-1.ims").write_bytes(b"".join(payloads))
        (set_dir / f"Frame{idx}-decoder.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<FrameDecoder>\n<id>raw-16-bit</id>\n</FrameDecoder>\n"
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
