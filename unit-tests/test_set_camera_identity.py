"""Cameras of a DaVis ``.set`` recording from its own camera labels.

A recording's ``MetaData{N}.attr`` files carry ``Acq.Camera.Label``,
``Acq.Camera.ID`` and ``CameraName`` per frame ("1A", "1B" for a double-frame
camera, "3" for a single-frame one). StreamSet.xml binds each file to a frame
stream or to the whole image, never ``N``. With labels the cameras come from
them; without, from the stream shapes (``test_set_stream_split.py``).

Evidence: attr-decode follow-up REPORT (2026-11-17) §2-§6, decisions of 17 Nov
(ticket ledger D7). Real fixtures: ``fixtures/davis_attr/`` (NTNU PIV plus PLIF
sample, DaVis 10.2.1; ``Recording_Date=261030_Time=145749``, DaVis 11.4.0.91,
camera keys on images 1..501 of 900, flagged ``SetIsSuspicious``), each with
lvpyio 1.3.1's attributes as the ground truth.
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

import json
from pathlib import Path

import numpy as np
import pytest
from fixtures.diagnosis import _inp, make_ctx, piv_route_response, write_config
from fixtures.lavision import (  # noqa: F401
    build_raw16_set,
    camera_keys,
    clean_set_cache,
    make_set_config,
    write_attr,
    write_manifest,
)

from pivtools_core.image_handling.load_images import read_pair
from pivtools_core.image_handling.readers.davis_attr import (
    AttrFormatError,
    CameraIdentityError,
)
from pivtools_core.image_handling.readers.set_reader import (
    SetConfigError,
    SetPairingError,
    camera_streams,
    read_set_attributes,
    read_set_frame,
    read_set_info,
    set_camera_streams,
    valid_stream_splits,
)
from pivtools_core.image_handling.source_diagnosis import (
    found_for,
    recording_warnings,
    rule_set_streams,
    scan_sentence,
    scan_source,
    set_stream_infos,
    streams_sentence,
)
from pivtools_core.validation import validate_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "davis_attr"
NTNU = FIXTURES / "REF_background_check_1006.set"
REC_145749 = FIXTURES / "Recording_Date=261030_Time=145749.set"
LVPYIO = json.loads((FIXTURES / "lvpyio_attributes.json").read_text())

# NTNU's four cameras, from attr-decode REPORT §2 C2 (ground truth independent
# of lvpyio: the four Settings_Acquisition_Camera files and Morgan's rig notes).
NTNU_CAMERAS = [
    (1, "Camera 1", "Camera.ImagerMX25M: 290187", "Imager MX 25M", (0, 1)),
    (2, "Camera 2", "Camera.ImagerMX25M: 290186", "Imager MX 25M", (2, 3)),
    (3, "Camera 3", "Camera.ImagerSCMOS5MCLHS: 61008639", "Imager sCMOS CLHS", (4,)),
    (4, "Camera 4", "Camera.ImagerSCMOS5MCLHS: 61009193", "Imager sCMOS CLHS", (5,)),
]

# Keys on which lvpyio and the files legitimately differ (attr-decode REPORT
# §2 B1-B3, ticket ledger V0). Made up by lvpyio on every buffer; LoadSetType
# only by lvpyio 1.5.0. ExpPathValue_* are parsed from a "Date=..._Time=..."
# folder name.
LVPYIO_INVENTED_BUFFER_KEYS = {
    "LoadSet",
    "LoadSetIndex",
    "LoadSetType",
    "_Date",
    "_Time",
    "ExpPathValue_Date",
    "ExpPathValue_Time",
}
LVPYIO_DROPPED_FRAME_KEYS = {"AcqTime"}  # B2: the same time is in Acq.Time
# B3: lvpyio's attribute transform, observed (not modelled) on DaVis 11.4 files
# that say Acq.AttributesTransformed = '0'.
LVPYIO_REWRITTEN_FRAME_KEYS = {
    "Acq.AttributesTransformed",
    "FrameRotation",
    "RealFrameSize",
}
LVPYIO_ADDED_FRAME_KEYS = {"FrameProcessing"}
CAMERA_KEYS = ("Acq.Camera.Label", "Acq.Camera.ID", "CameraName")

# Shapes of a PIV plus PLIF rig scaled down: four equal MX streams, two sCMOS.
PIV_PLIF_SHAPES = [(6, 6)] * 4 + [(4, 5)] * 2
PIV_PLIF_KEYS = [
    ("Camera 1", "Camera.X: 11", "1A: X"),
    ("Camera 1", "Camera.X: 11", "1B: X"),
    ("Camera 2", "Camera.X: 12", "2A: X"),
    ("Camera 2", "Camera.X: 12", "2B: X"),
    ("Camera 3", "Camera.S: 31", "3: S"),
    ("Camera 4", "Camera.S: 32", "4: S"),
]


def _labelled_set(
    tmp_path,
    keys,
    shapes=None,
    n_images=1,
    name="lab",
    labelled=None,
    transformers=(),
):
    """A raw-16-bit recording with one ``MetaData{k}.attr`` per frame stream.

    Args:
        keys: per frame stream, the (label, camera_id, CameraName) its images
            carry, or None for a stream whose images carry no camera keys.
        labelled: 0-based images that carry the keys; default every image.
            The others carry only a timing key, as DaVis 11.4 writes them.
        transformers: (id, prefix, stream) per transformer.
    """
    shapes = shapes or [(4, 6)] * len(keys)
    s = build_raw16_set(
        tmp_path, shapes, name=name, n_entries=n_images, fill_with_index=True
    )
    folder = tmp_path / name
    for k, triple in enumerate(keys):
        images = []
        for e in range(n_images):
            values = {"Acq.Time": np.array([[float(e)]], "<f8")}
            if triple is not None and (labelled is None or e in labelled):
                values.update(camera_keys(*triple))
            images.append(values)
        write_attr(folder / f"MetaData{k}.attr", images)
    write_manifest(
        folder,
        [(f"Frame{k}", k) for k in range(len(shapes))],
        transformers=transformers,
        attribute_readers=[(f"MetaData{k}", k) for k in range(len(keys))],
    )
    return s


# ---------------------------------------------------------------------------
# Real recordings
# ---------------------------------------------------------------------------


def test_ntnu_cameras_come_from_its_labels(clean_set_cache):
    """§6 item 5, and attr-decode REPORT C2."""
    info = read_set_info(NTNU)
    assert [
        (c.number, c.label, c.camera_id, c.model, c.frames) for c in info.cameras
    ] == NTNU_CAMERAS
    assert [c.double_frame for c in info.cameras] == [True, True, False, False]
    assert [c.labelled_images for c in info.cameras] == [1, 1, 1, 1]
    assert [f.frame_role for f in info.frames] == ["A", "B", "A", "B", "", ""]
    assert [f.camera_label for f in info.frames] == [
        "Camera 1",
        "Camera 1",
        "Camera 2",
        "Camera 2",
        "Camera 3",
        "Camera 4",
    ]
    assert info.frames[4].camera_id == "Camera.ImagerSCMOS5MCLHS: 61008639"
    assert info.attribute_files == tuple(f"MetaData{n}.attr" for n in range(7))
    assert info.suspicious_flag is None


def test_ntnu_split_is_the_labels_where_the_shape_rule_is_ambiguous(clean_set_cache):
    """The control: the same shapes give several splits at 4 cameras, so the
    shape rule cannot produce NTNU's layout (attr-decode REPORT C2)."""
    info = read_set_info(NTNU)
    assert len(valid_stream_splits([f.shape for f in info.frames], 4)) > 1
    assert set_camera_streams(NTNU, 4) == [(0, 1), (2, 3), (4,), (5,)]


def test_ntnu_wrong_camera_count_names_the_four_cameras(clean_set_cache):
    with pytest.raises(SetConfigError) as exc:
        set_camera_streams(NTNU, 3)
    message = str(exc.value)
    assert "4 cameras" in message and "Set the camera count to 4" in message
    for _n, label, camera_id, model, _frames in NTNU_CAMERAS:
        assert label in message and camera_id in message and model in message


def test_ntnu_single_frame_camera_cannot_pair_but_reads_alone(clean_set_cache):
    """A single-frame camera is refused for A/B pairing by name, and allowed
    time-resolved (17 Nov)."""
    with pytest.raises(SetPairingError) as exc:
        camera_streams(NTNU, 3, 4, pair=True)
    message = str(exc.value)
    assert "Camera 3" in message and "61008639" in message
    assert "single-frame camera" in message
    assert camera_streams(NTNU, 3, 4, pair=False) == (4,)
    assert camera_streams(NTNU, 2, 4, pair=True) == (2, 3)


def test_145749_partial_labels_group_from_the_labelled_image(clean_set_cache):
    """§6 item 9 on real bytes: image 501 carries the camera keys, image 502
    does not (DaVis 11.4 stopped writing them), and A/B sit in one stream file."""
    info = read_set_info(REC_145749)
    [cam] = info.cameras
    assert (cam.label, cam.camera_id, cam.model, cam.frames) == (
        "Camera 1",
        "Camera.ImagerCX25: 119775",
        "Imager CX-25",
        (0, 1),
    )
    assert cam.double_frame and cam.labelled_images == 1 and info.n_entries == 2
    assert [f.frame_role for f in info.frames] == ["A", "B"]
    assert set_camera_streams(REC_145749, 1) == [(0, 1)]


def test_145749_suspicious_flag_is_read_from_the_set_file(clean_set_cache):
    assert read_set_info(REC_145749).suspicious_flag == "1"


def _same(mine, theirs):
    if isinstance(theirs, str):
        return mine == theirs
    if not isinstance(mine, np.ndarray):
        return False
    return (
        mine.dtype.str == theirs["dtype"]
        and list(mine.shape) == theirs["shape"]
        and mine.tobytes().hex() == theirs["hex"]
    )


def _differences(mine, theirs):
    """(only_file, only_lvpyio, value_differs) key sets of one scope."""
    only_file = set(mine) - set(theirs)
    only_lvpyio = set(theirs) - set(mine)
    differs = {k for k in set(mine) & set(theirs) if not _same(mine[k], theirs[k])}
    return only_file, only_lvpyio, differs


@pytest.mark.parametrize("set_path", [NTNU, REC_145749], ids=lambda p: p.stem[:12])
def test_production_attributes_equal_lvpyio_but_for_classified_keys(
    set_path, clean_set_cache
):
    """§6 item 10 on the committed cuts: every key of every image and frame
    equals lvpyio 1.3.1's, except the keys of REPORT §2 B1-B3."""
    rec = read_set_attributes(set_path)
    oracle = LVPYIO["recordings"][set_path.stem]["images"]
    assert rec.n_entries == len(oracle)
    matched = 0
    for i, image in enumerate(oracle):
        only_file, only_lvpyio, differs = _differences(
            rec.buffer_attributes(i), image["buffer"]
        )
        assert (only_file, differs) == (set(), set()), f"image {i} buffer"
        assert only_lvpyio <= LVPYIO_INVENTED_BUFFER_KEYS, f"image {i} buffer"
        assert len(image["frames"]) == rec.n_frames
        for k, theirs in enumerate(image["frames"]):
            mine = rec.frame_attributes(i, k)
            only_file, only_lvpyio, differs = _differences(mine, theirs)
            assert only_file <= LVPYIO_DROPPED_FRAME_KEYS, f"image {i} frame {k}"
            assert only_lvpyio <= LVPYIO_ADDED_FRAME_KEYS, f"image {i} frame {k}"
            assert differs <= LVPYIO_REWRITTEN_FRAME_KEYS, f"image {i} frame {k}"
            for key in CAMERA_KEYS:
                assert (key in mine) == (key in theirs)
                if key in mine:
                    assert mine[key] == theirs[key]
            matched += len(set(mine) & set(theirs)) - len(differs)
    assert matched > 40


def test_oracle_comparison_fails_when_frames_are_bound_wrongly(clean_set_cache):
    """The negative control of REPORT §3: frame k compared with lvpyio's frame
    k + 1 disagrees on the camera keys, so the comparison above can fail."""
    rec = read_set_attributes(NTNU)
    oracle = LVPYIO["recordings"][NTNU.stem]["images"][0]["frames"]
    shifted = [
        _differences(rec.frame_attributes(0, k), oracle[(k + 1) % 6])[2]
        for k in range(6)
    ]
    assert all(set(CAMERA_KEYS) & d for d in shifted)


# ---------------------------------------------------------------------------
# Binding and merging (StreamSet.xml, not N)
# ---------------------------------------------------------------------------


def _merge_set(tmp_path, order):
    s = build_raw16_set(tmp_path, [(4, 6)], name=f"merge_{'_'.join(order)}")
    folder = s.with_suffix("")
    write_attr(folder / "MetaData7.attr", [{"k": "seven", "only7": "a"}])
    write_attr(folder / "MetaData2.attr", [{"k": "two", "only2": "b"}])
    write_attr(folder / "MetaData5.attr", [{"buffer_key": "c"}])
    write_manifest(
        folder,
        [("Frame0", 0)],
        attribute_readers=[(prefix, 0) for prefix in order] + [("MetaData5", None)],
    )
    return s


@pytest.mark.parametrize(
    "order, winner",
    [(("MetaData7", "MetaData2"), "two"), (("MetaData2", "MetaData7"), "seven")],
)
def test_files_bound_to_one_frame_merge_and_the_later_declared_wins(
    tmp_path, clean_set_cache, order, winner
):
    """§6 item 4: the union of both frame-bound files, the buffer-bound file
    apart, and for the shared key the file declared later, whatever N says."""
    rec = read_set_attributes(_merge_set(tmp_path, order))
    assert rec.frame_attributes(0, 0) == {"k": winner, "only7": "a", "only2": "b"}
    assert rec.buffer_attributes(0) == {"buffer_key": "c"}


def test_declared_attr_file_that_is_missing_is_refused_by_name(
    tmp_path, clean_set_cache
):
    """§6 item 8. lvpyio returns empty attributes here without a message."""
    s = _labelled_set(tmp_path, PIV_PLIF_KEYS, PIV_PLIF_SHAPES)
    (s.with_suffix("") / "MetaData4.attr").unlink()
    with pytest.raises(FileNotFoundError, match=r"MetaData4\.attr, which is missing"):
        read_set_info(s)


def test_attr_file_with_another_image_count_is_refused(tmp_path, clean_set_cache):
    s = _labelled_set(tmp_path, [("Camera 1", "Camera.X: 1", "1: X")], n_images=2)
    write_attr(s.with_suffix("") / "MetaData0.attr", [{"k": "v"}])
    with pytest.raises(AttrFormatError, match=r"describes 1 images.*hold 2"):
        read_set_info(s)


@pytest.mark.parametrize(
    "purpose",
    [
        'IsAssociatedToFrames="true" IsAssociatedToEntireImage="true" '
        'StartFrame="0" EndFrame="0"',
        'IsAssociatedToFrames="false" IsAssociatedToEntireImage="false"',
        'IsAssociatedToFrames="true" IsAssociatedToEntireImage="false" '
        'StartFrame="0" EndFrame="1"',
    ],
)
def test_attribute_binding_other_than_one_frame_or_the_buffer_is_refused(
    tmp_path, clean_set_cache, purpose
):
    s = build_raw16_set(tmp_path, [(4, 6)] * 2)
    folder = s.with_suffix("")
    write_attr(folder / "MetaData0.attr", [{"k": "v"}])
    manifest = (folder / "StreamSet.xml").read_text()
    reader = (
        '<ReaderInfo Name="AttributeReaderV1" Type="Core.Set.Recording.AttributeReader" '
        f'FilePrefix="MetaData0"><ContentPurpose {purpose}/></ReaderInfo>'
    )
    (folder / "StreamSet.xml").write_text(
        manifest.replace("</StreamCfg>", reader + "</StreamCfg>")
    )
    with pytest.raises(ValueError, match="MetaData0"):
        read_set_info(s)


def test_attribute_reader_of_another_name_is_refused(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, [(4, 6)])
    folder = s.with_suffix("")
    write_attr(folder / "MetaData0.attr", [{"k": "v"}])
    write_manifest(
        folder,
        [("Frame0", 0)],
        attribute_readers=[("MetaData0", 0)],
        attribute_reader_name="AttributeReaderV2",
    )
    with pytest.raises(AttrFormatError, match="'AttributeReaderV2'"):
        read_set_info(s)


# ---------------------------------------------------------------------------
# Identity rules (decisions of 17 Nov)
# ---------------------------------------------------------------------------


def test_labelled_piv_plus_plif_rig_splits_by_label(tmp_path, clean_set_cache):
    s = _labelled_set(tmp_path, PIV_PLIF_KEYS, PIV_PLIF_SHAPES)
    assert set_camera_streams(s, 4) == [(0, 1), (2, 3), (4,), (5,)]


def test_partial_labels_group_from_the_labelled_images(tmp_path, clean_set_cache):
    """§6 item 9: images 2..n without camera keys group from image 1."""
    s = _labelled_set(
        tmp_path, PIV_PLIF_KEYS, PIV_PLIF_SHAPES, n_images=3, labelled={0}
    )
    info = read_set_info(s)
    assert [c.frames for c in info.cameras] == [(0, 1), (2, 3), (4,), (5,)]
    assert [c.labelled_images for c in info.cameras] == [1, 1, 1, 1]


def test_a_label_that_changes_between_images_is_refused_naming_both(
    tmp_path, clean_set_cache
):
    """§6 item 9: image 2 with a different label raises, naming both images."""
    s = _labelled_set(tmp_path, [("Camera 1", "Camera.X: 1", "1: X")], n_images=2)
    write_attr(
        s.with_suffix("") / "MetaData0.attr",
        [
            camera_keys("Camera 1", "Camera.X: 1", "1: X"),
            camera_keys("Camera 2", "Camera.X: 2", "1: X"),
        ],
    )
    with pytest.raises(CameraIdentityError, match=r"on image 1 but .* on image 2"):
        read_set_info(s)


def test_an_image_with_some_camera_keys_but_not_all_is_refused(
    tmp_path, clean_set_cache
):
    s = _labelled_set(tmp_path, [("Camera 1", "Camera.X: 1", "1: X")], n_images=2)
    partial = camera_keys("Camera 1", "Camera.X: 1", "1: X")
    del partial["Acq.Camera.ID"]
    write_attr(
        s.with_suffix("") / "MetaData0.attr",
        [camera_keys("Camera 1", "Camera.X: 1", "1: X"), partial],
    )
    with pytest.raises(CameraIdentityError, match=r"image 2.*not Acq\.Camera\.ID"):
        read_set_info(s)


def test_attr_files_without_any_camera_keys_mean_the_shape_rule(
    tmp_path, clean_set_cache
):
    """Decision 2: an .attr with no camera keys on any image is treated as no
    .attr. The shape rule resolves a unique split and refuses an ambiguous one."""
    s = _labelled_set(tmp_path, [None] * 5, [(6, 8)] + [(4, 6)] * 4)
    info = read_set_info(s)
    assert info.cameras is None and len(info.attribute_files) == 5
    assert [f.camera_label for f in info.frames] == [None] * 5
    assert set_camera_streams(s, 3) == [(0,), (1, 2), (3, 4)]
    with pytest.raises(SetConfigError, match="ambiguous"):
        set_camera_streams(s, 4)


def test_some_streams_labelled_and_others_never_is_refused(tmp_path, clean_set_cache):
    s = _labelled_set(tmp_path, [("Camera 1", "Camera.X: 1", "1: X"), None])
    with pytest.raises(CameraIdentityError, match=r"frame streams \[1\] carry none"):
        read_set_info(s)


def test_camera_numbers_out_of_stream_order_are_refused(tmp_path, clean_set_cache):
    """Decision 3: numbered by first frame stream, CameraName must say 1..n."""
    s = _labelled_set(
        tmp_path,
        [("Camera 2", "Camera.X: 2", "2: X"), ("Camera 1", "Camera.X: 1", "1: X")],
    )
    with pytest.raises(CameraIdentityError, match=r"numbers \[2, 1\], not 1\.\.2"):
        read_set_info(s)


@pytest.mark.parametrize(
    "keys",
    [
        [("Camera 1", "Camera.X: 1", "1B: X"), ("Camera 1", "Camera.X: 1", "1A: X")],
        [("Camera 1", "Camera.X: 1", "1A: X")],
        [("Camera 1", "Camera.X: 1", "1A: X"), ("Camera 1", "Camera.X: 1", "1: X")],
        [
            ("Camera 1", "Camera.X: 1", "1A: X"),
            ("Camera 2", "Camera.X: 2", "2: X"),
            ("Camera 1", "Camera.X: 1", "1B: X"),
        ],
        [("Camera 1", "Camera.X: 1", "Imager")],
    ],
    ids=["B before A", "A alone", "mixed roles", "A and B apart", "bad CameraName"],
)
def test_impossible_camera_layouts_are_refused(tmp_path, clean_set_cache, keys):
    s = _labelled_set(tmp_path, keys)
    with pytest.raises(CameraIdentityError):
        read_set_info(s)


def test_no_attribute_reader_keeps_the_shape_rule(tmp_path, clean_set_cache):
    """§6 item 7 (test_set_stream_split.py holds the rest)."""
    s = build_raw16_set(tmp_path, [(6, 8)] + [(4, 6)] * 4)
    info = read_set_info(s)
    assert info.cameras is None and info.attribute_files == ()
    assert set_camera_streams(s, 3) == [(0,), (1, 2), (3, 4)]


# ---------------------------------------------------------------------------
# An unknown transformer blocks only its camera when labels name the cameras
# ---------------------------------------------------------------------------


def test_unknown_transformer_blocks_only_its_labelled_camera(tmp_path, clean_set_cache):
    s = _labelled_set(
        tmp_path,
        PIV_PLIF_KEYS,
        PIV_PLIF_SHAPES,
        transformers=[("rotate-45", "Transformer0", 3)],
    )
    info = read_set_info(s)
    assert set_camera_streams(s, 4) == [(0, 1), (2, 3), (4,), (5,)]
    assert read_set_frame(s, 1, 0)[0, 0] == 0  # camera 1 reads
    assert read_set_frame(s, 1, 4)[0, 0] == 400  # camera 3 reads
    for stream in (2, 3):  # both frames of camera 2 are blocked
        with pytest.raises(ValueError) as exc:
            read_set_frame(s, 1, stream)
        message = str(exc.value)
        assert "Camera 2" in message and "'rotate-45'" in message
        assert "frame stream 3" in message and "other cameras" in message
    with pytest.raises(ValueError, match="'rotate-45'"):
        info.frames[3].shape  # noqa: B018  (the output shape is unknown)
    cfg = make_set_config(
        tmp_path, s, camera_count=4, camera_numbers=[1, 2], frame_stride=0
    )
    assert read_pair(1, s, 1, cfg).shape == (2, 6, 6)
    with pytest.raises(ValueError, match="'rotate-45'"):
        read_pair(1, s, 2, cfg)


def test_unknown_transformer_without_labels_still_refuses_everything(
    tmp_path, clean_set_cache
):
    s = build_raw16_set(tmp_path, [(4, 6)] * 2)
    write_manifest(
        s.with_suffix(""),
        [("Frame0", 0), ("Frame1", 1)],
        transformers=[("rotate-45", "Transformer0", 1)],
    )
    with pytest.raises(ValueError, match="no camera of this recording can be read"):
        read_set_info(s)


# ---------------------------------------------------------------------------
# Source diagnosis: the stream table, the found line, the camera suggestion
# ---------------------------------------------------------------------------


def _set_inp(set_file, **ctx):
    return _inp(set_file, image_type="lavision_set", **ctx)


def test_stream_table_carries_each_streams_camera_label(clean_set_cache):
    infos = set_stream_infos(NTNU, 1)  # the configured count does not matter
    assert [i["camera"] for i in infos] == [1, 1, 2, 2, 3, 4]
    assert [i["role"] for i in infos] == ["A", "B", "A", "B", "", ""]
    assert [i["label"] for i in infos][3:5] == ["Camera 2", "Camera 3"]
    assert infos[5]["camera_id"] == "Camera.ImagerSCMOS5MCLHS: 61009193"
    assert infos[0]["model"] == "Imager MX 25M"
    assert (infos[4]["width"], infos[4]["height"]) == (6, 4)


def test_stream_table_without_labels_has_no_label_fields(tmp_path, clean_set_cache):
    s = build_raw16_set(tmp_path, [(6, 8)] + [(4, 6)] * 4)
    infos = set_stream_infos(s, 3)
    assert {(i["label"], i["camera_id"], i["model"], i["role"]) for i in infos} == {
        (None, None, None, None)
    }


def test_stream_table_says_size_unknown_for_an_unknown_transformer(
    tmp_path, clean_set_cache
):
    s = _labelled_set(
        tmp_path, PIV_PLIF_KEYS, PIV_PLIF_SHAPES, transformers=[("rotate-45", "T0", 3)]
    )
    infos = set_stream_infos(s, 4)
    assert [(i["width"], i["height"]) for i in infos][2:4] == [(6, 6), (None, None)]
    assert "3: size unknown raw-16-bit" in streams_sentence(s.name, infos)


def test_labelled_wrong_count_suggests_the_labels_and_their_double_frame_cameras(
    clean_set_cache,
):
    """§6 item 6: count 4, the double-frame cameras 1 and 2 pre-selected."""
    [x] = rule_set_streams(
        _set_inp(NTNU, camera_count=1, camera_numbers=(1,), pair_reader=True)
    )
    assert (x["apply"], x["label"]) == (
        {"camera_count": 4, "camera_numbers": [1, 2]},
        "Use 4 cameras, process 1 and 2",
    )
    assert "by the recording's own camera labels" in x["message"]
    assert "Camera 3 (Imager sCMOS CLHS" in x["message"]
    assert "single frame" in x["message"] and "PLIF" not in x["message"]


def test_labelled_right_count_deselects_the_single_frame_cameras(clean_set_cache):
    [x] = rule_set_streams(
        _set_inp(NTNU, camera_count=4, camera_numbers=(1, 2, 3, 4), pair_reader=True)
    )
    assert (x["apply"], x["label"]) == (
        {"camera_numbers": [1, 2]},
        "Process cameras 1 and 2",
    )
    assert (
        rule_set_streams(
            _set_inp(NTNU, camera_count=4, camera_numbers=(1, 2), pair_reader=True)
        )
        == []
    )


def test_labelled_time_resolved_allows_single_frame_cameras(clean_set_cache):
    [x] = rule_set_streams(
        _set_inp(NTNU, camera_count=2, camera_numbers=(1, 2), pair_reader=False)
    )
    assert x["apply"] == {"camera_count": 4, "camera_numbers": [1, 2, 3, 4]}
    assert (
        rule_set_streams(
            _set_inp(NTNU, camera_count=4, camera_numbers=(3, 4), pair_reader=False)
        )
        == []
    )


def test_labelled_single_frame_only_rig_offers_no_pair_selection(
    tmp_path, clean_set_cache
):
    s = _labelled_set(
        tmp_path,
        [("Camera 1", "Camera.S: 1", "1: S"), ("Camera 2", "Camera.S: 2", "2: S")],
    )
    assert (
        rule_set_streams(
            _set_inp(s, camera_count=2, camera_numbers=(1, 2), pair_reader=True)
        )
        == []
    )
    [x] = rule_set_streams(
        _set_inp(s, camera_count=1, camera_numbers=(1,), pair_reader=False)
    )
    assert x["apply"] == {"camera_count": 2, "camera_numbers": [1, 2]}


def _found(set_file, **ctx):
    ctx = make_ctx(set_file, image_type="lavision_set", **ctx)
    infos = set_stream_infos(set_file, ctx.camera_count)
    return found_for(scan_source(set_file), ctx, {"valid": True}, infos, None)


def test_found_line_names_the_labelled_cameras(clean_set_cache):
    text = _found(NTNU, camera_count=4)
    assert text.startswith("REF_background_check_1006.set: 6 streams")
    assert (
        "4 cameras from the recording's own labels: Camera 1 (Imager MX 25M, "
        "Camera.ImagerMX25M: 290187, double frame (A/B)) on streams 0–1;"
    ) in text
    assert (
        "Camera 4 (Imager sCMOS CLHS, Camera.ImagerSCMOS5MCLHS: 61009193, single frame) on stream 5."
        in text
    )
    assert "labels on" not in text and "suspicious" not in text


SUSPICIOUS_145749 = (
    "DaVis marked this recording as suspicious (SetIsSuspicious = 1 in "
    "Recording_Date=261030_Time=145749.set) without saying why; check its "
    "images before processing."
)


def test_found_line_reports_partial_labels_but_not_the_suspicious_flag(
    clean_set_cache,
):
    """``found`` holds facts; DaVis's warning goes in ``warnings``, so it can look like one."""
    text = _found(REC_145749, camera_count=1)
    assert "1 camera from the recording's own labels: Camera 1 (Imager CX-25" in text
    assert "Camera labels on 1 of 2 images; every labelled image agrees." in text
    assert "suspicious" not in text


def test_the_suspicious_flag_is_a_warning(clean_set_cache):
    assert recording_warnings(read_set_info(REC_145749)) == [SUSPICIOUS_145749]
    assert recording_warnings(read_set_info(NTNU)) == []


def _set_config(tmp_path, set_file, camera_count):
    return write_config(
        tmp_path,
        paths={
            "source_paths": [str(set_file)],
            "base_paths": [str(tmp_path / "out")],
            "active_paths": [0],
            "camera_count": camera_count,
            "camera_numbers": list(range(1, camera_count + 1)),
        },
        images={
            "image_type": "lavision_set",
            "image_format": ["data.set"],
            "pairing_preset": "pre_paired",
            "frame_stride": 0,
            "pair_stride": 1,
            "num_images": 2,
        },
    )


def test_cli_validation_reports_the_suspicious_flag_once(tmp_path, clean_set_cache):
    """The CLI's pre-run check passes DaVis's warning on, once per recording."""
    _ok, _error, warnings = validate_config(_set_config(tmp_path, REC_145749, 1))
    assert warnings.count(SUSPICIOUS_145749) == 1


def test_cli_validation_of_an_unflagged_recording_has_no_such_warning(
    tmp_path, clean_set_cache
):
    _ok, _error, warnings = validate_config(_set_config(tmp_path, NTNU, 4))
    assert not any("suspicious" in w for w in warnings)


def test_validate_files_reports_the_suspicious_flag_as_a_warning(
    tmp_path, monkeypatch, clean_set_cache
):
    body = piv_route_response(
        tmp_path,
        monkeypatch,
        paths={"source_paths": [str(REC_145749)]},
        images={
            "image_type": "lavision_set",
            "image_format": ["data.set"],
            "pairing_preset": "pre_paired",
            "frame_stride": 0,
        },
    ).get_json()
    cam = body["details"]["camera_1"]
    assert cam["warnings"] == [SUSPICIOUS_145749]
    assert "suspicious" not in cam["found"]


def test_found_line_says_when_cameras_come_from_shapes(tmp_path, clean_set_cache):
    plain = build_raw16_set(tmp_path, [(6, 8)] + [(4, 6)] * 4, name="plain")
    assert _found(plain, camera_count=3).endswith(
        "No camera labels: the recording has no .attr files, so cameras are "
        "matched to streams by shape."
    )
    blank = _labelled_set(tmp_path, [None] * 2, name="blank")
    assert _found(blank, camera_count=2).endswith(
        "No camera labels: its .attr files carry no camera keys, so cameras are "
        "matched to streams by shape."
    )


def test_scan_sentence_counts_stream_files_not_frame_streams(clean_set_cache):
    """D5 note: 145749 holds one stream file with A and B, two frame streams."""
    folder = REC_145749.with_suffix("")
    assert scan_sentence(scan_source(folder)) == (
        f"{folder.as_posix()} is the data folder of {REC_145749.as_posix()} "
        f"(1 stream file)."
    )
