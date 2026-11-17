"""Trigger alignment of ``.pivraw`` runs (ticket P2, decision 02-1).

Ground truth is the recorder's own record of each contract fixture
(``pivraw_contract/expected.json``): the pairs, the 1-based file frames per
camera, the skipped count, or the reason the run must be refused. The same
outcomes pin the acquisition app's ``export.align_run`` (A1), so the two
aligners agree case by case. Edited copies of the fixtures cover the cases the
recorder cannot be made to write (a software tick, a flipped camera clock).
"""

import numpy as np
import pytest
import yaml
from fixtures.diagnosis import make_ctx
from fixtures.pivraw import (
    CONTRACT_DIR,
    contract_cases,
    copy_case,
    edit_index,
    edit_run_yaml,
    edit_sidecar,
    expected_frames,
    write_piv_config,
)

from pivtools_core.config import Config
from pivtools_core.image_handling.load_images import _read_batch, read_pair
from pivtools_core.image_handling.path_utils import validate_images_generic
from pivtools_core.image_handling.readers.pivraw_reader import (
    PivrawAlignmentError,
    PivrawError,
    align_pivraw_run,
    clear_alignment_cache,
    read_pivraw_header,
    read_pivraw_single,
)
from pivtools_core.validation import validate_config

CASES = contract_cases()
# The fixtures were decided with the presets a user picks for each mode:
# double-frame skip_frames (frame_stride 1, pair_stride 2), time-resolved
# time_resolved (1, 1).
STRIDES = {"double_frame": (1, 2), "time_resolved": (1, 1)}


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_alignment_cache()
    yield
    clear_alignment_cache()


@pytest.mark.parametrize("case", sorted(CASES))
def test_alignment_matches_the_decided_outcome(case):
    truth = CASES[case]
    frame_stride, pair_stride = STRIDES[truth["mode"]]
    if truth["outcome"].startswith("must_raise:"):
        reason = truth["outcome"].split(":", 1)[1]
        with pytest.raises(PivrawAlignmentError) as info:
            align_pivraw_run(CONTRACT_DIR / case, frame_stride, pair_stride)
        assert info.value.reason == reason, str(info.value)
        return
    assert truth["outcome"] == "pairs"
    alignment = align_pivraw_run(CONTRACT_DIR / case, frame_stride, pair_stride)
    got = [
        (list(p.run_triggers), {str(s): list(f) for s, f in p.file_frames.items()})
        for p in alignment.pairs
    ]
    want = [(p["run_triggers"], p["file_frames"]) for p in truth["pairs"]]
    assert got == want
    assert alignment.skipped_incomplete == truth["skipped_incomplete"]


# ---------------------------------------------------------------------------
# Refusals name the camera, the trigger and the reason (PC section 7, T7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, words",
    [
        ("df_lost_first_frame", ["Camera1", "A->B", "camera clock", "first frame"]),
        (
            "df_2cam_lost_first_pair_cam1",
            ["Camera2 run trigger 0", "periods) from Camera1", "first frame(s) lost"],
        ),
        (
            "df_2cam_missed_trigger_cam2",
            ["Camera2 run trigger 1", "+1.0", "a trigger ignored"],
        ),
        ("df_2cam_cam2_empty", ["Camera2 holds no frames"]),
    ],
)
def test_refusal_message_names_camera_and_reason(case, words):
    with pytest.raises(PivrawAlignmentError) as info:
        align_pivraw_run(CONTRACT_DIR / case, 1, 2)
    for word in words:
        assert word in str(info.value), str(info.value)
    assert str(CONTRACT_DIR / case) in str(info.value)


def test_cross_camera_check_uses_host_time_not_the_camera_clocks():
    """The fixtures' camera clocks are hours apart (CAMERA_CLOCK_OFFSET_NS);
    a clean run must still align, so the cross-camera check is on host time."""
    clocks = [
        read_pivraw_header(CONTRACT_DIR / "df_2cam_clean" / f"Camera{c}.pivraw").index
        for c in (1, 2)
    ]
    hours_apart = abs(int(clocks[1]["camera_ns"][0]) - int(clocks[0]["camera_ns"][0]))
    assert hours_apart > 3600e9
    assert len(align_pivraw_run(CONTRACT_DIR / "df_2cam_clean", 1, 2).pairs) == 2


def _late_tick(tmp_path, trigger):
    """tr_bayer_clean with its third frame 30 ms late (70 ms after the second
    at a 40 ms period): a software tick that fired late, or a DAQ run whose
    camera ignored a trigger."""
    run = copy_case(tmp_path, "tr_bayer_clean", f"late_tick_{trigger}")
    host = read_pivraw_header(run / "Camera1.pivraw").index["host_ns"].copy()
    host[2] = host[1] + 70_000_000
    edit_index(run, 1, host_ns=host)
    edit_run_yaml(run, trigger=trigger)
    return run


def test_late_software_tick_points_to_the_daq(tmp_path):
    run = _late_tick(tmp_path, "software")
    with pytest.raises(PivrawAlignmentError) as info:
        align_pivraw_run(run, 1, 1)
    assert info.value.reason == "tr_interval"
    message = str(info.value)
    for word in [
        "Camera1 run triggers (1, 2)",
        "70.0 ms apart, expected 40.0 ms",
        "software-triggered",
        "DAQ trigger",
    ]:
        assert word in message, message


def test_late_frame_on_a_daq_run_says_a_trigger_was_ignored(tmp_path):
    run = _late_tick(tmp_path, "daq")
    with pytest.raises(PivrawAlignmentError, match="ignored a trigger") as info:
        align_pivraw_run(run, 1, 1)
    assert "software" not in str(info.value)


def test_phase_is_judged_on_the_camera_clock(tmp_path):
    """Host time says flipped, the camera clock says A then B: the camera clock
    decides. Stand-in: df_2cam_clean camera 1 with host A->B made 39 ms."""
    run = copy_case(tmp_path, "df_2cam_clean")
    host = read_pivraw_header(run / "Camera1.pivraw").index["host_ns"].copy()
    host[1] = host[0] + 39_000_000
    host[3] = host[2] + 39_000_000
    edit_index(run, 1, host_ns=host)
    alignment = align_pivraw_run(run, 1, 2)
    assert len(alignment.pairs) == 2 and alignment.notes == ()


def test_unprovable_phase_is_a_note_not_a_refusal(tmp_path):
    """Camera 1's A->B made 20 ms on its own clock, B->next A 20 ms: within the
    margin, so the phase cannot be proven and the pairs are still read."""
    run = copy_case(tmp_path, "df_2cam_clean")
    cam = read_pivraw_header(run / "Camera1.pivraw").index["camera_ns"].copy()
    cam[1] = cam[0] + 20_000_000
    cam[2] = cam[1] + 20_000_000
    cam[3] = cam[2] + 20_000_000
    edit_index(run, 1, camera_ns=cam)
    alignment = align_pivraw_run(run, 1, 2)
    assert len(alignment.pairs) == 2
    [note] = alignment.notes
    assert note.startswith("Camera1: A/B phase not provable") and "20.00 ms" in note


def test_phase_with_one_pair_per_run_is_a_note():
    alignment = align_pivraw_run(CONTRACT_DIR / "df_2cam_unequal_drops", 1, 2)
    assert alignment.notes == (
        "Camera1: A/B phase not provable (fewer than two consecutive pairs); "
        "pairs are read as recorded.",
    )


def test_unaccounted_triggers_become_a_note(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    edit_run_yaml(run, record_n=3)
    alignment = align_pivraw_run(run, 1, 2)
    assert alignment.triggers_requested == 3 and alignment.triggers_unaccounted == 1
    assert any(
        "1 trigger(s) were lost before the first or after the last" in n
        for n in alignment.notes
    )


def test_fewer_triggers_requested_than_recorded_is_refused(tmp_path):
    """run.yaml asks for 1 trigger, the cameras hold 2: the run.yaml is stale
    or belongs to another run, so the run is refused by name."""
    run = copy_case(tmp_path, "df_2cam_clean")
    edit_run_yaml(run, record_n=1)
    with pytest.raises(PivrawAlignmentError) as info:
        align_pivraw_run(run, 1, 2)
    assert info.value.reason == "run_file_count"
    message = str(info.value)
    for word in [str(run), "requested 1 trigger(s)", "account for 2", "run.yaml"]:
        assert word in message, message


@pytest.mark.parametrize("suffix", [".json", ".idx"])
def test_a_missing_camera_file_is_refused_by_name(tmp_path, suffix):
    """A run folder missing Camera2's sidecar or index: a PivrawError naming
    the file, which every route answers as 400, not a bare FileNotFoundError."""
    run = copy_case(tmp_path, "df_2cam_clean")
    missing = run / f"Camera2.pivraw{suffix}"
    missing.unlink()
    with pytest.raises(PivrawError) as info:
        align_pivraw_run(run, 1, 2)
    assert str(missing) in str(info.value)


def test_clearing_the_header_cache_drops_the_alignments_too(tmp_path):
    """clear_header_cache drops every cached view of the run's files, so the
    alignment is computed again (a new object, not the cached one)."""
    from pivtools_core.image_handling.readers.pivraw_reader import clear_header_cache

    run = copy_case(tmp_path, "df_2cam_clean")
    first = align_pivraw_run(run, 1, 2)
    assert align_pivraw_run(run, 1, 2) is first
    clear_header_cache()
    assert align_pivraw_run(run, 1, 2) is not first


def test_missing_run_yaml_is_refused(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    (run / "run.yaml").unlink()
    with pytest.raises(
        PivrawError, match=r"run\.yaml: cannot read the run description"
    ):
        align_pivraw_run(run, 1, 2)


def test_cameras_on_different_rates_are_refused(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    edit_sidecar(run, 2, rate_hz=10.0)
    with pytest.raises(PivrawAlignmentError, match="disagree") as info:
        align_pivraw_run(run, 1, 2)
    assert info.value.reason == "cameras_disagree"


def test_alignment_is_recomputed_when_a_file_changes(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    assert align_pivraw_run(run, 1, 2).triggers_unaccounted == 0
    edit_run_yaml(run, record_n=5)
    assert align_pivraw_run(run, 1, 2).triggers_unaccounted == 3


# ---------------------------------------------------------------------------
# Validation reports the alignment per camera and refuses what cannot align
# ---------------------------------------------------------------------------


def _validate(run, camera, pair_strides):
    path = run / f"Camera{camera}.pivraw"
    return validate_images_generic(
        camera_path=run,
        camera=camera,
        image_format="Camera%d.pivraw",
        image_type="pivraw",
        expected_count=4,
        zero_based_indexing=False,
        read_frame_fn=lambda idx: read_pivraw_single(path, idx),
        diagnosis=make_ctx(
            run,
            image_type="pivraw",
            image_format=("Camera%d.pivraw",),
            camera_count=2,
            camera_numbers=(1, 2),
        ),
        pair_strides=pair_strides,
    )


def test_validation_reports_skipped_pairs_per_camera(tmp_path):
    run = copy_case(tmp_path, "df_2cam_unequal_drops")
    for camera in (1, 2):
        result = _validate(run, camera, (1, 2))
        assert result["valid"] is True, result["error"]
        report = result["pivraw_alignment"]
        assert report["pairs"] == 1 and report["skipped_incomplete"] == 2
        assert report["skipped_by_camera"] == {"1": 1, "2": 1}  # str keys: jsonify
        assert report["summary"] == (
            "1 pair(s) aligned by trigger on every camera; 2 skipped as "
            "incomplete (Camera1 lacks a frame of 1, Camera2 lacks a frame of 1)."
        )
        assert report["notes"] == [
            "Camera1: A/B phase not provable (fewer than two consecutive pairs); "
            "pairs are read as recorded."
        ]


def test_validation_refuses_a_run_that_cannot_align(tmp_path):
    run = copy_case(tmp_path, "df_2cam_lost_first_pair_cam1")
    result = _validate(run, 1, (1, 2))
    assert result["valid"] is False
    assert "Camera2 run trigger 0" in result["error"]
    assert result["read_error"] == result["error"]
    assert result["pivraw_alignment"] is None


def test_a_preview_that_cannot_render_is_a_note_and_leaves_no_size(tmp_path):
    """Stand-in: a first frame of zero rows, which the preview renderer cannot
    draw. The user reads why there is no preview, and no size is reported
    for a frame that was not shown."""
    run = copy_case(tmp_path, "df_2cam_clean")
    result = validate_images_generic(
        camera_path=run,
        camera=1,
        image_format="Camera%d.pivraw",
        image_type="pivraw",
        expected_count=4,
        zero_based_indexing=False,
        read_frame_fn=lambda idx: np.zeros((0, 5)),
        diagnosis=make_ctx(run, image_type="pivraw", image_format=("Camera%d.pivraw",)),
        pair_strides=(1, 2),
    )
    assert result["valid"] is True, result["error"]
    assert result["image_size"] is None and result["first_image_preview"] is None
    assert any(
        n.startswith("No preview of Camera1.pivraw frame 1:") for n in result["notes"]
    ), result["notes"]


def test_piv_validation_of_a_run_needs_the_strides(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    with pytest.raises(ValueError, match="pair_strides"):
        _validate(run, 1, None)


def test_validate_config_warns_skipped_pairs_and_notes(tmp_path):
    run = _cam2_dropped_pair_2(tmp_path)
    cfg = _config_with_base(tmp_path, run, num_images=4)
    ok, error, warnings = validate_config(cfg)
    assert ok, error
    assert any(
        "2 pair(s) aligned by trigger on every camera; 1 skipped as incomplete "
        "(Camera2 lacks a frame of 1)." in w
        for w in warnings
    ), warnings
    assert any("Camera2: A/B phase not provable" in w for w in warnings), warnings


def test_validate_config_says_when_num_images_caps_the_pairs(tmp_path):
    run = _cam2_dropped_pair_2(tmp_path)
    cfg = _config_with_base(tmp_path, run, num_images=2)
    ok, error, warnings = validate_config(cfg)
    assert ok, error
    assert any(
        "Processing 1 of 2 aligned pairs: images.num_images (2) caps them." in w
        for w in warnings
    ), warnings


def test_validate_config_reports_every_runs_findings(tmp_path):
    """Two runs of 2 aligned pairs each: the first skips a pair and cannot prove
    Camera2's phase, the second cannot prove Camera1's phase. Both runs'
    findings reach the warnings, each named by its run."""
    first = _cam2_dropped_pair_2(tmp_path)
    second = copy_case(tmp_path, "df_2cam_clean", "second_run")
    cam = read_pivraw_header(second / "Camera1.pivraw").index["camera_ns"].copy()
    cam[1] = cam[0] + 20_000_000
    cam[2] = cam[1] + 20_000_000
    cam[3] = cam[2] + 20_000_000
    edit_index(second, 1, camera_ns=cam)
    cfg = _config_with_base(tmp_path, first, num_images=4)
    cfg.data["paths"]["source_paths"] = [str(first), str(second)]
    cfg.data["paths"]["base_paths"] = [str(tmp_path / "o1"), str(tmp_path / "o2")]
    ok, error, warnings = validate_config(cfg)
    assert ok, error
    assert any(
        w.startswith(f"{first}: 2 pair(s) aligned")
        and "Camera2 lacks a frame of 1" in w
        for w in warnings
    ), warnings
    assert any(
        w.startswith(f"{first}: Camera2: A/B phase not provable") for w in warnings
    ), warnings
    assert any(
        w.startswith(f"{second}: Camera1: A/B phase not provable") for w in warnings
    ), warnings
    assert not any(w.startswith(f"{second}: 2 pair(s) aligned") for w in warnings)


def test_validate_config_refuses_a_run_that_cannot_align(tmp_path):
    run = copy_case(tmp_path, "df_2cam_missed_trigger_cam2")
    cfg = _config_with_base(tmp_path, run, num_images=4)
    ok, error, _warnings = validate_config(cfg)
    assert not ok
    assert "Camera2 run trigger 1" in error and "a trigger ignored" in error


def _config_with_base(tmp_path, run, num_images):
    config_path = write_piv_config(tmp_path, run, 2, num_images)
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["base_paths"] = [str(tmp_path / "out")]
    config_path.write_text(yaml.dump(data))
    return Config(path=str(config_path))


# ---------------------------------------------------------------------------
# dropped_pairs against missing_indices (writer: len(missing) // 2 for
# double-frame, 0 for time-resolved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, camera, dropped",
    [
        ("df_2cam_unequal_drops", 1, 0),  # missing [2, 3] says 1
        ("df_2cam_unequal_drops", 1, 2),
        ("tr_lost_frame", 1, 1),  # time-resolved is always 0
    ],
)
def test_dropped_pairs_must_match_missing_indices(tmp_path, case, camera, dropped):
    run = copy_case(tmp_path, case)
    path = edit_sidecar(run, camera, dropped_pairs=dropped)
    with pytest.raises(PivrawError, match=r"dropped_pairs .* missing_indices"):
        read_pivraw_header(path)


def _cam2_dropped_pair_2(tmp_path):
    """A two-camera run whose cameras keep different file frames for one trigger.

    Stand-in built from ``df_2cam_missed_trigger_cam2``: camera 2's second file
    pair is relabelled run trigger 2 (ids 5004, 5005, with trigger 1 in
    ``missing_indices``). Its host times already sit on camera 1's trigger 2,
    so this is the run the recorder writes when camera 2 drops pair 2. Truth:
    trigger 0 is file pair (1, 2) on both cameras; trigger 2 is (5, 6) on
    camera 1 and (3, 4) on camera 2; trigger 1 is skipped (camera 2 lacks it).
    """
    run = copy_case(tmp_path, "df_2cam_missed_trigger_cam2", "cam2_dropped_pair_2")
    edit_index(run, 2, frame_id=[5000, 5001, 5004, 5005])
    edit_sidecar(run, 2, missing_indices=[2, 3], dropped_pairs=1)
    return run


def test_unequal_drops_give_each_camera_its_own_file_frames(tmp_path):
    run = _cam2_dropped_pair_2(tmp_path)
    alignment = align_pivraw_run(run, 1, 2)
    assert [p.run_triggers for p in alignment.pairs] == [(0, 0), (2, 2)]
    assert alignment.pairs[1].file_frames == {1: (5, 6), 2: (3, 4)}
    assert alignment.skipped_incomplete == 1
    assert alignment.skipped_by_camera == {1: 0, 2: 1}


def test_config_and_read_pair_follow_the_alignment_per_camera(tmp_path):
    run = _cam2_dropped_pair_2(tmp_path)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4)))
    assert cfg.num_frame_pairs == 2
    assert cfg.get_frame_pair_indices(2, camera=1) == (5, 6)
    assert cfg.get_frame_pair_indices(2, camera=2) == (3, 4)

    source = "df_2cam_missed_trigger_cam2"  # the pixels were not edited
    cam1, cam2 = expected_frames(source, 1), expected_frames(source, 2)
    np.testing.assert_array_equal(read_pair(2, run, 1, cfg), cam1[4:6])
    np.testing.assert_array_equal(read_pair(2, run, 2, cfg), cam2[2:4])
    batch = _read_batch(1, 2, run, 2, cfg)
    np.testing.assert_array_equal(batch[0], cam2[0:2])
    np.testing.assert_array_equal(batch[1], cam2[2:4])


def test_a_reader_refusal_reaches_shape_detection_unchanged(tmp_path):
    """The reader's PivrawError is what the caller sees, not a ValueError
    wrapping it in 'could not read image file' advice."""
    run = copy_case(tmp_path, "df_2cam_unequal_drops")
    edit_sidecar(run, 1, dropped_pairs=0)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4)))
    with pytest.raises(PivrawError) as info:
        cfg.image_shape_for(1)
    message = str(info.value)
    assert "dropped_pairs" in message and "missing_indices" in message
    assert "Could not read" not in message


def test_pivraw_pair_indices_need_the_camera(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4)))
    with pytest.raises(ValueError, match="per camera"):
        cfg.get_frame_pair_indices(1)


def test_num_images_caps_the_aligned_pairs(tmp_path):
    """num_images keeps its subset meaning: the first aligned pairs, at most as
    many as that many frames of a clean run would give."""
    run = _cam2_dropped_pair_2(tmp_path)
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 2)))
    assert cfg.num_frame_pairs == 1
    assert cfg.get_frame_pair_indices(1, camera=2) == (1, 2)


@pytest.mark.parametrize(
    "images, message",
    [
        ({"start_index": 0}, "start_index"),
        ({"num_loops": 2}, "num_loops"),
        ({"pairing_preset": "time_resolved", "pair_stride": 1}, "skip_frames"),
        ({"pairing_preset": "ab_format", "frame_stride": 0}, "skip_frames"),
    ],
)
def test_unsupported_pairing_is_refused_by_name(tmp_path, images, message):
    run = copy_case(tmp_path, "df_2cam_clean")
    cfg = Config(path=str(write_piv_config(tmp_path, run, 2, 4)))
    cfg.data["images"].update(images)
    with pytest.raises(PivrawError, match=message):
        cfg.num_frame_pairs
    with pytest.raises(PivrawError, match=message):
        read_pair(1, run, 1, cfg)


def test_two_runs_with_different_pair_counts_are_refused(tmp_path):
    """One config, two runs: 2 aligned pairs and 1. A single pair count would
    read past the end of one run, so the config is refused naming both."""
    clean = copy_case(tmp_path, "df_2cam_clean")
    dropped = copy_case(tmp_path, "df_2cam_unequal_drops")
    cfg = Config(path=str(write_piv_config(tmp_path, clean, 2, 4)))
    cfg.data["paths"]["source_paths"] = [str(clean), str(dropped)]
    with pytest.raises(PivrawError, match="same number of aligned pairs") as info:
        cfg.num_frame_pairs
    assert str(clean) in str(info.value) and str(dropped) in str(info.value)


def test_an_inactive_run_does_not_block_the_active_one(tmp_path):
    """source_paths [clean, refused], active_paths [0]: only the active run is
    aligned, so the refused one does not stop it."""
    clean = copy_case(tmp_path, "df_2cam_clean")
    refused = copy_case(tmp_path, "df_2cam_missed_trigger_cam2")
    cfg = _config_with_base(tmp_path, clean, num_images=4)
    cfg.data["paths"]["source_paths"] = [str(clean), str(refused)]
    cfg.data["paths"]["base_paths"] = [str(tmp_path / "o1"), str(tmp_path / "o2")]
    cfg.data["paths"]["active_paths"] = [0]
    assert cfg.num_frame_pairs == 2
    ok, error, _warnings = validate_config(cfg)
    assert ok, error


def test_the_alignment_covers_the_selected_cameras_only():
    """Camera 2 of df_2cam_cam2_empty holds no frames. Aligned over every
    camera the run is refused; over camera 1 alone it aligns."""
    run = CONTRACT_DIR / "df_2cam_cam2_empty"
    with pytest.raises(PivrawAlignmentError, match="Camera2 holds no frames"):
        align_pivraw_run(run, 1, 2)
    alignment = align_pivraw_run(run, 1, 2, cameras=(1,))
    assert alignment.cameras == (1,)
    assert len(alignment.pairs) >= 1


def test_deselecting_a_dead_camera_unblocks_the_config(tmp_path):
    run = copy_case(tmp_path, "df_2cam_cam2_empty")
    cfg = _config_with_base(tmp_path, run, num_images=4)
    cfg.data["paths"]["camera_numbers"] = [1]
    assert cfg.num_frame_pairs >= 1
    ok, error, warnings = validate_config(cfg)
    assert ok, error
    assert any(
        "Camera2 is in the run but not selected" in w and "holds no frames" in w
        for w in warnings
    ), warnings


def test_pairs_are_one_trigger_across_the_selected_cameras(tmp_path):
    """Camera 2 lacks trigger 1. Over cameras 1 and 2, pair 2 is trigger 2 on
    both; over camera 1 alone, nothing is skipped and pair 2 is trigger 1. The
    two alignments are cached apart."""
    run = _cam2_dropped_pair_2(tmp_path)
    both = align_pivraw_run(run, 1, 2, cameras=(1, 2))
    one = align_pivraw_run(run, 1, 2, cameras=(1,))
    assert [p.run_triggers for p in both.pairs] == [(0, 0), (2, 2)]
    assert both.pairs[1].file_frames == {1: (5, 6), 2: (3, 4)}
    assert [p.run_triggers for p in one.pairs] == [(0, 0), (1, 1), (2, 2)]
    assert one.pairs[1].file_frames == {1: (3, 4)}
    assert one.skipped_incomplete == 0


def test_a_selected_camera_missing_from_the_run_is_refused(tmp_path):
    run = copy_case(tmp_path, "df_2cam_clean")
    with pytest.raises(PivrawError, match="camera 3") as info:
        align_pivraw_run(run, 1, 2, cameras=(1, 3))
    assert "[1, 2]" in str(info.value)


def test_recorded_dropped_pairs_are_accepted():
    for case, truth in CASES.items():
        for camera, cam in truth["cameras"].items():
            header = read_pivraw_header(CONTRACT_DIR / case / f"Camera{camera}.pivraw")
            assert header.dropped_pairs == cam["dropped_pairs"]
