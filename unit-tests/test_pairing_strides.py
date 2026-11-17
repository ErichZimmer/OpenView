"""Every container reader reads exactly the frames the pairing strides ask for.

``Config.get_frame_pair_indices`` is the one statement of which frames make pair
n. ``load_images.read_pair`` must read those frames, or the config must refuse
the combination by name before anything is read. The 14 Nov review (fault 6,
``01-lavision-readers/REPORT.md`` §2) measured three readers that read other
frames with no error: ``.cine`` read A and A + 1 whatever ``frame_stride`` said,
and a pre-paired ``.set`` read entry n whatever ``pair_stride`` and
``start_index`` said. These cases are the rows of that reproduction
(``scripts/review_faults/fault6_strides_cine_set.py``).

A ``.set`` is synthetic: entry e (0-based) of stream k holds the value
100 k + e, so a read pixel names the 1-based entry it came from. No ``.cine``
sample exists here, so cinereader's two I/O calls are replaced by a stand-in
that returns a frame filled with its own frame number. Everything between
``read_pair`` and those calls is production code.

Run:
    pytest unit-tests/test_pairing_strides.py -v
"""

# ruff: noqa: F811  (the imported clean_set_cache fixture is re-bound as a test argument)

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fixtures.lavision import build_raw16_set, clean_set_cache  # noqa: F401

from pivtools_core.config import Config
from pivtools_core.image_handling import load_images
from pivtools_core.image_handling.readers import cine_reader

ENTRIES = 10
PAIRS_CHECKED = 3
FRAME_SHAPE = (4, 6)
# The synthetic .set marks stream k's entry e with 100 k + e.
STREAM_MARK = 100


def _config(tmp_path: Path, source: Path, images: dict) -> Config:
    cfg = {
        "paths": {
            "source_paths": [str(source)],
            "base_paths": [str(tmp_path / "out")],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {"num_images": ENTRIES, **images},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False))
    return Config(path=str(path))


def _strides(start_index: int, frame_stride: int, pair_stride: int) -> dict:
    return {
        "start_index": start_index,
        "frame_stride": frame_stride,
        "pair_stride": pair_stride,
    }


def _set_config(tmp_path, n_streams, strides):
    set_file = build_raw16_set(
        tmp_path, [FRAME_SHAPE] * n_streams, n_entries=ENTRIES, fill_with_index=True
    )
    images = {"image_format": [set_file.name], "image_type": "lavision_set", **strides}
    return set_file, _config(tmp_path, set_file, images)


def _set_entries_read(pair_no, set_file, cfg):
    pair = load_images.read_pair(pair_no, set_file, 1, cfg)
    return tuple(int(v) % STREAM_MARK + 1 for v in pair[:, 0, 0])


@pytest.fixture
def cine_stand_in(monkeypatch):
    """cinereader's metadata and frame reads, a frame filled with its number."""
    meta = SimpleNamespace(FirstImageNo=1, ImageCount=ENTRIES)
    monkeypatch.setattr(cine_reader, "_get_cached_metadata", lambda _path: meta)
    monkeypatch.setattr(
        cine_reader,
        "cr",
        SimpleNamespace(
            read_image=lambda _m, _p, n: np.full(FRAME_SHAPE, n, dtype=np.uint16)
        ),
    )


def _cine_config(tmp_path, strides):
    images = {"image_format": ["Camera%d.cine"], "image_type": "cine", **strides}
    return _config(tmp_path, tmp_path, images)


def _cine_frames_read(pair_no, tmp_path, cfg):
    pair = load_images.read_pair(pair_no, tmp_path, 1, cfg)
    return tuple(int(v) for v in pair[:, 0, 0])


@pytest.mark.parametrize(
    ("n_streams", "strides"),
    [
        (1, _strides(1, 1, 1)),
        (1, _strides(1, 1, 2)),
        (1, _strides(1, 3, 1)),
        (2, _strides(1, 0, 1)),
        (2, _strides(1, 0, 2)),
        (2, _strides(5, 0, 1)),
    ],
    ids=[
        "time-resolved",
        "time-resolved skip pairs",
        "time-resolved wide gap",
        "pre-paired",
        "pre-paired every 2nd entry",
        "pre-paired from entry 5",
    ],
)
def test_set_reads_the_entries_the_strides_ask_for(
    tmp_path, clean_set_cache, n_streams, strides
):
    set_file, cfg = _set_config(tmp_path, n_streams, strides)

    for pair_no in range(1, PAIRS_CHECKED + 1):
        assert _set_entries_read(pair_no, set_file, cfg) == (
            cfg.get_frame_pair_indices(pair_no)
        ), f"pair {pair_no}"


@pytest.mark.parametrize(
    "strides",
    [_strides(1, 1, 1), _strides(1, 1, 2), _strides(1, 3, 1)],
    ids=["time-resolved", "skip pairs", "wide gap"],
)
def test_cine_reads_the_frames_the_strides_ask_for(tmp_path, cine_stand_in, strides):
    cfg = _cine_config(tmp_path, strides)

    for pair_no in range(1, PAIRS_CHECKED + 1):
        assert _cine_frames_read(pair_no, tmp_path, cfg) == (
            cfg.get_frame_pair_indices(pair_no)
        ), f"pair {pair_no}"


@pytest.mark.parametrize("n_streams", [1, 2], ids=["time-resolved", "pre-paired"])
def test_set_refuses_start_index_zero_by_name(tmp_path, clean_set_cache, n_streams):
    frame_stride = 1 if n_streams == 1 else 0
    set_file, cfg = _set_config(tmp_path, n_streams, _strides(0, frame_stride, 1))

    with pytest.raises(ValueError, match=r"images\.start_index is 0.*\.set"):
        cfg.get_frame_pair_indices(1)
    with pytest.raises(ValueError, match=r"images\.start_index is 0.*\.set"):
        load_images.read_pair(1, set_file, 1, cfg)
    with pytest.raises(ValueError, match=r"images\.start_index is 0.*\.set"):
        cfg.num_frame_pairs


def test_cine_refuses_start_index_zero_by_name(tmp_path, cine_stand_in):
    cfg = _cine_config(tmp_path, _strides(0, 1, 1))

    with pytest.raises(ValueError, match=r"images\.start_index is 0.*\.cine"):
        cfg.get_frame_pair_indices(1)
    with pytest.raises(ValueError, match=r"images\.start_index is 0.*\.cine"):
        load_images.read_pair(1, tmp_path, 1, cfg)


def test_cine_refuses_pre_paired_strides_by_name(tmp_path, cine_stand_in):
    cfg = _cine_config(tmp_path, _strides(1, 0, 1))

    with pytest.raises(ValueError, match=r"images\.frame_stride is 0.*\.cine"):
        cfg.get_frame_pair_indices(1)
    with pytest.raises(ValueError, match=r"images\.frame_stride is 0.*\.cine"):
        cfg.num_frame_pairs


@pytest.mark.parametrize(
    ("strides", "key"),
    [
        (_strides(1, 1, 0), "pair_stride is 0"),
        (_strides(1, -1, 1), "frame_stride is -1"),
    ],
    ids=["pair_stride 0", "negative frame_stride"],
)
def test_impossible_strides_are_refused_by_name(tmp_path, strides, key):
    images = {"image_format": ["B%05d.tif"], "image_type": "standard", **strides}
    cfg = _config(tmp_path, tmp_path, images)

    with pytest.raises(ValueError, match=rf"images\.{key}"):
        cfg.get_frame_pair_indices(1)
    with pytest.raises(ValueError, match=rf"images\.{key}"):
        cfg.num_frame_pairs


@pytest.mark.parametrize(
    ("strides", "pairs"),
    [
        # 10 entries, A at 1, 3, 5, 7, 9.
        (_strides(1, 0, 2), 5),
        # 10 entries, A at 1, 4, 7, 10.
        (_strides(1, 0, 3), 4),
        (_strides(1, 0, 1), 10),
        # A at 1, 3, 5, 7 with B two later; A = 9 would need frame 11.
        (_strides(1, 2, 2), 4),
    ],
    ids=["pre-paired every 2nd", "pre-paired every 3rd", "pre-paired", "gap 2 step 2"],
)
def test_pair_count_honours_pair_stride(tmp_path, strides, pairs):
    images = {"image_format": ["B%05d.tif"], "image_type": "standard", **strides}
    cfg = _config(tmp_path, tmp_path, images)

    assert cfg.num_frame_pairs == pairs
