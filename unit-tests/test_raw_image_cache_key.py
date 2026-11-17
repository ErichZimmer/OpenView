"""The GUI's raw image cache key carries every setting that changes the image served.

CLAUDE.md gotcha 6: a prefetch cache keyed without a render-relevant setting
serves a stale image. The pairing settings decide which recorded frames pair
``idx`` reads (time_resolved pair 2 is frames 2 and 3, skip_frames pair 2 is
frames 3 and 4), so a key without them serves the old preset's frames.
"""

import pytest
import yaml

from pivtools_core.config import Config
from pivtools_gui.app import make_raw_cache_key


def _config(tmp_path) -> Config:
    cfg = {
        "paths": {
            "source_paths": [str(tmp_path / "src")],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {
            "image_type": "standard",
            "image_format": ["B%05d.tif"],
            "num_images": 10,
            "start_index": 1,
            "pairing_preset": "time_resolved",
            "frame_stride": 1,
            "pair_stride": 1,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    return Config(path=str(path))


def _key(cfg: Config) -> tuple:
    return make_raw_cache_key(0, 1, 2, "jpeg", None, cfg)


@pytest.mark.parametrize(
    "change",
    [
        {"pairing_preset": "skip_frames", "pair_stride": 2},
        {"frame_stride": 2},
        {"pair_stride": 3},
        {"start_index": 0},
        {"num_images": 20},
        {"image_format": ["C%05d.tif"]},
    ],
)
def test_a_changed_images_setting_gives_another_key(tmp_path, change):
    cfg = _config(tmp_path)
    before = _key(cfg)
    cfg.data["images"].update(change)
    assert _key(cfg) != before


def test_the_same_settings_give_the_same_key(tmp_path):
    cfg = _config(tmp_path)
    assert _key(cfg) == _key(cfg)
