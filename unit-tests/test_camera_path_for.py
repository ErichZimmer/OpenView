"""One camera-path rule for every reader of PIV images (14 Nov review, fault 1).

An ``.im7`` recording laid out one folder per camera (``src/Cam1/B00001.im7``,
``src/Cam2/B00001.im7``, ``images.use_camera_subfolders: true``) was read
correctly by the pipeline, and ``/validate_files`` was green, but the GUI looked
for camera 2 in the source root. ``GET /get_frame_pair`` answered 404,
``/filter_single_frame`` 500, and the preload cached nothing without a word.
The cause was four copies of an "im7 never has a subfolder" special case beside
the one rule (``Config.get_camera_folder``) that already knew better.

These tests drive the real routes on a real two-camera subfolder ``.im7`` set,
and compare what they serve with what the pipeline (``load_images``) reads.
"""

import base64
import io

import numpy as np
import pytest
from fixtures.diagnosis import write_config
from fixtures.lavision import write_dark_im7
from flask import Flask
from loguru import logger
from PIL import Image

import pivtools_gui.app as gui_app
from pivtools_core.image_handling.load_images import load_images
from pivtools_core.validation import validate_config

# Large enough for the display window's tile analysis (8 tiles of at least
# 3 px), so the ramp reads through the ordinary display path.
FRAME_SHAPE = (24, 32)
N_FRAMES = 3


def _frame(camera: int, frame: int) -> np.ndarray:
    """Recorded frame ``frame`` of ``camera``: a ramp offset by 1000*camera + 10*frame.

    Every (camera, frame) has its own value range, so a read from the wrong
    folder or the wrong frame cannot pass for the right one.
    """
    ramp = np.arange(FRAME_SHAPE[0] * FRAME_SHAPE[1]).reshape(FRAME_SHAPE)
    return (ramp + 1000 * camera + 10 * frame).astype(np.uint16)


def _subfolder_im7(tmp_path, use_camera_subfolders=True):
    """Two cameras, one single-frame ``.im7`` per instant in ``src/Cam{N}``."""
    src = tmp_path / "src"
    for camera in (1, 2):
        (src / f"Cam{camera}").mkdir(parents=True)
        for frame in range(1, N_FRAMES + 1):
            write_dark_im7(
                src / f"Cam{camera}" / f"B{frame:05d}.im7", _frame(camera, frame)
            )
    return write_config(
        tmp_path,
        paths={
            "source_paths": [str(src)],
            "base_paths": [str(tmp_path / "out")],
            "camera_count": 2,
            "camera_numbers": [1, 2],
        },
        images={
            "image_format": ["B%05d.im7"],
            "image_type": "lavision_im7",
            "use_camera_subfolders": use_camera_subfolders,
            "frame_stride": 1,
            "pair_stride": 1,
            "start_index": 1,
            "num_images": N_FRAMES,
        },
    )


@pytest.fixture
def subfolder_set(tmp_path, monkeypatch):
    """The subfolder set, with the GUI routes bound to its config.

    ``/filter_single_frame`` builds a fresh ``Config()`` from the working
    directory, hence the chdir.
    """
    cfg = _subfolder_im7(tmp_path)
    monkeypatch.setattr(gui_app, "get_config", lambda: cfg)
    monkeypatch.chdir(tmp_path)
    gui_app.raw_image_cache.clear()
    gui_app.FRAME_1_KEYS.clear()
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    yield cfg, app.test_client()
    gui_app.raw_image_cache.clear()
    gui_app.FRAME_1_KEYS.clear()


def _png_pixels(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def test_pipeline_reads_camera_2_from_its_subfolder(subfolder_set):
    """The reference the GUI is compared against: pair 1 of camera 2 is frames 1, 2."""
    cfg, _ = subfolder_set
    pair = load_images(2, cfg)[0].compute()
    np.testing.assert_array_equal(pair[0], _frame(2, 1))
    np.testing.assert_array_equal(pair[1], _frame(2, 2))


def test_get_frame_pair_serves_camera_2_as_the_pipeline_reads_it(subfolder_set):
    cfg, client = subfolder_set
    response = client.get("/backend/get_frame_pair?camera=2&idx=1&format=png")
    assert response.status_code == 200, response.get_json()
    body = response.get_json()

    pair = load_images(2, cfg)[0].compute()
    for role, frame in (("A", pair[0]), ("B", pair[1])):
        stats = body["stats"][role]
        assert (stats["data_min"], stats["data_max"]) == (frame.min(), frame.max())
        expected_png, _ = gui_app.encode_display_frame(frame, "png", None)
        np.testing.assert_array_equal(
            _png_pixels(body[role]), _png_pixels(expected_png)
        )


@pytest.fixture
def warnings_logged():
    """WARNING-or-above messages the GUI routes module logs while the test runs.

    Only ``pivtools_gui.app``: the display window (``pivtools_gui.utils``)
    rightly warns that a particle-free ramp has no particle peaks.
    """
    messages = []
    handler = logger.add(
        lambda m: messages.append(m.record["message"]),
        level="WARNING",
        filter="pivtools_gui.app",
    )
    yield messages
    logger.remove(handler)


def test_preload_caches_camera_2(subfolder_set, warnings_logged):
    cfg, _ = subfolder_set
    gui_app._preload_surrounding_frames(0, 2, 1, cfg, 10, "png")
    cached = {key[2] for key in gui_app.raw_image_cache if key[1] == 2}
    assert cached == {1, 2}
    assert warnings_logged == []


def test_preload_failure_is_a_warning_naming_camera_pairs_and_reason(
    subfolder_set, warnings_logged
):
    """A pair that will not read is reported, not dropped at debug level."""
    cfg, _ = subfolder_set
    missing = cfg.source_paths[0] / "Cam2" / "B00003.im7"
    missing.unlink()

    gui_app._preload_surrounding_frames(0, 2, 1, cfg, 10, "png")

    assert {key[2] for key in gui_app.raw_image_cache} == {1}
    assert len(warnings_logged) == 1
    message = warnings_logged[0]
    assert "camera 2" in message
    assert "1 of 2" in message
    assert "pair 2" in message
    assert str(missing) in message


def test_cli_preflight_validates_camera_subfolder_im7(tmp_path):
    """``validate_config`` (run before every CLI job) finds each camera's files."""
    cfg = _subfolder_im7(tmp_path)
    valid, error, _ = validate_config(cfg)
    assert valid, error


def test_raw_cache_key_changes_with_the_camera_subfolder_setting(tmp_path):
    """Turning ``use_camera_subfolders`` off reads another file, so another key.

    CLAUDE.md gotcha 6. With the old im7 special case both settings keyed on
    the source root, and the cache would serve the subfolder frame after the
    switch.
    """
    cfg = _subfolder_im7(tmp_path)
    keys = []
    processed_keys = []
    for flag in (True, False):
        cfg.data["images"]["use_camera_subfolders"] = flag
        keys.append(gui_app.make_raw_cache_key(0, 2, 1, "png", None, cfg))
        processed_keys.append(gui_app.cache_key(0, 2, cfg))
    assert keys[0] != keys[1]
    assert processed_keys[0] != processed_keys[1]


def test_filter_single_frame_reads_camera_2(subfolder_set):
    _, client = subfolder_set
    response = client.post(
        "/backend/filter_single_frame",
        json={"camera": 2, "frame_idx": 1, "filters": []},
    )
    assert response.status_code == 200, response.get_json()
    stats = response.get_json()["stats"]
    assert (stats["A"]["data_min"], stats["A"]["data_max"]) == (2010, 2777)
    assert (stats["B"]["data_min"], stats["B"]["data_max"]) == (2020, 2787)
