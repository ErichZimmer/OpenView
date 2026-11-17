"""POST /backend/filter builds its dask chunks from the batch's own shape (B7).

The preview reads one batch and needs no detected shape: chunking from the
batch itself cannot disagree with the frames, whatever camera they come from.
The route is driven for real; only the readers, the filter and the worker
thread are replaced.
"""

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest
from flask import Flask

from pivtools_core.config import Config

FRAME_SHAPE = (8, 10)


def test_filter_preview_chunks_ignore_detected_image_shape(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The batch reaches the filter whole even when ``image_shape_for`` is not usable."""
    from fixtures.diagnosis import write_config

    import pivtools_gui.app as gui_app

    write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    seen: List[tuple] = []

    def fake_apply(batch: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        seen.append(batch.shape)
        return batch

    class SyncThread:
        """Runs the target synchronously so the worker's outcome is observable."""

        def __init__(self, target: Any, **kwargs: Any) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    def forbidden(self: Config, camera: int) -> tuple:
        raise AssertionError("image_shape_for must not be consulted")

    monkeypatch.setattr(
        gui_app, "read_pair", lambda *a, **k: np.zeros((2, *FRAME_SHAPE), np.float32)
    )
    monkeypatch.setattr(gui_app, "load_mask_for_camera", lambda *a, **k: None)
    monkeypatch.setattr(gui_app, "apply_filters_to_batch", fake_apply)
    # Replace the module reference, not threading.Thread itself, which the
    # route's ThreadPoolExecutor also uses.
    monkeypatch.setattr(gui_app, "threading", SimpleNamespace(Thread=SyncThread))
    monkeypatch.setattr(Config, "image_shape_for", forbidden)

    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    response = app.test_client().post(
        "/backend/filter", json={"camera": 1, "start_idx": 1}
    )

    assert response.status_code == 200
    assert seen, "apply_filters_to_batch was never reached"
    assert seen[0][1:] == (2, *FRAME_SHAPE)


def _client(tmp_path: Any, monkeypatch: Any) -> Any:
    from fixtures.diagnosis import write_config

    import pivtools_gui.app as gui_app

    write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    return app.test_client()


@pytest.mark.parametrize("idx", [3, -1, "0"])
def test_filter_rejects_out_of_range_source_idx(
    tmp_path: Any, monkeypatch: Any, idx: Any
) -> None:
    """A bad index is a 400 before any worker thread starts (was a silent 200)."""
    import pivtools_gui.app as gui_app

    class NoThread:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise AssertionError("thread started")

    monkeypatch.setattr(gui_app, "threading", SimpleNamespace(Thread=NoThread))
    response = _client(tmp_path, monkeypatch).post(
        "/backend/filter", json={"camera": 1, "start_idx": 1, "source_path_idx": idx}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == (
        f"source_path_idx {idx} is out of range: 1 source paths configured"
    )


@pytest.mark.parametrize("idx", [-1, 3])
def test_filter_single_frame_rejects_out_of_range_source_idx(
    tmp_path: Any, monkeypatch: Any, idx: int
) -> None:
    """Negative (silently another source today) and too-large indices are a 400 before any read."""
    import pivtools_gui.app as gui_app

    def no_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("read attempted")

    monkeypatch.setattr(gui_app, "read_pair", no_read)
    monkeypatch.setattr(gui_app, "_resolve_loop_read", no_read)
    response = _client(tmp_path, monkeypatch).post(
        "/backend/filter_single_frame",
        json={"camera": 1, "frame_idx": 1, "filters": [], "source_path_idx": idx},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == (
        f"source_path_idx {idx} is out of range: 1 source paths configured"
    )
