"""POST /backend/filter builds its dask chunks from the batch's own shape (B7).

``cfg.image_shape`` is detected once from the first selected camera, so a camera
whose frames differ in size must not be chunked with it. The route is driven for
real; only the readers, the filter and the worker thread are replaced.
"""

from types import SimpleNamespace
from typing import Any, List

import numpy as np
from flask import Flask

from pivtools_core.config import Config

FRAME_SHAPE = (8, 10)


def test_filter_preview_chunks_ignore_cfg_image_shape(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The batch reaches the filter whole even when ``cfg.image_shape`` is not usable."""
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

    def forbidden(self: Config) -> tuple:
        raise AssertionError("image_shape must not be consulted")

    monkeypatch.setattr(
        gui_app, "read_pair", lambda *a, **k: np.zeros((2, *FRAME_SHAPE), np.float32)
    )
    monkeypatch.setattr(gui_app, "load_mask_for_camera", lambda *a, **k: None)
    monkeypatch.setattr(gui_app, "apply_filters_to_batch", fake_apply)
    # Replace the module reference, not threading.Thread itself, which the
    # route's ThreadPoolExecutor also uses.
    monkeypatch.setattr(gui_app, "threading", SimpleNamespace(Thread=SyncThread))
    monkeypatch.setattr(Config, "image_shape", property(forbidden))

    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    response = app.test_client().post(
        "/backend/filter", json={"camera": 1, "start_idx": 1}
    )

    assert response.status_code == 200
    assert seen, "apply_filters_to_batch was never reached"
    assert seen[0][1:] == (2, *FRAME_SHAPE)
