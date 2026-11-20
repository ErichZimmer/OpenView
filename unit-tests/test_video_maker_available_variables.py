"""``/backend/video/available_variables`` reports an unreadable result file.

The video maker's variable dropdown is filled from the first frame file. A file that
exists but cannot be read raises ``ResultFileError`` in the shared lister; the route
answers 400 with the message naming the file, so the GUI can show it. It used to fall
into the route's catch-all and answer 500.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from flask import Flask

import pivtools_gui.video_maker.app.views as V
from pivtools_core.paths import get_data_paths

NUM_FRAME_PAIRS = 3
VECTOR_FORMAT = "%05d.mat"


def _client():
    app = Flask(__name__)
    app.register_blueprint(V.video_maker_bp, url_prefix="/backend/video")
    return app.test_client()


def test_unreadable_frame_file_answers_400_naming_it(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        base_paths=[str(tmp_path)],
        num_frame_pairs=NUM_FRAME_PAIRS,
        vector_format=VECTOR_FORMAT,
    )
    monkeypatch.setattr(V, "get_config", lambda refresh=False: cfg)
    data_dir = Path(
        get_data_paths(
            base_dir=tmp_path,
            num_frame_pairs=NUM_FRAME_PAIRS,
            cam=1,
            type_name="instantaneous",
        )["data_dir"]
    )
    data_dir.mkdir(parents=True)
    bad = data_dir / (VECTOR_FORMAT % 1)
    bad.write_bytes(b"not a mat file")

    response = _client().get(
        "/backend/video/available_variables",
        query_string={"base_path": str(tmp_path), "camera": "1"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert str(bad) in body["error"]
