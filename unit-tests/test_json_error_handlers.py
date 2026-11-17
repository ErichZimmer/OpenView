"""``pivtools_gui.errors`` — every failure under ``/backend/`` is JSON.

Bare Flask app with throwaway routes; no config, no real blueprints. Pins the
contract the frontend's ``fetchJson`` relies on: real status code, JSON body with
``success: false`` and ``error``, and Flask's default (HTML) behaviour preserved
outside the API prefix so the SPA catch-all is unaffected.
"""

from __future__ import annotations

import pytest
from flask import Flask, abort

from pivtools_core.image_handling.readers.set_reader import SetPairingError
from pivtools_gui.errors import register_json_error_handlers

_MISSING = r"E:\bailey_calibration\Calibration\Cam3\cal_001.tif"
# The reader's text for a single-frame camera selected for A/B pairing (set_reader.py).
_SINGLE_FRAME = (
    "Camera 3 of rec.set, Camera 3 (Imager sCMOS CLHS, Camera.ImagerSCMOS5MCLHS: "
    "61008639, single frame) on stream 4, is a single-frame camera, so it has no "
    "A/B pairs. Deselect camera 3 in the PIV camera selection, or process it "
    "time-resolved."
)


@pytest.fixture()
def client():
    app = Flask(__name__)
    register_json_error_handlers(app)

    @app.route("/backend/boom")
    def _boom():
        raise FileNotFoundError(f"Image file not found: {_MISSING}")

    @app.route("/backend/gone")
    def _gone():
        abort(404, description="job not found")

    @app.route("/backend/single_frame_pair")
    def _single_frame_pair():
        raise SetPairingError(_SINGLE_FRAME)

    @app.route("/spa/boom")
    def _spa_boom():
        raise RuntimeError("outside the api")

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def _spa_catch_all(path):
        # Mirrors ``serve_react_app`` in app.py: the catch-all must refuse
        # unknown API paths, or they come back as index.html at HTTP 200.
        if path.startswith("backend/"):
            abort(404, description=f"no such backend route: /{path}")
        return "<html>index</html>"

    return app.test_client()


def test_unhandled_exception_is_json_500(client):
    res = client.get("/backend/boom")
    assert res.status_code == 500
    assert res.is_json
    body = res.get_json()
    assert body["success"] is False
    assert body["error"] == f"FileNotFoundError: Image file not found: {_MISSING}"


def test_http_exception_keeps_status_and_is_json(client):
    res = client.get("/backend/gone")
    assert res.status_code == 404
    assert res.is_json
    assert res.get_json() == {"success": False, "error": "job not found"}


def test_unknown_api_route_is_json_404(client):
    res = client.get("/backend/no_such_route")
    assert res.status_code == 404
    assert res.is_json
    assert res.get_json() == {
        "success": False,
        "error": "no such backend route: /backend/no_such_route",
    }


def test_a_set_pairing_refusal_is_a_400_with_its_own_text(client):
    """A camera selection the recording cannot pair is the user's to fix, not a crash."""
    res = client.get("/backend/single_frame_pair")
    assert res.status_code == 400
    assert res.get_json() == {"success": False, "error": _SINGLE_FRAME}


def test_paths_outside_api_keep_flask_defaults(client):
    res = client.get("/spa/boom")
    assert res.status_code == 500
    assert not res.is_json

    res = client.get("/spa/missing")
    assert res.status_code == 200
    assert not res.is_json
    assert b"index" in res.data
