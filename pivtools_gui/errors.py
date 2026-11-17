"""JSON error handlers for the Flask backend.

Every failure under ``/backend/`` must reach the browser as JSON. Without these
handlers an uncaught exception falls through to Werkzeug's HTML 500 page and the
frontend's ``res.json()`` dies with ``SyntaxError: Unexpected token '<'`` — the
real message never reaches the user.

Envelope: ``{"success": false, "error": "<message>"}`` with the real HTTP status.
Both keys are set so callers that test ``res.ok`` and callers that test
``data.success`` behave the same way.

Paths outside ``/backend/`` are left to Flask's defaults so the SPA catch-all in
``app.py`` keeps serving ``index.html``.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from loguru import logger
from werkzeug.exceptions import HTTPException, InternalServerError

from pivtools_core.config import CalibrationSourceError
from pivtools_core.image_handling.readers.pivraw_reader import PivrawError
from pivtools_core.image_handling.readers.set_reader import SetConfigError

API_PREFIX = "/backend/"
# Errors the user fixes in the configuration or the data: answered 400 with
# their own message (register_json_error_handlers).
BAD_INPUT_ERRORS = (CalibrationSourceError, PivrawError, SetConfigError)


def _is_api_request() -> bool:
    return request.path.startswith(API_PREFIX)


def register_json_error_handlers(app: Flask) -> None:
    """Attach the handlers to ``app``.

    ``HTTPException`` is registered first so ``abort()``, 404 and 405 keep their
    status code. An unset or blank calibration source is a configuration error the
    user fixes, so it is a 400 with its plain message. So is a ``.pivraw`` run the
    reader refuses (a run that cannot be paired by trigger, or a pairing that does
    not suit it): the message names the camera and the reason. So is a ``.set``
    camera count or selection the recording cannot split or pair
    (``SetConfigError``, e.g. ``SetPairingError`` for a single-frame camera
    selected for A/B pairs). Everything else
    becomes a 500 whose traceback goes to the loguru log, not to the browser.
    """

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        if not _is_api_request():
            return exc
        return jsonify({"success": False, "error": exc.description}), exc.code

    def _bad_input(exc: Exception):
        if not _is_api_request():
            return InternalServerError(original_exception=exc)
        return jsonify({"success": False, "error": str(exc)}), 400

    for error_type in BAD_INPUT_ERRORS:
        app.register_error_handler(error_type, _bad_input)

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        logger.exception("unhandled exception in {} {}", request.method, request.path)
        if not _is_api_request():
            return InternalServerError(original_exception=exc)
        return (
            jsonify({"success": False, "error": f"{type(exc).__name__}: {exc}"}),
            500,
        )
