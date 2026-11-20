"""Calibration 3D figures draw every camera, however many the rig has.

``_camera_records`` used to zip the cameras against the fixed ``CAM_COLORS``
palette, so a ninth camera was silently left out of the rig figures.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pivtools_gui.calibration.figures import CAM_COLORS, _camera_records


def test_more_cameras_than_colours_reuses_colours():
    n = len(CAM_COLORS) + 1
    models = [
        SimpleNamespace(R=np.eye(3), t=np.array([10.0 * i, 0.0, 1000.0]))
        for i in range(n)
    ]
    labels = [f"Cam{i + 1}" for i in range(n)]

    recs = _camera_records(models, labels, np.zeros((4, 3)))

    assert [r["label"] for r in recs] == labels
    assert recs[-1]["color"] == CAM_COLORS[0]
