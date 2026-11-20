"""Progress reporting in ``correlate_worker_batches`` fails loudly, but only once.

A failed Dask progress Variable used to be swallowed in silence. It now warns,
once per process and stage, never per batch, and never aborts the correlation.
"""

from __future__ import annotations

import logging

import pivtools_cli.processing.dask_pipeline as DP


def test_each_stage_warns_once_per_process(monkeypatch, caplog):
    monkeypatch.setattr(DP, "_progress_warned", {"create": False, "update": False})
    with caplog.at_level(logging.WARNING, logger=DP.logger.name):
        for _ in range(3):
            DP._warn_progress_once("create", ValueError("No clients found"))
        for _ in range(5):
            DP._warn_progress_once("update", OSError("comm closed"))

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "No clients found" in messages[0]
    assert "comm closed" in messages[1]
