"""Progress reporting in ``correlate_worker_batches`` fails loudly, once per call.

A failed Dask progress Variable used to be swallowed in silence. It now warns once
per ``correlate_worker_batches`` call (one ``_BatchProgress``), never per batch, and
never aborts the correlation. The flag lives with the call, not the process: a
worker process that runs a second pass or a second run warns again.
"""

from __future__ import annotations

import logging

import distributed

import pivtools_cli.processing.dask_pipeline as DP


class _FailingVariable:
    def __init__(self, err):
        self.err = err

    def set(self, value):
        raise self.err


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_no_client_warns_once_per_call(monkeypatch, caplog):
    def no_client():
        raise ValueError("No clients found")

    monkeypatch.setattr(distributed, "get_client", no_client)
    with caplog.at_level(logging.WARNING, logger=DP.logger.name):
        for _call in range(2):
            progress = DP._BatchProgress("progress-var")
            for batch in range(3):
                progress.set(batch + 1)

    messages = _messages(caplog)
    assert len(messages) == 2
    assert all("No clients found" in m for m in messages)


def test_failed_update_warns_once_per_call(monkeypatch, caplog):
    monkeypatch.setattr(distributed, "get_client", object)
    monkeypatch.setattr(
        distributed,
        "Variable",
        lambda name, client: _FailingVariable(OSError("comm closed")),
    )
    with caplog.at_level(logging.WARNING, logger=DP.logger.name):
        for _call in range(2):
            progress = DP._BatchProgress("progress-var")
            for batch in range(5):
                progress.set(batch + 1)

    messages = _messages(caplog)
    assert len(messages) == 2
    assert all("comm closed" in m for m in messages)


def test_no_variable_name_reports_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=DP.logger.name):
        progress = DP._BatchProgress(None)
        progress.set(1)
    assert _messages(caplog) == []
