"""The runners' Dask teardown hides Tornado chatter but never hides a failure.

``shutdown_cluster`` silences stderr while the client and cluster close. A close
that raises an ``Exception`` is kept and printed once stderr is back, and the
teardown carries on. A ``BaseException`` (``asyncio.CancelledError`` when a run is
cancelled, ``KeyboardInterrupt``) propagates, and stderr must be restored first,
or its traceback would be written into the discarded buffer.
"""

import asyncio
import sys

import pytest

from pivtools_cli.piv_cluster.cluster import shutdown_cluster


class _Closable:
    """A client or cluster stand-in whose ``close`` records the call and may raise."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.raises = raises
        self.closed = False

    def close(self, timeout: float) -> None:
        self.closed = True
        if self.raises is not None:
            raise self.raises


def test_failed_close_is_printed_after_stderr_is_restored(capsys):
    """A close that raises is reported on stdout and the other close still runs."""
    stderr_before = sys.stderr
    client = _Closable(raises=RuntimeError("scheduler gone"))
    cluster = _Closable()

    shutdown_cluster(client, cluster)

    assert sys.stderr is stderr_before
    assert client.closed and cluster.closed
    out = capsys.readouterr().out
    assert "client.close failed" in out
    assert "scheduler gone" in out


def test_cancelled_close_restores_stderr_and_propagates():
    """A CancelledError leaves the teardown with stderr pointing where it did before."""
    stderr_before = sys.stderr
    client = _Closable(raises=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        shutdown_cluster(client, _Closable())

    assert sys.stderr is stderr_before


def test_clean_close_prints_nothing(capsys):
    """Both objects close, nothing is printed, and ``None`` handles are skipped."""
    client = _Closable()
    shutdown_cluster(client, None)

    assert client.closed
    assert capsys.readouterr().out == ""
