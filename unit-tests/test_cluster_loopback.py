"""Local Dask clusters must listen on loopback only.

An all-interface listener raises a Windows Firewall prompt for every process that
opens one, and a standard user cannot answer it. A test run on 2026-11-14 left 57
such prompts on a workstation. These tests fail if a local cluster, built the way
production or the unit tests build it, opens a listening socket off loopback.
"""

import os
from types import SimpleNamespace

import psutil
from dask.distributed import Client

from pivtools_cli.piv_cluster.cluster import LOOPBACK_ANY_PORT, make_cluster

LOOPBACK_HOSTS = {"127.0.0.1", "::1"}


def _non_loopback_listeners() -> set[str]:
    """Return the local IPs, other than loopback, that this process tree listens on."""
    me = psutil.Process(os.getpid())
    found: set[str] = set()
    for proc in [me, *me.children(recursive=True)]:
        for conn in proc.net_connections(kind="inet"):
            if (
                conn.status == psutil.CONN_LISTEN
                and conn.laddr.ip not in LOOPBACK_HOSTS
            ):
                found.add(conn.laddr.ip)
    return found


def test_make_cluster_listens_on_loopback_only():
    """The production local cluster opens no listener off loopback."""
    before = _non_loopback_listeners()
    config = SimpleNamespace(
        cluster_type="local",
        dask_workers_per_node=1,
        dask_memory_limit="1GB",
        dask_nanny=False,
    )
    cluster, client = make_cluster(config)
    try:
        assert _non_loopback_listeners() - before == set()
    finally:
        client.close()
        cluster.close()


def test_in_process_client_listens_on_loopback_only():
    """The in-process client the unit tests use opens no listener off loopback."""
    before = _non_loopback_listeners()
    client = Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=LOOPBACK_ANY_PORT,
        worker_dashboard_address=LOOPBACK_ANY_PORT,
    )
    try:
        assert _non_loopback_listeners() - before == set()
    finally:
        client.close()
