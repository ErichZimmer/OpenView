"""Scattered data must never reuse a Dask key from an earlier scatter.

The drivers loop over path sets and cameras on one client, scattering the
correlator cache, config and masks once per iteration. When one iteration's
futures die, the client queues their release on its batched stream (10 ms
batches), while the next iteration's scatter travels as its own RPC. Nothing
orders the two. If the scattered data reused a key (a dict scattered under its
own keys "W", "H", ..., or equal content under its content hash), a release
that landed after the new scatter made the scheduler forget the new data, and
the next iteration's tasks failed with "lost dependencies" (about 1 run in 3
of the two-camera C1 test).

These tests slow the client's batched stream so the late release is certain,
which makes the old failure deterministic.

Usage:
    pytest unit-tests/test_scatter_unique_names.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_per_camera_image_shape import CAM_SHAPES, _make_config

# The client normally batches messages to the scheduler every 10 ms. At 300 ms
# a release queued just before the next scatter always reaches the scheduler
# after that scatter's RPC, the order that lost data.
SLOW_BATCHED_STREAM_S = 0.3
RESULT_TIMEOUT_S = 10


@pytest.fixture
def slow_release_client():
    """One in-process worker; client releases always arrive late."""
    from dask.distributed import Client

    from pivtools_cli.piv_cluster.cluster import LOOPBACK_ANY_PORT

    client = Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=LOOPBACK_ANY_PORT,
        worker_dashboard_address=LOOPBACK_ANY_PORT,
    )
    client.scheduler_comm.interval = SLOW_BATCHED_STREAM_S
    try:
        yield client
    finally:
        client.close()


def _cache_shape(cache: dict) -> tuple:
    return int(cache["H"]), int(cache["W"])


def test_next_cameras_scatter_survives_the_previous_cameras_release(
    slow_release_client, tmp_path
):
    """Camera 1's scattered data is dropped as its driver returns, then camera
    2 is scattered and used at once: camera 2's cache must be on the worker,
    with camera 2's shape."""
    from pivtools_cli.processing.dask_pipeline import scatter_immutable_data

    client = slow_release_client
    cfg = _make_config(tmp_path)
    first = scatter_immutable_data(client, cfg, 1)
    assert (
        client.submit(_cache_shape, first["cache"], pure=False).result(
            timeout=RESULT_TIMEOUT_S
        )
        == CAM_SHAPES[1]
    )
    del first

    second = scatter_immutable_data(client, cfg, 2)
    assert (
        client.submit(_cache_shape, second["cache"], pure=False).result(
            timeout=RESULT_TIMEOUT_S
        )
        == CAM_SHAPES[2]
    )


def test_equal_content_survives_the_release_of_its_earlier_copy(slow_release_client):
    """Equal content scattered again (the same config or mask for the next
    camera or path set) used to get the same content-hash key and was lost."""
    from pivtools_cli.processing.unique_scatter import scatter_unique

    client = slow_release_client
    first = scatter_unique(client, np.ones((64, 96)), broadcast=True)
    assert client.submit(np.sum, first, pure=False).result(timeout=RESULT_TIMEOUT_S)
    del first

    second = scatter_unique(client, np.ones((64, 96)), broadcast=True)
    assert (
        client.submit(np.sum, second, pure=False).result(timeout=RESULT_TIMEOUT_S)
        == 64 * 96
    )


@pytest.mark.parametrize(
    "make_data",
    [
        lambda: {"W": np.arange(3), "H": np.arange(4)},
        lambda: [np.zeros((3, 5), bool), np.ones((5, 3), bool)],
        lambda: (np.full(4, 2.0), np.full(4, 3.0)),
        lambda: np.ones((8, 8)),
    ],
    ids=["dict", "list", "tuple", "object"],
)
def test_two_scatters_of_equal_data_share_no_key(make_data):
    """Each call names its data afresh, so no key from an earlier scatter can
    come back; the futures keep the input's shape and the data round-trips."""
    from dask.distributed import Client

    from pivtools_cli.piv_cluster.cluster import LOOPBACK_ANY_PORT
    from pivtools_cli.processing.unique_scatter import scatter_unique

    def keys(futures):
        if isinstance(futures, dict):
            return {f.key for f in futures.values()}
        if isinstance(futures, (list, tuple)):
            return {f.key for f in futures}
        return {futures.key}

    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=LOOPBACK_ANY_PORT,
        worker_dashboard_address=LOOPBACK_ANY_PORT,
    ) as client:
        data = make_data()
        first = scatter_unique(client, data, broadcast=True)
        second = scatter_unique(client, make_data(), broadcast=True)
        assert keys(first).isdisjoint(keys(second))
        if isinstance(data, dict):
            assert set(first) == set(data)
            assert keys(first).isdisjoint(data)
        elif isinstance(data, (list, tuple)):
            assert type(first) is type(data)
        gathered = client.gather(second)
        if isinstance(data, dict):
            for key, value in data.items():
                np.testing.assert_array_equal(gathered[key], value)
        elif isinstance(data, (list, tuple)):
            for got, want in zip(gathered, data, strict=True):
                np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_array_equal(gathered, data)


def test_dict_keys_must_be_strings():
    from unittest.mock import MagicMock

    from pivtools_cli.processing.unique_scatter import scatter_unique

    with pytest.raises(TypeError, match=r"string dict keys"):
        scatter_unique(MagicMock(), {1: np.zeros(2)}, broadcast=True)


PACKAGES = ("pivtools_cli", "pivtools_core", "pivtools_gui")
HELPER = Path("pivtools_cli") / "processing" / "unique_scatter.py"


def test_no_production_module_scatters_directly():
    """Every scatter goes through scatter_unique; a direct client.scatter
    could reuse a key and bring the lost-dependencies race back."""
    import re

    root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"\bclient\.scatter\(")
    offenders = [
        f"{path.relative_to(root)}:{lineno}"
        for package in PACKAGES
        for path in sorted((root / package).rglob("*.py"))
        if path.relative_to(root) != HELPER
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, f"direct client.scatter calls: {offenders}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
