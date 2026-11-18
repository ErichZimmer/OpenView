"""Scatter data to Dask workers under keys no earlier scatter has used.

The PIV drivers loop over path sets and cameras on one client and scatter
their immutable data (correlator cache, config, masks, mean images,
accumulator slices) once per iteration. When an iteration's futures die, the
client queues their release on its batched stream, while the next scatter is
a separate RPC, and nothing orders the two. Had the new scatter reused a key,
a release that arrived after it made the scheduler forget the new data and the
iteration's tasks failed with "lost dependencies". ``client.scatter`` reuses
keys in two ways at distributed 2025.7 (``Client._scatter``): a dict is
scattered under its own keys verbatim, and anything else, with the default
``hash=True``, under a hash of its content.

``scatter_unique`` names every scattered value with a fresh ``uuid4``, so a key
from an earlier scatter can never come back.
"""

import uuid
from typing import Any

from dask.distributed import Client


def scatter_unique(client: Client, data: Any, *, broadcast: bool) -> Any:
    """Scatter ``data`` under fresh keys; return futures in the input's shape.

    A dict is scattered value by value as ``"<key>-<uuid4 hex>"``, one uuid per
    call, and the futures come back under the dict's original keys. Any other
    input is scattered with ``hash=False``, for which distributed names each
    value ``"<type>-<uuid4 hex>"``. A list or tuple returns a list or tuple of
    futures; a single object returns one future.

    Args:
        client: Dask distributed client.
        data: A dict with string keys, a list or tuple, or a single object.
        broadcast: Copy the data to every worker (True) or round-robin it.

    Raises:
        TypeError: a dict key is not a string, so it cannot carry the suffix.
    """
    if isinstance(data, dict):
        non_str = [key for key in data if not isinstance(key, str)]
        if non_str:
            raise TypeError(
                f"scatter_unique needs string dict keys to make unique Dask "
                f"keys; got {non_str!r}"
            )
        tag = uuid.uuid4().hex
        futures = client.scatter(
            {f"{key}-{tag}": value for key, value in data.items()},
            broadcast=broadcast,
        )
        return {key: futures[f"{key}-{tag}"] for key in data}
    return client.scatter(data, broadcast=broadcast, hash=False)
