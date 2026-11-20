"""The viewer's plot routes require ``b_mask`` on the pass they draw.

Every writer of a plotted result emits a ``b_mask`` field on every pass element: the
PIV writer (``save_results``, instantaneous and ensemble), calibration apply and
stereo reconstruction (``calibration/runio.py``), merging (``vector_merger``), mean
statistics and per-frame statistics (``instantaneous_statistics``, which copies the
frame file's fields). Two writers can leave it empty on a populated pass, when their
own input already had none: calibration apply and an ensemble resume. A missing,
non-boolean, empty or wrongly shaped ``b_mask`` therefore means a stale or broken
file, to be regenerated. It used to be replaced silently by an all-False mask,
drawing masked vectors as data; now it raises ``ValueError`` naming the file.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from pivtools_gui.plotting.app.shared_utils import b_mask_for, load_and_plot_data

GRID = (3, 4)


def _write_piv_result(path, fields):
    struct = np.empty((1,), dtype=[(f, object) for f in fields])
    for f, value in fields.items():
        struct[f][0] = value
    scipy.io.savemat(str(path), {"piv_result": struct}, oned_as="row")
    return path


def _load_element(path):
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    return mat["piv_result"]


def test_file_without_b_mask_raises_naming_it(tmp_path):
    path = _write_piv_result(
        tmp_path / "00001.mat", {"ux": np.ones(GRID), "uy": np.ones(GRID)}
    )
    with pytest.raises(ValueError, match=r"00001\.mat.*b_mask"):
        load_and_plot_data(
            mat_path=path, coords_path=None, var="ux", run=1, save_basepath=tmp_path
        )


def test_empty_b_mask_on_populated_pass_asks_for_regeneration(tmp_path):
    """An empty b_mask on a populated pass cannot mask it; the file must be rewritten.

    Calibration apply and an ensemble resume copy an empty source b_mask through,
    so the message names the file, says the mask is empty, and says to regenerate.
    """
    path = _write_piv_result(
        tmp_path / "00001.mat",
        {"ux": np.ones(GRID), "uy": np.ones(GRID), "b_mask": np.empty((0, 0))},
    )
    pr = _load_element(path)
    with pytest.raises(ValueError, match=r"00001\.mat.*empty b_mask.*regenerate"):
        b_mask_for(pr, np.asarray(pr.ux), path)


def test_mismatched_b_mask_asks_for_regeneration(tmp_path):
    path = tmp_path / "00001.mat"
    pr = type("Element", (), {"b_mask": np.zeros((2, 2), dtype=bool)})()
    with pytest.raises(
        ValueError, match=r"00001\.mat.*b_mask shape \(2, 2\).*\(3, 4\).*regenerate"
    ):
        b_mask_for(pr, np.ones(GRID), path)


def test_non_boolean_b_mask_raises_naming_it(tmp_path):
    path = tmp_path / "00001.mat"
    pr = type("Element", (), {"b_mask": np.full(GRID, "x")})()
    with pytest.raises(ValueError, match=r"00001\.mat.*b_mask"):
        b_mask_for(pr, np.ones(GRID), path)


def test_b_mask_is_returned_as_bool(tmp_path):
    stored = np.zeros(GRID, dtype=np.uint8)
    stored[0, 1] = 1
    path = _write_piv_result(
        tmp_path / "00001.mat",
        {"ux": np.ones(GRID), "uy": np.ones(GRID), "b_mask": stored},
    )
    pr = _load_element(path)
    mask = b_mask_for(pr, np.asarray(pr.ux), path)
    assert mask.dtype == bool
    assert mask.tolist() == stored.astype(bool).tolist()
