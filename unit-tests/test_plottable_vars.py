"""The one variable-list helper behind the viewer and video-maker dropdowns.

``pivtools_core.vector_loading.get_plottable_vars`` replaced three near-identical
copies (``check_vars`` and ``_extract_plottable_vars`` in the plotting views, and
``_extract_plottable_vars`` in the video maker). The copies fell back silently to an
empty list when a file could not be read, so a dropdown went blank with no reason.
These tests pin the contract on files laid out the way ``save_results`` writes them:
one struct element per pass, unsaved passes holding empty arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from pivtools_core.vector_loading import ResultFileError, get_plottable_vars

GRID = (3, 4)
N_PEAKS = 2


def _write_passes(path, n_passes, populated, peak_stack=False):
    """Write a piv_result with ``n_passes`` elements; only ``populated`` hold data."""
    fields = ["ux", "uy", "b_mask", "win_ctrs_x"]
    if peak_stack:
        fields += ["peak_mag", "peak_choice"]
    struct = np.empty((n_passes,), dtype=[(f, object) for f in fields])
    for i in range(n_passes):
        for f in fields:
            struct[f][i] = np.empty((0, 0))
    for i in populated:
        struct["ux"][i] = np.ones(GRID)
        struct["uy"][i] = np.ones(GRID)
        struct["b_mask"][i] = np.zeros(GRID)
        struct["win_ctrs_x"][i] = np.ones(GRID)
        if peak_stack:
            struct["peak_mag"][i] = np.ones((N_PEAKS, *GRID))
            struct["peak_choice"][i] = np.zeros(GRID)
    scipy.io.savemat(str(path), {"piv_result": struct}, oned_as="row")
    return path


def test_lists_2d_fields_of_the_populated_pass(tmp_path):
    """Pass 1 empty (not in runs_to_save), pass 2 populated: names come from pass 2."""
    path = _write_passes(tmp_path / "B00001.mat", 2, populated=[1])
    assert get_plottable_vars(path) == ["b_mask", "ux", "uy"]


def test_single_pass_file_is_read_as_one_struct(tmp_path):
    path = _write_passes(tmp_path / "B00001.mat", 1, populated=[0])
    assert get_plottable_vars(path) == ["b_mask", "ux", "uy"]


def test_peak_stack_listed_only_when_the_caller_can_index_it(tmp_path):
    """A multi-peak ``peak_mag`` is 3D; only the viewer resolves it via peak_choice."""
    path = _write_passes(tmp_path / "B00001.mat", 1, populated=[0], peak_stack=True)
    assert "peak_mag" not in get_plottable_vars(path)
    assert "peak_mag" in get_plottable_vars(path, include_peak_stack=True)


def test_absent_file_has_no_variables(tmp_path):
    assert get_plottable_vars(tmp_path / "B00001.mat") == []


@pytest.mark.parametrize(
    "content",
    # scipy 1.16 raises MatReadError for the first and IndexError for the second.
    [b"not a mat file", b"this is not a mat file"],
)
def test_unreadable_file_raises_naming_it(tmp_path, content):
    path = tmp_path / "B00001.mat"
    path.write_bytes(content)
    with pytest.raises(ResultFileError, match="B00001.mat"):
        get_plottable_vars(path)


def test_file_without_the_struct_raises_naming_it(tmp_path):
    path = tmp_path / "B00001.mat"
    scipy.io.savemat(str(path), {"coordinates": np.zeros(GRID)})
    with pytest.raises(ResultFileError, match=r"B00001\.mat.*piv_result"):
        get_plottable_vars(path)


def test_non_struct_variable_raises_naming_it(tmp_path):
    path = tmp_path / "B00001.mat"
    scipy.io.savemat(str(path), {"piv_result": np.zeros(GRID)})
    with pytest.raises(ResultFileError, match="not a struct"):
        get_plottable_vars(path)


def test_per_frame_statistics_file_lists_its_stats_with_the_default_fields(tmp_path):
    """A per-frame statistics file carries the frame file's ``ux``/``uy`` on every pass.

    The deleted video-maker lister searched ``u_prime``/``uu_inst`` for the first
    populated pass. The real writer (``_process_frame_parallel``) copies every field
    of the frame file and adds the stats on exactly the passes whose ``ux``/``uy`` are
    populated (``valid_runs`` comes from ``find_valid_piv_runs``, which checks
    ``ux``/``uy``), so the default ``("ux", "uy")`` finds the stats pass. Velocity
    saving is off here to show ``ux``/``uy`` are kept regardless.
    """
    from pivtools_gui.vector_statistics.instantaneous_statistics import (
        _process_frame_parallel,
    )

    frame = _write_passes(tmp_path / "00001.mat", 2, populated=[1])
    out = tmp_path / "stats_00001.mat"
    _process_frame_parallel(
        str(frame),
        str(out),
        means_dict={
            2: {"mean_ux": np.zeros(GRID), "mean_uy": np.zeros(GRID), "stereo": False}
        },
        calc_flags={"calc_inst_stresses": True, "save_inst_velocity": False},
        coords_dict={},
    )

    names = get_plottable_vars(out)
    assert {"ux", "uy", "uu_inst", "vv_inst", "uv_inst"} <= set(names)
