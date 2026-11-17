"""``pivtools-cli instantaneous`` and ``ensemble`` exit non-zero when processing fails.

Scripts and the GUI's job runner (``piv_runner.py``, success = return code 0)
read the exit code, so a failure that is only logged reports a run that did
nothing as a success. Each test runs the real command in a subprocess from a
folder holding the ``config.yaml``, as a user would.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fixtures.pivraw import copy_case, write_piv_config

REPO_ROOT = Path(__file__).resolve().parent.parent
# A processing subprocess imports Dask and the C libraries before it fails.
SUBPROCESS_TIMEOUT_S = 300


def _run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    env.pop("PIV_CAMERA", None)
    env.pop("PIV_ACTIVE_PATHS", None)
    return subprocess.run(
        [sys.executable, "-m", "pivtools_cli.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


def _pivraw_workspace(tmp_path: Path, case: str, **processing) -> Path:
    """A folder with a ``config.yaml`` that processes one contract run."""
    run = copy_case(tmp_path, case)
    work = tmp_path / "work"
    work.mkdir()
    config_path = write_piv_config(work, run, 2, 4)
    data = yaml.safe_load(config_path.read_text())
    data["paths"]["base_paths"] = [str(tmp_path / "out")]
    if processing:
        data["processing"] = processing
    config_path.write_text(yaml.dump(data))
    return work


def test_a_run_that_cannot_align_exits_non_zero_with_the_reason(tmp_path):
    """PC T6: the reader's refusal reaches the user and the exit code."""
    work = _pivraw_workspace(tmp_path, "df_2cam_missed_trigger_cam2")
    result = _run_cli(work, "instantaneous")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "Camera2 run trigger 1" in output and "a trigger ignored" in output


@pytest.mark.parametrize("command", ["instantaneous", "ensemble"])
def test_a_failure_after_validation_exits_non_zero_with_the_reason(tmp_path, command):
    """A config that validates but cannot be processed: a Slurm cluster with no
    node count fails inside ``main``'s processing block, after validation."""
    work = _pivraw_workspace(tmp_path, "df_2cam_clean", cluster_type="slurm")
    result = _run_cli(work, command)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "n_nodes must be set for Slurm cluster" in output
