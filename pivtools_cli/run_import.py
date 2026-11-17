"""Run import: a PIVTOOLs config for a ``.pivraw`` run recorded by the acquisition app.

The acquisition app records each run to ``<project>/<run name>/`` (one
``Camera<k>.pivraw`` with its ``.json`` sidecar and ``.idx`` per camera, plus
``run.yaml``) and the project's calibration images to
``<project>/calibration/Cam<k>/%05d.tif`` (pivtools-acquisition
``recorder/preflight.py``, ``server/captures.py``).

:func:`import_run` writes ``config.yaml`` from the CLI template plus the run's
facts, every one read through the strict ``.pivraw`` reader: the cameras, the
pairing preset of the run's mode, the image count, and the trigger alignment,
which must succeed before anything is written. It also writes ``rig.dt`` into
the calibration source's ``settings.yaml``. This is the only way a run becomes
a PIVTOOLs config: the recorder writes no config of its own into the run folder
(pivtools-acquisition ``docs/CONTRACTS.md`` section 4).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from pivtools_cli.cli import DEFAULT_CONFIG_YAML
from pivtools_core import calibration_settings as cs
from pivtools_core.config import Config
from pivtools_core.image_handling.readers.pivraw_reader import (
    DOUBLE_FRAME,
    DOUBLE_FRAME_STRIDES,
    PIVRAW_IMAGE_FORMAT,
    PivrawError,
    align_pivraw_run,
    pivraw_file_name,
    read_pivraw_header,
    run_cameras,
)
from pivtools_core.validation import validate_config

# Pairing preset per recorded mode, as (preset, frame_stride, pair_stride) in
# recorded frames. A double-frame run pairs A with B of each trigger (the only
# pairing the reader accepts for it); a time-resolved run pairs each trigger
# with the next.
PAIRING = {
    DOUBLE_FRAME: ("skip_frames",) + DOUBLE_FRAME_STRIDES,
    "time_resolved": ("time_resolved", 1, 1),
}
# Default output folder: <project>/results/<run name>, so the run folder keeps
# only what the recorder wrote.
RESULTS_DIRNAME = "results"
# The acquisition app's calibration layout (captures.py CALIBRATION_DIR_NAME,
# CALIBRATION_CAMERA_DIR, CALIBRATION_FILE_FORMAT). The files are plain TIFFs.
APP_CALIBRATION_DIRNAME = "calibration"
APP_CALIBRATION_CAMERA_DIR = "Cam{slot}"
APP_CALIBRATION_IMAGE_FORMAT = "%05d.tif"
APP_CALIBRATION_IMAGE_TYPE = "standard"


class RunImportConflict(FileExistsError):
    """The import would replace what exists: the config, a different ``rig.dt``, or both.

    Every conflict is collected before anything is written, so the user sees
    all of them before choosing ``force``, which replaces all of them.

    Attributes:
        conflicts: One plain sentence per conflict, without an interface's hint
            (the CLI's ``str()`` adds ``--force``; the GUI adds its own wording).
    """

    def __init__(self, conflicts: Sequence[str]):
        self.conflicts = tuple(conflicts)
        what = "it" if len(self.conflicts) == 1 else "them"
        super().__init__(f"{' '.join(self.conflicts)} Use --force to replace {what}.")


@dataclass(frozen=True)
class RunImport:
    """What :func:`import_run` wrote.

    Attributes:
        config_path: The config written.
        run_dir: The run folder, the config's source path.
        cameras: Camera slots of the run.
        mode: ``double_frame`` or ``time_resolved``.
        n_pairs: Pairs aligned by trigger on every camera.
        calibration_source: The config's calibration source.
        settings_path: The calibration settings holding ``rig.dt``.
        rig_dt: ``rig.dt`` in seconds.
        settings_created: The settings file did not exist; its image block was
            created from the acquisition app's calibration layout.
        warnings: Validation warnings (skipped pairs, alignment notes).
    """

    config_path: Path
    run_dir: Path
    cameras: Tuple[int, ...]
    mode: str
    n_pairs: int
    calibration_source: Path
    settings_path: Path
    rig_dt: float
    settings_created: bool
    warnings: Tuple[str, ...]

    def summary(self) -> str:
        """Plain lines for the CLI."""
        lines = [
            f"Wrote {self.config_path} for run {self.run_dir}",
            f"  cameras {list(self.cameras)}, {self.mode}, {self.n_pairs} pair(s) "
            "aligned by trigger",
            f"  calibration source {self.calibration_source}",
            f"  rig.dt {self.rig_dt!r} s in {self.settings_path}",
        ]
        if self.settings_created:
            lines.append(
                f"  created {self.settings_path.name} with image.image_format "
                f"{APP_CALIBRATION_IMAGE_FORMAT} and image.image_type "
                f"{APP_CALIBRATION_IMAGE_TYPE}, from the acquisition app's "
                "calibration layout (calibration/Cam<k>/%05d.tif); check them "
                "if the calibration images came from elsewhere"
            )
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def _run_dt_s(run_dir: Path, headers: Dict[int, Any]) -> float:
    """The run's ``dt_s``, which every camera's sidecar must agree on.

    Raises:
        PivrawError: The cameras record different ``dt_s``.
    """
    values = {h.dt_s for h in headers.values()}
    if len(values) != 1:
        listed = ", ".join(f"Camera{s} {h.dt_s!r} s" for s, h in headers.items())
        raise PivrawError(
            f"{run_dir}: the cameras record different dt_s ({listed}); one run "
            "has one pulse separation. Re-record the run."
        )
    return values.pop()


def _calibration_source(project: Path, cameras: Tuple[int, ...]) -> Path:
    """``<project>/calibration``, or its ``Cam<k>`` folder for a one-camera rig.

    The app always writes ``Cam<k>/``, while PIVTOOLs reads a one-camera
    standard source at its root (decision 02-3, Morgan 17 Nov).
    """
    root = project / APP_CALIBRATION_DIRNAME
    if len(cameras) == 1:
        return root / APP_CALIBRATION_CAMERA_DIR.format(slot=cameras[0])
    return root


def _stored_rig_dt(source: Path) -> Optional[float]:
    """``rig.dt`` of an existing settings file, None when the file or value is absent.

    Read as stored (:func:`calibration_settings.read_stored_settings`), not
    through ``load_settings``, which refuses a file still missing its image
    block.

    Raises:
        OSError: The settings file cannot be read.
        calibration_settings.CalibrationSettingsError: It is not a YAML mapping.
    """
    stored = cs.read_stored_settings(source)
    if stored is None:
        return None
    return (stored.get("rig") or {}).get("dt")


def _build_config(
    run_dir: Path,
    base_path: Path,
    cameras: Tuple[int, ...],
    num_images: int,
    mode: str,
    calibration_source: Path,
) -> Dict[str, Any]:
    """The CLI template with the run's paths, cameras, images and calibration source."""
    preset, frame_stride, pair_stride = PAIRING[mode]
    data = yaml.safe_load(DEFAULT_CONFIG_YAML)
    data["paths"].update(
        {
            "base_paths": [str(base_path)],
            "source_paths": [str(run_dir)],
            "active_paths": [0],
            "camera_numbers": list(cameras),
            "camera_count": len(cameras),
            "camera_subfolders": [],
        }
    )
    data["images"].update(
        {
            "num_images": num_images,
            "image_format": [PIVRAW_IMAGE_FORMAT],
            "image_type": "pivraw",
            "start_index": 1,
            "num_loops": 1,
            "frame_stride": frame_stride,
            "pair_stride": pair_stride,
            "pairing_preset": preset,
        }
    )
    data["calibration"]["calibration_sources"] = [str(calibration_source)]
    data["calibration"]["source_idx"] = 0
    return data


def _write_validated(
    config_path: Path,
    data: Dict[str, Any],
    before_commit: Callable[[], None],
) -> Tuple[int, Tuple[str, ...]]:
    """Write ``data`` to ``config_path`` only if it validates and ``before_commit`` succeeds.

    The config is staged beside the target and validated with the processing
    checks. Then ``before_commit`` runs (the calibration settings write), and
    only then is the staged file moved into place, as the last step. Any
    failure before the move leaves ``config_path`` as it was and no staged
    file behind. Validation creates the missing base path folders
    (``validate_config``); a failure removes the ones it created, so a refused
    import leaves no empty results folder.

    Returns:
        The config's frame-pair count and the validation warnings.

    Raises:
        ValueError: Validation failed; the message is the validation errors.
        Whatever ``before_commit`` raises, and ``OSError`` from the staging or
        the move.
    """
    created = [
        d
        for base in data["paths"]["base_paths"]
        for d in _missing_folders(Path(base))
    ]
    fd, tmp_path = tempfile.mkstemp(
        prefix=config_path.name + ".", suffix=".tmp", dir=config_path.parent
    )
    committed = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        staged = Config(path=tmp_path)
        ok, error, warnings = validate_config(staged)
        if not ok:
            raise ValueError(f"The imported config does not validate:\n{error}")
        n_pairs = staged.num_frame_pairs
        before_commit()
        os.replace(tmp_path, config_path)
        committed = True
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if not committed:
            _remove_empty_folders(created)
    return n_pairs, tuple(warnings)


def _missing_folders(path: Path) -> List[Path]:
    """``path`` and each ancestor that does not exist yet, deepest first."""
    missing = []
    while not path.exists() and path.parent != path:
        missing.append(path)
        path = path.parent
    return missing


def _remove_empty_folders(folders: Sequence[Path]) -> None:
    """Remove each folder of ``folders`` (deepest first) that exists and is empty.

    A folder something else wrote into since is kept.
    """
    for folder in folders:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def import_run(
    run_dir: Path,
    config_path: Path,
    base_path: Optional[Path] = None,
    force: bool = False,
) -> RunImport:
    """Write ``config_path`` for the ``.pivraw`` run in ``run_dir``.

    Nothing is written unless the run aligns by trigger, the cameras agree on
    ``dt_s``, and the config validates. The calibration settings are written
    before the config is moved into place, so a failed settings write leaves
    the config as it was.

    - Source path: ``run_dir`` (absolute). Output: ``base_path``, by default
      ``<project>/results/<run name>`` with ``<project>`` the run's parent.
    - Cameras: the ``Camera<k>.pivraw`` files. Pairing: ``skip_frames`` for a
      double-frame run, ``time_resolved`` for a time-resolved run, with
      ``start_index`` 1 and ``num_loops`` 1. ``num_images`` is the largest
      ``n_frames`` over the cameras, so it never caps the aligned pairs.
    - Calibration source ``<project>/calibration`` (``.../Cam<k>`` for one
      camera), whose ``settings.yaml`` gets ``rig.dt`` in seconds: ``dt_s``
      (double-frame) or ``frame_stride * dt_s`` (time-resolved). Settings
      created here also name the app's calibration images (``%05d.tif``,
      standard), so they load at once.

    Args:
        run_dir: The run folder.
        config_path: The config to write.
        base_path: Where PIV results go; None for the default.
        force: Overwrite an existing config, and replace a different ``rig.dt``
            in existing calibration settings.

    Raises:
        PivrawError: The reader refuses the run (``PivrawAlignmentError`` when
            it cannot be paired by trigger), or the cameras disagree on dt_s.
        RunImportConflict: ``force`` is False and ``config_path`` exists, or
            the calibration settings hold a different ``rig.dt``; it names
            every such conflict.
        calibration_settings.CalibrationSettingsError: The calibration
            settings file is malformed; nothing is written.
        ValueError: The config does not validate.
        OSError: A file cannot be read or written; the config is unchanged.
    """
    run_dir = Path(os.path.abspath(run_dir))
    config_path = Path(os.path.abspath(config_path))
    project = run_dir.parent

    cameras = run_cameras(run_dir)
    headers = {s: read_pivraw_header(run_dir / pivraw_file_name(s)) for s in cameras}
    first_mode = headers[cameras[0]].mode
    _preset, frame_stride, pair_stride = PAIRING[first_mode]
    alignment = align_pivraw_run(run_dir, frame_stride, pair_stride)
    dt_s = _run_dt_s(run_dir, headers)
    rig_dt = dt_s if alignment.mode == DOUBLE_FRAME else frame_stride * dt_s

    calibration_source = _calibration_source(project, cameras)
    settings_file = cs.settings_path(calibration_source)
    stored_dt = _stored_rig_dt(calibration_source)
    conflicts = []
    if config_path.exists() and not force:
        conflicts.append(f"{config_path} already exists.")
    if stored_dt is not None and stored_dt != rig_dt and not force:
        conflicts.append(
            f"{settings_file} holds rig.dt {stored_dt!r} s, and run {run_dir.name} "
            f"was recorded at {rig_dt!r} s. Velocity scales with dt, so the "
            "calibration of the run(s) recorded at the old value would change."
        )
    if conflicts:
        raise RunImportConflict(conflicts)

    data = _build_config(
        run_dir,
        (
            Path(base_path)
            if base_path is not None
            else project / RESULTS_DIRNAME / run_dir.name
        ),
        cameras,
        max(h.n_frames for h in headers.values()),
        alignment.mode,
        calibration_source,
    )
    partial: Dict[str, Any] = {"rig": {"dt": rig_dt}}
    settings_created = not settings_file.exists()
    if settings_created:
        partial["image"] = {
            "image_format": APP_CALIBRATION_IMAGE_FORMAT,
            "image_type": APP_CALIBRATION_IMAGE_TYPE,
        }
    n_pairs, warnings = _write_validated(
        config_path, data, lambda: cs.save_settings(calibration_source, partial)
    )

    return RunImport(
        config_path=config_path,
        run_dir=run_dir,
        cameras=cameras,
        mode=alignment.mode,
        n_pairs=n_pairs,
        calibration_source=calibration_source,
        settings_path=settings_file,
        rig_dt=rig_dt,
        settings_created=settings_created,
        warnings=warnings,
    )
