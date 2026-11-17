"""calibration.frame_match — a calibration applies only to PIV images in its own pixel frame.

A calibration model maps pixels of its calibration images to world mm. Applied to
PIV vectors measured on images that were turned, dark-treated or taken by another
camera, it puts every vector in the wrong place without any error. This module
proves the two image sets share one pixel frame before an apply starts, and
refuses by name when it cannot.

What each side says about its frame:

- **A .pivraw run** (``image_type: pivraw``): per camera, ``view_rotation`` from
  the run's ``run.yaml`` slot, ``dark_subtracted`` from whether the run holds an
  in-run dark, ``pixel_rule`` and ``camera_serial`` from the camera's sidecar.
- **An exported TIFF run** (the acquisition export): the camera folder's
  ``image_note.json``.
- **Acquisition calibration images**: one note per image, ``<n>.tif.json``.
- **DaVis .set recordings**: each camera's geometry transformer.
- **Anything else** (TIFF without notes, ``.im7``, ``.cine``): nothing.

The rules (:func:`_compare_camera`):

- app calibration notes against an app run (``.pivraw`` or exported): every note
  must agree on ``view_rotation``, ``dark_subtracted``, ``pixel_rule`` and
  ``camera_serial``. ``dark_id`` is never compared: a calibration is taken at
  another exposure with its own dark.
- ``.set`` calibration against ``.set`` PIV: each camera's geometry must agree.
- DaVis against plain, either way: applied, nothing compared, logged.
- plain against plain: applied, nothing to compare.
- every other pairing: refused, because the frames cannot be proven to match.

The apply planner (``runio.plan_apply_units``) calls :func:`check_calibration_frames`
once per apply, so the GUI route and both CLI apply commands check alike.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pivtools_core.calibration_settings import try_load_settings
from pivtools_core.image_handling.path_utils import (
    _glob_images,
    build_calibration_camera_path,
    format_to_glob,
)
from pivtools_core.image_handling.readers.image_note import (
    PIXEL_RULES,
    ImageNoteError,
    calibration_note_path,
    folder_note_path,
    read_image_note,
)
from pivtools_core.image_handling.readers.pivraw_reader import (
    PivrawError,
    read_pivraw_header,
    read_pivraw_run_file,
)
from pivtools_core.image_handling.readers.set_reader import (
    read_set_info,
    set_camera_streams,
    stream_geometry,
)

logger = logging.getLogger(__name__)

# The note fields that fix the pixel frame. dark_id is provenance only.
COMPARED_FIELDS = ("view_rotation", "dark_subtracted", "pixel_rule", "camera_serial")

# Kinds of image source.
PIVRAW = "pivraw"  # PIV only: a PIVTOOLS Acquisition run
EXPORTED = "exported"  # PIV only: TIFFs exported by the app, one folder note
NOTED = "noted"  # calibration only: app calibration TIFFs, one note per image
DAVIS = "davis"  # a LaVision .set recording
PLAIN = "plain"  # nothing recorded about the frame

# Geometry of a .set camera whose streams carry no geometry transformer.
NO_GEOMETRY = "none"


class CalibrationFrameMismatch(ValueError):
    """The calibration's images and the PIV images are not proven to share a pixel frame.

    The message names the calibration source, the PIV source, and per problem the
    camera, the image and the field with both values.
    """


@dataclass(frozen=True)
class _AppFrame:
    """What the acquisition app applied to one camera's images (the compared fields)."""

    view_rotation: str
    dark_subtracted: bool
    pixel_rule: str
    camera_serial: str

    def text(self, field: str) -> str:
        """A field's value as the note writes it (booleans as ``true``/``false``)."""
        value = getattr(self, field)
        return json.dumps(value) if isinstance(value, bool) else str(value)


@dataclass(frozen=True)
class _PivCamera:
    """One PIV camera's frame facts. ``frame`` for app runs, ``geometry`` for DaVis."""

    kind: str
    where: str
    frame: Optional[_AppFrame] = None
    geometry: Optional[str] = None


@dataclass(frozen=True)
class _CalibrationCamera:
    """One calibration camera's frame facts.

    ``images`` (noted): each image with its note's path, None when it has none.
    ``geometry`` (davis). ``why_plain``: what the calibration is, for a refusal.
    """

    kind: str
    images: Tuple[Tuple[Path, Optional[Path]], ...] = ()
    geometry: Optional[str] = None
    why_plain: str = ""


# ---------------------------------------------------------------------------
# PIV side
# ---------------------------------------------------------------------------


def _pivraw_camera(source: Path, image_format: str, camera: int) -> _PivCamera:
    """Camera ``camera`` of the .pivraw run ``source``: run.yaml slot and dark, sidecar."""
    path = source / (image_format % camera if "%" in image_format else image_format)
    header = read_pivraw_header(path)
    run = read_pivraw_run_file(path.parent)
    rotation = run.view_rotations.get(header.slot)
    if rotation is None:
        raise PivrawError(
            f"{run.path}: records no view_rotation for Camera{header.slot}, so the "
            "run's pixel frame is unknown."
        )
    return _PivCamera(
        kind=PIVRAW,
        where=f"{path.name}, run.yaml",
        frame=_AppFrame(
            view_rotation=rotation,
            dark_subtracted=run.dark_id is not None,
            pixel_rule=PIXEL_RULES[header.pixel_format],
            camera_serial=header.camera_serial,
        ),
    )


def _set_geometry(set_path: Path, camera: int, num_cameras: int) -> str:
    """The geometry transformer of one camera of a .set, :data:`NO_GEOMETRY` when none.

    Raises:
        ValueError: The camera's streams carry different geometries, or a
            transformer PIVTOOLs does not know.
    """
    info = read_set_info(set_path)
    streams = set_camera_streams(set_path, num_cameras, info)[camera - 1]
    found = {stream_geometry(info.frames[s]) or NO_GEOMETRY for s in streams}
    if len(found) != 1:
        raise ValueError(
            f"{set_path.name}: the streams of camera {camera} carry different "
            f"geometries ({', '.join(sorted(found))})."
        )
    return found.pop()


def _piv_camera(full_cfg, source: Path, camera: int) -> _PivCamera:
    """What the PIV source ``source`` records about camera ``camera``'s pixel frame."""
    image_type = full_cfg.image_type
    if image_type == "pivraw":
        return _pivraw_camera(source, full_cfg.image_format[0], camera)
    if image_type == "lavision_set":
        return _PivCamera(
            kind=DAVIS,
            where=source.name,
            geometry=_set_geometry(source, camera, full_cfg.camera_count),
        )
    if image_type == "standard":
        folder = full_cfg.get_camera_folder(camera)
        note_path = folder_note_path(source / folder if folder else source)
        if note_path.is_file():
            note = read_image_note(note_path)
            return _PivCamera(
                kind=EXPORTED,
                where=str(note_path),
                frame=_AppFrame(
                    view_rotation=note.view_rotation,
                    dark_subtracted=note.dark_subtracted,
                    pixel_rule=note.pixel_rule,
                    camera_serial=note.camera_serial,
                ),
            )
    return _PivCamera(kind=PLAIN, where=str(source))


# ---------------------------------------------------------------------------
# Calibration side
# ---------------------------------------------------------------------------


def _calibration_camera(
    source: Path, image: Optional[dict], camera: int, num_cameras: int
) -> _CalibrationCamera:
    """What the calibration source records about camera ``camera``'s pixel frame.

    ``image`` is the source's settings ``image`` block, None when it has no
    settings (an image-less model, or a source never opened in the GUI).
    """
    if image is None:
        return _CalibrationCamera(
            kind=PLAIN, why_plain="has no settings.yaml, so no calibration images"
        )
    image_type = image["image_type"]
    camera_path = build_calibration_camera_path(source, image, camera, num_cameras)
    if image_type == "lavision_set":
        return _CalibrationCamera(
            kind=DAVIS, geometry=_set_geometry(camera_path, camera, num_cameras)
        )
    if image_type != "standard":
        return _CalibrationCamera(
            kind=PLAIN,
            why_plain=f"holds {image_type} images, which carry no image notes",
        )
    images = sorted(_glob_images(camera_path, format_to_glob(image["image_format"])))
    pairs = tuple(
        (img, note if note.is_file() else None)
        for img in images
        for note in (calibration_note_path(img),)
    )
    if not any(note for _, note in pairs):
        return _CalibrationCamera(
            kind=PLAIN,
            why_plain=(
                f"holds {len(images)} image(s) in {camera_path} without image notes "
                "(not saved by PIVTOOLS Acquisition, or saved before notes existed)"
            ),
        )
    return _CalibrationCamera(kind=NOTED, images=pairs)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

_KIND_WORDS = {
    PIVRAW: "a PIVTOOLS Acquisition .pivraw run",
    EXPORTED: "TIFFs exported by PIVTOOLS Acquisition",
    NOTED: "images saved by PIVTOOLS Acquisition (with image notes)",
    DAVIS: "a DaVis .set recording",
    PLAIN: "images that record nothing about their frame",
}


def _compare_notes(camera: int, cal: _CalibrationCamera, piv: _PivCamera) -> List[str]:
    """Problems of each calibration image against the app run's camera."""
    problems = []
    for image_path, note_path in cal.images:
        prefix = f"camera {camera}, {image_path.name}"
        if note_path is None:
            problems.append(
                f"{prefix}: no image note ({calibration_note_path(image_path).name} "
                "is missing)"
            )
            continue
        try:
            note = read_image_note(note_path)
        except ImageNoteError as exc:
            problems.append(f"{prefix}: {exc}")
            continue
        noted = _AppFrame(
            view_rotation=note.view_rotation,
            dark_subtracted=note.dark_subtracted,
            pixel_rule=note.pixel_rule,
            camera_serial=note.camera_serial,
        )
        for field in COMPARED_FIELDS:
            if getattr(noted, field) != getattr(piv.frame, field):
                problems.append(
                    f"{prefix}: {field} {noted.text(field)} in its note, "
                    f"{piv.frame.text(field)} in the run ({piv.where})"
                )
    return problems


def _compare_camera(camera: int, cal: _CalibrationCamera, piv: _PivCamera) -> List[str]:
    """Problems for one camera; empty when the frames match or nothing can be compared."""
    if cal.kind == NOTED and piv.kind in (PIVRAW, EXPORTED):
        return _compare_notes(camera, cal, piv)
    if cal.kind == DAVIS and piv.kind == DAVIS:
        if cal.geometry != piv.geometry:
            return [
                f"camera {camera}: geometry transformer {cal.geometry} in the "
                f"calibration recording, {piv.geometry} in the PIV recording "
                f"({piv.where})"
            ]
        return []
    if cal.kind == PLAIN and piv.kind == PLAIN:
        return []
    if {cal.kind, piv.kind} == {DAVIS, PLAIN}:
        logger.warning(
            "calibration frame check: camera %d calibration is %s and the PIV source "
            "is %s; DaVis records its geometry only inside a .set, so the two frames "
            "are not compared",
            camera,
            _KIND_WORDS[cal.kind],
            _KIND_WORDS[piv.kind],
        )
        return []
    calibration = (
        f"the calibration {cal.why_plain}"
        if cal.why_plain
        else (f"the calibration holds {_KIND_WORDS[cal.kind]}")
    )
    return [
        f"camera {camera}: {calibration}, and the PIV source is {_KIND_WORDS[piv.kind]} "
        f"({piv.where}). Their pixel frames (rotation, dark, pixel rule, camera) "
        "cannot be proven to match. Calibrate with images of the same kind as the "
        "PIV images: for a PIVTOOLS Acquisition run, calibration images captured "
        "in the acquisition app"
    ]


def check_calibration_frames(
    full_cfg,
    calibration_source: Path,
    cameras: Sequence[int],
    piv_sources: Sequence[Path],
) -> None:
    """Refuse a calibration whose pixel frame is not proven to be the PIV images'.

    Args:
        full_cfg: the PIV ``Config`` (image type and format, camera count, folders).
        calibration_source: the calibration source the model was built from.
        cameras: the cameras the apply covers.
        piv_sources: the PIV source paths whose vectors are being calibrated.

    Raises:
        CalibrationFrameMismatch: Any camera of any PIV source fails a rule in the
            module docstring. Every problem is listed, grouped by PIV source.
            A PIV run, note or recording that cannot be read is reported too.
    """
    calibration_source = Path(calibration_source)
    settings = try_load_settings(calibration_source)
    image = None if settings is None else settings["image"]
    calibration: Dict[int, _CalibrationCamera] = {}
    unreadable = []
    for cam in cameras:
        try:
            calibration[cam] = _calibration_camera(
                calibration_source, image, cam, full_cfg.camera_count
            )
        except (OSError, ValueError) as exc:
            unreadable.append(
                f"camera {cam}: the calibration's pixel frame cannot be read: {exc}"
            )
    if unreadable:
        raise CalibrationFrameMismatch(
            f"The calibration in {calibration_source} is not applied: "
            + "; ".join(unreadable)
        )
    failures = []
    for source in piv_sources:
        problems = []
        for cam in cameras:
            try:
                piv = _piv_camera(full_cfg, Path(source), cam)
            except (OSError, ValueError) as exc:
                problems.append(
                    f"camera {cam}: the PIV source's pixel frame cannot be read: {exc}"
                )
                continue
            problems.extend(_compare_camera(cam, calibration[cam], piv))
        if problems:
            failures.append(f"against the PIV source {source}: " + "; ".join(problems))
    if failures:
        raise CalibrationFrameMismatch(
            f"The calibration in {calibration_source} is not applied, because its "
            "images are not proven to be in the same pixel frame as the PIV images "
            + " | ".join(failures)
            + ". A calibration applied to images turned, dark-treated or taken by "
            "another camera puts every vector in the wrong place."
        )
