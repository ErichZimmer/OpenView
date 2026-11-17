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

The rules (:data:`RULES`, applied by :func:`_compare_camera`):

- app calibration notes against an app run (``.pivraw`` or exported): every note
  must agree on ``view_rotation``, ``dark_subtracted``, ``pixel_rule`` and
  ``camera_serial``. ``dark_id`` is never compared: a calibration is taken at
  another exposure with its own dark.
- ``.set`` calibration against ``.set`` PIV: each recording is split into
  cameras by its own labels, camera k must be the same physical camera
  (``Acq.Camera.ID``) on both, and its geometry must agree. When either has no
  labels, camera k is taken as camera k, with a warning saying so.
- DaVis against plain, either way: applied, nothing compared, with a warning
  returned for the apply result.
- plain against plain: applied, nothing to compare.
- every other pairing: refused, because the frames cannot be proven to match.

The apply planner (``runio.plan_apply_units``) and stereo self-calibration
(``self_cal.check_self_cal_frames``) call :func:`check_calibration_frames`, so the
GUI routes and the CLI commands check alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pivtools_core.calibration_settings import try_load_settings
from pivtools_core.image_handling.path_utils import (
    glob_images,
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
from pivtools_core.image_handling.readers.davis_attr import (
    LabelledCamera,
    camera_list_text,
)
from pivtools_core.image_handling.readers.set_reader import (
    SetConfigError,
    read_set_info,
    set_camera_streams,
    stream_geometry,
)

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
    """One PIV camera's frame facts. ``frame`` for app runs, ``geometry`` for DaVis.

    ``identity``: a DaVis camera as its recording's labels name it, None
    without labels.
    """

    kind: str
    where: str
    frame: Optional[_AppFrame] = None
    geometry: Optional[str] = None
    identity: Optional[LabelledCamera] = None


@dataclass(frozen=True)
class _CalibrationCamera:
    """One calibration camera's frame facts.

    ``images`` (noted): each image with its note's path, None when it has none.
    ``geometry`` and ``identity`` (davis), as on :class:`_PivCamera`.
    ``why_plain``: what the calibration is, for a refusal.
    """

    kind: str
    images: Tuple[Tuple[Path, Optional[Path]], ...] = ()
    geometry: Optional[str] = None
    identity: Optional[LabelledCamera] = None
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


def _set_camera(
    set_path: Path, camera: int, num_cameras: int, own_count: bool
) -> Tuple[str, Optional[LabelledCamera]]:
    """(geometry, identity) of one camera of a .set recording.

    The geometry is the camera's geometry transformer, :data:`NO_GEOMETRY` when
    none. The identity is the camera as the recording's own labels name it,
    None for a recording without labels.

    Args:
        set_path: The .set file.
        camera: 1-based camera number.
        num_cameras: The camera count of the config the recording is read
            with. It splits an unlabelled recording.
        own_count: True for a calibration recording: a labelled one is split
            by its own labels, whatever ``num_cameras`` says, so a calibration
            of 2 cameras serves a PIV rig of 4. False for the PIV recording,
            whose labels must agree with ``num_cameras``.

    Raises:
        ValueError: The recording holds no camera ``camera``, the camera count
            does not fit it (``SetConfigError``), the camera's streams carry
            different geometries, or a transformer PIVTOOLs does not know.
    """
    info = read_set_info(set_path)
    count = len(info.cameras) if own_count and info.cameras else num_cameras
    split = set_camera_streams(set_path, count, info)
    if not 1 <= camera <= len(split):
        listing = (
            f" by its own camera labels: {camera_list_text(info.cameras)}"
            if info.cameras
            else ""
        )
        raise SetConfigError(
            f"{set_path.name} holds {len(split)} camera(s){listing}. It has no "
            f"camera {camera}."
        )
    streams = split[camera - 1]
    found = {stream_geometry(info.frames[s]) or NO_GEOMETRY for s in streams}
    if len(found) != 1:
        raise ValueError(
            f"{set_path.name}: the streams of camera {camera} carry different "
            f"geometries ({', '.join(sorted(found))})."
        )
    identity = None if info.cameras is None else info.cameras[camera - 1]
    return found.pop(), identity


def _piv_camera(full_cfg, source: Path, camera: int) -> _PivCamera:
    """What the PIV source ``source`` records about camera ``camera``'s pixel frame."""
    image_type = full_cfg.image_type
    if image_type == "pivraw":
        return _pivraw_camera(source, full_cfg.image_format[0], camera)
    if image_type == "lavision_set":
        geometry, identity = _set_camera(
            source, camera, full_cfg.camera_count, own_count=False
        )
        return _PivCamera(
            kind=DAVIS, where=source.name, geometry=geometry, identity=identity
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
    ``num_cameras`` is the PIV config's camera count: it places a camera's
    folder, and splits a ``.set`` recording without camera labels, as the
    calibration loader read it. A labelled recording is split by its own labels.
    """
    if image is None:
        return _CalibrationCamera(
            kind=PLAIN, why_plain="has no settings.yaml, so no calibration images"
        )
    image_type = image["image_type"]
    camera_path = build_calibration_camera_path(source, image, camera, num_cameras)
    if image_type == "lavision_set":
        geometry, identity = _set_camera(
            camera_path, camera, num_cameras, own_count=True
        )
        return _CalibrationCamera(kind=DAVIS, geometry=geometry, identity=identity)
    if image_type != "standard":
        return _CalibrationCamera(
            kind=PLAIN,
            why_plain=f"holds {image_type} images, which carry no image notes",
        )
    images = sorted(glob_images(camera_path, format_to_glob(image["image_format"])))
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


# What happens for each (calibration kind, PIV kind). Any pair not listed refuses.
COMPARE_NOTES = "compare notes"  # every calibration note against the app run's camera
COMPARE_GEOMETRY = "compare geometry"  # each camera's .set geometry transformer
APPLY = "apply"  # nothing on either side to compare, as before K1
WARN = "warn"  # applied, with a warning the user reads in the apply result
REFUSE = "refuse"
RULES: Dict[Tuple[str, str], str] = {
    (NOTED, PIVRAW): COMPARE_NOTES,
    (NOTED, EXPORTED): COMPARE_NOTES,
    (DAVIS, DAVIS): COMPARE_GEOMETRY,
    (PLAIN, PLAIN): APPLY,
    # DaVis records its geometry only inside a .set; DaVis users calibrate from
    # DaVis's own Properties/Calibration .im7 images (lead decision, 17 Nov; the
    # option of refusing is put to Morgan).
    (DAVIS, PLAIN): WARN,
    (PLAIN, DAVIS): WARN,
}

# Short names for the warning sentence.
_KIND_LABEL = {
    DAVIS: "DaVis .set recording",
    PLAIN: "images with no frame record",
}


# COMPARE_GEOMETRY when either recording has no camera labels.
IDENTITY_NOT_COMPARED_WARNING = (
    "The calibration recording or the PIV recording carries no camera labels, so "
    "camera k of the calibration is taken to be camera k of the PIV recording "
    "without checking that they are the same physical camera."
)


def frames_not_compared_warning(cal_kind: str, piv_kind: str) -> str:
    """The warning an applied-but-uncompared calibration carries in its result."""
    return (
        f"The calibration ({_KIND_LABEL[cal_kind]}) and the PIV images "
        f"({_KIND_LABEL[piv_kind]}) carry no shared record of rotation or dark, so "
        "their pixel frames could not be compared."
    )


def _compare_davis(
    camera: int, cal: _CalibrationCamera, piv: _PivCamera
) -> Tuple[List[str], Optional[str]]:
    """(problems, warning) for one camera of a .set calibration against .set PIV.

    With labels on both recordings, camera ``camera`` of each must be the same
    physical camera (``Acq.Camera.ID``, which carries the serial), and then its
    geometry must agree. Without labels on either, camera k is taken as camera
    k and the warning says so.
    """
    problems = []
    warning = None
    if cal.identity is not None and piv.identity is not None:
        if cal.identity.camera_id != piv.identity.camera_id:
            problems.append(
                f"camera {camera}: the calibration recording's camera {camera} is "
                f"{cal.identity.camera_id}, the PIV recording's camera {camera} is "
                f"{piv.identity.camera_id} ({piv.where}). A model of one camera "
                "cannot calibrate another: order the calibration's cameras as the "
                "PIV recording's"
            )
            return problems, None
    else:
        warning = IDENTITY_NOT_COMPARED_WARNING
    if cal.geometry != piv.geometry:
        problems.append(
            f"camera {camera}: geometry transformer {cal.geometry} in the "
            f"calibration recording, {piv.geometry} in the PIV recording "
            f"({piv.where})"
        )
    return problems, warning


def _compare_camera(
    camera: int, cal: _CalibrationCamera, piv: _PivCamera
) -> Tuple[List[str], Optional[str]]:
    """(problems, warning) for one camera, by :data:`RULES`."""
    rule = RULES.get((cal.kind, piv.kind), REFUSE)
    if rule == COMPARE_NOTES:
        return _compare_notes(camera, cal, piv), None
    if rule == COMPARE_GEOMETRY:
        return _compare_davis(camera, cal, piv)
    if rule == APPLY:
        return [], None
    if rule == WARN:
        return [], frames_not_compared_warning(cal.kind, piv.kind)
    calibration = (
        f"the calibration {cal.why_plain}"
        if cal.why_plain
        else f"the calibration holds {_KIND_WORDS[cal.kind]}"
    )
    return [
        f"camera {camera}: {calibration}, and the PIV source is "
        f"{_KIND_WORDS[piv.kind]} ({piv.where}). Their pixel frames (rotation, dark, "
        "pixel rule, camera) cannot be proven to match. Calibrate with images of the "
        "same kind as the PIV images: for a PIVTOOLS Acquisition run, calibration "
        "images captured in the acquisition app"
    ], None


def check_calibration_frames(
    full_cfg,
    calibration_source: Path,
    cameras: Sequence[int],
    piv_sources: Sequence[Path],
) -> Tuple[str, ...]:
    """Refuse a calibration whose pixel frame is not proven to be the PIV images'.

    Args:
        full_cfg: the PIV ``Config`` (image type and format, camera count, folders).
        calibration_source: the calibration source the model was built from.
        cameras: the cameras the apply covers.
        piv_sources: the PIV source paths whose vectors are being calibrated.

    Returns:
        The warnings of the pairs applied without a comparison (:data:`WARN`),
        each once, for the caller to put in the apply result the user reads.

    Raises:
        PivrawError: The .pivraw reader refuses the PIV run (its own text).
        CalibrationFrameMismatch: Any camera of any PIV source fails a rule of
            :data:`RULES`. Every problem is listed, grouped by PIV source. A note
            or recording that cannot be read is reported too.
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
    warnings: List[str] = []
    for source in piv_sources:
        problems = []
        for cam in cameras:
            try:
                piv = _piv_camera(full_cfg, Path(source), cam)
            except PivrawError:
                raise
            except (OSError, ValueError) as exc:
                problems.append(
                    f"camera {cam}: the PIV source's pixel frame cannot be read: {exc}"
                )
                continue
            found, warning = _compare_camera(cam, calibration[cam], piv)
            problems.extend(found)
            if warning is not None and warning not in warnings:
                warnings.append(warning)
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
    return tuple(warnings)
