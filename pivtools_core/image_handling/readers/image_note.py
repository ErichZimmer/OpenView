"""The image note: what PIVTOOLS Acquisition applied to the images it saved.

Contract: pivtools-acquisition ``docs/CONTRACTS.md`` §3.2, written by
``pivtools_acq.common.image_note``. Two places carry one:

- a calibration image ``<n>.tif`` has ``<n>.tif.json`` beside it;
- an exported run's camera folder ``Camera<k>/`` has one ``image_note.json``
  for all its TIFFs.

A note names the decode rule, the dark subtracted (or none), the view rotation
and the camera. PIVTOOLs compares it with the PIV source when it applies a
calibration (``pivtools_gui/calibration/frame_match.py``), so a calibration is
only applied to PIV images in the same pixel frame. This reader follows the
writer's own strict reader rule for rule; ``unit-tests/test_image_note.py``
pins the constants to the writer's (``fixtures/image_note_contract``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .pivraw_reader import BAYER_RG8, MONO12P
from .pivraw_reader import VIEW_ROTATIONS as _PIVRAW_VIEW_ROTATIONS

NOTE_FORMAT_VERSION = 1
NOTE_KEYS = (
    "format_version",
    "view_rotation",
    "dark_id",
    "dark_subtracted",
    "pixel_rule",
    "camera_serial",
)
# A camera's wire pixel format -> the note's pixel_rule (the decode rule).
PIXEL_RULES = {
    MONO12P: "mono12p",
    BAYER_RG8: "bayer_rggb_superpixel_sum",
}
# The note's rotations are the run.yaml slot rotations the .pivraw reader applies.
VIEW_ROTATIONS = tuple(_PIVRAW_VIEW_ROTATIONS)
# <n>.tif -> <n>.tif.json
CALIBRATION_NOTE_SUFFIX = ".json"
FOLDER_NOTE_NAME = "image_note.json"

PathLike = Union[str, Path]


class ImageNoteError(ValueError):
    """A note that cannot be read or is not exactly the contract. The message names the file."""


@dataclass(frozen=True)
class ImageNote:
    """One note, validated. ``dark_subtracted`` is ``dark_id is not None``."""

    path: Path
    view_rotation: str
    dark_id: Optional[str]
    pixel_rule: str
    camera_serial: str

    @property
    def dark_subtracted(self) -> bool:
        """Whether a dark was subtracted from the images (derived from ``dark_id``)."""
        return self.dark_id is not None


def calibration_note_path(image_path: PathLike) -> Path:
    """The note of one calibration image: ``<n>.tif`` -> ``<n>.tif.json``."""
    image_path = Path(image_path)
    return image_path.with_name(image_path.name + CALIBRATION_NOTE_SUFFIX)


def folder_note_path(folder: PathLike) -> Path:
    """The note of an exported camera folder: ``<folder>/image_note.json``."""
    return Path(folder) / FOLDER_NOTE_NAME


def is_note_file(name: str) -> bool:
    """True for a note file name of either kind (``*.tif.json``, ``image_note.json``)."""
    lower = name.lower()
    return lower == FOLDER_NOTE_NAME or lower.endswith(".tif" + CALIBRATION_NOTE_SUFFIX)


def read_image_note(path: PathLike) -> ImageNote:
    """Read a note strictly.

    Exactly :data:`NOTE_KEYS`; ``format_version`` the int 1; ``dark_subtracted``
    a bool equal to ``dark_id is not None``; ``dark_id`` None or a non-empty
    string; ``view_rotation`` one of :data:`VIEW_ROTATIONS`; ``pixel_rule``
    one of :data:`PIXEL_RULES`' values; ``camera_serial`` a non-empty string.

    Raises:
        ImageNoteError: The file cannot be read, is not JSON, or breaks any rule
            above. The message names the file and the key.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ImageNoteError(f"cannot read the image note {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ImageNoteError(f"{path}: not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ImageNoteError(
            f"{path}: an image note must be a JSON object, got {type(data).__name__}"
        )
    missing = [k for k in NOTE_KEYS if k not in data]
    unknown = sorted(set(data) - set(NOTE_KEYS))
    if missing or unknown:
        raise ImageNoteError(
            f"{path}: not an image note of format {NOTE_FORMAT_VERSION}: "
            f"missing {', '.join(missing) or 'none'}, unknown {', '.join(unknown) or 'none'}"
        )
    version = data["format_version"]
    if type(version) is not int or version != NOTE_FORMAT_VERSION:
        raise ImageNoteError(
            f"{path}: format_version must be {NOTE_FORMAT_VERSION}, got {version!r}"
        )
    dark_id = data["dark_id"]
    if dark_id is not None and (not isinstance(dark_id, str) or not dark_id):
        raise ImageNoteError(
            f"{path}: dark_id must be a dark's id or null, got {dark_id!r}"
        )
    if data["dark_subtracted"] is not (dark_id is not None):
        raise ImageNoteError(
            f"{path}: dark_subtracted is {json.dumps(data['dark_subtracted'])} but "
            f"dark_id is {json.dumps(dark_id)}"
        )
    rotation = data["view_rotation"]
    if rotation not in VIEW_ROTATIONS:
        raise ImageNoteError(
            f"{path}: unknown view rotation {rotation!r}; known: {VIEW_ROTATIONS}"
        )
    rule = data["pixel_rule"]
    if rule not in PIXEL_RULES.values():
        raise ImageNoteError(
            f"{path}: unknown pixel rule {rule!r}; known: {tuple(PIXEL_RULES.values())}"
        )
    serial = data["camera_serial"]
    if not isinstance(serial, str) or not serial:
        raise ImageNoteError(
            f"{path}: camera_serial must be the camera's serial, got {serial!r}"
        )
    return ImageNote(
        path=path,
        view_rotation=rotation,
        dark_id=dark_id,
        pixel_rule=rule,
        camera_serial=serial,
    )
