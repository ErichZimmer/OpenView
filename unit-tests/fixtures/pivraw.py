"""Synthetic PIVTOOLS Acquisition run folders (``.pivraw``) for reader tests.

Builds the container from the format specification (pivtools-acquisition
``docs/CONTRACTS.md`` section 4) without touching the reader under test:

    <run_dir>/Camera{k}.pivraw       frame j at j * frame_stride, zero padded
    <run_dir>/Camera{k}.pivraw.json  header sidecar
    <run_dir>/Camera{k}.pivraw.idx   "<qQ" (host ns, camera frame id) per frame
    <run_dir>/run.yaml, pivtools_config.yaml

Mono12p is packed here byte by byte (LSB-first, two pixels in three bytes), so a
reader that decodes it back is checked against an independent construction.
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import yaml

STRIDE_ALIGNMENT = 4096
INDEX_RECORD = struct.Struct("<qQ")
# Sentinel for ``sidecar_overrides``: remove the field from the sidecar.
DROP = object()


def pack_mono12p(img: np.ndarray) -> bytes:
    """uint16 (H, W) 12-bit counts -> PFNC Mono12p bytes.

    Pixel pair (p0, p1) -> bytes ``b0 = p0 & 0xFF``,
    ``b1 = (p0 >> 8) | (p1 & 0xF) << 4``, ``b2 = p1 >> 4``.
    """
    flat = np.asarray(img, dtype=np.uint16).reshape(-1)
    if flat.size % 2 or flat.max(initial=0) > 0x0FFF:
        raise ValueError("Mono12p needs an even pixel count and values below 4096")
    p0 = flat[0::2].astype(np.uint32)
    p1 = flat[1::2].astype(np.uint32)
    packed = np.empty((p0.size, 3), dtype=np.uint8)
    packed[:, 0] = p0 & 0xFF
    packed[:, 1] = ((p0 >> 8) & 0x0F) | ((p1 & 0x0F) << 4)
    packed[:, 2] = (p1 >> 4) & 0xFF
    return packed.tobytes()


def stride_for(bytes_per_frame: int) -> int:
    """``ceil(bytes_per_frame / 4096) * 4096``."""
    return -(-bytes_per_frame // STRIDE_ALIGNMENT) * STRIDE_ALIGNMENT


def write_camera(
    run_dir: Path,
    camera: int,
    frames: Sequence[np.ndarray],
    pixel_format: str = "Mono12p",
    mode: str = "double_frame",
    sidecar_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``Camera{camera}.pivraw`` + ``.json`` + ``.idx`` and return the data path.

    ``frames`` are uint16 12-bit images for Mono12p, uint8 RGGB mosaics for
    BayerRG8, each (height, width) at sensor resolution.
    ``sidecar_overrides`` replaces sidecar values; a value of :data:`DROP`
    deletes the field.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape
    if pixel_format == "Mono12p":
        payloads = [pack_mono12p(f) for f in frames]
    elif pixel_format == "BayerRG8":
        payloads = [np.asarray(f, dtype=np.uint8).tobytes() for f in frames]
    else:
        raise ValueError(f"fixture does not build {pixel_format!r}")
    bytes_per_frame = len(payloads[0])
    stride = stride_for(bytes_per_frame)
    padding = b"\0" * (stride - bytes_per_frame)

    data_path = run_dir / f"Camera{camera}.pivraw"
    data_path.write_bytes(b"".join(p + padding for p in payloads))
    Path(str(data_path) + ".idx").write_bytes(
        b"".join(INDEX_RECORD.pack(1_000_000 * k, 100 + k) for k in range(len(frames)))
    )

    sidecar: Dict[str, Any] = {
        "format": "pivraw",
        "format_version": 1,
        "camera_model": "Synthetic",
        "camera_serial": f"SN{camera:04d}",
        "slot": camera - 1,
        "width": width,
        "height": height,
        "pixel_format": pixel_format,
        "bytes_per_frame": bytes_per_frame,
        "frame_stride": stride,
        "mode": mode,
        "exposure_us": 10.0,
        "exposure_effective_us": 10.0,
        "max_rate_hz": 25.0,
        "rate_hz": 15.0,
        "dt_s": 2.0e-5 if mode == "double_frame" else 1.0 / 15.0,
        "camera_settings": {"gain": 1},
        "created_at": "2026-10-26T12:00:00+00:00",
        "n_frames": len(frames),
        "finalised": True,
        "dropped_pairs": 0,
        "missing_indices": [],
    }
    for key, value in (sidecar_overrides or {}).items():
        if value is DROP:
            sidecar.pop(key)
        else:
            sidecar[key] = value
    Path(str(data_path) + ".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return data_path


def write_run(
    run_dir: Path,
    frames_per_camera: Sequence[Sequence[np.ndarray]],
    pixel_format: str = "Mono12p",
    mode: str = "double_frame",
) -> Path:
    """A whole run folder: one container per camera plus the two YAML files."""
    for camera, frames in enumerate(frames_per_camera, start=1):
        write_camera(run_dir, camera, frames, pixel_format, mode)
    (run_dir / "run.yaml").write_text(yaml.safe_dump({"mode": mode}), encoding="utf-8")
    preset = "skip_frames" if mode == "double_frame" else "time_resolved"
    (run_dir / "pivtools_config.yaml").write_text(
        yaml.safe_dump(
            {
                "source_path": str(run_dir),
                "image_format": ["Camera%d.pivraw"],
                "pairing_preset": preset,
                "camera_count": len(frames_per_camera),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def mono_frames(n: int, height: int, width: int, seed: int = 0) -> list:
    """``n`` random 12-bit uint16 frames, each distinct."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 4096, size=(height, width), dtype=np.uint16) for _ in range(n)
    ]


def bayer_frames(n: int, height: int, width: int, seed: int = 0) -> list:
    """``n`` random uint8 RGGB mosaics."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(height, width), dtype=np.uint8) for _ in range(n)
    ]


def write_piv_config(
    tmp_path: Path,
    run_dir: Path,
    camera_count: int,
    num_images: int,
    preset: str = "skip_frames",
) -> Path:
    """A ``config.yaml`` pointing PIVTOOLs at ``run_dir`` the way a user would."""
    strides = {"skip_frames": (1, 2), "time_resolved": (1, 1), "ab_format": (0, 1)}
    frame_stride, pair_stride = strides[preset]
    cfg = {
        "paths": {
            "source_paths": [str(run_dir)],
            "camera_count": camera_count,
            "camera_numbers": list(range(1, camera_count + 1)),
        },
        "images": {
            "image_format": ["Camera%d.pivraw"],
            "image_type": "pivraw",
            "num_images": num_images,
            "start_index": 1,
            "pairing_preset": preset,
            "frame_stride": frame_stride,
            "pair_stride": pair_stride,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path
