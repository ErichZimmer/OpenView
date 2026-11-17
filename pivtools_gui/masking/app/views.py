from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from pivtools_core.config import get_config
from pivtools_core.image_handling.load_images import (
    MaskError,
    load_mask_for_camera,
    require_mask_fits_camera,
)
from pivtools_core.vector_loading import read_mask_from_mat, save_mask_to_mat

from ...utils import camera_number, numpy_to_png_base64

masking_bp = Blueprint("masking", __name__)


def _cfg():
    return get_config()


@masking_bp.route("/save_mask_array", methods=["POST"])
def upload_mask():
    """
    Expects JSON with: meta (basePathIdx, camera, index, frame), width, height, polygons.
    Optionally accepts 'data' (flat mask array) for backward compatibility.
    Saves mask as .mat file.
    """
    import cv2

    payload = request.get_json(silent=True) or {}
    width = payload.get("width")
    height = payload.get("height")
    flat = payload.get("data")
    meta = payload.get("meta", {})
    polygons = payload.get("polygons") or []

    # Validate dimensions
    if not (
        isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0
    ):
        return jsonify({"error": "width and height must be positive integers"}), 400

    if flat is not None:
        # Legacy path: rasterize from flat array
        if not (isinstance(flat, list) and len(flat) == width * height):
            return jsonify({"error": "data must be a list of length width*height"}), 400
        try:
            mask = np.asarray(flat, dtype=bool).reshape((height, width))
        except Exception as e:
            return jsonify({"error": f"invalid mask data: {e}"}), 400
    elif len(polygons) > 0:
        # Server-side rasterization from polygon data
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            pts = poly.get("points", [])
            if len(pts) >= 3:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(mask, [pts_arr], color=1)
        mask = mask.astype(bool)
    else:
        # Empty mask (e.g., after clear all)
        mask = np.zeros((height, width), dtype=bool)

    cfg = _cfg()
    try:
        basePathIdx = meta["basePathIdx"]
        camera_num = camera_number(meta["camera"])
        mask_path = cfg.get_mask_path(camera_num, basePathIdx)
    except Exception:
        return jsonify({"error": "invalid or missing meta fields"}), 400

    # Refuse before writing: a mask of another size than this camera's images
    # would be refused at every load anyway (MaskError is a 400, errors.py).
    require_mask_fits_camera(mask, cfg, camera_num, "being saved")

    # Ensure storage directory exists (especially for .set files with per-file storage)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    save_mask_to_mat(mask_path, mask, np.asarray(polygons))

    true_count = int(mask.sum())
    return jsonify(
        {
            "status": "ok",
            "shape": [height, width],
            "true_count": true_count,
            "fraction_true": true_count / (width * height),
            "meta": meta,
        }
    )


@masking_bp.route("/mask_status", methods=["GET"])
def mask_status():
    """Whether PIV can use this camera's mask on this dataset, without running it.

    Query params (both required):
      - basepath_idx: dataset index into ``paths.source_paths``
      - camera: camera number, ``2`` or ``Cam2``

    Returns 200 with ``{masking_enabled, mode, ok, message}``. ``ok`` is the
    outcome of ``load_mask_for_camera``, the check PIV and Run PIV make, and
    ``message`` is its MaskError text when refused, else null. With masking
    off nothing is needed, so ``ok`` is true. A bad index or camera is a 400.
    """
    cfg = _cfg()
    missing = [k for k in ("basepath_idx", "camera") if request.args.get(k) is None]
    if missing:
        return jsonify({"error": f"mask_status needs {' and '.join(missing)}"}), 400
    try:
        basepath_idx = int(request.args["basepath_idx"])
        camera = camera_number(request.args["camera"])
    except ValueError as e:
        return (
            jsonify({"error": f"basepath_idx and camera must be integers ({e})"}),
            400,
        )
    if not 0 <= basepath_idx < len(cfg.source_paths):
        return (
            jsonify(
                {
                    "error": f"basepath_idx {basepath_idx} is out of range: "
                    f"{len(cfg.source_paths)} source paths configured"
                }
            ),
            400,
        )

    try:
        load_mask_for_camera(camera, cfg, source_path_idx=basepath_idx)
        ok, message = True, None
    except MaskError as e:
        ok, message = False, str(e)
    return jsonify(
        {
            "masking_enabled": bool(cfg.masking_enabled),
            "mode": cfg.mask_mode,
            "ok": ok,
            "message": message,
        }
    )


@masking_bp.route("/load_mask", methods=["GET"])
def load_mask():
    """
    Loads a mask and polygon data from a .mat file.
    Query params:
      - path: full path to mask .mat file (preferred)
      - basepath_idx, camera: optional, used to construct path if 'path' not given
      - polygons_only: if 'true', skip loading the full mask array (faster for editor)
    Returns: { mask: [0|1,...], width, height, polygons: [...] }
    """
    cfg = _cfg()
    path = request.args.get("path", default=None, type=str)
    polygons_only = (
        request.args.get("polygons_only", default="false", type=str).lower() == "true"
    )
    camera = None
    if request.args.get("camera") is not None:
        try:
            camera = camera_number(request.args.get("camera"))
        except ValueError as e:
            return jsonify({"error": f"Could not resolve mask path: {e}"}), 400

    # Optionally reconstruct path if not provided
    if not path or not Path(path).exists():
        try:
            basepath_idx = int(request.args.get("basepath_idx", 0))
            base_paths = cfg.source_paths
            if basepath_idx < 0 or basepath_idx >= len(base_paths):
                return jsonify({"error": "basepath_idx out of range"}), 400
            if camera is None:
                raise ValueError("no camera given")
            path = str(cfg.get_mask_path(camera, basepath_idx))
        except Exception as e:
            return jsonify({"error": f"Could not resolve mask path: {e}"}), 400

    if not Path(path).exists():
        return jsonify({"error": f"Mask file not found: {path}"}), 404

    try:
        mask, polygons = read_mask_from_mat(path)
        if camera is not None:
            # A stale mask (another camera's, or drawn before this camera was
            # rotated) is refused, so its polygons never land on the wrong pixels.
            require_mask_fits_camera(np.asarray(mask), cfg, camera, path)

        def serialize_polygon(poly):
            return {
                "index": int(poly["index"]),
                "name": str(poly["name"]),
                "points": [list(map(float, pt)) for pt in poly["points"]],
            }

        polygons_serializable = [serialize_polygon(p) for p in polygons]
        mask_arr = np.asarray(mask)
        height, width = mask_arr.shape

        # If polygons_only is requested, skip the expensive mask flattening
        if polygons_only:
            return jsonify(
                {
                    "width": width,
                    "height": height,
                    "polygons": polygons_serializable,
                }
            )

        # Encode mask as base64 PNG (0/255 for proper PNG format) — ~5-20KB vs ~8MB JSON array
        mask_b64 = numpy_to_png_base64(mask_arr.astype(np.uint8) * 255)
        return jsonify(
            {
                "mask_image": mask_b64,
                "width": width,
                "height": height,
                "polygons": polygons_serializable,
            }
        )
    except MaskError as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        print("Exception in load_mask:", e)
        return jsonify({"error": f"Failed to load mask: {e}"}), 500
