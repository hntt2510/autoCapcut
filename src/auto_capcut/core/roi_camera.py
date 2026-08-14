from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from auto_capcut.models import CameraFraming, MotionTransform, TargetROI
from auto_capcut.core.camera_frame import project_camera_frame_center


def _camera_logger() -> logging.Logger:
    logger = logging.getLogger("auto_capcut.roi_camera")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    try:
        root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AutoCapCut" / "logs"
        root.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            root / "roi-camera.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        # Diagnostics must never make a project fail.
        logger.addHandler(logging.NullHandler())
    return logger


def log_roi_effect(image_index: int, effect_type: str, target_id: str, framing: CameraFraming) -> None:
    """Write a redacted, structured camera diagnostic without affecting builds."""
    _camera_logger().info(
        "Image %03d Effect=%s Target=%s original_roi=(%.6f,%.6f,%.6f,%.6f) "
        "adjusted_roi=(%.6f,%.6f,%.6f,%.6f) target_camera=(scale=%.6f,x=%.6f,y=%.6f) clamped=%s",
        image_index, effect_type, target_id,
        framing.original_roi.x, framing.original_roi.y, framing.original_roi.width, framing.original_roi.height,
        framing.adjusted_roi.x, framing.adjusted_roi.y, framing.adjusted_roi.width, framing.adjusted_roi.height,
        framing.transform.relative_scale, framing.transform.position_x, framing.transform.position_y, framing.clamped,
    )


def log_camera_projection(
    image_index: int,
    effect_type: str,
    target_id: str,
    frame: TargetROI,
    transform: MotionTransform,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> None:
    projected_x, projected_y = project_camera_frame_center(frame, transform, source_size, canvas_size)
    expected_x, expected_y = canvas_size[0] / 2, canvas_size[1] / 2
    _camera_logger().info(
        "Image %03d Effect=%s Target=%s frame=(x=%.8f,y=%.8f,w=%.8f,h=%.8f) "
        "source=(%d,%d) center_source=(%.3f,%.3f) scale=%.8f position=(%.8f,%.8f) "
        "projected=(%.6f,%.6f) expected=(%.6f,%.6f) error=(dx=%.6f,dy=%.6f)",
        image_index, effect_type, target_id, frame.x, frame.y, frame.width, frame.height,
        source_size[0], source_size[1], frame.center_x * source_size[0], frame.center_y * source_size[1],
        transform.relative_scale, transform.position_x, transform.position_y,
        projected_x, projected_y, expected_x, expected_y, projected_x - expected_x, projected_y - expected_y,
    )


def _validate_roi(roi: TargetROI) -> None:
    values = (roi.x, roi.y, roi.width, roi.height)
    if any(value != value for value in values) or min(values) < 0 or roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI must have positive normalized bounds")
    if roi.x + roi.width > 1.0 or roi.y + roi.height > 1.0:
        raise ValueError("ROI must remain inside normalized image bounds")


def _centered_rect(center_x: float, center_y: float, width: float, height: float) -> TargetROI:
    x = max(0.0, min(1.0 - width, center_x - width / 2.0))
    y = max(0.0, min(1.0 - height, center_y - height / 2.0))
    return TargetROI(x, y, width, height)


def _cover_scale(source_width: int, source_height: int, canvas_width: int, canvas_height: int) -> float:
    width_fit = canvas_width / source_width
    height_fit = canvas_height / source_height
    return max(width_fit, height_fit) / min(width_fit, height_fit)


def _rendered_size(source_width: int, source_height: int, canvas_width: int, canvas_height: int, relative_scale: float) -> tuple[float, float]:
    fit = min(canvas_width / source_width, canvas_height / source_height)
    cover = _cover_scale(source_width, source_height, canvas_width, canvas_height)
    return source_width * fit * cover * relative_scale, source_height * fit * cover * relative_scale


def _clamp_camera_transform_to_cover(
    transform: MotionTransform,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[MotionTransform, bool]:
    """Clamp a uniform camera transform so the fitted source always covers canvas."""
    scale = max(1.0, float(transform.relative_scale))
    rendered_width, rendered_height = _rendered_size(source_width, source_height, canvas_width, canvas_height, scale)
    # pycapcut positions are expressed in half-canvas units.
    bound_x = max(0.0, (rendered_width - canvas_width) / (2.0 * canvas_width))
    bound_y = max(0.0, (rendered_height - canvas_height) / (2.0 * canvas_height))
    clamped = scale != transform.relative_scale
    x = max(-bound_x, min(bound_x, transform.position_x))
    y = max(-bound_y, min(bound_y, transform.position_y))
    clamped = clamped or x != transform.position_x or y != transform.position_y
    return MotionTransform(scale, x, y), clamped


def clamp_camera_transform_to_cover(
    transform: MotionTransform,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
) -> MotionTransform:
    """Return a cover-safe transform (uniform scale and bounded translation)."""
    result, _ = _clamp_camera_transform_to_cover(
        transform, source_width, source_height, canvas_width, canvas_height
    )
    return result


def calculate_roi_framing(
    roi: TargetROI,
    source_width: int | tuple[int, int] | None = None,
    source_height: int | tuple[int, int] | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    framing_padding: float = 0.10,
    max_zoom: float = 2.5,
    *,
    source_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> CameraFraming:
    """Convert a normalized semantic ROI into an aspect-correct camera target."""
    # Accept both the explicit-width API and the compact ``(w, h)`` API.
    if source_size is not None:
        source_width, source_height = source_size
    if canvas_size is not None:
        canvas_width, canvas_height = canvas_size
    elif isinstance(source_width, (tuple, list)) and isinstance(source_height, (tuple, list)):
        # ``calculate_roi_framing(roi, (sw, sh), (cw, ch), padding, max_zoom)``
        source_dims = source_width
        canvas_dims = source_height
        old_padding = canvas_width
        old_max_zoom = canvas_height
        source_width, source_height = source_dims
        canvas_width, canvas_height = canvas_dims
        if old_padding is not None:
            framing_padding = float(old_padding)
        if old_max_zoom is not None:
            max_zoom = float(old_max_zoom)
    if None in (source_width, source_height, canvas_width, canvas_height):
        raise TypeError("source and canvas dimensions are required")
    source_width = int(source_width)  # type: ignore[arg-type]
    source_height = int(source_height)  # type: ignore[arg-type]
    canvas_width = int(canvas_width)  # type: ignore[arg-type]
    canvas_height = int(canvas_height)  # type: ignore[arg-type]
    if min(source_width, source_height, canvas_width, canvas_height) <= 0:
        raise ValueError("source and canvas dimensions must be positive")
    _validate_roi(roi)
    if not 0.0 <= framing_padding <= 0.50:
        raise ValueError("framing padding must be between 0 and 0.50")
    if max_zoom < 1.0:
        raise ValueError("max zoom must be at least 1.0")
    source_aspect = source_width / source_height
    canvas_aspect = canvas_width / canvas_height
    padded_width = min(1.0, roi.width * (1.0 + 2.0 * framing_padding))
    padded_height = min(1.0, roi.height * (1.0 + 2.0 * framing_padding))
    width = max(padded_width, padded_height * canvas_aspect / source_aspect)
    height = max(padded_height, padded_width * source_aspect / canvas_aspect)
    if width > 1.0:
        width = 1.0
        height = min(1.0, width * source_aspect / canvas_aspect)
    if height > 1.0:
        height = 1.0
        width = min(1.0, height * canvas_aspect / source_aspect)
    adjusted = _centered_rect(roi.center_x, roi.center_y, width, height)
    fit = min(canvas_width / source_width, canvas_height / source_height)
    cover = _cover_scale(source_width, source_height, canvas_width, canvas_height)
    required_scale_x = canvas_width / (source_width * fit * cover * adjusted.width)
    required_scale_y = canvas_height / (source_height * fit * cover * adjusted.height)
    # Uniform scaling must satisfy both viewport dimensions.  Using the larger
    # requirement prevents a tall/wide ROI from leaving uncovered canvas.
    required_scale = max(required_scale_x, required_scale_y)
    relative_scale = max(1.0, min(float(max_zoom), required_scale))
    rendered_width, rendered_height = _rendered_size(source_width, source_height, canvas_width, canvas_height, relative_scale)
    transform = MotionTransform(
        relative_scale,
        -(adjusted.center_x - 0.5) * rendered_width / canvas_width,
        (adjusted.center_y - 0.5) * rendered_height / canvas_height,
    )
    transform, clamped = _clamp_camera_transform_to_cover(transform, source_width, source_height, canvas_width, canvas_height)
    _camera_logger().info(
        "ROI framing original=(%.6f,%.6f,%.6f,%.6f) adjusted=(%.6f,%.6f,%.6f,%.6f) "
        "target=(scale=%.6f,x=%.6f,y=%.6f) clamped=%s",
        roi.x, roi.y, roi.width, roi.height, adjusted.x, adjusted.y, adjusted.width, adjusted.height,
        transform.relative_scale, transform.position_x, transform.position_y, clamped,
    )
    return CameraFraming(roi, adjusted, transform, clamped)


def medium_roi_framing(
    roi: TargetROI,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
    max_zoom: float = 2.5,
    occupancy: float = 0.65,
) -> CameraFraming:
    """Return a less-tight framing for ALERT emphasis."""
    base = calculate_roi_framing(roi, source_width, source_height, canvas_width, canvas_height, 0.10, max_zoom)
    scale = max(1.0, base.transform.relative_scale * max(0.55, min(1.0, occupancy)))
    transform, clamped = _clamp_camera_transform_to_cover(
        MotionTransform(scale, base.transform.position_x, base.transform.position_y),
        source_width, source_height, canvas_width, canvas_height,
    )
    return CameraFraming(base.original_roi, base.adjusted_roi, transform, base.clamped or clamped)
