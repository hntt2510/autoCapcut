from __future__ import annotations

from dataclasses import dataclass

from auto_capcut.models import CameraFrame, MotionTransform


@dataclass(frozen=True)
class CameraFrameValidation:
    valid: bool
    needs_reframing: bool = False
    reason: str = ""


def frame_aspect(frame: CameraFrame, source_size: tuple[int, int]) -> float:
    width, height = source_size
    return frame.width * width / (frame.height * height)


def canvas_aspect(canvas_size: tuple[int, int]) -> float:
    return canvas_size[0] / canvas_size[1]


def _rendered_dimensions(source_size: tuple[int, int], canvas_size: tuple[int, int], relative_scale: float) -> tuple[float, float]:
    source_width, source_height = source_size
    canvas_width, canvas_height = canvas_size
    fit = min(canvas_width / source_width, canvas_height / source_height)
    cover = max(canvas_width / source_width, canvas_height / source_height) / fit
    return source_width * fit * cover * relative_scale, source_height * fit * cover * relative_scale


def project_source_point_to_canvas(
    point: tuple[float, float],
    transform: MotionTransform,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[float, float]:
    """Project normalized source coordinates using pycapcut's half-canvas units."""
    canvas_width, canvas_height = canvas_size
    rendered_width, rendered_height = _rendered_dimensions(source_size, canvas_size, transform.relative_scale)
    return (
        canvas_width / 2 + (point[0] - 0.5) * rendered_width + transform.position_x * canvas_width / 2,
        canvas_height / 2 + (point[1] - 0.5) * rendered_height - transform.position_y * canvas_height / 2,
    )


def project_camera_frame_center(
    frame: CameraFrame,
    transform: MotionTransform,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[float, float]:
    return project_source_point_to_canvas(
        (frame.center_x, frame.center_y), transform, source_size, canvas_size
    )


def validate_camera_frame(
    frame: CameraFrame,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    *,
    pixel_tolerance: float = 1.0,
    relative_tolerance: float = 0.01,
) -> CameraFrameValidation:
    source_width, source_height = source_size
    if min(source_width, source_height, canvas_size[0], canvas_size[1]) <= 0:
        return CameraFrameValidation(False, reason="source and canvas dimensions must be positive")
    values = (frame.x, frame.y, frame.width, frame.height)
    if any(value != value for value in values) or min(values) < 0 or frame.width <= 0 or frame.height <= 0:
        return CameraFrameValidation(False, reason="frame must have positive normalized bounds")
    if frame.x + frame.width > 1.0 or frame.y + frame.height > 1.0:
        return CameraFrameValidation(False, reason="frame must remain inside the source image")
    expected_ratio = canvas_aspect(canvas_size)
    actual_pixel_width = frame.width * source_width
    actual_pixel_height = frame.height * source_height
    ratio_error_pixels = abs(actual_pixel_width - actual_pixel_height * expected_ratio)
    # Persisted normalized values originate from integer preview pixels.  Keep
    # a one-pixel floor plus a small relative tolerance for that round-trip.
    allowed_error = max(pixel_tolerance, actual_pixel_width * relative_tolerance)
    if ratio_error_pixels > allowed_error:
        return CameraFrameValidation(False, True, "frame aspect does not match project canvas")
    # At centered COVER, a viewport wider/taller than the source crop cannot be
    # represented without dropping below the required natural cover transform.
    cover_crop_width = min(1.0, canvas_size[0] / canvas_size[1] * source_height / source_width)
    cover_crop_height = min(1.0, canvas_size[1] / canvas_size[0] * source_width / source_height)
    if frame.width > cover_crop_width + 1e-9 or frame.height > cover_crop_height + 1e-9:
        return CameraFrameValidation(False, True, "frame is larger than the natural COVER viewport")
    return CameraFrameValidation(True)


def calculate_camera_transform(
    frame: CameraFrame,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> MotionTransform:
    """Turn an exact saved camera frame into a uniform CapCut transform."""
    validation = validate_camera_frame(frame, source_size, canvas_size)
    if not validation.valid:
        raise ValueError(validation.reason)
    source_width, source_height = source_size
    canvas_width, canvas_height = canvas_size
    cover_crop_width = min(1.0, canvas_width / canvas_height * source_height / source_width)
    cover_crop_height = min(1.0, canvas_height / canvas_width * source_width / source_height)
    relative_scale = max(cover_crop_width / frame.width, cover_crop_height / frame.height)
    rendered_width, rendered_height = _rendered_dimensions(source_size, canvas_size, relative_scale)
    # pycapcut position values are half-canvas units; positive Y is upward.
    transform = MotionTransform(
        relative_scale,
        -2 * (frame.center_x - 0.5) * rendered_width / canvas_width,
        2 * (frame.center_y - 0.5) * rendered_height / canvas_height,
    )
    projected = project_camera_frame_center(frame, transform, source_size, canvas_size)
    tolerance = 1e-6
    if abs(projected[0] - canvas_width / 2) > tolerance or abs(projected[1] - canvas_height / 2) > tolerance:
        raise ValueError("camera frame center cannot be projected to canvas center")
    return transform
