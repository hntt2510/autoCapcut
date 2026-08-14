from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from auto_capcut.models import (
    EffectDirection,
    EffectPhase,
    EffectPhaseType,
    MotionKeyframe,
    MotionPlan,
    MotionStrength,
    MotionTransform,
    TargetROI,
)
from auto_capcut.core.roi_resolver import RoiResolver
from auto_capcut.core.motion_resolution import infer_direction as _shared_infer_direction
from auto_capcut.core.camera_frame import calculate_camera_transform
from auto_capcut.core.roi_camera import calculate_roi_framing, log_camera_projection


@dataclass(frozen=True)
class StrengthValues:
    generic: float
    micro: float
    focus: float
    focus_move: float
    max_pan: float
    max_roi_zoom: float


STRENGTHS = {
    MotionStrength.VERY_SUBTLE.value.casefold(): StrengthValues(1.02, 1.015, 1.03, 1.04, 0.015, 2.0),
    MotionStrength.SUBTLE.value.casefold(): StrengthValues(1.035, 1.025, 1.05, 1.06, 0.03, 2.5),
    MotionStrength.NORMAL.value.casefold(): StrengthValues(1.04, 1.03, 1.06, 1.08, 0.05, 3.0),
}


def _strength(value: str) -> StrengthValues:
    return STRENGTHS.get(value.casefold(), STRENGTHS[MotionStrength.SUBTLE.value.casefold()])


def infer_direction(text: str) -> tuple[int, int]:
    value = text.casefold().replace("’", "'")
    horizontal = 0
    vertical = 0
    if re.search(r"\bleft\b.*\b(?:to|toward|through)\b.*\bright\b|toward the right|right panel", value):
        horizontal = -1
    elif re.search(r"\bright\b.*\b(?:to|toward|through)\b.*\bleft\b|toward the left|left panel", value):
        horizontal = 1
    if re.search(r"pan\s+(?:down|downward)|toward the bottom|lower", value):
        vertical = 1
    elif re.search(r"pan\s+(?:up|upward)|toward the top|upper", value):
        vertical = -1
    return horizontal, vertical


def infer_direction(text: str) -> tuple[int, int]:
    return _shared_infer_direction(text)


class MotionEngine:
    def __init__(self, canvas_width: int, canvas_height: int, strength: str = MotionStrength.SUBTLE.value, roi_resolver: RoiResolver | None = None) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.strength = _strength(strength)
        self.roi_resolver = roi_resolver

    @staticmethod
    def capcut_cover_scale(canvas_width: int, canvas_height: int, image_width: int, image_height: int) -> float:
        """Return COVER in CapCut's fit-normalized scale coordinates."""
        width_fit = canvas_width / image_width
        height_fit = canvas_height / image_height
        return max(width_fit, height_fit) / min(width_fit, height_fit)

    def plan_generic(self, mode: str, image_path: Path, image_width: int, image_height: int, duration_us: int, seed: str) -> MotionPlan:
        mode_key = mode.casefold()
        mode_key = {
            "random": "random light",
            "zoom in": "subtle zoom in",
            "zoom out": "subtle zoom out",
            "pan left": "subtle pan left",
            "pan right": "subtle pan right",
        }.get(mode_key, mode_key)
        if mode_key in {"none", "subtle zoom out"}:
            return MotionPlan((MotionKeyframe(0, MotionTransform()),))
        if mode_key == "random light":
            digest = hashlib.sha256(seed.encode("utf-8")).digest()[0] % 3
            mode_key = ("none", "subtle zoom in", "subtle pan left")[digest]
            if mode_key == "none":
                return MotionPlan((MotionKeyframe(0, MotionTransform()),))
        if mode_key == "subtle zoom in":
            return self._moving_plan(duration_us, MotionTransform(relative_scale=self.strength.generic), MotionTransform())
        if mode_key in {"subtle pan left", "subtle pan right"}:
            target_scale = self.strength.generic
            bound_x, _ = self._bounds(image_width, image_height, target_scale)
            amount = min(self.strength.max_pan, bound_x)
            direction = -1 if mode_key.endswith("left") else 1
            return self._moving_plan(
                duration_us,
                MotionTransform(target_scale, direction * amount, 0.0),
                MotionTransform(),
            )
        return MotionPlan((MotionKeyframe(0, MotionTransform()),))

    def plan_effect(self, effect: EffectDirection, image_path: Path, image_width: int, image_height: int, duration_us: int) -> MotionPlan:
        current = MotionTransform()
        keyframes: list[MotionKeyframe] = [MotionKeyframe(0, current)]
        warnings: list[str] = []
        manual_override = False
        for phase in effect.effects:
            start = min(duration_us, phase.local_start_us)
            end = min(duration_us, phase.local_end_us)
            if end <= start:
                continue
            phase_type = phase.type.value if isinstance(phase.type, EffectPhaseType) else phase.type
            if phase_type in {"HOLD", "SETTLE"}:
                keyframes.append(MotionKeyframe(start, current))
                keyframes.append(MotionKeyframe(end, current))
                continue
            if phase_type == "SUBTLE_ZOOM_IN":
                target = MotionTransform(max(current.relative_scale, self.strength.generic), current.position_x, current.position_y)
            elif phase_type == "SUBTLE_ZOOM_OUT":
                # COVER has no overflow.  Any existing pan must therefore
                # settle back to center while the relative scale approaches
                # 1.00; retaining it would expose the canvas edge.
                target = MotionTransform(1.0, 0.0, 0.0)
            elif phase_type in {"PAN_LEFT", "PAN_RIGHT"}:
                direction = -1 if phase_type.endswith("LEFT") else 1
                _, bound_y = self._bounds(image_width, image_height, current.relative_scale)
                bound_x, _ = self._bounds(image_width, image_height, current.relative_scale)
                target = MotionTransform(current.relative_scale, direction * min(self.strength.max_pan, bound_x), current.position_y)
            elif phase_type in {"FOCUS_ZOOM", "PAN_TO", "PULL_TO", "ALERT"}:
                roi = self.roi_resolver.resolve(image_path, phase.target_id, effect.image_index) if self.roi_resolver and phase.target_id else None
                target = self._roi_target(phase, roi, image_width, image_height, effect.image_index)
            else:
                target_scale = self._effect_scale(phase)
                roi = self.roi_resolver.resolve(image_path, phase.target_id, effect.image_index) if self.roi_resolver and phase.target_id else None
                target_x, target_y = self._target_position(
                    roi, f"{phase.target_id} {phase.instruction_text}", image_width, image_height,
                    target_scale, manual_override=manual_override,
                )
                target = MotionTransform(target_scale, target_x, target_y)
            if phase_type == "PULL_TO":
                mid_time = start + round((end - start) * 0.40)
                pull_scale = max(1.0, current.relative_scale * 0.82)
                midpoint = MotionTransform(
                    pull_scale,
                    current.position_x + (target.position_x - current.position_x) * 0.35,
                    current.position_y + (target.position_y - current.position_y) * 0.35,
                )
                keyframes.extend(self._smooth_phase(start, mid_time, current, midpoint))
                keyframes.extend(self._smooth_phase(mid_time, end, midpoint, target))
            else:
                keyframes.extend(self._smooth_phase(start, end, current, target))
            current = target
        keyframes.append(MotionKeyframe(duration_us, current))
        return MotionPlan(tuple(self._coalesce(keyframes)), tuple(warnings))

    def _roi_target(self, effect, roi: TargetROI | None, image_width: int, image_height: int, image_index: int) -> MotionTransform:
        if roi is None:
            return MotionTransform()
        effect_type = effect.type.value if isinstance(effect.type, EffectPhaseType) else effect.type
        try:
            transform = calculate_camera_transform(
                roi,
                (image_width, image_height),
                (self.canvas_width, self.canvas_height),
            )
            log_camera_projection(image_index, effect_type, effect.target_id, roi, transform, (image_width, image_height), (self.canvas_width, self.canvas_height))
            return transform
        except ValueError:
            # Direct legacy callers may still hand the engine a free-form ROI.
            # Project builds validate saved Camera Frames before reaching here;
            # this fallback only preserves the old in-memory API behavior.
            return calculate_roi_framing(
                roi,
                image_width,
                image_height,
                self.canvas_width,
                self.canvas_height,
                0.10,
                self.strength.max_roi_zoom,
            ).transform

    def _effect_scale(self, effect) -> float:
        effect_type = effect.type.value if isinstance(effect.type, EffectPhaseType) else effect.type
        default = {"MICRO FOCUS": self.strength.micro, "FOCUS": self.strength.focus, "FOCUS MOVE": self.strength.focus_move, "FOCUS_ZOOM": self.strength.focus, "PAN_TO": self.strength.focus, "PULL_TO": self.strength.focus, "ALERT": self.strength.focus}.get(effect_type, self.strength.generic)
        try:
            explicit = float(effect.params.get("zoom", default))
        except (ValueError, TypeError):
            explicit = default
        return max(1.0, min(default, explicit))

    def _bounds(self, image_width: int, image_height: int, relative_scale: float) -> tuple[float, float]:
        fit_scale = min(self.canvas_width / image_width, self.canvas_height / image_height)
        cover_scale = max(self.canvas_width / image_width, self.canvas_height / image_height) / fit_scale
        return (
            max(0.0, (image_width * fit_scale * cover_scale * relative_scale - self.canvas_width) / (2 * self.canvas_width)),
            max(0.0, (image_height * fit_scale * cover_scale * relative_scale - self.canvas_height) / (2 * self.canvas_height)),
        )

    def _target_position(
        self,
        roi: TargetROI | None,
        instruction: str,
        image_width: int,
        image_height: int,
        relative_scale: float,
        *,
        manual_override: bool = False,
    ) -> tuple[float, float]:
        fit_scale = min(self.canvas_width / image_width, self.canvas_height / image_height)
        cover_scale = max(self.canvas_width / image_width, self.canvas_height / image_height) / fit_scale
        rendered_width = image_width * fit_scale * cover_scale * relative_scale
        rendered_height = image_height * fit_scale * cover_scale * relative_scale
        bound_x = max(0.0, (rendered_width - self.canvas_width) / (2 * self.canvas_width))
        bound_y = max(0.0, (rendered_height - self.canvas_height) / (2 * self.canvas_height))
        if roi:
            target_x = -(roi.center_x - 0.5) * rendered_width / self.canvas_width * 0.35
            target_y = (roi.center_y - 0.5) * rendered_height / self.canvas_height * 0.35
        else:
            horizontal, vertical = infer_direction(instruction)
            target_x = horizontal * min(self.strength.max_pan, bound_x)
            target_y = vertical * min(self.strength.max_pan, bound_y)
        return max(-bound_x, min(bound_x, target_x)), max(-bound_y, min(bound_y, target_y))

    @staticmethod
    def _smooth_phase(start: int, end: int, source: MotionTransform, target: MotionTransform) -> list[MotionKeyframe]:
        length = end - start
        keys: list[MotionKeyframe] = []
        for fraction in (0.0, 0.25, 0.75, 1.0):
            smooth = fraction * fraction * (3 - 2 * fraction)
            transform = MotionTransform(
                source.relative_scale + (target.relative_scale - source.relative_scale) * smooth,
                source.position_x + (target.position_x - source.position_x) * smooth,
                source.position_y + (target.position_y - source.position_y) * smooth,
            )
            keys.append(MotionKeyframe(start + round(length * fraction), transform))
        return keys

    @staticmethod
    def _moving_plan(duration_us: int, target: MotionTransform, source: MotionTransform) -> MotionPlan:
        keys = MotionEngine._smooth_phase(0, duration_us, source, target)
        return MotionPlan(tuple(MotionEngine._coalesce(keys)))

    @staticmethod
    def _coalesce(keyframes: list[MotionKeyframe]) -> list[MotionKeyframe]:
        result: list[MotionKeyframe] = []
        for keyframe in sorted(keyframes, key=lambda item: item.local_time_us):
            if result and result[-1].local_time_us == keyframe.local_time_us:
                result[-1] = keyframe
            else:
                result.append(keyframe)
        return result
