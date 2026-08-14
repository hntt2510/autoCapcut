from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class AudioMode(str, Enum):
    SINGLE = "single"
    FOLDER = "folder"


class MotionStrength(str, Enum):
    VERY_SUBTLE = "Very Subtle"
    SUBTLE = "Subtle"
    NORMAL = "Normal"


class RoiRequirement(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True)
class EffectDefinition:
    type: str
    roi_requirement: RoiRequirement

    @property
    def requires_roi(self) -> bool:
        return self.roi_requirement is RoiRequirement.REQUIRED


EFFECT_REGISTRY: dict[str, EffectDefinition] = {
    name: EffectDefinition(name, requirement)
    for name, requirement in {
        "HOLD": RoiRequirement.NONE,
        "SUBTLE_ZOOM_IN": RoiRequirement.NONE,
        "SUBTLE_ZOOM_OUT": RoiRequirement.NONE,
        "PAN_LEFT": RoiRequirement.NONE,
        "PAN_RIGHT": RoiRequirement.NONE,
        "SETTLE": RoiRequirement.NONE,
        "FOCUS_ZOOM": RoiRequirement.REQUIRED,
        "PAN_TO": RoiRequirement.REQUIRED,
        "PULL_TO": RoiRequirement.REQUIRED,
        "ALERT": RoiRequirement.REQUIRED,
        "MICRO FOCUS": RoiRequirement.OPTIONAL,
        "FOCUS": RoiRequirement.OPTIONAL,
        "FOCUS MOVE": RoiRequirement.OPTIONAL,
    }.items()
}


class EffectPhaseType(str, Enum):
    HOLD = "HOLD"
    SUBTLE_ZOOM_IN = "SUBTLE_ZOOM_IN"
    SUBTLE_ZOOM_OUT = "SUBTLE_ZOOM_OUT"
    PAN_LEFT = "PAN_LEFT"
    PAN_RIGHT = "PAN_RIGHT"
    FOCUS_ZOOM = "FOCUS_ZOOM"
    PAN_TO = "PAN_TO"
    PULL_TO = "PULL_TO"
    ALERT = "ALERT"
    MICRO_FOCUS = "MICRO FOCUS"
    FOCUS = "FOCUS"
    FOCUS_MOVE = "FOCUS MOVE"
    SETTLE = "SETTLE"


@dataclass(frozen=True)
class CameraFrame:
    """A normalized, aspect-locked camera viewport selected by the user."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


# Compatibility name retained for existing source callers and sidecars.
TargetROI = CameraFrame


@dataclass(frozen=True)
class RoiTarget:
    image_index: int
    target_id: str
    effect_types: tuple[str, ...] = ()
    requirement: RoiRequirement = RoiRequirement.REQUIRED


@dataclass(frozen=True)
class VisualEffect:
    type: str | EffectPhaseType
    local_start_us: int
    local_end_us: int
    target_id: str = ""
    params: dict[str, str] = field(default_factory=dict)
    instruction_text: str = ""

    @property
    def definition(self) -> EffectDefinition:
        key = self.type.value if isinstance(self.type, Enum) else self.type
        return EFFECT_REGISTRY[key]

    @property
    def roi_required(self) -> bool:
        return self.definition.requires_roi

    @property
    def duration_us(self) -> int:
        return self.local_end_us - self.local_start_us


@dataclass(frozen=True)
class EffectCue:
    image_index: int
    global_start_us: int
    global_end_us: int
    target_text: str
    effects: tuple[VisualEffect, ...]
    transition_out: str | None = None
    declared_duration_us: int | None = None

    @property
    def duration_us(self) -> int:
        return self.global_end_us - self.global_start_us

    @property
    def required_roi_targets(self) -> tuple[RoiTarget, ...]:
        return _dedupe_targets(self, RoiRequirement.REQUIRED)

    @property
    def optional_roi_targets(self) -> tuple[RoiTarget, ...]:
        return _dedupe_targets(self, RoiRequirement.OPTIONAL)

    @property
    def phases(self) -> tuple[VisualEffect, ...]:
        return self.effects


def _dedupe_targets(cue: EffectCue, requirement: RoiRequirement) -> tuple[RoiTarget, ...]:
    found: dict[str, RoiTarget] = {}
    for effect in cue.effects:
        if effect.definition.roi_requirement is not requirement or not effect.target_id:
            continue
        current = found.get(effect.target_id)
        if current is None:
            found[effect.target_id] = RoiTarget(cue.image_index, effect.target_id, (effect.type,), requirement)
        elif effect.type not in current.effect_types:
            found[effect.target_id] = RoiTarget(cue.image_index, current.target_id, current.effect_types + (effect.type,), requirement)
    return tuple(found.values())


# Compatibility name for callers that used the old model.
EffectPhase = VisualEffect
EffectDirection = EffectCue


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


RESOLUTIONS = {
    "1920x1080": Resolution(1920, 1080),
    "1080x1920": Resolution(1080, 1920),
    "1080x1080": Resolution(1080, 1080),
}


@dataclass(frozen=True)
class ImageTiming:
    index: int
    start_us: int
    end_us: int

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_us: int
    end_us: int
    text: str

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class MotionTransform:
    relative_scale: float = 1.0
    position_x: float = 0.0
    position_y: float = 0.0


# Public semantic alias for callers that describe camera work explicitly.
CameraTransform = MotionTransform


@dataclass(frozen=True)
class CameraFraming:
    """Aspect-correct ROI framing result in source-normalized coordinates."""

    original_roi: TargetROI
    adjusted_roi: TargetROI
    transform: MotionTransform
    clamped: bool = False

    @property
    def target_transform(self) -> MotionTransform:
        return self.transform

    @property
    def original_rect(self) -> TargetROI:
        return self.original_roi

    @property
    def adjusted_rect(self) -> TargetROI:
        return self.adjusted_roi

    @property
    def camera_transform(self) -> MotionTransform:
        return self.transform


@dataclass(frozen=True)
class MotionKeyframe:
    local_time_us: int
    transform: MotionTransform


@dataclass(frozen=True)
class MotionPlan:
    keyframes: tuple[MotionKeyframe, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlertOverlayPlan:
    image_index: int
    target_id: str
    start_us: int
    end_us: int
    path: Path
    style: str = "red_warning"
    pulse: bool = False


@dataclass
class ProjectConfig:
    project_name: str = ""
    resolution: Resolution = field(default_factory=lambda: RESOLUTIONS["1920x1080"])
    image_folders: list[Path] = field(default_factory=list)
    audio_mode: AudioMode = AudioMode.SINGLE
    audio_path: Optional[Path] = None
    subtitle_srt: Optional[Path] = None
    import_subtitles: bool = False
    use_image_timing: bool = False
    image_timing_srt: Optional[Path] = None
    motion_enabled: bool = True
    motion_mode: str = "Random Light"
    motion_strength: str = MotionStrength.SUBTLE.value
    effect_direction_srt: Optional[Path] = None
    transition_enabled: bool = True
    transition_type: str = "Blur"
    transition_duration_us: int = 300_000
    logo_enabled: bool = False
    logo_path: Optional[Path] = None
    music_enabled: bool = False
    music_folder: Optional[Path] = None
    music_volume: float = 0.15
    draft_folder: Optional[Path] = None


@dataclass(frozen=True)
class ProjectJob:
    name: str
    images: tuple[Path, ...]
    audio_path: Path
    subtitle_srt: Optional[Path]
    image_timing_srt: Optional[Path]
    config: ProjectConfig


@dataclass(frozen=True)
class BuildResult:
    project_name: str
    project_path: Path
    duration_us: int
    warnings: tuple[str, ...] = ()


ProgressCallback = Callable[[int, str], None]
