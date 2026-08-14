from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class DrawMode(str, Enum):
    BASIC = "basic_draw"
    ADVANCED = "advanced_draw"


class DrawStyle(str, Enum):
    V1 = "v1"
    V2 = "v2"


class FinalRevealMode(str, Enum):
    LINE_ONLY = "line_only"
    LINE_THEN_COLOR = "line_then_color"
    ORIGINAL_REVEAL = "original_reveal"


class TextMode(str, Enum):
    KEEP = "keep"
    SIMPLIFIED = "simplified"
    TRACE = "trace"


class DrawObjectEffect(str, Enum):
    DRAW = "draw"
    SLIDE_IN = "slide_in"
    DROP_IN = "drop_in"
    PUSH_IN = "push_in"
    TOSS_IN = "toss_in"
    POP_IN = "pop_in"


class DrawObjectDirection(str, Enum):
    AUTO = "auto"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class ObjectEffectDurationMode(str, Enum):
    INHERIT = "inherit"
    AUTO = "auto"
    FIXED = "fixed"


class DrawActionType(str, Enum):
    DRAW = "DRAW"
    FOCUS = "FOCUS"
    PAN_TO = "PAN_TO"
    PULL_TO = "PULL_TO"
    FULL_VIEW = "FULL_VIEW"
    SETTLE = "SETTLE"


@dataclass(frozen=True)
class DrawAction:
    type: DrawActionType
    start_us: int
    end_us: int
    params: dict[str, str] = field(default_factory=dict)
    instruction: str = ""

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class DrawImagePlan:
    image_index: int
    image_name: str | None
    start_us: int
    end_us: int
    mode: DrawMode
    style: DrawStyle
    objects: str
    actions: tuple[DrawAction, ...]
    object_effects: tuple["ObjectEffectOverride", ...] = ()

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def draw_action(self) -> DrawAction:
        return next(action for action in self.actions if action.type is DrawActionType.DRAW)


@dataclass(frozen=True)
class DrawEffectFile:
    path: Path
    images: tuple[DrawImagePlan, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectEffectOverride:
    target: str
    effect: str
    direction: str = "auto"
    duration_us: int | None = None
    pause_after_us: int | None = None
    duration_mode: str = ObjectEffectDurationMode.INHERIT.value


@dataclass(frozen=True)
class NormalizedRect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class SceneObject:
    id: str
    type: str
    box: NormalizedRect
    camera_frame: NormalizedRect | None = None
    render_effect: str = DrawObjectEffect.DRAW.value
    direction: str = DrawObjectDirection.AUTO.value
    duration_us: int | None = None
    pause_after_us: int | None = None
    behavior_fields_present: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SceneImage:
    filename: str
    source_size: tuple[int, int]
    objects: tuple[SceneObject, ...]
    draw_order: tuple[str, ...]
    source_sha256: str | None = None

    @property
    def object_map(self) -> dict[str, SceneObject]:
        return {item.id: item for item in self.objects}


@dataclass(frozen=True)
class SceneDocument:
    schema_version: int
    images: dict[str, SceneImage]
    path: Path | None = None


@dataclass(frozen=True)
class DrawProjectConfig:
    image_folder: Path
    effect_file: Path
    output_folder: Path
    scene_file: Path | None = None
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    remove_background: bool = False
    fallback_basic: bool = True
    advanced_diagnostics: bool = False


ProgressCallback = Callable[[int, str], None]
