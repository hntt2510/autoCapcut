from __future__ import annotations

import re
from pathlib import Path

from auto_capcut.core.draw_models import (
    DrawAction,
    DrawActionType,
    DrawEffectFile,
    CameraAfterDirective,
    DrawAction,
    DrawActionType,
    DrawEffectFile,
    DrawImagePlan,
    DrawMode,
    DrawStyle,
    ObjectEffectDurationMode,
    ObjectEffectOverride,
)
from auto_capcut.core.errors import DrawParseError
from auto_capcut.core.srt_parser import parse_srt

_HEADER = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_SPACE_HEADER = re.compile(r"^(MODE|STYLE|OBJECTS)\s+(.+?)\s*$", re.IGNORECASE)
_BLOCK_HEADER = re.compile(r"^Image\s+(\d+)\s+DRAW$", re.IGNORECASE)
_OBJECT_EFFECT = re.compile(r"^OBJECT_EFFECT\s*:?\s*(.*)$", re.IGNORECASE)
_CAMERA_AFTER = re.compile(r"^CAMERA_AFTER\s*:?\s*(.*)$", re.IGNORECASE)
_ACTION = re.compile(
    r"^([A-Z_]+)\s+(\d+(?:\.\d+)?)s\s*(?:-|–|—)\s*(\d+(?:\.\d+)?)s\s*:\s*(.*)$",
    re.IGNORECASE,
)
_PARAM = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s]+)")
_KNOWN = {item.value for item in DrawActionType}


def _seconds(value: str, label: str) -> int:
    try:
        result = float(value)
    except ValueError as exc:
        raise DrawParseError(f"{label} must be a number of seconds") from exc
    if result < 0:
        raise DrawParseError(f"{label} cannot be negative")
    return round(result * 1_000_000)


def _properties(text: str, image_index: int, action_name: str) -> dict[str, str]:
    params: dict[str, str] = {}
    cursor = 0
    for match in _PARAM.finditer(text):
        if text[cursor:match.start()].strip(" ,"):
            raise DrawParseError(f"Image {image_index} {action_name}: malformed parameter list")
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        params[match.group(1).casefold()] = value
        cursor = match.end()
    if text[cursor:].strip(" ,"):
        raise DrawParseError(f"Image {image_index} {action_name}: malformed parameter list")
    return params


def _parse_action(line: str, image_index: int) -> DrawAction:
    match = _ACTION.match(line.strip())
    if not match:
        raise DrawParseError(f"Image {image_index}: malformed action: {line.strip()}")
    name = match.group(1).upper()
    if name not in _KNOWN:
        raise DrawParseError(f"Image {image_index}: unsupported action {name}")
    start = _seconds(match.group(2), f"Image {image_index} {name} start")
    end = _seconds(match.group(3), f"Image {image_index} {name} end")
    if end <= start:
        raise DrawParseError(f"Image {image_index} {name}: end must be after start")
    instruction = match.group(4).strip()
    params = {} if name == "SETTLE" and instruction.casefold() in {"", "hold final composition"} else _properties(instruction, image_index, name)
    allowed = {
        "DRAW": {"order", "pause_each", "text", "final", "direction", "unmatched"},
        "FOCUS": {"target", "framing", "easing"},
        "PAN_TO": {"target", "framing", "easing"},
        "PULL_TO": {"target", "framing", "easing"},
        "FULL_VIEW": {"easing"},
        "SETTLE": set(),
    }[name]
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise DrawParseError(f"Image {image_index} {name}: unsupported parameter(s): {', '.join(unknown)}")
    if name == "DRAW":
        params.setdefault("direction", "auto")
        params.setdefault("unmatched", "last")
        try:
            pause = float(params.get("pause_each", "0"))
        except ValueError as exc:
            raise DrawParseError(f"Image {image_index} DRAW pause_each must be numeric") from exc
        if pause < 0:
            raise DrawParseError(f"Image {image_index} DRAW pause_each cannot be negative")
        if params.get("text", "keep").casefold() not in {"keep", "simplified", "trace"}:
            raise DrawParseError(f"Image {image_index} DRAW text must be keep, simplified, or trace")
        if params.get("final", "line_then_color").casefold() not in {"line_only", "line_then_color", "original_reveal"}:
            raise DrawParseError(f"Image {image_index} DRAW final mode is invalid")
        if params.get("direction", "auto").casefold() not in {"auto", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}:
            raise DrawParseError(f"Image {image_index} DRAW direction is invalid")
        if params.get("unmatched", "last").casefold() not in {"last", "first", "ignore"}:
            raise DrawParseError(f"Image {image_index} DRAW unmatched policy is invalid")
    elif name in {"FOCUS", "PAN_TO", "PULL_TO"}:
        if not params.get("target"):
            raise DrawParseError(f"Image {image_index} {name} requires target")
        if params.get("framing", "camera_frame").casefold() not in {"camera_frame", "object_box"}:
            raise DrawParseError(f"Image {image_index} {name} framing is invalid")
    if params.get("easing", "ease_in_out").casefold() not in {"linear", "ease_in_out"}:
        raise DrawParseError(f"Image {image_index} {name} easing is invalid")
    return DrawAction(DrawActionType(name), start, end, params, instruction)


def _parse_object_effect(line: str, image_index: int) -> ObjectEffectOverride:
    match = _OBJECT_EFFECT.fullmatch(line.strip())
    if not match:
        raise DrawParseError(f"Image {image_index}: malformed OBJECT_EFFECT directive")
    params = _properties(match.group(1).strip(), image_index, "OBJECT_EFFECT")
    allowed = {"target", "effect", "direction", "duration", "pause_after"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise DrawParseError(f"Image {image_index} OBJECT_EFFECT: unsupported parameter(s): {', '.join(unknown)}")
    target = params.get("target", "").strip()
    effect = params.get("effect", "").casefold()
    if not target:
        raise DrawParseError(f"Image {image_index} OBJECT_EFFECT requires target")
    if effect not in {"draw", "slide_in", "drop_in", "push_in", "toss_in", "pop_in"}:
        raise DrawParseError(f"Image {image_index} OBJECT_EFFECT target {target}: invalid effect {params.get('effect', '')}")
    direction = params.get("direction", "auto").casefold()
    if direction not in {"auto", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"}:
        raise DrawParseError(f"Image {image_index} OBJECT_EFFECT target {target}: invalid direction {params.get('direction', '')}")
    duration_us = None
    duration_mode = ObjectEffectDurationMode.INHERIT.value
    if "duration" in params:
        if params["duration"].casefold() == ObjectEffectDurationMode.AUTO.value:
            duration_mode = ObjectEffectDurationMode.AUTO.value
        else:
            duration_us = _seconds(params["duration"], f"Image {image_index} OBJECT_EFFECT {target} duration")
            if duration_us <= 0:
                raise DrawParseError(f"Image {image_index} OBJECT_EFFECT {target} duration must be positive")
            duration_mode = ObjectEffectDurationMode.FIXED.value
    pause_after_us = None
    if "pause_after" in params:
        pause_after_us = _seconds(params["pause_after"], f"Image {image_index} OBJECT_EFFECT {target} pause_after")
    return ObjectEffectOverride(target, effect, direction, duration_us, pause_after_us, duration_mode)


def _parse_camera_after(line: str, image_index: int) -> CameraAfterDirective:
    match = _CAMERA_AFTER.fullmatch(line.strip())
    if not match:
        raise DrawParseError(f"Image {image_index}: malformed CAMERA_AFTER directive")
    params = _properties(match.group(1).strip(), image_index, "CAMERA_AFTER")
    allowed = {"object", "action", "target", "duration", "hold", "framing", "easing"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise DrawParseError(f"Image {image_index} CAMERA_AFTER: unsupported parameter(s): {', '.join(unknown)}")
    object_id = params.get("object", "").strip()
    action = params.get("action", "").casefold()
    if not object_id:
        raise DrawParseError(f"Image {image_index} CAMERA_AFTER requires object")
    if action not in {"focus", "pan_to", "pull_to", "full_view"}:
        raise DrawParseError(f"Image {image_index} CAMERA_AFTER object {object_id}: invalid action {params.get('action', '')}")
    target = params.get("target", "").strip()
    if not target and action in {"focus", "pan_to", "pull_to"}:
        target = object_id
    duration_us = None
    duration_mode = "auto"
    if "duration" in params:
        if params["duration"].casefold() == "auto":
            duration_mode = "auto"
        else:
            duration_us = _seconds(params["duration"], f"Image {image_index} CAMERA_AFTER {object_id} duration")
            if duration_us <= 0:
                raise DrawParseError(f"Image {image_index} CAMERA_AFTER {object_id} duration must be positive")
            duration_mode = "fixed"
    hold_us = 0
    if "hold" in params:
        hold_us = _seconds(params["hold"], f"Image {image_index} CAMERA_AFTER {object_id} hold")
    framing = params.get("framing", "camera_frame").casefold()
    if framing not in {"camera_frame", "object_frame", "object_box"}:
        raise DrawParseError(f"Image {image_index} CAMERA_AFTER {object_id} framing is invalid")
    easing = params.get("easing", "ease_in_out").casefold()
    if easing not in {"ease_in_out", "linear"}:
        raise DrawParseError(f"Image {image_index} CAMERA_AFTER {object_id} easing is invalid")
    return CameraAfterDirective(object_id, action, target, duration_us, duration_mode, hold_us, framing, easing)


def parse_draw_effect(path: str | Path) -> DrawEffectFile:
    source = Path(path)
    try:
        cues = parse_srt(source)
    except Exception as exc:
        if isinstance(exc, DrawParseError):
            raise
        raise DrawParseError(str(exc)) from exc
    if not cues:
        raise DrawParseError("Draw effect file is empty")
    plans: list[DrawImagePlan] = []
    warnings: list[str] = []
    previous_end = 0
    for cue in cues:
        image_index = cue.index + 1
        if cue.start_us != previous_end:
            raise DrawParseError(f"Image {image_index}: SRT blocks must be contiguous and start at zero")
        previous_end = cue.end_us
        headers: dict[str, str] = {}
        actions: list[DrawAction] = []
        object_effects: list[ObjectEffectOverride] = []
        camera_after: list[CameraAfterDirective] = []
        lines = [raw.strip() for raw in cue.text.splitlines() if raw.strip()]
        if lines:
            block_header = _BLOCK_HEADER.fullmatch(lines[0])
            if block_header:
                declared_image = int(block_header.group(1))
                if declared_image != image_index:
                    warnings.append(f"Image {image_index}: block header declares Image {declared_image} DRAW")
                lines = lines[1:]
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if match := (_HEADER.match(line) or _SPACE_HEADER.match(line)):
                key, value = match.group(1).casefold(), match.group(2).strip()
                if key in headers:
                    raise DrawParseError(f"Image {image_index}: duplicate header {key}")
                headers[key] = value
            elif _OBJECT_EFFECT.fullmatch(line):
                override = _parse_object_effect(line, image_index)
                if any(item.target == override.target for item in object_effects):
                    raise DrawParseError(f"Image {image_index}: duplicate OBJECT_EFFECT target {override.target}")
                object_effects.append(override)
            elif _CAMERA_AFTER.fullmatch(line):
                directive = _parse_camera_after(line, image_index)
                if any(item.object_id == directive.object_id for item in camera_after):
                    raise DrawParseError(f"Image {image_index}: duplicate CAMERA_AFTER for object {directive.object_id}")
                camera_after.append(directive)
            else:
                actions.append(_parse_action(line, image_index))
        unknown_headers = sorted(set(headers) - {"image", "mode", "style", "objects"})
        if unknown_headers:
            raise DrawParseError(f"Image {image_index}: unsupported header(s): {', '.join(unknown_headers)}")
        if headers.get("mode", "").casefold() not in {item.value for item in DrawMode}:
            raise DrawParseError(f"Image {image_index}: MODE must be basic_draw or advanced_draw")
        if headers.get("style", "").casefold() not in {item.value for item in DrawStyle}:
            raise DrawParseError(f"Image {image_index}: STYLE must be v1 or v2")
        mode = DrawMode(headers["mode"].casefold())
        style = DrawStyle(headers["style"].casefold())
        objects = headers.get("objects", "manual" if mode is DrawMode.ADVANCED else "auto").casefold()
        if objects not in {"manual", "auto"}:
            raise DrawParseError(f"Image {image_index}: OBJECTS must be manual or auto")
        expected_objects = "manual" if mode is DrawMode.ADVANCED else "auto"
        if objects != expected_objects:
            raise DrawParseError(f"Image {image_index}: {mode.value} requires OBJECTS={expected_objects}")
        if "image" in headers and not headers["image"]:
            raise DrawParseError(f"Image {image_index}: IMAGE cannot be empty")
        if len([action for action in actions if action.type is DrawActionType.DRAW]) != 1:
            raise DrawParseError(f"Image {image_index}: exactly one DRAW action is required")
        draw = next(action for action in actions if action.type is DrawActionType.DRAW)
        if draw.start_us != 0:
            raise DrawParseError(f"Image {image_index}: DRAW must begin at 0.00s")
        if not actions or max(action.end_us for action in actions) != cue.duration_us:
            raise DrawParseError(f"Image {image_index}: final action must end at the SRT cue duration")
        camera = [action for action in actions if action.type is not DrawActionType.DRAW]
        for left_index, left in enumerate(camera):
            for right in camera[left_index + 1 :]:
                if left.start_us < right.end_us and right.start_us < left.end_us:
                    raise DrawParseError(f"Image {image_index}: camera actions overlap ({left.type.value}, {right.type.value})")
        if object_effects and mode is not DrawMode.ADVANCED:
            raise DrawParseError(f"Image {image_index}: OBJECT_EFFECT directives require advanced_draw")
        if camera_after and mode is not DrawMode.ADVANCED:
            raise DrawParseError(f"Image {image_index}: CAMERA_AFTER directives require advanced_draw")
        plans.append(DrawImagePlan(image_index, headers.get("image"), cue.start_us, cue.end_us, mode, style, objects, tuple(actions), tuple(object_effects), tuple(camera_after)))
    return DrawEffectFile(source, tuple(plans), tuple(warnings))


# Naming parallel to the existing Effect Direction parser for callers that
# prefer to make the SRT format explicit.
parse_draw_effect_srt = parse_draw_effect
