from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from auto_capcut.core.errors import EffectDirectionError
from auto_capcut.core.srt_parser import parse_srt
from auto_capcut.models import EFFECT_REGISTRY, EffectCue, EffectDefinition, EffectPhaseType, RoiRequirement, VisualEffect

TOLERANCE_US = 50_000
_IMAGE_LINE = re.compile(r"^\s*Image\s+(\d+)\s+FX\b", re.IGNORECASE)
_TARGET_LINE = re.compile(r"^\s*Target\s*:\s*(.*)$", re.IGNORECASE)
_DURATION_LINE = re.compile(r"^\s*Duration\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$", re.IGNORECASE)
_TRANSITION_LINE = re.compile(r"^\s*Transition\s+out\s*:\s*(.*)$", re.IGNORECASE)
_EFFECT_LINE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 _-]*)\s+([0-9]+(?:\.[0-9]+)?)\s*s?\s*[^0-9A-Za-z\s]+\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*s?\s*:\s*(.*)$",
    re.IGNORECASE,
)
_PROPERTY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(.*?)\s*$")
_INLINE_PROPERTY = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|.*?)(?=(?:\s+[A-Za-z_][A-Za-z0-9_-]*\s*=)|[,;]|$)"
)
_UNSUPPORTED = re.compile(r"\b(?:ASSET|CONNECTOR|asset_id|slot|sheet|source\s*=\s*sheet)\b", re.IGNORECASE)
_LEGACY_MAP = {
    "MICRO FOCUS": "MICRO FOCUS",
    "FOCUS": "FOCUS",
    "FOCUS MOVE": "FOCUS MOVE",
}


def _seconds_to_us(value: str) -> int:
    return round(float(value) * 1_000_000)


def _normalize_effect(value: str) -> str:
    return re.sub(r"[_\s]+", " ", value.strip().upper())


def effect_definition(effect_type: str) -> EffectDefinition:
    key = _normalize_effect(effect_type)
    # New names use underscores; legacy names use spaces.
    key = key.replace(" ", "_") if key in {"SUBTLE ZOOM IN", "SUBTLE ZOOM OUT", "PAN LEFT", "PAN RIGHT", "FOCUS ZOOM", "PAN TO", "PULL TO"} else key
    if key not in EFFECT_REGISTRY:
        raise EffectDirectionError(f"Effect SRT error: unsupported effect type {effect_type.strip()}")
    return EFFECT_REGISTRY[key]


def _canonical_type(value: str) -> str:
    normalized = _normalize_effect(value)
    aliases = {
        "SUBTLE ZOOM IN": "SUBTLE_ZOOM_IN",
        "SUBTLE ZOOM OUT": "SUBTLE_ZOOM_OUT",
        "PAN LEFT": "PAN_LEFT",
        "PAN RIGHT": "PAN_RIGHT",
        "FOCUS ZOOM": "FOCUS_ZOOM",
        "PAN TO": "PAN_TO",
        "PULL TO": "PULL_TO",
    }
    return aliases.get(normalized, _LEGACY_MAP.get(normalized, normalized))


def _parse_effect_line(line: str, cue_index: int) -> tuple[str, int, int, str] | None:
    match = _EFFECT_LINE.match(line)
    if not match:
        return None
    effect_type = _canonical_type(match.group(1))
    try:
        effect_definition(effect_type)
    except EffectDirectionError:
        raise EffectDirectionError(f"Effect SRT error: cue {cue_index}: unsupported effect type {match.group(1).strip()}")
    return effect_type, _seconds_to_us(match.group(2)), _seconds_to_us(match.group(3)), match.group(4).strip()


def _finish_effect(active: dict | None, effects: list[VisualEffect], cue_duration: int, cue_index: int) -> None:
    if active is None:
        return
    effect_type = active["type"]
    start, end = active["start"], active["end"]
    if start < 0 or end <= start or end > cue_duration + TOLERANCE_US:
        raise EffectDirectionError(f"Effect SRT error: Image {cue_index} has an invalid {effect_type} range")
    definition = effect_definition(effect_type)
    params = dict(active["params"])
    target = params.pop("target", "").strip()
    if definition.requires_roi and not target:
        raise EffectDirectionError(f"Effect SRT error: Image {cue_index} {effect_type} requires target")
    for key, lower, upper, label in (("padding", 0.0, 0.50, "padding"), ("max_zoom", 1.0, float("inf"), "max_zoom")):
        if key not in params:
            continue
        try:
            value = float(params[key])
        except (TypeError, ValueError) as exc:
            raise EffectDirectionError(f"Effect SRT error: Image {cue_index} {label} must be numeric") from exc
        if not lower <= value <= upper:
            raise EffectDirectionError(f"Effect SRT error: Image {cue_index} {label} is out of range")
    effects.append(VisualEffect(effect_type, start, end, target, params, active["instruction"]))


def _inline_properties(instruction: str) -> tuple[dict[str, str], str]:
    """Extract all key/value properties from an inline effect instruction.

    Unknown properties deliberately remain in ``params`` so callers can extend
    the grammar without making the parser reject an otherwise valid cue.
    """
    params: dict[str, str] = {}
    spans: list[tuple[int, int]] = []
    for match in _INLINE_PROPERTY.finditer(instruction):
        value = match.group("value").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        params[match.group("key").casefold()] = value
        spans.append(match.span())
    if not spans:
        return params, instruction.strip()
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(instruction[cursor:start])
        cursor = end
    pieces.append(instruction[cursor:])
    return params, " ".join(" ".join(pieces).replace(",", " ").split()).strip()


def parse_effect_direction_srt(path: str | Path) -> list[EffectCue]:
    effects: list[EffectCue] = []
    for cue in parse_srt(path):
        image_number: int | None = None
        target_text = ""
        declared_duration_us: int | None = None
        transition_out: str | None = None
        parsed: list[VisualEffect] = []
        active: dict | None = None
        for raw_line in cue.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _UNSUPPORTED.search(line):
                raise EffectDirectionError("Unsupported legacy ASSET command. Asset-sheet workflow has been removed.")
            if match := _IMAGE_LINE.match(raw_line):
                image_number = int(match.group(1))
                continue
            if match := _TARGET_LINE.match(raw_line):
                target_text = match.group(1).strip()
                continue
            if match := _DURATION_LINE.match(raw_line):
                declared_duration_us = _seconds_to_us(match.group(1))
                continue
            if match := _TRANSITION_LINE.match(raw_line):
                transition_out = match.group(1).strip() or None
                continue
            parsed_line = _parse_effect_line(raw_line, cue.index + 1)
            if parsed_line:
                _finish_effect(active, parsed, cue.duration_us, cue.index + 1)
                effect_type, start, end, instruction = parsed_line
                params, instruction = _inline_properties(instruction)
                active = {"type": effect_type, "start": start, "end": end, "instruction": instruction, "params": params}
                continue
            if active is not None and (match := _PROPERTY.match(raw_line)):
                active["params"][match.group(1).casefold()] = match.group(2).strip()
                continue
            if active is not None:
                active["instruction"] = f"{active['instruction']} {line}".strip()
                continue
            if line.upper().startswith(("HOLD", "FOCUS", "SETTLE", "PAN", "PULL", "ALERT", "SUBTLE")):
                raise EffectDirectionError(f"Effect SRT error: malformed effect in cue {cue.index + 1}")
        _finish_effect(active, parsed, cue.duration_us, cue.index + 1)
        if target_text:
            parsed = [
                replace(effect, target_id=target_text)
                if effect.definition.roi_requirement is RoiRequirement.OPTIONAL and not effect.target_id
                else effect
                for effect in parsed
            ]
        expected_index = cue.index + 1
        if image_number is not None and image_number != expected_index:
            raise EffectDirectionError(f"Effect SRT error: cue {expected_index} is numbered Image {image_number}")
        if not parsed:
            raise EffectDirectionError(f"Effect SRT error: Image {expected_index} has no effects")
        if declared_duration_us is not None and abs(declared_duration_us - cue.duration_us) > TOLERANCE_US:
            raise EffectDirectionError(f"Effect SRT error: Image {expected_index} duration does not match its cue")
        previous_end = 0
        for effect in parsed:
            if effect.local_start_us < previous_end:
                raise EffectDirectionError(f"Effect SRT error: Image {expected_index} effects overlap")
            previous_end = effect.local_end_us
        effects.append(EffectCue(expected_index, cue.start_us, cue.end_us, target_text, tuple(parsed), transition_out, declared_duration_us))
    if not effects:
        raise EffectDirectionError("Effect SRT error: file is empty")
    return effects


def validate_effect_timing(effects: list[EffectCue], image_timings, tolerance_us: int = TOLERANCE_US) -> None:
    if len(effects) != len(image_timings):
        raise EffectDirectionError(f"Effect Direction mismatch: {len(image_timings)} images / {len(effects)} effect cues")
    for effect, timing in zip(effects, image_timings):
        if abs(effect.global_start_us - timing.start_us) > tolerance_us or abs(effect.global_end_us - timing.end_us) > tolerance_us:
            raise EffectDirectionError(f"Effect timing does not match image timing for Image {effect.image_index}.")


def required_roi_targets(effects: list[EffectCue]) -> tuple[RoiTarget, ...]:
    from auto_capcut.models import RoiTarget

    output: dict[tuple[int, str], RoiTarget] = {}
    for cue in effects:
        for target in cue.required_roi_targets:
            output[(target.image_index, target.target_id)] = target
    return tuple(output.values())


def optional_roi_targets(effects: list[EffectCue]) -> tuple[RoiTarget, ...]:
    from auto_capcut.models import RoiTarget

    output: dict[tuple[int, str], RoiTarget] = {}
    for cue in effects:
        for target in cue.optional_roi_targets:
            output[(target.image_index, target.target_id)] = target
    required = {(target.image_index, target.target_id) for target in required_roi_targets(effects)}
    return tuple(target for key, target in output.items() if key not in required)
