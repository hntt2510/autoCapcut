from __future__ import annotations

import re

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionResolutionDecision:
    mode: str
    direction_x: int = 0
    direction_y: int = 0
    reason: str = ""


def infer_direction(text: str) -> tuple[int, int]:
    value = text.casefold().replace("â€™", "'")
    horizontal = 0
    vertical = 0
    if re.search(r"\bpan\s+(?:to\s+)?left(?:ward)?\b|\bleftward\s+pan\b", value):
        horizontal = -1
    elif re.search(r"\bpan\s+(?:to\s+)?right(?:ward)?\b|\brightward\s+pan\b", value):
        horizontal = 1
    elif re.search(r"\bleft\b.*\b(?:to|toward|through)\b.*\bright\b|toward the right|right panel", value):
        horizontal = -1
    elif re.search(r"\bright\b.*\b(?:to|toward|through)\b.*\bleft\b|toward the left|left panel", value):
        horizontal = 1
    if re.search(r"\bpan\s+(?:down|downward)\b|\bdownward\s+pan\b|toward the bottom|\blower\b", value):
        vertical = 1
    elif re.search(r"\bpan\s+(?:up|upward)\b|\bupward\s+pan\b|toward the top|\bupper\b", value):
        vertical = -1
    return horizontal, vertical


def _generic_target(target: str) -> bool:
    value = target.casefold().strip()
    if not value:
        return True
    if re.fullmatch(r"(?:generic|scene|image|frame|full|overall|general)(?:\s+(?:scene|image|frame|composition))?", value):
        return True
    return any(
        phrase in value
        for phrase in (
            "full frame",
            "overall scene",
            "whole image",
            "main composition",
            "center composition",
            "center of frame",
            "general scene",
        )
    )


def deterministic_motion_decision(target_text: str, instruction_text: str) -> MotionResolutionDecision:
    combined = f"{target_text} {instruction_text}".casefold()
    direction_x, direction_y = infer_direction(combined)
    explicit_pan = bool(re.search(r"\b(?:pan|move|slide|track|follow)\b", combined))
    if explicit_pan and (direction_x or direction_y):
        return MotionResolutionDecision("deterministic", direction_x, direction_y, "explicit directional instruction")
    center_push = re.search(
        r"\b(?:push[- ]?in|zoom[- ]?in|focus)\b[^.\n]{0,40}\b(?:center|centre|middle)\b"
        r"|\b(?:center|centre|middle)\b[^.\n]{0,40}\b(?:push[- ]?in|zoom[- ]?in|focus)\b",
        combined,
    )
    if _generic_target(target_text) and (center_push or re.search(r"\b(?:push[- ]?in|zoom[- ]?in|focus\s+center|center\s+focus)\b", combined)):
        return MotionResolutionDecision("deterministic", reason="generic center push-in")
    return MotionResolutionDecision("manual_roi", reason="object or target localization required")
