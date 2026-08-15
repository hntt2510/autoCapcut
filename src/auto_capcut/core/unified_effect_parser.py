"""
unified_effect_parser.py
========================
Routes each SRT cue to the correct parser (draw or standard effect direction)
based on cue content.

A cue is classified as a **draw cue** when its text contains a line matching::

    MODE  basic_draw|advanced_draw

(case-insensitive, leading whitespace allowed).

All other cues are classified as **standard effect cues** and are delegated to
``effect_direction_parser.parse_effect_direction_srt``.

Usage::

    unified = parse_unified_effect(path)
    for cue in unified.cues:
        if cue.kind == "draw":
            plan = cue.draw_plan         # DrawImagePlan
        else:
            effect = cue.effect_cue      # EffectCue
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import DrawImagePlan
from auto_capcut.core.effect_direction_parser import parse_effect_direction_srt
from auto_capcut.core.errors import DrawParseError, EffectDirectionError, SRTParseError, ValidationError
from auto_capcut.core.srt_parser import parse_srt
from auto_capcut.models import EffectCue

# Matches "MODE basic_draw" or "MODE advanced_draw" (space or = separator)
_DRAW_MODE = re.compile(
    r"^\s*MODE\s*[=\s]\s*(basic_draw|advanced_draw)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches explicit legacy standard effect cues
_EXPLICIT_STANDARD = re.compile(
    r"^\s*(?:Image\s+\d+\s+FX\b|(?:HOLD|ZOOM|PAN|CAMERA|EFFECT)\s+\d+(?:\.\d+)?s\s*(?:-|–|—))",
    re.IGNORECASE | re.MULTILINE,
)

# Matches draw directives
_DRAW_DIRECTIVE = re.compile(
    r"^\s*(?:DRAW|COMPLETE_BEFORE_END|POST_MOTION|OBJECT_EFFECT|CAMERA_AFTER|STYLE)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Matches the optional "Image N FX" header line inside standard effect cue text.
_IMG_HDR = re.compile(r"^\s*Image\s+\d+\s+FX\b", re.IGNORECASE)


def _is_draw_cue(text: str) -> bool:
    if _DRAW_MODE.search(text) or _DRAW_DIRECTIVE.search(text):
        return True
    if not _EXPLICIT_STANDARD.search(text):
        return True
    return False



@dataclass(frozen=True)
class UnifiedCue:
    """One SRT block classified and parsed."""
    index: int                                # 1-based image index
    start_us: int
    end_us: int
    kind: Literal["draw", "standard"]
    draw_plan: DrawImagePlan | None = None    # set when kind=="draw"
    effect_cue: EffectCue | None = None       # set when kind=="standard"


@dataclass(frozen=True)
class UnifiedEffectFile:
    path: Path
    cues: tuple[UnifiedCue, ...]
    draw_warnings: tuple[str, ...]
    effect_warnings: tuple[str, ...]

    @property
    def has_draw_cues(self) -> bool:
        return any(c.kind == "draw" for c in self.cues)

    @property
    def has_standard_cues(self) -> bool:
        return any(c.kind == "standard" for c in self.cues)

    @property
    def draw_plans(self) -> list[DrawImagePlan]:
        """Ordered draw plans (None-safe)."""
        return [c.draw_plan for c in self.cues if c.kind == "draw" and c.draw_plan is not None]

    @property
    def effect_cues(self) -> list[EffectCue]:
        """Ordered standard effect cues (None-safe)."""
        return [c.effect_cue for c in self.cues if c.kind == "standard" and c.effect_cue is not None]


def parse_unified_effect(path: str | Path) -> UnifiedEffectFile:
    """Parse a unified SRT file that may contain both draw and standard cues.

    Each SRT block is classified independently; classification is based on
    whether the block text contains a ``MODE basic_draw|advanced_draw`` line.

    Raises
    ------
    ValidationError (or subclass)
        On any parse failure from either sub-parser.
    """
    source = Path(path)
    try:
        raw_cues = parse_srt(source)
    except Exception as exc:
        raise ValidationError(f"Cannot read SRT file: {source}: {exc}") from exc

    if not raw_cues:
        raise ValidationError(f"SRT file is empty: {source}")

    # Partition cue indices into draw vs standard groups.
    draw_indices: list[int] = []
    standard_indices: list[int] = []
    for raw in raw_cues:
        if _is_draw_cue(raw.text):
            draw_indices.append(raw.index)
        else:
            standard_indices.append(raw.index)

    # ── Parse draw subset ────────────────────────────────────────────────────
    draw_plans_by_index: dict[int, DrawImagePlan] = {}
    draw_warnings: tuple[str, ...] = ()
    if draw_indices:
        # Build a temporary SRT containing only the draw cues.
        # parse_draw_effect requires contiguous blocks starting at t=0, so we
        # renumber and re-timestamp each cue so they start at 0 and are
        # contiguous.  Each cue retains its original duration.
        draw_lines: list[str] = []
        cursor_us = 0
        draw_cues_ordered = [raw for raw in raw_cues if raw.index in set(draw_indices)]
        for seq, raw in enumerate(draw_cues_ordered):
            duration_us = raw.end_us - raw.start_us
            block_start = cursor_us
            block_end = cursor_us + duration_us
            cursor_us = block_end

            def _fmt(us: int) -> str:
                h = us // 3_600_000_000
                m = (us // 60_000_000) % 60
                s = (us // 1_000_000) % 60
                ms = (us // 1_000) % 1_000
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            draw_lines.append(f"{seq + 1}")
            draw_lines.append(f"{_fmt(block_start)} --> {_fmt(block_end)}")
            draw_lines.append(raw.text)
            draw_lines.append("")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("\n".join(draw_lines))
        tmp_path = Path(tmp.name)
        try:
            draw_effect = parse_draw_effect(tmp_path)
            draw_warnings = draw_effect.warnings
            from dataclasses import replace as _dc_replace
            for seq, original_index in enumerate(sorted(draw_indices)):
                if seq < len(draw_effect.images):
                    plan = draw_effect.images[seq]
                    draw_plans_by_index[original_index] = _dc_replace(plan, image_index=original_index + 1)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Parse standard subset ────────────────────────────────────────────────
    effect_cues_by_index: dict[int, EffectCue] = {}
    effect_warnings: tuple[str, ...] = ()
    if standard_indices:
        # Build a temp SRT with only standard cues. We preserve the original SRT
        # block number (raw.index + 1) because effect_direction_parser validates the
        # "Image N FX" declaration inside the cue text against the SRT ordinal.
        # If the body says "Image 3 FX" the SRT block number must also be 3 (or 2
        # if the parser is 0-based — the parser uses cue.index+1, so we just keep it).
        std_lines: list[str] = []
        for seq, raw in enumerate(raw_cue for raw_cue in raw_cues if raw_cue.index in set(standard_indices)):
            std_lines.append(f"{seq + 1}")  # sequential numbering for the temp file
            std_lines.append(
                f"{raw.start_us // 3600_000_000:02d}:{(raw.start_us // 60_000_000) % 60:02d}:"
                f"{(raw.start_us // 1_000_000) % 60:02d},{(raw.start_us // 1_000) % 1_000:03d}"
                f" --> "
                f"{raw.end_us // 3600_000_000:02d}:{(raw.end_us // 60_000_000) % 60:02d}:"
                f"{(raw.end_us // 1_000_000) % 60:02d},{(raw.end_us // 1_000) % 1_000:03d}"
            )
            # Remove any "Image N FX" header line from the text so the parser
            # doesn't reject the renumbered cue for image-number mismatch.
            filtered_lines = [line for line in raw.text.splitlines() if not _IMG_HDR.match(line)]
            std_lines.append("\n".join(filtered_lines))
            std_lines.append("")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("\n".join(std_lines))
        tmp_path = Path(tmp.name)
        try:
            parsed_effects = parse_effect_direction_srt(tmp_path)
            from dataclasses import replace as _dc_replace
            # Map results back to original index: parsed_effects[i] corresponds to the
            # i-th standard cue in original order.
            for seq, original_index in enumerate(sorted(standard_indices)):
                if seq < len(parsed_effects):
                    cue_obj = parsed_effects[seq]
                    effect_cues_by_index[original_index] = _dc_replace(cue_obj, image_index=original_index + 1)
        except EffectDirectionError as exc:
            raise ValidationError(str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)



    # ── Assemble output in original SRT order ─────────────────────────────────
    cues: list[UnifiedCue] = []
    for raw in raw_cues:
        if raw.index in draw_plans_by_index:
            plan = draw_plans_by_index[raw.index]
            cues.append(UnifiedCue(
                index=raw.index + 1,
                start_us=raw.start_us,
                end_us=raw.end_us,
                kind="draw",
                draw_plan=plan,
            ))
        elif raw.index in effect_cues_by_index:
            cues.append(UnifiedCue(
                index=raw.index + 1,
                start_us=raw.start_us,
                end_us=raw.end_us,
                kind="standard",
                effect_cue=effect_cues_by_index[raw.index],
            ))
        else:
            # Cue was classified but failed to parse (error should have been raised already)
            cues.append(UnifiedCue(
                index=raw.index + 1,
                start_us=raw.start_us,
                end_us=raw.end_us,
                kind="standard" if raw.index in set(standard_indices) else "draw",
            ))

    return UnifiedEffectFile(
        path=source,
        cues=tuple(cues),
        draw_warnings=draw_warnings,
        effect_warnings=effect_warnings,
    )
