"""Tests for unified_effect_parser.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from auto_capcut.core.unified_effect_parser import parse_unified_effect


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


STANDARD_ONLY = """\
1
00:00:00,000 --> 00:00:05,000
Image 1 FX
HOLD 0s - 5s :

2
00:00:05,000 --> 00:00:10,000
Image 2 FX
HOLD 0s - 5s :
"""

DRAW_ONLY = """\
1
00:00:00,000 --> 00:00:05,000
MODE=basic_draw
STYLE=v1
DRAW 0s-5s:

2
00:00:05,000 --> 00:00:10,000
MODE=basic_draw
STYLE=v1
DRAW 0s-5s:
"""

MIXED = """\
1
00:00:00,000 --> 00:00:05,000
Image 1 FX
HOLD 0s - 5s :

2
00:00:05,000 --> 00:00:10,000
MODE=basic_draw
STYLE=v1
DRAW 0s-5s:

3
00:00:10,000 --> 00:00:15,000
Image 3 FX
HOLD 0s - 5s :
"""


def test_standard_cue_only(tmp_path: Path) -> None:
    path = _write(tmp_path / "effect.srt", STANDARD_ONLY)
    result = parse_unified_effect(path)
    assert result.has_standard_cues
    assert not result.has_draw_cues
    assert all(c.kind == "standard" for c in result.cues)
    assert len(result.cues) == 2


def test_draw_cue_only(tmp_path: Path) -> None:
    path = _write(tmp_path / "effect.srt", DRAW_ONLY)
    result = parse_unified_effect(path)
    assert result.has_draw_cues
    assert not result.has_standard_cues
    assert all(c.kind == "draw" for c in result.cues)
    assert len(result.cues) == 2
    for cue in result.cues:
        assert cue.draw_plan is not None


def test_mixed_standard_and_draw(tmp_path: Path) -> None:
    path = _write(tmp_path / "effect.srt", MIXED)
    result = parse_unified_effect(path)
    assert len(result.cues) == 3
    assert result.cues[0].kind == "standard"
    assert result.cues[1].kind == "draw"
    assert result.cues[2].kind == "standard"
    assert result.cues[1].draw_plan is not None
    assert result.has_standard_cues
    assert result.has_draw_cues


def test_effect_cues_property(tmp_path: Path) -> None:
    path = _write(tmp_path / "effect.srt", STANDARD_ONLY)
    result = parse_unified_effect(path)
    assert len(result.effect_cues) == 2
    assert all(ec is not None for ec in result.effect_cues)


def test_draw_plans_property(tmp_path: Path) -> None:
    path = _write(tmp_path / "effect.srt", DRAW_ONLY)
    result = parse_unified_effect(path)
    assert len(result.draw_plans) == 2


def test_empty_file_raises(tmp_path: Path) -> None:
    from auto_capcut.core.errors import ValidationError
    path = tmp_path / "empty.srt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError):
        parse_unified_effect(path)
