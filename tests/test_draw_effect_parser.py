from pathlib import Path

import pytest

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import DrawActionType, DrawMode, DrawStyle
from auto_capcut.core.errors import DrawParseError


def write_effect(path: Path, body: str) -> Path:
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def test_parse_draw_effect_supports_defaults_unicode_dash_and_parallel_camera(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "draw_effect.srt",
        """
        1
        00:00:00,000 --> 00:00:08,000
        IMAGE=001.png
        MODE=advanced_draw
        STYLE=v2
        DRAW 0.00s–4.20s: order=alex,part_ab pause_each=0.10 text=trace final=original_reveal
        FOCUS 3.50s-5.20s: target=part_ab framing=camera_frame easing=ease_in_out
        SETTLE 5.20s-8.00s: hold final composition
        """,
    )
    parsed = parse_draw_effect(path)
    plan = parsed.images[0]
    assert plan.mode is DrawMode.ADVANCED
    assert plan.style is DrawStyle.V2
    assert plan.objects == "manual"
    assert plan.draw_action.params["order"] == "alex,part_ab"
    assert [action.type for action in plan.actions] == [DrawActionType.DRAW, DrawActionType.FOCUS, DrawActionType.SETTLE]


def test_basic_mode_defaults_to_auto_and_rejects_manual_objects(tmp_path: Path) -> None:
    valid = write_effect(
        tmp_path / "basic.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        MODE=basic_draw
        STYLE=v1
        DRAW 0s-1s:
        """,
    )
    assert parse_draw_effect(valid).images[0].objects == "auto"
    invalid = write_effect(
        tmp_path / "invalid.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        MODE=basic_draw
        STYLE=v1
        OBJECTS=manual
        DRAW 0s-1s:
        """,
    )
    with pytest.raises(DrawParseError, match="requires OBJECTS=auto"):
        parse_draw_effect(invalid)


def test_parser_rejects_camera_overlap_and_missing_draw(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "invalid.srt",
        """
        1
        00:00:00,000 --> 00:00:02,000
        MODE=advanced_draw
        STYLE=v1
        FOCUS 0s-1s: target=a
        PAN_TO 0.5s-2s: target=a
        """,
    )
    with pytest.raises(DrawParseError, match="exactly one DRAW"):
        parse_draw_effect(path)


@pytest.mark.parametrize("header", ["Image 001 DRAW", "Image 1 DRAW", "IMAGE    001    DRAW"])
def test_optional_image_block_header_is_metadata(tmp_path: Path, header: str) -> None:
    path = write_effect(
        tmp_path / "header.srt",
        f"""
        1
        00:00:00,000 --> 00:00:01,000
        {header}
        MODE basic_draw
        STYLE v1
        DRAW 0s-1s:
        """,
    )
    parsed = parse_draw_effect(path)
    assert len(parsed.images) == 1
    assert parsed.images[0].image_name is None
    assert parsed.warnings == ()


def test_block_header_is_optional_and_multiple_blocks_are_numbered(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "multiple.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        Image 001 DRAW
        MODE=basic_draw
        STYLE=v1
        DRAW 0s-1s:

        2
        00:00:01,000 --> 00:00:02,000
        MODE basic_draw
        STYLE v1
        DRAW 0s-1s:
        """,
    )
    parsed = parse_draw_effect(path)
    assert len(parsed.images) == 2
    assert parsed.warnings == ()


def test_mismatched_block_header_is_a_warning(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "mismatch.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        Image 009 DRAW
        MODE basic_draw
        STYLE v1
        DRAW 0s-1s:
        """,
    )
    parsed = parse_draw_effect(path)
    assert parsed.warnings == ("Image 1: block header declares Image 9 DRAW",)


def test_draw_direction_and_unmatched_defaults_and_validation(tmp_path: Path) -> None:
    valid = write_effect(
        tmp_path / "direction.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        MODE advanced_draw
        STYLE v1
        DRAW 0s-1s: direction=right_to_left unmatched=ignore
        """,
    )
    params = parse_draw_effect(valid).images[0].draw_action.params
    assert params["direction"] == "right_to_left"
    assert params["unmatched"] == "ignore"

    defaults = write_effect(
        tmp_path / "defaults.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        MODE basic_draw
        STYLE v1
        DRAW 0s-1s:
        """,
    )
    default_params = parse_draw_effect(defaults).images[0].draw_action.params
    assert default_params["direction"] == "auto"
    assert default_params["unmatched"] == "last"

    invalid = write_effect(
        tmp_path / "invalid-direction.srt",
        """
        1
        00:00:00,000 --> 00:00:01,000
        MODE basic_draw
        STYLE v1
        DRAW 0s-1s: direction=diagonal
        """,
    )
    with pytest.raises(DrawParseError, match="direction is invalid"):
        parse_draw_effect(invalid)


@pytest.mark.parametrize("duration, expected_mode, expected_us", [("auto", "auto", None), ("0.70", "fixed", 700_000), ("1.25", "fixed", 1_250_000)])
def test_object_effect_duration_forms(tmp_path: Path, duration: str, expected_mode: str, expected_us: int | None) -> None:
    path = write_effect(tmp_path / f"duration-{duration}.srt", f"""
        1
        00:00:00,000 --> 00:00:01,000
        MODE advanced_draw
        STYLE v1
        OBJECT_EFFECT target=part_c effect=toss_in duration={duration}
        DRAW 0s-1s:
        """)
    override = parse_draw_effect(path).images[0].object_effects[0]
    assert override.duration_mode == expected_mode
    assert override.duration_us == expected_us


def test_object_effect_rejects_invalid_duration(tmp_path: Path) -> None:
    path = write_effect(tmp_path / "duration-invalid.srt", """
        1
        00:00:00,000 --> 00:00:01,000
        MODE advanced_draw
        STYLE v1
        OBJECT_EFFECT target=part_c effect=toss_in duration=fast
        DRAW 0s-1s:
        """)
    with pytest.raises(DrawParseError, match="duration must be a number of seconds"):
        parse_draw_effect(path)


def test_camera_after_parsing_valid(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "camera_after_valid.srt",
        """
        1
        00:00:00,000 --> 00:00:05,000
        MODE advanced_draw
        STYLE v2
        CAMERA_AFTER object=object_1 action=focus target=object_1 duration=0.55 hold=0.15 framing=camera_frame easing=ease_in_out
        CAMERA_AFTER object=object_2 action=pan_to target=object_2 duration=auto hold=0.10
        CAMERA_AFTER object=object_6 action=full_view duration=0.70 hold=0.20 easing=linear
        DRAW 0s-5s:
        """,
    )
    plan = parse_draw_effect(path).images[0]
    assert len(plan.camera_after) == 3

    c1 = plan.camera_after[0]
    assert c1.object_id == "object_1"
    assert c1.action == "focus"
    assert c1.target == "object_1"
    assert c1.duration_us == 550_000
    assert c1.duration_mode == "fixed"
    assert c1.hold_us == 150_000
    assert c1.framing == "camera_frame"
    assert c1.easing == "ease_in_out"

    c2 = plan.camera_after[1]
    assert c2.object_id == "object_2"
    assert c2.action == "pan_to"
    assert c2.target == "object_2"
    assert c2.duration_us is None
    assert c2.duration_mode == "auto"
    assert c2.hold_us == 100_000
    assert c2.framing == "camera_frame"
    assert c2.easing == "ease_in_out"

    c3 = plan.camera_after[2]
    assert c3.object_id == "object_6"
    assert c3.action == "full_view"
    assert c3.target == ""
    assert c3.duration_us == 700_000
    assert c3.duration_mode == "fixed"
    assert c3.hold_us == 200_000
    assert c3.easing == "linear"


@pytest.mark.parametrize(
    "cue_body, expected_err",
    [
        ("CAMERA_AFTER action=focus", "requires object"),
        ("CAMERA_AFTER object=obj1 action=jump", "invalid action"),
        ("CAMERA_AFTER object=obj1 action=focus duration=fast", "duration must be a number of seconds"),
        ("CAMERA_AFTER object=obj1 action=focus duration=-0.5", "duration cannot be negative"),
        ("CAMERA_AFTER object=obj1 action=focus hold=-0.1", "hold cannot be negative"),
        ("CAMERA_AFTER object=obj1 action=focus framing=diagonal", "framing is invalid"),
        ("CAMERA_AFTER object=obj1 action=focus easing=bounce", "easing is invalid"),
        ("CAMERA_AFTER object=obj1 action=focus foo=bar", "unsupported parameter"),
    ],
)
def test_camera_after_rejects_invalid_parameters(tmp_path: Path, cue_body: str, expected_err: str) -> None:
    path = write_effect(
        tmp_path / "invalid_cam.srt",
        f"""
        1
        00:00:00,000 --> 00:00:02,000
        MODE advanced_draw
        STYLE v1
        {cue_body}
        DRAW 0s-2s:
        """,
    )
    with pytest.raises(DrawParseError, match=expected_err):
        parse_draw_effect(path)


def test_camera_after_requires_advanced_mode(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "basic_cam.srt",
        """
        1
        00:00:00,000 --> 00:00:02,000
        MODE basic_draw
        STYLE v1
        CAMERA_AFTER object=obj1 action=focus
        DRAW 0s-2s:
        """,
    )
    with pytest.raises(DrawParseError, match="require advanced_draw"):
        parse_draw_effect(path)


def test_camera_after_rejects_duplicate_object(tmp_path: Path) -> None:
    path = write_effect(
        tmp_path / "dup_cam.srt",
        """
        1
        00:00:00,000 --> 00:00:02,000
        MODE advanced_draw
        STYLE v1
        CAMERA_AFTER object=obj1 action=focus
        CAMERA_AFTER object=obj1 action=pan_to
        DRAW 0s-2s:
        """,
    )
    with pytest.raises(DrawParseError, match="duplicate CAMERA_AFTER"):
        parse_draw_effect(path)

