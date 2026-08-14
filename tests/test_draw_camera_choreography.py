from __future__ import annotations

import math
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import (
    CameraActionType,
    CameraAfterDirective,
    CameraEasing,
    CameraFramingMode,
    CameraState,
    DrawAction,
    DrawActionType,
    DrawImagePlan,
    DrawMode,
    DrawProjectConfig,
    DrawStyle,
    NormalizedRect,
    SceneDocument,
    SceneImage,
    SceneObject,
    TextMode,
)
from auto_capcut.core.draw_renderer import (
    DrawRenderer,
    FULL_VIEW_STATE,
    Stroke,
    _auto_camera_duration_us,
    _build_advanced_schedule,
    _camera_state_at,
    _interpolate_camera_state,
    _resolve_camera_target,
    prepare_image,
)
from auto_capcut.core.draw_scene import save_scene
from auto_capcut.core.errors import DrawParseError


# ==============================================================================
# 1. PURE CAMERA STATE GEOMETRY & TARGET RESOLUTION TESTS
# ==============================================================================

def test_full_view_state_is_exact_unit_square() -> None:
    assert FULL_VIEW_STATE.viewport == (0.0, 0.0, 1.0, 1.0)
    assert FULL_VIEW_STATE.x == 0.0
    assert FULL_VIEW_STATE.y == 0.0
    assert FULL_VIEW_STATE.w == 1.0
    assert FULL_VIEW_STATE.h == 1.0
    assert FULL_VIEW_STATE.center_x == 0.5
    assert FULL_VIEW_STATE.center_y == 0.5
    assert FULL_VIEW_STATE.scale == pytest.approx(1.0)


def test_focus_resolution_priority_saved_camera_frame() -> None:
    obj = SceneObject(
        "central_hero",
        "art",
        NormalizedRect(0.20, 0.20, 0.40, 0.50),
        camera_frame=NormalizedRect(0.15, 0.10, 0.50, 0.50),
    )
    target_state, frame_src, is_clamped = _resolve_camera_target(
        "focus",
        obj,
        framing="camera_frame",
        current_state=FULL_VIEW_STATE,
        canvas_aspect=16 / 9,
        source_size=(1920, 1080),
    )
    assert frame_src == "saved_camera_frame"
    assert target_state.viewport == (0.15, 0.10, 0.50, 0.50)
    assert not is_clamped


def test_focus_resolution_derived_object_frame_with_max_zoom_constraint() -> None:
    # A very tiny object (0.02 x 0.02)
    small_obj = SceneObject(
        "tiny_badge",
        "art",
        NormalizedRect(0.49, 0.49, 0.02, 0.02),
        camera_frame=None,
    )
    target_state, frame_src, is_clamped = _resolve_camera_target(
        "focus",
        small_obj,
        framing="object_frame",
        current_state=FULL_VIEW_STATE,
        canvas_aspect=16 / 9,
        source_size=(1920, 1080),
        max_zoom=2.4,
    )
    assert frame_src == "derived_object_frame"
    # Max zoom 2.4 means min width is 1.0 / 2.4 ~= 0.4167
    assert target_state.w >= 1.0 / 2.4 - 1e-4
    assert target_state.center_x == pytest.approx(0.50, abs=0.01)
    assert target_state.center_y == pytest.approx(0.50, abs=0.01)
    assert target_state.x >= 0.0
    assert target_state.y >= 0.0
    assert target_state.x + target_state.w <= 1.0
    assert target_state.y + target_state.h <= 1.0


def test_pan_to_preserves_zoom_and_centers_on_target() -> None:
    current_state = CameraState((0.10, 0.10, 0.50, 0.50))
    target_obj = SceneObject(
        "mid_card",
        "art",
        NormalizedRect(0.50, 0.40, 0.20, 0.20),
        camera_frame=None,
    )
    target_state, frame_src, is_clamped = _resolve_camera_target(
        "pan_to",
        target_obj,
        framing="camera_frame",
        current_state=current_state,
        canvas_aspect=16 / 9,
        source_size=(1920, 1080),
    )
    assert frame_src == "derived_object_center"
    # Viewport width and height preserved
    assert target_state.w == pytest.approx(0.50)
    assert target_state.h == pytest.approx(0.50)
    # Center aligned with target object center (0.60, 0.50)
    assert target_state.center_x == pytest.approx(0.60, abs=0.01)
    assert target_state.center_y == pytest.approx(0.50, abs=0.01)
    assert not is_clamped


def test_pan_to_minimally_zooms_out_if_target_too_large() -> None:
    current_state = CameraState((0.10, 0.10, 0.30, 0.30))  # Tight zoom
    large_obj = SceneObject(
        "big_chart",
        "art",
        NormalizedRect(0.20, 0.20, 0.60, 0.50),  # Larger than 0.30
        camera_frame=None,
    )
    target_state, _, _ = _resolve_camera_target(
        "pan_to",
        large_obj,
        framing="camera_frame",
        current_state=current_state,
        canvas_aspect=16 / 9,
        source_size=(1920, 1080),
    )
    # Must have expanded viewport to encompass the larger object
    assert target_state.w >= 0.60
    assert target_state.h >= 0.50


def test_pull_to_fallback_when_target_requires_zoom_in() -> None:
    current_state = CameraState((0.0, 0.0, 1.0, 1.0))  # Full view
    small_obj = SceneObject(
        "detail",
        "art",
        NormalizedRect(0.40, 0.40, 0.20, 0.20),
        camera_frame=None,
    )
    target_state, frame_src, _ = _resolve_camera_target(
        "pull_to",
        small_obj,
        framing="object_frame",
        current_state=current_state,
        canvas_aspect=16 / 9,
        source_size=(1920, 1080),
    )
    assert "focus_fallback" in frame_src


def test_interpolate_camera_state_smooth_and_clamped() -> None:
    start = CameraState((0.0, 0.0, 1.0, 1.0))
    end = CameraState((0.25, 0.25, 0.50, 0.50))

    # Boundary values
    assert _interpolate_camera_state(start, end, 0.0).viewport == start.viewport
    assert _interpolate_camera_state(start, end, 1.0).viewport == end.viewport
    assert _interpolate_camera_state(start, end, -0.5).viewport == start.viewport
    assert _interpolate_camera_state(start, end, 1.5).viewport == end.viewport

    # Midpoint values
    mid = _interpolate_camera_state(start, end, 0.5, easing="ease_in_out")
    assert 0.0 < mid.x < 0.25
    assert 0.0 < mid.y < 0.25
    assert 0.50 < mid.w < 1.00
    assert 0.50 < mid.h < 1.00
    assert mid.x + mid.w <= 1.0
    assert mid.y + mid.h <= 1.0


# ==============================================================================
# 2. TIMELINE SCHEDULING & CHOREOGRAPHY INTEGRATION TESTS
# ==============================================================================

def _make_test_scene(order=("obj_a", "obj_b", "obj_c")) -> SceneImage:
    objects = (
        SceneObject("obj_a", "art", NormalizedRect(0.10, 0.10, 0.25, 0.80), NormalizedRect(0.05, 0.05, 0.35, 0.90), "draw"),
        SceneObject("obj_b", "art", NormalizedRect(0.40, 0.10, 0.25, 0.80), NormalizedRect(0.35, 0.05, 0.35, 0.90), "push_in", "left"),
        SceneObject("obj_c", "art", NormalizedRect(0.70, 0.10, 0.25, 0.80), NormalizedRect(0.65, 0.05, 0.35, 0.90), "draw"),
    )
    return SceneImage("001.png", (1920, 1080), objects, tuple(order))


def test_camera_choreography_sequential_timeline() -> None:
    scene = _make_test_scene()
    strokes = (
        Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),
        Stroke(((0.75, 0.20), (0.85, 0.40)), object_id="obj_c"),
    )
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=500_000, duration_mode="fixed", hold_us=150_000),
        CameraAfterDirective(object_id="obj_b", action="pan_to", target="obj_b", duration_us=400_000, duration_mode="fixed", hold_us=100_000),
        CameraAfterDirective(object_id="obj_c", action="full_view", target="", duration_us=600_000, duration_mode="fixed", hold_us=200_000),
    )
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        5_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 5_000_000, {"final": "line_then_color"}),),
        camera_after=camera_directives,
    )
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    # Verify sequential phase structure
    phase_kinds = [p.kind for p in schedule.phases]
    assert "object" in phase_kinds
    assert "camera" in phase_kinds
    assert "camera_hold" in phase_kinds

    # Find obj_a sequence
    obj_a_idx = [i for i, p in enumerate(schedule.phases) if p.object_id == "obj_a"]
    assert len(obj_a_idx) >= 3  # object -> finalize -> camera -> camera_hold
    obj_phase = schedule.phases[obj_a_idx[0]]
    fin_phase = schedule.phases[obj_a_idx[1]]
    cam_phase = schedule.phases[obj_a_idx[2]]
    hold_phase = schedule.phases[obj_a_idx[3]]

    assert obj_phase.kind == "object"
    assert fin_phase.kind == "finalize"
    assert cam_phase.kind == "camera"
    assert hold_phase.kind == "camera_hold"

    # Strict non-overlapping ordering: object finishes -> local finalize -> camera moves -> camera holds
    assert obj_phase.end_us == fin_phase.start_us
    assert fin_phase.end_us == cam_phase.start_us
    assert cam_phase.end_us == hold_phase.start_us
    assert cam_phase.duration_us == 500_000
    assert hold_phase.duration_us == 150_000


def test_camera_continuity_across_timeline() -> None:
    scene = _make_test_scene()
    strokes = (
        Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),
        Stroke(((0.75, 0.20), (0.85, 0.40)), object_id="obj_c"),
    )
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=500_000, duration_mode="fixed", hold_us=100_000),
        CameraAfterDirective(object_id="obj_c", action="full_view", target="", duration_us=500_000, duration_mode="fixed", hold_us=100_000),
    )
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        5_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 5_000_000, {"final": "line_then_color"}),),
        camera_after=camera_directives,
    )
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    # Before camera moves on obj_a: must be FULL_VIEW
    st_0 = _camera_state_at(schedule, plan, scene, 100_000, 16 / 9, (1920, 1080))
    assert st_0.viewport == FULL_VIEW_STATE.viewport

    # During obj_b (which has no camera_after): camera must maintain obj_a's focused camera state (continuity!)
    cam_a_phase = next(p for p in schedule.phases if p.kind == "camera" and p.object_id == "obj_a")
    obj_b_phase = next(p for p in schedule.phases if p.kind == "object" and p.object_id == "obj_b")

    st_during_b = _camera_state_at(schedule, plan, scene, (obj_b_phase.start_us + obj_b_phase.end_us) // 2, 16 / 9, (1920, 1080))
    assert st_during_b.viewport == cam_a_phase.camera_end.viewport


def test_final_reconciliation_requires_full_view_state() -> None:
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    # obj_a focuses, but no full_view return directive before final reconciliation
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=500_000, duration_mode="fixed"),
    )
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        3_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 3_000_000, {"final": "line_then_color"}),),
        camera_after=camera_directives,
    )
    with pytest.raises(DrawParseError, match="Final reconciliation requires FULL_VIEW"):
        _build_advanced_schedule(strokes, plan, scene, (1920, 1080))


def test_budget_overflow_error_with_explicit_durations() -> None:
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    # Fixed camera duration of 5.0s on a 2.0s DRAW cue
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=5_000_000, duration_mode="fixed"),
    )
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        2_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 2_000_000, {"final": "line_only"}),),
        camera_after=camera_directives,
    )
    with pytest.raises(DrawParseError, match="Advanced choreography exceeds DRAW interval"):
        _build_advanced_schedule(strokes, plan, scene, (1920, 1080))


# ==============================================================================
# 3. END-TO-END MEDICARE ACCEPTANCE TEST WITH FULL 6 OBJECTS
# ==============================================================================

def test_medicare_scene_camera_choreography_acceptance(tmp_path: Path) -> None:
    """Acceptance test verifying full 6-object Medicare scene with camera choreography."""
    # 1. Create clean Medicare fixture
    image_path = tmp_path / "001.png"
    img = Image.new("RGB", (640, 360), (242, 245, 248))
    draw = ImageDraw.Draw(img)

    # Object 1: Alex + US map
    draw.rectangle((220, 90, 420, 270), fill=(100, 140, 200), outline=(30, 40, 60), width=3)
    draw.text((260, 170), "ALEX + MAP", fill=(255, 255, 255))
    # Object 2: Left card
    draw.rectangle((40, 90, 180, 230), fill=(220, 100, 90), outline=(40, 20, 20), width=2)
    # Object 3: Right card
    draw.rectangle((460, 90, 600, 230), fill=(90, 180, 110), outline=(20, 40, 20), width=2)
    # Object 4: Top icon
    draw.ellipse((280, 20, 360, 80), fill=(240, 200, 60), outline=(50, 40, 10), width=2)
    # Object 5: Bottom badge
    draw.rectangle((260, 290, 380, 340), fill=(180, 120, 220), outline=(40, 20, 50), width=2)
    # Object 6: Summary checkmark
    draw.rectangle((500, 260, 580, 330), fill=(70, 160, 240), outline=(20, 30, 60), width=2)
    img.save(image_path)

    # 2. Scene metadata
    scene_objects = (
        SceneObject("object_1", "art", NormalizedRect(220 / 640, 90 / 360, 200 / 640, 180 / 360), NormalizedRect(0.30, 0.20, 0.40, 0.60), "draw"),
        SceneObject("object_2", "art", NormalizedRect(40 / 640, 90 / 360, 140 / 640, 140 / 360), NormalizedRect(0.04, 0.20, 0.30, 0.50), "push_in", "left"),
        SceneObject("object_3", "art", NormalizedRect(460 / 640, 90 / 360, 140 / 640, 140 / 360), NormalizedRect(0.66, 0.20, 0.30, 0.50), "push_in", "right"),
        SceneObject("object_4", "art", NormalizedRect(280 / 640, 20 / 360, 80 / 640, 60 / 360), NormalizedRect(0.40, 0.03, 0.20, 0.30), "slide_in", "top"),
        SceneObject("object_5", "art", NormalizedRect(260 / 640, 290 / 360, 120 / 640, 50 / 360), NormalizedRect(0.38, 0.75, 0.24, 0.20), "pop_in"),
        SceneObject("object_6", "art", NormalizedRect(500 / 640, 260 / 360, 80 / 640, 70 / 360), NormalizedRect(0.75, 0.70, 0.20, 0.25), "toss_in"),
    )
    scene = SceneImage("001.png", (640, 360), scene_objects, ("object_1", "object_2", "object_3", "object_4", "object_5", "object_6"))

    # 3. Plan with camera choreography
    camera_directives = (
        CameraAfterDirective("object_1", "focus", "object_1", duration_us=400_000, duration_mode="fixed", hold_us=100_000),
        CameraAfterDirective("object_2", "pan_to", "object_2", duration_us=350_000, duration_mode="fixed", hold_us=80_000),
        CameraAfterDirective("object_6", "full_view", "", duration_us=500_000, duration_mode="fixed", hold_us=100_000),
    )
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        6_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 6_000_000, {"final": "line_then_color"}),),
        camera_after=camera_directives,
    )

    # 4. Render
    cache_root = tmp_path / "cache"
    renderer = DrawRenderer(cache_root)
    output_mp4 = tmp_path / "output.mp4"
    config = DrawProjectConfig(
        tmp_path,
        tmp_path / "effect.srt",
        tmp_path,
        resolution=(640, 360),
        fps=24,
        advanced_diagnostics=True,
    )

    result_path = renderer.render(image_path, plan, config, output_mp4, scene)
    assert result_path.is_file()

    # 5. Verify diagnostics
    diag_file = cache_root / "debug" / "001_draw_schedule.txt"
    assert diag_file.is_file()
    diag_text = diag_file.read_text(encoding="utf-8")
    assert "IMAGE 001 CAMERA CHOREOGRAPHY" in diag_text
    assert "CAMERA_AFTER:" in diag_text
    assert "Action: focus" in diag_text
    assert "Action: pan_to" in diag_text
    assert "Action: full_view" in diag_text
    assert "Final camera state:\n  FULL_VIEW" in diag_text
