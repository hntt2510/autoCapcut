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
    _auto_camera_return_duration_us,
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


def test_auto_camera_return_duration_in_expected_range() -> None:
    focused = CameraState((0.20, 0.20, 0.40, 0.40))
    ret_dur = _auto_camera_return_duration_us(focused, FULL_VIEW_STATE)
    # Range must be 0.35s – 0.55s (350_000 – 550_000 us)
    assert 350_000 <= ret_dur <= 550_000


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
    assert target_state.w == pytest.approx(0.50)
    assert target_state.h == pytest.approx(0.50)
    assert target_state.center_x == pytest.approx(0.60, abs=0.01)
    assert target_state.center_y == pytest.approx(0.50, abs=0.01)
    assert not is_clamped


def test_pan_to_minimally_zooms_out_if_target_too_large() -> None:
    current_state = CameraState((0.10, 0.10, 0.30, 0.30))
    large_obj = SceneObject(
        "big_chart",
        "art",
        NormalizedRect(0.20, 0.20, 0.60, 0.50),
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
    assert target_state.w >= 0.60
    assert target_state.h >= 0.50


def test_pull_to_fallback_when_target_requires_zoom_in() -> None:
    current_state = CameraState((0.0, 0.0, 1.0, 1.0))
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

    assert _interpolate_camera_state(start, end, 0.0).viewport == start.viewport
    assert _interpolate_camera_state(start, end, 1.0).viewport == end.viewport
    assert _interpolate_camera_state(start, end, -0.5).viewport == start.viewport
    assert _interpolate_camera_state(start, end, 1.5).viewport == end.viewport

    mid = _interpolate_camera_state(start, end, 0.5, easing="ease_in_out")
    assert 0.0 < mid.x < 0.25
    assert 0.0 < mid.y < 0.25
    assert 0.50 < mid.w < 1.00
    assert 0.50 < mid.h < 1.00
    assert mid.x + mid.w <= 1.0
    assert mid.y + mid.h <= 1.0


# ==============================================================================
# 2. TRANSIENT EMPHASIS CAMERA & SCHEDULE TESTS (A through J)
# ==============================================================================

def _make_test_scene(order=("obj_a", "obj_b", "obj_c")) -> SceneImage:
    objects = (
        SceneObject("obj_a", "art", NormalizedRect(0.10, 0.10, 0.25, 0.80), NormalizedRect(0.05, 0.05, 0.35, 0.90), "draw"),
        SceneObject("obj_b", "art", NormalizedRect(0.40, 0.10, 0.25, 0.80), NormalizedRect(0.35, 0.05, 0.35, 0.90), "slide_in", "left"),
        SceneObject("obj_c", "art", NormalizedRect(0.70, 0.10, 0.25, 0.80), NormalizedRect(0.65, 0.05, 0.35, 0.90), "push_in", "left"),
    )
    return SceneImage("001.png", (1920, 1080), objects, tuple(order))


def test_a_transient_focus_lifecycle_and_schedule_phases() -> None:
    """Test A: FULL_VIEW -> object_1 draw -> focus -> hold -> focus_out -> FULL_VIEW -> object_2 slide."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    camera_directives = (
        CameraAfterDirective(
            object_id="obj_a",
            action="focus",
            target="obj_a",
            duration_us=500_000,
            duration_mode="fixed",
            hold_us=150_000,
            return_duration_us=450_000,
            return_duration_mode="fixed",
        ),
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
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    # Collect phase kinds for obj_a
    obj_a_phases = [p for p in schedule.phases if p.object_id == "obj_a"]
    kinds = [p.kind for p in obj_a_phases]
    assert kinds == ["object", "finalize", "camera", "camera_hold", "camera_return"]

    cam_in = next(p for p in obj_a_phases if p.kind == "camera")
    hold = next(p for p in obj_a_phases if p.kind == "camera_hold")
    cam_out = next(p for p in obj_a_phases if p.kind == "camera_return")

    assert cam_in.camera_start == FULL_VIEW_STATE
    assert cam_in.camera_end.viewport == (0.05, 0.05, 0.35, 0.90)
    assert cam_in.duration_us == 500_000

    assert hold.camera_start == cam_in.camera_end
    assert hold.camera_end == cam_in.camera_end
    assert hold.duration_us == 150_000

    assert cam_out.camera_start == cam_in.camera_end
    assert cam_out.camera_end == FULL_VIEW_STATE
    assert cam_out.duration_us == 450_000


def test_b_subsequent_object_entrance_begins_in_full_view() -> None:
    """Test B: object_2 slide begins with camera FULL_VIEW."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=500_000, duration_mode="fixed", hold_us=100_000),
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
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    obj_b_phase = next(p for p in schedule.phases if p.object_id == "obj_b" and p.kind == "object")
    assert obj_b_phase.camera_start == FULL_VIEW_STATE
    assert obj_b_phase.camera_end == FULL_VIEW_STATE

    cam_at_start = _camera_state_at(schedule, plan, scene, obj_b_phase.start_us, 16 / 9, (1920, 1080))
    assert cam_at_start.viewport == FULL_VIEW_STATE.viewport


def test_c_object_without_camera_after_creates_no_camera_movement() -> None:
    """Test C: object without CAMERA_AFTER creates NO camera movement."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    # CAMERA_AFTER only on obj_a; obj_b has none
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=400_000, duration_mode="fixed", hold_us=100_000),
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
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    obj_b_phases = [p for p in schedule.phases if p.object_id == "obj_b"]
    assert all(p.kind not in {"camera", "camera_hold", "camera_return", "camera_staging"} for p in obj_b_phases)

    obj_b_phase = next(p for p in obj_b_phases if p.kind == "object")
    mid_time = (obj_b_phase.start_us + obj_b_phase.end_us) // 2
    cam_mid = _camera_state_at(schedule, plan, scene, mid_time, 16 / 9, (1920, 1080))
    assert cam_mid.viewport == FULL_VIEW_STATE.viewport


def test_d_focus_return_ends_exactly_at_full_view() -> None:
    """Test D: focus return ends exactly at FULL_VIEW without precision drift."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=500_000, duration_mode="fixed", hold_us=100_000, return_duration_us=400_000, return_duration_mode="fixed"),
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

    cam_return_phase = next(p for p in schedule.phases if p.kind == "camera_return" and p.object_id == "obj_a")
    st_at_end = _camera_state_at(schedule, plan, scene, cam_return_phase.end_us, 16 / 9, (1920, 1080))
    assert st_at_end.viewport == (0.0, 0.0, 1.0, 1.0)


def test_e_two_separate_focus_events_do_not_accumulate_camera_drift() -> None:
    """Test E: two separate focus events do not accumulate camera drift."""
    scene = _make_test_scene()
    strokes = (
        Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),
    )
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=400_000, duration_mode="fixed", hold_us=100_000, return_duration_us=400_000, return_duration_mode="fixed"),
        CameraAfterDirective(object_id="obj_c", action="focus", target="obj_c", duration_us=400_000, duration_mode="fixed", hold_us=100_000, return_duration_us=400_000, return_duration_mode="fixed"),
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
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    cam_ret_a = next(p for p in schedule.phases if p.kind == "camera_return" and p.object_id == "obj_a")
    st_after_a = _camera_state_at(schedule, plan, scene, cam_ret_a.end_us, 16 / 9, (1920, 1080))
    assert st_after_a.viewport == (0.0, 0.0, 1.0, 1.0)

    cam_ret_c = next(p for p in schedule.phases if p.kind == "camera_return" and p.object_id == "obj_c")
    st_after_c = _camera_state_at(schedule, plan, scene, cam_ret_c.end_us, 16 / 9, (1920, 1080))
    assert st_after_c.viewport == (0.0, 0.0, 1.0, 1.0)


def test_f_push_in_completes_before_focus_starts() -> None:
    """Test F: push_in completes before focus starts."""
    scene = _make_test_scene()
    camera_directives = (
        CameraAfterDirective(object_id="obj_c", action="focus", target="obj_c", duration_us=450_000, duration_mode="fixed", hold_us=150_000),
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
    schedule = _build_advanced_schedule((), plan, scene, (1920, 1080))

    obj_c_phase = next(p for p in schedule.phases if p.object_id == "obj_c" and p.kind == "object")
    cam_c_phase = next(p for p in schedule.phases if p.object_id == "obj_c" and p.kind == "camera")

    assert obj_c_phase.end_us <= cam_c_phase.start_us
    assert obj_c_phase.camera_start == FULL_VIEW_STATE
    assert obj_c_phase.camera_end == FULL_VIEW_STATE


def test_g_focus_out_completes_before_next_object_begins() -> None:
    """Test G: focus_out completes before next object begins."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=450_000, duration_mode="fixed", hold_us=100_000, return_duration_us=400_000, return_duration_mode="fixed"),
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
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    cam_return = next(p for p in schedule.phases if p.kind == "camera_return" and p.object_id == "obj_a")
    obj_b = next(p for p in schedule.phases if p.kind == "object" and p.object_id == "obj_b")

    assert cam_return.end_us <= obj_b.start_us


def test_h_timeline_budget_includes_focus_out_duration() -> None:
    """Test H: timeline budget includes focus-out duration."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    # Fixed focus-in 1.0s + hold 0.5s + return 1.0s = 2.5s > 2.0s DRAW cue
    camera_directives = (
        CameraAfterDirective(
            object_id="obj_a",
            action="focus",
            target="obj_a",
            duration_us=1_000_000,
            duration_mode="fixed",
            hold_us=500_000,
            return_duration_us=1_000_000,
            return_duration_mode="fixed",
        ),
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


def test_i_legacy_explicit_persist_retains_camera_when_requested() -> None:
    """Test I: legacy explicit persist=true retains camera when requested."""
    objects = (
        SceneObject("obj_a", "art", NormalizedRect(0.10, 0.10, 0.20, 0.40), NormalizedRect(0.05, 0.05, 0.60, 0.60), "draw"),
        SceneObject("obj_b", "art", NormalizedRect(0.30, 0.10, 0.20, 0.40), NormalizedRect(0.05, 0.05, 0.60, 0.60), "push_in", "left"),
        SceneObject("obj_c", "art", NormalizedRect(0.70, 0.10, 0.20, 0.40), NormalizedRect(0.50, 0.05, 0.50, 0.60), "draw"),
    )
    scene = SceneImage("001.png", (1920, 1080), objects, ("obj_a", "obj_b", "obj_c"))
    strokes = (
        Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),
        Stroke(((0.75, 0.20), (0.85, 0.40)), object_id="obj_c"),
    )
    camera_directives = (
        CameraAfterDirective(
            object_id="obj_a",
            action="focus",
            target="obj_a",
            duration_us=500_000,
            duration_mode="fixed",
            hold_us=100_000,
            persist=True,
        ),
        CameraAfterDirective(
            object_id="obj_c",
            action="full_view",
            target="",
            duration_us=500_000,
            duration_mode="fixed",
        ),
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
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    # obj_a should have no camera_return phase
    obj_a_phases = [p for p in schedule.phases if p.object_id == "obj_a"]
    assert all(p.kind != "camera_return" for p in obj_a_phases)

    # obj_b runs in focused camera state of obj_a
    obj_b_phase = next(p for p in schedule.phases if p.kind == "object" and p.object_id == "obj_b")
    cam_during_b = _camera_state_at(schedule, plan, scene, obj_b_phase.start_us, 16 / 9, (1920, 1080))
    assert cam_during_b.viewport == (0.05, 0.05, 0.60, 0.60)


def test_j_files_with_no_camera_after_remain_identical_to_baseline() -> None:
    """Test J: files with no CAMERA_AFTER remain in FULL_VIEW at all times."""
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        5_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 5_000_000, {"final": "line_then_color"}),),
        camera_after=(),
    )
    schedule = _build_advanced_schedule(strokes, plan, scene, (1920, 1080))

    camera_phases = [p for p in schedule.phases if p.kind in {"camera", "camera_hold", "camera_return", "camera_staging"}]
    assert len(camera_phases) == 0

    for t_sample in [0, 500_000, 1_500_000, 3_000_000, 4_500_000, 5_000_000]:
        st = _camera_state_at(schedule, plan, scene, t_sample, 16 / 9, (1920, 1080))
        assert st.viewport == FULL_VIEW_STATE.viewport


def test_final_reconciliation_requires_full_view_state_with_persisted_camera() -> None:
    scene = _make_test_scene()
    strokes = (Stroke(((0.15, 0.20), (0.25, 0.40)), object_id="obj_a"),)
    # obj_a focuses with persist=True, leaving camera zoomed at settle
    camera_directives = (
        CameraAfterDirective(object_id="obj_a", action="focus", target="obj_a", duration_us=500_000, duration_mode="fixed", persist=True),
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


def test_entrance_staging_pans_to_offscreen_upcoming_object() -> None:
    objects = (
        SceneObject("left_hero", "art", NormalizedRect(0.05, 0.20, 0.25, 0.60), NormalizedRect(0.02, 0.10, 0.35, 0.80), "draw"),
        SceneObject("right_card", "art", NormalizedRect(0.70, 0.20, 0.25, 0.60), NormalizedRect(0.65, 0.10, 0.35, 0.80), "slide_in", "right"),
        SceneObject("final_badge", "art", NormalizedRect(0.40, 0.40, 0.20, 0.20), None, "draw"),
    )
    scene = SceneImage("001.png", (1920, 1080), objects, ("left_hero", "right_card", "final_badge"))
    strokes = (
        Stroke(((0.10, 0.25), (0.20, 0.45)), object_id="left_hero"),
        Stroke(((0.45, 0.45), (0.55, 0.55)), object_id="final_badge"),
    )
    # Persist camera on left_hero to verify staging pan when right_card is off-screen
    camera_directives = (
        CameraAfterDirective(object_id="left_hero", action="focus", target="left_hero", duration_us=500_000, duration_mode="fixed", hold_us=100_000, persist=True),
        CameraAfterDirective(object_id="final_badge", action="full_view", target="", duration_us=500_000, duration_mode="fixed"),
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

    staging_phases = [p for p in schedule.phases if p.kind == "camera_staging"]
    assert len(staging_phases) == 2
    assert [p.object_id for p in staging_phases] == ["right_card", "final_badge"]
    assert staging_phases[0].camera_action == "staging_pan"
    assert staging_phases[0].camera_end.x >= 0.50


# ==============================================================================
# 3. END-TO-END MEDICARE ACCEPTANCE TEST WITH FULL 6 OBJECTS
# ==============================================================================

def test_medicare_scene_camera_choreography_acceptance(tmp_path: Path) -> None:
    """Acceptance test verifying full 6-object Medicare scene with restrained editorial camera choreography."""
    # 1. Create clean Medicare fixture
    image_path = tmp_path / "001.png"
    img = Image.new("RGB", (640, 360), (242, 245, 248))
    draw = ImageDraw.Draw(img)

    # Object 1: Alex + US map
    draw.rectangle((220, 90, 420, 270), fill=(100, 140, 200), outline=(30, 40, 60), width=3)
    draw.text((260, 170), "ALEX + MAP", fill=(255, 255, 255))
    # Object 2: Left card (Part A&B)
    draw.rectangle((40, 90, 180, 230), fill=(220, 100, 90), outline=(40, 20, 20), width=2)
    # Object 3: Right card (Part C)
    draw.rectangle((460, 90, 600, 230), fill=(90, 180, 110), outline=(20, 40, 20), width=2)
    # Object 4: Left bottom (Part D)
    draw.rectangle((40, 250, 180, 340), fill=(240, 200, 60), outline=(50, 40, 10), width=2)
    # Object 5: Bottom badge (Medigap)
    draw.rectangle((260, 290, 380, 340), fill=(180, 120, 220), outline=(40, 20, 50), width=2)
    # Object 6: Right bottom (Penalty)
    draw.rectangle((460, 250, 600, 340), fill=(70, 160, 240), outline=(20, 30, 60), width=2)
    img.save(image_path)

    # 2. Scene metadata
    scene_objects = (
        SceneObject("object_1", "art", NormalizedRect(220 / 640, 90 / 360, 200 / 640, 180 / 360), NormalizedRect(0.30, 0.20, 0.40, 0.60), "draw"),
        SceneObject("object_2", "art", NormalizedRect(40 / 640, 90 / 360, 140 / 640, 140 / 360), NormalizedRect(0.04, 0.20, 0.30, 0.50), "slide_in", "left"),
        SceneObject("object_3", "art", NormalizedRect(460 / 640, 90 / 360, 140 / 640, 140 / 360), NormalizedRect(0.66, 0.20, 0.30, 0.50), "toss_in", "top"),
        SceneObject("object_4", "art", NormalizedRect(40 / 640, 250 / 360, 140 / 640, 90 / 360), NormalizedRect(0.04, 0.65, 0.30, 0.30), "push_in", "left"),
        SceneObject("object_5", "art", NormalizedRect(260 / 640, 290 / 360, 120 / 640, 50 / 360), NormalizedRect(0.38, 0.75, 0.24, 0.20), "pop_in"),
        SceneObject("object_6", "art", NormalizedRect(460 / 640, 250 / 360, 140 / 640, 90 / 360), NormalizedRect(0.66, 0.65, 0.30, 0.30), "push_in", "top"),
    )
    scene = SceneImage("001.png", (640, 360), scene_objects, ("object_1", "object_2", "object_3", "object_4", "object_5", "object_6"))

    # 3. Plan with restrained editorial camera choreography (focus on 1, 4, 6 only; all transient)
    camera_directives = (
        CameraAfterDirective("object_1", "focus", "object_1", duration_us=500_000, duration_mode="fixed", hold_us=150_000, return_duration_us=450_000, return_duration_mode="fixed"),
        CameraAfterDirective("object_4", "focus", "object_4", duration_us=450_000, duration_mode="fixed", hold_us=150_000, return_duration_us=400_000, return_duration_mode="fixed"),
        CameraAfterDirective("object_6", "focus", "object_6", duration_us=450_000, duration_mode="fixed", hold_us=180_000, return_duration_us=450_000, return_duration_mode="fixed"),
    )
    plan = DrawImagePlan(
        1,
        "001.png",
        0,
        12_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V2,
        "manual",
        (
            DrawAction(DrawActionType.DRAW, 0, 11_400_000, {"final": "line_then_color", "pause_each": "0.08"}),
            DrawAction(DrawActionType.SETTLE, 11_400_000, 12_000_000, {}),
        ),
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
    assert "FOCUS:" in diag_text
    assert "HOLD:" in diag_text
    assert "FOCUS RETURN:" in diag_text
    assert "Camera after focus: FULL_VIEW" in diag_text
    assert "Final camera state:\n  FULL_VIEW" in diag_text
