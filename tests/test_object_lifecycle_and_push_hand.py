from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import (
    DrawAction,
    DrawActionType,
    DrawImagePlan,
    DrawMode,
    DrawObjectDirection,
    DrawObjectEffect,
    DrawProjectConfig,
    DrawStyle,
    FinalRevealMode,
    NormalizedRect,
    ObjectEffectOverride,
    SceneDocument,
    SceneImage,
    SceneObject,
    TextMode,
)
from auto_capcut.core.draw_renderer import (
    DrawRenderer,
    Stroke,
    _box_pixels,
    _build_advanced_schedule,
    _crop,
    _effect_state,
    _prepare_object_layers,
    _resolve_object_effects,
    build_advanced_schedule,
    prepare_image,
)
from auto_capcut.core.draw_scene import save_scene


def _two_object_scene(tmp_path: Path) -> tuple[Path, SceneImage]:
    image = tmp_path / "001.png"
    source = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(source)
    # Object 1 (left): bright red box
    draw.rectangle((10, 10, 40, 90), fill=(255, 0, 0), outline=(20, 20, 20), width=2)
    # Object 2 (right): bright blue box
    draw.rectangle((60, 10, 90, 90), fill=(0, 0, 255), outline=(20, 20, 20), width=2)
    source.save(image)
    scene = SceneImage(
        image.name,
        source.size,
        (
            SceneObject("obj_red", "art", NormalizedRect(0.10, 0.10, 0.30, 0.80), None, "draw", "auto", 500_000, None),
            SceneObject("obj_blue", "art", NormalizedRect(0.60, 0.10, 0.30, 0.80), None, "slide_in", "right", 500_000, None),
        ),
        ("obj_red", "obj_blue"),
    )
    return image, scene


def test_draw_object_finalizes_and_remains_color_while_next_object_animates(tmp_path: Path) -> None:
    image, scene = _two_object_scene(tmp_path)
    plan = DrawImagePlan(
        1,
        image.name,
        0,
        1_500_000,
        DrawMode.ADVANCED,
        DrawStyle.V1,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 1_500_000, {"final": "line_then_color"}),),
    )
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(image, renderer.cache_root, plan.style, TextMode.KEEP, False)
    schedule = build_advanced_schedule(artifact.strokes, plan, scene, (100, 100), TextMode.KEEP)

    red_phase = next(p for p in schedule.phases if p.object_id == "obj_red" and p.kind == "object")
    blue_phase = next(p for p in schedule.phases if p.object_id == "obj_blue" and p.kind == "object")

    # 1. During obj_red drawing: obj_red is sketch, obj_blue not started
    frame_during_red = renderer._frame(artifact, plan, scene, red_phase.start_us + 100_000, (100, 100), schedule)
    arr_during_red = np.asarray(frame_during_red)
    # Inside red box: pixel is not full red yet
    red_box_sample = arr_during_red[50, 25]
    assert not (red_box_sample[0] > 200 and red_box_sample[1] < 50 and red_box_sample[2] < 50)

    # 2. After obj_red finalize (t = red_phase.end_us + 250_000): obj_red MUST be full color (red)
    t_red_done = red_phase.end_us + 250_000
    frame_after_red = renderer._frame(artifact, plan, scene, t_red_done, (100, 100), schedule)
    arr_after_red = np.asarray(frame_after_red)
    red_sample_done = arr_after_red[50, 25]
    assert red_sample_done[0] > 200 and red_sample_done[1] < 50 and red_sample_done[2] < 50, f"Expected red, got {red_sample_done}"

    # 3. While obj_blue is actively sliding in: obj_red MUST STILL be full color (red)
    t_blue_mid = (blue_phase.start_us + blue_phase.end_us) // 2
    frame_during_blue = renderer._frame(artifact, plan, scene, t_blue_mid, (100, 100), schedule)
    arr_during_blue = np.asarray(frame_during_blue)
    red_sample_during_blue = arr_during_blue[50, 25]
    assert red_sample_during_blue[0] > 200 and red_sample_during_blue[1] < 50 and red_sample_during_blue[2] < 50, "obj_red must NOT revert to sketch while obj_blue is animating"


def test_push_hand_state_and_contact_anchor() -> None:
    renderer = DrawRenderer(Path("cache"))
    box = NormalizedRect(0.30, 0.20, 0.40, 0.50)

    from auto_capcut.core.draw_renderer import ScheduledGroup
    group_left = ScheduledGroup("obj_push", "art", (), 1_000_000, 0.0, "push_in", "left", box)
    group_right = ScheduledGroup("obj_push", "art", (), 1_000_000, 0.0, "push_in", "right", box)
    group_top = ScheduledGroup("obj_push", "art", (), 1_000_000, 0.0, "push_in", "top", box)

    # 1. Approach phase (progress = 0.05)
    st_app = renderer._push_hand_state(group_left, 0.05)
    assert st_app is not None
    pt_app, op_app = st_app
    assert 0.0 < op_app < 1.0
    # Hand is approaching from the left of the starting object x
    x_start = _effect_state(box, "push_in", "left", 0.0)[0]
    assert pt_app[0] < x_start

    # 2. Joint push phase (progress = 0.50)
    st_push = renderer._push_hand_state(group_left, 0.50)
    assert st_push is not None
    pt_push, op_push = st_push
    assert op_push == pytest.approx(1.0)
    curr_x, curr_y, _, curr_h, _, _ = _effect_state(box, "push_in", "left", 0.50)
    # Hand contact point touches left edge: (curr_x, curr_y + curr_h/2)
    assert pt_push[0] == pytest.approx(curr_x)
    assert pt_push[1] == pytest.approx(curr_y + curr_h / 2)

    # 3. Retract phase (progress = 0.95)
    st_ret = renderer._push_hand_state(group_left, 0.95)
    assert st_ret is not None
    pt_ret, op_ret = st_ret
    assert 0.0 < op_ret < 1.0
    # Hand has travelled slightly past box.x
    assert pt_ret[0] >= box.x

    # 4. Test right push contact point on right edge
    st_r = renderer._push_hand_state(group_right, 0.50)
    assert st_r is not None
    curr_xr, curr_yr, curr_wr, curr_hr, _, _ = _effect_state(box, "push_in", "right", 0.50)
    assert st_r[0][0] == pytest.approx(curr_xr + curr_wr)
    assert st_r[0][1] == pytest.approx(curr_yr + curr_hr / 2)

    # 5. Test top push contact point on top edge
    st_t = renderer._push_hand_state(group_top, 0.50)
    assert st_t is not None
    curr_xt, curr_yt, curr_wt, _, _, _ = _effect_state(box, "push_in", "top", 0.50)
    assert st_t[0][0] == pytest.approx(curr_xt + curr_wt / 2)
    assert st_t[0][1] == pytest.approx(curr_yt)


def test_diagnostics_contain_lifecycle_and_push_hand_details(tmp_path: Path) -> None:
    image = tmp_path / "001.png"
    source = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(source).rectangle((10, 10, 40, 90), fill=(255, 0, 0))
    ImageDraw.Draw(source).rectangle((60, 10, 90, 90), fill=(0, 255, 0))
    source.save(image)
    scene = SceneImage(
        image.name,
        source.size,
        (
            SceneObject("draw_obj", "art", NormalizedRect(0.10, 0.10, 0.30, 0.80), None, "draw", "auto", 500_000, None),
            SceneObject("push_obj", "art", NormalizedRect(0.60, 0.10, 0.30, 0.80), None, "push_in", "left", 700_000, None),
        ),
        ("draw_obj", "push_obj"),
    )
    plan = DrawImagePlan(
        1,
        image.name,
        0,
        2_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V1,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 2_000_000, {"final": "line_then_color"}),),
    )
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(image, renderer.cache_root, plan.style, TextMode.KEEP, False)
    schedule = build_advanced_schedule(artifact.strokes, plan, scene, (100, 100), TextMode.KEEP)
    renderer._write_schedule_diagnostics(schedule, plan, renderer.cache_root)

    diag_file = renderer.cache_root / "debug" / "001_draw_schedule.txt"
    assert diag_file.is_file()
    content = diag_file.read_text(encoding="utf-8")

    assert "Object: draw_obj" in content
    assert "Local color reveal:" in content
    assert "DONE at:" in content

    assert "Object: push_obj" in content
    assert "Resolved effect: push_in" in content
    assert "Direction: left" in content
    assert "Push hand asset: push_hand_side.png" in content
    assert "Z-order: hand above object" in content


def test_push_fallback_logged_loudly(tmp_path: Path) -> None:
    # bottom push is unsupported in V1 and must fall back to draw with clear logging
    image = tmp_path / "001.png"
    source = Image.new("RGB", (60, 60), "white")
    source.save(image)
    scene = SceneImage(
        image.name,
        source.size,
        (SceneObject("obj_bot", "art", NormalizedRect(0.2, 0.2, 0.6, 0.6), None, "push_in", "bottom", 500_000, None),),
        ("obj_bot",),
    )
    plan = DrawImagePlan(1, image.name, 0, 1_000_000, DrawMode.ADVANCED, DrawStyle.V1, "manual", (DrawAction(DrawActionType.DRAW, 0, 1_000_000, {}),))
    configs, fallbacks = _resolve_object_effects(plan, scene)
    assert configs["obj_bot"].effective_effect == DrawObjectEffect.DRAW.value
    assert any(f.object_id == "obj_bot" and f.requested_effect == "push_in" and f.effective_effect == "draw" for f in fallbacks)


def test_draw_object_finalization_no_rectangular_background_seam(tmp_path: Path) -> None:
    # Fixture: slightly off-white/light-gray background (e.g. 235, 240, 242)
    bg_color = (235, 240, 242)
    image = tmp_path / "001.png"
    source = Image.new("RGB", (100, 100), bg_color)
    draw = ImageDraw.Draw(source)
    # Inside ROI (20..80, 20..80), place a foreground red square at (35..65, 35..65)
    draw.rectangle((35, 35, 65, 65), fill=(255, 0, 0), outline=(20, 20, 20), width=2)
    source.save(image)

    scene = SceneImage(
        image.name,
        source.size,
        (SceneObject("center_obj", "art", NormalizedRect(0.20, 0.20, 0.60, 0.60), None, "draw", "auto", 500_000, None),),
        ("center_obj",),
    )
    plan = DrawImagePlan(
        1,
        image.name,
        0,
        1_200_000,
        DrawMode.ADVANCED,
        DrawStyle.V1,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, 1_200_000, {"final": "line_then_color"}),),
    )
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(image, renderer.cache_root, plan.style, TextMode.KEEP, False)
    schedule = build_advanced_schedule(artifact.strokes, plan, scene, (100, 100), TextMode.KEEP)

    obj_phase = next(p for p in schedule.phases if p.object_id == "center_obj" and p.kind == "object")

    # Check at a frame AFTER local color finalization (t = end + 250_000) but BEFORE final scene reconciliation
    t_test = obj_phase.end_us + 250_000
    assert t_test < plan.draw_action.end_us - 200_000  # Before final reconciliation!

    frame = renderer._frame(artifact, plan, scene, t_test, (100, 100), schedule)
    arr = np.asarray(frame)

    # 1. Verify foreground pixels recovered original colors (red)
    fg_pixel = arr[50, 50]
    assert fg_pixel[0] > 200 and fg_pixel[1] < 50 and fg_pixel[2] < 50, f"Foreground should be red, got {fg_pixel}"

    # 2. Verify no rectangular background seam:
    # Pixel just inside ROI background (e.g. at (25, 25)) vs pixel just outside ROI (e.g. at (15, 15))
    inside_bg_pixel = arr[25, 25].astype(float)
    outside_bg_pixel = arr[15, 15].astype(float)
    diff = np.linalg.norm(inside_bg_pixel - outside_bg_pixel)
    assert diff < 5.0, f"Rectangular seam detected: inside ROI bg {inside_bg_pixel} vs outside ROI {outside_bg_pixel}, diff={diff}"

    # Verify boundaries on top-left and bottom-right
    diff_tl = np.linalg.norm(arr[22, 22].astype(float) - arr[18, 18].astype(float))
    diff_br = np.linalg.norm(arr[78, 78].astype(float) - arr[82, 82].astype(float))
    assert diff_tl < 5.0
    assert diff_br < 5.0


@pytest.mark.skipif(not Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\img\001.png").is_file(), reason="Medicare fixture unavailable")
def test_medicare_scene_draw_object_finalization_no_roi_rectangle(tmp_path: Path) -> None:
    img_path = Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\img\001.png")
    effect_path = Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\draw_effect.srt")

    plan = parse_draw_effect(effect_path).images[0]
    scene = SceneImage(
        "001.png",
        (1376, 768),
        (
            SceneObject("object_1", "art", NormalizedRect(0.25, 0.15, 0.45, 0.70), None, "draw", "auto"),
            SceneObject("object_2", "art", NormalizedRect(0.0, 0.087, 0.38, 0.384), None, "slide_in", "left", 700000),
            SceneObject("object_3", "art", NormalizedRect(0.64, 0.0, 0.358, 0.362), None, "toss_in", "top_right", 700000),
            SceneObject("object_4", "art", NormalizedRect(0.0, 0.521, 0.36, 0.362), None, "push_in", "left", 700000),
            SceneObject("object_5", "art", NormalizedRect(0.0, 0.13, 0.344, 0.347), None, "pop_in", "auto", 500000),
            SceneObject("object_6", "art", NormalizedRect(0.627, 0.614, 0.36, 0.362), None, "push_in", "top", 700000),
        ),
        ("object_1", "object_2", "object_3", "object_4", "object_5", "object_6"),
    )

    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(img_path, renderer.cache_root, plan.style, TextMode.KEEP, False)
    effect_configs, config_fallbacks = _resolve_object_effects(plan, scene)
    layers, effect_configs, layer_fallbacks = _prepare_object_layers(artifact, scene, effect_configs)
    schedule = _build_advanced_schedule(artifact.strokes, plan, scene, (1376, 768), TextMode.KEEP, effect_configs, layers, config_fallbacks + layer_fallbacks)

    alex_phase = next(p for p in schedule.phases if p.object_id == "object_1" and p.kind == "object")
    # After Alex is done in color (t = alex_phase.end_us + 250_000, e.g. ~1.43s), before final reconciliation
    t_test = alex_phase.end_us + 250_000
    frame = renderer._frame(artifact, plan, scene, t_test, (1376, 768), schedule)
    arr = np.asarray(frame)

    left, top, right, bottom = _box_pixels(NormalizedRect(0.25, 0.15, 0.45, 0.70), (1376, 768))
    bg_out = arr[50, 50].astype(float)

    # 1. Check all 4 empty background corners inside Alex ROI against canvas background
    for corner_name, py, px in [
        ("top-left", top + 20, left + 20),
        ("top-right", top + 20, right - 20),
        ("bottom-left", bottom - 20, left + 20),
        ("bottom-right", bottom - 20, right - 20),
    ]:
        p = arr[py, px].astype(float)
        diff = np.linalg.norm(p - bg_out)
        assert diff < 5.0, f"Seam detected in Alex ROI {corner_name} at ({px}, {py}): diff={diff:.2f}"

    # 2. Alex center foreground color is preserved
    alex_center = arr[round(0.50 * 768), round(0.50 * 1376)]
    orig = np.asarray(Image.open(artifact.cleaned_path).convert("RGB"))
    orig_center = orig[round(0.50 * 768), round(0.50 * 1376)]
    assert np.linalg.norm(alex_center.astype(float) - orig_center.astype(float)) < 2.0


