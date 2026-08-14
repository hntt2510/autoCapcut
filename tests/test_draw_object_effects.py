from pathlib import Path
import json

import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import (
    DrawAction,
    DrawActionType,
    DrawImagePlan,
    DrawMode,
    DrawObjectEffect,
    DrawStyle,
    TextMode,
    NormalizedRect,
    SceneDocument,
    SceneImage,
    SceneObject,
)
from auto_capcut.core.draw_renderer import (
    DrawRenderer,
    Stroke,
    _effect_state,
    _resolve_object_effects,
    build_advanced_schedule,
    prepare_image,
)
from auto_capcut.core.draw_scene import load_scene, save_scene


def _scene(tmp_path: Path, effect: str = "draw") -> tuple[Path, SceneImage]:
    image = tmp_path / "001.png"
    source = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(source).rectangle((10, 10, 35, 45), fill=(30, 100, 210))
    ImageDraw.Draw(source).rectangle((60, 15, 90, 50), fill=(210, 50, 40))
    source.save(image)
    scene = SceneImage(
        image.name,
        source.size,
        (
            SceneObject("left", "art", NormalizedRect(.1, .16, .26, .60), None, effect, "left", 700_000, 100_000),
            SceneObject("right", "warning", NormalizedRect(.6, .25, .30, .60), None, "pop_in", "auto", 500_000, None),
        ),
        ("left", "right"),
    )
    return image, scene


def test_scene_effect_metadata_round_trips(tmp_path: Path) -> None:
    image, scene = _scene(tmp_path, "slide_in")
    path = tmp_path / "scene.json"
    save_scene(SceneDocument(1, {image.name: scene}, path), path)
    loaded = load_scene(path).images[image.name]
    assert loaded.objects[0].render_effect == "slide_in"
    assert loaded.objects[0].direction == "left"
    assert loaded.objects[0].duration_us == 700_000
    assert loaded.objects[0].pause_after_us == 100_000
    assert loaded.objects[1].render_effect == "pop_in"


def test_new_spatial_objects_omit_behavior_fields(tmp_path: Path) -> None:
    image, _ = _scene(tmp_path)
    scene_path = tmp_path / "new-scene.json"
    save_scene(SceneDocument(1, {image.name: SceneImage(image.name, (100, 60), (SceneObject("new", "art", NormalizedRect(.1, .1, .2, .2)),), ("new",))}, scene_path), scene_path)
    record = json.loads(scene_path.read_text(encoding="utf-8"))["images"][image.name]["objects"][0]
    assert "render_effect" not in record
    assert "direction" not in record
    assert "duration" not in record
    assert "pause_after" not in record


def test_parser_reads_untimed_object_effect_directives_and_priority(tmp_path: Path) -> None:
    effect = tmp_path / "draw_effect.srt"
    effect.write_text(
        """1
00:00:00,000 --> 00:00:02,000
MODE=advanced_draw
STYLE=v1
OBJECT_EFFECT: target=left effect=toss_in direction=top_right duration=0.70 pause_after=0.10
DRAW 0s-2s: final=line_only
""",
        encoding="utf-8",
    )
    plan = parse_draw_effect(effect).images[0]
    assert plan.object_effects[0].target == "left"
    assert plan.object_effects[0].duration_us == 700_000
    assert plan.object_effects[0].pause_after_us == 100_000


def test_unknown_scene_effect_falls_back_only_to_draw(tmp_path: Path) -> None:
    _, scene = _scene(tmp_path, "fly_spin_999")
    plan = DrawImagePlan(1, "001.png", 0, 1_000_000, DrawMode.ADVANCED, DrawStyle.V1, "manual", (DrawAction(DrawActionType.DRAW, 0, 1_000_000, {"final": "line_only"}),))
    configs, fallbacks = _resolve_object_effects(plan, scene)
    assert configs["left"].effective_effect == DrawObjectEffect.DRAW.value
    assert configs["right"].effective_effect == DrawObjectEffect.POP_IN.value
    assert any(item.object_id == "left" for item in fallbacks)


@pytest.mark.parametrize("effect", ["slide_in", "drop_in", "toss_in", "pop_in"])
def test_entrance_effects_reach_exact_final_state(effect: str) -> None:
    box = NormalizedRect(.3, .25, .25, .4)
    state = _effect_state(box, effect, "left", 1.0)
    assert state[:4] == pytest.approx((box.x, box.y, box.w, box.h))
    assert state[4] == pytest.approx(0.0)
    assert state[5] == pytest.approx(1.0)


def test_entrance_effects_have_off_canvas_or_scaled_start_states() -> None:
    box = NormalizedRect(.3, .25, .25, .4)
    slide = _effect_state(box, "slide_in", "left", 0.0)
    drop = _effect_state(box, "drop_in", "top", 0.0)
    pop = _effect_state(box, "pop_in", "auto", 0.0)
    assert slide[0] < box.x
    assert drop[1] < box.y
    assert pop[2] < box.w and pop[3] < box.h


def test_mixed_effect_schedule_is_sequential_and_fills_sketch_budget(tmp_path: Path) -> None:
    image, scene = _scene(tmp_path, "slide_in")
    strokes = (Stroke(((.1, .2), (.2, .2))), Stroke(((.65, .3), (.8, .3))))
    duration = 2_000_000
    plan = DrawImagePlan(1, image.name, 0, duration, DrawMode.ADVANCED, DrawStyle.V1, "manual", (DrawAction(DrawActionType.DRAW, 0, duration, {"final": "line_only", "pause_each": "0.10"}),))
    schedule = build_advanced_schedule(strokes, plan, scene, scene.source_size)
    object_phases = [phase for phase in schedule.phases if phase.kind == "object"]
    assert [phase.object_id for phase in object_phases] == ["left", "right"]
    assert sum(phase.duration_us for phase in schedule.phases) == schedule.sketch_end_us


def test_advanced_render_writes_object_cache_and_preserves_final_source(tmp_path: Path) -> None:
    image, scene = _scene(tmp_path, "slide_in")
    plan = DrawImagePlan(1, image.name, 0, 1_000_000, DrawMode.ADVANCED, DrawStyle.V1, "manual", (DrawAction(DrawActionType.DRAW, 0, 1_000_000, {"final": "line_then_color"}),))
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(image, renderer.cache_root, plan.style, TextMode.KEEP, False)
    output = tmp_path / "out" / "001_draw.mp4"
    from auto_capcut.core.draw_models import DrawProjectConfig

    renderer.render(image, plan, DrawProjectConfig(tmp_path, tmp_path / "effect.srt", tmp_path / "out", resolution=(100, 60), fps=10), output, scene)
    assert output.is_file()
    assert list(artifact.folder.glob("objects/*/*/crop.png"))
