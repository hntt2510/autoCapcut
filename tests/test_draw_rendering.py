from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import DrawProjectConfig
from auto_capcut.core.draw_renderer import DrawRenderer, _crop, _draw_phase_timing, _ordered_strokes, prepare_image
from auto_capcut.core.draw_renderer import DrawRenderService
from auto_capcut.core.draw_scene import save_scene
from auto_capcut.core.draw_models import DrawAction, DrawActionType, DrawImagePlan, DrawMode, DrawStyle, FinalRevealMode, NormalizedRect, SceneDocument, SceneImage, SceneObject, TextMode


def _line_then_color_fixture(tmp_path: Path, duration: str = "2") -> tuple[Path, DrawImagePlan, DrawRenderer]:
    image = tmp_path / "001.png"
    source = Image.new("RGB", (64, 64), "white")
    pixels = source.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (220, 30 + x, 40 + y) if x < 32 else (20, 90 + y, 210 - x)
    ImageDraw.Draw(source).rectangle((8, 8, 56, 56), outline="black", width=3)
    source.save(image)
    draw_end = int(float(duration) * 1_000_000)
    plan = DrawImagePlan(1, image.name, 0, draw_end, DrawMode.BASIC, DrawStyle.V1, "auto", (DrawAction(DrawActionType.DRAW, 0, draw_end, {"final": "line_then_color"}),))
    return image, plan, DrawRenderer(tmp_path / "cache")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_basic_renderer_writes_deterministic_mp4_and_cache(tmp_path: Path) -> None:
    image = tmp_path / "001.png"
    source = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(source).line((8, 8, 56, 56), fill="black", width=3)
    source.save(image)
    effect_path = tmp_path / "effect.srt"
    effect_path.write_text("1\n00:00:00,000 --> 00:00:00,200\nMODE=basic_draw\nSTYLE=v1\nDRAW 0s-0.2s: final=line_only\n", encoding="utf-8")
    effect = parse_draw_effect(effect_path)
    output = tmp_path / "out" / "001_draw.mp4"
    config = DrawProjectConfig(tmp_path, effect_path, tmp_path / "out", resolution=(64, 64), fps=10)
    renderer = DrawRenderer(config.output_folder / ".autocapcut_draw_cache")
    renderer.render(image, effect.images[0], config, output)
    assert output.is_file() and output.stat().st_size > 0
    assert (config.output_folder / ".autocapcut_draw_cache").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_advanced_renderer_uses_scene_object_and_camera_target(tmp_path: Path) -> None:
    image = tmp_path / "001.png"
    source = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(source).rectangle((16, 16, 48, 48), outline="black", width=3)
    source.save(image)
    effect_path = tmp_path / "effect.srt"
    effect_path.write_text("1\n00:00:00,000 --> 00:00:00,200\nIMAGE=001.png\nMODE=advanced_draw\nSTYLE=v1\nDRAW 0s-0.1s: order=shape final=line_only\nFOCUS 0.1s-0.2s: target=shape framing=camera_frame\n", encoding="utf-8")
    scene_path = tmp_path / "scene.json"
    save_scene(SceneDocument(1, {"001.png": SceneImage("001.png", (64, 64), (SceneObject("shape", "art", NormalizedRect(.2, .2, .6, .6), NormalizedRect(.0, .0, 1.0, 1.0)),), ("shape",))}, scene_path))
    config = DrawProjectConfig(tmp_path, effect_path, tmp_path / "out", scene_file=scene_path, resolution=(64, 64), fps=10, fallback_basic=False)
    outputs = DrawRenderService().render_project(config, parse_draw_effect(effect_path), [image])
    assert outputs[0].is_file()


def test_line_then_color_uses_full_frame_original_crossfade(tmp_path: Path) -> None:
    image, plan, renderer = _line_then_color_fixture(tmp_path)
    artifact = prepare_image(image, renderer.cache_root, plan.style, TextMode.KEEP, False)
    strokes = _ordered_strokes(artifact.strokes, plan, None)
    color_start = plan.duration_us - 600_000
    sketch = renderer._frame(artifact, plan, None, color_start, (64, 64), strokes)
    midpoint = renderer._frame(artifact, plan, None, color_start + 300_000, (64, 64), strokes)
    final = renderer._frame(artifact, plan, None, plan.duration_us, (64, 64), strokes)
    original = _crop(Image.open(artifact.cleaned_path).convert("RGB"), (0.0, 0.0, 1.0, 1.0), (64, 64))
    expected_midpoint = Image.blend(sketch.convert("RGBA"), original.convert("RGBA"), 0.5).convert("RGB")
    assert np.array_equal(np.asarray(midpoint), np.asarray(expected_midpoint))
    assert np.array_equal(np.asarray(final), np.asarray(original))
    assert not np.array_equal(np.asarray(sketch), np.asarray(original))


def test_line_then_color_preserves_original_after_settle_and_short_draw_has_both_phases(tmp_path: Path) -> None:
    image, draw_plan, renderer = _line_then_color_fixture(tmp_path, "2")
    settle_plan = DrawImagePlan(
        1,
        image.name,
        0,
        3_000_000,
        DrawMode.BASIC,
        DrawStyle.V1,
        "auto",
        (draw_plan.actions[0], DrawAction(DrawActionType.SETTLE, 2_000_000, 3_000_000)),
    )
    artifact = prepare_image(image, renderer.cache_root, settle_plan.style, TextMode.KEEP, False)
    strokes = _ordered_strokes(artifact.strokes, settle_plan, None)
    final = renderer._frame(artifact, settle_plan, None, 3_000_000, (64, 64), strokes)
    original = _crop(Image.open(artifact.cleaned_path).convert("RGB"), (0.0, 0.0, 1.0, 1.0), (64, 64))
    assert np.array_equal(np.asarray(final), np.asarray(original))
    stroke_duration, color_duration = _draw_phase_timing(200_000, FinalRevealMode.LINE_THEN_COLOR, 0, 10)
    assert stroke_duration > 0 and color_duration > 0 and stroke_duration + color_duration <= 200_000


@pytest.mark.skipif(not Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\img\001.png").is_file(), reason="Medicare fixture unavailable")
def test_medicare_final_frame_matches_original_source(tmp_path: Path) -> None:
    source_path = Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\img\001.png")
    effect = parse_draw_effect(Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\draw_effect.srt"))
    plan = effect.images[0]
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(source_path, renderer.cache_root, plan.style, TextMode.KEEP, False)
    strokes = _ordered_strokes(artifact.strokes, plan, None)
    size = (256, 144)
    final = renderer._frame(artifact, plan, None, plan.draw_action.end_us, size, strokes)
    original = _crop(Image.open(artifact.cleaned_path).convert("RGB"), (0.0, 0.0, 1.0, 1.0), size)
    assert np.array_equal(np.asarray(final), np.asarray(original))
