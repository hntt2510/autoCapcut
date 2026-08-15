from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import (
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
    _basic_schedule,
    _build_advanced_schedule,
    _camera_state_at,
    calculate_completion_buffer_us,
    prepare_image,
)
from auto_capcut.core.draw_scene import save_scene
from auto_capcut.core.unified_effect_parser import parse_unified_effect
from auto_capcut.models import ProjectConfig, ProjectJob, Resolution
from auto_capcut.core.capcut_builder import CapCutBuilder


def _make_dummy_image(path: Path, color: str = "white") -> Path:
    img = Image.new("RGB", (120, 80), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 100, 60), fill="blue", outline="black", width=2)
    img.save(path)
    return path


def test_missing_mode_defaults_to_basic_draw(tmp_path: Path) -> None:
    srt_path = tmp_path / "effect.srt"
    srt_path.write_text(
        """1
00:00:00,000 --> 00:00:05,000
COMPLETE_BEFORE_END 2.0s
POST_MOTION subtle_zoom_in
""",
        encoding="utf-8",
    )
    parsed = parse_unified_effect(srt_path)
    assert len(parsed.cues) == 1
    cue = parsed.cues[0]
    assert cue.kind == "draw"
    assert cue.draw_plan is not None
    assert cue.draw_plan.mode is DrawMode.BASIC
    assert cue.draw_plan.complete_before_end_us == 2_000_000
    assert cue.draw_plan.post_motion == "subtle_zoom_in"


def test_completion_buffer_calculation() -> None:
    # Explicit buffer
    assert calculate_completion_buffer_us(10_000_000, 2_500_000) == 2_500_000
    # Default buffer on 10s clip: 25% of 10s = 2.5s -> 2_500_000
    assert calculate_completion_buffer_us(10_000_000, None) == 2_500_000
    # Default buffer on 4s clip: 25% of 4s = 1.0s -> clamped min 1.5s -> 1_500_000
    assert calculate_completion_buffer_us(4_000_000, None) == 1_500_000
    # Default buffer on 20s clip: 25% of 20s = 5.0s -> clamped max 3.0s -> 3_000_000
    assert calculate_completion_buffer_us(20_000_000, None) == 3_000_000


def test_basic_draw_schedule_respects_completion_buffer(tmp_path: Path) -> None:
    img_path = _make_dummy_image(tmp_path / "001.png")
    plan = DrawImagePlan(
        1,
        img_path.name,
        0,
        6_000_000,
        DrawMode.BASIC,
        DrawStyle.V1,
        "auto",
        (),
        complete_before_end_us=2_000_000,
        post_motion="subtle_pan_right",
    )
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(img_path, renderer.cache_root, plan.style, TextMode.KEEP, False)
    schedule = _basic_schedule(artifact.strokes, plan)
    # Drawing finishes at 4.0s (6.0s - 2.0s buffer)
    assert schedule.draw_end_us == 4_000_000

    # Before draw end: camera is full view (0, 0, 1, 1)
    cam_mid = _camera_state_at(schedule, plan, None, 2_000_000, 1.5, (120, 80))
    assert cam_mid.viewport == (0.0, 0.0, 1.0, 1.0)

    # During completion buffer (e.g. t = 5.0s, half-way through buffer): camera moves according to subtle_pan_right
    cam_buffer = _camera_state_at(schedule, plan, None, 5_000_000, 1.5, (120, 80))
    assert cam_buffer.viewport != (0.0, 0.0, 1.0, 1.0)
    # At t = 6.0s (end of buffer)
    cam_end = _camera_state_at(schedule, plan, None, 6_000_000, 1.5, (120, 80))
    assert cam_end.viewport != (0.0, 0.0, 1.0, 1.0)


def test_advanced_draw_schedule_respects_completion_buffer(tmp_path: Path) -> None:
    img_path = _make_dummy_image(tmp_path / "001.png")
    scene = SceneImage(
        img_path.name,
        (120, 80),
        (SceneObject("obj1", "art", NormalizedRect(0.2, 0.2, 0.6, 0.6), None, "draw"),),
        ("obj1",),
    )
    plan = DrawImagePlan(
        1,
        img_path.name,
        0,
        8_000_000,
        DrawMode.ADVANCED,
        DrawStyle.V1,
        "manual",
        (),
        complete_before_end_us=2_000_000,
        post_motion="subtle_zoom_in",
    )
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(img_path, renderer.cache_root, plan.style, TextMode.KEEP, False)
    schedule = _build_advanced_schedule(artifact.strokes, plan, scene, (120, 80))
    assert schedule.draw_end_us == 6_000_000

    # Motion progresses from 6.0s to 8.0s
    cam_start = _camera_state_at(schedule, plan, scene, 6_000_000, 1.5, (120, 80))
    cam_end = _camera_state_at(schedule, plan, scene, 8_000_000, 1.5, (120, 80))
    assert cam_start.viewport == (0.0, 0.0, 1.0, 1.0)
    assert cam_end.viewport[2] < 1.0  # Zoomed in (viewport width < 1.0)


def test_production_three_image_unified_draw_workflow(tmp_path: Path) -> None:
    # 3 images:
    # 1. Cue 1: basic_draw (whole image draw)
    # 2. Cue 2: advanced_draw (object-based draw)
    # 3. Cue 3: basic_draw with explicit COMPLETE_BEFORE_END and POST_MOTION
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = _make_dummy_image(img_dir / "001.png", "white")
    img2 = _make_dummy_image(img_dir / "002.png", "lightyellow")
    img3 = _make_dummy_image(img_dir / "003.png", "lightblue")

    scene_path = tmp_path / "draw_scene.json"
    save_scene(
        SceneDocument(
            1,
            {
                "002.png": SceneImage(
                    "002.png",
                    (120, 80),
                    (SceneObject("card", "art", NormalizedRect(0.2, 0.2, 0.6, 0.6), None, "slide_in", "left"),),
                    ("card",),
                )
            },
            scene_path,
        )
    )

    srt_path = tmp_path / "effect.srt"
    srt_path.write_text(
        """1
00:00:00,000 --> 00:00:04,000
MODE basic_draw
COMPLETE_BEFORE_END 1.5s
POST_MOTION subtle_zoom_in

2
00:00:04,000 --> 00:00:09,000
MODE advanced_draw
COMPLETE_BEFORE_END 2.0s
POST_MOTION none

3
00:00:09,000 --> 00:00:13,000
MODE basic_draw
COMPLETE_BEFORE_END 1.5s
POST_MOTION subtle_pan_left
""",
        encoding="utf-8",
    )

    import wave
    dummy_audio = tmp_path / "audio.wav"
    with wave.open(str(dummy_audio), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00\x00" * 44100 * 13)


    draft_folder = tmp_path / "draft"
    config = ProjectConfig(
        project_name="TestUnifiedDraft",
        resolution=Resolution(120, 80),
        draft_folder=draft_folder,
        use_image_timing=True,
        image_timing_srt=srt_path,
        draw_enabled=True,
        draw_effect_srt=srt_path,
        draw_scene_json=scene_path,
        draw_fallback_basic=True,
        draw_reuse_cache=True,
    )
    job = ProjectJob(
        name="TestUnifiedDraft",
        images=(img1, img2, img3),
        audio_path=dummy_audio,
        subtitle_srt=None,
        image_timing_srt=srt_path,
        config=config,
    )


    draft_folder.mkdir(parents=True, exist_ok=True)
    builder = CapCutBuilder()
    result = builder.build_job(job)
    assert result is not None
    assert result.project_path.is_dir()
    draft_json = result.project_path / "draft_content.json"
    assert draft_json.is_file()
    content = json.loads(draft_json.read_text(encoding="utf-8"))
    videos = content.get("materials", {}).get("videos", [])
    assert len(videos) == 3
    for v in videos:
        assert Path(v["path"]).is_file()

