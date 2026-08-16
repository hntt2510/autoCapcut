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
        motion_enabled=True,
        effect_direction_srt=srt_path,
        draw_enabled=True,
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


def test_invalid_advanced_scene_fallback_off_blocks_build(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = _make_dummy_image(img_dir / "001.png", "white")

    srt_path = tmp_path / "effect.srt"
    srt_path.write_text(
        """1
00:00:00,000 --> 00:00:04,000
MODE advanced_draw
DRAW 0s-4s: order=missing_target
""",
        encoding="utf-8",
    )

    import wave
    dummy_audio = tmp_path / "audio.wav"
    with wave.open(str(dummy_audio), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00\x00" * 44100 * 4)

    draft_folder = tmp_path / "draft"
    draft_folder.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_name="TestBlockDraft",
        resolution=Resolution(120, 80),
        draft_folder=draft_folder,
        use_image_timing=True,
        image_timing_srt=srt_path,
        motion_enabled=True,
        effect_direction_srt=srt_path,
        draw_enabled=True,
        draw_scene_json=None,  # No scene JSON provided!
        draw_fallback_basic=False,  # Fallback disabled!
        draw_reuse_cache=False,
    )
    job = ProjectJob(
        name="TestBlockDraft",
        images=(img1,),
        audio_path=dummy_audio,
        subtitle_srt=None,
        image_timing_srt=srt_path,
        config=config,
    )

    builder = CapCutBuilder()
    from auto_capcut.core.errors import DrawRenderError, ValidationError
    with pytest.raises((ValidationError, DrawRenderError)):
        builder.build_job(job)


def test_invalid_advanced_scene_fallback_on_downgrades_to_basic(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = _make_dummy_image(img_dir / "001.png", "white")

    srt_path = tmp_path / "effect.srt"
    srt_path.write_text(
        """1
00:00:00,000 --> 00:00:04,000
MODE advanced_draw
DRAW 0s-4s: order=missing_target
""",
        encoding="utf-8",
    )

    import wave
    dummy_audio = tmp_path / "audio.wav"
    with wave.open(str(dummy_audio), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00\x00" * 44100 * 4)

    draft_folder = tmp_path / "draft"
    draft_folder.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_name="TestFallbackDraft",
        resolution=Resolution(120, 80),
        draft_folder=draft_folder,
        use_image_timing=True,
        image_timing_srt=srt_path,
        motion_enabled=True,
        effect_direction_srt=srt_path,
        draw_enabled=True,
        draw_scene_json=None,  # No scene JSON provided!
        draw_fallback_basic=True,  # Fallback enabled!
        draw_reuse_cache=False,
    )
    job = ProjectJob(
        name="TestFallbackDraft",
        images=(img1,),
        audio_path=dummy_audio,
        subtitle_srt=None,
        image_timing_srt=srt_path,
        config=config,
    )

    builder = CapCutBuilder()
    result = builder.build_job(job)
    assert result is not None
    assert result.project_path.is_dir()


def test_production_uses_effect_direction_srt_and_ignores_conflicting_draw_effect_srt(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = _make_dummy_image(img_dir / "001.png", "white")

    # Authoritative production effect SRT (valid basic_draw cue)
    main_effect_srt = tmp_path / "main_effect.srt"
    main_effect_srt.write_text(
        """1
00:00:00,000 --> 00:00:03,000
MODE basic_draw
COMPLETE_BEFORE_END 1.0s
POST_MOTION none
""",
        encoding="utf-8",
    )

    # Conflicting legacy draw SRT (broken syntax / impossible duration that would fail if parsed/used)
    conflicting_draw_srt = tmp_path / "conflicting_legacy_draw.srt"
    conflicting_draw_srt.write_text(
        """1
00:00:00,000 --> 00:00:99,000
INVALID_DIRECTIVE_THAT_WOULD_FAIL
""",
        encoding="utf-8",
    )

    import wave
    dummy_audio = tmp_path / "audio.wav"
    with wave.open(str(dummy_audio), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00\x00" * 44100 * 3)

    draft_folder = tmp_path / "draft"
    draft_folder.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_name="TestConflictDraft",
        resolution=Resolution(120, 80),
        draft_folder=draft_folder,
        use_image_timing=True,
        image_timing_srt=main_effect_srt,
        motion_enabled=True,
        effect_direction_srt=main_effect_srt,       # Main authoritative source
        draw_effect_srt=conflicting_draw_srt,        # Legacy field populated with conflicting/broken content
        draw_enabled=True,
        draw_reuse_cache=False,
    )
    job = ProjectJob(
        name="TestConflictDraft",
        images=(img1,),
        audio_path=dummy_audio,
        subtitle_srt=None,
        image_timing_srt=main_effect_srt,
        config=config,
    )

    builder = CapCutBuilder()
    # Must succeed by using main_effect_srt and ignoring conflicting_draw_srt
    result = builder.build_job(job)
    assert result is not None
    assert result.project_path.is_dir()
    draft_json = result.project_path / "draft_content.json"
    assert draft_json.is_file()
    content = json.loads(draft_json.read_text(encoding="utf-8"))
    videos = content.get("materials", {}).get("videos", [])
    assert len(videos) == 1
    assert Path(videos[0]["path"]).is_file()



@pytest.mark.parametrize("preset", [
    "none",
    "random_light",
    "subtle_zoom_in",
    "subtle_zoom_out",
    "subtle_pan_left",
    "subtle_pan_right",
])
def test_all_six_post_motion_presets_evaluated_in_buffer(tmp_path: Path, preset: str) -> None:
    img_path = _make_dummy_image(tmp_path / f"001_{preset}.png")
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
        post_motion=preset,
    )
    renderer = DrawRenderer(tmp_path / "cache")
    artifact = prepare_image(img_path, renderer.cache_root, plan.style, TextMode.KEEP, False)
    schedule = _basic_schedule(artifact.strokes, plan)

    # During draw (t=2s): always full view
    cam_draw = _camera_state_at(schedule, plan, None, 2_000_000, 1.5, (120, 80))
    assert cam_draw.viewport == (0.0, 0.0, 1.0, 1.0)

    # At completion buffer end (t=6s):
    cam_end = _camera_state_at(schedule, plan, None, 6_000_000, 1.5, (120, 80))
    if preset == "none":
        assert cam_end.viewport == (0.0, 0.0, 1.0, 1.0)
    elif preset == "subtle_zoom_in":
        assert cam_end.viewport[2] < 1.0  # Zoomed in
    elif preset == "subtle_zoom_out":
        assert cam_end.viewport[2] <= 1.0
    elif preset in {"subtle_pan_left", "subtle_pan_right"}:
        assert cam_end.viewport[0] != 0.0 or cam_end.viewport[1] != 0.0 or cam_end.viewport[2] != 1.0


def test_resolve_timings_asymmetric_main_effect_srt(tmp_path: Path) -> None:
    from auto_capcut.core.planning import resolve_timings
    import wave

    img1 = _make_dummy_image(tmp_path / "001.png")
    img2 = _make_dummy_image(tmp_path / "002.png")
    img3 = _make_dummy_image(tmp_path / "003.png")

    srt_path = tmp_path / "main_effect.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nMODE basic_draw\n\n"
        "2\n00:00:03,000 --> 00:00:11,000\nMODE basic_draw\n\n"
        "3\n00:00:11,000 --> 00:00:15,000\nMODE basic_draw\n\n",
        encoding="utf-8",
    )

    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 15)

    config = ProjectConfig(
        project_name="TestAsym",
        image_folders=[tmp_path],
        audio_path=audio_path,
        effect_direction_srt=srt_path,
    )
    job = ProjectJob(
        name="TestAsym",
        images=(img1, img2, img3),
        audio_path=audio_path,
        subtitle_srt=None,
        image_timing_srt=None,
        config=config,
    )

    timings, total_us = resolve_timings(job)
    assert total_us == 15_000_000
    assert len(timings) == 3
    assert timings[0].start_us == 0 and timings[0].end_us == 3_000_000 and timings[0].duration_us == 3_000_000
    assert timings[1].start_us == 3_000_000 and timings[1].end_us == 11_000_000 and timings[1].duration_us == 8_000_000
    assert timings[2].start_us == 11_000_000 and timings[2].end_us == 15_000_000 and timings[2].duration_us == 4_000_000


def test_resolve_timings_validations(tmp_path: Path) -> None:
    from auto_capcut.core.planning import resolve_timings
    from auto_capcut.core.errors import ValidationError
    import wave

    img1 = _make_dummy_image(tmp_path / "001.png")
    img2 = _make_dummy_image(tmp_path / "002.png")

    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 10)

    # 1. Cue count mismatch (3 cues for 2 images)
    srt_mismatch = tmp_path / "mismatch.srt"
    srt_mismatch.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nMODE basic_draw\n\n"
        "2\n00:00:03,000 --> 00:00:06,000\nMODE basic_draw\n\n"
        "3\n00:00:06,000 --> 00:00:10,000\nMODE basic_draw\n\n",
        encoding="utf-8",
    )
    job_mismatch = ProjectJob("test", (img1, img2), audio_path, None, None, ProjectConfig(effect_direction_srt=srt_mismatch))
    with pytest.raises(ValidationError, match="Main Effect SRT mismatch"):
        resolve_timings(job_mismatch)

    # 2. Non-zero start (starts at 1s)
    srt_nonzero = tmp_path / "nonzero.srt"
    srt_nonzero.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\nMODE basic_draw\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\nMODE basic_draw\n\n",
        encoding="utf-8",
    )
    job_nonzero = ProjectJob("test", (img1, img2), audio_path, None, None, ProjectConfig(effect_direction_srt=srt_nonzero))
    with pytest.raises(ValidationError, match="First cue must start at 00:00:00,000"):
        resolve_timings(job_nonzero)

    # 3. Gap between cues (0-4s and 5-10s)
    srt_gap = tmp_path / "gap.srt"
    srt_gap.write_text(
        "1\n00:00:00,000 --> 00:00:04,000\nMODE basic_draw\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\nMODE basic_draw\n\n",
        encoding="utf-8",
    )
    job_gap = ProjectJob("test", (img1, img2), audio_path, None, None, ProjectConfig(effect_direction_srt=srt_gap))
    with pytest.raises(ValidationError, match="Gap or overlap between cue 1"):
        resolve_timings(job_gap)

    # 4. Total duration mismatch vs audio (SRT ends at 8s, audio is 10s)
    srt_dur = tmp_path / "dur.srt"
    srt_dur.write_text(
        "1\n00:00:00,000 --> 00:00:04,000\nMODE basic_draw\n\n"
        "2\n00:00:04,000 --> 00:00:08,000\nMODE basic_draw\n\n",
        encoding="utf-8",
    )
    job_dur = ProjectJob("test", (img1, img2), audio_path, None, None, ProjectConfig(effect_direction_srt=srt_dur))
    with pytest.raises(ValidationError, match="Main Effect SRT / audio duration mismatch"):
        resolve_timings(job_dur)


def test_missing_main_effect_srt_blocks_production(tmp_path: Path) -> None:
    """Requirement A & 3: 3 images + audio + Main Effect SRT=None MUST raise ValidationError."""
    from auto_capcut.core.errors import ValidationError
    from auto_capcut.core.media import validate_config_paths
    from auto_capcut.core.planning import resolve_timings
    import wave

    img1 = _make_dummy_image(tmp_path / "001.png")
    img2 = _make_dummy_image(tmp_path / "002.png")
    img3 = _make_dummy_image(tmp_path / "003.png")

    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 15)

    config = ProjectConfig(
        project_name="no_srt_test",
        image_folders=[tmp_path],
        audio_path=audio_path,
        draft_folder=tmp_path,
        effect_direction_srt=None,
    )

    with pytest.raises(ValidationError, match="Choose a Main Effect SRT."):
        validate_config_paths(config)

    job = ProjectJob("no_srt_job", (img1, img2, img3), audio_path, None, None, config)
    with pytest.raises(ValidationError, match="Choose a Main Effect SRT."):
        resolve_timings(job)


def test_legacy_standard_cue_blocks_production_in_planning(tmp_path: Path) -> None:
    """Requirement B: Explicit legacy standard FX cues MUST block production planning."""
    from auto_capcut.core.errors import ValidationError
    from auto_capcut.core.planning import resolve_timings
    import wave

    img1 = _make_dummy_image(tmp_path / "001.png")
    img2 = _make_dummy_image(tmp_path / "002.png")

    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 4)

    legacy_srt = tmp_path / "legacy.srt"
    legacy_srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nMODE basic_draw\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nImage 2 FX\nHOLD 0s - 2s :\n",
        encoding="utf-8",
    )
    config = ProjectConfig(
        project_name="legacy_cue_test",
        image_folders=[tmp_path],
        audio_path=audio_path,
        effect_direction_srt=legacy_srt,
    )
    job = ProjectJob("legacy_cue_job", (img1, img2), audio_path, None, None, config)
    with pytest.raises(ValidationError, match="Image 002 uses a legacy standard FX cue"):
        resolve_timings(job)


def test_minimal_cue_no_mode_defaults_to_basic_draw(tmp_path: Path) -> None:
    """Requirement C: Minimal cue without explicit MODE line defaults to basic_draw."""
    from auto_capcut.core.unified_effect_parser import parse_unified_effect
    srt_path = tmp_path / "minimal.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nDRAW 0s-3s:\n\n",
        encoding="utf-8",
    )
    unified = parse_unified_effect(srt_path)
    assert len(unified.cues) == 1
    cue = unified.cues[0]
    assert cue.kind == "draw"
    assert cue.draw_plan is not None
    assert cue.draw_plan.mode == DrawMode.BASIC


def test_production_ui_config_always_has_draw_enabled_true() -> None:
    """Requirement D: ProjectConfig created by production UI always sets draw_enabled=True."""
    from PyQt6.QtWidgets import QApplication
    from auto_capcut.ui.main_window import MainWindow
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    cfg = win._config()
    assert cfg.draw_enabled is True


