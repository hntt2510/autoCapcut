"""
Real end-to-end smoke test for the unified Draw + CapCut Draft workflow.

Renders a real mixed project containing:
- Image 1: standard FX (CapCut image track + motion)
- Image 2: basic_draw (DrawRenderer + FFmpeg -> rendered MP4)
- Image 3: standard FX (CapCut image track + motion)

Inspects the resulting CapCut draft JSON, timeline segments, materials,
durations, and keyframe structures to verify end-to-end correctness.
"""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.capcut_builder import CapCutBuilder
from auto_capcut.core.media import probe_duration_us
from auto_capcut.models import ProjectConfig, ProjectJob, RESOLUTIONS


def _create_dummy_image(path: Path, color: str, text: str) -> Path:
    img = Image.new("RGB", (640, 360), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], outline="black", width=4)
    draw.text((60, 60), text, fill="black")
    img.save(path, format="PNG")
    return path


def _create_dummy_wav(path: Path, duration_seconds: float = 6.0, sample_rate: int = 44100) -> Path:
    total_samples = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # 16-bit silent samples
        data = struct.pack(f"<{total_samples * 2}h", *([0] * (total_samples * 2)))
        wav.writeframes(data)
    return path


def _create_unified_srt(path: Path) -> Path:
    path.write_text(
        "\n".join([
            "1",
            "00:00:00,000 --> 00:00:02,000",
            "Image 1 FX",
            "HOLD 0s - 2s :",
            "",
            "2",
            "00:00:02,000 --> 00:00:04,000",
            "MODE=basic_draw",
            "STYLE=v1",
            "DRAW 0s-2s:",
            "",
            "3",
            "00:00:04,000 --> 00:00:06,000",
            "Image 3 FX",
            "HOLD 0s - 2s :",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def test_real_mixed_draw_capcut_smoke(tmp_path: Path) -> None:
    """Full end-to-end smoke test verifying real DrawRenderer + FFmpeg + CapCut draft creation."""
    # 1. Create assets
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img1 = _create_dummy_image(img_dir / "001.png", "white", "Slide 1")
    img2 = _create_dummy_image(img_dir / "002.png", "lightblue", "Draw Slide")
    img3 = _create_dummy_image(img_dir / "003.png", "lightyellow", "Slide 3")

    audio_file = _create_dummy_wav(tmp_path / "audio.wav", duration_seconds=6.0)
    effect_srt = _create_unified_srt(tmp_path / "unified_effect.srt")
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build configuration
    config = ProjectConfig(
        project_name="smoke_test",
        resolution=RESOLUTIONS["1920x1080"],
        image_folders=[img_dir],
        audio_path=audio_file,
        draft_folder=drafts_dir,
        motion_enabled=True,
        motion_mode="Effect Direction SRT",
        effect_direction_srt=effect_srt,
        draw_enabled=True,
        transition_enabled=True,
        transition_duration_us=300_000,
    )
    job = ProjectJob(
        name="smoke_test",
        images=(img1, img2, img3),
        audio_path=audio_file,
        subtitle_srt=None,
        image_timing_srt=None,
        config=config,
    )

    # 3. Execute real build (renders draw clip with FFmpeg and creates draft)
    builder = CapCutBuilder()
    progress_calls: list[tuple[int, str]] = []
    result = builder.build_job(job, lambda val, msg: progress_calls.append((val, msg)))

    assert result.project_name.startswith("smoke_test")
    assert result.project_path.is_dir()
    assert result.duration_us == 6_000_000

    # 4. Verify draft files on disk
    draft_content_file = result.project_path / "draft_content.json"
    draft_meta_file = result.project_path / "draft_meta_info.json"
    assert draft_content_file.is_file()
    assert draft_meta_file.is_file()

    content = json.loads(draft_content_file.read_text(encoding="utf-8"))

    # 5. Verify materials mapping
    videos = content.get("materials", {}).get("videos", [])
    video_map = {v["id"]: v.get("path", "") for v in videos}

    # 6. Verify timeline tracks & segments
    tracks = content.get("tracks", [])
    images_track = next((t for t in tracks if t.get("name") == "Images" or t.get("type") == "video"), None)
    assert images_track is not None, "Images track not found in draft"

    segments = images_track.get("segments", [])
    assert len(segments) == 3, f"Expected 3 visual segments, got {len(segments)}"

    # Segment 1: Image 001
    seg1 = segments[0]
    mat1_path = video_map.get(seg1.get("material_id"), "")
    assert Path(mat1_path).name == "001.png"
    assert seg1["target_timerange"]["start"] == 0
    assert seg1["target_timerange"]["duration"] == 2_000_000

    # Segment 2: Generated Draw MP4
    seg2 = segments[1]
    mat2_path = video_map.get(seg2.get("material_id"), "")
    assert "002_draw.mp4" in mat2_path or mat2_path.endswith(".mp4")
    assert Path(mat2_path).is_file()
    assert seg2["target_timerange"]["start"] == 2_000_000
    assert seg2["target_timerange"]["duration"] == 2_000_000

    # Draw segment must receive NO duplicate CapCut camera keyframes
    # (keyframe references in segment should be empty or default)
    keyframe_list = seg2.get("keyframe_list", [])
    assert len(keyframe_list) == 0, f"Draw clip should not have keyframes, got {keyframe_list}"

    # Segment 3: Image 003
    seg3 = segments[2]
    mat3_path = video_map.get(seg3.get("material_id"), "")
    assert Path(mat3_path).name == "003.png"
    assert seg3["target_timerange"]["start"] == 4_000_000
    assert seg3["target_timerange"]["duration"] == 2_000_000

    # 7. Verify real rendered draw MP4 on disk
    draw_mp4 = Path(mat2_path)
    assert draw_mp4.is_file()
    assert draw_mp4.stat().st_size > 1000  # Non-trivial MP4 file

    # Probed duration must match 2.0s within tolerance (~70ms)
    probed_us = probe_duration_us(draw_mp4)
    assert abs(probed_us - 2_000_000) <= 70_000, f"Probed draw MP4 duration {probed_us}us differs from 2,000,000us"

    # 8. Verify audio track
    audio_track = next((t for t in tracks if t.get("name") == "Main Audio" or (t.get("type") == "audio" and len(t.get("segments", [])) > 0)), None)
    assert audio_track is not None, f"Main Audio track not found in tracks: {[(t.get('name'), t.get('type')) for t in tracks]}"
    audio_segs = audio_track.get("segments", [])
    assert len(audio_segs) >= 1
    assert audio_segs[0]["target_timerange"]["start"] == 0
    assert audio_segs[0]["target_timerange"]["duration"] == 6_000_000


def test_production_three_image_advanced_draw_workflow(tmp_path: Path) -> None:
    """Requirement 9: Production workflow covering 3 images, 1 unified effect.srt, 1 draw_scene.json
    with 6, 5, and 6 objects respectively rendered through DrawRenderer + CapCutBuilder."""
    from auto_capcut.core.draw_models import NormalizedRect, SceneDocument, SceneImage, SceneObject
    from auto_capcut.core.draw_scene import save_scene

    # 1. Create 3 images
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img1 = _create_dummy_image(img_dir / "001.png", "white", "Slide 1 (6 objs)")
    img2 = _create_dummy_image(img_dir / "002.png", "lightblue", "Slide 2 (5 objs)")
    img3 = _create_dummy_image(img_dir / "003.png", "lightyellow", "Slide 3 (6 objs)")

    audio_file = _create_dummy_wav(tmp_path / "audio.wav", duration_seconds=6.0)

    # 2. Unified effect.srt with 3 advanced_draw cues
    effect_srt = tmp_path / "unified_effect.srt"
    effect_srt.write_text(
        "\n".join([
            "1",
            "00:00:00,000 --> 00:00:02,000",
            "MODE advanced_draw",
            "STYLE v1",
            "OBJECT_EFFECT target=obj_1 effect=draw",
            "DRAW 0s-2s:",
            "",
            "2",
            "00:00:02,000 --> 00:00:04,000",
            "MODE advanced_draw",
            "STYLE v1",
            "OBJECT_EFFECT target=obj_1 effect=draw",
            "DRAW 0s-2s:",
            "",
            "3",
            "00:00:04,000 --> 00:00:06,000",
            "MODE advanced_draw",
            "STYLE v1",
            "OBJECT_EFFECT target=obj_1 effect=draw",
            "DRAW 0s-2s:",
            "",
        ]),
        encoding="utf-8",
    )

    # 3. One canonical draw_scene.json with records for all 3 images:
    # 001.png: 6 objects, 002.png: 5 objects, 003.png: 6 objects
    def _make_objs(count: int) -> tuple[SceneObject, ...]:
        return tuple(
            SceneObject(id=f"obj_{i}", type="art", box=NormalizedRect(0.05 * i, 0.1, 0.04, 0.04))
            for i in range(1, count + 1)
        )

    scene_file = tmp_path / "draw_scene.json"
    images_map = {
        "001.png": SceneImage("001.png", (640, 360), _make_objs(6), tuple(f"obj_{i}" for i in range(1, 7))),
        "002.png": SceneImage("002.png", (640, 360), _make_objs(5), tuple(f"obj_{i}" for i in range(1, 6))),
        "003.png": SceneImage("003.png", (640, 360), _make_objs(6), tuple(f"obj_{i}" for i in range(1, 7))),
    }
    doc = SceneDocument(schema_version=1, images=images_map, path=scene_file)
    save_scene(doc, scene_file)

    # 4. Build CapCut draft
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_name="adv_smoke_test",
        resolution=RESOLUTIONS["1920x1080"],
        image_folders=[img_dir],
        audio_path=audio_file,
        draft_folder=drafts_dir,
        motion_enabled=True,
        motion_mode="Effect Direction SRT",
        effect_direction_srt=effect_srt,
        draw_enabled=True,
        draw_scene_json=scene_file,
        draw_fallback_basic=False,
    )
    job = ProjectJob(
        name="adv_smoke_test",
        images=(img1, img2, img3),
        audio_path=audio_file,
        subtitle_srt=None,
        image_timing_srt=None,
        config=config,
    )

    builder = CapCutBuilder()
    result = builder.build_job(job)

    assert result.project_path.is_dir()
    content = json.loads((result.project_path / "draft_content.json").read_text(encoding="utf-8"))
    tracks = content.get("tracks", [])
    images_track = next((t for t in tracks if t.get("name") == "Images" or t.get("type") == "video"), None)
    assert images_track is not None
    segments = images_track.get("segments", [])
    assert len(segments) == 3

    videos = content.get("materials", {}).get("videos", [])
    video_map = {v["id"]: v.get("path", "") for v in videos}

    for idx, seg in enumerate(segments, 1):
        mat_path = video_map.get(seg.get("material_id"), "")
        assert mat_path.endswith(f"{idx:03d}_draw.mp4")
        assert Path(mat_path).is_file()
        assert seg["target_timerange"]["start"] == (idx - 1) * 2_000_000
        assert seg["target_timerange"]["duration"] == 2_000_000


