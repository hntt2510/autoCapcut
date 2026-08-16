"""
G07 — Full CapCut draft E2E reliability smoke test.

Covers:
- 3 images: cue1=basic_draw, cue2=advanced_draw, cue3=basic_draw
- 1 Main Effect SRT
- Audio (WAV), Subtitles (SRT), Logo (PNG), BGM folder (WAV)
- Transitions, Post-draw motion, Completion buffer
Verifies draft JSON tracks, draw clip durations, subtitle/logo/BGM placement.
"""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from auto_capcut.core.capcut_builder import CapCutBuilder
from auto_capcut.core.draw_models import NormalizedRect, SceneDocument, SceneImage, SceneObject
from auto_capcut.core.draw_scene import save_scene
from auto_capcut.core.media import probe_duration_us
from auto_capcut.models import ProjectConfig, ProjectJob, RESOLUTIONS


def _make_image(path, color="white", label=""):
    img = Image.new("RGB", (640, 360), color)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 200, 200], outline="black", width=3)
    d.text((30, 30), label, fill="black")
    img.save(path, format="PNG")
    return path


def _make_wav(path, duration_seconds=15.0, sample_rate=22050):
    total = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{total}h", *([0] * total)))
    return path


def _make_subtitle_srt(path):
    path.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nFirst slide narration.\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\nDraw animation in progress.\n\n"
        "3\n00:00:10,000 --> 00:00:15,000\nFinal slide summary.\n\n",
        encoding="utf-8",
    )
    return path


def _make_main_effect_srt(path):
    """3 cues: cue1=basic_draw(0-5s), cue2=advanced_draw(5-10s), cue3=basic_draw(10-15s)."""
    path.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n"
        "MODE=basic_draw\nSTYLE=v1\n"
        "COMPLETE_BEFORE_END 1.5s\nPOST_MOTION subtle_zoom_in\n"
        "DRAW 0s-3.5s:\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\n"
        "MODE=advanced_draw\nSTYLE=v1\n"
        "OBJECT_EFFECT target=obj_1 effect=draw\n"
        "DRAW 0s-5s:\n\n"
        "3\n00:00:10,000 --> 00:00:15,000\n"
        "MODE=basic_draw\nSTYLE=v1\n"
        "DRAW 0s-5s:\n\n",
        encoding="utf-8",
    )
    return path


def _make_scene_json(path, img2):
    objs = (SceneObject(id="obj_1", type="art", box=NormalizedRect(0.1, 0.1, 0.3, 0.3)),)
    images_map = {img2.name: SceneImage(img2.name, (640, 360), objs, ("obj_1",))}
    doc = SceneDocument(schema_version=1, images=images_map, path=path)
    save_scene(doc, path)
    return path


@pytest.mark.slow
def test_g07_full_feature_e2e_smoke(tmp_path):
    """G07: Full E2E smoke — 3 images, main effect SRT, audio, subtitles, logo, BGM, transitions."""

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = _make_image(img_dir / "001.png", "white", "Slide 1")
    img2 = _make_image(img_dir / "002.png", "lightblue", "Draw Slide")
    img3 = _make_image(img_dir / "003.png", "lightyellow", "Slide 3")

    audio = _make_wav(tmp_path / "audio.wav", duration_seconds=15.0)
    subtitle_srt = _make_subtitle_srt(tmp_path / "subtitles.srt")
    main_effect_srt = _make_main_effect_srt(tmp_path / "main_effect.srt")
    scene_json = _make_scene_json(tmp_path / "draw_scene.json", img2)

    logo_path = tmp_path / "logo.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 200)).save(logo_path)

    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    _make_wav(bgm_dir / "bgm1.wav", duration_seconds=16.0)

    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()

    config = ProjectConfig(
        project_name="g07_smoke",
        resolution=RESOLUTIONS["1920x1080"],
        image_folders=[img_dir],
        audio_path=audio,
        import_subtitles=True,
        subtitle_srt=subtitle_srt,
        draft_folder=drafts_dir,
        motion_enabled=True,
        motion_mode="Effect Direction SRT",
        effect_direction_srt=main_effect_srt,
        transition_enabled=True,
        transition_duration_us=300_000,
        logo_enabled=True,
        logo_path=logo_path,
        music_enabled=True,
        music_folder=bgm_dir,
        music_volume=0.12,
        draw_enabled=True,
        draw_scene_json=scene_json,
        draw_fallback_basic=True,
    )
    job = ProjectJob(
        name="g07_smoke",
        images=(img1, img2, img3),
        audio_path=audio,
        subtitle_srt=subtitle_srt,
        image_timing_srt=None,
        config=config,
    )

    builder = CapCutBuilder()
    progress_log = []
    result = builder.build_job(job, lambda v, m: progress_log.append((v, m)))

    assert result.project_path.is_dir()
    assert result.duration_us == 15_000_000
    assert result.project_name.startswith("g07_smoke")

    draft_content_file = result.project_path / "draft_content.json"
    draft_meta_file = result.project_path / "draft_meta_info.json"
    assert draft_content_file.is_file()
    assert draft_meta_file.is_file()

    content = json.loads(draft_content_file.read_text(encoding="utf-8"))
    tracks = content.get("tracks", [])
    track_names = [t.get("name", "") for t in tracks]

    # Images track: 3 draw MP4 segments
    images_track = next((t for t in tracks if t.get("name") == "Images"), None)
    assert images_track is not None, f"Images track missing. Got: {track_names}"
    segments = images_track.get("segments", [])
    assert len(segments) == 3

    videos = content.get("materials", {}).get("videos", [])
    video_map = {v["id"]: v.get("path", "") for v in videos}

    expected_timings = [(0, 5_000_000), (5_000_000, 5_000_000), (10_000_000, 5_000_000)]
    for i, (seg, (expected_start, expected_dur)) in enumerate(zip(segments, expected_timings), 1):
        mat = video_map.get(seg.get("material_id"), "")
        assert mat.endswith(".mp4"), f"Cue{i} must be draw MP4, got: {mat}"
        assert seg["target_timerange"]["start"] == expected_start, f"Cue{i} start mismatch"
        assert seg["target_timerange"]["duration"] == expected_dur, f"Cue{i} duration mismatch"
        assert len(seg.get("keyframe_list", [])) == 0, f"Cue{i} draw clip must not have camera keyframes"
        mp4 = Path(mat)
        assert mp4.is_file(), f"Draw MP4 for cue{i} not found at {mp4}"
        actual_us = probe_duration_us(mp4)
        assert abs(actual_us - expected_dur) <= 70_000, (
            f"Draw MP4 cue{i} duration: expected {expected_dur}us, got {actual_us}us"
        )

    # Audio
    audio_track = next((t for t in tracks if t.get("name") == "Main Audio"), None)
    assert audio_track is not None, f"Main Audio track missing. Got: {track_names}"
    assert audio_track["segments"][0]["target_timerange"]["duration"] == 15_000_000

    # Subtitles
    sub_track = next((t for t in tracks if t.get("name") == "Subtitles"), None)
    assert sub_track is not None, f"Subtitles track missing. Got: {track_names}"
    assert len(sub_track.get("segments", [])) == 3

    # Logo
    logo_track = next((t for t in tracks if t.get("name") == "Logo Overlay"), None)
    assert logo_track is not None, f"Logo Overlay track missing. Got: {track_names}"
    assert len(logo_track.get("segments", [])) >= 1

    # BGM
    bgm_track = next((t for t in tracks if t.get("name") == "Background Music"), None)
    assert bgm_track is not None, f"Background Music track missing. Got: {track_names}"
    assert len(bgm_track.get("segments", [])) >= 1

    # Progress
    assert progress_log[-1][0] == 100
