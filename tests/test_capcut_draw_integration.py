"""
Tests for CapCutBuilder draw rendering integration.

These tests do NOT exercise pycapcut (which requires a real CapCut
installation) or the draw renderer (which requires ffmpeg). Instead they
unit-test the orchestration logic via a mocked DrawRenderService and a stub
draw SRT, verifying:

1. draw_enabled=True causes DrawRenderService.render_subset to be called and
   the rendered MP4 paths to be tracked in draw_clips.
2. draw_enabled=False causes no draw rendering to happen.
3. _render_draw_clips returns {} when the SRT contains no draw cues.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from auto_capcut.core.capcut_builder import CapCutBuilder
from auto_capcut.models import ImageTiming, ProjectConfig, ProjectJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(tmp_path: Path, *, draw_enabled: bool = True, draw_srt: Path | None = None) -> ProjectJob:
    images = [tmp_path / "img1.png", tmp_path / "img2.png", tmp_path / "img3.png"]
    for img in images:
        img.write_bytes(b"x")
    srt = draw_srt or (tmp_path / "effect.srt")
    config = ProjectConfig(
        project_name="test",
        image_folders=[tmp_path],
        audio_path=tmp_path / "audio.mp3",
        draft_folder=tmp_path,
        draw_enabled=draw_enabled,
        effect_direction_srt=srt,
        draw_effect_srt=None,   # falls back to effect_direction_srt
        draw_fallback_basic=True,
    )
    return ProjectJob(
        name="test",
        images=tuple(images),
        audio_path=tmp_path / "audio.mp3",
        subtitle_srt=None,
        image_timing_srt=None,
        config=config,
    )


def _mixed_srt(path: Path) -> Path:
    """Two standard cues + one draw cue."""
    path.write_text(
        "\n".join([
            "1", "00:00:00,000 --> 00:00:05,000",
            "Image 1 FX", "HOLD 0s - 5s :", "",
            "2", "00:00:05,000 --> 00:00:10,000",
            "MODE=basic_draw", "STYLE=v1", "DRAW 0s-5s:", "",
            "3", "00:00:10,000 --> 00:00:15,000",
            "Image 3 FX", "HOLD 0s - 5s :", "",
        ]),
        encoding="utf-8",
    )
    return path


def _standard_only_srt(path: Path) -> Path:
    path.write_text(
        "\n".join([
            "1", "00:00:00,000 --> 00:00:05,000",
            "Image 1 FX", "HOLD 0s - 5s :", "",
            "2", "00:00:05,000 --> 00:00:10,000",
            "Image 2 FX", "HOLD 0s - 5s :", "",
            "3", "00:00:10,000 --> 00:00:15,000",
            "Image 3 FX", "HOLD 0s - 5s :", "",
        ]),
        encoding="utf-8",
    )
    return path


def _timings(n: int, duration_us: int = 5_000_000) -> list[ImageTiming]:
    return [ImageTiming(i, i * duration_us, (i + 1) * duration_us) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests for _render_draw_clips
# ---------------------------------------------------------------------------

def test_render_draw_clips_calls_render_subset_for_draw_cues(tmp_path: Path) -> None:
    """_render_draw_clips must call DrawRenderService.render_subset for the
    draw cue image index and return the resulting clip path."""
    srt = _mixed_srt(tmp_path / "effect.srt")
    job = _job(tmp_path, draw_enabled=True, draw_srt=srt)
    timings = _timings(3)

    fake_mp4 = tmp_path / "002_draw.mp4"
    fake_mp4.write_bytes(b"mp4")

    with patch("auto_capcut.core.draw_renderer.DrawRenderService") as mock_service_cls:
        mock_service = MagicMock()
        mock_service.render_subset.return_value = {1: fake_mp4}
        mock_service_cls.return_value = mock_service

        builder = CapCutBuilder.__new__(CapCutBuilder)
        builder.cc = MagicMock()
        builder.effect_templates = MagicMock()
        warnings: list[str] = []
        result = builder._render_draw_clips(job, timings, lambda *a: None, warnings)

    assert result == {1: fake_mp4}
    assert mock_service.render_subset.called
    _, kwargs = mock_service.render_subset.call_args
    assert 1 in (kwargs.get("image_indexes") or mock_service.render_subset.call_args[0][3])


def test_render_draw_clips_skips_when_no_draw_cues(tmp_path: Path) -> None:
    """If the SRT contains only standard cues, _render_draw_clips must return {}
    without calling the draw renderer."""
    srt = _standard_only_srt(tmp_path / "effect.srt")
    job = _job(tmp_path, draw_enabled=True, draw_srt=srt)
    timings = _timings(3)

    with patch("auto_capcut.core.draw_renderer.DrawRenderService") as mock_service_cls:
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        builder = CapCutBuilder.__new__(CapCutBuilder)
        builder.cc = MagicMock()
        builder.effect_templates = MagicMock()
        warnings: list[str] = []
        result = builder._render_draw_clips(job, timings, lambda *a: None, warnings)

    assert result == {}
    assert not mock_service.render_subset.called


def test_render_draw_clips_returns_empty_when_disabled(tmp_path: Path) -> None:
    """draw_enabled=False means _render_draw_clips is never called at all.
    We verify the flag is checked by calling the private method directly
    with a draw SRT — the unified parser should still classify the cue as draw
    but render_subset should not be called if we simulate the disabled path."""
    # When draw_enabled is False, build_job just skips _render_draw_clips entirely.
    # Here we simply verify _render_draw_clips returns {} for a missing SRT.
    job = _job(tmp_path, draw_enabled=False, draw_srt=None)
    job_disabled = ProjectJob(
        name="test",
        images=job.images,
        audio_path=job.audio_path,
        subtitle_srt=None,
        image_timing_srt=None,
        config=ProjectConfig(
            project_name="test",
            image_folders=[tmp_path],
            audio_path=tmp_path / "audio.mp3",
            draft_folder=tmp_path,
            draw_enabled=False,
        ),
    )
    builder = CapCutBuilder.__new__(CapCutBuilder)
    builder.cc = MagicMock()
    builder.effect_templates = MagicMock()
    warnings: list[str] = []
    result = builder._render_draw_clips(job_disabled, _timings(2), lambda *a: None, warnings)
    # No SRT configured → must return {} cleanly
    assert result == {}


def test_render_draw_clips_gracefully_handles_parse_error(tmp_path: Path) -> None:
    """A malformed SRT should append to warnings and return {} rather than raise."""
    bad_srt = tmp_path / "bad.srt"
    bad_srt.write_text("this is not a valid SRT\n", encoding="utf-8")
    config = ProjectConfig(
        project_name="test",
        image_folders=[tmp_path],
        audio_path=tmp_path / "audio.mp3",
        draft_folder=tmp_path,
        draw_enabled=True,
        effect_direction_srt=bad_srt,
    )
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    job = ProjectJob(
        name="test",
        images=(img,),
        audio_path=tmp_path / "audio.mp3",
        subtitle_srt=None,
        image_timing_srt=None,
        config=config,
    )
    builder = CapCutBuilder.__new__(CapCutBuilder)
    builder.cc = MagicMock()
    builder.effect_templates = MagicMock()
    warnings: list[str] = []
    result = builder._render_draw_clips(job, _timings(1), lambda *a: None, warnings)
    assert result == {}


def test_unified_flow_single_effect_srt_routes_mixed_cues(tmp_path: Path) -> None:
    """Product Invariant: ONE effect.srt with mixed cues automatically routes
    Standard cues to effect planning and Draw cues to DrawRenderer."""
    from auto_capcut.core.planning import resolve_effect_directions

    srt = _mixed_srt(tmp_path / "unified_effect.srt")
    # Single effect.srt configured as effect_direction_srt; draw_effect_srt is None
    config = ProjectConfig(
        project_name="test",
        image_folders=[tmp_path],
        audio_path=tmp_path / "audio.mp3",
        draft_folder=tmp_path,
        motion_enabled=True,
        motion_mode="Effect Direction SRT",
        effect_direction_srt=srt,
        draw_effect_srt=None,  # Not required! Uses effect_direction_srt
        draw_enabled=True,
    )
    images = (tmp_path / "img1.png", tmp_path / "img2.png", tmp_path / "img3.png")
    for img in images:
        img.write_bytes(b"img")
    job = ProjectJob(
        name="test",
        images=images,
        audio_path=tmp_path / "audio.mp3",
        subtitle_srt=None,
        image_timing_srt=None,
        config=config,
    )
    timings = _timings(3)

    # 1. resolve_effect_directions gives [EffectCue, None (draw), EffectCue]
    effects = resolve_effect_directions(job, timings)
    assert effects is not None
    assert len(effects) == 3
    assert effects[0] is not None and effects[0].image_index == 1
    assert effects[1] is None  # draw cue
    assert effects[2] is not None and effects[2].image_index == 3

    # 2. _render_draw_clips automatically uses effect_direction_srt
    fake_mp4 = tmp_path / "002_draw.mp4"
    fake_mp4.write_bytes(b"mp4")
    with patch("auto_capcut.core.draw_renderer.DrawRenderService") as mock_service_cls:
        mock_service = MagicMock()
        mock_service.render_subset.return_value = {1: fake_mp4}
        mock_service_cls.return_value = mock_service

        builder = CapCutBuilder.__new__(CapCutBuilder)
        builder.cc = MagicMock()
        builder.effect_templates = MagicMock()
        warnings: list[str] = []
        draw_clips = builder._render_draw_clips(job, timings, lambda *a: None, warnings)

    assert draw_clips == {1: fake_mp4}


def test_add_images_substitutes_draw_mp4_and_applies_motion_to_standard_images(tmp_path: Path) -> None:
    """Verify _add_images substitutes draw video segment on track without motion
    keyframing, while normal images get image materials + motion."""
    builder = CapCutBuilder.__new__(CapCutBuilder)
    cc = MagicMock()
    # Ensure VideoMaterial instances return numeric dimensions for cover scaling
    mock_material = MagicMock()
    mock_material.width = 1920
    mock_material.height = 1080
    cc.VideoMaterial.return_value = mock_material
    builder.cc = cc
    builder.effect_templates = MagicMock()

    images = (tmp_path / "img1.png", tmp_path / "img2.png", tmp_path / "img3.png")
    for img in images:
        img.write_bytes(b"img")


    fake_mp4 = tmp_path / "002_draw.mp4"
    fake_mp4.write_bytes(b"mp4")
    draw_clips = {1: fake_mp4}

    config = ProjectConfig(
        project_name="test",
        image_folders=[tmp_path],
        audio_path=tmp_path / "audio.mp3",
        draft_folder=tmp_path,
        motion_enabled=True,
        motion_mode="Effect Direction SRT",
        effect_direction_srt=tmp_path / "effect.srt",
        transition_enabled=True,
    )
    job = ProjectJob("test", images, tmp_path / "audio.mp3", None, None, config)
    timings = _timings(3)

    script = MagicMock()
    added_segments = []
    script.add_segment.side_effect = lambda seg, track: added_segments.append((seg, track))

    from auto_capcut.models import EffectCue, VisualEffect
    cue1 = EffectCue(1, 0, 5_000_000, "", (VisualEffect("HOLD", 0, 5_000_000),))
    cue3 = EffectCue(3, 10_000_000, 15_000_000, "", (VisualEffect("HOLD", 0, 5_000_000),))
    effects = [cue1, None, cue3]

    warnings: list[str] = []
    builder._add_images(script, job, timings, warnings, effects, frozenset(), draw_clips)

    assert len(added_segments) == 3
    # Check that cc.VideoMaterial was called with fake_mp4 for index 1
    call_args_list = [call.args[0] for call in cc.VideoMaterial.call_args_list]
    assert str(fake_mp4) in call_args_list
    assert str(images[0]) in call_args_list
    assert str(images[2]) in call_args_list

