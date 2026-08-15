from __future__ import annotations

from pathlib import Path

from auto_capcut.core.errors import ValidationError
from auto_capcut.core.effect_direction_parser import validate_effect_timing
from auto_capcut.core.unified_effect_parser import parse_unified_effect
from auto_capcut.core.roi_resolver import ManualRoiResolver, roi_sidecar_path, validate_saved_frame
from auto_capcut.core.media import collect_audio, collect_images, probe_duration_us, validate_config_paths
from auto_capcut.core.srt_parser import parse_image_timing_srt
from auto_capcut.models import AudioMode, EffectCue, ImageTiming, ProjectConfig, ProjectJob
from auto_capcut.utils.paths import safe_name

TIMING_TOLERANCE_US = 50_000


def calculate_ranges(image_count: int, duration_us: int, timings: list[ImageTiming] | None = None) -> list[ImageTiming]:
    if image_count <= 0:
        raise ValidationError("No images found")
    if timings is not None:
        if len(timings) != image_count:
            raise ValidationError(f"Image Timing mismatch: {image_count} images / {len(timings)} timing cues")
        return timings
    return [
        ImageTiming(index, (index * duration_us) // image_count, ((index + 1) * duration_us) // image_count)
        for index in range(image_count)
    ]


def create_jobs(config: ProjectConfig) -> list[ProjectJob]:
    validate_config_paths(config)
    audio_mode = AudioMode(config.audio_mode)
    images = tuple(collect_images(config.image_folders))
    if not images:
        raise ValidationError("No images found")
    timing_path = config.image_timing_srt if config.use_image_timing else None
    if audio_mode is AudioMode.SINGLE:
        assert config.audio_path is not None
        return [ProjectJob(safe_name(config.project_name), images, config.audio_path.resolve(), config.subtitle_srt if config.import_subtitles else None, timing_path, config)]
    assert config.audio_path is not None
    jobs: list[ProjectJob] = []
    base = safe_name(config.project_name)
    for audio in collect_audio(config.audio_path):
        matched_srt = audio.with_suffix(".srt") if config.import_subtitles and audio.with_suffix(".srt").is_file() else None
        jobs.append(ProjectJob(safe_name(f"{base}_{audio.stem}"), images, audio, matched_srt, timing_path, config))
    return jobs


def resolve_timings(job: ProjectJob) -> tuple[list[ImageTiming], int]:
    audio_duration = probe_duration_us(job.audio_path)
    if job.image_timing_srt:
        timings = parse_image_timing_srt(job.image_timing_srt)
        ranges = calculate_ranges(len(job.images), timings[-1].end_us, timings)
        if abs(ranges[-1].end_us - audio_duration) > TIMING_TOLERANCE_US:
            raise ValidationError(f"Image timing/audio mismatch: timing ends at {ranges[-1].end_us / 1_000_000:.3f}s, audio is {audio_duration / 1_000_000:.3f}s")
        return ranges, ranges[-1].end_us
    return calculate_ranges(len(job.images), audio_duration), audio_duration


def resolve_effect_directions(job: ProjectJob, timings: list[ImageTiming]) -> list[EffectCue | None] | None:
    mode = str(getattr(job.config, "motion_mode", "")).casefold()
    if not job.config.motion_enabled or mode != "effect direction srt":
        return None
    effect_path = job.config.effect_direction_srt
    if effect_path is None:
        raise ValidationError("Effect Direction SRT does not exist")
    unified = parse_unified_effect(effect_path)
    if len(unified.cues) != len(timings):
        raise ValidationError(f"Effect Direction mismatch: {len(timings)} images / {len(unified.cues)} effect cues")
    effects: list[EffectCue | None] = []
    for cue, timing in zip(unified.cues, timings):
        if cue.kind == "standard" and cue.effect_cue is not None:
            effect = cue.effect_cue
            directive_end = [phase.local_end_us for phase in effect.effects]
            max_directive_end = max(directive_end, default=0)
            if max_directive_end > timing.duration_us + 50_000:
                raise ValidationError(f"Effect SRT error: Image {effect.image_index} phase exceeds image duration")
            effects.append(effect)
        else:
            effects.append(None)
    if job.image_timing_srt:
        std_effects = [e for e in effects if e is not None]
        if std_effects:
            std_timings = [t for e, t in zip(effects, timings) if e is not None]
            validate_effect_timing(std_effects, std_timings)
    return effects


def validate_required_rois(job: ProjectJob, effects: list[EffectCue | None] | None) -> None:
    if not effects or job.config.effect_direction_srt is None:
        return
    resolver = ManualRoiResolver(roi_sidecar_path(job.config.effect_direction_srt))
    canvas_size = (job.config.resolution.width, job.config.resolution.height)
    missing: dict[int, list[str]] = {}
    for cue in effects:
        if cue is None:
            continue
        for target in cue.required_roi_targets:
            image = job.images[target.image_index - 1]
            frame = resolver.resolve(image, target.target_id, target.image_index)
            if frame is None:
                missing.setdefault(target.image_index, []).append(f"{target.target_id} (missing camera frame)")
                continue
            valid, reason = validate_saved_frame(frame, image, canvas_size)
            if not valid:
                label = "needs reframing" if reason == "frame aspect does not match project canvas" else reason
                missing.setdefault(target.image_index, []).append(f"{target.target_id} ({label})")
    if missing:
        lines = ["Missing ROI definitions:"]
        for image_index, targets in missing.items():
            lines.append(f"Image {image_index:03d}:")
            lines.extend(f"- {target}" for target in targets)
        raise ValidationError("\n".join(lines))

