from __future__ import annotations

import random
import re
import shutil
import tempfile
from pathlib import Path

from auto_capcut.core.capcut_compat import patch_capcut_metadata
from auto_capcut.core.errors import ValidationError
from auto_capcut.core.media import AUDIO_EXTENSIONS, probe_duration_us
from auto_capcut.core.motion_engine import MotionEngine
from auto_capcut.core.alert_overlay import create_alert_overlay
from auto_capcut.core.captured_effect_template import (
    CapturedEffectTemplateCloner,
    CapturedEffectTemplateRepository,
    ResolvedCapturedEffectPreset,
)
from auto_capcut.core.planning import resolve_effect_directions, resolve_timings, validate_required_rois
from auto_capcut.core.roi_resolver import ManualRoiResolver, roi_sidecar_path
from auto_capcut.core.srt_parser import parse_srt
from auto_capcut.core.validation import validate_draft_json
from auto_capcut.models import BuildResult, ImageTiming, ProjectJob, ProgressCallback
from auto_capcut.utils.paths import unique_project_name


class CapCutBuilder:
    def __init__(self) -> None:
        try:
            import pycapcut as cc
        except ImportError as exc:  # pragma: no cover - exercised in packaging/runtime
            raise ValidationError("pycapcut is not installed. Run pip install -e .") from exc
        self.cc = cc
        self.effect_templates = CapturedEffectTemplateRepository()

    def build_job(self, job: ProjectJob, progress_callback: ProgressCallback | None = None) -> BuildResult:
        cc = self.cc
        draft_root = job.config.draft_folder
        if draft_root is None or not draft_root.is_dir():
            raise ValidationError("CapCut draft folder not found")

        def progress(value: int, message: str) -> None:
            if progress_callback:
                progress_callback(value, message)

        progress(15, "Reading audio...")
        timings, duration_us = resolve_timings(job)
        effects = resolve_effect_directions(job, timings)
        warnings: list[str] = []
        captured_effects, captured_keys = self._resolve_captured_effects(effects, timings, warnings)
        validate_required_rois(job, effects)
        progress(25, "Creating CapCut Draft...")
        final_name = unique_project_name(draft_root, job.name)
        staging_parent = Path(tempfile.mkdtemp(prefix="autocapcut-", dir=str(draft_root)))
        staging_name = final_name
        try:
            draft_folder = cc.DraftFolder(str(staging_parent))
            script = draft_folder.create_draft(staging_name, job.config.resolution.width, job.config.resolution.height, fps=30)
            fallback_alerts = bool(effects and any(
                effect.type == "ALERT" and (cue_index, effect_index) not in captured_keys
                for cue_index, cue in enumerate(effects)
                for effect_index, effect in enumerate(cue.effects)
            ))
            self._add_tracks(script, fallback_alerts)
            progress(35, "Adding images...")
            self._add_images(script, job, timings, warnings, effects, captured_keys)
            progress(58, "Applying motion and transitions...")
            # Motion/transitions are attached while image segments are created.
            progress(67, "Adding main audio...")
            self._add_audio(script, job.audio_path, duration_us, "Main Audio", 1.0)
            if job.subtitle_srt:
                progress(74, "Adding subtitles...")
                self._add_subtitles(script, job.subtitle_srt)
            if job.config.music_enabled and job.config.music_folder:
                progress(82, "Adding music...")
                self._add_music(script, job.config.music_folder, duration_us, job.config.music_volume)
            if job.config.logo_enabled and job.config.logo_path:
                progress(88, "Adding logo...")
                self._add_logo(script, job.config.logo_path, duration_us, job.config.resolution.width, job.config.resolution.height)
            progress(93, "Saving project...")
            script.save()
            staging_project = staging_parent / staging_name
            for preset, effect_start_us, effect_duration_us in captured_effects:
                CapturedEffectTemplateCloner.inject_file(
                    staging_project,
                    preset,
                    start_us=effect_start_us,
                    duration_us=effect_duration_us,
                    track_name="Captured Effects",
                )
            validate_draft_json(staging_project, duration_us)
            destination = draft_root / final_name
            staging_project.replace(destination)
            try:
                patch_capcut_metadata(destination, final_name, duration_us)
                validate_draft_json(destination, duration_us)
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                raise
            progress(100, "Project created successfully.")
            return BuildResult(final_name, destination, duration_us, tuple(warnings))
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"Unable to create project: {exc}") from exc
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    def _add_tracks(self, script, warning_overlays: bool = False) -> None:
        cc = self.cc
        script.add_track(cc.TrackType.video, "Images", absolute_index=0)
        script.add_track(cc.TrackType.audio, "Main Audio", absolute_index=10)
        script.add_track(cc.TrackType.text, "Subtitles", absolute_index=20)
        script.add_track(cc.TrackType.audio, "Background Music", absolute_index=5)
        script.add_track(cc.TrackType.video, "Logo Overlay", absolute_index=30)
        if warning_overlays:
            script.add_track(cc.TrackType.video, "Warning Overlays", absolute_index=25)

    def _add_images(self, script, job: ProjectJob, timings: list[ImageTiming], warnings: list[str], effects=None, captured_keys=frozenset()) -> None:
        cc = self.cc
        manual_resolver = ManualRoiResolver(roi_sidecar_path(job.config.effect_direction_srt)) if effects and job.config.effect_direction_srt else None
        roi_resolver = manual_resolver
        engine = MotionEngine(job.config.resolution.width, job.config.resolution.height, job.config.motion_strength, roi_resolver)
        for index, (image_path, timing) in enumerate(zip(job.images, timings)):
            material = cc.VideoMaterial(str(image_path))
            base_scale = engine.capcut_cover_scale(job.config.resolution.width, job.config.resolution.height, material.width, material.height)
            segment = cc.VideoSegment(
                material,
                cc.Timerange(timing.start_us, timing.duration_us),
                clip_settings=cc.ClipSettings(scale_x=base_scale, scale_y=base_scale),
            )
            if effects:
                plan = engine.plan_effect(effects[index], image_path, material.width, material.height, timing.duration_us)
                warnings.extend(plan.warnings)
            elif job.config.motion_enabled:
                plan = engine.plan_generic(job.config.motion_mode, image_path, material.width, material.height, timing.duration_us, f"{job.name}:{index}:{image_path}")
            else:
                plan = None
            if plan:
                self._apply_motion_plan(segment, plan, base_scale)
            if effects:
                self._add_alert_overlays(script, effects[index], image_path, timing, job, base_scale, plan, manual_resolver, index, captured_keys)
            if job.config.transition_enabled and index < len(job.images) - 1:
                soft_cut = effects and effects[index].transition_out and effects[index].transition_out.casefold() == "soft cut"
                transition_duration = min(
                    job.config.transition_duration_us,
                    timing.duration_us // 2,
                    timings[index + 1].duration_us // 2,
                )
                if transition_duration > 0 and not soft_cut:
                    transition_type = self._blur_transition_type()
                    if transition_type is not None:
                        segment.add_transition(transition_type, duration=transition_duration)
                        if transition_duration < job.config.transition_duration_us:
                            warnings.append(f"Transition {index + 1} shortened to {transition_duration / 1_000_000:.3f}s")
            script.add_segment(segment, "Images")
        if manual_resolver:
            warnings.extend(dict.fromkeys(manual_resolver.warnings))


    def _apply_motion_plan(self, segment, plan, base_scale: float) -> None:
        cc = self.cc
        if len(plan.keyframes) <= 1:
            return
        has_scale = len({key.transform.relative_scale for key in plan.keyframes}) > 1
        has_x = len({key.transform.position_x for key in plan.keyframes}) > 1
        has_y = len({key.transform.position_y for key in plan.keyframes}) > 1
        for keyframe in plan.keyframes:
            offset = keyframe.local_time_us
            transform = keyframe.transform
            if has_scale:
                segment.add_keyframe(cc.KeyframeProperty.uniform_scale, offset, base_scale * transform.relative_scale)
            if has_x:
                segment.add_keyframe(cc.KeyframeProperty.position_x, offset, transform.position_x)
            if has_y:
                segment.add_keyframe(cc.KeyframeProperty.position_y, offset, transform.position_y)

    def _add_alert_overlays(self, script, cue, image_path: Path, timing: ImageTiming, job: ProjectJob, base_scale: float, plan, resolver, cue_index: int = 0, captured_keys=frozenset()) -> None:
        if resolver is None:
            return
        cc = self.cc
        for effect_index, effect in enumerate(cue.effects):
            if effect.type != "ALERT" or not effect.target_id:
                continue
            if effect.params.get("preset", "").strip():
                continue
            if (cue_index, effect_index) in captured_keys:
                continue
            roi = resolver.resolve(image_path, effect.target_id, cue.image_index)
            if roi is None:
                continue
            overlay_dir = job.config.effect_direction_srt.with_suffix(".overlays") if job.config.effect_direction_srt else image_path.parent / ".overlays"
            overlay_path = create_alert_overlay(image_path, roi, effect, overlay_dir)
            material = cc.VideoMaterial(str(overlay_path))
            start = timing.start_us + effect.local_start_us
            duration = effect.duration_us
            segment = cc.VideoSegment(material, cc.Timerange(start, duration), clip_settings=cc.ClipSettings(scale_x=base_scale, scale_y=base_scale))
            if plan:
                local_plan = type(plan)(tuple(key for key in plan.keyframes if effect.local_start_us <= key.local_time_us <= effect.local_end_us), plan.warnings)
                if local_plan.keyframes:
                    shifted = type(local_plan)(tuple(type(key)(key.local_time_us - effect.local_start_us, key.transform) for key in local_plan.keyframes), local_plan.warnings)
                    self._apply_motion_plan(segment, shifted, base_scale)
            if effect.params.get("pulse", "0").casefold() in {"1", "true", "yes"}:
                segment.add_keyframe(cc.KeyframeProperty.alpha, 0, 0.78)
                segment.add_keyframe(cc.KeyframeProperty.alpha, duration // 2, 1.0)
                segment.add_keyframe(cc.KeyframeProperty.alpha, duration, 0.78)
            script.add_segment(segment, "Warning Overlays")

    @staticmethod
    def _effect_time(value: str, *, image_index: int, field: str) -> int:
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*", value, re.IGNORECASE)
        if not match:
            raise ValidationError(f"Effect SRT error: Image {image_index} {field} must be a time in seconds")
        return round(float(match.group(1)) * 1_000_000)

    def _resolve_captured_effects(self, effects, timings: list[ImageTiming], warnings: list[str]):
        resolved: list[tuple[ResolvedCapturedEffectPreset, int, int]] = []
        keys: set[tuple[int, int]] = set()
        unresolved: list[str] = []
        if not effects:
            return resolved, keys
        for cue_index, (cue, timing) in enumerate(zip(effects, timings)):
            for effect_index, effect in enumerate(cue.effects):
                preset_name = effect.params.get("preset", "").strip()
                if not preset_name:
                    continue
                if effect.type != "ALERT":
                    unresolved.append(preset_name)
                    continue
                preset = self.effect_templates.resolve_effect_preset(preset_name)
                if preset is None:
                    unresolved.append(preset_name)
                    continue
                local_start = effect.local_start_us
                local_end = effect.local_end_us
                if "effect_start" in effect.params:
                    local_start = self._effect_time(effect.params["effect_start"], image_index=cue.image_index, field="effect_start")
                if "effect_end" in effect.params:
                    local_end = self._effect_time(effect.params["effect_end"], image_index=cue.image_index, field="effect_end")
                if local_start < effect.local_start_us or local_end > effect.local_end_us or local_end <= local_start:
                    raise ValidationError(f"Effect SRT error: Image {cue.image_index} captured effect timing must be inside ALERT range")
                resolved.append((preset, timing.start_us + local_start, local_end - local_start))
                keys.add((cue_index, effect_index))
        if unresolved:
            unique = list(dict.fromkeys(unresolved))
            lines = ["Unresolved CapCut effect presets:"] + [f"- {name}" for name in unique]
            raise ValidationError("\n".join(lines))
        return resolved, keys

    def _blur_transition_type(self):
        transition_type = getattr(self.cc.TransitionType, "转场_模糊", None)
        if transition_type is not None:
            return transition_type
        for name in dir(self.cc.TransitionType):
            if "模糊" in name:
                return getattr(self.cc.TransitionType, name)
        return None

    def _add_audio(self, script, path: Path, duration_us: int, track_name: str, volume: float) -> None:
        cc = self.cc
        source_duration = probe_duration_us(path)
        target_duration = min(duration_us, source_duration)
        segment = cc.AudioSegment(str(path), cc.Timerange(0, target_duration), volume=volume)
        script.add_segment(segment, track_name)

    def _add_subtitles(self, script, path: Path) -> None:
        cc = self.cc
        style = cc.TextStyle(size=8.0, bold=True, color=(1.0, 1.0, 1.0), align=1, auto_wrapping=True, max_line_width=0.82)
        border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=45.0)
        for cue in parse_srt(path):
            segment = cc.TextSegment(
                cue.text,
                cc.Timerange(cue.start_us, cue.duration_us),
                style=style,
                border=border,
                clip_settings=cc.ClipSettings(transform_y=-0.76),
            )
            script.add_segment(segment, "Subtitles")

    def _add_music(self, script, folder: Path, duration_us: int, volume: float) -> None:
        tracks = [path for path in folder.iterdir() if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS]
        if not tracks:
            raise ValidationError("No supported background music files found")
        random.SystemRandom().shuffle(tracks)
        current = 0
        index = 0
        while current < duration_us:
            path = tracks[index % len(tracks)]
            available = probe_duration_us(path)
            segment_duration = min(available, duration_us - current)
            self._add_audio_at(script, path, current, segment_duration, "Background Music", volume)
            current += segment_duration
            index += 1

    def _add_audio_at(self, script, path: Path, start_us: int, duration_us: int, track_name: str, volume: float) -> None:
        cc = self.cc
        script.add_segment(cc.AudioSegment(str(path), cc.Timerange(start_us, duration_us), volume=volume), track_name)

    def _add_logo(self, script, path: Path, duration_us: int, canvas_width: int, canvas_height: int) -> None:
        cc = self.cc
        material = cc.VideoMaterial(str(path))
        scale = min((canvas_width * 0.12) / material.width, (canvas_height * 0.12) / material.height)
        settings = cc.ClipSettings(scale_x=scale, scale_y=scale, transform_x=0.80, transform_y=0.80)
        script.add_segment(cc.VideoSegment(material, cc.Timerange(0, duration_us), clip_settings=settings), "Logo Overlay")
