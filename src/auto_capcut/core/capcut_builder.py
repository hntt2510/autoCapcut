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
from auto_capcut.core.planning import resolve_effect_directions, resolve_timings, validate_required_rois
from auto_capcut.core.roi_resolver import ManualRoiResolver, roi_sidecar_path
from auto_capcut.core.srt_parser import parse_srt
from auto_capcut.core.validation import validate_draft_json
from auto_capcut.models import BuildResult, ImageTiming, ProjectConfig, ProjectJob, ProgressCallback
from auto_capcut.utils.paths import unique_project_name


class CapCutBuilder:
    def __init__(self) -> None:
        try:
            import pycapcut as cc
        except ImportError as exc:  # pragma: no cover - exercised in packaging/runtime
            raise ValidationError("pycapcut is not installed. Run pip install -e .") from exc
        self.cc = cc

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

        # ── Draw rendering integration ─────────────────────────────────────
        draw_clips: dict[int, Path] = {}   # {0-based image index -> rendered MP4}
        if job.config.draw_enabled:
            draw_clips = self._render_draw_clips(job, timings, progress, warnings)
        # ──────────────────────────────────────────────────────────────────

        validate_required_rois(job, effects)
        progress(25, "Creating CapCut Draft...")
        final_name = unique_project_name(draft_root, job.name)
        staging_parent = Path(tempfile.mkdtemp(prefix="autocapcut-", dir=str(draft_root)))
        staging_name = final_name
        try:
            draft_folder = cc.DraftFolder(str(staging_parent))
            script = draft_folder.create_draft(staging_name, job.config.resolution.width, job.config.resolution.height, fps=30)
            has_alert_overlays = bool(effects and any(
                effect.type == "ALERT"
                for cue in effects
                if cue is not None
                for effect in cue.effects
            ))

            self._add_tracks(script, has_alert_overlays)
            progress(35, "Adding images...")
            self._add_images(script, job, timings, warnings, effects, draw_clips)
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

    def _add_images(self, script, job: ProjectJob, timings: list[ImageTiming], warnings: list[str], effects=None, draw_clips: dict[int, Path] | None = None) -> None:
        cc = self.cc
        main_srt = getattr(job.config, 'main_effect_srt', job.config.effect_direction_srt)
        manual_resolver = ManualRoiResolver(roi_sidecar_path(main_srt)) if effects and main_srt else None
        roi_resolver = manual_resolver
        engine = MotionEngine(job.config.resolution.width, job.config.resolution.height, job.config.motion_strength, roi_resolver)
        for index, (image_path, timing) in enumerate(zip(job.images, timings)):
            # ── Draw clip substitution ─────────────────────────────────────
            draw_mp4 = (draw_clips or {}).get(index)
            if draw_mp4 is not None and draw_mp4.is_file():
                material = cc.VideoMaterial(str(draw_mp4))
                base_scale = engine.capcut_cover_scale(job.config.resolution.width, job.config.resolution.height, material.width, material.height)
                segment = cc.VideoSegment(
                    material,
                    cc.Timerange(timing.start_us, timing.duration_us),
                    clip_settings=cc.ClipSettings(scale_x=base_scale, scale_y=base_scale),
                )
                # Draw clips do not get additional motion effects; the renderer
                # controls all visual motion internally.
                if effects and job.config.transition_enabled and index < len(job.images) - 1:
                    self._maybe_add_transition(segment, effects[index], timing, timings[index + 1], job, warnings, index)
                elif job.config.transition_enabled and index < len(job.images) - 1:
                    self._maybe_add_transition(segment, None, timing, timings[index + 1], job, warnings, index)
                script.add_segment(segment, "Images")
                continue

            # If this cue was classified as a DRAW cue in the unified SRT, failure to
            # have a rendered draw clip must NEVER silently fall back to a static image.
            if effects is not None and effects[index] is None:
                raise ValidationError(f"Image {index + 1}: DRAW cue failed to produce a valid rendered draw clip.")

            # ── Standard image path ────────────────────────────────────────
            material = cc.VideoMaterial(str(image_path))
            base_scale = engine.capcut_cover_scale(job.config.resolution.width, job.config.resolution.height, material.width, material.height)
            segment = cc.VideoSegment(
                material,
                cc.Timerange(timing.start_us, timing.duration_us),
                clip_settings=cc.ClipSettings(scale_x=base_scale, scale_y=base_scale),
            )
            if effects and effects[index] is not None:
                plan = engine.plan_effect(effects[index], image_path, material.width, material.height, timing.duration_us)
                warnings.extend(plan.warnings)
            elif job.config.motion_enabled:
                plan = engine.plan_generic(job.config.motion_mode, image_path, material.width, material.height, timing.duration_us, f"{job.name}:{index}:{image_path}")
            else:
                plan = None
            if plan:
                self._apply_motion_plan(segment, plan, base_scale)
            if effects and effects[index] is not None:
                self._add_alert_overlays(script, effects[index], image_path, timing, job, base_scale, plan, manual_resolver, index)
            if job.config.transition_enabled and index < len(job.images) - 1:
                self._maybe_add_transition(segment, effects[index] if effects else None, timing, timings[index + 1], job, warnings, index)
            script.add_segment(segment, "Images")

        if manual_resolver:
            warnings.extend(dict.fromkeys(manual_resolver.warnings))

    def _maybe_add_transition(self, segment, effect, timing: ImageTiming, next_timing: ImageTiming, job: ProjectJob, warnings: list[str], index: int) -> None:
        """Conditionally attach a blur transition to *segment*."""
        soft_cut = effect and effect.transition_out and effect.transition_out.casefold() == "soft cut"
        transition_duration = min(
            job.config.transition_duration_us,
            timing.duration_us // 2,
            next_timing.duration_us // 2,
        )
        if transition_duration > 0 and not soft_cut:
            transition_type = self._blur_transition_type()
            if transition_type is not None:
                segment.add_transition(transition_type, duration=transition_duration)
                if transition_duration < job.config.transition_duration_us:
                    warnings.append(f"Transition {index + 1} shortened to {transition_duration / 1_000_000:.3f}s")

    def _render_draw_clips(self, job: ProjectJob, timings: list[ImageTiming], progress, warnings: list[str]) -> dict[int, Path]:
        """Pre-render draw clips for all draw cues and return a map of index -> MP4 path."""
        from auto_capcut.core.unified_effect_parser import parse_unified_effect
        from auto_capcut.core.draw_models import DrawProjectConfig
        from auto_capcut.core.draw_renderer import DrawRenderService
        import logging
        logger = logging.getLogger(__name__)

        # Production draw rendering always uses effect_direction_srt (single-effect production contract)
        draw_srt = job.config.effect_direction_srt
        if draw_srt is None or not draw_srt.is_file():
            return {}

        unified = parse_unified_effect(draw_srt)
        if not unified.has_draw_cues:
            return {}

        # Map cue index (1-based) to 0-based image index, then filter draw cues
        draw_plans_by_img: dict[int, object] = {}   # {0-based image index -> DrawImagePlan}

        draw_indexes: list[int] = []
        for cue in unified.cues:
            img_idx = cue.index - 1  # 0-based
            if cue.kind == "draw" and cue.draw_plan is not None and img_idx < len(job.images):
                draw_plans_by_img[img_idx] = cue.draw_plan
                draw_indexes.append(img_idx)

        if not draw_indexes:
            return {}

        # Build parallel draw_plans list indexed same as job.images
        aligned_plans = [draw_plans_by_img.get(i) for i in range(len(job.images))]
        # Fill None slots with a placeholder (they won't be rendered since only draw_indexes are passed)
        from auto_capcut.core.draw_models import DrawMode, DrawStyle, DrawImagePlan as _DIP
        for i in range(len(aligned_plans)):
            if aligned_plans[i] is None:
                aligned_plans[i] = _DIP(
                    i + 1, None, 0,
                    timings[i].duration_us if i < len(timings) else 1_000_000,
                    DrawMode.BASIC, DrawStyle.V2, "auto",
                    (),  # no actions — placeholder only, not rendered
                )

        resolution = (job.config.resolution.width, job.config.resolution.height)
        cache_root = (job.config.draft_folder / ".autocapcut_draw_cache") if job.config.draft_folder else Path(tempfile.mkdtemp())
        draw_output_folder = cache_root / "clips"
        draw_output_folder.mkdir(parents=True, exist_ok=True)

        draw_config = DrawProjectConfig(
            image_folder=job.images[0].parent if job.images else Path("."),
            effect_file=draw_srt,
            output_folder=draw_output_folder,
            scene_file=job.config.draw_scene_json,
            resolution=resolution,
            fps=30,
            remove_background=job.config.draw_remove_background,
            fallback_basic=job.config.draw_fallback_basic,
            advanced_diagnostics=job.config.draw_diagnostics,
            reuse_cache=job.config.draw_reuse_cache,
            post_motion=job.config.motion_mode if job.config.motion_enabled else "none",
        )


        progress(20, f"Rendering {len(draw_indexes)} draw clip(s)...")

        def draw_progress(value: int, message: str) -> None:
            progress(20 + round(value * 0.15), message)

        service = DrawRenderService()
        rendered = service.render_subset(
            draw_config,
            aligned_plans,
            list(job.images),
            draw_indexes,
            progress=draw_progress,
        )

        # Validate that all requested draw cues produced valid MP4s with correct duration
        DURATION_TOLERANCE_US = 70_000  # ~2 frames at 30fps tolerance for frame rounding
        for img_idx in draw_indexes:
            clip_path = rendered.get(img_idx)
            if clip_path is None or not clip_path.is_file() or clip_path.stat().st_size == 0:
                raise ValidationError(f"Failed to produce draw clip for Image {img_idx + 1}: output file is missing or empty")

            actual_duration_us = probe_duration_us(clip_path)
            expected_duration_us = timings[img_idx].duration_us
            diff_us = abs(actual_duration_us - expected_duration_us)
            logger.info(
                "Image %03d draw clip duration validation: expected=%dus, actual=%dus, diff=%dus",
                img_idx + 1,
                expected_duration_us,
                actual_duration_us,
                diff_us,
            )
            if diff_us > DURATION_TOLERANCE_US:
                raise ValidationError(
                    f"Draw clip {clip_path.name} (Image {img_idx + 1}) duration mismatch: "
                    f"expected {expected_duration_us / 1_000_000:.3f}s, "
                    f"actual {actual_duration_us / 1_000_000:.3f}s "
                    f"(difference {diff_us / 1_000_000:.3f}s exceeds tolerance)"
                )

        progress(35, "Draw clips ready.")
        return rendered




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

    def _add_alert_overlays(self, script, cue, image_path: Path, timing: ImageTiming, job: ProjectJob, base_scale: float, plan, resolver, cue_index: int = 0) -> None:
        if resolver is None:
            return
        cc = self.cc
        for effect_index, effect in enumerate(cue.effects):
            if effect.type != "ALERT" or not effect.target_id:
                continue
            if effect.params.get("preset", "").strip():
                continue
            roi = resolver.resolve(image_path, effect.target_id, cue.image_index)
            if roi is None:
                continue
            _main_srt = getattr(job.config, 'main_effect_srt', job.config.effect_direction_srt)
            overlay_dir = _main_srt.with_suffix(".overlays") if _main_srt else image_path.parent / ".overlays"
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
