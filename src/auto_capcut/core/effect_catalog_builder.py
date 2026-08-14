"""Isolated CapCut effect catalog draft builder.

This module intentionally does not share the production project builder.  It
probes and applies the real pyCapCut effect enum members, so a missing resource
can be reported for that candidate while the remaining catalog continues.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from auto_capcut.core.capcut_compat import patch_capcut_metadata
from auto_capcut.core.captured_effect_template import (
    CapturedEffectTemplateCloner,
    CapturedEffectTemplateRepository,
    ResolvedCapturedEffectPreset,
)
from auto_capcut.core.errors import ValidationError
from auto_capcut.core.validation import validate_draft_json
from auto_capcut.utils.paths import unique_project_name


@dataclass(frozen=True)
class CatalogBuildFailure:
    stable_key: str
    display_name: str
    error: str


@dataclass(frozen=True)
class CatalogBuildPart:
    project_name: str
    project_path: Path
    selected: int
    built: int
    failures: tuple[CatalogBuildFailure, ...] = ()


@dataclass(frozen=True)
class CatalogBuildSummary:
    selected: int
    built: int
    failures: tuple[CatalogBuildFailure, ...]
    parts: tuple[CatalogBuildPart, ...]
    report_path: Path | None = None

    @property
    def failed(self) -> int:
        return len(self.failures)


class CatalogDraftBuilder:
    """Build one or more isolated effect-catalog drafts.

    ``entries`` may be the catalog model or any object exposing the fields
    recorded by the scanner (``source``, ``enum_name``, ``stable_key`` and
    ``display_name``).  This keeps the builder independent of catalog storage.
    """

    def __init__(self, capcut_module: Any | None = None) -> None:
        if capcut_module is None:
            try:
                import pycapcut as capcut_module
            except ImportError as exc:  # pragma: no cover
                raise ValidationError("pycapcut is not installed") from exc
        self.cc = capcut_module

    def build_catalog(
        self,
        entries: Iterable[Any],
        image_path: str | Path,
        draft_folder: str | Path,
        *,
        duration_seconds: float = 2.0,
        batch_size: int = 30,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        project_prefix: str = "EffectCatalog",
        progress_callback: Callable[[int, str], None] | None = None,
        captured_templates: dict[str, ResolvedCapturedEffectPreset] | None = None,
    ) -> CatalogBuildSummary:
        image = Path(image_path).resolve()
        root = Path(draft_folder).resolve()
        if not image.is_file():
            raise ValidationError(f"Catalog test image not found: {image}")
        if not root.is_dir():
            raise ValidationError(f"CapCut draft folder not found: {root}")
        if duration_seconds <= 0:
            raise ValidationError("Catalog segment duration must be positive")
        duration_us = max(1, round(float(duration_seconds) * 1_000_000))
        batch_size = max(1, min(30, int(batch_size)))
        ordered = sorted(list(entries), key=lambda item: int(getattr(item, "test_index", 0) or 0))
        failures: list[CatalogBuildFailure] = []
        parts: list[CatalogBuildPart] = []
        total = len(ordered)
        for batch_start in range(0, total, batch_size):
            batch = ordered[batch_start : batch_start + batch_size]
            part_name = unique_project_name(root, f"{project_prefix}_{batch_start // batch_size + 1}")
            part, part_failures = self._build_part(
                batch, image, root, part_name, duration_us, width, height, fps,
                batch_start, total, progress_callback,
                captured_templates,
            )
            parts.append(part)
            failures.extend(part_failures)
        report_path = root / f"{project_prefix}_build_report.json"
        report = {
            "selected": total,
            "built": total - len(failures),
            "failed": [asdict(item) for item in failures],
            "parts": [
                {"project_name": p.project_name, "project_path": str(p.project_path), "selected": p.selected, "built": p.built}
                for p in parts
            ],
            "generated_at": int(time.time()),
        }
        self._atomic_json(report_path, report)
        if progress_callback:
            progress_callback(100, f"Effect catalog complete: {total - len(failures)}/{total} built")
        return CatalogBuildSummary(total, total - len(failures), tuple(failures), tuple(parts), report_path)

    def _build_part(
        self, batch: list[Any], image: Path, root: Path, name: str,
        duration_us: int, width: int, height: int, fps: int,
        offset: int, total: int, progress_callback: Callable[[int, str], None] | None,
        captured_templates: dict[str, ResolvedCapturedEffectPreset] | None = None,
    ) -> tuple[CatalogBuildPart, list[CatalogBuildFailure]]:
        cc = self.cc
        failures: list[CatalogBuildFailure] = []
        # Captured templates are deliberately loaded once per part. Entries
        # without a registered captured template continue through pyCapCut.
        repository = CapturedEffectTemplateRepository()
        captured_templates = dict(captured_templates or {})
        for entry in batch:
            stable_key = str(getattr(entry, "stable_key", ""))
            if stable_key and stable_key not in captured_templates:
                resolved = repository.template_for_stable_key(stable_key)
                if resolved is not None:
                    captured_templates[stable_key] = resolved
        staging_parent = Path(tempfile.mkdtemp(prefix="autocapcut-effect-", dir=str(root)))
        try:
            draft_folder = cc.DraftFolder(str(staging_parent))
            script = draft_folder.create_draft(name, width, height, fps=fps)
            script.add_track(cc.TrackType.video, "Images", absolute_index=0)
            script.add_track(cc.TrackType.effect, "Catalog Effects", absolute_index=10000)
            script.add_track(cc.TrackType.text, "Catalog Labels", absolute_index=15000)
            material = cc.VideoMaterial(str(image))
            cover = self._cover_scale(width, height, material.width, material.height)
            style = cc.TextStyle(size=8.0, bold=True, color=(1.0, 1.0, 1.0), align=0, auto_wrapping=True, max_line_width=0.82)
            border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=45.0)
            # Baseline is always the first, isolated segment in every part.
            baseline = cc.VideoSegment(material, cc.Timerange(0, duration_us), clip_settings=cc.ClipSettings(scale_x=cover, scale_y=cover))
            script.add_segment(baseline, "Images")
            script.add_segment(cc.TextSegment("BASELINE - NO EFFECT", cc.Timerange(0, duration_us), style=style, border=border, clip_settings=cc.ClipSettings(transform_x=-0.78, transform_y=0.78)), "Catalog Labels")
            built = 0
            injected: list[tuple[ResolvedCapturedEffectPreset, int, int, str, str]] = []
            for local_index, entry in enumerate(batch):
                label = str(getattr(entry, "display_name", None) or getattr(entry, "enum_name", None) or getattr(entry, "effect_id", "effect"))
                stable_key = str(getattr(entry, "stable_key", "")) or self._stable_key(entry)
                start = duration_us * (local_index + 1)
                target = cc.Timerange(start, duration_us)
                try:
                    captured_preset = captured_templates.get(stable_key)
                    captured_warning = captured_preset is not None
                    enum_member = None if captured_warning else self._resolve_effect(entry)
                    # Keep the timeline slot and native label even when this
                    # particular resource cannot be instantiated. This makes
                    # failures inspectable while allowing all later candidates
                    # to be tested in the same draft.
                    segment = cc.VideoSegment(material, target, clip_settings=cc.ClipSettings(scale_x=cover, scale_y=cover))
                    script.add_segment(segment, "Images")
                    script.add_segment(cc.TextSegment(label, target, style=style, border=border, clip_settings=cc.ClipSettings(transform_x=-0.78, transform_y=0.78)), "Catalog Labels")
                    if captured_warning:
                        # The normal pyCapCut path emits a placeholder material
                        # for Warning.  Its native segment is replaced after
                        # serialization with the exact material payload
                        # captured from CapCut's working test_8 project.
                        injected.append((captured_preset, start, duration_us, stable_key, label))
                    else:
                        # Probe/serialize this candidate independently before
                        # adding it to the shared script. Unsupported resources
                        # are skipped.
                        self._probe_effect(enum_member, duration_us, width, height, fps, self._params(entry))
                        script.add_effect(enum_member, target, "Catalog Effects", params=self._params(entry))
                    built += 1
                except Exception as exc:
                    failures.append(CatalogBuildFailure(stable_key, label, str(exc)[:500]))
                if progress_callback and total:
                    progress_callback(min(95, int((offset + local_index + 1) * 95 / total)), f"Testing {offset + local_index + 1}/{total}: {label}")
            script.save()
            staging_project = staging_parent / name
            if injected:
                # Remove the placeholder material (if any) and inject the
                # captured resource-backed record.  Older pyCapCut versions do
                # not create one for the bypass path, so injection is safe in
                # either case.
                self._remove_warning_placeholders(staging_project)
                for preset, start_us, effect_duration, stable_key, label in injected:
                    try:
                        CapturedEffectTemplateCloner.inject_file(
                            staging_project,
                            preset,
                            start_us=start_us,
                            duration_us=effect_duration,
                            track_name="Catalog Effects",
                        )
                    except Exception as exc:
                        failures.append(CatalogBuildFailure(stable_key, label, str(exc)[:500]))
                        built = max(0, built - 1)
            validate_draft_json(staging_project, duration_us * (len(batch) + 1))
            destination = root / name
            staging_project.replace(destination)
            try:
                patch_capcut_metadata(destination, name, duration_us * (len(batch) + 1))
                validate_draft_json(destination, duration_us * (len(batch) + 1))
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                raise
            return CatalogBuildPart(name, destination, len(batch), built, tuple(failures)), failures
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    @staticmethod
    def _remove_warning_placeholders(project: Path) -> None:
        """Remove synthetic placeholder Warning records before injection."""
        path = project / "draft_content.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        materials = data.get("materials", {})
        rows = materials.get("video_effects", [])
        if isinstance(rows, list):
            materials["video_effects"] = [
                row for row in rows
                if not (isinstance(row, dict) and str(row.get("path", "")).startswith("##_material_placeholder_"))
            ]
        for track in data.get("tracks", []):
            if not isinstance(track, dict) or track.get("type") != "effect":
                continue
            track["segments"] = [
                segment for segment in track.get("segments", [])
                if not (isinstance(segment, dict) and str(segment.get("material_id", "")).startswith("##_material_placeholder_"))
            ]
        CatalogDraftBuilder._atomic_json(path, data)

    def _resolve_effect(self, entry: Any):
        source = str(getattr(entry, "source", "scene")).casefold()
        if source == "local":
            if not bool(getattr(entry, "buildable", False)) or not bool(getattr(entry, "pycapcut_match", False)):
                raise ValueError("CapCut Local effect is discovery-only; no safe draft material adapter is available")
            for enum_type in (self.cc.VideoSceneEffectType, self.cc.VideoCharacterEffectType):
                for member in enum_type:
                    if str(getattr(member.value, "effect_id", "")) == str(getattr(entry, "effect_id", "")):
                        return member
            raise ValueError("CapCut Local effect has no installed pyCapCut match")
        enum_name = str(getattr(entry, "enum_name", ""))
        enum_type = self.cc.VideoCharacterEffectType if source == "character" else self.cc.VideoSceneEffectType
        try:
            return enum_type[enum_name]
        except (KeyError, TypeError):
            effect_id = str(getattr(entry, "effect_id", ""))
            for member in enum_type:
                if str(getattr(member.value, "effect_id", "")) == effect_id:
                    return member
            raise ValueError(f"Effect is not installed: {source}:{enum_name or effect_id}")

    def _probe_effect(self, effect, duration_us: int, width: int, height: int, fps: int, params=None) -> None:
        cc = self.cc
        probe = cc.ScriptFile(width, height, fps)
        probe.add_track(cc.TrackType.effect, "Probe Effects")
        probe.add_effect(effect, cc.Timerange(0, duration_us), "Probe Effects", params=params)
        probe.dumps()

    @staticmethod
    def _params(entry: Any):
        params = getattr(entry, "params", None)
        if not params:
            return None
        # pyCapCut's ``params`` input is expressed as a 0..100 control value,
        # while catalog metadata stores normalized default values.  Passing
        # those normalized floats would unintentionally select near-zero
        # settings; ``None`` explicitly requests each enum's native default.
        return [None] * len(params)

    @staticmethod
    def _stable_key(entry: Any) -> str:
        return f"{getattr(entry, 'source', 'scene')}:{getattr(entry, 'effect_id', getattr(entry, 'enum_name', 'effect'))}"

    @staticmethod
    def _cover_scale(canvas_w: int, canvas_h: int, image_w: int, image_h: int) -> float:
        fit_w = canvas_w / max(1, image_w)
        fit_h = canvas_h / max(1, image_h)
        return max(fit_w, fit_h) / min(fit_w, fit_h)

    @staticmethod
    def _atomic_json(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

