"""Build production-safe captured templates from local CapCut effects.

The promoter is intentionally conservative: local metadata is accepted only
when it can be paired with a complete cached effect package and the captured
Warning material schema.  Unknown or incomplete packages remain unresolved.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from auto_capcut.core.captured_effect_template import (
    CapturedEffectTemplate,
    CapturedEffectTemplateRepository,
    ResolvedCapturedEffectPreset,
    SEGMENT_ID_PLACEHOLDER,
    MATERIAL_ID_PLACEHOLDER,
    validate_resource_path,
    load_captured_template,
    slugify_preset_name,
)


@dataclass(frozen=True)
class PreparedLocalEffect:
    stable_key: str
    preset_key: str
    display_name: str
    template: CapturedEffectTemplate | None
    reason: str = ""
    effect_id: str = ""
    resource_id: str = ""

    @property
    def resolved(self) -> bool:
        return self.template is not None


@dataclass(frozen=True)
class PromotionSummary:
    prepared: tuple[PreparedLocalEffect, ...]
    build_result: Any | None
    unresolved: tuple[PreparedLocalEffect, ...]


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LocalEffectTemplatePromoter:
    def __init__(self, repository: CapturedEffectTemplateRepository | None = None) -> None:
        self.repository = repository or CapturedEffectTemplateRepository()

    def allocate_preset_keys(self, entries: Iterable[Any]) -> dict[str, str]:
        existing = self.repository.load_registry()
        by_stable = {item.stable_key: key for key, item in existing.items() if item.stable_key}
        used = {key: item.stable_key for key, item in existing.items()}
        output: dict[str, str] = {}
        ordered = sorted(entries, key=lambda item: (int(getattr(item, "test_index", 0) or 0), str(getattr(item, "stable_key", ""))))
        for entry in ordered:
            stable_key = str(getattr(entry, "stable_key", ""))
            if not stable_key:
                continue
            if stable_key in by_stable:
                output[stable_key] = by_stable[stable_key]
                continue
            base = slugify_preset_name(str(getattr(entry, "display_name", "effect")))
            key = base
            if key in used and used[key] != stable_key:
                suffix = re.sub(r"[^a-z0-9]", "", str(getattr(entry, "effect_id", "" ) or getattr(entry, "resource_id", "" )).casefold())[-8:] or "effect"
                key = f"{base}_{suffix}"
            counter = 2
            while key in used and used[key] != stable_key:
                key = f"{base}_{counter}"
                counter += 1
            used[key] = stable_key
            output[stable_key] = key
        return output

    def prepare(self, entries: Iterable[Any]) -> list[PreparedLocalEffect]:
        keys = self.allocate_preset_keys(entries)
        return [self._prepare_one(entry, keys.get(str(getattr(entry, "stable_key", "")), "effect")) for entry in entries]

    def _prepare_one(self, entry: Any, preset_key: str) -> PreparedLocalEffect:
        stable_key = str(getattr(entry, "stable_key", ""))
        display_name = str(getattr(entry, "display_name", "") or getattr(entry, "effect_id", "effect"))
        if str(getattr(entry, "source", "")).casefold() != "local":
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "Only CapCut Local effects can be promoted")
        raw = dict(getattr(entry, "raw_metadata_subset", {}) or {})
        effect_id = str(getattr(entry, "effect_id", "") or raw.get("effect_id", ""))
        resource_id = str(getattr(entry, "resource_id", "") or raw.get("resource_id", "") or effect_id)
        if not effect_id or not resource_id or effect_id != resource_id:
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "effect_id/resource_id are missing or inconsistent", effect_id, resource_id)
        if str(raw.get("effect_type", "7")) != "7":
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "record is not a video effect", effect_id, resource_id)
        md5 = str(raw.get("md5", "")).strip().casefold()
        root = Path(getattr(entry, "local_resource_path", "") or "")
        package = root / md5 if md5 else Path("")
        if root.name != resource_id:
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "cached resource root does not match resource_id", effect_id, resource_id)
        if not md5 or not package.is_dir():
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "cached resource package is missing", effect_id, resource_id)
        if not (package / "config.json").is_file() or not (package / "AmazingFeature").is_dir():
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "resource package is not compatible with captured video-effect schema", effect_id, resource_id)
        resource = validate_resource_path(package)
        if not resource.valid:
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "cached resource package failed validation", effect_id, resource_id)
        warning_path = self.repository.root / "captured_templates" / "warning" / "captured_effect_template.json"
        canonical = load_captured_template(warning_path)
        if canonical is None:
            return PreparedLocalEffect(stable_key, preset_key, display_name, None, "captured Warning template is unavailable", effect_id, resource_id)
        material = copy.deepcopy(canonical.material)
        material["id"] = MATERIAL_ID_PLACEHOLDER
        material["effect_id"] = effect_id
        material["resource_id"] = resource_id
        material["name"] = display_name
        material["path"] = package.as_posix()
        material["source_platform"] = int(raw.get("source_platform", raw.get("source", material.get("source_platform", 1))) or 0)
        material["type"] = "video_effect"
        material["request_id"] = ""
        for key in ("category_id", "category_name", "platform", "sub_type", "item_effect_type"):
            if raw.get(key) is not None:
                material[key] = raw[key]
        sdk_extra = raw.get("sdk_extra") or raw.get("extra")
        sdk_payload: dict[str, Any] | None = None
        if isinstance(sdk_extra, dict):
            sdk_payload = sdk_extra
            material["sdk_extra"] = json.dumps(sdk_extra, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(sdk_extra, str):
            material["sdk_extra"] = sdk_extra
            try:
                parsed = json.loads(sdk_extra)
                if isinstance(parsed, dict):
                    sdk_payload = parsed
            except json.JSONDecodeError:
                pass
        if sdk_payload:
            settings = sdk_payload.get("setting")
            if isinstance(settings, dict) and isinstance(settings.get("effect_adjust_params"), list):
                material["adjust_params"] = copy.deepcopy(settings["effect_adjust_params"])
        segment = copy.deepcopy(canonical.segment)
        segment["id"] = SEGMENT_ID_PLACEHOLDER
        segment["material_id"] = MATERIAL_ID_PLACEHOLDER
        fingerprint = _json_hash({
            "material_keys": sorted(material),
            "config": (package / "config.json").read_text(encoding="utf-8", errors="ignore"),
        })
        template = CapturedEffectTemplate(
            material=material,
            segment=segment,
            companion_records=[],
            resource_validation=resource,
            source_draft="local-cache",
            source_effect_name=display_name,
            source_effect_id=effect_id,
            provenance="local_reconstructed",
            source_stable_key=stable_key,
            compatibility_fingerprint=fingerprint,
        )
        return PreparedLocalEffect(stable_key, preset_key, display_name, template, "", effect_id, resource_id)

    def persist(self, prepared: Iterable[PreparedLocalEffect], built_keys: set[str]) -> None:
        for item in prepared:
            if item.template is None:
                self.repository.record_unresolved(
                    item.preset_key,
                    stable_key=item.stable_key,
                    display_name=item.display_name,
                    effect_id=item.effect_id,
                    resource_id=item.resource_id,
                    error=item.reason,
                )
                continue
            if item.preset_key in built_keys:
                self.repository.register_template(
                    item.preset_key,
                    item.template,
                    stable_key=item.stable_key,
                    display_name=item.display_name,
                    state="promoted",
                )
            else:
                self.repository.record_unresolved(
                    item.preset_key,
                    stable_key=item.stable_key,
                    display_name=item.display_name,
                    effect_id=str(item.template.source_effect_id),
                    resource_id=str(item.template.material.get("resource_id", "")),
                    error="test draft candidate failed validation",
                )
