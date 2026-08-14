"""Capture and clone real CapCut effect payloads.

This module is intentionally independent from the production camera/effect
engine.  CapCut effect materials contain fields which pyCapCut does not expose
when serialising a synthetic effect.  The service snapshots a known-good
effect, keeps those fields intact, and can inject a cloned segment into a
staged draft.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


CAPTURE_VERSION = "1"
MATERIAL_ID_PLACEHOLDER = "__MATERIAL_ID__"
SEGMENT_ID_PLACEHOLDER = "__SEGMENT_ID__"
START_US_PLACEHOLDER = "__START_US__"
DURATION_US_PLACEHOLDER = "__DURATION_US__"
WARNING_EFFECT_ID = "7399465244088618245"
PRESET_REGISTRY_VERSION = 1
PRESET_STATES = {"promoted", "draft_recognized", "render_confirmed", "unresolved"}


@dataclass(frozen=True)
class ResourceValidation:
    path: str
    exists: bool
    is_file: bool = False
    file_count: int = 0
    total_bytes: int = 0
    missing_paths: tuple[str, ...] = ()
    valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"missing_paths": list(self.missing_paths)}


@dataclass
class CapturedEffectTemplate:
    material: dict[str, Any]
    segment: dict[str, Any]
    companion_records: list[dict[str, Any]] = field(default_factory=list)
    resource_validation: ResourceValidation | None = None
    source_draft: str = ""
    source_effect_name: str = ""
    source_effect_id: str = ""
    capture_version: str = CAPTURE_VERSION
    provenance: str = "captured"
    source_stable_key: str = ""
    compatibility_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_version": self.capture_version,
            "source_draft": self.source_draft,
            "source_effect_name": self.source_effect_name,
            "source_effect_id": self.source_effect_id,
            "provenance": self.provenance,
            "source_stable_key": self.source_stable_key,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "material": copy.deepcopy(self.material),
            "segment": copy.deepcopy(self.segment),
            "companion_records": copy.deepcopy(self.companion_records),
            "resource_validation": self.resource_validation.to_dict() if self.resource_validation else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CapturedEffectTemplate":
        rv = payload.get("resource_validation") or None
        validation = ResourceValidation(
            path=str(rv.get("path", "")), exists=bool(rv.get("exists", False)),
            is_file=bool(rv.get("is_file", False)), file_count=int(rv.get("file_count", 0) or 0),
            total_bytes=int(rv.get("total_bytes", 0) or 0),
            missing_paths=tuple(str(x) for x in rv.get("missing_paths", []) or []),
            valid=bool(rv.get("valid", False)),
        ) if isinstance(rv, dict) else None
        return cls(
            material=copy.deepcopy(dict(payload.get("material") or {})),
            segment=copy.deepcopy(dict(payload.get("segment") or {})),
            companion_records=copy.deepcopy(list(payload.get("companion_records") or [])),
            resource_validation=validation,
            source_draft=str(payload.get("source_draft", "")),
            source_effect_name=str(payload.get("source_effect_name", "")),
            source_effect_id=str(payload.get("source_effect_id", "")),
            capture_version=str(payload.get("capture_version", CAPTURE_VERSION)),
            provenance=str(payload.get("provenance", "captured")),
            source_stable_key=str(payload.get("source_stable_key", "")),
            compatibility_fingerprint=str(payload.get("compatibility_fingerprint", "")),
        )


@dataclass(frozen=True)
class ResolvedCapturedEffectPreset:
    key: str
    effect_id: str
    template: CapturedEffectTemplate
    template_path: Path
    state: str = "render_confirmed"
    stable_key: str = ""


@dataclass(frozen=True)
class PresetRegistryEntry:
    preset_key: str
    stable_key: str
    display_name: str
    effect_id: str
    resource_id: str
    template_path: str
    resource_fingerprint: str = ""
    schema_fingerprint: str = ""
    state: str = "promoted"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PresetRegistryEntry":
        state = str(payload.get("state", "unresolved"))
        if state == "captured":
            state = "promoted"
        if state not in PRESET_STATES:
            state = "unresolved"
        return cls(
            preset_key=str(payload.get("preset_key", "")).strip().casefold(),
            stable_key=str(payload.get("stable_key", "")),
            display_name=str(payload.get("display_name", "")),
            effect_id=str(payload.get("effect_id", "")),
            resource_id=str(payload.get("resource_id", "")),
            template_path=str(payload.get("template_path", "")),
            resource_fingerprint=str(payload.get("resource_fingerprint", "")),
            schema_fingerprint=str(payload.get("schema_fingerprint", "")),
            state=state,
            error=str(payload.get("error", "")),
        )


def preset_registry_path(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root) / "preset_registry.json"
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "AutoCapCut" / "effect_catalog" / "preset_registry.json"


def slugify_preset_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_") or "effect"


class CapturedEffectTemplateRepository:
    """Resolve production-safe preset keys to captured, resource-backed templates."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_template_path().parents[2]
        self.registry_path = preset_registry_path(self.root)

    def load_registry(self) -> dict[str, PresetRegistryEntry]:
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}
        rows = payload.get("presets", {}) if isinstance(payload, dict) else {}
        if isinstance(rows, list):
            rows = {str(row.get("preset_key", "")): row for row in rows if isinstance(row, dict)}
        return {
            key: PresetRegistryEntry.from_dict(row)
            for key, row in rows.items()
            if isinstance(row, dict) and str(key).strip()
        }

    def save_registry(self, entries: dict[str, PresetRegistryEntry]) -> None:
        _atomic_json(self.registry_path, {
            "schema_version": PRESET_REGISTRY_VERSION,
            "presets": {key: entry.to_dict() for key, entry in sorted(entries.items())},
        })

    def ensure_warning_bootstrap(self) -> PresetRegistryEntry | None:
        key = "warning"
        path = self.root / "captured_templates" / key / "captured_effect_template.json"
        template = load_captured_template(path)
        if template is None:
            return None
        effect_id = str(template.source_effect_id or template.material.get("effect_id", ""))
        resource_id = str(template.material.get("resource_id", effect_id))
        resource = validate_resource_path(str(template.material.get("path", "")))
        if effect_id != WARNING_EFFECT_ID or resource_id != WARNING_EFFECT_ID or not resource.valid:
            return None
        entries = self.load_registry()
        existing = entries.get(key)
        if existing is not None and existing.resource_fingerprint and existing.resource_fingerprint != f"{resource.file_count}:{resource.total_bytes}":
            return existing
        entry = PresetRegistryEntry(
            preset_key=key,
            stable_key=existing.stable_key if existing else f"local:{effect_id}",
            display_name=existing.display_name if existing else str(template.material.get("name", "Warning")),
            effect_id=effect_id,
            resource_id=resource_id,
            template_path=str(path),
            resource_fingerprint=f"{resource.file_count}:{resource.total_bytes}",
            schema_fingerprint=template.compatibility_fingerprint,
            state=existing.state if existing else "render_confirmed",
            error="",
        )
        entries[key] = entry
        self.save_registry(entries)
        return entry

    def register_template(
        self,
        preset_key: str,
        template: CapturedEffectTemplate,
        *,
        stable_key: str,
        display_name: str,
        state: str = "promoted",
        error: str = "",
    ) -> PresetRegistryEntry:
        key = str(preset_key).strip().casefold()
        if not key:
            raise ValueError("preset key is required")
        if state == "captured":
            state = "promoted"
        if state not in PRESET_STATES:
            raise ValueError(f"unknown preset state: {state}")
        path = self.root / "captured_templates" / key / "captured_effect_template.json"
        save_template(template, path)
        resource = template.resource_validation or validate_resource_path(str(template.material.get("path", "")))
        entry = PresetRegistryEntry(
            preset_key=key,
            stable_key=stable_key,
            display_name=display_name,
            effect_id=str(template.source_effect_id or template.material.get("effect_id", "")),
            resource_id=str(template.material.get("resource_id", "")),
            template_path=str(path),
            resource_fingerprint=f"{resource.file_count}:{resource.total_bytes}" if resource else "",
            schema_fingerprint=template.compatibility_fingerprint,
            state=state,
            error=error,
        )
        entries = self.load_registry()
        entries[key] = entry
        self.save_registry(entries)
        return entry

    def record_unresolved(self, preset_key: str, *, stable_key: str, display_name: str, effect_id: str = "", resource_id: str = "", error: str = "") -> PresetRegistryEntry:
        key = str(preset_key).strip().casefold()
        entries = self.load_registry()
        current = entries.get(key)
        entry = PresetRegistryEntry(
            preset_key=key,
            stable_key=stable_key,
            display_name=display_name,
            effect_id=effect_id,
            resource_id=resource_id,
            template_path=current.template_path if current else "",
            resource_fingerprint=current.resource_fingerprint if current else "",
            schema_fingerprint=current.schema_fingerprint if current else "",
            state="unresolved",
            error=error,
        )
        entries[key] = entry
        self.save_registry(entries)
        return entry

    def mark_state(self, preset_key: str, state: str, error: str = "") -> PresetRegistryEntry:
        key = str(preset_key).strip().casefold()
        if state == "captured":
            state = "promoted"
        if state not in PRESET_STATES:
            raise ValueError(f"unknown preset state: {state}")
        entries = self.load_registry()
        current = entries.get(key)
        if current is None:
            raise KeyError(key)
        if state == "draft_recognized" and current.state not in {"promoted", "draft_recognized"}:
            raise ValueError("preset must be promoted before draft recognition")
        if state == "render_confirmed" and current.state not in {"draft_recognized", "render_confirmed"}:
            raise ValueError("preset must be draft-recognized before render confirmation")
        updated = PresetRegistryEntry(**{**current.to_dict(), "state": state, "error": error})
        entries[key] = updated
        self.save_registry(entries)
        return updated

    def resolve_effect_preset(self, preset: str) -> ResolvedCapturedEffectPreset | None:
        key = slugify_preset_name(preset)
        self.ensure_warning_bootstrap()
        entry = self.load_registry().get(key)
        if entry is None or entry.state != "render_confirmed":
            return None
        def invalidate(reason: str) -> None:
            self.record_unresolved(
                key,
                stable_key=entry.stable_key,
                display_name=entry.display_name,
                effect_id=entry.effect_id,
                resource_id=entry.resource_id,
                error=reason,
            )
        path = Path(entry.template_path)
        template = load_captured_template(path)
        actual_id = "" if template is None else str(
            template.source_effect_id
            or template.material.get("effect_id", "")
            or template.material.get("resource_id", "")
        )
        if template is None or actual_id != entry.effect_id or str(template.material.get("resource_id", "")) != entry.resource_id:
            invalidate("template IDs or schema do not match registry")
            return None
        resource_path = str(template.material.get("path", "")).strip()
        validation = validate_resource_path(resource_path)
        if not resource_path or not validation.valid:
            invalidate("captured resource is missing or invalid")
            return None
        if entry.resource_fingerprint and entry.resource_fingerprint != f"{validation.file_count}:{validation.total_bytes}":
            invalidate("captured resource fingerprint changed")
            return None
        if entry.schema_fingerprint and template.compatibility_fingerprint and entry.schema_fingerprint != template.compatibility_fingerprint:
            invalidate("captured template schema fingerprint changed")
            return None
        return ResolvedCapturedEffectPreset(key, entry.effect_id, template, path, entry.state, entry.stable_key)

    def template_for_stable_key(self, stable_key: str) -> ResolvedCapturedEffectPreset | None:
        for entry in self.load_registry().values():
            if entry.stable_key != stable_key:
                continue
            template = load_captured_template(entry.template_path)
            if template is None:
                return None
            return ResolvedCapturedEffectPreset(
                entry.preset_key, entry.effect_id, template, Path(entry.template_path), entry.state, entry.stable_key,
            )
        return None


class CapturedEffectTemplateCloner:
    """Shared draft injector used by catalog tests and production builds."""

    @staticmethod
    def inject_file(
        draft_path: str | Path,
        resolved: ResolvedCapturedEffectPreset,
        *,
        start_us: int,
        duration_us: int,
        track_name: str,
    ) -> None:
        inject_captured_effect_file(
            draft_path,
            resolved.template,
            start_us=start_us,
            duration_us=duration_us,
            track_name=track_name,
        )


@dataclass(frozen=True)
class EffectDiff:
    changes: tuple[dict[str, Any], ...]
    categories: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"changes": [dict(item) for item in self.changes], "categories": dict(self.categories)}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _draft_content_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "draft_content.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"draft_content.json not found: {candidate}")
    return candidate


def load_draft_content(path: str | Path) -> dict[str, Any]:
    source = _draft_content_path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def _record_id(record: Any) -> str:
    return str(record.get("id", "")) if isinstance(record, dict) else ""


def _material_records(content: dict[str, Any]) -> list[dict[str, Any]]:
    materials = content.get("materials") or {}
    rows = materials.get("video_effects") or []
    return [row for row in rows if isinstance(row, dict)]


def _effect_segments(content: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for track in content.get("tracks") or []:
        if not isinstance(track, dict) or str(track.get("type", "")).casefold() != "effect":
            continue
        for segment in track.get("segments") or []:
            if isinstance(segment, dict):
                yield segment


def extract_effect_template(
    draft_path: str | Path,
    *,
    effect_name: str = "Warning",
    effect_id: str | None = None,
) -> CapturedEffectTemplate:
    """Extract an effect by following segment material references.

    Matching is by ``effect_id`` when supplied, then by exact display name.
    A segment is never selected merely because its track contains a similar
    label.  This prevents an unrelated effect from becoming the anchor.
    """
    source = _draft_content_path(draft_path)
    content = json.loads(source.read_text(encoding="utf-8"))
    materials = {str(row.get("id")): row for row in _material_records(content) if row.get("id")}
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for segment in _effect_segments(content):
        material = materials.get(str(segment.get("material_id", "")))
        if not material:
            continue
        if effect_id and str(material.get("effect_id", "")) == str(effect_id):
            candidates.append((segment, material))
        elif not effect_id and str(material.get("name", "")).casefold() == effect_name.casefold():
            candidates.append((segment, material))
    if not candidates:
        raise ValueError(f"Effect not found in draft: {effect_name or effect_id}")
    if len(candidates) > 1:
        # Multiple segments may intentionally reference one material.  The
        # material is still unambiguous; choose the first stable timeline use.
        candidates.sort(key=lambda item: int((item[0].get("target_timerange") or {}).get("start", 0) or 0))
    segment, material = candidates[0]
    refs = set(str(x) for x in segment.get("extra_material_refs", []) or [])
    refs.discard(str(material.get("id", "")))
    companions: list[dict[str, Any]] = []
    for rows in (content.get("materials") or {}).values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and str(row.get("id", "")) in refs:
                companions.append(copy.deepcopy(row))
    resource = validate_resource_path(str(material.get("path", ""))) if material.get("path") else None
    return CapturedEffectTemplate(
        material=copy.deepcopy(material), segment=copy.deepcopy(segment),
        companion_records=companions, resource_validation=resource,
        source_draft=str(source), source_effect_name=str(material.get("name", effect_name)),
        source_effect_id=str(material.get("effect_id", effect_id or "")),
    )


def validate_resource_path(path: str | Path, required_paths: Iterable[str | Path] = ()) -> ResourceValidation:
    """Validate an absolute cached resource file/folder without modifying it."""
    value = Path(path) if str(path) else Path("")
    exists = value.exists() and bool(str(path))
    is_file = value.is_file() if exists else False
    files: list[Path] = []
    if exists:
        files = [value] if is_file else [p for p in value.rglob("*") if p.is_file()]
    missing: list[str] = []
    for required in required_paths:
        target = Path(required)
        if not target.is_absolute() and exists and not is_file:
            target = value / target
        if not target.exists():
            missing.append(str(target))
    total = sum(p.stat().st_size for p in files if p.exists())
    return ResourceValidation(str(value), exists, is_file, len(files), total, tuple(missing), bool(exists and files and not missing))


_ID_KEYS = {"id", "material_id", "segment_id", "track_id", "request_id"}
_TIME_KEYS = {"start", "duration", "time_offset"}


def _category(path: str, key: str, missing: bool = False, empty: bool = False) -> str:
    p = path.casefold()
    k = key.casefold()
    if missing:
        return "missing_fields"
    if empty:
        return "generated_empty"
    if k in _ID_KEYS or any(token in p for token in ("/id", "uuid")):
        return "project_ids"
    if k in _TIME_KEYS or "timerange" in p or "timeline" in p:
        return "timeline"
    if k in {"path", "file_uri", "resource_id", "effect_id", "source_platform", "request_id"} or any(token in p for token in ("resource", "cache", "file_uri")):
        return "resource"
    return "effect_payload"


def recursive_effect_diff(working: Any, generated: Any, path: str = "") -> EffectDiff:
    changes: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, current: str, key: str = "") -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(set(left) | set(right)):
                child = f"{current}.{name}" if current else name
                if name not in left:
                    changes.append({"path": child, "kind": "missing_in_working", "working": None, "generated": right[name], "category": _category(child, name, missing=True)})
                elif name not in right:
                    changes.append({"path": child, "kind": "missing_in_generated", "working": left[name], "generated": None, "category": _category(child, name, missing=True)})
                else:
                    walk(left[name], right[name], child, name)
            return
        if isinstance(left, list) and isinstance(right, list):
            for index in range(max(len(left), len(right))):
                child = f"{current}[{index}]"
                if index >= len(left):
                    changes.append({"path": child, "kind": "missing_in_working", "working": None, "generated": right[index], "category": "missing_fields"})
                elif index >= len(right):
                    changes.append({"path": child, "kind": "missing_in_generated", "working": left[index], "generated": None, "category": "missing_fields"})
                else:
                    walk(left[index], right[index], child, key)
            return
        if left != right:
            empty = right in (None, "", [], {})
            category = _category(current, key, empty=empty)
            # Keep resource changes in the resource bucket while also exposing
            # the useful ``generated_empty`` aggregate for diagnostics.
            if empty and (key.casefold() in {"path", "file_uri", "resource_id", "effect_id"} or "resource" in current.casefold() or "cache" in current.casefold()):
                category = "resource"
            row = {"path": current, "kind": "value_changed", "working": left, "generated": right, "category": category}
            if empty:
                row["generated_empty"] = True
            changes.append(row)

    walk(working, generated, path)
    counts: dict[str, int] = {}
    for item in changes:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
        if item.get("generated_empty"):
            counts["generated_empty"] = counts.get("generated_empty", 0) + 1
    return EffectDiff(tuple(changes), counts)


def canonicalize_template(template: CapturedEffectTemplate) -> CapturedEffectTemplate:
    material = copy.deepcopy(template.material)
    segment = copy.deepcopy(template.segment)
    material["id"] = MATERIAL_ID_PLACEHOLDER
    segment["id"] = SEGMENT_ID_PLACEHOLDER
    segment["material_id"] = MATERIAL_ID_PLACEHOLDER
    target = segment.get("target_timerange")
    if isinstance(target, dict):
        target["start"] = START_US_PLACEHOLDER
        target["duration"] = DURATION_US_PLACEHOLDER
    return CapturedEffectTemplate(material, segment, copy.deepcopy(template.companion_records), template.resource_validation, template.source_draft, template.source_effect_name, template.source_effect_id, template.capture_version)


def clone_effect_template(template: CapturedEffectTemplate, start_us: int, duration_us: int, *, material_id: str | None = None, segment_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a fresh material/segment pair with cloned IDs and timerange."""
    if duration_us <= 0:
        raise ValueError("duration_us must be positive")
    material = copy.deepcopy(template.material)
    segment = copy.deepcopy(template.segment)
    new_material = material_id or str(uuid.uuid4()).upper()
    new_segment = segment_id or str(uuid.uuid4()).upper()
    old_material = str(material.get("id", ""))
    material["id"] = new_material
    segment["id"] = new_segment
    if str(segment.get("material_id", "")) in {old_material, MATERIAL_ID_PLACEHOLDER, ""}:
        segment["material_id"] = new_material
    target = segment.setdefault("target_timerange", {})
    target["start"] = int(start_us)
    target["duration"] = int(duration_us)
    return material, segment


def inject_effect_template(content: dict[str, Any], template: CapturedEffectTemplate, start_us: int, duration_us: int, *, track_name: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inject a cloned material and segment into an in-memory draft payload."""
    material, segment = clone_effect_template(template, start_us, duration_us)
    materials = content.setdefault("materials", {})
    materials.setdefault("video_effects", []).append(material)
    tracks = content.setdefault("tracks", [])
    track = next((item for item in tracks if isinstance(item, dict) and str(item.get("type", "")).casefold() == "effect" and (not track_name or item.get("name") == track_name)), None)
    if track is None:
        track = {"id": str(uuid.uuid4()).upper(), "type": "effect", "name": track_name or "Effects", "segments": [], "flag": 0, "attribute": 0, "is_default_name": False}
        tracks.append(track)
    track.setdefault("segments", []).append(segment)
    return material, segment


def save_template(template: CapturedEffectTemplate, path: str | Path) -> Path:
    destination = Path(path)
    _atomic_json(destination, template.to_dict())
    return destination


def load_template(path: str | Path) -> CapturedEffectTemplate:
    return CapturedEffectTemplate.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_template_path(effect: str = "warning") -> Path:
    """Return the user-local canonical template location."""
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "AutoCapCut" / "effect_catalog" / "captured_templates" / effect / "captured_effect_template.json"


def load_captured_template(path: str | Path | None = None, *, effect: str = "warning") -> CapturedEffectTemplate | None:
    candidate = Path(path) if path else default_template_path(effect)
    try:
        return load_template(candidate)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def is_warning_entry(entry: Any) -> bool:
    # Local Warning is the only catalog item allowed through the captured
    # template path.  A pyCapCut enum with the same name must use its normal
    # serializer unless it is explicitly marked as a local discovery row.
    return str(getattr(entry, "source", "")).casefold() == "local" and (
        str(getattr(entry, "effect_id", "")) == "7399465244088618245" or
        str(getattr(entry, "display_name", "")).strip().casefold() == "warning"
    )


def inject_captured_effect_file(
    draft_path: str | Path,
    template: CapturedEffectTemplate,
    start_us: int,
    duration_us: int,
    *,
    track_name: str | None = "Catalog Effects",
) -> None:
    """Inject a cloned captured record into an already-saved draft folder."""
    path = _draft_content_path(draft_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    inject_effect_template(data, template, start_us, duration_us, track_name=track_name)
    _atomic_json(path, data)


def catalog_template_path(root: Path | None = None, *, effect: str = "warning") -> Path:
    if root is None:
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        root = base / "AutoCapCut" / "effect_catalog"
    return Path(root) / "captured_templates" / effect / "captured_effect_template.json"


def load_captured_template(path: Path | None = None, *, effect: str = "warning") -> CapturedEffectTemplate | None:
    candidate = Path(path) if path else catalog_template_path(effect=effect)
    try:
        return load_template(candidate)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_captured_template(template: CapturedEffectTemplate, path: Path | None = None, *, effect: str = "warning") -> Path:
    return save_template(template, path or catalog_template_path(effect=effect))


def inject_captured_effect(draft_path: str | Path, template: CapturedEffectTemplate, *, start_us: int, duration_us: int, track_name: str = "Catalog Effects") -> None:
    path = _draft_content_path(draft_path)
    content = json.loads(path.read_text(encoding="utf-8"))
    inject_effect_template(content, template, start_us, duration_us, track_name=track_name)
    _atomic_json(path, content)


def capture_warning_template(working_draft: str | Path, generated_draft: str | Path, output_dir: str | Path, *, effect_id: str = "7399465244088618245") -> CapturedEffectTemplate:
    """Capture Warning snapshots/diff and write a reusable canonical template."""
    out = Path(output_dir)
    working = extract_effect_template(working_draft, effect_id=effect_id)
    generated = extract_effect_template(generated_draft, effect_id=effect_id)
    _atomic_json(out / "working_warning_segment.json", working.segment)
    _atomic_json(out / "working_warning_material.json", working.material)
    _atomic_json(out / "generated_warning_segment.json", generated.segment)
    _atomic_json(out / "generated_warning_material.json", generated.material)
    diff = recursive_effect_diff({"segment": working.segment, "material": working.material}, {"segment": generated.segment, "material": generated.material})
    _atomic_json(out / "warning_effect_diff.json", diff.to_dict())
    lines = ["Warning effect diff", "", *[f"[{item['category']}] {item['path']}: {item['kind']}" for item in diff.changes]]
    (out / "warning_effect_diff.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    canonical = canonicalize_template(working)
    save_template(canonical, out / "captured_effect_template.json")
    return canonical


# Friendly aliases for callers that use the service terminology.
extract_captured_effect = extract_effect_template
clone_captured_effect = clone_effect_template
diff_effect_templates = recursive_effect_diff
