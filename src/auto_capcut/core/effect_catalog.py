"""Effect catalog persistence and discovery for the optional tester tool.

This module is deliberately independent from the production project builder.
It only inspects the effect enums shipped by the installed pyCapCut package and
stores user review metadata in a small, mergeable JSON document.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ReviewStatus(str, Enum):
    UNTESTED = "Untested"
    CAPCUT_OK = "CapCut OK"
    CAPCUT_MISSING = "CapCut Missing"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    BROKEN = "Broken"


class EffectLifecycle(str, Enum):
    """Discovery/build lifecycle for catalog entries.

    ``buildable`` remains as a compatibility property for older callers;
    lifecycle is the richer state used by captured local templates.
    """
    DISCOVERED = "discovered"
    METADATA_ONLY = "metadata_only"
    TEMPLATE_CAPTURED = "template_captured"
    PROMOTED = "promoted"
    DRAFT_RECOGNIZED = "draft_recognized"
    RENDER_UNVERIFIED = "render_unverified"
    RENDER_CONFIRMED = "render_confirmed"
    UNRESOLVED = "unresolved"


@dataclass
class EffectCatalogEntry:
    """Serializable metadata for one real pyCapCut effect enum member."""

    stable_key: str
    test_index: int
    display_name: str
    enum_name: str
    effect_id: str
    resource_id: str
    source: str  # ``scene`` or ``character``
    is_vip: bool = False
    md5: str = ""
    params: list[dict[str, Any]] = field(default_factory=list)
    supported: bool = True
    installed: bool = True
    build_status: str = "unbuilt"
    review_status: str = ReviewStatus.UNTESTED.value
    favorite: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    error: str | None = None
    validation_state: str = "discovered"
    buildable: bool = False
    local_resource_path: str = ""
    source_file: str = ""
    source_type: str = ""
    raw_metadata_subset: dict[str, Any] = field(default_factory=dict)
    pycapcut_match: bool = False

    @property
    def key(self) -> str:
        """Compatibility alias for the stable catalog identifier."""
        return self.stable_key

    @property
    @property
    def internal_type(self) -> str:
        return self.enum_name

    @property
    def tested(self) -> bool:
        return self.build_status in {"build_ok", "build_failed"} or self.review_status != ReviewStatus.UNTESTED.value

    @property
    def status(self) -> str:
        return self.review_status

    @property
    def lifecycle_state(self) -> str:
        """Canonical lifecycle alias (old catalogs use validation_state)."""
        return self.validation_state or EffectLifecycle.DISCOVERED.value

    @lifecycle_state.setter
    def lifecycle_state(self, value: str) -> None:
        self.validation_state = str(value or EffectLifecycle.DISCOVERED.value)

    @property
    def category(self) -> str:
        """UI-friendly category from captured metadata or source."""
        return str(self.raw_metadata_subset.get("category_name") or self.source.capitalize())

    @classmethod
    def from_member(cls, member: Any, source: str, test_index: int = 0) -> "EffectCatalogEntry":
        """Construct metadata for a pyCapCut enum member without probing it."""
        meta = getattr(member, "value", None)
        effect_id = str(getattr(meta, "effect_id", "") or "")
        source = str(source).casefold()
        return cls(
            stable_key=f"{source}:{effect_id}", test_index=test_index,
            display_name=str(getattr(meta, "name", getattr(member, "name", effect_id))),
            enum_name=str(getattr(member, "name", "")), effect_id=effect_id,
            resource_id=str(getattr(meta, "resource_id", effect_id) or effect_id), source=source,
            is_vip=bool(getattr(meta, "is_vip", False)), md5=str(getattr(meta, "md5", "") or ""),
            params=[_param_dict(param) for param in (getattr(meta, "params", None) or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        value["params"] = [dict(p) for p in self.params]
        return value

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EffectCatalogEntry":
        """Read current and early catalog schemas leniently."""
        source = str(raw.get("source", raw.get("source_type", "scene"))).lower()
        if source in {"videosceneeffecttype", "video_scene", "scene_effect"}:
            source = "scene"
        elif source in {"videocharactereffecttype", "video_character", "character_effect"}:
            source = "character"
        effect_id = str(raw.get("effect_id", raw.get("id", "")))
        enum_name = str(raw.get("enum_name", raw.get("enum", raw.get("name", ""))))
        stable_key = str(raw.get("stable_key") or f"{source}:{effect_id}")
        status = raw.get("review_status", ReviewStatus.UNTESTED.value)
        if isinstance(status, ReviewStatus):
            status = status.value
        # Unknown statuses should not make the whole catalog unreadable.
        if status not in {item.value for item in ReviewStatus}:
            status = ReviewStatus.UNTESTED.value
        return cls(
            stable_key=stable_key,
            test_index=int(raw.get("test_index", raw.get("index", 0)) or 0),
            display_name=str(raw.get("display_name", raw.get("name", enum_name))),
            enum_name=enum_name,
            effect_id=effect_id,
            resource_id=str(raw.get("resource_id", effect_id)),
            source=source,
            is_vip=bool(raw.get("is_vip", raw.get("vip", False))),
            md5=str(raw.get("md5", "")),
            params=[dict(p) for p in (raw.get("params") or []) if isinstance(p, dict)],
            supported=bool(raw.get("supported", True)),
            installed=bool(raw.get("installed", True)),
            build_status=str(raw.get("build_status", "unbuilt")),
            review_status=str(status),
            favorite=bool(raw.get("favorite", False)),
            notes=str(raw.get("notes", "")),
            tags=[str(tag) for tag in (raw.get("tags") or [])],
            error=(str(raw["error"]) if raw.get("error") else None),
            validation_state=str(raw.get("validation_state", "discovered")),
            buildable=bool(raw.get("buildable", False)),
            local_resource_path=str(raw.get("local_resource_path", "")),
            source_file=str(raw.get("source_file", "")),
            source_type=str(raw.get("source_type", "")),
            raw_metadata_subset=dict(raw.get("raw_metadata_subset") or {}),
            pycapcut_match=bool(raw.get("pycapcut_match", False)),
        )


def normalize_effect_name(value: str) -> str:
    """Normalize human effect names and SRT preset slugs for exact matching."""
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s_-]+", " ", value).strip()


@dataclass(frozen=True)
class PresetCatalogMatch:
    preset_key: str
    normalized_name: str
    candidates: tuple[EffectCatalogEntry, ...] = ()
    selected_local: EffectCatalogEntry | None = None
    status: str = "not_found"

    @property
    def unique(self) -> bool:
        return self.status == "unique" and self.selected_local is not None

    @property
    def reason(self) -> str:
        if self.status == "not_found":
            return "No exact normalized catalog display-name match"
        if self.status == "ambiguous":
            return "Multiple local effects match; choose one"
        if self.status == "not_promotable":
            return "Match exists but no CapCut Local resource entry is available"
        return ""


@dataclass
class EffectCatalog:
    entries: list[EffectCatalogEntry] = field(default_factory=list)
    schema_version: int = 1

    def by_key(self) -> dict[str, EffectCatalogEntry]:
        return {entry.stable_key: entry for entry in self.entries}


def catalog_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "AutoCapCut" / "effect_catalog"


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _param_dict(param: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(param, "name", "")),
        "default_value": getattr(param, "default_value", None),
        "min_value": getattr(param, "min_value", None),
        "max_value": getattr(param, "max_value", None),
    }


def scan_effect_types() -> list[EffectCatalogEntry]:
    """Enumerate only effects exposed by the installed pyCapCut package.

    Instantiation/export is used as a lightweight capability probe. A failed
    probe remains in the catalog with ``supported=False`` so the UI can show
    the real failure instead of inventing a missing effect.
    """
    try:
        import pycapcut as cc
    except Exception:
        return []
    entries: list[EffectCatalogEntry] = []
    index = 1
    for source, enum_name in (("scene", "VideoSceneEffectType"), ("character", "VideoCharacterEffectType")):
        enum_type = getattr(cc, enum_name, None)
        if enum_type is None:
            continue
        for member in enum_type:
            meta = getattr(member, "value", None)
            effect_id = str(getattr(meta, "effect_id", "") or "")
            if not effect_id:
                # pyCapCut's enum members are the source of truth; skip only
                # malformed package metadata rather than fabricating an ID.
                continue
            error: str | None = None
            supported = True
            try:
                segment = cc.EffectSegment(member, cc.Timerange(0, 1_000_000))
                segment.export_json()
            except Exception as exc:  # capability failures are catalog data
                supported = False
                error = f"{type(exc).__name__}: {exc}"[:500]
            key = f"{source}:{effect_id}"
            entries.append(
                EffectCatalogEntry(
                    stable_key=key,
                    test_index=index,
                    display_name=str(getattr(meta, "name", member.name)),
                    enum_name=str(member.name),
                    effect_id=effect_id,
                    resource_id=str(getattr(meta, "resource_id", effect_id) or effect_id),
                    source=source,
                    is_vip=bool(getattr(meta, "is_vip", False)),
                    md5=str(getattr(meta, "md5", "") or ""),
                    params=[_param_dict(param) for param in (getattr(meta, "params", None) or [])],
                    supported=supported,
                    error=error,
                )
            )
            index += 1
    return entries


def resolve_effect_member(entry: EffectCatalogEntry):
    """Resolve a catalog entry back to the exact installed enum member.

    Catalog files intentionally persist metadata, not Python objects.  This
    helper is used by the tester builder and returns ``None`` when the package
    no longer exposes the recorded member.
    """
    try:
        import pycapcut as cc
        enum_type = getattr(cc, "VideoSceneEffectType" if entry.source == "scene" else "VideoCharacterEffectType")
        member = getattr(enum_type, entry.enum_name, None)
        if member is not None and str(getattr(getattr(member, "value", None), "effect_id", "")) == str(entry.effect_id):
            return member
    except Exception:
        pass
    return None


class CatalogStore:
    """Load, atomically save, merge scans and export approved effects."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else catalog_root()
        self.path = self.root / "effect_catalog.json"

    def load(self) -> EffectCatalog:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return EffectCatalog()
        rows = raw.get("entries", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            rows = []
        entries: list[EffectCatalogEntry] = []
        for item in rows:
            if isinstance(item, dict):
                try:
                    entries.append(EffectCatalogEntry.from_dict(item))
                except (TypeError, ValueError):
                    continue
        entries.sort(key=lambda e: (e.test_index or 10**9, e.stable_key))
        return EffectCatalog(entries, int(raw.get("schema_version", 1)) if isinstance(raw, dict) else 1)

    def save(self, catalog: EffectCatalog | Iterable[EffectCatalogEntry]) -> EffectCatalog:
        if isinstance(catalog, EffectCatalog):
            value = catalog
        else:
            value = EffectCatalog(list(catalog))
        value.entries.sort(key=lambda e: (e.test_index, e.stable_key))
        _atomic_json_write(self.path, {"schema_version": value.schema_version, "entries": [e.to_dict() for e in value.entries]})
        return value

    def merge_scan(self, discovered: Iterable[EffectCatalogEntry] | None = None, *, mark_missing: bool = True) -> EffectCatalog:
        discovered_list = list(discovered if discovered is not None else scan_effect_types())
        current = self.load()
        old = current.by_key()
        max_index = max((entry.test_index for entry in current.entries), default=0)
        merged: list[EffectCatalogEntry] = []
        seen: set[str] = set()
        for fresh in discovered_list:
            previous = old.get(fresh.stable_key)
            if previous is None:
                max_index += 1
                fresh.test_index = max_index
                merged.append(fresh)
            else:
                # Metadata is refreshed while all user-owned review fields stay.
                fresh.test_index = previous.test_index
                fresh.review_status = previous.review_status
                fresh.build_status = previous.build_status
                fresh.validation_state = previous.validation_state
                fresh.buildable = previous.buildable
                fresh.error = previous.error
                fresh.favorite = previous.favorite
                fresh.notes = previous.notes
                fresh.tags = list(previous.tags)
                merged.append(fresh)
            seen.add(fresh.stable_key)
        for previous in current.entries:
            if previous.stable_key not in seen:
                if mark_missing:
                    previous.installed = False
                merged.append(previous)
        return self.save(EffectCatalog(merged, current.schema_version))

    def scan(self) -> EffectCatalog:
        return self.merge_scan(scan_effect_types())

    def scan_local(self) -> EffectCatalog:
        from auto_capcut.core.local_effect_scanner import CapCutLocalEffectScanner
        scanner = CapCutLocalEffectScanner()
        local_entries = scanner.scan()
        converted = [
            EffectCatalogEntry(
                stable_key=item.stable_key, test_index=0,
                display_name=item.display_name, enum_name="", effect_id=item.effect_id,
                resource_id=item.resource_id, source="local", supported=item.buildable,
                installed=True, build_status="unbuilt", validation_state=item.validation_state,
                buildable=item.buildable, local_resource_path=str(item.local_resource_path or ""),
                source_file=str(item.source_file or ""), source_type=item.source_type,
                raw_metadata_subset=dict(item.raw_metadata_subset), pycapcut_match=item.pycapcut_match,
            )
            for item in local_entries
        ]
        result = self.merge_scan(converted, mark_missing=False)
        scanner.write_report(local_entries)
        return result

    def scan_all(self) -> EffectCatalog:
        py_entries = scan_effect_types()
        from auto_capcut.core.local_effect_scanner import CapCutLocalEffectScanner
        scanner = CapCutLocalEffectScanner()
        local_entries = scanner.scan()
        converted = [EffectCatalogEntry(
            stable_key=item.stable_key, test_index=0, display_name=item.display_name,
            enum_name="", effect_id=item.effect_id, resource_id=item.resource_id, source="local",
            supported=item.buildable, installed=True, validation_state=item.validation_state,
            buildable=item.buildable, local_resource_path=str(item.local_resource_path or ""),
            source_file=str(item.source_file or ""), source_type=item.source_type,
            raw_metadata_subset=dict(item.raw_metadata_subset), pycapcut_match=item.pycapcut_match,
        ) for item in local_entries]
        result = self.merge_scan(py_entries + converted)
        scanner.write_report(local_entries)
        return result

    def update(self, stable_key: str, **changes: Any) -> EffectCatalogEntry:
        catalog = self.load()
        entry = next((item for item in catalog.entries if item.stable_key == stable_key), None)
        if entry is None:
            raise KeyError(stable_key)
        for name in ("review_status", "favorite", "notes", "tags", "build_status", "validation_state", "buildable"):
            if name in changes:
                setattr(entry, name, changes[name])
        self.save(catalog)
        return entry

    def capture_warning_template(
        self,
        working_draft: Path | str | None = None,
        generated_draft: Path | str | None = None,
    ):
        """Capture the known-good Warning material into the catalog cache.

        Defaults point at the local ``test_8`` and ``EffectCatalog_1`` drafts,
        but callers may provide explicit paths for tests or another CapCut
        profile.  The draft files are opened read-only by the capture service.
        """
        from auto_capcut.core.captured_effect_template import capture_warning_template
        capcut_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "CapCut/User Data/Projects/com.lveditor.draft"
        working = Path(working_draft) if working_draft else capcut_root / "test_8"
        generated = Path(generated_draft) if generated_draft else capcut_root / "EffectCatalog_1"
        output = self.root / "captured_templates" / "warning"
        template = capture_warning_template(working, generated, output)
        catalog = self.load()
        entry = next((item for item in catalog.entries if str(item.effect_id) == template.source_effect_id and item.source == "local"), None)
        if entry is not None:
            entry.validation_state = "template_captured"
            entry.buildable = True
            entry.error = None
            self.save(catalog)
        return output / "captured_effect_template.json"

    def mark_lifecycle(self, stable_key: str, state: str) -> EffectCatalogEntry:
        """Record a human verification state without claiming render success."""
        allowed = {"discovered", "metadata_only", "template_captured", "promoted", "draft_recognized", "render_unverified", "render_confirmed", "unresolved"}
        if state not in allowed:
            raise ValueError(f"Unknown effect lifecycle state: {state}")
        catalog = self.load()
        entry = next((item for item in catalog.entries if item.stable_key == stable_key), None)
        if entry is None:
            raise KeyError(stable_key)
        entry.validation_state = state
        if state in {"template_captured", "promoted", "draft_recognized", "render_unverified", "render_confirmed"}:
            entry.buildable = True
        elif state == "unresolved":
            entry.buildable = False
        self.save(catalog)
        return entry

    def filter_entries(
        self,
        query: str = "",
        *,
        source: str | None = None,
        review_status: str | ReviewStatus | None = None,
        build_status: str | None = None,
        installed_only: bool = False,
    ) -> list[EffectCatalogEntry]:
        """Return deterministic UI-ready catalog rows matching simple filters."""
        needle = normalize_effect_name(query)
        source_value = source.casefold() if source else None
        status_value = review_status.value if isinstance(review_status, ReviewStatus) else review_status
        rows = []
        for entry in self.load().entries:
            haystack = " ".join(normalize_effect_name(value) for value in (entry.display_name, entry.enum_name, entry.source, entry.effect_id))
            if needle and needle not in haystack:
                continue
            if source_value and entry.source.casefold() != source_value:
                continue
            if status_value and entry.review_status != status_value:
                continue
            if build_status and entry.build_status != build_status:
                continue
            if installed_only and not entry.installed:
                continue
            rows.append(entry)
        return rows

    def match_preset_keys(self, preset_keys: Iterable[str]) -> list[PresetCatalogMatch]:
        """Find exact display-name matches for production preset slugs."""
        entries = [entry for entry in self.load().entries if entry.installed]
        output: list[PresetCatalogMatch] = []
        for preset_key in dict.fromkeys(str(value).strip() for value in preset_keys if str(value).strip()):
            normalized = normalize_effect_name(preset_key)
            matches = tuple(entry for entry in entries if normalize_effect_name(entry.display_name) == normalized)
            local = tuple(entry for entry in matches if entry.source.casefold() == "local")
            if len(local) == 1:
                output.append(PresetCatalogMatch(preset_key, normalized, local, local[0], "unique"))
            elif len(local) > 1:
                output.append(PresetCatalogMatch(preset_key, normalized, local, None, "ambiguous"))
            elif matches:
                linked = tuple(
                    entry for entry in entries
                    if entry.source.casefold() == "local"
                    and any(
                        str(entry.effect_id) == str(match.effect_id)
                        or str(entry.resource_id) == str(match.resource_id)
                        for match in matches
                    )
                )
                if len(linked) == 1:
                    output.append(PresetCatalogMatch(preset_key, normalized, linked, linked[0], "unique"))
                elif len(linked) > 1:
                    output.append(PresetCatalogMatch(preset_key, normalized, linked, None, "ambiguous"))
                else:
                    output.append(PresetCatalogMatch(preset_key, normalized, matches, None, "not_promotable"))
            else:
                output.append(PresetCatalogMatch(preset_key, normalized, (), None, "not_found"))
        return output

    def export_approved(self, output_path: Path | None = None) -> Path:
        output = Path(output_path) if output_path is not None else self.root / "approved_effects.json"
        approved = [entry for entry in self.load().entries if entry.review_status == ReviewStatus.APPROVED.value and entry.installed]
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in approved:
            tags = [tag.strip() for tag in entry.tags if str(tag).strip()] or ["uncategorized"]
            for tag in tags:
                groups.setdefault(tag, []).append(entry.to_dict())
        for rows in groups.values():
            rows.sort(key=lambda row: (row["test_index"], row["stable_key"]))
        _atomic_json_write(output, {"schema_version": 1, "groups": groups})
        return output


# Functional aliases make the service convenient for workers and tests.
def load_catalog(path: Path | None = None) -> EffectCatalog:
    return CatalogStore(path.parent if path and path.suffix else path).load() if path else CatalogStore().load()


def load_effect_catalog(path: Path | None = None) -> EffectCatalog:
    return load_catalog(path)


def save_effect_catalog(catalog: EffectCatalog, path: Path | None = None) -> EffectCatalog:
    store = CatalogStore(path.parent if path and path.suffix else path)
    return store.save(catalog)


def scan_effect_catalog(root: Path | None = None) -> EffectCatalog:
    return CatalogStore(root).scan()


# Short aliases used by workers and integrations.
scan_effects = scan_effect_types


def export_approved_effects(output_path: Path | None = None, root: Path | None = None) -> Path:
    return CatalogStore(root).export_approved(output_path)
