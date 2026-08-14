"""Read-only discovery of CapCut's locally cached video-effect catalog."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class LocalCapCutEffectEntry:
    stable_key: str
    display_name: str
    effect_id: str
    resource_id: str
    category: str = ""
    local_resource_path: Path | None = None
    source_file: Path | None = None
    source_type: str = "sqlite_http_cache"
    raw_metadata_subset: dict[str, Any] = field(default_factory=dict)
    validation_state: str = "discovered"
    buildable: bool = False
    pycapcut_match: bool = False
    error: str = ""


def _capcut_user_data() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "CapCut" / "User Data"


def default_anchor_path() -> Path:
    return _capcut_user_data() / "Projects" / "com.lveditor.draft" / "test_8"


def _safe_subset(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "id", "effect_id", "resource_id", "title", "name", "category_id",
        "category_name", "effect_type", "source", "resource_source", "sub_type",
        "md5", "type", "platform", "source_platform", "item_effect_type",
        "path", "file_uri", "request_id", "item_urls", "tags", "tag_list",
        "sdk_extra", "extra", "category_ids",
    }
    return {key: value[key] for key in allowed if key in value and key not in {"request_id", "item_urls"}}


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return ""


def extract_anchor(anchor_path: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """Extract the effect material from test_8 without modifying it."""
    root = Path(anchor_path or default_anchor_path())
    content_path = root / "draft_content.json"
    if not content_path.is_file():
        return {}, [f"Anchor project not found: {root}"]
    try:
        data = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Unable to read anchor draft: {exc}"]
    materials = data.get("materials", {}).get("video_effects", [])
    effects = [item for item in materials if isinstance(item, dict)]
    if len(effects) != 1:
        return {}, [f"Expected exactly one Video Effect in anchor, found {len(effects)}"]
    anchor = dict(effects[0])
    anchor["anchor_project"] = str(root)
    anchor["material_references"] = [
        segment.get("material_id")
        for track in data.get("tracks", []) if track.get("type") == "effect"
        for segment in track.get("segments", []) if isinstance(segment, dict)
    ]
    return anchor, []


def _pycapcut_ids() -> set[str]:
    try:
        import pycapcut as cc
        return {
            str(member.value.effect_id)
            for enum_type in (cc.VideoSceneEffectType, cc.VideoCharacterEffectType)
            for member in enum_type
            if getattr(member.value, "effect_id", None)
        }
    except Exception:
        return set()


def _walk_common_attr(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        common = value.get("common_attr")
        if isinstance(common, dict):
            yield common
        for child in value.values():
            yield from _walk_common_attr(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_common_attr(child)


class CapCutLocalEffectScanner:
    def __init__(self, user_data: Path | None = None, anchor_path: Path | None = None):
        self.user_data = Path(user_data) if user_data else _capcut_user_data()
        self.anchor_path = Path(anchor_path) if anchor_path else default_anchor_path()
        self.anchor, self.warnings = extract_anchor(self.anchor_path)

    def scan(self) -> list[LocalCapCutEffectEntry]:
        records: dict[tuple[str, str, str], LocalCapCutEffectEntry] = {}
        anchor_id = str(self.anchor.get("effect_id", ""))
        anchor_title = str(self.anchor.get("name", ""))
        anchor_category = str(self.anchor.get("category_name", ""))
        if anchor_id:
            records[(anchor_id, anchor_id, anchor_title.casefold())] = self._entry(
                self.anchor, self.anchor_path / "draft_content.json", "anchor_draft", anchor_id, anchor_category,
            )
        db_root = self.user_data / "Cache" / "ressdk_db"
        for db_path in db_root.rglob("rp.db") if db_root.is_dir() else ():
            self._scan_db(db_path, records)
        return sorted(records.values(), key=lambda item: (item.display_name.casefold(), item.effect_id))

    def _scan_db(self, db_path: Path, records: dict[tuple[str, str, str], LocalCapCutEffectEntry]) -> None:
        try:
            con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
            tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
            if "http_cache" not in tables:
                return
            for url, body in con.execute("select url,response_body from http_cache"):
                if not isinstance(body, str):
                    continue
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for common in _walk_common_attr(payload):
                    if str(common.get("effect_type", "")) != "7":
                        continue
                    effect_id = str(common.get("effect_id") or common.get("id") or "")
                    resource_id = str(common.get("resource_id") or effect_id)
                    if not effect_id:
                        continue
                    title = str(common.get("title") or common.get("name") or "").strip()
                    key = (resource_id, "", "")
                    records.setdefault(key, self._entry(common, db_path, "sqlite_http_cache", resource_id, str(common.get("category_name") or ""), url))
        except (OSError, sqlite3.Error) as exc:
            self.warnings.append(f"Unable to inspect {db_path}: {exc}")
        finally:
            try: con.close()
            except Exception: pass

    def _entry(self, raw: dict[str, Any], source_file: Path | None, source_type: str, resource_id: str, category: str, url: str = "") -> LocalCapCutEffectEntry:
        effect_id = str(raw.get("effect_id") or raw.get("id") or resource_id)
        path = self.user_data / "Cache" / "effect" / resource_id
        valid = bool(resource_id and effect_id and str(raw.get("effect_type", "7")) == "7")
        has_resource = path.is_dir()
        matched = effect_id in _pycapcut_ids()
        state = "validated" if valid else "discovered"
        buildable = bool(matched)
        if buildable: state = "buildable"
        return LocalCapCutEffectEntry(
            stable_key=f"local:{resource_id or effect_id}",
            display_name=str(raw.get("name") or raw.get("title") or effect_id),
            effect_id=effect_id, resource_id=resource_id, category=category,
            local_resource_path=path if has_resource else None, source_file=source_file,
            source_type=source_type, raw_metadata_subset={**_safe_subset(raw), **({"url": _redact_url(url)} if url else {})},
            validation_state=state, buildable=buildable, pycapcut_match=matched,
        )

    def write_report(self, entries: list[LocalCapCutEffectEntry], path: Path | None = None) -> Path:
        output = path or (self.user_data.parent / "AutoCapCut" / "effect_catalog" / "capcut_effect_discovery_report.txt")
        output.parent.mkdir(parents=True, exist_ok=True)
        anchor_json = output.parent / "warning_anchor.json"
        anchor_json.write_text(json.dumps(_safe_subset(self.anchor), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        anchor_id = self.anchor.get("effect_id", "")
        matched = sum(entry.pycapcut_match for entry in entries)
        sources = sorted({str(item.source_file) for item in entries if item.source_file})
        lines = ["CapCut Effect Discovery Report", "", f"Warning anchor found: {'yes' if anchor_id else 'no'}", f"Anchor display_name: {self.anchor.get('name', '')}", f"Anchor effect_id: {anchor_id}", f"Anchor resource_id: {self.anchor.get('resource_id', '')}", f"Anchor local_path: {self.anchor.get('path', '')}", "", "Catalog sources:"] + [f"- {source}" for source in sources[:200]] + ["", f"Local effects discovered: {len(entries)}", f"Matched existing pyCapCut effects: {matched}", f"New effects not in pyCapCut: {len(entries) - matched}", f"Validated/buildable: {sum(entry.buildable for entry in entries)}", "", "Warnings:"] + [f"- {warning}" for warning in self.warnings]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output


def scan_capcut_local_effects(user_data: Path | None = None, anchor_path: Path | None = None) -> list[LocalCapCutEffectEntry]:
    return CapCutLocalEffectScanner(user_data, anchor_path).scan()
