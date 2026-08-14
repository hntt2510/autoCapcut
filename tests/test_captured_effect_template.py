from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path

from auto_capcut.core.captured_effect_template import (
    CapturedEffectTemplateRepository,
    CapturedEffectTemplate,
    canonicalize_template,
    clone_effect_template,
    extract_effect_template,
    inject_effect_template,
    recursive_effect_diff,
    validate_resource_path,
    save_template,
)
from auto_capcut.core.local_effect_promoter import LocalEffectTemplatePromoter


def _draft(path: Path, *, generated: bool = False) -> None:
    material_id = "generated-material" if generated else "working-material"
    segment_id = "generated-segment" if generated else "working-segment"
    material = {
        "id": material_id,
        "effect_id": "7399465244088618245",
        "resource_id": "7399465244088618245",
        "name": "Warning",
        "path": "##_material_placeholder_##" if generated else str(path.parent / "resource"),
        "category_id": "" if generated else "1111",
        "category_name": "" if generated else "Video effects",
        "source_platform": 0 if generated else 1,
        "request_id": "" if generated else "request",
        "nested_unknown": {"keep": True},
    }
    segment = {
        "id": segment_id,
        "material_id": material_id,
        "target_timerange": {"start": 2_000_000 if generated else 6_233_333, "duration": 2_000_000 if generated else 966_667},
    }
    content = {
        "materials": {"video_effects": [material]},
        "tracks": [{"id": "track", "type": "effect", "segments": [segment]}],
    }
    (path / "draft_content.json").write_text(json.dumps(content), encoding="utf-8")


def test_extract_follows_material_reference_and_validates_resources(tmp_path: Path) -> None:
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "effect.bin").write_bytes(b"x")
    draft = tmp_path / "test_8"
    draft.mkdir()
    _draft(draft)
    template = extract_effect_template(draft, effect_id="7399465244088618245")
    assert template.material["id"] == "working-material"
    assert template.segment["material_id"] == template.material["id"]
    assert template.resource_validation is not None and template.resource_validation.valid
    assert template.resource_validation.file_count == 1


def test_diff_classifies_empty_resource_and_ids() -> None:
    left = {"id": "a", "path": "C:/cache/effect", "category_id": "1111"}
    right = {"id": "b", "path": "", "category_id": ""}
    diff = recursive_effect_diff(left, right)
    assert diff.categories["project_ids"] == 1
    assert diff.categories["resource"] == 1
    assert diff.categories["generated_empty"] >= 1


def test_canonical_clone_replaces_ids_and_time_but_preserves_unknown_fields() -> None:
    template = CapturedEffectTemplate(
        {"id": "old", "effect_id": "e", "nested": {"x": 1}},
        {"id": "segment", "material_id": "old", "target_timerange": {"start": 1, "duration": 2}},
    )
    canonical = canonicalize_template(template)
    assert canonical.material["id"] == "__MATERIAL_ID__"
    assert canonical.segment["material_id"] == "__MATERIAL_ID__"
    material, segment = clone_effect_template(canonical, 2_000_000, 2_000_000, material_id="m2", segment_id="s2")
    assert material["id"] == "m2" and segment["id"] == "s2"
    assert segment["material_id"] == "m2"
    assert segment["target_timerange"] == {"start": 2_000_000, "duration": 2_000_000}
    assert material["nested"] == {"x": 1}
    assert "##_material_placeholder" not in json.dumps(material)


def test_inject_adds_material_and_effect_track_without_mutating_template() -> None:
    template = CapturedEffectTemplate({"id": "m", "effect_id": "e"}, {"id": "s", "material_id": "m", "target_timerange": {}})
    original = copy.deepcopy(template.to_dict())
    content = {"materials": {"video_effects": []}, "tracks": []}
    material, segment = inject_effect_template(content, template, 2_000_000, 2_000_000, track_name="Catalog Effects")
    assert content["materials"]["video_effects"] == [material]
    assert content["tracks"][0]["segments"] == [segment]
    assert template.to_dict() == original


def test_validate_resource_reports_missing_required_file(tmp_path: Path) -> None:
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "present.bin").write_bytes(b"ok")
    result = validate_resource_path(resource, ["present.bin", "missing.bin"])
    assert not result.valid
    assert result.missing_paths == (str(resource / "missing.bin"),)


def test_warning_preset_resolves_by_stable_effect_id_and_resource(tmp_path: Path) -> None:
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "effect.bin").write_bytes(b"ok")
    template = CapturedEffectTemplate(
        {"id": "m", "effect_id": "7399465244088618245", "resource_id": "7399465244088618245", "path": str(resource)},
        {"id": "s", "material_id": "m", "target_timerange": {}},
        source_effect_name="Renamed display metadata",
        source_effect_id="7399465244088618245",
    )
    repository = CapturedEffectTemplateRepository(tmp_path)
    repository.register_template("warning", template, stable_key="local:7399465244088618245", display_name="Warning", state="render_confirmed")
    resolved = repository.resolve_effect_preset("WARNING")
    assert resolved is not None
    assert resolved.effect_id == "7399465244088618245"
    assert CapturedEffectTemplateRepository(tmp_path).resolve_effect_preset("not-real") is None


def test_preset_registry_requires_draft_recognized_before_render_confirmed(tmp_path: Path) -> None:
    resource = tmp_path / "resource"; resource.mkdir(); (resource / "effect.bin").write_bytes(b"ok")
    template = CapturedEffectTemplate(
        {"id": "m", "effect_id": "e", "resource_id": "r", "path": str(resource)},
        {"id": "s", "material_id": "m", "target_timerange": {}},
        source_effect_id="e",
    )
    repository = CapturedEffectTemplateRepository(tmp_path)
    repository.register_template("serious_error", template, stable_key="local:e", display_name="Serious Error")
    assert repository.resolve_effect_preset("serious_error") is None
    repository.mark_state("serious_error", "draft_recognized")
    assert repository.resolve_effect_preset("serious_error") is None
    repository.mark_state("serious_error", "render_confirmed")
    assert repository.resolve_effect_preset("serious_error") is not None


def test_preset_registry_rejects_resource_fingerprint_changes(tmp_path: Path) -> None:
    resource = tmp_path / "resource"; resource.mkdir(); (resource / "effect.bin").write_bytes(b"ok")
    template = CapturedEffectTemplate(
        {"id": "m", "effect_id": "e", "resource_id": "r", "path": str(resource)},
        {"id": "s", "material_id": "m", "target_timerange": {}}, source_effect_id="e",
    )
    repository = CapturedEffectTemplateRepository(tmp_path)
    repository.register_template("changed", template, stable_key="local:e", display_name="Changed", state="render_confirmed")
    (resource / "changed.bin").write_bytes(b"changed")
    assert repository.resolve_effect_preset("changed") is None


def test_legacy_captured_registry_state_migrates_to_promoted(tmp_path: Path) -> None:
    repository = CapturedEffectTemplateRepository(tmp_path)
    repository.registry_path.parent.mkdir(parents=True, exist_ok=True)
    repository.registry_path.write_text(json.dumps({"schema_version": 1, "presets": {
        "legacy": {"preset_key": "legacy", "stable_key": "local:e", "state": "captured"}
    }}), encoding="utf-8")
    assert repository.load_registry()["legacy"].state == "promoted"


def test_local_promoter_clones_complete_warning_schema_for_compatible_package(tmp_path: Path) -> None:
    resource_root = tmp_path / "e"; package = resource_root / "abc123"
    (package / "AmazingFeature").mkdir(parents=True)
    (package / "config.json").write_text('{"effect":{"Link":[{"path":"AmazingFeature/"}]}}', encoding="utf-8")
    (package / "AmazingFeature" / "main.scene").write_text("scene", encoding="utf-8")
    canonical_resource = tmp_path / "warning-resource"; canonical_resource.mkdir(); (canonical_resource / "x").write_bytes(b"x")
    canonical_dir = tmp_path / "captured_templates" / "warning"; canonical_dir.mkdir(parents=True)
    canonical = CapturedEffectTemplate(
        {"id": "__MATERIAL_ID__", "effect_id": "w", "resource_id": "w", "name": "Warning", "type": "video_effect", "path": str(canonical_resource), "category_name": "Video effects", "adjust_params": []},
        {"id": "__SEGMENT_ID__", "material_id": "__MATERIAL_ID__", "target_timerange": {"start": "__START_US__", "duration": "__DURATION_US__"}},
        source_effect_id="w",
    )
    save_template(canonical, canonical_dir / "captured_effect_template.json")
    entry = SimpleNamespace(
        stable_key="local:e", display_name="Serious Error", source="local", effect_id="e", resource_id="e",
        local_resource_path=resource_root, raw_metadata_subset={"effect_type": 7, "md5": "abc123", "source": 1}, test_index=1,
    )
    prepared = LocalEffectTemplatePromoter(CapturedEffectTemplateRepository(tmp_path)).prepare([entry])[0]
    assert prepared.resolved
    assert prepared.preset_key == "serious_error"
    assert prepared.template.material["type"] == "video_effect"
    assert prepared.template.material["path"].endswith("abc123")
    assert prepared.template.material["category_name"] == "Video effects"
    assert "placeholder" not in json.dumps(prepared.template.material).casefold()


def test_local_promoter_rejects_missing_or_incompatible_package(tmp_path: Path) -> None:
    entry = SimpleNamespace(
        stable_key="local:missing", display_name="Missing Effect", source="local", effect_id="missing", resource_id="missing",
        local_resource_path=tmp_path / "missing", raw_metadata_subset={"effect_type": 7, "md5": "absent"}, test_index=1,
    )
    prepared = LocalEffectTemplatePromoter(CapturedEffectTemplateRepository(tmp_path)).prepare([entry])[0]
    assert not prepared.resolved
    assert "package" in prepared.reason


def test_local_promoter_slug_collision_uses_stable_id_suffix(tmp_path: Path) -> None:
    promoter = LocalEffectTemplatePromoter(CapturedEffectTemplateRepository(tmp_path))
    entries = [
        SimpleNamespace(stable_key="local:one", display_name="Glow", effect_id="one", test_index=1),
        SimpleNamespace(stable_key="local:two", display_name="Glow", effect_id="two", test_index=2),
    ]
    keys = promoter.allocate_preset_keys(entries)
    assert keys["local:one"] == "glow"
    assert keys["local:two"] == "glow_two"
