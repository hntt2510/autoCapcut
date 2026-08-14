from __future__ import annotations

import pytest

from auto_capcut.core.effect_catalog import (
    CatalogStore,
    EffectCatalog,
    EffectCatalogEntry,
    normalize_effect_name,
)


def _entry(key: str, name: str, source: str = "local", effect_id: str | None = None) -> EffectCatalogEntry:
    effect_id = effect_id or key
    return EffectCatalogEntry(
        stable_key=f"{source}:{key}", test_index=1, display_name=name, enum_name=name,
        effect_id=effect_id, resource_id=effect_id, source=source,
    )


def test_effect_name_normalization_is_slug_and_search_compatible() -> None:
    assert normalize_effect_name("  Zoom__Lens ") == "zoom lens"
    assert normalize_effect_name("zoom-lens") == "zoom lens"
    assert normalize_effect_name("Zoom Lens") == "zoom lens"


def test_catalog_matching_unique_not_found_and_ambiguous(tmp_path) -> None:
    store = CatalogStore(tmp_path)
    store.save(EffectCatalog([
        _entry("zoom", "Zoom Lens"),
        _entry("a", "Duplicate", effect_id="a"),
        _entry("b", "Duplicate", effect_id="b"),
        _entry("scene", "Scene Only", source="scene", effect_id="scene"),
    ]))
    matches = {item.preset_key: item for item in store.match_preset_keys(["zoom_lens", "duplicate", "missing", "scene_only"])}
    assert matches["zoom_lens"].status == "unique"
    assert matches["zoom_lens"].selected_local.display_name == "Zoom Lens"
    assert matches["duplicate"].status == "ambiguous"
    assert matches["missing"].status == "not_found"
    assert matches["scene_only"].status == "not_promotable"
    assert len(store.filter_entries("zoom_lens")) == 1
    assert len(store.filter_entries("ZOOM-LENS")) == 1


def test_catalog_matching_prefers_local_link_for_cross_source_duplicate(tmp_path) -> None:
    store = CatalogStore(tmp_path)
    store.save(EffectCatalog([
        _entry("scene", "Zoom Lens", source="scene", effect_id="shared"),
        _entry("local", "Other Name", source="local", effect_id="shared"),
    ]))
    result = store.match_preset_keys(["zoom_lens"])[0]
    assert result.status == "unique"
    assert result.selected_local.source == "local"


def test_resolution_dialog_shows_bulk_promotion_action(tmp_path) -> None:
    pytest.importorskip("PyQt6", reason="PyQt6 is optional")
    from PyQt6.QtWidgets import QApplication
    from auto_capcut.ui.effect_catalog_dialog import EffectCatalogDialog

    app = QApplication.instance() or QApplication([])
    store = CatalogStore(tmp_path)
    store.save(EffectCatalog([_entry("zoom", "Zoom Lens")]))
    dialog = EffectCatalogDialog(resolution_presets=["zoom_lens"])
    dialog._store = store
    dialog._load_resolution_matches()
    assert dialog.promote_button.text() == "Promote All Matched Effects"
    assert dialog.resolution_table.rowCount() == 1
    dialog.close()
