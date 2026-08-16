"""
test_draft_draw_ui.py
=====================
UI contract tests for the MainWindow draw animation panel.

Updated for new SRT-driven draw setup panel.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from auto_capcut.core.draw_models import NormalizedRect, SceneDocument, SceneImage, SceneObject
from auto_capcut.core.draw_scene import save_scene
from auto_capcut.ui.draw_animation import DrawObjectEditorDialog
from auto_capcut.ui.main_window import MainWindow


@pytest.fixture
def qt_app():
    return QApplication.instance() or QApplication([])


def _create_images(folder: Path, count: int = 3) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(1, count + 1):
        img = folder / f"{i:03d}.png"
        Image.new("RGB", (320, 180), color="white").save(img)
        images.append(img)
    return images


def test_main_window_draw_ui_elements_exist(qt_app: QApplication, tmp_path: Path) -> None:
    """Verify Configure Advanced Images button and scene status label exist in MainWindow."""
    window = MainWindow()
    # configure_advanced_btn replaces edit_draw_objects_btn
    assert hasattr(window, "configure_advanced_btn")
    assert "Advanced" in window.configure_advanced_btn.text() or "required" in window.configure_advanced_btn.text().lower()
    # alias still exists for backward compat
    assert hasattr(window, "edit_draw_objects_btn")
    assert hasattr(window, "draw_setup_status")
    assert hasattr(window, "draw_scene_status")  # alias
    assert hasattr(window, "draw_source_label")
    assert window.draw_source_label.text() == "Uses Main Effect SRT (timing + draw mode source)"
    window.deleteLater()


def test_configure_advanced_warns_if_no_images(qt_app: QApplication, tmp_path: Path) -> None:
    """If no images exist, clicking Configure Advanced Images displays warning dialog."""
    window = MainWindow()
    window.image_list.clear()

    with patch.object(QMessageBox, "warning") as mock_warn:
        window._configure_advanced_images()
        assert mock_warn.called
        assert "Add an image folder first" in str(mock_warn.call_args)
    window.deleteLater()


def test_configure_advanced_auto_resolves_scene_path(qt_app: QApplication, tmp_path: Path) -> None:
    """Empty Scene JSON field auto-resolves to <effect_dir>/draw_scene.json."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 3)

    effect_folder = tmp_path / "effects"
    effect_folder.mkdir(parents=True, exist_ok=True)
    effect_file = effect_folder / "unified_effect.srt"
    effect_file.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nDRAW 0s-2s:\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nMODE advanced_draw\nDRAW 0s-2s:\n\n"
        "3\n00:00:04,000 --> 00:00:06,000\nMODE basic_draw\n",
        encoding="utf-8",
    )

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(effect_file))
    window.draw_scene_path.setText("")
    window._update_draw_scene_status()  # populate _draw_summary

    with patch.object(DrawObjectEditorDialog, "exec"):
        window._configure_advanced_images()

    expected_scene = effect_folder / "draw_scene.json"
    assert window.draw_scene_path.text() == str(expected_scene)
    window.deleteLater()


def test_draw_scene_status_summary_updates(qt_app: QApplication, tmp_path: Path) -> None:
    """Verify scene status shows SRT-driven BASIC / ADVANCED classification."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 3)

    # SRT: cue 1 advanced, cue 2 advanced, cue 3 basic
    effect_file = tmp_path / "effect.srt"
    effect_file.write_text(
        "\n".join([
            "1",
            "00:00:00,000 --> 00:00:02,000",
            "MODE advanced_draw",
            "DRAW 0s-2s:",
            "",
            "2",
            "00:00:02,000 --> 00:00:04,000",
            "MODE advanced_draw",
            "DRAW 0s-2s:",
            "",
            "3",
            "00:00:04,000 --> 00:00:06,000",
            "MODE basic_draw",
            "",
        ]),
        encoding="utf-8",
    )

    # Create scene with only 001.png configured (1 object)
    scene_file = tmp_path / "draw_scene.json"
    objects = [
        SceneObject(
            id="object_1",
            type="art",
            box=NormalizedRect(0.1, 0.1, 0.2, 0.2),
            camera_frame=None,
        )
    ]
    scene_doc = SceneDocument(
        schema_version=1,
        images={"001.png": SceneImage("001.png", (1920, 1080), tuple(objects), ("object_1",))},
        path=scene_file,
    )
    save_scene(scene_doc, scene_file)

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(effect_file))
    window.draw_scene_path.setText(str(scene_file))

    window._update_draw_scene_status()
    status_text = window.draw_scene_status.text()

    # Should show DRAW SETUP header
    assert "DRAW SETUP" in status_text
    # Should show all 3 images
    assert "001.png" in status_text
    assert "002.png" in status_text
    assert "003.png" in status_text
    # 001 advanced and has an object → Ready
    assert "ADVANCED" in status_text
    # 003 is basic → BASIC
    assert "BASIC" in status_text
    # Summary line
    assert "3 images" in status_text

    window.deleteLater()


def test_draw_object_editor_queue_filtering_and_required_objects(qt_app: QApplication, tmp_path: Path) -> None:
    """Verify DrawObjectEditorDialog restricts image combo to allowed queue and provides required objects dropdown."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 5)
    scene_file = tmp_path / "draw_scene.json"

    allowed = ["002.png", "004.png"]
    req_ids = {
        "002.png": ["title", "card_left", "card_right"],
        "004.png": ["alex", "warning"],
    }
    cam_ids = {
        "004.png": {"warning"},
    }

    dialog = DrawObjectEditorDialog(
        images,
        scene_file,
        (1920, 1080),
        allowed_image_names=allowed,
        required_ids_by_image=req_ids,
        camera_frame_ids_by_image=cam_ids,
    )

    # Combo only has allowed images
    combo_items = [dialog.image_combo.itemText(i) for i in range(dialog.image_combo.count())]
    assert combo_items == ["002.png", "004.png"]

    # Current image is 002.png
    assert dialog.images[dialog.current_image].name == "002.png"

    # Required dropdown exists and has all 3 required IDs
    assert dialog._add_required_combo is not None
    add_req_items = [dialog._add_required_combo.itemText(i) for i in range(dialog._add_required_combo.count())]
    assert set(add_req_items) == {"title", "card_left", "card_right"}

    # Add required object 'title'
    dialog._add_required_combo.setCurrentText("title")
    dialog._add_required_object()
    obj_ids = [o["id"] for o in dialog._record()["objects"]]
    assert "title" in obj_ids

    # Remaining in combo
    add_req_remaining = [dialog._add_required_combo.itemText(i) for i in range(dialog._add_required_combo.count())]
    assert "title" not in add_req_remaining
    assert set(add_req_remaining) == {"card_left", "card_right"}

    dialog.deleteLater()


def test_draw_object_editor_auto_advances_queue_on_save(qt_app: QApplication, tmp_path: Path) -> None:
    """Saving an image in queue mode advances to next incomplete image if ready."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 3)
    scene_file = tmp_path / "draw_scene.json"

    allowed = ["002.png", "003.png"]
    req_ids = {
        "002.png": ["title"],
        "003.png": ["card"],
    }

    dialog = DrawObjectEditorDialog(
        images,
        scene_file,
        (1920, 1080),
        allowed_image_names=allowed,
        required_ids_by_image=req_ids,
    )

    # Initially at 002.png (idx 0)
    assert dialog.current_image == 0
    assert dialog.images[0].name == "002.png"

    # Add 'title' to 002.png so 002 is ready
    dialog._add_required_combo.setCurrentText("title")
    dialog._add_required_object()
    assert dialog._is_image_ready(0)

    # Save -> should auto-advance to 003.png (idx 1)
    with patch.object(dialog, "accept") as mock_accept:
        dialog._save()
        assert not mock_accept.called
        assert dialog.current_image == 1
        assert dialog.images[dialog.current_image].name == "003.png"

    # Now on 003.png: add 'card'
    dialog._add_required_combo.setCurrentText("card")
    dialog._add_required_object()
    assert dialog._is_image_ready(1)

    # Save -> both are ready, should call accept() to close
    with patch.object(dialog, "accept") as mock_accept:
        dialog._save()
        assert mock_accept.called

    dialog.deleteLater()


def test_preflight_blocks_when_advanced_setup_missing(qt_app: QApplication, tmp_path: Path) -> None:
    """Preflight blocks project creation when an advanced_draw image is missing setup in default production UI."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 3)

    effect_file = tmp_path / "effect.srt"
    effect_file.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nDRAW 0s-2s:\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nMODE advanced_draw\nOBJECT_EFFECT target=warning effect=draw\n\n"
        "3\n00:00:04,000 --> 00:00:06,000\nMODE basic_draw\n",
        encoding="utf-8",
    )

    window = MainWindow()
    # Verify default UI-created ProjectConfig has draw_fallback_basic=False
    assert window._config().draw_fallback_basic is False

    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(effect_file))
    window.draw_scene_path.setText("")
    window._update_draw_scene_status()

    # Preflight should show QMessageBox warning and not start worker
    with patch.object(QMessageBox, "exec") as mock_exec, patch.object(QMessageBox, "addButton") as mock_btn:
        window._create_project()
        assert mock_exec.called
        # Worker thread should not have started
        assert window.worker is None

    window.deleteLater()


def test_preflight_blocks_on_stale_srt_when_effect_path_changes(qt_app: QApplication, tmp_path: Path) -> None:
    """Changing effect_path to an SRT requiring new objects must block create without stale cached bypass."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 1)

    # SRT A: requires only 'title'
    srt_a = tmp_path / "effect_a.srt"
    srt_a.write_text("1\n00:00:00,000 --> 00:00:02,000\nMODE advanced_draw\nOBJECT_EFFECT target=title effect=draw\n", encoding="utf-8")

    # SRT B: requires 'title' AND 'warning'
    srt_b = tmp_path / "effect_b.srt"
    srt_b.write_text("1\n00:00:00,000 --> 00:00:02,000\nMODE advanced_draw\nOBJECT_EFFECT target=title effect=draw\nOBJECT_EFFECT target=warning effect=draw\n", encoding="utf-8")

    # Scene JSON configured with only 'title'
    scene_file = tmp_path / "draw_scene.json"
    doc = SceneDocument(
        schema_version=1,
        images={"001.png": SceneImage("001.png", (1920, 1080), (
            SceneObject("title", "art", NormalizedRect(0.1, 0.1, 0.2, 0.2)),
        ), ("title",))},
        path=scene_file,
    )
    save_scene(doc, scene_file)

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.draw_scene_path.setText(str(scene_file))

    # Load SRT A -> ready state
    window.effect_path.setText(str(srt_a))
    assert window._draw_summary is not None
    assert window._draw_summary.all_ready is True

    # Switch to SRT B (do NOT manually call _update_draw_scene_status)
    window.effect_path.setText(str(srt_b))

    # Trigger create -> must strictly block for missing 'warning'
    with patch.object(QMessageBox, "exec") as mock_exec, patch.object(QMessageBox, "addButton"):
        window._create_project()
        assert mock_exec.called
        assert window.worker is None

    window.deleteLater()


def test_preflight_fails_closed_on_invalid_scene_json(qt_app: QApplication, tmp_path: Path) -> None:
    """Invalid/corrupt scene JSON must block project creation with a critical error (fail closed)."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 1)

    srt = tmp_path / "effect.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nMODE advanced_draw\nOBJECT_EFFECT target=title effect=draw\n", encoding="utf-8")

    scene_file = tmp_path / "draw_scene.json"
    scene_file.write_text("{corrupt json content}", encoding="utf-8")

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(srt))
    window.draw_scene_path.setText(str(scene_file))

    with patch.object(QMessageBox, "critical") as mock_crit:
        window._create_project()
        assert mock_crit.called
        assert window.worker is None

    window.deleteLater()


def test_preflight_fails_closed_on_missing_or_malformed_srt(qt_app: QApplication, tmp_path: Path) -> None:
    """Missing or malformed SRT must block project creation (fail closed)."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 1)

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(tmp_path / "non_existent.srt"))

    with patch.object(QMessageBox, "critical") as mock_crit:
        window._create_project()
        assert mock_crit.called
        assert window.worker is None

    window.deleteLater()


def test_effect_path_text_changed_refreshes_draw_setup(qt_app: QApplication, tmp_path: Path) -> None:
    """Updating effect_path text automatically triggers _update_draw_scene_status."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 2)

    srt_basic = tmp_path / "basic.srt"
    srt_basic.write_text("1\n00:00:00,000 --> 00:00:02,000\nMODE basic_draw\n\n2\n00:00:02,000 --> 00:00:04,000\nMODE basic_draw\n", encoding="utf-8")

    srt_adv = tmp_path / "adv.srt"
    srt_adv.write_text("1\n00:00:00,000 --> 00:00:02,000\nMODE advanced_draw\nOBJECT_EFFECT target=hero effect=draw\n\n2\n00:00:02,000 --> 00:00:04,000\nMODE basic_draw\n", encoding="utf-8")

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))

    # Setting effect_path to basic.srt
    window.effect_path.setText(str(srt_basic))
    assert "2 Basic" in window.draw_setup_status.text()
    assert "0 Advanced" in window.draw_setup_status.text()

    # Setting effect_path to adv.srt updates status table automatically
    window.effect_path.setText(str(srt_adv))
    assert "1 Basic" in window.draw_setup_status.text()
    assert "1 Advanced" in window.draw_setup_status.text()
    assert "001.png" in window.draw_setup_status.text()
    assert "ADVANCED" in window.draw_setup_status.text()
    assert "Setup needed" in window.draw_setup_status.text()
    assert "1 Needs Setup" in window.draw_setup_status.text()

    window.deleteLater()


