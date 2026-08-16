from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
    """Verify Edit Draw Objects button and scene status label exist in MainWindow."""
    window = MainWindow()
    assert hasattr(window, "edit_draw_objects_btn")
    assert window.edit_draw_objects_btn.text() == "Edit Draw Objects"
    assert hasattr(window, "draw_scene_status")
    assert hasattr(window, "draw_source_label")
    assert window.draw_source_label.text() == "Uses Main Effect SRT (timing + draw mode source)"
    window.deleteLater()


def test_edit_draw_objects_warns_if_no_images(qt_app: QApplication, tmp_path: Path) -> None:
    """If no images exist, clicking Edit Draw Objects displays warning dialog."""
    window = MainWindow()
    window.image_list.clear()

    with patch.object(QMessageBox, "warning") as mock_warn:
        window._edit_draw_objects()
        assert mock_warn.called
        assert "Add an image folder first" in str(mock_warn.call_args)
    window.deleteLater()


def test_edit_draw_objects_auto_resolves_scene_path(qt_app: QApplication, tmp_path: Path) -> None:
    """Empty Scene JSON field auto-resolves to <effect_dir>/draw_scene.json or <image_dir>/draw_scene.json."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 3)

    effect_folder = tmp_path / "effects"
    effect_folder.mkdir(parents=True, exist_ok=True)
    effect_file = effect_folder / "unified_effect.srt"
    effect_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nHOLD 0s-2s:\n", encoding="utf-8")

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(effect_file))
    window.draw_scene_path.setText("")

    with patch.object(DrawObjectEditorDialog, "exec"):
        window._edit_draw_objects()

    expected_scene = effect_folder / "draw_scene.json"
    assert window.draw_scene_path.text() == str(expected_scene)
    window.deleteLater()


def test_draw_scene_status_summary_updates(qt_app: QApplication, tmp_path: Path) -> None:
    """Verify scene status accurately summarizes configured objects, camera frames, and fallback warnings."""
    img_folder = tmp_path / "images"
    images = _create_images(img_folder, 3)

    effect_file = tmp_path / "effect.srt"
    effect_file.write_text(
        "\n".join([
            "1",
            "00:00:00,000 --> 00:00:02,000",
            "MODE advanced_draw",
            "STYLE v1",
            "DRAW 0s-2s:",
            "",
            "2",
            "00:00:02,000 --> 00:00:04,000",
            "MODE advanced_draw",
            "STYLE v1",
            "DRAW 0s-2s:",
            "",
            "3",
            "00:00:04,000 --> 00:00:06,000",
            "HOLD 0s-2s:",
            "",
        ]),
        encoding="utf-8",
    )

    # Create scene with only 001.png configured (6 objects, 2 camera frames)
    scene_file = tmp_path / "draw_scene.json"
    objects = [
        SceneObject(
            id=f"object_{i}",
            type="art",
            box=NormalizedRect(0.1 * i, 0.1, 0.08, 0.08),
            camera_frame=NormalizedRect(0.1 * i, 0.1, 0.16, 0.09) if i <= 2 else None,
        )
        for i in range(1, 7)
    ]
    scene_doc = SceneDocument(
        schema_version=1,
        images={"001.png": SceneImage("001.png", (1920, 1080), tuple(objects), tuple(o.id for o in objects))},
        path=scene_file,
    )
    save_scene(scene_doc, scene_file)

    window = MainWindow()
    window.image_list.clear()
    window.image_list.addItem(str(img_folder))
    window.effect_path.setText(str(effect_file))
    window.draw_scene_path.setText(str(scene_file))
    window.draw_fallback_basic.setChecked(True)

    window._update_draw_scene_status()
    status_text = window.draw_scene_status.text()

    assert "3 project images (1 configured)" in status_text
    assert "001.png: 6 objects / 2 camera frames" in status_text
    assert "002.png: advanced scene missing → will fallback to basic_draw" in status_text
    assert "003.png: not configured" in status_text

    # When fallback is disabled, missing scene becomes blocking
    window.draw_fallback_basic.setChecked(False)
    window._update_draw_scene_status()
    status_text_blocking = window.draw_scene_status.text()
    assert "002.png: advanced scene missing (blocking error)" in status_text_blocking

    window.deleteLater()
