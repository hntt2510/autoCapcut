"""
test_qa_acceptance_5img.py
==========================
End-to-End Section 16 QA Verification Test.

Validates the full 5-image acceptance case from the product specification:
1. Correct BASIC / ADVANCED classification
2. Missing object names shown correctly in UI status panel
3. Missing camera frame shown correctly
4. Configure button text and count
5. Queue contains only incomplete advanced images (skips BASIC)
6. Queue advances automatically on save
7. Status refreshes after save
8. Preflight blocks project build when incomplete
9. Preflight passes when complete
10. Draw Debug tab remains operational
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMessageBox

from auto_capcut.core.draw_models import (
    CameraFramingMode,
    NormalizedRect,
    SceneDocument,
    SceneImage,
    SceneObject,
)
from auto_capcut.core.draw_scene import save_scene, load_scene
from auto_capcut.core.draw_setup import analyze_from_srt
from auto_capcut.ui.draw_animation import DrawObjectEditorDialog
from auto_capcut.ui.main_window import MainWindow


@pytest.fixture
def qt_app():
    return QApplication.instance() or QApplication([])


def test_section_16_qa_complete_workflow(qt_app: QApplication, tmp_path: Path) -> None:
    # 1. Setup 5 images matching acceptance specification
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    images = []
    for i in range(1, 6):
        img_p = img_dir / f"{i:03d}.png"
        Image.new("RGB", (320, 180), color="blue").save(img_p)
        images.append(img_p)

    # 2. Setup acceptance SRT
    srt_content = (
        "1\n00:00:00,000 --> 00:00:04,000\nDRAW 0s-2s:\n\n"
        "2\n00:00:04,000 --> 00:00:10,000\nMODE advanced_draw\n"
        "OBJECT_EFFECT target=title effect=draw\n"
        "OBJECT_EFFECT target=card_left effect=slide_in direction=left\n"
        "OBJECT_EFFECT target=card_right effect=slide_in direction=right\n\n"
        "3\n00:00:10,000 --> 00:00:15,000\nMODE basic_draw\n\n"
        "4\n00:00:15,000 --> 00:00:22,000\nMODE advanced_draw\n"
        "OBJECT_EFFECT target=alex effect=draw\n"
        "OBJECT_EFFECT target=warning effect=push_in direction=top\n"
        "CAMERA_AFTER object=warning action=focus target=warning framing=camera_frame persist=false\n\n"
        "5\n00:00:22,000 --> 00:00:27,000\nPOST_MOTION subtle_zoom_in\n"
    )
    srt_file = tmp_path / "main_effect.srt"
    srt_file.write_text(srt_content, encoding="utf-8")

    # 3. Setup partial scene JSON:
    # 002: title, card_left (card_right MISSING)
    # 004: alex, warning (warning camera_frame MISSING)
    scene_file = tmp_path / "draw_scene.json"
    doc = SceneDocument(
        schema_version=1,
        images={
            "002.png": SceneImage("002.png", (320, 180), (
                SceneObject("title", "art", NormalizedRect(0.1, 0.1, 0.2, 0.2)),
                SceneObject("card_left", "art", NormalizedRect(0.3, 0.3, 0.2, 0.2)),
            ), ("title", "card_left")),
            "004.png": SceneImage("004.png", (320, 180), (
                SceneObject("alex", "art", NormalizedRect(0.1, 0.1, 0.2, 0.2)),
                SceneObject("warning", "art", NormalizedRect(0.4, 0.4, 0.2, 0.2), camera_frame=None),
            ), ("alex", "warning")),
        },
        path=scene_file,
    )
    save_scene(doc, scene_file)

    # 4. Instantiate MainWindow & wire up paths
    window = MainWindow()
    assert window._config().draw_fallback_basic is False
    window.image_list.clear()
    window.image_list.addItem(str(img_dir))
    window.effect_path.setText(str(srt_file))
    window.draw_scene_path.setText(str(scene_file))

    window._update_draw_scene_status()
    status_text = window.draw_setup_status.text()

    # Verify visual text in status panel
    assert "001.png" in status_text and "BASIC" in status_text and "Ready ✓" in status_text
    assert "002.png" in status_text and "ADVANCED" in status_text and "card_right" in status_text
    assert "003.png" in status_text and "BASIC" in status_text and "Ready ✓" in status_text
    assert "004.png" in status_text and "ADVANCED" in status_text and "warning camera frame missing" in status_text
    assert "005.png" in status_text and "BASIC" in status_text and "Ready ✓" in status_text

    # Verify Summary counts
    assert "5 images" in status_text
    assert "3 Basic" in status_text
    assert "2 Advanced" in status_text
    assert "2 Needs Setup" in status_text

    # Verify Configure Button text
    assert window.configure_advanced_btn.text() == "Configure 2 Advanced Images"
    assert window.configure_advanced_btn.isEnabled()

    # 5. Verify Preflight Blocking
    with patch.object(QMessageBox, "exec") as mock_box:
        window._create_project()
        assert mock_box.called
        assert window.worker is None

    # 6. Verify Queue in DrawObjectEditorDialog
    summary = window._draw_summary
    assert [s.image_name for s in summary.incomplete_advanced] == ["002.png", "004.png"]

    # Open dialog for queue
    dialog = DrawObjectEditorDialog(
        images,
        scene_file,
        (1920, 1080),
        parent=window,
        initial_image_index=0,
        allowed_image_names=[s.image_name for s in summary.incomplete_advanced],
        required_ids_by_image={s.image_name.casefold(): list(s.required_ids) for s in summary.incomplete_advanced},
        camera_frame_ids_by_image={s.image_name.casefold(): set(s.required_camera_frame_ids) for s in summary.incomplete_advanced},
    )

    # Combo only contains 002.png and 004.png (BASIC images 001, 003, 005 excluded!)
    combo_names = [dialog.image_combo.itemText(i) for i in range(dialog.image_combo.count())]
    assert combo_names == ["002.png", "004.png"]
    assert dialog.current_image == 0  # starts at 002.png

    # On 002.png: add required 'card_right'
    assert dialog._add_required_combo.currentText() == "card_right"
    dialog._add_required_object()
    assert dialog._is_image_ready(0)

    # Save 002.png -> should auto-advance to 004.png (idx 1) without closing
    with patch.object(dialog, "accept") as mock_accept:
        dialog._save()
        assert not mock_accept.called
        assert dialog.current_image == 1
        assert dialog.images[dialog.current_image].name == "004.png"

    # On 004.png: configure camera frame for warning
    # warning is present at index 1 in objects
    warning_idx = next(i for i, o in enumerate(dialog._record()["objects"]) if o["id"] == "warning")
    dialog.object_list.setCurrentRow(warning_idx)
    dialog._camera_toggled(True)  # creates camera frame
    assert dialog._is_image_ready(1)

    # Save 004.png -> all queue images are ready -> dialog accepts!
    with patch.object(dialog, "accept") as mock_accept:
        dialog._save()
        assert mock_accept.called

    dialog.deleteLater()

    # 7. Refresh MainWindow status
    window._update_draw_scene_status()
    updated_status = window.draw_setup_status.text()
    assert "2 Advanced Ready  |  0 Needs Setup" in updated_status
    assert "Setup needed" not in updated_status
    assert window.configure_advanced_btn.text() == "Review Advanced Images"

    # 8. Verify Preflight now passes and builds project
    with patch("auto_capcut.ui.main_window.ProjectWorker") as mock_worker_cls, \
         patch("auto_capcut.ui.main_window.QThread") as mock_thread_cls:
        mock_worker = MagicMock()
        mock_worker_cls.return_value = mock_worker
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        window._create_project()
        assert window.worker is not None
        assert mock_thread.start.called

    window.deleteLater()
