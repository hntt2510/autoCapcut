from __future__ import annotations

from pathlib import Path
import json

import pytest
from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from auto_capcut.core.draw_models import NormalizedRect
from auto_capcut.ui.draw_animation import DrawCanvas, DrawObjectEditorDialog


@pytest.fixture
def qt_app():
    return QApplication.instance() or QApplication([])


def _canvas(tmp_path: Path, qt_app: QApplication) -> DrawCanvas:
    image = tmp_path / "001.png"
    Image.new("RGB", (400, 200), "white").save(image)
    canvas = DrawCanvas(image, 16 / 9)
    canvas.resize(600, 500)
    canvas.set_objects([{"id": "one", "type": "art", "box": NormalizedRect(.25, .25, .25, .25), "camera": None}], 0, False)
    canvas.show()
    qt_app.processEvents()
    canvas.displayed_image_rect = canvas._display_rect()
    return canvas


def test_letterboxed_coordinate_mapping_uses_displayed_image_rect(tmp_path: Path, qt_app: QApplication) -> None:
    canvas = _canvas(tmp_path, qt_app)
    display = canvas.displayed_image_rect
    rect = NormalizedRect(.1, .2, .3, .4)
    pixels = canvas._rect_pixels(rect, display)
    assert canvas._normalized(pixels, display) == rect
    assert display.top() > 0
    canvas.deleteLater()


def test_mouse_move_selects_and_clamps_object_to_all_edges(tmp_path: Path, qt_app: QApplication) -> None:
    canvas = _canvas(tmp_path, qt_app)
    display = canvas.displayed_image_rect
    center = canvas._rect_pixels(canvas.objects[0]["box"], display).center().toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(canvas, QPoint(int(display.left()), int(display.top())))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(int(display.left()), int(display.top())))
    moved = canvas.objects[0]["box"]
    assert moved.x == pytest.approx(0)
    assert moved.y == pytest.approx(0)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=canvas._rect_pixels(moved, display).center().toPoint())
    bottom_right = QPoint(int(display.right()), int(display.bottom()))
    QTest.mouseMove(canvas, bottom_right)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=bottom_right)
    moved = canvas.objects[0]["box"]
    assert moved.x + moved.w == pytest.approx(1)
    assert moved.y + moved.h == pytest.approx(1)
    canvas.deleteLater()


@pytest.mark.parametrize("handle", [
    "resize_top_left", "resize_top_right", "resize_bottom_left", "resize_bottom_right",
    "resize_left", "resize_right", "resize_top", "resize_bottom",
])
def test_object_resize_handles_are_free_aspect_and_clamped(tmp_path: Path, qt_app: QApplication, handle: str) -> None:
    canvas = _canvas(tmp_path, qt_app)
    display = canvas.displayed_image_rect
    start = canvas._rect_pixels(canvas.objects[0]["box"], display)
    canvas.selected = 0
    canvas.start_rect = start
    canvas.interaction = handle
    point = {
        "resize_top_left": start.topLeft(), "resize_top_right": start.topRight(),
        "resize_bottom_left": start.bottomLeft(), "resize_bottom_right": start.bottomRight(),
        "resize_left": QPointF(start.left(), start.center().y()), "resize_right": QPointF(start.right(), start.center().y()),
        "resize_top": QPointF(start.center().x(), start.top()), "resize_bottom": QPointF(start.center().x(), start.bottom()),
    }[handle]
    point = QPointF(point.x() + (35 if "right" in handle or "left" not in handle else -35), point.y() + (20 if "bottom" in handle or "top" not in handle else -20))
    resized = canvas._resize_rect(point)
    assert resized.width() >= canvas.MIN_RECT_SIZE
    assert resized.height() >= canvas.MIN_RECT_SIZE
    assert resized.left() >= display.left() and resized.top() >= display.top()
    assert resized.right() <= display.right() and resized.bottom() <= display.bottom()
    if not canvas.camera_mode:
        assert resized.width() / resized.height() != pytest.approx(16 / 9, rel=0.01)
    canvas.deleteLater()


@pytest.mark.parametrize("handle", ["resize_top_left", "resize_top_right", "resize_bottom_left", "resize_bottom_right", "resize_left", "resize_right", "resize_top", "resize_bottom"])
def test_camera_resize_handles_preserve_project_aspect(tmp_path: Path, qt_app: QApplication, handle: str) -> None:
    canvas = _canvas(tmp_path, qt_app)
    canvas.camera_mode = True
    display = canvas.displayed_image_rect
    canvas.objects[0]["camera"] = NormalizedRect(.25, .25, .25, .140625)
    start = canvas._rect_pixels(canvas.objects[0]["camera"], display)
    canvas.selected = 0; canvas.start_rect = start; canvas.interaction = handle
    resized = canvas._resize_rect(QPointF(start.right() + 30, start.bottom() + 20))
    assert resized.width() / resized.height() == pytest.approx(16 / 9, rel=0.01)
    assert resized.left() >= display.left() and resized.top() >= display.top()
    assert resized.right() <= display.right() and resized.bottom() <= display.bottom()
    canvas.deleteLater()


def test_canvas_and_list_selection_stay_synchronized(tmp_path: Path, qt_app: QApplication) -> None:
    image = tmp_path / "001.png"; Image.new("RGB", (400, 200), "white").save(image)
    dialog = DrawObjectEditorDialog([image], tmp_path / "scene.json", (1920, 1080))
    dialog._add(); dialog._add()
    assert dialog.object_list.currentRow() == 1
    display = dialog.canvas._display_rect()
    second = dialog.canvas._rect_pixels(dialog.records["001.png"]["objects"][1]["box"], display).center().toPoint()
    QTest.mousePress(dialog.canvas, Qt.MouseButton.LeftButton, pos=second)
    QTest.mouseRelease(dialog.canvas, Qt.MouseButton.LeftButton, pos=second)
    assert dialog.current_object == 1
    dialog.object_list.setCurrentRow(0)
    assert dialog.canvas.selected == 0
    dialog.deleteLater()


def test_camera_frame_initialization_uses_source_pixel_aspect(tmp_path: Path, qt_app: QApplication) -> None:
    image = tmp_path / "001.png"; Image.new("RGB", (400, 200), "white").save(image)
    dialog = DrawObjectEditorDialog([image], tmp_path / "scene.json", (1920, 1080))
    dialog._add(); dialog.camera.setChecked(True)
    frame = dialog.records["001.png"]["objects"][0]["camera"]
    assert frame.w / frame.h * (400 / 200) == pytest.approx(1920 / 1080, rel=0.01)
    dialog.deleteLater()


def test_add_rename_save_reopen_preserves_normalized_geometry(tmp_path: Path, qt_app: QApplication) -> None:
    image = tmp_path / "001.png"; Image.new("RGB", (400, 200), "white").save(image)
    scene_path = tmp_path / "scene.json"
    dialog = DrawObjectEditorDialog([image], scene_path, (1920, 1080))
    dialog._add()
    expected = NormalizedRect(.04, .18, .27, .19)
    dialog.records["001.png"]["objects"][0]["box"] = expected
    dialog.name.setText("part_ab"); dialog._rename(); dialog._save()
    reopened = DrawObjectEditorDialog([image], scene_path, (1920, 1080))
    record = reopened.records["001.png"]["objects"][0]
    assert record["id"] == "part_ab"
    assert record["box"] == expected
    reopened.deleteLater()


def test_object_editor_is_spatial_only_and_preserves_legacy_behavior_fields(tmp_path: Path, qt_app: QApplication) -> None:
    image = tmp_path / "001.png"; Image.new("RGB", (400, 200), "white").save(image)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"schema_version": 1, "images": {"001.png": {"source_size": {"width": 400, "height": 200}, "objects": [{"id": "part_ab", "type": "art", "box": {"x": .1, "y": .1, "w": .3, "h": .3}, "draw_order": [], "render_effect": "slide_in", "direction": "left", "duration": .7, "pause_after": .1}], "draw_order": ["part_ab"]}}}), encoding="utf-8")
    dialog = DrawObjectEditorDialog([image], scene_path, (1920, 1080))
    assert not hasattr(dialog, "render_effect")
    assert not hasattr(dialog, "direction")
    assert not hasattr(dialog, "duration")
    assert not hasattr(dialog, "pause_after")
    dialog.records["001.png"]["objects"][0]["box"] = NormalizedRect(.2, .2, .25, .25)
    dialog._save()
    raw = json.loads(scene_path.read_text(encoding="utf-8"))["images"]["001.png"]["objects"][0]
    assert raw["render_effect"] == "slide_in"
    assert raw["direction"] == "left"
    assert raw["duration"] == pytest.approx(.7)
    assert raw["pause_after"] == pytest.approx(.1)


@pytest.mark.skipif(not Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\img\001.png").is_file(), reason="Medicare fixture unavailable")
def test_medicare_part_ab_geometry_survives_editor_reopen(tmp_path: Path, qt_app: QApplication) -> None:
    image = Path(r"D:\HOCTAP\latvat\VIDEO YTB\Insurance\test\img\001.png")
    scene_path = tmp_path / "scene.json"
    dialog = DrawObjectEditorDialog([image], scene_path, (1920, 1080))
    dialog._add()
    expected = NormalizedRect(54 / 1376, 145 / 768, (425 - 54) / 1376, (293 - 145) / 768)
    dialog.records[image.name.casefold()]["objects"][0]["box"] = expected
    dialog.name.setText("part_ab"); dialog._rename(); dialog._save()
    reopened = DrawObjectEditorDialog([image], scene_path, (1920, 1080))
    record = reopened.records[image.name.casefold()]["objects"][0]
    assert record["id"] == "part_ab"
    assert record["box"] == expected
    reopened.deleteLater()
