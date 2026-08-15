from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QSettings, QThread, Qt, QUrl, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import (
    DrawEffectFile,
    DrawImagePlan,
    DrawMode,
    DrawProjectConfig,
    NormalizedRect,
    SceneDocument,
    SceneImage,
    SceneObject,
)
from auto_capcut.core.draw_renderer import DrawRenderer
from auto_capcut.core.draw_scene import load_scene, save_scene, sha256_file
from auto_capcut.core.errors import SceneValidationError
from auto_capcut.core.media import collect_images
from auto_capcut.core.draw_renderer import DrawRenderService
from auto_capcut.models import RESOLUTIONS
from auto_capcut.workers.draw_worker import DrawWorker


def _qimage(path: Path) -> QImage:
    return QImage(str(path))


class DrawCanvas(QWidget):
    changed = pyqtSignal(object)
    selection_changed = pyqtSignal(int)
    geometry_changed = pyqtSignal(int, str, object)

    HANDLE_SIZE = 12.0
    MIN_RECT_SIZE = 12.0

    def __init__(self, image: Path, camera_aspect: float = 16 / 9, parent=None) -> None:
        super().__init__(parent)
        self.image = image
        self._pixmap = QPixmap(str(image)) if image.is_file() else QPixmap()
        self.objects: list[dict] = []
        self.selected = -1
        self.camera_mode = False
        self.camera_aspect = camera_aspect if camera_aspect > 0 else 16 / 9
        self.interaction = "idle"
        self.press_point: QPointF | None = None
        self.start_rect: QRectF | None = None
        self.displayed_image_rect = QRectF()
        self.setMinimumSize(520, 320)
        self.setMouseTracking(True)

    def set_image(self, image: Path) -> None:
        self.image = image
        self._pixmap = QPixmap(str(image)) if image.is_file() else QPixmap()
        self.update()

    def set_objects(self, objects: list[dict], selected: int, camera_mode: bool) -> None:
        self.objects = objects
        self.selected = selected
        self.camera_mode = camera_mode
        self.interaction = "idle"
        self.press_point = None
        self.start_rect = None
        self.update()

    def source_aspect(self) -> float:
        if self._pixmap.isNull() or self._pixmap.height() <= 0:
            return 1.0
        return self._pixmap.width() / self._pixmap.height()

    def _display_rect(self) -> QRectF:
        if self._pixmap.isNull() or self._pixmap.width() <= 0 or self._pixmap.height() <= 0:
            return QRectF()
        scale = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        width, height = self._pixmap.width() * scale, self._pixmap.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    @staticmethod
    def _rect_pixels(rect: NormalizedRect, display: QRectF) -> QRectF:
        return QRectF(display.left() + rect.x * display.width(), display.top() + rect.y * display.height(), rect.w * display.width(), rect.h * display.height())

    @staticmethod
    def _normalized(rect: QRectF, display: QRectF) -> NormalizedRect:
        if display.width() <= 0 or display.height() <= 0:
            return NormalizedRect(0.0, 0.0, 1.0, 1.0)
        x = max(0.0, min(1.0, (rect.left() - display.left()) / display.width()))
        y = max(0.0, min(1.0, (rect.top() - display.top()) / display.height()))
        w = max(0.000001, min(1.0 - x, rect.width() / display.width()))
        h = max(0.000001, min(1.0 - y, rect.height() / display.height()))
        return NormalizedRect(x, y, w, h)

    def _active_rect(self, item: dict) -> NormalizedRect | None:
        return (item.get("camera") or item.get("box")) if self.camera_mode else item.get("box")

    def _active_key(self) -> str:
        return "camera" if self.camera_mode else "box"

    def _handle_rects(self, rect: QRectF) -> dict[str, QRectF]:
        half = self.HANDLE_SIZE / 2
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        center_x, center_y = rect.center().x(), rect.center().y()
        points = {
            "resize_top_left": QPointF(left, top), "resize_top_right": QPointF(right, top),
            "resize_bottom_left": QPointF(left, bottom), "resize_bottom_right": QPointF(right, bottom),
            "resize_left": QPointF(left, center_y), "resize_right": QPointF(right, center_y),
            "resize_top": QPointF(center_x, top), "resize_bottom": QPointF(center_x, bottom),
        }
        return {name: QRectF(point.x() - half, point.y() - half, self.HANDLE_SIZE, self.HANDLE_SIZE) for name, point in points.items()}

    def _handle_at(self, point: QPointF) -> str | None:
        if not (0 <= self.selected < len(self.objects)):
            return None
        rect = self._active_rect(self.objects[self.selected])
        if rect is None:
            return None
        pixels = self._rect_pixels(rect, self.displayed_image_rect)
        for name, handle in self._handle_rects(pixels).items():
            if handle.contains(point):
                return name
        return None

    def _object_at(self, point: QPointF) -> int:
        if not self.displayed_image_rect.contains(point):
            return -1
        indexes = ([self.selected] if 0 <= self.selected < len(self.objects) else [])
        indexes.extend(index for index in range(len(self.objects) - 1, -1, -1) if index != self.selected)
        for index in indexes:
            rect = self._active_rect(self.objects[index])
            if rect and self._rect_pixels(rect, self.displayed_image_rect).contains(point):
                return index
        return -1

    def _clamp_point(self, point: QPointF) -> QPointF:
        display = self.displayed_image_rect
        return QPointF(max(display.left(), min(display.right(), point.x())), max(display.top(), min(display.bottom(), point.y())))

    def _clamp_rect(self, rect: QRectF, preserve_aspect: bool = False) -> QRectF:
        display = self.displayed_image_rect
        width, height = max(1.0, rect.width()), max(1.0, rect.height())
        if preserve_aspect:
            aspect = self.camera_aspect
            minimum_width = self.MIN_RECT_SIZE
            minimum_height = minimum_width / aspect
            if width < minimum_width:
                width, height = minimum_width, minimum_width / aspect
            elif height < minimum_height:
                width, height = minimum_height * aspect, minimum_height
            if width > display.width() or height > display.height():
                scale = min(display.width() / width, display.height() / height)
                width, height = width * scale, height * scale
        else:
            width = min(max(self.MIN_RECT_SIZE, width), display.width())
            height = min(max(self.MIN_RECT_SIZE, height), display.height())
        width = min(width, display.width())
        height = min(height, display.height())
        left = max(display.left(), min(display.right() - width, rect.left()))
        top = max(display.top(), min(display.bottom() - height, rect.top()))
        return QRectF(left, top, width, height)

    def _move_rect(self, point: QPointF) -> QRectF:
        assert self.start_rect is not None and self.press_point is not None
        return self._clamp_rect(self.start_rect.translated(point - self.press_point))

    def _resize_free(self, point: QPointF) -> QRectF:
        assert self.start_rect is not None
        start = self.start_rect
        left, right, top, bottom = start.left(), start.right(), start.top(), start.bottom()
        minimum = self.MIN_RECT_SIZE
        if self.interaction in {"resize_top_left", "resize_left", "resize_bottom_left"}:
            left = min(point.x(), right - minimum)
        if self.interaction in {"resize_top_right", "resize_right", "resize_bottom_right"}:
            right = max(point.x(), left + minimum)
        if self.interaction in {"resize_top_left", "resize_top", "resize_top_right"}:
            top = min(point.y(), bottom - minimum)
        if self.interaction in {"resize_bottom_left", "resize_bottom", "resize_bottom_right"}:
            bottom = max(point.y(), top + minimum)
        return self._clamp_rect(QRectF(left, top, right - left, bottom - top))

    def _resize_camera(self, point: QPointF) -> QRectF:
        assert self.start_rect is not None
        start = self.start_rect
        aspect = self.camera_aspect
        if self.interaction in {"resize_left", "resize_right"}:
            width = max(self.MIN_RECT_SIZE, abs((start.right() if self.interaction == "resize_left" else start.left()) - point.x()))
            height = width / aspect
            center_y = start.center().y()
            left = start.right() - width if self.interaction == "resize_left" else start.left()
            return self._clamp_rect(QRectF(left, center_y - height / 2, width, height), True)
        if self.interaction in {"resize_top", "resize_bottom"}:
            height = max(self.MIN_RECT_SIZE, abs((start.bottom() if self.interaction == "resize_top" else start.top()) - point.y()))
            width = height * aspect
            center_x = start.center().x()
            top = start.bottom() - height if self.interaction == "resize_top" else start.top()
            return self._clamp_rect(QRectF(center_x - width / 2, top, width, height), True)

        if self.interaction == "resize_top_left":
            anchor, raw_width, raw_height, signs = QPointF(start.right(), start.bottom()), start.right() - point.x(), start.bottom() - point.y(), (-1, -1)
        elif self.interaction == "resize_top_right":
            anchor, raw_width, raw_height, signs = QPointF(start.left(), start.bottom()), point.x() - start.left(), start.bottom() - point.y(), (1, -1)
        elif self.interaction == "resize_bottom_left":
            anchor, raw_width, raw_height, signs = QPointF(start.right(), start.top()), start.right() - point.x(), point.y() - start.top(), (-1, 1)
        else:
            anchor, raw_width, raw_height, signs = QPointF(start.left(), start.top()), point.x() - start.left(), point.y() - start.top(), (1, 1)
        width = max(self.MIN_RECT_SIZE, raw_width, raw_height * aspect)
        max_width = (anchor.x() - self.displayed_image_rect.left()) if signs[0] < 0 else (self.displayed_image_rect.right() - anchor.x())
        max_height = (anchor.y() - self.displayed_image_rect.top()) if signs[1] < 0 else (self.displayed_image_rect.bottom() - anchor.y())
        width = min(width, max_width, max_height * aspect)
        width = max(1.0, width)
        height = width / aspect
        left = anchor.x() - width if signs[0] < 0 else anchor.x()
        top = anchor.y() - height if signs[1] < 0 else anchor.y()
        return self._clamp_rect(QRectF(left, top, width, height), True)

    def _resize_rect(self, point: QPointF) -> QRectF:
        return self._resize_camera(point) if self.camera_mode else self._resize_free(point)

    def _cursor_for(self, point: QPointF) -> Qt.CursorShape:
        handle = self._handle_at(point)
        if handle in {"resize_top_left", "resize_bottom_right"}:
            return Qt.CursorShape.SizeFDiagCursor
        if handle in {"resize_top_right", "resize_bottom_left"}:
            return Qt.CursorShape.SizeBDiagCursor
        if handle in {"resize_left", "resize_right"}:
            return Qt.CursorShape.SizeHorCursor
        if handle in {"resize_top", "resize_bottom"}:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeAllCursor if self._object_at(point) >= 0 else Qt.CursorShape.ArrowCursor

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eeeeee"))
        self.displayed_image_rect = self._display_rect()
        if not self.displayed_image_rect.isEmpty():
            painter.drawPixmap(self.displayed_image_rect.toRect(), self._pixmap)
        indexes = [index for index in range(len(self.objects)) if index != self.selected]
        if 0 <= self.selected < len(self.objects):
            indexes.append(self.selected)
        for index in indexes:
            rect = self._active_rect(self.objects[index])
            if rect is None or self.displayed_image_rect.isEmpty():
                continue
            pixels = self._rect_pixels(rect, self.displayed_image_rect)
            selected = index == self.selected
            painter.setPen(QPen(QColor("#ff4d4d") if selected else QColor("#36a269"), 2 if selected else 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(pixels)
            painter.drawText(pixels.topLeft() + QPointF(3, 15), self.objects[index].get("id", ""))
            if selected:
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QPen(QColor("#ff4d4d"), 1))
                for handle in self._handle_rects(pixels).values():
                    painter.drawRect(handle)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.displayed_image_rect = self._display_rect()
        point = event.position()
        if self.displayed_image_rect.isEmpty() or not self.displayed_image_rect.contains(point):
            self.interaction = "idle"; self.setCursor(Qt.CursorShape.ArrowCursor); return
        handle = self._handle_at(point)
        index = self.selected if handle else self._object_at(point)
        if index < 0:
            self.selected = -1; self.selection_changed.emit(-1); self.interaction = "idle"; self.update(); return
        if index != self.selected:
            self.selected = index; self.selection_changed.emit(index); handle = self._handle_at(point)
        rect = self._active_rect(self.objects[index])
        if rect is None:
            return
        self.start_rect = self._rect_pixels(rect, self.displayed_image_rect)
        self.press_point = point
        self.interaction = handle or "moving_object"
        self.setCursor(self._cursor_for(point)); self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self.displayed_image_rect = self._display_rect()
        point = event.position()
        if self.interaction == "idle":
            self.setCursor(self._cursor_for(point)); return
        if self.selected < 0 or self.start_rect is None:
            return
        point = self._clamp_point(point)
        pixels = self._move_rect(point) if self.interaction == "moving_object" else self._resize_rect(point)
        normalized = self._normalized(pixels, self.displayed_image_rect)
        key = self._active_key()
        self.objects[self.selected][key] = normalized
        self.geometry_changed.emit(self.selected, key, normalized)
        self.changed.emit(self.objects); self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.interaction = "idle"; self.start_rect = None; self.press_point = None
            self.setCursor(self._cursor_for(event.position())); self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self.interaction == "idle":
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)


class DrawObjectEditorDialog(QDialog):
    def __init__(self, images: list[Path], scene_path: Path, canvas_size: tuple[int, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Draw Objects")
        self.resize(900, 650)
        self.images = images
        self.scene_path = scene_path
        self.canvas_size = canvas_size
        self.records: dict[str, dict] = {}
        self.current_image = 0
        self.current_object = -1
        self.camera_mode = False
        self._load_records()
        root = QVBoxLayout(self)
        image_row = QHBoxLayout()
        self.image_combo = QComboBox(); self.image_combo.addItems([path.name for path in images]); self.image_combo.currentIndexChanged.connect(self._image_changed)
        image_row.addWidget(QLabel("Image")); image_row.addWidget(self.image_combo, 1)
        root.addLayout(image_row)
        body = QHBoxLayout()
        self.object_list = QListWidget(); self.object_list.currentRowChanged.connect(self._object_changed); body.addWidget(self.object_list, 0)
        self.canvas = DrawCanvas(images[0] if images else Path(), canvas_size[0] / canvas_size[1]); self.canvas.selection_changed.connect(self._canvas_selected); self.canvas.geometry_changed.connect(self._canvas_geometry_changed); body.addWidget(self.canvas, 1)
        controls = QVBoxLayout()
        self.name = QLineEdit(); self.name.editingFinished.connect(self._rename)
        self.kind = QComboBox(); self.kind.addItems(["art", "text", "warning"]); self.kind.currentTextChanged.connect(self._kind_changed)
        controls.addWidget(QLabel("Name")); controls.addWidget(self.name); controls.addWidget(QLabel("Type")); controls.addWidget(self.kind)
        add = QPushButton("Add Object"); add.clicked.connect(self._add); delete = QPushButton("Delete Object"); delete.clicked.connect(self._delete)
        up = QPushButton("Move Up"); up.clicked.connect(lambda: self._move(-1)); down = QPushButton("Move Down"); down.clicked.connect(lambda: self._move(1))
        self.camera = QCheckBox("Edit camera frame"); self.camera.toggled.connect(self._camera_toggled)
        for button in (add, delete, up, down): controls.addWidget(button)
        controls.addWidget(self.camera); controls.addStretch(1); body.addLayout(controls, 0)
        root.addLayout(body, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save); buttons.rejected.connect(self.reject); root.addWidget(buttons)
        self._image_changed(0)

    def _load_records(self) -> None:
        if self.scene_path.is_file():
            document = load_scene(self.scene_path)
            for key, image in document.images.items():
                self.records[key.casefold()] = {"filename": key, "size": image.source_size, "hash": image.source_sha256, "objects": [{"id": obj.id, "type": obj.type, "box": obj.box, "camera": obj.camera_frame, "render_effect": obj.render_effect, "direction": obj.direction, "duration_us": obj.duration_us, "pause_after_us": obj.pause_after_us, "behavior_fields_present": obj.behavior_fields_present} for obj in image.objects], "order": list(image.draw_order)}
        for image in self.images:
            key = image.name.casefold()
            if key not in self.records:
                with Image.open(image) as source:
                    size = source.size
                self.records[key] = {"filename": image.name, "size": size, "hash": sha256_file(image), "objects": [], "order": []}

    def _record(self) -> dict:
        return self.records[self.images[self.current_image].name.casefold()]

    @staticmethod
    def _object_label(item: dict) -> str:
        cam_str = "Camera ✓" if item.get("camera") is not None else "Camera —"
        return f"{item['id']}  ({cam_str})"

    def _refresh_object_labels(self) -> None:
        record = self._record()
        for idx, item in enumerate(record.get("objects", [])):
            if idx < self.object_list.count():
                self.object_list.item(idx).setText(self._object_label(item))

    def _image_changed(self, index: int) -> None:
        if not self.images:
            return
        self.current_image = index
        self.canvas.set_image(self.images[index])
        self.object_list.clear()
        record = self._record()
        self.object_list.addItems([self._object_label(item) for item in record["objects"]])
        self.current_object = -1
        self._object_changed(-1)
        self._sync_canvas()

    def _object_changed(self, index: int) -> None:
        self.current_object = index
        record = self._record()
        if 0 <= index < len(record["objects"]):
            item = record["objects"][index]
            if self.camera_mode and item["camera"] is None:
                self._ensure_camera_frame(item)
                self._refresh_object_labels()
            self.name.setText(item["id"]); self.kind.setCurrentText(item["type"])
        else:
            self.name.clear()
        self._sync_canvas()

    def _sync_canvas(self, *_args) -> None:
        if self.images:
            self.canvas.set_objects(self._record()["objects"], self.current_object, self.camera_mode)

    def _canvas_selected(self, index: int) -> None:
        self.current_object = index
        if self.object_list.currentRow() != index:
            self.object_list.setCurrentRow(index)
        else:
            self._object_changed(index)

    def _canvas_geometry_changed(self, index: int, key: str, rect: NormalizedRect) -> None:
        record = self._record()
        if 0 <= index < len(record["objects"]):
            record["objects"][index][key] = rect
            if key == "camera":
                self._refresh_object_labels()

    def _rename(self) -> None:
        if not (0 <= self.current_object < len(self._record()["objects"])):
            return
        value = self.name.text().strip()
        if not value or any(item["id"] == value for index, item in enumerate(self._record()["objects"]) if index != self.current_object):
            return
        old = self._record()["objects"][self.current_object]["id"]
        self._record()["objects"][self.current_object]["id"] = value
        self._record()["order"] = [value if item == old else item for item in self._record()["order"]]
        self.object_list.item(self.current_object).setText(self._object_label(self._record()["objects"][self.current_object]))

    def _kind_changed(self, value: str) -> None:
        if 0 <= self.current_object < len(self._record()["objects"]):
            self._record()["objects"][self.current_object]["type"] = value

    def _add(self) -> None:
        record = self._record()
        base = "object"
        number = 1
        while any(item["id"] == f"{base}_{number}" for item in record["objects"]):
            number += 1
        item = {"id": f"{base}_{number}", "type": "art", "box": NormalizedRect(0.35, 0.35, 0.3, 0.3), "camera": None, "behavior_fields_present": frozenset()}
        record["objects"].append(item); record["order"].append(item["id"]); self.object_list.addItem(self._object_label(item)); self.object_list.setCurrentRow(len(record["objects"]) - 1)

    def _delete(self) -> None:
        if 0 <= self.current_object < len(self._record()["objects"]):
            deleted = self._record()["objects"].pop(self.current_object)["id"]; self._record()["order"] = [item for item in self._record()["order"] if item != deleted]; self._image_changed(self.current_image)

    def _move(self, delta: int) -> None:
        record = self._record()
        if not (0 <= self.current_object < len(record["objects"])):
            return
        current = record["objects"][self.current_object]["id"]
        position = record["order"].index(current)
        target = position + delta
        if 0 <= target < len(record["order"]):
            record["order"][position], record["order"][target] = record["order"][target], record["order"][position]
            by_id = {item["id"]: item for item in record["objects"]}
            record["objects"][:] = [by_id[item_id] for item_id in record["order"]]
            self._image_changed(self.current_image)
            self.object_list.setCurrentRow(target)

    def _ensure_camera_frame(self, item: dict) -> None:
        box = item["box"]
        aspect = self.canvas_size[0] / self.canvas_size[1]
        source_aspect = self.canvas.source_aspect()
        width = min(1.0, max(box.w, box.h * aspect / source_aspect))
        height = min(1.0, width * source_aspect / aspect)
        x = min(max(0.0, box.center_x - width / 2), 1.0 - width)
        y = min(max(0.0, box.center_y - height / 2), 1.0 - height)
        item["camera"] = NormalizedRect(x, y, width, height)

    def _camera_toggled(self, value: bool) -> None:
        self.camera_mode = value
        if value and 0 <= self.current_object < len(self._record()["objects"]):
            self._ensure_camera_frame(self._record()["objects"][self.current_object])
        self._sync_canvas()

    def _save(self) -> None:
        images: dict[str, SceneImage] = {}
        for record in self.records.values():
            objects = tuple(SceneObject(item["id"], item["type"], item["box"], item["camera"], item.get("render_effect", "draw"), item.get("direction", "auto"), item.get("duration_us"), item.get("pause_after_us"), frozenset(item.get("behavior_fields_present", ()))) for item in record["objects"])
            images[record["filename"]] = SceneImage(record["filename"], record["size"], objects, tuple(record["order"]), record["hash"])
        save_scene(SceneDocument(1, images, self.scene_path), self.scene_path)
        self.accept()



class DrawPreviewDialog(QDialog):
    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw Preview")
        self.resize(900, 600)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self); self.player.setVideoOutput(self.video)
        layout = QVBoxLayout(self); layout.addWidget(self.video, 1)
        close = QPushButton("Close"); close.clicked.connect(self.accept); layout.addWidget(close)
        self.player.setSource(QUrl.fromLocalFile(str(path))); self.player.play()


class DrawAnimationWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.settings = QSettings("AutoCapCut", "AutoCapCutDraw")
        self.thread: QThread | None = None
        self.worker: DrawWorker | None = None
        self.images: list[Path] = []
        self.effect: DrawEffectFile | None = None
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self); root.setContentsMargins(18, 14, 18, 18)
        title = QLabel("DRAW ANIMATION"); title.setStyleSheet("font-size: 22px; font-weight: 700; letter-spacing: 2px;"); root.addWidget(title)
        project = QGroupBox("PROJECT"); form = QFormLayout(project)
        self.resolution = QComboBox(); self.resolution.addItems(list(RESOLUTIONS)); self.fps = QComboBox(); self.fps.addItems(["24", "30", "60"]); self.fps.setCurrentText("30")
        self.remove_background = QCheckBox("Remove simple background"); self.fallback_basic = QCheckBox("Fallback invalid advanced scenes to basic"); self.fallback_basic.setChecked(True); self.advanced_diagnostics = QCheckBox("Write advanced scheduling diagnostics")
        form.addRow("Resolution", self.resolution); form.addRow("FPS", self.fps); form.addRow(self.remove_background); form.addRow(self.fallback_basic); form.addRow(self.advanced_diagnostics); root.addWidget(project)
        images = QGroupBox("IMAGES"); image_form = QFormLayout(images)
        self.image_folder = QLineEdit(); browse_images = QPushButton("Browse"); browse_images.clicked.connect(self._browse_images); image_row = QHBoxLayout(); image_row.addWidget(self.image_folder); image_row.addWidget(browse_images); image_form.addRow("Image Folder", image_row)
        self.image_list = QListWidget(); self.image_list.currentRowChanged.connect(lambda *_: self._update_status()); image_form.addRow(self.image_list); root.addWidget(images)
        effects = QGroupBox("DRAW EFFECTS"); effect_form = QFormLayout(effects)
        self.effect_path = QLineEdit(); load_effect = QPushButton("Load Draw Effect File"); load_effect.clicked.connect(self._load_effect); effect_row = QHBoxLayout(); effect_row.addWidget(self.effect_path); effect_row.addWidget(load_effect); effect_form.addRow(effect_row)
        self.scene_path = QLineEdit(); load_scene_button = QPushButton("Load Scene JSON"); load_scene_button.clicked.connect(self._load_scene); scene_row = QHBoxLayout(); scene_row.addWidget(self.scene_path); scene_row.addWidget(load_scene_button); effect_form.addRow(scene_row)
        self.edit_objects = QPushButton("Edit Objects"); self.edit_objects.clicked.connect(self._edit_objects); effect_form.addRow(self.edit_objects); self.effect_status = QLabel("Draw effect: not loaded"); self.effect_status.setWordWrap(True); effect_form.addRow(self.effect_status); root.addWidget(effects)
        output = QGroupBox("OUTPUT"); output_form = QFormLayout(output); self.output_path = QLineEdit(); browse_output = QPushButton("Browse"); browse_output.clicked.connect(lambda: self._browse_folder(self.output_path)); output_row = QHBoxLayout(); output_row.addWidget(self.output_path); output_row.addWidget(browse_output); output_form.addRow("Output Folder", output_row); root.addWidget(output)
        actions = QGroupBox("ACTIONS"); action_row = QHBoxLayout(actions)
        self.preview_button = QPushButton("Preview Current Image"); self.preview_button.clicked.connect(self._preview); self.render_current = QPushButton("Render Current"); self.render_current.clicked.connect(lambda: self._render(True)); self.render_all = QPushButton("Render All"); self.render_all.clicked.connect(lambda: self._render(False)); self.open_output = QPushButton("Open Output Folder"); self.open_output.clicked.connect(self._open_output)
        for button in (self.preview_button, self.render_current, self.render_all, self.open_output): action_row.addWidget(button)
        root.addWidget(actions); self.progress = QLabel("Ready"); self.progress.setWordWrap(True); root.addWidget(self.progress); root.addStretch(1)

    def _browse_folder(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path: target.setText(path)

    def _browse_images(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path: return
        self.image_folder.setText(path); self._reload_images()

    def _reload_images(self) -> None:
        self.image_list.clear(); self.images = []
        try: self.images = collect_images([Path(self.image_folder.text())])
        except Exception as exc: self.progress.setText(str(exc)); return
        for image in self.images: self.image_list.addItem(f"{len(self.image_list)+1:03d}  {image.name}")
        if self.images: self.image_list.setCurrentRow(0)
        self._update_status()

    def _load_effect(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select draw effect file", filter="SRT files (*.srt)")
        if not path: return
        self.effect_path.setText(path)
        try: self.effect = parse_draw_effect(path); self._update_status()
        except Exception as exc: self.effect = None; self.effect_status.setText(f"Draw effect: {exc}")

    def _load_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select scene JSON", filter="JSON files (*.json)")
        if path: self.scene_path.setText(path); self._update_status()

    def _update_status(self) -> None:
        if self.effect is None and self.effect_path.text():
            try: self.effect = parse_draw_effect(self.effect_path.text())
            except Exception as exc: self.effect_status.setText(f"Draw effect: {exc}"); return
        if self.effect:
            count = len(self.images); cues = len(self.effect.images); status = f"Draw effect: {count} images / {cues} cues {'✓' if count == cues else '— mismatch'}"
            if self.effect.warnings:
                status += "\nWarnings: " + "; ".join(self.effect.warnings)
            self.effect_status.setText(status)
        else: self.effect_status.setText("Draw effect: not loaded")

    def _config(self, output: Path | None = None) -> DrawProjectConfig:
        if not self.images: raise ValueError("Choose an image folder")
        if self.effect is None: self.effect = parse_draw_effect(self.effect_path.text())
        destination_text = str(output) if output is not None else self.output_path.text().strip()
        if not destination_text: raise ValueError("Choose an output folder")
        destination = Path(destination_text)
        return DrawProjectConfig(Path(self.image_folder.text()), Path(self.effect_path.text()), destination, Path(self.scene_path.text()) if self.scene_path.text() else None, (RESOLUTIONS[self.resolution.currentText()].width, RESOLUTIONS[self.resolution.currentText()].height), int(self.fps.currentText()), self.remove_background.isChecked(), self.fallback_basic.isChecked(), self.advanced_diagnostics.isChecked())

    def _edit_objects(self) -> None:
        if not self.images: self._reload_images()
        if not self.images: return
        path = Path(self.scene_path.text()) if self.scene_path.text() else Path(self.image_folder.text()) / "scene.json"
        try: dialog = DrawObjectEditorDialog(self.images, path, (RESOLUTIONS[self.resolution.currentText()].width, RESOLUTIONS[self.resolution.currentText()].height), self); dialog.exec(); self.scene_path.setText(str(path)); self._update_status()
        except Exception as exc: QMessageBox.critical(self, "Scene JSON", str(exc))

    def _render(self, current: bool) -> None:
        try: config = self._config(); selection = self.image_list.currentRow() if current else None
        except Exception as exc: QMessageBox.warning(self, "Draw Animation", str(exc)); return
        self._save_settings(); self._set_busy(True); self.thread = QThread(self); self.worker = DrawWorker(config, selection); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(lambda value, message: self.progress.setText(f"{value}% · {message}")); self.worker.finished.connect(self._render_finished); self.worker.failed.connect(self._render_failed); self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.finished.connect(self._thread_finished); self.thread.start()

    def _render_finished(self, outputs) -> None: self.progress.setText(f"Rendered {len(outputs)} clip(s): {', '.join(path.name for path in outputs)}")
    def _render_failed(self, message: str) -> None: self.progress.setText(f"Error: {message}"); QMessageBox.critical(self, "Draw Animation", message)
    def _thread_finished(self) -> None: self.thread = None; self.worker = None; self._set_busy(False)
    def _set_busy(self, busy: bool) -> None:
        for button in (self.preview_button, self.render_current, self.render_all, self.edit_objects): button.setEnabled(not busy)

    def _preview(self) -> None:
        index = self.image_list.currentRow()
        if index < 0:
            QMessageBox.warning(self, "Preview", "Select an image first."); return
        try:
            config = self._config(); effect = self.effect or parse_draw_effect(config.effect_file); plan = effect.images[index]; scene = None
            if plan.mode is DrawMode.ADVANCED and config.scene_file and config.scene_file.is_file():
                document = load_scene(config.scene_file); scene = next((item for key, item in document.images.items() if key.casefold() == self.images[index].name.casefold()), None)
            scene_signature = sha256_file(config.scene_file) if config.scene_file and config.scene_file.is_file() else "none"
            signature = f"{sha256_file(config.effect_file)}-{scene_signature}-{index}-{config.resolution}-{config.fps}"
            output = config.output_folder / ".autocapcut_draw_cache" / "previews" / f"{index + 1:03d}-{signature[:16]}.mp4"; output.parent.mkdir(parents=True, exist_ok=True)
            scale = min(1.0, 960 / config.resolution[0], 540 / config.resolution[1]); preview_size = (max(2, round(config.resolution[0] * scale)), max(2, round(config.resolution[1] * scale)))
            DrawRenderer(config.output_folder / ".autocapcut_draw_cache").render(self.images[index], plan, DrawProjectConfig(config.image_folder, config.effect_file, config.output_folder, config.scene_file, preview_size, min(30, config.fps), config.remove_background, config.fallback_basic, config.advanced_diagnostics), output, scene, lambda value, message: self.progress.setText(f"Preview {value}% · {message}"))
            DrawPreviewDialog(output, self).exec()
        except Exception as exc: QMessageBox.critical(self, "Preview", str(exc))

    def _open_output(self) -> None:
        path = Path(self.output_path.text())
        if path.is_dir(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_settings(self) -> None:
        for key, value in {"image_folder": self.image_folder.text(), "effect_path": self.effect_path.text(), "scene_path": self.scene_path.text(), "output_path": self.output_path.text(), "resolution": self.resolution.currentText(), "fps": self.fps.currentText(), "remove_background": self.remove_background.isChecked(), "fallback_basic": self.fallback_basic.isChecked(), "advanced_diagnostics": self.advanced_diagnostics.isChecked()}.items(): self.settings.setValue(key, value)

    def _load_settings(self) -> None:
        for key, widget in (("image_folder", self.image_folder), ("effect_path", self.effect_path), ("scene_path", self.scene_path), ("output_path", self.output_path)):
            widget.setText(str(self.settings.value(key, "")))
        self.resolution.setCurrentText(str(self.settings.value("resolution", "1920x1080"))); self.fps.setCurrentText(str(self.settings.value("fps", "30"))); self.remove_background.setChecked(self.settings.value("remove_background", False, type=bool)); self.fallback_basic.setChecked(self.settings.value("fallback_basic", True, type=bool)); self.advanced_diagnostics.setChecked(self.settings.value("advanced_diagnostics", False, type=bool))
        if self.image_folder.text(): self._reload_images()
        if self.effect_path.text(): self._update_status()
