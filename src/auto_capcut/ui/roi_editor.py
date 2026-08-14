from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from auto_capcut.core.roi_resolver import ManualRoiResolver
from auto_capcut.models import EffectCue, MotionStrength, RoiRequirement, RoiTarget, TargetROI


class RoiPreview(QWidget):
    """Canvas-aspect-locked camera frame editor."""

    HANDLE_SIZE = 12

    def __init__(self, canvas_aspect: float = 16 / 9) -> None:
        super().__init__()
        self.setMinimumSize(520, 300)
        self.canvas_aspect = canvas_aspect
        self._source_aspect = 1.0
        self._pixmap = QPixmap()
        self._display_rect = QRect()
        self._selection: QRect | None = None
        self._roi: TargetROI | None = None
        self._interaction: str | None = None
        self._press_point = QPoint()
        self._start_selection = QRect()
        self._start_roi: TargetROI | None = None
        self.frame_changed = None

    def set_canvas_aspect(self, aspect: float) -> None:
        if aspect > 0:
            self.canvas_aspect = aspect
            self.update()

    def set_image(self, path: Path, roi: TargetROI | None) -> None:
        self._pixmap = QPixmap(str(path))
        if not self._pixmap.isNull() and self._pixmap.height() > 0:
            self._source_aspect = self._pixmap.width() / self._pixmap.height()
        self._roi = roi
        self._selection = None
        self._interaction = None
        self.update()

    def _layout_rect(self) -> QRect:
        if self._pixmap.isNull():
            return QRect()
        size = self._pixmap.size()
        size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect((self.width() - size.width()) // 2, (self.height() - size.height()) // 2, size.width(), size.height())

    def _ensure_selection(self) -> None:
        if self._selection is not None or self._roi is None or not self._display_rect.isValid():
            return
        self._selection = QRect(
            round(self._roi.x * self._display_rect.width()),
            round(self._roi.y * self._display_rect.height()),
            max(1, round(self._roi.width * self._display_rect.width())),
            max(1, round(self._roi.height * self._display_rect.height())),
        )

    def frame(self) -> TargetROI | None:
        if self._selection is None or not self._display_rect.isValid() or self._selection.width() <= 0 or self._selection.height() <= 0:
            return self._roi
        rect = self._selection.intersected(QRect(0, 0, self._display_rect.width(), self._display_rect.height()))
        return TargetROI(
            max(0.0, min(1.0, rect.left() / self._display_rect.width())),
            max(0.0, min(1.0, rect.top() / self._display_rect.height())),
            max(0.000001, min(1.0, rect.width() / self._display_rect.width())),
            max(0.000001, min(1.0, rect.height() / self._display_rect.height())),
        )

    # Compatibility for existing callers.
    roi = frame

    def reset(self) -> None:
        self._selection = None
        self._roi = None
        self._interaction = None
        self.update()

    def _point_inside(self, point: QPoint) -> bool:
        return self._display_rect.contains(point)

    def _clamp_point(self, point: QPoint) -> QPoint:
        return QPoint(
            max(0, min(self._display_rect.width(), point.x())),
            max(0, min(self._display_rect.height(), point.y())),
        )

    def _handle_at(self, point: QPoint) -> str | None:
        if self._selection is None:
            return None
        rect = self._selection
        handles = {
            "top_left": QPoint(rect.left(), rect.top()),
            "top_right": QPoint(rect.right(), rect.top()),
            "bottom_left": QPoint(rect.left(), rect.bottom()),
            "bottom_right": QPoint(rect.right(), rect.bottom()),
        }
        radius = self.HANDLE_SIZE
        for name, handle in handles.items():
            if abs(point.x() - handle.x()) <= radius and abs(point.y() - handle.y()) <= radius:
                return name
        return None

    def _aspect_size(self, width: int, height: int, prefer_width: bool = True) -> tuple[int, int]:
        # The preview itself is rendered at the source image aspect.  Therefore
        # the displayed rectangle ratio is the desired output/canvas ratio.
        # Conversion back to normalized values then preserves the same ratio in
        # source pixel coordinates.
        preview_aspect = self.canvas_aspect
        width = max(self.HANDLE_SIZE, width)
        height = max(self.HANDLE_SIZE, height)
        if prefer_width:
            height = max(self.HANDLE_SIZE, round(width / preview_aspect))
        else:
            width = max(self.HANDLE_SIZE, round(height * preview_aspect))
        return width, height

    def _fit_rect_to_bounds(self, rect: QRect) -> QRect:
        """Uniformly shrink if necessary, otherwise translate without distortion."""
        max_width = self._display_rect.width()
        max_height = self._display_rect.height()
        if rect.width() > max_width or rect.height() > max_height:
            scale = min(max_width / rect.width(), max_height / rect.height())
            rect.setSize(rect.size() * scale)
        rect.moveLeft(max(0, min(max_width - rect.width(), rect.left())))
        rect.moveTop(max(0, min(max_height - rect.height(), rect.top())))
        return rect

    def _create_rect(self, anchor: QPoint, point: QPoint) -> QRect:
        dx = point.x() - anchor.x()
        dy = point.y() - anchor.y()
        width, height = self._aspect_size(abs(dx), abs(dy), abs(dx) >= abs(dy) * self.canvas_aspect)
        x = anchor.x() + (width if dx < 0 else 0)
        y = anchor.y() + (height if dy < 0 else 0)
        rect = QRect(x, y, width, height).normalized()
        return self._fit_rect_to_bounds(rect)

    def _move_rect(self, point: QPoint) -> QRect:
        assert self._selection is not None
        rect = QRect(self._start_selection).translated(point - self._press_point)
        return self._fit_rect_to_bounds(rect)

    def _resize_rect(self, point: QPoint) -> QRect:
        assert self._selection is not None
        start = self._start_selection
        handle = self._interaction or "bottom_right"
        if handle == "top_left":
            opposite = QPoint(start.right(), start.bottom())
            width, height = self._aspect_size(abs(opposite.x() - point.x()), abs(opposite.y() - point.y()), True)
            point = QPoint(opposite.x() - width, opposite.y() - height)
            rect = QRect(point, opposite).normalized()
        elif handle == "top_right":
            opposite = QPoint(start.left(), start.bottom())
            width, height = self._aspect_size(abs(point.x() - opposite.x()), abs(opposite.y() - point.y()), True)
            rect = QRect(opposite.x(), opposite.y() - height, width, height)
        elif handle == "bottom_left":
            opposite = QPoint(start.right(), start.top())
            width, height = self._aspect_size(abs(opposite.x() - point.x()), abs(point.y() - opposite.y()), True)
            rect = QRect(opposite.x() - width, opposite.y(), width, height)
        else:
            opposite = QPoint(start.left(), start.top())
            width, height = self._aspect_size(abs(point.x() - opposite.x()), abs(point.y() - opposite.y()), True)
            rect = QRect(opposite.x(), opposite.y(), width, height)
        return self._fit_rect_to_bounds(rect.normalized())

    def _notify_changed(self) -> None:
        if callable(self.frame_changed):
            self.frame_changed(self.frame())

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        self._display_rect = self._layout_rect()
        if not self._display_rect.isValid():
            return
        painter.drawPixmap(self._display_rect, self._pixmap)
        self._ensure_selection()
        if self._selection and self._selection.isValid():
            shifted = self._selection.translated(self._display_rect.topLeft())
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            painter.drawRect(shifted)
            painter.setBrush(Qt.GlobalColor.yellow)
            for point in (shifted.topLeft(), shifted.topRight(), shifted.bottomLeft(), shifted.bottomRight()):
                painter.drawRect(QRect(point.x() - 4, point.y() - 4, 8, 8))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position().toPoint() - self._display_rect.topLeft()
        if not self._point_inside(point):
            return
        self._ensure_selection()
        handle = self._handle_at(point)
        if handle:
            self._interaction = handle
        elif self._selection and self._selection.contains(point):
            self._interaction = "move"
        else:
            self._interaction = "create"
            self._roi = None
            self._selection = QRect(point, point)
        self._press_point = point
        self._start_selection = QRect(self._selection)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._interaction:
            return
        point = self._clamp_point(event.position().toPoint() - self._display_rect.topLeft())
        if self._interaction == "create":
            self._selection = self._create_rect(self._press_point, point)
        elif self._interaction == "move":
            self._selection = self._move_rect(point)
        else:
            self._selection = self._resize_rect(point)
        self._notify_changed()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._interaction = None
            self._notify_changed()


class RoiEditorDialog(QDialog):
    def __init__(self, images: list[Path], effects: list[EffectCue], sidecar_path: Path, parent=None, canvas_size: tuple[int, int] = (1920, 1080), strength: str = MotionStrength.SUBTLE.value) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Camera Frames")
        self.resize(700, 650)
        self.images = images
        self.effects = effects
        self.canvas_size = canvas_size
        self.strength = strength
        self.resolver = ManualRoiResolver(sidecar_path)
        self.queue: list[RoiTarget] = []
        for cue in effects:
            self.queue.extend(cue.required_roi_targets)
            self.queue.extend(cue.optional_roi_targets)
        deduped: dict[tuple[int, str], RoiTarget] = {}
        for item in self.queue:
            key = (item.image_index, item.target_id)
            current = deduped.get(key)
            if current is None:
                deduped[key] = item
            else:
                requirement = RoiRequirement.REQUIRED if RoiRequirement.REQUIRED in (current.requirement, item.requirement) else RoiRequirement.OPTIONAL
                effect_types = tuple(dict.fromkeys(current.effect_types + item.effect_types))
                deduped[key] = RoiTarget(item.image_index, item.target_id, effect_types, requirement)
        self.queue = sorted(deduped.values(), key=lambda item: (item.requirement is not RoiRequirement.REQUIRED, item.image_index, item.target_id))
        self.current = 0
        layout = QVBoxLayout(self)
        self.heading = QLabel()
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.preview = RoiPreview(canvas_size[0] / canvas_size[1])
        layout.addWidget(self.preview, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        buttons = QHBoxLayout()
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self._move(-1))
        next_missing = QPushButton("Next Missing Frame")
        next_missing.clicked.connect(self._next_missing)
        save_next = QPushButton("Save Frame & Next")
        save_next.clicked.connect(self._save_next)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        for button in (reset, previous, next_missing, save_next, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._show_current()

    def _show_current(self) -> None:
        if not self.queue:
            self.heading.setText("No camera frames required.")
            self.preview.set_image(Path(), None)
            self.info.setText("")
            return
        target = self.queue[self.current]
        image = self.images[target.image_index - 1]
        frame = self.resolver.resolve(image, target.target_id, target.image_index)
        effects = [effect.type for cue in self.effects if cue.image_index == target.image_index for effect in cue.effects if effect.target_id == target.target_id]
        status = "required" if target.requirement is RoiRequirement.REQUIRED else "optional"
        frame_state = "missing"
        reason = ""
        if frame is not None:
            try:
                from auto_capcut.core.roi_resolver import validate_saved_frame
                valid, reason = validate_saved_frame(frame, image, self.canvas_size)
                frame_state = "configured" if valid else "Needs reframing"
            except OSError as exc:
                reason = str(exc)
        self.heading.setText(f"Image {target.image_index:03d} · Frame {self.current + 1} / {len(self.queue)}\nEffect: {', '.join(effects)}\nTarget: {target.target_id} ({status})")
        if frame is not None:
            zoom = 1.0 / max(frame.width, frame.height)
            self.info.setText(f"Canvas aspect: {self.canvas_size[0]}:{self.canvas_size[1]} · {frame_state} · estimated zoom {zoom:.2f}x" + (f"\n{reason}" if reason and frame_state != "configured" else ""))
        else:
            self.info.setText(f"Canvas aspect: {self.canvas_size[0]}:{self.canvas_size[1]} · missing")
        self.preview.set_image(image, frame)

    def _save_current(self) -> None:
        if self.queue:
            target = self.queue[self.current]
            self.resolver.save(self.images[target.image_index - 1], target, self.preview.frame())

    def _save_next(self) -> None:
        self._save_current()
        if self.current < len(self.queue) - 1:
            self.current += 1
            self._show_current()
        else:
            self.accept()

    def _move(self, delta: int) -> None:
        self._save_current()
        if self.queue:
            self.current = max(0, min(len(self.queue) - 1, self.current + delta))
            self._show_current()

    def _next_missing(self) -> None:
        self._save_current()
        if not self.queue:
            return
        from auto_capcut.core.roi_resolver import validate_saved_frame
        for offset in range(1, len(self.queue) + 1):
            index = (self.current + offset) % len(self.queue)
            target = self.queue[index]
            frame = self.resolver.resolve(self.images[target.image_index - 1], target.target_id, target.image_index)
            if frame is None:
                self.current = index
                self._show_current()
                return
            valid, _ = validate_saved_frame(frame, self.images[target.image_index - 1], self.canvas_size)
            if not valid:
                self.current = index
                self._show_current()
                return

    def _reset(self) -> None:
        if self.queue:
            target = self.queue[self.current]
            self.resolver.save(self.images[target.image_index - 1], target, None)
            self.preview.reset()
            self._show_current()
