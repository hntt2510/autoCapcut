from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import DrawProjectConfig
from auto_capcut.core.draw_renderer import DrawRenderService
from auto_capcut.core.media import collect_images


class DrawWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: DrawProjectConfig, selection: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.selection = selection

    @pyqtSlot()
    def run(self) -> None:
        try:
            images = collect_images([self.config.image_folder])
            effect = parse_draw_effect(self.config.effect_file)

            def report(value: int, message: str) -> None:
                self.progress.emit(value, message)

            outputs = DrawRenderService().render_project(self.config, effect, images, self.selection, report)
            self.finished.emit(outputs)
        except Exception as exc:
            self.failed.emit(str(exc))
