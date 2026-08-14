from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from auto_capcut.core.capcut_builder import CapCutBuilder
from auto_capcut.core.errors import AutoCapCutError
from auto_capcut.core.planning import create_jobs
from auto_capcut.models import ProjectConfig


class ProjectWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config

    @pyqtSlot()
    def run(self) -> None:
        try:
            jobs = create_jobs(self.config)
            builder = CapCutBuilder()
            results = []
            for index, job in enumerate(jobs):
                offset = int(index * 100 / max(1, len(jobs)))

                def report(value: int, message: str, offset=offset, count=len(jobs)) -> None:
                    self.progress.emit(min(99, offset + int(value / max(1, count))), message)

                results.append(builder.build_job(job, report))
            self.finished.emit(results)
        except AutoCapCutError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unable to create project: {exc}")
