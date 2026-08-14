"""Background worker for the isolated effect catalog tester."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from auto_capcut.core.effect_catalog_builder import CatalogDraftBuilder
from auto_capcut.core.effect_catalog import CatalogStore
from auto_capcut.core.captured_effect_template import ResolvedCapturedEffectPreset
from auto_capcut.core.local_effect_promoter import LocalEffectTemplatePromoter, PromotionSummary
from auto_capcut.core.errors import AutoCapCutError


class EffectCatalogWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, entries=None, image_path=None, draft_folder=None, *, duration_seconds=2.0,
                 batch_size=30, width=1920, height=1080, fps=30,
                 project_prefix="EffectCatalog"):
        super().__init__()
        self.entries = entries
        self.image_path = image_path
        self.draft_folder = draft_folder
        self.duration_seconds = duration_seconds
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.fps = fps
        self.project_prefix = project_prefix
        self.operation = "build"

    @classmethod
    def for_promote(cls, entries, image_path, draft_folder, **kwargs):
        worker = cls(entries, image_path, draft_folder, **kwargs)
        worker.operation = "promote"
        return worker

    @classmethod
    def for_scan(cls, mode="py"):
        worker = cls()
        worker.operation = f"scan_{mode}"
        return worker

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.operation.startswith("scan"):
                store = CatalogStore()
                catalog = {"scan_py": store.scan, "scan_local": store.scan_local, "scan_all": store.scan_all}.get(self.operation, store.scan)()
                self.finished.emit(catalog)
                return
            if self.operation == "promote":
                promoter = LocalEffectTemplatePromoter()
                prepared = promoter.prepare(self.entries or [])
                buildable = [item for item in prepared if item.template is not None]
                if buildable:
                    templates = {
                        item.stable_key: ResolvedCapturedEffectPreset(
                            item.preset_key,
                            item.template.source_effect_id,
                            item.template,
                            Path("<pending>"),
                            "promoted",
                            item.stable_key,
                        )
                        for item in buildable
                    }
                    try:
                        result = CatalogDraftBuilder().build_catalog(
                            [entry for entry in self.entries if str(getattr(entry, "stable_key", "")) in templates],
                            self.image_path,
                            self.draft_folder,
                            duration_seconds=self.duration_seconds,
                            batch_size=self.batch_size,
                            width=self.width,
                            height=self.height,
                            fps=self.fps,
                            project_prefix=self.project_prefix,
                            progress_callback=lambda value, message: self.progress.emit(value, message),
                            captured_templates=templates,
                        )
                    except Exception:
                        promoter.persist(prepared, set())
                        raise
                    failed = {item.stable_key for item in result.failures}
                else:
                    result = None
                    failed = set()
                promoter.persist(prepared, {item.preset_key for item in buildable if item.stable_key not in failed})
                catalog_store = CatalogStore()
                for item in prepared:
                    try:
                        catalog_store.update(
                            item.stable_key,
                            validation_state="promoted" if item.template is not None and item.stable_key not in failed else "unresolved",
                            buildable=bool(item.template is not None and item.stable_key not in failed),
                            build_status="build_ok" if item.template is not None and item.stable_key not in failed else "build_failed",
                        )
                    except KeyError:
                        pass
                unresolved = tuple(item for item in prepared if item.template is None or item.stable_key in failed)
                self.finished.emit(PromotionSummary(tuple(prepared), result, unresolved))
                return
            builder = CatalogDraftBuilder()
            result = builder.build_catalog(
                self.entries,
                self.image_path,
                self.draft_folder,
                duration_seconds=self.duration_seconds,
                batch_size=self.batch_size,
                width=self.width,
                height=self.height,
                fps=self.fps,
                project_prefix=self.project_prefix,
                progress_callback=lambda value, message: self.progress.emit(value, message),
            )
            self.finished.emit(result)
        except AutoCapCutError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unable to build effect catalog: {exc}")
