from __future__ import annotations

"""Internal Effect Catalog Tester dialog.

This module is deliberately isolated from the production project worker.  The
catalog backend is imported lazily so the main application remains usable when
pyCapCut metadata is unavailable (for example, during source development).
"""

from pathlib import Path
from typing import Any, Iterable

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
        QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class EffectCatalogDialog(QDialog):
    """Search, review and build isolated drafts for installed CapCut effects."""

    HEADERS = ("Index", "Effect", "Source", "Effect ID", "Build", "Lifecycle", "Review", "Favorite", "Preset", "Preset State", "Preset Error")

    def __init__(self, parent=None, *, resolution_presets: Iterable[str] = (), test_image_path: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Effect Catalog Tester")
        self.setMinimumSize(860, 560)
        self._entries: list[Any] = []
        self._store = None
        self._catalog_error = ""
        self._thread = None
        self._worker = None
        self._resolution_presets = tuple(dict.fromkeys(str(value).strip() for value in resolution_presets if str(value).strip()))
        self._resolution_matches = []
        self._matched_entries: dict[str, Any] = {}
        self._test_image_path = Path(test_image_path) if test_image_path else None
        self._build_ui()
        if self._test_image_path:
            self.test_image.setText(str(self._test_image_path))
        self._load_catalog()
        if self._resolution_presets:
            self._load_resolution_matches()

    # -- UI -------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.scan_button = QPushButton("Scan pyCapCut Effects")
        self.scan_button.clicked.connect(self.scan_catalog)
        toolbar.addWidget(self.scan_button)
        self.scan_local_button = QPushButton("Scan CapCut Local Catalog")
        self.scan_local_button.clicked.connect(self.scan_local_catalog)
        toolbar.addWidget(self.scan_local_button)
        self.scan_all_button = QPushButton("Scan All Effects")
        self.scan_all_button.clicked.connect(self.scan_all_catalog)
        toolbar.addWidget(self.scan_all_button)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search effect name, enum, ID...")
        self.search.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self.search, 1)
        self.source_filter = QComboBox()
        self.source_filter.addItems(["All sources", "Scene", "Character", "Local"])
        self.source_filter.currentTextChanged.connect(self._apply_filter)
        toolbar.addWidget(self.source_filter)
        self.review_filter = QComboBox()
        self.review_filter.addItems(["All review states", "Untested", "CapCut OK", "CapCut Missing", "Approved", "Rejected", "Broken"])
        self.review_filter.currentTextChanged.connect(self._apply_filter)
        toolbar.addWidget(self.review_filter)
        self.build_filter = QComboBox()
        self.build_filter.addItems(["All build states", "unbuilt", "build_ok", "build_failed", "buildable", "discovery-only", "promoted", "template_captured", "draft_recognized", "render_unverified", "render_confirmed", "unresolved"])
        self.build_filter.currentTextChanged.connect(self._apply_filter)
        toolbar.addWidget(self.build_filter)
        layout.addLayout(toolbar)

        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        self.resolution_table = QTableWidget(0, 5)
        self.resolution_table.setHorizontalHeaderLabels(("Preset", "Catalog Effect", "Source", "Effect ID", "Status"))
        self.resolution_table.setVisible(bool(self._resolution_presets))
        self.resolution_label.setVisible(bool(self._resolution_presets))
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_table)

        self.table = QTableWidget(0, len(self.HEADERS) + 2)
        self.table.setHorizontalHeaderLabels(self.HEADERS + ("Validation", "Buildable"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        form = QFormLayout()
        self.test_image = QLineEdit()
        choose_image = QPushButton("Browse")
        choose_image.clicked.connect(self._browse_image)
        image_row = QHBoxLayout(); image_row.addWidget(self.test_image, 1); image_row.addWidget(choose_image)
        form.addRow("Test image", image_row)
        self.duration = QSpinBox(); self.duration.setRange(1, 60); self.duration.setValue(2); self.duration.setSuffix(" s")
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 30); self.batch_size.setValue(30)
        form.addRow("Segment duration", self.duration); form.addRow("Effects per draft", self.batch_size)
        self.notes = QLineEdit(); self.notes.setPlaceholderText("Notes for selected effect(s)")
        self.tags = QLineEdit(); self.tags.setPlaceholderText("Semantic tags, comma separated")
        form.addRow("Notes", self.notes); form.addRow("Tags", self.tags)
        self.review = QComboBox(); self.review.addItems(["Untested", "CapCut OK", "CapCut Missing", "Approved", "Rejected", "Broken"])
        self.favorite = QComboBox(); self.favorite.addItems(["Not favorite", "Favorite"])
        form.addRow("Review", self.review); form.addRow("Favorite", self.favorite)
        self.save_metadata = QPushButton("Save Review / Notes / Tags")
        self.save_metadata.clicked.connect(self._save_metadata)
        form.addRow(self.save_metadata)
        layout.addLayout(form)

        self.status = QLabel("Catalog not scanned")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.build_button = QPushButton("Build Draft")
        self.build_button.clicked.connect(self.build_draft)
        self.export_button = QPushButton("Export approved_effects.json")
        self.export_button.clicked.connect(self.export_approved)
        self.open_button = QPushButton("Open Catalog Folder")
        self.open_button.clicked.connect(self.open_catalog_folder)
        self.capture_template_button = QPushButton("Capture Warning Template")
        self.capture_template_button.setToolTip("Capture the Warning material from the read-only test_8 draft")
        self.capture_template_button.clicked.connect(self.capture_warning_template)
        self.promote_button = QPushButton("Promote Local Effects")
        self.promote_button.setToolTip("Reconstruct compatible local CapCut materials and build test drafts")
        self.promote_button.clicked.connect(self.promote_local_effects)
        if self._resolution_presets:
            self.promote_button.setText("Promote All Matched Effects")
        self.mark_recognized_button = QPushButton("Mark Draft Recognized")
        self.mark_recognized_button.clicked.connect(lambda: self.mark_lifecycle("draft_recognized"))
        self.mark_render_button = QPushButton("Mark Render Confirmed")
        self.mark_render_button.clicked.connect(lambda: self.mark_lifecycle("render_confirmed"))
        actions.addWidget(self.build_button); actions.addWidget(self.promote_button); actions.addWidget(self.export_button); actions.addWidget(self.open_button)
        actions.addWidget(self.capture_template_button)
        actions.addWidget(self.mark_recognized_button); actions.addWidget(self.mark_render_button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- Catalog operations --------------------------------------------
    def _store_instance(self):
        if self._store is not None:
            return self._store
        try:
            from auto_capcut.core.effect_catalog import CatalogStore
            self._store = CatalogStore()
        except Exception as exc:  # pragma: no cover - optional backend
            self._catalog_error = str(exc)
            self._store = None
        return self._store

    def _load_catalog(self) -> None:
        store = self._store_instance()
        if store is None:
            self.status.setText(f"Catalog unavailable: {self._catalog_error}")
            return
        try:
            loaded = store.load()
            try:
                from auto_capcut.core.captured_effect_template import CapturedEffectTemplateRepository
                CapturedEffectTemplateRepository().ensure_warning_bootstrap()
            except Exception:
                pass
            self._entries = list(loaded if isinstance(loaded, (list, tuple)) else getattr(loaded, "entries", loaded))
            self._populate()
            self.status.setText(f"{len(self._entries)} effects loaded")
        except Exception as exc:
            self.status.setText(f"Catalog load failed: {exc}")

    def scan_catalog(self) -> None:
        self._start_scan("py")

    def scan_local_catalog(self) -> None:
        self._start_scan("local")

    def scan_all_catalog(self) -> None:
        self._start_scan("all")

    def _start_scan(self, mode: str) -> None:
        store = self._store_instance()
        if store is None:
            self.status.setText(f"Catalog unavailable: {self._catalog_error}")
            return
        if self._thread is not None:
            return
        try:
            from auto_capcut.workers.effect_catalog_worker import EffectCatalogWorker
            self._thread = QThread(self)
            self._worker = EffectCatalogWorker.for_scan(mode)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._scan_finished)
            self._worker.failed.connect(lambda message: self.status.setText(f"Effect scan failed: {message}"))
            self._worker.finished.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._worker_done)
            self.scan_button.setEnabled(False); self.scan_local_button.setEnabled(False); self.scan_all_button.setEnabled(False)
            self.status.setText("Scanning CapCut effect catalog..."); self._thread.start()
        except Exception as exc:
            self.status.setText(f"Effect scan failed: {exc}")

    def _scan_finished(self, result) -> None:
        self._entries = list(result.entries if hasattr(result, "entries") else result)
        self._populate()
        if self._resolution_presets:
            self._load_resolution_matches()
        self.status.setText(f"{len(self._entries)} effects scanned")

    def _load_resolution_matches(self) -> None:
        store = self._store_instance()
        if store is None:
            return
        self._resolution_matches = store.match_preset_keys(self._resolution_presets)
        self._matched_entries = {}
        self.resolution_table.setRowCount(0)
        for match in self._resolution_matches:
            row = self.resolution_table.rowCount()
            self.resolution_table.insertRow(row)
            self.resolution_table.setItem(row, 0, QTableWidgetItem(match.preset_key))
            candidates = list(match.candidates)
            if match.status == "ambiguous":
                choice = QComboBox()
                choice.addItem("Choose a catalog effect", None)
                for candidate in candidates:
                    choice.addItem(f"{candidate.display_name} ({candidate.effect_id})", candidate)
                choice.currentIndexChanged.connect(lambda _index, key=match.preset_key, combo=choice: self._resolution_choice_changed(key, combo))
                self.resolution_table.setCellWidget(row, 1, choice)
            elif candidates:
                candidate = candidates[0]
                self.resolution_table.setItem(row, 1, QTableWidgetItem(candidate.display_name))
                self.resolution_table.setItem(row, 2, QTableWidgetItem("CapCut Local" if candidate.source.casefold() == "local" else str(candidate.source).title()))
                self.resolution_table.setItem(row, 3, QTableWidgetItem(candidate.effect_id))
                if match.unique:
                    self._matched_entries[match.preset_key] = candidate
            else:
                self.resolution_table.setItem(row, 1, QTableWidgetItem("—"))
            if candidates and match.status == "ambiguous":
                self.resolution_table.setItem(row, 2, QTableWidgetItem("CapCut Local"))
                self.resolution_table.setItem(row, 4, QTableWidgetItem(match.reason))
            elif candidates:
                candidate = candidates[0]
                resource = match.reason if match.status == "not_promotable" else ("resource available" if getattr(candidate, "local_resource_path", "") else "resource missing")
                self.resolution_table.setItem(row, 4, QTableWidgetItem(resource))
            else:
                self.resolution_table.setItem(row, 4, QTableWidgetItem(match.reason))
        matched = len(self._matched_entries)
        ambiguous = sum(match.status == "ambiguous" for match in self._resolution_matches)
        self.resolution_label.setText(f"Preset mappings: {matched} unique matched, {ambiguous} ambiguous; unmatched presets remain blocked.")

    def _resolution_choice_changed(self, preset_key: str, combo: QComboBox) -> None:
        candidate = combo.currentData()
        if candidate is None:
            self._matched_entries.pop(preset_key, None)
        else:
            self._matched_entries[preset_key] = candidate
        for row in range(self.resolution_table.rowCount()):
            if self.resolution_table.item(row, 0) and self.resolution_table.item(row, 0).text() == preset_key:
                if candidate is not None:
                    self.resolution_table.setItem(row, 2, QTableWidgetItem("CapCut Local" if candidate.source.casefold() == "local" else str(candidate.source).title()))
                    self.resolution_table.setItem(row, 3, QTableWidgetItem(candidate.effect_id))
                    self.resolution_table.setItem(row, 4, QTableWidgetItem("Selected; resource available" if getattr(candidate, "local_resource_path", "") else "Selected; resource missing"))
                break

    def _populate(self) -> None:
        self.table.setRowCount(0)
        for entry in self._entries:
            row = self.table.rowCount(); self.table.insertRow(row)
            values = [
                getattr(entry, "test_index", ""), getattr(entry, "display_name", getattr(entry, "enum_name", "")),
                str(getattr(entry, "source", "")).title(), getattr(entry, "effect_id", ""),
                getattr(entry, "build_status", ""),
                getattr(entry, "lifecycle", getattr(entry, "lifecycle_state", getattr(entry, "validation_state", "discovered"))),
                getattr(entry, "review_status", "Untested"),
                "★" if getattr(entry, "favorite", False) else "",
            ]
            preset_key, preset_state, preset_error = self._preset_info(entry)
            values.extend([preset_key, preset_state, preset_error, getattr(entry, "validation_state", ""), "Yes" if getattr(entry, "buildable", False) else "Discovery only"])
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); item.setData(Qt.ItemDataRole.UserRole, entry)
                self.table.setItem(row, col, item)
        self._apply_filter()

    def _preset_info(self, entry: Any) -> tuple[str, str, str]:
        try:
            from auto_capcut.core.captured_effect_template import CapturedEffectTemplateRepository
            repository = CapturedEffectTemplateRepository()
            for key, item in repository.load_registry().items():
                if item.stable_key == str(getattr(entry, "stable_key", "")):
                    return key, item.state, item.error
        except Exception:
            pass
        return "", "", ""

    def _apply_filter(self) -> None:
        from auto_capcut.core.effect_catalog import normalize_effect_name
        query = normalize_effect_name(self.search.text())
        source = self.source_filter.currentText().casefold()
        review = self.review_filter.currentText().casefold()
        build = self.build_filter.currentText().casefold()
        for row in range(self.table.rowCount()):
            entry = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            haystack = " ".join(normalize_effect_name(str(getattr(entry, key, ""))) for key in ("display_name", "enum_name", "effect_id", "resource_id", "category", "validation_state"))
            matches = (not query or query in haystack)
            matches = matches and (source.startswith("all") or str(getattr(entry, "source", "")).casefold() == source)
            matches = matches and (review.startswith("all") or str(getattr(entry, "review_status", "")).casefold() == review)
            matches = matches and (build.startswith("all") or str(getattr(entry, "build_status", "")).casefold() == build or (build == "buildable" and bool(getattr(entry, "buildable", False))) or (build == "discovery-only" and not bool(getattr(entry, "buildable", False))))
            # Template lifecycle is persisted independently from the legacy
            # build_status field.  Accept either field so older catalogs and
            # newly captured local entries remain searchable.
            lifecycle = str(getattr(entry, "lifecycle", getattr(entry, "lifecycle_state", getattr(entry, "validation_state", "")))).casefold()
            if build in {"promoted", "template_captured", "draft_recognized", "render_unverified", "render_confirmed"}:
                matches = lifecycle == build
            if build == "unresolved":
                _, preset_state, _ = self._preset_info(entry)
                matches = preset_state == "unresolved" or lifecycle == "unresolved"
            self.table.setRowHidden(row, not matches)

    def _selection_changed(self) -> None:
        selected = self._selected_entries()
        if len(selected) != 1:
            self.notes.clear(); self.tags.clear(); self.save_metadata.setEnabled(bool(selected)); return
        entry = selected[0]
        self.notes.setText(str(getattr(entry, "notes", "")))
        self.tags.setText(", ".join(getattr(entry, "tags", []) or []))
        self.review.setCurrentText(str(getattr(entry, "review_status", "Untested")))
        self.favorite.setCurrentIndex(1 if getattr(entry, "favorite", False) else 0)
        self.save_metadata.setEnabled(True)

    def _save_metadata(self) -> None:
        selected = self._selected_entries()
        if not selected: return
        store = self._store_instance()
        if store is None: return
        tags = [tag.strip() for tag in self.tags.text().split(",") if tag.strip()]
        for entry in selected:
            store.update(entry.stable_key, review_status=self.review.currentText(), favorite=self.favorite.currentIndex() == 1, notes=self.notes.text().strip(), tags=tags)
        self._entries = list(store.load().entries)
        self._populate()
        self.status.setText("Catalog metadata saved")

    def capture_warning_template(self) -> None:
        """Capture the real Warning material from the test_8 anchor.

        The capture implementation lives in the catalog service so this
        dialog never parses or writes CapCut drafts itself.  Keeping the call
        optional also lets the UI open with older catalogs while the feature
        is unavailable.
        """
        store = self._store_instance()
        capture = getattr(store, "capture_warning_template", None) if store is not None else None
        if capture is None:
            self.status.setText("Warning template capture is unavailable")
            return
        self.capture_template_button.setEnabled(False)
        try:
            result = capture()
            # A capture may return an entry, a path, or a structured result;
            # only surface a compact human-readable status here.
            detail = getattr(result, "path", result)
            self.status.setText(f"Warning template captured: {detail}")
            self._load_catalog()
        except Exception as exc:
            self.status.setText(f"Warning template capture failed: {exc}")
        finally:
            self.capture_template_button.setEnabled(True)

    def mark_lifecycle(self, state: str) -> None:
        """Mark selected effects with a manual template lifecycle state."""
        selected = self._selected_entries()
        if not selected:
            QMessageBox.warning(self, "Effect Catalog", "Select an effect first.")
            return
        store = self._store_instance()
        marker = getattr(store, "mark_lifecycle", None) if store is not None else None
        if marker is None:
            self.status.setText("Template lifecycle actions are unavailable")
            return
        failures = []
        try:
            from auto_capcut.core.captured_effect_template import CapturedEffectTemplateRepository
            preset_repository = CapturedEffectTemplateRepository()
        except Exception:
            preset_repository = None
        for entry in selected:
            try:
                preset_key, _, _ = self._preset_info(entry)
                if str(getattr(entry, "source", "")).casefold() == "local" and (not preset_key or preset_repository is None):
                    raise ValueError("effect has no registered captured preset")
                if preset_key and preset_repository is not None:
                    preset_repository.mark_state(preset_key, state)
                marker(entry.stable_key, state)
            except Exception as exc:
                failures.append(f"{getattr(entry, 'display_name', entry.stable_key)}: {exc}")
        self._load_catalog()
        if failures:
            self.status.setText(f"Lifecycle update failed for {len(failures)} effect(s): {failures[0]}")
        else:
            self.status.setText(f"Marked {len(selected)} effect(s) as {state}")

    def _selected_entries(self) -> list[Any]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in rows if not self.table.isRowHidden(row)]

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select test image", filter="Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.test_image.setText(path)

    def build_draft(self) -> None:
        selected = self._selected_entries()
        image = Path(self.test_image.text()) if self.test_image.text() else None
        parent = self.parentWidget()
        draft = Path(parent.draft_path.text()) if parent is not None and hasattr(parent, "draft_path") else None
        resolution = getattr(getattr(parent, "resolution", None), "currentText", lambda: "1920x1080")()
        try:
            width, height = (int(value) for value in str(resolution).lower().split("x", 1))
        except Exception:
            width, height = 1920, 1080
        if not selected:
            QMessageBox.warning(self, "Effect Catalog", "Select at least one effect."); return
        if image is None or not image.is_file():
            QMessageBox.warning(self, "Effect Catalog", "Choose a valid test image."); return
        if draft is None or not draft.is_dir():
            QMessageBox.warning(self, "Effect Catalog", "Configure a valid CapCut Draft Folder first."); return
        if self._thread is not None:
            return
        try:
            from auto_capcut.workers.effect_catalog_worker import EffectCatalogWorker
            self._thread = QThread(self)
            self._worker = EffectCatalogWorker(selected, image, draft, duration_seconds=float(self.duration.value()), batch_size=self.batch_size.value(), width=width, height=height)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(lambda value, message: self.status.setText(message))
            self._worker.finished.connect(self._build_finished)
            self._worker.failed.connect(self._build_failed)
            self._worker.finished.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._worker_done)
            self.build_button.setEnabled(False); self.status.setText("Building effect catalog..."); self._thread.start()
        except Exception as exc:
            self._build_failed(str(exc))

    def promote_local_effects(self) -> None:
        if self._resolution_presets:
            selected = []
            seen: set[str] = set()
            for entry in self._matched_entries.values():
                key = str(getattr(entry, "stable_key", ""))
                if key not in seen:
                    selected.append(entry); seen.add(key)
        else:
            selected = self._selected_entries()
        if not selected or any(str(getattr(entry, "source", "")).casefold() != "local" for entry in selected):
            if self._resolution_presets:
                QMessageBox.warning(self, "Effect Catalog", "Choose one catalog effect for each ambiguous preset before promoting.")
            else:
                QMessageBox.warning(self, "Effect Catalog", "Select one or more CapCut Local effects.")
            return
        image = Path(self.test_image.text()) if self.test_image.text() else None
        parent = self.parentWidget()
        draft = Path(parent.draft_path.text()) if parent is not None and hasattr(parent, "draft_path") else None
        resolution = getattr(getattr(parent, "resolution", None), "currentText", lambda: "1920x1080")()
        try:
            width, height = (int(value) for value in str(resolution).lower().split("x", 1))
        except Exception:
            width, height = 1920, 1080
        if image is None or not image.is_file() or draft is None or not draft.is_dir():
            QMessageBox.warning(self, "Effect Catalog", "Choose a valid test image and Draft Folder first.")
            return
        if self._thread is not None:
            return
        try:
            from auto_capcut.workers.effect_catalog_worker import EffectCatalogWorker
            self._thread = QThread(self)
            self._worker = EffectCatalogWorker.for_promote(
                selected, image, draft, duration_seconds=float(self.duration.value()),
                batch_size=self.batch_size.value(), width=width, height=height,
            )
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(lambda value, message: self.status.setText(message))
            self._worker.finished.connect(self._promotion_finished)
            self._worker.failed.connect(self._build_failed)
            self._worker.finished.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._worker_done)
            self.build_button.setEnabled(False); self.promote_button.setEnabled(False)
            self.status.setText("Promoting local CapCut effects...")
            self._thread.start()
        except Exception as exc:
            self._build_failed(str(exc))

    def _promotion_finished(self, result) -> None:
        unresolved = list(getattr(result, "unresolved", ()) or ())
        built = getattr(getattr(result, "build_result", None), "built", 0)
        self._load_catalog()
        if self._resolution_presets:
            self._load_resolution_matches()
        if unresolved:
            self.status.setText(f"Promotion complete: {built} built, {len(unresolved)} unresolved")
        else:
            self.status.setText(f"Promotion complete: {built} built; review drafts and mark render confirmed")

    def _build_finished(self, result) -> None:
        store = self._store_instance()
        if store:
            failed = {item.stable_key for item in getattr(result, "failures", ())}
            for entry in self._selected_entries():
                store.update(entry.stable_key, build_status="build_failed" if entry.stable_key in failed else "build_ok")
                # A successful JSON build is intentionally not equivalent to
                # visual rendering.  Keep the Warning lifecycle explicitly
                # unverified until the user confirms it in CapCut.
                if entry.stable_key not in failed and str(getattr(entry, "effect_id", "")) == "7399465244088618245":
                    marker = getattr(store, "mark_lifecycle", None)
                    if marker is not None:
                        try:
                            marker(entry.stable_key, "render_unverified")
                        except Exception:
                            pass
            self._entries = list(store.load().entries); self._populate()
        self.status.setText(f"Effect Catalog Build Complete — Selected: {result.selected}, Built: {result.built}, Failed: {result.failed}")

    def _build_failed(self, message: str) -> None:
        self.status.setText(f"Catalog build failed: {message}")

    def _worker_done(self) -> None:
        if self._thread is not None: self._thread.deleteLater()
        self._thread = None; self._worker = None; self.build_button.setEnabled(True); self.promote_button.setEnabled(True); self.scan_button.setEnabled(True); self.scan_local_button.setEnabled(True); self.scan_all_button.setEnabled(True)

    def export_approved(self) -> None:
        store = self._store_instance()
        if store is None: return
        try:
            path = store.export_approved()
            self.status.setText(f"Exported: {path}")
        except Exception as exc:
            self.status.setText(f"Export failed: {exc}")

    def open_catalog_folder(self) -> None:
        store = self._store_instance()
        folder = Path(getattr(store, "root", Path.home())) if store is not None else Path.home()
        try:
            import os
            os.startfile(str(folder))
        except Exception as exc:
            QMessageBox.warning(self, "Effect Catalog", f"Unable to open catalog folder: {exc}")


# Compatibility alias for callers/tests using the shorter name.
EffectCatalogTesterDialog = EffectCatalogDialog
