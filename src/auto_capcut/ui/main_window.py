from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QThread, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSlider, QDoubleSpinBox, QVBoxLayout, QWidget,
    QMenuBar,
    QTabWidget,
)

from auto_capcut.core.effect_direction_parser import parse_effect_direction_srt, required_roi_targets, optional_roi_targets
from auto_capcut.core.media import collect_images
from auto_capcut.core.planning import validate_required_rois
from auto_capcut.core.roi_resolver import ManualRoiResolver, roi_sidecar_path, validate_saved_frame
from auto_capcut.models import AudioMode, MotionStrength, ProjectConfig, RESOLUTIONS
from auto_capcut.utils.paths import default_draft_folder
from auto_capcut.workers.project_worker import ProjectWorker
from auto_capcut.ui.roi_editor import RoiEditorDialog
from auto_capcut.ui.draw_animation import DrawAnimationWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings()
        self.thread: QThread | None = None
        self.worker: ProjectWorker | None = None
        self.setWindowTitle("Auto CapCut")
        self.setMinimumSize(560, 760)
        self.resize(650, 900)
        self._build_ui()
        self._build_tools_menu()
        self._load_settings()
        self._update_enabled()

    def _build_tools_menu(self) -> None:
        """Install isolated developer tools without touching project controls."""
        tools_menu = self.menuBar().addMenu("Tools")
        action = tools_menu.addAction("Effect Catalog Tester")
        action.triggered.connect(self._open_effect_catalog)

    def _open_effect_catalog(self) -> None:
        from auto_capcut.ui.effect_catalog_dialog import EffectCatalogDialog
        dialog = EffectCatalogDialog(self)
        dialog.exec()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        draw_tab = DrawAnimationWidget(self)
        root = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        tabs.addTab(scroll, "CapCut Draft")
        tabs.addTab(draw_tab, "Draw Animation")
        self.setCentralWidget(tabs)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(10)
        title = QLabel("AUTO CAPCUT")
        title.setStyleSheet("font-size: 22px; font-weight: 700; letter-spacing: 2px;")
        layout.addWidget(title)

        project = QGroupBox("PROJECT")
        form = QFormLayout(project)
        self.project_name = QLineEdit()
        self.resolution = QComboBox(); self.resolution.addItems(list(RESOLUTIONS))
        form.addRow("Project name", self.project_name); form.addRow("Resolution", self.resolution)
        layout.addWidget(project)

        images = QGroupBox("IMAGES")
        image_layout = QVBoxLayout(images)
        self.image_list = QListWidget(); self.image_list.setMaximumHeight(120); image_layout.addWidget(self.image_list)
        image_buttons = QHBoxLayout(); add_images = QPushButton("+ Add Image Folder"); remove_images = QPushButton("Remove Selected")
        add_images.clicked.connect(self._add_image_folder); remove_images.clicked.connect(lambda: self.image_list.takeItem(self.image_list.currentRow()))
        image_buttons.addWidget(add_images); image_buttons.addWidget(remove_images); image_layout.addLayout(image_buttons)
        self.image_count = QLabel("Number of images: 0"); image_layout.addWidget(self.image_count); layout.addWidget(images)

        timing = QGroupBox("IMAGE TIMING")
        timing_form = QFormLayout(timing); self.use_timing = QCheckBox("Use Image Timing SRT"); self.timing_path = QLineEdit(); timing_browse = QPushButton("Browse")
        timing_browse.clicked.connect(lambda: self._browse_file(self.timing_path, "SRT files (*.srt)")); timing_row = QHBoxLayout(); timing_row.addWidget(self.timing_path); timing_row.addWidget(timing_browse)
        timing_form.addRow(self.use_timing); timing_form.addRow("Image Timing SRT", timing_row); layout.addWidget(timing)

        audio = QGroupBox("AUDIO"); audio_layout = QVBoxLayout(audio); mode_row = QHBoxLayout()
        self.single_audio = QRadioButton("Single Audio"); self.folder_audio = QRadioButton("Audio Folder"); self.single_audio.setChecked(True)
        mode_row.addWidget(self.single_audio); mode_row.addWidget(self.folder_audio); audio_layout.addLayout(mode_row)
        audio_form = QFormLayout(); self.audio_path = QLineEdit(); audio_browse = QPushButton("Browse"); audio_browse.clicked.connect(self._browse_audio)
        audio_row = QHBoxLayout(); audio_row.addWidget(self.audio_path); audio_row.addWidget(audio_browse); audio_form.addRow("Main Audio / Folder", audio_row); audio_layout.addLayout(audio_form); layout.addWidget(audio)

        subtitles = QGroupBox("SUBTITLES"); subtitle_form = QFormLayout(subtitles); self.import_subtitles = QCheckBox("Import subtitles"); self.subtitle_path = QLineEdit(); subtitle_browse = QPushButton("Browse")
        subtitle_browse.clicked.connect(lambda: self._browse_file(self.subtitle_path, "SRT files (*.srt)")); subtitle_row = QHBoxLayout(); subtitle_row.addWidget(self.subtitle_path); subtitle_row.addWidget(subtitle_browse)
        subtitle_form.addRow(self.import_subtitles); subtitle_form.addRow("Subtitle SRT", subtitle_row); layout.addWidget(subtitles)

        motion = QGroupBox("VISUAL EFFECTS"); motion_form = QFormLayout(motion)
        self.motion_enabled = QCheckBox("Enable Visual Effects"); self.motion_enabled.setChecked(True)
        self.motion_mode = QComboBox(); self.motion_mode.addItems(["None", "Random Light", "Subtle Zoom In", "Subtle Zoom Out", "Subtle Pan Left", "Subtle Pan Right"])
        self.effect_path = QLineEdit(); effect_browse = QPushButton("Browse"); effect_browse.clicked.connect(lambda: self._browse_file(self.effect_path, "SRT files (*.srt)")); effect_row = QHBoxLayout(); effect_row.addWidget(self.effect_path); effect_row.addWidget(effect_browse)
        self.effect_status = QLabel("Effect status: not selected"); self.effect_status.setWordWrap(True)
        self.roi_status = QLabel("Camera Frame status: not selected"); self.roi_status.setWordWrap(True)
        self.edit_roi_button = QPushButton("Edit Camera Frames"); self.edit_roi_button.clicked.connect(self._edit_roi)
        self.motion_strength = QComboBox(); self.motion_strength.addItems([strength.value for strength in MotionStrength])
        motion_form.addRow(self.motion_enabled); motion_form.addRow("Post-draw motion", self.motion_mode); motion_form.addRow("Effect Direction SRT", effect_row)
        motion_form.addRow("Effect status", self.effect_status); motion_form.addRow("Camera Frame status", self.roi_status); motion_form.addRow(self.edit_roi_button); motion_form.addRow("Motion strength", self.motion_strength); layout.addWidget(motion)


        transitions = QGroupBox("TRANSITIONS"); transition_form = QFormLayout(transitions); self.transition_enabled = QCheckBox("Enable Transitions"); self.transition_enabled.setChecked(True); self.transition_type = QComboBox(); self.transition_type.addItem("Blur"); self.transition_duration = QDoubleSpinBox(); self.transition_duration.setRange(0.01, 5.0); self.transition_duration.setSingleStep(0.05); self.transition_duration.setValue(0.30)
        transition_form.addRow(self.transition_enabled); transition_form.addRow("Transition", self.transition_type); transition_form.addRow("Duration (s)", self.transition_duration); layout.addWidget(transitions)

        logo = QGroupBox("LOGO"); logo_form = QFormLayout(logo); self.logo_enabled = QCheckBox("Add Logo"); self.logo_path = QLineEdit(); logo_browse = QPushButton("Browse"); logo_browse.clicked.connect(lambda: self._browse_file(self.logo_path, "Images (*.png *.jpg *.jpeg *.webp)")); logo_row = QHBoxLayout(); logo_row.addWidget(self.logo_path); logo_row.addWidget(logo_browse); logo_form.addRow(self.logo_enabled); logo_form.addRow("Logo file", logo_row); layout.addWidget(logo)
        music = QGroupBox("BACKGROUND MUSIC"); music_form = QFormLayout(music); self.music_enabled = QCheckBox("Add Background Music"); self.music_path = QLineEdit(); music_browse = QPushButton("Browse"); music_browse.clicked.connect(lambda: self._browse_folder(self.music_path)); music_row = QHBoxLayout(); music_row.addWidget(self.music_path); music_row.addWidget(music_browse); self.music_volume = QSlider(Qt.Orientation.Horizontal); self.music_volume.setRange(0, 100); self.music_volume.setValue(15); music_form.addRow(self.music_enabled); music_form.addRow("Music folder", music_row); music_form.addRow("Volume", self.music_volume); layout.addWidget(music)
        output = QGroupBox("OUTPUT"); output_form = QFormLayout(output); self.draft_path = QLineEdit(str(default_draft_folder())); draft_browse = QPushButton("Browse"); draft_browse.clicked.connect(lambda: self._browse_folder(self.draft_path)); draft_row = QHBoxLayout(); draft_row.addWidget(self.draft_path); draft_row.addWidget(draft_browse); output_form.addRow("CapCut Draft Folder", draft_row); layout.addWidget(output)
        self.create_button = QPushButton("CREATE CAPCUT PROJECT"); self.create_button.setMinimumHeight(42); self.create_button.clicked.connect(self._create_project); layout.addWidget(self.create_button); self.progress = QProgressBar(); self.status = QLabel("Ready"); self.status.setWordWrap(True); layout.addWidget(self.progress); layout.addWidget(self.status)
        for widget in (self.use_timing, self.import_subtitles, self.motion_enabled, self.transition_enabled, self.logo_enabled, self.music_enabled, self.single_audio, self.folder_audio): widget.toggled.connect(self._update_enabled)
        self.motion_mode.currentTextChanged.connect(lambda *_: (self._update_enabled(), self._update_effect_status()))
        self.effect_path.textChanged.connect(lambda *_: (self._update_enabled(), self._update_effect_status()))
        self.image_list.model().rowsInserted.connect(lambda *_: self._image_list_changed()); self.image_list.model().rowsRemoved.connect(lambda *_: self._image_list_changed())

        # ── DRAW ANIMATION group ──────────────────────────────────────────
        draw_grp = QGroupBox("DRAW ANIMATION")
        draw_form = QFormLayout(draw_grp)

        self.draw_enabled = QCheckBox("Enable draw rendering")
        self.draw_enabled.setChecked(True)
        draw_form.addRow(self.draw_enabled)

        self.draw_source_label = QLabel("Uses main Effect Direction SRT")
        self.draw_source_label.setStyleSheet("color: #666; font-style: italic;")
        draw_form.addRow("Effect source:", self.draw_source_label)

        # Internal / backwards-compat holder (empty by default)
        self.draw_effect_path = QLineEdit()

        self.draw_scene_path = QLineEdit()
        draw_scene_browse = QPushButton("Browse")
        draw_scene_browse.clicked.connect(lambda: self._browse_file(self.draw_scene_path, "JSON files (*.json)"))
        draw_scene_row = QHBoxLayout(); draw_scene_row.addWidget(self.draw_scene_path); draw_scene_row.addWidget(draw_scene_browse)
        draw_form.addRow("Scene JSON:", draw_scene_row)

        self.edit_draw_objects_btn = QPushButton("Edit Draw Objects")
        self.edit_draw_objects_btn.clicked.connect(self._edit_draw_objects)
        draw_form.addRow("", self.edit_draw_objects_btn)

        self.draw_scene_status = QLabel("Draw scene: not configured")
        self.draw_scene_status.setWordWrap(True)
        self.draw_scene_status.setStyleSheet("color: #444; padding: 2px 0;")
        draw_form.addRow("Draw scene status:", self.draw_scene_status)

        self.draw_remove_bg = QCheckBox("Remove simple background")
        self.draw_fallback_basic = QCheckBox("Fallback invalid advanced scenes to basic")
        self.draw_fallback_basic.setChecked(True)
        self.draw_diagnostics = QCheckBox("Write draw diagnostics")
        self.draw_reuse_cache = QCheckBox("Reuse draw render cache")
        self.draw_reuse_cache.setChecked(True)
        draw_form.addRow(self.draw_remove_bg)
        draw_form.addRow(self.draw_fallback_basic)
        draw_form.addRow(self.draw_diagnostics)
        draw_form.addRow(self.draw_reuse_cache)

        # Insert draw group between TRANSITIONS and LOGO
        transitions_index = layout.indexOf(transitions)
        layout.insertWidget(transitions_index + 1, draw_grp)

        self.draw_enabled.toggled.connect(lambda *_: (self._update_enabled(), self._update_draw_scene_status()))
        self.draw_scene_path.textChanged.connect(lambda *_: self._update_draw_scene_status())
        self.draw_fallback_basic.toggled.connect(lambda *_: self._update_draw_scene_status())


    def _add_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if folder: self.image_list.addItem(folder); self._image_list_changed()

    def _browse_folder(self, target: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder: target.setText(folder)

    def _browse_file(self, target: QLineEdit, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", filter=file_filter)
        if path: target.setText(path)

    def _browse_audio(self) -> None:
        self._browse_folder(self.audio_path) if self.folder_audio.isChecked() else self._browse_file(self.audio_path, "Audio (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.aiff)")

    def _current_images(self) -> list[Path]:
        try: return collect_images([Path(self.image_list.item(row).text()) for row in range(self.image_list.count())])
        except Exception: return []

    def _image_list_changed(self) -> None:
        self.image_count.setText(f"Number of images: {len(self._current_images())}")
        self._update_effect_status()
        self._update_draw_scene_status()

    def _update_draw_scene_status(self) -> None:
        if not hasattr(self, "draw_scene_status"):
            return
        if not self.draw_enabled.isChecked():
            self.draw_scene_status.setText("Draw scene: draw rendering disabled")
            return

        images = self._current_images()
        if not images:
            self.draw_scene_status.setText("Draw scene: add image folder first")
            return

        scene_text = self.draw_scene_path.text().strip()
        scene_doc = None
        if scene_text:
            path = Path(scene_text)
            if path.is_file():
                try:
                    from auto_capcut.core.draw_scene import load_scene
                    scene_doc = load_scene(path)
                except Exception as exc:
                    self.draw_scene_status.setText(f"Draw scene: invalid scene JSON ({exc})")
                    return

        # Check effect SRT for which images have advanced_draw cues
        draw_cues_by_img: dict[str, str] = {}
        effect_text = self.effect_path.text().strip()
        if effect_text and Path(effect_text).is_file():
            try:
                from auto_capcut.core.unified_effect_parser import parse_unified_effect
                unified = parse_unified_effect(effect_text)
                for cue in unified.cues:
                    if cue.kind == "draw" and cue.draw_plan is not None:
                        idx = cue.index - 1
                        if 0 <= idx < len(images):
                            draw_cues_by_img[images[idx].name.casefold()] = cue.draw_plan.mode.value
            except Exception:
                pass

        configured_count = 0
        fallback_enabled = self.draw_fallback_basic.isChecked()
        lines: list[str] = []

        for img in images:
            key = img.name.casefold()
            rec = None
            if scene_doc and scene_doc.images:
                rec = scene_doc.images.get(key)
                if rec is None:
                    for k, v in scene_doc.images.items():
                        if k.casefold() == key or k == img.name:
                            rec = v
                            break

            req_mode = draw_cues_by_img.get(key)
            if rec:
                configured_count += 1
                obj_count = len(rec.objects)
                cam_count = sum(1 for o in rec.objects if o.camera_frame is not None)
                cam_str = f" / {cam_count} camera frame{'s' if cam_count != 1 else ''}" if cam_count else ""
                lines.append(f"{img.name}: {obj_count} objects{cam_str}")
            else:
                if req_mode == "advanced_draw":
                    if fallback_enabled:
                        lines.append(f"{img.name}: advanced scene missing → will fallback to basic_draw")
                    else:
                        lines.append(f"{img.name}: advanced scene missing (blocking error)")
                else:
                    lines.append(f"{img.name}: not configured")

        header = f"Draw scene: {len(images)} project images ({configured_count} configured)" if len(images) > 0 else "Draw scene: not configured"
        self.draw_scene_status.setText(header + ("\n" + "\n".join(lines) if lines else ""))

    def _edit_draw_objects(self) -> None:
        images = self._current_images()
        if not images:
            QMessageBox.warning(self, "Draw Objects", "Add an image folder first.")
            return

        scene_text = self.draw_scene_path.text().strip()
        if scene_text:
            scene_path = Path(scene_text)
        else:
            effect_text = self.effect_path.text().strip()
            if effect_text and Path(effect_text).is_file():
                scene_path = Path(effect_text).parent / "draw_scene.json"
            elif self.image_list.count() > 0:
                scene_path = Path(self.image_list.item(0).text()) / "draw_scene.json"
            else:
                scene_path = images[0].parent / "draw_scene.json"
            self.draw_scene_path.setText(str(scene_path))
            self.settings.setValue("draw_scene_path", str(scene_path))

        resolution = RESOLUTIONS[self.resolution.currentText()]
        from auto_capcut.ui.draw_animation import DrawObjectEditorDialog
        dialog = DrawObjectEditorDialog(images, scene_path, (resolution.width, resolution.height), parent=self)
        dialog.exec()
        self._update_draw_scene_status()

    def _update_effect_status(self) -> None:
        path = Path(self.effect_path.text()) if self.effect_path.text() else None
        if path is None or not path.is_file():
            self.effect_status.setText("Effect status: choose an Effect Direction SRT")
            self.roi_status.setText("Camera Frame status: not selected")
            return
        try:
            from auto_capcut.core.unified_effect_parser import parse_unified_effect
            unified = parse_unified_effect(path)
            images = self._current_images()
            total_cues = len(unified.cues)
            draw_count = len(unified.draw_plans)
            cue_label = f"{total_cues} cues ({draw_count} draw)" if draw_count else f"{total_cues} effect cues"
            self.effect_status.setText(f"{len(images)} images / {cue_label} {'✓' if len(images) == total_cues else '!'}")
            effects = [c.effect_cue for c in unified.cues if c.kind == "standard" and c.effect_cue is not None]
            targets = required_roi_targets(effects)
            resolver = ManualRoiResolver(roi_sidecar_path(path))
            configured = 0
            reframing = 0
            resolution = RESOLUTIONS[self.resolution.currentText()]
            for target in targets:
                if target.image_index > len(images):
                    continue
                valid, reason = validate_saved_frame(resolver.resolve(images[target.image_index - 1], target.target_id, target.image_index), images[target.image_index - 1], (resolution.width, resolution.height)) if resolver.resolve(images[target.image_index - 1], target.target_id, target.image_index) else (False, "missing")
                configured += int(valid)
                reframing += int(not valid and reason == "frame aspect does not match project canvas")
            missing = len(targets) - configured
            if not targets:
                self.roi_status.setText("Camera Frame status: ready ✓ (No camera frames required)" if draw_count else "Camera Frame status: 0 / 0 ready ✓")
            elif missing == 0:
                self.roi_status.setText(f"Camera Frame status: {configured} / {len(targets)} ready ✓")
            else:
                suffix = f"\n{reframing} needs reframing" if reframing else ""
                self.roi_status.setText(f"Camera Frame status: {len(targets)} targets / {configured} configured\n{missing} missing{suffix}")
        except Exception as exc:
            self.effect_status.setText(f"Effect status: {exc}")
            self.roi_status.setText("Camera Frame status: invalid Effect Direction")

    def _edit_roi(self) -> None:
        path = Path(self.effect_path.text()) if self.effect_path.text() else None
        images = self._current_images()
        if path is None or not path.is_file():
            QMessageBox.warning(self, "Camera Frames", "Choose a valid Effect Direction SRT first.")
            return
        try:
            from auto_capcut.core.unified_effect_parser import parse_unified_effect
            unified = parse_unified_effect(path)
            effects = [c.effect_cue for c in unified.cues if c.kind == "standard" and c.effect_cue is not None]
            if not effects:
                QMessageBox.information(self, "Camera Frames", "No standard camera frames in the selected effect file.")
                return
        except Exception as exc:
            QMessageBox.critical(self, "Camera Frames", str(exc))
            return
        resolution = RESOLUTIONS[self.resolution.currentText()]
        dialog = RoiEditorDialog(images, effects, roi_sidecar_path(path), self, (resolution.width, resolution.height), self.motion_strength.currentText())
        dialog.exec()
        self._update_effect_status()


    def _show_missing_roi_editor(self) -> None:
        """Open the same project-wide queue used by the ROI button."""
        self._edit_roi()

    def _update_enabled(self) -> None:
        self.motion_mode.setEnabled(self.motion_enabled.isChecked())
        self.effect_path.setEnabled(True)
        self.edit_roi_button.setEnabled(Path(self.effect_path.text()).is_file() and bool(self._current_images()))
        self.motion_strength.setEnabled(self.motion_enabled.isChecked())
        self.transition_type.setEnabled(self.transition_enabled.isChecked())
        self.transition_duration.setEnabled(self.transition_enabled.isChecked())
        self.audio_path.setPlaceholderText("Folder containing audio files" if self.folder_audio.isChecked() else "Audio file")
        draw_on = self.draw_enabled.isChecked()
        for w in (self.draw_source_label, self.draw_scene_path, self.edit_draw_objects_btn, self.draw_scene_status, self.draw_remove_bg, self.draw_fallback_basic, self.draw_diagnostics, self.draw_reuse_cache):
            w.setEnabled(draw_on)



    def _config(self) -> ProjectConfig:
        return ProjectConfig(
            project_name=self.project_name.text(),
            resolution=RESOLUTIONS[self.resolution.currentText()],
            image_folders=[Path(self.image_list.item(row).text()) for row in range(self.image_list.count())],
            audio_mode=AudioMode.FOLDER if self.folder_audio.isChecked() else AudioMode.SINGLE,
            audio_path=Path(self.audio_path.text()) if self.audio_path.text() else None,
            subtitle_srt=Path(self.subtitle_path.text()) if self.subtitle_path.text() else None,
            import_subtitles=self.import_subtitles.isChecked(),
            use_image_timing=self.use_timing.isChecked(),
            image_timing_srt=Path(self.timing_path.text()) if self.timing_path.text() else None,
            motion_enabled=self.motion_enabled.isChecked(),
            motion_mode=self.motion_mode.currentText(),
            motion_strength=self.motion_strength.currentText(),
            effect_direction_srt=Path(self.effect_path.text()) if self.effect_path.text() else None,
            transition_enabled=self.transition_enabled.isChecked(),
            transition_type=self.transition_type.currentText(),
            transition_duration_us=round(self.transition_duration.value() * 1_000_000),
            logo_enabled=self.logo_enabled.isChecked(),
            logo_path=Path(self.logo_path.text()) if self.logo_path.text() else None,
            music_enabled=self.music_enabled.isChecked(),
            music_folder=Path(self.music_path.text()) if self.music_path.text() else None,
            music_volume=self.music_volume.value() / 100,
            draft_folder=Path(self.draft_path.text()) if self.draft_path.text() else None,
            # ── Draw Animation ──────────────────────────────────────────
            draw_enabled=self.draw_enabled.isChecked(),
            draw_effect_srt=Path(self.draw_effect_path.text()) if self.draw_effect_path.text() else None,
            draw_scene_json=Path(self.draw_scene_path.text()) if self.draw_scene_path.text() else None,
            draw_remove_background=self.draw_remove_bg.isChecked(),
            draw_fallback_basic=self.draw_fallback_basic.isChecked(),
            draw_diagnostics=self.draw_diagnostics.isChecked(),
            draw_reuse_cache=self.draw_reuse_cache.isChecked(),
        )

    def _create_project(self) -> None:
        try: config = self._config()
        except Exception as exc: QMessageBox.critical(self, "Invalid configuration", str(exc)); return
        self._save_settings(); self.create_button.setEnabled(False); self.progress.setValue(0); self.status.setText("Loading inputs..."); self.thread = QThread(self); self.worker = ProjectWorker(config); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress); self.worker.finished.connect(self._on_finished); self.worker.failed.connect(self._on_failed); self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.finished.connect(self._thread_finished); self.thread.start()

    def _on_progress(self, value: int, message: str) -> None: self.progress.setValue(value); self.status.setText(message)
    def _on_finished(self, results: list) -> None: self.progress.setValue(100); self.status.setText(f"Project created successfully: {', '.join(result.project_name for result in results)}")
    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        if message.startswith("Unresolved CapCut effect presets:"):
            presets = [line[2:].strip() for line in message.splitlines()[1:] if line.startswith("- ")]
            images = self._current_images()
            from auto_capcut.ui.effect_catalog_dialog import EffectCatalogDialog
            dialog = EffectCatalogDialog(self, resolution_presets=presets, test_image_path=images[0] if images else None)
            dialog.exec()
            return
        if message.startswith("Missing ROI definitions:"):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Missing Target ROIs")
            box.setText("Required target ROIs are missing. Configure them before creating the draft.")
            details = QPushButton("View Missing Targets", box)
            box.addButton(details, QMessageBox.ButtonRole.ActionRole)
            edit = box.addButton("Edit Missing ROIs", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Close)
            details.clicked.connect(lambda: box.setDetailedText(message))
            box.exec()
            if box.clickedButton() is edit:
                self._show_missing_roi_editor()
            return
        QMessageBox.critical(self, "Auto CapCut", message)
    def _thread_finished(self) -> None: self.create_button.setEnabled(True); self.thread = None; self.worker = None; self._update_enabled()

    def _save_settings(self) -> None:
        values = {
            "project_name": self.project_name.text(), "resolution": self.resolution.currentText(),
            "audio_path": self.audio_path.text(), "subtitle_path": self.subtitle_path.text(),
            "timing_path": self.timing_path.text(), "logo_path": self.logo_path.text(),
            "music_path": self.music_path.text(), "draft_path": self.draft_path.text(),
            "motion_enabled": self.motion_enabled.isChecked(), "transition_enabled": self.transition_enabled.isChecked(),
            "effect_path": self.effect_path.text(), "motion_mode": self.motion_mode.currentText(),
            "motion_strength": self.motion_strength.currentText(),
            # Draw Animation settings
            "draw_enabled": self.draw_enabled.isChecked(),
            "draw_effect_path": self.draw_effect_path.text(),
            "draw_scene_path": self.draw_scene_path.text(),
            "draw_remove_bg": self.draw_remove_bg.isChecked(),
            "draw_fallback_basic": self.draw_fallback_basic.isChecked(),
            "draw_diagnostics": self.draw_diagnostics.isChecked(),
            "draw_reuse_cache": self.draw_reuse_cache.isChecked(),
        }
        for key, value in values.items(): self.settings.setValue(key, value)
        if self.image_list.count(): self.settings.setValue("image_folder", self.image_list.item(0).text())

    def _load_settings(self) -> None:
        # One-time cleanup for settings written by removed Asset/Vision UI.
        self.settings.remove("asset_folder")
        self.settings.remove("visual_manifest_path")
        self.project_name.setText(self.settings.value("project_name", "")); resolution = self.settings.value("resolution", "1920x1080"); index = self.resolution.findText(str(resolution)); self.resolution.setCurrentIndex(index if index >= 0 else 0)
        for key, widget in (("audio_path", self.audio_path), ("subtitle_path", self.subtitle_path), ("timing_path", self.timing_path), ("logo_path", self.logo_path), ("music_path", self.music_path), ("draft_path", self.draft_path)): widget.setText(str(self.settings.value(key, widget.text())))
        self.effect_path.setText(str(self.settings.value("effect_path", ""))); saved_mode = str(self.settings.value("motion_mode", "Random Light")); legacy = {"Random": "Random Light", "Zoom In": "Subtle Zoom In", "Zoom Out": "Subtle Zoom Out", "Pan Left": "Subtle Pan Left", "Pan Right": "Subtle Pan Right"}; saved_mode = legacy.get(saved_mode, saved_mode); self.motion_mode.setCurrentText(saved_mode if self.motion_mode.findText(saved_mode) >= 0 else "Random Light"); self.motion_strength.setCurrentText(str(self.settings.value("motion_strength", MotionStrength.SUBTLE.value))); self.motion_enabled.setChecked(self.settings.value("motion_enabled", True, type=bool)); self.transition_enabled.setChecked(self.settings.value("transition_enabled", True, type=bool)); image_folder = str(self.settings.value("image_folder", ""));
        if image_folder: self.image_list.addItem(image_folder); self._image_list_changed()
        # Draw Animation settings
        self.draw_enabled.setChecked(self.settings.value("draw_enabled", True, type=bool))
        self.draw_effect_path.setText(str(self.settings.value("draw_effect_path", "")))

        self.draw_scene_path.setText(str(self.settings.value("draw_scene_path", "")))
        self.draw_remove_bg.setChecked(self.settings.value("draw_remove_bg", False, type=bool))
        self.draw_fallback_basic.setChecked(self.settings.value("draw_fallback_basic", True, type=bool))
        self.draw_diagnostics.setChecked(self.settings.value("draw_diagnostics", False, type=bool))
        self.draw_reuse_cache.setChecked(self.settings.value("draw_reuse_cache", True, type=bool))
        self._update_draw_scene_status()

