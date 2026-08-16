from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QThread, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSlider, QDoubleSpinBox, QVBoxLayout, QWidget,
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
        self._load_settings()
        self._update_enabled()


    def _build_ui(self) -> None:
        tabs = QTabWidget()
        draw_tab = DrawAnimationWidget(self)
        root = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        tabs.addTab(scroll, "CapCut Draft")
        tabs.addTab(draw_tab, "Draw Debug")
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

        # Audio
        audio = QGroupBox("AUDIO"); audio_layout = QVBoxLayout(audio); mode_row = QHBoxLayout()
        self.single_audio = QRadioButton("Single Audio"); self.folder_audio = QRadioButton("Audio Folder"); self.single_audio.setChecked(True)
        mode_row.addWidget(self.single_audio); mode_row.addWidget(self.folder_audio); audio_layout.addLayout(mode_row)
        audio_form = QFormLayout(); self.audio_path = QLineEdit(); audio_browse = QPushButton("Browse"); audio_browse.clicked.connect(self._browse_audio)
        audio_row = QHBoxLayout(); audio_row.addWidget(self.audio_path); audio_row.addWidget(audio_browse); audio_form.addRow("Main Audio / Folder", audio_row); audio_layout.addLayout(audio_form); layout.addWidget(audio)

        subtitles = QGroupBox("SUBTITLES"); subtitle_form = QFormLayout(subtitles); self.import_subtitles = QCheckBox("Import subtitles"); self.subtitle_path = QLineEdit(); subtitle_browse = QPushButton("Browse")
        subtitle_browse.clicked.connect(lambda: self._browse_file(self.subtitle_path, "SRT files (*.srt)")); subtitle_row = QHBoxLayout(); subtitle_row.addWidget(self.subtitle_path); subtitle_row.addWidget(subtitle_browse)
        subtitle_form.addRow(self.import_subtitles); subtitle_form.addRow("Subtitle SRT", subtitle_row); layout.addWidget(subtitles)

        motion = QGroupBox("MAIN EFFECT SRT & MOTION"); motion_form = QFormLayout(motion)
        self.motion_enabled = QCheckBox("Enable post-draw motion"); self.motion_enabled.setChecked(True)
        self.motion_mode = QComboBox(); self.motion_mode.addItems(["None", "Random Light", "Subtle Zoom In", "Subtle Zoom Out", "Subtle Pan Left", "Subtle Pan Right"])
        self.effect_path = QLineEdit(); effect_browse = QPushButton("Browse"); effect_browse.clicked.connect(lambda: self._browse_file(self.effect_path, "SRT files (*.srt)")); effect_row = QHBoxLayout(); effect_row.addWidget(self.effect_path); effect_row.addWidget(effect_browse)
        self.effect_status = QLabel("Effect status: not selected"); self.effect_status.setWordWrap(True)
        self.motion_strength = QComboBox(); self.motion_strength.addItems([strength.value for strength in MotionStrength])
        motion_form.addRow(self.motion_enabled); motion_form.addRow("Post-draw motion", self.motion_mode); motion_form.addRow("Main Effect SRT", effect_row)
        motion_form.addRow("Effect status", self.effect_status); motion_form.addRow("Motion strength", self.motion_strength); layout.addWidget(motion)

        transitions = QGroupBox("TRANSITIONS"); transition_form = QFormLayout(transitions); self.transition_enabled = QCheckBox("Enable Transitions"); self.transition_enabled.setChecked(True); self.transition_type = QComboBox(); self.transition_type.addItem("Blur"); self.transition_duration = QDoubleSpinBox(); self.transition_duration.setRange(0.01, 5.0); self.transition_duration.setSingleStep(0.05); self.transition_duration.setValue(0.30)
        transition_form.addRow(self.transition_enabled); transition_form.addRow("Transition", self.transition_type); transition_form.addRow("Duration (s)", self.transition_duration); layout.addWidget(transitions)

        logo = QGroupBox("LOGO"); logo_form = QFormLayout(logo); self.logo_enabled = QCheckBox("Add Logo"); self.logo_path = QLineEdit(); logo_browse = QPushButton("Browse"); logo_browse.clicked.connect(lambda: self._browse_file(self.logo_path, "Images (*.png *.jpg *.jpeg *.webp)")); logo_row = QHBoxLayout(); logo_row.addWidget(self.logo_path); logo_row.addWidget(logo_browse); logo_form.addRow(self.logo_enabled); logo_form.addRow("Logo file", logo_row); layout.addWidget(logo)
        music = QGroupBox("BACKGROUND MUSIC"); music_form = QFormLayout(music); self.music_enabled = QCheckBox("Add Background Music"); self.music_path = QLineEdit(); music_browse = QPushButton("Browse"); music_browse.clicked.connect(lambda: self._browse_folder(self.music_path)); music_row = QHBoxLayout(); music_row.addWidget(self.music_path); music_row.addWidget(music_browse); self.music_volume = QSlider(Qt.Orientation.Horizontal); self.music_volume.setRange(0, 100); self.music_volume.setValue(15); music_form.addRow(self.music_enabled); music_form.addRow("Music folder", music_row); music_form.addRow("Volume", self.music_volume); layout.addWidget(music)
        output = QGroupBox("OUTPUT"); output_form = QFormLayout(output); self.draft_path = QLineEdit(str(default_draft_folder())); draft_browse = QPushButton("Browse"); draft_browse.clicked.connect(lambda: self._browse_folder(self.draft_path)); draft_row = QHBoxLayout(); draft_row.addWidget(self.draft_path); draft_row.addWidget(draft_browse); output_form.addRow("CapCut Draft Folder", draft_row); layout.addWidget(output)
        self.create_button = QPushButton("CREATE CAPCUT PROJECT"); self.create_button.setMinimumHeight(42); self.create_button.clicked.connect(self._create_project); layout.addWidget(self.create_button); self.progress = QProgressBar(); self.status = QLabel("Ready"); self.status.setWordWrap(True); layout.addWidget(self.progress); layout.addWidget(self.status)
        for widget in (self.import_subtitles, self.motion_enabled, self.transition_enabled, self.logo_enabled, self.music_enabled, self.single_audio, self.folder_audio): widget.toggled.connect(self._update_enabled)
        self.motion_mode.currentTextChanged.connect(lambda *_: (self._update_enabled(), self._update_effect_status()))
        self.effect_path.textChanged.connect(lambda *_: (self._update_enabled(), self._update_effect_status()))
        self.image_list.model().rowsInserted.connect(lambda *_: self._image_list_changed()); self.image_list.model().rowsRemoved.connect(lambda *_: self._image_list_changed())


        # ── DRAW ANIMATION / SETUP group ─────────────────────────────────────
        draw_grp = QGroupBox("DRAW ANIMATION")
        draw_form = QFormLayout(draw_grp)

        self.draw_source_label = QLabel("Uses Main Effect SRT (timing + draw mode source)")
        self.draw_source_label.setStyleSheet("color: #666; font-style: italic;")
        draw_form.addRow("Effect source:", self.draw_source_label)

        self.draw_scene_path = QLineEdit()
        draw_scene_browse = QPushButton("Browse")
        draw_scene_browse.clicked.connect(lambda: self._browse_file(self.draw_scene_path, "JSON files (*.json)"))
        draw_scene_row = QHBoxLayout()
        draw_scene_row.addWidget(self.draw_scene_path)
        draw_scene_row.addWidget(draw_scene_browse)
        draw_form.addRow("Scene JSON:", draw_scene_row)

        # ── Per-image draw setup status panel (SRT-driven) ──────────────────
        self.draw_setup_status = QLabel("DRAW SETUP\n\nAdd images and a Main Effect SRT first.")
        self.draw_setup_status.setWordWrap(True)
        self.draw_setup_status.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f8f8f8; "
            "border: 1px solid #ddd; padding: 6px; color: #222;"
        )
        self.draw_setup_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        draw_form.addRow(self.draw_setup_status)

        # ── Configure Advanced Images button ─────────────────────────────────
        self.configure_advanced_btn = QPushButton("Configure Advanced Images")
        self.configure_advanced_btn.clicked.connect(self._configure_advanced_images)
        draw_form.addRow("", self.configure_advanced_btn)

        # Keep legacy draw_scene_status for _update_enabled compat
        self.draw_scene_status = self.draw_setup_status

        # ── Debug / options ──────────────────────────────────────────────────
        self.draw_remove_bg = QCheckBox("Remove simple background")
        self.draw_fallback_basic = QCheckBox("Fallback invalid advanced scenes to basic (debug)")
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

        self.draw_scene_path.textChanged.connect(lambda *_: self._update_draw_scene_status())
        self.draw_fallback_basic.toggled.connect(lambda *_: self._update_draw_scene_status())
        # edit_draw_objects_btn alias removed; _update_enabled must not ref it
        self.edit_draw_objects_btn = self.configure_advanced_btn  # alias for compat




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
        """Refresh the SRT-driven per-image draw setup status panel."""
        if not hasattr(self, "draw_setup_status"):
            return

        images = self._current_images()
        effect_text = self.effect_path.text().strip()

        if not images:
            self.draw_setup_status.setText("DRAW SETUP\n\nAdd an image folder first.")
            self.configure_advanced_btn.setText("Configure Advanced Images")
            self.configure_advanced_btn.setEnabled(False)
            return

        if not effect_text or not Path(effect_text).is_file():
            self.draw_setup_status.setText("DRAW SETUP\n\nSelect a Main Effect SRT first.")
            self.configure_advanced_btn.setText("Configure Advanced Images")
            self.configure_advanced_btn.setEnabled(False)
            return

        # Load scene doc (optional)
        scene_doc = None
        scene_text = self.draw_scene_path.text().strip()
        if scene_text and Path(scene_text).is_file():
            try:
                from auto_capcut.core.draw_scene import load_scene
                scene_doc = load_scene(Path(scene_text))
            except Exception as exc:
                self.draw_setup_status.setText(f"DRAW SETUP\n\nInvalid scene JSON: {exc}")
                self.configure_advanced_btn.setEnabled(False)
                return

        # Analyze
        try:
            from auto_capcut.core.draw_setup import analyze_from_srt
            summary = analyze_from_srt(images, effect_text, scene_doc)
        except Exception as exc:
            self.draw_setup_status.setText(f"DRAW SETUP\n\nSRT parse error: {exc}")
            self.configure_advanced_btn.setEnabled(False)
            return

        # Compose per-image status table
        col_w = max(len(s.image_name) for s in summary.statuses) if summary.statuses else 8
        lines = ["DRAW SETUP", ""]
        for s in summary.statuses:
            mode_str = "BASIC   " if s.is_basic else "ADVANCED"
            ready_str = "Ready ✓" if s.is_ready else "Setup needed"
            detail = "" if s.is_ready else f"  ← {s.message.replace('Setup needed — ', '')}"
            lines.append(f"{s.image_name:<{col_w}}  {mode_str}  {ready_str}{detail}")

        lines.append("")
        lines.append(f"{summary.total} images  |  {summary.basic_count} Basic  |  {summary.advanced_count} Advanced")
        if summary.advanced_count:
            lines.append(f"{summary.advanced_ready} Advanced Ready  |  {summary.advanced_needs_setup} Needs Setup")

        self.draw_setup_status.setText("\n".join(lines))
        self.draw_setup_status.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f8f8f8; "
            "border: 1px solid #ddd; padding: 6px; color: #222;"
        )

        # Update configure button
        n_missing = summary.advanced_needs_setup
        n_adv = summary.advanced_count
        if n_adv == 0:
            self.configure_advanced_btn.setText("No advanced setup required ✓")
            self.configure_advanced_btn.setEnabled(False)
        elif n_missing == 0:
            self.configure_advanced_btn.setText("Review Advanced Images")
            self.configure_advanced_btn.setEnabled(True)
        elif n_missing == 1:
            self.configure_advanced_btn.setText("Configure 1 Advanced Image")
            self.configure_advanced_btn.setEnabled(True)
        else:
            self.configure_advanced_btn.setText(f"Configure {n_missing} Advanced Images")
            self.configure_advanced_btn.setEnabled(True)

        # Store summary for quick preflight
        self._draw_summary = summary




    def _configure_advanced_images(self) -> None:
        """Open the Draw Object Editor focused on incomplete advanced images.

        Builds a queue of advanced images that are not ready, opens the editor
        restricted to those images, starting at the first incomplete one.
        When invoked as 'Review' (all ready), opens all advanced images.
        """
        images = self._current_images()
        if not images:
            QMessageBox.warning(self, "Configure Advanced Images", "Add an image folder first.")
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
        canvas_size = (resolution.width, resolution.height)

        # Get current summary (computed during status update)
        summary = getattr(self, "_draw_summary", None)

        # Determine queue
        if summary is not None and summary.advanced_needs_setup > 0:
            # Queue = incomplete advanced images only
            incomplete = summary.incomplete_advanced
            allowed_names = [s.image_name for s in incomplete]
            # Required IDs context
            req_ids: dict[str, list[str]] = {
                s.image_name.casefold(): list(s.required_ids)
                for s in incomplete
            }
            cam_ids: dict[str, set[str]] = {
                s.image_name.casefold(): set(s.required_camera_frame_ids)
                for s in incomplete
            }
            initial_idx = 0  # start at first incomplete
        elif summary is not None and summary.advanced_count > 0:
            # Review mode: all advanced images
            advanced = summary.all_advanced
            allowed_names = [s.image_name for s in advanced]
            req_ids = {
                s.image_name.casefold(): list(s.required_ids)
                for s in advanced
            }
            cam_ids = {
                s.image_name.casefold(): set(s.required_camera_frame_ids)
                for s in advanced
            }
            initial_idx = 0
        else:
            # No SRT summary available — open all images (legacy fallback)
            allowed_names = None
            req_ids = {}
            cam_ids = {}
            initial_idx = 0

        from auto_capcut.ui.draw_animation import DrawObjectEditorDialog
        dialog = DrawObjectEditorDialog(
            images,
            scene_path,
            canvas_size,
            parent=self,
            initial_image_index=initial_idx,
            allowed_image_names=allowed_names,
            required_ids_by_image=req_ids,
            camera_frame_ids_by_image=cam_ids,
        )
        dialog.exec()
        self._update_draw_scene_status()

    def _update_effect_status(self) -> None:
        path = Path(self.effect_path.text()) if self.effect_path.text() else None
        if path is None or not path.is_file():
            self.effect_status.setText("Effect status: choose a Main Effect SRT")
            return
        try:
            from auto_capcut.core.unified_effect_parser import parse_unified_effect
            from auto_capcut.core.draw_models import DrawMode
            unified = parse_unified_effect(path)
            images = self._current_images()
            total_cues = len(unified.cues)
            draw_cues = [c for c in unified.cues if c.kind == "draw" and c.draw_plan is not None]
            basic_count = sum(1 for c in draw_cues if c.draw_plan.mode is DrawMode.BASIC)
            advanced_count = sum(1 for c in draw_cues if c.draw_plan.mode is DrawMode.ADVANCED)
            match_icon = "✓" if len(images) == total_cues else "!"
            lines = [
                f"{len(images)} images / {total_cues} cues {match_icon}",
                f"{basic_count} Basic Draw  |  {advanced_count} Advanced Draw",
            ]
            if unified.draw_warnings:
                lines.append("Warnings: " + "; ".join(unified.draw_warnings[:2]))
            self.effect_status.setText("\n".join(lines))
        except Exception as exc:
            self.effect_status.setText(f"Effect status: {exc}")



    def _update_enabled(self) -> None:
        self.motion_mode.setEnabled(self.motion_enabled.isChecked())
        self.effect_path.setEnabled(True)
        self.motion_strength.setEnabled(self.motion_enabled.isChecked())
        self.transition_type.setEnabled(self.transition_enabled.isChecked())
        self.transition_duration.setEnabled(self.transition_enabled.isChecked())
        self.audio_path.setPlaceholderText("Folder containing audio files" if self.folder_audio.isChecked() else "Audio file")
        for w in (self.draw_source_label, self.draw_scene_path, self.edit_draw_objects_btn, self.draw_scene_status, self.draw_remove_bg, self.draw_fallback_basic, self.draw_diagnostics, self.draw_reuse_cache):
            w.setEnabled(True)

    def _config(self) -> ProjectConfig:
        return ProjectConfig(
            project_name=self.project_name.text(),
            resolution=RESOLUTIONS[self.resolution.currentText()],
            image_folders=[Path(self.image_list.item(row).text()) for row in range(self.image_list.count())],
            audio_mode=AudioMode.FOLDER if self.folder_audio.isChecked() else AudioMode.SINGLE,
            audio_path=Path(self.audio_path.text()) if self.audio_path.text() else None,
            subtitle_srt=Path(self.subtitle_path.text()) if self.subtitle_path.text() else None,
            import_subtitles=self.import_subtitles.isChecked(),
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
            # ── Draw Animation (Always active in normal production) ─────
            draw_enabled=True,
            draw_scene_json=Path(self.draw_scene_path.text()) if self.draw_scene_path.text() else None,
            draw_remove_background=self.draw_remove_bg.isChecked(),
            draw_fallback_basic=self.draw_fallback_basic.isChecked(),
            draw_diagnostics=self.draw_diagnostics.isChecked(),
            draw_reuse_cache=self.draw_reuse_cache.isChecked(),
        )

    def _create_project(self) -> None:
        try:
            config = self._config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return

        # ── Advanced draw setup preflight ─────────────────────────────────
        # Only block if fallback_basic is NOT checked (strict mode)
        if not self.draw_fallback_basic.isChecked():
            summary = getattr(self, "_draw_summary", None)
            if summary is None and config.effect_direction_srt and config.effect_direction_srt.is_file():
                try:
                    from auto_capcut.core.draw_setup import analyze_from_srt
                    from auto_capcut.core.draw_scene import load_scene
                    images = self._current_images()
                    scene_doc = None
                    if config.draw_scene_json and config.draw_scene_json.is_file():
                        scene_doc = load_scene(config.draw_scene_json)
                    summary = analyze_from_srt(images, config.effect_direction_srt, scene_doc)
                except Exception:
                    summary = None

            if summary is not None and not summary.all_ready:
                # Build detailed error message
                detail_lines = ["Advanced draw setup incomplete:\n"]
                for s in summary.incomplete_advanced:
                    detail_lines.append(f"● {s.image_name}:")
                    if s.missing_ids:
                        detail_lines.append(f"   Missing objects: {', '.join(s.missing_ids)}")
                    if s.missing_camera_frame_ids:
                        detail_lines.append(f"   Camera frame missing: {', '.join(s.missing_camera_frame_ids)}")
                    if not s.missing_ids and not s.missing_camera_frame_ids:
                        detail_lines.append(f"   {s.message}")

                msg = QMessageBox(self)
                msg.setWindowTitle("Advanced Draw Setup Incomplete")
                msg.setText("\n".join(detail_lines))
                msg.setIcon(QMessageBox.Icon.Warning)
                configure_btn = msg.addButton("Configure Advanced Images", QMessageBox.ButtonRole.ActionRole)
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                if msg.clickedButton() is configure_btn:
                    self._configure_advanced_images()
                return

        self.create_button.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Starting...")
        self.thread = QThread()
        self.worker = ProjectWorker(config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda v, m: (self.progress.setValue(v), self.status.setText(m)))
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()


    def _on_finished(self, results: list) -> None: self.progress.setValue(100); self.status.setText(f"Project created successfully: {', '.join(result.project_name for result in results)}")
    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Auto CapCut", message)
    def _thread_finished(self) -> None: self.create_button.setEnabled(True); self.thread = None; self.worker = None; self._update_enabled()

    def _save_settings(self) -> None:
        values = {
            "project_name": self.project_name.text(), "resolution": self.resolution.currentText(),
            "audio_path": self.audio_path.text(), "subtitle_path": self.subtitle_path.text(),
            "logo_path": self.logo_path.text(), "music_path": self.music_path.text(),
            "draft_path": self.draft_path.text(), "motion_enabled": self.motion_enabled.isChecked(),
            "transition_enabled": self.transition_enabled.isChecked(),
            "effect_path": self.effect_path.text(), "motion_mode": self.motion_mode.currentText(),
            "motion_strength": self.motion_strength.currentText(),
            # Draw Animation settings
            "draw_scene_path": self.draw_scene_path.text(),
            "draw_remove_bg": self.draw_remove_bg.isChecked(),
            "draw_fallback_basic": self.draw_fallback_basic.isChecked(),
            "draw_diagnostics": self.draw_diagnostics.isChecked(),
            "draw_reuse_cache": self.draw_reuse_cache.isChecked(),
        }
        for key, value in values.items(): self.settings.setValue(key, value)
        if self.image_list.count(): self.settings.setValue("image_folder", self.image_list.item(0).text())

    def _load_settings(self) -> None:
        self.settings.remove("asset_folder")
        self.settings.remove("visual_manifest_path")
        self.settings.remove("timing_path")
        self.settings.remove("draw_effect_path")
        self.settings.remove("draw_enabled")
        self.project_name.setText(self.settings.value("project_name", "")); resolution = self.settings.value("resolution", "1920x1080"); index = self.resolution.findText(str(resolution)); self.resolution.setCurrentIndex(index if index >= 0 else 0)
        for key, widget in (("audio_path", self.audio_path), ("subtitle_path", self.subtitle_path), ("logo_path", self.logo_path), ("music_path", self.music_path), ("draft_path", self.draft_path)): widget.setText(str(self.settings.value(key, widget.text())))
        self.effect_path.setText(str(self.settings.value("effect_path", ""))); saved_mode = str(self.settings.value("motion_mode", "Random Light")); legacy = {"Random": "Random Light", "Zoom In": "Subtle Zoom In", "Zoom Out": "Subtle Zoom Out", "Pan Left": "Subtle Pan Left", "Pan Right": "Subtle Pan Right"}; saved_mode = legacy.get(saved_mode, saved_mode); self.motion_mode.setCurrentText(saved_mode if self.motion_mode.findText(saved_mode) >= 0 else "Random Light"); self.motion_strength.setCurrentText(str(self.settings.value("motion_strength", MotionStrength.SUBTLE.value))); self.motion_enabled.setChecked(self.settings.value("motion_enabled", True, type=bool)); self.transition_enabled.setChecked(self.settings.value("transition_enabled", True, type=bool)); image_folder = str(self.settings.value("image_folder", ""));
        if image_folder: self.image_list.addItem(image_folder); self._image_list_changed()
        # Draw Animation settings
        self.draw_scene_path.setText(str(self.settings.value("draw_scene_path", "")))
        self.draw_remove_bg.setChecked(self.settings.value("draw_remove_bg", False, type=bool))
        self.draw_fallback_basic.setChecked(self.settings.value("draw_fallback_basic", True, type=bool))
        self.draw_diagnostics.setChecked(self.settings.value("draw_diagnostics", False, type=bool))
        self.draw_reuse_cache.setChecked(self.settings.value("draw_reuse_cache", True, type=bool))
        self._update_draw_scene_status()
