from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from auto_capcut.core.errors import ValidationError
from auto_capcut.models import ProjectConfig
from auto_capcut.utils.natural_sort import natural_sorted

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}


def collect_images(folders: list[str | Path]) -> list[Path]:
    images: list[Path] = []
    for folder in folders:
        root = Path(folder)
        if not root.is_dir():
            raise ValidationError(f"Image folder does not exist: {root}")
        folder_images = (
            path.resolve()
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.casefold() in IMAGE_EXTENSIONS
            # Asset-sheet files from older projects are intentionally inert and
            # must never become timeline images after the asset workflow was
            # removed.
            and not re.match(r"^\d+_assets\.[^.]+$", path.name, re.IGNORECASE)
        )
        images.extend(natural_sorted(folder_images, key=lambda path: path.name))
    return images


def collect_audio(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise ValidationError(f"Audio folder does not exist: {root}")
    return natural_sorted((path.resolve() for path in root.iterdir() if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS), key=lambda path: path.name)


def probe_duration_us(path: str | Path) -> int:
    source = Path(path)
    try:
        import pymediainfo

        info = pymediainfo.MediaInfo.parse(str(source))
        tracks = [track for track in info.tracks if getattr(track, "duration", None) is not None and track.track_type in {"Audio", "Video"}]
        if tracks:
            return max(1, int(float(tracks[0].duration) * 1000))
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                seconds = float(json.loads(result.stdout)["format"]["duration"])
                return max(1, round(seconds * 1_000_000))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    raise ValidationError(f"Unable to read media duration: {source}")


def validate_audio(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Main audio file does not exist: {path}")
    if path.suffix.casefold() not in AUDIO_EXTENSIONS:
        raise ValidationError(f"Unsupported audio format: {path.suffix}")


def validate_config_paths(config: ProjectConfig) -> None:
    if not config.image_folders:
        raise ValidationError("Add at least one image folder")
    collect_images(config.image_folders)
    audio_mode = str(getattr(config.audio_mode, "value", config.audio_mode))
    if audio_mode == "single":
        if config.audio_path is None:
            raise ValidationError("Choose a main audio file")
        validate_audio(config.audio_path)
    else:
        if config.audio_path is None:
            raise ValidationError("Choose an audio folder")
        if not collect_audio(config.audio_path):
            raise ValidationError("No supported audio files found")
    if config.import_subtitles and audio_mode == "single" and config.subtitle_srt and not config.subtitle_srt.is_file():
        raise ValidationError("Subtitle SRT does not exist")
    if config.use_image_timing and (config.image_timing_srt is None or not config.image_timing_srt.is_file()):
        raise ValidationError("Image Timing SRT does not exist")
    motion_mode = str(getattr(config, "motion_mode", "")).casefold()
    if config.motion_enabled and motion_mode == "effect direction srt":
        if config.effect_direction_srt is None or not config.effect_direction_srt.is_file():
            raise ValidationError("Effect Direction SRT does not exist")
    if config.logo_enabled and (config.logo_path is None or not config.logo_path.is_file()):
        raise ValidationError("Logo file is invalid")
    if config.music_enabled and (config.music_folder is None or not config.music_folder.is_dir()):
        raise ValidationError("Music folder does not exist")
    if config.draft_folder is None or not config.draft_folder.is_dir():
        raise ValidationError("CapCut draft folder not found")
