from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from auto_capcut.models import CameraFrame, EffectCue, RoiRequirement, RoiTarget, TargetROI
from auto_capcut.core.camera_frame import validate_camera_frame


def roi_sidecar_path(effect_srt: Path) -> Path:
    return effect_srt.with_suffix(".roi.json")


def _valid_roi(value) -> TargetROI | None:
    if not isinstance(value, dict):
        return None
    try:
        x, y, width, height = (float(value[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    if min(x, y, width, height) < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        return None
    return TargetROI(x, y, width, height)


def validate_saved_frame(
    frame: CameraFrame,
    image_path: Path,
    canvas_size: tuple[int, int],
) -> tuple[bool, str]:
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            result = validate_camera_frame(frame, image.size, canvas_size)
    except (OSError, ValueError) as exc:
        return False, str(exc)
    return result.valid, result.reason


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        Path(temp_name).replace(path)
    finally:
        temp = Path(temp_name)
        if temp.exists():
            temp.unlink()


def discover_roi_targets(effects: list[EffectCue]) -> tuple[RoiTarget, ...]:
    from auto_capcut.core.effect_direction_parser import required_roi_targets, optional_roi_targets

    return required_roi_targets(effects) + optional_roi_targets(effects)


class RoiResolver:
    def resolve(self, image_path: Path, target_id: str, image_index: int) -> TargetROI | None:
        raise NotImplementedError


class ManualRoiResolver(RoiResolver):
    def __init__(self, sidecar_path: Path | None) -> None:
        self.sidecar_path = sidecar_path
        self.records: dict[str, dict] = {}
        self.warnings: list[str] = []
        if sidecar_path and sidecar_path.is_file():
            try:
                raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
                self.records = self._normalize_records(raw)
            except (OSError, json.JSONDecodeError):
                self.warnings.append(f"ROI sidecar could not be read: {sidecar_path}")

    @staticmethod
    def _normalize_records(raw) -> dict[str, dict]:
        if not isinstance(raw, dict):
            return {}
        if isinstance(raw.get("images"), dict):
            return {str(index): value for index, value in raw["images"].items() if isinstance(value, dict)}
        # Legacy v1 records were one ROI per image. Keep source-less records;
        # old AI records are deliberately ignored after the manual-only migration.
        output: dict[str, dict] = {}
        for index, value in raw.items():
            if not isinstance(value, dict) or str(value.get("source", "")).casefold() == "vision":
                continue
            roi = _valid_roi(value)
            if roi is not None:
                output[str(index)] = {"image_path": value.get("image_path", ""), "targets": {"__legacy__": value}}
        return output

    def resolve(self, image_path: Path, target_id: str, image_index: int) -> TargetROI | None:
        record = self.records.get(str(image_index))
        if not isinstance(record, dict):
            return None
        stored_path = record.get("image_path")
        if stored_path and Path(stored_path).resolve() != image_path.resolve():
            self.warnings.append(f"ROI for Image {image_index} does not match the current image")
            return None
        targets = record.get("targets", {})
        if not isinstance(targets, dict):
            return None
        value = targets.get(target_id)
        # Vision records from the removed workflow are never authoritative for
        # this manual-only engine, including when they appear in a v2 sidecar.
        if isinstance(value, dict) and str(value.get("source", "")).casefold() == "vision":
            return None
        if value is None and target_id and "__legacy__" in targets:
            value = targets["__legacy__"]
        roi = _valid_roi(value)
        if value is not None and roi is None:
            self.warnings.append(f"ROI for Image {image_index} target {target_id} is invalid")
        return roi

    def configured(self, target: RoiTarget, image_path: Path) -> bool:
        return self.resolve(image_path, target.target_id, target.image_index) is not None

    def frame_status(self, target: RoiTarget, image_path: Path, canvas_size: tuple[int, int]) -> tuple[bool, str]:
        frame = self.resolve(image_path, target.target_id, target.image_index)
        if frame is None:
            return False, "missing"
        return validate_saved_frame(frame, image_path, canvas_size)

    def save(self, image_path: Path, target: RoiTarget, roi: TargetROI | None) -> None:
        if self.sidecar_path is None:
            raise ValueError("ROI sidecar path is not configured")
        records = self.records
        record = records.setdefault(str(target.image_index), {"image_path": str(image_path.resolve()), "targets": {}})
        record["image_path"] = str(image_path.resolve())
        targets = record.setdefault("targets", {})
        # Saving an explicit target upgrades a migrated one-ROI record to the
        # canonical v2 target map instead of retaining an ambiguous alias.
        targets.pop("__legacy__", None)
        if roi is None:
            targets.pop(target.target_id, None)
        else:
            targets[target.target_id] = {"x": roi.x, "y": roi.y, "w": roi.width, "h": roi.height}
        payload = {"schema_version": 2, "images": records}
        _atomic_write(self.sidecar_path, payload)


class NullRoiResolver(RoiResolver):
    def resolve(self, image_path: Path, target_id: str, image_index: int) -> TargetROI | None:
        return None
