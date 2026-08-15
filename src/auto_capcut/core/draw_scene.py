from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from auto_capcut.core.draw_models import DrawObjectDirection, DrawObjectEffect, NormalizedRect, SceneDocument, SceneImage, SceneObject
from auto_capcut.core.errors import SceneValidationError

_OBJECT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rect(value, label: str, errors: list[str]) -> NormalizedRect | None:
    if not isinstance(value, dict):
        errors.append(f"{label}: rectangle must be an object")
        return None
    try:
        result = NormalizedRect(*(float(value[key]) for key in ("x", "y", "w", "h")))
    except (KeyError, TypeError, ValueError):
        errors.append(f"{label}: rectangle requires numeric x, y, w, h")
        return None
    if result.w <= 0 or result.h <= 0 or min(result.x, result.y) < 0 or result.x + result.w > 1 or result.y + result.h > 1:
        errors.append(f"{label}: rectangle must be positive and inside 0..1")
        return None
    return result


def _optional_seconds(value, label: str, errors: list[str], positive: bool = False) -> int | None:
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}: duration must be numeric")
        return None
    if not (seconds >= 0) or (positive and seconds <= 0):
        errors.append(f"{label}: duration must be {'positive' if positive else 'non-negative'}")
        return None
    return round(seconds * 1_000_000)


def load_scene(path: str | Path) -> SceneDocument:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneValidationError(f"Unable to read scene JSON: {source}: {exc}") from exc
    errors: list[str] = []
    # Accept both schema_version=1 (canonical) and version=1 (legacy alias) silently.
    _schema_v = raw.get("schema_version") if "schema_version" in raw else raw.get("version")
    if not isinstance(raw, dict) or _schema_v != 1:
        raise SceneValidationError("scene.json: schema_version must be 1")
    raw_images = raw.get("images")
    if not isinstance(raw_images, dict):
        raise SceneValidationError("scene.json: images must be an object")
    images: dict[str, SceneImage] = {}
    for filename, record in raw_images.items():
        label = f"Image {filename}"
        if not isinstance(filename, str) or not filename:
            errors.append("scene.json: image keys must be non-empty filenames")
            continue
        if not isinstance(record, dict):
            errors.append(f"{label}: record must be an object")
            continue
        size = record.get("source_size")
        try:
            source_size = (int(size["width"]), int(size["height"]))
            if min(source_size) <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: source_size requires positive width and height")
            continue
        raw_objects = record.get("objects")
        raw_order = record.get("draw_order")
        if not isinstance(raw_objects, list):
            errors.append(f"{label}: objects must be an array")
            raw_objects = []
        if not isinstance(raw_order, list) or not all(isinstance(item, str) for item in raw_order):
            errors.append(f"{label}: draw_order must be an array of object IDs")
            raw_order = []
        objects: list[SceneObject] = []
        ids: set[str] = set()
        for object_index, value in enumerate(raw_objects, 1):
            object_label = f"{label} object {object_index}"
            if not isinstance(value, dict):
                errors.append(f"{object_label}: must be an object")
                continue
            object_id = value.get("id")
            object_type = value.get("type")
            if not isinstance(object_id, str) or not _OBJECT_ID.fullmatch(object_id):
                errors.append(f"{object_label}: id is invalid")
                continue
            if object_id in ids:
                errors.append(f"{label}: duplicate object id {object_id}")
                continue
            if object_type not in {"art", "text", "warning"}:
                errors.append(f"{object_label} {object_id}: type must be art, text, or warning")
                continue
            box = _rect(value.get("box"), f"{object_label} {object_id} box", errors)
            camera = None
            if value.get("camera_frame") is not None:
                camera = _rect(value["camera_frame"], f"{object_label} {object_id} camera_frame", errors)
            if box is not None:
                effect = value.get("render_effect", DrawObjectEffect.DRAW.value)
                direction = value.get("direction", DrawObjectDirection.AUTO.value)
                if not isinstance(effect, str) or not effect:
                    errors.append(f"{object_label} {object_id}: render_effect must be a string")
                    effect = DrawObjectEffect.DRAW.value
                if not isinstance(direction, str) or not direction:
                    errors.append(f"{object_label} {object_id}: direction must be a string")
                    direction = DrawObjectDirection.AUTO.value
                duration_us = _optional_seconds(value.get("duration"), f"{object_label} {object_id} duration", errors, positive=True)
                pause_after_us = _optional_seconds(value.get("pause_after"), f"{object_label} {object_id} pause_after", errors)
                ids.add(object_id)
                behavior_fields = frozenset(key for key in ("render_effect", "direction", "duration", "pause_after") if key in value)
                objects.append(SceneObject(object_id, object_type, box, camera, effect.casefold(), direction.casefold(), duration_us, pause_after_us, behavior_fields))
        order = tuple(raw_order)
        if set(order) != ids or len(order) != len(ids):
            errors.append(f"{label}: draw_order must contain every object exactly once")
        images[filename] = SceneImage(filename, source_size, tuple(objects), order, record.get("source_sha256"))
        if record.get("source_sha256") is not None and not isinstance(record.get("source_sha256"), str):
            errors.append(f"{label}: source_sha256 must be a string")
    if errors:
        raise SceneValidationError("Invalid scene.json:\n" + "\n".join(f"- {item}" for item in errors))
    return SceneDocument(1, images, source)


load_scene_json = load_scene


def validate_scene_document(scene: SceneDocument, image_paths: list[Path], canvas_size: tuple[int, int]) -> list[str]:
    errors: list[str] = []
    expected_aspect = canvas_size[0] / canvas_size[1]
    by_name = {path.name.casefold(): path for path in image_paths}
    for key, image in scene.images.items():
        path = by_name.get(key.casefold())
        if path is None:
            errors.append(f"Image {key}: not found in image folder")
            continue
        try:
            from PIL import Image

            with Image.open(path) as source:
                if source.size != image.source_size:
                    errors.append(f"Image {key}: source size changed ({source.size} != {image.source_size})")
        except OSError as exc:
            errors.append(f"Image {key}: unable to inspect source: {exc}")
        if image.source_sha256:
            try:
                if sha256_file(path).casefold() != str(image.source_sha256).casefold():
                    errors.append(f"Image {key}: source hash changed")
            except OSError as exc:
                errors.append(f"Image {key}: unable to hash source: {exc}")
        for obj in image.objects:
            if obj.camera_frame and abs(obj.camera_frame.w / obj.camera_frame.h - expected_aspect) > 0.01:
                errors.append(f"Image {key} object {obj.id}: camera_frame aspect does not match output")
    return errors


def _rect_dict(rect: NormalizedRect) -> dict[str, float]:
    return {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h}


def scene_to_dict(scene: SceneDocument) -> dict:
    images = {}
    for filename, image in scene.images.items():
        images[filename] = {
            "source_size": {"width": image.source_size[0], "height": image.source_size[1]},
            **({"source_sha256": image.source_sha256} if image.source_sha256 else {}),
            "objects": [
                {
                    "id": obj.id,
                    "type": obj.type,
                    "box": _rect_dict(obj.box),
                    **({"camera_frame": _rect_dict(obj.camera_frame)} if obj.camera_frame else {}),
                    **({"render_effect": obj.render_effect} if "render_effect" in obj.behavior_fields_present or obj.render_effect != "draw" else {}),
                    **({"direction": obj.direction} if "direction" in obj.behavior_fields_present or obj.direction != "auto" else {}),
                    **({"duration": obj.duration_us / 1_000_000} if obj.duration_us is not None else {}),
                    **({"pause_after": obj.pause_after_us / 1_000_000} if obj.pause_after_us is not None else {}),
                }
                for obj in image.objects
            ],
            "draw_order": list(image.draw_order),
        }
    return {"schema_version": 1, "images": images}


def save_scene(scene: SceneDocument, path: str | Path | None = None) -> Path:
    destination = Path(path or scene.path or "scene.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(scene_to_dict(scene), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        Path(temporary).replace(destination)
    finally:
        temp = Path(temporary)
        if temp.exists():
            temp.unlink()
    return destination
