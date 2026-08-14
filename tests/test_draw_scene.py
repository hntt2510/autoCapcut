import json
from pathlib import Path

from PIL import Image
import pytest

from auto_capcut.core.draw_models import NormalizedRect, SceneDocument, SceneImage, SceneObject
from auto_capcut.core.draw_scene import load_scene, save_scene, validate_scene_document
from auto_capcut.core.errors import SceneValidationError


def test_scene_round_trip_and_validation(tmp_path: Path) -> None:
    image = tmp_path / "001.png"
    Image.new("RGB", (100, 100), "white").save(image)
    scene_path = tmp_path / "scene.json"
    document = SceneDocument(1, {"001.png": SceneImage("001.png", (100, 100), (SceneObject("label", "text", NormalizedRect(.1, .1, .5, .2)),), ("label",))}, scene_path)
    save_scene(document)
    loaded = load_scene(scene_path)
    assert loaded.images["001.png"].objects[0].id == "label"
    assert validate_scene_document(loaded, [image], (1920, 1080)) == []


def test_scene_loader_reports_invalid_order_and_rect(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"schema_version": 1, "images": {"001.png": {"source_size": {"width": 10, "height": 10}, "objects": [{"id": "bad id", "type": "art", "box": {"x": 0, "y": 0, "w": 2, "h": 1}}], "draw_order": []}}}), encoding="utf-8")
    with pytest.raises(SceneValidationError, match="invalid|inside|draw_order"):
        load_scene(path)
