from pathlib import Path

import pytest

from auto_capcut.core.draw_effect_parser import parse_draw_effect
from auto_capcut.core.draw_models import DrawProjectConfig
from auto_capcut.core.draw_renderer import DrawRenderService
from auto_capcut.core.errors import DrawRenderError


class FakeRenderer:
    def __init__(self) -> None:
        self.calls = []

    def render(self, image, plan, config, output, scene=None, progress=None):
        self.calls.append((image.name, plan.mode.value, output.name, scene))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake")
        return output


def effect_file(path: Path) -> Path:
    path.write_text(
        """
        1
        00:00:00,000 --> 00:00:01,000
        MODE=advanced_draw
        STYLE=v1
        DRAW 0s-1s:

        2
        00:00:01,000 --> 00:00:02,000
        MODE=basic_draw
        STYLE=v1
        DRAW 0s-1s:
        """.strip() + "\n",
        encoding="utf-8",
    )
    return path


def test_render_all_orders_outputs_and_falls_back_to_basic(tmp_path: Path) -> None:
    effect = parse_draw_effect(effect_file(tmp_path / "effect.srt"))
    images = [tmp_path / "2.png", tmp_path / "10.png"]
    for image in images:
        image.write_bytes(b"image")
    fake = FakeRenderer()
    config = DrawProjectConfig(tmp_path, tmp_path / "effect.srt", tmp_path / "out", fallback_basic=True)
    outputs = DrawRenderService(fake).render_project(config, effect, images)
    assert [output.name for output in outputs] == ["001_draw.mp4", "002_draw.mp4"]
    assert fake.calls[0][1] == "basic_draw"
    assert fake.calls[1][1] == "basic_draw"


def test_advanced_missing_scene_can_block(tmp_path: Path) -> None:
    effect = parse_draw_effect(effect_file(tmp_path / "effect.srt"))
    image = tmp_path / "001.png"
    image.write_bytes(b"image")
    config = DrawProjectConfig(tmp_path, tmp_path / "effect.srt", tmp_path / "out", fallback_basic=False)
    with pytest.raises(DrawRenderError, match="scene record missing"):
        DrawRenderService(FakeRenderer()).render_project(config, effect, [image, image])


# ── New tests for render_subset ────────────────────────────────────────────

def test_render_subset_by_index(tmp_path: Path) -> None:
    """render_subset only renders the requested indexes."""
    effect = parse_draw_effect(effect_file(tmp_path / "effect.srt"))
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    for img in images:
        img.write_bytes(b"image")
    fake = FakeRenderer()
    config = DrawProjectConfig(tmp_path, tmp_path / "effect.srt", tmp_path / "out", fallback_basic=True)
    result = DrawRenderService(fake).render_subset(
        config,
        list(effect.images),
        images,
        [1],  # only render second image
    )
    assert len(result) == 1
    assert 1 in result
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "b.png"


def test_render_subset_all_indexes(tmp_path: Path) -> None:
    """render_subset with all indexes renders everything."""
    effect = parse_draw_effect(effect_file(tmp_path / "effect.srt"))
    images = [tmp_path / "img1.png", tmp_path / "img2.png"]
    for img in images:
        img.write_bytes(b"image")
    fake = FakeRenderer()
    config = DrawProjectConfig(tmp_path, tmp_path / "effect.srt", tmp_path / "out", fallback_basic=True)
    result = DrawRenderService(fake).render_subset(
        config,
        list(effect.images),
        images,
        [0, 1],
    )
    assert sorted(result.keys()) == [0, 1]
    assert len(fake.calls) == 2


def test_scene_json_version_key_compat(tmp_path: Path) -> None:
    """scene.json using 'version': 1 (legacy) is accepted without error."""
    from auto_capcut.core.draw_scene import load_scene

    scene_json = tmp_path / "scene.json"
    scene_json.write_text(
        '{"version": 1, "images": {"img.png": {"source_size": {"width": 100, "height": 100}, "objects": [], "draw_order": []}}}',
        encoding="utf-8",
    )
    doc = load_scene(scene_json)
    assert doc.schema_version == 1

