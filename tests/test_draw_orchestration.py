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
