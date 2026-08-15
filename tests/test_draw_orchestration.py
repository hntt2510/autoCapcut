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


def test_draw_renderer_cache_reuse_and_invalidation(tmp_path: Path) -> None:
    """Requirement 4: draw_reuse_cache=True reuses rendered clip on identical signature,
    while draw_reuse_cache=False or changed parameters forces fresh render."""
    from unittest.mock import MagicMock, patch
    from PIL import Image as PILImage
    from auto_capcut.core.draw_renderer import DrawRenderer

    # Create dummy PNG
    img_path = tmp_path / "test.png"
    PILImage.new("RGB", (64, 64), color="white").save(img_path)

    effect = parse_draw_effect(effect_file(tmp_path / "effect.srt"))
    plan = effect.images[1]  # basic_draw

    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    renderer = DrawRenderer(cache_dir)

    # 1. First render with reuse_cache=True (cache miss -> calls FFmpeg)
    config = DrawProjectConfig(tmp_path, tmp_path / "effect.srt", out_dir, reuse_cache=True)
    out1 = out_dir / "001_draw.mp4"

    with patch("subprocess.Popen") as mock_popen, \
         patch("auto_capcut.core.draw_renderer._ffmpeg_exe", return_value="ffmpeg"), \
         patch("auto_capcut.core.draw_renderer.prepare_image") as mock_prep:
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Create dummy partial output so replace succeeds
        def fake_write(*args, **kwargs):
            partial = out1.with_name(f"{out1.stem}.partial{out1.suffix}")
            partial.write_bytes(b"rendered_video")
        mock_proc.stdin.close.side_effect = fake_write

        mock_art = MagicMock()
        mock_art.source_hash = "fakehash123"
        mock_art.strokes = []
        mock_prep.return_value = mock_art

        with patch.object(renderer, "_frame", return_value=PILImage.new("RGB", (1920, 1080))):
            renderer.render(img_path, plan, config, out1)
        assert mock_popen.call_count == 1


    # 2. Second render with reuse_cache=True on identical inputs (cache hit -> NO FFmpeg call)
    out2 = out_dir / "002_draw.mp4"
    with patch("subprocess.Popen") as mock_popen, \
         patch("auto_capcut.core.draw_renderer.prepare_image") as mock_prep:
        mock_art = MagicMock()
        mock_art.source_hash = "fakehash123"
        mock_art.strokes = []
        mock_prep.return_value = mock_art

        renderer.render(img_path, plan, config, out2)
        assert mock_popen.call_count == 0  # Reused from cache!
        assert out2.is_file()
        assert out2.read_bytes() == b"rendered_video"

    # 3. Third render with reuse_cache=False (force fresh render -> calls FFmpeg)
    config_no_cache = DrawProjectConfig(tmp_path, tmp_path / "effect.srt", out_dir, reuse_cache=False)
    out3 = out_dir / "003_draw.mp4"
    with patch("subprocess.Popen") as mock_popen, \
         patch("auto_capcut.core.draw_renderer._ffmpeg_exe", return_value="ffmpeg"), \
         patch("auto_capcut.core.draw_renderer.prepare_image") as mock_prep:

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        def fake_write3(*args, **kwargs):
            partial = out3.with_name(f"{out3.stem}.partial{out3.suffix}")
            partial.write_bytes(b"fresh_video")
        mock_proc.stdin.close.side_effect = fake_write3

        mock_art = MagicMock()
        mock_art.source_hash = "fakehash123"
        mock_art.strokes = []
        mock_prep.return_value = mock_art

        with patch.object(renderer, "_frame", return_value=PILImage.new("RGB", (1920, 1080))):
            renderer.render(img_path, plan, config_no_cache, out3)
        assert mock_popen.call_count == 1
        assert out3.read_bytes() == b"fresh_video"


def test_validate_targets_blocks_unknown_draw_object(tmp_path: Path) -> None:
    """Requirement 8: Explicit OBJECT_EFFECT or CAMERA_AFTER targeting unknown object must BLOCK with clear error."""
    from auto_capcut.core.draw_models import NormalizedRect, SceneImage, SceneObject
    from auto_capcut.core.draw_renderer import DrawRenderer

    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"png")

    srt = tmp_path / "effect.srt"
    srt.write_text(
        "\n".join([
            "1",
            "00:00:00,000 --> 00:00:05,000",
            "MODE advanced_draw",
            "STYLE v1",
            "OBJECT_EFFECT target=object_6 effect=draw",
            "DRAW 0s-5s:",
            "",
        ]),
        encoding="utf-8",
    )
    effect = parse_draw_effect(srt)
    plan = effect.images[0]

    # Scene only contains object_1 .. object_5
    objects = [
        SceneObject(id=f"object_{i}", type="art", box=NormalizedRect(0.1 * i, 0.1, 0.08, 0.08))
        for i in range(1, 6)
    ]
    scene = SceneImage((1920, 1080), "hash", objects, [f"object_{i}" for i in range(1, 6)])

    renderer = DrawRenderer(tmp_path / "cache")
    config = DrawProjectConfig(tmp_path, srt, tmp_path / "out")

    with pytest.raises(
        DrawRenderError,
        match=r'Image 001: unknown draw object "object_6"\. Available objects: object_1, object_2, object_3, object_4, object_5',
    ):
        renderer.render(img_path, plan, config, tmp_path / "out" / "001_draw.mp4", scene)



