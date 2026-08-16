from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from auto_capcut.core.alert_overlay import create_alert_overlay
from auto_capcut.core.effect_direction_parser import parse_effect_direction_srt, required_roi_targets
from auto_capcut.core.errors import EffectDirectionError, ValidationError
from auto_capcut.core.motion_engine import MotionEngine
from auto_capcut.core.media import collect_images
from auto_capcut.core.roi_camera import calculate_roi_framing, clamp_camera_transform_to_cover
from auto_capcut.core.planning import validate_required_rois
from auto_capcut.core.roi_resolver import ManualRoiResolver, roi_sidecar_path
from auto_capcut.models import AudioMode, MotionStrength, ProjectConfig, ProjectJob, RESOLUTIONS, TargetROI
from auto_capcut.core.camera_frame import calculate_camera_transform, project_camera_frame_center, validate_camera_frame
from auto_capcut.core.capcut_builder import CapCutBuilder
from auto_capcut.models import EffectCue, ImageTiming, VisualEffect


def _srt(text: str, tmp_path: Path) -> Path:
    path = tmp_path / "effects.srt"
    path.write_text(text, encoding="utf-8")
    return path


def test_modern_effects_parse_and_deduplicate_targets(tmp_path: Path) -> None:
    path = _srt(
        "1\n00:00:00,000 --> 00:00:08,000\nImage 001 FX\n"
        "HOLD 0.00s–1.00s: show full composition\n"
        "FOCUS_ZOOM 1.00s–2.60s:\n" "target=part_ab\nzoom=1.05\n"
        "PAN_TO 2.60s–4.10s:\n" "target=part_c\nzoom=1.05\n"
        "PULL_TO 4.10s–5.70s:\n" "target=part_d\nstrength=subtle\n"
        "ALERT 5.70s–7.20s:\n" "target=penalty\nstyle=red_warning\npulse=1\n"
        "SETTLE 7.20s–8.00s: hold final composition\n"
    , tmp_path)
    cue = parse_effect_direction_srt(path)[0]
    assert [effect.type for effect in cue.effects] == ["HOLD", "FOCUS_ZOOM", "PAN_TO", "PULL_TO", "ALERT", "SETTLE"]
    assert [target.target_id for target in required_roi_targets([cue])] == ["part_ab", "part_c", "part_d", "penalty"]


def test_target_reuse_is_one_roi(tmp_path: Path) -> None:
    path = _srt("1\n00:00:00,000 --> 00:00:02,000\nFOCUS_ZOOM 0s–1s:\ntarget=penalty\nPAN_TO 1s–2s:\ntarget=penalty\n", tmp_path)
    targets = required_roi_targets(parse_effect_direction_srt(path))
    assert len(targets) == 1
    assert targets[0].effect_types == ("FOCUS_ZOOM", "PAN_TO")


def test_no_roi_effects_and_legacy_focus_are_valid(tmp_path: Path) -> None:
    path = _srt("1\n00:00:00,000 --> 00:00:02,000\nHOLD 0s–1s:\nfull\nSUBTLE_ZOOM_IN 1s–2s:\nsubtle\n", tmp_path)
    assert required_roi_targets(parse_effect_direction_srt(path)) == ()
    legacy = _srt("1\n00:00:00,000 --> 00:00:01,000\nTarget: calendar\nFOCUS 0s–1s: push in\n", tmp_path)
    assert parse_effect_direction_srt(legacy)[0].optional_roi_targets[0].target_id == "calendar"


def test_required_modern_effect_without_target_fails(tmp_path: Path) -> None:
    path = _srt("1\n00:00:00,000 --> 00:00:01,000\nFOCUS_ZOOM 0s–1s:\nzoom=1.05\n", tmp_path)
    with pytest.raises(EffectDirectionError, match="requires target"):
        parse_effect_direction_srt(path)


def test_legacy_asset_is_explicitly_rejected(tmp_path: Path) -> None:
    path = _srt("1\n00:00:00,000 --> 00:00:01,000\nASSET 0s–1s:\nid=card\nslot=1\n", tmp_path)
    with pytest.raises(EffectDirectionError, match="Unsupported legacy ASSET command"):
        parse_effect_direction_srt(path)


def test_roi_sidecar_v2_round_trip_and_legacy_manual_migration(tmp_path: Path) -> None:
    image = tmp_path / "001.png"
    Image.new("RGB", (100, 100), "white").save(image)
    sidecar = tmp_path / "effects.roi.json"
    sidecar.write_text(json.dumps({"1": {"image_path": str(image), "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}), encoding="utf-8")
    resolver = ManualRoiResolver(sidecar)
    assert resolver.resolve(image, "target", 1) == TargetROI(0.1, 0.2, 0.3, 0.4)
    from auto_capcut.models import RoiTarget
    target = RoiTarget(1, "target")
    resolver.save(image, target, TargetROI(0.2, 0.2, 0.2, 0.2))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["images"]["1"]["targets"]["target"]["x"] == 0.2


def test_vision_sidecar_records_are_ignored(tmp_path: Path) -> None:
    image = tmp_path / "001.png"
    image.write_bytes(b"image")
    sidecar = tmp_path / "effects.roi.json"
    sidecar.write_text(json.dumps({"1": {"source": "vision", "image_path": str(image), "x": 0, "y": 0, "w": 1, "h": 1}}), encoding="utf-8")
    assert ManualRoiResolver(sidecar).resolve(image, "target", 1) is None


def test_missing_required_roi_blocks_build(tmp_path: Path) -> None:
    image = tmp_path / "001.png"; image.write_bytes(b"image")
    effect_path = _srt("1\n00:00:00,000 --> 00:00:01,000\nFOCUS_ZOOM 0s–1s:\ntarget=calendar\n", tmp_path)
    config = ProjectConfig(image_folders=[tmp_path], audio_mode=AudioMode.SINGLE, effect_direction_srt=effect_path)
    job = ProjectJob("test", (image,), image, None, None, config)
    with pytest.raises(ValidationError, match="Image 001"):
        validate_required_rois(job, parse_effect_direction_srt(effect_path))


def test_motion_continuity_and_cover_bounds(tmp_path: Path) -> None:
    path = _srt("1\n00:00:00,000 --> 00:00:08,000\nFOCUS_ZOOM 0s–2s:\ntarget=a\nPAN_TO 2s–4s:\ntarget=b\nPULL_TO 4s–6s:\ntarget=c\nSETTLE 6s–8s:\nhold\n", tmp_path)
    cue = parse_effect_direction_srt(path)[0]
    sidecar = roi_sidecar_path(path)
    resolver = ManualRoiResolver(sidecar)
    image = tmp_path / "001.png"; image.write_bytes(b"image")
    from auto_capcut.models import RoiTarget
    for target in required_roi_targets([cue]): resolver.save(image, target, TargetROI(0.2, 0.2, 0.2, 0.2))
    engine = MotionEngine(1920, 1080, MotionStrength.SUBTLE.value, resolver)
    plan = engine.plan_effect(cue, image, 1376, 768, 8_000_000)
    assert plan.keyframes[0].transform.relative_scale == 1.0
    assert all(key.transform.relative_scale >= 1.0 for key in plan.keyframes)
    assert plan.keyframes[-1].transform == plan.keyframes[-2].transform


def test_alert_overlay_is_deterministic_and_preserves_dimensions(tmp_path: Path) -> None:
    image = tmp_path / "001.png"; Image.new("RGB", (320, 180), "white").save(image)
    effect_path = _srt("1\n00:00:00,000 --> 00:00:01,000\nALERT 0s–1s:\ntarget=warning\nstyle=red_warning\npulse=1\n", tmp_path)
    effect = parse_effect_direction_srt(effect_path)[0].effects[0]
    output = create_alert_overlay(image, TargetROI(0.25, 0.25, 0.5, 0.5), effect, tmp_path / "effects.overlays")
    assert output == create_alert_overlay(image, TargetROI(0.25, 0.25, 0.5, 0.5), effect, tmp_path / "effects.overlays")
    with Image.open(output) as overlay:
        assert overlay.mode == "RGBA"
        assert overlay.size == (320, 180)


def test_alert_overlay_honors_dim_setting(tmp_path: Path) -> None:
    image = tmp_path / "001.png"; Image.new("RGB", (100, 100), "white").save(image)
    effect_path = _srt("1\n00:00:00,000 --> 00:00:01,000\nALERT 0s–1s:\ntarget=warning\ndim_others=0.30\n", tmp_path)
    effect = parse_effect_direction_srt(effect_path)[0].effects[0]
    output = create_alert_overlay(image, TargetROI(0.25, 0.25, 0.5, 0.5), effect, tmp_path / "overlays")
    with Image.open(output) as overlay:
        assert overlay.getpixel((0, 0))[3] == pytest.approx(round(255 * 0.30), abs=1)
        assert overlay.getpixel((50, 50))[3] == 0


def test_alert_medium_framing_is_less_tight_than_roi_cover() -> None:
    from auto_capcut.core.roi_camera import medium_roi_framing
    roi = TargetROI(0.4, 0.4, 0.1, 0.1)
    full = calculate_roi_framing(roi, (1752, 978), (1920, 1080), 0.10, 2.5)
    medium = medium_roi_framing(roi, 1752, 978, 1920, 1080, 2.5, 0.65)
    assert 1.0 <= medium.transform.relative_scale < full.transform.relative_scale


def test_cover_scale() -> None:
    assert MotionEngine.capcut_cover_scale(1920, 1080, 1376, 768) == pytest.approx(1.0078125)


def test_medicare_acceptance_requires_four_rois(tmp_path: Path) -> None:
    path = _srt(
        "1\n00:00:00,000 --> 00:00:08,000\nImage 001 FX\n"
        "HOLD 0.00s–1.00s:\nshow full composition\n"
        "FOCUS_ZOOM 1.00s–2.60s:\ntarget=part_ab\nzoom=1.05\n"
        "PAN_TO 2.60s–4.10s:\ntarget=part_c\nzoom=1.05\n"
        "PULL_TO 4.10s–5.70s:\ntarget=part_d\nstrength=subtle\n"
        "ALERT 5.70s–7.20s:\ntarget=penalty\nstyle=red_warning\npulse=1\n"
        "SETTLE 7.20s–8.00s:\nhold final composition\n", tmp_path)
    targets = required_roi_targets(parse_effect_direction_srt(path))
    assert len(targets) == 4
    assert {target.target_id for target in targets} == {"part_ab", "part_c", "part_d", "penalty"}


def test_inline_properties_and_asset_sheets_are_not_timeline_images(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "001.png")
    Image.new("RGB", (8, 8), "green").save(tmp_path / "001_assets.png")
    assert [path.name for path in collect_images([tmp_path])] == ["001.png"]
    effect = _srt(
        "1\n00:00:00,000 --> 00:00:01,000\n"
        "FOCUS_ZOOM 0s–1s: target=part_ab zoom=1.08 easing=smoothstep\n",
        tmp_path,
    )
    parsed = parse_effect_direction_srt(effect)[0].effects[0]
    assert parsed.target_id == "part_ab"
    assert parsed.params == {"zoom": "1.08", "easing": "smoothstep"}


@pytest.mark.parametrize("roi", [
    TargetROI(0.0, 0.0, 0.18, 0.12),
    TargetROI(0.82, 0.88, 0.18, 0.12),
    TargetROI(0.45, 0.45, 0.01, 0.01),
    TargetROI(0.05, 0.10, 0.90, 0.80),
])
def test_roi_framing_is_uniform_and_cover_safe(roi: TargetROI) -> None:
    framing = calculate_roi_framing(roi, (1752, 978), (1920, 1080), 0.10, 2.5)
    transform = framing.transform
    assert transform.relative_scale >= 1.0
    safe = clamp_camera_transform_to_cover(transform, 1752, 978, 1920, 1080)
    assert safe == transform
    pixel_aspect = (framing.adjusted_roi.width * 1752) / (framing.adjusted_roi.height * 978)
    assert pixel_aspect == pytest.approx(16 / 9, rel=1e-5)


def test_roi_framing_respects_max_zoom() -> None:
    framing = calculate_roi_framing(TargetROI(0.49, 0.49, 0.01, 0.01), (1752, 978), (1920, 1080), 0.10, 1.4)
    assert framing.transform.relative_scale <= 1.4


def test_roi_framing_uses_both_uniform_scale_axes() -> None:
    framing = calculate_roi_framing(TargetROI(0.40, 0.40, 0.20, 0.02), (1000, 2000), (1920, 1080), 0.10, 3.0)
    assert framing.transform.relative_scale >= 1.0
    assert clamp_camera_transform_to_cover(framing.transform, 1000, 2000, 1920, 1080) == framing.transform


def test_roi_effects_use_target_framing_and_pull_midpoint(tmp_path: Path) -> None:
    path = _srt(
        "1\n00:00:00,000 --> 00:00:08,000\n"
        "HOLD 0s–1s:\n"
        "FOCUS_ZOOM 1s–2.6s:\ntarget=a\n"
        "PAN_TO 2.6s–4.1s:\ntarget=b\n"
        "PULL_TO 4.1s–5.7s:\ntarget=c\n"
        "ALERT 5.7s–7.2s:\ntarget=d\n"
        "SETTLE 7.2s–8s:\n",
        tmp_path,
    )
    cue = parse_effect_direction_srt(path)[0]
    image = tmp_path / "001.png"
    Image.new("RGB", (1752, 978), "white").save(image)
    resolver = ManualRoiResolver(roi_sidecar_path(path))
    from auto_capcut.models import RoiTarget
    for target, roi in zip(required_roi_targets([cue]), [
        TargetROI(0.05, 0.05, 0.25, 0.25), TargetROI(0.60, 0.05, 0.25, 0.25),
        TargetROI(0.05, 0.60, 0.25, 0.25), TargetROI(0.60, 0.60, 0.25, 0.25),
    ]):
        resolver.save(image, target, roi)
    plan = MotionEngine(1920, 1080, MotionStrength.SUBTLE.value, resolver).plan_effect(cue, image, 1752, 978, 8_000_000)
    by_time = {key.local_time_us: key.transform for key in plan.keyframes}
    assert by_time[0].relative_scale == 1.0
    assert by_time[2_600_000] != by_time[1_000_000]
    assert any(key.local_time_us == 4_740_000 for key in plan.keyframes)
    assert by_time[8_000_000] == by_time[max(by_time)]


@pytest.mark.parametrize("source,canvas", [
    ((1752, 978), (1920, 1080)),
    ((1752, 978), (1080, 1920)),
    ((1752, 978), (1080, 1080)),
])
def test_camera_frame_validation_uses_source_pixel_aspect(source, canvas) -> None:
    source_width, source_height = source
    canvas_width, canvas_height = canvas
    pixel_width = 320
    pixel_height = round(pixel_width / (canvas_width / canvas_height))
    frame = TargetROI(20 / source_width, 20 / source_height, pixel_width / source_width, pixel_height / source_height)
    assert validate_camera_frame(frame, source, canvas).valid


def test_camera_frame_validation_accepts_integer_rounding() -> None:
    frame = TargetROI(0.1, 0.1, 333 / 1752, 187 / 978)
    assert validate_camera_frame(frame, (1752, 978), (1920, 1080)).valid


@pytest.mark.parametrize("frame", [
    TargetROI(0.02, 0.02, 0.30, 0.30), TargetROI(0.68, 0.02, 0.30, 0.30),
    TargetROI(0.02, 0.68, 0.30, 0.30), TargetROI(0.68, 0.68, 0.30, 0.30),
    TargetROI(0.35, 0.35, 0.30, 0.30),
])
@pytest.mark.parametrize("zoom", [1.2, 1.8, 2.4, 3.0])
def test_camera_frame_center_projects_exactly(frame: TargetROI, zoom: float) -> None:
    # Use a source frame whose pixel aspect exactly matches 16:9.
    frame = TargetROI(frame.x, frame.y, frame.width, frame.height * (1752 / 978) / (16 / 9))
    transform = calculate_camera_transform(frame, (1752, 978), (1920, 1080))
    projected = project_camera_frame_center(frame, transform, (1752, 978), (1920, 1080))
    assert projected[0] == pytest.approx(960.0, abs=1e-6)
    assert projected[1] == pytest.approx(540.0, abs=1e-6)


@pytest.mark.parametrize("handle", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_camera_frame_resize_keeps_canvas_aspect_from_each_corner(handle) -> None:
    from PyQt6.QtCore import QPoint, QRect
    from PyQt6.QtWidgets import QApplication
    from auto_capcut.ui.roi_editor import RoiPreview
    app = QApplication.instance() or QApplication([])
    preview = RoiPreview(16 / 9)
    preview._display_rect = QRect(0, 0, 520, 290)
    preview._selection = QRect(100, 70, 200, 112)
    preview._start_selection = QRect(preview._selection)
    preview._interaction = handle
    rect = preview._resize_rect(QPoint(8, 8))
    assert rect.width() / rect.height() == pytest.approx(16 / 9, rel=0.02)
    assert rect.left() >= 0 and rect.top() >= 0
    assert rect.right() <= preview._display_rect.width()
    assert rect.bottom() <= preview._display_rect.height()
    preview.deleteLater()


@pytest.mark.parametrize("anchor,point", [
    ("top_left", (180, 110)), ("top_right", (-180, 110)),
    ("bottom_left", (180, -110)), ("bottom_right", (-180, -110)),
])
def test_camera_frame_creation_near_all_edges_translates_without_distortion(anchor, point) -> None:
    from PyQt6.QtCore import QPoint, QRect
    from PyQt6.QtWidgets import QApplication
    from auto_capcut.ui.roi_editor import RoiPreview
    app = QApplication.instance() or QApplication([])
    preview = RoiPreview(16 / 9)
    preview._display_rect = QRect(0, 0, 520, 290)
    anchors = {
        "top_left": QPoint(0, 0), "top_right": QPoint(520, 0),
        "bottom_left": QPoint(0, 290), "bottom_right": QPoint(520, 290),
    }
    start = anchors[anchor]
    rect = preview._create_rect(start, start + QPoint(*point))
    assert rect.width() / rect.height() == pytest.approx(16 / 9, rel=0.02)
    assert rect.left() >= 0 and rect.top() >= 0
    assert rect.right() <= preview._display_rect.width()
    assert rect.bottom() <= preview._display_rect.height()
    preview.deleteLater()
