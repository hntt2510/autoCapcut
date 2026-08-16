"""
test_draw_setup_analyzer.py
===========================
Tests for the SRT-driven per-image draw setup analysis (draw_setup.py).

Tests A-L as specified in the production contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from auto_capcut.core.draw_models import (
    CameraAfterDirective,
    DrawImagePlan,
    DrawMode,
    DrawStyle,
    NormalizedRect,
    ObjectEffectOverride,
    SceneDocument,
    SceneImage,
    SceneObject,
)
from auto_capcut.core.draw_setup import (
    ImageSetupStatus,
    ProjectSetupSummary,
    analyze_from_srt,
    analyze_project,
    classify_image,
    required_camera_frame_ids,
    required_object_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan(
    mode: DrawMode = DrawMode.BASIC,
    object_effects: tuple = (),
    camera_after: tuple = (),
) -> DrawImagePlan:
    return DrawImagePlan(
        image_index=1,
        image_name="001.png",
        start_us=0,
        end_us=3_000_000,
        mode=mode,
        style=DrawStyle.V1,
        objects="all",
        actions=(),
        object_effects=object_effects,
        camera_after=camera_after,
    )


def _oe(target: str) -> ObjectEffectOverride:
    return ObjectEffectOverride(target=target, effect="draw")


def _ca(object_id: str, target: str = "", framing: str = "camera_frame") -> CameraAfterDirective:
    return CameraAfterDirective(object_id=object_id, action="focus", target=target, framing=framing)


def _scene(filename: str, objects: list[tuple[str, NormalizedRect | None]]) -> SceneDocument:
    """Build a simple SceneDocument with one image."""
    scene_objs = tuple(
        SceneObject(id=oid, type="art", box=NormalizedRect(0.1, 0.1, 0.2, 0.2), camera_frame=cam)
        for oid, cam in objects
    )
    img = SceneImage(
        filename=filename,
        source_size=(1920, 1080),
        objects=scene_objs,
        draw_order=tuple(oid for oid, _ in objects),
    )
    return SceneDocument(schema_version=1, images={filename: img})


def _box() -> NormalizedRect:
    return NormalizedRect(0.1, 0.1, 0.4, 0.3)


# ---------------------------------------------------------------------------
# Test A: Missing MODE (no plan) → BASIC, Ready
# ---------------------------------------------------------------------------

class TestA_MissingMode:
    def test_none_plan_is_basic_ready(self):
        """A: Missing MODE => BASIC Ready without scene JSON."""
        status = classify_image("001.png", None, None)
        assert status.is_basic
        assert status.is_ready
        assert "Ready" in status.message

    def test_basic_plan_is_basic_ready(self):
        """A: Explicit basic_draw plan → BASIC Ready."""
        status = classify_image("001.png", _plan(DrawMode.BASIC), None)
        assert status.is_basic
        assert status.is_ready


# ---------------------------------------------------------------------------
# Test B: basic_draw never requires scene setup
# ---------------------------------------------------------------------------

class TestB_BasicNeverRequiresScene:
    def test_basic_no_scene_ready(self):
        """B: basic_draw never requires scene setup."""
        status = classify_image("001.png", _plan(DrawMode.BASIC), None)
        assert status.is_ready
        assert not status.missing_ids

    def test_basic_with_empty_scene_still_ready(self):
        """B: basic_draw with empty scene doc → still ready."""
        scene = _scene("001.png", [])
        status = classify_image("001.png", _plan(DrawMode.BASIC), scene)
        assert status.is_ready


# ---------------------------------------------------------------------------
# Test C: advanced_draw with no scene → Setup needed
# ---------------------------------------------------------------------------

class TestC_AdvancedNoScene:
    def test_advanced_no_scene_not_ready(self):
        """C: advanced_draw with no scene => Setup needed."""
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"), _oe("card_left")))
        status = classify_image("002.png", plan, None)
        assert status.is_advanced
        assert not status.is_ready
        assert "Setup needed" in status.message

    def test_advanced_no_scene_lists_missing_ids(self):
        """C: missing IDs should be listed when scene is absent."""
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"), _oe("card_left")))
        status = classify_image("002.png", plan, None)
        assert "title" in status.missing_ids
        assert "card_left" in status.missing_ids


# ---------------------------------------------------------------------------
# Test D: advanced_draw with partial scene → show exact missing IDs
# ---------------------------------------------------------------------------

class TestD_AdvancedPartialScene:
    def test_partial_scene_missing_ids(self):
        """D: advanced_draw with partial scene shows exact missing IDs."""
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"), _oe("card_left"), _oe("card_right")))
        scene = _scene("002.png", [("title", None), ("card_left", None)])
        status = classify_image("002.png", plan, scene)
        assert not status.is_ready
        assert "card_right" in status.missing_ids
        assert "title" not in status.missing_ids
        assert "card_left" not in status.missing_ids

    def test_partial_message_contains_missing_name(self):
        """D: missing object name appears in the status message."""
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("warning_box"),))
        scene = _scene("002.png", [])
        status = classify_image("002.png", plan, scene)
        assert "warning_box" in status.message


# ---------------------------------------------------------------------------
# Test E: advanced_draw with all objects → Ready
# ---------------------------------------------------------------------------

class TestE_AdvancedAllReady:
    def test_all_objects_present_no_camera_req(self):
        """E: advanced_draw with all object IDs present => Ready."""
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"), _oe("card_left")))
        scene = _scene("002.png", [("title", None), ("card_left", None)])
        status = classify_image("002.png", plan, scene)
        assert status.is_ready
        assert "Ready" in status.message

    def test_required_ids_extraction(self):
        """E: required_object_ids captures all referenced IDs."""
        plan = _plan(
            DrawMode.ADVANCED,
            object_effects=(_oe("title"), _oe("card_a")),
            camera_after=(_ca("card_a"),),
        )
        ids = required_object_ids(plan)
        assert "title" in ids
        assert "card_a" in ids
        # no duplicates
        assert ids.count("card_a") == 1


# ---------------------------------------------------------------------------
# Test F: CAMERA_AFTER with camera_frame → require camera_frame rect
# ---------------------------------------------------------------------------

class TestF_CameraFrameRequired:
    def test_camera_frame_missing_not_ready(self):
        """F: required CAMERA_AFTER camera_frame missing => Setup needed."""
        plan = _plan(
            DrawMode.ADVANCED,
            object_effects=(_oe("alex"), _oe("warning")),
            camera_after=(_ca("warning", framing="camera_frame"),),
        )
        scene = _scene("004.png", [
            ("alex", None),
            ("warning", None),  # object exists, camera_frame=None
        ])
        status = classify_image("004.png", plan, scene)
        assert not status.is_ready
        assert "warning" in status.missing_camera_frame_ids
        assert "camera frame missing" in status.message.lower()

    def test_camera_frame_present_is_ready(self):
        """F: all objects present with camera_frame → Ready."""
        plan = _plan(
            DrawMode.ADVANCED,
            object_effects=(_oe("warning"),),
            camera_after=(_ca("warning", framing="camera_frame"),),
        )
        scene = _scene("004.png", [
            ("warning", _box()),  # camera_frame set
        ])
        status = classify_image("004.png", plan, scene)
        assert status.is_ready

    def test_camera_frame_not_required_when_framing_is_object_box(self):
        """F: CAMERA_AFTER with framing=object_box does NOT require camera_frame rect."""
        plan = _plan(
            DrawMode.ADVANCED,
            object_effects=(_oe("title"),),
            camera_after=(_ca("title", framing="object_box"),),
        )
        scene = _scene("001.png", [("title", None)])  # no camera_frame
        status = classify_image("001.png", plan, scene)
        # camera_frame not required for object_box framing
        assert "title" not in status.missing_camera_frame_ids

    def test_required_camera_frame_ids(self):
        """F: required_camera_frame_ids returns correct IDs."""
        plan = _plan(
            DrawMode.ADVANCED,
            camera_after=(
                _ca("alex", framing="camera_frame"),
                _ca("warning", framing="object_box"),
            ),
        )
        cam_ids = required_camera_frame_ids(plan)
        assert "alex" in cam_ids
        assert "warning" not in cam_ids


# ---------------------------------------------------------------------------
# Test G: Configure queue contains only incomplete advanced images
# ---------------------------------------------------------------------------

class TestG_ConfigureQueue:
    def _make_summary(self) -> ProjectSetupSummary:
        images = [
            Path("001.png"),  # basic
            Path("002.png"),  # advanced missing
            Path("003.png"),  # basic
            Path("004.png"),  # advanced ready
            Path("005.png"),  # advanced missing
        ]
        plans = [
            _plan(DrawMode.BASIC),
            _plan(DrawMode.ADVANCED, object_effects=(_oe("title"),)),
            _plan(DrawMode.BASIC),
            _plan(DrawMode.ADVANCED, object_effects=(_oe("alex"),)),
            _plan(DrawMode.ADVANCED, object_effects=(_oe("warning"),)),
        ]
        scene_doc = SceneDocument(
            schema_version=1,
            images={
                "004.png": SceneImage(
                    "004.png", (1920, 1080),
                    (SceneObject("alex", "art", NormalizedRect(0.1, 0.1, 0.1, 0.1)),),
                    ("alex",),
                ),
            }
        )
        return analyze_project(images, plans, scene_doc)

    def test_queue_contains_only_incomplete_advanced(self):
        """G: Configure queue contains only incomplete advanced images."""
        summary = self._make_summary()
        queue = summary.incomplete_advanced
        names = [s.image_name for s in queue]
        assert "002.png" in names
        assert "005.png" in names
        assert "004.png" not in names  # ready

    def test_queue_excludes_basic(self):
        """H: Basic images never appear in configure queue."""
        summary = self._make_summary()
        queue = summary.incomplete_advanced
        names = [s.image_name for s in queue]
        assert "001.png" not in names
        assert "003.png" not in names

    def test_all_advanced_accessible(self):
        """G: all_advanced includes ready + not-ready advanced images."""
        summary = self._make_summary()
        all_adv = summary.all_advanced
        names = [s.image_name for s in all_adv]
        assert "002.png" in names
        assert "004.png" in names
        assert "005.png" in names

    def test_summary_counts(self):
        """G: Summary counts are correct."""
        summary = self._make_summary()
        assert summary.total == 5
        assert summary.basic_count == 2
        assert summary.advanced_count == 3
        assert summary.advanced_ready == 1
        assert summary.advanced_needs_setup == 2
        assert not summary.all_ready


# ---------------------------------------------------------------------------
# Test H: Basic images never appear in queue (already in G but explicit)
# ---------------------------------------------------------------------------

class TestH_BasicNotInQueue:
    def test_all_basic_summary_has_empty_queue(self):
        """H: All-basic project → empty configure queue."""
        images = [Path(f"{i:03d}.png") for i in range(1, 4)]
        plans: list[DrawImagePlan | None] = [_plan(DrawMode.BASIC)] * 3
        summary = analyze_project(images, plans, None)
        assert len(summary.incomplete_advanced) == 0
        assert summary.all_ready


# ---------------------------------------------------------------------------
# Test J: All advanced scenes ready → no preflight block
# ---------------------------------------------------------------------------

class TestJ_AllReadyNoBlock:
    def test_all_ready_summary(self):
        """J: All advanced scenes configured → summary.all_ready is True."""
        images = [Path("002.png")]
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"),))
        scene = _scene("002.png", [("title", None)])
        summary = analyze_project(images, [plan], scene)
        assert summary.all_ready


# ---------------------------------------------------------------------------
# Test: analyze_from_srt end-to-end with acceptance SRT
# ---------------------------------------------------------------------------

class TestAcceptanceCase:
    SRT_CONTENT = """\
1
00:00:00,000 --> 00:00:04,000
DRAW 0s-2s:

2
00:00:04,000 --> 00:00:10,000
MODE advanced_draw
OBJECT_EFFECT target=title effect=draw
OBJECT_EFFECT target=card_left effect=slide_in direction=left
OBJECT_EFFECT target=card_right effect=slide_in direction=right

3
00:00:10,000 --> 00:00:15,000
MODE basic_draw

4
00:00:15,000 --> 00:00:22,000
MODE advanced_draw
OBJECT_EFFECT target=alex effect=draw
OBJECT_EFFECT target=warning effect=push_in direction=top
CAMERA_AFTER object=warning action=focus target=warning framing=camera_frame persist=false

5
00:00:22,000 --> 00:00:27,000
POST_MOTION subtle_zoom_in
"""

    def _write_srt(self, tmp_path: Path) -> Path:
        p = tmp_path / "effect.srt"
        p.write_text(self.SRT_CONTENT, encoding="utf-8")
        return p

    def _make_images(self, tmp_path: Path) -> list[Path]:
        from PIL import Image
        imgs = []
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(1, 6):
            p = img_dir / f"{i:03d}.png"
            Image.new("RGB", (100, 100), color="white").save(p)
            imgs.append(p)
        return imgs

    def test_classification(self, tmp_path: Path):
        """Acceptance: correct BASIC / ADVANCED classification for 5-image SRT."""
        srt = self._write_srt(tmp_path)
        imgs = self._make_images(tmp_path)
        summary = analyze_from_srt(imgs, srt, None)
        modes = [s.mode for s in summary.statuses]
        assert modes == [DrawMode.BASIC, DrawMode.ADVANCED, DrawMode.BASIC, DrawMode.ADVANCED, DrawMode.BASIC]

    def test_required_ids_for_cue2(self, tmp_path: Path):
        """Acceptance: cue 2 requires title, card_left, card_right."""
        srt = self._write_srt(tmp_path)
        imgs = self._make_images(tmp_path)
        summary = analyze_from_srt(imgs, srt, None)
        s2 = summary.statuses[1]  # image 002 (0-indexed)
        assert set(s2.required_ids) == {"title", "card_left", "card_right"}

    def test_required_ids_for_cue4_with_camera(self, tmp_path: Path):
        """Acceptance: cue 4 requires alex, warning; warning needs camera_frame."""
        srt = self._write_srt(tmp_path)
        imgs = self._make_images(tmp_path)
        summary = analyze_from_srt(imgs, srt, None)
        s4 = summary.statuses[3]  # image 004 (0-indexed)
        assert "alex" in s4.required_ids
        assert "warning" in s4.required_ids
        assert "warning" in s4.required_camera_frame_ids

    def test_partial_scene_002_missing_card_right(self, tmp_path: Path):
        """Acceptance: scene with title+card_left but missing card_right → setup needed."""
        srt = self._write_srt(tmp_path)
        imgs = self._make_images(tmp_path)
        scene = SceneDocument(
            schema_version=1,
            images={
                "002.png": SceneImage("002.png", (1920, 1080), (
                    SceneObject("title", "art", NormalizedRect(0.1, 0.1, 0.2, 0.2)),
                    SceneObject("card_left", "art", NormalizedRect(0.3, 0.3, 0.2, 0.2)),
                ), ("title", "card_left")),
                "004.png": SceneImage("004.png", (1920, 1080), (
                    SceneObject("alex", "art", NormalizedRect(0.1, 0.1, 0.2, 0.2)),
                    SceneObject("warning", "art", NormalizedRect(0.4, 0.4, 0.2, 0.2), camera_frame=NormalizedRect(0.3, 0.3, 0.4, 0.4)),
                ), ("alex", "warning")),
            }
        )
        summary = analyze_from_srt(imgs, srt, scene)
        s2 = summary.statuses[1]
        assert not s2.is_ready
        assert "card_right" in s2.missing_ids
        # 004 should be ready: alex present, warning present with camera_frame
        s4 = summary.statuses[3]
        assert s4.is_ready

    def test_configure_queue_for_acceptance_case(self, tmp_path: Path):
        """Acceptance: incomplete_advanced queue = [002, 004] when both need setup."""
        srt = self._write_srt(tmp_path)
        imgs = self._make_images(tmp_path)
        summary = analyze_from_srt(imgs, srt, None)
        queue_names = [s.image_name for s in summary.incomplete_advanced]
        assert queue_names == ["002.png", "004.png"]


# ---------------------------------------------------------------------------
# Test I: After fixing image, queue advances to next incomplete
# ---------------------------------------------------------------------------

class TestI_QueueAdvance:
    def test_after_saving_one_queue_shrinks(self, tmp_path: Path):
        """I: After saving 002, reanalyzing should show only 004 in queue."""
        from PIL import Image as PILImage
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(1, 4):
            PILImage.new("RGB", (100, 100)).save(img_dir / f"{i:03d}.png")

        srt_path = tmp_path / "effect.srt"
        srt_path.write_text("""
1
00:00:00,000 --> 00:00:04,000
DRAW 0s-2s:

2
00:00:04,000 --> 00:00:08,000
MODE advanced_draw
OBJECT_EFFECT target=title effect=draw

3
00:00:08,000 --> 00:00:12,000
MODE advanced_draw
OBJECT_EFFECT target=card effect=draw
""".strip(), encoding="utf-8")

        imgs = [img_dir / f"{i:03d}.png" for i in range(1, 4)]

        # Before fix: both 002 and 003 need setup
        summary1 = analyze_from_srt(imgs, srt_path, None)
        assert len(summary1.incomplete_advanced) == 2

        # Simulate saving 002 (add 'title' object to scene)
        scene = SceneDocument(
            schema_version=1,
            images={
                "002.png": SceneImage("002.png", (100, 100), (
                    SceneObject("title", "art", NormalizedRect(0.1, 0.1, 0.2, 0.2)),
                ), ("title",)),
            }
        )
        # After fix: only 003 needs setup
        summary2 = analyze_project(imgs, [None, _plan(DrawMode.ADVANCED, object_effects=(_oe("title"),)), _plan(DrawMode.ADVANCED, object_effects=(_oe("card"),))], scene)
        incomplete_names = [s.image_name for s in summary2.incomplete_advanced]
        assert "002.png" not in incomplete_names
        assert "003.png" in incomplete_names


# ---------------------------------------------------------------------------
# Test K: Missing advanced setup → summary.all_ready is False → preflight blocks
# ---------------------------------------------------------------------------

class TestK_PreflightBlock:
    def test_incomplete_advanced_prevents_all_ready(self):
        """K: Missing advanced setup → summary.all_ready is False."""
        images = [Path("002.png")]
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"),))
        summary = analyze_project(images, [plan], None)
        assert not summary.all_ready
        assert len(summary.incomplete_advanced) == 1

    def test_complete_advanced_all_ready(self):
        """K: Configured advanced scene → summary.all_ready is True."""
        images = [Path("002.png")]
        plan = _plan(DrawMode.ADVANCED, object_effects=(_oe("title"),))
        scene = _scene("002.png", [("title", None)])
        summary = analyze_project(images, [plan], scene)
        assert summary.all_ready

    def test_incomplete_advanced_details_in_summary(self):
        """K: incomplete_advanced shows correct missing IDs for blocking dialog."""
        images = [Path("002.png"), Path("004.png")]
        plans = [
            _plan(DrawMode.ADVANCED, object_effects=(_oe("title"), _oe("card"))),
            _plan(DrawMode.ADVANCED, object_effects=(_oe("alex"),), camera_after=(_ca("alex"),)),
        ]
        scene = _scene("004.png", [
            ("alex", _box()),  # has camera_frame
        ])
        summary = analyze_project(images, plans, scene)
        assert not summary.all_ready
        incomplete = {s.image_name: s for s in summary.incomplete_advanced}
        assert "002.png" in incomplete
        assert "004.png" not in incomplete  # 004 is ready (alex present with camera_frame)
        s2 = incomplete["002.png"]
        assert "title" in s2.missing_ids
        assert "card" in s2.missing_ids


# ---------------------------------------------------------------------------
# Test L: Asymmetric Main Effect SRT timing E2E
# ---------------------------------------------------------------------------

class TestL_AsymmetricSRTTiming:
    """L: Existing asymmetric Main Effect SRT timing E2E still passes."""

    ASYMMETRIC_SRT = """\
1
00:00:00,000 --> 00:00:03,000
DRAW 0s-2s:

2
00:00:03,000 --> 00:00:11,000
DRAW 0s-5s:

3
00:00:11,000 --> 00:00:15,000
DRAW 0s-2s:
"""

    def test_asymmetric_srt_classifies_all_basic(self, tmp_path: Path):
        """L: Asymmetric basic SRT → all images BASIC."""
        from PIL import Image as PILImage
        srt = tmp_path / "effect.srt"
        srt.write_text(self.ASYMMETRIC_SRT, encoding="utf-8")
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        imgs = []
        for i in range(1, 4):
            p = img_dir / f"{i:03d}.png"
            PILImage.new("RGB", (100, 100)).save(p)
            imgs.append(p)
        summary = analyze_from_srt(imgs, srt, None)
        assert all(s.is_basic for s in summary.statuses)
        assert summary.all_ready
        assert summary.advanced_count == 0
