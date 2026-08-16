"""
draw_setup.py
=============
SRT-driven per-image draw setup analysis.

Classifies each production image as BASIC or ADVANCED purely from the parsed
Main Effect SRT (DrawImagePlan), then validates whether the required objects
and camera frames exist in the SceneDocument.

No image-content analysis, no AI, no OCR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from auto_capcut.core.draw_models import (
    CameraAfterDirective,
    CameraFramingMode,
    DrawImagePlan,
    DrawMode,
    ObjectEffectOverride,
    SceneDocument,
    SceneImage,
    SceneObject,
)


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageSetupStatus:
    """Per-image draw setup status derived from SRT + scene document."""
    image_name: str
    mode: DrawMode                      # BASIC or ADVANCED
    is_ready: bool
    required_ids: tuple[str, ...]       # all object IDs required by SRT
    required_camera_frame_ids: tuple[str, ...]  # IDs that also need camera_frame
    missing_ids: tuple[str, ...]        # required IDs absent from scene
    missing_camera_frame_ids: tuple[str, ...]   # IDs present but camera_frame=None
    message: str                        # human-readable summary line

    @property
    def is_basic(self) -> bool:
        return self.mode is DrawMode.BASIC

    @property
    def is_advanced(self) -> bool:
        return self.mode is DrawMode.ADVANCED

    @property
    def needs_setup(self) -> bool:
        return not self.is_ready


@dataclass(frozen=True)
class ProjectSetupSummary:
    """Summary of all images in the current production project."""
    statuses: tuple[ImageSetupStatus, ...]
    total: int
    basic_count: int
    advanced_count: int
    advanced_ready: int
    advanced_needs_setup: int

    @property
    def all_ready(self) -> bool:
        return self.advanced_needs_setup == 0

    @property
    def incomplete_advanced(self) -> tuple[ImageSetupStatus, ...]:
        """Only advanced images that are not ready, in order."""
        return tuple(s for s in self.statuses if s.is_advanced and not s.is_ready)

    @property
    def all_advanced(self) -> tuple[ImageSetupStatus, ...]:
        return tuple(s for s in self.statuses if s.is_advanced)


# ---------------------------------------------------------------------------
# Required ID extraction
# ---------------------------------------------------------------------------

def required_object_ids(draw_plan: DrawImagePlan) -> list[str]:
    """Return all object IDs explicitly referenced in the draw plan.

    Sources:
    - OBJECT_EFFECT target=<id>
    - CAMERA_AFTER object=<id>
    - CAMERA_AFTER target=<id>  (when target is an object, not a position)

    IDs are deduplicated and returned in the order they first appear.
    If no explicit IDs are referenced but mode is advanced_draw, returns [].
    """
    seen: dict[str, None] = {}  # ordered set
    for oe in draw_plan.object_effects:
        t = oe.target.strip()
        if t:
            seen[t] = None
    for ca in draw_plan.camera_after:
        oid = ca.object_id.strip()
        if oid:
            seen[oid] = None
        # target is sometimes an object id (when framing=camera_frame / object_frame)
        t = ca.target.strip()
        if t and t not in seen:
            # only treat target as object ID if it's not a positional keyword
            _positional = {"center", "top", "bottom", "left", "right", "top_left",
                           "top_right", "bottom_left", "bottom_right", "full"}
            if t.lower() not in _positional:
                seen[t] = None
    return list(seen)


def required_camera_frame_ids(draw_plan: DrawImagePlan) -> list[str]:
    """Return object IDs that require a camera_frame rect.

    An ID requires camera_frame when a CAMERA_AFTER directive references it
    with framing=camera_frame (the default).
    """
    ids: list[str] = []
    seen: set[str] = set()
    for ca in draw_plan.camera_after:
        framing = ca.framing.strip().lower()
        if framing == CameraFramingMode.CAMERA_FRAME.value:
            oid = ca.object_id.strip()
            if oid and oid not in seen:
                ids.append(oid)
                seen.add(oid)
    return ids


# ---------------------------------------------------------------------------
# Scene lookup
# ---------------------------------------------------------------------------

def _lookup_scene_image(scene_doc: SceneDocument | None, filename: str) -> SceneImage | None:
    if scene_doc is None:
        return None
    key = filename.casefold()
    # exact match
    if filename in scene_doc.images:
        return scene_doc.images[filename]
    # case-insensitive fallback
    for k, v in scene_doc.images.items():
        if k.casefold() == key:
            return v
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_image(
    image_name: str,
    draw_plan: DrawImagePlan | None,
    scene_doc: SceneDocument | None,
) -> ImageSetupStatus:
    """Compute the setup status for a single production image.

    Parameters
    ----------
    image_name:
        Filename of the image (e.g. ``"001.png"``).
    draw_plan:
        The parsed ``DrawImagePlan`` from the Main Effect SRT for this image.
        ``None`` means the cue is treated as basic_draw.
    scene_doc:
        The loaded ``SceneDocument`` (may be ``None`` if no JSON is configured).
    """
    # --- Determine mode ---
    if draw_plan is None or draw_plan.mode is DrawMode.BASIC:
        return ImageSetupStatus(
            image_name=image_name,
            mode=DrawMode.BASIC,
            is_ready=True,
            required_ids=(),
            required_camera_frame_ids=(),
            missing_ids=(),
            missing_camera_frame_ids=(),
            message="Ready ✓",
        )

    # --- Advanced ---
    req_ids = required_object_ids(draw_plan)
    req_cam_ids = required_camera_frame_ids(draw_plan)
    scene_img = _lookup_scene_image(scene_doc, image_name)

    if scene_img is None:
        # No scene record at all
        if not req_ids:
            # advanced but no explicit refs → need at least one usable object
            return ImageSetupStatus(
                image_name=image_name,
                mode=DrawMode.ADVANCED,
                is_ready=False,
                required_ids=(),
                required_camera_frame_ids=tuple(req_cam_ids),
                missing_ids=(),
                missing_camera_frame_ids=tuple(req_cam_ids),
                message="Setup needed — scene missing",
            )
        return ImageSetupStatus(
            image_name=image_name,
            mode=DrawMode.ADVANCED,
            is_ready=False,
            required_ids=tuple(req_ids),
            required_camera_frame_ids=tuple(req_cam_ids),
            missing_ids=tuple(req_ids),
            missing_camera_frame_ids=tuple(req_cam_ids),
            message=f"Setup needed — missing {len(req_ids)} required object(s)",
        )

    obj_map = scene_img.object_map  # {id: SceneObject}

    # --- Check missing IDs ---
    if not req_ids:
        # advanced but no explicit targets → just need at least one object
        if not scene_img.objects:
            return ImageSetupStatus(
                image_name=image_name,
                mode=DrawMode.ADVANCED,
                is_ready=False,
                required_ids=(),
                required_camera_frame_ids=tuple(req_cam_ids),
                missing_ids=(),
                missing_camera_frame_ids=tuple(req_cam_ids),
                message="Setup needed — at least one object required",
            )
        # Has objects, ready
        return ImageSetupStatus(
            image_name=image_name,
            mode=DrawMode.ADVANCED,
            is_ready=True,
            required_ids=(),
            required_camera_frame_ids=tuple(req_cam_ids),
            missing_ids=(),
            missing_camera_frame_ids=(),
            message=f"Ready ✓  ({len(scene_img.objects)} object(s))",
        )

    missing_ids = [oid for oid in req_ids if oid not in obj_map]

    # --- Check camera frames ---
    # Only check for camera_frame IDs that actually exist in the scene
    missing_cam: list[str] = []
    for oid in req_cam_ids:
        if oid in obj_map:
            obj = obj_map[oid]
            if obj.camera_frame is None:
                missing_cam.append(oid)
        elif oid not in missing_ids:
            # referenced in camera_after but not in object_effects → still missing
            missing_cam.append(oid)

    # --- Compose message ---
    parts: list[str] = []
    n_req = len(req_ids)
    n_present = n_req - len(missing_ids)

    if missing_ids:
        parts.append(f"Missing: {', '.join(missing_ids)}")
    if missing_cam:
        parts.append(f"Camera frame missing: {', '.join(missing_cam)}")

    if not missing_ids and not missing_cam:
        if req_cam_ids:
            cam_info = f" + {len(req_cam_ids)} camera frame(s)"
        else:
            cam_info = ""
        message = f"Ready ✓  ({n_req}/{n_req} objects{cam_info})"
        is_ready = True
    else:
        detail = "; ".join(parts)
        message = f"Setup needed — {detail}"
        is_ready = False

    if not is_ready and not missing_ids:
        # Objects present, only camera frames missing
        message = f"Setup needed — {', '.join(missing_cam)} camera frame missing"

    return ImageSetupStatus(
        image_name=image_name,
        mode=DrawMode.ADVANCED,
        is_ready=is_ready,
        required_ids=tuple(req_ids),
        required_camera_frame_ids=tuple(req_cam_ids),
        missing_ids=tuple(missing_ids),
        missing_camera_frame_ids=tuple(missing_cam),
        message=message,
    )


# ---------------------------------------------------------------------------
# Project-level analysis
# ---------------------------------------------------------------------------

def analyze_project(
    images: Sequence[Path],
    draw_plans: Sequence[DrawImagePlan | None],
    scene_doc: SceneDocument | None,
) -> ProjectSetupSummary:
    """Analyze all images in a production project.

    Parameters
    ----------
    images:
        Ordered list of production image paths.
    draw_plans:
        Corresponding ``DrawImagePlan`` for each image (None = basic_draw).
        Must have same length as *images*.
    scene_doc:
        Loaded ``SceneDocument`` or ``None``.
    """
    assert len(images) == len(draw_plans), "images and draw_plans must align"
    statuses: list[ImageSetupStatus] = []
    for img, plan in zip(images, draw_plans):
        statuses.append(classify_image(img.name, plan, scene_doc))

    basic = sum(1 for s in statuses if s.is_basic)
    advanced = sum(1 for s in statuses if s.is_advanced)
    adv_ready = sum(1 for s in statuses if s.is_advanced and s.is_ready)
    adv_needs = sum(1 for s in statuses if s.is_advanced and not s.is_ready)

    return ProjectSetupSummary(
        statuses=tuple(statuses),
        total=len(statuses),
        basic_count=basic,
        advanced_count=advanced,
        advanced_ready=adv_ready,
        advanced_needs_setup=adv_needs,
    )


def analyze_from_srt(
    images: Sequence[Path],
    srt_path: Path | str,
    scene_doc: SceneDocument | None,
) -> ProjectSetupSummary:
    """Convenience: parse SRT and analyze.

    Raises ``ValidationError`` on SRT parse failure.
    """
    from auto_capcut.core.unified_effect_parser import parse_unified_effect
    unified = parse_unified_effect(srt_path)

    # Map cue index (1-based) → draw_plan
    plan_by_cue_index: dict[int, DrawImagePlan] = {}
    for cue in unified.cues:
        if cue.kind == "draw" and cue.draw_plan is not None:
            plan_by_cue_index[cue.index] = cue.draw_plan

    # Align to images list (cue index == image position 1-based)
    plans: list[DrawImagePlan | None] = []
    for i, _img in enumerate(images, 1):
        plans.append(plan_by_cue_index.get(i))

    return analyze_project(images, plans, scene_doc)
