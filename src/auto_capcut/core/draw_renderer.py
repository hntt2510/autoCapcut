from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from auto_capcut.core.draw_models import (
    DrawActionType,
    DrawImagePlan,
    DrawMode,
    DrawObjectDirection,
    DrawObjectEffect,
    ObjectEffectDurationMode,
    DrawProjectConfig,
    DrawStyle,
    FinalRevealMode,
    NormalizedRect,
    ObjectEffectOverride,
    SceneImage,
    TextMode,
)
from auto_capcut.core.errors import DrawRenderError, SceneValidationError
from auto_capcut.core.draw_scene import load_scene, sha256_file, validate_scene_document

LOGGER = logging.getLogger(__name__)

ALGORITHM_VERSION = "draw-v2-object-schedule"
OBJECT_EFFECT_PREPROCESS_VERSION = "object-entrance-v2-transparent"
COLOR_REVEAL_MAX_US = 600_000


@dataclass(frozen=True)
class Stroke:
    points: tuple[tuple[float, float], ...]
    object_id: str = ""

    @property
    def length(self) -> float:
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.points, self.points[1:]))

    @property
    def center(self) -> tuple[float, float]:
        if not self.points:
            return 0.5, 0.5
        return tuple(sum(point[index] for point in self.points) / len(self.points) for index in (0, 1))

    def pixel_length(self, size: tuple[int, int]) -> float:
        width, height = size
        return sum(math.hypot((b[0] - a[0]) * width, (b[1] - a[1]) * height) for a, b in zip(self.points, self.points[1:]))


@dataclass(frozen=True)
class ProcessedImage:
    source_hash: str
    folder: Path
    cleaned_path: Path
    line_path: Path
    text_mask_path: Path
    foreground_path: Path | None
    strokes: tuple[Stroke, ...]


@dataclass(frozen=True)
class ObjectEffectConfig:
    object_id: str
    requested_effect: str
    effective_effect: str
    requested_direction: str
    effective_direction: str
    duration_us: int | None
    pause_after_us: int | None
    warning: str = ""


@dataclass(frozen=True)
class ObjectLayerArtifact:
    object_id: str
    box: NormalizedRect
    crop_path: Path
    background_patch_path: Path
    confidence: float
    safe: bool
    reason: str = ""
    alpha_coverage: float = 0.0
    border_background_ratio: float = 0.0

    @property
    def rgba_path(self) -> Path:
        """Explicit name for the transparent layer (crop_path is legacy)."""
        return self.crop_path


@dataclass(frozen=True)
class ObjectEffectFallback:
    object_id: str
    requested_effect: str
    requested_direction: str
    effective_effect: str
    reason: str
    effective_direction: str = DrawObjectDirection.AUTO.value


@dataclass(frozen=True)
class ScheduledGroup:
    """One non-interleaved object stroke group in an advanced draw."""

    object_id: str
    object_type: str
    strokes: tuple[Stroke, ...]
    duration_us: int
    path_length: float
    effect: str = DrawObjectEffect.DRAW.value
    direction: str = DrawObjectDirection.AUTO.value
    object_box: NormalizedRect | None = None
    pause_after_us: int | None = None
    layer: ObjectLayerArtifact | None = None
    requested_effect: str = DrawObjectEffect.DRAW.value
    requested_direction: str = DrawObjectDirection.AUTO.value
    fallback_reason: str = ""

    @property
    def render_effect(self) -> str:
        return self.effect

    @property
    def first_point(self) -> tuple[float, float] | None:
        return self.strokes[0].points[0] if self.strokes and self.strokes[0].points else None

    @property
    def last_point(self) -> tuple[float, float] | None:
        return self.strokes[-1].points[-1] if self.strokes and self.strokes[-1].points else None


@dataclass(frozen=True)
class SchedulePhase:
    kind: str
    start_us: int
    end_us: int
    group_index: int | None = None
    object_id: str = ""
    from_point: tuple[float, float] | None = None
    to_point: tuple[float, float] | None = None

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class DrawSchedule:
    groups: tuple[ScheduledGroup, ...]
    phases: tuple[SchedulePhase, ...]
    sketch_start_us: int
    sketch_end_us: int
    color_start_us: int
    color_end_us: int
    resolved_order: tuple[str, ...] = ()
    unmatched_count: int = 0
    unmatched_policy: str = "last"
    fallbacks: tuple[ObjectEffectFallback, ...] = ()

    @property
    def color_duration_us(self) -> int:
        return max(0, self.color_end_us - self.color_start_us)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(item.reason for item in self.fallbacks if item.reason)


def _hash_signature(*parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _edge_mask(image: Image.Image, style: DrawStyle) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    try:
        import cv2

        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        if style is DrawStyle.V2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            first = cv2.Canny(gray, 45, 130, apertureSize=3, L2gradient=True)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            second = cv2.Canny(blurred, 25, 90, apertureSize=3, L2gradient=True)
            mask = cv2.bitwise_or(first, second)
            minimum = max(3, round(min(image.size) * 0.000012))
        else:
            median = float(np.median(gray))
            low = int(max(0, 0.66 * median))
            high = int(min(255, max(low + 1, 1.33 * median)))
            mask = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True)
            minimum = max(6, round(min(image.size) * 0.000025))
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask >= 128
    except ImportError:
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        threshold = 28 if style is DrawStyle.V2 else 45
        return np.asarray(edges, dtype=np.uint8) >= threshold


def _contour_strokes(mask: np.ndarray, style: DrawStyle) -> list[Stroke]:
    try:
        import cv2

        cv2.setNumThreads(1)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        height, width = mask.shape
        minimum = max(3, round(min(width, height) * (0.000006 if style is DrawStyle.V2 else 0.000015)))
        output: list[Stroke] = []
        for contour in contours:
            if len(contour) < minimum:
                continue
            epsilon = (0.5 if style is DrawStyle.V2 else 1.5) * max(width, height) / 1080
            simplified = cv2.approxPolyDP(contour, epsilon, False).reshape(-1, 2)
            if len(simplified) < 2:
                continue
            output.append(Stroke(tuple((float(x) / width, float(y) / height) for x, y in simplified)))
        return output
    except ImportError:
        height, width = mask.shape
        output: list[Stroke] = []
        minimum = 4 if style is DrawStyle.V2 else 7
        for y, row in enumerate(mask):
            starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
            ends = np.flatnonzero(row & ~np.r_[row[1:], False])
            for start, end in zip(starts, ends):
                if end - start + 1 >= minimum:
                    output.append(Stroke(((float(start) / width, float(y) / height), (float(end) / width, float(y) / height))))
        return output


def _text_mask(image: Image.Image) -> Image.Image:
    """A conservative deterministic mask for rows of small dark components."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    try:
        import cv2

        _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
        components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        candidates: list[tuple[int, int, int, int]] = []
        for index in range(1, components):
            x, y, width, height, area = stats[index]
            if 2 <= height <= max(8, gray.shape[0] // 10) and 2 <= width <= gray.shape[1] // 3 and 4 <= area <= width * height * 0.8:
                candidates.append((x, y, width, height))
        mask = np.zeros_like(gray)
        for x, y, width, height in candidates:
            cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
        return Image.fromarray(mask, mode="L")
    except ImportError:
        return Image.new("L", image.size, 0)


def _remove_simple_background(image: Image.Image) -> Image.Image | None:
    try:
        import cv2

        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        source = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        h, w = source.shape[:2]
        border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]), axis=0).astype(np.float32)
        color = np.median(border, axis=0)
        distance = np.linalg.norm(source.astype(np.float32) - color, axis=2)
        if float(np.mean(distance < 28)) < 0.20:
            return None
        mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        mask[distance < 28] = cv2.GC_BGD
        mask[max(1, h // 20): max(2, h - h // 20), max(1, w // 20): max(2, w - w // 20)] = cv2.GC_PR_FGD
        if h > 4 and w > 4:
            mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = cv2.GC_FGD
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(source, mask, None, background, foreground, 5, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if float(np.mean(alpha > 0)) < 0.05 or float(np.mean(alpha > 0)) > 0.95:
            return None
        rgba = np.dstack((source[:, :, ::-1], alpha))
        return Image.fromarray(rgba, mode="RGBA")
    except Exception:
        return None

def _hash_signature(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:20]


def _ease_out(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 1.0 - (1.0 - value) ** 3


def _save_strokes(path: Path, strokes: Iterable[Stroke]) -> None:
    points_list: list[float] = []
    offsets: list[int] = [0]
    objects: list[str] = []
    for stroke in strokes:
        for point in stroke.points:
            points_list.extend(point)
        offsets.append(len(points_list) // 2)
        objects.append(stroke.object_id)
    np.savez_compressed(path, points=np.asarray(points_list, dtype=np.float32).reshape(-1, 2), offsets=np.asarray(offsets, dtype=np.int32), objects=np.asarray(objects, dtype=object))


def _stroke_objects(path: Path) -> list[str]:
    with np.load(path, allow_pickle=True) as archive:
        if "objects" in archive:
            objects = [str(value) for value in archive["objects"]]
        else:
            objects = []
    return objects


def _load_strokes(path: Path, objects: list[str]) -> tuple[Stroke, ...]:
    with np.load(path) as archive:
        points = archive["points"]
        offsets = archive["offsets"]
    return tuple(Stroke(tuple(tuple(float(value) for value in point) for point in points[offsets[i]: offsets[i + 1]]), objects[i] if i < len(objects) else "") for i in range(len(offsets) - 1))


def _estimate_canvas_background(image: Image.Image) -> tuple[int, int, int, int]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = pixels.shape[:2]
    ring = max(1, min(6, min(h, w) // 50))
    border = np.concatenate((
        pixels[:ring, :].reshape(-1, 3),
        pixels[-ring:, :].reshape(-1, 3),
        pixels[:, :ring].reshape(-1, 3),
        pixels[:, -ring:].reshape(-1, 3),
    ), axis=0)
    median = np.median(border, axis=0)
    std = np.std(border, axis=0)
    if float(np.mean(std)) < 40.0:
        r, g, b = [int(np.clip(round(v), 0, 255)) for v in median]
        return (r, g, b, 255)
    return (255, 255, 255, 255)


def _draw_line_image(size: tuple[int, int], strokes: Iterable[Stroke], bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(image)
    width, height = size
    line_width = max(1, round(min(size) / 900))
    for stroke in strokes:
        points = [(round(x * width), round(y * height)) for x, y in stroke.points]
        if len(points) > 1:
            draw.line(points, fill=(24, 24, 24), width=line_width, joint="curve")
    return image


def prepare_image(image_path: Path, cache_root: Path, style: DrawStyle, text_mode: TextMode, remove_background: bool, scene: SceneImage | None = None) -> ProcessedImage:
    source_hash = sha256_file(image_path)
    # Strokes are deliberately cached before scene assignment.  A scene edit
    # must only change scheduling, not trigger a second edge extraction (and
    # a stale scene-assigned cache can never leak into another scene).
    signature = _hash_signature(ALGORITHM_VERSION, style.value, text_mode.value, remove_background)
    folder = cache_root / source_hash / signature
    manifest_path = folder / "manifest.json"
    files = [folder / name for name in ("cleaned.png", "line.png", "text_mask.png", "strokes.npz")]
    if manifest_path.is_file() and all(item.is_file() for item in files):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("algorithm") != ALGORITHM_VERSION:
                raise ValueError("stale draw cache")
            strokes = _load_strokes(folder / "strokes.npz", manifest.get("stroke_objects", []))
            foreground = folder / "foreground.png"
            return ProcessedImage(source_hash, folder, files[0], files[1], files[2], foreground if foreground.is_file() else None, strokes)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    canvas_bg = _estimate_canvas_background(image)
    background = Image.new("RGBA", image.size, canvas_bg)
    background.alpha_composite(image)
    cleaned = background.convert("RGB")
    text = _text_mask(cleaned)
    mask = _edge_mask(cleaned, style)
    strokes = _contour_strokes(mask, style)
    cleaned.save(folder / "cleaned.png", format="PNG")
    _draw_line_image(cleaned.size, strokes, canvas_bg[:3]).save(folder / "line.png", format="PNG")
    text.save(folder / "text_mask.png", format="PNG")
    foreground = _remove_simple_background(cleaned) if remove_background else None
    if foreground:
        foreground.save(folder / "foreground.png", format="PNG")
    _save_strokes(folder / "strokes.npz", strokes)
    (folder / "manifest.json").write_text(json.dumps({"algorithm": ALGORITHM_VERSION, "source_hash": source_hash, "size": cleaned.size, "stroke_objects": _stroke_objects(folder / "strokes.npz")}, indent=2), encoding="utf-8")
    return ProcessedImage(source_hash, folder, folder / "cleaned.png", folder / "line.png", folder / "text_mask.png", folder / "foreground.png" if foreground else None, tuple(strokes))


def _box_pixels(rect: NormalizedRect, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, min(width - 1, round(rect.x * width)))
    top = max(0, min(height - 1, round(rect.y * height)))
    right = max(left + 1, min(width, round((rect.x + rect.w) * width)))
    bottom = max(top + 1, min(height, round((rect.y + rect.h) * height)))
    return left, top, right, bottom


def _reconstruct_background_patch(source: Image.Image, box: tuple[int, int, int, int], ring: int) -> Image.Image:
    left, top, right, bottom = box
    pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    sides: list[tuple[np.ndarray, np.ndarray]] = []
    if top > 0:
        y0 = max(0, top - ring)
        sides.append((np.full((right - left, 1), float(top), dtype=np.float32), np.median(pixels[y0:top, left:right], axis=(0, 1))))
    if bottom < height:
        y1 = min(height, bottom + ring)
        sides.append((np.full((right - left, 1), float(bottom - 1), dtype=np.float32), np.median(pixels[bottom:y1, left:right], axis=(0, 1))))
    if left > 0:
        x0 = max(0, left - ring)
        sides.append((np.full((bottom - top, 1), float(left), dtype=np.float32), np.median(pixels[top:bottom, x0:left], axis=(0, 1))))
    if right < width:
        x1 = min(width, right + ring)
        sides.append((np.full((bottom - top, 1), float(right - 1), dtype=np.float32), np.median(pixels[top:bottom, right:x1], axis=(0, 1))))
    fallback = np.median(pixels[max(0, top - ring):min(height, bottom + ring), max(0, left - ring):min(width, right + ring)].reshape(-1, 3), axis=0) if sides else np.asarray([255, 255, 255], dtype=np.float32)
    yy, xx = np.mgrid[top:bottom, left:right].astype(np.float32)
    result = np.zeros((bottom - top, right - left, 3), dtype=np.float32)
    weights = np.zeros((bottom - top, right - left), dtype=np.float32)
    for position, color in sides:
        boundary = float(position[0, 0])
        if boundary == top:
            distance = np.maximum(1.0, yy - top + 1.0)
        elif boundary == bottom - 1:
            distance = np.maximum(1.0, bottom - yy)
        elif boundary == left:
            distance = np.maximum(1.0, xx - left + 1.0)
        else:
            distance = np.maximum(1.0, right - xx)
        weight = 1.0 / distance
        result += weight[..., None] * color
        weights += weight
    if not sides:
        result[...] = fallback
        weights[...] = 1.0
    patch = np.clip(result / np.maximum(weights[..., None], 1e-6), 0, 255).astype(np.uint8)
    return Image.fromarray(patch, mode="RGB")


def _prepare_object_layer(artifact: ProcessedImage, obj, cache_root: Path, other_boxes: tuple[NormalizedRect, ...] = ()) -> ObjectLayerArtifact:
    with Image.open(artifact.cleaned_path) as opened:
        source = opened.convert("RGB")
    size = source.size
    box = _box_pixels(obj.box, size)
    canvas_bg = _estimate_canvas_background(source)
    signature = _hash_signature(OBJECT_EFFECT_PREPROCESS_VERSION, artifact.source_hash, obj.box, size, other_boxes)
    folder = artifact.folder / "objects" / OBJECT_EFFECT_PREPROCESS_VERSION / signature
    manifest_path = folder / "manifest.json"
    rgba_path = folder / "object_rgba.png"
    patch_path = folder / "background_patch.png"
    if manifest_path.is_file() and rgba_path.is_file() and patch_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != OBJECT_EFFECT_PREPROCESS_VERSION:
                raise ValueError("stale object layer")
            if manifest.get("source_hash") not in {None, artifact.source_hash} or tuple(manifest.get("source_size", size)) != tuple(size):
                raise ValueError("stale object geometry")
            expected_size = (box[2] - box[0], box[3] - box[1])
            with Image.open(rgba_path) as cached_rgba:
                cached_rgba.verify()
            with Image.open(rgba_path) as cached_rgba:
                if cached_rgba.size != expected_size or cached_rgba.mode not in {"RGBA", "LA"}:
                    raise ValueError("invalid cached object alpha layer")
                cached_alpha = np.asarray(cached_rgba.convert("RGBA"))[:, :, 3]
                cached_coverage = float(np.mean(cached_alpha > 0))
                cached_border = np.concatenate((cached_alpha[0], cached_alpha[-1], cached_alpha[:, 0], cached_alpha[:, -1]))
                cached_border_ratio = float(np.mean(cached_border == 0)) if len(cached_border) else 0.0
                if not np.isfinite(cached_coverage) or not np.isfinite(cached_border_ratio):
                    raise ValueError("invalid cached object alpha mask")
                if "alpha_coverage" in manifest and abs(cached_coverage - float(manifest["alpha_coverage"])) > 0.02:
                    raise ValueError("cached object alpha metrics do not match manifest")
                if "border_background_ratio" in manifest and abs(cached_border_ratio - float(manifest["border_background_ratio"])) > 0.02:
                    raise ValueError("cached object border metrics do not match manifest")
                if manifest.get("safe") and (cached_coverage < 0.005 or cached_coverage > 0.90 or cached_border_ratio < 0.40):
                    raise ValueError("cached object alpha mask no longer passes safety checks")
            with Image.open(patch_path) as cached_patch:
                cached_patch.verify()
            return ObjectLayerArtifact(obj.id, obj.box, rgba_path, patch_path, float(manifest.get("confidence", 0.0)), bool(manifest.get("safe", False)), str(manifest.get("reason", "")), float(manifest.get("alpha_coverage", 0.0)), float(manifest.get("border_background_ratio", 0.0)))
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    left, top, right, bottom = box
    ring = max(2, min(16, round(min(right - left, bottom - top) * 0.05)))
    expanded = (max(0, left - ring), max(0, top - ring), min(size[0], right + ring), min(size[1], bottom + ring))
    pixels = np.asarray(source, dtype=np.int16)
    sample = pixels[expanded[1]:expanded[3], expanded[0]:expanded[2]]
    ring_mask = np.ones(sample.shape[:2], dtype=bool)
    ring_mask[top - expanded[1]:bottom - expanded[1], left - expanded[0]:right - expanded[0]] = False
    ring_pixels = sample[ring_mask]
    confidence = 0.0
    median = np.asarray(canvas_bg[:3], dtype=np.float32)
    if len(ring_pixels):
        sample_median = np.median(ring_pixels, axis=0)
        distances = np.linalg.norm(ring_pixels.astype(np.float32) - sample_median, axis=1)
        confidence = float(np.mean(distances <= 24.0))
        if confidence >= 0.70:
            median = sample_median
    else:
        confidence = 0.85

    crop = pixels[top:bottom, left:right].astype(np.float32)
    crop_distance = np.linalg.norm(crop - median, axis=2)

    # Exclude other scene objects from this object's foreground
    exclusion_mask = np.zeros(crop.shape[:2], dtype=bool)
    for other_b in other_boxes:
        o_left, o_top, o_right, o_bottom = _box_pixels(other_b, size)
        int_l = max(left, o_left) - left
        int_r = min(right, o_right) - left
        int_t = max(top, o_top) - top
        int_b = min(bottom, o_bottom) - top
        if int_l < int_r and int_t < int_b:
            exclusion_mask[int_t:int_b, int_l:int_r] = True

    candidates = (crop_distance <= 20.0) | exclusion_mask
    try:
        import cv2

        labels_count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=4)
        background = np.zeros_like(candidates, dtype=bool)
        if labels_count > 1:
            border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
            background = np.isin(labels, border_labels[border_labels != 0])
        fg_binary = ((~background) & (~exclusion_mask)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 25:
                fg_binary[labels == i] = 0
        distances = cv2.distanceTransform(fg_binary, cv2.DIST_L2, 3).astype(np.float32)
        feather = np.round(np.minimum(distances, 2.0) / 2.0 * 255.0)
        alpha = np.where(distances >= 2.0, 255.0, np.where(distances > 0, feather, 0.0)).astype(np.uint8)
    except ImportError:
        background = np.zeros_like(candidates, dtype=bool)
        queue = deque()
        height, width = candidates.shape
        for x in range(width):
            if candidates[0, x]: queue.append((0, x))
            if candidates[height - 1, x]: queue.append((height - 1, x))
        for y in range(height):
            if candidates[y, 0]: queue.append((y, 0))
            if candidates[y, width - 1]: queue.append((y, width - 1))
        while queue:
            y, x = queue.popleft()
            if background[y, x]:
                continue
            background[y, x] = True
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and candidates[next_y, next_x] and not background[next_y, next_x]:
                    queue.append((next_y, next_x))
        alpha = np.where((~background) & (~exclusion_mask), 255, 0).astype(np.uint8)

    alpha_coverage = float(np.mean(alpha > 0))
    border_alpha = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
    border_background_ratio = float(np.mean(border_alpha == 0)) if len(border_alpha) else 0.0
    safe = confidence >= 0.70 and 0.005 <= alpha_coverage <= 0.90 and border_background_ratio >= 0.40
    reason = ""
    if not safe:
        if confidence < 0.70:
            reason = f"background confidence {confidence:.3f} is below 0.700"
        elif alpha_coverage < 0.005:
            reason = f"foreground alpha coverage {alpha_coverage:.3f} is too small"
        elif alpha_coverage > 0.90:
            reason = f"foreground alpha coverage {alpha_coverage:.3f} is too large"
        else:
            reason = f"crop-border background ratio {border_background_ratio:.3f} is too small"
    rgba = np.dstack((crop.astype(np.uint8), alpha))
    Image.fromarray(rgba, mode="RGBA").save(rgba_path, format="PNG")
    Image.fromarray(rgba, mode="RGBA").save(folder / "crop.png", format="PNG")
    _reconstruct_background_patch(source, box, ring).save(patch_path, format="PNG")
    manifest_path.write_text(json.dumps({"version": OBJECT_EFFECT_PREPROCESS_VERSION, "source_hash": artifact.source_hash, "source_size": list(size), "box": [left, top, right, bottom], "background_rgb": [float(value) for value in median], "confidence": confidence, "alpha_coverage": alpha_coverage, "border_background_ratio": border_background_ratio, "safe": safe, "reason": reason}, indent=2), encoding="utf-8")
    return ObjectLayerArtifact(obj.id, obj.box, rgba_path, patch_path, confidence, safe, reason, alpha_coverage, border_background_ratio)


def _prepare_object_layers(artifact: ProcessedImage, scene: SceneImage, configs: dict[str, ObjectEffectConfig]) -> tuple[dict[str, ObjectLayerArtifact], dict[str, ObjectEffectConfig], tuple[ObjectEffectFallback, ...]]:
    layers: dict[str, ObjectLayerArtifact] = {}
    resolved = dict(configs)
    fallbacks: list[ObjectEffectFallback] = []
    for obj in scene.objects:
        config = resolved[obj.id]
        is_draw = config.effective_effect == DrawObjectEffect.DRAW.value
        other_boxes = tuple(other.box for other in scene.objects if other.id != obj.id)
        try:
            layer = _prepare_object_layer(artifact, obj, artifact.folder, other_boxes)
            layers[obj.id] = layer
        except Exception as exc:
            if not is_draw:
                reason = f"transparent extraction failed: {exc}"
                resolved[obj.id] = replace(config, effective_effect=DrawObjectEffect.DRAW.value, warning=reason)
                fallbacks.append(ObjectEffectFallback(obj.id, config.requested_effect, config.requested_direction, DrawObjectEffect.DRAW.value, reason, config.effective_direction))
            continue
        if not is_draw and not layer.safe:
            reason = layer.reason or "background reconstruction confidence too low"
            resolved[obj.id] = replace(config, effective_effect=DrawObjectEffect.DRAW.value, warning=reason)
            fallbacks.append(ObjectEffectFallback(obj.id, config.requested_effect, config.requested_direction, DrawObjectEffect.DRAW.value, reason, config.effective_direction))
    safe_entrance_layers = tuple((key, value.box) for key, value in sorted(layers.items()) if value.safe and resolved.get(key, ObjectEffectConfig("", "draw", "draw", "auto", "auto", None, None)).effective_effect != DrawObjectEffect.DRAW.value)
    safe_signature = _hash_signature(OBJECT_EFFECT_PREPROCESS_VERSION, safe_entrance_layers)
    base_folder = artifact.folder / "bases"
    base_folder.mkdir(parents=True, exist_ok=True)
    base_path = base_folder / f"{safe_signature}.png"
    if not base_path.is_file():
        with Image.open(artifact.cleaned_path) as opened:
            base = opened.convert("RGB")
        for object_id, layer in layers.items():
            if not layer.safe or resolved.get(object_id, ObjectEffectConfig("", "draw", "draw", "auto", "auto", None, None)).effective_effect == DrawObjectEffect.DRAW.value:
                continue
            left, top, right, bottom = _box_pixels(layer.box, base.size)
            with Image.open(layer.background_patch_path) as patch:
                base.paste(patch.convert("RGB"), (left, top))
        base.save(base_path, format="PNG")
    return layers, resolved, tuple(fallbacks)


def _rect_viewport(rect: NormalizedRect, aspect: float, padding: float = 0.10) -> tuple[float, float, float, float]:
    x = max(0.0, rect.x - rect.w * padding)
    y = max(0.0, rect.y - rect.h * padding)
    width = min(1.0 - x, rect.w * (1 + padding * 2))
    height = min(1.0 - y, rect.h * (1 + padding * 2))
    if width / height > aspect:
        height = min(1.0, width / aspect)
    else:
        width = min(1.0, height * aspect)
    x = min(max(0.0, rect.center_x - width / 2), 1.0 - width)
    y = min(max(0.0, rect.center_y - height / 2), 1.0 - height)
    return x, y, width, height


def _interpolate(a: tuple[float, float, float, float], b: tuple[float, float, float, float], value: float, easing: str) -> tuple[float, float, float, float]:
    if easing == "ease_in_out":
        value = value * value * (3 - 2 * value)
    return tuple(left + (right - left) * value for left, right in zip(a, b))


def _viewport_at(plan: DrawImagePlan, scene: SceneImage | None, time_us: int, aspect: float) -> tuple[float, float, float, float]:
    full = (0.0, 0.0, 1.0, 1.0)
    current = full
    object_map = scene.object_map if scene else {}
    for action in plan.actions:
        if action.type is DrawActionType.DRAW or time_us < action.start_us:
            continue
        if action.type is DrawActionType.SETTLE:
            continue
        if action.type is DrawActionType.FULL_VIEW:
            target = full
        else:
            obj = object_map.get(action.params.get("target", ""))
            if obj is None:
                continue
            rect = obj.camera_frame if action.params.get("framing", "camera_frame").casefold() == "camera_frame" else obj.box
            if rect is None:
                continue
            target = _rect_viewport(rect, aspect)
            if action.type is DrawActionType.PAN_TO:
                target = (target[0], target[1], current[2], current[3])
        if time_us < action.end_us:
            fraction = (time_us - action.start_us) / action.duration_us
            if action.type is DrawActionType.PULL_TO:
                if fraction < 0.4:
                    return _interpolate(current, full, fraction / 0.4, action.params.get("easing", "ease_in_out"))
                return _interpolate(full, target, (fraction - 0.4) / 0.6, action.params.get("easing", "ease_in_out"))
            return _interpolate(current, target, fraction, action.params.get("easing", "ease_in_out"))
        current = target
    return current


def _crop(image: Image.Image, viewport: tuple[float, float, float, float], size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    x, y, w, h = viewport
    crop = image.crop((round(x * width), round(y * height), max(round((x + w) * width), 1), max(round((y + h) * height), 1)))
    return ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _stroke_prefix(stroke: Stroke, fraction: float) -> tuple[tuple[float, float], ...]:
    if fraction >= 1:
        return stroke.points
    if fraction <= 0 or len(stroke.points) < 2:
        return stroke.points[:1]
    target = stroke.length * fraction
    output = [stroke.points[0]]
    travelled = 0.0
    for left, right in zip(stroke.points, stroke.points[1:]):
        distance = math.hypot(right[0] - left[0], right[1] - left[1])
        if travelled + distance >= target:
            ratio = (target - travelled) / max(distance, 1e-9)
            output.append((left[0] + (right[0] - left[0]) * ratio, left[1] + (right[1] - left[1]) * ratio))
            break
        output.append(right)
        travelled += distance
    return tuple(output)


def _group_prefixes(group: ScheduledGroup, progress: float) -> list[tuple[Stroke, tuple[tuple[float, float], ...]]]:
    lengths = [max(stroke.length, 1e-9) for stroke in group.strokes]
    target = max(0.0, min(1.0, progress)) * sum(lengths)
    travelled = 0.0
    output: list[tuple[Stroke, tuple[tuple[float, float], ...]]] = []
    for stroke, length in zip(group.strokes, lengths):
        fraction = max(0.0, min(1.0, (target - travelled) / length))
        output.append((stroke, _stroke_prefix(stroke, fraction)))
        travelled += length
        if travelled >= target and target < sum(lengths):
            break
    return output


def _draw_phase_timing(draw_duration_us: int, final_mode: FinalRevealMode, pause_us: int, group_count: int) -> tuple[int, int]:
    """Return (stroke_duration, color_duration) in microseconds."""
    if final_mode is not FinalRevealMode.LINE_THEN_COLOR:
        return max(1, draw_duration_us), 0
    color_duration = min(COLOR_REVEAL_MAX_US, max(1, draw_duration_us // 2))
    stroke_duration = max(1, draw_duration_us - color_duration - pause_us * max(0, group_count - 1))
    return stroke_duration, color_duration


def _resolved_object_order(plan: DrawImagePlan, scene: SceneImage) -> list[str]:
    requested = [item.strip() for item in plan.draw_action.params.get("order", "").split(",") if item.strip()]
    if requested:
        # Keep the directive's order authoritative, but do not duplicate an ID
        # if a hand-edited directive contains it twice. Objects omitted from a
        # partial directive follow the saved scene order so their strokes are
        # not silently discarded.
        result = list(dict.fromkeys(requested))
        fallback = list(scene.draw_order) if scene.draw_order else [item.id for item in scene.objects]
        return result + [item for item in fallback if item not in result]
    return list(scene.draw_order) if scene.draw_order else [item.id for item in scene.objects]


_EFFECTS = {item.value for item in DrawObjectEffect}
_DIRECTIONS = {item.value for item in DrawObjectDirection}


def _resolve_effect_direction(effect: str, direction: str) -> tuple[str, str, str]:
    requested_effect = str(effect or DrawObjectEffect.DRAW.value).casefold()
    requested_direction = str(direction or DrawObjectDirection.AUTO.value).casefold()
    if requested_effect not in _EFFECTS:
        return DrawObjectEffect.DRAW.value, DrawObjectDirection.AUTO.value, f"unknown render_effect {effect!r}"
    if requested_direction not in _DIRECTIONS:
        return requested_effect, DrawObjectDirection.AUTO.value, f"invalid direction {direction!r}"
    if requested_effect in {DrawObjectEffect.DRAW.value, DrawObjectEffect.POP_IN.value} and requested_direction != DrawObjectDirection.AUTO.value:
        return requested_effect, DrawObjectDirection.AUTO.value, f"{requested_effect} does not use direction"
    compatible = {
        DrawObjectEffect.DROP_IN.value: {DrawObjectDirection.AUTO.value, DrawObjectDirection.TOP.value},
        # V1 has side and top push assets only.  A bottom push cannot be
        # represented by either asset, so it is an explicit draw fallback.
        DrawObjectEffect.PUSH_IN.value: {DrawObjectDirection.AUTO.value, DrawObjectDirection.LEFT.value, DrawObjectDirection.RIGHT.value, DrawObjectDirection.TOP.value},
    }
    if requested_effect in compatible and requested_direction not in compatible[requested_effect]:
        if requested_effect == DrawObjectEffect.PUSH_IN.value and requested_direction == DrawObjectDirection.BOTTOM.value:
            return DrawObjectEffect.DRAW.value, DrawObjectDirection.AUTO.value, "push_in direction 'bottom' is unsupported in V1; falling back to draw"
        return requested_effect, DrawObjectDirection.AUTO.value, f"direction {direction!r} is incompatible with {requested_effect}"
    return requested_effect, requested_direction, ""


def _resolve_object_effects(plan: DrawImagePlan, scene: SceneImage) -> tuple[dict[str, ObjectEffectConfig], tuple[ObjectEffectFallback, ...]]:
    overrides = {item.target: item for item in plan.object_effects}
    configs: dict[str, ObjectEffectConfig] = {}
    fallbacks: list[ObjectEffectFallback] = []
    for override in plan.object_effects:
        if override.target not in scene.object_map:
            fallbacks.append(ObjectEffectFallback(override.target, override.effect, override.direction, DrawObjectEffect.DRAW.value, "OBJECT_EFFECT target is missing from scene"))
    for obj in scene.objects:
        override = overrides.get(obj.id)
        requested_effect = override.effect if override else obj.render_effect
        requested_direction = override.direction if override else obj.direction
        if override and override.duration_mode == ObjectEffectDurationMode.AUTO.value:
            duration_us = None
        elif override and (override.duration_mode == ObjectEffectDurationMode.FIXED.value or override.duration_us is not None):
            duration_us = override.duration_us
        else:
            duration_us = obj.duration_us
        pause_after_us = override.pause_after_us if override and override.pause_after_us is not None else obj.pause_after_us
        effective_effect, effective_direction, warning = _resolve_effect_direction(requested_effect, requested_direction)
        # Auto entry direction is resolved against the object's actual ROI.
        # Keep auto pushes on an available edge; an explicit bottom direction
        # was rejected above and therefore never reaches this branch.
        if effective_effect == DrawObjectEffect.PUSH_IN.value and effective_direction == DrawObjectDirection.AUTO.value:
            distances = {
                DrawObjectDirection.LEFT.value: obj.box.center_x,
                DrawObjectDirection.RIGHT.value: 1.0 - obj.box.center_x,
                DrawObjectDirection.TOP.value: obj.box.center_y,
            }
            effective_direction = min(distances, key=lambda item: (distances[item], ("left", "right", "top").index(item)))
        configs[obj.id] = ObjectEffectConfig(obj.id, str(requested_effect).casefold(), effective_effect, str(requested_direction).casefold(), effective_direction, duration_us, pause_after_us, warning)
        if warning:
            fallbacks.append(ObjectEffectFallback(obj.id, str(requested_effect), str(requested_direction), effective_effect, warning, effective_direction))
    return configs, tuple(fallbacks)


def _rect_contains(rect: NormalizedRect, point: tuple[float, float]) -> bool:
    return rect.x <= point[0] <= rect.x + rect.w and rect.y <= point[1] <= rect.y + rect.h


def _assign_strokes_to_objects(strokes: tuple[Stroke, ...], scene: SceneImage, resolved_order: Iterable[str] | None = None) -> tuple[dict[str, tuple[Stroke, ...]], tuple[Stroke, ...]]:
    """Assign each raw stroke to at most one scene object.

    Centroid matches are preferred.  Otherwise a stroke is assigned only when
    at least half of its sampled points are inside an object box.  Sorting the
    candidate tuple makes overlap/tie resolution independent of dictionary or
    contour enumeration order.
    """
    order = list(resolved_order) if resolved_order is not None else list(scene.draw_order or [item.id for item in scene.objects])
    order_index = {key: index for index, key in enumerate(order)}
    objects = scene.objects
    assigned: dict[str, list[Stroke]] = {obj.id: [] for obj in objects}
    unmatched: list[Stroke] = []
    for stroke in strokes:
        if not stroke.points:
            unmatched.append(stroke)
            continue
        center = stroke.center
        samples = stroke.points
        candidates: list[tuple[float, float, int, object]] = []
        centroid_candidates: list[tuple[float, float, int, object]] = []
        for index, obj in enumerate(objects):
            overlap = sum(_rect_contains(obj.box, point) for point in samples) / len(samples)
            candidate = (overlap, obj.box.w * obj.box.h, order_index.get(obj.id, len(order) + index), obj)
            if _rect_contains(obj.box, center):
                centroid_candidates.append(candidate)
            if overlap >= 0.5:
                candidates.append(candidate)
        choices = centroid_candidates or candidates
        if not choices:
            unmatched.append(stroke)
            continue
        _, _, _, selected = min(choices, key=lambda item: (-item[0], item[1], item[2]))
        assigned[selected.id].append(Stroke(stroke.points, selected.id))
    return {key: tuple(value) for key, value in assigned.items()}, tuple(unmatched)


def assign_strokes_to_objects(strokes: tuple[Stroke, ...], scene: SceneImage, resolved_order: Iterable[str] | None = None) -> tuple[dict[str, tuple[Stroke, ...]], tuple[Stroke, ...]]:
    """Publicly importable wrapper used by diagnostics/tests."""
    return _assign_strokes_to_objects(strokes, scene, resolved_order)


def _sort_group_strokes(strokes: Iterable[Stroke], direction: str) -> tuple[Stroke, ...]:
    items = list(strokes)
    direction = direction.casefold()
    if direction == "left_to_right":
        key = lambda item: (item.center[0], item.center[1], -item.length)
    elif direction == "right_to_left":
        key = lambda item: (-item.center[0], item.center[1], -item.length)
    elif direction == "bottom_to_top":
        key = lambda item: (-item.center[1], item.center[0], -item.length)
    else:  # auto and top_to_bottom
        key = lambda item: (item.center[1], item.center[0], -item.length)
    return tuple(sorted(items, key=key))


def _transition_us(left: ScheduledGroup, right: ScheduledGroup) -> int:
    if left.last_point is None or right.first_point is None:
        return 100_000
    distance = math.hypot(right.first_point[0] - left.last_point[0], right.first_point[1] - left.last_point[1]) / math.sqrt(2.0)
    return round(100_000 + min(1.0, distance) * 200_000)


def _allocate_weighted(total_us: int, weights: list[float]) -> list[int]:
    if not weights:
        return []
    values = [max(0.0, value) for value in weights]
    weight_sum = sum(values)
    if weight_sum <= 0:
        values = [1.0] * len(weights)
        weight_sum = float(len(values))
    output = [math.floor(total_us * value / weight_sum) for value in values]
    output[-1] += total_us - sum(output)
    return output


def _entry_direction(box: NormalizedRect, effect: str, direction: str) -> str:
    if effect == DrawObjectEffect.DROP_IN.value:
        return DrawObjectDirection.TOP.value
    if effect in {DrawObjectEffect.DRAW.value, DrawObjectEffect.POP_IN.value}:
        return DrawObjectDirection.AUTO.value
    if direction != DrawObjectDirection.AUTO.value:
        return direction
    distances = {
        DrawObjectDirection.LEFT.value: box.center_x,
        DrawObjectDirection.RIGHT.value: 1.0 - box.center_x,
        DrawObjectDirection.TOP.value: box.center_y,
        DrawObjectDirection.BOTTOM.value: 1.0 - box.center_y,
    }
    return min(distances, key=lambda item: (distances[item], ("left", "right", "top", "bottom").index(item)))


def _auto_effect_duration(config: ObjectEffectConfig, group: ScheduledGroup) -> int:
    if config.duration_us is not None:
        if config.effective_effect == DrawObjectEffect.PUSH_IN.value:
            return max(400_000, config.duration_us)
        return config.duration_us
    if config.effective_effect == DrawObjectEffect.DRAW.value:
        return 0
    box = config_box = group.object_box
    if box is None:
        return 550_000
    direction = _entry_direction(box, config.effective_effect, config.effective_direction)
    distance = {
        DrawObjectDirection.LEFT.value: box.center_x + box.w,
        DrawObjectDirection.RIGHT.value: 1.0 - box.center_x + box.w,
        DrawObjectDirection.TOP.value: box.center_y + box.h,
        DrawObjectDirection.BOTTOM.value: 1.0 - box.center_y + box.h,
    }.get(direction, 0.5)
    if config.effective_effect == DrawObjectEffect.PUSH_IN.value:
        return round(max(0.70, min(1.00, 0.55 + 0.30 * distance + 0.15)) * 1_000_000)
    if config.effective_effect in {DrawObjectEffect.SLIDE_IN.value, DrawObjectEffect.TOSS_IN.value}:
        return round(max(0.55, min(0.85, 0.45 + 0.30 * distance)) * 1_000_000)
    if config.effective_effect == DrawObjectEffect.DROP_IN.value:
        return 750_000
    return 550_000


def _build_advanced_schedule(strokes: tuple[Stroke, ...], plan: DrawImagePlan, scene: SceneImage, source_size: tuple[int, int] | None = None, text_mode: TextMode = TextMode.KEEP, effect_configs: dict[str, ObjectEffectConfig] | None = None, layers: dict[str, ObjectLayerArtifact] | None = None, fallbacks: tuple[ObjectEffectFallback, ...] = ()) -> DrawSchedule:
    draw = plan.draw_action
    final_mode = FinalRevealMode(draw.params.get("final", FinalRevealMode.LINE_THEN_COLOR.value).casefold())
    color_duration = min(200_000, max(1, draw.duration_us // 4)) if final_mode is FinalRevealMode.LINE_THEN_COLOR else 0
    color_start = draw.end_us - color_duration
    order = _resolved_object_order(plan, scene)
    assigned, unmatched = _assign_strokes_to_objects(strokes, scene, order)
    pixel_size = source_size or scene.source_size
    direction = draw.params.get("direction", "auto")
    if effect_configs is None:
        effect_configs, resolved_fallbacks = _resolve_object_effects(plan, scene)
        fallbacks = tuple(fallbacks) + resolved_fallbacks
    groups: list[ScheduledGroup] = []
    for object_id in order:
        obj = scene.object_map.get(object_id)
        if obj is None:
            continue
        config = effect_configs[obj.id]
        local_direction = _entry_direction(obj.box, config.effective_effect, config.effective_direction)
        ordered = _sort_group_strokes(assigned.get(object_id, ()), direction)
        pixel_length = sum(stroke.pixel_length(pixel_size) for stroke in ordered)
        groups.append(ScheduledGroup(object_id, obj.type, ordered, 0, pixel_length, config.effective_effect, local_direction, obj.box, config.pause_after_us, (layers or {}).get(object_id), config.requested_effect, config.requested_direction, config.warning))
    policy = draw.params.get("unmatched", "last").casefold()
    unmatched_group = ScheduledGroup("__unmatched__", "unmatched", _sort_group_strokes(unmatched, direction), 0, sum(stroke.pixel_length(pixel_size) for stroke in unmatched))
    if unmatched and policy == "first":
        groups.insert(0, unmatched_group)
    elif unmatched and policy == "last":
        groups.append(unmatched_group)
    active = [group for group in groups if group.strokes or group.effect != DrawObjectEffect.DRAW.value]
    budget = max(0, color_start - draw.start_us)
    global_pause = max(0, round(float(draw.params.get("pause_each", "0")) * 1_000_000))
    desired = [_auto_effect_duration(effect_configs.get(group.object_id, ObjectEffectConfig(group.object_id, group.effect, group.effect, group.direction, group.direction, None, None)), group) if group.object_id != "__unmatched__" else 0 for group in active]
    auto_draw = [index for index, group in enumerate(active) if group.effect == DrawObjectEffect.DRAW.value and effect_configs.get(group.object_id, ObjectEffectConfig("", "draw", "draw", "auto", "auto", None, None)).duration_us is None]
    pauses = [effect_configs.get(left.object_id, ObjectEffectConfig("", "draw", "draw", "auto", "auto", None, None)).pause_after_us if effect_configs.get(left.object_id) and effect_configs[left.object_id].pause_after_us is not None else global_pause for left in active[:-1]]
    transitions = [_transition_us(left, right) if left.effect == DrawObjectEffect.DRAW.value and right.effect == DrawObjectEffect.DRAW.value else 0 for left, right in zip(active, active[1:])]
    fixed_total = sum(desired) + sum(pauses) + sum(transitions)
    warnings = list(fallbacks)
    if fixed_total > budget and fixed_total:
        scale = budget / fixed_total
        scaled_values = _allocate_weighted(budget, desired + pauses + transitions)
        desired = scaled_values[:len(desired)]
        pauses = scaled_values[len(desired):len(desired) + len(pauses)]
        transitions = scaled_values[len(desired) + len(pauses):]
        warnings.append(ObjectEffectFallback("__schedule__", "", "", "", f"requested sequence duration {fixed_total / 1_000_000:.3f}s exceeded sketch budget; scaled by {scale:.3f}"))
    used = sum(desired) + sum(pauses) + sum(transitions)
    remaining = max(0, budget - used)
    if auto_draw:
        extras = _allocate_weighted(remaining, [active[index].path_length or len(active[index].strokes) or 1 for index in auto_draw])
        for index, extra in zip(auto_draw, extras):
            desired[index] += extra
        remaining = 0
    durations_by_group = {id(group): duration for group, duration in zip(active, desired)}
    groups = [replace(group, duration_us=durations_by_group.get(id(group), 0)) for group in groups]
    active = [group for group in groups if group.strokes or group.effect != DrawObjectEffect.DRAW.value]
    phases: list[SchedulePhase] = []
    cursor = draw.start_us
    for index, group in enumerate(active):
        phases.append(SchedulePhase("object", cursor, cursor + group.duration_us, index, group.object_id))
        cursor += group.duration_us
        if index < len(active) - 1:
            pause, travel = pauses[index], transitions[index]
            if pause:
                phases.append(SchedulePhase("pause", cursor, cursor + pause, index, group.object_id, group.last_point, group.last_point)); cursor += pause
            next_group = active[index + 1]
            if travel:
                phases.append(SchedulePhase("travel", cursor, cursor + travel, index + 1, next_group.object_id, group.last_point, next_group.first_point)); cursor += travel
    if cursor < color_start:
        if active:
            phases.append(SchedulePhase("hold", cursor, color_start, len(active) - 1, active[-1].object_id))
        else:
            phases.append(SchedulePhase("hold", cursor, color_start, None, ""))
        cursor = color_start
    return DrawSchedule(tuple(groups), tuple(phases), draw.start_us, color_start, color_start, draw.end_us, tuple(order), len(unmatched), policy, tuple(warnings))


def build_advanced_schedule(strokes: tuple[Stroke, ...], plan: DrawImagePlan, scene: SceneImage, source_size: tuple[int, int] | None = None, text_mode: TextMode = TextMode.KEEP) -> DrawSchedule:
    return _build_advanced_schedule(strokes, plan, scene, source_size, text_mode)


def _basic_schedule(strokes: tuple[Stroke, ...], plan: DrawImagePlan) -> DrawSchedule:
    ordered = tuple(sorted(strokes, key=lambda item: (item.center[1], item.center[0], -item.length)))
    group = ScheduledGroup("", "art", ordered, 0, sum(item.length for item in ordered))
    draw = plan.draw_action
    color_duration = min(COLOR_REVEAL_MAX_US, max(1, draw.duration_us // 2)) if FinalRevealMode(draw.params.get("final", FinalRevealMode.LINE_THEN_COLOR.value).casefold()) is FinalRevealMode.LINE_THEN_COLOR else 0
    color_start = draw.end_us - color_duration
    group = ScheduledGroup(group.object_id, group.object_type, group.strokes, max(0, color_start - draw.start_us), group.path_length)
    phase = (SchedulePhase("object", draw.start_us, color_start, 0, ""),) if ordered and color_start > draw.start_us else ()
    return DrawSchedule((group,), phase, draw.start_us, color_start, color_start, draw.end_us)


def _ordered_strokes(strokes: tuple[Stroke, ...], plan: DrawImagePlan, scene: SceneImage | None) -> list[Stroke]:
    """Compatibility helper retaining basic whole-image ordering."""
    if scene and plan.mode is DrawMode.ADVANCED:
        return [stroke for group in _build_advanced_schedule(strokes, plan, scene).groups for stroke in group.strokes]
    return list(_basic_schedule(strokes, plan).groups[0].strokes)


def _ease_out(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 1.0 - (1.0 - value) ** 3


def _effect_state(box: NormalizedRect, effect: str, direction: str, progress: float) -> tuple[float, float, float, float, float, float]:
    """Return x, y, w, h, rotation-degrees, opacity in normalized source space."""
    progress = max(0.0, min(1.0, progress))
    if effect == DrawObjectEffect.POP_IN.value:
        if progress < 0.75:
            scale = 0.80 + (1.06 - 0.80) * (progress / 0.75)
        else:
            scale = 1.06 + (1.0 - 1.06) * ((progress - 0.75) / 0.25)
        return box.center_x - box.w * scale / 2, box.center_y - box.h * scale / 2, box.w * scale, box.h * scale, 0.0, 1.0
    vector = {
        "left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0),
        "top_left": (-1.0, -1.0), "top_right": (1.0, -1.0), "bottom_left": (-1.0, 1.0), "bottom_right": (1.0, 1.0),
    }.get(direction, (0.0, -1.0))
    distance = max(box.w if vector[0] else 0.0, box.h if vector[1] else 0.0) + 0.05
    start_x = box.x + vector[0] * distance
    start_y = box.y + vector[1] * distance
    if effect == DrawObjectEffect.DROP_IN.value:
        if progress < 0.82:
            value = progress / 0.82
            y = (box.y - box.h - 0.05) + (box.y + box.h * 0.04 - (box.y - box.h - 0.05)) * (value * value)
        else:
            value = (progress - 0.82) / 0.18
            y = box.y + box.h * 0.04 + (box.y - (box.y + box.h * 0.04)) * _ease_out(value)
        return box.x, y, box.w, box.h, 0.0, 1.0
    if effect == DrawObjectEffect.PUSH_IN.value:
        if progress < 0.15:
            movement = 0.0
        elif progress < 0.85:
            movement = _ease_out((progress - 0.15) / 0.70)
        else:
            movement = 1.0
    else:
        movement = _ease_out(progress)
    x = start_x + (box.x - start_x) * movement
    y = start_y + (box.y - start_y) * movement
    rotation = 0.0
    if effect == DrawObjectEffect.TOSS_IN:
        rotation = (10.0 if vector[0] < 0 or vector[1] < 0 else -10.0) if vector[0] and vector[1] else (8.0 if vector[0] < 0 or vector[1] < 0 else -8.0)
        rotation *= 1.0 - _ease_out(progress)
    return x, y, box.w, box.h, rotation, 1.0


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise DrawRenderError("FFmpeg is unavailable. Install the bundled renderer dependencies or put ffmpeg on PATH.")


class DrawRenderer:
    def __init__(self, cache_root: Path, hand_asset: Path | None = None) -> None:
        self.cache_root = cache_root
        asset_root = Path(__file__).resolve().parents[1] / "assets"
        if not (asset_root / "draw_hand.png").is_file() and getattr(sys, "_MEIPASS", None):
            asset_root = Path(sys._MEIPASS) / "auto_capcut" / "assets"
        self.hand_asset = hand_asset or asset_root / "draw_hand.png"
        self.push_side_asset = asset_root / "push_hand_side.png"
        self.push_top_asset = asset_root / "push_hand_top.png"
        self.hand_anchor = self._read_anchor(self.hand_asset, "nib_anchor", (0.12, 0.72))
        self.push_side_anchor = self._read_anchor(self.push_side_asset, "contact_anchor", (0.88, 0.50))
        self.push_top_anchor = self._read_anchor(self.push_top_asset, "contact_anchor", (0.50, 0.88))

    @staticmethod
    def _read_anchor(asset: Path | None, key: str, default: tuple[float, float]) -> tuple[float, float]:
        metadata = asset.with_suffix(".json") if asset else None
        if metadata and metadata.is_file():
            try:
                anchor = json.loads(metadata.read_text(encoding="utf-8")).get(key, {})
                return float(anchor.get("x", default[0])), float(anchor.get("y", default[1]))
            except (OSError, ValueError, TypeError):
                pass
        return default

    def _hand(self, size: tuple[int, int], role: str = "draw", mirror: bool = False) -> Image.Image:
        """Load a purpose-specific hand; push phases never use the marker hand."""
        if role == "push_side":
            asset = self.push_side_asset
        elif role == "push_top":
            asset = self.push_top_asset
        else:
            asset = self.hand_asset
        if asset and asset.is_file():
            with Image.open(asset) as image:
                result = image.convert("RGBA")
            if mirror:
                result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return result.resize(size, Image.Resampling.LANCZOS)
        # Missing push assets are intentionally transparent.  Falling back to
        # the marker hand would mix the two visual roles and produce a false
        # push affordance.
        if role.startswith("push"):
            return Image.new("RGBA", size, (0, 0, 0, 0))
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((size[0] * 0.2, size[1] * 0.2, size[0] * 0.8, size[1] * 0.95), fill=(236, 190, 155, 255))
        draw.line((size[0] * 0.55, size[1] * 0.35, size[0] * 0.95, 0), fill=(20, 20, 20, 255), width=max(2, size[1] // 14))
        return image

    def _layer_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGBA")

    def _composite_effect_layer(self, frame: Image.Image, group: ScheduledGroup, progress: float, source_size: tuple[int, int]) -> tuple[float, float] | None:
        if not group.layer or progress <= 0 or not group.object_box:
            return None
        x, y, width, height, rotation, _ = _effect_state(group.object_box, group.effect, group.direction, progress)
        crop = self._layer_image(group.layer.crop_path)
        target_size = (max(1, round(width * source_size[0])), max(1, round(height * source_size[1])))
        crop = crop.resize(target_size, Image.Resampling.LANCZOS)
        if rotation:
            crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        destination = (round((x + width / 2) * source_size[0] - crop.width / 2), round((y + height / 2) * source_size[1] - crop.height / 2))
        frame.alpha_composite(crop, destination)
        return x + width / 2, y + height / 2

    def _push_hand_state(self, group: ScheduledGroup, progress: float) -> tuple[tuple[float, float], float] | None:
        if not group.object_box or group.effect != DrawObjectEffect.PUSH_IN.value or progress < 0.0 or progress > 1.0:
            return None
        box = group.object_box
        x, y, width, height, _, _ = _effect_state(box, group.effect, group.direction, progress)

        # 1. Approach (~15%): hand enters from approach offset, fades in
        if progress < 0.15:
            frac = progress / 0.15
            opacity = max(0.0, min(1.0, frac))
            approach_dist = 0.08 * (1.0 - _ease_out(frac))
            if group.direction == DrawObjectDirection.LEFT.value:
                return (x - approach_dist, y + height / 2), opacity
            elif group.direction == DrawObjectDirection.RIGHT.value:
                return (x + width + approach_dist, y + height / 2), opacity
            elif group.direction == DrawObjectDirection.TOP.value:
                return (x + width / 2, y - approach_dist), opacity
            return (x, y + height / 2), opacity

        # 2. Joint push (~70%): hand contacts moving object edge, full opacity
        elif progress < 0.85:
            opacity = 1.0
            if group.direction == DrawObjectDirection.LEFT.value:
                return (x, y + height / 2), opacity
            elif group.direction == DrawObjectDirection.RIGHT.value:
                return (x + width, y + height / 2), opacity
            elif group.direction == DrawObjectDirection.TOP.value:
                return (x + width / 2, y), opacity
            return (x, y + height / 2), opacity

        # 3. Retract / Release (~15%): object settled at box, hand extends slightly, fades out
        else:
            frac = (progress - 0.85) / 0.15
            opacity = max(0.0, min(1.0, 1.0 - frac))
            retract_dist = 0.03 * _ease_out(frac)
            if group.direction == DrawObjectDirection.LEFT.value:
                return (x + retract_dist, y + height / 2), opacity
            elif group.direction == DrawObjectDirection.RIGHT.value:
                return (x + width - retract_dist, y + height / 2), opacity
            elif group.direction == DrawObjectDirection.TOP.value:
                return (x + width / 2, y + retract_dist), opacity
            return (x, y + height / 2), opacity

    def _push_hand_point(self, group: ScheduledGroup, progress: float) -> tuple[float, float] | None:
        state = self._push_hand_state(group, progress)
        return state[0] if state is not None else None

    def _composite_hand_at(self, result: Image.Image, point: tuple[float, float], viewport: tuple[float, float, float, float], size: tuple[int, int], opacity: float = 1.0, role: str = "draw", direction: str = "auto") -> None:
        if role == "push_side":
            asset = self.push_side_asset
            mirror = direction == DrawObjectDirection.RIGHT.value
            anchor_x, anchor_y = self.push_side_anchor
            if mirror:
                anchor_x = 1.0 - anchor_x
        elif role == "push_top":
            asset = self.push_top_asset
            mirror = False
            anchor_x, anchor_y = self.push_top_anchor
        else:
            asset = self.hand_asset
            mirror = False
            anchor_x, anchor_y = self.hand_anchor
        if not asset or not asset.is_file():
            return
        with Image.open(asset) as opened:
            source_size = opened.size
        hand_height = max(24, round(size[1] * (0.24 if role == "draw" else 0.32)))
        aspect = source_size[0] / max(1, source_size[1])
        hand_size = (max(24, round(hand_height * aspect)), hand_height)
        hand = self._hand(hand_size, role, mirror)
        if opacity < 1.0:
            alpha = hand.getchannel("A").point(lambda value: round(value * max(0.0, min(1.0, opacity))))
            hand.putalpha(alpha)
        x, y, width, height = viewport
        transformed = ((point[0] - x) / max(width, 1e-6) * size[0], (point[1] - y) / max(height, 1e-6) * size[1])
        result.alpha_composite(hand, (round(transformed[0] - hand.width * anchor_x), round(transformed[1] - hand.height * anchor_y)))

    def _write_schedule_diagnostics(self, schedule: DrawSchedule, plan: DrawImagePlan, cache_root: Path) -> None:
        debug = cache_root / "debug"
        debug.mkdir(parents=True, exist_ok=True)
        lines = [
            f"Image {plan.image_index:03d}",
            f"Mode: {plan.mode.value}",
            "",
            "Resolved object order:",
            *[f"{index} {object_id}" for index, object_id in enumerate(schedule.resolved_order, 1)],
            "",
            "Groups & Per-Object Lifecycle:",
        ]
        group_phases: dict[int, SchedulePhase] = {}
        for phase in schedule.phases:
            if phase.kind == "object" and phase.group_index is not None:
                group_phases[phase.group_index] = phase

        for idx, group in enumerate(schedule.groups):
            phase = group_phases.get(idx)
            t_start = phase.start_us if phase else 0
            t_end = phase.end_us if phase else 0
            duration_s = (t_end - t_start) / 1_000_000

            if group.effect == DrawObjectEffect.DRAW.value:
                local_reveal_us = 250_000
                done_us = t_end + local_reveal_us
                lines.append(f"Object: {group.object_id}")
                lines.append(f"  Effect: draw")
                lines.append(f"  Draw phase: {t_start} - {t_end} us ({len(group.strokes)} strokes, {group.path_length:.2f} px path, {duration_s:.3f}s)")
                lines.append(f"  Local color reveal: {t_end} - {done_us} us (0.250s)")
                lines.append(f"  DONE at: {done_us} us")
            elif group.effect == DrawObjectEffect.PUSH_IN.value:
                asset_name = "push_hand_top.png" if group.direction == DrawObjectDirection.TOP.value else "push_hand_side.png"
                asset_file = self.push_top_asset if group.direction == DrawObjectDirection.TOP.value else self.push_side_asset
                loaded = "YES" if asset_file and asset_file.is_file() else "NO"
                anchor = self.push_top_anchor if group.direction == DrawObjectDirection.TOP.value else self.push_side_anchor
                lines.append(f"Object: {group.object_id}")
                lines.append(f"  Resolved effect: push_in")
                lines.append(f"  Direction: {group.direction}")
                lines.append(f"  Fallback: NO")
                lines.append(f"  Push hand asset: {asset_name}")
                lines.append(f"  Asset loaded: {loaded}")
                lines.append(f"  Hand anchor: {anchor}")
                lines.append(f"  Hand visible frame range: {t_start} - {t_end} us (approach=15% joint_push=70% retract=15%)")
                lines.append(f"  Object frame range: {t_start} - {t_end} us ({duration_s:.3f}s)")
                lines.append(f"  Z-order: hand above object")
                lines.append(f"  DONE at: {t_end} us")
            else:
                lines.append(f"Object: {group.object_id}")
                lines.append(f"  Resolved effect: {group.effect}")
                lines.append(f"  Direction: {group.direction}")
                lines.append(f"  Object frame range: {t_start} - {t_end} us ({duration_s:.3f}s)")
                lines.append(f"  Hand: None")
                lines.append(f"  DONE at: {t_end} us")

        if schedule.fallbacks:
            lines.extend(("", "Fallbacks:"))
            for fallback in schedule.fallbacks:
                lines.append(f"FALLBACK DETECTED:")
                lines.append(f"  Object: {fallback.object_id}")
                lines.append(f"  REQUESTED: {fallback.requested_effect}")
                lines.append(f"  ACTUAL: {fallback.effective_effect}")
                lines.append(f"  Direction: {fallback.requested_direction} -> {fallback.effective_direction}")
                lines.append(f"  REASON: {fallback.reason}")

        lines.extend(("", f"Unmatched strokes: {schedule.unmatched_count}", f"Policy: {schedule.unmatched_policy}", f"Sketch: {schedule.sketch_start_us}-{schedule.sketch_end_us}us", f"Color: {schedule.color_start_us}-{schedule.color_end_us}us", "", "Phases:"))
        for phase in schedule.phases:
            endpoints = f" from={phase.from_point} to={phase.to_point}" if phase.from_point or phase.to_point else ""
            lines.append(f"{phase.kind}: {phase.start_us}-{phase.end_us}us ({phase.duration_us / 1_000_000:.3f}s) object={phase.object_id}{endpoints}")
        destination = debug / f"{plan.image_index:03d}_draw_schedule.txt"
        temporary = destination.with_suffix(".partial")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def _get_draw_foreground_mask(
        self,
        crop_rgb: np.ndarray,
        layer: ObjectLayerArtifact | None,
        strokes: tuple[Stroke, ...],
        box_pixels: tuple[int, int, int, int],
        source_size: tuple[int, int],
        canvas_bg: tuple[int, int, int],
        is_text: bool = False,
        exclusion_rects: tuple[tuple[int, int, int, int], ...] = (),
    ) -> Image.Image:
        h, w = crop_rgb.shape[:2]
        left, top, right, bottom = box_pixels

        # 1. Use extracted layer alpha mask if valid and safe
        if layer is not None and layer.rgba_path.is_file():
            try:
                with Image.open(layer.rgba_path) as layer_img:
                    alpha = np.asarray(layer_img.convert("RGBA"))[:, :, 3].copy()
                if alpha.shape == (h, w):
                    coverage = float(np.mean(alpha > 0))
                    border = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
                    border_bg = float(np.mean(border == 0)) if len(border) else 0.0
                    if 0.005 <= coverage <= 0.90 and border_bg >= 0.40:
                        return Image.fromarray(alpha, mode="L")
            except Exception:
                pass

        # 2. Derive deterministic border-connected background removal mask with exclusion rects
        bg_ref = np.asarray(canvas_bg[:3], dtype=np.float32)
        diff = np.linalg.norm(crop_rgb.astype(np.float32) - bg_ref, axis=2)

        exclusion_mask = np.zeros((h, w), dtype=bool)
        for o_l, o_t, o_r, o_b in exclusion_rects:
            int_l = max(left, o_l) - left
            int_r = min(right, o_r) - left
            int_t = max(top, o_t) - top
            int_b = min(bottom, o_b) - top
            if int_l < int_r and int_t < int_b:
                exclusion_mask[int_t:int_b, int_l:int_r] = True

        thresh = 14.0 if is_text else 18.0
        candidates = (diff <= thresh) | exclusion_mask

        stroke_mask = np.zeros((h, w), dtype=bool)
        if strokes:
            s_img = Image.new("L", (w, h), 0)
            s_draw = ImageDraw.Draw(s_img)
            line_w = max(3, round(min(source_size) / 300))
            for stroke in strokes:
                pts = [(round(x * source_size[0] - left), round(y * source_size[1] - top)) for x, y in stroke.points]
                if len(pts) > 1:
                    s_draw.line(pts, fill=255, width=line_w, joint="curve")
            stroke_mask = (np.asarray(s_img) > 0) & (~exclusion_mask)

        try:
            import cv2

            labels_count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=4)
            bg_connected = np.zeros((h, w), dtype=bool)
            if labels_count > 1:
                border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
                border_labels = border_labels[border_labels != 0]
                bg_connected = np.isin(labels, border_labels)

            fg_mask = ((~bg_connected) & (~exclusion_mask)) | stroke_mask
            fg_binary = fg_mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_binary, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 25:
                    fg_binary[labels == i] = 0
            distances = cv2.distanceTransform(fg_binary, cv2.DIST_L2, 3).astype(np.float32)
            feather = np.round(np.minimum(distances, 2.0) / 2.0 * 255.0)
            alpha = np.where(distances >= 2.0, 255.0, np.where(distances > 0, feather, 0.0)).astype(np.uint8)
            return Image.fromarray(alpha, mode="L")
        except ImportError:
            fg_mask = (diff > thresh) & (~exclusion_mask) | stroke_mask
            alpha = np.where(fg_mask, 255, 0).astype(np.uint8)
            return Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=1.0))

    def _frame(self, artifact: ProcessedImage, plan: DrawImagePlan, scene: SceneImage | None, time_us: int, size: tuple[int, int], strokes: list[Stroke] | DrawSchedule) -> Image.Image:
        # The color layer is always the cleaned source RGB image.
        source = Image.open(artifact.cleaned_path).convert("RGBA")
        canvas_bg = _estimate_canvas_background(source)
        original = Image.new("RGBA", source.size, canvas_bg)
        original.alpha_composite(source)
        draw = plan.draw_action
        final_mode = FinalRevealMode(draw.params.get("final", FinalRevealMode.LINE_THEN_COLOR.value).casefold())
        text_mode = TextMode(draw.params.get("text", TextMode.KEEP.value).casefold())
        schedule = strokes if isinstance(strokes, DrawSchedule) else (_build_advanced_schedule(artifact.strokes, plan, scene, original.size, text_mode) if scene and plan.mode is DrawMode.ADVANCED else _basic_schedule(tuple(strokes), plan))
        line = Image.new("RGBA", original.size, canvas_bg)
        line_draw = ImageDraw.Draw(line)
        # Entrance objects remain active even when contour assignment found no
        # strokes; their transparent layer still has a scheduled phase.
        active_groups = [group for group in schedule.groups if group.strokes or group.effect != DrawObjectEffect.DRAW.value]
        group_progress = [0.0] * len(active_groups)
        prefix_point: tuple[float, float] | None = None
        current_phase: SchedulePhase | None = None
        for phase in schedule.phases:
            if phase.start_us <= time_us < phase.end_us:
                current_phase = phase
                break
        for phase in schedule.phases:
            if phase.kind == "object" and phase.group_index is not None and phase.group_index < len(active_groups):
                if time_us >= phase.end_us:
                    group_progress[phase.group_index] = 1.0
                elif phase.start_us <= time_us < phase.end_us:
                    fraction = (time_us - phase.start_us) / max(1, phase.duration_us)
                    group_progress[phase.group_index] = max(0.0, min(1.0, fraction))
        if current_phase is not None:
            if current_phase.kind == "travel" and current_phase.from_point and current_phase.to_point:
                fraction = (time_us - current_phase.start_us) / max(1, current_phase.duration_us)
                prefix_point = tuple(left + (right - left) * fraction for left, right in zip(current_phase.from_point, current_phase.to_point))
            elif current_phase.from_point:
                prefix_point = current_phase.from_point
        line_width = max(1, round(min(original.size) / 900))
        for group_index, (group, progress) in enumerate(zip(active_groups, group_progress)):
            if plan.mode is DrawMode.ADVANCED and group.object_type == "text" and text_mode is TextMode.KEEP:
                continue
            if group.effect != DrawObjectEffect.DRAW.value:
                continue
            for stroke, prefix in _group_prefixes(group, progress):
                if len(prefix) > 1:
                    line_draw.line([(round(x * original.width), round(y * original.height)) for x, y in prefix], fill=(24, 24, 24, 255), width=line_width, joint="curve")
                    if current_phase is not None and current_phase.kind == "object" and current_phase.group_index == group_index:
                        prefix_point = prefix[-1]
                elif prefix and progress > 0:
                    if current_phase is not None and current_phase.kind == "object" and current_phase.group_index == group_index:
                        prefix_point = prefix[-1]

        if plan.mode is DrawMode.ADVANCED and scene is not None:
            frame = line.copy()
            group_phases = {}
            for phase in schedule.phases:
                if phase.kind == "object" and phase.group_index is not None:
                    group_phases[phase.group_index] = phase

            local_reveal_us = 250_000  # 0.25s per-draw-object color reveal

            for group_index, group in enumerate(active_groups):
                phase = group_phases.get(group_index)
                if phase is None:
                    continue
                progress = group_progress[group_index]

                if group.effect == DrawObjectEffect.DRAW.value:
                    if final_mode is FinalRevealMode.ORIGINAL_REVEAL:
                        pass
                    elif final_mode is FinalRevealMode.LINE_ONLY:
                        pass
                    else:
                        box = group.object_box or (scene.object_map[group.object_id].box if scene and group.object_id in scene.object_map else None)
                        if box:
                            left, top, right, bottom = _box_pixels(box, original.size)
                            crop_orig = original.crop((left, top, right, bottom))
                            crop_arr = np.asarray(crop_orig.convert("RGB"))
                            is_text_obj = group.object_type == "text"
                            other_rects = tuple(_box_pixels(other.box, original.size) for other in scene.objects if other.id != group.object_id)
                            fg_mask = self._get_draw_foreground_mask(
                                crop_arr,
                                group.layer,
                                group.strokes,
                                (left, top, right, bottom),
                                original.size,
                                canvas_bg[:3],
                                is_text=is_text_obj,
                                exclusion_rects=other_rects,
                            )
                            if is_text_obj and text_mode in {TextMode.KEEP, TextMode.SIMPLIFIED} and progress > 0 and time_us < phase.end_us:
                                text_mask = fg_mask.point(lambda v: round(v * progress)) if text_mode is TextMode.SIMPLIFIED else fg_mask
                                frame.paste(crop_orig, (left, top), text_mask)
                            elif time_us >= phase.end_us:
                                reveal_frac = min(1.0, (time_us - phase.end_us) / local_reveal_us)
                                if reveal_frac > 0.0:
                                    scaled_mask = fg_mask.point(lambda v: round(v * reveal_frac)) if reveal_frac < 1.0 else fg_mask
                                    frame.paste(crop_orig, (left, top), scaled_mask)
                else:
                    if progress > 0 and group.layer:
                        self._composite_effect_layer(frame, group, progress, original.size)

            if final_mode is FinalRevealMode.ORIGINAL_REVEAL:
                mask = Image.new("L", original.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                for group, progress in zip(active_groups, group_progress):
                    for stroke, prefix in _group_prefixes(group, progress):
                        if len(prefix) > 1:
                            mask_draw.line([(round(x * original.width), round(y * original.height)) for x, y in prefix], fill=255, width=max(2, round(min(original.size) / 100)))
                frame = Image.composite(original, line, mask)
                if time_us >= draw.end_us:
                    frame = original.copy()
            elif final_mode is FinalRevealMode.LINE_THEN_COLOR:
                recon_duration = 200_000
                recon_start = max(draw.start_us, draw.end_us - recon_duration)
                if time_us >= draw.end_us:
                    frame = original.copy()
                elif time_us >= recon_start:
                    recon_frac = min(1.0, max(0.0, (time_us - recon_start) / max(1, draw.end_us - recon_start)))
                    frame = Image.blend(frame, original, recon_frac)
        else:
            for group_index, (group, progress) in enumerate(zip(active_groups, group_progress)):
                if group.effect == DrawObjectEffect.DRAW.value or not group.layer:
                    continue
                self._composite_effect_layer(line, group, progress, original.size)
            frame = line
            if final_mode is FinalRevealMode.ORIGINAL_REVEAL:
                mask = Image.new("L", original.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                for group, progress in zip(active_groups, group_progress):
                    for stroke, prefix in _group_prefixes(group, progress):
                        if len(prefix) > 1:
                            mask_draw.line([(round(x * original.width), round(y * original.height)) for x, y in prefix], fill=255, width=max(2, round(min(original.size) / 100)))
                frame = Image.composite(original, line, mask)
                if time_us >= draw.end_us:
                    frame = original.copy()
            else:
                if text_mode is TextMode.KEEP and scene and plan.mode is DrawMode.BASIC:
                    for obj in scene.objects:
                        if obj.type == "text":
                            box = (round(obj.box.x * original.width), round(obj.box.y * original.height), round((obj.box.x + obj.box.w) * original.width), round((obj.box.y + obj.box.h) * original.height))
                            frame.alpha_composite(original.crop(box), box[:2])
                text_mask = Image.open(artifact.text_mask_path).convert("L")
                if scene and plan.mode is DrawMode.BASIC:
                    manual_text = Image.new("L", original.size, 0)
                    manual_draw = ImageDraw.Draw(manual_text)
                    for obj in scene.objects:
                        if obj.type in {"text", "warning"}:
                            manual_draw.rectangle((round(obj.box.x * original.width), round(obj.box.y * original.height), round((obj.box.x + obj.box.w) * original.width), round((obj.box.y + obj.box.h) * original.height)), fill=255)
                    text_mask = ImageChops.lighter(text_mask, manual_text)
                if text_mode is TextMode.KEEP:
                    frame = Image.composite(original, frame, text_mask)
                elif text_mode is TextMode.SIMPLIFIED and any(group_progress):
                    frame = Image.composite(original, frame, text_mask)
                if final_mode is FinalRevealMode.LINE_THEN_COLOR:
                    color_start = schedule.color_start_us
                    color_duration = schedule.color_duration_us
                    color_fraction = min(1.0, max(0.0, (time_us - color_start) / max(1, color_duration)))
                    if color_fraction >= 1.0:
                        frame = original.copy()
                    elif color_fraction > 0.0:
                        frame = Image.blend(frame, original, color_fraction)

        viewport = _viewport_at(plan, scene, time_us, size[0] / size[1])
        result = _crop(frame, viewport, size)

        # Hand overlay (composited above all frame elements on the cropped viewport)
        if current_phase is not None and current_phase.group_index is not None and current_phase.group_index < len(active_groups):
            current_group = active_groups[current_phase.group_index]
            if current_phase.kind == "object":
                if current_group.effect == DrawObjectEffect.PUSH_IN.value:
                    prog = group_progress[current_phase.group_index]
                    push_state = self._push_hand_state(current_group, prog)
                    if push_state is not None:
                        push_pt, push_op = push_state
                        role = "push_top" if current_group.direction == DrawObjectDirection.TOP.value else "push_side"
                        self._composite_hand_at(result, push_pt, viewport, size, push_op, role, current_group.direction)
                elif current_group.effect == DrawObjectEffect.DRAW.value:
                    if prefix_point and (final_mode is not FinalRevealMode.LINE_THEN_COLOR or time_us < schedule.color_start_us):
                        self._composite_hand_at(result, prefix_point, viewport, size)
            elif current_phase.kind == "travel":
                if current_group.effect == DrawObjectEffect.DRAW.value and prefix_point:
                    self._composite_hand_at(result, prefix_point, viewport, size)
            elif current_phase.kind == "pause":
                if current_group.effect == DrawObjectEffect.DRAW.value and prefix_point:
                    self._composite_hand_at(result, prefix_point, viewport, size)
        return result.convert("RGB")

    def render(self, image_path: Path, plan: DrawImagePlan, config: DrawProjectConfig, output_path: Path, scene: SceneImage | None = None, progress=None) -> Path:
        cache_root = self.cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        text_mode = TextMode(plan.draw_action.params.get("text", TextMode.KEEP.value).casefold())
        artifact = prepare_image(image_path, cache_root, plan.style, text_mode, config.remove_background, scene)
        if progress:
            progress(20, f"Prepared {image_path.name}")
        text_mode = TextMode(plan.draw_action.params.get("text", TextMode.KEEP.value).casefold())
        if scene is not None and plan.mode is DrawMode.ADVANCED:
            with Image.open(artifact.cleaned_path) as cleaned:
                source_size = cleaned.size
            effect_configs, config_fallbacks = _resolve_object_effects(plan, scene)
            layers, effect_configs, layer_fallbacks = _prepare_object_layers(artifact, scene, effect_configs)
            schedule = _build_advanced_schedule(artifact.strokes, plan, scene, source_size, text_mode, effect_configs, layers, config_fallbacks + layer_fallbacks)
            if progress:
                for warning in schedule.fallbacks:
                    progress(20, f"Object {warning.object_id}: {warning.reason}")
            for warning in schedule.fallbacks:
                warning_group = next((item for item in schedule.groups if item.object_id == warning.object_id), None)
                metrics = ""
                if warning_group and warning_group.layer:
                    metrics = f" alpha_coverage={warning_group.layer.alpha_coverage:.3f} background_confidence={warning_group.layer.confidence:.3f}"
                LOGGER.warning(
                    "Image %03d Object %s Requested effect %s direction %s Fallback %s Reason %s%s",
                    plan.image_index,
                    warning.object_id,
                    warning.requested_effect or "draw",
                    warning.requested_direction or "auto",
                    warning.effective_effect or "draw",
                    warning.reason,
                    metrics,
                )
        else:
            schedule = _basic_schedule(artifact.strokes, plan)
        if config.advanced_diagnostics and plan.mode is DrawMode.ADVANCED:
            self._write_schedule_diagnostics(schedule, plan, cache_root)
        width, height = config.resolution
        frames = max(1, math.ceil(plan.duration_us / 1_000_000 * config.fps))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
        if temporary.exists():
            temporary.unlink()
        ffmpeg = _ffmpeg_exe()
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(config.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1", "-map_metadata", "-1", str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            assert process.stdin is not None
            for index in range(frames):
                time_us = min(plan.duration_us, round((index + 1) * 1_000_000 / config.fps))
                frame = self._frame(artifact, plan, scene, time_us, (width, height), schedule)
                process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                if progress:
                    progress(20 + round((index + 1) / frames * 75), f"Rendering {image_path.name} ({index + 1}/{frames})")
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            return_code = process.wait()
            if return_code != 0:
                raise DrawRenderError(f"FFmpeg failed for {image_path.name}: {stderr.strip() or return_code}")
            temporary.replace(output_path)
        except BrokenPipeError as exc:
            stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            process.wait()
            temporary.unlink(missing_ok=True)
            raise DrawRenderError(f"FFmpeg stopped while rendering {image_path.name}: {stderr.strip() or exc}") from exc
        except Exception:
            try:
                process.kill()
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
        if progress:
            progress(100, f"Wrote {output_path.name}")
        return output_path


class DrawRenderService:
    def __init__(self, renderer: DrawRenderer | None = None) -> None:
        self.renderer = renderer

    def render_project(self, config: DrawProjectConfig, effect, images: list[Path], selection: int | None = None, progress=None) -> list[Path]:
        if len(images) != len(effect.images):
            raise DrawRenderError(f"Draw effect mismatch: {len(images)} images / {len(effect.images)} effect cues")
        scene_document = None
        scene_errors: list[str] = []
        if config.scene_file and config.scene_file.is_file() and any(item.mode is DrawMode.ADVANCED for item in effect.images):
            try:
                scene_document = load_scene(config.scene_file)
                scene_errors = validate_scene_document(scene_document, images, config.resolution)
            except SceneValidationError as exc:
                if not config.fallback_basic:
                    raise DrawRenderError(str(exc)) from exc
                scene_errors = [str(exc)]
        renderer = self.renderer or DrawRenderer(config.output_folder / ".autocapcut_draw_cache")
        indexes = [selection] if selection is not None else list(range(len(images)))
        outputs: list[Path] = []
        for index in indexes:
            if index < 0 or index >= len(images):
                raise DrawRenderError(f"Image selection is out of range: {index}")
            plan = effect.images[index]
            if plan.image_name and plan.image_name.casefold() != images[index].name.casefold():
                raise DrawRenderError(f"Image {index + 1}: IMAGE={plan.image_name} does not match {images[index].name}")
            image_scene = None
            if plan.mode is DrawMode.ADVANCED:
                image_scene = next((value for key, value in scene_document.images.items() if key.casefold() == images[index].name.casefold()), None) if scene_document else None
                missing = [item for item in scene_errors if item.casefold().startswith(f"image {images[index].name}".casefold())]
                if image_scene is not None:
                    object_ids = set(image_scene.object_map)
                    requested = [item.strip() for item in plan.draw_action.params.get("order", "").split(",") if item.strip()]
                    for target in requested:
                        if target not in object_ids:
                            missing.append(f"Image {images[index].name}: DRAW order target {target} is missing")
                    for action in plan.actions:
                        target = action.params.get("target")
                        if target and target not in object_ids:
                            missing.append(f"Image {images[index].name}: {action.type.value} target {target} is missing")
                        if target and action.params.get("framing", "camera_frame").casefold() == "camera_frame" and image_scene.object_map.get(target) and image_scene.object_map[target].camera_frame is None:
                            missing.append(f"Image {images[index].name}: object {target} has no camera_frame")
                if image_scene is None or missing:
                    if not config.fallback_basic:
                        raise DrawRenderError("Advanced draw scene errors:\n" + "\n".join(f"- {item}" for item in (missing or [f'Image {images[index].name}: scene record missing'])) )
                    plan = DrawImagePlan(plan.image_index, plan.image_name, plan.start_us, plan.end_us, DrawMode.BASIC, plan.style, "auto", plan.actions)
            output = config.output_folder / f"{index + 1:03d}_draw.mp4"
            outputs.append(renderer.render(images[index], plan, config, output, image_scene, progress))
        return outputs
