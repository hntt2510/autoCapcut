from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from auto_capcut.models import AlertOverlayPlan, EffectCue, TargetROI, VisualEffect

OVERLAY_VERSION = "alert-v2"


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_alert_overlay(image_path: Path, roi: TargetROI, effect: VisualEffect, output_dir: Path) -> Path:
    image_hash = _image_hash(image_path)
    try:
        dim = max(0.0, min(1.0, float(effect.params.get("dim_others", "0.30"))))
    except (TypeError, ValueError):
        dim = 0.30
    pulse = effect.params.get("pulse", "0").casefold() in {"1", "true", "yes"}
    key = f"{image_hash}:{roi.x:.8f}:{roi.y:.8f}:{roi.width:.8f}:{roi.height:.8f}:{effect.params.get('style', 'red_warning')}:{dim:.6f}:{pulse}:{OVERLAY_VERSION}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{image_path.stem}-{hashlib.sha256(key.encode()).hexdigest()[:20]}.png"
    if output.is_file():
        return output
    with Image.open(image_path) as source:
        width, height = source.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, round(255 * dim)))
        mask = Image.new("L", (width, height), round(255 * dim))
        draw_mask = ImageDraw.Draw(mask)
        box = (round(roi.x * width), round(roi.y * height), round((roi.x + roi.width) * width), round((roi.y + roi.height) * height))
        draw_mask.rectangle(box, fill=0)
        overlay.putalpha(mask)
        outline = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        outline_draw = ImageDraw.Draw(outline)
        outline_draw.rectangle(box, outline=(235, 50, 45, 240), width=max(4, round(min(width, height) * 0.006)))
        glow = outline.filter(ImageFilter.GaussianBlur(max(2, round(min(width, height) * 0.012))))
        result = Image.alpha_composite(overlay, glow)
        result = Image.alpha_composite(result, outline)
        result.save(output, format="PNG", optimize=True)
    return output


def alert_overlay_plan(image_path: Path, cue: EffectCue, timing_start_us: int, output_dir: Path) -> list[AlertOverlayPlan]:
    plans: list[AlertOverlayPlan] = []
    for effect in cue.effects:
        if effect.type != "ALERT" or not effect.target_id:
            continue
        # The caller supplies the resolved ROI when constructing the final plan.
        # This helper intentionally only names the deterministic derived location.
        plans.append(AlertOverlayPlan(cue.image_index, effect.target_id, timing_start_us + effect.local_start_us, timing_start_us + effect.local_end_us, output_dir / f"{image_path.stem}-{effect.target_id}.png", effect.params.get("style", "red_warning"), effect.params.get("pulse", "0") == "1"))
    return plans
