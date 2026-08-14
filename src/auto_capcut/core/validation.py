from __future__ import annotations

import json
from pathlib import Path

from auto_capcut.core.errors import ValidationError


def validate_draft_json(project_path: Path, expected_duration_us: int) -> None:
    content_path = project_path / "draft_content.json"
    if not content_path.is_file():
        raise ValidationError("Unable to create project: draft_content.json is missing")
    try:
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("Unable to create project: draft metadata is invalid") from exc
    if not content.get("id") or not content.get("tracks"):
        raise ValidationError("Unable to create project: incomplete draft metadata")
    if abs(int(content.get("duration", 0)) - expected_duration_us) > 1_000:
        raise ValidationError("Unable to create project: timeline duration is invalid")
    for track in content["tracks"]:
        for segment in track.get("segments", []):
            timerange = segment.get("target_timerange", {})
            if int(timerange.get("duration", 0)) <= 0 or int(timerange.get("start", 0)) < 0:
                raise ValidationError("Unable to create project: invalid segment timing")

