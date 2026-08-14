from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def patch_capcut_metadata(project_path: Path, project_name: str, duration_us: int) -> None:
    """Fill discovery metadata expected by current CapCut Desktop builds.

    pycapcut owns the draft-content schema. This small patch only updates the
    folder/discovery fields and deliberately leaves machine-specific platform
    identifiers untouched or absent.
    """
    content_path = project_path / "draft_content.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    project_path = project_path.resolve()
    content["id"] = str(uuid.uuid4()).upper()
    content["name"] = project_name
    content["duration"] = duration_us
    # Current CapCut keeps this content field empty and discovers the folder
    # through draft_meta_info.json.
    content["path"] = ""
    content["update_time"] = int(time.time())
    content["new_version"] = "179.0.0"
    platform = content.setdefault("platform", {})
    platform.update({"os": "windows", "app_id": 359289, "app_source": "cc", "app_version": "6.7.0"})
    last_platform = content.setdefault("last_modified_platform", {})
    last_platform.update({"os": "windows", "app_id": 359289, "app_source": "cc", "app_version": "9.1.0"})
    content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = project_path / "draft_meta_info.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"draft_id": str(uuid.uuid4()).upper()}
    meta["draft_name"] = project_name
    meta["draft_root_path"] = str(project_path.parent).replace("\\", "/")
    meta["draft_fold_path"] = str(project_path).replace("\\", "/")
    meta["tm_duration"] = duration_us
    meta["draft_new_version"] = "164.0.0"
    meta["draft_id"] = str(uuid.uuid4()).upper()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
