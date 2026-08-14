from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def default_draft_folder() -> Path:
    local_app_data = Path.home() / "AppData" / "Local"
    return local_app_data / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"


def safe_name(value: str, fallback: str | None = None) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" .")
    value = re.sub(r"\s+", " ", value)
    if not value:
        value = fallback or f"Auto_{datetime.now():%Y%m%d_%H%M%S}"
    return value[:120]


def unique_project_name(root: Path, desired: str) -> str:
    name = safe_name(desired)
    candidate = name
    index = 2
    while (root / candidate).exists():
        candidate = f"{name}_{index}"
        index += 1
    return candidate

