from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, TypeVar

T = TypeVar("T")
_PARTS = re.compile(r"(\d+)")


def natural_key(value: str | Path) -> tuple[object, ...]:
    text = str(value).casefold()
    return tuple(int(part) if part.isdigit() else part for part in _PARTS.split(text))


def natural_sorted(values: Iterable[T], key=None) -> list[T]:
    if key is None:
        return sorted(values, key=natural_key)
    return sorted(values, key=lambda value: natural_key(key(value)))

