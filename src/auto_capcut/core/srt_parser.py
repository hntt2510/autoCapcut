from __future__ import annotations

import re
from pathlib import Path

from auto_capcut.core.errors import SRTParseError
from auto_capcut.models import ImageTiming, SubtitleCue

_TIMESTAMP = re.compile(
    r"^(?P<hour>\d{1,3}):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)[,.](?P<milli>\d{1,3})$"
)
_ARROW = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)(?:\s+.*)?$")
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1")


def _decode(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SRTParseError(f"Unable to read SRT file: {path}") from exc
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SRTParseError(f"Subtitle SRT cannot be decoded: {path}")


def parse_timestamp(value: str) -> int:
    match = _TIMESTAMP.match(value.strip())
    if not match:
        raise SRTParseError(f"Invalid SRT timestamp: {value}")
    milliseconds = int(match.group("milli").ljust(3, "0"))
    return (
        int(match.group("hour")) * 3_600_000_000
        + int(match.group("minute")) * 60_000_000
        + int(match.group("second")) * 1_000_000
        + milliseconds * 1_000
    )


def _blocks(text: str) -> list[list[str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    return [block.split("\n") for block in re.split(r"\n\s*\n", text) if block.strip()]


def parse_srt(path: str | Path) -> list[SubtitleCue]:
    source = Path(path)
    text = _decode(source)
    cues: list[SubtitleCue] = []
    for block_index, lines in enumerate(_blocks(text), 1):
        if not lines:
            continue
        timing_line_index = 1 if len(lines) > 1 and "-->" not in lines[0] else 0
        if timing_line_index >= len(lines):
            raise SRTParseError(f"Missing timing line in SRT cue {block_index}")
        match = _ARROW.match(lines[timing_line_index])
        if not match:
            raise SRTParseError(f"Invalid timing line in SRT cue {block_index}")
        start_us = parse_timestamp(match.group(1))
        end_us = parse_timestamp(match.group(2))
        if end_us <= start_us:
            raise SRTParseError(f"SRT cue {block_index} has a non-positive duration")
        cue_text = "\n".join(lines[timing_line_index + 1 :]).strip()
        if not cue_text:
            raise SRTParseError(f"SRT cue {block_index} is empty")
        cues.append(SubtitleCue(len(cues), start_us, end_us, cue_text))
    if not cues:
        raise SRTParseError(f"SRT file is empty: {source}")
    return cues


def parse_image_timing_srt(path: str | Path) -> list[ImageTiming]:
    cues = parse_srt(path)
    timings = [ImageTiming(cue.index, cue.start_us, cue.end_us) for cue in cues]
    if timings[0].start_us != 0:
        raise SRTParseError("Image Timing SRT must start at 00:00:00,000")
    previous_end = 0
    for timing in timings:
        if timing.start_us != previous_end:
            raise SRTParseError("Image Timing SRT contains a gap or overlap")
        if timing.end_us <= timing.start_us:
            raise SRTParseError("Image Timing SRT contains a non-positive duration")
        previous_end = timing.end_us
    return timings

