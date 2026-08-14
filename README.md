# Auto CapCut

Auto CapCut is a small Windows/PyQt6 utility that creates editable CapCut Desktop drafts directly from image folders, voice audio, SRT subtitles, optional image timing, music, and a logo. It does not open or control CapCut.

## Development

Use Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest
python -m auto_capcut
```

On Windows, the equivalent source-development command is `py -3.12 -m auto_capcut`.

The default draft directory is `%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft`. The configured directory must already exist.

Image motion supports `None`, `Random Light`, subtle zoom/pan modes, and `Effect Direction SRT`. Effect cues are mapped sequentially to naturally sorted images. Optional manual target rectangles are saved beside the effect file as `<effect-file>.roi.json`.

Effect Direction projects use the `VISUAL EFFECTS` card. Modern effects such as `FOCUS_ZOOM`, `PAN_TO`, `PULL_TO`, and `ALERT` use manually configured Camera Frames in the sidecar next to the SRT. Each saved frame is locked to the project canvas aspect and directly defines the final viewport. Legacy focus cues remain compatible with a safe center/directional fallback when no frame is configured. ALERT overlays are generated as editable RGBA media beside the SRT.

Asset-sheet and Cockpit Vision workflows are intentionally not part of the production application. Existing asset caches, visual manifests, and vision sidecars on disk are left untouched and are ignored.

## Portable build

Install PyInstaller, then run `./build.ps1`. The output is a one-folder application under `dist/AutoCapCut/`; keep the whole folder together when moving it to another Windows machine.

## Draw Animation

The Draw Animation tab renders one silent clip per naturally sorted image. Load
an image folder, a `draw_effect.srt`, an optional `scene.json`, and an output
folder. The effect file uses one contiguous SRT cue per image. Each cue contains
`MODE=basic_draw|advanced_draw`, `STYLE=v1|v2`, optional `OBJECTS=auto|manual`,
one `DRAW` action, and optional `FOCUS`, `PAN_TO`, `PULL_TO`, `FULL_VIEW`, or
`SETTLE` actions. DRAW and camera actions may overlap; camera actions may not
overlap one another. Local action times are decimal seconds, for example:

```srt
1
00:00:00,000 --> 00:00:08,000
IMAGE=001.png
MODE=advanced_draw
STYLE=v2
OBJECTS=manual
DRAW 0.00s-4.20s: order=alex,part_ab pause_each=0.10 text=keep final=line_then_color
FOCUS 3.50s-5.20s: target=part_ab framing=camera_frame easing=ease_in_out
SETTLE 5.20s-8.00s: hold final composition
```

Advanced object boxes and draw order are stored in `scene.json` as normalized
`x`, `y`, `w`, `h` values. The Draw Object Editor writes the file atomically.
Rendered intermediates live under `.autocapcut_draw_cache/<source-sha256>/`;
final clips are `001_draw.mp4`, `002_draw.mp4`, and so on. Preview uses the same
renderer at a reduced profile and plays the cached MP4 in the application.
