# Auto CapCut

Auto CapCut is a production-ready Windows/PyQt6 application that creates editable CapCut Desktop drafts directly from image folders, voice audio, a single Main Effect SRT, subtitle tracks, background music, and logos. It builds native CapCut Desktop draft project files directly without automating or opening CapCut.

## Development

Requires Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest
python -m auto_capcut
```

On Windows, the source-development run command is:
```powershell
py -3.12 -m auto_capcut
```

The default draft directory is `%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft`. The configured directory must exist.

## Production Workflow

### 1. Single Main Effect SRT
In normal production, a single **Main Effect SRT** file serves as the sole source of truth for both:
1. **Image Timing**: The start and end timestamps of each SRT cue directly dictate the on-screen duration of each corresponding image.
2. **Visual & Draw Choreography**: Directives within each cue specify whether the slide uses `basic_draw` (default) or `advanced_draw` (optional), the completion buffer, and post-draw camera motions.

**Validation Rules**:
- The number of SRT cues must equal the number of images.
- Cue 1 must start at `00:00:00,000`.
- Cues must be contiguous (no gaps or overlaps).
- The final cue end time must match the main audio duration within tolerance.

### 2. Draw-First Pipeline
- **`basic_draw` (Default)**: Automatically analyzes the image, detects background colors, synthesizes authentic stroke trajectories, and renders progressive marker drawing animations.
- **`advanced_draw` (Optional)**: Orchestrates distinct scene objects with entrance effects (`draw`, `push_in`, `slide_in`, `pop_in`, `toss_in`) and transient focus camera zoom/restore.
- **Scene JSON**: Only required when using `advanced_draw`. Created and visually adjusted via the **Edit Draw Objects** dialog.
- **Completion Buffer & Post-Draw Motion**: Drawing finishes before the slide boundary (via `COMPLETE_BEFORE_END`), allowing smooth post-draw camera motions during the remaining buffer duration.
- **Six Post-Draw Motions**:
  - `none`
  - `random_light`
  - `subtle_zoom_in`
  - `subtle_zoom_out`
  - `subtle_pan_left`
  - `subtle_pan_right`

### 3. Audio, Subtitles, Logo, and Background Music
- **Main Audio**: Supports single audio files or batch audio folders.
- **Subtitles**: Imports standard `.srt` subtitle tracks onto the CapCut draft timeline.
- **Logo**: Positions and overlays custom logo branding across the draft duration.
- **BGM**: Plays background music from a dedicated folder with adjustable volume.
- **Transitions**: Smooth blur transitions between adjacent visual segments.

### 4. Developer Tools (Draw Debug)
The **Draw Debug** tab is an internal developer panel for testing isolated image renders, inspecting intermediate stroke masks in `.autocapcut_draw_cache/`, and verifying object bounding boxes. It is not part of the normal production pipeline.

## Portable Build

To build a standalone Windows one-folder distribution:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

The output is written to `dist\AutoCapCut\AutoCapCut.exe`. Keep the folder structure together when deploying to other Windows machines.

## Testing

Run the full automated test suite (199 tests):

```powershell
py -3.12 -m pytest -q
```