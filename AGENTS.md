AGENTS.md — AutoCapCut Autonomous Completion Contract

This file is the source of truth for Antigravity and delegated subagents.

0. Mission

Finish AutoCapCut as a draw-first, production-ready Windows tool that creates editable CapCut drafts from images, audio and exactly one main effect.srt.

Normal production flow:

Choose image folder(s).

Choose audio and optional subtitles/logo/BGM.

Choose one main effect.srt.

Configure draw_scene.json only for images using advanced_draw.

Click CREATE CAPCUT PROJECT.

AutoCapCut renders draw clips, inserts them at exact cue positions, preserves audio/subtitles/logo/BGM/transitions, and produces a valid editable CapCut draft.

The user must not need to understand internal render engines, effect catalogs, captured CapCut materials, or separate Draw-vs-CapCut pipelines.

1. Product invariants

These override older architecture/tests when they conflict.

1.1 One production effect file

Normal production uses exactly one main effect.srt.

Its SRT cue timestamps are the per-image timing source.

Cue order maps to natural-sorted image order.

Separate Draw Effect SRT is not required.

Separate Image Timing SRT is not required in normal production.

Subtitle SRT remains separate.

Legacy fields may exist internally only if they do not create a second normal workflow.

1.2 Draw-first by default

Every production image cue is a draw cue.

Missing MODE => basic_draw.

basic_draw => whole-image whiteboard draw; no object setup.

advanced_draw => object-aware draw/entrance/camera choreography; scene objects required.

Most narrator images should work with basic_draw automatically.

1.3 Advanced draw is optional

Use advanced_draw only for scenes that need spatial choreography, such as:

list/comparison scenes,

multiple cards/panels,

narration explicitly referring to a region/object,

push/slide/toss/pop actions,

short target focus.

Do not force object setup for normal images.

1.4 Completion buffer is mandatory

For one narrator sentence = one image, drawing must finish before cue end.

Priority:

COMPLETE_BEFORE_END <seconds> if explicit.

Otherwise clamp(25% of cue duration, min=1.5s, max=3.0s).

During this final buffer:

composition is complete and full-color,

no pending object appears,

only allowed subtle post-draw motion may run,

transitions occur only at cue boundary.

1.5 Only six production post-draw motion choices

Production UI may expose only:

None

Random Light

Subtle Zoom In

Subtle Zoom Out

Subtle Pan Left

Subtle Pan Right

These are post-draw motions, never replacements for drawing.

They must not run while marker drawing, push-hand motion, object entrance, object finalization, or transient focus is active.

Per-cue POST_MOTION overrides project-level selection.

Random Light must be deterministic for the same cue/input.

1.6 Remove old CapCut 24xx/captured-effect production dependency

Final production must not depend on:

local CapCut effect scanning,

24xx effect IDs,

effect catalog promotion,

captured effect template injection,

preset matching,

Effect Catalog Tester,

unresolved CapCut effect presets.

Delete obsolete modules/tests/UI when truly unused. Do not leave hidden production fallbacks.

1.7 Full view is the neutral camera state

FULL_VIEW is default.

Object entrances normally happen in full view.

Focus is short temporary emphasis.

Normal lifecycle: focus-in -> hold -> focus-out -> exact full view.

Camera must not move while marker/push hand is active.

New image starts from full view with no leaked state.

2. Preserve stable systems unless a verified defect requires change

Do not rewrite stable code merely for style.

Preserve:

DrawRenderer / FFmpeg backend.

basic_draw / advanced_draw.

object editor and one multi-image canonical draw_scene.json.

object lifecycle PENDING -> ACTIVE -> FINALIZING -> DONE.

local full-color reveal after object completion.

scene-aware ROI seam fixes.

transparent foreground extraction.

draw / slide_in / drop_in / push_in / toss_in / pop_in.

dedicated push hands.

marker hand only for drawing.

transient focus and exact Camera Frame geometry.

target validation.

draw render cache + invalidation.

rendered MP4 duration validation.

explicit DRAW never silently becoming static image.

audio/subtitles/logo/BGM/transitions.

atomic/staging draft publication.

3. Autonomous operating mode

The user does not want to supervise individual implementation steps.

3.1 Full non-destructive CLI authorization

Antigravity/subagents may, without asking:

inspect the full repo,

edit source/tests/docs/build files,

run PowerShell/cmd/Python/Git/FFmpeg/FFprobe/PyInstaller,

use rg, git grep, Get-ChildItem, scripts/temp fixtures,

install project/dev dependencies,

launch the desktop app,

create temporary test drafts/renders,

inspect generated JSON/MP4 metadata/frames,

use Git status/diff/log/blame/fetch/pull,

add regression tests,

delegate to subagents,

repeat repair loops until acceptance passes.

Do not ask the user for implementation choices already defined here.

3.2 Destructive-command guardrails

Full CLI permission does not allow destroying unrelated/user work.

Never use without unavoidable documented reason:

git reset --hard

git clean -fd / git clean -fdx

force-push

recursive deletion outside generated temp/cache/build output

deletion/modification of unrelated CapCut projects

global OS configuration changes

At every run start:

git status --short
git branch --show-current
git log -5 --oneline
git diff --stat
git diff

Preserve unrelated uncommitted work.

3.3 Git policy

Commit only verified milestones/coherent repairs.

Do not mix unrelated formatting/refactors.

Push verified commits if remote is configured and normal push is possible.

Never force-push.

Never deploy.

Commit/push is not evidence of PASS; independent QA still required.

4. Supervisor/subagent workflow

Primary Antigravity agent is the supervisor.

For every goal:

Phase A — Baseline

Inspect architecture/tests before editing.

Confirm branch/HEAD/worktree.

Reproduce behavior where possible.

Identify smallest compatible change.

Phase B — Builder subagent

Delegate a narrow implementation spec.

Builder must:

preserve unrelated behavior,

add/update tests,

report changed files,

run focused tests.

Phase C — Reviewer subagent

Use a different subagent/role to inspect:

actual diff,

architecture fit,

hidden fallbacks,

test quality,

dead code,

regressions.

Builder does not self-approve.

Phase D — Independent QA

For desktop/render features test real runtime/output, not mocks only.

Inspect:

UI layout/controls,

interaction,

progress/loading,

error states,

generated MP4,

generated CapCut draft JSON/material/timeline,

runtime exceptions/logs,

representative video frames.

If CapCut is installed, open generated draft as final smoke when relevant.

Phase E — Repair loop

If QA fails:

exact reproduction,

expected vs actual,

evidence/log/frame/JSON,

delegate only verified defect,

rerun complete relevant QA.

Maximum five repair attempts per defect/goal.

After five failures, mark BLOCKED with evidence and continue independent goals. Never weaken acceptance criteria.

Phase F — Close milestone

Before PASS:

focused tests pass,

full suite passes,

compile check passes,

real-runtime QA passes,

final diff reviewed,

no unrelated changes,

no temp/debug artifacts committed.

5. Standard verification commands

Current Python target: 3.12.

Minimum:

py -3.12 -m pytest -q
py -3.12 -m compileall src

Inspect pyproject.toml before requiring lint/type tools. If configured, run them.

Packaging:

powershell -ExecutionPolicy Bypass -File .\build.ps1

Runtime:

auto-capcut
# or
py -3.12 -m auto_capcut.main

Use FFmpeg/FFprobe or existing media probing for duration/codec/readability.

Prefer temp dirs for automated QA.

6. Completion goals

Re-inspect current HEAD before deciding if a goal is already partially complete.

G01 — Remove legacy 24xx/captured CapCut effects

Objective

No old effect-catalog/captured-effect dependency in production.

Required work

Remove production captured-effect repository/cloner imports/calls.

Remove local effect scan/preset matching production paths.

Remove Effect Catalog Tester from app UI.

Delete obsolete catalog/scanner/captured-effect modules/tests when unused.

Remove ALERT behavior that depends on captured CapCut effects; if needed, implement emphasis inside draw renderer or omit.

No 24xx IDs/material payloads required.

Acceptance

rg/git grep finds no production dependency.

Project builds with local CapCut effect database/cache absent.

Full tests pass.

G02 — One effect.srt = timing + visual source

Objective

Normal production must not require Effect Direction SRT + Image Timing SRT + Draw SRT.

Required

Main SRT timestamps define image timing automatically.

Same SRT defines draw mode/choreography/post-motion.

Cue count == image count.

Cue order == natural image order.

Production UI has one Main Effect SRT selector.

Separate Image Timing SRT may be removed or clearly legacy/advanced only.

Separate Draw Effect SRT must not appear in normal UI.

Rename misleading Effect Direction SRT labels to Effect SRT / Main Effect SRT.

Minimal cue:

1
00:00:00,000 --> 00:00:06,000
Image 001 DRAW

Defaults:

MODE basic_draw
STYLE v1
OBJECTS auto

Supported production directives:

MODE basic_draw|advanced_draw
STYLE v1|v2
COMPLETE_BEFORE_END <seconds>
POST_MOTION none|random_light|subtle_zoom_in|subtle_zoom_out|subtle_pan_left|subtle_pan_right
OBJECT_EFFECT ...   # advanced only
CAMERA_AFTER ...    # advanced only

Acceptance

A multi-image project builds using one main effect.srt and no second timing/effect SRT.

G03 — Production-grade basic_draw visual fidelity

Objective

Basic draw must be good enough for most images with no object setup.

Required

Frame 0 does not reveal a mostly complete source image.

Start from clean/estimated background + draw state.

Progressive reveal.

Text/icons do not appear as ugly source rectangles.

No gray ROI blocks, seams, black wedges, transparency gaps.

Final reconciles to original visually losslessly/exactly.

Preserve off-white backgrounds; never hard-code pure white.

text=keep must not reveal all text at frame 0.

Must handle infographic-heavy images.

Visual acceptance

Use a representative infographic fixture with title, 3 panels, icons/checkmarks, character, off-white background.

Inspect frames at:

start,

25%,

50%,

75%,

draw completion,

completion buffer,

final.

Numeric tests alone are insufficient.

G04 — Optional polished advanced_draw

Objective

Advanced mode adds deliberate object choreography only where useful.

Required

One canonical multi-image draw_scene.json.

Production Edit Draw Objects handles all project images.

Scene optional for basic cues.

Validate explicit targets before FFmpeg.

Unknown target blocks with available names.

Advanced -> basic fallback only when explicit setting allows.

Completed objects never revert.

Entrance objects are foreground-only, no rectangular crop backgrounds.

Push uses dedicated non-marker hands.

Marker hand only for draw.

Focus transient; exact full view before next major object unless explicit persistence.

Camera does not overlap marker/push hand in normal choreography.

Acceptance

Maintain a 6-object scene covering:
draw, slide_in, toss_in, side push, pop_in, top push, transient focus, final full view.

Visual inspection required.

G05 — Lock completion buffer + post-draw motion

Required

Deterministic completion buffer.

Explicit buffer overrides default.

All auto work finishes before buffer.

Impossible fixed durations fail clearly; no silent overlap/truncation.

Composition is fully finished during buffer.

Post-motion starts only after draw/choreography completion.

No mask/foreground recomputation per frame from post-motion.

Per-cue POST_MOTION overrides global.

None = static hold.

Random Light deterministic.

Acceptance

Test short/medium/long cues and explicit overrides; visually confirm final 1.5–3.0s (or explicit buffer) show finished image.

G06 — Simplify production UI

Objective

New user should understand workflow without implementation history.

Production sections should center around:

Project

Images

Main Effect SRT

Audio

Subtitles

Draw Scene / Edit Draw Objects

Post-draw Motion

Transitions

Logo

BGM

Output

Create CapCut Project

Cleanup

Remove old Effect Direction terminology.

Remove effect catalog tester/menu.

No separate Draw Effect SRT.

Scene JSON clearly optional / advanced-only.

Scene status shows basic vs advanced requirements.

Pre-build parse/timing/scene validation.

Disable irrelevant controls.

Draw Animation tab is debug-only; rename e.g. Draw Debug or move to developer tools.

QA

Check common desktop sizes:

no clipped controls,

scrolling,

enable/disable,

readable errors,

progress updates,

no duplicate/misleading SRT fields.

G07 — Full CapCut draft E2E reliability

Required

Draw clips inserted at exact cue positions.

No duplicate CapCut camera keyframes on draw clips.

Real clip duration within frame tolerance.

Audio aligned.

Subtitle timing unchanged.

Logo correct.

BGM correct.

Transitions work across draw boundaries.

No missing media.

Atomic publish only after validation.

Any cue failure prevents misleading partial final draft.

Real smoke

At minimum:

3 images,

one main SRT,

cue1 basic,

cue2 advanced,

cue3 basic,

audio,

subtitles,

logo,

BGM,

transitions,

post-motion,

completion buffers.

Inspect draft JSON, media paths, segments, tracks, rendered frames. Open in CapCut if installed.

G08 — Batch scale/cache/progress/fault isolation

QA sizes

1 image

10 images

50 images

200 images (synthetic lightweight acceptable)

Requirements

deterministic cache reuse,

invalidation on image/effect/scene/resolution/FPS/render-option/algorithm-version changes,

basic cues avoid advanced overhead,

scene loaded once per batch,

bounded memory,

progress shows image/cue + overall %,

errors identify exact image/cue/object,

failed build leaves no corrupted final draft,

no generated artifacts committed.

Record rough timing/memory; optimize only verified bottlenecks.

G09 — Windows portable build acceptance

Required

build.ps1 succeeds.

Packaged app launches outside repo CWD.

PyQt assets load.

marker/push-hand assets load.

FFmpeg draw works without manual system FFmpeg configuration if bundled/imageio behavior is intended.

pycapcut imports.

Packaged runtime can create a small basic_draw CapCut draft.

No source-tree-only paths.

Source-only success is not DONE.

G10 — Final product acceptance/cleanup

Project is DONE only when all pass:

full pytest,

compileall,

Windows portable build,

3-image full-feature real smoke,

batch stress,

basic visual acceptance,

advanced visual acceptance,

no 24xx/captured production dependency,

one Main Effect SRT in production UI,

README matches final workflow/DSL,

obsolete code/tests/docs cleaned safely,

worktree clean except intentional work,

no MP4/PNG/cache/draft/temp/debug artifacts in final diff.

7. Definition of visual PASS

For render goals, PASS requires all three:

unit/integration tests,

real renderer/draft smoke,

human-equivalent frame/video inspection by QA subagent.

Do not substitute synthetic metrics for visible QA.

8. Scope control

Do not expand scope until G01–G10 are PASS.

Not required for core DONE:

OCR/AI automatic object detection,

LLM narration-to-choreography,

cloud rendering,

new CapCut effect catalog features,

arbitrary cinematic camera,

extra entrance-effect families,

deployment/update service.

Auto Choreography Planner is post-core only.

9. Reporting format

After each autonomous cycle:

STATUS: PASS / PARTIAL / BLOCKED / FAIL

GOAL:
G0X — <name>

IMPLEMENTATION:
- ...

TESTS:
- focused: ...
- full pytest: ...
- compileall: ...
- build/package if relevant: ...

QA:
- runtime flow: ...
- visual/output inspection: ...
- defects repaired: ...

FILES CHANGED:
- ...

GIT:
- branch: ...
- commit: ...
- push: ...
- worktree: clean/dirty

REMAINING RISKS:
- ...

NEXT GOAL:
G0Y — <name>

Do not stop after planning. Continue autonomously through implementation, independent review, QA and repair until the current goal is PASS or genuinely BLOCKED.

10. Observed baseline when authored

Observed main around commit 6015b55a8fbb1afce2d9bd3fd3ca081b2cf474af already had partial draw-first work:

default basic_draw,

COMPLETE_BEFORE_END,

completion-buffer calculation,

post-draw motion,

six-value production motion dropdown,

draw clip insertion into CapCut draft,

production object editing,

render cache/duration validation,

synthetic 3-image draw-first tests.

Agents must re-inspect current HEAD. At authoring time, legacy captured/effect-catalog production code still existed, so the project was not DONE.