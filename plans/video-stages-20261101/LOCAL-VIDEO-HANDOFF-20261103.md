# Current local video delivery handoff

Updated 3 November 2026 after the 17:39 UTC user visual verdict. This closes
the single photographic MiniMax comparison and shared audio-upload follow-up.
It does not close the broader catalog, hardware or publication release gate.

## Ready for testing

- Windows app: the existing local server on port **8194**.
- Linux video app: port **8218** on Linux, forwarded to **8298** on Windows.
- Both run tested backend **`496f3aed`**, client product **`75080b73`**. Later
  canonical backend commits only document outcomes; no additional model rerun or
  deployment is needed for documentation labels.
- The original LTX/Wan app and worker remain untouched. Use the updated video
  app for the repaired upload flow.
- **Template — Pika photographic cyclist keyframes** and **Template — Pika
  photographic image video audio references** remain delivered and visually
  accepted. Their portable source packages intentionally require user media.

In either updated app, add Video.Load or Audio.Load and upload an MP4 that has
an actual soundtrack. Upload should succeed; video or soundtrack previews should
load. Save/reopen should retain the selected input. Removing an input clears its
preview. Input inspection needs no large-model execution. A graph containing
only those input nodes remains blocked until it has a connected output.

## Single MiniMax comparison

The user authorized exactly one photographic H3 keyframe run using accepted
Pika task `thUUt9jxaheu`'s exact prompt and opening/ending image bytes.
MiniMax task `DUC96v3XXjU-` completes in **11,825.66 native seconds (3h17m06)**.
Strict native checks pass: 960×544, 124 decoded frames, 24 FPS, stereo 32 kHz,
nonzero finite audio and exact source/task/attempt/input/owner/cache/durable/
served-byte bindings. It retains its actual execution source `ce0fb40c`.

The private comparison presents MiniMax beside unchanged accepted Pika
keyframes. Geometry and native sampling differ; this is not an equal-compute
benchmark. Twelve samples show photographic cycling, parallax and jacket
movement, with a tight ending crop. Fully visible ordinary built-app Gallery
playback presents all 124 frames with zero drops/errors. The earlier clipped
canvas-player result remains separately diagnosed and is not a canvas pass.

Normal scoped release preserves all 123 then-current app files and durable
media, clears owned caches and removes temporary offload scratch. No model
cache is retired. The private review now contains **18 clips/25 video instances**;
all 17 earlier objects/23 earlier instances and nine accepted/eight rejected
verdicts remain unchanged. **The user now visually accepts the MiniMax clip**,
bringing the current verdict to ten accepted and eight rejected clips. The
exact verdict is "looks good, what next?"; original media remain unchanged. No
extra H3 run, new H3 template, listening or publication approval is inferred.

## Upload fix and validation

An explicit Audio.Load MP4 upload was rejected as video even when it contained
real audio. The server now admits that explicit soundtrack upload, keeps the
original container classification/bytes and rejects silent or wrong-kind media.
Ordinary strict probes and original-file delivery remain strict.

- Full backend gate at `496f3aed`: Ruff, dependency checks, preflight and
  `python -B -m pytest -q` pass: **8,993 tests, 762 skips, 13,194 subtests**.
- All four backend CI jobs at that commit and all four client CI jobs at
  `66b5152` pass. Previous browser checks remain scoped to their original source
  and unchanged client product; no broad suite was repeated for documentation.
- Windows: one protected cutover, 119 exact disk/118 served assets; real upload,
  full muted playback, clear/rebind and complete Save/reload pass without models.
- Linux: one pidfd-bound graceful cutover to the exact tested commit. Runtime,
  packages, configuration, 119 disk/118 served assets and the original worker
  remain preserved. Actual video/audio uploads retain the original 4,509,153-byte
  file. One video/two audio previews load; the soundtrack finishes muted playback.
  Normal import, Save/reload and blocked Run pass with **zero graph submissions**.
- All 57 earlier Linux workflow bodies remain exact; all 59 current bodies are
  exact after reload. All 123 earlier app files remain exact apart from the
  explicitly authorized restart queue fields. Six owned additions comprise two
  uploaded files, two ordinary managed preview derivatives and two QA records.
- All 18 private clips still serve their exact original hashes after adoption.

## Preserved diagnostic lessons

Retain failures and identify the changed hypothesis before repeating a check.
The already inspected supervisor was reparented after its deployment controller
exited; its birth/executable/argv/cwd/uid and worker parent remained exact. Bind
fresh current ownership; do not adopt an exited controller's PID as authority.

The final read-only auditor initially expected five additions and omitted the
normal managed video-preview MP4. The sixth exact file is bound to existing
source-digest/conversion options and actual response bytes. Original data remain
exact; the failed counter is retained. Do not delete an expected derivative or
relax the check to allow arbitrary extra files.

Playback admission must confirm that the actual player is visible and
unobstructed. Cache-release observation must follow the real `/cache` route and
recover original requests after observer failure instead of resubmitting an
already successful mutation. Do not rerun expensive models for helper failures
or newer documentation/source labels.

## Still separate

The whole-app Gallery has 73 recorded failures; current release qualification
covers a 90-template/203-workflow/60-resource denominator with substantial
missing current evidence. Unavailable Apple Silicon cannot be qualified on
Windows/Linux. Wider Auto/offload/resolution claims and extra H3 variants remain
outside this bounded delivery. Human listening and public video publication
remain separate. Both existing PR7s remain open, with no new branch/worktree or
merge implied. See [the current release closeout](RELEASE-CLOSEOUT-20261103.md).
