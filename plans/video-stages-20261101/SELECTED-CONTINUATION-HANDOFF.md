# Corrected Pika and Service CLI handoff

Updated 3 November 2026. The user-selected local continuation is ready for
testing: corrected Pika reference, first-frame-only acceptance, real model-backed
Service CLI verification and this handoff. Extra H3 motion runs are deferred.
This does not mark the entire video plan or the whole application released.

The subsequent optional-reference repair, exact MiniMax/Pika comparison and
advanced workflow delivery are recorded in
[Advanced Pika handoff](ADVANCED-PIKA-HANDOFF.md). That newer handoff supersedes
the optional-input Fix hold and application-source identities below. Original
native receipts retain their recorded source scope.

## Completed results

| Check                                   | Measured native time                   | Result                                                                                                                                   |
| --------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Pika reference, natural drift v3        | 32m 42s                                | 832×480, 124 frames, 24 FPS, original stereo 32 kHz audio; whole editor playback with zero drops/errors                                  |
| Pika first-frame-only, natural drift v3 | 27m 08s                                | Same output geometry/audio; opening image connected, no last-frame source/value; whole editor playback with zero drops/errors            |
| Wan image-to-video through Service CLI  | About 10m 06s including client polling | Actual CLI exited zero and returned named `video`; 512×320, 121 frames, 24 FPS, video-only; whole editor playback with zero drops/errors |

Both Pika workflows use the [version 3 creative recipes](motion-recipes.v3.json),
the retained paper-boat input, immutable model/required-adapter pins and the
official fixed six-evaluation sampler. Full prompts were saved and reopened in
the actual built editor. Existing model settings and wires were retained.

The reference clip now shows gentle drift, small ripples and modest camera
travel rather than a powered-looking wake. Its wider framing keeps the boat
small. The first-frame clip retains the opening boat's identity and plausible
gentle movement; its apparent size grows slightly toward the end while the
whole boat remains visible. These are local creative assessments. The user's
visual review of these two new files is still pending.

The Service proof includes a real current-source UI export, normal CLI
inspect/build, then actual CLI execution through the existing graph executor.
Named inputs are `image` and `prompt`; the named output is `video`. Observed
original HTTP bytes match the package and input projection. All 50 sampling
steps execute; this is not merely metadata inspection or a cached graph result.
The resulting bytes equal the previously approved Wan image clip with the same
inputs and seed. Exact task/attempt/source/cache/durable/served-media joins and
full frame decoding pass. This proves this Wan service route, not every model's
CLI or cross-runtime portability.

## What to test

Open the private review page and play the two new v3 clips first. Compare the
corrected reference against the preserved rejected v2: expect gentle paper-boat
motion and ripples without speedboat-style acceleration/splash. In the
first-frame clip, compare the opening image, follow the boat through the whole
duration, and confirm there is no forced ending frame. Listen to the original
audio separately; muted playback and audio-stream checks do not establish
human listening acceptance.

In **My workflows → Video motion**, the saved titles are:

- **Pika — Paper boat natural drift — v3**
- **Pika — First-frame natural drift — v3**

Both use the existing generic stages. Model-specific conditioning remains
backend-owned. Inspect the opening-image-only connection in the first-frame
workflow; Last Frame must remain absent. Large fresh generations run on Linux,
serially, with the tested resource settings. Reviewing retained clips does not
require another model run.

To reproduce service use on an admitted instance, export its current Service
package, supply named `image` and `prompt` values, and use the documented
[`modiff.service` commands](../../docs/service-prototyping.md). Exported files and
runtime/source manifests are instance-specific private artifacts. Do not use an
old package merely because its model name matches.

## Preservation and readiness

Normal Model Setup and independent input scoped releases, followed by ordinary
idle cleanup, preserve all 64 current app files and durable media. Both deployed
instances are ready, queues/caches empty, and downloads inactive. The original
LTX/Wan worker remains unchanged. Service-created cache controls were visible
after ordinary workflow open, and acknowledged release without browser reload;
this is not a claim about every passive-tab scenario.

All seven human-approved clips and the rejected v2 remain unchanged. There are
ten review clips, including the two new Pika v3 outputs. All ten served files
match their retained hashes. Of 22 pre-existing workflow bodies, 21 remain exact;
all 22 snapshots/settings/graph/media references remain exact. One original
first-frame workflow has only `clientId`, `updatedAt` and `revision` envelope
changes from ordinary reopen. Original/current raw bodies are preserved. No
blanket browser or full-body preservation claim is made.

Application bytes remain backend `579bd123` and client artifact `d357832a`, with
client source/test head `5ad393f3`. The previously passed 8,976-test backend gate,
client check, full 16+322 browser gate and four backend CI jobs are reused.
Current changes are creative descriptors/documentation; no app/runtime/package
update or broad gate rerun was needed. Changes remain in the same two PR #7s;
neither is merged to `develop` by this handoff.

## Remaining and deferred work

- Human visual review of the two new Pika v3 clips, and human listening of audio
  clips. No video publication has been approved.
- H3 camera/action, first-frame-only and reference motion variants are explicitly
  deferred. Earlier H3 technical evidence keeps its actual source scope.
- The optional Reference Assembler Fix defect identified during this checkpoint
  is now fixed and verified through an owned old-schema graph, selected missing
  video/audio cases, Undo and exact Save/reload. See the newer advanced handoff.
- Public video-template qualification, wider resource/Auto combinations,
  unavailable hardware and the existing whole-app Gallery holds remain separate
  (83 retained errors plus five earlier stage-card lock discrepancies).
- The superseded Pika first-frame v2 motion recipe remains unrun; the selected
  first-frame acceptance is closed by its genuine v3 replacement, not relabelled
  as v2 execution.

## Postmortem

The rejected recipe asked a folded-paper boat to behave like a powered boat.
The correction changed the physical action and camera brief, preserving model,
input and sampler settings so the new hypothesis was testable.

A Windows browser-command multiline fill initially kept only the first
paragraph. Full-input/save/reopen comparison caught this before dispatch; a
single-line escaped browser fill preserved all sections. An incorrect helper
assumption about an empty unbound sampler field also stopped before dispatch;
comparison with the original graph confirmed unchanged sampler semantics.

A read-only Windows status watcher later failed on a connection reset while the
independent Linux guard and model continued normally. Bounded read retries and
the genuine terminal response recovered observation without duplicate execution
or model restart. Original browser reconnect events remain evidence; an
uninterrupted original WebSocket is not claimed. Future runs must distinguish
observer failure from model failure before retrying.

Strict native audits were executed once per successful case before owned
release. Resource floors stayed intact. Private original receipts, failed
commands and append-only checkpoint logs remain on both machines. No new model
downloads, cache retirements, extra H3 renders or public uploads were performed
for this continuation.
