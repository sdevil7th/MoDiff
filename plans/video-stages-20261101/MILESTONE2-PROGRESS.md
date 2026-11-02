# MiniMax/Pika progress and evidence boundaries

Updated 2 November 2026 IST. Work remains on the same two PRs, with source authored
in the canonical Windows pair and native models on Linux. The first LTX/Wan
testing app has not been replaced by this in-progress milestone.

## Completed in the current slice

| Change                  | Evidence and scope                                                                                                                                                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H3 Model Setup          | Native task `RotZTltUc8G4`, 118.70 seconds; loading only                                                                                                                                                                                |
| Conditioner entrypoints | Source-declared leaf hooks for the actual nested Qwen3-VL call; tiny numerical red/green regression and four exact-provider Linux checks                                                                                                |
| Audio VAE placement     | Honor the official group-offload opt-out; reviewed resident placement for the small float32 H3/Pika audio VAE, with actual per-component cache policy. 193 focused checks, one skip, 32 subtests; four exact-provider Linux checks      |
| H3 engineering export   | Task `7BojyLbXZkAp`, backend `b770e165c7b665ba9006ec98df89e744841d8543`; 512×288, 124 frames, 24 fps, two sigma points / one model evaluation. Completed in 552.83 seconds without a resource guard                                     |
| Actual saved media      | SHA-256 `35468d91636462bbd701df90ca6479e94432de14972bc1da23d63e3c1e37c65c`; 2,687,607 bytes. All 124 frames decode and differ; stereo 32 kHz audio is finite and non-silent. Exact cache-owner/file bytes agree                         |
| Same-compute reuse      | Task `40RBQUbQ2JpZ`, 0.375 seconds; actual WebSocket events report all five nodes cached. One genuine persisted output record. This is API/executor evidence, not a visible browser or quality signoff                                  |
| Inline adapter message  | LTX/Wan Model Setup accurately reports its unsupported inline LoRA editor instead of blaming valid consumers. 387 focused checks and two mounted-control browser cases; graph/export unchanged                                          |
| Video input history     | Record LTX condition strength separately from denoise strength and identify the ordered frozen PIL source-video sequence/count. 44 backend checks; 43 client receipt/history checks. No path/media-byte capture or old-history backfill |
| Export preservation     | Borrowed audio and prior completed outputs survive failed muxing; owned temporary intermediates, alias rejection and atomic destination replacement. Previously retained tiny real-media regressions                                    |

The engineering clip is blurred and does **not** meet the recognizable-sailboat
brief. The first raw API graph also omitted the normal client preview
`display`/`sourceKey` metadata, so its original persisted outputs list stays empty.
The separate cached replay adds that metadata without changing compute inputs;
its genuine receipt is not retroactively assigned to the first task.

## Latest paired integration checkpoint

The ordinary video template card path had still opened a legacy Generate graph,
although the split-stage factory passed its tests. One shared recipe predicate
now covers normal cards, secondary creation and the Gallery harness. Five retained
LTX/Wan cards preserve their creative settings and finishing branches through
save/reopen. Verification includes 368 related unit checks, one normal-card/reload
browser case and two retained managed-document browser cases. Seven additional
H3/Pika/Wan image candidates still need normal-entrypoint/template acceptance;
this correction does not qualify them.

The new Wan image adapter also extended a raw loader selector, which changed
nine older compiled Block schemas despite all 292 older execution specifications
remaining exact. Older compiled recipes now retain their original selector domain;
the new image recipe retains its explicit choice. The strict generated Block
catalog matches its unchanged bytes. Focused preservation checks: 78 passed and
904 subtests.

Complete client `npm run check` passed at `f557b9f9`, paired with backend
`4329a759`. The 119-file production bundle is mirrored exactly. Five dependent
ledgers changed only their hashes; semantic and qualification rows are unchanged.
Measured gzip growth is 20 bytes in the largest startup file and 21 bytes in total
startup code; deferred code decreases by 93 bytes. Loading membership, dependency
versions, minification and both deferred caps are unchanged.

The complete baseline browser gate passed 16 shared-control and 320 mocked cases
at the frozen earlier pair. Current focused browser results above are additional
coverage; they are not a current complete browser-gate claim. Backend full v6
reported 8,941 passed, 763 skipped and three stale public-inventory assertions.
Those assertions and the compiled-schema defect are corrected with focused checks;
a complete current backend PASS remains required.

## Latest bounded results

Pika's first/last-frame route now completed through the corrected ordinary AV
exporter: 124 distinct decoded frames at 832×480/24 fps, synchronized finite
stereo audio, exact saved/served bytes, and muted Chromium playback to the end
with zero dropped frames. An unchanged repeat reused all nine nodes and retained
the exact MP4. A seed-only Stop case reached genuine sampler progress, cancelled,
recovered ready/idle, and released its owned offload scratch while preserving
saved media. These are native/API lifecycle and playback results; audio listening,
full visible-editor acceptance and public publication remain separate.

H3 first/last-frame inference is active on Linux. Its reference partition now
has a complete full-hash acquisition receipt. Pika reference remains pending
capacity admission; the user approved retiring the older Wan 2.1 FLF2V cache
after H3 becomes idle. No deletion during active model execution is permitted.

The missing Wan 5B image route now has an explicit loader/profile using official
WanImageToVideoPipeline and the existing shared stages. It preserves all 292
pre-existing complete execution specification bodies, with one new image
specification. No new model-named node, sampler implementation, runtime upgrade,
Auto grant or native qualification was introduced. The prior text route and its
Auto rules remain exact. No-weight focused and cross-contract checks passed;
guarded Linux execution remains required before this route is accepted.

The earlier complete client and browser checks passed with a verified frozen
backend copy. Its isolated CPU interpreter was explicit; older CUDA-fixture and
default-Expert-versus-Auto test mistakes remain archived. The latest paired
integration checkpoint above distinguishes current source from that baseline.

The H3 quality task `eBU924y4iqro` completed in 7,979.47 seconds at 960×544,
124 frames and 24 fps, without triggering a resource guard. Its saved MP4
SHA-256 is `22e7aed778073eb834921bc6309ccdd9ac6a708f9f98c4aa7de90bd301e6ab6f`.
All frames decode; sampled frames show a clear red sailboat and stable motion.
The audio is finite and non-silent but quiet. Full playback/listening and public
creative qualification remain open; the earlier failed engineering brief remains
failed.

The exact saved H3 clip also played to the end in isolated Chromium: 124 decoded
frames, zero dropped frames. Playback was muted; this closes the technical browser
playback row only, not audio listening or full workflow-editor acceptance.

Fresh H3 authoring now selects version-two shared-stage specifications with an
explicit reviewed provider. All three saved version-one specification hashes and
all 139 existing library hashes remain exact. New task starters use the shared
reference helper, while the saved legacy helper schema remains unchanged. The
source-admission audit resolves a candidate's declared profile rather than a newer
default. Focused verification: 94 passed and 1,009 subtests. Visible authoring,
paired integration and final template acceptance remain open.

Native Pika exposed a progress binding defect: the public upstream `blocks`
property returns a copy. The corrected bridge binds the owned executing tree;
real-pipeline red/green checks cover progress, Stop and restoration without
changing model math. The current native attempt retains its original source and
resource monitor; the correction requires subsequent guarded deployment and
native lifecycle proof.

The complete client `npm run check` passed at `26a426b5`, paired against backend
`3499b26d`. Its exact 119-file bundle is committed in the backend. Five derived
ledgers were refreshed with hash-only changes and no qualification promotion.
The private app was refreshed with protected data checks; the original LTX/Wan
testing app remains unchanged. Current complete backend and final browser gates
remain open.

Pika's first attempt failed before weights because its published tokenizer type
and the official block's base type differed in preflight. After that correction,
task `oN5ClcZ_oOTh` loaded the components but failed at tokenizer registry lookup
in about 122 seconds, without OOM or a resource guard. The adapter had flattened
a three-field Modular loading declaration into a standard two-field class pair;
upstream consequently skipped the tokenizer. The correction preserves that
declaration and is checked before another native attempt. A separate rejected
dispatch during runtime-lock contention submitted no model task. Runtime checks
and app startup are now serialized and readiness includes the exact active
optional provider.

New history recognition covers exact shared H3/Pika routes while retaining saved
H3 actions. Focused task/Auto/profile checks: 123 passed. No old output receipt is
rewritten, and recognizing a task grants no Auto, template or runtime eligibility.

## In progress and still open

Pika task `jpqAjXcnc1GI` completed the six-step 832×480 / 124-frame model route
in 1,559.65 seconds without a resource guard. The decoder retained all 124 frames,
but its first video/audio export contained only 121. The strict audit failed and
that original remains unchanged. A cached ordinary video export confirmed all 124
frames; ordinary retained-video audio mux task `pRAwqJi-SwLF` then copied that
complete video stream with the retained generated soundtrack, without a model
rerun. The repaired media SHA-256 is
`217bf479436e2c41a50cf9c5618af020c29eda8e38164a7e17e99d54f19d463a`.
All 124 decoded frames match the complete retained video export; stereo32kHz audio
is finite/non-silent and its duration agrees within one video frame. Sampled
frames show a clear sailboat and stable scene. Actual isolated Chromium playback
passed for both the H3 clip and repaired Pika clip: 124 decoded frames, zero
dropped, playback ended normally. These checks were muted; listening and public
qualification remain open. Both saved clips remained byte-exact over HTTP after
ordinary scoped cache release; all 17 private app data files were preserved.

The permanent shared exporter correction keeps all video packets and fits audio
to video duration. Tiny real-media regressions reproduced the truncation without
models. Related retained-video mux cleanup now preserves existing source-adjacent
audio. Export/composition checks: 48 passed. The corrected `ExportWithAudio` still
needs current Linux native integration; the repaired file is explicitly a separate
deterministic finishing result, not a relabelled successful original export.

The complete client check passed again at `9bc8fa7`, using the reviewed compatible
backend contract `2b5f1353`. The retained-spec catalog audit includes only exact
id/hash/profile joins and reproduces the unchanged catalog. The first gate attempt
exposed that missing retained-spec join; the next caught a stale documented CI pin.
Both failed logs remain in private evidence. The paired bundle has 119 exact disk
files. The complete backend gate then exposed 15 failures: missing catalog-only
Pika disposition and tests with older inventory, optional-provider and image-only
replacement assumptions. A new hash-bound Pika held-family review preserves the
structural/execution distinction. All affected checks now pass: 195 passed, two
skipped, 2,132 subtests. This focused result is not a complete backend gate PASS.
The read-only-input regression guard also fails immediately rather than causing
Windows temporary-directory creation to retry indefinitely. Guarded private
deployment and remaining product acquisition/native checks continue.

- H3 quality playback/listening, remaining conditioned routes and lifecycle.
  The completed quality graph SHA-256 remains
  `f6a7fb473ee0228f661f3b151459352885aa66c8b9f4678b347514468c389e95`.
- Pika: all 35 selected files, 146,806,845,173 bytes, downloaded and fully
  hash-verified at its immutable revision. First model output and bounded repair
  evidence are described above.
  Source-accurate catalog and shared-stage starter checks pass without rewriting
  the 139 existing definition hashes. The exact-provider Expert profile is
  declared. Its original 32-block hierarchy preserves all 43 placement contexts;
  all 101 workflows pass the existing nested-graph/control/subtree round-trip
  suite. Executable Cluster admission, visible hierarchy editing and actual
  adapter/sampler acceptance remain open.
- H3 keyframe/reference routes, the remaining Wan image route, finishing stages,
  all twelve final creative recipes, final lifecycle and visible browser checks.
- Complete current backend/client gates, exact paired bundle, protected refresh,
  supported/held matrix and final manual handoff. The earlier complete backend
  attempts exposed stale image-catalog fingerprints and an incremental-loader
  accessor assumption. The semantic-preserving refresh and authoritative component
  map fix pass their affected tests. Neither attempt is a complete current gate PASS.
  A subsequent backend attempt was deliberately interrupted after the client gate
  exposed Pika's missing source-specific hierarchy. That integration is now fixed
  in both repositories; complete current gates still require successful reruns.
- Public clip publication requires exact-file review; existing image approvals
  do not approve these videos.

## Logging and mistakes retained

The complete client check passed at `c614595`, against compatible backend
contract `73b4fe8b`. Shared parsing avoids separate legacy/generic artifact
envelopes. The measured feature allowance is 262/267/297 gzip bytes for the
largest startup file/total startup/deferred code, preserving previous headroom.
All four startup and 90 deferred asset names retain their loading membership;
only the shared contract/policy and model-manager chunks change beyond import
versions. No dependency, minifier or loading-rule change was made. The packaged
119-file bundle is byte-exact; all five derived-ledger changes are hashes only,
with qualification rows unchanged. The earlier size-gate failures remain logged.

The complete UI check initially had one cold lazy-editor loading timeout (15
other shared-control cases passed). Its screenshot, network trace and original
failure are archived. A targeted reproduction passed without a source change or
timeout increase; the complete UI rerun remains required.

The native keyframe log also exposed an ignored frame-count argument. Both
shared families and legacy H3 now carry that declared control through official
`PipelineState` to before-denoise, rather than passing it to an image-only
preprocessor. Four regressions failed first; 36 focused tests and nine subtests
then passed. The ongoing native 124-frame candidate retains its actual older
source identity; no running checkout was changed or output receipt relabelled.

The current-source first/last-frame attempt `aO0X_KxV7YkF` stopped before
denoising: its starter incorrectly inserted an empty reference-list assembler.
The generic Before Encode stage now projects only the selected task's inputs;
hidden auxiliary inputs must not create supplier nodes. Keyframe tasks keep
first/last images, while reference tasks keep their ordered-reference assembler.
Two earlier launcher attempts had missing transferred private files and submitted
no model task. Their failure logs remain retained; file presence must be checked
on the destination host, not inferred from the authoring workspace.
The task-port correction and related app-acquisition/contract checks now pass:
85 tests and 343 subtests. Fresh conditioned graphs must be compiled again from
that corrected starter; an older graph must not be patched and relabelled.

App acquisition is being integrated through finite task artifact selections and
the existing Hub API. The first endpoint regression attempt used an unavailable
async pytest plugin; the corrected synchronous harness reproduced 12 failures
with six valid selections passing. The fix passes 64 focused checks. The generic
client inventory passes 127 authoring tests and two real browser selector cases,
including distinct installed status for text and reference partitions. These are
application contract/UI checks, not new pretrained model outputs.

Raw attempts, source/dependency identities, selected-file receipts, resource
samples, terminal API responses, original WebSocket events, media hashes and
commands remain in the private campaign evidence. The append-only event log
records failures and harness corrections rather than replacing them with a
later PASS. No private logs, media, model weights or host configuration belong
in Git.

The fixed native defects were empty optional-adapter dispatch, conditioner calls
bypassing outer group hooks and an official audio-VAE offload opt-out being
ignored. Initial overly coarse transient-swap guards and missing test-preview
metadata were harness mistakes, not model fit or product quality failures.
The resource monitor keeps a 20 GiB available-RAM floor, bounded worker swap,
175 GB owned scratch and 160 GB free-filesystem reserve. Concurrent Pika download
growth is measured separately from the owned model scratch. Do not override
official model flags, modify model math or rerun unchanged failing allocations.
