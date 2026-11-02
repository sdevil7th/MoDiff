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

The complete packaged backend gate passed at `73b649ea`: pinned Ruff E9/F,
package compatibility, CPU preflight and maintained pytest. Result: **8,948 passed,
763 skipped, 33 warnings and 13,176 passing subtests**, in 1,510.09 seconds.
The current browser gate initially retained one cold layout-editor timeout:
15 shared-control cases passed, one module request remained pending, and the
screenshot still showed loading. After backend CPU work ended, the unchanged
isolated case passed; the current full rerun passed all 16 shared controls and
passed 57 mocked cases before an intentional stop for the Pika saved-form defect
below. No timeout or validator was loosened.

## Latest bounded results

Pika's first/last-frame route now completed through the corrected ordinary AV
exporter: 124 distinct decoded frames at 832×480/24 fps, synchronized finite
stereo audio, exact saved/served bytes, and muted Chromium playback to the end
with zero dropped frames. An unchanged repeat reused all nine nodes and retained
the exact MP4. A seed-only Stop case reached genuine sampler progress, cancelled,
recovered ready/idle, and released its owned offload scratch while preserving
saved media. These are native/API lifecycle and playback results; audio listening,
full visible-editor acceptance and public publication remain separate.

H3 first/last task `jPn69jG3gIIF` completed in 8,292.63 seconds on backend
`86efa329`: 960×544, 124 distinct decoded frames, 24 fps and finite stereo 32 kHz
audio. Its MP4 SHA-256 is
`4865320db9fdc1a6962f29ea714cd554068bdf703bfc2cf2b721040ff8d703c5`.
Ten contact-sheet samples and both source views show a coherent red sailboat and
gentle leftward motion; muted Chromium playback reached the end with zero dropped
frames. Listening, full editor acceptance and publication remain open. Normal
release cleared all nine owned nodes and scratch while retaining all 24 prior
application data files exactly.

The explicitly approved older Wan 2.1 FLF2V 14B cache was then retired through the
existing exact-plan-hash API. It recovered 90.10 GB, leaving 450.23 GB before new
reference work. The first plan had correctly refused because this isolated app
lacked its canonical workflow catalog. The complete exact published 209-workflow
catalog was supplied; its one legacy Wan image route now needs a redownload,
while no saved-workflow dependency was found. All 25 protected data files stayed
exact. No other model was retired, and the original LTX/Wan testing app was not
changed.

H3 reference task `auSuWM6x5LJT` is now active, using its fully hash-verified
reference partition. Pika's finite reference download is active concurrently.
Admission reserves its entire pending payload plus 175 GB offload scratch,
160 GB operating space and 20 GB staging/output space. Native reference output
acceptance remains open.

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

## Saved candidate reopening defect (historical diagnosis)

Seven local candidate workflow packages were prepared through the existing
starter/registry/stage-grouping path without model dispatch. Reopening exposed
that Pika had no frontend presentation profile: its graph survived but its form
model was coerced to Z-Image. The current full browser run was intentionally
stopped before source changes; this is a real persistence/presentation gap to
fix before deployment. The Linux reference task and download remain untouched.
The first private comparison also exposed harness-only JSON undefined/display
metadata differences; those are retained separately from the real model-identity
failure. Prepared packages are not yet app-persisted or user-ready.

## Reference-mode checkpoint, 2 November

The saved Pika model/profile correction is published in client `0f702adf`.
All seven private candidates now survive the real store/form/API round trip;
this remains a no-dispatch check, not editor/model acceptance. A related fresh
draft defect is being fixed: a joint video/audio producer must connect both
typed streams to the existing joined exporter rather than separate previews.

H3 keyframes completed in 8,292.63 seconds with 124 distinct frames and stereo
32 kHz audio. Sampled motion/input continuity and muted Chromium playback passed.
Listening, visible-editor acceptance and publication remain open. Its original
source identity is retained; it is not a current-source rerun claim.

H3 reference attempt `auSuWM6x5LJT` failed during vision prompt encoding after
880.76 seconds: eager attention requested 40.64 GiB beyond available GPU memory.
The failed request, trace, inputs and resource samples remain unchanged. Normal
owned-cache/scratch release preserved the 25 protected application files.
The trained 2,048-pixel reference short edge has not been reduced to hide this.

A reference-only ROCm/BF16 attention policy now delegates to the official
Transformers SDPA implementation with PyTorch's scoped BF16 math backend.
Component cache identity and explicit sharing bind that policy; existing text,
keyframe and image loading retain their prior behavior. The fused kernel failed
the large numerical probe and CK is unavailable. The selected math path passed
the exact 26,112-token attention allocation/numerical probe at 55.11 GB peak;
that is random-weight evidence, not pretrained workflow acceptance. The fix has
89 focused CPU tests passing, including tiny official-model execution, failure
restoration, unchanged-route controls and sharing identity. Full final gates and
the real reference retry remain required.

The specifically approved old Wan 14B cache retirement freed 90.10 GB through
the normal plan-bound API after a complete dependency-catalog check. No other
model cache was deleted. Pika's reference selection has now completed all
35 file hashes (146.81 GB selected); available filesystem space is 381.18 GB.
No second download or completed text/keyframe rerun is needed.

## Paired-code and saved-editor checkpoint, 2 November

Backend `8351b62a` packages the exact 119-file client artifact from `d357832a`.
The complete backend CPU/source gate passed: 8,975 tests, 763 skips and 13,176
subtests in 1,543.27 seconds, with Ruff, dependency consistency and CPU preflight
also passing. The complete client `npm run check` passed at `d357832a` against
the frozen compatible backend `9168e6de`. The later client commits `b83e7db6`
and `5ad393f3` change only two browser-test files; all production source,
dependencies and rebuilt bundle bytes remain exact. The full browser rerun at
`5ad393f3` against frozen backend `8351b62a` is still running at this checkpoint.

The first complete browser attempt retained one cold lazy-editor readiness
failure. Its scoped setup now waits for the actual editor to load before the
unchanged field and interaction assertions. A subsequent run passed all 16
shared-control cases and 321 of 322 mocked-app cases. The remaining test still
used the new stage-card creator to reconstruct a historical legacy LTX graph;
it now explicitly uses the legacy test creator while keeping the original
model identity, numeric FPS and tampered-session recovery assertions. Its
focused reproduction passes. Neither correction changes product code or proves
the complete rerun passed before its terminal result exists.

All seven local candidates now pass ordinary Open, Save and reload in the real
built client against the Linux application. The saved bodies also pass normal
store/form/API round trips against their frozen executable graphs. The private
candidate compiler initially overlapped output and setup nodes; the existing
Arrange action corrected presentation positions while preserving form values,
wires and executable inputs. Browser-derived connection/display metadata was
retained rather than overwritten. These are editor/persistence proofs with no
model dispatch, not native browser Run/Stop acceptance or public qualification.
Use [the testing checklist](SECOND-MILESTONE-TESTING.md) for the later handoff.

All seven saved executable graphs also pass real HTTP Service inspection,
package creation and preparation with exact nodes, paths, effective hints and
immutable model pins. Ordinary built-client Wan image Service export passes
with required `prompt` and `image` inputs and a named `video` output. The server
first correctly refused an embedded private file path; naming it as a required
input removes that value from the portable package. This closes metadata/export
inspection only. No service graph was submitted for inference.

The whole-app `npm run check:acceptance` remains failed at Gallery: 83 retained
errors plus five earlier LTX/Wan stage-conversion lock discrepancies. Comparing
the actual template-lock runtime before and after the current Pika changes
produced 90 identical locks. Do not refresh old public-media hashes or relabel
clips to hide this hold. No public video approval exists.

The original LTX/Wan application has a fresh read-only preservation snapshot:
631 application files, 2,271 tracked source files and 105 model snapshots. Its
source/process refresh is prepared but has not executed at this checkpoint.
It is separate from the active H3 reference worker. Deployment must wait for the
current browser gate, recheck protected state and bind the exact published pair;
no package installation, weight download or model inference is needed for that
refresh. Browser-roster preservation and a global preservation PASS are not
claimed by this bounded snapshot.

## Reference timing and monitor handoff

The H3 reference retry passed the original vision-encoding failure and reached
14 of 49 evaluations by 10:19 UTC. Available RAM was 109.00 GB, free filesystem
space 237.62 GB and owned offload scratch 143.41 GB. At about 437 seconds per
evaluation, total reference inference is estimated at 6–6½ hours on this Linux
machine. The earlier text/keyframe timing was an inadequate estimate for this
task. Pika reference and Wan image runs have not started.

After measured reference progress, the whole-task time bound increased from
five to eight hours without changing the task, worker, source, graph, creative
inputs, sampler or resource floors. An independent successor resource guard
was ready before the two owned old observers exited. Original raw captures are
retained as separate segments; a genuine WebSocket connection rejoined the same
session. There was no new graph submission or Stop request. The serial observer
continues only after terminal success, technical media audit and normal scoped
cache release. Failure stops the sequence without retries or further deletion.

The reference-specific step cost should have been measured with a labelled
one-evaluation timing probe at trained geometry before scheduling the full
quality run. Future estimates must use that admission step. Do not repeat this
active full run merely to obtain a better estimate or newer source label.

## Clean-checkout catalog hash correction

GitHub CI exposed six catalog failures on Windows and macOS despite the local
backend gate above passing. The Pika review hash had been computed from a CRLF
working copy; the existing Git policy commits and checks out LF bytes. The
strict validator correctly rejected that clean-checkout mismatch. The Pika
review body is semantically unchanged, its binding now names the existing LF
Git blob, and only that family hash plus the ledger's canonical content hash
change. No admission, blocker, model-execution or validation rule changes.

A regression covering all seven evidence families failed first on Pika's CRLF
bytes. All seven catalog tests and seven subtests pass after the correction.
The complete backend gate must run again on the committed correction. The
ongoing client browser suite uses its frozen backend fixture; the active Linux
inference checkout remains unchanged. Do not deploy the prior hash mismatch or
represent the earlier local PASS as a clean-checkout/CI PASS.

## Corrected final gates and original-app deployment

The corrected backend gate at `579bd123` passed 8,976 tests, 763 skips and
13,183 subtests in 1,604.50 seconds, plus Ruff, dependency consistency and CPU
preflight. The complete client check passed at `d357832a`; the full browser
gate at test-only `5ad393f3` passed all 16 shared and 322 mocked-app cases.
The later backend `17a784ce` changes only the Ubuntu CI timeout from 35 to
45 minutes after its test process passed but the job exceeded its deadline.
These are code/browser gates, not new native or listening acceptance.

The original LTX/Wan app is deployed with backend `579bd123` and the exact
119-file `d357832a` product artifact. Its retained LTX workflow survived actual
built-client reopen/reload; normal save-envelope revisions are recorded
separately from exact snapshot content. Protected application/model contents
remain preserved within the recorded scope. The active H3 reference instance
still uses `9168e6de`; its final idle-only refresh remains pending.

## Motion feedback and revised recipes, 2 November

The user finds the four H3/Pika clips visually good but nearly static. This
supersedes the earlier sampled-motion sufficiency assessment; technical media
checks and original bytes remain valid in their narrower scope. The original
prompts requested gentle drifting and a steady camera. Do not infer a model
capability limit from those briefs or rerun them unchanged.

Official Pika and H3 prompting guidance was reviewed at immutable revisions,
alongside the pinned Diffusers contract. Eight new
[motion candidates](motion-recipes.v2.json) cover concrete subject action,
camera travel and parallax, first-frame-only conditioning, identity references,
and an eight-second two-scene Pika variation with a cut at four seconds. H3/Pika
use their documented structured prompt sections; Wan retains its own prompt
format. Pika keeps the official six-evaluation sampler, scale-one task adapter
and 24 FPS. No invented motion control, Steps, CFG or negative-prompt override
is introduced. No runtime source change is required to pass these prompts or
omit the last frame through the existing stages.

The old unstarted Pika-reference/Wan-image dispatcher was stopped by exact
owned-process identity. The current H3 reference task, graph, worker and guard
are unchanged. A baseline-only observer will audit that run and release its
owned cache; it submits no follow-on graph. The original three-case refresh
precondition is obsolete and must be replaced by a versioned, bound admission.

All eight revised candidates are now normal saved workflows in the Linux-backed
app. They pass actual built-client Open, Save and reload, exact full prompt and
frame-count checks, and post-UI normal store/form/executable-graph round trips.
All 12 prior saved workflow bodies remain exact. One new empty draft appeared
during the browser-check window; it was retained because its browser client
identity was not captured before startup. No model graph was submitted by these
checks. Fresh Pika Run/Stop/recovery, native motion and audio review remain open.
Start with the short Pika action case, then admit the longer two-scene case
against measured headroom. Do not automatically schedule another six-hour H3
reference run before reviewing the cheaper motion results. Original clips,
recipes, failures and raw captures remain preserved. Publication is not approved.

GitHub CI for `17a784ce` has now completed successfully: lint and backend jobs
on Ubuntu, Windows and macOS all pass. This closes the CI timeout hold, not
native or whole-app Gallery acceptance. The motion changes are creative recipe
descriptors, documentation and local saved workflows; application code and
installed runtimes are unchanged, so the passing full code gates are reused.

## H3 reference completion and current video-app deployment

The retained H3 reference task completed in 22,861.73 seconds of native execution
(about 6 hours 21 minutes). Its original source remains `9168e6de`. Strict
task/history/cache/durable/served-byte checks pass with 960×544, 124 decoded
frames at 24 FPS and stereo 32 kHz audio. Ordinary scoped release cleared its
owned cache and scratch while preserving all 46 application files. This closes
technical execution at trained reference geometry; visual identity, sufficient
motion and human listening are not inferred from those checks. Its quiet brief
remains a baseline, with the user's revised motion request still open.

The H3/Pika video instance now serves the tested `579bd123` / `d357832a` pair:
119 disk bundle files and 118 served assets are exact. Its 46 protected files
remain exact except the expected queue worker ID and update timestamp; queue
task bodies, configuration, runtime and packages are preserved. The original
LTX/Wan instance and worker were untouched. No package install, weight download
or model deletion was part of this refresh.

Two maintenance-helper holds are retained. First, the evaluation Python lacked
the required process-descriptor API; it stopped before any process/source/data
mutation. The system Python supplied that maintenance API without changing the
model interpreter. Next, a probe rejected closing `TIME_WAIT` sockets after a
verified graceful shutdown. No listener remained; an explicit stopped-state
continuation used an address-reuse probe, rechecked exact data/process/source
state and completed the same deployment without a second stop or force kill.

The revised Pika camera/action workflow has now been submitted by the actual
built client's Run button. Its executable nodes and paths exactly match the
frozen normal saved-workflow export. An independent Linux resource guard was
ready before allocation; only the genuine browser session captures WebSocket
frames. This first attempt is the explicit UI Stop probe, followed by unchanged
Run recovery and a fresh current joined AV export. Neither the probe nor model
loading is a new motion-quality PASS.

The actual UI Stop probe subsequently cancelled at evaluation 1 of 6 with zero
outputs. A ready replacement worker retained the same application/runtime;
the browser reconnected after four transient handshake resets. Ordinary idle
accelerator cleanup removed 1,300 temporary disk-offload files left by process
termination while preserving all 47 current application files. Free space
returned to 380.81 GB; no model cache was deleted. The same unchanged workflow
is now in its fresh ordinary UI Run, with a new independent guard ready before
allocation. Both actual UI requests match the frozen executable nodes and paths.
Full current-export frame/audio checks, useful motion and the longer timed-cut
case remain pending.

## First revised motion clip and complete current UI lifecycle

The revised Pika camera/action task completed in 1,629.85 native seconds
(about 27 minutes 10 seconds). The current joined exporter retains all 124
decoded frames at 24 FPS with original stereo 32 kHz audio. Strict native
task/attempt/owner/cache/source/durable/served-byte checks pass. The actual
built-client Video Output played the whole 5.167-second clip at normal speed,
muted, with no dropped frames or media error. No finishing repair or model rerun
was used to produce this result.

Whole-duration samples show pedalling and wheel/subject movement, foreground
parallax and a camera move from side-on toward a front three-quarter view.
This demonstrates useful primary motion. The ending frames crop the bicycle
more tightly than requested; the complete visual brief and human listening
remain unaccepted. The new clip is available in the private review alongside
the four unchanged original baselines. It is not a public Gallery admission.

The genuine native required-adapter receipt reports the exact pinned Pika
artifact at scale one. The official guide maps both text and endpoint-frame
tasks to `fl2va/adapter.safetensors`; reference generation uses its distinct
adapter. This identity check is not general user-LoRA qualification.

One unchanged ordinary UI repeat completed in 4.10 native seconds. Original
browser events report all five executable nodes as cached with unchanged
inputs; the resident recipe remained reusable and the media bytes are identical.
Actual Model Setup scoped release cleared the five nodes and their temporary
scratch while preserving both durable outputs and all 50 current application
files. The app's durable previews remain available after release. No model cache
was deleted.

The separate eight-second, 192-frame Pika two-scene workflow is now admitted
and running through the normal built UI. Its actual nodes/paths match the frozen
saved workflow and its independent guard was ready before loading. The requested
forest-to-plaza cut near four seconds, identity, movement in both shots and audio
still require actual output review. Pika reference and Wan image remain open;
the old unstarted quiet recipes are not silently resubmitted.

## Native timed cut and remaining reference execution

The eight-second Pika case completed successfully. Strict media checks retain
all 192 frames at 24 FPS with original stereo 32 kHz audio and exact native
task/owner/source/durable/served-byte bindings. The actual built-client Video
Output played all eight seconds, muted, with 192 presented frames, zero dropped
frames and no media error.

Whole-duration samples show forest cycling, the root-crossing action and a
distinct plaza/fountain shot with the same red jacket, black helmet and blue
bicycle. Analysis of the unmodified native video detects the cut at frame 100,
about 4.17 seconds. This is a generated scene transition, without stitching or
review re-encoding. The forest camera tracks the subject; the plaza camera is
largely fixed, so the requested strong arc in that shot is not clearly achieved.
The complete creative brief and human listening remain open. Both new motion
clips are available privately alongside the four unchanged original baselines.

Actual Model Setup scoped release and ordinary idle cleanup cleared this case's
cache/scratch while preserving its durable bytes and all 52 current application
files. The already completed five-node cached-repeat proof is reused; this case
was not rerun just to repeat it. No model cache was retired.

The revised Pika reference-motion workflow is now running through the normal
built UI with nine executable nodes. Its guard checks the complete cached
35-file reference selection against the committed closure and binds the retained
512×288 boat input to its original staging hash. It keeps the trained 1,024-pixel
reference preprocessing, existing ROCm/BF16 reference attention policy, required
reference adapter and official six evaluations. The new task passed reference
encoding and reached sampling; full output/motion review is pending. No new
weights were downloaded. The revised Wan image case remains next after its
technical review and release.
