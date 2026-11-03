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
| H3 engineering export   | Task `7BojyLbXZkAp`, backend `b770e165c7b665ba9006ec98df89e744841d8543`; 512Ã—288, 124 frames, 24 fps, two sigma points / one model evaluation. Completed in 552.83 seconds without a resource guard                                    |
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
exporter: 124 distinct decoded frames at 832Ã—480/24 fps, synchronized finite
stereo audio, exact saved/served bytes, and muted Chromium playback to the end
with zero dropped frames. An unchanged repeat reused all nine nodes and retained
the exact MP4. A seed-only Stop case reached genuine sampler progress, cancelled,
recovered ready/idle, and released its owned offload scratch while preserving
saved media. These are native/API lifecycle and playback results; audio listening,
full visible-editor acceptance and public publication remain separate.

H3 first/last task `jPn69jG3gIIF` completed in 8,292.63 seconds on backend
`86efa329`: 960Ã—544, 124 distinct decoded frames, 24 fps and finite stereo 32 kHz
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

The H3 quality task `eBU924y4iqro` completed in 7,979.47 seconds at 960Ã—544,
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

Pika task `jpqAjXcnc1GI` completed the six-step 832Ã—480 / 124-frame model route
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
evaluation, total reference inference is estimated at 6â€“6Â½ hours on this Linux
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
task/history/cache/durable/served-byte checks pass with 960Ã—544, 124 decoded
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
512Ã—288 boat input to its original staging hash. It keeps the trained 1,024-pixel
reference preprocessing, existing ROCm/BF16 reference attention policy, required
reference adapter and official six evaluations. The new task passed reference
encoding and reached sampling; full output/motion review is pending. No new
weights were downloaded. The revised Wan image case remains next after its
technical review and release.

## Reference motion passes; Wan image execution active

The revised Pika reference-motion case completed in about 32 minutes 23 seconds
through the ordinary saved-workflow Run. Strict checks join its actual task,
attempt, owner, source and durable/served bytes; all 124 frames at 24 FPS and
original stereo 32 kHz audio are retained. Whole built-client muted playback
presented all 124 frames with zero drops and no media error. The native
application receipt contains the required reference adapter's exact immutable
artifact, hash and scale one.

Twelve whole-duration samples show the retained red sailboat travelling with a
wake, changing view and background parallax. Useful primary reference motion
passes for local testing. Human listening and final creative/publication
approval remain separate. The result is first on the seven-clip private review
page; the previous six media records and original outputs remain unchanged.

Actual Model Setup scoped release cleared its model-dependent nodes. The
independent image/reference preparation caches correctly remained until the
normal source-node scoped release; an early helper assertion expecting an empty
cache stopped before cleanup. This was a helper scope assumption, not an
application or preservation failure. Both ordinary releases and idle cleanup
then preserved all 54 current app files and durable served video bytes, with
empty cache and no remaining owned scratch. No model cache was deleted.

The revised Wan 5B image-to-video workflow is now running through the actual
built UI. Its six-node submitted graph matches the frozen saved workflow
exactly. Before Run, a separate monitor checked its retained immutable cache,
complete indexed safetensors, original image hash, active reviewed runtime,
idle queue and memory/disk reserves. It uses 512Ã—320, 121 frames, 24 FPS,
50 steps, guidance five, BF16 and resident placement, with video-only export.
No Pika sampler/audio assumptions or new downloads are introduced. Its native
media, conditioning/motion review, cleanup and final bounded handoff remain
open. Earlier H3 technical evidence keeps its true source and is not rerun
merely for labels; unrun H3/first-frame motion variants stay unaccepted.

## Wan image output delivered; queue ready for local testing

The revised Wan opening-image case completed in 10 minutes 24 seconds through
the ordinary saved-workflow Run. Strict checks join actual task/attempt/owner,
source and durable/served video bytes: all 121 frames, 512Ã—320, 24 FPS and no
audio stream, as expected. Whole actual built-client playback presented all
121 frames with zero drops and no media error. The input boat remains
recognizable, with travel, a wake and a strong camera approach. The ending
heavily crops the sail; the requested controlled arc and full creative brief
remain open. No unchanged retry was submitted to hide that limitation.

Actual Model Setup and independent source-node scoped releases, followed by
ordinary idle cleanup, preserve all 56 current app files and the durable served
video. Cache and queue are empty; the original LTX/Wan process is untouched.
Both reference and Wan outputs are available privately, with eight clip records
including the four unchanged baselines. No models were downloaded or deleted.

This closes the two remaining native reference/Wan rows in bounded local scope.
Revised Pika first-frame-only and H3 camera/action, first-frame-only and reference
motion candidates remain unrun. Historical H3 technical outputs keep their
actual source and do not establish new motion. The current results and existing
lifecycle proofs are ready for manual review; full creative/audio acceptance,
model-backed service CLI verification, public-template qualification and the
retained whole-app Gallery/hardware holds remain separate. The complete video
plan is not marked verified from this local checkpoint.

## User visual review â€” reference boat rejected

The user visually accepts seven of the eight displayed outputs and specifically
praises both cycling videos. The Pika referenced-boat result is rejected because
it looks like an unrealistic paper speedboat. This supersedes the earlier local
creative assessment for that exact output; its technical/media checks still
pass and the original recipe, input and video remain preserved.

The recipe asked a paper-boat reference to move several hull lengths, throw a
bow splash and leave a widening V-shaped wake. That mismatch is the hypothesis
for a future version: physically plausible slow motion and gentle ripples, with
more of the visible action supplied by camera movement. No new render or input
replacement was started. The other seven displayed clips do not need reruns
solely for the previously noted framing/camera deviations. Visual approval does
not establish audio listening, publication permission or acceptance of the four
unrun revised motion candidates.

## Authorized corrected Pika and Service CLI continuation

The user selected a finite continuation: corrected Pika reference, first-frame
acceptance, actual model-backed Service CLI verification and handoff, preserving
approved clips and deferring extra H3 renders. Two version 3 creative recipes
keep the paper-boat input and fixed native settings but ask for plausible slow
drift, gentle ripples and modest camera travel instead of powered-looking motion.

The corrected reference was duplicated, edited, saved and reopened through the
built client. Its full six-section prompt and native exported graph are checked;
only prompt/output filename differ from the previous executable recipe. The
first-frame version is separately saved with its full four-section prompt,
opening image connected and last frame intentionally absent. Its eight-node
export likewise preserves all other execution values and wires. The corrected
reference has started through actual UI Run with its independent resource guard
ready before allocation. No second model is dispatched concurrently.

A current-source Wan Service package was exported through the normal dialog
with named image/prompt inputs and video output. It will be used for actual CLI
inference after both Pika cases are reviewed and released; metadata export alone
does not close that proof. Existing code gates and approved output evidence are
reused. Browser authoring failures and narrower preservation results remain in
the private log; no raw evidence or old recipe is overwritten.

## Corrected reference v3 delivered; first-frame v3 active

The corrected reference completed in 32 minutes 42 seconds. Strict native
task/media checks retain all 124 frames at 24 FPS and original stereo 32 kHz
audio. Whole actual built-client muted playback presented all 124 frames with
zero drops or media errors. Whole-duration samples show the original folded
paper boat drifting gently, small ripples and modest camera travel, without the
previous powered-looking wake. The framing is wider and the boat remains small;
final human visual/audio acceptance is still separate from this local review.

Actual Model Setup and independent input scoped releases plus ordinary idle
cleanup preserved all 60 current app files and the durable video. The new clip
is first on the private review page; all eight prior media records and seven
user approvals remain unchanged, including the rejected v2 diagnostic.

First-frame v3 then started through the normal built UI with its own guard ready
before allocation. Its exact eight-node graph has the opening image connected
and no last-frame value/source. The guard uses the finite fl2va artifact closure
and required frame adapter, not reference-only preprocessing. No concurrent
model or extra H3 run was started. Model-backed Service CLI verification remains
next after this case's actual output review and owned release.

## Selected continuation complete and ready for testing

First-frame v3 completed in 27 minutes 8 seconds through actual built-client
Run. Its eight-node graph connects the opening image and supplies no last frame.
Strict native task/source/media checks pass all 124 frames at 832Ã—480, 24 FPS,
with original stereo 32 kHz audio. Full editor playback presented all frames
with no drops or media errors. Opening identity, gentle movement and whole-boat
framing pass local review; human visual/audio acceptance remains pending. Owned
loader/input releases preserved all 62 current app files and durable media.

The subsequent real `modiff.service` CLI run completed in about 10 minutes
6 seconds including client polling, exited zero and returned named `video`
from the named image/prompt inputs. Original observed HTTP bytes match the
current-source exported package and submitted graph. Actual fresh computation
includes all 50 sampling steps. Its 121-frame, 512Ã—320, 24 FPS video-only output
passes strict task/attempt/source/durable/served-byte joins and complete editor
playback with zero drops/errors. The bytes equal the already approved Wan clip
under the same recipe. This closes actual service inference for this route,
not every model or runtime. Service-created cache controls were visible after
ordinary workflow open and acknowledged scoped release without reload.

Final scoped releases/idle cleanup preserve all 64 app files and durable media.
Both deployed apps are ready with idle queues/empty caches and no downloads;
the original LTX/Wan worker is unchanged. All ten private review files remain
hash-exact, including seven human-approved clips, rejected v2 and two new v3s.
All 22 pre-existing snapshots remain exact; 21 full bodies are exact. The sole
envelope-only first-frame v2 change remains explicitly retained, not restored or
promoted into a blanket preservation PASS.

The [selected handoff](SELECTED-CONTINUATION-HANDOFF.md) records tests, expected
results, command-observer incidents and remaining limits. Recipe/document-only
changes reuse the full passing application gates. No downloads, deletions,
new runtime, broad gate reruns, extra H3 renders or public uploads occurred.
Human review of the two new clips, listening/publication, the optional-input Fix
diagnostic and separate whole-app/Gallery/hardware qualification remain open.

## Optional-reference repair and advanced Pika workflows

The optional-input Fix defect is now fixed, deployed and verified on an owned
old-schema reference graph. Selected image/video/audio requirements remain
enforced, and Undo/Save/reload preserves graph values/wires. The complete backend
gate passes 8,982 tests/13,185 subtests with 763 skips; matched client CI passes
all four jobs including the full 16+322 browser suite. Prior failed Windows
navigation evidence retains its own source and outcome.

Three serial Pika runs add four private outputs: exact retained MiniMax keyframe
prompt/two-image comparison (23m08), mixed image/video/audio references (53m35),
and actual last-frame cyclist continuation with separate composed timeline
(27m12). Exact native media audits and full editor playback pass, with 124 native
frames or 248 composed frames at 832Ã—480, 24 FPS, stereo 32 kHz. The timeline seam
and container timing pass local inspection. Camera arc/pullback limitations
remain explicit. No extra MiniMax run, download, retirement or public upload was
performed. All original media objects and seven user visual approvals remain.

Normal scoped releases preserve 71/77/85 app files and durable media. Arranged
6/12/14-node presentations survive save/reload without overlaps. The
[source examples](../../examples/video/README.md) are ordinary packages with
empty media fields, unchanged sealed definitions and verified parser/form/
execution roundtrips; private media rebinding reproduces the actual native
recipes. They do not promote public Gallery or hardware qualification.

Empty portable-template import exposed a final literal-file readiness issue.
The client-only repair passes focused tests, all five affected browser cases
and the complete client check. An abandoned raw-schema prototype and incomplete
gate are retained as failures, not signoff; all raw schema additions were
reverted so backend runtime and immutable Block identities remain unchanged.
Final pairing/deployment and actual empty-template save/reopen closeout follow
the [advanced handoff](ADVANCED-PIKA-HANDOFF.md). Model output is not rerun to
validate a presence check or a new documentation source label.

The final protected refresh and actual template closeout now pass: backend
`328d06f8` / client product `ba8f7d52`, exact 119 disk/118 served files and 86
protected app files except permitted queue worker/timestamp. Both applications
are ready, queues/caches/downloads idle, original LTX/Wan process unchanged.
All three portable templates import/save/reopen with missing-media Run blocked,
exact complete snapshots and executable graphs, preserving other workflow bodies
and submitting no model task. Fourteen review media files, ten prior clip objects,
seven approvals and all 24 full workflow bodies from the preceding current
baseline remain exact. The finite local update is ready for user testing;
new visual/audio review, publication and wider release gates remain separate.

## Latest human verdict: paper-boat recipes rejected

The user rejects all eight paper-boat clips and accepts the six other displayed
outputs: two conventional sailboat text outputs and four cyclist outputs,
including the continuation and composed timeline. This supersedes prior visual
approval and local positive assessments for those exact paper outputs. Original
media, recipes and technical evidence remain unchanged. No new render was run.

The keyframe and mixed-reference paper recipes are held for creative replacement;
the continuation template is visually accepted. The next proposed step is one
photographic cyclist-reference Pika pilot using retained accepted media, reviewed
before any further variants. New rendering waits for user direction. Audio and
publication approval remain separate; code/CI fixes require no repetition.

## Single photographic reference pilot in progress

Following the user's go-ahead, one separate Pika reference pilot uses frame zero
from the accepted cyclist camera/action clip. Source video bytes are verified;
the deterministic PNG extraction preserves all 92 existing app files. The
ordinary nine-node reference graph changes only input image, complete prompt and
output filename. Standard Block unpack/regroup preserves all other executable
values and connections; an earlier attempt to edit sealed graph contents was
correctly rejected by the graph-hash validator and was never imported or run.

Actual built-client import, Arrange, Save and reload preserve the entire new
snapshot, exact executable nodes/paths and all 41 prior saved workflow bodies.
The independent Linux guard admitted the unchanged memory/disk/swap/time floors
before the sole genuine UI Run. Rendering and creative acceptance remain pending.
No further variants, extra H3 runs, downloads, model retirements or public uploads
are part of this pilot. The six approved clips and eight rejected paper clips
remain preserved. Technical audits and human review will be recorded separately.

The sole photographic pilot has now completed in 35m46s without a resource stop.
Exact task/attempt/source/adapter/owner/cache/source-file/durable/served joins and
full decoding pass: 124 frames at 24 FPS, 832Ã—480, stereo 32 kHz. Whole built-editor
muted playback presents all 124 frames with zero drops/errors. Normal owned
ModelSetup and ImageLoad release preserves all 98 app files and durable media;
cache is empty and temporary offload is absent. A private auditor's initial raw
cache URL failed on a slash-bearing Block owner; matching the application's
existing segment encoding fixed the helper, with the failed partial evidence
retained and every identity/hash check unchanged. No application fix or rerun
was required.

The private review adds the genuine photographic pilot as awaiting user review,
preserving all 14 previous clip objects, 20 video references, comparisons and
six accepted/eight rejected verdicts. Cycling and foreground parallax are visible;
camera angle remains mostly side-on and the ending crops the wheels. The next
step is the user's visual verdict before further recipe replacements. This is
candidate delivery, not accepted-template, listening, publication or whole-app
signoff. The source brief is versioned separately from all private media/evidence.

## Photographic keyframe and mixed-reference replacements authorized

The user accepts the photographic cyclist pilot and authorizes both replacements
and reusable templates from good outputs. Seven clips are now visually accepted;
all eight paper clips remain rejected. New keyframes are the exact first and
last frames of that approved photographic pilot; the mixed recipe uses its
opening image, video motion and original audio together. Deterministic extraction
preserves all 98 existing app files. Old workflows and media remain untouched.

The keyframe replacement passes ordinary import, Arrange, complete Save/reload
and exact executable-graph comparison, preserving all 43 prior workflow bodies.
Its sole UI render is running on Linux, with the independent resource guard
ready before allocation. The 15-node mixed recipe is prepared, not submitted;
it waits for first-case review, release and fresh idle admission. All fixed model,
adapter, sampler and memory/disk/swap/time limits remain unchanged. Templates
will be checked as ordinary portable packages with empty required media, after
their real outputs are reviewed. No extra H3 or unaffected-model/gate reruns.

The photographic keyframe replacement completed in 25m09s. Strict actual
task/attempt/source/adapter/owner/cache/source/durable/served-byte checks pass,
with 124 frames at 24 FPS, 832Ã—480 and stereo 32 kHz. Full built-editor muted
playback has zero drops/errors. Samples show realistic continuous cycling and
foreground parallax, with framing matching the supplied wider opening and closer
ending. Normal scoped release preserves all 104 app files and durable media.
The output is delivered privately as awaiting user visual review; all 15 earlier
clip objects, 21 video references/comparisons and seven accepted/eight rejected
human verdicts remain exact.

The new portable photographic keyframe package has two empty required media
fields, exact sealed definitions/form/parser roundtrip and exact private media
rebinding to the admitted native graph. Actual import/Arrange/Save/reload under
**Template â€” Pika photographic cyclist keyframes** preserves its complete
snapshot/executable graph and all 45 prior workflow bodies. Empty inputs block
Run before and after reload; no graph is submitted. The mixed-reference recipe
also passes complete Save/reload while preserving all 47 prior workflow bodies
and now runs serially under the same Linux guard. Its three-input portable
package passes static checks but awaits output quality and native/UI closeout.

The mixed-reference replacement also completed, in 48m50s, without a resource
stop. All 15 submitted nodes appear in the actual resolved-execution record,
including all three reference assemblers, video trim, audio trim and media loads.
Strict task/source/adapter/owner/media joins and full decoding pass: 124 frames
at 24 FPS, 832Ã—480, stereo 32 kHz. Complete built-editor muted playback has zero
drops/media errors. Samples show a realistic recognizable cyclist, repeated
pedalling and foreground parallax; the camera remains mostly side-on and some
ending wheel crop remains. The new clip is delivered privately pending human
visual/audio review, preserving all 16 prior clip objects, 22 video references
and seven accepted/eight rejected verdicts. Original paper outputs remain held.

Normal ModelSetup plus independent image/video/audio root releases preserve all
112 app files and durable media, with empty cache and no temporary offload.
The mixed portable source and actual **Template â€” Pika photographic image video
audio references** pass complete parser/form/sealed-definition/private native
rebind checks and ordinary import/Arrange/Save/reload. All three required media
inputs are blank; Run stays blocked before/after reload, no graph is submitted,
and all 49 prior workflow bodies remain exact. Both new templates are delivered;
no extra H3, model download/retirement, install or public upload was performed.

The run also exposes a separate existing input-preview defect: requesting an
audio preview from an AV input returns 422, and some loader players request raw
backend paths as relative URLs (404). Actual input consumption and generated
output playback succeed. Preserve those console/HTTP failures and keep this
visible as remaining app work; do not reinterpret output playback as proof that
input previews are correct. No application patch or additional model run was
performed for that diagnosis. Current delivery is the two photographic examples
and reusable templates, with human review and this preview defect still explicit.

## 3 November â€” input-preview repair delivered

The separately diagnosed input-preview hold above is now closed. Selected video
and audio paths resolve through the managed preview endpoint, including Windows
paths, scalar and picker-array selections. Clearing an input clears its stale
preview without changing saved parameters during display. Converted audio
previews/exports can extract a real AV soundtrack; ordinary probes, original-file
delivery and silent-video checks retain their strict types and file boundaries.

Backend core `e6eaffb3` and paired delivery `ce0fb40c`, client product `75080b73`
and measured-budget follow-up `1cd170f2` are in the same two PR7s. The dedicated
video app was refreshed with the exact 119-file bundle, including 118 served
assets. The five dependent ledgers changed hashes only; semantic rows and
qualification remain unchanged. Runtime packages and configuration were retained.

Validation includes 43 focused backend tests/3 subtests, 28 media unit tests and
three focused browser cases. The complete backend gate passes: lint, dependency
check, preflight, **8,986 passed / 763 skipped / 13,185 subtests**. Client
`npm run check` passes in CI at `1cd170f2`; the full local browser gate passes
**16 shared-control + 325 editor cases**. The initial local size-gate failure and
its measured-only correction remain recorded; dependencies, loading boundaries,
four startup/90 deferred assets and prior budget headroom are retained.

Actual built-app verification imports only two ordinary media loaders and reuses
the accepted cyclist MP4. Video plays all 124 frames at 1x with zero drops/errors;
both AV soundtrack players complete. Normal Save/reload, clear/Save/reload with
Run blocked, and array-valued rebind/Save/reload pass with correct managed URLs.
No graph POST or large-model run occurs. A saved **QA â€” Retained video/audio input
previews** workflow provides the same small test without model execution.

Final preservation checks retain all 114 earlier app files except the expected
supervisor worker/timestamp fields, all 51 earlier saved workflow bodies, both
source/saved templates, and all 17 review clips/23 video URLs. Only three owned
QA workflow records and one soundtrack preview cache are added. Review text now
records the repair, with original page/manifest backups; all clip objects,
comparisons and seven accepted/eight rejected/two pending verdicts remain exact.
The original LTX/Wan worker is untouched. No model download, retirement, runtime
installation, public upload, new rendering or qualification promotion occurred.
Human review/listening/publication and deferred model work retain their prior
scope; this repair does not extend their acceptance.

## 3 November â€” photographic approval and release-tooling closeout

The user now explicitly accepts **both photographic replacements visually**.
The private review records nine accepted clips and eight rejected paper-boat
clips, preserving all 17 media identities and comparisons. Both portable/saved
templates remain intact. Audio listening and publication were not inferred.
The task ledger and handoff now close the stale input-preview and visual-review
items; historical checkpoints remain preserved.

Gallery diagnosis establishes a real decoder-producer portability defect:
identical decoded LTX frame rows have different raw `framemd5` hashes solely
because of the FFmpeg software header. Client `a0145878` retains the actual raw
digest and strictly checks reviewed historical-header variants without changing
frame data, media, PCM, locks or provenance. All 60 Gallery harness tests and
the complete client check pass. The real report improves from 88 failures to
73; this is not a Gallery pass.

Client `66b5152` fixes a second release-tooling defect: removed modules leaving
only empty/cache directories no longer prevent static bundled-node inspection.
Incomplete source modules still fail closed. The original failure and 22 passing
focused tests are retained; its complete client gate also passes. Downstream reports now enumerate 90 templates, 203 workflows and
60 resource recipes. Before retained-proof recovery, only one template/workflow
is release-eligible; supported physical profile evidence is absent. These
denominators remain unchanged, with a blocked release result.

Two recovered FLUX3 proofs pass the full V2 provenance validator. Ordinary
registration correctly refuses their experimental canonical routes; the existing
explicit blocked-probe path retains the still-current reference-edit receipt
without promoting its route. The T2I receipt's creator lock is now stale and is
held as historical. No model is rerun or lock rewritten for this recovery.

See [the current closeout](RELEASE-CLOSEOUT-20261103.md) for the exact remaining
work and constraints. No extra H3 render, download, deletion, installation or
public upload occurs during this tooling/status work. App source and approved
media remain unchanged.

## 3 November — Windows upload diagnosis and single authorized H3 comparison

Native Windows now runs `fc0c9fcc` / client product `75080b73`; a separate
read-only audit verifies all 105 original saved workflows, 369 prior app files
and 119 disk/118 served assets. Preserve the earlier cutover observer hold;
current deployment acceptance is an independent observation.

Actual Audio.Load MP4 upload exposed a server kind-check defect. Fix `a43e5e98`
extracts a soundtrack only for explicit audio uploads containing real audio,
preserves uploaded bytes and rejects silent video/renamed images. The focused
46-test/five-subtest media check passes. A full backend gate has one independent
reporting failure among 8,989 passing tests: ten deliberately unmapped canonical
routes were not representable in the reconciliation report. The narrow report
fix records them as held, without weakening qualification. Full verification
and Windows deployment of the upload fix remain pending at this checkpoint.

The user lifts the extra-H3 restriction for **one** photographic keyframe
comparison. Task `DUC96v3XXjU-` is genuinely submitted through the current built
Linux UI at 13:06 UTC, after resource admission and exact Save/reload/request
checks. It uses the accepted Pika prompt and identical first/last photos,
immutable cached H3 weights and native geometry/sampling. No other H3 render,
download or media publication is authorized. Approved clips/templates and
original rejected media remain unchanged; native output/review are pending.
