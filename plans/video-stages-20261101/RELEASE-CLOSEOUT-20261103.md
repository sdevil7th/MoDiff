# Current video delivery and remaining release work

Updated 3 November 2026. This checkpoint supersedes older pending-work summaries;
historical recipes, verdicts and raw receipts remain unchanged. Source changes
stay in the canonical Windows repositories on `fix-ui-ux-issues`, in the existing
[backend PR7](https://github.com/sdevil7th/MoDiff/pull/7) and
[client PR7](https://github.com/sdevil7th/MoDiff-client/pull/7).

## Delivered

- Shared video stages and selected LTX/Wan/H3/Pika route implementations, exact
  Pika task adapters and fixed sampler, guarded Linux loading/offload, AV export,
  cancellation/recovery, eligible cache reuse and owned release have bounded
  implementation and native evidence. Original source scopes remain explicit.
- The optional-reference Fix-suggestion defect is repaired. Model-backed Wan
  Service CLI verification and the selected continuation handoff are delivered.
- Both photographic replacements are **visually accepted by the user**. The
  private review contains nine accepted clips and eight rejected paper-boat
  clips. All originals remain preserved. Muted playback and visual approval do
  not establish audio listening or grant publication permission.
- **Template â€” Pika photographic cyclist keyframes** and **Template â€” Pika
  photographic image video audio references** are delivered as portable source
  packages and saved workflows. Required media remain blank; ordinary import,
  Arrange, complete Save/reload and blocked Run pass without submitting a model.
- Input-preview bugs are repaired and deployed at backend `ce0fb40c` / client
  product `75080b73`. Actual retained-video/soundtrack playback, clear/rebind and
  Save/reload pass with zero model submissions. The complete backend gate passes
  8,986 tests / 13,185 subtests with 763 skips; the browser gate passes 16 shared
  plus 325 editor cases. All four backend and four client delivery CI jobs pass.

## Release tooling defects corrected

Client `a0145878` corrects decoded-video comparison across reviewed FFmpeg
producers. The retained LTX file has identical decoded frame rows on different
FFmpeg versions; only the `framemd5` software banner changes. The verifier retains
the actual raw digest and reports a narrowly validated historical-banner match.
Frame checksums, timestamps, geometry, stream metadata, PCM hashes, media bytes,
recipe locks and provenance remain strict. No public manifest or media hash was
refreshed. All 60 Gallery harness tests and the complete client check pass.

The real Gallery report drops from **88 to 73 failures**: 13 video digest and two
AV digest false mismatches are resolved. Remaining failures are 63 current
template-lock differences, two prompt-lock differences, two audio-duration
mismatches and six provenance checks across two older Qwen examples.

Client `66b5152` corrects release-contract source inspection when a removed
module leaves an empty directory or only `__pycache__`. Other missing
initializers/source or unresolved registries still fail. A failing reproduction
and 22 passing focused registry/release-contract tests and a complete client check pass. This allows
the downstream release reports to run; it does not qualify any model or device.

## Remaining work, in priority order

| Work                                                  | Approach and completion requirement                                                                                                                                                                                                                                                                                                                                         | Estimate and constraint                                                                                                                                                                              |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Finish tooling verification and status reconciliation | The frozen client gate passes; verify generated reports, retain originals and update the same PRs. No model or app restart is needed for these script/doc changes.                                                                                                                                                                                                          | About 0.5â€“1 hour active, plus the measured gate wait.                                                                                                                                              |
| Recover existing current evidence                     | Compare retained native source/model/input/graph/task/output/measurement identities with the current creator and canonical graph. Use existing recorders only when every required binding passes. Retain stale or incomplete evidence as historical; never refresh locks to make it current.                                                                                | Approximately 2â€“4 hours for an initial inventory; recovery yield is unknown.                                                                                                                       |
| Correct ACE example qualification                     | The old public files are 30s and a 15s tail; current creators promise 75s and a joined 90s track. A retained 75s source exists, but a file alone is not execution proof. Recover matching native receipts first. Otherwise admit the cached 11.1 GB ACE selection on Linux and exercise the actual generation/continuation/loudness/join graph with exact creator settings. | Roughly 1â€“3 hours preparation/analysis plus unmeasured native wait; listening and exact-file publication remain separate. Do not pad/loop old audio or lower promised durations.                   |
| Close old Qwen provenance holds                       | Recover the four original duplicate runs, complete executed model sets and contemporaneous source/runtime evidence. The currently retained public records lack required complete model-set authority. Use fail-closed retained-evidence repair only if those originals exist.                                                                                               | About 1â€“2 hours search/validation; absent originals are a real hold. No large Windows Qwen rerun.                                                                                                  |
| Current Gallery and canonical-workflow qualification  | The current creator lock deliberately includes stage execution selection. Previous-recipe display is not current native proof. Batch missing genuine runs by cached family on Linux after preservation, capacity and exact-input admission; diagnose failures before changing a recipe.                                                                                     | Requires an enumerated campaign and per-family measured timings; 73 failures are not 73 independent code bugs or a short test rerun. Retired/missing models and licensing may block particular rows. |
| Resource and physical hardware qualification          | Recover complete receipts before rerunning. Use genuinely small cached workloads for CPU/NVIDIA where all runtime, model, source and measurement bindings can be captured. Linux/ROCm cannot stand in for NVIDIA or MPS. Keep other Windows applications' RAM allowance.                                                                                                    | Small workload preparation about 1â€“2 hours per available profile; 60 resource recipes need individual evidence. Apple Silicon remains unavailable.                                                 |
| Extra H3 motion variants                              | Still deferred because the user explicitly said not to start extra H3 runs. Existing H3 technical routes and model comparisons remain available. Any authorized photographic replacement needs a new exact recipe, guarded Linux run and fresh review, preserving rejected paper clips.                                                                                     | Prior keyframe run was about 2h18; trained-geometry reference about 6â€“6.5h. These are native waits, not evidence that a new creative result will pass.                                             |
| Human audio and publication decisions                 | Keep visual acceptance separate from listening and exact-file rights/publication. Prepare reviewable exact-file records before requesting approval. No public video upload is authorized by the current visual verdicts.                                                                                                                                                    | Depends on human review; never auto-approve.                                                                                                                                                         |

The newly generated whole-app report covers **90 templates, 203 canonical
workflows and 60 resource recipes**, rather than just the 80 retained Gallery
examples. Before retained-proof recovery, one template/workflow is
release-eligible, 89 templates and 202 workflows lack complete current release
evidence, and zero resource recipes have registered qualifications. CPU, NVIDIA
and Apple MPS have no qualifying physical receipt in that report. These are
registry/report observations, not a claim that every workflow is broken or that
none has ever executed. The available Linux profile remains conditional.

The report generation preserves the advertised denominator and returns
**blocked**. It must not become green by dropping supported hardware, counting
unit/mocked tests as native output, replacing source identities, normalizing old
receipts or treating queued generation as completion. The Apple MPS requirement
cannot be closed on the available Windows/Linux machines. Whole-app completion
is therefore not a credible fixed-hour promise on the current hardware.

## Evidence and lessons

### 3 November follow-up: native Windows upload and one MiniMax comparison

The actual Windows application on port 8194 has been refreshed to backend
`fc0c9fcc` and product `75080b73`. A separate read-only audit verifies the running
worker/source, all 119 disk/118 served bundle assets and preservation of the
105 earlier saved workflows and 369 earlier app files (apart from expected
worker/timestamp fields). The original cutover controller's final observer
exited during a short-lived validator query; that hold remains recorded. The
independent audit establishes current deployment/preservation, not graceful
shutdown of the former worker.

Normal Windows UI upload then exposed a separate defect: Audio.Load advertises
MP4 soundtrack input, but the server rejected a real audio/video MP4 as video.
Backend `a43e5e98` permits soundtrack extraction only for an explicit audio
upload with a real audio stream. Original uploaded bytes remain unchanged;
silent video, renamed images and wrong-kind uploads remain rejected. Focused
media tests pass 46 tests and five subtests. The full backend gate reports
8,989 passed, 762 skipped and 13,187 passing subtests, with one failure in the
offline public-template reconciliation report. The upload fix is not yet
deployed to the Windows worker at this checkpoint.

The report failure is diagnosed: ten experimental public templates explicitly
have no supported canonical workflow, a held release status and historical
`no_canonical_workflow` evidence. The reconciliation tool previously aborted
instead of reporting that state. It now retains a separate
`canonical_workflow_required` bucket, rejects contradictory release/current
authority, counts historical records accurately and schedules no canary for
those entries. No workflow, receipt, lock or eligibility is promoted. Focused
red/green tests cover the original failure and malformed/contradictory records;
the complete backend gate must pass before the next Windows cutover.

The user explicitly authorizes **one MiniMax photographic keyframe comparison**.
The ordinary saved workflow uses the accepted Pika keyframe clip's exact prompt
and opening/ending image bytes. One actual UI Run starts task `DUC96v3XXjU-` on
Linux at 13:06 UTC, with independent resource admission before allocation,
cached immutable weights, 960×544 / 124 frames / 24 FPS and the native H3 sampler.
The running source remains `ce0fb40c`; no Linux restart or source update is made.
Its native geometry/sampler differ from Pika, so this is a same-brief comparison,
not an equal-compute benchmark. The result is pending, with no automatic retry
or extra H3 variants authorized. All nine accepted/eight rejected clips and
both photographic templates remain preserved. Listening/publication stay open.

The append-only private closeout records retain the original Gallery result,
decoder producer comparison, focused red/green logs, complete check logs,
generated release reports, retained-proof recovery attempts and the exact user
visual verdict. Generated reports are local evidence, not public-media or source
authority. Tests and reports do not change the executing application.

Avoid repeating the earlier mistakes: do not rerun large Windows models, repeat
closed Stop/cache/playback proofs, run broad gates for recipe-only changes,
substitute old paper briefs, bypass a stale creator lock, infer audio approval
from a muted player, or call the whole plan complete because local templates are
ready. Preserve approved clips/templates and keep a single source-authoring pair.
