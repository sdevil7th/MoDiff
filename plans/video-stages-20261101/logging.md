# Video execution logging and postmortem

Use one canonical task ledger and one private evidence root per campaign. This directory contains source-controlled planning data and empty templates only. Actual prompts, local paths, original media, logs, credentials and host/process details belong outside public Git.

## Directory layout

Choose the private root during V0 and record it in the local handoff. Use this layout on the authoring machine and a verified evidence copy on Linux:

```text
campaign/
  baseline/                  # source, runtime, cache and preservation inventory
  decisions.jsonl            # append-only scope, recipe and resource decisions
  events.jsonl               # append-only task/session events and timings
  current.json               # derived current task view, with event pointers
  cases/<case-id>/recipe.json
  cases/<case-id>/<attempt-id>/
    run-record.json
    submitted-graph.json
    original-terminal-response.json
    consumed-inputs.json
    resource-samples.jsonl
    stdout.log
    stderr.log
    inputs/                  # ordered originals and hashes
    outputs/                 # raw video/audio and exported media, hashes
    screenshots/             # actual output preview, stage controls, errors
    media-inspection.json
    verdict.json
  checks/<check-id>/          # command, environment identity, exit and log
  deployments/<pair-id>/      # before/after inventories and served identities
  incidents/<incident-id>/   # original failure, reproduction, diagnosis, fix
  handoff.md
  postmortem.md
```

Do not copy credentials with the evidence. Preserve original byte streams and build derived summaries separately. Use unique attempt IDs and exclusive creation for original receipts/media. A capture error is its own incident; recover the already completed task by exact identity rather than submitting another costly job.

## Task and time records

`tasks.json` is the planning baseline, not an inference log. Allowed status values are `planned`, `in_progress`, `implemented`, `verified`, `blocked` and `deferred`. `verified` requires the task's exit evidence; `implemented` alone is not delivery. Snapshot the original estimate and append changes with a reason rather than rewriting history.

Every start/pause/resume/end event includes:

- UTC timestamp and IST display timestamp; task ID, session ID and event ID.
- Source pair and dirty-diff hash, where applicable; exact runtime/host evidence reference.
- Concrete action, expected outcome and evidence destination.
- Active engineering seconds, download wait seconds, model wait seconds, test wait seconds and blocked seconds as separate counters.
- Whether a wait overlapped active work. Do not add overlapping intervals twice when reporting calendar elapsed time.
- Result, remaining work, revised estimate and the next action.

At session start, reconcile unfinished task IDs with the real queue before editing or stopping anything. At session end, record active work, exact running tasks, durable evidence locations and the safe next operation. Never infer a terminal status from a disconnected browser or SSH connection.

## Model run record

Copy [run-record.template.json](run-record.template.json) for each attempt. Fill fields from real sources; keep unknown values null. Never turn a template field into claimed measured evidence.

Required before dispatch:

1. Immutable recipe ID/version and hash; creative brief and acceptance criteria.
2. Exact exported graph hash, ordered source-media hashes, model/component/adapter revisions and file selections.
3. Runtime/dependency, backend source and served client identity. Record actual source paths only in private evidence.
4. Requested controls: prompts, seed, geometry, legal frames, FPS, precision, quantization, offload, adapter order/scales and any fixed schedule.
5. Current free disk, system/shared memory, OS reserve, task timeout and cancellation policy; queue current/queued owners and download state.
6. Hypothesis/change from the previous attempt. An unchanged failed retry is not a new experiment.

Required after dispatch:

1. Genuine backend task, attempt and workflow identity, exact submitted/consumed values and terminal receipt.
2. Loading, encoding, denoising, decoding and export timings when observable. Record unavailable stage timing as null, not an estimated measurement.
3. Sampled resource peaks and sampling interval/gaps. Keep RAM, GPU shared accounting and swap distinct; sampled peaks are lower bounds.
4. Original raw video/audio and encoded output hashes, exact preview/task match and ordinary browser playback.
5. Measured frame count, dimensions, rational FPS, duration, codec, audio sample count/rate/channels, sync observations and visual review.
6. Separate verdicts for execution, repeatability/cache reuse, visual/motion/audio quality, UI delivery, cleanup, resource qualification and publication approval.
7. Whether post-run cache release preserved outputs and unrelated owners; evidence transfer hash/readback.

A completed HTTP task is not visual acceptance. A playing input or placeholder is not output proof. A second cached result is not fresh computation. An offload success on Linux is not Windows or universal Auto qualification.

## Test and deployment records

Record the exact command, working directory, package/fixture identities, relevant environment variables with secrets omitted, start/end, exit code, pass/fail/skip counts and complete logs. Isolate test data/ports from production. Name harness failures separately but retain their original failing result. Do not label segmented runs as a one-shot full-gate pass.

Use a test-impact table per change: changed contract → affected unit/dispatch tests → affected browser actions → affected model routes. Run focused checks while fixing; full gates at stable checkpoints. Reuse completed native evidence when the changed code did not affect its execution, recording why. A runtime upgrade changes the applicability of older runtime evidence even when the graph looks identical.

Before deployment preserve saved graphs/User Nodes, input/output bytes and indices, queue history, model inventories, approval records, settings, runtime identity and user fields. Restart only verified owned idle processes. After deployment verify source pair, process-start source, served bundle, readiness, output recovery and exact protected file comparisons. Record byte changes separately from inode/mtime changes; do not claim global preservation from a narrower check.

## Incident record

Create an incident as soon as a failure changes the estimate or risks data/qualification. Include:

```text
Incident ID and first observed time:
Task, source pair and runtime:
Expected user behavior:
Actual behavior and original evidence:
Classification: product / harness / source mismatch / resources / access / quality
Impact and affected scope:
Reproduction and controls:
Hypotheses considered and observations that rejected them:
Root cause, or explicitly still unknown:
Fix and why it is shared or model-specific:
Regression evidence:
Unchanged and affected historical evidence:
Active/wait time spent and avoidable repetition:
Remaining risk and next action:
```

Keep failed source snapshots or hashes where needed to reproduce. Do not normalize original JSON, replace missing receipts or alter a prompt/seed without a versioned decision. Preserve logs of dependency-optimizer collisions, wrong API methods, capture errors or unknown task outcomes as such; they are not model-quality failures.

## Milestone postmortem

At each milestone, answer these questions in at most one page before linking detailed incidents:

1. What users can now do, and which exact build proves it.
2. Which task estimates differed materially from actual active and elapsed time, and why.
3. Which existing implementation/evidence was reused and what duplicate work was avoided.
4. Which defects escaped focused tests; which minimal regression now catches them.
5. Whether shared memory, load peaks, download duplication, fixed schedules, source mismatch or UI state caused surprises.
6. Whether any settings, graph data, model caches, outputs or approvals changed unintentionally; exact recovery and remaining preservation holds.
7. Completed, failed, blocked and deferred matrix with one actionable next step per open item.

Do not conclude with a single all-done flag when a family, task, media approval or global release gate remains open. Freeze the handoff and retain its hashes on both machines after a verified transfer.

## 2 November continuation lessons

- Empty dependency arrays in a **blocked** cache-deletion plan do not prove that
  dependencies are absent. Check every blocker first. An isolated app missing its
  workflow-library catalog must receive the exact complete published catalog;
  never create an empty catalog or suppress the guard. Retain the rejected plan,
  exact source hash, actual dependency warning and successful transaction.
- A raw loader enum can leak into older compiled Block definitions even when
  every execution-specification body is unchanged. Compare both proof levels and
  require the strict generated catalog to stay exact; do not regenerate old
  hashes to conceal the change.
- Preserve cold browser failures before rerunning. Here the screenshot remained
  on the lazy editor's loading state and the trace had an unfinished module
  request. The unchanged isolated case passed after the backend CPU gate ended.
  Run the complete relevant browser gate without inflating its timeouts.
- Saved-candidate authoring checks compare actual wire JSON. JavaScript-only
  undefined properties disappear during serialization; transport display metadata
  is not an executable input. Record declared default/unconnected fields and
  intentional new output filenames explicitly rather than asserting that every
  raw object is byte-identical to a hand-authored native request.

## Reference attention and ordinary draft checkpoints

- Preserve the original native OOM and exact trained preprocessing geometry.
  A successful random-weight attention allocation is a numerical/resource probe,
  not pretrained reference generation. Record the deployed callback check and
  actual workflow separately. Fused SDPA non-finites and unavailable CK remain
  failed/unsupported alternatives; do not enable them globally.
- Bind the reviewed reference-only attention choice to component/cache identity.
  Keep completed text/keyframe routes on their existing policy rather than
  silently changing them and repeating hours of unaffected model work.
- New private GPU probes must set the repository working directory before
  importing MoDiff modules. One registered-callback probe failed before compute
  because it ran outside the repository; its failure is retained. The corrected
  callback probe passed without a product-code change.
- Test the ordinary New Workflow path, not only a starter factory. A joint
  producer needs two distinct typed streams connected to the joined exporter.
  An ambiguous union port is not evidence of simultaneous video and audio.
- Use a GPU transport schema for browser-only tests of upstream CPU-offload
  defaults. The initial CPU fixture correctly refused its invalid offload plan;
  retain that failure and do not weaken the exporter device check.
- Client check v20 passed its source/unit/build stages but exceeded only startup
  size caps. Record exact growth (202 bytes each for largest/total startup), keep
  loading membership and independent deferred caps, then freeze the final pair.

## Reference timing and final-browser lessons

- Measure task-specific cost after a loading/attention fix. H3 reference took
  about 437 seconds per evaluation, materially more than text/keyframes. A
  labelled one-evaluation timing admission at trained geometry should precede
  the full quality run. Preserve the underestimated bound and its reason; do
  not restart identical long inference to improve the estimate.
- Extending an observer deadline is not a new model run. Establish successor
  resource sampling before stopping exact owned observers, retain both raw
  capture segments and reconnect to the genuine original session. Record the
  unchanged task/worker/graph and zero additional dispatches or Stop requests.
- Readiness waits and behavioral assertions are separate. The cold lazy editor
  now has a scoped visible-readiness wait; the field/interaction assertion
  limits remain unchanged. Do not add sleeps, retries or a suite-wide timeout
  increase, and retain the original failing trace.
- Migrate historical fixtures explicitly when normal template creation changes.
  One missed legacy LTX test used the new stage-card path; its corrected legacy
  creator retains the original identity/FPS/recovery assertions. New normal-card
  tests remain separate. A focused PASS does not close the full browser gate.
- Test candidate presentation in the actual built app. A private compiler's
  missing layout position caused overlapping nodes; the existing Arrange action
  resolved it without changing executable inputs. Re-read current documents
  before updates so browser-derived display metadata is preserved.
- Compare actual template-lock functions before attributing Gallery failures.
  All 90 locks remain exact across the current Pika changes. The retained 83
  errors and five earlier stage-card lock holds remain failures; do not fix
  them by blindly refreshing hashes or relabelling public clips.
- Hash source-controlled review files using the exact Git/LF byte policy. Pika's
  review was hashed from a CRLF working copy, letting a local gate pass while
  clean Windows/macOS CI checkouts failed six strict catalog tests. Preserve the
  byte validator, bind the existing committed LF bytes and test every evidence
  family for line-ending drift. No review semantics or admission may change as
  part of this correction. Compare committed blobs as well as working files.

## Motion-feedback lessons

- Review the creative brief before expensive inference. Gentle drift and a
  steady camera cannot demonstrate camera travel, strong action or scene cuts.
  Record camera path, subject action and scene transitions as separate checks;
  distinct frames and successful playback are technical evidence only.
- User review can overturn an earlier sampled visual judgment. Preserve the
  old clips and audit records, append the new verdict, and create a versioned
  recipe instead of retroactively labelling the old output satisfactory.
- Pin the prompt-guide source as well as the sampler/model. Use the local
  pipeline's documented syntax and input-image roles; web-product tags or
  invented motion-strength parameters are not a library contract.
- Hold unstarted stale recipes after material feedback. Stop only the exact
  owned dispatcher, keep the active task and resource guard intact, and replace
  obsolete closeout/refresh prerequisites explicitly. Do not fabricate a
  successful serial receipt for cases that were never submitted.
- First test the cheaper short Pika motion candidate, then the longer timed
  two-scene case. Admit duration against measured memory/storage headroom and
  preserve task-specific reference preprocessing. A new full H3 reference run
  is not the first way to validate a prompt improvement.
