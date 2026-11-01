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
