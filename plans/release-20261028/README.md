# Release preparation for 28 October 2026

This plan was requested for execution in separate Windows and Linux chats. The
original deadline was **28 October 2026, 18:00 IST (12:30 UTC)** for a deployed,
testable candidate, with 18:00–21:00 IST reserved for manual testing. On 28 October,
the maintainer explicitly approved continuing unfinished work past 18:00 and
delaying manual testing. The schedule and cutoff references below retain the
original planning checkpoints; that manual-testing window is now superseded.
Provide a freshly verified deployment and handoff before beginning the maintainer's
2–3 hours of manual testing. This extension does not close unavailable-hardware,
catalog-evidence or publication-approval requirements.
The review started around 01:00 IST, leaving about 17 hours under the original
cutoff, not two working days.

This is a plan, not a statement that its acceptance checks have passed. The
canonical copy is in the backend repository; the client carries an identical copy
so either checkout can supply the handoff. Keep the five files together.

## Read and use these files

1. [Review and feedback chronology](review-and-feedback.md): what was inspected,
   what is already implemented, actual gaps, and mistakes to avoid.
2. [Windows execution plan](windows.md): setup, constrained-memory generation,
   frontend acceptance, final integration and deployment. Includes a chat prompt.
3. [Linux execution plan](linux.md): backend/service/custom-node acceptance and
   the bounded remaining Linux cases. Includes a chat prompt.
4. [Deferred release work](remaining-release.md): every broader open workstream,
   its implementation boundary and prerequisites. Deferred does not mean done.

`BE/` and `FE/` below mean repository-relative paths in MoDiff and MoDiff-client.
They are not literal directories to create. No private logs, credentials, model
weights, user workflows or generated media belong in these plan commits.

## Reviewed starting point and publication

| Repository | Reviewed application source                | Starting branch                |
| ---------- | ------------------------------------------ | ------------------------------ |
| Backend    | `ae09c2014f01198d359a1310d63a1d8d619796be` | `work/manual-testing-20261020` |
| Client     | `fd3d3dacbcda0b4f44ff23497276658f0233eea8` | `work/manual-testing-20261020` |

Both reviewed heads were pushed to that branch before authoring this plan. The
existing `codex/windows-delivery-20261026` remote branches also contain that code
pair. The subsequent plan commits contain documentation only. Read their actual
Git heads when starting; do not invent a future commit ID or treat branch names as
immutable execution evidence. The deployed Windows application still has the
reviewed application pair until the Windows owner deliberately deploys another.

Both active primary checkouts were clean. Historical worktrees retain preserved
overlays already incorporated or superseded; they are not the starting point for
new work. Their exact disposition is in the review. No new actionable runtime
defect was established by the clean-baseline review, so no speculative runtime
patch was made. One confirmed documentation defect remains: ordinary installation
instructions still select the older `fix-ui-ux-issues` branch. W1 corrects it.

For an existing clean checkout, update its remote references, inspect the branch
and start the lane branch from the published working branch:

```text
git status --short
git fetch origin work/manual-testing-20261020
git log -1 origin/work/manual-testing-20261020
```

Run these in each repository. Preserve any dirty local work first; do not reset,
clean or overwrite an existing Linux checkout. Use the lane-specific branch
instructions in the two execution plans. Never run a generic CPU/CUDA environment
sync on an existing specialist ROCm installation.

## What can be delivered by the deadline

The achievable target is a **reviewed Windows update with an explicitly bounded
support statement**, tested installation instructions, representative image
prototyping, preserved user data, and a usable manual-test checklist. Available
Linux results strengthen the shared implementation and retain their Linux scope.

The original expanded release means 80 public templates, 203 ordinary workflows,
58 declared resource tuples and the supported hardware predicates. That is not a
credible promise within this window. Apple MPS has no available host; general
canonical evidence collection remains incomplete; Gallery has unresolved evidence
and quality issues. The whole-app acceptance gate must retain those failures.
Do not change its denominator, skip predicates silently or call this candidate
the completed all-catalog/all-hardware release.

If the maintainer requires the original full-release claim at 21:00, the present
plan has a **no-go** until those requirements are actually met. A date cannot turn
an untested route into a supported one. A bounded Windows release must state its
tested routes and remaining limitations plainly.

## Priorities and ownership

| Task                                                                   | Owner                                                  |                                   Estimate | Deadline importance                                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------ | -----------------------------------------: | ------------------------------------------------------------------------------------- |
| W0/L0: establish exact source, app, evidence and ownership             | Both, separate hosts                                   |                             30–45 min each | Required before mutations                                                             |
| W1: correct paired setup/distribution guidance                         | Windows                                                |                                  45–90 min | Required for delivery                                                                 |
| W2: untouched Windows CUDA setup, base libraries, no-Triton eager path | Windows                                                |              1.5–3 h, downloads additional | Required for installation claim                                                       |
| W3: Qwen cold/repeated Auto; distinguish the FLUX3 capacity hold       | Windows; Linux assists only by assigned code ownership |                          3–6 h if admitted | Required to close the original memory feedback                                        |
| W4: representative Block/model/port/persistence journey                | Windows                                                |   2–4 h, sharing W3 model runs where valid | Required for prototyping claim                                                        |
| W5: refresh/recovery and optional installation status                  | Windows                                                | 1–2 h; full optional reinstall 1–2 h extra | Required affected behavior; optional reinstall conditional                            |
| L1: model-backed custom Block and service journey                      | Linux                                                  |                                      2–4 h | Required for the service/custom integration claim; Windows persistence still needs W4 |
| L2: remaining joint/reverse adapter controls                           | Linux                                                  |   1.5–3 h including deployment and capture | Bounded follow-up after critical L1 blockers                                          |
| L3: disk-offload/lifecycle and Linux Auto refusal                      | Linux                                                  |                                    1–2.5 h | Bounded follow-up if fresh capacity admits it                                         |
| R1: integrate, full applicable software gates, deploy and handoff      | Windows integrator                                     |           Reserve 3 h plus 1 h contingency | Cannot be traded for more catalog runs                                                |

The rows are not an additive elapsed-time promise: several observations can be
collected in one genuine journey, and Windows/Linux run independently. Plan for
about **20–30 person-hours across the two lanes**, attempting to fit the critical
slice into **12–16 elapsed hours**. Native fit, download time and newly diagnosed
defects can exceed this. The earlier 15–25-hour estimate covered representative
feedback closure, not every model transition or the expanded catalog. Re-estimate
at the checkpoints below instead of promising completion from feature presence.

## Schedule, all times IST on 28 October

| Time                   | Windows lane                                                | Linux lane                                                | Required decision/result                                |
| ---------------------- | ----------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| By 02:00               | W0, W1; start isolated setup                                | L0; obtain exact pair and prerequisites                   | Named owners, safe app/data boundaries, reachable hosts |
| 02:00–04:00            | W2; establish W3 first-route admission                      | Begin L1 custom/service journey                           | First install/runtime and model-admission result        |
| 04:00 checkpoint       | Classify a real failure before further attempts             | Report genuine code failure versus fixture/access failure | No blind rerun of the same failed setup or recipe       |
| 04:00–08:00            | W3 causal fix if needed; cold/warm sequence                 | Finish L1; focused backend fixes with assigned files      | Native evidence or an exact unresolved blocker          |
| 08:00–11:00            | W4 integrated UI journey; W5 refresh                        | L2, then L3 only if prerequisites pass                    | Stable candidate changes; retained failures             |
| 11:00–12:00            | Finish required targeted checks and release notes           | Finish current bounded case; no broad campaign            | List passed, failed, blocked and unrun separately       |
| 12:00 scope checkpoint | Choose deliverable claims from evidence                     | Send ordered commit/validation handoff                    | No new features or catalog expansion                    |
| By 13:00               | Receive Linux patches and inspect conflicts                 | Last source handoff; stop source mutations                | Both lanes agree the exact pair to integrate            |
| 13:00–14:00            | Integrate source, regenerate affected contracts/build       | Independent review of merged backend changes              | Freeze application source by 14:00                      |
| 14:00–16:00            | Required final gates and emitted-bundle checks              | Run only assigned isolated checks on the frozen pair      | No competing inference or editing served source         |
| 16:00–17:00            | Deploy exact pair at idle; affected native checks           | Close evidence and cleanup only owned jobs                | Actual served candidate verified                        |
| 17:00–18:00            | Buffer, final preservation and manual-test handoff          | No new runs that jeopardize handoff                       | URL/build IDs/checklist; idle app ready at 18:00        |
| 18:00–21:00            | Maintainer manual testing; targeted fixes only by agreement | Available for diagnosed issues                            | No autonomous broad testing/restarts                    |

If work starts late, shorten deferred scope, not preservation, required checks or
manual testing. By 12:00, an unresolved first image, save/data-loss issue or
repeated-run crash is a release blocker. Revert only the lane's diagnosed change
through a normal reviewed commit when appropriate; preserve unrelated changes.
Never stop an unknown active job to meet a calendar time.

## Parallel work contract

- Windows is the sole release integrator and owner of FE product source, setup
  documentation, CI pairing, generated client assets and final Windows deployment.
- Linux initially owns BE service/custom-extension defects: `modiff/service.py`,
  `modiff/service_package.py`, the diagnosed custom-extension modules and their
  focused tests. It owns its private native evidence and Linux app lifecycle.
- Shared critical files (`modiff/server.py`, `NodeBase.py`, memory/offload modules,
  `modules/ModularDiffusers/loaders.py`, receipt producers/parsers, manifests and
  lockfiles) need one explicit owner in the shared handoff before editing. Default
  owner is Windows/integrator. Linux reports a reproduction and proposed patch
  scope; it must not independently rewrite them while Windows diagnoses memory.
- Linux may inspect FE evidence tools but does not edit them concurrently with
  Windows. Transfer a file's ownership explicitly if a receipt defect needs work.
- Each lane uses its own branch and app/test data root. No shared writable runtime,
  user data, model offload directory, qualification registry or browser storage.
- One heavy inference job per host. Avoid running full suites alongside model
  generation on that host. Tests must not instantiate supervisors against live
  operator data. Source/collector bytes remain frozen during captured runs.
- Handoffs contain base/commit IDs for both repos, changed files, diagnosis,
  commands/results, proof level, retained failures, deployment state and remaining
  work. Push ordinary branches; no force push. Only the integrator merges into the
  release candidate. Direct cross-chat messaging requires the user's authorization;
  a pushed handoff can be carried by the user without such authorization.

## Verification budget and final admission

Reuse existing evidence at its original scope. The reviewed pair already passed
8,540 backend tests, 12,978 subtests, 11 static/generated checks, the client quality
gate and 328 browser checks. The latest measured full gate cost was approximately
26 minutes backend + 8.5 minutes client + 33 minutes browser, before integration
and deployment. Do not rerun that campaign merely to inspect documentation.

For an actual change: reproduce narrowly, add the causal regression, run focused
checks, then run the applicable complete CONTRIBUTING gates on the final source.
Update both immutable FE CI backend refs after a coordinated BE code commit. Keep
historical source-audit fixtures separate from the installed runtime. Run any
required active-optional-runtime checks with the supported validated runner;
base-environment skips do not prove optional execution.

`npm run check:acceptance` is the whole-app gate. Record its real result and the
first failing stage; later short-circuited stages are **unrun**, not passed. Do not
repeat a known failing Gallery stage without changed evidence, or silently replace
the whole gate with `npm run check`. For a bounded candidate, independently run
the applicable software and asset checks and disclose the unmet full-release gate.

The 18:00 handoff must include:

1. Verified application URL, exact BE/FE commits, runtime, and served-bundle identity.
2. Changes since the already testable pair, plus known unchanged limitations.
3. Required native results with model/task/settings and actual consumed values.
4. Preservation result for existing workflows, outputs, custom nodes and runtime.
5. A 2–3-hour manual checklist with expected results and recovery instructions.
6. Failed/blocked/unrun items, including the Mac and full-catalog boundaries.

No final source edits after testing without updating the impact assessment and
rerunning the checks affected by those edits. A passing old bundle is not evidence
for a newer deployed one.
