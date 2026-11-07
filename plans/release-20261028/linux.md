# Linux lane: bounded backend and native acceptance

Use the [shared plan](README.md), [review](review-and-feedback.md) and
[deferred backlog](remaining-release.md). Deadline: **28 October 2026, 18:00 IST**.
Deliver source handoff by **13:00 IST**; shared source freeze is **14:00 IST**.
Do not move these later because an old handoff proposes a different schedule.

The purpose of a separate Linux chat is real parallelism: available Linux model
capacity and backend/service checks can run while Windows handles setup, UI and
constrained-memory behavior. Linux success never substitutes for Windows proof.

## L0 — source parity, current machine and preserved state

**Budget: 30–45 minutes, plus diagnosis if the host is unavailable.**

Start with both remote working branches updated by the clean-baseline review.
Run branch creation only in a separate non-served checkout/worktree. A clean tree
can still be serving the app: fetch/inspect first and defer any served-source
switch until the admitted idle stop and preservation capture.

```text
git status --short
git fetch origin work/manual-testing-20261020
git switch -c codex/release-linux-20261028 origin/work/manual-testing-20261020
git rev-parse HEAD
```

Run in each clean repository. If the branch exists, inspect and resume it; preserve
dirty code before any switch. Do not reset the current Linux app checkout merely
to obtain the plan. The plan's reviewed application pair is BE `ae09c201` / FE
`fd3d3dac`; later plan-only commits do not mean the running server already changed.
The last retained Linux app pair was older and needs deliberate source parity.

Read AGENTS, CONTRIBUTING, SECURITY, the complete mirrored cluster lessons,
`docs/custom-nodes.md`, `docs/service-prototyping.md` and workbench acceptance.
Resolve current host/process/launcher/runtime, app/control origins, current/queued
tasks, downloads, installed model revisions, data/cache/offload roots and available
memory/disk. Do not assume a historical PID, forwarded port or receipt count.

Keep app and supervisor on loopback. If accessed from Windows, both authenticated
forwards must lead to the same remote app/control pair; a working app forward alone
does not prove that Stop/restart is safe. A connection reset does not authorize
starting another worker or repeating an unknown task.

Before an idle source refresh, privately preserve source state, workflows/User
Nodes, inputs/outputs, exact models, runtime/launcher/config and prior evidence.
Obtain the actual machine-local launcher/environment bindings, supervisor/worker
ownership, app/control ports, runtime contract, configuration/data roots, latest
preservation baseline and existing lifecycle procedure privately. Published source
alone does not reconstruct the specialist runtime. If any required item cannot be
established, stop before source/process changes and name it; do not substitute a
generic installer or guessed startup command.

Use that existing guarded ordinary refresh procedure. Deploy the admitted backend
and its verified client bundle; verify the exact client-source binding, emitted
and served assets, process-start source and unchanged runtime. Rebuild only when
the selected source changed or the accepted bundle is unavailable, with the
integrator owning that bundle.
Do not reinstall ROCm or apply generic `uv sync --extra cpu/cuda` to that environment.

**Important prior boundaries:** retained Linux records include allowed runtime
observation growth and an unresolved raw-preservation hold. Preserve both; do not
blanket-normalize JSON or silently relabel the old hold as a pass. Fresh current
inventory is required. Scope cleanup to owned tasks/caches only.

## L1 — one actual custom-model, Block and service journey

**Budget: 30-minute prerequisite selection, then 2–3 hours. Type: integrated
acceptance gap, with backend fixes only for diagnosed behavior.**

**Where:** BE `modiff/custom_extension_source.py`, `custom_extensions.py`,
`custom_extension_api.py`, `custom_modular_extension.py`, `custom_modular_models.py`,
`custom_modular_inspection.py`, `service.py`, `service_api.py`, `service_package.py`;
`examples/custom_nodes/ModularImageReconstruction/`; FE normal custom-node,
Block/persistence and Service Export UI.

The existing model-consuming example calls a connected **AutoencoderKL** to
reconstruct an image. It requires a compatible cached component. The existing
FLUX3 VAE must not be assumed compatible. The source is a runnable starting point,
not a new integration to invent. See the example README and actual `block.py`.

1. Select one approved cached compatible model/component and permitted input.
   Record immutable model/source identity and installed dependency contract.
2. For a Hub-import claim, identify and inspect an actual compatible custom-Block
   source and its immutable commit. A model repository alone is not custom-node
   code. The inspected local records did not establish a ready-to-use immutable
   Hub source pin. If none can be admitted in the 30-minute slot, record this
   prerequisite; a local example does not close the Hub-specific acceptance.
3. Use the normal Add/Load/Reload action after source/dependency review. The action
   is code consent; discovery or workflow import is not. Do not automatically
   install dependencies or execute arbitrary repository Python.
4. Connect model components → Models, approved input → Image, output → Preview.
   Put the supported graph in a Block and expose Amount. Run with a nonzero Amount;
   inspect actual model execution and decoded result, not only a successful status.
5. Save/reopen, change Amount and confirm a fresh affected computation while
   retaining the compatible model owner. Save as User Node and reinsert; compare
   effective values/graph. At idle, restart and verify the same graph/source/model
   identity. Exercise actual visible controls rather than injected stores.
6. Export the actual API graph and Service package through the normal UI. Bind
   only declared Amount/input/output fields. Use the existing service CLI against
   the local backend, with an explicit bound. Match returned named outputs to its
   actual task and durable media. Preserve the source manifest/trust boundary.
7. Release only owned caches and inspect output persistence, actual manager state,
   app responsiveness and visible release acknowledgment.

The CLI accepts these arguments. Define `MODIFF_PYTHON` as the verified interpreter
and `MODIFF_ORIGIN`/`CASE_DIR` as the observed local origin/new private case directory.
The examples below use the repository wrapper with the interpreter supplied as
its command argument. A private managed-overlay launcher can have a different
argument contract; inspect and reproduce that actual launcher rather than assuming
this wrapper reconstructs its environment. The graph/inputs/bindings are actual exports and values,
not fixtures to fabricate. Output paths must be new:

```bash
./scripts/with-runtime-env.sh "$MODIFF_PYTHON" -m modiff.service inspect "$CASE_DIR/api-graph.json" \
  --server "$MODIFF_ORIGIN" --output "$CASE_DIR/service-inspection.json"
./scripts/with-runtime-env.sh "$MODIFF_PYTHON" -m modiff.service build "$CASE_DIR/api-graph.json" \
  --interface "$CASE_DIR/interface.json" --server "$MODIFF_ORIGIN" \
  --output "$CASE_DIR/service-package.json"
./scripts/with-runtime-env.sh "$MODIFF_PYTHON" -m modiff.service run "$CASE_DIR/service-package.json" \
  --inputs "$CASE_DIR/inputs.json" --server "$MODIFF_ORIGIN" --timeout 900 \
  --output "$CASE_DIR/service-result.json"
```

`--timeout` does not cancel the task. It reports its identity; inspect and use the
normal Stop path before any retry. CLI service invocation is the product's real
service route, not a substitute hidden browser submission. The service remains a
local MoDiff graph; standalone Diffusers-Python export is not implemented here.

**Focused checks if corresponding code changes:**

```text
python -m pytest -q tests/test_custom_extensions.py tests/test_custom_extension_source.py tests/test_custom_modular_identity.py tests/test_service_package.py tests/test_service_modular_fields.py
```

Use an isolated reviewed test environment; do not point these at live app data or
assume the active optional-runtime environment is the baseline test environment.

**Exit:** a real model-backed edited/restarted graph and callable service with
correct named outputs, or a concrete retained prerequisite/failure. Local-only,
Linux-only and model-free proofs remain separately labeled.

## L2 — the two remaining adapter-order controls

**Budget: 1.5–3 hours including capture/recovery. Type: finite native acceptance.**

Twelve finite executions are already retained: ten original cases plus NASA and
Monet scale-zero controls. Do not repeat them just to refresh timestamps. NASA
style remains unconfirmed; Monet's observed difference is narrow. None grants
publication approval.

Use the retained exact control lock and original permitted input/media inventory.
Its SHA-256 is
`78d77b2ee6b4cbefddbb26a33d473d605965ccfe4fff93314488cb9337d574a0`.
The private relative record is
`final-release-20261027/linux/four-lora-controls-preparation-20261027/control-locks-and-observation-plan.json`.
Recover it from the originating private evidence store if absent on Linux. Git
contains this plan, not private inputs or raw receipts. Do not reconstruct absent
model revisions, adapter hashes, prompts or task evidence from this prose.

1. Deploy the reviewed applied-adapter producer/parser before these runs. Old
   receipts cannot retroactively acquire native activation observations.
2. **Joint:** actual retained T05 NASA recipe, NASA scale 0.75 followed by Outpaint
   scale 1.5; 512×512, 16 steps, seed 481005, BF16/FP8r, CPU loading, no offload,
   Auto false. Exact pinned artifacts and all other values come from the lock.
3. **Reverse:** same owned graph/IDs/settings, retaining first-run caches; reverse
   only the adapter chain. Require actual changed application identity/order and
   fresh downstream execution. Timing or a plausible image alone cannot prove it.
4. Retain original graph/creator/output packages, native task/attempt, resolved
   inputs, applied adapter observations, model/runtime/source identities, decoded
   bytes and actual visual assessment. Verify no stale or duplicate adapters.

Each model task has a 900-second whole-task bound and normal Stop available. One
visible ordinary Run at a time; no hidden POST/store mutation to bypass UI behavior.
Do not resubmit when task outcome is unknown. Capture failure before cleanup.

**Where if a defect appears:** `BE/modiff/applied_adapter_provenance.py`,
`native_disk_lora.py`, `modules/ModularDiffusers/loaders.py`; FE
`scripts/applied-adapter-provenance.mjs`, `live-proof-provenance.mjs` and
`src/studio/resolvedExecutionInputs.ts`. These are shared files: report a precise
reproduction and obtain explicit single-editor ownership from the integrator.

**Checks if changed:** `tests/test_applied_adapter_provenance.py`, affected native
adapter tests and client receipt/provenance tests. Keep the strict recorder's
single-owner scope; do not loosen it for an unrelated multi-owner graph.

**Exit:** original joint and reversed-order observations correctly match actual
application state and outputs, with any visual failure retained. This establishes
those controls, not NASA's style efficacy or public-template quality.

## L3 — one adapter-bearing Group Disk lifecycle and truthful Auto refusal

**Budget: 1–2.5 hours. Type: unclosed native resource/lifecycle acceptance.**

The old disk-capacity obstacle was resolved by an already authorized, exact unused
model-cache deletion. Historical free space was 19,397,165,056 bytes against the
then required payload-plus-reserve floor of 17,454,649,984 bytes. The small margin
must be checked again on the actual writable offload filesystem. Do not schedule
more cleanup automatically or assume the old hold is still current.

1. Obtain fresh capacity/artifact admission. If refused, record current numbers
   and stop that row; do not erase important files or bypass the floor.
2. Save a separate owned copy of the retained approved Outpaint variant, preserving
   exact input, seed, scale and other settings. Explicitly select `group_disk` with
   AutoOffload true. It is a separately scoped resource variant, not the public
   no-offload recipe.
   Recover its original saved graph/creator package, input bytes, model/adapter
   pins, seed, scale and resource settings in the private handoff. The adapter-order
   lock alone is not this disk-case inventory. Missing originals block this row;
   do not reconstruct them from prose or borrow an unrelated run's output.
3. Run with a 900-second bound. Observe component/load-specific offload directories,
   actual output and applied-adapter state. Distinct owners/components must not
   reuse equal-shaped wrong weights or stale group files.
4. Release the owned graph normally; inspect registry/manager ownership and actual
   file cleanup. Retain shared owners and durable outputs. Empty cache counts alone
   do not prove memory/file cleanup. Check the updated Linux Setup release indicator;
   the old source retained a stale enabled indicator after backend cleanup.
5. Inspect the existing Linux FLUX3 Auto restriction through the normal UI without
   submitting an unsupported model run. A truthful refusal on unsupported hardware
   is the expected result; do not add Linux Auto authority merely to make it green.

**Where:** `BE/modiff/diffusers_offload.py`, disk-offload/lifecycle modules,
`workflow_auto_lifecycle.py`, `flux3_auto_resource.py`, native loader hooks. Required
focused regression entry points include `tests/test_diffusers_disk_offload_lifecycle.py`
and `tests/test_pipeline_disk_offload_isolation.py` if those paths change.

## Code ownership, handoff and stop conditions

Default to evidence-only execution until a real defect is diagnosed. This lane
may own the L1 custom/service modules, selected example and corresponding tests.
Do not edit shared executor, Auto/resource, loader/provenance, FE source, lockfiles,
generated ledgers or web assets without recorded ownership transfer. Windows is
the sole final integrator. Read-only review is always useful during waits.

By 13:00, push the Linux branch normally and supply:

- Exact BE/FE base and head commits, changed files and ordered commit list.
- Reproduction, causal fix, commands/results, failed/skipped checks and source
  impact; no full-suite claims from focused tests.
- For each native case: source/runtime/model/task/output identities, consumed
  values, proof level, visible result and retained quality/capacity holds.
- Preservation and cleanup findings, active task/queue state and actual deployed
  pair. Keep private logs/media out of public Git.
- Remaining prerequisites, estimated fix time and whether final Windows checks
  must repeat because a shared execution path changed.

Stop starting new cases at the handoff cutoff. Diagnose two same-category failures
instead of a third blind retry. Missing reviewed Hub source, incompatible cached
VAE, unreachable host, absent normal browser access, missing private locks or
insufficient disk are specific blockers. Report them immediately and continue only
independent admitted work. Do not hide them behind a substituted model-free demo.

Cap this lane at eight hours and the 13:00 source handoff, whichever comes first:
L1, then L2, then L3 if time and capacity permit. The upper individual estimates
can sum to almost ten hours before fixes, so all three are not guaranteed. Leave
the last row explicitly unrun rather than overrun integration. Six to eight hours
is the target with cached compatible artifacts and healthy deployment. There is
no promise to finish Gallery/canonical/all-model work in this budget, and no
new-media publication is authorized by this plan.

## Prompt for the Linux chat

> Execute the bounded Linux lane in `plans/release-20261028/linux.md`, reading
> its README, review and required repository guides first. My delivery cutoff is
> 28 October 2026 at 18:00 IST, with 18:00–21:00 reserved for manual testing.
> Start from the pushed `work/manual-testing-20261020` pair in both repositories;
> preserve local work and use dedicated Linux branches. Establish current host,
> app/control ports, runtime, source, queues, data and offload capacity before
> changing the application. Tell me the exact task before acting. Preserve
> existing workflows, outputs, inputs, model caches, runtime and original failure
> evidence. Execute L1's real custom-model/Block/service journey, then the two
> remaining locked joint/reverse adapter controls and one Group Disk lifecycle
> when their prerequisites pass. Do not repeat completed scale-zero or original
> cases merely to refresh evidence. Admit concrete custom source through the
> normal deliberate Add/Load/Reload flow; never guess a Hub revision or substitute
> a model-free result. Use normal visible UI runs for model cases and the existing
> service CLI for the service case. Timeouts do not cancel unknown tasks.
> Own only the listed custom/service files and focused tests for diagnosed bugs;
> coordinate shared-file changes with the Windows integrator. Push a compact
> source/evidence handoff by 13:00 IST; source freeze is 14:00. Do not run duplicate
> broad gates, expand the catalog, weaken validators, change creator defaults or
> publish media. Report concrete blockers early and continue independent work
> within this finite plan. Routine actions already authorized by this execution
> request do not need another permission question.
