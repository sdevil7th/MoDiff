# Windows lane: implementation, acceptance and final delivery

Follow the [shared deadline/ownership plan](README.md) and
[review findings](review-and-feedback.md). All deadlines are 28 October 2026 IST:
Linux handoff by 13:00, code freeze 14:00, verified deployment by 18:00, manual
testing 18:00–21:00. This lane is the release integrator.

## W0 — establish the candidate and protect the existing app

**Budget: 30–45 minutes.** No product defect is implied by this task.

Start from the clean published working branch in both repositories. Run branch
creation only in a non-served checkout/worktree; a clean tree can still be serving
the application. Defer any served-source switch until an idle, preserved cutover.
Use separate lane branches if no branch with that name exists:

```text
git fetch origin work/manual-testing-20261020
git switch -c codex/release-windows-20261028 origin/work/manual-testing-20261020
git status --short
git rev-parse HEAD
```

For an existing lane branch, inspect it and resume deliberately; do not force
recreate it. Review local AGENTS/CONTRIBUTING/SECURITY and the complete mirrored
cluster lessons. Resolve actual worktree and `.venv` paths, including junctions.

Record privately the serving source pair, worker/supervisor identity, app/control
origins, queues and downloads, runtime/profile, model cache roots, disk, system RAM
and dedicated VRAM. Preserve hashes of existing workflows, reusable Blocks, inputs,
outputs, configuration, launcher/runtime and backend-owned `web/user` files.
Inspect the actual `queued` map as well as the current task. A historical PID,
port number, empty `pending` property or remembered count is not current evidence.

Keep the already testable app available while preparing isolated setup/test work.
Use the existing deployment procedure at an idle boundary. Do not introduce a new
server/controller, change global Python/Node installations or close user apps.

**Exit:** known source/runtime/data boundaries, named Linux owner, safe separate
test storage and a compact before-state. If these cannot be established, diagnose
the prerequisite before running tests against an unknown app.

## W1 — make the advertised installation obtain this update

**Budget: 45–90 minutes. Type: diagnosed documentation/contract defect.**

**Where:** `BE/README.md`, `BE/docs/developer-setup.md`, related current installation
examples such as `BE/docs/windows-fashion-demo.md`; `FE/README.md`,
`FE/CONTRIBUTING.md`, `FE/docs/windows-support.md` and
`FE/scripts/backend-installer-contract.test.mjs`.

Those live instructions still clone/link `fix-ui-ux-issues`, whose remote heads
predate the reviewed pair. A user following them can reproduce older bugs even
though the running local app is updated.

**Approach:**

1. Verify remote refs again. Use the already published delivery branch for the
   interim candidate, or a deliberately published reviewed release ref. Do not
   imply that `main`/`develop` already contain the changes. Never force-push those
   branches to repair documentation.
2. Update both ordinary quick starts and active setup links to the same compatible
   pair. Explain the exact paired commits for reproducibility; a matching branch
   name alone is insufficient. Preserve historical dated evidence as historical.
3. Update the existing command-contract assertions. Keep native `uv`/`npm` commands,
   the same accelerator extra on sync/run, correct Windows Python path, and
   specialist AMD instructions separate. No installer wrapper in the advertised
   direct path; no unnecessary global version/package upgrade.
4. Test the published instructions from the fresh checkout used by W2. Verify
   referenced paths and ordinary Git fetch/checkout from the remote, not merely
   a local branch name. Keep the upcoming release target and plan links consistent.

**Checks:** focused `node --test scripts/backend-installer-contract.test.mjs`,
Markdown/format/link checks, and `tests/test_dev_entrypoint.py` if entrypoint
behavior changes. These documentation-command assertions do not prove installation.
Include the required full client gate at final freeze for changed setup/config
examples, as CONTRIBUTING specifies.

**Exit:** a reviewer can obtain the tested source pair from the published commands.
**Blocker:** choosing a public release branch/tag is separate from changing docs;
use the existing authorized candidate branch rather than inventing a release.

## W2 — ordinary fresh Windows setup and readiness

**Budget: 1.5–3 hours; network/package download time is additional. Type: native
acceptance gap; most source implementation already exists.**

**Where:** `BE/pyproject.toml`, `uv.lock`, `modiff/base_runtime.py`,
`runtime_environment.py`, `runtime_compilation.py`, `optimization_packages.py`,
`docs/developer-setup.md`; FE `src/components/RuntimeOptimizationsCard.tsx`,
`src/studio/templateReadiness.ts` and `src/studio/optionalRuntimePolling.ts`.

Use a genuinely separate checkout/environment/data directory, not a second folder
whose `.venv` resolves into the live runtime. Preserve package/download caches;
sharing immutable package-cache bytes does not make the new environment non-fresh.

1. Follow W1's checkout instructions. Verify Node/npm versions and ordinary
   operator-installed uv. Install the declared CUDA profile without repository
   installer wrappers:

   ```text
   uv sync --extra cuda --locked
   uv pip check --python .venv/Scripts/python.exe
   uv run --extra cuda python -m modiff.preflight --json --fail-on-error
   ```

2. Configure an isolated data root and unused loopback ports using the existing
   documented configuration. Check the intended port before launch. Start with
   the same CUDA extra; do not launch another backend on an existing app's port.
3. Inspect dependency versions and capability readiness. Transformers and PEFT
   must already be present. Startup, browsing and Auto inspection must not install
   anything. An ordinary template must not ask the user to Activate those core
   libraries. Model weights and gated access remain separate prerequisites.
4. Record the actual allocator settings and whether Triton is absent. Run one
   supported eager/SDPA image through the normal UI with an already approved cached
   model. A small model is suitable for setup plumbing, but label that scope;
   it cannot replace W3's Qwen workload or establish optional compiler support.
5. In this disposable environment only, exercise the documented missing-base
   repair path and ordinary same-profile update. Do not uninstall a foundational
   package from the user's active environment or intentionally upgrade all packages.
6. Launch the matching FE with `npm ci` then `npm run dev`; cover the normal Vite
   path separately from the backend-served production client in W5.

**Expected:** reproducible setup, accurate healthy/missing-base UI, no foundational
Install/Activate ceremony, a genuine image, and ordinary Windows operation without
Triton or unsupported allocator warnings injected by MoDiff.

**Blockers:** package download/access, insufficient isolated-disk capacity, GPU
driver/runtime incompatibility, a currently busy GPU. Report these directly.
After two failures in one setup category, inspect cause instead of starting a
third installation. Native uv CUDA and the existing managed composite are distinct
runtime contracts; do not merge their proof.

## W3 — original Windows Qwen and repeat-run memory complaint

**Budget: 3–6 hours if the exact route is admitted. Initial capacity verdict by
04:00. Type: open native acceptance; a new memory bug is not yet established.**

**Where to trace:**

- `BE/modiff/NodeBase.py`: input snapshots, implementation identity, cache reasons.
- `BE/modiff/server.py`: per-node execution/reuse events, exact cached owner
  bindings, normal cache release and memory observations.
- `BE/modiff/workflow_auto_resource.py`: separate RAM/VRAM, candidate scope,
  loading/conversion allocations and once-only resident storage credit.
- `BE/modiff/workflow_auto_lifecycle.py`: ownership removal and actual next-owner
  capacity, not assumed reclaimable bytes.
- `BE/modules/ModularDiffusers/loaders.py`: component reuse, placement, optional
  quantization and pre-allocation guards.
- For the separate FLUX3 case: `BE/modiff/flux3_auto_resource.py` and the bounded
  CUDA loading helpers. FLUX3 evidence does not close Qwen feedback.

**Approach:**

1. Identify the exact original Qwen text-to-image template/route, immutable model,
   full creator settings and approved installed artifacts. Do not substitute Qwen
   2.1, a different model, reduced resolution/steps or Custom mode and call it the
   original Auto acceptance. Record a separately chosen supported recipe separately.
2. Obtain live Auto assessment and idle RAM/dedicated VRAM. If refused, preserve the
   exact requirement/availability and source before any retry. Trace cold loading,
   retained owners and actual process/OS memory to identify an overcount, leak or
   genuine shortage. A capacity floor is not automatically a bug.
3. For a diagnosed defect, make a minimal shared correction with a failing focused
   regression. Never remove safety reserve, count shared memory twice, credit a
   partial/dead owner, close arbitrary user applications or restart to disguise a
   warm-run memory failure. Contact the Linux owner before shared-file changes.
4. Once admitted, run A–E below consecutively without intermediate cache release,
   worker restart or unrelated cleanup. Capture actual node events, model load and
   encoder observations, memory and terminal receipts. An unchanged cache hit is
   legitimate, but label it as delivered cached output.

| Case                      | User action                                | Expected result                                                                                               |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| A                         | Cold run of exact saved recipe             | Successful load, complete output set and responsive UI at full settings                                       |
| B                         | Run again unchanged                        | Valid cached delivery or actual compute identified; no unnecessary weight reload or crash                     |
| C                         | Change seed only                           | New downstream computation; model and unchanged prompt encoding reused where supported; consumed seed matches |
| D                         | Change positive prompt and seed            | Prompt re-encoded, new generation; unaffected settings/models retained                                        |
| E                         | Restore original values                    | Correct restored execution or honest cache reuse; no stale embeddings/adapters/guider state                   |
| F/G, only if time permits | Switch to another supported model and back | Reviewed compatible values/wires preserved; valid owner replacement/reuse or accurate refusal                 |

5. Share W4's save/reopen and LoRA observations with these runs only when they
   really use the same graph/settings/source. After A–E, exercise scoped Release;
   output bytes persist and unrelated owners remain. Then verify next-run behavior.

**Focused tests when affected:** `tests/test_node_base.py`,
`test_modular_workflow_reuse.py`, `test_workflow_auto_resource.py`,
`test_workflow_auto_cuda_recovery.py`, `test_node_cache_cleanup.py`, and exact
FLUX3/quantization tests only for changes to that separate route.

**Separate FLUX3 capacity diagnosis:** the retained W01 refusal happened before
generation, with a 2.75-billion-byte RAM shortfall. The earlier loading-overlap
fix is already in the baseline. First attribute the dispatch-to-load memory drop;
do not replay the same recipe hoping free memory changes. If the real envelope
cannot fit with supported policies, preserve the hold and its practical user
guidance. This prevents claiming that the original Windows memory feedback is
fully resolved.

## W4 — one continuous image-prototyping and persistence journey

**Budget: 2–4 hours, reusing genuine W3 observations where applicable. Type:
integrated acceptance plus narrowly diagnosed UX fixes.**

**Where:** FE `src/components/BlockNodeV2.tsx`,
`src/components/CustomExtensionsPanel.tsx`, `src/components/NodeSearchDialog.tsx`,
`src/studio/blockPersistenceV2.ts`, `src/workflow/operationAuthoring.ts`,
`src/workflow/operationLegacyBlockChange.ts`,
`src/workflow/nodeConnectionMatching.ts`, the existing workflow save/sync store
modules and `src/components/ServiceExportDialog.tsx`. BE custom/composition/service source is owned according
to the shared plan, not edited concurrently.

1. Create an image workflow through the normal task browser. Show its generic
   load/setup → encode → denoise → output stages. Advanced options should remain
   discoverable. Do not redesign the product or enforce four boxes everywhere.
2. Drag from an unconnected image/text port onto empty canvas. Verify compatible
   suggestions and insertion with the correct wire. Check that unrelated types
   are excluded by default. Preserve the explicit unverified-match option.
3. Use a reviewed native modular Block and expand its actual upstream hierarchy;
   an ordinary visual group around a whole-pipeline call does not establish this
   acceptance. Edit prompt, seed and another applicable control through visible
   inputs; commit focus. Collapse/expand, resize, move a supported internal node and make
   one supported connection edit. Inspect usable internal controls, all ancestor
   bounds and semantic connections. No hidden-store injection is acceptance.
4. Use an already reviewed custom node or the exact source admitted by L1. For a
   true Hub-Block claim, exercise the actual pinned Hub Add/Load path on Windows;
   a local text helper alone cannot supply that claim. If the source/model is
   unavailable, retain the partial Windows result and precise missing prerequisite.
5. Review a supported cross-family or text-to-image/image-to-image change. Test
   Cancel before Apply; apply one compatible change; Undo/Redo. Preserve authored
   values, user nodes, layout and connections; require missing input media openly.
   Observe the first Run click after a model change rather than silently clicking
   again if no task is submitted.
6. Save the workflow, explicitly Save as User Node, close its tab and reopen from
   My workflows without reloading the browser. Insert the reusable Block. Compare
   effective defaults plus overrides and exported execution graph, not just the
   overrides dictionary. Plain edits must not overwrite the library definition.
7. At an idle boundary, refresh and restart the actual candidate. Reopen/reinsert
   and Run with saved model/parameters. Match the restored generated preview to
   the actual task/media, excluding input images and thumbnails.
8. Verify model-backed service execution on Windows if claimed for this release;
   otherwise document L1's Linux-only scope. The existing service package is a
   local MoDiff API graph, not standalone Diffusers Python generation.
9. Reserve 45–90 minutes within the W3/W4 model budget for one ordinary Windows
   LoRA case if compatible approved artifacts are already cached. Freeze the base
   model, adapter revision/hash/name and full recipe. Compare base/zero and nonzero
   scale with the same seed, then change scale and restore/remove the adapter.
   Confirm actual activation, affected fresh computation, persisted settings and
   no stale adapter after removal. If a second approved adapter is available,
   exercise replacement as a distinct observation. Do not infer effect from a
   descriptor alone. If fit/artifact/time admission fails, leave Windows LoRA
   acceptance explicitly held; PEFT installation and Linux L2 cannot close it.

**Checks if changed:** existing operation-authoring, block-persistence, graph
export, node-connection-search and workflow-backend-sync tests, plus the relevant
native/browser gesture. No new source change is required merely because a proof
row was previously missing.

**Exit:** one continuous, recorded journey with exact values and output after
restart. A defect receives a causal fix; an unrun transition remains unrun.
Arbitrary family/task permutations are outside this deadline.

## W5 — refresh, progress, release and installation status

**Budget: 1–2 hours. A real optional reinstall adds 1–2 hours and is admitted only
if needed for an affected release claim.**

**Where:** FE `scripts/backend-proxy-lifecycle.ts`,
`src/studio/optionalRuntimePolling.ts`, `src/components/RuntimeOptimizationsCard.tsx`, connection
and supervisor helpers, `src/utils/serverActions.ts`, `src/components/CustomNode.tsx`; BE websocket,
optional-runtime job/replacement-worker and cache teardown paths.

- Observe production startup and five refreshes during a genuine owned run. The
  same task/progress/output recovers; another edited tab stays selected and intact.
  Repeat the disconnect observation through normal Vite serving. Record emitting
  process/stack if ECONNABORTED recurs. Do not globally suppress network errors.
- On a separate owned run, ordinary Stop reaches a durable terminal state and the
  queue drains before retry. Distinguish graceful cancellation from worker
  replacement. Retain the failed attempt; never retry an unknown active task.
- After genuinely loaded-model execution, Release cache updates the UI from the
  acknowledged removed-node set, preserves outputs, stays responsive and shows an
  error if it fails. The current full cache-release request timeout is 120 seconds; ordinary
  requests and recompute retain their narrower policy.
- In an isolated runtime only, if a real optional install/activate is necessary,
  capture durable job ID and visible install → verify → activate/restart → ready
  stages, refresh recovery, failure recovery and replacement-worker identity.
  HTTP 202, an installed directory or a returned Activate button is not readiness.
  Ordinary base dependencies must never require this ceremony.

Use existing request/proxy/polling regressions when their code changes. A mocked
polling pass alone does not close actual installation or native refresh acceptance.

## R1 — integrate once, verify, deploy and hand over

**Reserve 14:00–18:00.** Read Linux's exact commit/validation handoff by 13:00.
Inspect semantic conflicts even if Git merges cleanly. Preserve old evidence and
user data; do not merge historical dirty overlays again.

1. Freeze BE source, then update both immutable FE CI backend refs to the real
   compatible BE commit when coordinated code changes require it. Run focused
   conflict/contract checks; build the client from source.
2. Mirror final client `dist/` into BE `web/` through the established procedure,
   preserving `web/user` and local Gallery installs. Never hand-merge minified JS.
3. Check all maintained generated contracts using the exact contributor/CI
   commands, including operation inventory and image-prototyping readiness.
   Regenerate only affected outputs. The dependent ledger chain is upstream
   coverage → template candidates → authoring specs → Comfy contract/evidence
   resolution; operation inventory/readiness also require their own checks.
   Compare semantic changes; hash refreshes grant no qualification.
4. Run complete BE CONTRIBUTING validation and compatible FE `npm run check` /
   required browser gate on frozen source. The same interpreter/runtime cannot
   be assumed valid for base and active optional tests. Preserve skips/failures.
5. Record the whole-app gate limitation separately. Inspect Gallery, release and
   asset stages when their inputs changed; do not describe short-circuited stages
   as successful. An unavailable Mac remains blocked.
6. At idle, deploy the exact accepted pair using the existing guarded process.
   Compare source identity, disk and HTTP asset bytes, main/control worker identity,
   saved-data before/after and ordinary output access. Recheck only affected native
   paths on the deployed view. Keep a verified prior candidate for rollback.
7. Push both source branches normally and publish the exact paired identities.
   Clean active checkouts must contain no forgotten implementation changes.

**18:00 manual checklist:** allocate about 20 minutes setup/launch, 40 minutes
generation/repeats, 40 minutes graph/model/custom edits, 30 minutes save/restart/
reopen, 20 minutes Stop/refresh/cache release, plus 30 minutes for observations.
Give concrete workflow names, settings, expected result and known limitations in
the actual handoff. This is a 3-hour maximum guide, not a demand to rerun a broad
qualification suite. Leave the app idle and ready for the maintainer.

## Prompt for the Windows chat

> Execute the Windows lane in `plans/release-20261028/windows.md` together with
> its README and review. Deadline is 28 October 2026 at 18:00 IST; reserve
> 18:00–21:00 for my manual testing. Begin from the published clean
> `work/manual-testing-20261020` pair and create/resume the dedicated Windows
> branches safely. Read the required repository guides and establish current
> app/runtime/data ownership before changing anything. Tell me the exact task
> before working on it. W1–W5 are the scope; diagnose actual failures and implement
> focused fixes, rather than repeatedly running full tests. Preserve user data,
> runtime, model settings and all original failed evidence. Keep Qwen acceptance
> separate from the FLUX3 capacity hold. You own frontend, setup/distribution and
> final integration; coordinate shared backend files with the Linux chat through
> the agreed handoff. Source freeze is 14:00, deployment verification before
> 18:00. Use one final stable-pair gate cycle, report blocked/unrun requirements
> honestly, and provide the actual URL, changes and expected manual-test results.
> Do not expand into the full catalog or publish new Gallery media. Raise concrete
> missing access, source approval or hardware prerequisites promptly; do not ask
> again for routine actions already authorized by this execution request.
