# Creator / Developer workspaces implementation plan

## Status and relationship to previous work

The Creator / Developer direction is accepted. This document is the implementation
and acceptance contract; unchecked work is not delivered. Keep this document
identical in the backend and client repositories and continue on their existing
`feat/generic-diffusers-workbench` branches.

This supersedes the audience, starting-point and discovery decisions in the
[earlier workbench plan](generic-diffusers-workbench-plan.md). Preserve its completed
runtime work and historical evidence, including unresolved findings. Do not reset
the repositories, rewrite saved workflows, or relabel old validation as a new pass.

The user specifically selected a **Workflows** modal for Developer startup, with
actions such as text-to-image and image editing. It must not open the Templates
modal as the Developer starting point.

## Product contract

### Two workspaces, one application

| Surface              | Creator                                                 | Developer                                                                  |
| -------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------- |
| Purpose              | Compose and run creative workflows                      | Implement, inspect and test nodes and workflows                            |
| Fresh starting point | Templates, recent workflows, empty workflow             | Workflows modal with task choices, recent/open and empty workflow          |
| Canvas               | Editable nodes and Blocks                               | The same editable nodes and Blocks                                         |
| Discovery            | Common operations, installed custom nodes, Saved Blocks | Generic operations, installed custom nodes, Saved Blocks                   |
| Inspector            | Parameters, connections, outputs                        | Same controls plus interface, implementation, source, docs and run details |
| Extensions           | Installed nodes usable normally; management accessible  | Prominent Add from Hugging Face, inspect, enable and reload                |
| Export               | Workflows, Blocks and outputs                           | Same exports plus API graph and service package                            |
| Resources            | Memory: Automatic / Custom                              | Memory: Automatic / Custom                                                 |

Use a labelled **Workspace: Creator / Developer** segmented control. Memory
settings remain separate. Changing workspace changes presentation and defaults,
not execution authority, model choice, values, resource policy, caches or topology.
Keep the same sidebar positions, workflow tabs, Run/Stop and queue controls.
Templates remain deliberately accessible to developers; they are not startup.
Creators retain ordinary graph editing and precise parameter controls.

### Vocabulary and discovery

- Workflow: the editable, executable document.
- Template: an optional starting copy of a workflow.
- Node: an operation with inputs, outputs and controls. Encode Prompt and Denoise
  are nodes, not a new public object called Stage.
- Block: a reusable graph composition with exposed inputs/outputs.
- Saved Blocks: saved compositions within the Nodes library.
- Cluster: preserve existing identifiers, user names and search aliases; avoid a
  competing catalog for the same compositions.
- Custom node: an enabled code-defined extension, possibly wrapping a native
  Modular Diffusers block. Code without an editable graph does not gain fake children.

Replace the Stages / Essentials / Advanced library selector with one Nodes library
using categories, search, favorites and provenance filters. Match one canonical
entry to its real operations; do not create Creator/Developer copies or one public
copy per model family. Keep distinct implementation/legacy access contextual where
equivalent replacement is not established. Do not bulk-delete backend adapters.

### Developer Workflows modal

1. Open on a fresh Developer session without a restorable workflow and on explicit
   New workflow. Restore existing work before considering a startup chooser.
2. Title: **Workflows**. Present actions including Text to image, Image to image,
   Image edit, Inpaint, Outpaint, Image conditioning and other declared tasks.
   Add video/audio/other task groups from available capabilities; do not maintain
   a second hard-coded family/task support list in the client.
3. Always provide Empty workflow and Open workflow; show recent workflows when
   present. Dismiss/Escape leaves an empty editable canvas without creating models.
4. Selecting an action shows compatible models, prioritizing downloaded models.
   Separate missing files, missing runtime and unqualified resource recipes.
   Unsupported tasks show an explanation rather than an empty unexplained list.
5. Show the intended node graph and required user inputs. Create the minimal
   connected workflow through the existing operation starter API. Include the
   appropriate ordinary Preview/Save/Export node where its contract is known.
   Use existing node factories, connection validation and history transactions.
6. For a decomposable text-to-image route, show Load Models, Encode Prompt,
   Denoise, Decode Latents and Preview as ordinary visible nodes. Actual component
   wires remain explicit. Whole-pipeline-only routes show their honest load/run
   operations and output node; do not manufacture a Modular decomposition.
7. Opening, filtering or inspecting this modal never installs packages, downloads
   weights, imports custom Python or allocates a model. Explicit setup actions
   remain separate from creating an editable draft.
8. Apply creation once. Cancel, double clicks, delayed responses, retries, tab
   switches and offline errors cannot add duplicates or overwrite another graph.

Changing workspace on a nonempty workflow must not open a modal over it. Empty
workflow switches may show the new workspace's chooser once; dismissal and reload
must not create a recurring modal loop. Test both startup and later New workflow.

### Generic model and task authoring

The selected loader or Block owns model/task changes. Reuse the existing canonical
operation contracts and backend adapters; frontend presentation reads capabilities.
Derive controls from upstream declarations where available, with small documented
backend adaptations where necessary. Audit family-name restrictions and replace
only unnecessary ones with real input/component capability checks.

Keep compatible fields and connections. Preserve unsupported values for Undo or
switch-back without submitting them. Structural changes require a clear preview
and one atomic apply/Undo. Retain custom nodes and explain incompatible edges.
Never reuse embeddings or latents from the previous incompatible model.

Common text/image processors must not require per-family registration. Real tensor
layout, component, guidance and task constraints remain enforced. Metadata support,
execution support, hardware fit and output quality are separate properties.
Selecting a standard pipeline without decomposition keeps a generic whole-pipeline
node rather than forcing users into a family-specific workflow.

## Milestones and completion tracker

Only check a milestone after its implementation and required acceptance pass.
Attach exact commits and sanitized results to its review; retain raw receipts,
machine inventories and media in ignored review storage. A blocked case remains
visible and does not count as a pass or a completed release requirement.

| ID  | Milestone                                                    | Status                                   | Depends on                      |
| --- | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------- |
| W0  | Accepted design and detailed implementation/test plan        | Complete: plan only                      | User direction                  |
| W1  | Baseline, artifact classification and executable test ledger | Planned                                  | W0                              |
| W2  | Workspace switch, independent memory labels and migration    | Planned                                  | W1                              |
| W3  | Creator entry and Developer Workflows modal                  | Planned                                  | W2                              |
| W4  | Unified Nodes library and contextual inspector               | Planned                                  | W2                              |
| W5  | Generic model/task changes and workflow modification         | Planned                                  | W3, W4                          |
| W6  | HF/local custom-node workflow                                | Planned                                  | W4, W5                          |
| W7  | Concurrent authoring, reuse and recovery                     | Planned                                  | W5, W6                          |
| W8  | All-local-image execution and modification campaign          | Planned                                  | Stable W2–W7 build              |
| W9  | Other modalities and real service-export campaign            | Planned                                  | Stable W2–W7 build; W8 fixtures |
| W10 | Release acceptance, documentation and publication            | Planned                                  | W8, W9                          |
| H1  | Windows Qwen 16 GB VRAM / 32 GB RAM qualification            | Deferred until main UI/UX implementation | W2–W7; Windows host             |

### W1 — Baseline and test ledger

- [ ] Record paired source revisions, dirty-file ownership, served bundle bytes
      and running-process identity before changing behavior.
- [ ] Inventory existing cached snapshots, exact revisions, selected shards,
      tokenizer/config requirements, references, adapters and dependencies.
- [ ] Classify complete pipelines, multimodal pipelines, auxiliary processors,
      components/LoRAs, partial artifacts and unsupported upstream integrations.
      Do not count a component or repository name as a runnable image model.
- [ ] Resolve each complete model to task contracts, generic operations, editable
      decomposition or whole-pipeline fallback and applicable resource recipes.
- [ ] Create one ledger entry per artifact revision/task/runtime path; include
      missing integrations instead of silently filtering them out of totals.
- [ ] Preserve existing workflows, Blocks, model files and old failure evidence.
      Separate the Wan video delivery failure from successful backend completion.

Acceptance: every inventoried artifact has a classification and disposition;
every required model/task has queued test scenarios or a named blocker. No new
downloads, deletions, dependency upgrades or execution-support claims from scanning.

### W2 — Workspace and memory controls

- [ ] Implement Creator/Developer presentation policy centrally; audit all old
      Auto/Expert checks so audience and resource policy cannot be conflated.
- [ ] Migrate old Auto view -> Creator and Expert view -> Developer once. Preserve
      independently stored resource settings and legacy graph/API representations.
- [ ] Preserve graph values/topology, viewport, collapsed state, queue, trust and
      component owners during workspace changes and across reload.
- [ ] Retain per-workspace panel preferences without letting imported workflows
      change the user's workspace. Expose Automatic/Custom memory in both.
- [ ] Replace outdated user-facing copy without renaming upstream class names,
      serialized identifiers or historical evidence.

Acceptance: native switch tests cover old/new storage, both memory policies,
multiple tabs, loaded models and active runs. Compare execution graph and effective
settings before/after; verify no load/encode work triggered by presentation alone.

### W3 — Entry flows and Workflows modal

- [ ] Reuse the task-launcher and shared dialog primitives for the Developer
      Workflows modal; route creation through operation capabilities/starters.
- [ ] Implement task-first selection, downloaded-model ranking, graph/input preview,
      Empty/Open/recent workflows and explicit missing-dependency actions.
- [ ] Keep Creator's Templates/recent/empty path and preserve authored template
      layout. Do not force compact Blocks or a full-screen parameter form.
- [ ] Add appropriate output nodes and validate actual required component wiring.
      Keep task distinctions such as image-to-image versus instruction editing.
- [ ] Cover startup restoration, Escape, keyboard navigation, small viewports,
      retry, double click, stale responses and switching tabs while resolving.

Acceptance: new Developer starts at Workflows, never Templates automatically.
Text-to-image, image-edit, image-to-image and a non-image task create the exact
visible executable graph. Opening/selecting tasks performs no model allocation.

### W4 — One Nodes library and inspector

- [ ] Consolidate canonical discovery, utilities, installed custom nodes and Saved
      Blocks under one searchable library. Keep distinct saved revisions.
- [ ] Support click/drag insertion, canvas search and dangling-link suggestions
      in both directions with the same connection validation as the canvas.
- [ ] Retain legacy search aliases and actionable access to distinct implementations
      without presenting duplicates as separate generic operations.
- [ ] Use the selected node/Block inspector for Parameters, Interface, Implementation,
      Docs and Run details. Developer emphasizes diagnostic/source tools.
- [ ] Keep nested graph editing, public sockets, output previews and ordinary node
      behavior shared across both workspaces. Do not confuse Python blocks with
      editable graph compositions.

Acceptance: library and canvas tests cover built-ins, enabled custom nodes, saved
compositions, type aliases, unsupported connections, keyboard use and legacy load.
Inspecting metadata never mutates execution or assumes the library was already open.

### W5 — Model/task changes and editable workflows

- [ ] Put model/task changes on the owning loader/Block, reusing current graph-change
      preview/apply machinery and one Undo/rollback transaction.
- [ ] Make compatible model changes retain inputs and adaptable fields; preserve
      unsupported values without execution. Handle disconnected and shared loaders.
- [ ] Add required image/mask/reference operations through actual task contracts.
      Respect dimensions, shared generator identity and upstream state writers.
- [ ] Preserve custom nodes and explicit diagnostic edges when no safe automatic
      adaptation exists. Reject stale previews rather than overwriting edits.
- [ ] Qualify create/modify/save/reinsert/execute on generic nodes and nested Blocks,
      including crossing connections, selected-Block execution and legacy instances.

Acceptance: the modification matrix below passes in contract/native tests before
expensive model qualification. Selected paths additionally execute with real models;
graph-shaped JSON or a successful inspector is not execution evidence.

### W6 — Custom-node development

- [ ] Provide Add from Hugging Face and local-source entry points from Developer
      without requiring a template or existing custom-node placeholder.
- [ ] Resolve user-entered repo URL/ID to an inspectable immutable revision through
      existing Hub/source mechanisms; show revision, source and dependencies.
- [ ] Render existing Diffusers/Mellon UI metadata with normal registry/field tools.
      Require exact code approval before imports; importing a workflow grants none.
- [ ] Show installed source location and review/reload changes. Preserve targeted
      cache invalidation and reject stale approval or missing dependencies.
- [ ] Explain idle-only activation/reload without blocking source inspection during
      inference. Verify edit failures preserve unrelated graph/component owners.

Acceptance: an approved local Python node and an HF Modular block execute inside
existing image workflows, including a saved Block. A generic prompt/image processor
is tested across distinct model families without family-specific registration.
Missing dependency, changed helper file, failed import, cancellation and stale
approval paths retain useful diagnostics. No Python sandbox claim.

### W7 — Responsiveness, reuse and recovery

- [ ] Reproduce/profile model/task preview and loader-contract timeouts during
      actual inference; trace event-loop work, locking and metadata dependencies.
- [ ] Keep read-only metadata resolution independent of live model allocation and
      execution locks. Cache immutable metadata with correct identity/invalidation.
- [ ] Treat submitted runs as immutable snapshots; edits target the next draft.
      Progress/history cannot replace newer values or steal the active editing tab.
- [ ] Preserve per-node/component reuse, exact consumed-input identities, immutable
      cached generators/state and bounded multi-owner memory behavior.
- [ ] Retain safe serialization for extension activation, model release and teardown.
      Do not fix latency by weakening validation or merely increasing timeouts.
- [ ] Test Stop/restart, durable cancellation, OOM recovery and a successful new run.

Acceptance: while a real model denoises, create another workflow, search/add nodes,
preview model/task changes, inspect custom sources, edit/save and switch workspace.
Measure request and gesture latency separately. Initial budgets: local interactive
read requests p95 <= 1 second; warm metadata preview <= 3 seconds; cold metadata
preview <= 10 seconds, excluding explicit network/source downloads. Capture any
breach and investigate; budgets are targets, not claims about current performance.
Keep native tests for stale responses, cleanup cancellation and queue attribution.

### W8 — Every complete local image model

- [ ] Freeze a validated paired build and finalize the W1 artifact/task ledger.
- [ ] Execute the mandatory per-model cases below through the revised native UI.
- [ ] Execute every declared applicable task for each complete local image model;
      test task-specific inputs and output interpretation rather than one T2I proxy.
- [ ] Qualify auxiliary image processors and cached LoRAs/controls in compatible
      containing workflows; record actual dependency identities and effects.
- [ ] Execute structural/custom-node tests for every distinct implementation path,
      and targeted cross-model transitions that cover contract differences.
- [ ] Assess real outputs and retain original media, not just completion statuses.
- [ ] Fix failures with regression tests and rerun the affected matrix. Maintain
      blockers visibly until resolved; do not lower settings to erase a failure.

Acceptance: all required ledger rows pass at their declared proof level. A model
that cannot run due to a missing integration, dependency or memory recipe remains
a blocker with a concrete follow-up, not an omitted denominator or passing skip.

### W9 — Other modalities and real service export

- [ ] Run baseline, unchanged, parameter-change and save/reopen cases for each
      complete supported cached video/audio/other pipeline; track incomplete and
      unsupported artifacts separately. Prioritize image qualification first.
- [ ] Exercise representative distinct non-image runtime paths with modified/nested
      graphs, custom processors and workspace switching. Expand coverage for every
      newly exposed contract difference or regression.
- [ ] Recover the already-generated heavy Wan output and diagnose browser decoding
      before deciding whether inference needs repeating; retain failed evidence.
- [ ] Export real image workflows natively as service packages and execute with
      named inputs; compare consumed values and exact-task outputs with UI runs.
- [ ] Cover at least native Modular, standard whole-pipeline, image editing, custom
      node and nested-Block service paths, then representative video/audio outputs.
- [ ] Exercise changed inputs, repeated calls, sequential queued invocations,
      cancellation/recovery, restart, code/package drift and missing model identity.

Acceptance: actual downloadable package, actual invocation and decoded retained
outputs are all proven. The existing model-free smoke and parser tests remain
separate evidence. Do not claim standalone Python export, cross-device pixel
identity, multi-user hosting or arbitrary concurrent model execution.

### W10 — Release and publication

- [ ] Run final backend base/verified-optional gates and client quality/browser gates.
- [ ] Mirror the final production client bundle while preserving backend-owned user
      fields; verify served bytes and source identity from a fresh idle backend.
- [ ] Close relevant Gallery asset/provenance and release-contract evidence failures
      without changing expected hashes merely to obtain a green result.
- [ ] Update both READMEs and durable UI/custom-node/service guides to implemented
      labels, setup flow and measured support. Keep this plan mirrored.
- [ ] Audit preservation of model files, user workflows, Blocks and settings.
- [ ] Commit focused reviewed changes and push paired feature-branch revisions;
      record exact validation and unresolved platform restrictions.

H1 remains a separate Windows hardware requirement: after main UI work, verify
Qwen on 16 GB VRAM / 32 GB RAM with an explicit suitable recipe, repeat/edit/save,
Stop/recovery and real output inspection. Do not use shared-memory Linux evidence
as a substitute or claim that H1 passed because the interface milestones passed.

## Rigorous execution and modification matrix

### Coverage unit and record

The coverage unit is **artifact revision + task + implementation path + resource
recipe + surface + scenario**. Track built-in nodes, graph Blocks and custom Python
wrappers distinctly. Different revisions/distillation variants do not inherit live
qualification solely because their pipeline class matches.

Each private ledger row records source revisions, model/dependency revisions,
download completeness, prompt/input-media identity, requested and consumed values,
precision/quantization/offload, task ID, node-call counts, component reuse reasons,
timings/memory, output identity/shape, decoded-media result, visual assessment,
failure details and evidence paths. Never store credentials in the ledger.

Statuses: Planned, Running, Passed, Failed, Blocked, Not applicable. Not applicable
requires a contract-based reason (for example, no text encoder), not a test failure.
An interrupted/unknown task is reconciled before retrying; never blindly resubmit.

### Mandatory baseline and edit cases per complete local image model

| Case | Native action and execution                                                     | Required evidence                                                                                  |
| ---- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| I01  | Create workflow through Developer Workflows; choose exact downloaded model; run | Visible minimal graph, actual artifact/inputs, decoded output                                      |
| I02  | Run unchanged                                                                   | Same consumed inputs/output; measured node/component reuse or explicit memory-driven reload reason |
| I03  | Edit prompt with real field gestures and run, where supported                   | Encoder invalidation where appropriate; models retained when feasible; consumed text exact         |
| I04  | Change seed and run, where supported                                            | Effective seed; no stale generator reuse; unaffected conditioning retained                         |
| I05  | Change steps/guidance or another supported sampler control and run              | Actual effective values; only dependent recomputation                                              |
| I06  | Change size/batch or a task-specific output control where supported             | Valid model granularity; measured output dimensions/count                                          |
| I07  | Save, reload page/reopen workflow, edit and run                                 | Persisted values and connections; durable preview restored; new execution after edit               |
| I08  | Save processing selection as Block, reinsert, edit exposed input and run        | Equivalent effective graph; independent instance values; correct output ownership                  |
| I09  | Open the same workflow in Creator, edit/run, then return to Developer           | Shared editable graph; mode change alone causes no resource/model change                           |
| I10  | Deliberately recompute same supported inputs                                    | Fresh compute, stable or documented-tolerance output comparison; not a cache-only claim            |
| I11  | Applicable image/edit/control tasks with actual reference/mask inputs           | Task-specific effect, preserved regions or structure as appropriate                                |
| I12  | Insert a compatible generic text/image processor and run                        | Real custom/built-in operation output consumed, not an unused decorative node                      |

For unconditional, layered, auxiliary or otherwise different models, replace
inapplicable prompt/sampler assertions with their declared controls and output
semantics, recording why. Do not fake a text encoder or VAE. All other applicable
cases remain required; the task sweep is not replaced by one small generation.

### Structural modifications across distinct runtime contracts

Run contract/native tests broadly and real inference on every distinct affected
runtime path. Ensure multiple model families are exercised; one SDXL fixture is
insufficient for Qwen/Flux-specific latent and state behavior.

| Modification                                                  | Assertions before and after actual execution                                             |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Add/remove/reconnect compatible nodes                         | Effective input changes; downstream invalidation; unaffected branches preserved          |
| Insert between existing nodes                                 | Real execution through inserted processor; no stale bypass or duplicate edge             |
| Invalid connection / missing required input                   | Precise diagnostic; no submission; reconnection restores runnable graph                  |
| Same-family and cross-family model switch, then switch back   | Compatible values retained; unsupported values recoverable; no old conditioning reused   |
| T2I -> img2img/edit/inpaint where declared                    | Correct encoders/masks/references and dimensions; no accidental task substitution        |
| Duplicate workflow/Block and edit one                         | Independent values/output routing; intentional model sharing only                        |
| Save Block -> reinsert -> modify -> save again                | Stable identities and public interface; saved baseline not implicitly overwritten        |
| Nested Block, move in/out, public/crossing wires              | Same graph ownership and execution scope; no hidden extra model requirements             |
| Resize, expand/collapse, arrange and move                     | Usable fields/ports; no semantic changes or recomputation from layout                    |
| Undo/Redo after value and structural edits                    | Exact restored graph and execution inputs; redo survives async layout/save               |
| Run selected Block versus whole workflow                      | Correct dependency scope and all intended terminal outputs                               |
| Add/reload custom HF/local node                               | Actual new behavior, targeted cache invalidation and unchanged unrelated owners          |
| Shared loader, multiple consumers and two independent loaders | Correct ownership, no duplicate loading, pressure-aware lifetime behavior                |
| Recompute outputs versus release model cache                  | Different intended effects; actual component references/allocations checked              |
| Edit another tab during long run; refresh/reconnect           | Active run immutable, draft preserved, no tab stealing or stale response overwrite       |
| Stop or OOM then retry                                        | Durable terminal status, safe cleanup, correct restored settings and successful next run |
| Open legacy workflows/Blocks                                  | No unsolicited conversion; explicit touched changes are atomic and undoable              |

Select cross-model transitions by contract differences: pooled versus non-pooled
conditioning, multiple encoders, packed/unpacked latents, negative-prompt support,
distilled/full guidance, edit references, whole-pipeline versus Modular execution.
Test incompatible transitions visibly. Pairwise structural coverage is acceptable
only after every model has its mandatory baseline/edit cases and every distinct
runtime contract has structural execution coverage; it is not a shortcut around
the all-model requirement.

### Workloads and output assessment

- Use complex prompts with spatial relations, multiple objects/materials, lighting,
  composition and preservation/edit instructions appropriate to each task.
- Use meaningful model-supported resolutions and steps. Tiny low-step probes can
  diagnose plumbing but cannot be the only acceptance for a high-quality workload.
- Verify every UI edit reaches the consumed input, including text, booleans, zero,
  scalar/list controls, uploaded media, effective size and seed sharing.
- Check NaN/Inf/empty/black outputs, decoded dimensions/count, task relevance and
  actual intended modification. Assess masked/unmodified regions separately.
- Pixel hashes establish identity, not image quality. Seed/prompt edits should have
  appropriate effects, but do not impose invalid universal pixel inequalities on
  deterministic processors or models that ignore an unsupported control.
- For video/audio/3D, decode/play or inspect the actual generated output, measuring
  frames/duration/channels/sample rate/geometry as applicable. Input previews and
  placeholders cannot count as generated-media evidence.
- Use exact reproducibility where the path supports it; otherwise define measured
  numerical/visual tolerances before accepting a comparison. Keep failures visible.

## Implementation ownership and validation commands

Reuse the current owners rather than adding another runtime or workflow format:

| Area                               | Existing owners to inspect/change                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Workspace / entry / discovery      | Client settings and Studio stores, TopBar, TaskLauncher, TemplateBrowserDialog, NodeList                                                                           |
| Canonical creation and changes     | OperationCatalogPanel, OperationGraphControls, operationAuthoring, operationGraphTransaction, backend operation_catalog / operation_starters / operation_contracts |
| Dynamic controls and compatibility | Registry field actions, ModelSelectField, shared connection matching, operation schemas                                                                            |
| Context and extensions             | OperationStageInspector, CustomExtensionsPanel, backend custom_extensions / custom_extension_api                                                                   |
| Reuse and responsiveness           | Backend server handlers, node_cache_identity, workflow_auto_resource / workflow_auto_lifecycle, existing executor and component manager                            |
| Persistence and services           | Existing flow/Studio/Block stores and lowering, service_package / service_api / service CLI                                                                        |

Before editing, follow the mirrored [engineering procedure](cluster-engineering-lessons.md).
Add failing reproductions for actual defects; run focused checks during iteration,
then complete shared gates at stable checkpoints. Do not repeatedly rerun unrelated
expensive inference after documentation changes; use an explicit impact matrix.

Backend baseline from the backend repository:

```bash
uvx --from ruff==0.12.7 ruff check . --select E9,F
uv pip check --python .venv/bin/python
./scripts/with-runtime-env.sh .venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error
./scripts/with-runtime-env.sh .venv/bin/python -m pytest -q
```

Verified already-approved optional runtime, separately:

```bash
./scripts/with-runtime-env.sh .venv/bin/python scripts/test_reviewed_optional_runtime.py -q tests
```

Client from the client repository:

```text
npm run check
npm run check:ui
npm run check:acceptance
```

Windows uses the corresponding managed `.venv/Scripts/python.exe` commands.
Review current CONTRIBUTING instructions before executing these gates. Hardware
tests use isolated paths/ports set before any server or supervisor construction.
Do not run runtime cleanup suites concurrently with live model/offload execution.
GPU runs are serialized unless concurrency is the explicit measured scenario.
Native UI tests must use real gestures; read-only state inspection is allowed,
but mutating stores to conceal a failed gesture is not evidence.

## Release decision

Implementation complete, all-model execution complete, output quality reviewed,
Windows memory qualified and release-ready are separate conclusions. A milestone
may close within its explicit scope, but outstanding required model rows, active-run
authoring failures, undelivered outputs or failed release gates remain open.
Publish both feature branches with exact paired revisions and the ledger summary;
never change evidence or silently reduce support to make the aggregate pass.
