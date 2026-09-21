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
| W1  | Baseline, artifact classification and executable test ledger | In progress: inventory/path ledger       | W0                              |
| W2  | Workspace switch, independent memory labels and migration    | In progress: controls and persistence    | W1                              |
| W3  | Creator entry and Developer Workflows modal                  | Complete: entry flows                    | W2                              |
| W4  | Unified Nodes library and contextual inspector               | Complete: authoring acceptance           | W2                              |
| W5  | Generic model/task changes and workflow modification         | In progress: loader authoring            | W3, W4                          |
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
- [x] Migrate old Auto view -> Creator and Expert view -> Developer once. Preserve
      independently stored resource settings and legacy graph/API representations.
- [ ] Preserve graph values/topology, viewport, collapsed state, queue, trust and
      component owners during workspace changes and across reload.
- [x] Retain per-workspace panel preferences without letting imported workflows
      change the user's workspace. Expose Automatic/Custom memory in both.
- [ ] Replace outdated user-facing copy without renaming upstream class names,
      serialized identifiers or historical evidence.

Acceptance: native switch tests cover old/new storage, both memory policies,
multiple tabs, loaded models and active runs. Compare execution graph and effective
settings before/after; verify no load/encode work triggered by presentation alone.

Implemented W2 foundation: keyboard-accessible Creator/Developer control,
Automatic/Custom memory labels, compatible preference migration, and per-workspace
panel restoration. Unit and native browser checks cover migration, graph/value
preservation and memory-policy independence. Full W2 acceptance remains open for
loaded models/active execution and the remaining audience-control audit. W4 now
keeps diagnostic inspection in the selected workspace; the native Run-blocked and
Fix paths are covered separately. W3 startup is complete; discovery consolidation
remains in W4.

### W3 — Entry flows and Workflows modal

- [x] Reuse the task-launcher and shared dialog primitives for the Developer
      Workflows modal; route creation through operation capabilities/starters.
- [x] Implement task-first selection, downloaded-model ranking, graph/input preview,
      Empty/Open/recent workflows and explicit missing-dependency actions.
- [x] Keep Creator's Templates/recent/empty path and preserve authored template
      layout. Do not force compact Blocks or a full-screen parameter form.
- [x] Add appropriate output nodes and validate actual required component wiring.
      Keep task distinctions such as image-to-image versus instruction editing.
- [x] Cover startup restoration, Escape, keyboard navigation, small viewports,
      retry, double click, stale responses and switching tabs while resolving.

Acceptance: new Developer starts at Workflows, never Templates automatically.
Text-to-image, image-edit, image-to-image and a non-image task create the exact
visible executable graph. Opening/selecting tasks performs no model allocation.

### W4 — One Nodes library and inspector

- [x] Consolidate canonical discovery, utilities, installed custom nodes and Saved
      Blocks under one searchable library. Keep distinct saved revisions.
- [x] Support click/drag insertion, canvas search and dangling-link suggestions
      in both directions with the same connection validation as the canvas.
- [x] Retain legacy search aliases and actionable access to distinct implementations
      without presenting duplicates as separate generic operations.
- [x] Preserve registry aliases in canvas search; share Saved Block name, revision
      and historical-name search with the library.
- [x] Add Saved Blocks through canvas search and typed suggestions in either
      direction, with fresh instances, final canvas validation and one-step Undo.
- [x] Share transient pipeline/task selection and additive discovery filters between
      sidebar and canvas search in both workspaces; resolve bound nodes on insertion
      and cancel pending requests when selection or the destination document changes.
- [x] Drag bound library nodes onto the canvas or into expanded nested Blocks in
      both workspaces, retaining exact contracts, one-step Undo and reload; cancel
      resolution when its selection, document, catalog or destination changes.
- [x] Include reviewed catalog Blocks and upstream implementations in canvas search;
      use exact compiled public ports for typed suggestions and one atomic insertion
      transaction, with retry and cancellation in both workspaces.
- [x] Use the selected node/Block inspector for Parameters, Interface, Implementation,
      Docs and Run details. Developer emphasizes diagnostic/source tools.
- [x] Keep nested graph editing, public sockets, output previews and ordinary node
      behavior shared across both workspaces. Do not confuse Python blocks with
      editable graph compositions.

Acceptance: library and canvas tests cover built-ins, enabled custom nodes, saved
compositions, type aliases, unsupported connections, keyboard use and legacy load.
Inspecting metadata never mutates execution or assumes the library was already open.

W4 inspection checkpoint: one shared inspector now serves ordinary nodes,
registered/saved Blocks and enabled custom nodes through the Studio side panel and
canvas dialog in both workspaces. The operation-only inspector is removed.
Parameters reuse the canvas controls; metadata sections show declared sockets,
connections, implementation/source identities, retained settings, help and current
run diagnostics. Opening details sends no field actions and grants no code consent.
Run-blocked and Fix actions focus the affected node without switching workspace or
memory policy. Native checks cover unchanged graph contents, keyboard/Escape focus,
parameter edits/Undo, Block controls and approved custom source metadata. A cold
production Qwen workflow passed inspection in both workspaces, edit/save/reload and
narrow layout without inference. The complete model campaign remains W8.

Checkpoint validation: `npm run check` passed; `npm run check:ui` passed all
174 cases (2 shared controls and 172 Studio cases). The backend base gate passed
3,160 tests and 9,767 subtests, with 510 skips; lint, dependency and preflight checks
passed. The published bundle matches all 78 client build files; regenerated
coverage ledgers changed only 18 hash fields. All 104 downloaded-model snapshot
file lists remain unchanged. These results do not complete real-model execution,
Windows memory qualification or release acceptance.

W4 canvas-search checkpoint: the canvas picker now includes registry nodes and
Saved Blocks in both workspaces. Exact registry aliases remain searchable after
contract deduplication. Saved Blocks share the library's search terms, retain
separate definition/revision identities, display the actual V2 hash suffix rather
than its shared schema prefix, and use the same instance factory as
library click and drag insertion. Suggestions read public Block sockets; insertion
uses the existing canvas connection commit, rolls back rejected connections, and
records the node and wire together for Undo/Redo. A changed workflow invalidates
an open search's insertion context. Browsing never enables custom source.

Canvas-search validation: `npm run check` passed with unchanged bundle limits.
The complete browser campaign passed 2 shared-control cases and 175 Studio cases;
one SDXL task-change case encountered a correctly rejected stale preview. Its
test-only settled-source synchronization was corrected, and both model-change
cases passed three consecutive native repetitions (6 passes). The final 4 Saved
Block cases also passed with keyboard/click selection in both workspaces. The
backend base gate passed 3,160 tests and 9,767 subtests, with 510 skips. Cold
production checks used the existing 160 Saved Blocks and verified insertion,
revision labels, Undo/Redo, save/reload and exact served bytes without inference.
All 78 build files match the backend; ledgers changed only 18 hash fields; all
104 downloaded-model snapshot file lists remain unchanged. Original failures
remain in review evidence. Concurrent metadata authoring and the model-execution
campaign are still separate pending acceptance.

W4 shared-library checkpoint: the four catalog tabs are replaced by one common
library in Creator and Developer, with independent options to include implementation
and experimental entries. Generic nodes stay visible before selecting a pipeline.
A bound operation suppresses only the exact canonical registry contract and its
aliases for the selected pipeline/task; distinct schemas and saved revisions remain
separate. Sidebar and canvas searches retain historical aliases and share pipeline,
task and filters. Bound canvas insertion uses the existing resolver, provenance,
connection validation and atomic Undo; stale resolutions cannot enter another
document. A declared editable value control becomes an input socket when a new
bound node is inserted from a compatible wire. Binding constants, hidden/disabled
controls and signal fields remain protected. The pipeline/task dropdown inside
canvas search now opens above its owning popover. These are discovery preferences, not execution or code consent.

Shared-library validation: the complete browser campaign passed 181 cases (2
shared controls and 179 Studio cases) on frozen source. A subsequent targeted
regression reproduced a declared prompt input that was not exposed as a socket.
After that correction, 12 affected browser cases passed, followed by 2 final
compatibility checks; the final `npm run check` passed. Backend validation passed
3,160 tests and 9,767 subtests with 510 skips; final bundle-contract checks passed
60 tests and 848 subtests with 1 skip. Native production checks used real backend
contracts and the existing 160 Saved Blocks, covering bound component/value
connections, pipeline/task dropdowns, Undo/Redo, reload and shared discovery.
Served bytes match the build; all 81 bundle files match, the ledgers changed only
18 hash fields, and all 104 model snapshot file lists remain unchanged. This
increment includes no model-generation or Windows-memory qualification.

W4 bound-drag checkpoint: bound library rows now support native drag in both
workspaces. A single-use local gesture identifies the backend-declared operation;
the existing resolver and ordinary node factory retain its defaults and provenance.
The shared drop path adopts it into the deepest expanded Block through the existing
ownership/history transaction. Selection, workflow, catalog and destination changes
invalidate pending metadata; a collapsed destination cannot redirect a delayed node
into the outer graph. Errors leave no orphan and permit retry. Resolution code is
lazy-loaded, retaining the existing startup and deferred bundle budgets.

Bound-drag validation: `npm run check` passed. All 16 focused native browser cases
passed on frozen product source, including Qwen and Anima root insertion, nested
insertion in both workspaces, unchanged siblings/source definitions, Undo/Redo,
reload, delayed selection/workflow changes, failed resolution/retry and destination
collapse. Existing catalog drag, click insertion, Saved Block typed insertion and
prompt-socket regressions passed in that same campaign. Initial test-harness and
bundle-budget failures remain in review evidence. The backend baseline passed
3,160 tests and 9,767 subtests with 510 skips; after bundle publication, affected
contract checks passed 60 tests and 848 subtests with 1 skip. All 81 bundle files
match; dependent ledgers changed only 18 hash fields. The 104 recorded downloaded
model snapshot file lists remain unchanged. These are authoring/contract checks,
not model-generation qualification. Production browser checks against the restarted
backend also passed Qwen and Anima bound drag, exact defaults/provenance, Undo/Redo
and reload across both workspaces. Served entry/library bytes match the build;
no generation, installation or download was requested.

At the bound-drag checkpoint, catalog search insertion and remaining shared
editing acceptance were still open.

W4 catalog-search checkpoint: canvas search now includes reviewed task Blocks
and the same upstream leaf/container implementations as the sidebar. Normal
search shows graph-qualified task Blocks; Show implementation nodes reveals
composable upstream entries and catalog-only structural Blocks. Distinct Saved
Block revisions remain distinct. Typed suggestions read exact public interfaces
from a compact backend index, verify all source/admission/compiled pins, and use
the actual leaf node sockets. Containers without public sockets remain available
through plain search. No internal socket or model-family adapter is invented.

Insertion uses the existing hash-verified factories and canvas connection commit.
The node and wire share one Undo/Redo transaction; a failure leaves the picker and
original connection available for retry. Escape and workflow/discovery changes
cancel pending insertion. Library, picker and factory callers share metadata
requests. Metadata browsing does not load models or enable source code.

Catalog-search validation: the full client gate passed. All 19 focused native
browser cases passed on frozen product source, covering both workspaces, both
wire directions, keyboard/click, saved revisions, exact catalog identities,
upstream leaves/containers, catalog-only Blocks, Undo/Redo/reload and delayed
failure/cancellation. Contract checks cover the public interfaces of all 122
compiled admissions; they do not establish model execution. The backend gate
passed 3,161 tests and 9,767 subtests with 510 skips. Final bundle checks passed
60 tests and 848 subtests with 1 skip. All 82 published build files match;
regenerated ledgers changed only 18 hash fields. Startup remains within its
existing budget; this feature adds 1.8 KiB of deferred code, bounded at 222 KiB.
All 104 downloaded-model snapshot file lists remain unchanged. Original failures
and corrected test fixtures are retained in review evidence.

Cold production checks against the real backend passed Qwen catalog insertion in
Creator and Anima insertion in Developer from an image input, with exact public
wiring, atomic Undo/Redo, explicit Save as and reload. The canonical qualified
entry remains distinguishable from existing Saved Block revisions. Served build
bytes match. These checks requested no generation, installation or download.

W4 shared-editing acceptance is complete. The detailed Saved Block lifecycle,
collapsed-control layout, connected/disconnected adoption, registered public-port
editing, protected replacements and nested preview-interface cases now run in
both Creator and Developer. Each case additionally switches workspaces and back,
comparing the full authored nodes/edges and independent memory policy. The tests
retain their existing Undo/Redo, reload, unchanged-sibling/source, invalid-edit,
geometry and visible-connector assertions. No separate workspace renderer or
runtime behavior was needed.

Validation: all 12 expanded native cases and the full client gate passed. A cold
production Qwen authoring check passed prompt editing, expanded public sockets,
workspace switching, root and nested interface inspection, explicit Save as and
reload. The production build remains byte-identical across all 82 files. Both
READMEs now describe the implemented entry flows; obsolete Nodes → Stages guidance
is removed, and service export uses Developer workspace terminology. All 104
recorded model snapshot file lists remain unchanged. No inference or generation
quality is claimed by this authoring checkpoint; W2 loaded-model/active-run
acceptance, W5–W10 and Windows memory qualification remain open.

### W5 — Model/task changes and editable workflows

- [x] Put model/task changes on the owning top-level generic loader in both
      workspaces, reusing current graph-change preview/apply machinery and one
      Undo/rollback transaction.
- [x] Protect existing Block route changes from stale workflow/graph replies,
      cancellation and active gestures; validate crossing wires against the actual
      restored draft interface before the atomic history transaction.
- [x] Preserve generic operation metadata and shared input relationships through
      Block projection, nested adoption and reuse; update hidden shared controls
      atomically and retain independent instance seed identity.
- [x] Add owning-Block preview/apply for graphs of generic nodes through existing
      instance reducers and the canvas transaction, preserving nested ownership,
      definitions, compatible interfaces and crossing connections.
- [ ] Complete model/task replacement for existing upstream Modular compositions
      and the full interface/modification matrix; retain their current editing path.
- [x] Preserve compatible outside inputs during generic Block task changes;
      extend shared input connections from their existing source without changing
      public port identities, and reject competing sources or sealed destinations.
- [ ] Make compatible model changes retain inputs and adaptable fields; preserve
      unsupported values without execution. Handle disconnected and shared loaders.
- [x] Resolve the observed Automatic task-identity gap for recognizable generic
      Modular operation graphs. Match existing reviewed operations/state edges and
      map task aliases through their reviewed workflow to existing resource modes.
      SDXL text-to-image → image-to-image now runs under Automatic memory.
- [x] Record recognizable model-owner tasks separately from consumed values and
      apply the correct task label in output history without rewriting saved forms.
      Ambiguous/partial compositions do not gain a recognized task or resource proof.
- [ ] Extend task recognition/acceptance to remaining upstream compositions and
      ambiguous graph signatures; one SDXL run does not qualify every task/model.
- [x] Correct stale model display labels and numeric-string summaries in history.
      Gallery uses captured identities, separates custom pipeline filters and
      displays bounded decimal strings without changing receipts or saved forms.
      Creator/Developer browser checks and the persisted live SDXL output verify
      the model, task, seed, steps and guidance after reload.
- [ ] Add required image/mask/reference operations through actual task contracts.
      Respect dimensions, shared generator identity and upstream state writers.
- [ ] Preserve custom nodes and explicit diagnostic edges when no safe automatic
      adaptation exists. Reject stale previews rather than overwriting edits.
- [ ] Qualify create/modify/save/reinsert/execute on generic nodes and nested Blocks,
      including crossing connections, selected-Block execution and legacy instances.

Acceptance: the modification matrix below passes in contract/native tests before
expensive model qualification. Selected paths additionally execute with real models;
graph-shaped JSON or a successful inspector is not execution evidence.

W5 owning-Block checkpoint: generic operation graphs expose model/task changes on
collapsed Blocks and expanded loader inspectors. The shared planner updates one
instance, retains semantic IDs even when implementation classes change, and uses
the existing Undo/rollback transaction. Incompatible public bindings and bound
values are rejected before mutation. Outside inputs are included as read-only
sources in the existing planner, so they
keep precedence over starter connections. Added shared consumers receive reviewed
crossing wires from the same source. Competing sources, sealed controls and
incompatible bound fields still stop the preview without changing the workflow.
This does not claim that
arbitrary upstream Modular compositions can be switched through the generic path,
or qualify any model execution. Remaining W5 modification and execution rows stay
open.

W5 loader checkpoint: the selected top-level generic loader now exposes
**Inspect node → Parameters → Change model / task**, also available in the Studio
side inspector. Replacement choices are local to that owner and use the backend's
declared pipeline/task contracts. Preview and Apply reuse the existing graph
planner and history transaction; no second graph representation, executor or
family dispatch was introduced. The library entry point remains available.
Nested and legacy Blocks retain their existing composition tools and are not
silently converted by this control.

The native acceptance covers both workspaces, Qwen → Flux, text-to-image →
image-to-image, prompt retention, preview cancellation, one-step Undo/Redo,
explicit Save as and reload. Delayed replies are discarded after a target change,
inspector close or intervening graph edit, and a fresh request can succeed. The
new flow exposed and fixed a stale toolbar inspector owner that otherwise reopened
its dialog after graph replacement and reselection. SDXL shared-seed and keyboard
inspection regressions were also repeated. An initial SDXL attempt rejected a
stale preview; the strict guard remains, and wider asynchronous-publication
acceptance is still part of W5/W7 rather than treated as proven by later passes.

Packaged-browser acceptance uses actual backend metadata for the same Qwen →
Flux → image-to-image sequence, workspace switches and save/reopen. This is
**authoring evidence, not model execution**. The complete client gate and affected
backend bundle/ledger contracts are required for publication. The remaining
Block-owned switching, modification matrix, W8/W9 model runs and deferred Windows
memory acceptance remain unchecked.

W5 Block transaction checkpoint: existing saved route-selection Blocks now
capture their workflow and graph before asynchronous save/metadata work and reject
obsolete replies. Cancel remains available while the destination loads; an
unmounted owner aborts its request. Switches cannot merge into an unfinished
canvas gesture. The route reducer validates connected ports against the actual
restored draft, including its customized interface, rather than the destination's
registered defaults. Rejected changes preserve the current graph and Undo history;
accepted switches retain one Undo/Redo step.

Seven contract cases cover workflow replacement, intervening edits, cancellation,
active gestures, normal commit/Undo/Redo and both preserved/removed custom output
ports. Native browser cases use exact backend Qwen/Flux catalog definitions in
Creator and Developer, with one-time fixture setup representing an existing saved
route-selection workflow. They cover Cancel/retry, prompt retention, both draft
and Save-active-route actions, Undo/Redo and Save as/reload. A packaged browser
check against the live backend validates the same compatibility path without
inference. The full client gate stays within its existing bundle limits.

This fixes the existing switch transaction; it does not extend the older fixed
route list or claim generic nested Block/task switching. That implementation and
all-model execution remain open. New modelVariant Blocks keep their current
checkpoint selector, and no source definition or downloaded model is modified.

W5 shared-input checkpoint: generic operation hints now survive visible and
execution projections of Blocks. Loader references follow exact semantic IDs when
a Block is adopted into another Block and receive the destination instance prefix
when projected. Shared controls use the existing field/value reducers and history;
root, nested-container and leaf edits update hidden members together. Sealed
controls reject the whole edit, competing loaders do not gain shared ownership,
and conflicting shared values still block execution export. Immutable definitions
remain unchanged. This is a prerequisite for nested model/task changes, not their
completion. Model execution qualification remains in W8.

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
| Workspace / entry / discovery      | Client settings and Studio stores, TopBar, DeveloperWorkflowLauncher, WorkflowEntryActions, TemplateBrowserDialog, NodeList                                        |
| Canonical creation and changes     | OperationCatalogPanel, OperationGraphControls, operationAuthoring, operationGraphTransaction, backend operation_catalog / operation_starters / operation_contracts |
| Dynamic controls and compatibility | Registry field actions, ModelSelectField, shared connection matching, operation schemas                                                                            |
| Context and extensions             | GraphNodeInputs, NodeInspectorDialog, NodeInspectionDetails, CustomExtensionsPanel, backend custom_extensions / custom_extension_api                               |
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

### W3 Developer entry implementation checkpoint — 2026-09-20

Implemented the task-first **Workflows** chooser using backend operation support,
with model search and downloaded-file ranking, required-input and connected-node
preview, Empty/Open/recent actions, and separate setup/model-file actions.
Creation uses ordinary nodes, typed image/audio/video outputs and the existing
atomic graph transaction. Existing work restores before a chooser can open.
Dismissal is persisted per workflow, including empty workflows; explicit New
workflow offers the chooser again.

The starter API now accepts an optional exact execution profile. This fixes the
case where selecting two models sharing a pipeline class could create the same
default loader. Metadata tests exercise all 275 advertised profile/task pairs,
including immutable repository revisions, without constructing model nodes.
Six focused native browser tests passed for image generation, image-to-image,
instruction editing, audio, reload, small viewport/keyboard/Escape, retry,
double-click and delayed-response cancellation. These tests use backend-generated
schemas and mocked transport; they are **not model execution qualification**.

At this checkpoint, W3 remained in progress: Creator entry and the remaining
acceptance cases were still open; see the shared-entry checkpoint below. W8/W9 real-model and modification
campaigns remain unstarted for this workspace redesign. Existing execution
receipts do not qualify the new chooser or prove all downloaded models work.

The real production-browser check exposed a rapid Qwen → FLUX navigation
failure: pending field-schema HTTP calls and a signal lookup to a departed
browser delayed reload. The regression fix aborts nonqueued client waits when
their document changes or navigation begins, and resolves backend signal futures
for the disconnected session. Queued action acknowledgements, unrelated sessions
and the executor's ownership lease remain intact. Regression tests cover each
boundary. This is a partial recovery fix; it does not close W7's active-inference
latency, memory reuse or OOM requirements.

After that fix, native production-browser checks passed against the real backend
for Qwen modular image nodes, FLUX schnell whole-pipeline nodes and Stable Audio:
exact profile selection, preview, ordinary connected outputs, creation and reload.
Served entry/chooser JavaScript matched the built files. No inference, install or
download requests were submitted. All 104 downloaded model snapshot file lists
remain unchanged.

Validation for this checkpoint:

- Client `npm run check`: passed, including final build and bundle budgets.
- Shared-control browser suite: 2 passed. Full Studio suite: 166 passed and one
  notification test retained two old expectations that revisiting a dismissed
  empty tab reopened its chooser. Both assertions were updated for the accepted
  per-document dismissal behavior; the complete notification test then passed.
  All seven new Developer Workflows cases passed in the full run, including
  recovery through the existing startup-error dialog and capability retry.
- Backend full base gate: 3,160 tests and 9,767 subtests passed; 510 skipped.
  Ruff, dependency compatibility and preflight passed. Final regenerated bundle
  ledgers passed 59 focused tests and 817 subtests, with one skipped; only hash
  fields changed in those ledgers.
- Production browser: Qwen modular, FLUX schnell and Stable Audio model selection,
  connected output creation and reload passed with real backend metadata and
  exact served-build hashes. No model inference occurred. Test-created workflow
  documents were archived with the traces and removed from My Workflows.

The optional-runtime execution suite, all-model inference/modification campaign
and Windows memory qualification were not run for this checkpoint. Keep their
milestones open.

### W3 shared entry implementation checkpoint — 2026-09-20

Creator now opens the existing Templates browser directly, with shared Empty,
Open and recent-workflow actions. The separate task grid is removed. Developer
retains its task-first Workflows chooser and can deliberately browse Templates.
Both paths keep the same graph document, node factory and execution system.
Template creation dismisses the chooser for its new document before asynchronous
preparation, so it cannot flash a second startup modal while building the graph.
On narrow screens, expandable filters preserve room for template cards and show
an active-filter count; desktop filters remain visible.

The real-backend reload check reproduced a startup deadlock: late authoritative
hydration invalidated the in-flight memory plan without changing its visible form.
Planning now waits for hydration and tracks document/form epochs, retaining the
stale-response guard while retrying for the restored owner. A focused browser
test reproduces the original stuck gate and passes after the fix.

Focused native tests pass for Creator startup, keyboard Empty, narrow viewports,
recent/Open actions, template creation and unchanged graphs across workspace
switches. A delayed Developer preview is abandoned on Escape/document change;
its eventual response cannot replace another document's preview or graph.
Existing managed-graph regression tests seed their legacy documents explicitly;
that fixture setup is not counted as native entry-flow proof.

W3 is complete for entry flows and graph-authoring contracts. Validation:

- Client `npm run check` passed, including lint, type checking, unit/contract
  tests, the production build and existing bundle limits.
- Shared controls: 2 passed. Full Studio browser run: 170 passed; one model-change
  test captured a late schema/validation publication after its preview. Its setup
  now waits for a settled source document; all original preservation assertions
  passed in three consecutive native reruns. The graph ownership guard remains
  unchanged. All 173 unique browser cases are covered across these runs.
- Backend base gate: 3,160 passed, 510 skipped, 9,767 subtests passed; Ruff,
  dependency checks and preflight passed. After the final bundle refresh, the
  dependent ledger tests passed: 59 passed, 1 skipped, 817 subtests. All 18 ledger
  changes are fingerprints; no qualification status was promoted.
- Native production checks passed for Creator template creation, workspace
  switching, reload, recents, keyboard Empty and narrow-screen browsing, plus
  Developer Qwen modular, FLUX schnell and Stable Audio creation/reload. Served
  assets matched the final build. Reload comparisons retain all authored values,
  positions and connections while normalizing refreshed callback metadata and
  materialized defaults.
- All 104 downloaded model snapshot file lists remain unchanged. No inference,
  installation or model download was submitted. Sixteen verified smoke-created
  workflow documents were archived and removed from My Workflows.

W1/W2 remaining acceptance, W4 onward and W8/W9 real-model/modification campaigns
remain open. These checks do not qualify model execution or the deferred Windows
Qwen memory target. A W5 follow-up should surface stale model-change preview errors
inside the review dialog while preserving its ownership guard.
