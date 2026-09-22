# Creator / Developer workspaces implementation plan

## Immediate delivery: image demo

The current priority is a two-hour image-demo checkpoint. Qualify a representative
image workflow on the upgraded runtime, exercise native prompt/seed edits,
reopening and service export, and test Qwen 2.1 attention reuse with real weights.
Keep a previously proven image pipeline available as a fallback. Publish only
verified behavior with its exact limits. The exhaustive image/audio matrix below
is deferred beyond this demo checkpoint; W8/W9/W10 are not thereby complete.

- [ ] Qualify and retain a working image demo workflow on the upgraded runtime.
- [ ] Run full-weight Qwen 2.1 and inspect cache on/off behavior and real output.
- [ ] Verify demo browser flows and served production bundle; preserve user/model data.
- [ ] Push the tested paired revisions and document the shortest demo sequence.

## Current completion scope: image and audio

W8, W9 and W10 now target image and audio models, nodes and tasks. Resume the
remaining image campaign and audio coverage. Defer video generation, video edits,
video upscaling and video-specific release qualification; preserve their existing
results and downloaded artifacts. Deferred video rows must not become passing
skips or disappear from the ledger. Windows Qwen memory qualification remains H1.

- [ ] Review and pin the current Diffusers/Modular Diffusers snapshot
      `fbf49e7f35857f76bc57b177e26f12b03687c668`, including required dependency changes,
      component lifecycle compatibility and affected image/audio contracts.
- [x] Add Qwen-Image 2.1 through existing generic image operations: generation,
      single/multiple reference editing and alpha-preserving output. Bind the
      reviewed model revision `790c92633540aa0cb11d9abf19eb46d861714758`. Adapter
      and tiny native denoiser/VAE tests pass; full-weight acceptance is tracked below.
- [ ] Expose and capture upstream context KV caching, verify per-run cache lifetime,
      changed prompt/reference invalidation, cache on/off output validity and
      measured speed/memory. Do not require cross-setting pixel identity or claim
      the illustrated A100 benchmark on another accelerator.
- [x] Keep Qwen 2.1's ordinary pipeline route distinct from native Modular support:
      the reviewed upstream snapshot has `QwenImage21Pipeline` but no native
      Qwen 2.1 Modular pipeline. Do not invent a Modular compatibility claim.
- [ ] Finish applicable W8 image and W9 audio model/task/modification ledgers on
      the reviewed runtime, with a dependency-impact record for earlier results.
- [ ] Finish W10 scoped quality/browser/acceptance gates, production bundle and
      derived catalogs, README/support documentation, preservation and paired pushes.

Implementation checkpoint: the managed Linux AMD base upgrade and isolated
Transformers 5.17 profile qualification passed. The verified optional-runtime
suite passed 4,085 tests and 10,820 subtests (21 skips). All 122 registered Block
routes recompile with unchanged public inputs, outputs and controls. Original
model-file metadata and operator approval preservation checks pass. These checks
do not close the pending live model/task matrix or release gates.

Qwen 2.1 weights carry the Qwen Research License; runtime integration and model
weight licensing are separate. Do not publish model weights or generated Gallery
assets as part of registering its generic node support.

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
| W5  | Generic model/task changes and workflow modification         | Complete: W5 acceptance                  | W3, W4                          |
| W6  | HF/local custom-node workflow                                | Complete                                 | W4, W5                          |
| W7  | Concurrent authoring, reuse and recovery                     | Complete                                 | W5, W6                          |
| W8  | All-local-image execution and modification campaign          | In progress: image scope resumed         | Stable W2–W7 build              |
| W9  | Other modalities and real service-export campaign            | In progress                              | Stable W2–W7 build; W8 fixtures |
| W10 | Release acceptance, documentation and publication            | In progress: upgrade gates               | W8, W9                          |
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
- [x] Complete model/task replacement for existing upstream Modular compositions
      and the full interface/modification matrix; retain their current editing path.
- [x] Preserve compatible outside inputs during generic Block task changes;
      extend shared input connections from their existing source without changing
      public port identities, and reject competing sources or sealed destinations.
- [x] Make compatible model changes retain inputs and adaptable fields; preserve
      unsupported values without execution. Handle disconnected and shared loaders.
- [x] Resolve the observed Automatic task-identity gap for recognizable generic
      Modular operation graphs. Match existing reviewed operations/state edges and
      map task aliases through their reviewed workflow to existing resource modes.
      SDXL text-to-image → image-to-image now runs under Automatic memory.
- [x] Record recognizable model-owner tasks separately from consumed values and
      apply the correct task label in output history without rewriting saved forms.
      Ambiguous/partial compositions do not gain a recognized task or resource proof.
- [x] Extend task recognition/acceptance to remaining upstream compositions and
      ambiguous graph signatures; one SDXL run does not qualify every task/model.
- [x] Correct stale model display labels and numeric-string summaries in history.
      Gallery uses captured identities, separates custom pipeline filters and
      displays bounded decimal strings without changing receipts or saved forms.
      Creator/Developer browser checks and the persisted live SDXL output verify
      the model, task, seed, steps and guidance after reload.
- [x] Publish task-required media on generic operation ports and block missing
      image/mask/reference inputs in the resolved execution scope, including
      unexposed Block internals and stale saved declarations. Both workspaces and
      memory policies retain targeted, distinct Fix findings without inventing inputs.
- [x] Correct conditional Preview VAE suggestions in Fix: an image input should
      not be diagnosed as missing the latent decoder, nor offered an unrelated
      wildcard source. Observed during packaged required-media acceptance.
- [x] Add required image/mask/reference operations through actual task contracts.
      Respect dimensions, shared generator identity and upstream state writers.
- [x] Preserve custom nodes and explicit diagnostic edges when no safe automatic
      adaptation exists. Reject stale previews rather than overwriting edits.
- [x] Qualify create/modify/save/reinsert/execute on generic nodes and nested Blocks,
      including crossing connections, selected-Block execution and legacy instances.

Acceptance: the modification matrix below passes in contract/native tests before
expensive model qualification. Selected paths additionally execute with real models;
graph-shaped JSON or a successful inspector is not execution evidence.

W5 implementation: generic loaders and their owning Blocks expose **Change model /
task** in both workspaces. Choices come from backend pipeline/task contracts and
exact declared model profiles. Compatible values, literal media and outside wires
survive a change; shared consumers use the same supplied seed. Unsupported values
remain recoverable outside execution. Preview/Apply uses one existing graph/history
transaction, with cancellation and rejection of stale replies or active gestures.

Upstream Modular compositions expose a workflow replacement chooser through the
existing catalog/compiler. A replacement keeps the exact edited composition in a
bounded set of eight inactive drafts; restoring a draft recovers custom nodes and
connections. This is distinct from pretending arbitrary internals can be translated
across models. Saved Blocks can be reinserted, edited and explicitly updated without
silently modifying their reusable definition. Legacy route-selection instances
retain their compatibility path.

Required image, mask and reference inputs are checked against executable ports and
connections, including nested/unexposed inputs. Fix no longer asks for a VAE when
Preview already receives an image, or proposes an arbitrary wildcard connection.
Run Block honors transitive outside inputs and required internal siblings while
excluding downstream outputs and unrelated drafts; reusable saving remains scoped
to contained nodes. The live nested Qwen case exposed and regression-tested this
selected-run dependency correction.

Backend contracts share the published task adapter definitions. Exact components,
state edges and reviewed upstream placements identify tasks; partial/ambiguous
compositions remain unresolved. A loader projection supplies the required inpaint
VAE connection. Split decoding can forecast explicit denoiser dimensions during
Automatic planning, with the existing executor checking actual connected dimensions
before allocation. No frontend family switch or separate executor was added.

Validation distinguishes authoring from execution:

- The no-download modification matrix covers all 86 published Modular task starters
  in ordinary and Block form. A further 81 directed model/task changes check public
  inputs, exact drafts and preserved custom graph edits. These are contract proofs.
- Native Creator/Developer checks cover owning loaders/Blocks, explicit model
  profiles, required media, shared outside seed, nested interfaces, stale previews,
  Undo/Redo, legacy routes and upstream draft restoration. Browser tests also inspect
  selected-run exports after native media wiring and Save as/reload.
- Real packaged-browser runs cover the following downloaded models and paths.
  Inputs, submitted graphs, source identities, receipts, screenshots and output
  hashes are retained in ignored review storage; output images are inspected.

| Live path                           | Modification and persistence exercised                                                                              | Parameters / memory                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| SDXL generic inpaint                | T2I → inpaint; connect image/mask and shared external seed; save/reload; run and inspect history                    | 1024², 32 requested steps, guidance 6.5, strength 0.62, seed 7319; Automatic                       |
| Flux generic image-to-image         | T2I → img2img; image, prompt, guidance and external denoiser seed; save/reload                                      | 1024², 32 steps, guidance 3.5, strength 0.62, seed 7319; Automatic                                 |
| SDXL whole Diffusers image-to-image | Change whole pipeline/task; image, prompt/negative prompt and connected seed; save/reload                           | 1024², 32 steps, guidance 6.5, strength 0.62, seed 7319; Custom, bfloat16, CUDA, model CPU offload |
| Reviewed upstream Flux composition  | Switch from Qwen; save Block, delete/reinsert, edit, explicitly update, save/reload and Run Block                   | 1024², 30 steps, guidance 3.5, seed 42; Automatic                                                  |
| Qwen generic multi-reference edit   | T2I → multi-reference edit; connect both encoders, shared seed, save/reload and Run Block; outside Preview excluded | 1024² output, two references, 30 steps, guidance 4, shared external seed 7319; Automatic           |

The complete backend gate passes **3,188 tests and 9,864 subtests** (510 skips),
with 75 additional optional resource/task checks. The complete client gate passes,
including its bundle limits; six focused native media/seed regressions pass after
the selected-run correction. Other W5 native cases and exact run receipts are
recorded in the completion review. All 104 downloaded model snapshots remain intact.

The final Qwen run completed in approximately 31 minutes. Its execution receipt
records both references, seed 7319 at both consumers and the exact pinned model.
Qwen Edit Plus declares the same action/input signature for `edit_image` and
`multi_image_reference_edit`, so the captured task label deliberately stays unset
instead of guessing; the pipeline identity and actual two-image inputs are recorded.

Scope limits: standard Diffusers SDXL image editing has no declared Automatic
recipe for this exact path, so its live proof uses explicit Custom memory. This
does not qualify every model or every modification on each model. W8's mandatory
per-model I01–I12 campaign, W6/W7 custom development and concurrent authoring,
W9 service export/other modalities, W10 release acceptance, and H1 Windows Qwen
16 GB VRAM / 32 GB RAM remain separately tracked. Windows memory acceptance is
deferred until the main UI/UX work is ready.

### W6 — Custom-node development

- [x] Provide Add from Hugging Face and local-source entry points from Developer
      without requiring a template or existing custom-node placeholder.
- [x] Resolve user-entered repo URL/ID to an inspectable immutable revision through
      existing Hub/source mechanisms; show revision, source and dependencies.
- [x] Render existing Diffusers/Mellon UI metadata with normal registry/field tools.
      Require exact code approval before imports; importing a workflow grants none.
- [x] Show installed source location and review/reload changes. Preserve targeted
      cache invalidation and reject stale approval or missing dependencies.
- [x] Explain idle-only activation/reload without blocking source inspection during
      inference. Verify edit failures preserve unrelated graph/component owners.

- [x] Derive one Models socket after approval for Modular blocks whose sidecar
      omits model ports, and publish already-loaded Pipeline Components from every
      Modular Load Models route. Preserve declared fields, source bytes, loader
      ownership and runtime type checks; no family-specific UI or implicit weights.
- [x] Provision additional block-specific model repositories through the existing
      model/resource boundary, with explicit selection and immutable identities.
      Approved blocks with official pretrained component types expose a matching
      Load Models node. Sources are explicit, pinned and cache-only; precision,
      device and offload remain explicit. Custom memory is required for these
      unqualified suppliers. Florence's model and processor now load through this
      path; connecting an image-generation VAE alone is still insufficient.

Acceptance: an approved local Python node and an HF Modular block execute inside
existing image workflows, including a saved Block. A generic prompt/image processor
is tested across distinct model families without family-specific registration.
Missing dependency, changed helper file, failed import, cancellation and stale
approval paths retain useful diagnostics. No Python sandbox claim.

W6 implementation progress: Developer Workflows now opens the shared Custom nodes
review directly for Hub or local sources. Hub metadata resolution accepts a repo
ID/root URL or tree revision, returns an exact commit, and never stages or imports
code. Editing, cancelling or closing invalidates an outstanding lookup. The existing
exact-code approval, dependency checks and typed-node registration remain in place.
Both READMEs and the custom-node guide describe this flow and independent memory
policy.

Focused evidence: six native browser tests with mocked backend responses cover
entry points, pinned staging, typed discovery in both workspaces, late responses,
missing dependencies, stale approval and import failure with an unchanged graph.
Backend tests cover source validation, exact approvals, execution/reload boundaries,
targeted invalidation and cancellation. The current full backend gate passes 3,236
tests and 9,866 subtests, with 510 skips. The installed optional custom-source and
runtime gate passes 205 tests and 310 subtests; the final optional NodeBase/isolation
check adds a separate 28 passing tests and 2 subtests. These are overlapping focused
suites, not an aggregate coverage count. Startup counts now include enabled
custom nodes; clean-base tests isolate operator extensions, including child
registry imports. The full client gate passes. The published bundle still matches
all 87 generated files; this checkpoint changes no client executable code. The 104 originally inventoried model snapshots retain all their files.

Live acceptance completed so far:

- [x] Locally staged Python Prompt Prefix and upstream Modular Prompt execute in
      the same prompt chain with SDXL and Flux, inside saved/reopened Blocks.
      Initial compositions are fixtures from live schemas; source approval, saving,
      reopening, parameter edits and Run actions use the native packaged UI.
- [x] Six runs at 1024 × 1024, 30 steps and fixed seed 98131: baseline, unchanged
      reuse and changed custom-prefix input for each model. SDXL took 35.57 / 4.19 /
      25.81 seconds; Flux took 144.39 / 4.28 / 123.85 seconds. Unchanged runs kept
      identical image bytes and reused the resident recipe; edited inputs produced
      different images with recomputation. Original images were visually inspected.
- [x] Native source inspection during SDXL denoising completed in about 2.09 seconds
      including navigation. A real Hub lookup took 0.35 seconds while that same task
      remained active. This is a sampled interaction, not W7's full latency campaign.
- [x] Real Hub URL resolution and disabled staging/review of
      [Florence2 Image Annotator](https://huggingface.co/diffusers/Florence2-image-Annotator/tree/6d110d638eac3be25dbbab5a39667fefcd1e3f82).
      Staging during an active run returned the intended idle-queue correction and
      installed nothing. After completion, source review and cancellation succeeded
      without importing that Python. A discovered omitted `model_input_names` field
      is normalized only in the custom-source adapter; source bytes and ordinary
      declarative pipeline validation remain unchanged.
- [x] A locally staged Modular Image Reconstruction example receives a connected
      VAE through its derived Models socket, without a model-family selector.
      SDXL and Flux execute it inside saved/reopened Blocks at 1024 × 1024,
      30 steps and fixed seed. Unchanged input reuses identical image bytes;
      editing reconstruction amount reruns the custom block while reusing
      denoising. Actual AutoencoderKL dispatch, shared ownership, missing/wrong
      component types, source reload and precision restoration have regressions.
      The initial SDXL half-precision failure is retained; the example now honors
      the component's force_upcast configuration and restores its original dtype.
- [x] Verify final-backend Flux component execution, reuse and a persisted amount
      edit after fixing stale idle RAM sampling. The baseline ran in 135.6
      seconds; the final edited run took 6.1 seconds, including 2.0 seconds in the
      custom block and cached denoising. Match retained image bytes to exact run
      receipts and inspect originals. This is component execution evidence with
      a native-delivery gap: the first capture stalled after submission, so that
      baseline was recovered from its backend receipt. Unchanged reuse completed
      natively; the amount edit resumed in a fresh browser with the Workflows panel
      closed. A later Auto rejection and the original stalled capture remain
      failures to investigate, not a passed uninterrupted UI lifecycle.
- [x] Execute the real pinned Florence Hub block in a saved/reopened image Block,
      using its separate pinned model and processor. Four native runs cover an
      initial "gloves" annotation, unchanged reuse, editing to "lamp", and reviewed
      source reload with the edited workflow preserved. The input is a retained
      1024 × 1024 Flux-generated image; these runs execute annotation, not a new
      Flux denoising job. Original outputs visibly bound the requested objects.
      Image hashes are identical for unchanged reuse and for the edited output
      after reload; the prompt edit changes the image. Backend execution took
      7.18 / 3.84 / 4.03 / 4.64 seconds. Initial graph setup used live-schema
      fixtures; approval, memory selection, saves, reopening, edits, reload and
      Run were native UI gestures.
- [x] Preserve and fix the live cache-selection and root-subfolder failures:
      cached model selection must not demand unrelated repository files such as
      .gitattributes, and native Transformers loading requires an empty root
      subfolder string. Regressions retain cache-only loading, exact revisions,
      safe snapshot paths and configuration-sensitive reuse.
- [x] Isolate pytest's default extension store before test collection. Importing
      the node registry in a base test environment must not disable an operator's
      approved extension that needs the installed optional runtime. Explicit
      fixture roots remain unchanged. The cold registry subprocess is isolated too;
      the full suite must preserve the real approval file byte for byte.

W6 acceptance and checkpoint gates are complete. The final full base suite and
optional isolation checks preserved the installed extension approval file byte for
byte, including cold child-process registry discovery.
The earlier six prompt-chain image runs used explicit
component metadata before the optional-metadata compatibility correction. The
new image-reconstruction runs exercise omitted metadata and derived Models ports;
those use local staging. The separate four-run Hub campaign now verifies the
staged annotator's loading, execution and reload. One additional pinned Florence
weight snapshot was downloaded for that campaign; include it when refreshing W8's
artifact/task inventory. The original 104 snapshot inventories are preserved. These
checks do not qualify all local models (W8), Windows memory limits (H1), or the full
concurrent-authoring and service-export campaigns (W7/W9).

### W7 — Responsiveness, reuse and recovery

Status: complete. The all-local-model campaign is W8; Windows Qwen on 16 GB VRAM /
32 GB RAM remains the separate H1 qualification.

- [x] Refresh idle host-memory availability without changing runtime identity or
      probing accelerators during inference. Cover released capacity, external
      pressure and unavailable OS samples.
- [x] Keep Auto preparation failures in the Run blocked dialog until dismissed;
      retry from fresh state and ignore responses belonging to an older draft.
- [x] Page saved workflows in groups of 50 while searching the full inventory.
      Cover a 1,000-entry library, paging, empty results and narrow layouts.
- [x] Batch atomic history serialization without changing its format, and bound
      intermediate loading progress to four updates per second. Preserve failed
      serialization recovery and initial/final/component progress messages.
- [x] Complete conservative resident-owner accounting. Count shared loaders once,
      independent owners separately, and shared/unified memory as one pool.
      Validate combined retained requirements, actual capacity before scheduled
      allocations, incompatible recipes, external pressure and survivor ownership.
      Do not treat process RSS as reclaimable model RAM.
- [x] Profile saved-workflow reopening, prompt editing and exact-task result
      delivery with Workflows open. Retain the original failures; independently
      reproduce and correct unbounded row rendering, loading-progress bursts,
      history serialization and metadata connection starvation.
- [x] Isolate reviewed built-in metadata callbacks from model execution. Audit the
      public registry across Modular selectors and ordinary image/audio/video/3D
      contracts. Use presentation contexts without executable constructors,
      destructors or component ownership; clear schemas on model disconnection.
- [x] Keep queued/custom callbacks and explicit model-layer construction serialized.
      Preserve authoritative authorization, optional-runtime checks, cancellation
      draining, runtime activation and custom-source mutation gates.
- [x] Carry workflow, canvas and form ownership on every dynamic schema message,
      including legacy Dynamic Block labels/styles. Apply ownership checks before
      queue-recovery progress resolves a node by ID or label.
- [x] Treat submitted runs as immutable snapshots. Preserve next-draft edits,
      autosave, selected tabs and restored workflows while background work runs.
- [x] Verify node/component reuse and exact consumed inputs. Isolate cached RNG,
      scheduler, guider and continuation state while sharing appropriate weights.
- [x] Test native Stop/restart, durable active/queued cancellation, a real controlled
      GPU OOM and a successful same-settings native retry.
- [x] Show success/information as node status rather than warnings. Clear stale
      node error details on a new attempt; retain failed-task history.

Evidence and scope:

| Case                         | Evidence                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Concurrent native authoring  | Cold Flux inference alongside model/task previews, connected workflow creation, Nodes search, insertion/Undo, autosave, reload and tab switching. Sampled previews took 0.26–0.93 s; connected creation 210 ms; Nodes readiness 843 ms; queue p95 10.1 ms across 125 samples.                                                                                                                |
| Diagnosed latency            | The original 7.38 s prompt-edit failure is retained. A reproduced cold-loading burst dropped from 5.89 s / 451 weight messages to 76 ms / nine messages, with identical image bytes. Paging was profiled against 4,553 saved workflows. These are sampled flows, not every-model latency guarantees.                                                                                         |
| Generic Flux reuse           | A native 1024 × 1024, 30-step, bfloat16/model-CPU-offload workflow completed with a complex prompt. Unchanged execution took 4.1 s, reused all five nodes and reproduced identical bytes. Prompt-only changes retained models; seed-only changes also retained conditioning. Consumed receipts, persistent drafts and exact-task previews were checked.                                      |
| Metadata during inference    | Ordinary Flux, SDXL and Qwen image-contract HTTP callbacks completed in 3–5 ms during denoising. Contract tests cover all declared image/audio/video/3D adapters; that is separate from executing those models.                                                                                                                                                                              |
| Stop and snapshots           | Native Stop cancelled both the active and queued runs, replaced the worker, and preserved the selected workflow across reconnect/reload. A fresh workflow ran successfully afterward.                                                                                                                                                                                                        |
| OOM and recovery             | A recorded allocator-budget injection caused a real GPU OOM during prompt encoding. Removing only that injected budget allowed a native retry with unchanged model/creative inputs and byte-identical output to the preceding seed run. The final published client additionally verified decoded browser media, cleared error details and correct status labels.                             |
| Multiple owners              | A real backend API composition with two Flux owners passed an unchanged cached repeat (4.7 s) and release of one owner while preserving the survivor's cache (3.0 s). Auto ran the same composition with two actual owner releases in 260.2 s; both material output hashes matched Custom execution. This is live executor proof, not native two-owner authoring or all-model qualification. |
| Mutable state and boundaries | Regression tests cover actual pinned Modular continuation/state contracts, fresh generator attempts, shared-owner release, unknown/custom callback serialization, metadata cancellation draining and runtime activation exclusion.                                                                                                                                                           |

Final validation: 3,293 backend tests and 10,029 subtests pass, with 511 skips.
The approved optional-runtime selection passes 232 tests and 414 subtests.
All 82 coordinator tests, five focused browser regressions and the full client
quality gate pass. Ruff, dependency checks and runtime preflight pass. All 87
client bundle files match the served copy; dependent catalog changes are hashes
only and do not promote qualification. Original model inventories, additional
Florence weights and operator custom-source approvals remain preserved.

Original failed captures remain retained. Harness mistakes involving numeric-text
controls, duplicate workflow titles and fixture initialization were corrected
without relabeling the failed captures. Successful images, exact input receipts,
source identities, per-node status messages and recovery records are retained
separately. W8 still owns the complete local image-model/modification matrix;
these W7 checks do not certify every model or the deferred Windows memory target.

Acceptance: while a real model denoises, create another workflow, search/add nodes,
preview model/task changes, inspect custom sources, edit/save and switch workspace.
Measure request and gesture latency separately. Initial budgets: local interactive
read requests p95 <= 1 second; warm metadata preview <= 3 seconds; cold metadata
preview <= 10 seconds, excluding explicit network/source downloads. Capture any
breach and investigate; budgets are targets, not claims about current performance.
Keep native tests for stale responses, cleanup cancellation and queue attribution.

### W8 — Every complete local image model

- [x] Freeze a validated paired build and verify served client bytes and runtime identity.
- [ ] Finalize the W1 artifact/task ledger, including exact referenced dependencies.
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

W8 is paused at the retained checkpoint below. The inventory has 112 local snapshots and
208 preliminary artifact/task/path rows; components, adapters and custom source
snapshots are included in that inventory and are not all runnable image models.
Exact dependencies and applicable-task dispositions remain under review.

Completed execution checks within the SDXL Modular text-to-image path:

- [x] Native Developer creation, unchanged repeat, prompt/seed/step/size edits,
      save/reopen and subsequent execution.
- [x] Creator/Developer switching preserves graph contracts and values; explicit
      denoiser recomputation retains compatible upstream cached outputs.
- [x] Add and connect Resize through normal discovery, then execute at the
      requested output size.
- [x] Save the six-node graph as a Block, insert it into another workflow, edit
      its seed and execute; the saved definition remains unchanged.

These are native UI and decoded-output checks on the recorded Linux ROCm recipe,
not Windows qualification. The unchanged run reused all five nodes and identical
image bytes. Prompt edits reused the loader; seed, step and size edits reused
conditioning. Complex-prompt visual assessment remains partial: the requested
vase placement was not fully achieved. Retain that failed visual requirement
separately from successful parameter consumption and execution. SDXL's other
applicable tasks and the remaining local-model matrix are still open.

Completed execution checks within the Sana Sprint whole-pipeline text-to-image path:

- [x] Native Developer creation and four-step baseline; unchanged repeat;
      prompt, seed, guidance and size changes; save/reopen/edit/run.
- [x] Creator/Developer switching, explicit generation recomputation, connected
      Resize, and Saved Block reinsertion with an independent seed edit.
- [x] Preserve the original failed four-step attempt, fix the backend schedule
      adapter, and rerun the same workload without lowering its settings.

The baseline used bfloat16, model CPU offload, 1024 square and four steps;
subsequent edits used guidance 4.5 and 1152 × 896. The unchanged run reused all
three nodes and identical bytes. Input edits reused the loader; whole-pipeline
execution does not separately cache prompt encoding. Recomputed generation
returned identical bytes; Resize produced 576 × 448, and the saved Block
baseline stayed unchanged after its new instance ran with a different seed.
Original media was decoded and visually inspected. Complex prompt adherence is
partial, so successful execution is not full visual approval. The image-to-image
checks are recorded below; the remaining local-model/task rows stay open.

Completed execution checks within the LCM DreamShaper whole-pipeline path:

- [x] Native Developer creation, unchanged repeat, prompt/seed/step/size edits,
      save/reopen/edit/run, and Creator/Developer switching.
- [x] Explicit generation recomputation, connected Resize, Saved Block
      reinsertion with an independent seed, and exact restored Block media.
- [x] Diagnose wrong new-workflow defaults; seed new ordinary image nodes and
      starters from the selected backend profile, including shared-class aliases.
- [x] Compare the same baseline workload with pinned upstream Diffusers: decoded
      pixels are identical. The prompt fits the encoder context; weak scene
      adherence remains a model-quality limitation for this tested workload.

LCM used float32, model CPU offload, guidance 8.5 and four steps at 512 square;
subsequent edits used six steps and 640 × 384. Resize produced 320 × 192. All
11 core native execution checks passed; image-to-image checks are recorded below. The original
incorrect-default runs and preview/disclosure harness failures remain retained.
Preview recovery verifies both declared runtime/durable references and exact
media bytes. No downloaded model or operator approval was changed.

The defaults fix is authoring-only. Existing workflows retain their edited values;
resource policy stays independent. New individual nodes and connected starters
are covered, including distinct model profiles sharing one pipeline class. Its
focused gate passed 27 tests and 1,661 subtests. The complete backend gate passed
3,300 tests and 10,029 subtests, with 512 skips; the complete client gate passed.
These are additional checks, not proof that the remaining model matrix passed.

Additional completed native core campaigns:

- [x] SD 1.5: all 11 core execution cases, including unchanged reuse, parameter
      changes, mode switching, persistence, Resize and independent Saved Block edits.
- [x] PixArt Sigma: the same 11 core cases using the selected float32 defaults,
      20 then 22 steps, and 1024 square then 1152 × 896 output.
- [x] Sana 600M and FLUX Schnell: all 11 core cases each, with exact consumed
      parameters, unchanged reuse, persistence, recomputation and Saved Block checks.
- [x] DreamLite Mobile and Base: all 11 core cases each; decoded media and exact
      execution receipts audited. Requested vase placement is only partially achieved.
- [x] Lumina Next and Lumina 2: all 11 core cases each, including exact recomputation, 1152 × 896
      output, connected Resize and independently edited Saved Block execution.
- [x] CogView3 Plus: all 11 core cases, including parameter changes, unchanged
      reuse, exact recomputation, Resize, and independently edited Saved Block
      execution with exact restored media. The requested vase behind the instrument is present;
      fine-detail quality is assessed separately.
- [x] Correct the diagnosed history/Saved Block responsiveness paths and repeat
      native concurrent workflow creation, wiring and saving during PixArt generation.
- [ ] Complete their applicable task variants and separate visual assessment.

- [x] PRX: recover all 11 core native cases after publishing the selected
      adapter's inference-step bounds in both starter and dynamic controls.
      The original UI-accepted 30-step rejection remains retained. The recovery
      keeps 28 steps and tests a guidance edit at that reviewed cap, then size,
      save/reopen, Creator, recomputation, Resize and Saved Block changes.
- [x] Both locally cached Depth Anything V2 models: seven native cases each,
      covering baseline, unchanged cache reuse, processing resolution, source
      replacement, save/reopen, Creator editing and Saved Block reinsertion.
      Original media, exact revisions, consumed values and all 14 outputs audited.
- [x] Both depth Saved Blocks: native model switching and return, exact output
      hashes after reload, and unchanged saved definitions. Switching back
      reproduces each original output. The legacy model-control and stale-preview
      defects have regression coverage and successful live recovery evidence.
- [ ] Compatible containing control workflows for both depth models. Preview
      inspection does not establish metric accuracy or downstream compatibility.

Generic depth discovery now accepts backend-published model descriptors outside
legacy Studio's closed model list. Execution still uses the reviewed optional
Transformers runtime and bounded native processor adapter; this is not a claim
that every third-party processor or model configuration works automatically.

The PixArt campaign exposed intermittent authoring stalls during generation.
Original native preview failures and a GIL profile are retained. The trace shows
monolithic history JSON work, Saved Block validation on the HTTP thread, and
unnecessary copying of unrelated model defaults. The correction preserves history
and preview contracts. The full backend gate
passes 3,306 tests and 10,029 subtests (512 skips), and the client quality gate
passes. During the repeated PixArt workload, two native previews opened in 0.37
and 0.96 seconds against the original five-second bound. The second session also
created, wired and saved the image-edit workflow while generation continued.
The original failures and an image-layout timing mistake in the test remain
retained. Cold asset loading still reached 5.54 seconds in one session; this is
not a claim that every interaction now meets every responsiveness target.
Prepared image-edit drafts remain authoring evidence until they execute.

Completed native image-to-image task checks for LCM DreamShaper, Sana Sprint and
SD 1.5:

- [x] Baseline, unchanged repeat, and changed prompt/seed execution for each model.
- [x] Replace the source through normal image controls, change strength, and run.
- [x] Save/reopen, verify the exact restored output, change seed and run again.
- [x] Audit all 15 outputs, consumed parameters, cache messages and media hashes.

LCM and SD 1.5 followed source dimensions from 1024 square to 1328 square; Sana
Sprint retained its requested 1024-square output. Every unchanged repeat reused
all executed nodes and identical image bytes. The original accidental two-image
LCM submission was rejected before generation and is retained with its recovery.
These are descriptive image-to-image workflows, not instruction-edit fidelity
claims. Visual inspection shows source influence but partial prompt adherence;
SD 1.5's larger result has noticeable distortion. The remaining edit, mask,
control, processor, adapter and model paths are still open.

Three diagnosed defects are fixed: normal discovery now includes deterministic
image utilities, and the Sana Sprint backend adapters preserve the requested
step count while selecting the compatible upstream schedule, and new ordinary
image workflows receive their selected model defaults. Focused regression
tests reproduced each defect before correction. Validation passed:

- `npm run check` and two focused mocked browser regressions for shared discovery.
- Backend Ruff, package consistency and preflight checks; the full base suite:
  3,293 tests and 10,029 subtests passed, with 512 skips.
- Verified optional-runtime schedule/registry suite: 130 tests and 545 subtests
  passed, including real pinned text-to-image and image-to-image input/scheduler
  validation for one through four steps. This does not qualify live img2img.
- All 87 generated client files match the backend bundle; the five dependent
  ledgers contain only 18 fingerprint changes, with no qualification promotion.
- All 112 inventoried snapshots retain their 1,696 files, targets, sizes, mtimes
  and recorded metadata hashes; custom-source approvals are byte-identical.

Additional W8 task and correction checkpoint:

- [x] CogView4 6B: all 11 native core executions at 50 inference steps, including
      1024-square and 1152 × 896 generation, exact unchanged/recomputed output,
      Creator editing, connected Resize, independent Saved Block edits and
      restored media. Visual fine-detail limitations remain separately recorded.

- [x] Marigold depth: nine native executions covering unchanged reuse, processing
      resolution, source replacement, save/reopen, Creator editing, Saved Block
      reinsertion, seed and inference-step edits. All output hashes, dimensions,
      consumed settings and the unchanged saved definition were audited.
- [x] SDXL Canny ControlNet: five native executions covering unchanged reuse,
      prompt/seed edits, source replacement and save/reopen. The exact base and
      fp16 control weights were checked; all nine nodes reused their unchanged
      results. Source/seed edits retained compatible prompt conditioning.
- [ ] Complete the processors' containing workflows and ControlNet scale-effect
      isolation. Marigold's coherent relative depth is not metric-depth validation;
      ControlNet's partial visual fidelity is retained separately.
- [x] Reproduce missing Image upscale discovery: its existing combined loading
      and computation action had no operation-catalog entry.
- [x] Implement a backend-declared integrated model owner for the existing
      Upscaler action and shared client ownership handling. Its workflow uses one
      Upscale node and a connected Preview; it keeps the exact reviewed artifact
      selector, required source image and editable tile/output controls.
- [x] Extend the authoring/Block modification matrix to integrated operations.
      It exposed a shared reducer error on identical internal-node replacement
      and restoration to the definition. Preserve no-op state and recompute the
      actual customization state without weakening schema validation.
- [x] Final client quality gate and two targeted browser checks. Bundle limits
      remain unchanged; the final total is 859,426 compressed JavaScript bytes.
- [x] Backend Ruff, package consistency and preflight; 3,421 tests and 10,057
      subtests passed with 516 skips. The verified optional-runtime focused suite
      separately passed 56 tests and 2,514 subtests.
- [x] Publish the paired correction and complete native Real-ESRGAN x2 recovery:
      eight executions covering baseline, identical cached repeat, tile/overlap
      changes, post-scaling, source replacement, save/reopen, Creator editing and
      Saved Block reinsertion. Exact output hashes, dimensions and submitted
      artifact/settings were audited; the saved definition stayed unchanged.
      The baseline retained the source composition on visual inspection. Other
      cached upscaler weights and detailed tiled-output quality remain open.

The upscale correction adds authoring metadata to an existing executor. It does
not qualify every Spandrel weight. Original failed discovery evidence remains
retained. The remaining image-model/task campaigns are still in
progress; this checkpoint does not close W8.

Native image regression follow-up:

- [x] Audit LongCat's 11 native core cases, including exact recomputation,
      connected Resize, independently edited Saved Blocks and restored media.
      Complex-prompt objects and placement were visually checked separately.
- [x] Reproduce Hunyuan's clamped dimension edit and Ovis's rejection of unrelated
      cached-repository weight folders. Preserve both original failed attempts.
- [x] Add focused regressions for explicit bounded dimensions, immutable local
      pipeline loading, and indeterminate progress when upstream has no callback.
- [x] Run the full client gate and five browser workflow creation/reload checks;
      all 89 generated bundle files match the existing served build exactly.
- [x] Final backend gate: 3,435 tests and 10,057 subtests passed, with 521 skips.
      Verified optional-runtime checks separately passed 145 tests and 545
      subtests, including the installed classes and their call signatures.
- [x] Repeat all 11 Hunyuan native core cases on the fix, including exact
      1152 × 896 output, unchanged default-baseline bytes, Resize and Saved Blocks.
- [x] Complete all 11 Ovis native recovery cases, including prompt/seed/guidance
      edits, 1152 × 896 output, save/reopen, Creator editing, exact recomputation,
      connected Resize and independently edited Saved Blocks. Saved definitions
      remain unchanged and restored output bytes match. The original receipt
      timeout is retained; the successful repeat preserves its baseline image.
- [x] Verify LongCat's visible indeterminate progress during a 50-step,
      1024-square generation. The baseline completes in 223.5 seconds and the
      unchanged cached repeat in 4.9 seconds, with identical output bytes.
      Step counts and ETA remain unknown when upstream provides no callback;
      the corrected run preserves the original pre-correction baseline image.

These corrections keep generic nodes and put upstream differences behind backend
contracts. Prepared recovery campaigns and unit checks are not live acceptance.
The remaining model/task, auxiliary and structural matrix remains open.

Additional concurrency and dimension regressions:

- [x] Native authoring reproduces fixed-size clamping for Kandinsky 3, ERNIE
      and GLM. Their upstream contracts support explicit non-square sizes.
- [x] Preserve Ovis and LongCat receipt timeouts after completed generation.
      Eight concurrent retained-run lookups reproduce a 15.8-second response.
- [x] Add a regression proving that one run lookup unnecessarily decodes all
      history records. Index candidates in the existing immutable record cache;
      retain strict identity validation and external-file invalidation.
- [x] Author a connected Text Value → custom Modular prompt → custom Python
      prefix → Generate graph during generation; edit, undo/redo, save and reopen
      through native controls. No graph was submitted by this authoring check.
- [x] Final backend gate passed 3,444 tests and 10,057 subtests (524 skips);
      verified optional checks passed 151 tests and 545 subtests. The complete
      client gate and five workflow creation/reload browser checks passed.
- [x] Regenerate the one affected ERNIE compiled Block definition after reviewing
      its eight dimension-bound changes. The other 121 definitions are unchanged;
      all 89 served files match and dependent ledgers have only 18 hash changes.
- [x] Verify native 1152 × 896 authoring through Kandinsky 3, ERNIE, GLM and
      ERNIE's equivalent-standard workflow. All four retain the requested values;
      this is authoring evidence, not generation qualification.
- [x] Repeat eight concurrent run lookups against the real retained history.
      Maximum response time falls from 15.80 seconds to 2.73 seconds cold and
      0.041 seconds warm; all eight original output hashes are preserved.
- [x] Repeat native custom-node wiring, edits, undo/redo, save and reopen during
      the Ovis seed generation that previously exposed a receipt timeout. Both
      the authoring session and generation receipt complete successfully; the
      authoring browser submits no competing graph. Audit all 11 Ovis recovery
      cases and both LongCat progress/reuse cases against exact task outputs.
- [ ] Complete live non-square generation for Kandinsky 3, ERNIE and GLM.
      Their native authoring checks remain separate from generation acceptance.
- [ ] Execute the prepared custom compositions across standard and Modular
      model paths, including execution after undo/redo and node removal.

The core batch now has 165 independently audited native execution checks across
15 model paths. These counts exclude separately recorded SDXL/Sana Sprint,
processor and task campaigns; they do not mean all local models are qualified.
The latest checkpoint preserves all 112 original snapshots, 1,696 files and
operator approvals. Remaining model/task, auxiliary, custom-composition and
visual-assessment requirements keep W8 open.

Completed cached-weight selection and Saved Block field-action recovery:

- [x] Reproduce the uncatalogued cached x4 upscaler selection failure. Resolve
      its exact installed revision, hash and size when the user selects it;
      persist that identity without weakening execution validation or loading
      weights during metadata inspection.
- [x] Reproduce suppressed callbacks in collapsed Saved Blocks and incorrectly
      routed inspector callbacks. Resolve declared exposed fields to their
      existing internal runtime owner and map metadata replies back to the
      workflow instance. Preserve saved definitions, peers and stale-reply guards.
- [x] Pass 200 related client regressions, the full client gate and five final
      focused browser checks. The paired backend gate passes 3,458 tests and
      10,057 subtests with 524 skips; the already-reviewed selector runtime
      contract separately passed 94 optional-runtime tests and 281 subtests.
- [x] Complete and independently audit ten native x4 executions: baseline,
      identical repeat, tile/overlap edits, post-scaling, replacement input,
      save/reopen, Creator editing, Saved Block reuse, canvas x4-to-x2 switching
      and inspector return to x4. The returned output is byte-identical to the
      prior x4 Block output. Exact metadata requests, persisted artifact pins,
      dimensions, cache reuse, saved-definition preservation and restored output
      bytes are checked.

The x4 baseline is 4096 square; the Block return produces 2656 square after
post-scaling. Visual inspection retains the source composition; fine-detail and
full-resolution tile-seam quality are not established by that inspection.
Both original failures remain recorded. This closes the selector/routing
recovery, not W8's remaining model/task, auxiliary and structural campaigns.

Additional guidance-default correction checkpoint:

- [x] Trace the native Krea baseline mismatch to new-workflow defaults: the
      recommended distilled-guidance value was seeded into a legacy True CFG
      field, while distilled guidance remained implicit.
- [x] Route new ordinary FLUX workflow recommendations into the existing
      distilled-guidance override and initialize True CFG to its pinned upstream
      default. Preserve saved values, dynamic updates and legacy invocation.
- [x] Cover nine model/task selections, single-node insertion, exact consumed
      parameter names, old calls and pinned upstream defaults. Pass the full
      backend gate (3,470 tests / 10,057 subtests; 524 optional skips), 217 approved
      optional-runtime tests / 2,214 subtests, the client gate and five browser
      workflow creation/reload checks.
- [x] Reject absent numeric receipt values instead of converting null to zero.
      Retain the original Krea failure and remove Schnell's earlier guidance
      default acceptance pending recovery. DreamLite Mobile's absent guidance
      is explicitly reviewed as an unused control, not reported as consumed zero.
- [x] Complete fresh native Krea and Schnell recovery campaigns and independently
      audit their explicit guidance receipts, edits and reusable Blocks.

These corrections do not close the remaining model/task, auxiliary, structural
or visual portions of W8. The downloaded Z-Image realism LoRA now has a prepared
exact-revision containing-workflow campaign; preparation is not execution proof.

### W8 — Paused checkpoint and remaining work

This earlier W8 pause checkpoint is retained; the current image/audio completion scope above resumes its image work. The audited core
checkpoint contains 176 runs across 16 model paths, including completed Krea and
Schnell recovery. Chroma additionally has retained baseline, unchanged, prompt,
seed and sampler results; its campaign remains partial. The last task completed
before all campaign controllers stopped and the queue was verified empty.

- [ ] Finish Chroma dimensions, persistence, workspace switching, recomputation,
      processor wiring and Saved Block reuse.
- [ ] Complete remaining image models, variants, unconditional workflows and
      standard/Modular task changes from the retained matrix.
- [ ] Complete depth/custom processor compositions, LoRA effects, IP-Adapters,
      Union controls, Layered workflows and alternate-source Redux cases.
- [ ] Qualify Florence native custom-node authoring and persistence.
- [ ] Resolve and test the reviewed runtime/component binding for cached 4-bit
      Klein; runtime availability alone does not establish model compatibility.
- [ ] Reconcile exact-task/dependency/media audits and remaining visual review;
      retain incomplete artifact blockers and original failures separately.

Resume from retained evidence with a fresh source/impact review. Prepared
campaigns are not passing tests. Do not overwrite prior evidence or repeat all
completed models after unrelated changes. Windows memory qualification remains H1.

### W9 — Other modalities and real service export

- [ ] Run baseline, unchanged, parameter-change and save/reopen cases for each
      complete supported cached video/audio/other pipeline; track incomplete and
      unsupported artifacts separately. W9 now takes priority while W8 is paused.
- [x] Exercise representative non-image paths: an approved custom prompt processor
      across nested internal containers in an audio Block V2; workspace switching
      on audio, video and rendered-3D workflows. Expand testing if new contract
      differences or regressions appear. Nested composite instances remain unsupported.
- [x] Recover the already-generated heavy Wan output and diagnose browser decoding
      before deciding whether inference needs repeating; retain failed evidence.
- [x] Export real image workflows natively as service packages and execute with
      named inputs; compare consumed values and exact-task outputs with UI runs.
- [x] Cover native Modular, standard whole-pipeline, image editing, custom-node and
      supported Block V2 internal hierarchy service paths, plus representative
      generated video/audio/rendered-3D outputs. This is bounded path coverage,
      not qualification of every cached model/task below.
- [x] Exercise changed inputs, repeated calls, sequential queued invocations,
      cancellation/recovery, restart, code/package drift and missing model identity.

Checkpoint coverage (W9 remains in progress):

- [x] Standard image service: native download, API and CLI execution, unchanged
      calls, prompt/seed edits and exact UI/service output comparisons.
- [x] Representative service lifecycle: observed sequential queue, cancellation,
      worker replacement, replay after restart, code/package drift and missing
      immutable model identity rejection. This does not qualify every runtime.
- [x] Recovered Wan media: full decode, native upload/preview, unchanged/reopened
      workflow and service output playback with changed frame rate. This proves
      delivery of historical output, not a new Wan generation.
- [x] Fix service export of text inputs and completed preview observations;
      preserve private-file input validation and original failed evidence.
- [x] Correct new audio workflow defaults after AudioLDM2 rejected the shared
      duration; preserve edited values in saved workflows. AudioLDM2 native
      generation, save/reopen, exported service, repeat and prompt/seed edits pass.
- [x] Capture consumed audio steps/guidance and requested duration/delivery rate;
      preserve connected origins and validate the matching client receipt.
- [x] Recover LongCat loading after its inherited VAE tiling hook reported no
      implementation; distinguish unsupported optional hooks from real failures.
      Native/service execution, repeats, prompt/seed edits, reopen and workspace
      invariance pass. Requested duration remains subject to upstream VAE rounding.
- [x] Recover native Modular service export using reviewed dynamic controls:
      Z-Image passes native/API/CLI execution, repeats, prompt/seed changes and
      exact UI/service output comparisons.
- [x] Execute an approved custom prompt processor through native service export;
      a legacy composed Block also passes image plus metadata delivery, changed
      inputs and reopen. Legacy composition flattens children and does not prove
      nested Block V2 execution.
- [x] ACE-Step XL text-to-audio: baseline, repeats, prompt/seed changes, reopen,
      workspace switch and native/service output comparisons; eight retained
      stereo WAVs decode at 48 kHz / 30 seconds. Semantic listening is pending.
- [x] Correct new video/3D workflow defaults using reviewed capability metadata
      after CogVideoX rejected the shared precision. Saved edits remain unchanged;
      the original failure is retained. CogVideoX recovery passes baseline,
      unchanged/reopen, service and prompt/seed/Creator edits. Its 25-frame,
      720×480 clip fully decodes at 8 FPS; sampled visual inspection is recorded.
- [x] Native JSON import, edits, persistence and service export of an audio Block V2
      with two nested internal containers and the approved custom prompt processor:
      seven cases pass with exact output comparisons and decoded audio. This does
      not represent independently nested composite Block instances.
- [x] SDXL image-edit service: uploaded image exposed as a required named input,
      eight native/service/reopen/edit/workspace cases and exact output comparisons.
      Visual inspection shows a meaningful edit with partial instruction adherence.
- [x] Shap-E rendered-3D workflow: eight native/service/reopen/edit/workspace cases;
      the 20-frame, 256×256 orbit fully decodes and shows a coherent red chair.
      This does not qualify native mesh export or perfect geometry adherence.
- [ ] Finish the remaining cached non-image model/task matrix: other ACE-Step
      artifacts and conditioned audio tasks; MiniMax Music3; Sana; Wan first/last
      frame, VACE, Animate full/distilled, I2V and TI2V; CogVideoX video-to-video;
      AnimateDiff/PAG/LCM variants and conditioned tasks.
- [ ] Resolve and test specialized Whisper transcription/translation, wav2vec CTC
      and Real-ESRGAN video-upscale authoring paths; finish media/identity audits.
      Incomplete Mochi/Cosmos/Stable Audio/Stable Video artifacts remain separate.
      W9 is not complete; W8 remains paused.

This checkpoint tested production code at backend `68a41d8` / client `4cc26bd`.
The video/3D defaults fix passed the full backend suite (3507 tests, 10057 subtests,
524 optional skips), 163 approved optional tests / 253 subtests, Ruff, dependency
and preflight checks, and the relevant Workflows browser check. Earlier service
fixes and matching client code have their separate full gates recorded above.
Preservation recheck: 112 snapshots / 1696 original files retain their targets,
sizes, modification times and recorded metadata hashes; operator approvals remain
byte-identical. Original failed evidence is retained separately from recoveries.

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
