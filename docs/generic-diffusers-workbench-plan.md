# Generic Diffusers workbench implementation plan

## Purpose and status

This is the accepted direction for implementation, not a description of features
already delivered. Build a small, transparent workbench for Diffusers developers
and a guided composition experience for creators. Both use the same workflow,
backend executor, model components, and persistent Block representation.

Maintain an identical copy of this plan in the backend and client repositories.
Implement on `feat/generic-diffusers-workbench`, branched from `develop` in each
repository. Keep changes in reviewable commits with their validation results.
No model weights, local configuration, generated media, or user workflows are
part of this change or its cleanup scope.

The first implementation increment delivers M1's catalog-view changes: Expert
defaults to Stages; Auto/Essentials scopes the HF catalog to insertable,
graph-qualified task Blocks; implementation entries remain in Advanced; search
and counts follow the selected view. Existing insertion paths and saved User
Nodes are retained. Unit and native browser regressions cover the new policy.

M1 is complete: audience-appropriate discovery and shared typed drag-to-add
matching are implemented and validated. M2 is complete: independent authoring
and resource controls, stable field lifetimes, and persistence/Run regressions
passed their final validation.
M3 is complete: canonical operations, scoped port semantics, pipeline/task
coverage, existing-node resolution, Expert discovery consolidation and reviewed
whole-pipeline fallbacks are implemented and validated. M4 is complete: connected
starters, model/task previews, ordinary graph transactions, retained settings and
shared generator inputs passed their authoring and scoped real-execution checks.
M5 is complete: selective recomputation, component ownership, isolated mutable
state, bounded memory and failure recovery passed their scoped validation.
M6 is complete: unified custom-source review, explicit code approval, ordinary
node and Modular block execution, reload and targeted invalidation passed their
scoped validation. M7 implementation and Linux validation are complete; Windows
clean-install execution is deferred by the operator until a Windows machine is
available. M8 implementation and scoped local qualification are complete;
feature-branch publication is blocked by Git authentication. This does not qualify every declared
model/task, hardware configuration or third-party extension.

## Progress tracker

Check a task only after its implementation and applicable validation pass, then
record its commit under the milestone. **In progress** remains unchecked.
Keep this tracker and the detailed checklists below synchronized in both repos.
Model execution, hardware qualification, and UI tests are separate evidence.

| Milestone                                          | Status      | Remaining work                                                                  |
| -------------------------------------------------- | ----------- | ------------------------------------------------------------------------------- |
| M1 — Audience-appropriate discovery                | Complete    | None within M1; model/task authoring is delivered in M4                         |
| M2 — Independent authoring/resource modes          | Complete    | None within M2; representative runtime reuse is validated in M5                 |
| M3 — Canonical operations and capability inventory | Complete    | None within M3; stage authoring is delivered in M4                              |
| M4 — Stage authoring and model/task switching      | Complete    | None within M4; broader model/hardware qualification remains explicitly scoped  |
| M5 — Reuse and selective recomputation             | Complete    | None within M5; broader family/hardware qualification remains scoped            |
| M6 — Custom-node developer experience              | Complete    | None within M6; arbitrary extensions still need code/dependency/resource review |
| M7 — Developer setup and service prototyping       | Implemented | Windows qualification deferred by the operator until after publication          |
| M8 — Consolidation and product qualification       | Implemented | Push blocked; DDPM recipe follow-up and Windows qualification remain open       |

Completed foundation: plan committed in both repos; implementation branches
created from `develop`. M1 catalog views were committed in client `df50a30` and backend
`aa2340c`: client quality gate and 148 browser tests passed; backend base gate
passed with 3,038 tests and 509 skips; fresh HTTP serving matched the client
bundle. This evidence does not qualify live model generation.

## Diagnosis

The application exposes several overlapping authoring layers: task-level
Diffusers nodes, generic Modular stages, specialized workflow-stage adapters,
family/task clusters, individual upstream blocks, component reference entries,
and user compositions. Exact aliases are already deduplicated. Different
implementations with similar purposes are not necessarily execution duplicates.

Before M1, the client defaulted Expert to the broad Advanced catalog. Its ordinary
node visibility filter did not govern the separately rendered Hugging Face
catalog, so restricted views exposed internal blocks and reference-only
components. M1's catalog increment corrected this. Both repositories contain the smaller
Load Models / Encode Prompt / Denoise / Decode Latents / Preview template.

Before M2, the Auto/Expert control also changed resource planning policy.
Custom local/Git modules and custom Hub Modular blocks use different installation
and execution paths. Existing cache reuse coexists with pressure- and
schedule-driven model eviction. Each issue needs a distinct change and proof.

## Product contract

### Terminology

- **Node:** one operation with declared inputs, outputs, and parameters.
- **Block:** a reusable composition with a public interface; retain the existing
  BlockDefinitionV2 / BlockInstanceV2 implementation.
- **Workflow:** the editable, executable document containing nodes and Blocks.
- **Template:** a starting copy with useful defaults, not a prerequisite for
  executing or validating a graph.
- **Implementation:** the concrete upstream pipeline or block behind an operation.

Use Blocks as the eventual public name for registered clusters and saved
compositions. Keep legacy identifiers readable. Avoid adding another executor,
serialized graph format, or parallel library for the two authoring modes.

### Auto and Expert

| Surface            | Auto target                            | Expert target                                            |
| ------------------ | -------------------------------------- | -------------------------------------------------------- |
| Starting point     | Task or compact starter                | Empty canvas or small stage starter                      |
| Default library    | Tasks, media, processing, saved Blocks | Generic stages, utilities, installed custom nodes        |
| Default canvas     | Compact task Blocks                    | Editable processing stages                               |
| Model choice       | Compatible model selector              | Model/component loader with dynamic downstream contracts |
| Parameters         | Common controls and an advanced drawer | Complete supported controls, grouped by operation        |
| Internal blocks    | Outside normal discovery               | Contextual implementation editor                         |
| Memory policy      | Automatic by default                   | Automatic by default; explicit overrides available       |
| Custom development | Use installed nodes                    | Create, install, reload, inspect, and debug              |
| Export             | Workflow and media                     | Workflow, API graph, reproducible execution package      |

Changing authoring mode changes presentation and available editing tools. It must
not change model choice, effective inputs, execution authority, component cache,
or saved topology. Existing workflows without an equivalent stage decomposition
remain editable in their current form. Structural conversion is explicit,
previewable, atomic, and undoable.

### User journeys

Auto: select a task, select a compatible model, enter text/media, run, inspect the
output, connect another task or processor, optionally open the underlying stages.
Show required installs or incompatible inputs without making the user browse
implementation catalogs. Opening a template never installs packages or weights.

Expert: insert Load Models, select a model, drag a compatible port to insert
Encode Prompt, connect Denoise and Decode Latents, add a Preview, and run. Allow
manual construction without a template. Add image/mask/reference conditioning
through compatible ports and explain the selected task. Inspect a stage's actual
upstream blocks only on request.

On a model change, retain stable node identities, layout, prompt text, compatible
parameters, and compatible wires. Preserve unsupported values for Undo or a later
switch without passing them to execution. Diagnose incompatible custom stages;
do not silently remove them or reuse embeddings/latents from the previous model.

## Architecture and coverage

Generic operations select model-specific implementations in the backend. Extend
the existing registries, execution specifications, Modular adapters, and node
schema publication rather than introducing frontend family dispatch.

A capability description must include stable operation identity, supported tasks,
required components, input/output semantics, defaults, implementation binding,
cache dependencies, and supported decomposition depth. Derive declarations from
upstream Modular metadata where possible; use small reviewed adapters for missing
semantics. Do not infer interchangeability from a raw `tensor` or `embeddings` type.

Support the entire pipeline inventory of the pinned Diffusers revision, with
separate evidence for discovery, callable execution, stage decomposition, task
variants, and hardware. Do not limit the architecture to Qwen and Flux.

| Upstream capability                               | Public execution surface                                     |
| ------------------------------------------------- | ------------------------------------------------------------ |
| Native Modular blocks                             | Generic stages and optional deeper block editing             |
| Standard pipeline without supported decomposition | Load Pipeline and Run Pipeline with dynamic task fields      |
| Additional modality stages                        | Relevant image, video, audio, reference, or other operations |
| Explicitly installed custom block                 | Node generated from declared schema and dependencies         |

A whole-pipeline call is not proof of editable Modular execution. Not every model
has text encoding, a VAE, or the same four processing stages. Metadata discovery
does not qualify every repository or hardware configuration.

## Milestones

### M1 — Make discovery match the selected audience

Client ownership: `nodeCatalog`, `huggingFaceNodeCatalog`, `NodeList`, connection
search, and their existing unit/browser fixtures. Backend ownership: publish the
validated client bundle and keep the paired plan/documentation aligned.

- [x] **M1.1** Reproduce internal HF catalog leakage in Auto/Essentials and broad Expert
      startup with focused catalog and rendered/browser assertions.
- [x] **M1.2** Give Expert an explicit default Stages view containing existing generic Modular
      stages, useful ordinary operations, and installed custom nodes. Keep full
      implementation discovery available through explicit catalog selection.
- [x] **M1.3** Apply the same effective view to ordinary entries and HF sections. Auto keeps
      usable task Blocks; implementation blocks and reference components are opt-in.
- [x] **M1.4** Preserve exact identifiers and existing insertion functions. Do not delete
      backend classes, user Blocks, catalog definitions, or saved graph formats.
- [x] **M1.5** Make search and empty states explain the selected scope; selecting an advanced
      view must not change execution policy or graph state.
- [x] **M1.6** Add direction-aware, type-compatible drag-to-add search using the same
      compatibility rules as connection insertion. Installed custom nodes participate.

Acceptance: native mode/catalog navigation exposes the intended entries; a generic
stage can be inserted; hidden internals are still available deliberately; existing
graphs and saved Blocks survive. Record catalog/UI evidence separately from model
execution. No additional runtime capability is claimed by this milestone.

Completion evidence:

- Catalog views: client `df50a30`, backend bundle `aa2340c`.
- Shared typed suggestions and insertion: client `7320ab5`, backend bundle
  `5d03622`. Installed custom controls, normalized types, wildcard contracts,
  direction filtering, exact-alias deduplication and contextual search are covered.
- `npm run check`: passed, including 1,111 Node tests and the unchanged bundle
  budgets. `npm run check:ui`: final full run passed 2 shared-control and 148
  Studio browser tests. Native drag-to-add coverage checks both directions,
  keyboard/mouse selection, incompatible results and refresh persistence.
- The first full browser attempt had one existing template-dropdown dismissal
  failure. That test then passed three isolated runs on both the unchanged branch
  and changed code, followed by the complete passing run. No assertion was relaxed.
- Fresh-backend HTTP checks matched the exact built bytes for `/`,
  `/assets/index.js`, `/assets/NodeList.js` and `/assets/NodeSearchDialog.js`.
  Backend-owned user fields and local Gallery content were preserved.
- This is source, mocked-browser and HTTP proof. No new live model execution,
  cross-family adaptation or model-residency claim is made.

### M2 — Separate authoring mode from execution resource policy

- [x] **M2.1** Inventory all reads/writes of `studioViewMode` and `resourceMode`, including Run,
      selected-node execution, workflow tabs, persistence, imports, and recovery.
- [x] **M2.2** Introduce explicit authoring state and resource-policy state with a migration
      that preserves existing execution settings. Define old-client compatibility.
- [x] **M2.3** Remove mode-driven model/schema rebuilds; preserve the active graph and cache.
- [x] **M2.4** Expose automatic resource management in Expert and a clear override control.
- [x] **M2.5** Update Auto/Expert design documentation and policy tests together.

Acceptance: mode changes preserve exported graph, effective values, active task,
and loaded-model identity; saved documents round-trip through refresh and Undo.
Unsupported automatic planning remains explicit, without forcing a UI mode change.

Inventory and migration decisions:

| Existing owner                                                                            | Responsibility and required change                                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `useSettingsStore.studioViewMode`                                                         | Persisted global authoring preference; retain its key and existing `manual`-to-`expert` compatibility reader. Library, inspector, export and workspace visibility read it.                                         |
| `useStudioStore.form.resourceMode`                                                        | Per-workflow execution setting; retained in tab snapshots, saved workflows, imports, output restoration and packages. Keep the existing legacy resource-mode normalizer and backend values.                        |
| `TopBar`                                                                                  | View changes now update only the authoring preference. A separate Resources handler retains deliberate graph synchronization. Removed policy-to-view mirroring and silent fallback for invalid automatic planning. |
| `blockAutoAuthorityV2`                                                                    | Uses only the workflow's resource policy. Opening Expert authoring tools cannot bypass registered-Block planning or rewrite the saved policy.                                                                      |
| `useStudioRunActions`, `runCoordinator`, `workflowAutoExecutionV2`, `blockRuntimeHintsV2` | Use resource policy for planning, selected-node/whole-workflow execution and submitted hints. Preserve these execution boundaries.                                                                                 |
| `GraphFixDialog`, `RunIssuesDialog`                                                       | Opening an inspector selects Expert presentation. It must not change resource policy or trigger graph rebuilding.                                                                                                  |

The two persisted fields already exist; avoid a second settings store or a new
workflow schema. Preserve both saved values, remove implicit cross-writes, and
show separate authoring and resource controls. Existing workflows keep their
resource settings; new workflows retain the existing automatic default. Older
clients can read the same workflow fields but retain their coupled UI behavior;
independent controls require the matching updated client bundle.

Implementation and validation sequence:

1. Add regression coverage for an Expert authoring preference paired with automatic
   resources, and an Auto authoring preference paired with explicit overrides.
   Assert the effective run policy instead of merely inspecting switch labels.
2. Make the top-bar authoring switch update only `studioViewMode`. Introduce a
   separately labelled resource-policy control in both presentations. Keep managed
   graph synchronization behind deliberate resource or generation-input edits.
3. Remove the authoring override from registered-Block Auto authority preparation.
   Planner failures must remain failures; selecting Expert presentation must not
   silently bypass them. Unsupported automatic planning stays visible and can be
   resolved by an explicit execution-policy choice.
4. Remove the permanently disabled duplicate Studio resource header and its unused
   handlers. Retain existing saved keys, legacy normalization and tab restoration.
   Keep field groups mounted across disclosure/view changes so unchanged advanced
   custom controls do not repeat initial schema actions.
5. Audit browser helpers and expectations: tests that intend to change execution
   must operate the resource control explicitly. Presentation tests must not use
   helpers that change both settings and conceal accidental coupling.

| Regression case                                                | Required invariant                                                                                                                            |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Managed workflow: Auto to Expert and back                      | Same graph IDs, topology, user values, task and effective resource policy; no schema refresh or planner request caused solely by presentation |
| Registered Block: Expert presentation with automatic resources | Automatic planning and receipt validation still apply; a planner rejection never falls through to manual execution                            |
| Auto presentation with explicit overrides                      | Resource summary reports the actual overrides; presentation does not overwrite them                                                           |
| Inspect a run issue or graph fix                               | Inspector opens without changing the saved execution policy or rebuilding the graph                                                           |
| Save, refresh and switch workflow tabs                         | Global authoring preference and each workflow's resource settings restore independently                                                       |
| Legacy values and older packages                               | Existing normalizers preserve the intended execution policy; no mass graph conversion or storage-key replacement                              |
| Undo/Redo after a presentation switch                          | Graph edit history remains intact; switching presentation adds no topology transaction                                                        |
| Whole-workflow and selected-node Run                           | Both use the workflow resource policy, regardless of authoring presentation                                                                   |

These checks establish state and request-boundary behavior. Actual retained
component/cache behavior still requires M5's runtime instrumentation and execution
proof; unchanged UI values alone do not establish that models stayed resident.

Completion evidence:

- Client `19eeebf` and backend bundle/API documentation `9e289b6`.
  Auto/Expert view writes only the global editing preference. Resources explicitly
  selects the workflow's automatic planning or Expert overrides policy.
- Both existing persisted fields and their legacy normalizers are retained;
  no new workflow format or settings migration is necessary. Older clients can
  read the values but need the updated bundle for independent controls.
- Native regressions cover unchanged graph/form/plan state across view changes,
  independent Save/reopen and tab restoration, Undo/Redo, and automatic planning
  in Expert view for whole-workflow and selected-Block Run. Planner rejection
  does not fall through to an override policy.
- A custom advanced control reproduced a second schema request on an unchanged
  Expert → Auto → Expert round trip. Stable mounted field groups fix it; native
  view/disclosure toggles now preserve initialization while real edits still
  dispatch their declared actions. Existing tests that intended a resource edit
  now operate Resources explicitly instead of relying on the old coupled switch.
- Removed the permanently disabled duplicate Studio resource header, unused
  planning summaries and handlers. The top bar owns the policy selector.
- `npm run check`: passed, including **1,112 Node tests**, production build and
  unchanged bundle budgets. Final `npm run check:ui`: **2 shared-control and 150
  Studio browser tests passed**. Updated documentation passed formatting and local
  Markdown-link validation.
- A fresh backend using the normal optional-runtime activation path returned
  HTTP 200 and exact client build bytes for `/`, `/assets/index.js`,
  `/assets/studio-templates.js`, `/assets/NodeList.js` and
  `/assets/NodeSearchDialog.js`. The bundle mirror preserved `web/user/` and
  installed `web/template-gallery/` content.
- Backend Python execution code did not change in M2. These checks establish
  UI/state/request and HTTP behavior; actual retained model components,
  selective recomputation, constrained-memory execution and crash recovery
  remain M5 work. No downloaded models were deleted or installed by this work.

### M3 — Establish canonical operations and complete capability inventory

- [x] **M3.1** Inventory ordinary actions, stage adapters, registered Blocks, and aliases.
      Classify exact aliases separately from semantically overlapping implementations.
- [x] **M3.2** Extend backend-owned contracts with operation identity, stage capabilities,
      semantic ports, and execution/decomposition support levels.
- [x] **M3.2a** Project registered generic Modular stage declarations into the existing
      capabilities response and validate/store them in the client without family dispatch.
- [x] **M3.2b** Extend the same operation contract to loaders, standard pipelines and
      specialized stages through their existing owners; declare richer port semantics
      needed for compatibility instead of relying on socket type or name alone.
- [x] **M3.2b1** Project standard image/video/audio/rendered-3D loaders and whole-pipeline
      actions by task; preserve port visibility, requiredness and pipeline handles.
      Validate the version 2 client contract and version 1 compatibility.
- [x] **M3.2b2** Project Modular loader details and specialized workflow stages; add
      richer conditioning/latent semantics needed for compatibility decisions.
- [x] **M3.2c** Publish per-pipeline/task execution and decomposition support, linked to
      actual adapters and optional-runtime requirements independently of template receipts.
- [x] **M3.3** Map all pinned upstream pipelines and tasks into the coverage matrix. Keep
      unavailable optional dependencies distinct from missing adapter support.
- [x] **M3.4** Bind existing generic stage nodes to supported implementations; consolidate
      discovery without changing persisted backend action names prematurely.
- [x] **M3.5** Add standard-pipeline fallback operations for non-Modular paths. Preserve
      special stages for video, audio, 3D, and unusual conditioning.

Acceptance: parsers reject malformed capabilities; declarations agree with actual
upstream components and inputs; every inventory entry has an honest support state.
No template-specific receipt is required merely to author an ordinary valid graph.

M3 completion:

- The backend publishes operation schema 3 and pipeline-support schema 1 through
  `/model_capabilities`. **861 declarations use 24 canonical operation IDs across
  157 pipeline/alias classes**. Discovery constructs no nodes or pipelines and
  performs no installation or model download.
- Modular loaders and all reviewed task stage sequences retain their existing
  actions. Component inventories, model-owned conditioning/latents, sealed state
  predecessors and opaque media references remain explicit. Numeric latent
  controls are not misclassified as latent tensors.
- `/operations/resolve` returns one ordinary node schema with its existing action,
  exact selectors, dynamic fields, canonical label and correct loader signals.
  It does not create a graph, template receipt or execution permission. Every
  published binding is exercised by the no-construction resolver regression.
- The pinned inventory accounts for **all 330 upstream exports**, declared
  AutoPipeline mappings (including conditional entries) and Modular tasks.
  Together with local aliases, support discovery has **335 pipeline records and
  524 task records**. Exports without named tasks retain `pipeline_call`; missing
  bindings and runtime requirements stay separate. This does not infer arbitrary
  modes from every optional pipeline-call argument.
- Reviewed standard fallbacks preserve exact public profile modes and aliases.
  A Modular stage path is not mixed with a standard generator requiring a
  different loader. Video/audio, media preparation/postprocessing, reference
  assembly and rendered-3D operations remain separate where needed.
- Expert → Stages provides a pipeline/task picker and one entry per bound
  operation. It initially chooses a task with an operation binding when one
  exists. Clicking adds an ordinary node. Pending insertion is cancelled after
  selection/workflow changes. Advanced retains underlying implementations;
  Auto/Essentials and saved action IDs retain their behavior.
- Client schemas validate bindings, scoped semantics and support references;
  malformed refreshes clear both operation and support state. Older operation
  schemas remain readable. No frontend model-family dispatch table was added.

M3 completion evidence:

- Client code: `7fed2d6`; backend code and matching served bundle: `1ebe608`.
- Backend: `uvx --from ruff==0.12.7 ruff check . --select E9,F`,
  `uv pip check --python .venv/bin/python`, and
  `./scripts/with-runtime-env.sh .venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error`
  passed. Complete base suite:
  `./scripts/with-runtime-env.sh .venv/bin/python -m pytest -q` —
  **3,065 passed, 509 skipped, 9,061 subtests passed**.
- Verified installed optional runtime: `scripts/test_reviewed_optional_runtime.py`
  over operation catalog, Modular catalog, upstream-contract/workflow truth,
  standard operations and Modular workflow-block tests — **112 passed, 2,203
  subtests passed**. It performed no installation or model inference.
- Pinned inventory generator `--check` passed. All 200 canonical workflows and
  78 public template records remained unchanged when refreshing bundle hashes.
  After the final bundle mirror, coverage/authoring/Comfy ledgers and operation
  regressions passed **77 tests and 1,844 subtests**; one separate full-source
  ledger regeneration test remained skipped because it requires additional
  reviewed source/wheel inputs. The M3 Diffusers inventory check itself passed.
- Client: `npm ci` and `npm run check` passed, including **1,126 Node tests**.
  The complete production startup remains under **602 KiB**; the new lazy picker,
  resolver and contract validation measure about **198.1 KiB** across deferred
  code, bounded at **199 KiB**. Individual chunk limits remain unchanged.
- Native Chromium: shared controls plus canonical insertion/cancellation and
  Auto/Expert catalog regressions — **4 passed**. Actual backend HTTP checks
  resolved Modular stages/loaders, a standard fallback and a contract-only
  schema, rejected invalid selection, and served matching production assets.
  A production browser inserted an Anima Denoise node through the real endpoint
  with no page errors. The source client parsers accepted the full HTTP catalog.

These are source, schema, no-weight runtime, HTTP and browser proofs. They do not
establish new live model output, universal cross-family tensor compatibility,
artifact availability, hardware qualification or cache performance. **M4** owns
small starters, connected graph authoring improvements and atomic model/task
adaptation. **M5** owns selective recomputation and model residency. **M6–M8**
remain pending. No downloaded AI models were deleted or changed.

M3.2a implementation (earlier checkpoint):

The existing `/model_capabilities` endpoint now exposes versioned operation
contracts derived from `ModiffPipelineRegistry` / `MoDiffPipelineConfig`, without
another executor or implementation catalog. The inspected runtime publishes
**55 declarations across 13 pipeline classes and seven generic stage operations**.
Saved module/action identities remain the same. The shared operation descriptors
also supply the existing Studio role mapping and Modular stage labels.

Declarations preserve the formatted port identity, original semantic name,
direction, requiredness, declared types, and value/component roles. One socket
may supply both conditioning values and model components. They retain their
pipeline scope and explicitly report `support: declared`; matching names/types
alone does not establish cross-model compatibility or runtime readiness. Block
and bundle decomposition are distinguished without constructing pipelines.

The client validates and stores these declarations without its Studio family
union. Older backends may omit the catalog; malformed or failed refreshes clear
operation metadata. The parser loads on demand and concurrent discovery callers
still share one request. At this earlier checkpoint, insertion and richer
support/compatibility semantics were not included. The M3 completion section
above records their delivery; graph-wide model/task switching remains M4,
model reuse remains M5 and custom extension work remains M6.

The matching client build was mirrored into `web/`. The coverage generator's
workflow/template projection confirmed all **200 workflow records and 78 public
template records** were unchanged before updating the bundle fingerprint and its
content hash. The four dependent ledgers were refreshed with their existing
generators; their diffs contain source/content hashes only, with no new admission,
execution or qualification claims.

M3.2a completion evidence:

- Client code: `4d3826f`; backend code and matching served bundle: `0f16764`.
- Backend gate passed: `uvx --from ruff==0.12.7 ruff check . --select E9,F`,
  `uv pip check --python .venv/bin/python`,
  `./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error`,
  and `./scripts/with-runtime-env.sh ./.venv/bin/python -m pytest -q`.
  The final suite reports **3,045 passed, 509 skipped, 7,358 subtests passed**.
- The validated installed optional runtime passed **124 tests and 755 subtests**
  through `scripts/test_reviewed_optional_runtime.py -q` against
  `tests/test_operation_contracts.py`, `tests/test_model_capabilities.py`,
  `tests/test_pipeline_schema.py`, `tests/test_studio_execution_specs.py` and
  `tests/test_modular_diffusers_upstream_contract.py`. The wrapper was
  `./scripts/with-runtime-env.sh .venv/bin/python`; no runtime was installed.
  Discovery was also tested with pipeline constructors and network access blocked.
- The six ledger/helper test files passed **65 tests and 817 subtests**, with
  one optional full-source-regeneration test skipped. That test requires the
  separately reviewed source checkouts/wheel inputs. The later M3 inventory audit
  has its own reproducible Diffusers-only check described above.
- Client `npm run check` passed **1,117 Node tests**, formatting, lint, types,
  catalog/style checks, production build and unchanged bundle budgets.
  Startup JavaScript is **601.8 KiB** (602 KiB limit); deferred JavaScript is
  **193.5 KiB** (194 KiB limit). The request suite accounts for **95 passing tests**.
- A fresh isolated backend passed three capability HTTP checks (55 unfiltered
  declarations, 20 Flux declarations, and an empty unmatched query), five exact
  served-file comparisons, and a Chromium production startup check. The browser
  loaded the deferred parser, completed workspace startup and reported no page
  errors. The client parser separately accepted all 55 captured HTTP declarations.
- These are metadata, contract, HTTP and startup checks. They do not qualify model
  generation, model swapping or cache reuse. No downloaded model files were changed.

M3.2b1 implementation (earlier checkpoint):

The existing standard adapter owners now project their loaders and task-level
operations through the same discovery endpoint. Schema version 2 adds task
identity, port visibility, pipeline-handle roles, and explicit loader versus
whole-pipeline decomposition. Image, video, audio and rendered-3D declarations
retain their real task fields and outputs; synchronized audio is limited to
capable adapters, prediction maps retain their type, and rendered 3D advertises
an orbit video rather than a mesh. Deprecated aliases remain callable without
creating duplicate entries. No new runtime actions or frontend family dispatch
were added.

Discovery and ordinary dynamic nodes share their existing adapter field overlays.
The parity regression exposed a missing `update_adapter_modes` callback on
`GenerateVideoAudio`; that node now delegates to the existing video callback,
which also fixes the retained `GenerateLTX2` alias. Query filtering includes
pipeline classes linked from matching capabilities as well as direct class
matches, including adapters without Studio catalog rows. The client validates
version 2 and normalizes the previous stage-only schema.

This earlier checkpoint covered only standard projection. The M3 completion
above adds Modular loaders, specialized stages, richer semantics, readiness and
ordinary insertion. Atomic model/task switching, selective recomputation and
developer extension work remain M4–M6. Presentation visibility and
`support: declared` confer no execution permission or cross-model compatibility.

M3.2b1 completion evidence:

- Client code: `c7fcef2`; backend code and matching served bundle: `b08c703`.
- Public discovery now reports **389 declarations across 136 pipeline classes**:
  55 Modular stages, 164 standard loader/task pairs and 170 standard pipeline/task
  operations. This is adapter metadata, not 136 newly qualified executable models.
- Backend gate passed: `uvx --from ruff==0.12.7 ruff check . --select E9,F`,
  `uv pip check --python .venv/bin/python`,
  `./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error`,
  and `./scripts/with-runtime-env.sh ./.venv/bin/python -m pytest -q`.
  The final suite reports **3,052 passed, 509 skipped, 8,200 subtests passed**.
- `./scripts/with-runtime-env.sh .venv/bin/python scripts/test_reviewed_optional_runtime.py -q`
  passed **131 tests and 1,597 subtests** against `test_operation_contracts.py`,
  `test_standard_operation_contracts.py`, `test_model_capabilities.py`,
  `test_pipeline_schema.py`, `test_studio_execution_specs.py` and
  `test_modular_diffusers_upstream_contract.py` in `tests/`. The installed runtime
  was verified, not reinstalled. Tests compare actual dynamic field callbacks
  with declarations and exercise discovery with network/process/node creation blocked.
- Client `npm run check` passed **1,119 Node tests**, formatting, lint, types,
  catalog/style checks, production build and unchanged bundle limits. Startup
  JavaScript is **601.8 KiB** (602 limit), deferred **193.7 KiB** (194 limit).
  The focused node-store request suite passed **46 tests**, including schema 1
  normalization, task identity, pipeline handles and rejection of malformed data.
- A fresh backend with isolated data paths passed three capability HTTP checks
  (389 unfiltered records, 76 Flux records, zero unmatched records), five exact
  served-file comparisons, and synchronous field-action HTTP checks for both
  `GenerateVideoAudio` and `GenerateLTX2`. A production Chromium startup completed,
  loaded the deferred parser and reported no page/asset errors. The client parser
  accepted all 389 captured HTTP records with exact value equality.
- All 200 canonical workflow and 78 public template records were unchanged before
  refreshing bundle/content fingerprints and the four dependent generated ledgers.
  Their changes are hashes only; no qualification or admission was upgraded.
- These are metadata, contract, HTTP and startup checks. No live model inference,
  cross-model swapping, cache-lifetime qualification or new installation was run.
  No downloaded model files were changed.

Inventory findings (M3.1):

The backend registry at runtime baseline `9e289b6` contains **228 ordinary actions
in 24 built-in modules**. This is the application inventory, not the complete
upstream pipeline inventory required by M3.3. It was inspected through the normal
verified optional-runtime entry point without downloading weights or running
model inference. All 228 action names resolved to their backend classes.

| Existing surface                                | Count | Owner and consolidation direction                                                                                                                                                                                                                                                    |
| ----------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Modular stages and supporting operations        |    18 | `modules/ModularDiffusers`: includes `ModelsLoader`, `EncodePrompt`, `Denoise`, `DecodeLatents`, `ImageEncode`, component/adaptor utilities, `DynamicBlockNode` and `ReviewedModularWorkflowStep`; preserve these execution identities while defining public operation capabilities. |
| Modular workflow-stage adapters                 |    57 | `modules/ModularDiffusers/workflow_blocks.py`: shared and family-specific `Workflow*` actions. Bind their real input/state/component contracts behind generic operations; do not infer equivalence from their labels.                                                                |
| Standard image pipelines and tasks              |    15 | `modules/DiffusersImage`: `LoadPipeline`, generation/edit/control/prediction tasks and supporting operations. Reuse this path for whole-pipeline execution where decomposition is unavailable.                                                                                       |
| Standard video pipelines and orchestration      |     8 | `modules/DiffusersVideo`: loading, video/audio generation, shots and sequences. Preserve additional temporal and audio contracts.                                                                                                                                                    |
| Standard audio pipelines and adapters           |     5 | `modules/DiffusersAudio`: loading, generation and adapter controls. Keep audio-specific inputs and outputs.                                                                                                                                                                          |
| Standard 3D pipelines                           |     2 | `modules/DiffusersThreeD`: loading and rendered-artifact generation; do not force a text/image four-stage representation.                                                                                                                                                            |
| Diffusers resource operations                   |     9 | `modules/DiffusersRuntime`: component inspection/loading, execution recipes, memory planning and release. Most belong in resource configuration or deliberate advanced authoring.                                                                                                    |
| Diffusers adapter operations                    |     7 | `modules/DiffusersAdapters`: LoRA loading/mixing, inspection, hotswap, fuse, unload and comparison jobs; consolidate discovery without replacing runtime semantics.                                                                                                                  |
| Other official-library tasks                    |    10 | `modules/HuggingFaceSpeech` and `modules/HuggingFaceTransformers`; keep their optional-runtime and task contracts distinct.                                                                                                                                                          |
| Media, tensor, primitive and workflow utilities |    97 | Remaining built-in modules; these are useful composition operations, not duplicate diffusion stages.                                                                                                                                                                                 |

The separately validated HF library at Diffusers revision
`2f7e0154a9db246e95c9ede43edba7db5b130805` contains **49 task contracts,
135 workflow definitions (127 Diffusers and 8 Transformers), and 559 reusable
block definitions**. Its authoritative source is
`modiff/huggingface_node_library.py` plus the reviewed workflow/block snapshots.
A workflow definition binds a pipeline and task; block definitions describe shared
or nested implementation pieces. These counts do not represent 559 independent
user operations or 135 qualified model executions. M1 already keeps these deeper
layers out of the default discovery views.

| Apparent duplication                                                                                | Classification and required treatment                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multiple catalog keys for the same runtime identity                                                 | Exact discovery aliases: the client's `nodeCatalogEntries` already deduplicates them and retains their search terms. No two of the 228 inspected backend action names were bound to the same class object; that alone does not prove distinct semantics. |
| `DiffusersVideo.GenerateLTX2` and `GenerateVideoAudio`                                              | An explicit deprecated compatibility subclass, not another public operation. Keep the saved action readable; consolidate new discovery and retire only after M8 compatibility proof.                                                                     |
| `Denoise`, `WorkflowImageDenoise`, family-specific `Workflow*Denoise`, and reviewed block execution | Overlapping purpose with different bindings, states and component contracts. M3.2/M3.4 need common operation metadata and reviewed backend adapter selection, rather than deleting classes or renaming serialized graphs.                                |
| Standard `Generate` and a Modular stage chain                                                       | Different decomposition levels. Expose callable whole-pipeline support honestly when a stage implementation is unavailable.                                                                                                                              |
| Family/task workflow definitions and their nested blocks                                            | Composition and implementation layers. Public task/stage selection should resolve these underneath; deeper inspection remains deliberate.                                                                                                                |

Existing extension points are `ModiffPipelineRegistry` / `PipelineConfig.node_specs`
for generic Modular fields and implementations, modality adapter registries for
standard pipelines, and the bounded reviewed HF workflow/block/task contracts.
M3 adds a common operation/capability description to these owners without
creating independent runtime implementations. The completed M3.3 inventory
reconciles every pinned upstream pipeline/task, including unavailable
optional dependencies, contract-only entries and genuinely missing adapters.

### M4 — Deliver stage-first authoring and atomic model/task changes

- [x] **M4.1** Create small generic starters from capabilities and support the equivalent
      graph constructed manually through typed connections.
- [x] **M4.2** Adapt fields/ports from the selected loader contract. Preserve connected values
      and distinguish defaults, user overrides, and retained unsupported settings.
- [x] **M4.3** Plan model changes before applying them. Commit compatible changes as one
      transaction; preserve incompatible custom nodes with actionable diagnostics.
- [x] **M4.4** Extend tasks by connecting required conditioning. Use actual upstream task
      selection rules; do not assume img2img and instruction editing are identical.
- [x] **M4.5** Expose implementation inspection separately from optional structural editing.
      Preserve existing deep Blocks without lossy automatic conversion.

M4 completion:

- Connected drafts derive ordinary nodes and exact state/component wires from
  the existing backend owners (`POST /operations/starter`). All 250 task bindings
  with operations pass endpoint and single-writer checks without constructing nodes
  or accessing the network. Output Preview/Save/Export nodes remain an explicit
  typed connection, documented in the starter preview.
- Native manual wiring and reload pass for Qwen, Anima and standard Stable Audio.
  The Qwen → Flux → image-to-image journey passes native prompt editing, retained
  negative-prompt inspection, Undo/Redo and Save/reopen.
- Pure planning and the existing history transaction cover custom-node retention,
  incompatible wires, unchanged IDs/positions, new IDs for changed Python actions,
  stale previews, shared-loader rejection and rollback. Initial field defaults
  and the native random-seed object have separate regression coverage.
- Implementation inspection remains read-only. Existing nested Blocks retain
  their structure and their existing composition/structural editing commands.
- The live SDXL check exposed a missing required VAE wire. Drafts now complete
  exact required component bundles from their operation contracts, with regression
  coverage that does not invent that dependency for Flux or Qwen.
- Stateful task changes exposed a seed mismatch between a new image encoder and
  the retained denoiser. Backend-declared shared seed groups now preserve the
  authored value, mirror native edits under one Undo transaction, reuse one
  random draw per run, and preserve a connected custom source. Imported hints
  cannot cross loader branches, act on malformed bindings, or rewrite Blocks.
  Runtime seed/state validation is unchanged.

- [x] **M4 acceptance** Validate the integrated build and representative real
      executions, retaining submitted and consumed inputs separately from UI/schema proof.

Acceptance: native editing, drag-to-connect, Save/reopen, Undo/Redo, model switching,
and task changes pass across representative distinct families and modalities.
Real executions verify effective inputs, not just successful schema construction.

M4 implementation commits: backend `3d89d99026bf35177ec508bab76bc4ed9bbda48d`
(includes the matching compiled client), client
`c08b02b81d8d869c8ceef96be9486e40fb979a38` (pins CI to that backend).
Both remain on `feat/generic-diffusers-workbench`.

M4 validation:

- Backend gate: `uvx --from ruff==0.12.7 ruff check . --select E9,F`,
  `uv pip check --python .venv/bin/python`, wrapped `python -m modiff.preflight
--json --check-port 8088 --fail-on-error`, and wrapped `python -m pytest -q`
  pass. The base suite records **3,072 passed, 509 skipped and 9,311 subtests**.
- Verified optional-runtime entry point: operation starters/catalogs, Modular
  upstream/workflow contracts, standard operations and workflow blocks record
  **118 passed and 2,453 subtests**. The skipped base tests are not counted as
  model execution coverage.
- Client: `npm ci`, `npm run check`, and final formatting/lint/type checks pass.
  Thirteen authoring regression tests include rollback, retained values, default
  initialization, native seed objects, shared edits/random export, connected
  sources, persistence and malformed/detached hints. All **250** fresh backend
  starter payloads also pass the strict client parser.
- Native Chromium: **six tests** pass via `npx playwright test
tests/e2e/studio-mocked/studio-mocked.spec.ts --grep 'operation starters preserve|operation stages can|Expert resolves canonical|stateful task changes'`.
  These cover model/task changes, seed editing, manual wiring, Undo/Redo,
  Save/reopen and stale selection; they use genuine backend schemas with mocked
  execution and make no weight/runtime claim.
- Production HTTP/UI: six served assets match the checked build byte-for-byte.
  Startup JavaScript is **603.0 KiB gzip**, deferred JavaScript **203.9 KiB**;
  bounded ceilings are 604/204 KiB with unchanged individual chunk limits.
  Bundle evidence refresh preserves all **200 workflow** and **78 template**
  semantic records; only their bundle-dependent hashes change.
- Real execution: production Chromium authors the SDXL four-stage graph, adds a
  typed Preview, changes it to image-to-image, and uploads the generated image
  through a normal Load Image node. SDXL revision
  `462165984030d82259a11f4367a4eed129e94a7b` runs locally on ROCm/gfx1151 using
  existing cached weights, float16 and model CPU offload. Submitted and consumed
  receipts agree on the authored prompts, 512×512 dimensions, 20 configured steps,
  fixed seeds 4109/4111, and one shared random seed in both stateful stages.
  Text-to-image and both fixed/random image-to-image runs complete with retained
  image artifacts and no browser errors. Logs, receipts, media and isolated
  server workspaces stay outside the repositories. No weights are downloaded
  or removed. This qualifies those authored checks on this host, not every
  family, modality, operating system, GPU or M5 cache behavior.

### M5 — Make repeated runs reuse components and unaffected results

- [x] **M5.1** Reproduce repeated-run behavior and capture loader, encoder, denoiser, and
      decoder execution counts alongside component ownership and memory placement.
- [x] **M5.2** Separate component lifetime from node-output cache lifetime. Cache identity
      includes consumed inputs, model/component revisions, relevant adapters, dtype,
      implementation revision, and custom-code revision.
- [x] **M5.3** Exclude layout, collapse, selection, mode presentation, and progress metadata.
      Keep mutable generators per execution; never share stale mutable pipeline state.
- [x] **M5.4** Prefer retained components when memory permits. Make pressure eviction and
      multi-owner scheduling explicit and preserve shared-owner references.
- [x] **M5.5** Publish bounded reuse/reload/offload reasons and provide deliberate output
      recomputation and model-release controls.

Acceptance matrix: seed-only and step-only edits reuse unaffected conditioning;
prompt edits re-encode only affected stages; model/adapter/code changes invalidate
dependent data; mode/layout changes do not reload; cancellation and OOM recovery
do not reuse invalid state. Test shared components and real constrained-memory
runs separately. CPU offload is not equivalent to destruction/reload.

M5 implementation and scoped acceptance (**Complete**):

- [x] Measure real SDXL stage execution, component object identity, collection
      ownership and placement across unchanged, seed, step and prompt edits.
- [x] Separate explicit output invalidation from node destruction. The backend
      retains model owners and current previews; native context/toolbar actions
      expose **Recompute on next Run** and **Release node cache**.
- [x] Publish bounded per-node reuse/recompute reasons through the existing
      execution messages and node status, without serializing runtime objects.
- [x] Retain independent Auto owners when the combined memory envelope fits.
      Existing release scheduling still handles lower-capacity cases; shared
      memory is counted against system RAM. Validate both strategies and shared
      ownership with focused executor/planner tests.
- [x] Reproduce and fix cached generator mutation in official split-workflow
      continuations. Each consuming call clones the saved _current_ random state,
      retaining stream position without advancing the reusable upstream snapshot.
      Seed/device checks remain enforced. This is contract/test evidence for the
      affected workflow adapter, not live LTX-2.5 model qualification.
- [x] Complete the consumed-input and implementation/custom-code identity audit,
      including mutable inputs and adapter/component replacement.
- [x] Complete the expanded Pipeline State and scheduler lifetime audit. Preserve
      exact route-state authority and deliberate sharing within one execution.
- [x] Qualify retained/shared owners, targeted release, pressure eviction and
      cancellation/OOM recovery with real model execution. Individual loader
      removal does not establish that downstream pipeline references are freed;
      verify actual live references and memory, not just registry removal.

The real SDXL production-browser matrix uses cached pinned weights, 512×512,
float16 and model CPU offload. Counts below exclude the ordinary preview node:

| Run/edit                                | Loader calls | Prompt encoder calls | Denoiser calls | Decoder calls |
| --------------------------------------- | ------------ | -------------------- | -------------- | ------------- |
| Cold run                                | 1            | 1                    | 1              | 1             |
| Unchanged run                           | 0            | 0                    | 0              | 0             |
| Seed only                               | 0            | 0                    | 1              | 1             |
| Steps only                              | 0            | 0                    | 1              | 1             |
| Prompt only                             | 0            | 1                    | 1              | 1             |
| Auto/Expert view toggle and Arrange     | 0            | 0                    | 0              | 0             |
| Explicit prompt-output recomputation    | 0            | 1                    | 1              | 1             |
| Explicit loader cache release, then Run | 1            | 1                    | 1              | 1             |

Component objects stayed identical through the first seven runs. Explicit loader
release caused new component objects on the eighth run. All eight runs completed
with retained output receipts and no browser errors. This establishes selective
recomputation for this SDXL path; it does not qualify every family or prove that
all references were freed by individual loader removal. Downloaded model files
were preserved. An unchanged Run reused outputs and is not an additional
independent inference-quality proof.

The initial memory investigation found approximately 113 MB of additional
accelerator allocation per recomputation and retained downstream pipelines during
loader release. Allocation traces identified per-thread math workspaces as the
source of the incremental growth. Graph model work and teardown now use one
worker; downloads preserve the existing I/O serialization policy. Explicit cache
release follows cached input dependencies and preserves other owners' components.

Final production-browser SDXL runs, without execution-method instrumentation,
kept allocation between **7.165 and 7.167 GB** across the eight cases above.
Release/reload returned to **7.165 GB**. The same seed/settings produced the same
pixel hashes after output recomputation and model reload. Browser errors: **zero**.
The separate expanded SDXL graph produced identical pixels to the canonical
four-stage graph at both tested seeds; denoiser-only recomputation also matched.
Expanded continuations copy state, schedulers and guiders while sharing weights.

Separate live resource acceptance used an enforced **12 GiB PyTorch allocator
budget** on the existing shared-memory accelerator:

- Two loaders shared all seven managed components. Releasing one retained the
  other's exact component identities; its next seed-edited run reused the loader.
- Cold and seed-edited inference completed, with allocation differing by less
  than 1 MB. A deliberate **5 GiB** budget caused a genuine allocator OOM.
- The original OOM reproduction exposed traceback-held tensors after registries
  were cleared. Recovery now releases completed synchronous frames after
  recording diagnostics, preserving suspended queue coroutines and error types.
- After both OOM and backend cancellation (`POST /stop`) during denoising, node caches
  and collections were empty and allocation fell to **113,246,208 bytes**.
  Restoring the 12 GiB budget completed the unchanged failed workload. Retrying
  the cancelled 100-step workload and explicitly recomputing it produced identical
  pixels and approximately **7.165 GB** retained allocation.

These are real SDXL executions on the available ROCm shared-memory host with a
process-enforced accelerator budget. They do not qualify a physical Windows
16 GB GPU/32 GB RAM configuration or every Diffusers family. Automatic owner
scheduling and shared/disk-owner rules also have focused contract coverage;
this is not blanket live qualification of every Auto recipe. All downloaded
models were preserved, and the live processes used offline cached pinned weights.

M5 checkpoint implementation commits: backend
`897e27a9f3292f6ecd84fa54bb088c3c4ebf5fba` and client
`7523fc347502176b8f91fc3b104a46f75e13c960`. Both remain on
`feat/generic-diffusers-workbench`; no branch was pushed. These introduced the initial recomputation controls; the completion fixes and
validation are recorded below.

M5 checkpoint validation:

- Backend: `uvx --from ruff==0.12.7 ruff check . --select E9,F`,
  `uv pip check --python .venv/bin/python` (67 packages), and the managed-runtime
  preflight with `--check-port 8088 --fail-on-error` passed.
  Final managed-runtime `python -m pytest -q`: **3,083 passed, 509 skipped,
  9,311 subtests passed**, with two existing upstream/runtime warnings.
- Verified optional-runtime gate: **126 passed, 6 subtests passed** across
  workflow continuation, LTX-2.5 contract, NodeBase, owner planning and cache
  cleanup tests. No runtime repair or model installation was performed. An initial
  concurrent activation could not acquire the startup lease; the sequential
  retry passed without changing the runtime.
- Client: `npm ci`, `npm run check`, and the final documentation formatting gate
  passed. Three native Playwright scenarios passed: cache controls, model/task
  authoring, and shared-seed task changes. The response parser rejects an older
  backend's incompatible recomputation response.
- The first full backend gate found a historical assertion requiring identical
  Generator objects. Its replacement retains exact continuous-stream comparison
  and adds cached-state preservation/retry assertions; the final full gate passed.
- Production assets matched all **67** generated files and six fresh HTTP asset
  responses. Dependent coverage ledgers changed only their bundle/source hashes;
  all **200** canonical workflows and **78** template entries stayed unchanged.
- The new controls measured **204.1 KiB** of deferred JavaScript against the
  previous 204 KiB cap. The bounded cap is now **205 KiB**; startup (603.3 KiB)
  remains within 604 KiB, and both individual chunk limits are unchanged.

M5 completion validation:

- Mutable container/image/array/tensor input changes and loaded implementation
  changes invalidate cached results. Exact Modular route/component authority
  checks remain enforced before reuse. Custom source edits require explicit
  module refresh; M6's new development flow remains separate.
- Loader adoption transfers manager collections, cached dependency ownership and
  exclusive disk-offload metadata without replacing hooks or their storage.
  Standard-manager release also preserves shared owners and avoids CPU copies.
- Ruff E9/F, dependency compatibility (67 packages), preflight and client
  `npm ci` / `npm run check` passed. The rebuilt client matches all 67 backend
  bundle files, so no bundle or coverage-ledger changes are required.
- Verified optional-runtime suite: **218 passed, 248 subtests passed**, with
  three upstream/runtime warnings. The final SDXL pipeline fixture also passed
  the verified optional-runtime gate; it is correctly skipped in the base
  environment where Transformers is deliberately absent.
- Native mocked browser checks: cache controls and stateful task changes passed;
  model/task preview hit its existing graph-change guard once and passed on an
  isolated retry against unchanged client source. The original failure and trace
  were retained privately; no timeout or guard was weakened.
- Final full base-runtime `python -m pytest -q`: **3,093 passed, 510 skipped,
  9,311 subtests passed**, with two existing upstream/runtime warnings. The first
  run found a missing optional-runtime guard on the new real SDXL fixture;
  the corrected base gate passed and that fixture separately passed with the
  verified optional runtime. No runtime package was installed to bypass the
  boundary.
- The fresh backend served six checked HTTP assets byte-for-byte, all 67 client
  bundle files matched, and its process-start source fingerprint matched the
  final implementation. The owned smoke server was stopped only after the queue
  was empty. No downloaded model files were deleted.

M5 completion implementation commits: backend
`b2e8ec5af979f1a08088a82d406c6515da72a155` and client
`c3a6fb128c3987fc5b2dca7d950d236c270ae273`. The client CI backend pin
references this completed runtime implementation. Both repositories remain on
`feat/generic-diffusers-workbench`; no branch was pushed.

### M6 — Make custom node development a coherent product flow

- [x] **M6.1** Document and unify discovery of existing local/Git Python modules and Hub
      Modular blocks. Reuse schema-derived UI and compatible Diffusers metadata.
- [x] **M6.2** Design an explicit install/enable boundary for executable code. Expert mode
      alone is not authorization; browsing/import preview does not execute code.
- [x] **M6.3** Revise SECURITY/AGENTS/contributor policy together with the executable path.
      Retain immutable Hub revisions, dependency visibility, input/path validation,
      and the local single-user boundary. Do not bypass the current checks ad hoc.
- [x] **M6.4** Provide local developer reload, import diagnostics, exact-code identity, and
      targeted cache invalidation. Do not mislabel the process as a Python sandbox.
- [x] **M6.5** Register installed custom nodes in normal search and typed suggestions; allow
      installed nodes to be used in Auto without granting additional permissions.

Acceptance: create a small local node and a pinned custom Modular block, render
their declared UI without frontend changes, connect and execute them, edit/reload,
and observe correct cache invalidation. Test rejection/cancellation paths and no
installation during discovery. Custom support must not create another executor.

M6 completion:

- Expert → Nodes → Custom nodes and Models share one source review panel for
  local folders, immutable Git commits and pinned Hub Modular blocks. Stage and
  inspect read bounded source/metadata without importing submitted Python,
  installing dependencies, copying model weights or granting approval. Existing
  directories require review before startup imports resume.
- Enable and reload require explicit consent for the inspected source and declared
  installed dependency hash. Approvals live outside source packages. Changed code
  is rejected before graph/cached-node or custom-field execution. Missing packages,
  failed imports and malformed metadata have visible diagnostics. Moving-branch
  updates are retired; a new remote revision is staged and reviewed explicitly.
- Enabled definitions use the existing node registry, schema renderer, normal
  search, typed suggestions, executor and cache. The new panel loads on demand.
  Auto uses approved data nodes and connected-component contracts without new
  trust; manual resource declarations require Expert. These declarations do not
  grant model/hardware qualification or execute custom suppliers during planning.
- Custom Modular blocks use native Diffusers `from_config`, `init_pipeline`,
  pipeline calls and the existing ComponentsManager. Isolated approved Python
  snapshots avoid stale bytecode and shared upstream local-package aliases;
  source `main.py` also stays separate from MoDiff's dispatch adapter. Connected
  weights remain shared; the adapter never implicitly loads default model repos.
  The historical contract-only Dynamic Block path remains fail closed.
- Reload/disable use the existing executor/cache lease, reject running or queued
  work, and release only affected cached nodes and transitive consumers. A cancelled
  HTTP request cannot drop the lease while an approved import is still executing.
  Unrelated component owners remain resident. Python import side effects can still
  require a restart; this is not a sandbox. Schema edits require reinserting or
  updating existing nodes rather than silently rewriting saved parameters.
- Backend `docs/custom-nodes.md` includes runnable PromptTools and ModularPrompt
  examples, source layout, typed fields, dependency/resource declarations and API
  requests. SECURITY, AGENTS and contributor guidance were updated in both repos.

M6 completion evidence:

- Backend full base gate: `./scripts/with-runtime-env.sh .venv/bin/python -m pytest -q`
  — **3,121 passed, 510 skipped, 9,311 subtests passed**, with two existing
  upstream/runtime warnings. Ruff E9/F, dependency compatibility (67 packages)
  and managed preflight passed.
- Verified optional-runtime gate: `scripts/test_reviewed_optional_runtime.py -q`
  over `tests/test_custom_extensions.py`, `tests/test_custom_modular_identity.py`,
  `tests/test_workflow_auto_resource.py` and `tests/test_node_cache_cleanup.py`
  — **165 passed, 133 subtests passed**, with one upstream warning. It used the
  approved runtime wrapper and installed no packages. The **28** extension cases
  include dependency/source drift, approval cancellation, import failure, relative
  helpers, separate packages, `main.py`, disabled browser assets, bounded paths,
  Windows Git drive paths, native connected Torch components and startup approvals.
- Client `npm ci` and `npm run check` passed, including **1,142 Node tests** across
  the unit and bundle checks. `npm run check:ui` passed **160 native Chromium
  scenarios**: 2 shared-control and 158 mocked-backend scenarios. The new scenario
  verifies review cancellation/consent, normal insertion, typed wiring and Auto
  discovery. The complete production startup measures **604.1 KiB**, bounded at
  **605 KiB**; deferred code measures **206.2 KiB**, bounded at **207 KiB**.
  Individual startup/deferred chunk limits are unchanged.
- A fresh production browser staged and enabled both model-free examples, then
  authored Text Value → Prompt Prefix → Modular Prompt → Data Viewer using normal
  controls and typed drag suggestions. Four real backend runs verified cold output,
  unchanged reuse, changed output after approved reload, and unchanged reuse in
  Auto. Reload released exactly the three custom/downstream cached nodes; the Text
  Value cache survived. There were no browser page errors. Screenshots, graph
  requests, terminal receipts and read-only cache observations were retained privately.
- Git staging exercised an actual local Git transport at an exact commit without
  checking out weights. Hub staging used a pinned snapshot transport fixture and
  executed its real native Modular block; this is not a live external Hub download
  or a third-party model qualification claim. Earlier live harness attempts had
  incorrect control/accessible-name selectors and canvas timing; corrected native
  interactions passed without changing production behavior or weakening guards.
- All **69** generated client files match the backend bundle. The fresh server's
  HTML and six checked HTTP assets match, and its process-start source fingerprint
  matches the final executable source. Existing Gallery data and downloaded models
  were preserved. Five dependent ledgers changed only **18 hash bindings**; all
  **200** canonical workflow and **78** public template records stayed unchanged.

At M6 close, M7 and M8 were still pending; the current M7 status is recorded below.
No new live diffusion-model or Windows hardware qualification is claimed.

M6 implementation commits: backend
`12fb46efd9c886bb55e70121a0f003d071972a7f` and client
`1e232ac6a5f88a69e7c3d549ebbcb8fb2fbea80b`. The client CI backend pin
references this completed implementation. Both repositories remain on
`feat/generic-diffusers-workbench`; no branch was pushed. The owned live smoke
server was stopped after verifying that its queue was empty.

### M7 — Add transparent developer setup and service prototyping

- [x] **M7.1** Design a documented `uv` backend path and `npm` client path for the supported
      runtime profiles. Keep guided installation and existing environments working.
- [x] **M7.2** Reconcile `uv` management, constraints, platform Torch sources, optional
      runtimes, and reproducibility before documenting commands as supported.
- [ ] **M7.3** Publish working Windows and Linux commands; test clean environments without
      replacing a user's accelerator packages or downloading inference weights.
- [x] **M7.4** Reuse API graph export for a reproducible execution package with dependencies,
      model revisions, custom-node identities, and named service inputs/outputs.
- [x] **M7.5** Treat standalone Python generation for arbitrary graphs as a separate feature;
      avoid claiming that a MoDiff API package is independent Diffusers Python.

Acceptance: clean setup and health checks on supported platforms; a saved workflow
executes through the documented API with equivalent resolved inputs. Credentials
and local machine paths do not enter portable packages.

#### M7 implementation and validation — 2026-09-18

The implementation is complete on both feature branches. The operator deferred
Windows execution until after implementation and publication, when a Windows
machine will be provided. M7.3 remains an explicit qualification task; it does
not block M8 implementation or the authorized feature-branch push.

M7.3 platform evidence:

- [x] Publish the exact uv/npm developer flow in the backend guide and client
      Windows/Linux guides, including preservation of existing environments and
      the optional guided launchers. Contributor and documentation indexes link
      to this path.
- [x] Linux: isolated clean CPU installation through the exact documented uv command;
      preflight reports ready; script-free supervised launch responds to health;
      saved API graph and Manual/Auto service calls produce equivalent text output.
- [x] Linux client: `npm ci` and the complete client quality gate.
- [ ] Windows: run the committed CPU setup/check/service CI job or the documented
      commands in a clean Windows environment and record the result. This host
      is Linux with no Windows runtime. GitHub repository access is available,
      and publication is authorized. This shell has no GitHub credentials for a
      history-preserving push. The connected GitHub integration also rejects blob
      creation with HTTP 403. No feature branch has been published or Windows job
      started. Windows execution evidence is pending.
      Accelerator/model qualification remains separate from this CPU setup check.

Delivered behavior:

- `modiff.dev` is a standard-library bootstrap for the existing installer and
  runtime. `plan` is read-only, `setup` refuses an existing `.venv` unless `--repair`
  is explicit, and `check`/`run` use the installed Python and accelerator process
  environment. Guided launchers, reviewed Torch sources, optional-runtime consent,
  staged promotion and rollback remain shared.
- Expert's lazy **Export → Service package** dialog reuses the existing API export
  and Modular/Block lowering. It declares required named scalar inputs and named
  persisted preview outputs, rejects duplicate/invalid names, and refuses stale
  graph/context changes. Auto retains its existing Export menu.
- Backend `/service_package` inspection/build/preparation records the existing
  concrete graph, backend source identity, Python/profile/dependency contract,
  observed installed versions, required optional profiles, explicit immutable
  model references and approved custom-node code/dependency identities. Active
  overlay distributions take precedence over shadowed base packages.
- Named inputs omit their current values. Session/UI snapshots are excluded;
  execution hints remain subject to portability checks. Detected credentials,
  local paths/models, mutable model revisions and connected model selectors block
  export with an actionable error. This first interface supports scalar values;
  custom Python's hidden dependencies cannot be inferred automatically.
- The CLI prepares and submits through ordinary `/graph`, then retrieves exact
  task/node/field results from `/runs`. Multi-item durable media references are
  projected without copying private graph snapshots. Timeouts do not resubmit or
  cancel work. Requirements are checked again when execution begins; Auto gets a
  fresh receipt for the invocation's resolved values. No alternate executor,
  automatic package/code install, model download, or code approval was added.
- Developer setup and service-prototyping guides plus a model-free API graph,
  interface and inputs example are committed. Standalone arbitrary-graph Python
  generation is explicitly separate.

Validation:

- Backend final gate: Ruff E9/F and `uv pip check` passed; preflight ready;
  **3,147 passed, 510 skipped, 9,311 subtests**, two existing warnings.
- Verified optional-runtime entry point, service/setup/custom-extension tests:
  **54 passed**. These are no-download contracts/custom-node tests, not diffusion
  model output qualification.
- Client `npm ci` and `npm run check` passed, including **1,144 Node tests**.
- `npm run check:ui`: shared controls **2 passed**; initial mocked suite
  **155 passed, 3 failed**. The StableAudio native-drag case passed on unchanged
  rerun. Two template cases encountered remote asset HTTP 503/queue errors and
  passed using the existing `MODIFF_E2E_TEMPLATE_INPUT_CACHE` hook against the
  installed content-addressed assets, with normal checksums enforced. Every case
  has a passing result; the initial full sweep was not a single green run.
- Fresh production browser: authored Text Value → Data Viewer using native
  controls, exported raw API and named service JSON, verified equivalent prepared
  node parameters/paths, and completed the service through the real queue with
  the expected durable output. No browser page errors.
- HTTP smoke in both existing and clean Linux CPU environments: health, saved
  graph, Manual/Auto service reuse, explicitly approved custom source, and rejection
  after custom source edits passed. No inference weights were downloaded.
- **70 generated client files** match the backend bundle; HTML and its **six**
  referenced assets matched HTTP bytes, and the worker source fingerprint matched
  current source. All owned test queues were empty before shutdown.
- Startup gzip is **604.2 KiB** under the unchanged **605 KiB** ceiling. The lazy
  service dialog takes deferred aggregate gzip to **208.3 KiB**, bounded at
  **209 KiB**; both per-chunk ceilings stay unchanged. The five generated evidence
  files changed only bundle/content-hash bindings: all **200 workflows** and
  **78 templates** retain their semantic evidence.

Downloaded models, the installed Gallery cache and backend-owned `web/user`
were preserved. M8 was not started. Both branches remain local; no push occurred.

M7 implementation commits: backend
`875176b3596ee569debfb0f185df3e68a1e24591` and client
`8061db1a90210af1467f45fccda0e7782f12b878`. Client CI pins this backend implementation.
M7.3 Windows execution remains unchecked. These local commits do not constitute
Windows or new model/hardware qualification.

### M8 — Retire redundant public surfaces and qualify the product

- [x] **M8.1** Promote canonical operations after replacement coverage exists. Keep legacy
      identities loadable and offer explicit migration where semantics change.
- [x] **M8.2** Rename cluster presentation to Blocks consistently, without bulk ID rewriting.
- [x] **M8.3** Replace default implementation catalogs with contextual inspection. Retain
      templates as optional starters and user-owned Blocks as independent revisions.
- [x] **M8.4** Validate the complete Auto and Expert journeys, accessibility, nested editing,
      persistence, API export, repeated runs, and compatibility with older workflows.
- [x] **M8.5** Retire backend implementations only after proving replacement equivalence and
      migration behavior; similar labels are not sufficient evidence for deletion.

#### M8 implementation checklist

- [x] Consolidate raw-node discovery behind exact bound canonical operations;
      preserve Advanced access and fallback for older/incomplete catalogs.
- [x] Include ordinary text/value utilities in Expert Stages.
- [x] Present registered compositions as Blocks and the reusable library as Saved
      Blocks; retain serialized identities, names, revisions and legacy group keys.
- [x] Open stage implementation/settings from its canvas selection, independently
      of the pipeline picker; keep inspection read-only and scoped to the workflow.
- [x] Verify the deprecated video/audio alias still delegates to its canonical
      implementation. Retain distinct backend adapters and legacy readers.
- [x] Run local contract/browser journeys, rebuild the paired bundle, validate
      backend gates and served bytes, and commit the reviewed result.
- [ ] Push both feature branches and record their exact published commits.

Windows clean installation and live Windows/model qualification are deferred by
operator instruction. Local UI, API and contract checks remain required. No new
hardware/model support is implied by consolidation or by publishing this branch.

#### M8 implementation and local evidence — 2026-09-18

Local implementation commits: backend
`9931f22b5c0a11a42ce55ea6c37106a984886b35`; client
`2cd222d86c74cdc2932b236296bf191e58e67701`. The subsequent client
publication-preparation commit pins CI to that backend commit. Documentation and
CI-pin commits do not change the tested runtime or the bundled frontend source.
Final bundle SHA-256:
`b93f21468d0147c2f0f21bc3916627657db69ff2d3c5d5367440d11c06d2e23b`.

Expert Stages now suppresses an ordinary runtime entry only when an exact bound
canonical operation has matching pipeline/task support. Incomplete or older
catalogs retain the ordinary fallback; Advanced keeps implementation access.
Text/value utilities and enabled custom nodes remain available in Stages.
The full runtime registry and saved workflow action identities are untouched.

Public composition controls use Blocks and Saved Blocks. Serialized categories,
library group keys, IDs, revisions and embedded snapshots remain compatible.
Existing migration preview/apply and legacy readers are preserved. The stage
inspector is shared between the operation panel and the canvas selection toolbar;
it works with the library closed, uses native dialog focus management, and closes
when its selection, workflow or authoring mode becomes invalid. Inspection does
not resolve a new graph, load models or change saved parameters.

M8.5 review found no further backend deletion justified by replacement evidence.
The deprecated `GenerateLTX2` action already inherits the generic
`GenerateVideoAudio` implementation. A strengthened verified-runtime test checks
its execute method, parameter object and adapter callback identity. The hidden
alias stays loadable. Other adapters have distinct state/component contracts;
similar presentation is not evidence that they are interchangeable.

Validation commands and scope:

- Client `npm run check`: 1,145 Node tests, formatting, lint, types, catalog byte
  golden checks, production build and bundle budgets. Startup is 604.3 KiB gzip;
  deferred code is 209.3 KiB. The lazy shared inspector increases the deferred
  ceiling from 209 to 210 KiB; startup and individual-chunk ceilings stay fixed.
- `MODIFF_E2E_TEMPLATE_INPUT_CACHE=<existing-installed-input-cache> npm run check:ui`:
  two shared-control tests passed; the full mocked sweep initially passed 158/159.
  The remaining Anima wiring test captured handles during Arrange animation.
  Native actionability checks now precede coordinate capture; all three pipeline
  wiring cases passed twice (six passes), preserving the actual pointer gesture.
- The new canvas inspector test verifies visible content, Enter/Escape, focus
  restoration, unchanged graph and Auto/Expert visibility. Its initial repeated
  run exposed asynchronous model validation changing the baseline snapshot.
  Waiting for the fixture's `ModelsLoader` validation before capture fixed the
  harness; three final repetitions passed. No application focus workaround was
  introduced. Initial failure traces are retained outside the repositories.
- Backend `uvx --from ruff==0.12.7 ruff check . --select E9,F`, `uv pip check
--python .venv/bin/python`, runtime-wrapped `python -m modiff.preflight --json
--check-port 8088 --fail-on-error`, and runtime-wrapped `python -m pytest -q`:
  ready; 3,147 passed, 510 skipped, 9,311 subtests and two existing warnings.
- Verified optional-runtime alias regression through
  `scripts/test_reviewed_optional_runtime.py`: one passed, 115 deselected.
- A fresh isolated production server and native browser checked stage inspection
  with the library closed, Text Value discovery in Stages, typed Data Viewer
  insertion, named service export, API graph equivalence, normal queue execution
  and durable text output. CLI repetition returned the same named result.
  The process source fingerprint and served HTML/assets matched disk; all 71
  emitted client files matched the backend bundle. The owned server was stopped
  only after its queue was empty. This is model-free API/UI proof.
- Bundle evidence regeneration changes hashes only: 200 canonical workflows and
  78 public template records retain their existing semantics and qualification.
  Models, Gallery media, saved Blocks and user workflows were preserved.

Publication is authorized but blocked by credentials. Noninteractive HTTPS Git
push fails because no username/credential helper is configured. SSH has no agent
or default identity configured. The connected GitHub integration can read the
repositories, but `create_blob` returns HTTP 403, "Resource not accessible by
integration." No remote source, branch or snapshot commit was created. Keep the
publication checkbox open until both branch tips are verified on GitHub. The
client CI must pin the paired backend implementation commit, and the backend
branch must be pushed first so that pin is reachable. All original local feature
history remains intact.

#### Downloaded-model validation on Linux — 2026-09-18

- [x] Inventory existing cached revisions and use an isolated production server,
      native browser gestures and ordinary graph dispatch with Hub offline mode.
- [x] Validate SDXL, FLUX.2 Klein, Qwen Image 2512 and Z-Image Turbo through the
      same four Modular stages plus Preview; preserve graph node identities and
      compatible user inputs when switching families through the operation panel.
- [x] Verify unchanged, seed-edit, prompt-edit and explicit-recompute behavior for
      each family. SDXL additionally covers step edits, layout-only changes and
      explicit model release/reload. Preserve actual consumed inputs, component
      identities, durable receipts, pixel hashes and native screenshots locally.
- [x] Fix and retest offline sharded component loading; backend implementation
      commit `3424faf` preserves normal online loading behavior.
- [x] Execute the DDPM whole-pipeline fallback with explicit float32, offload Off
      and Auto offload disabled; cold, unchanged and seed-edit runs passed.
- [ ] Resolve DDPM starter recipe compatibility: model CPU offload fails with a
      CPU/CUDA tensor mismatch, and resident bfloat16 fails during NumPy conversion.
      The explicit float32 resident recipe is a separate result, not a default pass.
- [x] Verify the DDPM Auto resource boundary: the planner reports that no Auto
      recipe is declared for this exact model/task pair and no graph is submitted.
- [ ] Qualify Windows in the separate Windows session; do not infer its results
      from this Linux host or these reduced-resolution image tests.

The four Modular families completed 23 execution/lifecycle checks. All used
512 × 512 images: SDXL at 20 steps (21 for the step edit), FLUX.2 Klein at four,
Qwen at 20, and Z-Image at nine. These explicit test settings do not change
creator defaults or promote public template qualification. DDPM uses its native
32 × 32 output at 50 steps; its pixels are visibly noisy, so this is execution
and reuse evidence, not image-quality qualification. Cold means first execution of the newly loaded graph
with weights already downloaded; it does not mean a cleared OS disk cache.

| Pipeline        | Cold execution | Unchanged repeat | Scope                                 |
| --------------- | -------------: | ---------------: | ------------------------------------- |
| SDXL            |         8.47 s |           0.38 s | Eight lifecycle cases                 |
| FLUX.2 Klein 4B |        11.31 s |           0.42 s | Five lifecycle cases                  |
| Qwen Image 2512 |        65.72 s |           0.76 s | Five cases after the offline fix      |
| Z-Image Turbo   |        18.96 s |           0.47 s | Five lifecycle cases                  |
| DDPM CIFAR-10   |         2.13 s |           0.46 s | Explicit float32 resident recipe only |

Unchanged runs reused cached outputs. Seed/step edits retained prompt encoding;
prompt edits retained loaded models. Explicit recomputation reproduced pixels
for the same inputs, and SDXL release/reload replaced the component identities.
Restored FLUX, Qwen and Z-Image previews matched the durable output byte hashes.
Generated pixels were inspected separately from execution status. This Radeon
8060S ROCm shared-memory machine has approximately 121 GiB system RAM; Qwen
retained approximately 57.9 GB in Torch allocations. This does not establish fit
on a discrete 16 GB GPU. No downloaded weights or user workflows were removed.

Qwen's original run failed before inference despite all required shards being
cached. The pinned Diffusers sharded loader queries Hub metadata unless
`local_files_only=True` is explicit. Pipeline and standalone Modular component
loading now pass that flag when `HF_HUB_OFFLINE` is enabled, while online calls
retain their prior kwargs. Focused regressions passed in both base and verified
optional runtimes; the complete base gate passed 3,149 tests with 510 skips and
9,314 subtests. Lint, dependency checks and runtime preflight also passed.
Original failures and successful retests remain in isolated local evidence.
The DDPM Expert sequence passed twice; an initial Auto harness expected a graph
submission instead of inspecting the planner denial. The captured denial is
preserved and is not counted as successful Auto inference. Final checks matched
all 71 client bundle files on disk and the 69 served HTML/asset files over HTTP;
all 90 inventoried cached-weight snapshots remain present without broken links.

## Sequencing and validation

M1 is independently deliverable. M2 and M3 establish the contracts required by M4.
Investigate M5 early, then integrate it with M3/M4 before claiming fast prototyping.
M6 needs the common discovery/contracts and a coordinated trust-policy change.
M7 follows stable execution contracts. M8 depends on replacement and migration
evidence, not simply completion of the new UI.

For every implementation slice:

1. Capture a focused failing test or document why faithful reproduction requires
   unavailable hardware/data. Preserve existing user state.
2. Implement the smallest complete behavior with regression coverage and docs.
3. Run the client `npm run check` and relevant Playwright tests for client changes.
   Run the backend gate in CONTRIBUTING for runtime changes and the verified
   optional-runtime entry point when those libraries are part of the claim.
4. Diagnose failures against the unchanged branch baseline. Never weaken unrelated
   checks, fake qualification, or regenerate semantic catalogs to hide a failure.
5. Build the paired frontend bundle from source when publishing integrated UI
   changes, preserve backend-owned fields/media, and verify served bytes/HTTP.
6. Review the complete staged diff, check for private/generated artifacts, and
   commit only the validated increment. Record exact checks and any baseline
   blockers. Do not label partial milestones complete.

Keep raw logs and screenshots outside tracked documentation. This plan records
scope and acceptance; user guides describe only delivered behavior. Use existing
theme/UI primitives, bounded parsers, and the single graph source of truth.

## End-to-end release scenarios

- Auto creator selects a task/model, runs, composes another block, and reopens it.
- Expert builds a supported graph from scratch using typed port suggestions.
- A seed-only rerun reuses models/conditioning while producing new sampling work.
- A compatible model switch preserves the graph and user text while invalidating
  incompatible model-derived results.
- Image/mask/reference inputs select the supported task and execute the expected
  upstream path; audio/video workflows expose their genuine additional stages.
- A developer adds a typed custom node, reloads it, and exports the executable
  workflow without adding frontend code.
- A pre-existing nested/legacy workflow remains editable and runnable without
  automatic destructive conversion.
- A constrained-memory run reports actual offload/release decisions and recovers
  from failure without stale caches or misleading readiness.
