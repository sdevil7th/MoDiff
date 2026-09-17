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
matching are implemented and validated. M2's dependency inventory is complete;
its behavior changes remain pending. M3–M8 have not started.
Catalog changes do not establish generic cross-family execution, mode
independence, cache correctness, or new custom-code support.

## Progress tracker

Check a task only after its implementation and applicable validation pass, then
record its commit under the milestone. **In progress** remains unchecked.
Keep this tracker and the detailed checklists below synchronized in both repos.
Model execution, hardware qualification, and UI tests are separate evidence.

| Milestone                                          | Status      | Remaining work                                                                         |
| -------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------- |
| M1 — Audience-appropriate discovery                | Complete    | None within M1; generic execution and model adaptation remain in M3/M4                 |
| M2 — Independent authoring/resource modes          | In progress | Inventory complete; implement separate controls and preserve saved execution settings  |
| M3 — Canonical operations and capability inventory | Not started | Backend contracts, complete pinned-upstream inventory, adapters and fallback           |
| M4 — Stage authoring and model/task switching      | Not started | Small starters, dynamic ports, atomic compatible changes, implementation inspection    |
| M5 — Reuse and selective recomputation             | Not started | Reproduction, cache identities, component lifetime, eviction diagnostics               |
| M6 — Custom-node developer experience              | Not started | Unified install/discovery, explicit code trust, reload/debug and invalidation          |
| M7 — Developer setup and service prototyping       | Not started | Tested uv/npm setup, platform guidance and reproducible API export                     |
| M8 — Consolidation and product qualification       | Not started | Legacy-compatible retirement, terminology, complete user journeys and runtime evidence |

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

The Auto/Expert control currently also changes resource planning policy.
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
- [ ] **M2.2** Introduce explicit authoring state and resource-policy state with a migration
      that preserves existing execution settings. Define old-client compatibility.
- [ ] **M2.3** Remove mode-driven model/schema rebuilds; preserve the active graph and cache.
- [ ] **M2.4** Expose automatic resource management in Expert and a clear override control.
- [ ] **M2.5** Update Auto/Expert design documentation and policy tests together.

Acceptance: mode changes preserve exported graph, effective values, active task,
and loaded-model identity; saved documents round-trip through refresh and Undo.
Unsupported automatic planning remains explicit, without forcing a UI mode change.

Inventory and migration decisions:

| Existing owner                                                                            | Responsibility and required change                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `useSettingsStore.studioViewMode`                                                         | Persisted global authoring preference; retain its key and existing `manual`-to-`expert` compatibility reader. Library, inspector, export and workspace visibility read it.                                                                               |
| `useStudioStore.form.resourceMode`                                                        | Per-workflow execution setting; retained in tab snapshots, saved workflows, imports, output restoration and packages. Keep the existing legacy resource-mode normalizer and backend values.                                                              |
| `TopBar`                                                                                  | The switch handler currently writes both states and synchronizes/rebuilds managed graphs. A reconciliation effect mirrors resource mode back into authoring and forces invalid graphs into Expert. Split these paths and keep invalid planning explicit. |
| `blockAutoAuthorityV2`                                                                    | Currently reads the authoring preference and can rewrite saved resource policy before execution. Make only the workflow's resource policy authoritative.                                                                                                 |
| `useStudioRunActions`, `runCoordinator`, `workflowAutoExecutionV2`, `blockRuntimeHintsV2` | Use resource policy for planning, selected-node/whole-workflow execution and submitted hints. Preserve these execution boundaries.                                                                                                                       |
| `GraphFixDialog`, `RunIssuesDialog`                                                       | Opening an inspector selects Expert presentation. It must not change resource policy or trigger graph rebuilding.                                                                                                                                        |

The two persisted fields already exist; avoid a second settings store or a new
workflow schema. Preserve both saved values, remove implicit cross-writes, and
show separate authoring and resource controls. Existing workflows keep their
resource settings; new workflows retain the existing automatic default. Older
clients can read the same workflow fields but retain their coupled UI behavior;
independent controls require the matching updated client bundle.

Implementation sequence for the next increment:

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
4. Reconcile Studio resource summaries and inspector actions with the independent
   controls. Retain existing saved keys, legacy normalization and tab restoration.
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

### M3 — Establish canonical operations and complete capability inventory

- [ ] **M3.1** Inventory ordinary actions, stage adapters, registered Blocks, and aliases.
      Classify exact aliases separately from semantically overlapping implementations.
- [ ] **M3.2** Extend backend-owned contracts with operation identity, stage capabilities,
      semantic ports, and execution/decomposition support levels.
- [ ] **M3.3** Map all pinned upstream pipelines and tasks into the coverage matrix. Keep
      unavailable optional dependencies distinct from missing adapter support.
- [ ] **M3.4** Bind existing generic stage nodes to supported implementations; consolidate
      discovery without changing persisted backend action names prematurely.
- [ ] **M3.5** Add standard-pipeline fallback operations for non-Modular paths. Preserve
      special stages for video, audio, 3D, and unusual conditioning.

Acceptance: parsers reject malformed capabilities; declarations agree with actual
upstream components and inputs; every inventory entry has an honest support state.
No template-specific receipt is required merely to author an ordinary valid graph.

### M4 — Deliver stage-first authoring and atomic model/task changes

- [ ] **M4.1** Create small generic starters from capabilities and support the equivalent
      graph constructed manually through typed connections.
- [ ] **M4.2** Adapt fields/ports from the selected loader contract. Preserve connected values
      and distinguish defaults, user overrides, and retained unsupported settings.
- [ ] **M4.3** Plan model changes before applying them. Commit compatible changes as one
      transaction; preserve incompatible custom nodes with actionable diagnostics.
- [ ] **M4.4** Extend tasks by connecting required conditioning. Use actual upstream task
      selection rules; do not assume img2img and instruction editing are identical.
- [ ] **M4.5** Expose implementation inspection separately from optional structural editing.
      Preserve existing deep Blocks without lossy automatic conversion.

Acceptance: native editing, drag-to-connect, Save/reopen, Undo/Redo, model switching,
and task changes pass across representative distinct families and modalities.
Real executions verify effective inputs, not just successful schema construction.

### M5 — Make repeated runs reuse components and unaffected results

- [ ] **M5.1** Reproduce repeated-run behavior and capture loader, encoder, denoiser, and
      decoder execution counts alongside component ownership and memory placement.
- [ ] **M5.2** Separate component lifetime from node-output cache lifetime. Cache identity
      includes consumed inputs, model/component revisions, relevant adapters, dtype,
      implementation revision, and custom-code revision.
- [ ] **M5.3** Exclude layout, collapse, selection, mode presentation, and progress metadata.
      Keep mutable generators per execution; never share stale mutable pipeline state.
- [ ] **M5.4** Prefer retained components when memory permits. Make pressure eviction and
      multi-owner scheduling explicit and preserve shared-owner references.
- [ ] **M5.5** Publish bounded reuse/reload/offload reasons and provide deliberate output
      recomputation and model-release controls.

Acceptance matrix: seed-only and step-only edits reuse unaffected conditioning;
prompt edits re-encode only affected stages; model/adapter/code changes invalidate
dependent data; mode/layout changes do not reload; cancellation and OOM recovery
do not reuse invalid state. Test shared components and real constrained-memory
runs separately. CPU offload is not equivalent to destruction/reload.

### M6 — Make custom node development a coherent product flow

- [ ] **M6.1** Document and unify discovery of existing local/Git Python modules and Hub
      Modular blocks. Reuse schema-derived UI and compatible Diffusers metadata.
- [ ] **M6.2** Design an explicit install/enable boundary for executable code. Expert mode
      alone is not authorization; browsing/import preview does not execute code.
- [ ] **M6.3** Revise SECURITY/AGENTS/contributor policy together with the executable path.
      Retain immutable Hub revisions, dependency visibility, input/path validation,
      and the local single-user boundary. Do not bypass the current checks ad hoc.
- [ ] **M6.4** Provide local developer reload, import diagnostics, exact-code identity, and
      targeted cache invalidation. Do not mislabel the process as a Python sandbox.
- [ ] **M6.5** Register installed custom nodes in normal search and typed suggestions; allow
      installed nodes to be used in Auto without granting additional permissions.

Acceptance: create a small local node and a pinned custom Modular block, render
their declared UI without frontend changes, connect and execute them, edit/reload,
and observe correct cache invalidation. Test rejection/cancellation paths and no
installation during discovery. Custom support must not create another executor.

### M7 — Add transparent developer setup and service prototyping

- [ ] **M7.1** Design a documented `uv` backend path and `npm` client path for the supported
      runtime profiles. Keep guided installation and existing environments working.
- [ ] **M7.2** Reconcile `uv` management, constraints, platform Torch sources, optional
      runtimes, and reproducibility before documenting commands as supported.
- [ ] **M7.3** Publish working Windows and Linux commands; test clean environments without
      replacing a user's accelerator packages or downloading inference weights.
- [ ] **M7.4** Reuse API graph export for a reproducible execution package with dependencies,
      model revisions, custom-node identities, and named service inputs/outputs.
- [ ] **M7.5** Treat standalone Python generation for arbitrary graphs as a separate feature;
      avoid claiming that a MoDiff API package is independent Diffusers Python.

Acceptance: clean setup and health checks on supported platforms; a saved workflow
executes through the documented API with equivalent resolved inputs. Credentials
and local machine paths do not enter portable packages.

### M8 — Retire redundant public surfaces and qualify the product

- [ ] **M8.1** Promote canonical operations after replacement coverage exists. Keep legacy
      identities loadable and offer explicit migration where semantics change.
- [ ] **M8.2** Rename cluster presentation to Blocks consistently, without bulk ID rewriting.
- [ ] **M8.3** Replace default implementation catalogs with contextual inspection. Retain
      templates as optional starters and user-owned Blocks as independent revisions.
- [ ] **M8.4** Validate the complete Auto and Expert journeys, accessibility, nested editing,
      persistence, API export, repeated runs, and compatibility with older workflows.
- [ ] **M8.5** Retire backend implementations only after proving replacement equivalence and
      migration behavior; similar labels are not sufficient evidence for deletion.

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
