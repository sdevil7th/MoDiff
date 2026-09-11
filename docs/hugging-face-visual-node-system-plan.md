# Hugging Face Visual Node System Plan

> **Architecture correction — 2026-09-01:** Registered Cluster Nodes and User
> Nodes must use one composite-node schema, canvas type, renderer, boundary,
> expansion, and persistence system. The dedicated Cluster renderer and
> destructive Cluster-to-User reconstruction recorded later in this historical
> tracker are superseded and are not considered complete product behavior. The
> authoritative replacement contract and active implementation status are in
> [Unified Composite Node Contract and Implementation Plan](unified-composite-node-implementation-plan-2026-09-01.md).

## Purpose

MoDiff is a local visual authoring and execution surface for reviewed Hugging
Face Diffusers, Modular Diffusers, and Transformers workflows. This plan adds a
first-party node library that exposes Cluster Nodes containing complete
pipelines, reusable Block Nodes, Component Nodes, and direct Transformers Nodes
without storing built-in definitions as User Nodes or introducing another graph
executor.

This document is an implementation tracker. Except in sections explicitly
marked historical or superseded, a checked item means the code, contract tests,
compatible client work, and stated proof level are complete. It does not claim
that every upstream-visible pipeline is executable in MoDiff.

## Upstream contracts reviewed for this work

The design follows the public contracts documented by Hugging Face:

- [Diffusers pipeline overview](https://huggingface.co/docs/diffusers/main/api/pipelines/overview): a standard `DiffusionPipeline` is an end-to-end inference object containing models, schedulers, and processors.
- [Modular Diffusers overview](https://huggingface.co/docs/diffusers/main/modular_diffusers/overview): blocks are reusable, composable, and intended to be mixed, matched, swapped, and shared.
- [Modular Diffusers quickstart](https://huggingface.co/docs/diffusers/main/modular_diffusers/quickstart): pipelines expose `blocks`, `available_workflows`, `get_workflow()`, and nested `sub_blocks`.
- [Modular pipeline blocks API](https://huggingface.co/docs/diffusers/main/api/modular_diffusers/pipeline_blocks): leaf, sequential, loop, conditional, and Auto block types retain different execution semantics and build pipelines through `init_pipeline()`.
- [Modular Diffusers state guide](https://huggingface.co/docs/diffusers/main/modular_diffusers/modular_diffusers_states): `PipelineState` and `BlockState` are the public communication mechanism between blocks.
- [ModularPipeline guide](https://huggingface.co/docs/diffusers/main/modular_diffusers/modular_pipeline): component loading is lazy, components may be shared, and a modified block definition must be used to construct a new pipeline.
- [ControlNet guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlnet) and [SDXL ControlNet API](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl): a prepared control image and separately loaded ControlNet component condition the base denoiser; conditioning scale and guidance start/end are runtime controls.
- [Transformers pipeline guide](https://huggingface.co/docs/transformers/main/pipeline_tutorial): direct Transformers inference is task-oriented and uses generic or task-specific pipeline contracts.

MoDiff remains pinned to the reviewed Diffusers revision declared by the
executable dependency contract. Documentation for upstream `main` is research
input; code is admitted only after it matches the pinned revision and the
generated no-weight compatibility snapshot.

## Product terminology and ownership

Custom blocks created by a user are **User Nodes**. First-party Hugging Face
definitions remain immutable library registrations. Every row below inserts
the same canonical composite `block` canvas type; the section controls
discovery, ownership, provenance, and permissions, not rendering or interface
inference. The intended library sections are:

| Section                       | Ownership                                                                                                   | Examples                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Diffusers Cluster Nodes       | Immutable, generated from reviewed Diffusers contracts; each contains a complete upstream pipeline/workflow | Qwen Image — Text to Image                          |
| Modular Diffusers Block Nodes | Immutable, generated from reviewed block contracts                                                          | Text Encoder, VAE Encoder, Denoise, Decode          |
| Diffusers Component Nodes     | Immutable MoDiff adapters over reviewed library components                                                  | Load Components, Scheduler, Guider, Adapter         |
| Transformers Cluster Nodes    | Immutable composites over an exact reviewed task graph, artifact revision, and optional runtime             | SmolLM2 — Text Generation, Whisper — Speech to Text |
| Transformers Nodes            | Immutable task-generic MoDiff adapters used inside Transformers Clusters or directly                        | Model Loader, Generate Text, Transcribe Audio       |
| Utilities                     | Built into MoDiff                                                                                           | Preview, media processing, export                   |
| User Nodes                    | User-owned, mutable custom block definitions                                                                | A modified Qwen Control cluster                     |

The existing `/studio/blocks` storage remains user-owned. Built-in definitions
must use a separate read-only contract and must never be written into that
directory.

## Reuse model: shared core plus narrow deltas

Pipeline data is model/workflow-specific only where upstream behavior differs.
There is significant overlap across Qwen, Flux, Wan, LTX, Cosmos, Helios, and
other families. The implementation must represent that overlap explicitly.

The contract is layered as follows:

1. **Generic task contract** — task identity and stable user concepts such as
   text-to-image, image-to-image, inpaint, text-to-video, image-to-video,
   prompt, source media, mask, seed, size, duration, and output media.
2. **Reusable block-role contract** — common roles such as text encoding, image
   encoding, VAE encoding, latent preparation, timestep preparation, denoising,
   decoding, and postprocessing.
3. **Pipeline/workflow definition** — the exact upstream blocks selected by
   `get_workflow()`, their hierarchy and order, field aliases, state flow,
   optional stages, and component requirements.
4. **Model/artifact execution profile** — immutable repository revision,
   component classes, dtype, quantization, device/offload limits, optional
   runtime requirements, qualification, and Auto eligibility.
5. **Composite block instance state** — prompt, seed, dimensions, steps, user
   media, exposed component choices, presentation overrides, and a copy-on-write
   effective graph when the workflow instance is structurally customized.

The frontend renderer, canvas type, public-boundary model, graph compiler,
persistence, expansion, state validation, and Auto canonicalization are
generic. Pipeline and model identity remain backend-declared data. Source kind
must never select a different renderer or re-infer inputs, outputs, or controls.

### Text-to-image overlap

Qwen Image and Flux text-to-image share the task, prompt-to-latent-to-image
shape, loader/component lifecycle, seed and sizing concepts, denoising progress,
preview/output handling, persistence, and most graph interactions. They differ
in exact text encoders, prompt fields, latent preparation, guidance semantics,
transformer inputs, scheduler compatibility, default values, and artifact/resource
profiles. Those differences are adapter/profile data, not separate frontend
node implementations.

The current pinned catalog groups 16 workflow definitions under the shared
`diffusers.task.text_to_image.v1` contract. Flux and Qwen have common semantic
paths such as text encoding, denoise input, denoiser, after-denoiser, and
decode, but their exact upstream leaf classes are respectively Flux- and
Qwen-specific. MoDiff therefore reuses the task UI, graph mechanics, and block
roles while retaining the exact class and hierarchy declared by each reviewed
workflow.

### Image-to-video overlap

Wan image-to-video overlaps with Cosmos, Helios, Hunyuan Video, LTX, and other
image-to-video workflows in the generic source-image + prompt -> temporal
conditioning -> latent denoising -> video decode/export flow. Reusable concepts
include source-media validation, prompt encoding, frame geometry, seed, temporal
length, denoising progress, video output, preview/export, resource planning, and
workflow persistence.

Pipeline-specific deltas include image-encoder requirements, one-image versus
first/last-frame conditioning, VAE latent layout, frame packing, temporal RoPE,
noise and timestep schedules, transformer signatures, decode tiling/chunking,
supported resolutions/frame counts, audio coupling, and exact component
artifacts. A shared image-to-video contract therefore does not imply that a Wan
block can be connected to an LTX or Cosmos block without compatibility proof.

The current pinned catalog groups nine workflow definitions under
`diffusers.task.image_to_video.v1`, including Wan, Wan 2.2, LTX, Cosmos,
Helios, and Hunyuan Video families. Wan and LTX share semantic paths for text
encoding, VAE encoding, timestep and latent preparation, the denoise loop, and
decode, while Cosmos uses a different vision packing/update structure. These
are variations of one reusable task flow, not nine unrelated frontend
implementations and not one interchangeable set of model-specific leaf blocks.

## User experience

First-party catalog rows are searchable and immutable. Every reviewed
definition is insertable as an exact structural Cluster, including
**Catalog only** entries. Structural insertion permits expansion, block
inspection/editing, composition, and persistence; it does not authorize Run.
An inserted Cluster can prepare a qualification run only when its publication
is **Graph qualified** and its exact runtime, artifact, dependency, graph, and
resource receipts pass. This volatile authority does not change the immutable
public executability claim.

### Complete Cluster Node

1. The user opens **Diffusers Cluster Nodes** or **Transformers Cluster Nodes**
   and selects a reviewed entry.
2. MoDiff inserts one collapsed Cluster Node with important inputs and output.
3. Run is available only for an exact graph-qualified execution admission;
   Auto applies only a qualified resource recipe. Catalog-only Clusters remain
   safely editable structural graphs.
4. Expanding a Diffusers Cluster reveals the selected top-level Modular blocks;
   expanding a Transformers Cluster reveals its exact MoDiff loader, task
   action, input, and preview nodes.
5. In a Modular Diffusers Cluster, expanding a compound or loop block reveals
   nested `sub_blocks` without changing sequential/loop/conditional semantics.
6. Editing a child parameter updates the collapsed projection and only that
   Cluster Node instance.
7. Collapse/expand is presentation state. Both views export the same semantic
   graph and produce the same cache identity.

### Workflow and model changes

- Changing a workflow uses the reviewed equivalent of `get_workflow()`, keeps
  compatible instance values, adds newly required inputs, and removes stale
  workflow-only state.
- Changing to a compatible artifact within the same pipeline family keeps the
  block structure but invalidates loaded components and the Auto plan.
- Changing pipeline family is an explicit replacement with a preview of values
  retained, reset, and removed.

### Composition

In Expert mode, a user may insert, remove, replace, or reorder compatible
blocks. The editor validates required and produced state keys, component
requirements, container semantics, and outputs. A structural edit updates the
same workflow instance copy-on-write; it never mutates the built-in definition,
changes the canvas renderer, or creates a reusable User Node implicitly.

Canonical Cluster Nodes and parameter overrides may remain Auto-managed.
Unreviewed structural edits switch to Expert until the resulting composition
matches a reviewed workflow or composition recipe.

## Technical architecture

### First-party library manifest

A bounded read-only manifest owns provider, library revision, pipeline class,
blocks class, workflow, generic task contract, hierarchy, inputs, outputs,
components, integration status, and a content hash. Stable definition identity
is derived from the provider, pinned revision, pipeline class, workflow, and
block path.

The existing generated `data/modular-workflow-contracts.json` remains the
upstream structural source. Runtime/profile registries remain authoritative for
execution and Auto eligibility. Visibility never implies runnability.

The resolved workflow and block snapshots intentionally remain stable. The
separate reviewed `data/modular-conditional-contracts.json` companion retains
the original unpruned `pipeline.blocks` trees for UI and composition consumers:
34 pipeline classes, 94 workflows, 632 static definitions, 1,051 placements,
and 73 conditional selectors with complete presence/absence truth tables and
workflow execution traces. This avoids silently changing existing Cluster
identities while preserving official inactive branches and skipped-block
semantics.

The current manifest contains 102 immutable Cluster definitions and 505
deduplicated block contracts. The 94 pinned Modular Diffusers workflows account
for 483 exact contracts: 393 visible child placements plus 90 aggregate root
contracts that retain selected-workflow `PipelineState` initialization
metadata. Eight reviewed Transformers composites contribute 22 task-graph
blocks. These Transformers children are MoDiff's exact generic task nodes;
Transformers does not expose the Modular Diffusers `blocks`/`sub_blocks` API.
Schema v5 additionally publishes 321 content-addressed Block Role adapters for
all 48 contract-only workflows and the seven equivalent-standard definitions.
Nine custom loop-state adapters supplement the 12 Helios, MiniMax Music, and
Wan Animate workflows. All remain explicitly structural; they do not imply an
artifact, resource recipe, or executable Studio route.

### Graph representation

The visible `useFlowStore` graph remains the only executable graph. A generic
compiler expands a reviewed pipeline/workflow definition into existing MoDiff
nodes and edges. It adds explicit component bindings and state connections;
there is no hidden browser runtime and no second pipeline executor.

`PipelineState`-like data must be represented by an explicit, typed graph value
whose declared keys are inspectable and validated. Loop blocks remain control
flow containers. Runtime tensors and component handles are process-local and
are regenerated on the next execution after a refresh.

### Persistence

A saved first-party instance stores the exact definition reference and embedded
snapshot, library revision, content hash, instance parameter values, explicit
public boundary, selected static execution admission and Studio-spec receipt,
copy-on-write effective graph when structurally changed, and presentation
state. Loading checks all receipts before applying values. Unsupported drift
produces a migration-required state and never silently resets parameters or
re-derives ports. The canonical persistence target is `BlockInstanceV2` in the
[unified contract](unified-composite-node-implementation-plan-2026-09-01.md).

### Auto

Before graph fingerprinting, readiness, managed-control synchronization, or
resource planning, first-party Cluster Node containers are normalized to stable
semantic child identities. Expansion, collapse, and container layout are
excluded from the execution fingerprint. Structural changes are included.

## Implementation tracker

### P0 — Research and durable contract

- [x] Review repository graph, security, persistence, and Auto constraints.
- [x] Review the official Diffusers/Modular Diffusers/Transformers documents listed above.
- [x] Confirm the pinned snapshot discovers 34 Modular pipeline classes and 94 upstream workflows without loading weights.
- [x] Define the shared-task versus pipeline-delta architecture in this document.
- [x] Add the first-party library manifest and strict validation.
- [x] Add tests proving different pipeline families reference shared generic task contracts.

### P1 — Read-only backend catalog

- [x] Publish the reviewed first-party manifest through a bounded read-only API.
- [x] Keep contract-only and equivalent-standard routes distinguishable from reviewed MoDiff contracts.
- [x] Add response-size, schema, revision, hash, and no-model-import tests.
- [x] Document the API contract.

### P2 — Client catalog and ownership separation

- [x] Add a strict client parser/store for the first-party manifest.
- [x] Add Diffusers Cluster Nodes, Modular Diffusers Block Nodes, Diffusers Component Nodes, Transformers Cluster Nodes, and Transformers Nodes sections.
- [x] Keep custom blocks named User Nodes.
- [x] Keep first-party definitions immutable and separate from `/studio/blocks`.
- [x] Add search, readiness, accessibility, and mocked client contract tests.
- [x] Add the full mocked-browser catalog interaction test to the existing
      Studio harness.

### P3 — Legacy Cluster composite (historical; V2 remediation open)

The checked rows in this section record the implementation that existed before
the 2026-09-01 architecture correction. They are not proof of the unified
composite acceptance contract. In particular, a separate root type/renderer
and destructive reconstruction into a User Node are now rejected designs.

- [x] Define a strict persisted Cluster Node instance contract with exact
      definition revision/hash, isolated parameter overrides, presentation-only
      expansion state, and stable semantic child identities derived from block
      paths.
- [x] Add a generic nested hierarchy projection, collapsed input parameter
      projection, JSON round-trip proof, and a semantic snapshot that excludes
      expand/collapse state.
- [x] Extract provider-neutral child identity, ownership, and descendant
      mechanics while preserving existing User Node child IDs.
- [x] Materialize the stable semantic child identities as recursively nested,
      definition-backed graph containers with deterministic layout.
- [ ] Remove the legacy dedicated Cluster Node renderer. Route registered and
      user-owned definitions through the canonical `block` renderer and common
      V2 store actions while retaining catalog ownership and authority as
      metadata/capabilities.
- [x] Rebuild derived child containers from the exact reviewed definition after
      persistence, retain per-instance parameter overrides, and fail closed on a
      revision/content-hash mismatch.
- [x] Prove discovery-only Cluster containers are excluded from execution and
      that expanding/collapsing them cannot change the surrounding executable
      graph export.
- [x] Persist an exact per-instance execution-mode/spec receipt and only the
      backend-declared tunable binding sources; reject artifact, pipeline-class,
      and revision overrides, migrate schema-v1 instances, and invalidate stale
      derived execution children after a mode or execution-parameter change.
- [x] Annotate every materialized bound child field with its backend binding
      source and persistence policy. Expanded child edits now update the owning
      root instance (and collapsed projection) for upstream inputs or execution
      overrides, while reviewed artifact/model/revision fields are disabled and
      rejected as sealed.
- [x] Give exact upstream leaf Block Nodes persisted parameter disclosures.
      Only workflow inputs declared by both the pinned block and owning Cluster are
      editable; intermediate tensor/state fields are not presented as ordinary
      values. A block edit updates the isolated root instance and collapsed
      projection.
- [x] Define a stable execution fingerprint over the exact definition,
      admission/spec receipt, persisted parameters, sealed artifact values, and
      auxiliary dependencies; presentation state and layout are excluded, while
      mode or parameter changes alter the fingerprint.
- [x] Project execution-child progress and preview updates onto the collapsed
      Cluster root, while expanded Clusters keep updates on the exact child node;
      derived preview fields are removed when execution authority is invalidated or
      rebuilt after refresh.
- [x] Prove collapsed/expanded equivalence again after P4 compiles Cluster
      children into executable nodes and edges.

### P4 — Workflow-to-graph compiler

- [x] Enrich the pinned no-weight snapshot with each block's declared inputs,
      intermediate outputs, required inputs, component/config requirements, and
      ordered container membership from the official block APIs.
- [x] Capture the selected workflow's aggregate root block contract so
      `PipelineState` initialization retains named and variadic `kwargs_type`
      metadata before any child block runs.
- [x] Add a fail-closed structural compiler plan with exact block identities,
      state sequencing, container entry/exit, loop feedback, component/config
      bindings, unresolved adapter identities, and no execution claim.
- [x] Parse and structurally compile all 94 pinned workflows without importing
      model libraries or downloading weights: 82 are statically state-closed and
      12 truthfully report upstream container-local state that requires an exact
      runtime adapter.
- [x] Join 33 existing reviewed generic action/state-edge contracts to their 31
      exact pipeline/workflow definitions as backend-declared discovery metadata;
      keep every definition at `executionClaim: discovery_only` until
      materialization and runtime qualification pass.
- [ ] Compile reviewed pipeline/workflow definitions into the existing graph.
- [ ] Represent component bindings and declared state flow explicitly.
- [ ] Preserve sequential, loop, conditional, and Auto block semantics.
- [ ] Reject unknown block classes, paths, revisions, or incompatible state.
- [ ] Cover pinned upstream parity without model downloads.

The first materialization phase now exists for statically admitted definitions.
It consumes the sealed Studio execution-spec receipt and the live generic node
registry, creates stable per-Cluster node and edge identities, applies only
backend-declared instance-input and execution-parameter bindings, and reports
missing dynamic fields or binding sources. The client has no handwritten
prompt/media alias table. Selection and tunable values survive JSON
save/refresh independently per Cluster instance. Materialized nodes remain
disabled and excluded from graph export until a volatile runtime/resource
authority matches them. Admission schema v3 also declares the
ordered loader/input field actions and value sources required to obtain each
dynamic Modular node schema. A provider-neutral runner now attaches the
isolated skeleton to the visible graph, applies those backend-declared actions,
waits for the WebSocket node-definition updates, and rebinds the exact fields
and typed edges with stable IDs. It still returns `executable: false`;
Admission schema v3 now supplies every sealed literal/artifact binding and its
exact auxiliary model dependencies; planner values are accepted only for
declared execution-parameter sources and cannot override prompts, artifacts,
pipeline classes, or revisions. When reconciliation is complete it can issue the same bounded
`studioExecutionSpec` node-ID receipt that the existing backend graph admission
already validates. Completing this P4 item requires component/state proof plus
resource/runtime admission.

The 12 custom-container structural closures are now represented by nine exact
loop-state adapters: six Helios variants, MiniMax Music 3, Wan Animate 2, and
Wan Animate 2 Distilled. They retain the upstream iteration source,
initializers, published `latent_chunks` or `segment_frames` state, and progress
semantics from each subclass `__call__`. This closes static planning without
inventing a generic edge. Runtime Studio/profile/artifact admission for those
12 workflows remains open.

All 94 reviewed Diffusers definitions are now structurally insertable from the
live frontend independently of execution admission. A visible browser proof
inserted **Helios Pyramid Distilled — Text To Video**, expanded its pinned
block hierarchy, expanded the exact `HeliosTextEncoderStep` parameter
disclosure, edited the prompt inside that block, saved, refreshed, and verified
the same prompt and disclosure state before collapsing back to the identical
root projection. Evidence is preserved under
`data/qualification/local-review/hugging-face-clusters/`.

#### Diffusers readiness inventory — 2026-08-26

The pinned manifest still contains 94 exact Modular Diffusers workflow
definitions. They are intentionally split by proof level rather than treated
as 94 interchangeable executions:

| Proof level                           | Definitions | Current meaning                                                                                                  |
| ------------------------------------- | ----------: | ---------------------------------------------------------------------------------------------------------------- |
| Reviewed MoDiff action/state contract |          39 | Exact generic action roles and state edges exist for the pinned workflow.                                        |
| Contract only                         |          47 | Exact upstream block hierarchy exists, but a MoDiff execution adapter/profile has not yet been admitted.         |
| Equivalent standard route             |           7 | MoDiff has an exact workflow-scoped standard Diffusers route; that does not claim split-block Modular execution. |
| Official whole-workflow route         |           1 | MiniMax Music 3 uses its package-owned Modular workflow through a sealed top-level block execution contract.      |

Forty-eight execution admissions across 47 definitions now pass the static
gate: the original 40 reviewed-action joins, seven exact equivalent-standard
routes, and the sealed MiniMax whole-workflow route. Every admission remains graph-qualified and
insertable but publishes `executable: false`, `autoEligible: false`, and
`liveProof: false` pending exact runtime/resource authority, visible-frontend
evidence, and manual approval.

No definition in the reviewed-action bucket is waiting for a static graph
adapter. All 47 contract-only definitions and all seven equivalent-standard
definitions now have exact Block Role coverage, and the remaining custom-container
workflows additionally have their nine exact loop-state supplements. They
remain at their existing readiness levels because structural coverage is not a
split-block Studio execution route. The seven equivalent-standard definitions still need
separate split-block Modular execution admissions; a working standard Diffusers route is
not treated as proof of that equivalence.

### P5 — Initial executable Cluster Node publication

- [x] Validate all 40 admitted exact model-type/mode joins to declarative
      Studio execution specs: 11 Qwen joins, six FLUX-family joins, three Wan
      joins, two Z-Image joins, and all 18 pinned SDXL Modular workflows. Validate
      role/state compatibility because identifier equality alone is not proof.
- [x] Publish expandable composites only for workflows with an existing reviewed MoDiff execution contract.
- [x] Keep contract-only entries visible only at their truthful proof level.
- [x] Add live parity evidence separately for each currently admitted model/workflow/artifact/hardware recipe.

The static admission audit now accepts all 39 definitions with reviewed MoDiff
actions. This includes Qwen Image text-to-image and image-conditioned routes,
Qwen Image Edit inpainting, FLUX Kontext, FLUX.2 Klein, Wan text and single-
image video, and both Z-Image workflows in addition to the earlier joins. The
Wan FLF graph explicitly joins its
user-facing `image_to_video` Studio mode to the upstream `flf2v` state-flow
adapter and FLF2V artifact, plus all 18 pinned SDXL workflows: base,
ControlNet, ControlNet Union, IP-Adapter, and the ordinary/Union ControlNet plus
IP-Adapter compositions for text-to-image, image-to-image, and inpainting.
The inpainting graphs explicitly carry the source image, mask, encoded mask,
masked-image latents, image latents, and route state. Public executable and
Auto-eligible publication flags remain false until their exact live evidence is
manually approved.

The current Z-Image Studio spec remains rejected because it uses the standard
`ZImagePipeline` loader/generator graph,
not `ZImageModularPipeline` actions, so it cannot materialize the expandable
Modular workflow without a separate reviewed Modular execution spec.

All 30 admitted routes currently declare `live_proof: false`; the additional
Z-Image Modular candidate is rejected against its standard-pipeline Studio
spec. Graph-qualified
definitions may be inserted, and an exact volatile authority may enable a
qualification run, but the immutable catalog
must not claim public executability or Auto eligibility until the evidence is
reviewed and the admission receipt is promoted.

### P6 — Auto and managed graph integration

- [ ] Canonicalize composites before binding divergence and fingerprints.
- [x] Keep Auto active across collapse/expand and managed parameter overrides.
- [ ] Replan resources after compatible artifact, workflow, or structural changes.
- [ ] Switch unreviewed structural User Node forks to Expert with an actionable reason.

The current static resource join finds exact declared Auto pairs for the
admitted Qwen and SDXL contracts. Wan FLF remains Expert-only until a measured
recipe is reviewed. The admitted execution profiles that declare an optional
overlay require the reviewed Transformers/PEFT runtime on this host;
the immutable node-library contract records no volatile installed/active
state, and catalog browsing must not install or activate that overlay.

### P7 — Composition editor

- [ ] Publish compatible block insertion/replacement points.
- [ ] Validate required/produced state, components, outputs, and container kinds.
- [ ] Rebuild a modified upstream block definition through `init_pipeline()`.
- [ ] Save modifications as User Nodes.

### P8 — Transformers composites

- [x] Present existing task-generic Transformers nodes under the correct provider.
- [x] Add composite loader/action/preview definitions for the seven initially
      reviewed text, vision-language, any-to-any, and speech routes.
- [x] Add a separately sealed Wav2Vec2 CTC speech-to-text Cluster using the
      official `AutoProcessor` and `AutoModelForCTC` boundary.
- [x] Preserve explicit optional-runtime consent, exact target-profile selection,
      digest verification, and activation/rollback through Setup.
- [x] Do not install Transformers during discovery, browsing, insertion, or Auto planning.

### P9 — Migration and release gates

- [ ] Preserve existing user-block and workflow formats through explicit migrations.
- [ ] Run backend, client, browser, graph round-trip, and security-boundary gates.
- [ ] Record static, mocked, artifact, runtime, and live-output evidence separately.
- [ ] Update user-facing documentation only for behavior that is actually shipped.

### Promotion and rollout plan

The remaining work is delivered through one fail-closed promotion path rather
than by changing catalog flags by hand. Each promoted Cluster Node must carry
the exact definition, graph-adapter, Studio execution-spec, artifact revision,
optional-runtime, resource-recipe, and live-evidence receipts that authorized
its publication. Missing or stale receipts return the entry to **Catalog only**
without changing saved instance parameters.

#### R1 — Qwen and Wan publication foundation

- [x] Add a backend-owned publication receipt that distinguishes discoverable,
      insertable, and executable admissions without consulting volatile installed
      state while browsing.
- [x] Make the client catalog derive its readiness and insertion behavior only
      from that receipt; the frontend must not contain a Qwen/Wan allowlist.
- [x] Insert a selected definition as one Cluster root with an isolated
      per-instance execution selection and deterministic child identities.
- [x] Finalize dynamic Modular fields through the existing backend actions,
      verify the exact Studio-spec node receipt, then enable execution children only
      after the required runtime and resource checks pass.
- [ ] Promote Qwen Image Edit first, followed by Qwen Image Edit Plus, Qwen
      Layered, Qwen Control, and Wan FLF. Each promotion remains independently
      reversible if its artifact, runtime, or evidence receipt changes.

The schema-v4 static admission now includes a nested publication receipt.
Thirty reviewed admissions publish as **Graph qualified** and insertable; the rejected
Z-Image join remains **Catalog only**. Every publication still has
`executable: false` and `autoEligible: false`. Insertion is generic, selects a
unique admission when possible, and projects visible Studio values only through
the backend-declared input and execution-parameter bindings. A prepared Cluster
materializes and finalizes its isolated child graph. Its children are enabled
only by a volatile authority joining the live runtime fingerprint, exact
optional runtime, installed artifact revision, backend Auto candidate,
dependencies, and Studio execution spec. This permits qualification runs
without mutating the immutable publication claim.

The visible frontend now prepares and executes all five admitted Qwen routes
independently. Qwen Image Edit, Edit Plus single- and multi-image, Layered, and
Control each have a completed Cluster-route output. The Layered route preserved
all four alpha outputs, and Control joined its exact pinned ControlNet
dependency. The preparation/live-execution matrix is stored beside the Qwen
Image Edit run in
`test-review/frontend-pipeline-e2e-2026-08-24/qwen-cluster-qualification/`.
The Layered pass found that Studio's fixed `add alpha` image-loader mode had
been classified as an execution parameter; admission schema v4 now seals both
alpha-operation literals and the strict client parser accepts them only as
sealed sources.

#### R2 — Semantic and persistence equivalence

- [x] Canonicalize a Cluster to the same executable graph irrespective of its
      collapsed or expanded presentation state.
- [x] Prove identical executable node/edge payloads and execution fingerprints
      for collapsed and expanded views.
- [x] Save and reload the workflow, rebuild derived children from the pinned
      definition, re-run admission/finalization, and prove the executable payload
      and per-instance parameters are unchanged.
- [x] Prove that editing one Cluster, changing its model/workflow selection, or
      invalidating its runtime authority cannot modify another Cluster instance.

The materializer now emits a canonical execution snapshot that excludes
presentation, layout, selection, progress, preview, and pre-admission disabled
state. Contract tests prove this snapshot is identical for collapsed and
expanded materializations and after JSON round-trip/rematerialization. Enabled
contract tests now also prove byte-equivalent API node/path exports for
collapsed and expanded views and after persisted-root rematerialization plus
reauthorization. The visible-frontend Qwen qualification run proves the
collapsed live path; a second full model run is not required for the
presentation-only parity already established at the exported API boundary.

SDXL Modular now adds a full live equivalence proof. A collapsed 12-step run
was saved, reopened through **My workflows** in a fresh frontend session, and
expanded into ten upstream Modular blocks. The reopened root retained its
prompt and step override. After exact Expert requalification, the expanded run
used the same execution fingerprint and run-input hash as the collapsed run and
produced a byte-identical WebP. See
`test-review/frontend-pipeline-e2e-2026-08-24/sdxl-cluster-qualification/receipt.json`.

The independently admitted SDXL `image2image` workflow now has the same proof.
The visible frontend imported its source image, inserted the seven-node Cluster,
and ran an eight-step recipe with strength `0.65`. The saved workflow was then
reopened in a fresh browser session and expanded into 11 upstream Modular block
containers. Source image, prompt, steps, dtype, and strength all survived. The
collapsed and expanded exports were identical after removing only volatile
session/check timestamps, both runs used execution fingerprint
`hfcluster_3d5f730b` and run-input hash `run_352e4da3`, and both produced the
same WebP hash. See
`test-review/frontend-pipeline-e2e-2026-08-24/sdxl-image2image-cluster-qualification/receipt.json`.

This run also closed an overlapping-parameter ambiguity. When a Studio binding
source is itself an exact upstream workflow input (`dtype` and `strength` for
this SDXL workflow), admission now persists it in the Cluster instance rather
than in a second execution-only override. The collapsed field, expanded child,
saved snapshot, and executable graph therefore read one value.

The independently admitted SDXL `inpainting` workflow extends that proof to a
second visible media input and mask-specific state. The frontend imported the
source image and mask through **Assets**, persisted `image`, `mask_image`,
`strength`, `dtype`, prompt, dimensions, and steps on the Cluster instance,
and ran its eight-node API graph as task `Dgwx0JmMEHTT`. The saved workflow was
then opened from **My workflows** in a fresh browser, expanded, requalified,
and run as task `WQgtLGPh8TXP`. After removing only volatile `checkedAt` and
`sid` fields, both authorized API graphs have SHA-256
`04e2401ae3d947d469bd0aa74d98244fc12cab6ec237ebebd07b667a749e13d8`;
both runs produced the same 126,790-byte WebP with SHA-256
`578d5dfbc7c11ee32054e2c72d093067273bb9f6c889195a4379c45c50f00810`.
See
`test-review/frontend-pipeline-e2e-2026-08-24/sdxl-inpainting-cluster-qualification/receipt.json`.

The independently admitted SDXL `controlnet_text2image` workflow adds a pinned
standalone component without borrowing Qwen's different routed-ControlNet
shape. Its eight-node graph loads the ordinary Canny ControlNet at revision
`eb115a19a10d14909256db740ed109532ab1483c` with the sealed `fp16` weight
variant, passes the prepared control image directly into the upstream SDXL
ControlNet block, and passes only its `controlnet_bundle` to Denoise. There is
no invented base-VAE-to-ControlNet or ControlNet route-state edge. The visible
frontend saved workflow `DYyGqzuYGEmNLNTPGwoRd`, reopened it in a fresh
browser, expanded 11 upstream block containers, and reran it. Both runs used
execution fingerprint `hfcluster_dbf1ea60`; after removing only volatile
`checkedAt` and `sid` fields the API graphs have SHA-256
`bd27db2b205d8d7b6bc18917f2cd2bc2570242aed3d79aae2bd72d83f4dfc88b`,
and both runs produced the same 84,534-byte WebP with SHA-256
`6ce69421c7d167cd1acd2810434148403fd40dfdf7e028ce63d425d3088746b2`.
See
`test-review/frontend-pipeline-e2e-2026-08-24/sdxl-controlnet-cluster-qualification/receipt.json`.

The independently admitted SDXL `controlnet_image2image` workflow composes
that ordinary ControlNet route with the reviewed source-image VAE encoding
route. Its ten-node API graph sends source-image latents and route state to
Denoise and sends only the separately loaded ControlNet's `controlnet_bundle`
to Denoise. The visible frontend imported distinct source and control images,
saved workflow `ZC3xJjRzAbS0DXi12AuTB`, reopened it in a fresh browser,
expanded 12 upstream block containers, and reran it. Both runs used execution
fingerprint `hfcluster_d4e29c61` and run-input hash `run_00529272`; after
removing only volatile `checkedAt` and `sid` fields the API graphs have SHA-256
`0408406437a87c41a84170c2837b1f12f80d19fb8f6bee6bc3ce43cc87696b31`,
and both runs produced the same 85,516-byte WebP with SHA-256
`1dc341e295498735693d8c5b4991489bea7da7882610b0407ea251404e439a51`.
See
`test-review/frontend-pipeline-e2e-2026-08-24/sdxl-controlnet-image2image-cluster-qualification/receipt.json`.

The independently admitted SDXL `controlnet_inpainting` workflow completes the
ordinary ControlNet trio. Its eleven-node API graph joins the reviewed source
image, mask, masked-image latents, image latents, and route state with the
standalone ControlNet bundle only at Denoise. The visible frontend imported all
three distinct media inputs, saved workflow `W7DH2wiuUEdtoG82i2JCI`, reopened
it in a fresh browser, expanded 12 upstream block containers, and reran it.
Both runs used execution fingerprint `hfcluster_ab00644e` and run-input hash
`run_bfa5ef83`; their normalized API graphs have SHA-256
`5a23f0d57fc6a486c810d5e92d02ab884ae72f2d8f4bac7d7ca67371f9a4d82b`,
and both produced the same 217,654-byte WebP with SHA-256
`69b2db4e4546120fd407c27f5d0549d6d6520e06c007bf1b66daf6b7b8e9b77e`.
See
`test-review/frontend-pipeline-e2e-2026-08-24/sdxl-controlnet-inpainting-cluster-qualification/receipt.json`.

A visible browser save/refresh proof now covers the remaining lifecycle
boundary. Before refresh the root owned one volatile authority and seven
derived execution nodes. After refresh, its exact prompt, negative prompt,
source image, steps, seed, and guidance remained, while authority and derived
execution-node counts were both zero. Re-preparation restored one authority and
the same seven-node graph; the exported API nodes and paths were exactly equal
to the pre-refresh export. The receipt and screenshots are under
`test-review/frontend-pipeline-e2e-2026-08-24/qwen-cluster-qualification/`.
The store contract also exercises two simultaneous Cluster roots: editing the
first changes only its root values and clears only its authority, while the
second root's prompt and authority remain byte-for-byte intact.

#### R3 — Auto and resource integration

- [x] Request Auto plans from the canonical Cluster execution identity and bind
      only backend-declared execution parameter sources.
- [x] Preserve Auto across collapse/expand and ordinary parameter edits; replan
      after artifact, workflow, or structural changes.
- [x] Keep Wan FLF Expert-only until it has a reviewed Auto resource pair.
- [x] Keep Run disabled with an actionable Setup/install state when the exact
      optional runtime, model artifact, or auxiliary dependency is unavailable.

Wan FLF now follows the explicit Expert branch without creating a fake Auto
candidate. The backend issues a volatile qualification receipt only after the
exact cached commit and shard integrity, Modular execution profile, active
optional runtime, device/offload pairing, and dependencies pass. The client
strictly parses and cross-checks that receipt against the admission before it
enables the nine-node graph. A visible frontend preparation proof and screenshot
are stored in
`test-review/frontend-pipeline-e2e-2026-08-24/wan-cluster-qualification/`.
The same visible frontend then submitted task `6gfz2uw4hqt-` through the
collapsed Cluster route. All nine derived nodes completed in 73.6 seconds and
produced a hash-sealed five-frame H.264 MP4 plus first/middle/last review
frames in that folder. The immutable catalog remains non-executable and
non-Auto-eligible pending manual publication review; the runtime proof does not
weaken that boundary.

SDXL Modular follows the same fail-closed publication boundary. Qualification
runs may use an exact backend-declared Auto pair while immutable public
`executable`, `autoEligible`, and `liveProof` flags remain false. Its first live
run exposed an artifact/loader mismatch: Studio had installed the reviewed
`*.fp16.safetensors` selection, while `ModelsLoader` requested default weight
filenames. The loader now applies a repository-scoped reviewed `fp16` variant
before component-reuse matching and loading. This deliberately does not infer
variants from dtype for Qwen, Wan, or arbitrary repositories. A second live
diagnostic found and fixed the missing loader-VAE-to-denoise edge required by
the upstream SDXL denoise block. The subsequent inpainting promotion adds the
upstream source image, mask image, encoded mask, masked-image latents, image
latents, and route state without treating any of them as an implicit browser
side channel. The ordinary ControlNet promotions additionally seal their exact
standalone component revision and `fp16` filename variant while retaining the
upstream SDXL component/state boundaries. The image-to-image variant joins
source-image latents and route state at Denoise without inventing
VAE-to-ControlNet or ControlNet route-state edges. The inpainting variant adds
the exact mask and masked-image-latent edges at Denoise. All six boundaries
have regression coverage.

#### R4 — Remaining pinned Modular Diffusers definitions

- [x] Admit every definition that already has a reviewed generic Modular
      action/state contract: 40 workflow/mode joins across all 39 definitions.
- [x] Promote only the LTX-2 `text2video` and `image2video` workflows through
      their exact standard condition-pipeline overlap; keep `condition` and
      `in_context` contract-only.
- [x] Promote the remaining statically closed workflows by reusable task and
      block-role contracts, retaining model/workflow differences as backend data.
- [x] Add exact structural container-state adapters for the 12 custom-container
      workflows without making an execution claim.
- [ ] Add exact runtime action/profile/admission routes for those 12 workflows
      before considering them executable.
- [ ] Keep contract-only and equivalent-standard definitions at their truthful
      readiness levels until a reviewed Modular execution route exists.

All 18 SDXL R4 definitions are now graph-qualified and insertable through
reviewed task/state-flow adapters. Visible-frontend live evidence exists for the
base and ordinary ControlNet trios, ControlNet Union text-to-image, standalone
IP-Adapter text-to-image, and the maximal IP-Adapter + ControlNet Union
inpainting composition. The other admitted SDXL definitions still require
definition-specific live execution evidence. Every immutable publication claim
remains non-public-executable and non-Auto-eligible pending manual approval.

All 38 ordinary contract-only workflows now have complete backend-declared
Block Role coverage: 209 deduplicated exact block definitions map to the shared
workflow, text/VAE/image encoding, prompt transform, duration, condition,
reference, pre-encode, denoise, decode, and post-decode roles. The client
attaches only these sealed role receipts and reports whether a structural plan
is fully role-adapted; it contains no model-family role inference. These
workflows intentionally remain `contract_only` and non-executable until their
artifact, component, Studio execution, resource, and live-evidence gates are
individually admitted.

The same schema completes deduplicated role adapters for the custom-container
and equivalent-standard workflows, bringing the non-reviewed workflow role
catalog to 321. Nine exact container-state adapters make the 12
custom workflows statically closed while preserving their distinct loop
semantics. This completes structural adaptation only; runtime execution
adapters and admissions remain pending.

#### R5 — Transformers composites

- [x] Define provider-neutral composite receipts for the existing generic text
      generation, image-to-text, any-to-any, and speech nodes.
- [x] Publish graph-qualified loader/action/preview Cluster Nodes only for exact
      target optional-runtime profiles and immutable artifact revisions.
- [x] Preserve explicit installation consent and never install Transformers as
      a side effect of catalog discovery, insertion, or Auto planning.

The catalog now contains eight independently immutable Transformers
definitions: SmolLM2
135M text generation; SmolVLM 256M image-to-text; Janus Pro 1B text generation,
image-to-text, and text-to-image; and Whisper Tiny speech transcription and
translation; plus Wav2Vec2 Base 960h CTC speech transcription. They reuse five
task contracts and 22 deduplicated composite
blocks. This is coverage of the generic task graphs MoDiff currently reviews,
not a claim that all Transformers models or pipeline tasks are supported.

For the current delivery phase, Transformers work is narrowed to automatic
speech recognition and begins only after the Diffusers completion gates. The
app-managed Transformers runtime is pinned to commit
`96fe6dce36cc929a5ffd3e34296554c4cb6b669e`. Its official
`AutomaticSpeechRecognitionPipeline` has five distinct execution branches, so
the implementation order is architectural rather than a separate hard-coded
node for every checkpoint:

1. [x] the Whisper sequence-to-sequence branch through the official
       `AutoModelForSpeechSeq2Seq`/`AutomaticSpeechRecognitionPipeline` contract,
       with transcription first and speech-to-English translation second;
2. [x] the plain CTC branch through the official `AutoModelForCTC`
       contract, beginning with an immutable reviewed Wav2Vec2 artifact and then
       admitting compatible HuBERT, WavLM, MMS/XLS-R, or other CTC checkpoints only
       when their exact config architecture, processor, artifact revision, runtime,
       and frontend evidence pass the same gates;
3. [ ] the generic speech sequence-to-sequence branch for models in
       `MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES`, with no Whisper-only language or
       timestamp controls;
4. [ ] the transducer/TDT branch shared by the pinned Parakeet TDT/RNNT and
       Nemotron ASR model types, using their `generate()` and non-grouping decode
       contract; and
5. [ ] CTC with a language-model decoder as a separate optional dependency and
       artifact contract, because it requires `pyctcdecode`, decoder assets, and
       permits word timestamps only.

The branch list above is derived from the
[exact pinned Transformers ASR source](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/pipelines/automatic_speech_recognition.py),
not from a manually maintained list of every audio model name. Models such as
Speech2Text, Moonshine, Parakeet, Nemotron ASR, HuBERT, WavLM, MMS, or XLS-R are
admitted through the matching reviewed branch only when their immutable model
and processor contracts pass qualification.

The shared product surface is **Transformers Speech-to-Text Cluster Nodes**;
family-specific loader behavior stays inside the expanded Cluster. Translation
is shown only for a model family that explicitly supports it, so a CTC model
cannot accidentally receive Whisper-only `task` or `language` generation
arguments. SmolLM2, SmolVLM, Janus, and all non-ASR Transformers families are
deferred. Every admitted speech Cluster must receive the same
collapsed/expanded, save/refresh, runtime/artifact admission, Auto/Expert,
visible-frontend output, and preserved-review-asset proof before any public
flag changes.

The SmolLM2 definition has visible-frontend proof for insertion, typed numeric
parameter editing, exact runtime/artifact qualification, collapsed execution,
save plus fresh-page restoration, expanded execution, API-graph equivalence,
and deterministic output equivalence. Public `executable`, `autoEligible`, and
`liveProof` flags remain false until manual review. Both Whisper task variants
and the Wav2Vec2 CTC definition now also have visible-frontend output evidence.
The four non-speech definitions are deferred from the current phase.

#### Rollout gates

For every promoted definition, record these proof levels separately:

1. no-download upstream/source and schema parity;
2. graph materialization and dynamic-field reconciliation;
3. collapsed/expanded and save/refresh export equivalence;
4. optional-runtime, artifact, dependency, and resource admission;
5. visible-frontend live output on each claimed hardware recipe;
6. manual review approval before changing the public readiness claim.

### 2026-08-24 runtime recovery and app execution proof

With explicit user consent, the reviewed Transformers/PEFT optional environment
was repaired and activated using the frontend Setup flow. Live testing exposed
and fixed two generic client defects: Setup hid rollback-to-base when the broken
active optional environment had no previous optional environment, and graph
export generated random values outside declared Diffusers seed bounds. Focused
unit, typecheck, and browser regression tests pass, and a post-restart MoDiff
DDPM graph completed on ROCm with its output and run receipt preserved under
`test-review/hugging-face-cluster-initial-2026-08-24/runtime-repair/`.
The development proxy now exposes the read-only Hugging Face library endpoint,
and a fresh no-bridge browser smoke renders all 94 definitions.

A later backend restart correctly returned the active overlay to a state that
required verification. The visible Setup flow repaired and staged a replacement,
rolled back to the base process, and activated the repaired environment. That
flow exposed a client ambiguity when the catalog contained both an explicitly
named previous environment and a new same-spec staged environment. The client
now reserves the previous environment for the distinct **Rollback** action. A
catalog-only `staged_unchecked` environment is presented as **Repair**, not as
safely activatable; **Activate** is offered only for the environment ID returned
by the completed validation job in the current flow. Request-contract
regressions cover both cases, and activation still uses the backend's full
integrity verification rather than trusting the client-side selection alone.

This proves the generic app Diffusers execution path can run with the activated
optional overlay. It does not satisfy P5's per-Cluster live-parity requirements
and does not change an admission's `executable` flag. Insertability is now a
separate graph-qualified publication state.

### 2026-08-25 visible-frontend pipeline execution evidence

The current reviewed Studio execution graphs were exercised through visible
frontend controls, with receipts and generated media preserved under
`test-review/frontend-pipeline-e2e-2026-08-24/`. Successful runs cover Qwen
Image Edit, Qwen Image Edit Plus (single- and multi-reference), Qwen Image
Layered (four outputs), Qwen Image 2512 Control, Wan first/last-frame video,
and SDXL Modular text-to-image, image-to-image, inpainting, and ordinary
ControlNet Cluster routes.

Wan FLF was installed through Studio's visible missing-model action at the
pinned Hub revision `17c30769b1e0b5dcaa1799b117bf20a9c31f59d7`. Its final
visible-frontend smoke run selected Expert mode, two ordered images, two
denoising steps, five frames, and 512 x 512 geometry, then submitted task
`fEJrHEd2APrQ`. The exact nine-node Modular graph finalized with 19 managed
edges. Model loading, first/last-frame image embedding and VAE conditioning,
two denoising steps, route-authenticated decode, and video export completed in
57.6 seconds. The preserved output is a five-frame H.264 MP4 at 512 x 512 and
15 fps; extracted first, middle, and last review frames confirm that the first
and last source images reached their distinct upstream `image` and
`last_image` inputs.

Live testing fixed four generic frontend contracts rather than weakening
backend validation: Auto-to-Expert changes now rebuild the managed graph when
its fingerprint changes; programmatic dynamic signal actions are not executed
again by the rendered handle effect; number fields expose their generated
label target ID; and FLF's ordered two-image Studio value is split into one
first-frame loader and one last-frame loader while readiness requires both.
The implementation follows the pinned upstream Wan auto-block contract, where
the FLF workflow is selected only when both `image` and `last_image` are
provided.

Regression evidence after the live run: 58/58 graph-visual tests, 41/41 run
coordinator/issue-store tests, 119 backend Hugging Face download and Modular
route-state tests (one expected skip), TypeScript typecheck, production build,
and diff checks pass. The complete media hashes, screenshots, receipts, and
diagnostic attempts are indexed by the review-folder README.

This is live parity evidence for the existing reviewed Studio execution paths.
It is not collapsed-versus-expanded Cluster Node equivalence and therefore
does not by itself complete P3's remaining equivalence proof or promote any P5
catalog definition to executable.

### 2026-08-25 first executable Cluster qualification

The graph-qualified **Qwen Image Edit — Edit Image** definition now has a
fail-closed volatile execution authority. Preparing it runs the backend-declared
dynamic field actions and verifies the bounded Studio node receipt, live runtime
fingerprint, installed artifact revision, active optional runtime, exact model
dependencies, and selected backend Auto candidate before enabling its derived
children. Changing the Cluster input, execution parameters, mode, or children
revokes that authority and removes the derived executable graph.

The collapsed Cluster was inserted, prepared, and submitted through visible
frontend controls as task `NcYIqr7a53cu`. It completed on ROCm in 112.8 seconds
using `Qwen/Qwen-Image-Edit` at revision
`ac7f9318f633fc4b5778c59367c8128225f1e3de`, bfloat16, model CPU offload,
two denoising steps, seed 42, and guidance 4. The 1024 x 1024 output has SHA-256
`c75676c1d09bdf3d19bf8d12df082cdb864609ed11c06b703b07d9d7f37148c2`.
The complete receipt, source, output, and before/after screenshots are under
`test-review/frontend-pipeline-e2e-2026-08-24/qwen-cluster-qualification/`.

The run exposed a provenance-only client defect: the submitted graph carried
the correct Cluster Auto authority, but the output snapshot cloned the global
canvas form, which custom graphs label Expert. The coordinator now computes the
run-input hash from, and captures, the authority's effective Auto form. A
dedicated regression test covers this behavior. The original execution and
artifact remain valid; its receipt records the metadata discrepancy explicitly.

The subsequent refresh proof exposed and fixed two restored-root lifecycle
defects. Persisted checkpoints now omit all definition-derived block/execution
children and preview fields, and the root fetches its exact pinned definition
while disabling Prepare until it arrives. The backend generic prompt encoder
also now republishes its dynamic schema when a deterministic node ID requests
the same model type after refresh, matching the idempotent behavior already
implemented by the other generic Modular actions.

This is an internal qualification execution, not an immutable public
`executable` or `autoEligible` publication claim. Manual visual approval and the
remaining definition-by-definition promotions are still open.

### 2026-08-25 SDXL IP-Adapter and combined-workflow qualification

The SDXL IP-Adapter artifact contract now selects the three exact files at
`h94/IP-Adapter@018e402774aeeddd60609b4ecdb7e298259dc729`:

- `sdxl_models/ip-adapter_sdxl.safetensors` — 702,585,376 bytes, SHA-256
  `ba1002529e783604c5f326d49f0122025392d1d20ac8d573b3eeb3e6dea4ebb6`;
- `sdxl_models/image_encoder/config.json` — 2,013 bytes, SHA-256
  `1d53c2b4b74c5f85171d313adda3e3b8771ff5c698ee66a29710d0ac822298e4`;
- `sdxl_models/image_encoder/model.safetensors` — 3,689,912,664 bytes,
  SHA-256
  `657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d`.

The frontend Model Manager installed that bounded selection and changed its
exact requirement to Ready. The visible Setup flow then rolled a stale active
optional runtime back to base, repaired it from reviewed locked artifacts,
activated environment `runtime-1787677346-2d28819c`, and verified the restarted
process as active. Browsing, insertion, and Auto planning still perform neither
operation implicitly.

Visible-frontend execution exposed two normal upstream transformations that the
initial process-local receipt treated as tampering. Diffusers/Accelerate moves
the same adapter `Parameter` objects between CPU and the execution device under
model CPU offload, so device placement is no longer part of the immutable
parameter seal; identity, Torch mutation version, shape, and dtype remain
sealed. The pinned SDXL before-denoise block also assigns `torch.cat` results
back into both IP-Adapter embedding lists even at batch size one. Denoise now
passes shallow copies of only those list containers upstream, preserving the
backend-issued bundle and tensor provenance without weakening its validation.

The collapsed standalone `ip_adapter_text2image` Cluster completed as task
`3C9uMRbryjfR` in 35.3 seconds and produced one 512 x 512 WebP (119,354 bytes,
SHA-256
`88ff5d2521b0c8fc3b1b942bfd3df16b10de738832480473f1e61b1ce3fa310e`).
The maximal `ip_adapter_controlnet_union_inpainting` Cluster imported its source,
mask, control, and IP reference through the Assets UI, prepared in Auto mode,
expanded 13 upstream block containers, executed its 14-node API graph as task
`UGuDu4XZrwbe`, and saved workflow `Jl5be9qjnslt12x_enZMa`. A fresh browser
reopened that workflow, verified every persisted input, collapsed it, prepared
it again, and completed task `cuty3PzM8b4K`. Both runs used execution fingerprint
`hfcluster_585895fb`; their executable nodes and paths have the same SHA-256
`3f05ff24ac4d13bdfbbfcc3b2ea14256ff762bf9b8599f8cc1c3d5a55490ec1d`,
and both produced the same 119,216-byte WebP with SHA-256
`5e70bbf7f85bd171e073b48e9f1d2772da3b342ed22cee698d1f41069ef930da`.
Auto evidence correctly advanced from static “This should work” to local
“Ran here” after the first success; that observational metadata is not part of
the executable-graph equivalence projection.

Evidence is indexed under
`test-review/frontend-pipeline-e2e-2026-08-25/`. Consolidated verification after
these runs passed 371 backend tests with 1,537 parameterized subtests (19
expected skips and the existing Diffusers `torch_dtype` deprecation warning),
plus frontend typecheck, style audit, all unit suites, and the production build.
The remaining promotion work is manual approval of these outputs, individual
live proof for admitted workflows not yet executed, additional pinned Modular
workflow adapters, and definition-specific live proof for the remaining six
initial Transformers Cluster Nodes.

### 2026-08-25 Transformers Cluster qualification

The immutable catalog now merges 94 pinned Modular Diffusers workflow
definitions with seven reviewed Transformers composites. Its provider-neutral
parser, insertion, persistence, graph ownership, runtime authority, and API
export paths accept both providers while enforcing provider-specific IDs,
surfaces, task contracts, artifacts, and optional runtimes. Transformers
discovery performs no import, download, installation, activation, or model
load.

The visible Setup flow verified that Linux x86-64 requires the exact
`huggingface-transformers-main-96fe6dce-peft-0.20.0` target profile with digest
`sha256:189e8c337059de39d06025b1f29d641b45d5c7b78cdb54c0c9dd909d86d5e487`;
the earlier cross-platform 5.14.1 candidate was rolled back rather than being
silently accepted. The exact SmolLM2 artifact was already complete at commit
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.

Through visible frontend controls, a user inserted **SmolLM2 135M Instruct —
Text Generation**, edited `max new tokens` to 32, prepared the exact Expert
CUDA recipe, and ran task `AnTO3UhekQjK`. Numeric root controls are normalized
to their reviewed JSON integer/float types before persistence; a regression
test rejects invalid numeric overrides. The workflow was saved and opened in a
fresh page, where the prompt and token override remained while volatile derived
nodes and authority were correctly absent. Re-preparation restored the same
execution fingerprint, expansion exposed the three exact generic task nodes,
and task `QwwczUYWr_QA` completed from the expanded view.

Both views exported byte-identical API `nodes` and `paths` with SHA-256
`0fc7f74f6c97e0b5baaf07d00f62a5b0244c665657a0f76b9d9503bee6a5e9c2`,
used execution fingerprint `hfcluster_b7d88a02`, and generated identical text
with SHA-256
`9fba278361b8d347e1d96f767a0b3a9fa3c05018bb4664c616d72ea89e29f41c`.
The generated result independently records the exact model revision,
`cuda:0`, `float32`, 39 input tokens, and 32 generated tokens. Receipts,
focused state captures, the generated text/result, repeatable browser harness,
and screenshots are under
`test-review/frontend-pipeline-e2e-2026-08-25/transformers-cluster-smollm2-text-generation-v1/`.
This remains internal qualification evidence and does not change a public
publication flag.

### 2026-08-26 Diffusers-first and speech-to-text qualification

The scoped catalog now publishes 94 exact Diffusers Cluster definitions and
eight Transformers Cluster definitions backed by 505 deduplicated block
contracts. All Diffusers definitions are structurally insertable and
expandable. Structural support is deliberately distinct from executable
support: contract-only, equivalent-standard, and custom-container workflows
retain their truthful readiness until their exact runtime adapter, artifact,
resource recipe, and live evidence are admitted.

Five additional Diffusers qualification runs were submitted through visible
frontend controls and preserved under
`data/qualification/local-review/hugging-face-clusters/`:

| Cluster               | Task           | Preserved output SHA-256                                           |
| --------------------- | -------------- | ------------------------------------------------------------------ |
| LTX text-to-video     | `EeiDMyJJS9KS` | `063e3d69ce32a65fc404715b159e1b9b4d6c035db67d750d1bd77cf61da742dd` |
| LTX image-to-video    | `Arfu4PbTvHHf` | `ec30a24ad211245bebbd8c35ab7bb98626afbf75ae856396e256bba49e39c08c` |
| Wan2.2 image-to-video | `CU-hnQlxdCVQ` | `eea11d7f0f21f370a9f50c3500cacd709f00b5fbade520a341c92c287131345d` |
| Qwen Image Edit       | `wgs2hPSzAvrC` | `0e1a8007ca9ae7ac60cda9da3a659c0f78ce68bb2dcd7d38927a68039f4a9a93` |
| ERNIE Image           | `yFCZBqKXIqhy` | `8b7811e87bc36e749dd5fc4d9a9212e43b07966e1cb8490a5aa6184571310d55` |

These outputs are technical qualification evidence. In particular, the short,
low-step video runs prove media routing and temporal output but are not
showcase-quality generation examples. They are not part of a new visual-review
request: the user has already reviewed the earlier review queue and its
recorded decisions remain authoritative. Showcase prompts, durations, frame
counts, and quality parameters will be tuned only after the implementation and
qualification gates in this plan are complete.

The exact Wan2.2 text-to-video snapshot could not be installed safely: its 49
files total 126,200,628,126 bytes (117.53 GiB), while the host had 67 GiB free
at the check. No unrelated cache was deleted. The bounded blocker receipt is
`diffusers-wan22-text-to-video-blocker.json`; the consolidated successful and
blocked result index is
`diffusers-equivalent-video-clusters-live-results.json`.

Official LTX-2 documentation and the pinned `LTX2AutoBlocks` source establish
two exact overlaps: modular `text2video` and `image2video` can use the reviewed
standard `LTX2ConditionPipeline` graph and retain synchronized video/audio
output. Those two definitions are now independently graph-qualified. The
`condition` workflow remains contract-only because it accepts arbitrary lists
of image/video conditions at latent indices, and `in_context` remains
contract-only because it requires separate IC-LoRA reference-condition
semantics. The app therefore supports partial workflow promotion without
relabeling an entire model family.

The LTX-2 frontend Install contract is sealed to 45 exact standard-pipeline
files at revision `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`; it excludes the
unrelated root single-file variants and duplicate Diffusers-format text-encoder
weights. The bounded selection is still 92,074,139,514 bytes (85.75 GiB), so it
cannot fit in the current 65 GiB free space and was not partially downloaded.
The receipt is
`diffusers-ltx2-equivalent-cluster-storage-blocker.json`.

Transformers work in this phase is restricted to speech-to-text. The visible
frontend already completed Whisper Tiny transcription and speech-to-English
translation. It now also installed and activated the exact reviewed optional
runtime, installed `facebook/wav2vec2-base-960h` at immutable revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156` with the bounded file allowlist,
inserted the CTC Cluster, edited its audio and timestamp parameters, saved the
workflow, refreshed the page, and proved those effective values and overrides
were unchanged. The collapsed and expanded four-node API graph exports were
identical. Visible task `bstoqkySg96n` transcribed the 5.12-second review
fixture as “AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN” and
returned word timestamps. The receipt is
`transformers-asr/wav2vec2-ctc-cluster-frontend-result.json`.

Live qualification exposed and fixed two integration defects without relaxing
the contracts: a stale active optional environment may now be replaced only by
a freshly validated environment owned by the same trust class, and the client
now has an exact static CTC resource/model profile instead of dereferencing an
unknown profile during planning. Studio's explicit Install action also sends
the backend-declared immutable revision and bounded download selection.

On the same date, the visible Model Manager used a new explicit local-eviction
contract to remove six exact completed-proof or deferred model revisions while
retaining every saved workflow and canonical definition. The confirmation
warns that retained workflows must redownload the exact revision before their
next run; active graphs, queued work, downloads, ambiguous revisions, and stale
plan hashes still block deletion. The operation reclaimed 210,565,616,193
bytes and increased free space from about 65 GiB to 260.7 GiB. Its receipt and
screenshots are under `storage-turnover/`. This resolves the LTX-2 and Wan2.2
text-to-video capacity blockers; their exact frontend downloads and live runs
are now pending rather than storage-blocked.

The first unblocked install exposed a generic Model Manager gap: exact
Expert-only profiles were visible but could only show an inert **Expert** badge
when the Auto planner correctly returned `manual_only`. Model Manager now
offers an explicit Install/Repair action in Expert mode only when the
authoritative backend capability supplies one repository, one immutable
40-character revision, a bounded validated file selection, and an active exact
optional runtime. The client cannot invent or widen any of those fields. A
mocked visible-browser test proves the `/hf_download` request carries the exact
reviewed revision and files. LTX-2 installation is now running through that
visible frontend path; after it finishes the gated Cluster test will save,
refresh, compare effective parameters, compare collapsed and expanded API
graphs, execute the graph, and preserve the generated media as qualification-
only evidence. Wan2.2 text-to-video follows after LTX-2 is explicitly evicted
through the frontend to retain the configured free-space safety margin.

The shared Wan image-to-video class also exposed a second generic schema gap:
one upstream pipeline class can own multiple workflows whose immutable model
artifacts are different. Capabilities now publish validated per-workflow
`artifactSelections` instead of collapsing those repositories into one
ambiguous revision list. The first admission binds `single_image_to_video` to
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers@b184e23a8a16b20f108f727c902e769e873ffc73`
with 36 selected files (90,097,576,675 bytes), and `image_to_video` to
`Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers@17c30769b1e0b5dcaa1799b117bf20a9c31f59d7`
with 41 selected files (90,104,408,960 bytes). Studio readiness and Model
Manager resolve the exact selection by workflow mode, display separate Expert
install rows, and still reject duplicate mode ownership, mutable revisions,
unsafe paths, or incomplete selections. The backend app-download planner uses
the same selection records. A visible mocked-browser proof now renders both
rows with their distinct artifact labels and verifies that each Install action
sends only its matching repository, immutable revision, and bounded file list.
The same proof exposed and fixed a generic merge defect: an optional field
omitted by an otherwise authoritative backend capability can no longer erase a
known static profile value by replacing it with `undefined`.

The immutable Wan2.2 text-to-video repository contains 49 files totaling
126,200,628,126 bytes. Its executable install contract intentionally excludes
the six `assets/` illustrations, leaving the 43 runtime files and
126,199,294,813 bytes asserted by the backend planner and downloader. The
upcoming visible install must use that bounded selection rather than the full
repository tree.

MiniMax Music 3 is the first reviewed current-pin workflow whose faithful
execution graph is the package-owned Modular workflow itself rather than an
equivalent standard pipeline or MoDiff's older encode/denoise/decode adapters.
The exact repository commit is
`MiniMaxAI/MiniMax-Music3@fbdf52fbaaca799592917417eb05f1899f1255ec`.
Its full repository is 57,353,379,600 bytes; the reviewed runtime closure is 23
files and 28,517,608,999 bytes and excludes the legacy `qwen_7B/`,
`flowmatching_vae.pth`, `dav.pth`, examples, and publisher assets. The artifact
receipt is `data/minimax-music3-artifact-review.json`.

The repository uses the MiniMax-Music3 Community License, not Apache-2.0. It
requires revision-bound review before installation or execution, contains
commercial-product attribution and hosted-generation safeguard terms, and
requires separate authorization above its stated annual-revenue threshold.
The UI must display and record that exact immutable license before this model
can be exposed for download or Run.

The reusable official-workflow executor foundation is implemented as three
Block Nodes matching the pinned upstream top-level sequence:
`semantic_generator -> denoise -> decode`. Each node selects the reviewed
package block, calls `init_pipeline()`, and uses the normal `ModularPipeline`
call interface. The implementation does not call a block's internal
components/state protocol and does not reproduce upstream autoregressive,
flow-matching, scheduler, or vocoder loops. State passed between visual nodes
is process-local, sealed to the exact loader execution and workflow, immutable,
and deliberately nonserializable; saved graphs persist parameters and edges,
then recreate runtime state by rerunning from the first block.

MiniMax remains fail-closed for real execution while the following gates are
tracked independently:

- [x] Pin and hash the exact artifact, component descriptors, official block
      sources, workflow order, outputs, and license.
- [x] Add an all-component Models Loader bundle and official package-block
      executors with isolated contract tests.
- [x] Add the exact `text_to_audio` Studio execution specification and Cluster
      adapter: Models Loader -> Semantic Generation -> Denoise -> Decode Audio ->
      Export Audio.
- [x] Add revision-bound license acknowledgement to the frontend install,
      qualification, and Run flows.
- [x] Publish the exact 23-file bounded Model Manager selection only after the
      preceding policy gate is enforceable.
- [x] Add the reviewed optional-runtime/resource profile. Its activation must
      still be requalified after the profile digest change.
- [x] Qualify the
      collapsed/expanded, save/refresh, API-equivalence, and real CUDA frontend
      flow. Auto remains disabled until that evidence is manually approved.

The frontend parser and materializer now accept the exact `workflow` adapter
identity, the official whole-workflow integration status, its sealed workflow
and block paths, and the 44.1 kHz numeric output binding. A focused client test
materializes the five-node execution graph, proves its three pipeline-component
fan-out edges and two sealed state edges, and compares canonical collapsed and
expanded snapshots after a JSON save/restore round trip. A visible mocked
Model Manager test proves no download request occurs before acknowledgement
and that confirmation sends only the exact revision and 23-file allowlist.

### Demo acceptance matrix for the original one-node/customizable-node goal

Catalog size is not the completion metric. The current immutable library has
94 Diffusers Cluster definitions. Forty-eight workflow admissions across 47
Diffusers definitions have an exact graph-qualified execution contract; the
remaining 47 definitions are structurally insertable/expandable but catalog-
only. All new publication `liveProof` flags remain false pending explicit
approval.

The minimum demo set is one image, one video, and one audio Cluster. Each demo
must perform the same visible user journey in one test and preserve its own
receipt and generated asset:

1. Insert the first-party Cluster from **Diffusers Cluster Nodes**.
2. Run it collapsed as the single “magic” node experience.
3. Expand it, edit at least one parameter through an internal Modular block,
   and prove the owning Cluster instance changes.
4. Save the workflow, refresh the browser, reopen it, and prove the edited
   effective parameters are unchanged rather than reset to model defaults.
5. Prepare the exact runtime graph again (volatile weights/runtime state are
   correctly not serialized), compare collapsed and expanded canonical/API
   graphs, then generate through visible frontend controls.
6. Prove another Cluster instance and another workflow retain their own model,
   prompt, and parameters.

Current evidence against that combined standard:

| Modality | Candidate                      | Real frontend output                                                                                                                     | Save/refresh parameter proof                                                                                                                | One combined demo                                                                                                                                                                                                                      |
| -------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image    | Qwen Image text-to-image       | Cluster task `R5flQQqzyeQB` and customized User Node task `376IQyhhHcq1` generated real 256x256 WebP assets through the visible frontend | Prompt, width, height, steps, seed, device, dtype, guidance, and offload values are byte-for-byte equal before and after browser refresh    | Complete; collapsed/expanded, Cluster/customized User Node, and User-Node/save/refresh execution graphs are equal; an existing canvas node was adopted with a real pointer drag; both generated assets have SHA-256 `61cba26f...ccec2` |
| Video    | LTX-2 text-to-video with audio | Task `gcO67kuvAufh` generated a real MP4 through the visible frontend from the exact 92.07 GB snapshot                                   | Prompt, dimensions, steps, frame count/rate, seed, guidance, dtype, device, and offload values are equal before and after browser refresh   | Complete; collapsed/expanded graphs are equal and output hash is `8d6c532f...ea965`                                                                                                                                                    |
| Audio    | MiniMax Music 3                | Task `KqYefJpE0m8c` generated a real 2-second 44.1 kHz stereo WAV through the visible frontend from the exact 28.52 GB snapshot          | Prompt, multiline lyrics, duration, steps, seed, device, dtype, and offload values are byte-for-byte equal before and after browser refresh | Complete; collapsed/expanded graphs are equal and output hash is `a296989b...41f1`                                                                                                                                                     |

The existing generated image/video assets are technical qualification evidence,
not showcase media. Showcase prompt and quality tuning remains after these
implementation proofs.

Expanding a first-party Cluster exposes its exact upstream Modular hierarchy
and editable shared parameters. The legacy implementation then reconstructed a
topology-changing instance as a different User Node. That reconstruction is
superseded: it changed controls and public ports, could discard external edges,
and conflated workflow customization with reusable-definition creation. The
following bullets describe historical evidence, not the accepted product flow:

- **Use Cluster** keeps the reviewed first-party Cluster immutable and gives the
  one-node experience.
- **Customize as User Node** materialized the internal execution graph through
  the generic User Node derivation path, removed first-party execution authority,
  and replaced the wrapper. Automatic conversion on a child drag is now a known
  defect rather than intended behavior.
- Cluster Nodes and User Nodes cannot be nested. The converted instance receives
  a workflow-context name, and its Save action offers **Update existing User
  Node**, **Save as new User Node**, or **Keep only in this workflow**.
- Saving/refreshing either representation persists only graph structure,
  parameters, immutable artifact identities, and presentation state; model
  objects and intermediate tensors are recreated at Run time.
- **Import from Hugging Face Hub** must accept only an immutable commit and a
  declarative reviewed sidecar, show the discovered blocks before insertion,
  and import the result as User Nodes unless it exactly matches a first-party
  reviewed definition. Repository Python remains disabled without separate,
  fresh operator authorization.

The historical implementation checklist was:

- [x] Add **Customize as User Node**, preserving the exact finalized graph,
      parameters, artifact revision, and origin receipt while removing Cluster
      ownership and execution-admission authority.
- [x] Allow a normal library node dropped inside an expanded User Node to be
      adopted into its editable topology; prove the injected topology survives
      collapse and JSON save/restore without mutating the source Cluster.
- [x] Adopt both new library nodes and existing ordinary canvas nodes into an
      expanded User Node; automatically customize an expanded Cluster first;
      reject User Node/Cluster nesting; and keep each canvas gesture one
      undoable graph transaction.
- [x] Add the three reusable-definition choices: update the current reusable
      User Node, save a new opaque User Node definition, or keep the embedded
      change only in the current workflow.
- [x] Complete the visible live execution proof for the customized User Node. The
      Qwen combined frontend test executes the first-party Cluster, customizes it,
      saves and refreshes the User Node, and executes it through the ordinary
      graph executor. The two generated outputs are byte-identical for the same
      immutable model, parameters, seed, and internal graph.
- [x] Add a visible immutable Hub import entry that creates a pinned User Node,
      hands the exact repository to Model Manager, keeps repository Python off,
      and preserves repository/revision parameters across browser refresh.
- [x] Extend Hub import into a single install -> preview -> admission summary ->
      save-to-library wizard and translate the official
      `mellon_pipeline_config.json` sidecar without importing repository Python.
- [x] Complete one real official-Hub browser proof for the importer.
- [ ] Design a
      separately authorized sandbox boundary for repositories that require
      custom Python.
- [x] Complete the combined image, video, and audio live demo set.

The active corrective implementation is tracked in the
[unified composite-node plan](unified-composite-node-implementation-plan-2026-09-01.md):

As of 2026-09-01, the synchronized V2 schemas, backend User Node definition
store, registered-definition compiler, fail-closed capability resolver, pure
runtime/projector, workflow persistence canonicalizer, and shared V2
`BlockNodeFrame` renderer branch are implemented. The
runtime's 17 focused tests fail closed for stale/spoofed projections, missing
bindings, invalid internal connection semantics, duplicate identities, and ID
collisions; the compiler removes legacy Cluster-prefixed field options after
translating them into the source-neutral V2 contract. Already-V2 API export,
run readiness, and run-target selection now use the same concrete internal
graph in collapsed and expanded views. Public connector discovery and the
shared connect/remove/status/signal paths derive their sockets from the
explicit V2 boundary, and admitted same-owner internal edge changes update the
instance effective graph copy-on-write. Public port IDs are unique across both
input and output directions so the shared root handle namespace is
unambiguous; both client and backend V2 validators enforce that invariant.

These slices are intentionally not proof of broad production convergence: the
exact Qwen Image text-to-image catalog admission now uses the V2 route, while
the remaining registered catalog admissions and Auto still have legacy
Cluster routes. The
general structural capability remains disabled until replacement, move-out,
public-interface crossing, full reconnection, and their failure recovery all
durably update `effectiveGraph`; disconnected ordinary-node adoption and
eligible internal-node deletion are already copy-on-write, undoable, and
fail-closed at unsafe boundaries. Configure
interface remains disabled until `BlockInstanceV2` can persist a strictly
validated effective-interface snapshot. The three V2 save choices are now
wired: workflow-only performs no reusable-definition write, save-as-new
preserves the exact effective graph/interface/defaults/preview bindings and
parent provenance, and update-existing is limited to mutable user-owned
definitions. API rollback, semantic-race rejection, target-only rebasing,
open-workflow isolation, and independent click/drag reinsertion have focused
coverage; boundary-only valued inputs fail closed because `BlockPortV2` has no
reusable default. This is not yet a browser E2E claim. A migration-window
containment fix coalesces concurrent capability loads and preserves legacy
child layout across Save/refresh for two independent Qwen instances; that
regression is safety evidence, not acceptance of the legacy renderer.

- [ ] Persist registered and user-owned composites as `BlockDefinitionV2` and
      `BlockInstanceV2` through one canvas type and renderer.
- [x] Treat internal movement as presentation-only and retain internal layout
      by stable semantic node ID.
- [ ] Apply topology edits copy-on-write to the same workflow instance without
      replacing its wrapper or creating a reusable User Node.
- [ ] Preserve explicit controls, inputs, outputs, values, external edges,
      dimensions, preview, and provenance across customization.
- [ ] Separate reviewed/Auto authority invalidation from source identity and
      rendering.
- [ ] Migrate and requalify legacy Cluster and Cluster-derived User Node data.

### 2026-08-26 legacy implementation and live-test evidence

This section records the now-superseded implementation and remains only as
migration/test evidence. The first-party Cluster-to-User-Node customization was
implemented in the visible Cluster header and automatic structural-edit paths.
Customization materialized the exact backend-admitted execution skeleton,
finished backend-declared dynamic fields, stripped Cluster ownership and
authority, enabled ordinary parameters (including formerly sealed values),
saved the result in **User Nodes**, and replaced only that Cluster instance.
That behavior does not satisfy the V2 contract even though the source catalog
definition remained unchanged. Existing canvas nodes and new
library nodes can be adopted directly; nested User/Cluster Nodes are rejected.
Both forms use the existing MoDiff graph executor; User Nodes do not have a
second runtime.

The visible Hub import wizard is implemented under **User Nodes**. It accepts
only `owner/repository` plus an exact 40-character commit, requires an explicit
review checkbox, waits for Model Manager to install that revision, inspects the
bounded sidecar, translates official Mellon block/port metadata, previews
admission and remote-code status, and saves the result as a User Node. A mocked
browser test proves the complete flow and browser-refresh persistence. A
Mellon-only repository becomes a disabled visual User Node because its custom
Python is never imported. A MoDiff sidecar can become executable only when it
resolves entirely to already reviewed, installed official classes/components.

Focused client verification is green: typecheck, lint, and 34 focused graph and
Cluster/User-Node tests, including exact Cluster/customized API-export equality, editable unsealed
fields, normal-node adoption inside an expanded User Node, and persistence of
the modified topology after collapse and JSON serialization. The targeted
backend Cluster/Hub/Modular Diffusers suite passes 66 tests, 9 skips, and 78
subtests under the managed ROCm environment. Two focused mocked browser tests
also pass.

The exact LTX-2 snapshot finished installing through the visible Model Manager:
`Lightricks/LTX-2@47da56e2ad66ce4125a9922b4a8826bf407f9d0a`,
45 selected files and 92,074,139,514 bytes. The combined LTX frontend test now
performs internal-block prompt editing, save, browser refresh, parameter
comparison, collapsed/expanded API comparison, and real generation. During the
first live reruns, the current backend added four app-owned built-in capability
profiles whose `builtin://modiff/.../v1` artifact identity was rejected by the
client's Hugging Face-only repository validator. That production refresh bug
is fixed: versioned app-owned identities validate only for `artifactKind` =
`builtin`, malformed paths still fail closed, and the full live response now
parses 112 recognized profiles plus 223 task contracts. Workflow GET/PUT/DELETE
uses a bounded 120-second timeout so a cache-heavy startup cannot silently
turn an explicit Save into a 15-second false failure.

The LTX live run also found two execution-boundary defects that are now fixed.
`LTX2ModularPipeline` is explicitly qualified against its reviewed equivalent
standard `LTX2ConditionPipeline` executor. That executor resolves the exact
40-character commit to the already installed, bounded local snapshot and gives
Diffusers that local directory; it no longer asks Hugging Face Hub to validate
unrelated files outside MoDiff's reviewed 45-file closure. Focused backend
regressions cover both the equivalent-executor receipt and the exact local
snapshot boundary. The complete browser run generated task `gcO67kuvAufh` and
saved its receipt and MP4 under the phase review directory.

The combined Qwen Image test now covers insert -> expand the official
`text_encoder` block -> edit prompt -> save -> refresh -> compare effective
parameters -> compare collapsed/expanded API exports -> generate -> drag an
existing Data Viewer node into the expanded Cluster -> automatically customize
that one instance as a workflow-named User Node -> remove the injected proof
node -> expand/collapse -> save/refresh -> compare API export -> generate again
through the ordinary User Node executor. Its installed immutable artifact
is `Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26`.
The latest complete browser run passes in 1.5 minutes with Cluster task
`R5flQQqzyeQB` and customized User Node task `376IQyhhHcq1`. It preserves both generated WebP files and the
full receipt under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-cluster/`.

Run readiness now validates a collapsed User Node against the same expanded
internal execution graph used by graph export and Run. This fixes the final
refresh defect where the saved internal Preview existed but the collapsed
wrapper incorrectly reported that no output node was connected. Structure,
model, and device/offload checks all use that common execution view. The
58-test graph suite includes a collapsed internal-Preview regression.

The combined MiniMax Music 3 test now covers the same flow through the official
`semantic_generator -> denoise -> decode` Modular blocks, including editing the
prompt and lyrics in `semantic_generator.tokenize`, exact 23-file frontend
installation, collapsed/expanded equality, refresh persistence, and real audio
output. The user's exact revision-bound license acknowledgement was applied,
and the latest visible-frontend run passed in 2.0 minutes with task
`KqYefJpE0m8c`. It persisted structure tags on their own lines in the new
multiline lyrics control and produced a 16-bit stereo 44.1 kHz WAV. Future runs
remain gated on the same explicit legal acknowledgement environment switch.

The remote-code-disabled Hub wizard is complete. The real official-Hub proof
installs
`diffusers/gemini-prompt-expander-mellon@0562591cbb2144060ce641aaf101fea686a4cb71`,
translates `mellon_pipeline_config.json`, previews its block hierarchy and
`prompt`/`out_prompt` ports, reports preview-only admission, confirms a disabled
User Node with repository Python off, and proves its exact identity survives
Save/browser refresh. Repositories requiring repository Python remain a
separate explicit sandbox/authorization feature estimated at two to four
working days; they fail closed in this demo build.

All integrity ledgers were regenerated from the exact pinned Diffusers package,
the reviewed Transformers Git checkout, and the locked production wheel after
these admissions. The complete backend gate now passes 2,156 tests, 46 skips,
and 6,507 parameterized subtests. The reviewed LTX-2.5 and Wan Animate 2 source
receipts remain explicitly contract-only because they were authored against a
newer Diffusers revision than the app pin; their tests now prove that boundary
instead of conflating future source research with current runtime support.
The previously approved gallery runtime files were restored to the backend
asset endpoint for fixture resolution, but they remain outside this phase's
review queue.

The remaining repository gates are also green: Ruff's fatal-error/undefined-
name check passes, the managed backend environment has no dependency conflicts,
and preflight reports the AMD/ROCm runtime ready on an unused validation port
while the live frontend download continues on the intentionally reused app
port. The complete client gate passes formatting, lint, type checking, all unit
suites, production build, and bundle budgets; the final JavaScript total is
574,001 gzip bytes against the 575,488-byte cap. The per-workflow Model Manager
browser proof passes independently.

Public `executable`, `autoEligible`, and `liveProof` flags remain false for all
of these new receipts pending the explicit publication decision recorded by
the promotion gate; this does not reopen previously reviewed media. The remaining Diffusers work is
definition-by-definition runtime admission and live evidence for catalog-only
or not-yet-run workflows, including the 12 custom-container runtime adapters;
the newly unblocked Wan2.2 text-to-video and LTX-2 runs are next. The
remaining Transformers work in this phase is limited to the pinned pipeline's
generic speech-seq2seq, transducer/TDT, and CTC-with-LM branches after the
Diffusers gates; non-speech Transformers families remain deferred.

### 2026-08-27 FLUX.2 Klein Cluster qualification expansion

The next definition-by-definition Diffusers wave live-qualified both admitted
FLUX.2 Klein workflows without changing their immutable publication flags:

| Definition | Task | Expanded blocks | Generated WebP SHA-256 |
| ---------- | ---- | --------------: | ---------------------- |
| `diffusers.modular:Flux2KleinModularPipeline:text2image` | `06_fORfy8Bw2` | 10 | `407f58958b2f6d23556f08f3088db866e106252c6145b6270643d872cd5fdcac` |
| `diffusers.modular:Flux2KleinModularPipeline:image_conditioned` | `VTL8AsHT5m8b` | 13 | `d569d38cb4431bc4642c090f95fb86fa4b3557be9a461eeb1ff61f24ad96cff8` |

The visible frontend inserted each Cluster from **Diffusers Cluster Nodes**,
changed its prompt, dimensions, steps, seed, dtype, and offload recipe, saved
the workflow, refreshed the browser, and compared the complete effective input
and execution-override snapshots. Both snapshots survived exactly. Each
Cluster then expanded to its pinned upstream hierarchy without changing the API
graph, collapsed to the same graph, prepared against the authoritative runtime
capability, and generated a real 256 x 256 image from
`black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The edit route consumed the first task's preserved backend asset and visibly
changed the red cube into translucent green glass.

The run exposed and closed two generic integrity gaps. First, 37
image-oriented Modular Diffusers specifications had sealed the repository but
had not bound `ModelsLoader.revision`; all now bind the exact reviewed commit,
and all 48 admitted routes are covered by a regression requiring their sealed
revision to equal the admission artifact revision. Second, the pinned FLUX.2
model index serializes the legacy `Qwen2TokenizerFast` name while Transformers
5 consolidates that export into the tokenizers-backed `Qwen2Tokenizer` class.
MoDiff now admits only that exact repository-scoped official alias and still
rejects arbitrary tokenizer classes. The closed client model registry also now
retains the three admitted FLUX Modular capability families, and Cluster
preparation refreshes a missing authoritative capability snapshot before
failing.

The combined receipt, browser screenshots, and generated assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/flux2-klein-clusters/`.
They are qualification-only outputs with `showcaseApproved: false`.
`executable`, `autoEligible`, and `liveProof` remain false until the user
explicitly approves each definition's output and promotion receipt. The next
coverage candidates are the already-admitted FLUX.1-dev and FLUX Kontext
definitions; catalog-only definitions still require real runtime
action/profile/admission work before live qualification is possible.

The post-change release gate is green: all 2,206 backend tests ran with 47
intentional skips, the complete client lint/type/unit/build/bundle check
passed, and all 114 mocked Studio browser tests passed in one 6.2-minute run.
The browser regression now also requires a saved
`Flux2KleinModularPipeline` loader to retain that exact Modular identity after
hydration instead of falling back to the standard pipeline sharing its
repository.

## Acceptance invariants

- Registered Cluster and user-owned definitions use one canonical composite
  canvas type, renderer, public-boundary model, persistence contract, expansion
  mechanism, and connection behavior.
- Built-in Hugging Face definitions never appear as User Nodes.
- The frontend does not select Python classes or contain model-family graph branches.
- Shared task behavior is implemented once; family/model differences are narrow backend data or adapters.
- Upstream visibility does not imply MoDiff executability or Auto readiness.
- Collapsing or expanding a Cluster Node never changes its execution semantics.
- Moving an existing internal node changes only that instance's saved internal
  layout; it never reconstructs the wrapper or creates a reusable User Node.
- Structural customization is copy-on-write within the same instance and keeps
  controls, public ports, values, dimensions, preview, and external edges until
  the user explicitly changes the interface.
- Parameters and structurally customized User Node instances survive
  save/refresh without silent reset.
- A model change cannot mutate prompts or parameters in another graph instance.
- Browsing nodes performs no model download, weight load, optional-runtime installation, or remote-code execution.
- The existing MoDiff graph remains the only graph executor.
