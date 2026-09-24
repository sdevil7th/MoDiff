# Image prototyping readiness and Álvaro handoff plan

Status: **Planned; not an implementation-complete or qualification claim.**
Decision date: 24 September 2026.
Baseline: backend `59bc342`, client `6e93329`, plus the existing uncommitted changes in both worktrees.
Reviewed Diffusers snapshot: `fbf49e7f35857f76bc57b177e26f12b03687c668`.

This is the controlling plan for the next **image prototyping handoff**. It supersedes the completion criteria of earlier workbench/demo plans for this handoff, not their implementation history. It preserves the existing executor, generic nodes, Block representation, security boundaries, model installations, approvals, and user workflows. It is mirrored in both repositories.

## 1. Agreed outcome and boundaries

The user answered these scope questions explicitly:

1. Whole-pipeline exceptions are acceptable when there are no usable upstream Modular Diffusers stages. Document **each exact exception**, its reason, limitations, and evidence. Every advertised image family with usable native stages needs a real editable implementation; a standard pipeline producing an image does not satisfy this requirement.
2. Windows Qwen on 16 GB VRAM / 32 GB RAM may remain an explicitly documented limitation. It is **not** a blocker to inviting Álvaro to test.
3. Qualify **every advertised image model**, including additional downloads, not only models already cached locally.

The intended experience is a small, visible graph: **Load Models → Encode Prompt → Denoise → Decode Latents**, plus Preview Image. Users can build it by dragging typed connections; add their own Python nodes; change model, task, settings, or compatible components; and retain their work. Family-specific behavior lives behind generic stage interfaces. Some architectures genuinely require additional conditioning or route stages: expose those honestly, without making every user rebuild a family-specific graph.

Scope includes advertised image generation, editing, reconstruction, upscaling, image conditioning and adapters, image analysis/depth, and image-consuming caption/vision tasks. The latter may use other reviewed official Hugging Face libraries and need honest task-specific graphs, not a fictitious diffusion denoise stage. Pure audio/video/3D/text qualification is outside this handoff, but shared changes must pass their existing regression gates and must not cross-wire their components into image routes.

“No bugs” cannot be established universally. The handoff criterion is **no known unresolved defect in this agreed image scope**, with the model/task inventory, original reported defects, integrated journeys, runtime/resource behavior, and preservation gates all accounted for. A required model blocked by weights, license, runtime, or hardware remains a blocker; do not quietly remove it from the denominator. The agreed Windows hardware limitation is the explicit exception.

This document plans the work. No product implementation, model downloads, or new qualification campaign is claimed by writing it. Execute the implementation phases below as a coherent batch, then the consolidated testing/fix campaign requested by the user.

## 2. Historical audit: what went wrong

### 2.1 Documents and decisions reviewed

Read together, these plans describe several different milestones, not one completed universal-stage project:

| Earlier document                                                                                                         | What it actually established                                                                          | What it did not establish                                                        |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `generic-nodes-upstream-assets-completion-plan.md` (August)                                                              | Upstream inventory, generic surfaces, executable/equivalent/contract-only dispositions                | Equivalent generation output does not prove editable intermediate stages         |
| `unified-composite-node-implementation-plan-2026-09-01.md`                                                               | Shared Block representation, persistence and hierarchy                                                | Native stage integration for every model                                         |
| `exact-modular-diffusers-block-expansion-plan-2026-09-04.md` and `nested-modular-diffusers-migration-plan-2026-09-04.md` | Truthful upstream blocks and nested execution through one effective graph                             | A compact cross-family public stage contract                                     |
| `compact-modular-blocks-and-model-route-switch-plan-2026-09-04.md`                                                       | Compact presentation and same-family Qwen route switching                                             | Cross-family interchangeability; that scope was explicitly deferred              |
| `generic-diffusers-workbench-plan.md` (September 17 onward)                                                              | Small generic workbench, dynamic stage declarations, standard fallbacks, extensions and service tools | Mandatory stage parity across every advertised image model                       |
| `creator-developer-workspaces-plan.md` (September 20 onward)                                                             | Creator/Developer UX, migration, resources, custom nodes, planned image qualification                 | W1 inventory, W8 image campaign, W10 release and H1 hardware were not all closed |
| `image-demo.md` (September 22)                                                                                           | A bounded local demo checkpoint with representative native outputs                                    | All-model readiness or Windows Qwen memory qualification                         |
| Current `custom-nodes.md`, `workflow-authoring-ux.md`, `developer-setup.md`, `service-prototyping.md`                    | Existing authoring, installation and service behavior                                                 | Evidence that every documented journey works on the final combined revision      |

### 2.2 Concrete commit trail

Hashes below are repository-local; backend unless explicitly marked client.

| Commit / date                                        | Delivered decision                                                                       | Why it matters now                                                                                                                      |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `e9477bd`, August 15                                 | ERNIE Modular exports routed to the standard `ErnieImagePipeline` as equivalent coverage | Explicitly judged by public task/output coverage, not editable stage access. This is a MoDiff routing decision, not an upstream absence |
| `4cac48b`, September 17                              | Initial generic workbench plan                                                           | Allowed standard fallbacks and representative-family acceptance; did not require a per-advertised-model native-stage disposition        |
| `0f16764`, September 17                              | Generic Modular stage declarations from existing adapters                                | Generalized the interface to integrations already present; did not integrate all upstream families                                      |
| `b08c703`, September 17                              | Standard operations projected through existing task adapters                             | Increased catalog/model reach using whole pipelines, not new composable stages                                                          |
| `1ebe608` and `6e849c7`, September 17                | Canonical operations and M3 closure                                                      | Closed the scoped catalog milestone; equivalent/fallback routing remained                                                               |
| `3d89d99`, `1c9144f`; client `c08b02b`, September 17 | Connected starters and M4 closure                                                        | Representative graph construction was mistaken in later communication for broad model/stage readiness                                   |
| `897e27a`, `b2e8ec5`, September 17                   | Recompute separation and runtime reuse/retry work                                        | Real useful foundations; they still need final multi-model and edited-graph acceptance                                                  |
| `12fb46e`, `875176b`, `706aa12`, September 18        | Extensions, uv/service work, scoped Windows evidence                                     | Actual delivery, but not complete custom-stage portability or the 16/32 GB Qwen case                                                    |
| `594769b`, September 20                              | Creator/Developer plan                                                                   | Kept whole-pipeline fallbacks; exact W1 model/artifact ledger was still open while subsequent milestones progressed                     |
| `626cc4c`, September 21                              | W5 model/task/resource migration closure                                                 | Representative native results and migration tests, not all advertised models                                                            |
| `0134d5d`, September 22                              | W8 paused at operator request to proceed with W9                                         | Explicit reprioritization, not evidence of W8 completion. The recorded 176 runs over 16 paths left known image cases open               |
| `9cc1ba1`, September 22                              | Qwen 2.1 reviewed runtime integration                                                    | A documented standard-pipeline exception, not native Modular parity                                                                     |
| `4b4a430`, `1710efe`, September 22                   | Published image demo checkpoint                                                          | Narrow demo evidence; document explicitly says W8/W9/W10/H1 remain open                                                                 |
| `59bc342`; client `6e93329`, September 23            | Task-first authoring                                                                     | Better entry and model-selection UX; not a substitute for missing native profiles                                                       |

Uncommitted subsequent fixes have no committed release provenance yet. Review them as existing work, not as automatically accepted behavior.

### 2.3 Root causes and their corrective gates

1. **Two meanings of support were conflated.** “Can generate this task” and “can edit/reuse each stage” are different. The older equivalence table survived a change in product goal. Gate: separate execution and stage coverage in every inventory row; native availability cannot close as equivalent whole-pipeline output.
2. **The denominator was not frozen.** Exported pipeline classes, starter variants, repositories, downloaded snapshots, tasks and public selectable profiles were counted interchangeably. Gate: authoritative union of advertised artifact/profile/task routes before implementation closure.
3. **Earlier plans were locally reasonable but insufficient for this acceptance target.** Same-family switching, existing-adapter generalization and standard fallbacks were explicitly in scope. The missing step was revisiting those boundaries when the promise became cross-family prototyping across supported models.
4. **Proof was promoted beyond its level.** Thousands of migration/schema checks demonstrate transformations, not valid tensors, correct consumed settings, stage caching or UI usability. Gate: separately tracked static, automated-contract, browser, native-execution and resource evidence.
5. **Compatibility checked nominal types more thoroughly than semantic contracts.** A shared port label or `pipeline`/`model` type does not guarantee unconditional generation, a valid scheduler class, an image VAE, or matching latent layout. Gate: shared, versioned capability and role constraints.
6. **Work packages closed without a final integrated-user-journey gate.** Prompt widgets, dynamic migration, preview history and autosave/tab behavior interact across components. Gate: original screenshots and continuous authoring/reload/model-change journeys on the final bundle.
7. **Communication overstated completion.** Earlier “all nodes”/“fixed” language should have been bounded to declared constraints and tested paths. Report open rows and limitations even when catalog or unit gates pass.
8. **ERNIE should have been reopened during the September workbench planning.** Its exclusion was discoverable in `modular_contract_only_registry.py`; reviewing only existing executable adapters inherited the earlier decision. Gate: independently compare upstream native exports/workflows against executable native routes, including explicitly equivalent entries.

Do not respond by creating another executor, replacing all working code, or accumulating more representative pass counts. Fix the missing acceptance boundaries and the concrete integrations.

## 3. Already delivered: preserve and finish, do not restart

| Existing work                                                                                   | Current evidence/status                                                                                       | Remaining acceptance                                                                                |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Generic stage nodes, backend dynamic declarations, Block persistence and shared graph execution | Implemented and exercised on representative native families                                                   | Complete per-profile native stage coverage and unsupported-feature boundaries                       |
| Model/task migration and retained-node machinery                                                | Historical representative checks; current working-tree improvements for readable impacts and topology changes | Provenance-based values, correct semantic mapping, round trips, real cross-family execution         |
| Typed drag suggestions and direct-connection checks                                             | Implemented; current signal/capability refinements                                                            | All image node contracts, unknown/custom behavior, imported graphs, parameter-dependent constraints |
| Custom source staging, approval, enable/reload, example packages                                | Implemented; real custom-node evidence exists                                                                 | Fresh end-to-end source creation and edited-code journey in a native image graph                    |
| Reuse, offload/recompute controls, cancellation and retry                                       | Historical native checks and resource tests                                                                   | Final-revision cache matrix, no duplicate model ownership, honest hardware limits                   |
| uv-based developer setup, npm frontend setup and service export                                 | Existing documentation and scoped setup/service checks                                                        | Clean environment instructions and real image service after model/custom-node changes               |
| Checkbox spacing, compact topbar, picker footer, split search sections                          | Existing UI changes, some in working tree                                                                     | Viewport/keyboard/long-content acceptance with final combined data                                  |
| Absolute dirty marker; separation of history lightbox from latest preview                       | Existing working-tree fixes                                                                                   | Autosave cycles and historical-image/new-run event race acceptance                                  |
| Resizable PromptTools example and multiline field metadata                                      | Existing example/documentation changes                                                                        | Enabled installed copy, reload, existing saved instance migration and actual prompt consumption     |

The previously reported 255 starter schemas, 5,024 same-task migration combinations, focused test results and mocked UI cases are useful historical evidence, **not** the release gate. Do not add overlapping pass counts or describe schema coverage as GPU/model qualification.

## 4. Authoritative image coverage ledger

### 4.1 Freeze the denominator first

Extend the existing qualification/inventory tools and W8 receipts; do not invent a second catalog or execution model.

1. Collect the union of selectable image routes from model capabilities, operation contracts/starters, execution profiles, templates, task starters, node-model dropdowns, custom first-party examples and user-facing supported-model documentation.
2. Join by immutable repository/revision, implementation/profile, task and relevant adapter/component variant. Preserve distinct public routes even when weights are shared.
3. Distinguish advertised runnable routes from explicitly contract-only inventory. Experimental entries exposed as runnable remain required, with their experimental status retained.
4. Cross-check against the pinned upstream exports and actual block workflow contracts independently of MoDiff's current adapter list.
5. Record omissions, duplicate aliases, invalid “Downloaded” badges, missing optional runtime declarations and selectable-but-not-runnable entries as defects. Do not relabel/remove them merely to pass this release.
6. Freeze a snapshot/hash for the campaign. Any subsequent advertisement/runtime/profile change reopens affected rows and the aggregate gate.

Required ledger columns:

- Stable route ID, display name, repository and revision, task, implementation, advertisement entry points.
- Native upstream class/workflow/blocks and source revision; full-stage-chain availability, not merely one block present.
- MoDiff adapter/profile/action bindings, public stage roles and extra required conditioning.
- Disposition: native integrated / native integration missing / documented whole-pipeline exception / non-diffusion image task / blocked.
- Exact exception reason and upstream evidence; whether a standard alternative and native alternative coexist.
- Weights/component inventory, bytes, gated access/license approval, optional runtime/dependency profile and approved source hashes.
- Required inputs and controls, scheduler/guider/VAE constraints, dimensions/latent layout and output cardinality.
- Supported task transitions, preservation mappings, cache dependencies and resource policy.
- Structural, browser, native output, parameter-consumption, reuse, service and platform evidence separately.
- Known defects/blocker, accountable work package, receipt location, tested backend/client/bundle/runtime identities.

Completion is all required rows accepted, with no unclassified native export, unresolved advertised-route omission, or undocumented exception.

### 4.2 Audit snapshot: do not confuse these counts with the denominator

The current generated starter fixture contains **255 all-task variants**. Its text-to-image slice has **52 variants: 15 staged and 37 whole-pipeline variants**. These are not 52 distinct models, and not all advertised artifact profiles occur in this fixture.

The pinned contracts include these 19 image-capable native Modular classes (including image branches of multi-modality classes):

| Upstream native classes                                                                                                           | Required disposition work                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AnimaModularPipeline`                                                                                                            | Bind and qualify each advertised artifact/task, not just the exported class                                                                           |
| `Cosmos3DistilledModularPipeline`, `Cosmos3OmniModularPipeline`                                                                   | Audit image branch explicitly; whole-workflow discovery is not split-stage proof                                                                      |
| `FluxModularPipeline`, `FluxKontextModularPipeline`                                                                               | Audit dev/Schnell/Krea and edit/reference profiles individually; family resemblance is not qualification                                              |
| `Flux2ModularPipeline`, `Flux2KleinModularPipeline`, `Flux2KleinBaseModularPipeline`                                              | Separate base/distilled/size/quantized/KV/task variants; no inherited compatibility claims                                                            |
| `Ideogram4ModularPipeline`, `Krea2ModularPipeline`, `Krea2TurboModularPipeline`                                                   | Confirm advertised/profile status and implement every required exposed native route                                                                   |
| `QwenImageModularPipeline`, `QwenImageEditModularPipeline`, `QwenImageEditPlusModularPipeline`, `QwenImageLayeredModularPipeline` | Text/edit/multiple-reference/layered contracts separately; do not alias Qwen 2.1 into them                                                            |
| `StableDiffusion3ModularPipeline`, `StableDiffusionXLModularPipeline`                                                             | Per-artifact/task/adapter support; base availability does not prove PAG/ControlNet/Turbo compatibility                                                |
| `ZImageModularPipeline`                                                                                                           | Text and supported image/inpaint branches; scheduler/guider restrictions before execution                                                             |
| `ErnieImageModularPipeline`                                                                                                       | **Known integration gap:** replace equivalent-only disposition with real native stages for usable advertised profiles; verify ERNIE/Turbo differences |

The whole-pipeline text-to-image fixture contains the following classes. Each needs an exact profile/task disposition; this list is a starting audit, not permission to exempt an entire family:

- Native-family alternatives may exist: `ErnieImagePipeline`, `FluxPipeline`, `Flux2Pipeline`, `Flux2KleinPipeline`, `Flux2KleinKVPipeline`, `QwenImagePipeline`, `StableDiffusionXLPipeline`, `StableDiffusionXLTurboPipeline`, `StableDiffusionXLPAGPipeline`, `ZImagePipeline`. Verify exact variant support; do not presume PAG, KV or Turbo support from a base class.
- No matching native family identified in the pinned Modular export inventory: `AuraFlowPipeline`, `ChromaPipeline`, `CogView3PlusPipeline`, `CogView4Pipeline`, `DreamLitePipeline`, `DreamLiteMobilePipeline`, `GlmImagePipeline`, `HunyuanDiTPipeline`, `HunyuanDiTPAGPipeline`, `JoyImageEditPipeline`, `Kandinsky3Pipeline`, `LatentConsistencyModelPipeline`, `LongCatImagePipeline`, `LuminaPipeline`, `Lumina2Pipeline`, `NucleusMoEImagePipeline`, `OmniGenPipeline`, `OvisImagePipeline`, `PRXPipeline`, `PixArtSigmaPipeline`, `PixArtSigmaPAGPipeline`, `QwenImage21Pipeline`, `SanaPipeline`, `SanaPAGPipeline`, `SanaSprintPipeline`, `StableDiffusionPipeline`, `StableDiffusionPAGPipeline`.

Thus, at this pinned snapshot, upstream absence is **not limited to GLM and Qwen 2.1**. Conversely, many standard entries are alternatives to native implementations, and MoDiff cannot attribute those missing integrations to upstream.

Image routes outside this slice must also enter the ledger: control/inpaint/edit variants, image adapters and LoRAs, reconstruction, unconditional DDPM/DDIM/consistency, upscalers/Spandrel, Marigold/Depth Anything, image caption/vision/multimodal routes, and the first-party custom image examples. Enumerate the actual current catalog rather than deriving these from text-to-image counts.

### 4.3 Exception policy

For each whole-pipeline exception record:

1. Exact model/revision/task and checked Diffusers revision.
2. Missing or unusable native workflow, with source evidence. Missing MoDiff adapter alone is **not** an upstream exception.
3. Why the standard adapter is correct; supported inputs/settings and actual execution evidence.
4. Which stage editing, component substitution and cross-stage caching are unavailable.
5. Model-switch impact, values that migrate, values retained inactive, and how to return.
6. Revisit trigger: reviewed dependency update or new native implementation.

GLM's pipeline has meaningful internal boundaries (prior-token generation, text encoding, denoising, decode), but no native GLM Modular export at this pin. A hand-written adapter is possible future work, not a requirement to pretend it already has native stages. Prior-token generation also has its own conditioning/seed/cache dependencies. Qwen 2.1 is similarly a documented exception, not the same implementation as Qwen Image 2512.

Do not wrap a single whole-pipeline call in four cosmetic nodes. Do not upgrade Diffusers simply to change the inventory without an explicit reviewed pin migration and regression scope.

## 5. Implementation architecture and invariants

### 5.1 One backend-owned stage and compatibility contract

Extend existing operation contracts, stage bindings and schemas. Keep one executor and one model/resource owner.

Every stage/port declares as applicable:

- Stable semantic operation and field IDs, independent of display labels and Python class names.
- Data kind/cardinality plus role (image VAE, text encoders, denoiser, scheduler, guider, prompt bundle, route state, image/latent).
- Required capabilities and declared provided capabilities.
- Model/route lineage, latent layout/channel/scaling expectations, conditioning schema, image dimensionality and supported value constraints.
- Runtime-resolved/unknown fields explicitly, with dependency provenance.
- Authored/default/derived value provenance and any reviewed migration/conversion rules.
- Versioning for saved graph/schema compatibility.

Reuse existing nominal types and explicit aliases; do not collapse all tensors, models or embeddings into one universal type. Names such as “Pipeline”, “Model” and “VAE” are labels, not proof.

Connection assessment has three outcomes:

1. **Compatible under declared constraints:** offer and allow; still validate actual runtime data.
2. **Known incompatible:** exclude from automatic suggestions or show a clearly disabled reason; reject direct connections with a readable explanation.
3. **Unresolved until model/value/code is known:** mark as unverified, not guaranteed to work. Require resolution/preflight before Ready when required constraints are knowable. Runtime validation remains mandatory.

Use the same contract for drag discovery, direct connection, auto-fix, model/task reconciliation, imported/saved graphs, Block boundary ports, custom nodes, API execution and Run preflight. Parameter/model changes must invalidate compatibility/readiness even if no wire changes.

Do not infer semantic compatibility from matching field names or labels. Two scheduler-typed ports can require different classes; two tensor ports can carry incompatible latent formats.

### 5.2 Integrate native stages, keeping a small public graph

For each required native profile:

1. Map exact upstream workflow/sub-blocks to existing generic stage roles and component ownership.
2. Define conditioning and route-state dependencies explicitly. Do not hide mutable state in globals or untracked pipeline references.
3. Bind the upstream block execution through the existing adapters/manager. Use upstream implementations, not copied denoising loops.
4. Keep Load / Prompt / Denoise / Decode stable when those semantic roles exist. Additional image encoding, conditioning, prior or routing steps may be required and must be explainable/expandable.
5. Expose advanced exact blocks through current expansion/Developer mechanisms without forcing them into the default authoring surface.
6. Ensure a complete executable stage chain and exact artifact profile before labelling a route “Editable stages”. Current metadata detecting any block/bundle is insufficient.
7. Default new model selections to the qualified native option when available. Preserve an explicitly saved whole-pipeline choice; offer a reviewed conversion instead of silently changing existing graphs.
8. Integrate ERNIE early, then audit native variants and all other native rows; do not stop after ERNIE or the familiar Flux/Qwen families.

Mellon is a design reference for generic public nodes, backend-selected configurations and typed suggestions. Its Modular configuration also resolves registered pipelines and named upstream blocks; it is not evidence that arbitrary Diffusers models are interchangeable. Adopt the approach, not its implementation or support claims wholesale.

### 5.3 Model/task changes: transactional, semantic and lossless

Use the existing graph reconciliation transaction, with explicit preview/apply/cancel and undo.

**Identity and topology**

- Match stable semantic stage roles and lineage, not current labels or action IDs. Keep IDs/layout where the same role remains; a changed backend implementation does not inherently require replacing that node.
- Distinguish duplicate stages by ownership and role instance, not first label match.
- Limit automated edits to the selected operation's owned graph. Preserve custom/user-added nodes, branches, external references, manual wire overrides and unrelated loaders.
- Track actual per-instance managed wires. Do not reconstruct ownership solely from a pristine canonical starter.
- Add/remove required owned stages only when architecture/task changes demand it. Keep displaced user-bearing stages inactive with an explicit restore path; avoid accumulating invisible runnable orphans.
- Revalidate every retained edge against the target contracts, including route state and component lineage.
- Apply model, immutable revision, implementation/profile, task, fields and connections atomically; cancel and failed/stale apply leave the graph unchanged.

**Value precedence and preservation**

1. Preserve compatible user-authored values and external connections.
2. Apply explicit semantic conversion rules where units/meaning differ, with review where needed.
3. Use target model defaults only for fields never authored, new fields, or an explicit “Restore model defaults”.
4. Keep unsupported authored values in a persisted, model/task-scoped inactive draft and explain why they are inactive.
5. Restore those drafts when switching back, unless the user deliberately overrides them. Include them in save/load, undo/redo and workflow export.
6. Record authored provenance even when the user chose the previous default value; equality-to-default cannot prove “not edited”.
7. Never silently clamp dimensions/steps or map controls solely because both are named “guidance”. CFG, true CFG and distilled guidance are distinct.
8. Preserve compatible prompt/negative prompt, seed mode/value, image/mask/reference order, width/height, batch/cardinality, steps, strengths, meaningful guidance, scheduler settings, resource controls, export settings, node size/layout and custom code parameters.
9. Model repository/revision/class and model-specific derived defaults change together. Retaining an incompatible revision from the old repository is a defect.
10. When multiple prior fields compete for one target field, show the conflict; do not silently choose one.

**Native ↔ whole-pipeline changes**

A native-to-standard change is a genuine topology change: one Generate action may replace Encode/Denoise/Decode. That does not mean all values or all surrounding nodes must change. Map prompt/custom prompt output to the generator, map generation controls semantically, retain preview/export and image input branches, and preserve unsupported intermediate custom edits in inactive drafts. Explain precisely why an intermediate custom embedding/latent transform cannot remain executable.

The reverse transition restores native roles and compatible drafts. Neither direction may discard a custom prompt processor merely because its old consumer disappeared.

**Readable impact review**

- Show counts separately: retained/updated/added/inactivated nodes, disconnected/reconnected links, settings retained/inactive/changed. “12 changes” must not be presented as “12 nodes”.
- Use node titles and port labels, e.g. “Denoise · Route state → Decode Latents · Route state”. Add readable duplicate disambiguation and a focus-node action.
- Explain the specific reason: removed stage, unsupported task, changed conditioning schema, invalid scheduler class, or incompatible latent format.
- Keep opaque IDs in optional technical details only.
- Show important retained user values, not just destructive changes.
- Review and Apply stay in the bottom action area, with a bounded independently scrollable detail region so neither model choices nor buttons disappear.
- Recompute the preview if graph or catalog changes while the modal is open; never apply a stale diff.

### 5.4 Task changes should reuse the graph where genuinely supported

Adding an image can select a declared image-to-image/edit route only when the selected profile supports it. Offer the necessary Encode Image/reference/mask controls and explain missing inputs. Do not infer support from one image-typed optional port.

Cover text-to-image ↔ image-to-image, inpaint/mask, ordered multiple references, supported ControlNet/IP-Adapter/LoRA compositions and layered outputs. Preserve shared controls and source images across supported changes; explicitly retain incompatible task-specific settings inactive. Removing the image must not silently keep an image-only route runnable.

### 5.5 Runtime reuse and memory

- One manager owns loaded components and their revision/dtype/quantization/device/adapter identity; stages borrow them without independently loading duplicate models.
- Cache keys include actual consumed inputs, model/component and adapter identities, code/schema revision, tensor/array mutation version and relevant execution settings.
- Unchanged run: reuse unchanged results without loading/encoding again.
- Seed/steps/guidance change: rerun required generation/downstream work, not prompt encoding unless the route truly consumes that value while conditioning.
- Prompt/negative prompt/reference/image changes: invalidate exactly their dependants.
- Model, LoRA, tokenizer/text encoder, clip settings or conditioning implementation changes: invalidate dependent embeddings and generations correctly.
- Native prior/conditioning branches may depend on seed; do not impose Flux's dependency matrix on GLM or other architectures.
- Distinguish output cache, prompt cache, native within-generation KV reuse, component residency, CPU offload and explicit release. Standard exceptions may not permit independent prompt-stage reuse; state that limitation.
- Auto policy may offload for memory, but routine completion must not silently destroy reusable component ownership. A deliberate release/pressure policy must be visible and explained.
- Cancellation, OOM retry, partial failure, queue changes and workflow switching must leave the manager coherent and avoid treating incomplete outputs as reusable successes.
- Prove behavior using load/encode/denoise/decode counters and resource traces, not elapsed time alone.

## 6. Every reported issue: concrete treatment

| ID  | Report / feedback                                         | Required behavior and regression acceptance                                                                                                                                                                                                                                                                       |
| --- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U01 | Checkbox text/toggles run together                        | Explicit label/control and sibling gaps, wrapping on narrow widths, clickable labels and keyboard focus                                                                                                                                                                                                           |
| U02 | ACE audio models offered for image VAE reconstruction     | Dropdown filters exact loader role, class, task/modality and declared compatibility; unknown local/custom metadata is unverified, not “compatible”. Same class alone must not certify audio/video/image or latent geometry                                                                                        |
| U03 | FLUX pipeline wired to unconditional generation           | Capability-aware suggestion/direct/preflight rejection naming both nodes and required action; imported old invalid graphs get actionable issues                                                                                                                                                                   |
| U04 | Scheduler accepted then rejected by native pipeline       | Validate selected scheduler class/config against connected route before Run; parameter changes revalidate. Do not merely compare `scheduler` types                                                                                                                                                                |
| U05 | Generic Model connected to VAE / other coarse model ports | Required component-role constraints across loaders, encoders, denoisers, adapters and decode. Audit embeddings, latents, route state, control data, mask/image cardinality and dimensions too                                                                                                                     |
| U06 | “All suggested nodes will work?”                          | Distinguish declared compatibility from runtime-dependent unknowns; compatible port is not a promise that all other required inputs/weights are present                                                                                                                                                           |
| U07 | Suggestions not ranked by context                         | Image output → Preview Image first; deterministic semantic/task/direction ranking thereafter. Search relevance remains effective. Rank only after compatibility filtering, in both drag directions                                                                                                                |
| U08 | Custom section hidden below expanded built-ins            | Fixed visible section headers and bounded independent result scroll regions; flexible height for the sole expanded section while reserving other headers; empty/search states, both collapsed, keyboard navigation, small viewport and overflow acceptance                                                        |
| U09 | Workflows look like different node systems                | Explain native editable stages versus genuine whole-pipeline exceptions and exact-block expansion; consistent task/role titles, one underlying executor. Never call Generate Image a decomposed denoiser                                                                                                          |
| U10 | Model/Pipeline Type contradicts selected model            | Canonical model choice infers implementation/class. No second independent field on curated loaders. Raw/custom loading may need an explicit advanced override only if inference is ambiguous; validate model/class/revision as a unit                                                                             |
| U11 | Model picker alignment and premature review               | Align repository/status/implementation rows; nest alternatives inside their model card; selection list above bottom review; short labels with accessible explanations                                                                                                                                             |
| U12 | Crowded topbar                                            | No redundant Workspace label; compact Creator/Developer switch, accessible memory toggle, detailed policy/status on hover/focus; info action in right icon group. Do not silently destroy Expert/custom resource settings when toggling                                                                           |
| U13 | Custom node hard to find / wrong demo workflow            | Nodes → Custom nodes searchable section and exact example name; explicit new named native text-to-image workflow, originals unchanged; show Text Value → Prompt Prefix → Encode Prompt                                                                                                                            |
| U14 | Custom Prefix single-line, no resize, missing prompts     | Resizable generic node; textarea metadata for text/prefix; explicit prompt input socket and optional inline fallback with documented precedence. A connection supplies the prompt; blank unused fallback is not missing runtime text. Show connected-source state/effective preview without stale duplicated text |
| U15 | Tab asterisk causes horizontal movement                   | Reserved marker space or absolute positioning; tab text width, close button and selected-tab scroll position stable through repeated dirty/autosave/run cycles                                                                                                                                                    |
| U16 | Historical preview replaces/latest image disappears       | Latest completed result, gallery history, export selection and temporary lightbox selection have separate stable identities. Opening/closing old output never rewrites latest/history; completing a new run while a modal is open also safe                                                                       |
| U17 | Hashes and “12 changes” incomprehensible                  | Human node/port names, reason/action per change, category counts and focus links; retain exact IDs only in details. No assertion that all nodes change when only links differ                                                                                                                                     |
| U18 | Custom port types/docs/e2e insufficient                   | Maintained schema-backed type/role reference, explicit aliases and unknown limitations, actual create/stage/enable/reload/run e2e, compatible and incompatible custom examples                                                                                                                                    |
| U19 | uv/npm install without shell scripts                      | README discovery and relative-path clean-install instructions using reviewed managed runtime; Windows/POSIX commands and npm flow; no need to inspect/run a premade .sh/.ps1 to start                                                                                                                             |
| U20 | Small graph, cross-family prototyping and services        | Native stage coverage + transactional retention + actual service export/run from the edited graph. No hidden subgraph required to reach the four core roles                                                                                                                                                       |
| U21 | Re-loading, re-encoding, DiT unloading/crash              | Dependency-aware cache/resource acceptance in section 5.5, including changed/unchanged and memory-pressure cases; Windows Qwen limit disclosed                                                                                                                                                                    |
| U22 | Most model choices say whole-pipeline                     | Per-profile native inventory, complete integrations where usable, explicit upstream exceptions elsewhere; separate badge from download state and task support                                                                                                                                                     |

For U14, maintain one clear authoring contract: inline widget values are fallbacks when the relevant socket is unconnected; a connected prompt takes precedence. The example should teach this, and the renderer should make it visible. Do not keep two editable prompts that appear equally active.

For U16, also verify batched images, full-resolution URLs, persistent history after save/reopen, navigating between workflows, old results after node deletion/recreation, cache-hit runs, lightbox next/previous, clear/delete actions and delayed events from a prior run. Do not solve it by hiding old outputs.

## 7. Custom-node and documentation work

Keep `examples/custom_nodes/PromptTools` and `docs/custom-nodes.md` as the reusable source, not a private absolute-path recipe.

The guide and example must cover:

1. Relative source directory creation; `main.py`, `__init__.py` and `modiff_extension.json`; NodeBase execution signature and output dict keys.
2. Field metadata: scalar data type versus widget presentation, socket direction, defaults, required/optional values, textarea, resizing, output cardinality and inline-versus-connected precedence.
3. Supported built-in nominal types/aliases and semantic roles. Types are not arbitrary matching display labels. Link one maintained backend-owned reference; add contract checks keeping its examples synchronized with runtime schemas.
4. Safe custom type namespaces and constraints; what an undeclared/unknown contract cannot guarantee. No promise that arbitrary custom Python works with every model.
5. Stage source → inspect files/hash/dependencies/ports/runtime role → explicit code consent → Enable. Staging/importing a workflow must not execute unapproved source.
6. Find the installed node through the separate searchable Custom section; create a model-free smoke graph and a new native image workflow.
7. In the image workflow, Text Value → Prompt Prefix → Encode Prompt; no hidden unchanged prompt winning over the new connection.
8. Review reload of the installed path (clearly distinguished from the original copied source); new hash requires consent; dependency/schema changes require revalidation and a safe saved-instance migration story.
9. Errors, failed imports, disabled/missing extensions, renamed ports, runtime input validation, mutating inputs, declared resource ownership, unsafe file/network access and recovery.
10. A simple image transform and, where declared compatible, a custom component-consuming example showing real image/VAE interoperability. Do not imply a text-only example proves cross-family tensor portability.
11. Export/service requirements, exact extension version/hash availability, and why importing a workflow does not approve code.

Automated e2e must create a temporary package from documented content, stage it through the actual API/UI, verify no premature import, consent/enable, insert through drag/search, execute, edit/review/reload and execute changed output. Include refusal/stale approval, import failure, existing-instance schema changes, missing dependency, incompatible port and restart/reopen cases. A native image run then proves the prefixed prompt is actually consumed.

README/setup docs should describe the supported uv-managed developer entry, not invent `uv sync` if that is not the executable contract. Existing commands use `uv run --no-project --no-sync --python 3.12 -m modiff.dev` with plan/setup/check/run actions and frontend `npm ci` / `npm run dev`. Verify exact prerequisites, working directories, optional runtime consent, NVIDIA/AMD/CPU paths, restart/update and troubleshooting against the current implementation before publishing.

Keep hardware limitations prominent. Installation success on Windows is not Qwen inference qualification.

## 8. Work packages and implementation order

All packages use existing ownership boundaries. The file names below are entry points, not permission to scatter new parallel subsystems.

| Package                       | Dependencies                  | Implementation target                                                                                                  | Main entry points                                                                                                                                   |
| ----------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 Coverage freeze            | Agreed scope                  | Exact advertised route ledger; upstream/profile cross-check; per-exception record; source/dirty-worktree baseline      | operation catalog/contracts/starters, execution profiles, existing W8 qualification tools                                                           |
| P1 Shared contracts           | P0                            | Semantic roles, capabilities, provenance and unresolved state; backend validation and frontend consumption             | `modiff/operation_contracts.py`, `operation_catalog.py`, `modules/ModularDiffusers/pipeline_schema.py`, client `nodeConnectionMatching.ts`          |
| P2 Native integration         | P0–P1                         | ERNIE and every missing usable native image profile/task; native-first new selection; honest exceptions                | `modular_contract_only_registry.py`, `modular_task_adapters.py`, `modular_action_bindings.py`, ModularDiffusers stage modules, `workflowChoices.ts` |
| P3 Reconciliation             | P1–P2                         | Stable roles, semantic mapping, authored provenance, inactive drafts, connections and atomic readable review           | `operationAuthoring.ts`, graph transactions, task/route reconciliation, `OperationModelPicker.tsx`                                                  |
| P4 Discovery and readiness    | P1–P3                         | Shared matching across all paths, scheduler/model-role constraints, custom declarations, actionable errors             | `nodeConnectionMatching.ts`, `nodeConnectionSearch.ts`, `useWorkflowConnections.ts`, flow mutations and backend preflight                           |
| P5 Runtime/resources          | P1–P3                         | Correct stage dependencies and ownership for all new profiles; repeat/recompute/offload recovery                       | `NodeBase.py`, `node_cache_identity.py`, ModularDiffusers runtime/utilities, existing resource manager                                              |
| P6 UX completion              | P3–P4                         | U01–U17 remaining details; integrate/preserve existing fixes                                                           | NodeSearchDialog, model picker, TopBar, NodeContent, WorkflowTabsBar, UIImageField/history/lightbox                                                 |
| P7 Custom/dev/service/docs    | P1–P6                         | Runnable examples, installed-copy reload, maintained type guide, relative uv/npm instructions and edited-graph service | extension boundary, examples, custom-node/setup/service docs and existing export path                                                               |
| P8 Consolidated qualification | P0–P7 implementation complete | Full campaign below; grouped diagnosis/fixes; rerun affected and final gates                                           | existing backend/client/Playwright/native qualification tooling                                                                                     |
| P9 Demo and handoff           | P8 passes                     | Continuous rehearsed demo, new named workflows, limitation sheet, source/bundle proof and commit-ready review          | docs/image-demo.md, sanitized recipes, local receipts                                                                                               |

During P0–P7, inspect call paths, collect original reproductions and implement the associated regression cases alongside code. Do not repeatedly run large suites or a model campaign after every small edit. Once the integrated implementation is ready, run P8 in layers. Fix failures by root cause and rerun focused affected cases, then the final full gates. This follows the user's requested plan → implementation → testing/fixes order without treating untested code as complete.

If P0 reveals a materially new product choice, ask a specific question and wait. Technical effort alone is not a reason to narrow the agreed coverage. If an upstream/native incompatibility is discovered, document the exact evidence; do not silently convert an implementation gap into an approved exception.

## 9. Consolidated qualification campaign

### 9.1 Resource and download preparation

- Resolve all required repositories/revisions/components and missing byte counts; reuse verified existing caches.
- Separate download presence, completeness, permission, optional dependency readiness and actual model compatibility.
- Plan disk headroom, peak RAM/VRAM and per-profile resource policy before starting downloads/runs. Never evict user models or outputs to make space without a specific decision.
- Additional downloads are in scope, but license acceptance, gated-account access and remote-code approval remain explicit operator boundaries. Record required action instead of bypassing them.
- Run GPU qualification serially or under an explicitly measured resource budget; do not overlap expensive browser/model campaigns and then infer performance.
- Preserve existing workflow files, approvals, model snapshot metadata and output history. Receipts and generated images remain in ignored local review storage, not public Git assets.

### 9.2 Layer A — contracts and inventory

- Every advertised route resolves to an exact executable native chain, documented standard exception, or non-diffusion task with its correct graph.
- Fail on missing native integration; do not permit a test to pass because all candidates were classified incompatible.
- Round-trip contracts, saved graphs, Block boundaries, legacy aliases and custom schema declarations.
- Connection matrix covers role/capability/shape/cardinality/model lineage, both directions and all creation/import/migration/preflight paths.
- Model changes preserve field provenance and managed/user-owned wires; cancel/stale preview/undo/reopen restore exact values.
- Catalog/model dropdown filters include positive and negative cases for image/audio/video VAEs, wrong generic model components and scheduler classes.
- Validate documentation/example schema consistency without treating those checks as model execution.

### 9.3 Layer B — integrated browser journeys

Use the real rendered UI where feasible, with mocked tests reserved for deterministic races/errors/layout. Include:

- Empty-canvas construction through port dragging, both directions; intended preview ranking; searchable custom nodes; all section headers stay reachable at narrow viewport/zoom/long result list.
- Model picker long names, alternatives, large impact review, bottom action visibility, keyboard navigation and no raw IDs in ordinary messages.
- Model changes with edited/default-equal prompts, settings, external custom wires, renamed nodes, duplicate roles and user replacement components.
- Pending model-change modal while graph changes; cancel/apply/undo/redo; save/reopen and return to old model.
- U14 prompt consumption/resizing, U15 repeated dirty/autosave cycles and U16 complete historical-preview event sequence.
- Workspace/resource toggle persistence, optional install/code approval boundary and useful Ready/error states.
- A fresh documented custom package lifecycle, not only a seeded fixture graph.

### 9.4 Layer C — every advertised model/task native execution

Every required ledger row needs a real baseline output at its pinned artifact revision on the supported qualification machine. Also exercise each distinct advertised task/control contract; an inherited base-family result cannot certify a different adapter or architecture.

For generation/editing routes record:

| Case                             | Required observation                                                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Cold baseline                    | Correct components/route loaded; valid expected media and dimensions; no extra duplicate loaders      |
| Unchanged repeat                 | Expected output/stage reuse and load/encode counters; no unwanted release/reload                      |
| Prompt edit                      | Changed prompt is consumed; appropriate conditioning/generation recompute                             |
| Seed-only edit                   | Correct seed consumed; prompt cache retained where semantically valid                                 |
| Steps/guidance edit              | Actual backend values match controls; distinguish unavailable/distilled/true guidance                 |
| Size/batch/reference edit        | Supported non-square/bounds/cardinality handling; honest preflight errors; retained compatible inputs |
| Save/reopen and workspace switch | Same graph, values, implementation, code requirements and result ownership                            |
| Recompute/offload/release        | Distinct intended behavior, correct residency/cache invalidation and recovery                         |
| Task-specific branches           | Every advertised image/mask/control/reference/adapter/layered route has its own valid output          |
| Failure/cancel/retry             | No stale success, lost latest preview, leaked ownership or poisoned cache                             |

For analysis/upscale/vision tasks use equivalent task-appropriate variations and output assertions, not meaningless seed/denoise cases. Mark N/A with a reason, never count it as a pass.

Do not require all 5,024 structural migration pairs to run on a GPU. Native transition coverage must instead cover every distinct source/target semantic-contract change, every implementation topology change, each target artifact's real execution, and specific regression pairs. Include native↔native across families, native↔standard, standard↔standard, and round trips with custom prompt/image nodes and edited settings. A coverage mapping must show why any representative transition covers other identical contracts.

Validate outputs visually and structurally: finite tensors where inspectable, correct channels/size/count, no black/transparent/empty corruption, expected task behavior and broad prompt/edit adherence. Investigate ambiguity against upstream execution with the same settings; one aesthetically weak sample is not automatically an adapter defect, but valid file bytes alone do not prove correct generation.

### 9.5 Layer D — resources, services, setup and complete checks

- Instrument cache/load/encode counts and peak RAM/VRAM on native families and every distinct standard adapter behavior.
- Execute a real generated service package after custom-node and model/task changes; prompt/seed/file inputs and named image outputs must agree with the UI. Verify ordered multi-file references and code/runtime requirements.
- Clean uv/npm setup check using documented paths. Reuse historical platform evidence only with its exact version/scope; qualify changed installer paths again.
- Run backend gates from CONTRIBUTING, reviewed optional-runtime gates for required profiles, and client `npm run check`, `npm run check:ui`, `npm run check:acceptance` as applicable to the integrated candidate. Report skipped hardware/live cases separately.
- Verify the served production bundle matches the checked client build and backend contract revision.
- After fixes, rerun affected native rows and full required integrated gates; do not finish on “isolated failures passed” if the final full gate remains unrun.
- Record tested source/bundle hashes. A later execution-affecting change invalidates the appropriate receipts. Documentation-only changes do not need repeating model inference.

Historical native evidence can guide recipes and avoid rediscovery. It is not automatically final-revision acceptance when stage bindings, migration or cache dependencies changed.

## 10. Continuous demo and manual-test workflow

Create a **new**, clearly named workflow such as “Image prototyping — custom nodes and model changes”. Do not replace existing user workflows. Supply a relative-path recipe and source package so Álvaro can reproduce it, not just open a private saved graph.

Rehearse this journey end to end on the release candidate:

1. Start with an empty canvas in Developer mode. Select a qualified native image model; add/wire the four core stages and Preview using drag suggestions.
2. Enter a distinctive prompt and set user-owned seed, dimensions and a supported generation control. Run and inspect a real image.
3. Repeat unchanged, then change only seed. Show instrumentation confirming model/prompt reuse where expected.
4. Create/stage/review/enable PromptTools from the documented example. Find **Prompt Prefix** in the left Custom section and insert it between Text Value and Encode Prompt. Run and show the effective changed prompt/result.
5. Resize/edit the multiline custom node. Modify its Python behavior, review the installed-source reload hash, approve and rerun; prove the new code is used.
6. Add a simple custom image processor between decode and preview. It should survive a compatible model change without special frontend code.
7. Change to another qualified native family in the same graph. Use a prequalified pair selected from Z-Image, FLUX.2 Klein, SDXL/Qwen as appropriate to the machine. Review what remains, changes and becomes inactive. Run successfully; do not hide incompatible guidance/conditioning changes.
8. Switch back and demonstrate restoration of authored values, layout and custom nodes. Include a value deliberately equal to the first model's old default.
9. Add an input image and change to a genuinely supported image-to-image/edit route. Explain any required extra encoding/mask/reference control; reuse the surrounding graph.
10. Optionally show GLM or Qwen 2.1 as a labelled whole-pipeline exception: preview actual topology changes, retain custom prompt/image nodes and supported settings, then return to native. Do not make this the main demonstration of editable stages.
11. Generate a new image, open it, browse an older historical result, close it and prove the latest image is still available. Generate while the lightbox is open and repeat.
12. Save/reopen, create/reinsert a Saved Block where appropriate, and verify values and extension requirements remain intact.
13. Export a service package from this edited graph and call it with a new prompt/seed (and image files for the edit route). Show a real resulting image and accurate named outputs.
14. Finish with the limitations sheet: upstream exceptions, Windows Qwen 16/32 GB not qualified, hardware/runtime used and any optional approvals the recipient must make.

Prepare missing weights before recording; do not hide setup requirements in the published instructions. Measure demo duration on the final machine rather than promising a duration without a rehearsal. Keep recovery copies of the new workflow, approved example source and dependency identities. Public artifacts must exclude local absolute paths, tokens, private prompts/images and unlicensed model/output assets.

## 11. Release gates and reporting discipline

The handoff is ready only when all of the following are true:

- [ ] P0 authoritative image inventory frozen and reconciled with every advertisement entry point.
- [ ] Every required usable native workflow integrated as real stages; ERNIE and native variant omissions closed.
- [ ] Every allowed standard exception individually documented, correctly labelled and executed.
- [ ] Every advertised image artifact/task row has the required native output/control evidence, including newly downloaded models.
- [ ] Every U01–U22 issue accepted on the final integrated app; no known in-scope defect hidden behind a pass count.
- [ ] Model/task round trips preserve authored values and custom/user-owned graph content; losses require explicit review, never silent discard.
- [ ] Cache/resource/cancellation/retry observations match the declared dependency contracts.
- [ ] Fresh custom-package e2e, native image use, reload and saved-instance behavior accepted.
- [ ] uv/npm and type/role docs are reproducible, maintained and linked from README/help.
- [ ] Real service export/inference from the edited graph accepted.
- [ ] Full final required automated/browser gates passed; skips/failures separately accounted for.
- [ ] Served source/bundle/runtime identities match receipts; existing user data/models/approvals preserved.
- [ ] Continuous demo rehearsed and reproducible from an empty workflow.
- [ ] Windows Qwen limitation and exact per-model exceptions visible in the handoff.
- [ ] Paired repository diffs reviewed, generated assets handled by existing build policy, and commit-ready changes separated from unrelated user work.

Do not claim completion from “M3/M4 done”, catalog counts, static compatibility checks or a handful of attractive images. Report each package as planned / implemented-awaiting-qualification / accepted / blocked, with proof level and exact remaining rows. A newly discovered in-scope defect reopens its gate.

The user will decide when to commit and contact Álvaro. This plan does not authorize automatic commits, publishing outputs, accepting licenses, or sending external messages.

## 12. Research anchors

These are evidence for the design/audit, not claims that mutable upstream main equals the installed runtime:

- [Pinned Diffusers Modular exports](https://github.com/huggingface/diffusers/blob/fbf49e7f35857f76bc57b177e26f12b03687c668/src/diffusers/modular_pipelines/__init__.py).
- [Pinned GLM pipeline implementation](https://github.com/huggingface/diffusers/blob/fbf49e7f35857f76bc57b177e26f12b03687c668/src/diffusers/pipelines/glm_image/pipeline_glm_image.py): prior tokens, prompt encoding, denoising and decode are internal operations, not a native Modular export.
- [Diffusers Modular overview](https://huggingface.co/docs/diffusers/main/modular_diffusers/overview): upstream composition model; verify concrete availability against the pinned snapshot.
- [Mellon ModularDiffusers design](https://github.com/cubiq/Mellon/blob/main/modules/ModularDiffusers/README.md) and [configuration implementation](https://github.com/cubiq/Mellon/blob/main/modules/ModularDiffusers/modular_utils.py): design references reviewed on 24 September 2026, not executable MoDiff dependencies.
- Local evidence entry points: `modiff/modular_contract_only_registry.py`, `modiff/operation_catalog.py`, `data/modular-workflow-contracts.json`, `modiff/operation_starters.py`, `src/workflow/operationAuthoring.ts`, `src/workflow/workflowChoices.ts`, `src/workflow/nodeConnectionMatching.ts`.
- The earlier plans named in section 2 preserve detailed historical receipts and unfinished campaign records. Do not overwrite them to imply this new acceptance scope was previously completed.
