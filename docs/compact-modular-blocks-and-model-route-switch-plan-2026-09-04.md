# Compact Modular Blocks and Safe Model Route Switching

Date: 2026-09-04

## Outcome

An expanded Diffusers Cluster must remain a faithful visual projection of the reviewed Modular Diffusers workflow while being practical to edit:

- every upstream block remains an independently movable, reconnectable, replaceable node;
- blocks with no editable controls use a compact body instead of an arbitrary 500 px height;
- blocks with a few controls size to their content, while prompt, loader, and preview nodes retain useful working space;
- the compatible model choice is available as a normal instance control on the expanded loader node and collapsed Cluster;
- selecting a model resolves one reviewed repository + immutable revision pair, not an unsafe repository-string mutation;
- compatible prompts, parameters, public connections, internal-node sizes, and internal-node positions survive a route switch;
- same-family variants with an identical component/workflow contract keep the exact same nodes and edges;
- save, refresh, app restart, collapse, and re-expand preserve the selected variant and retained state.

## Upstream findings and constraints

The implementation follows the official Hugging Face contracts rather than treating similarly named repositories as interchangeable:

- `Qwen/Qwen-Image` and `Qwen/Qwen-Image-2512` are text-to-image repositories using the Qwen Image pipeline. The 2512 model card calls it the December update of the original model.
- `Qwen/Qwen-Image-Edit-2511` is an image-edit model. It belongs in the instruction-edit route set and cannot be substituted into a text-to-image Cluster.
- Modular Diffusers maps a repository to its default block collection through `ModularPipeline.from_pretrained`, supports alternate workflows through `blocks.get_workflow(...)`, and initializes a runnable pipeline from the resulting block graph. A model switch must therefore select a reviewed repository + workflow + immutable revision together.
- Internal block boundaries are semantic. Visual compaction must not merge executable blocks or bypass their state/component ports.

Primary references:

- <https://huggingface.co/Qwen/Qwen-Image>
- <https://huggingface.co/Qwen/Qwen-Image-2512>
- <https://huggingface.co/Qwen/Qwen-Image-Edit-2511>
- <https://huggingface.co/docs/diffusers/main/modular_diffusers/quickstart>
- <https://huggingface.co/docs/diffusers/main/modular_diffusers/modular_pipeline>

## Why the selector appeared disabled

The expanded loader displays the registered definition's `model_type`, `repo_id`, and immutable revision as sealed fields. Sealing is correct for those raw fields: editing only the repository would invalidate the reviewed `BlockDefinitionV2` graph, artifact pin, resource recipe, and execution contract.

The earlier loader had no ordinary instance parameter representing a reviewed same-family model variant. The user therefore saw a disabled repository identity with no usable replacement. The implementation now adds one `Model` control whose values are restricted to immutable, reviewed repositories for that exact pipeline + workflow. The raw definition repository and revision remain sealed evidence; execution resolves the selected variant to its catalog pin atomically.

The initial dynamic-field implementation also exposed a second failure mode: a transient backend field publication could contain only the default repository, causing the frontend to render read-only text. The final compiler therefore materializes the exact option list from the registered route's immutable `reviewedArtifacts` contract. Runtime and resource admission independently validate the selected repository and revision; the UI is no longer dependent on ambient node-registry state for this selector.

## Implementation stages

### 1. Content-aware node presentation

Add one deterministic sizing policy used by exact reviewed Modular Diffusers compilation and layout reset:

- state/loop blocks with no visible controls: compact;
- blocks with one to three scalar controls: small;
- blocks with more scalar controls: medium;
- prompt or other multiline controls: tall enough for useful editing;
- model/component loader: medium/tall because immutable artifact evidence and runtime controls are useful;
- preview/media nodes: media-sized.

Store these as default `presentation.internalLayout` values. Preserve any user-resized width/height already stored for an instance. Provide a `Compact internal layout` action so existing instances can explicitly adopt the current defaults without changing graph semantics.

### 2. Same-family model selector

Add a generic reviewed-variant parameter to the Modular Diffusers loader. The backend publishes options only when multiple immutable repositories implement the same installed pipeline and workflow contract. Registered `BlockDefinitionV2` compilation binds that field as an ordinary instance control, so the existing Block renderer shows the same control in collapsed and expanded modes without a special Cluster-only widget.

For Qwen text-to-image the first admitted options are:

- `Qwen/Qwen-Image@75e0b4be04f60ec59a75f475837eced720f823b6`;
- `Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26`.

Both publish the same `QwenImagePipeline` component classes. `Qwen/Qwen-Image-Edit-2511` is excluded because it is an image-edit pipeline. Catalog-only, unpinned, or structurally incompatible repositories are unavailable.

### 3. Compatibility-driven reconciliation

Replace the current `prompt`/`seed`-only carry rule with declarative reconciliation within a modality route set:

1. Match controls and boundary inputs by stable logical ID.
2. Require compatible `BlockValueTypeV2` declarations.
3. Carry the current value only when both conditions pass.
4. Never carry sealed artifact identity, pipeline class, workflow identity, or revision fields.
5. Preserve internal layout for target nodes with the same semantic role.
6. Use the target definition's nodes, edges, defaults, and immutable identity for everything unmatched.
7. Validate every existing external edge before committing the switch.

For the admitted Qwen text-to-image variants, the reviewed workflow schema is identical. The selected model changes as one bound instance value, so the effective graph, prompts, parameters, connections, node positions, and node sizes remain byte-for-byte unchanged. The generic matching rules remain available for a future same-task family whose reviewed variants genuinely require different internal stages, but this phase does not expose cross-family switching.

Switching never mutates or replaces the registered definition. For compatible Qwen text-to-image checkpoints, the workflow keeps the same `definitionRef`; only the ordinary `modelVariant` instance value changes. No route draft, node replacement, edge replacement, or layout reconciliation is involved. Structural user edits remain workflow-instance data or a User Node definition according to the existing save choice.

### 4. Tests

Unit and store tests:

- zero-control and scalar-control blocks receive compact deterministic defaults;
- manual child resizing is not overwritten by ordinary expand/collapse;
- compatible values and internal layouts survive route changes;
- incompatible values, sealed identity, and target-only defaults do not leak;
- external boundary edges either survive with identical type/port contracts or block the switch before mutation;
- switching back restores the route-specific draft;
- route-switch failure is atomic and undo/redo is one history operation.

Browser tests:

- insert Qwen text-to-image Cluster from an empty graph;
- expand it and change the model from the loader;
- verify the node and edge sets do not change and prompt/negative prompt/size/steps/seed are retained;
- resize/move an internal node, switch variants twice, and verify its layout is retained;
- save, refresh, and verify the selected model and values;
- collapse/re-expand and verify the same state;
- run through the frontend and inspect the generated asset.

Live routes:

- Qwen text-to-image: `Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26`.
- Same-family base variant: `Qwen/Qwen-Image@75e0b4be04f60ec59a75f475837eced720f823b6`, after its immutable artifact catalog record and execution/resource validation are in place.

All model acquisition must use the application's Hugging Face Hub snapshot flow. No `wget`, mutable branch, or unreviewed remote Python path is allowed.

## Acceptance criteria

- The first expanded loader has an enabled model selector with accessible busy/error feedback.
- Its immutable repository, revision, pipeline, and workflow fields remain visible and sealed.
- Empty upstream blocks are compact but remain separately selectable, movable, reconnectable, and replaceable.
- A Qwen-Image-2512 → Qwen-Image → Qwen-Image-2512 switch changes only the selected reviewed artifact value; node IDs, edges, prompts, parameters, and internal layout remain unchanged.
- A saved workflow survives browser refresh and backend restart without resetting its selected route or retained values.
- Focused unit/store/browser tests, client check/build, backend contract tests, and one real frontend generation pass.
- Screenshots, graph snapshots, receipts, and generated media are kept outside Git under a new review-only directory containing no historical assets.

## Implementation checkpoint

Current implementation scope is deliberately task-safe:

- `Qwen Image — Text To Image` now owns the `Model` instance control;
- its admitted choices are only `Qwen/Qwen-Image` and `Qwen/Qwen-Image-2512` at their exact reviewed commits;
- Qwen Image Edit 2511 is not an option and remains owned by its separate image-edit Cluster;
- newly inserted Qwen text-to-image Blocks suppress the older cross-family route selector and expose only the ordinary `Model` value; persisted historical cross-family drafts remain readable for recovery;
- reviewed model variants are now admitted by `(pipeline class, workflow ID)`, not by pipeline class alone; repositories for another task cannot appear in or pass the text-to-image selector;
- empty/no-control upstream Modular blocks retain separate execution identities and links, but can be reflowed using `Compact internal layout`;
- model selection propagates to execution, artifact readiness, resource planning, and runtime hints from the instance value.

Completed automated gates at this checkpoint:

- exact route/canonical-pin audit;
- 165 backend contract tests, 12 optional hardware/runtime skips;
- 115 focused client compiler, insertion, renderer, persistence, route, resource, runtime-hint, drag/drop, and library tests.

The post-restart visible-browser lifecycle proves that the selector contains exactly
`Qwen/Qwen-Image` and `Qwen/Qwen-Image-2512`, with no Edit model option. It
selected the base checkpoint from a collapsed Cluster, saved and refreshed the
workflow, verified unchanged graph/prompt/parameter/interface/layout state, and
switched back through the expanded loader. The permanent Playwright lifecycle
test passes in 26.2 seconds. Its current-only screenshots and machine-readable
receipt are in the review directory below.

The base checkpoint was then installed at its exact pinned revision by the
frontend-started Diffusers/Hugging Face Hub execution path. No `wget` or mutable
branch was used. The real run completed with the unchanged Qwen starter values:
1328 by 1328, 50 steps, guidance 4, seed 42, bfloat16, and model CPU offload.
Its current-only evidence is outside Git at
`/home/sayak/MoDiff/review/qwen-t2i-family-final-2026-09-04`.

The final production client bundle was also deployed to the backend `web/`
directory and tested directly at `http://127.0.0.1:8088` using only public UI
controls. The integrated smoke inserted the Cluster, observed the exact two
same-task options, chose the base model, edited the prompt, saved, refreshed,
restored both values, expanded all 14 internal nodes, and selected 2512 from the
expanded loader. The receipt and screenshot are in the same clean review
directory.

Remaining follow-up: forward Hugging Face shard progress and Modular denoising
step progress through the reviewed wrapper. The completed run remained at
coarse fixed queue percentages during both long phases even though heartbeat,
disk, network, and accelerator telemetry confirmed active work.

## Explicitly deferred

- Merging upstream Modular Diffusers blocks into fewer executable nodes. That would reduce composition granularity and requires a separate upstream-equivalence review.
- Treating Qwen Image Edit 2511 as a text-to-image model.
- Cross-family Qwen-to-FLUX switching.
- Allowing arbitrary repository text in a registered loader.
