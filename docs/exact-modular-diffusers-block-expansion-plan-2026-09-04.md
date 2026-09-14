# Exact Modular Diffusers expansion plan

Status: Qwen text-to-image implemented and qualified; generic rollout pending  
Owner: MoDiff  
First qualification target: `QwenImageModularPipeline` / `text2image`  
Pinned model: `Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26`

## Problem

The current registered Qwen Cluster is a valid `BlockDefinitionV2`, but its
effective graph is compiled from a five-node MoDiff execution skeleton:
`ModelsLoader`, `EncodePrompt`, `Denoise`, `DecodeLatents`, and `Preview`.
That graph is executable, but it is not the exact Modular Diffusers workflow
the Cluster claims to expose. The reviewed catalog already records the exact
upstream block classes, placement paths, fields, components, and ordering, but
the V2 compiler currently discards that information.

The first node labelled `models` is the model loader. Its current label hides
its purpose and its collapsed controls make the pinned repository difficult to
discover. The expanded graph also hides the actual component inventory and the
twelve Qwen text-to-image block placements.

## Required user-visible result

Collapsed mode remains one ordinary Cluster Node with creator starter values,
public inputs, public outputs, preview, resize, and the same action toolbar as a
User Node.

Expanded mode is one resizable container holding ordinary connected nodes:

1. `Load Qwen Image Components` (MoDiff infrastructure)
2. `QwenImageTextEncoderStep` (`text_encoder`)
3. `QwenImageTextInputsStep` (`denoise.input`)
4. `QwenImagePrepareLatentsStep` (`denoise.prepare_latents`)
5. `QwenImageSetTimestepsStep` (`denoise.set_timesteps`)
6. `QwenImageRoPEInputsStep` (`denoise.prepare_rope_inputs`)
7. `QwenImageDenoiseStep` (`denoise.denoise`, loop owner)
8. `QwenImageLoopBeforeDenoiser`
9. `QwenImageLoopDenoiser`
10. `QwenImageLoopAfterDenoiser`
11. `QwenImageAfterDenoiseStep` (`denoise.after_denoise`)
12. `QwenImageDecoderStep` (`decode.decode`)
13. `QwenImageProcessImagesOutputStep` (`decode.postprocess`)
14. `Preview Image` (MoDiff infrastructure)

The loader must visibly show the immutable repository/revision, pipeline and
blocks class, component list, dtype, device, offload, and quantization policy.
Component loading is an explicit MoDiff infrastructure node because upstream
Modular Diffusers separates `ModularPipeline.from_pretrained()` /
`load_components()` from the compute block tree.

## Contract invariants

- `Cluster` and `User Node` are catalog/ownership classifications, not
  different canvas types or renderers. Both use `BlockDefinitionV2`,
  `BlockInstanceV2`, the `block` canvas type, and `BlockNode`.
- `BlockDefinitionV2.graph` is the editable semantic graph shown to the user.
  It may not silently substitute broader convenience stages for reviewed
  upstream blocks.
- A registered Diffusers graph node identifies whether it is MoDiff
  infrastructure or an exact upstream block, and exact blocks carry the pinned
  class, definition id, placement path, block kind, and contract hash.
- Runtime lowering is explicit, versioned, and hash-bound. It may combine
  leaf steps for efficient execution only if every editable field and
  structural operation has a deterministic upstream meaning.
- Decorative or non-executing "fake block" nodes are prohibited. Until a
  structural operation can be rebuilt through upstream `init_pipeline()`, the
  editor must reject it with a precise Fix diagnostic instead of accepting an
  ineffective edit.
- A parameter-only edit changes only its bound value. It must not change node
  identity, sockets, layout, unrelated defaults, or model selection.
- A structural edit changes only that workflow instance to user ownership.
  The registered definition remains immutable; the modified instance and any
  saved User Node preserve all values and layout.
- Save, browser refresh, and backend restart preserve repository, revision,
  prompts, parameters, public interface, internal topology, and layout.
- Collapsed and expanded views execute the same effective graph.
- Cluster Nodes and User Nodes cannot be nested.
- Runtime/resource work must not rewrite creator defaults merely to make a
  model fit. Auto may propose an explicit preset; non-Auto reports actual
  errors and lets the operator decide.

## `BlockDefinitionV2` maintenance

`BlockGraphNodeV2` gains optional `modularDiffusers` metadata:

```text
kind: infrastructure | upstream_block
pipelineClass / blocksClass / workflowId
blockDefinitionId / blockClass / blockKind / blockContractHash
placementPath (array, preserving dotted top-level keys as one segment)
parentPlacementPath (for loop/container ownership)
componentNames
runtimeRole
```

The Python and TypeScript validators, canonical JSON/hash fixtures, API docs,
migration inventory, and normative unified-composite contract must change in
lockstep. Unknown metadata fields continue to fail closed.

This metadata is identity/provenance, not a second value store. Parameters
remain in `node.data.params`; instance values remain in
`BlockInstanceV2.values`; layout remains in
`BlockInstanceV2.presentation.internalLayout`.

A reviewed checkpoint choice within the same pipeline/workflow contract is a
normal bound `BlockControlV2`, not a mutation of `BlockDefinitionV2.source`.
The registered definition keeps its immutable baseline repository and content
hash; the workflow-owned `BlockInstanceV2.values` stores the selected variant.
Its backing graph field contains the deterministic option list copied from the
registered route's exact `reviewedArtifacts`; it must not depend on ambient or
transient node-registry options. Adding, removing, or reordering a reviewed
choice therefore changes the canonical `BlockDefinitionV2` and requires a new
content hash, canonical SHA-256 pin, and route audit.
Before execution the backend must resolve that value through the exact
pipeline-family allowlist and immutable artifact catalog. If changing a model
requires different blocks, ports, or edges, it is a different registered
definition/route and cannot use this same-family control.

## Runtime design

### Semantic graph

The registered compiler consumes the exact `blockPlacements` and
`blockDefinitions` from the reviewed Hugging Face node library. It builds
ordinary semantic nodes and explicit typed state/component edges. Loop
membership is metadata and visible links, not nested Cluster/User Nodes.

### Lowering

The executor lowers a hash-bound semantic graph to runnable MoDiff nodes:

- loader infrastructure loads/reuses components from the exact Hub revision;
- upstream blocks execute through reviewed package-owned adapters;
- loop children remain owned by the reviewed loop and are rebuilt through
  `init_pipeline()` when their order or membership changes;
- preview infrastructure consumes the exact declared workflow output.

The first Qwen delivery may only advertise the unmodified route and supported
parameter mappings as executable. Unsupported add/replace/delete/reconnect
operations must produce a visible compatibility diagnostic until the
composition recipe validates and rebuilds successfully.

## Implementation phases

### Phase 1 — contract and source pin

- Add and validate `modularDiffusers` node metadata in Python and TypeScript.
- Update canonical cross-runtime fixtures and contract documentation.
- Add an exact Qwen topology compiler fixture containing the 12 reviewed
  placements, component inventory, and immutable revisions.

### Phase 2 — Qwen semantic compiler

- Replace the five-card semantic definition with loader + 12 upstream blocks
  + preview.
- Use human-readable labels while retaining exact class/path in the node help.
- Lay out the loop children as ordinary nodes inside the Cluster frame and draw
  entry, sequence, feedback, and exit links.
- Project creator inputs to the exact owning block fields.

### Phase 3 — executable lowering

- Add reviewed Qwen step/loop adapters or an equivalent versioned lowering
  that preserves exact block semantics.
- Load components exclusively through `huggingface_hub`/Diffusers Hub APIs.
- Bind the exact model/revision and component types at runtime.
- Rebuild changed supported topology through upstream `init_pipeline()`.

### Phase 4 — editing and persistence

- Parameter edits update only the addressed binding.
- First structural edit changes instance ownership to a User Node without
  replacing the canvas root or public interface.
- Validate node removal/replacement/reconnection; highlight the first invalid
  block/edge and offer a precise Fix action.
- Preserve independent instances and all three save choices: update existing
  User Node, save as a new User Node, or keep workflow-only.

### Phase 5 — qualification

- Unit: strict schema, canonical hash, exact topology, component inventory,
  typed edges, loop ownership, and lowering validation.
- Browser: insert/click/drag feedback, collapse/expand, resize root and child,
  edit prompt/steps, save, refresh, restart, and compare exact persisted data.
- Real frontend/backend: collapsed run, expanded run, output fingerprint
  parity, parameter-only run, and one supported structural composition run.
- Regression: complete client unit/typecheck/build, backend suite, mocked
  Studio browser suite, and backend-served clean-browser smoke.

### Phase 6 — generic rollout

After Qwen text-to-image passes, apply the same compiler to the remaining Qwen
workflows, then other reviewed Diffusers families. A definition is draggable
only when its exact semantic graph and runtime lowering are admitted. Catalog
classification never changes its renderer.

## Acceptance criteria for Qwen text-to-image

- Expanded view visibly names the loader and shows
  `Qwen/Qwen-Image-2512` plus the exact revision.
- All 12 pinned upstream block placements appear as ordinary, resizable nodes
  inside the Cluster, with correct links and loop ownership.
- The public prompt/negative prompt/size/steps inputs and images output remain
  stable across collapse/expand.
- Editing one parameter changes no other value or interface field.
- Save + refresh + backend restart preserve values, topology, public sockets,
  child sizes, and child positions.
- Collapsed and expanded real runs produce equivalent execution receipts and
  output for the same seed.
- An unsupported structural edit cannot silently run the old graph.
- The production bundle served at `127.0.0.1:8088` passes the browser test;
  testing only the development source is insufficient.

## Known blockers and risks

- Modular Diffusers leaf blocks communicate through mutable `PipelineState`,
  including variadic kwargs bundles; those are not ordinary one-value edges.
- `QwenImageDenoiseStep` is a loop container. Its three children execute per
  timestep and cannot be flattened into one pass each.
- Component replacement must satisfy the exact upstream type hints and pinned
  artifact policy.
- Existing five-node Qwen instances need a hash-bound migration or an explicit
  recovery copy; silently reinterpreting them is prohibited.
- Real generation remains bounded by local accelerator memory and model
  availability, but those conditions do not block schema/compiler/UI work.

## Progress log

- 2026-09-04: confirmed the reviewed catalog contains all 12 exact Qwen
  placements and component contracts; confirmed the V2 compiler currently
  derives its graph from the five-role Studio execution skeleton.
- 2026-09-04: added matching Python and TypeScript
  `BlockGraphNodeV2.modularDiffusers` contracts and strict validators. The
  canonical definition hash now covers infrastructure/upstream identity,
  exact placement paths, block contract hashes, component names, and runtime
  roles.
- 2026-09-04: replaced the Qwen text-to-image five-stage semantic projection
  with one explicit component loader, all twelve reviewed upstream Modular
  Diffusers placements, and one preview node. The projection contains 14
  ordinary, resizable child nodes and 13 explicit state/component/loop links.
- 2026-09-04: added the reviewed runtime step adapter. It validates the pinned
  pipeline/workflow/block identities and hashes, retains upstream
  `PipelineState`, preserves `kwargs_type`, delegates loop iteration to the
  exact upstream loop owner, and normalizes JSON boundary values before
  calling Diffusers.
- 2026-09-04: real visible-frontend lifecycle passed against the pinned
  `Qwen/Qwen-Image-2512` revision: insert two instances, preserve publisher
  defaults, edit one instance, expand, move a child, collapse, save, refresh,
  re-expand, execute, save as a User Node, refresh, and execute again. Backend
  task ids: `4dkWzAlS3H59` and `93LO09k_0jQJ`.
- 2026-09-04: the original collapsed-resize regression is covered directly:
  resize the collapsed Block, expand, verify all 14 children are inside the
  recalculated frame, resize the loader child, move a child, collapse, refresh,
  expand, and recheck containment plus both child size and position. The
  focused mocked browser test, client typecheck/lint/build, and nine reviewed
  runtime tests pass.
- 2026-09-04: mirrored the production client bundle to the backend and ran a
  clean browser smoke against `127.0.0.1:8088`. It observed 14 child nodes, 13
  links, and zero children outside the Block frame. Current registered pin:
  `block-definition-v2-ec976096` /
  `sha256:fd2c47dddc69396efb255765303747bcea88b47385a3d0135ef7078e550ff74f`.
- 2026-09-04: completed the cross-route regression boundary. All 34 registered
  Block V2 adapter/insertion/live-catalog audit tests pass with the exact Qwen
  compiler enabled, alongside client typechecking and focused lint. The nine
  backend reviewed-step/promotion tests also pass against the currently served
  222-node backend registry.
